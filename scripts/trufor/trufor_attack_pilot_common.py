#!/usr/bin/env python3
"""Shared utilities for the TruFor native-resolution VRAM/gradient pilot.

Scientific purpose
------------------
This pilot is NOT the final adversarial experiment. It asks only whether the
Home-PC GPU can backpropagate through native-resolution TruFor on the largest
clean-correct forged images under the intended image-space threat model.

Important autograd detail
-------------------------
The official phase-3 TruFor config freezes NP++, the CMX backbone and the
localisation head. Upstream builder_np_conf.py therefore wraps those modules in
torch.no_grad() during normal inference. That is appropriate for inference but
blocks input gradients.

For this white-box attack pilot we do NOT modify upstream source or weights.
Instead, differentiable_forward() calls the exact same frozen submodules with
the same arithmetic while omitting only those torch.no_grad() wrappers. All
model parameters are requires_grad=False. Clean map/score parity against the
frozen Stage-2 outputs is mandatory before any backward pass is accepted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    INVENTORY,
    ROOT,
    load_trufor_model,
    project_git_head,
    relative_to_root,
    resolve_device,
    sha256_file,
    verify_inventory,
    verify_trufor_provenance,
    write_json,
)

PROTOCOL_ROOT = ROOT / "output" / "trufor_policy_c_frozen_protocol"
CALIB_ROOT = PROTOCOL_ROOT / "stage03_dev_calibration_accuracy"
THRESHOLD_JSON = CALIB_ROOT / "frozen_threshold.json"
STAGE4_ROOT = PROTOCOL_ROOT / "stage04_clean_evaluation_accuracy"
STAGE4_PROVENANCE = STAGE4_ROOT / "stage04_provenance.json"
CLEAN_CORRECT_ATTACKS = STAGE4_ROOT / "clean_correct_attacks.csv"
LOCALISATION_PER_IMAGE = STAGE4_ROOT / "localisation_per_image.csv"

EXPECTED_OBJECTIVE = "maximize pooled ordinary image-level accuracy on dev_val"

# Same physical threat model as the frozen ResNet localisation attack.
EPSILON_PHYSICAL = 1.0 / 255.0
ALPHA_PHYSICAL = 0.25 / 255.0

# TruFor canonical loader uses RGB uint8 / 256.0 rather than /255.
# Let z be conventional physical RGB in [0,1] and x be TruFor input:
#       x = z * 255/256
# Therefore a physical epsilon 1/255 maps exactly to 1/256 in x-space.
EPSILON_MODEL = 1.0 / 256.0
ALPHA_MODEL = 0.25 / 256.0
MODEL_MAX = 255.0 / 256.0

MAP_PARITY_ATOL = 5e-6
SCORE_PARITY_ATOL = 5e-7
METRIC_PARITY_ATOL = 5e-6

PILOT_IMAGE_PATHS = [
    # Largest eligible image overall — TextDiffuser.
    "test/attack/textdiffuserft_bfei/huawei/"
    "chinese2-flickr_M_33552319041_2631ab0a0c_c.jpg",

    # Largest eligible FaceDancer.
    "test/attack/facedancer/huawei/"
    "chinese2-flickr_F_16858102101_d4c5c1e7a2_c.jpg",

    # Large FaceDancer close to frozen threshold.
    "test/attack/facedancer/huawei/"
    "netherland-flickr_F_20301931855_4cd18f8166_k.jpg",

    # Largest eligible Digital-3.
    "test/attack/digital_3/huawei/"
    "usa-01_0038_000-0ee87df5_0.jpg",

    # Largest eligible Digital-1 and Digital-2 share stem/geometry.
    "train/attack/digital_1/huawei/turkiye-115_03.jpg",
    "train/attack/digital_2/huawei/turkiye-115_03.jpg",
]

PILOT_ROLE = {
    PILOT_IMAGE_PATHS[0]: "largest_textdiffuser_and_largest_overall",
    PILOT_IMAGE_PATHS[1]: "largest_facedancer",
    PILOT_IMAGE_PATHS[2]: "large_near_threshold_facedancer",
    PILOT_IMAGE_PATHS[3]: "largest_digital_3",
    PILOT_IMAGE_PATHS[4]: "largest_digital_1",
    PILOT_IMAGE_PATHS[5]: "largest_digital_2_same_stem_as_d1",
}


def safe_run_tag(tag: str) -> str:
    tag = str(tag).strip()
    if not tag or not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise RuntimeError(
            "run-tag must contain only letters, numbers, '.', '_' or '-'"
        )
    return tag


def pilot_root(run_tag: str) -> Path:
    return ROOT / "output" / safe_run_tag(run_tag) / "trufor_vram_gradient_pilot"


def load_frozen_threshold() -> Tuple[dict, float]:
    if not THRESHOLD_JSON.is_file():
        raise RuntimeError(f"Missing frozen threshold: {THRESHOLD_JSON}")
    payload = json.loads(THRESHOLD_JSON.read_text())
    if payload.get("status") != "FROZEN":
        raise RuntimeError("Accuracy-calibrated threshold is not FROZEN")
    if payload.get("calibration_split") != "dev_val":
        raise RuntimeError("Frozen threshold was not calibrated on dev_val")
    if payload.get("selection_objective") != EXPECTED_OBJECTIVE:
        raise RuntimeError(
            "Wrong threshold objective:\n"
            f"found {payload.get('selection_objective')!r}\n"
            f"expected {EXPECTED_OBJECTIVE!r}"
        )
    if payload.get("decision_rule") != "attack iff trufor_score >= threshold":
        raise RuntimeError("Unexpected threshold decision rule")
    threshold = float(payload["frozen_threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("Frozen threshold is non-finite")
    return payload, threshold


def validate_stage4_population() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not STAGE4_PROVENANCE.is_file():
        raise RuntimeError(f"Missing Stage-4 provenance: {STAGE4_PROVENANCE}")
    if not CLEAN_CORRECT_ATTACKS.is_file():
        raise RuntimeError(f"Missing clean-correct attacks: {CLEAN_CORRECT_ATTACKS}")
    if not LOCALISATION_PER_IMAGE.is_file():
        raise RuntimeError(f"Missing Stage-4 localisation table: {LOCALISATION_PER_IMAGE}")

    provenance = json.loads(STAGE4_PROVENANCE.read_text())
    if provenance.get("status") != "PASS":
        raise RuntimeError("Stage-4 provenance is not PASS")

    frozen_payload, threshold = load_frozen_threshold()
    if abs(float(provenance.get("frozen_threshold")) - threshold) > 1e-12:
        raise RuntimeError("Stage-4 threshold differs from frozen Stage-3A threshold")

    expected_sha = provenance.get("clean_correct_attacks_sha256")
    actual_sha = sha256_file(CLEAN_CORRECT_ATTACKS)
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError(
            "clean_correct_attacks.csv SHA256 mismatch:\n"
            f"frozen {expected_sha}\nactual {actual_sha}"
        )

    clean = pd.read_csv(CLEAN_CORRECT_ATTACKS, keep_default_na=False)
    loc = pd.read_csv(LOCALISATION_PER_IMAGE, keep_default_na=False)

    if len(clean) != 1330:
        # 153 + 135 + 786 + 107 + 149
        raise RuntimeError(f"Unexpected clean-correct attack count: {len(clean)} != 1330")
    if clean["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in clean-correct attacks")
    if loc["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in Stage-4 localisation table")

    return clean, loc, provenance


def load_regions() -> pd.DataFrame:
    verify_inventory()
    regions = pd.read_excel(INVENTORY, sheet_name="Regions")
    required = {
        "image_path",
        "field_name",
        "region_provenance_raw",
        "x",
        "y",
        "width",
        "height",
    }
    missing = required - set(regions.columns)
    if missing:
        raise RuntimeError(f"Frozen Regions schema mismatch: {sorted(missing)}")

    regions = regions.copy()
    regions["image_path"] = regions["image_path"].astype(str)
    regions["field_name"] = regions["field_name"].astype(str).str.strip().str.lower()
    regions["region_provenance_raw"] = (
        regions["region_provenance_raw"].astype(str).str.strip().str.lower()
    )
    return regions


def round_box(row: pd.Series) -> Tuple[int, int, int, int]:
    x = int(round(float(row["x"])))
    y = int(round(float(row["y"])))
    w = int(round(float(row["width"])))
    h = int(round(float(row["height"])))
    return x, y, x + w, y + h


def clip_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    x0, y0, x1, y1 = box
    clipped = (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )
    changed = clipped != box
    cx0, cy0, cx1, cy1 = clipped
    if cx1 <= cx0 or cy1 <= cy0:
        return None, changed
    return clipped, changed


def build_union_mask(
    regions: pd.DataFrame,
    image_path: str,
    height: int,
    width: int,
) -> Tuple[np.ndarray, int]:
    rows = regions.loc[
        (regions["image_path"].astype(str) == str(image_path))
        & (regions["region_provenance_raw"].astype(str) == "altered")
    ]
    if rows.empty:
        raise RuntimeError(f"No altered rectangles for {image_path}")

    mask = np.zeros((height, width), dtype=bool)
    clipped_count = 0
    valid = 0
    for _, row in rows.iterrows():
        box, changed = clip_box(round_box(row), width, height)
        if changed:
            clipped_count += 1
        if box is None:
            continue
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        valid += 1

    if valid == 0 or int(mask.sum()) == 0:
        raise RuntimeError(f"No valid altered union after clipping: {image_path}")
    return mask, clipped_count


def localisation_values_np(amap: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    amap = np.asarray(amap, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if amap.shape != mask.shape:
        raise RuntimeError(f"map/mask shape mismatch: {amap.shape} vs {mask.shape}")

    area = int(mask.sum())
    total = float(amap.sum())
    if area <= 0 or total <= 0 or not np.isfinite(total):
        raise RuntimeError("Invalid localisation denominator/area")

    A = area / float(mask.size)
    E = float(amap[mask].sum()) / total
    mu = E / A

    flat_index = int(np.argmax(amap.reshape(-1)))
    PG = float(mask.reshape(-1)[flat_index])
    return {"A": float(A), "E": float(E), "mu_w": float(mu), "PG": PG}


def load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB shape: {path}: {rgb.shape}")
    return rgb


def canonical_model_tensor_from_uint8(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Exactly reproduce canonical uint8 / 256.0 clean input."""
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(
        device=device,
        dtype=torch.float32,
    )
    return x.unsqueeze(0) / 256.0


def freeze_all_model_parameters(model) -> None:
    # Parameters are fixed for the attack. Input gradients still flow.
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None
    model.eval()


def differentiable_forward(model, rgb: torch.Tensor):
    """Numerically reproduce TruFor phase-3 forward while exposing d(output)/d(input).

    Upstream forward uses torch.no_grad() around modules listed in FIX_MODULES.
    Here we invoke the same modules directly without those wrappers. No weights
    are trainable and model.eval() remains active.
    """
    from lib.models.cmx.layer_utils import weighted_statistics_pooling

    modal_x = None
    if "NP++" in model.mods:
        modal_x = model.dncnn(rgb)
        if model.np_out_ch == 1:
            # Match upstream builder_np_conf.py exactly.
            modal_x = torch.tile(modal_x, (3, 1, 1))
        elif model.np_out_ch != 3:
            raise RuntimeError(f"Unexpected NP++ output channels: {model.np_out_ch}")

    rgb_branch = rgb
    if "RGB" not in model.mods:
        rgb_branch = None
    elif model.prepro is not None:
        rgb_branch = model.prepro(rgb_branch)

    orisize = rgb_branch.shape if rgb_branch is not None else modal_x.shape

    features = model.backbone(rgb_branch, modal_x)

    out = model.decode_head(features)
    out = F.interpolate(
        out,
        size=orisize[2:],
        mode="bilinear",
        align_corners=False,
    )

    if model.decode_head_conf is None:
        raise RuntimeError("Final TruFor model has no confidence head")
    conf = model.decode_head_conf(features)
    conf = F.interpolate(
        conf,
        size=orisize[2:],
        mode="bilinear",
        align_corners=False,
    )

    if model.detection is None or model.conf_detection != "confpool":
        raise RuntimeError("Unexpected TruFor detection head configuration")

    f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
    f2 = weighted_statistics_pooling(
        out[:, 1:2, :, :] - out[:, 0:1, :, :],
        F.logsigmoid(conf),
    ).view(out.shape[0], -1)
    det = model.detection(torch.cat((f1, f2), dim=-1))

    return out, conf, det


def anomaly_from_logits(out: torch.Tensor) -> torch.Tensor:
    return F.softmax(out, dim=1)[:, 1, :, :]


def differentiable_E(anomaly: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if anomaly.ndim != 3 or mask.ndim != 3:
        raise RuntimeError(
            f"Expected anomaly/mask [B,H,W], got {anomaly.shape}/{mask.shape}"
        )
    denom = anomaly.sum(dim=(1, 2))
    numer = (anomaly * mask).sum(dim=(1, 2))
    if torch.any(denom <= 0):
        raise RuntimeError("Non-positive anomaly-map mass")
    return numer / denom


def canonical_inference_from_tensor(model, x: torch.Tensor):
    """Use the original upstream model forward for no-grad candidate evaluation."""
    with torch.inference_mode():
        out, conf, det, _ = model(x, save_np=False)
        amap = anomaly_from_logits(out)
        score = torch.sigmoid(det).reshape(-1)
    return amap, score


def project_one_step_minimise_E(
    clean_x: torch.Tensor,
    grad_x: torch.Tensor,
) -> torch.Tensor:
    """One sign step in exact TruFor model coordinates.

    Physical threat budget is epsilon=1/255 in conventional RGB [0,1].
    Because canonical TruFor uses /256, this is epsilon_model=1/256.
    """
    with torch.no_grad():
        candidate = clean_x - ALPHA_MODEL * grad_x.sign()

        lower = torch.clamp(clean_x - EPSILON_MODEL, min=0.0, max=MODEL_MAX)
        upper = torch.clamp(clean_x + EPSILON_MODEL, min=0.0, max=MODEL_MAX)
        candidate = torch.maximum(torch.minimum(candidate, upper), lower)
        candidate = torch.clamp(candidate, min=0.0, max=MODEL_MAX)
    return candidate


def physical_linf(clean_x: torch.Tensor, adv_x: torch.Tensor) -> float:
    # z = x * 256/255
    return float(
        ((adv_x - clean_x).abs().amax() * (256.0 / 255.0)).detach().cpu().item()
    )


def mib_to_gib(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0 ** 3)


def cuda_snapshot(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "cuda_allocated_gib": mib_to_gib(torch.cuda.memory_allocated(device)),
        "cuda_reserved_gib": mib_to_gib(torch.cuda.memory_reserved(device)),
        "cuda_free_gib": mib_to_gib(free_b),
        "cuda_total_gib": mib_to_gib(total_b),
        "cuda_peak_allocated_gib": mib_to_gib(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_gib": mib_to_gib(torch.cuda.max_memory_reserved(device)),
    }


def reset_cuda_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def environment_record(device: torch.device) -> Dict[str, object]:
    record: Dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "project_git_head": project_git_head(),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        record.update(
            {
                "cuda_device": str(device),
                "gpu_name": props.name,
                "gpu_total_memory_gib": mib_to_gib(props.total_memory),
                "torch_cuda_version": torch.version.cuda,
            }
        )
    return record


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


class PilotOOM(RuntimeError):
    def __init__(self, phase: str, original: BaseException):
        self.phase = phase
        self.original_message = str(original)
        super().__init__(f"CUDA OOM during {phase}: {original}")
