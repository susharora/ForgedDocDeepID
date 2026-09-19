#!/usr/bin/env python3
"""Shared utilities for the LABPC TruFor localisation freeze + VRAM pilot.

These helpers intentionally mirror the scientific contract already established:
- exact frozen TruFor checkpoint
- exact frozen Policy-C JPEG inputs
- exact frozen dev-calibrated threshold (NOT recalibrated on LABPC)
- exact six-image worst-case pilot selection

This module is LAB-specific and reads clean baseline artifacts from:
    output/LABPC/trufor_pretrained_policy_c_native/
    output/LABPC/trufor_cross_machine_reproducibility/
    output/LABPC/trufor_policy_c_frozen_protocol/
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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
    resolve_device,
    sha256_file,
    verify_inventory,
    verify_trufor_provenance,
    write_json,
)

RUN_TAG = "LABPC"
LAB_STAGE2_ROOT = ROOT / "output" / RUN_TAG / "trufor_pretrained_policy_c_native"
LAB_REPRO_ROOT = ROOT / "output" / RUN_TAG / "trufor_cross_machine_reproducibility"
LAB_PROTOCOL_ROOT = ROOT / "output" / RUN_TAG / "trufor_policy_c_frozen_protocol"

THRESHOLD_JSON = (
    ROOT
    / "output"
    / "trufor_policy_c_frozen_protocol"
    / "stage03_dev_calibration_accuracy"
    / "frozen_threshold.json"
)

LAB_CLEAN_CORRECT_ATTACKS = LAB_REPRO_ROOT / "lab_clean_correct_attacks.csv"
LAB_CLEAN_CORRECT_BONAFIDES = LAB_REPRO_ROOT / "lab_clean_correct_bonafides.csv"
LAB_DECISION_POPULATION = LAB_REPRO_ROOT / "lab_clean_decision_population.csv"

STAGE11_ROOT = LAB_PROTOCOL_ROOT / "stage11_clean_evaluation_accuracy"
STAGE11_PROVENANCE = STAGE11_ROOT / "stage11_provenance.json"
STAGE11_LOCALISATION = STAGE11_ROOT / "localisation_per_image.csv"
STAGE11_CLEAN_ATTACKS = STAGE11_ROOT / "clean_correct_attacks.csv"
STAGE11_CLEAN_BONAFIDES = STAGE11_ROOT / "clean_correct_bonafides.csv"

EXPECTED_OBJECTIVE = "maximize pooled ordinary image-level accuracy on dev_val"

EPSILON_PHYSICAL = 1.0 / 255.0
ALPHA_PHYSICAL = 0.25 / 255.0
EPSILON_MODEL = 1.0 / 256.0
ALPHA_MODEL = 0.25 / 256.0
MODEL_MAX = 255.0 / 256.0

MAP_PARITY_ATOL = 5e-6
SCORE_PARITY_ATOL = 5e-7
METRIC_PARITY_ATOL = 5e-6

PILOT_IMAGE_PATHS = [
    "test/attack/textdiffuserft_bfei/huawei/"
    "chinese2-flickr_M_33552319041_2631ab0a0c_c.jpg",
    "test/attack/facedancer/huawei/"
    "chinese2-flickr_F_16858102101_d4c5c1e7a2_c.jpg",
    "test/attack/facedancer/huawei/"
    "netherland-flickr_F_20301931855_4cd18f8166_k.jpg",
    "test/attack/digital_3/huawei/usa-01_0038_000-0ee87df5_0.jpg",
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
        raise RuntimeError("run-tag must contain only letters, numbers, '.', '_' or '-'" )
    return tag


def pilot_root(run_tag: str) -> Path:
    return ROOT / "output" / safe_run_tag(run_tag) / "trufor_vram_gradient_pilot"


def load_frozen_threshold() -> Tuple[dict, float]:
    if not THRESHOLD_JSON.is_file():
        raise RuntimeError(f"Missing frozen threshold: {THRESHOLD_JSON}")
    payload = json.loads(THRESHOLD_JSON.read_text())
    if payload.get("status") != "FROZEN":
        raise RuntimeError("Frozen threshold JSON is not in FROZEN state")
    if payload.get("calibration_split") != "dev_val":
        raise RuntimeError("Threshold was not calibrated on dev_val")
    if payload.get("selection_objective") != EXPECTED_OBJECTIVE:
        raise RuntimeError(
            "Unexpected threshold objective:\n"
            f"found {payload.get('selection_objective')!r}\n"
            f"expected {EXPECTED_OBJECTIVE!r}"
        )
    if payload.get("decision_rule") != "attack iff trufor_score >= threshold":
        raise RuntimeError("Unexpected threshold decision rule")
    threshold = float(payload["frozen_threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("Frozen threshold is non-finite")
    return payload, threshold


def require_stage11_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    for path in [STAGE11_PROVENANCE, STAGE11_LOCALISATION, STAGE11_CLEAN_ATTACKS, STAGE11_CLEAN_BONAFIDES]:
        if not path.is_file():
            raise RuntimeError(f"Required Stage-11 file missing: {path}")
    provenance = json.loads(STAGE11_PROVENANCE.read_text())
    if provenance.get("status") != "PASS":
        raise RuntimeError("Stage-11 provenance is not PASS")
    _, threshold = load_frozen_threshold()
    if abs(float(provenance.get("frozen_threshold")) - threshold) > 1e-12:
        raise RuntimeError("Stage-11 threshold does not match frozen threshold")
    loc = pd.read_csv(STAGE11_LOCALISATION, keep_default_na=False)
    attacks = pd.read_csv(STAGE11_CLEAN_ATTACKS, keep_default_na=False)
    if loc["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in Stage-11 localisation table")
    if attacks["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in Stage-11 clean-correct attacks")
    return loc, attacks, provenance


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


def clip_box(box: Tuple[int, int, int, int], width: int, height: int):
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


def build_union_mask(regions: pd.DataFrame, image_path: str, height: int, width: int):
    rows = regions.loc[
        (regions["image_path"] == str(image_path))
        & (regions["region_provenance_raw"] == "altered")
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
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32)
    return x.unsqueeze(0) / 256.0


def freeze_all_model_parameters(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()


def differentiable_forward(model, rgb: torch.Tensor):
    from lib.models.cmx.layer_utils import weighted_statistics_pooling
    modal_x = None
    if "NP++" in model.mods:
        modal_x = model.dncnn(rgb)
        if model.np_out_ch == 1:
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
    out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)
    if model.decode_head_conf is None:
        raise RuntimeError("Final TruFor model has no confidence head")
    conf = model.decode_head_conf(features)
    conf = F.interpolate(conf, size=orisize[2:], mode="bilinear", align_corners=False)
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
    denom = anomaly.sum(dim=(1, 2))
    numer = (anomaly * mask).sum(dim=(1, 2))
    if torch.any(denom <= 0):
        raise RuntimeError("Non-positive anomaly map mass")
    return numer / denom


def canonical_inference_from_tensor(model, x: torch.Tensor):
    with torch.inference_mode():
        out, conf, det, _ = model(x, save_np=False)
        amap = anomaly_from_logits(out)
        score = torch.sigmoid(det).reshape(-1)
    return amap, score


def project_one_step_minimise_E(clean_x: torch.Tensor, grad_x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        candidate = clean_x - ALPHA_MODEL * grad_x.sign()
        lower = torch.clamp(clean_x - EPSILON_MODEL, min=0.0, max=MODEL_MAX)
        upper = torch.clamp(clean_x + EPSILON_MODEL, min=0.0, max=MODEL_MAX)
        candidate = torch.maximum(torch.minimum(candidate, upper), lower)
        candidate = torch.clamp(candidate, min=0.0, max=MODEL_MAX)
    return candidate


def physical_linf(clean_x: torch.Tensor, adv_x: torch.Tensor) -> float:
    return float((((adv_x - clean_x).abs().amax()) * (256.0 / 255.0)).detach().cpu().item())


def gib(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0 ** 3)


def cuda_snapshot(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "cuda_allocated_gib": gib(torch.cuda.memory_allocated(device)),
        "cuda_reserved_gib": gib(torch.cuda.memory_reserved(device)),
        "cuda_free_gib": gib(free_b),
        "cuda_total_gib": gib(total_b),
        "cuda_peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }


def reset_cuda_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def environment_record(device: torch.device) -> Dict[str, object]:
    record = {
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
                "gpu_total_memory_gib": gib(props.total_memory),
                "torch_cuda_version": torch.version.cuda,
            }
        )
    return record


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


class PilotOOM(RuntimeError):
    def __init__(self, phase: str, original: BaseException):
        self.phase = phase
        self.original_message = str(original)
        super().__init__(f"CUDA OOM during {phase}: {original}")
