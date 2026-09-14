#!/usr/bin/env python3
"""Shared utilities for the frozen pretrained TruFor / FantasyID baseline.

Scientific contract
-------------------
- Upstream TruFor checkout is used in-place from external/TruFor.
- Upstream commit is frozen.
- Final official TruFor checkpoint is frozen by SHA256 and linked back to the
  official weight archive by MD5 + archive-member SHA256.
- Input condition is ONLY the already-materialised FantasyID Policy-C JPEG.
- TruFor preprocessing remains canonical: native HxW RGB, float32 / 256.0.
- No ResNet 512x864 resize, padding, ImageNet transform, or extra JPEG stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

TRUFOR_REPO = ROOT / "external" / "TruFor"
TRUFOR_CODE = TRUFOR_REPO / "TruFor_train_test"
TRUFOR_CHECKPOINT = (
    TRUFOR_REPO / "test_docker" / "src" / "weights" / "trufor.pth.tar"
)
TRUFOR_ARCHIVE = (
    TRUFOR_REPO / "test_docker" / "src" / "TruFor_weights.zip"
)
TRUFOR_CONFIG = TRUFOR_CODE / "lib" / "config" / "trufor_ph3.yaml"

PROJECT_INDEX = ROOT / "output" / "policy_c_cache_index.csv"
OFFICIAL_INDEX = ROOT / "output" / "fantasyid_official_test_policy_c_index.csv"
INVENTORY = ROOT / "output" / "fantasyid_inventory_2026-09-05_023126.xlsx"

OUT_ROOT = ROOT / "output" / "trufor_pretrained_policy_c_native"

EXPECTED_TRUFOR_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
EXPECTED_ARCHIVE_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
EXPECTED_CHECKPOINT_SHA256 = (
    "ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
)
EXPECTED_INVENTORY_SHA256 = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
)

# Hashes measured from the user's clean upstream checkout at the frozen commit.
EXPECTED_CANONICAL_SHA256 = {
    "test.py": "403c554f96fe7923572543cc62cf9d50792fe1ef57e2b6eb8418baaa8577b18e",
    "dataset/dataset_test.py": "53de9aeb6a13727ada9b9d43cf7a0d85595e655951f48445ffa54802d875f72a",
    "lib/config/trufor_ph3.yaml": "a87108eb0df40d9bab6a303eb91419564b7c106d5105bbd5d8ecaec1567b5b8b",
    "lib/models/cmx/builder_np_conf.py": "b31b02a96417b424228f0301ac653a44d257e522c3ddede414d3c95209d80ded",
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file missing:\n{path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Required directory missing:\n{path}")


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return hash_file(path, "sha256")


def md5_file(path: Path) -> str:
    return hash_file(path, "md5")


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def project_git_head() -> str:
    try:
        return git_output(ROOT, "rev-parse", "HEAD")
    except Exception:
        return "unavailable"


def verify_trufor_provenance(check_archive_member: bool = False) -> Dict[str, object]:
    require_dir(TRUFOR_REPO)
    require_dir(TRUFOR_CODE)
    require_file(TRUFOR_CHECKPOINT)
    require_file(TRUFOR_ARCHIVE)
    require_file(TRUFOR_CONFIG)

    head = git_output(TRUFOR_REPO, "rev-parse", "HEAD")
    if head != EXPECTED_TRUFOR_COMMIT:
        raise RuntimeError(
            "Unexpected TruFor upstream commit:\n"
            f"expected {EXPECTED_TRUFOR_COMMIT}\nactual   {head}"
        )

    # Ignore untracked files deliberately. Only modifications/deletions of tracked
    # upstream source would invalidate provenance.
    tracked_status = git_output(
        TRUFOR_REPO,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(
            "Tracked files in external/TruFor are modified.\n"
            f"git status --porcelain --untracked-files=no:\n{tracked_status}"
        )

    archive_md5 = md5_file(TRUFOR_ARCHIVE)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError(
            "Official TruFor weight archive MD5 mismatch:\n"
            f"expected {EXPECTED_ARCHIVE_MD5}\nactual   {archive_md5}"
        )

    checkpoint_sha256 = sha256_file(TRUFOR_CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Extracted TruFor checkpoint SHA256 mismatch:\n"
            f"expected {EXPECTED_CHECKPOINT_SHA256}\nactual   {checkpoint_sha256}"
        )

    canonical_hashes = {}
    for rel, expected in EXPECTED_CANONICAL_SHA256.items():
        path = TRUFOR_CODE / rel
        require_file(path)
        actual = sha256_file(path)
        canonical_hashes[rel] = actual
        if actual != expected:
            raise RuntimeError(
                f"Canonical TruFor file hash mismatch: {rel}\n"
                f"expected {expected}\nactual   {actual}"
            )

    member_sha256 = None
    member_name = None
    if check_archive_member:
        with zipfile.ZipFile(TRUFOR_ARCHIVE, "r") as archive:
            candidates = [
                name for name in archive.namelist()
                if Path(name).name == "trufor.pth.tar"
                and not name.endswith("/")
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "Expected exactly one trufor.pth.tar in official archive; "
                    f"found {candidates}"
                )
            member_name = candidates[0]
            h = hashlib.sha256()
            with archive.open(member_name, "r") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            member_sha256 = h.hexdigest()

        if member_sha256 != checkpoint_sha256:
            raise RuntimeError(
                "Extracted checkpoint is not byte-identical to the official "
                "archive member:\n"
                f"archive member SHA256 {member_sha256}\n"
                f"extracted SHA256      {checkpoint_sha256}"
            )

    return {
        "status": "PASS",
        "project_git_head": project_git_head(),
        "trufor_repo": str(TRUFOR_REPO),
        "trufor_code": str(TRUFOR_CODE),
        "trufor_commit": head,
        "tracked_upstream_status": tracked_status,
        "official_archive": str(TRUFOR_ARCHIVE),
        "official_archive_md5": archive_md5,
        "checkpoint": str(TRUFOR_CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256,
        "archive_member": member_name,
        "archive_member_sha256": member_sha256,
        "canonical_source_sha256": canonical_hashes,
        "preprocessing": "native RGB HxW -> float32 / 256.0",
        "nuisance_condition": "Policy-C only: Q75 4:2:0 -> matched deterministic Q50-90 4:2:0",
    }


def verify_inventory() -> str:
    require_file(INVENTORY)
    actual = sha256_file(INVENTORY)
    if actual != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError(
            "Frozen FantasyID inventory SHA256 mismatch:\n"
            f"expected {EXPECTED_INVENTORY_SHA256}\nactual   {actual}"
        )
    return actual


def load_policy_c_population() -> pd.DataFrame:
    """Return exactly frozen dev_val + complete official test, Policy-C only."""
    require_file(PROJECT_INDEX)
    require_file(OFFICIAL_INDEX)

    project = pd.read_csv(PROJECT_INDEX, keep_default_na=False)
    official = pd.read_csv(OFFICIAL_INDEX, keep_default_na=False)

    if len(project) != 1899:
        raise RuntimeError(f"Expected 1899 project Policy-C rows, got {len(project)}")

    split_counts = project["split"].value_counts().to_dict()
    if split_counts != {"project_train": 1440, "dev_val": 459}:
        raise RuntimeError(f"Unexpected project Policy-C split counts: {split_counts}")

    dev = project.loc[project["split"] == "dev_val"].copy()
    if len(dev) != 459:
        raise RuntimeError("Frozen dev_val must contain 459 images")

    dev_class = dev["label"].value_counts().to_dict()
    if dev_class != {1: 306, 0: 153}:
        raise RuntimeError(f"Unexpected dev class counts: {dev_class}")

    dev_attacks = (
        dev.loc[dev["label"] == 1, "variant"].value_counts().to_dict()
    )
    if dev_attacks != {"digital_1": 153, "digital_2": 153}:
        raise RuntimeError(f"Unexpected dev attack families: {dev_attacks}")

    if len(official) != 1385:
        raise RuntimeError(f"Expected 1385 official-test rows, got {len(official)}")

    official_class = official["label"].value_counts().to_dict()
    if official_class != {1: 1085, 0: 300}:
        raise RuntimeError(f"Unexpected official-test class counts: {official_class}")

    official_attacks = (
        official.loc[official["label"] == 1, "variant"].value_counts().to_dict()
    )
    expected_families = {
        "digital_3": 786,
        "facedancer": 150,
        "textdiffuserft_bfei": 149,
    }
    if official_attacks != expected_families:
        raise RuntimeError(
            f"Unexpected official-test attack families: {official_attacks}"
        )

    for name, frame in [("dev_val", dev), ("official_test", official)]:
        expected_labels = (frame["traffic_type"] == "attack").astype(int)
        if not np.array_equal(frame["label"].astype(int).to_numpy(), expected_labels.to_numpy()):
            raise RuntimeError(f"Class polarity mismatch in {name}")

    dev.insert(0, "eval_split", "dev_val")
    official.insert(0, "eval_split", "official_test")

    # Project index already has a split column; official index does not need one.
    population = pd.concat([dev, official], ignore_index=True, sort=False)

    if len(population) != 1844:
        raise RuntimeError("Expected 1844 Policy-C native baseline images")
    if population["image_path"].duplicated().any():
        dups = population.loc[
            population["image_path"].duplicated(keep=False), "image_path"
        ].tolist()[:10]
        raise RuntimeError(f"Duplicate image_path values in baseline population: {dups}")

    return population


def resolve_device(gpu: int):
    import torch

    if gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
    if gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU {gpu} requested but only {torch.cuda.device_count()} CUDA devices exist"
        )
    return torch.device(f"cuda:{gpu}")


def _safe_torch_load(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_trufor_model(device):
    """Load the final TruFor model from canonical TruFor_train_test code only."""
    import torch

    code = str(TRUFOR_CODE)
    if code not in sys.path:
        sys.path.insert(0, code)

    # Avoid silently importing another package named `lib` from elsewhere.
    if "lib" in sys.modules:
        lib_file = getattr(sys.modules["lib"], "__file__", None)
        if lib_file is not None and not str(Path(lib_file).resolve()).startswith(
            str(TRUFOR_CODE.resolve())
        ):
            raise RuntimeError(
                "A non-TruFor module named 'lib' was already imported:\n"
                f"{lib_file}\nRun this stage in a fresh Python process."
            )

    from lib.config import config as base_config  # type: ignore
    from lib.utils import get_model  # type: ignore
    import lib  # type: ignore

    lib_file = Path(lib.__file__).resolve()
    if not str(lib_file).startswith(str(TRUFOR_CODE.resolve())):
        raise RuntimeError(f"TruFor import escaped canonical source tree: {lib_file}")

    cfg = base_config.clone()
    cfg.defrost()
    cfg.merge_from_file(str(TRUFOR_CONFIG))
    cfg.TEST.MODEL_FILE = str(TRUFOR_CHECKPOINT)
    cfg.freeze()

    if device.type == "cuda":
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = bool(cfg.CUDNN.BENCHMARK)
        cudnn.deterministic = bool(cfg.CUDNN.DETERMINISTIC)
        cudnn.enabled = bool(cfg.CUDNN.ENABLED)

    checkpoint = _safe_torch_load(TRUFOR_CHECKPOINT, device)
    if "state_dict" not in checkpoint:
        raise RuntimeError("TruFor checkpoint does not contain state_dict")

    model = get_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, cfg


def canonical_input_tensor(image_path: Path):
    """Reproduce upstream TestDataset exactly: RGB float tensor / 256.0."""
    import torch

    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB array shape for {image_path}: {rgb.shape}")

    x = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32) / 256.0
    return x.unsqueeze(0), (int(rgb.shape[0]), int(rgb.shape[1]))


def infer_one(model, image_path: Path, device, include_conf: bool = False) -> Dict[str, object]:
    """Canonical final TruFor inference, without any spatial transform."""
    import torch
    from torch.nn import functional as F

    x, image_hw = canonical_input_tensor(image_path)
    x = x.to(device)

    with torch.inference_mode():
        pred, conf, det, _ = model(x, save_np=False)

    if det is None:
        raise RuntimeError("Final TruFor model returned no detection head output")
    if conf is None:
        raise RuntimeError("Final TruFor model returned no confidence map")

    score = torch.sigmoid(det).item()

    pred = torch.squeeze(pred, 0)
    anomaly_map = F.softmax(pred, dim=0)[1].detach().cpu().numpy().astype(np.float32)

    if anomaly_map.shape != image_hw:
        raise RuntimeError(
            "TruFor anomaly map did not return at native input resolution: "
            f"image={image_hw}, map={anomaly_map.shape}"
        )
    if not np.isfinite(anomaly_map).all():
        raise RuntimeError(f"Non-finite values in TruFor anomaly map: {image_path}")
    if anomaly_map.min() < -1e-6 or anomaly_map.max() > 1.0 + 1e-6:
        raise RuntimeError(
            f"TruFor anomaly map outside [0,1] for {image_path}: "
            f"min={anomaly_map.min()}, max={anomaly_map.max()}"
        )
    if not np.isfinite(score) or score < -1e-6 or score > 1.0 + 1e-6:
        raise RuntimeError(f"Invalid TruFor detection score for {image_path}: {score}")

    result: Dict[str, object] = {
        "map": anomaly_map,
        "score": float(score),
        "imgsize": image_hw,
    }

    if include_conf:
        conf_map = torch.squeeze(conf, 0)
        conf_map = torch.sigmoid(conf_map)[0].detach().cpu().numpy().astype(np.float32)
        if conf_map.shape != image_hw:
            raise RuntimeError(
                f"TruFor confidence map shape mismatch: {conf_map.shape} vs {image_hw}"
            )
        result["conf"] = conf_map

    return result


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def map_output_path(eval_split: str, image_path: str) -> Path:
    # Mirror dataset-relative paths underneath a split prefix. Keep original
    # extension in the basename and append .npz for easy traceability.
    return OUT_ROOT / "maps" / eval_split / Path(image_path + ".npz")


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(str(tmp), **arrays)
    os.replace(tmp, path)


def stable_token(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))
