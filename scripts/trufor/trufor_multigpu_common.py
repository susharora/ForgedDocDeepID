#!/usr/bin/env python3
"""Shared utilities for fault-tolerant multi-GPU TruFor Stage-24 sharding."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import trufor_adversarial_attack_common as attack
from trufor_attack_pilot_common import validate_stage4_population
from trufor_common import ROOT, sha256_file

EXPECTED_PROTOCOL_SHA256 = (
    "5840aa9bee076b493c6a645e140edaad892373433c7e919e95ed4e60e44158d5"
)
EXPECTED_N = 1330
DEFAULT_N_SHARDS = 5

REQUIRED_NPZ_KEYS = {
    "adv_x_model",
    "clean_anomaly_map",
    "adv_anomaly_map",
    "altered_union_mask",
}


def multigpu_root(run_tag: str) -> Path:
    return attack.attack_root(run_tag) / "multigpu_sharding"


def shard_outputs_root(run_tag: str) -> Path:
    return attack.attack_root(run_tag) / "full_population_shards"


def legacy_images_root(run_tag: str) -> Path:
    return attack.attack_root(run_tag) / "full_population" / "images"


def plan_freeze_path(run_tag: str) -> Path:
    return multigpu_root(run_tag) / "multigpu_plan_freeze.json"


def shard_csv_path(run_tag: str, shard_id: int) -> Path:
    return multigpu_root(run_tag) / "shards" / f"shard_{shard_id:02d}.csv"


def shard_root(run_tag: str, shard_id: int) -> Path:
    return shard_outputs_root(run_tag) / f"shard_{shard_id:02d}"


def atomic_copy_bytes(src: Path, dst: Path) -> None:
    attack.atomic_write_bytes(dst, src.read_bytes())


def _process_lines() -> List[str]:
    try:
        p = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return p.stdout.splitlines()
    except Exception:
        return []


def conflicting_unsharded_stage24_processes() -> List[str]:
    return [
        line.strip()
        for line in _process_lines()
        if "24_run_trufor_adversarial_full.py" in line
    ]


def active_shard_workers() -> List[str]:
    return [
        line.strip()
        for line in _process_lines()
        if "26_run_trufor_multigpu_shard.py" in line
    ]


def build_full_selection() -> pd.DataFrame:
    """Reconstruct exact Stage-24 order from frozen clean_correct_attacks row order."""
    clean, loc, _ = validate_stage4_population()
    loc_by = loc.set_index("image_path", drop=False)

    rows = []
    for zero_idx, (_, c) in enumerate(clean.iterrows()):
        image_path = str(c["image_path"])
        if image_path not in loc_by.index:
            raise RuntimeError(f"Missing localisation row: {image_path}")
        l = loc_by.loc[image_path]
        global_order = zero_idx + 1

        rows.append(
            {
                "global_order": global_order,
                "pilot_order": global_order,
                "pilot_role": "full_population",
                "eval_split": str(c["eval_split"]),
                "variant": str(c["variant"]),
                "hardware_source": str(c["hardware_source"]),
                "file_stem": str(c["file_stem"]),
                "image_path": image_path,
                "cache_path": str(c["cache_path"]),
                "cache_sha256": str(c["cache_sha256"]),
                "clean_score_stage4": float(c["trufor_score"]),
                "native_height": int(c["native_height"]),
                "native_width": int(c["native_width"]),
                "native_pixels": int(c["native_height"]) * int(c["native_width"]),
                "A_union_stage4": float(l["A_union"]),
                "E_union_stage4": float(l["E_union"]),
                "mu_union_stage4": float(l["mu_union"]),
                "PG_union_stage4": float(l["PG_union"]),
            }
        )

    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} rows, got {len(frame)}")
    if frame["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in full selection")
    if frame["global_order"].tolist() != list(range(1, EXPECTED_N + 1)):
        raise RuntimeError("Global-order reconstruction failed")
    return frame


def validate_complete_dir(
    image_dir: Path,
    protocol_sha: str,
    expected_image_path: str | None = None,
    *,
    deep_npz: bool = False,
) -> dict:
    marker_path = image_dir / "COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError(f"Missing COMPLETE marker: {image_dir}")

    marker = json.loads(marker_path.read_text())
    image_path = str(marker.get("image_path", ""))

    if expected_image_path is not None and image_path != expected_image_path:
        raise RuntimeError(
            f"Marker image_path mismatch: {image_path} != {expected_image_path}"
        )

    if not attack.validate_completion_marker(image_dir, protocol_sha, image_path):
        raise RuntimeError(f"Hash-invalid COMPLETE directory: {image_dir}")

    result = json.loads((image_dir / "result.json").read_text())
    if str(result.get("image_path")) != image_path:
        raise RuntimeError(f"result.json image_path mismatch: {image_dir}")
    if str(result.get("protocol_sha256")) != protocol_sha:
        raise RuntimeError(f"result.json protocol SHA mismatch: {image_dir}")

    if deep_npz:
        npz_path = image_dir / "adversarial_result.npz"
        with np.load(npz_path) as z:
            if set(z.files) != REQUIRED_NPZ_KEYS:
                raise RuntimeError(
                    f"NPZ key mismatch in {image_dir}: {set(z.files)}"
                )
            x = z["adv_x_model"]
            clean = z["clean_anomaly_map"]
            adv = z["adv_anomaly_map"]
            gt = z["altered_union_mask"]

            if x.ndim != 3 or x.shape[0] != 3:
                raise RuntimeError(f"Bad adv_x_model shape in {image_dir}: {x.shape}")
            if clean.shape != adv.shape or clean.shape != gt.shape:
                raise RuntimeError(
                    f"Map/GT mismatch in {image_dir}: "
                    f"{clean.shape}/{adv.shape}/{gt.shape}"
                )
            if x.shape[1:] != clean.shape:
                raise RuntimeError(
                    f"Tensor/map mismatch in {image_dir}: {x.shape}/{clean.shape}"
                )

    return {
        "image_path": image_path,
        "image_dir": str(image_dir),
        "marker_sha256": sha256_file(marker_path),
        "result": result,
    }


def scan_complete_records(
    roots: Iterable[Tuple[str, Path]],
    protocol_sha: str,
    *,
    deep_npz: bool = False,
) -> List[dict]:
    records: List[dict] = []
    for source_name, images_root in roots:
        if not images_root.is_dir():
            continue
        for marker_path in images_root.rglob("COMPLETE.json"):
            validated = validate_complete_dir(
                marker_path.parent,
                protocol_sha,
                deep_npz=deep_npz,
            )
            validated["source_name"] = source_name
            records.append(validated)
    return records


def completed_snapshot_frame(
    run_tag: str,
    selection: pd.DataFrame,
    protocol_sha: str,
) -> pd.DataFrame:
    by_path = selection.set_index("image_path", drop=False)
    records = scan_complete_records(
        [("legacy_stage24", legacy_images_root(run_tag))],
        protocol_sha,
        deep_npz=False,
    )

    rows = []
    seen = set()

    for rec in records:
        image_path = rec["image_path"]
        if image_path in seen:
            raise RuntimeError(f"Duplicate legacy COMPLETE image: {image_path}")
        seen.add(image_path)

        if image_path not in by_path.index:
            raise RuntimeError(
                f"Legacy COMPLETE image outside full selection: {image_path}"
            )

        sel = by_path.loc[image_path]
        result = rec["result"]
        result_order = int(result.get("pilot_order", -1))
        expected_order = int(sel["global_order"])

        if result_order != expected_order:
            raise RuntimeError(
                f"Legacy order mismatch for {image_path}: "
                f"result={result_order}, expected={expected_order}"
            )

        rows.append(
            {
                "global_order": expected_order,
                "image_path": image_path,
                "variant": str(sel["variant"]),
                "hardware_source": str(sel["hardware_source"]),
                "native_pixels": int(sel["native_pixels"]),
                "image_dir": rec["image_dir"],
                "marker_sha256": rec["marker_sha256"],
                "wall_seconds": float(
                    result["timing"]["image_wall_seconds_before_artifact_hashing"]
                ),
                "gradient_seconds": float(
                    result["optimisation"]["gradient_seconds_total"]
                ),
                "reference_seconds": float(
                    result["optimisation"]["reference_seconds_total"]
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values("global_order").reset_index(drop=True)
    return frame


def estimate_remaining_costs(
    remaining: pd.DataFrame,
    completed: pd.DataFrame,
) -> pd.Series:
    if len(completed) == 0:
        fallback = {
            "huawei": 12.53 * 60.0,
            "iphone15pro": 2.84 * 60.0,
            "scan": 0.55 * 60.0,
        }
        return remaining["hardware_source"].map(fallback).fillna(5.38 * 60.0)

    preds = []
    global_median = float(completed["wall_seconds"].median())

    for _, row in remaining.iterrows():
        hw = str(row["hardware_source"])
        target = int(row["native_pixels"])
        pool = completed.loc[completed["hardware_source"] == hw].copy()

        if len(pool) == 0:
            preds.append(global_median)
            continue

        pool["pixel_distance"] = (
            pool["native_pixels"].astype(np.int64) - target
        ).abs()
        nearest = pool.nsmallest(min(25, len(pool)), "pixel_distance")
        pred = float(nearest["wall_seconds"].median())
        preds.append(max(pred, 1.0))

    return pd.Series(preds, index=remaining.index, dtype=float)


def greedy_lpt_assign(frame: pd.DataFrame, n_shards: int) -> pd.DataFrame:
    if n_shards < 1:
        raise RuntimeError("n_shards must be >= 1")

    loads = [0.0] * n_shards
    counts = [0] * n_shards
    assignments = {}

    ordered = frame.sort_values(
        ["estimated_seconds", "native_pixels", "global_order"],
        ascending=[False, False, True],
    )

    for idx, row in ordered.iterrows():
        shard = min(range(n_shards), key=lambda s: (loads[s], counts[s], s))
        assignments[idx] = shard
        loads[shard] += float(row["estimated_seconds"])
        counts[shard] += 1

    out = frame.copy()
    out["shard_id"] = [assignments[i] for i in out.index]
    out["estimated_shard_seconds"] = out["shard_id"].map(
        {i: loads[i] for i in range(n_shards)}
    )
    return out


def validate_plan(run_tag: str) -> Tuple[dict, str]:
    freeze_path = plan_freeze_path(run_tag)
    if not freeze_path.is_file():
        raise RuntimeError(
            f"Missing frozen multi-GPU plan:\n{freeze_path}\nRun Stage 25 first."
        )

    freeze = json.loads(freeze_path.read_text())
    freeze_sha = sha256_file(freeze_path)

    if freeze.get("status") != "FROZEN":
        raise RuntimeError("Multi-GPU plan is not FROZEN")
    if freeze.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Unexpected protocol SHA in multi-GPU plan")

    common_path = Path(attack.__file__).resolve()
    if sha256_file(common_path) != freeze.get("attack_common_sha256"):
        raise RuntimeError(
            "trufor_adversarial_attack_common.py differs from frozen plan"
        )

    root = multigpu_root(run_tag)
    for rel, expected_sha in freeze.get("manifest_sha256", {}).items():
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"Missing frozen manifest file: {path}")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"Frozen manifest SHA mismatch:\n{path}\n"
                f"expected={expected_sha}\nactual={actual}"
            )

    return freeze, freeze_sha


def existing_complete_image_paths(
    run_tag: str,
    protocol_sha: str,
) -> Dict[str, str]:
    roots: List[Tuple[str, Path]] = [
        ("legacy_stage24", legacy_images_root(run_tag))
    ]

    shard_base = shard_outputs_root(run_tag)
    if shard_base.is_dir():
        for p in sorted(shard_base.glob("shard_*/images")):
            roots.append((p.parent.name, p))

    records = scan_complete_records(roots, protocol_sha, deep_npz=False)

    result: Dict[str, str] = {}
    for rec in records:
        image_path = rec["image_path"]
        if image_path in result:
            raise RuntimeError(
                f"Duplicate COMPLETE image already exists: {image_path}\n"
                f"{result[image_path]}\n{rec['image_dir']}"
            )
        result[image_path] = rec["image_dir"]
    return result
