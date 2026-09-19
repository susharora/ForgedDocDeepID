#!/usr/bin/env python3
"""LABPC Stage 12: freeze the six-image native TruFor VRAM pilot selection.

This stage validates the already-frozen LAB clean baseline and writes:
    output/LABPC/trufor_vram_gradient_pilot/

It selects the same six worst-case clean-correct attacks previously used on HOME,
now anchored to LAB-generated maps and LAB Stage-11 localisation metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trufor_common import ROOT, sha256_file, verify_trufor_provenance, write_json
from trufor_labpc_attack_pilot_common import (
    ALPHA_MODEL,
    ALPHA_PHYSICAL,
    EPSILON_MODEL,
    EPSILON_PHYSICAL,
    LAB_STAGE2_ROOT,
    PILOT_IMAGE_PATHS,
    PILOT_ROLE,
    STAGE11_LOCALISATION,
    STAGE11_PROVENANCE,
    build_union_mask,
    load_frozen_threshold,
    load_regions,
    pilot_root,
    require_stage11_outputs,
    safe_run_tag,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    args = parser.parse_args()
    tag = safe_run_tag(args.run_tag)
    out_root = pilot_root(tag)
    out_root.mkdir(parents=True, exist_ok=True)

    verify_trufor_provenance(check_archive_member=False)
    _, threshold = load_frozen_threshold()
    loc, attacks, stage11 = require_stage11_outputs()
    regions = load_regions()

    by_path = attacks.set_index("image_path", drop=False)
    loc_by_path = loc.set_index("image_path", drop=False)

    rows = []
    for order, image_path in enumerate(PILOT_IMAGE_PATHS, start=1):
        if image_path not in by_path.index:
            raise RuntimeError(
                "Predeclared pilot image is not in LAB clean-correct attacks:\n"
                f"{image_path}"
            )
        if image_path not in loc_by_path.index:
            raise RuntimeError(f"Missing LAB Stage-11 localisation row: {image_path}")
        row = by_path.loc[image_path]
        loc_row = loc_by_path.loc[image_path]
        cache_path = ROOT / str(row["cache_path"])
        map_path = ROOT / str(row["map_path"])
        if not cache_path.is_file():
            raise RuntimeError(f"Missing Policy-C cache: {cache_path}")
        if not map_path.is_file():
            raise RuntimeError(f"Missing LAB Stage-09 NPZ: {map_path}")
        runtime_cache_sha = sha256_file(cache_path)
        expected_cache_sha = str(row["cache_sha256"])
        if runtime_cache_sha != expected_cache_sha:
            raise RuntimeError(
                "Policy-C cache SHA mismatch:\n"
                f"{image_path}\nexpected {expected_cache_sha}\nactual   {runtime_cache_sha}"
            )
        h = int(row["native_height"])
        w = int(row["native_width"])
        mask, clipped = build_union_mask(regions, image_path, h, w)
        score = float(row["trufor_score"])
        if score < threshold:
            raise RuntimeError(f"Pilot row is not clean-correct at frozen threshold: {image_path}")
        rows.append({
            "pilot_order": order,
            "pilot_role": PILOT_ROLE[image_path],
            "eval_split": str(row["eval_split"]),
            "variant": str(row["variant"]),
            "hardware_source": str(row["hardware_source"]),
            "file_stem": str(row["file_stem"]),
            "image_path": image_path,
            "cache_path": str(row["cache_path"]),
            "map_path": str(row["map_path"]),
            "cache_sha256": expected_cache_sha,
            "trufor_score": score,
            "frozen_threshold": threshold,
            "score_margin_above_threshold": score - threshold,
            "native_width": w,
            "native_height": h,
            "native_pixels": h * w,
            "stage11_A_union": float(loc_row["A_union"]),
            "stage11_E_union": float(loc_row["E_union"]),
            "stage11_mu_union": float(loc_row["mu_union"]),
            "stage11_PG_union": float(loc_row["PG_union"]),
            "altered_pixels": int(mask.sum()),
            "clipped_rectangles": int(clipped),
        })
    selection = pd.DataFrame(rows)
    if len(selection) != 6:
        raise RuntimeError(f"Expected 6 pilot images, got {len(selection)}")
    selection_path = out_root / "pilot_selection.csv"
    selection.to_csv(selection_path, index=False)

    config = {
        "status": "PREPARED",
        "stage": "12_prepare_labpc_vram_pilot",
        "run_tag": tag,
        "purpose": "native-resolution one-backward-pass VRAM feasibility pilot on LAB-generated clean baseline",
        "frozen_threshold": threshold,
        "stage11_provenance": str(STAGE11_PROVENANCE.relative_to(ROOT)),
        "stage11_provenance_sha256": sha256_file(STAGE11_PROVENANCE),
        "stage11_localisation": str(STAGE11_LOCALISATION.relative_to(ROOT)),
        "stage11_localisation_sha256": sha256_file(STAGE11_LOCALISATION),
        "pilot_selection": str(selection_path.relative_to(ROOT)),
        "pilot_selection_sha256": sha256_file(selection_path),
        "n_images": len(selection),
        "epsilon_physical_rgb01": EPSILON_PHYSICAL,
        "alpha_physical_rgb01": ALPHA_PHYSICAL,
        "epsilon_trufor_model_coordinates": EPSILON_MODEL,
        "alpha_trufor_model_coordinates": ALPHA_MODEL,
        "objective": "minimise native TruFor anomaly-map mass E inside the altered-union mask",
        "gradient_contract": [
            "All TruFor weights remain unchanged and requires_grad=False.",
            "The same frozen submodules/arithmetic are used; inference no_grad wrappers are bypassed only to expose input gradients.",
            "Clean anomaly-map/score parity with LAB Stage-09 clean output is mandatory before backward.",
            "Clean A/E/mu/PG parity with LAB Stage-11 localisation is mandatory before backward.",
            "Only one projected sign step is attempted.",
        ],
    }
    write_json(out_root / "pilot_config.json", config)

    print("LABPC STAGE 12 — LARGEST-IMAGE VRAM PILOT SELECTION")
    print(f"run tag: {tag}")
    print(f"frozen threshold: {threshold:.9f}")
    print(f"images: {len(selection)}")
    print()
    print(selection[[
        "pilot_order", "pilot_role", "eval_split", "variant", "hardware_source", "trufor_score",
        "score_margin_above_threshold", "native_width", "native_height", "native_pixels", "image_path"
    ]].to_string(index=False))
    print()
    print(f"selection: {selection_path}")
    print(f"config:    {out_root / 'pilot_config.json'}")
    print("STAGE 12 PASS")
    print("Proceed to Stage 13 only after this PASS.")


if __name__ == "__main__":
    main()
