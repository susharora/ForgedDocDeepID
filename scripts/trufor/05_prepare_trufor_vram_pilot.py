#!/usr/bin/env python3
"""Stage 05: freeze the six-image largest/native-resolution TruFor VRAM pilot.

No model inference occurs here. This stage validates:
- official TruFor provenance and frozen checkpoint;
- the accuracy-calibrated threshold;
- the Stage-4 clean-correct attack population SHA;
- the exact six predeclared pilot images;
- Policy-C cache SHA256 values;
- frozen Stage-2 NPZ presence;
- annotation/localisation baseline availability.

Stop if any selected image is not in the frozen clean-correct population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trufor_common import ROOT, sha256_file, verify_trufor_provenance, write_json
from trufor_attack_pilot_common import (
    ALPHA_MODEL,
    ALPHA_PHYSICAL,
    CLEAN_CORRECT_ATTACKS,
    EPSILON_MODEL,
    EPSILON_PHYSICAL,
    LOCALISATION_PER_IMAGE,
    PILOT_IMAGE_PATHS,
    PILOT_ROLE,
    STAGE4_PROVENANCE,
    build_union_mask,
    load_frozen_threshold,
    load_regions,
    pilot_root,
    safe_run_tag,
    validate_stage4_population,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="HOMEPC")
    args = parser.parse_args()
    tag = safe_run_tag(args.run_tag)
    out_root = pilot_root(tag)
    out_root.mkdir(parents=True, exist_ok=True)

    verify_trufor_provenance(check_archive_member=False)
    frozen, threshold = load_frozen_threshold()
    clean, loc, stage4 = validate_stage4_population()
    regions = load_regions()

    by_path = clean.set_index("image_path", drop=False)
    loc_by_path = loc.set_index("image_path", drop=False)

    rows = []
    for order, image_path in enumerate(PILOT_IMAGE_PATHS, start=1):
        if image_path not in by_path.index:
            raise RuntimeError(
                "Predeclared pilot image is not in frozen clean-correct attacks:\n"
                f"{image_path}"
            )
        if image_path not in loc_by_path.index:
            raise RuntimeError(f"Missing Stage-4 localisation row: {image_path}")

        row = by_path.loc[image_path]
        loc_row = loc_by_path.loc[image_path]

        cache_path = ROOT / str(row["cache_path"])
        map_path = ROOT / str(row["map_path"])
        if not cache_path.is_file():
            raise RuntimeError(f"Missing Policy-C cache: {cache_path}")
        if not map_path.is_file():
            raise RuntimeError(f"Missing frozen Stage-2 NPZ: {map_path}")

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
        if int(mask.sum()) <= 0:
            raise RuntimeError(f"Empty altered mask: {image_path}")

        clean_correct = bool(row["clean_correct"])
        score = float(row["trufor_score"])
        if not clean_correct or score < threshold:
            raise RuntimeError(
                f"Pilot row is not attack-clean-correct at frozen threshold: {image_path}"
            )

        rows.append(
            {
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
                "stage4_A_union": float(loc_row["A_union"]),
                "stage4_E_union": float(loc_row["E_union"]),
                "stage4_mu_union": float(loc_row["mu_union"]),
                "stage4_PG_union": float(loc_row["PG_union"]),
                "altered_pixels": int(mask.sum()),
                "clipped_rectangles": int(clipped),
            }
        )

    selection = pd.DataFrame(rows)
    if len(selection) != 6:
        raise RuntimeError(f"Expected 6 pilot images, got {len(selection)}")
    if selection["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image path in pilot selection")

    selection_path = out_root / "pilot_selection.csv"
    selection.to_csv(selection_path, index=False)

    config = {
        "status": "PREPARED",
        "stage": "05_prepare_trufor_vram_pilot",
        "run_tag": tag,
        "purpose": (
            "native-resolution one-backward-pass VRAM feasibility pilot; "
            "not the final adversarial experiment"
        ),
        "frozen_threshold": threshold,
        "threshold_json": str(
            Path("output/trufor_policy_c_frozen_protocol/")
            / "stage03_dev_calibration_accuracy/frozen_threshold.json"
        ),
        "clean_correct_attacks": str(CLEAN_CORRECT_ATTACKS.relative_to(ROOT)),
        "clean_correct_attacks_sha256": sha256_file(CLEAN_CORRECT_ATTACKS),
        "stage4_provenance": str(STAGE4_PROVENANCE.relative_to(ROOT)),
        "pilot_selection": str(selection_path.relative_to(ROOT)),
        "pilot_selection_sha256": sha256_file(selection_path),
        "n_images": len(selection),
        "epsilon_physical_rgb01": EPSILON_PHYSICAL,
        "alpha_physical_rgb01": ALPHA_PHYSICAL,
        "epsilon_trufor_model_coordinates": EPSILON_MODEL,
        "alpha_trufor_model_coordinates": ALPHA_MODEL,
        "objective": (
            "minimise native TruFor anomaly-map relevance mass E inside the "
            "frozen altered-union rectangle mask"
        ),
        "gradient_contract": [
            "All TruFor weights remain unchanged and requires_grad=False.",
            "Official phase-3 no_grad wrappers around FIX_MODULES are bypassed only to expose input gradients.",
            "The same frozen submodules and arithmetic are called in eval mode.",
            "Clean anomaly-map/score parity with frozen Stage-2 output is mandatory before backward.",
            "Only one projected sign step is attempted; no multi-step attack or hyperparameter search occurs here.",
            "No resize, crop, padding, re-JPEG or uint8 requantisation is introduced.",
        ],
    }
    write_json(out_root / "pilot_config.json", config)

    print("TRUFOR STAGE 05 — LARGEST-IMAGE VRAM PILOT SELECTION")
    print(f"run tag: {tag}")
    print(f"frozen threshold: {threshold:.9f}")
    print(f"images: {len(selection)}")
    print()
    print(
        selection[
            [
                "pilot_order",
                "pilot_role",
                "eval_split",
                "variant",
                "hardware_source",
                "trufor_score",
                "score_margin_above_threshold",
                "native_width",
                "native_height",
                "native_pixels",
                "image_path",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"selection: {selection_path}")
    print(f"config:    {out_root / 'pilot_config.json'}")
    print("STAGE 05 PASS")
    print("Proceed to Stage 06 only after this PASS.")


if __name__ == "__main__":
    main()
