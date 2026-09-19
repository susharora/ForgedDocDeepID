#!/usr/bin/env python3
"""
Stage 19: freeze the final TruFor adversarial-localisation pilot protocol.

This stage performs no model inference. It freezes:
- exact six-image pilot membership/order;
- frozen threshold and Stage-4 clean-correct population provenance;
- epsilon/alpha/steps/backtracking/objective-acceptance rule;
- exact memory-efficient gradient route;
- backend policy;
- continuous final adversarial representation;
- implementation hashes for Stages 19-21 + shared attack module.

Rerunning Stage 19 after modifying any implementation file deliberately creates
a new implementation freeze. Do not do that merely because a pilot effect size
is disappointing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trufor_common import (
    ROOT,
    sha256_file,
    verify_trufor_provenance,
)
from trufor_attack_pilot_common import (
    PILOT_IMAGE_PATHS,
    PILOT_ROLE,
    build_union_mask,
    load_frozen_threshold,
    load_regions,
    load_rgb_uint8,
    validate_stage4_population,
)
from trufor_adversarial_attack_common import (
    CLEAN_CORRECT_ATTACKS,
    LOCALISATION_PER_IMAGE,
    STAGE4_PROVENANCE,
    attack_root,
    attention_chunk_checkpoint_equivalence_self_test,
    atomic_write_csv,
    atomic_write_json,
    implementation_hashes,
    protocol_config,
    protocol_config_path,
    selection_path,
    stage19_provenance_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    args = parser.parse_args()

    out_root = attack_root(args.run_tag)
    out_root.mkdir(parents=True, exist_ok=True)

    trufor_prov = verify_trufor_provenance(check_archive_member=False)
    attention_checkpoint_self_test = (
        attention_chunk_checkpoint_equivalence_self_test()
    )
    frozen_threshold_payload, threshold = load_frozen_threshold()
    clean, loc, stage4 = validate_stage4_population()
    regions = load_regions()

    clean_by_path = clean.set_index("image_path", drop=False)
    loc_by_path = loc.set_index("image_path", drop=False)

    rows = []
    for order, image_path in enumerate(PILOT_IMAGE_PATHS, start=1):
        if image_path not in clean_by_path.index:
            raise RuntimeError(
                f"Predeclared pilot image missing from clean-correct population: {image_path}"
            )
        if image_path not in loc_by_path.index:
            raise RuntimeError(
                f"Predeclared pilot image missing Stage-4 localisation row: {image_path}"
            )

        c = clean_by_path.loc[image_path]
        l = loc_by_path.loc[image_path]
        cache_path = ROOT / str(c["cache_path"])
        if not cache_path.is_file():
            raise RuntimeError(f"Missing Policy-C cache: {cache_path}")

        actual_cache_sha = sha256_file(cache_path)
        if actual_cache_sha != str(c["cache_sha256"]):
            raise RuntimeError(
                f"Policy-C cache SHA mismatch for {image_path}"
            )

        rgb = load_rgb_uint8(cache_path)
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        mask, clipped = build_union_mask(regions, image_path, H, W)
        A = float(mask.mean())
        if abs(A - float(l["A_union"])) > 1e-12:
            raise RuntimeError(f"Stage-4 A mismatch for {image_path}")

        rows.append(
            {
                "pilot_order": order,
                "pilot_role": PILOT_ROLE[image_path],
                "eval_split": str(c["eval_split"]),
                "variant": str(c["variant"]),
                "hardware_source": str(c["hardware_source"]),
                "file_stem": str(c["file_stem"]),
                "image_path": image_path,
                "cache_path": str(c["cache_path"]),
                "cache_sha256": str(c["cache_sha256"]),
                "clean_score_stage4": float(c["trufor_score"]),
                "native_height": H,
                "native_width": W,
                "native_pixels": int(H * W),
                "A_union_stage4": float(l["A_union"]),
                "E_union_stage4": float(l["E_union"]),
                "mu_union_stage4": float(l["mu_union"]),
                "PG_union_stage4": float(l["PG_union"]),
                "clipped_gt_rectangles_runtime": int(clipped),
            }
        )

    selection = pd.DataFrame(rows)
    if selection["image_path"].tolist() != list(PILOT_IMAGE_PATHS):
        raise RuntimeError("Pilot order mismatch")

    cfg = protocol_config(threshold)
    cfg_path = protocol_config_path(args.run_tag)
    sel_path = selection_path(args.run_tag)

    atomic_write_json(cfg_path, cfg)
    atomic_write_csv(sel_path, selection)

    impl = implementation_hashes()

    provenance = {
        "status": "FROZEN",
        "stage": "19_freeze_trufor_adversarial_protocol",
        "run_tag": args.run_tag,
        "project_root": str(ROOT),
        "attack_protocol_config": str(cfg_path.relative_to(ROOT)),
        "attack_protocol_config_sha256": sha256_file(cfg_path),
        "pilot_selection": str(sel_path.relative_to(ROOT)),
        "pilot_selection_sha256": sha256_file(sel_path),
        "implementation_sha256": impl,
        "frozen_threshold": float(threshold),
        "frozen_threshold_payload": frozen_threshold_payload,
        "clean_correct_attacks": str(CLEAN_CORRECT_ATTACKS.relative_to(ROOT)),
        "clean_correct_attacks_sha256": sha256_file(CLEAN_CORRECT_ATTACKS),
        "localisation_per_image": str(LOCALISATION_PER_IMAGE.relative_to(ROOT)),
        "localisation_per_image_sha256": sha256_file(LOCALISATION_PER_IMAGE),
        "stage4_provenance": str(STAGE4_PROVENANCE.relative_to(ROOT)),
        "stage4_provenance_sha256": sha256_file(STAGE4_PROVENANCE),
        "trufor_provenance": trufor_prov,
        "attention_chunk_checkpoint_equivalence_self_test": (
            attention_checkpoint_self_test
        ),
        "n_pilot_images": int(len(selection)),
        "scientific_note": (
            "Hyperparameters and acceptance rule are frozen before the "
            "six-image multi-step pilot. Pilot effect size is not a pass criterion."
        ),
    }

    prov_path = stage19_provenance_path(args.run_tag)
    atomic_write_json(prov_path, provenance)

    print("TRUFOR STAGE 19 — ADVERSARIAL LOCALISATION PROTOCOL FREEZE")
    print(f"run tag: {args.run_tag}")
    print(f"threshold: {threshold:.15f}")
    print(f"protocol SHA256: {provenance['attack_protocol_config_sha256']}")
    print(f"selection SHA256: {provenance['pilot_selection_sha256']}")
    print()
    print(selection[
        [
            "pilot_order",
            "pilot_role",
            "variant",
            "native_width",
            "native_height",
            "clean_score_stage4",
            "E_union_stage4",
        ]
    ].to_string(index=False))
    print()
    print(
        "attention chunk checkpoint self-test: "
        f"{attention_checkpoint_self_test['status']}"
    )
    print("STAGE 19 PASS — scientific protocol unchanged; corrected implementation frozen.")
    print(
        "Next: Stage 20 largest-image ONE-STEP memory gate only. "
        "Do not launch the six-image pilot yet."
    )


if __name__ == "__main__":
    main()
