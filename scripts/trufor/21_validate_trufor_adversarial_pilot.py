#!/usr/bin/env python3
"""
Stage 21: validate and summarise the frozen six-image TruFor adversarial pilot.

Technical PASS criteria:
- all six atomic results are hash-valid COMPLETE;
- identical frozen protocol and implementation hashes;
- classification preserved for every pilot image;
- physical L_inf <= 1/255;
- final E did not increase;
- exact 32-attention / 32-MLP memory route observed;
- no NaN/integrity failure.

There is deliberately NO minimum E-degradation pass criterion.

If technical validation passes, this stage writes a pilot-validated protocol
freeze record. It still stops before the 1330-image population run so the pilot
evidence can be reviewed scientifically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trufor_common import ROOT, sha256_file
from trufor_adversarial_attack_common import (
    ATTACK_STEPS,
    EPSILON_PHYSICAL,
    EXPECTED_ATTENTION_MODULES,
    EXPECTED_MLP_MODULES,
    L_INF_AUDIT_ATOL,
    attack_root,
    atomic_write_csv,
    atomic_write_json,
    implementation_hashes,
    load_completed_result,
    pilot_root,
    verify_protocol_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    args = parser.parse_args()

    cfg, protocol_sha, selection, stage19 = verify_protocol_freeze(args.run_tag)

    rows = []
    marker_hashes = {}

    for _, sel in selection.sort_values("pilot_order").iterrows():
        result, trace, image_dir = load_completed_result(
            args.run_tag,
            sel,
            protocol_sha,
        )

        if result.get("status") != "PASS":
            raise RuntimeError(f"Non-PASS result: {sel['image_path']}")
        if len(trace) != ATTACK_STEPS:
            raise RuntimeError(
                f"Expected {ATTACK_STEPS} trace rows for {sel['image_path']}, "
                f"found {len(trace)}"
            )

        adv = result["adversarial"]
        clean = result["clean"]
        opt = result["optimisation"]

        if not bool(adv["classification_preserved"]):
            raise RuntimeError(f"Classification not preserved: {sel['image_path']}")
        if float(adv["score"]) < float(result["threshold"]):
            raise RuntimeError(f"Final score below threshold: {sel['image_path']}")
        if float(adv["physical_linf"]) > EPSILON_PHYSICAL + L_INF_AUDIT_ATOL:
            raise RuntimeError(f"L_inf violation: {sel['image_path']}")
        if float(adv["E"]) > float(clean["E"]) + 3e-6:
            raise RuntimeError(f"Final E increased: {sel['image_path']}")
        if (trace["attention_modules_observed"].astype(int) != EXPECTED_ATTENTION_MODULES).any():
            raise RuntimeError(f"Attention-route count mismatch: {sel['image_path']}")
        route = result.get("memory_route", {})
        if int(route.get("attention_modules", -1)) != EXPECTED_ATTENTION_MODULES:
            raise RuntimeError(f"Frozen attention-module count mismatch: {sel['image_path']}")
        if int(route.get("mlp_modules", -1)) != EXPECTED_MLP_MODULES:
            raise RuntimeError(f"Frozen MLP-module count mismatch: {sel['image_path']}")
        if int(opt["min_split_attention_modules"]) <= 0:
            raise RuntimeError(
                f"Pilot never exercised genuine multi-chunk attention: {sel['image_path']}"
            )
        if not np.isfinite(trace.select_dtypes(include=[np.number]).to_numpy()).all():
            raise RuntimeError(f"Non-finite numeric trace value: {sel['image_path']}")

        marker_path = image_dir / "COMPLETE.json"
        marker_hashes[str(sel["image_path"])] = sha256_file(marker_path)

        rows.append(
            {
                "pilot_order": int(result["pilot_order"]),
                "pilot_role": result["pilot_role"],
                "eval_split": result["eval_split"],
                "variant": result["variant"],
                "family": result["family_display"],
                "file_stem": result["file_stem"],
                "hardware_source": result["hardware_source"],
                "image_path": result["image_path"],
                "native_width": int(result["native_width"]),
                "native_height": int(result["native_height"]),
                "native_MP": float(result["native_pixels"]) / 1e6,
                "clean_score": float(clean["score"]),
                "adv_score": float(adv["score"]),
                "adv_score_margin": float(adv["score_margin"]),
                "classification_preserved": bool(adv["classification_preserved"]),
                "clean_E": float(clean["E"]),
                "adv_E": float(adv["E"]),
                "delta_E": float(result["degradation"]["delta_E"]),
                "relative_E_degradation": float(
                    result["degradation"]["relative_E"]
                ),
                "clean_mu_w": float(clean["mu_w"]),
                "adv_mu_w": float(adv["mu_w"]),
                "delta_mu_w": float(result["degradation"]["delta_mu_w"]),
                "relative_mu_w_degradation": float(
                    result["degradation"]["relative_mu_w"]
                ),
                "clean_PG": float(clean["PG"]),
                "adv_PG": float(adv["PG"]),
                "delta_PG": float(result["degradation"]["delta_PG"]),
                "physical_linf": float(adv["physical_linf"]),
                "accepted_steps": int(opt["accepted_steps"]),
                "boundary_pullbacks": int(opt["boundary_pullbacks"]),
                "reference_forwards": int(opt["reference_forwards_total"]),
                "gradient_seconds": float(opt["gradient_seconds_total"]),
                "reference_seconds": float(opt["reference_seconds_total"]),
                "max_gradient_peak_allocated_gib": float(
                    opt["max_gradient_peak_allocated_gib"]
                ),
                "max_gradient_peak_reserved_gib": float(
                    opt["max_gradient_peak_reserved_gib"]
                ),
                "min_split_attention_modules": int(
                    opt["min_split_attention_modules"]
                ),
                "max_query_chunks_one_attention": int(
                    opt["max_query_chunks_one_attention"]
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values("pilot_order").reset_index(drop=True)
    if len(results) != 6:
        raise RuntimeError(f"Expected exactly six completed pilot images, got {len(results)}")

    out_root = pilot_root(args.run_tag)

    # Runtime observable from the attack itself. Artifact compression time is
    # intentionally not used as a proxy for GPU attack compute.
    results["attack_compute_seconds"] = (
        results["gradient_seconds"] + results["reference_seconds"]
    )

    results_path = out_root / "pilot_results.csv"
    atomic_write_csv(results_path, results)
    median_seconds = float(results["attack_compute_seconds"].median())
    fastest_seconds = float(results["attack_compute_seconds"].min())
    slowest_seconds = float(results["attack_compute_seconds"].max())

    full_n = 1330
    runtime = {
        "pilot_n": 6,
        "full_population_n": full_n,
        "median_attack_compute_seconds_per_pilot": median_seconds,
        "fastest_attack_compute_seconds_per_pilot": fastest_seconds,
        "slowest_attack_compute_seconds_per_pilot": slowest_seconds,
        "median_extrapolated_hours_for_1330": median_seconds * full_n / 3600.0,
        "fastest_extrapolated_hours_for_1330": fastest_seconds * full_n / 3600.0,
        "slowest_extrapolated_hours_for_1330": slowest_seconds * full_n / 3600.0,
        "interpretation": (
            "Conservative size-stress extrapolation only: the six pilot images "
            "are deliberately among the largest native images and are not a "
            "representative runtime sample of all 1330 images."
        ),
    }

    grouped = (
        results.groupby("family", sort=False)
        .agg(
            n=("image_path", "size"),
            clean_E=("clean_E", "mean"),
            adv_E=("adv_E", "mean"),
            relative_E_degradation=("relative_E_degradation", "mean"),
            clean_mu_w=("clean_mu_w", "mean"),
            adv_mu_w=("adv_mu_w", "mean"),
            delta_PG=("delta_PG", "mean"),
            adv_score=("adv_score", "mean"),
            boundary_pullbacks=("boundary_pullbacks", "sum"),
        )
        .reset_index()
    )
    grouped_path = out_root / "pilot_grouped_descriptive.csv"
    atomic_write_csv(grouped_path, grouped)

    lines = []
    lines.append("TRUFOR SIX-IMAGE ADVERSARIAL LOCALISATION PILOT")
    lines.append("")
    lines.append(f"protocol SHA256: {protocol_sha}")
    lines.append("technical status: PASS")
    lines.append("effect-size pass criterion: NONE (by design)")
    lines.append("")
    lines.append("PER IMAGE")
    display_cols = [
        "pilot_order",
        "family",
        "native_MP",
        "clean_E",
        "adv_E",
        "relative_E_degradation",
        "clean_PG",
        "adv_PG",
        "clean_score",
        "adv_score",
        "adv_score_margin",
        "physical_linf",
        "accepted_steps",
        "boundary_pullbacks",
        "attack_compute_seconds",
    ]
    lines.append(results[display_cols].to_string(index=False))
    lines.append("")
    lines.append("GROUPED DESCRIPTIVE ONLY (pilot n is tiny)")
    lines.append(grouped.to_string(index=False))
    lines.append("")
    lines.append("RUNTIME")
    lines.append(json.dumps(runtime, indent=2, sort_keys=True))
    lines.append("")
    lines.append(
        "STOP HERE: technical pilot validation is complete. Review the six-image "
        "scientific evidence before implementing/launching the 1330-image run."
    )

    report_path = out_root / "pilot_report.txt"
    report_path.write_text("\n".join(lines) + "\n")

    freeze = {
        "status": "PILOT_VALIDATED_TECHNICALLY",
        "stage": "21_validate_trufor_adversarial_pilot",
        "run_tag": args.run_tag,
        "protocol_sha256": protocol_sha,
        "pilot_selection_sha256": stage19["pilot_selection_sha256"],
        "implementation_sha256": implementation_hashes(),
        "pilot_results": str(results_path.relative_to(ROOT)),
        "pilot_results_sha256": sha256_file(results_path),
        "pilot_grouped_descriptive": str(grouped_path.relative_to(ROOT)),
        "pilot_grouped_descriptive_sha256": sha256_file(grouped_path),
        "pilot_report": str(report_path.relative_to(ROOT)),
        "pilot_report_sha256": sha256_file(report_path),
        "completion_marker_sha256_by_image": marker_hashes,
        "runtime": runtime,
        "n_images": 6,
        "classification_preserved_n": int(results["classification_preserved"].sum()),
        "max_physical_linf": float(results["physical_linf"].max()),
        "min_adv_score_margin": float(results["adv_score_margin"].min()),
        "effect_size_not_a_pass_criterion": True,
        "next_boundary": (
            "scientific review of pilot E degradation, near-threshold behaviour, "
            "memory and runtime before full 1330-image implementation/run"
        ),
    }
    freeze_path = attack_root(args.run_tag) / "pilot_validated_protocol_freeze.json"
    atomic_write_json(freeze_path, freeze)

    print(report_path.read_text())
    print("VALIDATED PROTOCOL FREEZE")
    print(f"  {freeze_path}")
    print(f"  SHA256: {sha256_file(freeze_path)}")


if __name__ == "__main__":
    main()
