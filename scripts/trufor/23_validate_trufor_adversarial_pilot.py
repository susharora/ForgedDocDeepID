#!/usr/bin/env python3
"""
Stage 23: validate and summarize the frozen six-image TruFor adversarial pilot.

No minimum attack effect is required for PASS.

Technical PASS means:
- all six hash-valid atomic COMPLETE results exist;
- same Stage-19 scientific protocol SHA;
- same Stage-21 execution implementation hashes;
- 10 trace steps per image;
- all final classifications preserved;
- physical L_inf <= 1/255;
- final E does not exceed clean E beyond numerical tolerance;
- every gradient forward observed all 32 attention modules;
- validated v2 attention activation-checkpoint route recorded.

This stage stops at the scientific decision boundary before the 1330-image run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    load_completed_result,
    pilot_root,
    verify_protocol_freeze,
)

_STAGE21_PATH = Path(__file__).resolve().parent / "21_freeze_trufor_adversarial_pilot_execution.py"
_STAGE21_SPEC = importlib.util.spec_from_file_location(
    "trufor_stage21_pilot_freeze",
    _STAGE21_PATH,
)
if _STAGE21_SPEC is None or _STAGE21_SPEC.loader is None:
    raise RuntimeError(f"Cannot import Stage 21 helper: {_STAGE21_PATH}")
_STAGE21 = importlib.util.module_from_spec(_STAGE21_SPEC)
_STAGE21_SPEC.loader.exec_module(_STAGE21)
validate_execution_freeze = _STAGE21.validate_execution_freeze

FINAL_E_TOL = 3e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    args = parser.parse_args()

    cfg, protocol_sha, selection, stage19 = verify_protocol_freeze(args.run_tag)
    execution_freeze, execution_freeze_sha = validate_execution_freeze(
        args.run_tag
    )

    rows = []
    completion_marker_hashes = {}

    for _, sel in selection.sort_values("pilot_order").iterrows():
        result, trace, image_dir = load_completed_result(
            args.run_tag,
            sel,
            protocol_sha,
        )

        if result.get("status") != "PASS":
            raise RuntimeError(f"Non-PASS atomic result: {sel['image_path']}")
        if len(trace) != ATTACK_STEPS:
            raise RuntimeError(
                f"Expected {ATTACK_STEPS} trace steps for {sel['image_path']}; "
                f"found {len(trace)}"
            )

        clean = result["clean"]
        adv = result["adversarial"]
        degradation = result["degradation"]
        opt = result["optimisation"]
        route = result.get("memory_route", {})

        if route.get("attention_chunk_activation_checkpointing") is not True:
            raise RuntimeError(
                f"Image did not use v2 attention activation checkpointing: "
                f"{sel['image_path']}"
            )
        if int(route.get("attention_modules", -1)) != EXPECTED_ATTENTION_MODULES:
            raise RuntimeError(f"Attention-module route mismatch: {sel['image_path']}")
        if int(route.get("mlp_modules", -1)) != EXPECTED_MLP_MODULES:
            raise RuntimeError(f"MLP-module route mismatch: {sel['image_path']}")

        if not bool(adv["classification_preserved"]):
            raise RuntimeError(f"Classification not preserved: {sel['image_path']}")
        if float(adv["score"]) < float(result["threshold"]):
            raise RuntimeError(f"Final score below threshold: {sel['image_path']}")
        if float(adv["physical_linf"]) > EPSILON_PHYSICAL + L_INF_AUDIT_ATOL:
            raise RuntimeError(f"Physical L_inf violation: {sel['image_path']}")
        if float(adv["E"]) > float(clean["E"]) + FINAL_E_TOL:
            raise RuntimeError(f"Final E increased: {sel['image_path']}")

        if (
            trace["attention_modules_observed"].astype(int)
            != EXPECTED_ATTENTION_MODULES
        ).any():
            raise RuntimeError(
                f"Not all 32 attention modules observed at every step: "
                f"{sel['image_path']}"
            )

        numeric = trace.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Non-finite trace numeric value: {sel['image_path']}")

        # Accepted current E should never increase materially from one row to next.
        current_E = trace["current_E_after"].astype(float).to_numpy()
        if np.any(np.diff(current_E) > FINAL_E_TOL):
            raise RuntimeError(
                f"Accepted-current E increased across steps: {sel['image_path']}"
            )

        marker_path = image_dir / "COMPLETE.json"
        completion_marker_hashes[str(sel["image_path"])] = sha256_file(marker_path)

        gradient_seconds = float(opt["gradient_seconds_total"])
        reference_seconds = float(opt["reference_seconds_total"])

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
                "classification_preserved": bool(
                    adv["classification_preserved"]
                ),
                "clean_E": float(clean["E"]),
                "adv_E": float(adv["E"]),
                "delta_E": float(degradation["delta_E"]),
                "relative_E_degradation": float(
                    degradation["relative_E"]
                ),
                "clean_mu_w": float(clean["mu_w"]),
                "adv_mu_w": float(adv["mu_w"]),
                "delta_mu_w": float(degradation["delta_mu_w"]),
                "relative_mu_w_degradation": float(
                    degradation["relative_mu_w"]
                ),
                "clean_PG": float(clean["PG"]),
                "adv_PG": float(adv["PG"]),
                "delta_PG": float(degradation["delta_PG"]),
                "physical_linf": float(adv["physical_linf"]),
                "accepted_steps": int(opt["accepted_steps"]),
                "boundary_pullbacks": int(opt["boundary_pullbacks"]),
                "reference_forwards": int(opt["reference_forwards_total"]),
                "gradient_seconds": gradient_seconds,
                "reference_seconds": reference_seconds,
                "attack_compute_seconds": gradient_seconds + reference_seconds,
                "image_wall_seconds_before_artifact_hashing": float(
                    result["timing"][
                        "image_wall_seconds_before_artifact_hashing"
                    ]
                ),
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

    results = (
        pd.DataFrame(rows)
        .sort_values("pilot_order")
        .reset_index(drop=True)
    )
    if len(results) != 6:
        raise RuntimeError(f"Expected six completed pilot images; got {len(results)}")

    out_root = pilot_root(args.run_tag)
    results_path = out_root / "pilot_results.csv"
    atomic_write_csv(results_path, results)

    grouped = (
        results.groupby("family", sort=False)
        .agg(
            n=("image_path", "size"),
            clean_E=("clean_E", "mean"),
            adv_E=("adv_E", "mean"),
            delta_E=("delta_E", "mean"),
            relative_E_degradation=("relative_E_degradation", "mean"),
            clean_mu_w=("clean_mu_w", "mean"),
            adv_mu_w=("adv_mu_w", "mean"),
            delta_mu_w=("delta_mu_w", "mean"),
            clean_PG=("clean_PG", "mean"),
            adv_PG=("adv_PG", "mean"),
            delta_PG=("delta_PG", "mean"),
            clean_score=("clean_score", "mean"),
            adv_score=("adv_score", "mean"),
            min_adv_score_margin=("adv_score_margin", "min"),
            boundary_pullbacks=("boundary_pullbacks", "sum"),
            accepted_steps=("accepted_steps", "sum"),
            attack_compute_seconds=("attack_compute_seconds", "sum"),
        )
        .reset_index()
    )
    grouped_path = out_root / "pilot_grouped_descriptive.csv"
    atomic_write_csv(grouped_path, grouped)

    full_n = 1330
    median_compute = float(results["attack_compute_seconds"].median())
    min_compute = float(results["attack_compute_seconds"].min())
    max_compute = float(results["attack_compute_seconds"].max())
    total_pilot_compute = float(results["attack_compute_seconds"].sum())

    runtime = {
        "pilot_n": 6,
        "full_population_n": full_n,
        "pilot_total_attack_compute_hours": total_pilot_compute / 3600.0,
        "median_attack_compute_seconds_per_pilot_image": median_compute,
        "fastest_attack_compute_seconds_per_pilot_image": min_compute,
        "slowest_attack_compute_seconds_per_pilot_image": max_compute,
        "median_size_stress_extrapolated_hours_for_1330": (
            median_compute * full_n / 3600.0
        ),
        "fastest_size_stress_extrapolated_hours_for_1330": (
            min_compute * full_n / 3600.0
        ),
        "slowest_size_stress_extrapolated_hours_for_1330": (
            max_compute * full_n / 3600.0
        ),
        "interpretation": (
            "The six pilot images are deliberately among the largest native "
            "images (~6.5-6.8 MP). These extrapolations are size-stress bounds, "
            "not representative full-population runtime estimates."
        ),
    }

    display_cols = [
        "pilot_order",
        "family",
        "native_MP",
        "clean_E",
        "adv_E",
        "delta_E",
        "relative_E_degradation",
        "clean_mu_w",
        "adv_mu_w",
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

    lines = [
        "TRUFOR SIX-IMAGE x 10-STEP ADVERSARIAL LOCALISATION PILOT",
        "",
        f"protocol SHA256:        {protocol_sha}",
        f"execution freeze SHA:   {execution_freeze_sha}",
        "technical status:       PASS",
        "classification preserved: 6/6",
        "effect-size pass criterion: NONE (by design)",
        "",
        "PER IMAGE",
        results[display_cols].to_string(index=False),
        "",
        "GROUPED DESCRIPTIVE ONLY",
        grouped.to_string(index=False),
        "",
        "RUNTIME",
        json.dumps(runtime, indent=2, sort_keys=True),
        "",
        "SCIENTIFIC DECISION BOUNDARY",
        (
            "Review localisation degradation, near-threshold FaceDancer "
            "behaviour, boundary pullbacks, accepted-step counts and measured "
            "runtime before implementing/launching the 1330-image run."
        ),
        "",
        "STOP HERE. Do not start the full population attack from this stage.",
    ]

    report_path = out_root / "pilot_report.txt"
    report_path.write_text("\n".join(lines) + "\n")

    freeze = {
        "status": "PILOT_VALIDATED_TECHNICALLY",
        "stage": "23_validate_trufor_adversarial_pilot",
        "run_tag": args.run_tag,
        "protocol_sha256": protocol_sha,
        "execution_freeze_sha256": execution_freeze_sha,
        "n_images": 6,
        "classification_preserved_n": int(
            results["classification_preserved"].sum()
        ),
        "max_physical_linf": float(results["physical_linf"].max()),
        "min_adv_score_margin": float(results["adv_score_margin"].min()),
        "total_boundary_pullbacks": int(
            results["boundary_pullbacks"].sum()
        ),
        "total_accepted_steps": int(results["accepted_steps"].sum()),
        "pilot_results": str(results_path.relative_to(ROOT)),
        "pilot_results_sha256": sha256_file(results_path),
        "pilot_grouped_descriptive": str(grouped_path.relative_to(ROOT)),
        "pilot_grouped_descriptive_sha256": sha256_file(grouped_path),
        "pilot_report": str(report_path.relative_to(ROOT)),
        "pilot_report_sha256": sha256_file(report_path),
        "completion_marker_sha256_by_image": completion_marker_hashes,
        "runtime": runtime,
        "effect_size_not_a_pass_criterion": True,
        "next_boundary": (
            "scientific review before full 1330-image implementation/run"
        ),
    }

    freeze_path = (
        attack_root(args.run_tag)
        / "pilot_validated_protocol_freeze.json"
    )
    atomic_write_json(freeze_path, freeze)

    print(report_path.read_text())
    print("STAGE 23 PASS — SIX-IMAGE PILOT TECHNICALLY VALIDATED")
    print(f"validated freeze: {freeze_path}")
    print(f"validated freeze SHA256: {sha256_file(freeze_path)}")


if __name__ == "__main__":
    main()
