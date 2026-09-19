#!/usr/bin/env python3
"""
Stage 20: corrected TruFor largest-native-image ONE-STEP memory gate.

This stage intentionally does NOT run the six-image 10-step pilot.

It repeats the already-validated Stage-18 stress case on the exact largest
TextDiffuser image (3248x2088; 6.782 MP) after the v2 attention activation-
lifetime fix.

Hard engineering gates
----------------------
- frozen scientific protocol + implementation hashes validate;
- default allocator/backend contract validates;
- CPU-first dual-model load validates;
- full-native differentiable forward + first-order backward complete;
- forward chunk regression is exactly 148 total query chunks,
  32/32 attention modules split, max 10 chunks in one attention;
- attack-instance E matches the authoritative reference clean E;
- one projected alpha step obeys physical L_inf = 0.25/255 (within tolerance);
- authoritative unmodified TruFor evaluates the candidate;
- classification remains ATTACK;
- one-step candidate is consistent with the previous validated Stage-18
  rounded regression values within a deliberately loose engineering tolerance.

No attack hyperparameter is tuned here. Stop after this gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from trufor_common import ROOT, sha256_file
from trufor_attack_pilot_common import (
    PILOT_IMAGE_PATHS,
    build_union_mask,
    canonical_model_tensor_from_uint8,
    load_frozen_threshold,
    load_regions,
    load_rgb_uint8,
    localisation_values_np,
    physical_linf,
)
from trufor_adversarial_attack_common import (
    ALPHA_PHYSICAL,
    ATTACK_REFERENCE_E_PARITY_ATOL,
    EXPECTED_ATTENTION_MODULES,
    MAX_SCORE_ELEMS,
    ReferenceEvaluator,
    attack_root,
    atomic_write_json,
    environment_record,
    gradient_step,
    load_two_models_cpu_first,
    patch_memory_efficient_attack_model,
    physical_bounds_audit,
    propose_sign_step,
    resolve_cuda_device,
    selection_path,
    validate_backend_record,
    verify_protocol_freeze,
)


EXPECTED_STAGE18_TOTAL_QUERY_CHUNKS = 148
EXPECTED_STAGE18_SPLIT_ATTENTION_MODULES = 32
EXPECTED_STAGE18_MAX_CHUNKS_ONE_ATTENTION = 10

# Rounded values recorded from the previous validated Stage 18.
# These are an engineering regression reference, not an effect-size target.
STAGE18_ONE_STEP_E_ROUNDED = 0.052624
STAGE18_ONE_STEP_SCORE_ROUNDED = 0.850684
STAGE18_ROUNDED_REGRESSION_ATOL = 2.0e-3

ALPHA_LINF_ATOL = 2.0e-7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    backend = validate_backend_record()
    cfg, protocol_sha, selection, stage19 = verify_protocol_freeze(args.run_tag)
    _, threshold = load_frozen_threshold()

    row = (
        selection.sort_values("pilot_order")
        .loc[lambda x: x["pilot_order"] == 1]
        .iloc[0]
    )
    expected_path = PILOT_IMAGE_PATHS[0]
    if str(row["image_path"]) != expected_path:
        raise RuntimeError("Largest-image gate selection is not pilot order 1")

    device = resolve_cuda_device(args.gpu)
    torch.cuda.set_device(device)

    print("TRUFOR STAGE 20 — CORRECTED LARGEST-IMAGE ONE-STEP MEMORY GATE")
    print(f"run tag:       {args.run_tag}")
    print(f"device:        {device}")
    print(f"GPU:           {torch.cuda.get_device_name(device)}")
    print(f"protocol SHA:  {protocol_sha}")
    print(f"threshold:     {threshold:.15f}")
    print(f"MAX_SCORE_ELEMS: {MAX_SCORE_ELEMS:,}")
    print("scope:         ONE image, ONE gradient step, then STOP")
    print()

    t_load = time.perf_counter()
    attack_model, reference_model, load_meta = load_two_models_cpu_first(device)
    route_meta = patch_memory_efficient_attack_model(attack_model)
    load_seconds = time.perf_counter() - t_load

    print("CPU-first dual-model load PASS")
    print(json.dumps(load_meta, indent=2, sort_keys=True))
    print("Corrected memory route:")
    print(json.dumps(route_meta, indent=2, sort_keys=True))
    print(f"model load/patch wall time: {load_seconds:.1f}s")
    print()

    cache_path = ROOT / str(row["cache_path"])
    if not cache_path.is_file():
        raise RuntimeError(f"Missing Policy-C cache: {cache_path}")

    cache_sha = sha256_file(cache_path)
    if cache_sha != str(row["cache_sha256"]):
        raise RuntimeError("Policy-C cache SHA mismatch")

    rgb = load_rgb_uint8(cache_path)
    H, W = int(rgb.shape[0]), int(rgb.shape[1])
    if (W, H) != (3248, 2088):
        raise RuntimeError(f"Unexpected largest-image geometry: {W}x{H}")

    regions = load_regions()
    union_mask_np, clipped = build_union_mask(
        regions,
        str(row["image_path"]),
        H,
        W,
    )
    altered_mask = torch.from_numpy(
        union_mask_np.astype(np.float32, copy=False)
    ).unsqueeze(0).to(device=device)

    clean_x = canonical_model_tensor_from_uint8(rgb, device)

    evaluator = ReferenceEvaluator(
        reference_model,
        altered_mask,
        threshold,
        device,
    )

    # ------------------------------------------------------------------
    # Authoritative clean baseline.
    # ------------------------------------------------------------------
    clean_eval = evaluator.evaluate(clean_x, return_map=True)
    clean_metrics = localisation_values_np(clean_eval["map"], union_mask_np)

    saved_score_gap = abs(
        clean_eval["score"] - float(row["clean_score_stage4"])
    )
    saved_E_gap = abs(
        clean_metrics["E"] - float(row["E_union_stage4"])
    )
    if saved_score_gap > 2e-5:
        raise RuntimeError(
            f"Fresh clean score parity failed: gap={saved_score_gap}"
        )
    if saved_E_gap > 2e-5:
        raise RuntimeError(
            f"Fresh clean E parity failed: gap={saved_E_gap}"
        )
    if not clean_eval["feasible"]:
        raise RuntimeError("Largest pilot image is no longer clean-correct")

    print(
        f"clean reference PASS | score={clean_eval['score']:.9f} | "
        f"E={clean_metrics['E']:.9f}"
    )

    # ------------------------------------------------------------------
    # Corrected full-native differentiable forward + backward.
    # ------------------------------------------------------------------
    grad_state = gradient_step(
        attack_model,
        clean_x,
        altered_mask,
        device,
    )

    attack_reference_E_gap = abs(
        grad_state["attack_E"] - clean_eval["E"]
    )
    if attack_reference_E_gap > ATTACK_REFERENCE_E_PARITY_ATOL:
        raise RuntimeError(
            "Attack/reference clean E parity failed: "
            f"{attack_reference_E_gap}"
        )

    chunk_actual = {
        "attention_modules_observed": int(
            grad_state["attention_modules_observed"]
        ),
        "total_query_chunks": int(grad_state["total_query_chunks"]),
        "split_attention_modules": int(
            grad_state["split_attention_modules"]
        ),
        "max_chunks_one_attention": int(
            grad_state["max_chunks_one_attention"]
        ),
    }
    chunk_expected = {
        "attention_modules_observed": EXPECTED_ATTENTION_MODULES,
        "total_query_chunks": EXPECTED_STAGE18_TOTAL_QUERY_CHUNKS,
        "split_attention_modules": EXPECTED_STAGE18_SPLIT_ATTENTION_MODULES,
        "max_chunks_one_attention": EXPECTED_STAGE18_MAX_CHUNKS_ONE_ATTENTION,
    }
    if chunk_actual != chunk_expected:
        raise RuntimeError(
            "Stage-18 query-chunk regression mismatch:\n"
            f"actual={chunk_actual}\nexpected={chunk_expected}"
        )

    print("differentiable forward/backward PASS")
    print(
        "chunk regression PASS | "
        f"total={chunk_actual['total_query_chunks']} | "
        f"split={chunk_actual['split_attention_modules']}/32 | "
        f"max={chunk_actual['max_chunks_one_attention']}"
    )
    print(
        f"gradient timing | forward={grad_state['gradient_forward_seconds']:.1f}s | "
        f"backward={grad_state['gradient_backward_seconds']:.1f}s"
    )
    print(
        "memory peak | "
        f"allocated={grad_state['gradient_peak_allocated_gib']:.2f} GiB | "
        f"reserved={grad_state['gradient_peak_reserved_gib']:.2f} GiB"
    )

    # ------------------------------------------------------------------
    # One full projected alpha step, matching Stage 18's feasibility gate.
    # No classification bisection is used here.
    # ------------------------------------------------------------------
    grad = grad_state.pop("grad")
    candidate_x = propose_sign_step(clean_x, clean_x, grad)
    del grad
    gc.collect()
    torch.cuda.empty_cache()

    candidate_audit = physical_bounds_audit(clean_x, candidate_x)
    expected_alpha = ALPHA_PHYSICAL
    alpha_linf_error = abs(
        candidate_audit["physical_linf"] - expected_alpha
    )
    if alpha_linf_error > ALPHA_LINF_ATOL:
        raise RuntimeError(
            "One-step physical L_inf does not match alpha: "
            f"linf={candidate_audit['physical_linf']} "
            f"alpha={expected_alpha}"
        )

    candidate_eval = evaluator.evaluate(candidate_x, return_map=True)
    candidate_metrics = localisation_values_np(
        candidate_eval["map"],
        union_mask_np,
    )
    if not candidate_eval["feasible"]:
        raise RuntimeError(
            "Stage-18 regression failed: one projected alpha step no longer "
            "preserves classification"
        )

    stage18_E_abs_error = abs(
        candidate_metrics["E"] - STAGE18_ONE_STEP_E_ROUNDED
    )
    stage18_score_abs_error = abs(
        candidate_eval["score"] - STAGE18_ONE_STEP_SCORE_ROUNDED
    )

    if stage18_E_abs_error > STAGE18_ROUNDED_REGRESSION_ATOL:
        raise RuntimeError(
            "One-step E differs materially from prior validated Stage-18 "
            f"rounded value: actual={candidate_metrics['E']}, "
            f"reference={STAGE18_ONE_STEP_E_ROUNDED}"
        )
    if stage18_score_abs_error > STAGE18_ROUNDED_REGRESSION_ATOL:
        raise RuntimeError(
            "One-step score differs materially from prior validated Stage-18 "
            f"rounded value: actual={candidate_eval['score']}, "
            f"reference={STAGE18_ONE_STEP_SCORE_ROUNDED}"
        )

    print(
        "one projected step PASS | "
        f"Linf={candidate_audit['physical_linf']:.12f} | "
        f"E={clean_metrics['E']:.6f}->{candidate_metrics['E']:.6f} | "
        f"score={clean_eval['score']:.6f}->{candidate_eval['score']:.6f}"
    )
    print(
        "Stage-18 rounded regression PASS | "
        f"|ΔE_ref|={stage18_E_abs_error:.3g} | "
        f"|Δscore_ref|={stage18_score_abs_error:.3g}"
    )

    out_root = attack_root(args.run_tag) / "memory_gate"
    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "stage20_memory_gate_result.json"

    result = {
        "status": "PASS",
        "stage": "20_corrected_largest_image_one_step_memory_gate",
        "run_tag": args.run_tag,
        "protocol_sha256": protocol_sha,
        "scientific_protocol_changed": False,
        "engineering_revision": (
            "v2_attention_chunk_activation_checkpoint"
        ),
        "image": {
            "pilot_order": int(row["pilot_order"]),
            "image_path": str(row["image_path"]),
            "cache_path": str(row["cache_path"]),
            "cache_sha256": cache_sha,
            "native_width": W,
            "native_height": H,
            "native_pixels": int(H * W),
            "clipped_gt_rectangles": int(clipped),
        },
        "model_load": load_meta,
        "memory_route": route_meta,
        "backend": backend,
        "environment": environment_record(device),
        "clean_reference": {
            "score": float(clean_eval["score"]),
            "E": float(clean_metrics["E"]),
            "score_stage4_abs_error": float(saved_score_gap),
            "E_stage4_abs_error": float(saved_E_gap),
        },
        "gradient": {
            "attack_E": float(grad_state["attack_E"]),
            "attack_reference_E_abs_gap": float(
                attack_reference_E_gap
            ),
            "forward_seconds": float(
                grad_state["gradient_forward_seconds"]
            ),
            "backward_seconds": float(
                grad_state["gradient_backward_seconds"]
            ),
            "total_seconds": float(
                grad_state["gradient_seconds"]
            ),
            "peak_allocated_gib": float(
                grad_state["gradient_peak_allocated_gib"]
            ),
            "peak_reserved_gib": float(
                grad_state["gradient_peak_reserved_gib"]
            ),
            "memory_before_gradient": grad_state[
                "memory_before_gradient"
            ],
            "memory_after_forward": grad_state[
                "memory_after_forward"
            ],
            "memory_after_backward": grad_state[
                "memory_after_backward"
            ],
            "chunk_actual": chunk_actual,
            "chunk_expected": chunk_expected,
        },
        "one_projected_step": {
            "physical_linf": float(
                candidate_audit["physical_linf"]
            ),
            "expected_alpha_physical": float(expected_alpha),
            "alpha_linf_abs_error": float(alpha_linf_error),
            "score": float(candidate_eval["score"]),
            "classification_preserved": bool(
                candidate_eval["feasible"]
            ),
            "E": float(candidate_metrics["E"]),
            "previous_stage18_rounded_E": (
                STAGE18_ONE_STEP_E_ROUNDED
            ),
            "previous_stage18_rounded_score": (
                STAGE18_ONE_STEP_SCORE_ROUNDED
            ),
            "stage18_E_abs_error": float(
                stage18_E_abs_error
            ),
            "stage18_score_abs_error": float(
                stage18_score_abs_error
            ),
            "rounded_regression_atol": (
                STAGE18_ROUNDED_REGRESSION_ATOL
            ),
        },
        "next_boundary": (
            "STOP and review this memory-gate result before enabling the "
            "six-image 10-step pilot"
        ),
    }
    atomic_write_json(result_path, result)

    print()
    print("STAGE 20 PASS — CORRECTED MEMORY ROUTE VALIDATED ON 6.782 MP IMAGE")
    print(f"result: {result_path}")
    print(f"result SHA256: {sha256_file(result_path)}")
    print()
    print("STOP HERE.")
    print(
        "Do NOT run Stage 21 or the six-image 10-step pilot yet. "
        "Paste this Stage-20 log/result for review."
    )


if __name__ == "__main__":
    main()
