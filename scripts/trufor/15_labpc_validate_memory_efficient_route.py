#!/usr/bin/env python3
"""Stage 15: validate checkpointed/query-chunked TruFor against plain autograd.

The validation deliberately uses the smallest clean-correct image from each of
the five attack families so that the *ordinary* differentiable route can still
fit in GPU memory. For each image:

- compare ordinary differentiable TruFor to frozen LAB Stage-09 clean output;
- compare memory-efficient route to ordinary differentiable route;
- compare E objective;
- compare dE/dx input gradients (cosine + sign agreement).

Only after this PASS should Stage 16 test the six largest images.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trufor_common import (
    ROOT,
    load_trufor_model,
    resolve_device,
    sha256_file,
    verify_trufor_provenance,
    write_json,
)
from trufor_labpc_attack_pilot_common import (
    STAGE11_CLEAN_ATTACKS,
    build_union_mask,
    canonical_model_tensor_from_uint8,
    load_regions,
    load_rgb_uint8,
)
from trufor_memory_efficient_attack import (
    DEFAULT_DNCNN_SEGMENTS,
    DEFAULT_MAX_SCORE_ELEMS,
    anomaly_from_logits,
    attention_chunk_stats,
    configure_reproducible_float32,
    enable_memory_efficient_attack_route,
    freeze_all_parameters,
    gradient_similarity,
    localisation_E,
    memory_efficient_differentiable_forward,
    plain_differentiable_forward,
    require_default_allocator,
    score_from_det,
)

OUT_ROOT = ROOT / "output" / "LABPC" / "trufor_memory_efficient_validation"
FAMILIES = (
    "digital_1",
    "digital_2",
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
)

MAP_CACHED_ATOL = 5e-6
SCORE_CACHED_ATOL = 5e-7

# Validation gate: still very stringent, while allowing normal floating-point
# reordering from query chunking/checkpoint recomputation.
MAP_ROUTE_ATOL = 2e-5
SCORE_ROUTE_ATOL = 2e-5
E_ROUTE_ATOL = 2e-5
GRAD_COS_MIN = 0.9999
GRAD_SIGN_ALL_MIN = 0.999


def cuda_mem(device):
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "free_gib": free_b / 2**30,
        "total_gib": total_b / 2**30,
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def load_cached_map_score(row):
    path = ROOT / str(row["map_path"])
    with np.load(path, allow_pickle=False) as data:
        amap = np.asarray(data["map"], dtype=np.float32)
        score = float(np.asarray(data["score"]).item())
    return amap, score


def choose_validation_rows(clean: pd.DataFrame) -> pd.DataFrame:
    work = clean.copy()
    work["native_pixels"] = (
        work["native_width"].astype(int) * work["native_height"].astype(int)
    )
    rows = []
    for family in FAMILIES:
        sub = work.loc[work["variant"].astype(str) == family].copy()
        if sub.empty:
            raise RuntimeError(f"No clean-correct Stage-11 rows for {family}")
        sub = sub.sort_values(["native_pixels", "image_path"], kind="stable")
        rows.append(sub.iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(0, "validation_order", np.arange(1, len(out) + 1))
    return out


def run_route(model, row, regions, device, route: str):
    rgb = load_rgb_uint8(ROOT / str(row["cache_path"]))
    h, w = rgb.shape[:2]
    mask_np, clipped = build_union_mask(regions, str(row["image_path"]), h, w)

    x = canonical_model_tensor_from_uint8(rgb, device)
    x.requires_grad_(True)
    mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if route == "plain":
        out, conf, det = plain_differentiable_forward(model, x)
    elif route == "efficient":
        out, conf, det = memory_efficient_differentiable_forward(model, x)
    else:
        raise RuntimeError(route)

    amap_t = anomaly_from_logits(out)
    score_t = score_from_det(det)
    E_t = localisation_E(amap_t, mask).reshape(())

    amap = amap_t.detach().squeeze(0).float().cpu().numpy()
    score = float(score_t.detach().cpu().item())
    E = float(E_t.detach().cpu().item())

    grad, = torch.autograd.grad(
        E_t,
        x,
        retain_graph=False,
        create_graph=False,
    )
    torch.cuda.synchronize(device)
    mem = cuda_mem(device)
    chunks = attention_chunk_stats(model) if route == "efficient" else None

    grad_cpu = grad.detach().float().cpu()
    del grad, E_t, amap_t, score_t, out, conf, det, mask, x
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "map": amap,
        "score": score,
        "E": E,
        "grad": grad_cpu,
        "clipped_rectangles": int(clipped),
        "memory": mem,
        "chunks": chunks,
    }


def checkpoint_cpu_device(checkpoint) -> str:
    for value in checkpoint["state_dict"].values():
        if torch.is_tensor(value):
            return str(value.device)
    return "no_tensor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-score-elems", type=int, default=DEFAULT_MAX_SCORE_ELEMS)
    parser.add_argument("--dncnn-segments", type=int, default=DEFAULT_DNCNN_SEGMENTS)
    args = parser.parse_args()

    require_default_allocator()
    configure_reproducible_float32()
    verify_trufor_provenance(check_archive_member=False)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("Stage 15 requires CUDA")

    print("LABPC STAGE 15 — MEMORY-EFFICIENT TRUFOR EQUIVALENCE VALIDATION")
    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"max_score_elems: {args.max_score_elems}")
    print(f"dncnn_segments: {args.dncnn_segments}")
    print("allocator: default")
    print()

    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    print("CUDA trivial allocation PASS:", probe)
    del probe

    if not STAGE11_CLEAN_ATTACKS.is_file():
        raise RuntimeError(f"Missing Stage-11 clean attacks: {STAGE11_CLEAN_ATTACKS}")
    clean = pd.read_csv(STAGE11_CLEAN_ATTACKS, keep_default_na=False)
    selected = choose_validation_rows(clean)
    selected_path = OUT_ROOT / "stage15_validation_selection.csv"
    selected.to_csv(selected_path, index=False)

    print("\nValidation images — smallest clean-correct image per family:")
    print(
        selected[
            [
                "validation_order", "variant", "hardware_source",
                "native_width", "native_height", "native_pixels", "image_path",
            ]
        ].to_string(index=False)
    )

    regions = load_regions()

    # -------- Plain differentiable reference --------
    print("\nLoading PLAIN differentiable TruFor...")
    plain_model, plain_ckpt, _ = load_trufor_model(device)
    freeze_all_parameters(plain_model)
    if checkpoint_cpu_device(plain_ckpt) != "cpu":
        raise RuntimeError("CPU-first checkpoint invariant failed for plain model")

    plain_results = {}
    for _, row in selected.iterrows():
        key = str(row["image_path"])
        print(f"PLAIN {row['validation_order']}/5 | {row['variant']} | {key}")
        result = run_route(plain_model, row, regions, device, "plain")

        cached_map, cached_score = load_cached_map_score(row)
        cached_map_err = float(np.max(np.abs(result["map"] - cached_map)))
        cached_score_err = abs(result["score"] - cached_score)
        if cached_map_err > MAP_CACHED_ATOL or cached_score_err > SCORE_CACHED_ATOL:
            raise RuntimeError(
                "Plain differentiable route failed clean LAB parity:\n"
                f"{key}\nmap_error={cached_map_err}\nscore_error={cached_score_err}"
            )
        result["cached_map_error"] = cached_map_err
        result["cached_score_error"] = cached_score_err
        plain_results[key] = result
        print(
            f"  cached parity map={cached_map_err:.3g} "
            f"score={cached_score_err:.3g} | "
            f"peak_reserved={result['memory']['peak_reserved_gib']:.2f} GiB"
        )

    del plain_model, plain_ckpt
    gc.collect()
    torch.cuda.empty_cache()

    # -------- Memory-efficient attack route --------
    print("\nLoading MEMORY-EFFICIENT TruFor attack route...")
    efficient_model, efficient_ckpt, _ = load_trufor_model(device)
    freeze_all_parameters(efficient_model)
    if checkpoint_cpu_device(efficient_ckpt) != "cpu":
        raise RuntimeError("CPU-first checkpoint invariant failed for efficient model")

    route_info = enable_memory_efficient_attack_route(
        efficient_model,
        max_score_elems=args.max_score_elems,
        dncnn_segments=args.dncnn_segments,
    )
    print("route:", json.dumps(route_info, sort_keys=True))

    rows = []
    for _, row in selected.iterrows():
        key = str(row["image_path"])
        print(f"EFFICIENT {row['validation_order']}/5 | {row['variant']} | {key}")
        efficient = run_route(efficient_model, row, regions, device, "efficient")
        plain = plain_results[key]

        map_err = float(np.max(np.abs(efficient["map"] - plain["map"])))
        map_mean_err = float(np.mean(np.abs(efficient["map"] - plain["map"])))
        score_err = abs(efficient["score"] - plain["score"])
        E_err = abs(efficient["E"] - plain["E"])
        grad_cmp = gradient_similarity(plain["grad"], efficient["grad"])
        chunks = efficient["chunks"]

        passed = (
            map_err <= MAP_ROUTE_ATOL
            and score_err <= SCORE_ROUTE_ATOL
            and E_err <= E_ROUTE_ATOL
            and grad_cmp["gradient_cosine"] >= GRAD_COS_MIN
            and grad_cmp["gradient_sign_agreement_all"] >= GRAD_SIGN_ALL_MIN
        )

        rows.append(
            {
                "validation_order": int(row["validation_order"]),
                "eval_split": str(row["eval_split"]),
                "variant": str(row["variant"]),
                "hardware_source": str(row["hardware_source"]),
                "image_path": key,
                "native_width": int(row["native_width"]),
                "native_height": int(row["native_height"]),
                "native_pixels": int(row["native_pixels"]),
                "plain_cached_map_max_abs_error": plain["cached_map_error"],
                "plain_cached_score_abs_error": plain["cached_score_error"],
                "efficient_vs_plain_map_max_abs_error": map_err,
                "efficient_vs_plain_map_mean_abs_error": map_mean_err,
                "efficient_vs_plain_score_abs_error": score_err,
                "efficient_vs_plain_E_abs_error": E_err,
                **grad_cmp,
                "attention_modules_observed": chunks["modules_observed"],
                "total_query_chunks": chunks["total_query_chunks"],
                "max_chunks_in_one_attention": chunks["max_chunks_in_one_attention"],
                "max_score_elems_actual": chunks["max_score_elems_actual"],
                "plain_peak_reserved_gib": plain["memory"]["peak_reserved_gib"],
                "efficient_peak_reserved_gib": efficient["memory"]["peak_reserved_gib"],
                "passed": bool(passed),
            }
        )
        print(
            f"  map_max={map_err:.3g} score={score_err:.3g} E={E_err:.3g} | "
            f"grad_cos={grad_cmp['gradient_cosine']:.9f} "
            f"sign={grad_cmp['gradient_sign_agreement_all']:.6f} | "
            f"chunks={chunks['total_query_chunks']} | PASS={passed}"
        )

    result_df = pd.DataFrame(rows)
    result_path = OUT_ROOT / "stage15_equivalence_per_image.csv"
    result_df.to_csv(result_path, index=False)

    n_pass = int(result_df["passed"].sum())
    status = "PASS" if n_pass == len(result_df) == 5 else "FAIL"

    report_lines = [
        "LABPC STAGE 15 — CHECKPOINTED / QUERY-CHUNKED TRUFOR VALIDATION",
        "",
        f"STATUS: {status}",
        f"GPU: {torch.cuda.get_device_name(args.gpu)}",
        "allocator: default",
        f"max_score_elems: {args.max_score_elems}",
        f"dncnn_segments: {args.dncnn_segments}",
        f"images passed: {n_pass}/{len(result_df)}",
        "",
        result_df[
            [
                "variant", "native_width", "native_height",
                "efficient_vs_plain_map_max_abs_error",
                "efficient_vs_plain_score_abs_error",
                "efficient_vs_plain_E_abs_error",
                "gradient_cosine",
                "gradient_sign_agreement_all",
                "total_query_chunks",
                "plain_peak_reserved_gib",
                "efficient_peak_reserved_gib",
                "passed",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.9g}"),
        "",
        "Acceptance thresholds:",
        f"  map max abs <= {MAP_ROUTE_ATOL}",
        f"  score abs <= {SCORE_ROUTE_ATOL}",
        f"  E abs <= {E_ROUTE_ATOL}",
        f"  gradient cosine >= {GRAD_COS_MIN}",
        f"  gradient sign agreement(all elements) >= {GRAD_SIGN_ALL_MIN}",
        "",
        "Reviewer reference (not assumed as our result):",
        "  map abs difference <= 8.94e-07",
        "  gradient cosine >= 0.99999988",
        "  gradient sign agreement >= 0.999992",
        "",
    ]
    if status == "PASS":
        report_lines += [
            "SCIENTIFIC GATE:",
            "The memory-efficient route reproduces the ordinary differentiable",
            "TruFor route closely enough on images where both routes fit.",
            "Proceed to Stage 16 six-largest-image native VRAM pilot.",
        ]
    else:
        report_lines += [
            "SCIENTIFIC GATE:",
            "Equivalence validation failed. Do NOT use this route for the large",
            "pilot or full attack until the discrepancy is investigated.",
        ]

    report = "\n".join(report_lines) + "\n"
    report_path = OUT_ROOT / "stage15_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": status,
        "stage": "15_labpc_validate_memory_efficient_route",
        "route_info": route_info,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "selection": str(selected_path.relative_to(ROOT)),
        "selection_sha256": sha256_file(selected_path),
        "results": str(result_path.relative_to(ROOT)),
        "results_sha256": sha256_file(result_path),
        "acceptance": {
            "map_route_atol": MAP_ROUTE_ATOL,
            "score_route_atol": SCORE_ROUTE_ATOL,
            "E_route_atol": E_ROUTE_ATOL,
            "gradient_cosine_min": GRAD_COS_MIN,
            "gradient_sign_all_min": GRAD_SIGN_ALL_MIN,
        },
        "scientific_contract": [
            "Full native-resolution images; no spatial tiling.",
            "Exact query-row chunking inside SegFormer self-attention.",
            "Activation checkpoint/recomputation only; weights unchanged.",
            "First-order input gradient only; create_graph=False.",
            "Authoritative final attack evaluation remains on an unmodified TruFor instance.",
        ],
    }
    write_json(OUT_ROOT / "stage15_provenance.json", provenance)

    print("\n" + report)
    print(f"report: {report_path}")
    print(f"results: {result_path}")

    if status != "PASS":
        raise RuntimeError("STAGE 15 FAIL — equivalence gate not cleared")


if __name__ == "__main__":
    main()
