#!/usr/bin/env python3
"""Stage 17: forced-chunk equivalence + project-backend audit.

Why this exists
---------------
Stage 15 passed, but each of 32 attention modules used one query chunk. That
validated checkpointing and wrapper equivalence but did NOT exercise the actual
multi-chunk attention path needed by the ~6.8 MP pilot.

Stage 17:
- preserves this project's existing TruFor CUDNN semantics;
- records TF32/matmul state without changing it;
- forces genuine query splitting with max_score_elems=300M;
- compares the efficient route with the ordinary differentiable route;
- requires total_query_chunks > 32 AND at least one split attention module.

No attack parameters, threshold, inputs or localisation objective are changed.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from trufor_memory_efficient_attack_v2 import (
    DEFAULT_DNCNN_SEGMENTS,
    DEFAULT_MAX_SCORE_ELEMS,
    anomaly_from_logits,
    apply_memory_efficient_route,
    assert_project_trufor_backend,
    backend_state,
    chunk_stats,
    efficient_differentiable_forward,
    freeze_all_parameters,
    gradient_similarity,
    localisation_E,
    plain_differentiable_forward,
    require_default_allocator,
    score_from_det,
)

OUT_ROOT = ROOT / "output" / "LABPC" / "trufor_memory_efficient_validation_v2"
FAMILIES = ("digital_1", "facedancer", "textdiffuserft_bfei")

CACHED_MAP_ATOL = 5e-6
CACHED_SCORE_ATOL = 5e-7

ROUTE_MAP_ATOL = 5e-6
ROUTE_SCORE_ATOL = 5e-6
ROUTE_E_ATOL = 5e-6
GRAD_COS_MIN = 0.99999
GRAD_SIGN_MIN = 0.9999


def choose_rows(clean):
    work = clean.copy()
    work["native_pixels"] = (
        work["native_width"].astype(int)
        * work["native_height"].astype(int)
    )
    rows = []
    for family in FAMILIES:
        sub = work.loc[work["variant"].astype(str) == family].copy()
        if sub.empty:
            raise RuntimeError(f"No Stage-11 clean-correct rows for {family}")
        rows.append(
            sub.sort_values(
                ["native_pixels", "image_path"],
                kind="stable",
            ).iloc[0]
        )
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(0, "validation_order", np.arange(1, len(out) + 1))
    return out


def cached_map_score(row):
    with np.load(ROOT / str(row["map_path"]), allow_pickle=False) as data:
        return (
            np.asarray(data["map"], dtype=np.float32),
            float(np.asarray(data["score"]).item()),
        )


def run_route(model, row, regions, device, efficient):
    rgb = load_rgb_uint8(ROOT / str(row["cache_path"]))
    h, w = rgb.shape[:2]
    mask_np, _ = build_union_mask(
        regions,
        str(row["image_path"]),
        h,
        w,
    )

    x = canonical_model_tensor_from_uint8(rgb, device)
    x.requires_grad_(True)
    mask = (
        torch.from_numpy(mask_np.astype(np.float32))
        .unsqueeze(0)
        .to(device)
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if efficient:
        out, conf, det = efficient_differentiable_forward(model, x)
    else:
        out, conf, det = plain_differentiable_forward(model, x)

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

    peak_alloc = torch.cuda.max_memory_allocated(device) / 2**30
    peak_res = torch.cuda.max_memory_reserved(device) / 2**30

    chunks = chunk_stats(model) if efficient else None
    grad_cpu = grad.detach().float().cpu()

    del grad, E_t, amap_t, score_t, out, conf, det, mask, x
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "map": amap,
        "score": score,
        "E": E,
        "grad": grad_cpu,
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib": peak_res,
        "chunks": chunks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--max-score-elems",
        type=int,
        default=DEFAULT_MAX_SCORE_ELEMS,
    )
    ap.add_argument(
        "--dncnn-segments",
        type=int,
        default=DEFAULT_DNCNN_SEGMENTS,
    )
    args = ap.parse_args()

    require_default_allocator()
    verify_trufor_provenance(check_archive_member=False)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("Stage 17 requires CUDA")

    print("LABPC STAGE 17 — FORCED-CHUNK TRUFOR EQUIVALENCE")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"max_score_elems: {args.max_score_elems}")
    print(f"dncnn_segments: {args.dncnn_segments}")
    print("allocator: default")
    print("NOTE: backend state is observed/preserved, not overwritten.")
    print()

    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    print("CUDA trivial allocation PASS:", probe)
    del probe

    if not STAGE11_CLEAN_ATTACKS.is_file():
        raise RuntimeError(f"Missing {STAGE11_CLEAN_ATTACKS}")
    clean = pd.read_csv(STAGE11_CLEAN_ATTACKS, keep_default_na=False)
    selected = choose_rows(clean)
    selection_path = OUT_ROOT / "stage17_selection.csv"
    selected.to_csv(selection_path, index=False)

    print("\nRepresentative validation rows:")
    print(
        selected[
            [
                "validation_order", "variant", "hardware_source",
                "native_width", "native_height", "native_pixels",
                "image_path",
            ]
        ].to_string(index=False)
    )

    regions = load_regions()

    # ---------------- plain route ----------------
    print("\nLoading ordinary differentiable TruFor...")
    plain_model, plain_ckpt, plain_cfg = load_trufor_model(device)
    freeze_all_parameters(plain_model)
    backend_plain = assert_project_trufor_backend(plain_cfg)
    print("backend_after_plain_load:", json.dumps(backend_plain, sort_keys=True))

    if int(plain_ckpt.get("epoch", -1)) != 81:
        raise RuntimeError("Unexpected checkpoint epoch")

    plain = {}
    for _, row in selected.iterrows():
        key = str(row["image_path"])
        print(f"PLAIN {row['validation_order']}/3 | {row['variant']} | {key}")
        r = run_route(plain_model, row, regions, device, efficient=False)
        cmap, cscore = cached_map_score(row)
        map_err = float(np.max(np.abs(r["map"] - cmap)))
        score_err = abs(r["score"] - cscore)
        if map_err > CACHED_MAP_ATOL or score_err > CACHED_SCORE_ATOL:
            raise RuntimeError(
                f"Plain clean parity failed: {key}: "
                f"map={map_err}, score={score_err}"
            )
        r["cached_map_error"] = map_err
        r["cached_score_error"] = score_err
        plain[key] = r
        print(
            f"  cached map={map_err:.3g} score={score_err:.3g}"
        )

    del plain_model, plain_ckpt
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- efficient forced-chunk route ----------------
    print("\nLoading corrected memory-efficient TruFor...")
    eff_model, eff_ckpt, eff_cfg = load_trufor_model(device)
    freeze_all_parameters(eff_model)
    backend_eff_before_wrap = assert_project_trufor_backend(eff_cfg)

    route_info = apply_memory_efficient_route(
        eff_model,
        max_score_elems=args.max_score_elems,
        dncnn_segments=args.dncnn_segments,
    )
    backend_eff_after_wrap = assert_project_trufor_backend(eff_cfg)

    if backend_eff_after_wrap != backend_eff_before_wrap:
        raise RuntimeError(
            "Memory wrapper changed backend state unexpectedly"
        )
    if backend_eff_after_wrap != backend_plain:
        raise RuntimeError(
            "Plain and efficient routes do not share the same backend state"
        )

    print("route_info:", json.dumps(route_info, sort_keys=True))
    print(
        "backend_after_efficient_wrap:",
        json.dumps(backend_eff_after_wrap, sort_keys=True),
    )

    rows = []
    for _, row in selected.iterrows():
        key = str(row["image_path"])
        print(
            f"EFFICIENT {row['validation_order']}/3 | "
            f"{row['variant']} | {key}"
        )
        er = run_route(eff_model, row, regions, device, efficient=True)
        pr = plain[key]

        map_err = float(np.max(np.abs(er["map"] - pr["map"])))
        score_err = abs(er["score"] - pr["score"])
        E_err = abs(er["E"] - pr["E"])
        g = gradient_similarity(pr["grad"], er["grad"])
        c = er["chunks"]

        forced_chunking = bool(
            c["total_query_chunks"] > c["modules_observed"]
            and c["n_attention_modules_actually_split"] > 0
            and c["max_chunks_in_one_attention"] >= 2
        )

        passed = bool(
            map_err <= ROUTE_MAP_ATOL
            and score_err <= ROUTE_SCORE_ATOL
            and E_err <= ROUTE_E_ATOL
            and g["gradient_cosine"] >= GRAD_COS_MIN
            and g["gradient_sign_agreement"] >= GRAD_SIGN_MIN
            and forced_chunking
        )

        rows.append(
            {
                "validation_order": int(row["validation_order"]),
                "variant": str(row["variant"]),
                "image_path": key,
                "native_width": int(row["native_width"]),
                "native_height": int(row["native_height"]),
                "efficient_vs_plain_map_max_abs_error": map_err,
                "efficient_vs_plain_score_abs_error": score_err,
                "efficient_vs_plain_E_abs_error": E_err,
                **g,
                "attention_modules_observed": c["modules_observed"],
                "total_query_chunks": c["total_query_chunks"],
                "n_attention_modules_actually_split":
                    c["n_attention_modules_actually_split"],
                "max_chunks_in_one_attention":
                    c["max_chunks_in_one_attention"],
                "max_score_elems_actual": c["max_score_elems_actual"],
                "forced_chunking_exercised": forced_chunking,
                "plain_peak_allocated_gib": pr["peak_allocated_gib"],
                "efficient_peak_allocated_gib": er["peak_allocated_gib"],
                "passed": passed,
            }
        )

        print(
            f"  map={map_err:.3g} score={score_err:.3g} E={E_err:.3g} | "
            f"cos={g['gradient_cosine']:.9f} "
            f"sign={g['gradient_sign_agreement']:.9f} | "
            f"chunks={c['total_query_chunks']} "
            f"split_modules={c['n_attention_modules_actually_split']} | "
            f"PASS={passed}"
        )

    df = pd.DataFrame(rows)
    results_path = OUT_ROOT / "stage17_equivalence.csv"
    df.to_csv(results_path, index=False)

    n_pass = int(df["passed"].sum())
    status = "PASS" if n_pass == len(df) == 3 else "FAIL"

    report = "\n".join(
        [
            "LABPC STAGE 17 — FORCED-CHUNK EQUIVALENCE + BACKEND AUDIT",
            "",
            f"STATUS: {status}",
            f"GPU: {torch.cuda.get_device_name(args.gpu)}",
            f"max_score_elems: {args.max_score_elems}",
            f"dncnn_segments: {args.dncnn_segments}",
            f"rows passed: {n_pass}/{len(df)}",
            "",
            "Backend state preserved from this project's TruFor loader:",
            json.dumps(backend_eff_after_wrap, indent=2, sort_keys=True),
            "",
            df[
                [
                    "variant",
                    "efficient_vs_plain_map_max_abs_error",
                    "efficient_vs_plain_score_abs_error",
                    "efficient_vs_plain_E_abs_error",
                    "gradient_cosine",
                    "gradient_sign_agreement",
                    "total_query_chunks",
                    "n_attention_modules_actually_split",
                    "max_chunks_in_one_attention",
                    "forced_chunking_exercised",
                    "passed",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.9g}"),
            "",
            "Acceptance gate:",
            f"  map max abs <= {ROUTE_MAP_ATOL}",
            f"  score abs <= {ROUTE_SCORE_ATOL}",
            f"  E abs <= {ROUTE_E_ATOL}",
            f"  gradient cosine >= {GRAD_COS_MIN}",
            f"  gradient sign agreement >= {GRAD_SIGN_MIN}",
            "  genuine query splitting REQUIRED",
            "",
            (
                "Proceed to Stage 18 largest-image gate."
                if status == "PASS"
                else
                "STOP. Do not run the largest-image gate."
            ),
        ]
    ) + "\n"

    report_path = OUT_ROOT / "stage17_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": status,
        "stage": "17_labpc_forced_chunk_equivalence",
        "route_info": route_info,
        "backend_state": backend_eff_after_wrap,
        "backend_policy": (
            "Preserve ForgedDocDeepID load_trufor_model CUDNN settings. "
            "TF32/matmul state observed and recorded but not changed."
        ),
        "selection": str(selection_path.relative_to(ROOT)),
        "selection_sha256": sha256_file(selection_path),
        "results": str(results_path.relative_to(ROOT)),
        "results_sha256": sha256_file(results_path),
        "max_score_elems": int(args.max_score_elems),
        "dncnn_segments": int(args.dncnn_segments),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    write_json(OUT_ROOT / "stage17_provenance.json", provenance)

    print("\n" + report)
    print(f"report: {report_path}")

    if status != "PASS":
        raise RuntimeError("Stage 17 FAIL")


if __name__ == "__main__":
    main()
