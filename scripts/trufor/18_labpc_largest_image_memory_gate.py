#!/usr/bin/env python3
"""Stage 18: corrected route on the single largest 6.78 MP pilot image.

This is deliberately ONE image only: the largest clean-correct attack selected
earlier. It is the next empirical decision boundary after forced-chunk
equivalence.

If it passes, the next bundle can run the remaining six-image feasibility gate
or freeze the attack design. If it OOMs, stop and review memory/chunk settings.

Authoritative candidate evaluation uses a separate unmodified TruFor instance.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trufor_common import (
    ROOT,
    load_trufor_model,
    resolve_device,
    sha256_file,
    write_json,
)
from trufor_labpc_attack_pilot_common import (
    ALPHA_MODEL,
    ALPHA_PHYSICAL,
    EPSILON_MODEL,
    EPSILON_PHYSICAL,
    MODEL_MAX,
    build_union_mask,
    canonical_model_tensor_from_uint8,
    load_frozen_threshold,
    load_regions,
    load_rgb_uint8,
    localisation_values_np,
    physical_linf,
)
from trufor_memory_efficient_attack_v2 import (
    anomaly_from_logits,
    apply_memory_efficient_route,
    assert_project_trufor_backend,
    backend_state,
    chunk_stats,
    efficient_differentiable_forward,
    freeze_all_parameters,
    localisation_E,
    require_default_allocator,
    score_from_det,
)

STAGE17_ROOT = ROOT / "output" / "LABPC" / "trufor_memory_efficient_validation_v2"
STAGE17_PROV = STAGE17_ROOT / "stage17_provenance.json"
SELECTION = (
    ROOT / "output" / "LABPC" / "trufor_vram_gradient_pilot"
    / "pilot_selection.csv"
)
OUT_ROOT = (
    ROOT / "output" / "LABPC"
    / "trufor_memory_efficient_largest_image_gate"
)

MAP_ATOL = 5e-6
SCORE_ATOL = 5e-6
E_ATOL = 5e-6


def official_reference(model, x):
    with torch.inference_mode():
        out, conf, det, _ = model(x, save_np=False)
        amap = anomaly_from_logits(out)
        score = score_from_det(det)
    return amap, score


def project_step(clean_x, grad):
    with torch.no_grad():
        candidate = clean_x - ALPHA_MODEL * grad.sign()
        lower = torch.clamp(
            clean_x - EPSILON_MODEL,
            min=0.0,
            max=MODEL_MAX,
        )
        upper = torch.clamp(
            clean_x + EPSILON_MODEL,
            min=0.0,
            max=MODEL_MAX,
        )
        candidate = torch.maximum(
            torch.minimum(candidate, upper),
            lower,
        )
        return torch.clamp(
            candidate,
            min=0.0,
            max=MODEL_MAX,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    require_default_allocator()

    if not STAGE17_PROV.is_file():
        raise RuntimeError("Run Stage 17 first")
    stage17 = json.loads(STAGE17_PROV.read_text())
    if stage17.get("status") != "PASS":
        raise RuntimeError("Stage 17 is not PASS")

    max_score_elems = int(stage17["max_score_elems"])
    dncnn_segments = int(stage17["dncnn_segments"])
    expected_backend = stage17["backend_state"]

    if not SELECTION.is_file():
        raise RuntimeError(f"Missing canonical Stage-12 selection: {SELECTION}")
    selection = pd.read_csv(SELECTION, keep_default_na=False)
    if len(selection) != 6:
        raise RuntimeError(f"Expected six selection rows, got {len(selection)}")

    row = selection.sort_values("pilot_order").iloc[0]
    if int(row["pilot_order"]) != 1:
        raise RuntimeError("Largest-image selection order changed")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("Stage 18 requires CUDA")

    _, threshold = load_frozen_threshold()

    print("LABPC STAGE 18 — LARGEST NATIVE IMAGE MEMORY GATE")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"image: {row['image_path']}")
    print(
        f"native: {int(row['native_width'])}x{int(row['native_height'])} "
        f"({int(row['native_pixels'])/1e6:.3f} MP)"
    )
    print(f"max_score_elems: {max_score_elems}")
    print(f"dncnn_segments: {dncnn_segments}")
    print("allocator: default")
    print()

    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    print("CUDA trivial allocation PASS:", probe)
    del probe

    print("\nLoading unmodified authoritative reference model...")
    ref_model, ref_ckpt, ref_cfg = load_trufor_model(device)
    freeze_all_parameters(ref_model)
    backend_ref = assert_project_trufor_backend(ref_cfg)

    print("Loading corrected memory-efficient attack model...")
    atk_model, atk_ckpt, atk_cfg = load_trufor_model(device)
    freeze_all_parameters(atk_model)
    backend_atk_pre = assert_project_trufor_backend(atk_cfg)

    route_info = apply_memory_efficient_route(
        atk_model,
        max_score_elems=max_score_elems,
        dncnn_segments=dncnn_segments,
    )
    backend_atk_post = assert_project_trufor_backend(atk_cfg)

    if backend_ref != expected_backend:
        raise RuntimeError(
            "Reference backend differs from Stage-17 frozen observed state"
        )
    if backend_atk_pre != expected_backend:
        raise RuntimeError(
            "Attack backend differs from Stage-17 frozen observed state"
        )
    if backend_atk_post != expected_backend:
        raise RuntimeError(
            "Memory wrapper changed backend state"
        )

    print("route_info:", json.dumps(route_info, sort_keys=True))
    print("backend:", json.dumps(backend_atk_post, sort_keys=True))

    regions = load_regions()

    rgb = load_rgb_uint8(ROOT / str(row["cache_path"]))
    h, w = rgb.shape[:2]
    mask_np, clipped = build_union_mask(
        regions,
        str(row["image_path"]),
        h,
        w,
    )
    mask = (
        torch.from_numpy(mask_np.astype(np.float32))
        .unsqueeze(0)
        .to(device)
    )

    with np.load(ROOT / str(row["map_path"]), allow_pickle=False) as data:
        cached_map = np.asarray(data["map"], dtype=np.float32)
        cached_score = float(np.asarray(data["score"]).item())

    x = canonical_model_tensor_from_uint8(rgb, device)
    x.requires_grad_(True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    try:
        print("\nRunning memory-efficient differentiable forward...")
        out, conf, det = efficient_differentiable_forward(atk_model, x)
        amap_t = anomaly_from_logits(out)
        score_t = score_from_det(det)
        E_t = localisation_E(amap_t, mask).reshape(())
        torch.cuda.synchronize(device)

        attack_map = amap_t.detach().squeeze(0).float().cpu().numpy()
        attack_score = float(score_t.detach().cpu().item())
        attack_E = float(E_t.detach().cpu().item())

        cached_metrics = localisation_values_np(cached_map, mask_np)
        map_err = float(np.max(np.abs(attack_map - cached_map)))
        score_err = abs(attack_score - cached_score)
        E_err = abs(attack_E - float(cached_metrics["E"]))

        chunks = chunk_stats(atk_model)
        if chunks["n_attention_modules_actually_split"] <= 0:
            raise RuntimeError("Largest image did not exercise query splitting")

        if map_err > MAP_ATOL:
            raise RuntimeError(f"clean map parity failed: {map_err}")
        if score_err > SCORE_ATOL:
            raise RuntimeError(f"clean score parity failed: {score_err}")
        if E_err > E_ATOL:
            raise RuntimeError(f"clean E parity failed: {E_err}")

        print(
            f"  clean parity map={map_err:.3g} "
            f"score={score_err:.3g} E={E_err:.3g}"
        )
        print(
            f"  attention chunks={chunks['total_query_chunks']} "
            f"split_modules={chunks['n_attention_modules_actually_split']} "
            f"max_chunks={chunks['max_chunks_in_one_attention']}"
        )

        print("Running first-order input-gradient backward...")
        grad, = torch.autograd.grad(
            E_t,
            x,
            retain_graph=False,
            create_graph=False,
        )
        torch.cuda.synchronize(device)

        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite input gradient")
        if float(grad.abs().max().item()) == 0.0:
            raise RuntimeError("Input gradient is identically zero")

        peak_alloc = torch.cuda.max_memory_allocated(device) / 2**30
        peak_res = torch.cuda.max_memory_reserved(device) / 2**30

        candidate = project_step(x.detach(), grad.detach())
        linf = physical_linf(x.detach(), candidate)
        if linf > EPSILON_PHYSICAL + 1e-7:
            raise RuntimeError(f"Physical Linf violation: {linf}")

        del grad, E_t, amap_t, score_t, out, conf, det
        gc.collect()
        torch.cuda.empty_cache()

        print("Running authoritative candidate evaluation...")
        adv_map_t, adv_score_t = official_reference(ref_model, candidate)
        torch.cuda.synchronize(device)

        adv_map = adv_map_t.detach().squeeze(0).float().cpu().numpy()
        adv_score = float(adv_score_t.detach().cpu().item())
        adv_metrics = localisation_values_np(adv_map, mask_np)
        cls_preserved = bool(adv_score >= threshold)

        status = "PASS"
        failure = ""

        result = {
            "status": status,
            "failure": failure,
            "pilot_order": 1,
            "image_path": str(row["image_path"]),
            "variant": str(row["variant"]),
            "native_width": w,
            "native_height": h,
            "native_pixels": h * w,
            "max_score_elems": max_score_elems,
            "dncnn_segments": dncnn_segments,
            "clean_map_parity_max_abs_error": map_err,
            "clean_score_parity_abs_error": score_err,
            "clean_E_parity_abs_error": E_err,
            "attention_modules_observed": chunks["modules_observed"],
            "total_query_chunks": chunks["total_query_chunks"],
            "n_attention_modules_actually_split":
                chunks["n_attention_modules_actually_split"],
            "max_chunks_in_one_attention":
                chunks["max_chunks_in_one_attention"],
            "gradient_abs_max": float(x.grad.abs().max().item())
                if x.grad is not None else None,
            "realised_linf_physical": linf,
            "clean_score": cached_score,
            "candidate_score": adv_score,
            "classification_preserved": cls_preserved,
            "clean_E": float(cached_metrics["E"]),
            "candidate_E": float(adv_metrics["E"]),
            "E_drop_clean_minus_candidate":
                float(cached_metrics["E"] - adv_metrics["E"]),
            "peak_allocated_gib_reported": peak_alloc,
            "peak_reserved_gib_reported": peak_res,
            "elapsed_seconds": time.time() - t0,
            "clipped_rectangles": int(clipped),
        }

    except torch.cuda.OutOfMemoryError as exc:
        status = "OOM"
        failure = str(exc)
        result = {
            "status": status,
            "failure": failure,
            "pilot_order": 1,
            "image_path": str(row["image_path"]),
            "variant": str(row["variant"]),
            "native_width": w,
            "native_height": h,
            "native_pixels": h * w,
            "max_score_elems": max_score_elems,
            "dncnn_segments": dncnn_segments,
            "elapsed_seconds": time.time() - t0,
        }
        torch.cuda.empty_cache()

    result_path = OUT_ROOT / "stage18_result.json"
    write_json(result_path, result)

    report_lines = [
        "LABPC STAGE 18 — LARGEST-IMAGE MEMORY GATE",
        "",
        f"STATUS: {status}",
        f"image: {row['image_path']}",
        f"native: {w}x{h} ({h*w/1e6:.3f} MP)",
        f"max_score_elems: {max_score_elems}",
        f"dncnn_segments: {dncnn_segments}",
    ]

    if status == "PASS":
        report_lines += [
            f"clean map parity max abs: {result['clean_map_parity_max_abs_error']:.9g}",
            f"clean score parity abs:   {result['clean_score_parity_abs_error']:.9g}",
            f"clean E parity abs:       {result['clean_E_parity_abs_error']:.9g}",
            f"total query chunks:       {result['total_query_chunks']}",
            f"split attention modules:  {result['n_attention_modules_actually_split']}",
            f"max chunks/attention:     {result['max_chunks_in_one_attention']}",
            f"realised Linf physical:   {result['realised_linf_physical']:.9g}",
            f"E: {result['clean_E']:.6f} -> {result['candidate_E']:.6f}",
            f"score: {result['clean_score']:.6f} -> {result['candidate_score']:.6f}",
            f"classification preserved: {result['classification_preserved']}",
            "",
            "MEMORY GATE: PASS",
            "The largest native image completed forward + first-order backward",
            "+ one projected step using the corrected forced-chunk route.",
            "",
            "STOP HERE.",
            "Do not run the full six-image/full multi-step attack yet.",
        ]
    else:
        report_lines += [
            "",
            "MEMORY GATE: OOM",
            failure.splitlines()[0] if failure else "",
            "",
            "STOP HERE for review.",
        ]

    report = "\n".join(report_lines) + "\n"
    report_path = OUT_ROOT / "stage18_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": status,
        "stage": "18_labpc_largest_image_memory_gate",
        "stage17_provenance_sha256": sha256_file(STAGE17_PROV),
        "backend_state": backend_atk_post,
        "route_info": route_info,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256_file(result_path),
    }
    write_json(OUT_ROOT / "stage18_provenance.json", provenance)

    print("\n" + report)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
