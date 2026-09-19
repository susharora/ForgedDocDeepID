#!/usr/bin/env python3
"""Stage 16: six-largest-image native TruFor VRAM pilot with exact recomputation.

Attack-gradient instance:
- full native image
- CPU-first loaded model, frozen weights
- DnCNN activation checkpointing (4 segments)
- all 32 SegFormer MLPs checkpointed
- all 32 self-attention modules use exact query-row chunking

Reference instance:
- separate unmodified TruFor model
- no-grad authoritative candidate evaluation

This remains a ONE-STEP feasibility pilot. It does not freeze the final
multi-step optimiser.
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
from trufor_memory_efficient_attack import (
    DEFAULT_DNCNN_SEGMENTS,
    DEFAULT_MAX_SCORE_ELEMS,
    anomaly_from_logits,
    attention_chunk_stats,
    configure_reproducible_float32,
    enable_memory_efficient_attack_route,
    freeze_all_parameters,
    localisation_E,
    memory_efficient_differentiable_forward,
    require_default_allocator,
    score_from_det,
)

STAGE15_ROOT = ROOT / "output" / "LABPC" / "trufor_memory_efficient_validation"
STAGE15_PROV = STAGE15_ROOT / "stage15_provenance.json"
SELECTION = ROOT / "output" / "LABPC" / "trufor_vram_gradient_pilot" / "pilot_selection.csv"
OUT_ROOT = ROOT / "output" / "LABPC" / "trufor_memory_efficient_vram_pilot"

MAP_PARITY_ATOL = 2e-5
SCORE_PARITY_ATOL = 2e-5
E_PARITY_ATOL = 2e-5


class PilotOOM(RuntimeError):
    def __init__(self, phase, original):
        self.phase = phase
        self.original_message = str(original)
        super().__init__(f"CUDA OOM during {phase}: {original}")


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def cuda_snapshot(device):
    free_b, total_b = torch.cuda.mem_get_info(device)
    alloc = torch.cuda.memory_allocated(device)
    reserv = torch.cuda.memory_reserved(device)
    peak_a = torch.cuda.max_memory_allocated(device)
    peak_r = torch.cuda.max_memory_reserved(device)
    return {
        "free_gib": free_b / 2**30,
        "total_gib": total_b / 2**30,
        "allocated_gib": alloc / 2**30,
        "reserved_gib": reserv / 2**30,
        "peak_allocated_gib": peak_a / 2**30,
        "peak_reserved_gib": peak_r / 2**30,
        "peak_accounting_plausible": bool(max(peak_a, peak_r) <= total_b * 1.10),
    }


def official_reference_inference(model, x):
    with torch.inference_mode():
        out, conf, det, _ = model(x, save_np=False)
        amap = anomaly_from_logits(out)
        score = score_from_det(det)
    return amap, score


def project_step(clean_x, grad_x):
    with torch.no_grad():
        candidate = clean_x - ALPHA_MODEL * grad_x.sign()
        lower = torch.clamp(clean_x - EPSILON_MODEL, min=0.0, max=MODEL_MAX)
        upper = torch.clamp(clean_x + EPSILON_MODEL, min=0.0, max=MODEL_MAX)
        candidate = torch.maximum(torch.minimum(candidate, upper), lower)
        return torch.clamp(candidate, min=0.0, max=MODEL_MAX)


def validate_stage15():
    if not STAGE15_PROV.is_file():
        raise RuntimeError(f"Missing Stage-15 provenance: {STAGE15_PROV}")
    payload = json.loads(STAGE15_PROV.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError("Stage 15 is not PASS")
    return payload


def load_cached(row):
    with np.load(ROOT / str(row["map_path"]), allow_pickle=False) as data:
        return (
            np.asarray(data["map"], dtype=np.float32),
            float(np.asarray(data["score"]).item()),
        )


def run_one(atk_model, ref_model, row, regions, device, threshold):
    phase = "load_input"
    started = time.time()
    try:
        rgb = load_rgb_uint8(ROOT / str(row["cache_path"]))
        h, w = rgb.shape[:2]
        mask_np, clipped = build_union_mask(regions, str(row["image_path"]), h, w)
        cached_map, cached_score = load_cached(row)

        clean_x = canonical_model_tensor_from_uint8(rgb, device)
        clean_x.requires_grad_(True)
        mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(device)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = cuda_snapshot(device)

        phase = "memory_efficient_gradient_forward"
        out, conf, det = memory_efficient_differentiable_forward(atk_model, clean_x)
        anomaly = anomaly_from_logits(out)
        score_t = score_from_det(det)
        E_t = localisation_E(anomaly, mask).reshape(())
        torch.cuda.synchronize(device)
        forward_mem = cuda_snapshot(device)
        chunk_info = attention_chunk_stats(atk_model)

        attack_map = anomaly.detach().squeeze(0).float().cpu().numpy()
        attack_score = float(score_t.detach().cpu().item())
        attack_E = float(E_t.detach().cpu().item())

        map_err = float(np.max(np.abs(attack_map - cached_map)))
        score_err = abs(attack_score - cached_score)
        cached_metrics = localisation_values_np(cached_map, mask_np)
        E_err = abs(attack_E - float(cached_metrics["E"]))

        if map_err > MAP_PARITY_ATOL:
            raise RuntimeError(
                f"Memory-efficient clean-map parity failed: {map_err} > {MAP_PARITY_ATOL}"
            )
        if score_err > SCORE_PARITY_ATOL:
            raise RuntimeError(
                f"Memory-efficient clean-score parity failed: {score_err} > {SCORE_PARITY_ATOL}"
            )
        if E_err > E_PARITY_ATOL:
            raise RuntimeError(
                f"Memory-efficient clean-E parity failed: {E_err} > {E_PARITY_ATOL}"
            )

        phase = "backward"
        grad, = torch.autograd.grad(
            E_t,
            clean_x,
            retain_graph=False,
            create_graph=False,
        )
        torch.cuda.synchronize(device)
        backward_mem = cuda_snapshot(device)

        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite input gradient")
        grad_abs_max = float(grad.detach().abs().amax().cpu().item())
        grad_abs_mean = float(grad.detach().abs().mean().cpu().item())
        if grad_abs_max == 0:
            raise RuntimeError("Input gradient is identically zero")

        phase = "project_one_step"
        candidate = project_step(clean_x.detach(), grad.detach())
        linf = physical_linf(clean_x.detach(), candidate)
        if linf > EPSILON_PHYSICAL + 1e-7:
            raise RuntimeError(f"Physical L_inf exceeded: {linf}")

        # Destroy attack graph before authoritative reference evaluation.
        del grad, E_t, anomaly, score_t, out, conf, det, mask
        clean_x.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        phase = "official_reference_candidate_eval"
        adv_map_t, adv_score_t = official_reference_inference(ref_model, candidate)
        torch.cuda.synchronize(device)
        ref_mem = cuda_snapshot(device)

        adv_map = adv_map_t.detach().squeeze(0).float().cpu().numpy()
        adv_score = float(adv_score_t.detach().cpu().item())
        adv_metrics = localisation_values_np(adv_map, mask_np)
        cls_preserved = bool(adv_score >= threshold)

        elapsed = time.time() - started
        return {
            "status": "PASS",
            "oom_phase": "",
            "pilot_order": int(row["pilot_order"]),
            "pilot_role": str(row["pilot_role"]),
            "eval_split": str(row["eval_split"]),
            "variant": str(row["variant"]),
            "hardware_source": str(row["hardware_source"]),
            "file_stem": str(row["file_stem"]),
            "image_path": str(row["image_path"]),
            "native_width": w,
            "native_height": h,
            "native_pixels": h * w,
            "frozen_threshold": threshold,
            "clean_score_cached": cached_score,
            "clean_score_attack_route": attack_score,
            "clean_score_parity_abs_error": score_err,
            "clean_map_parity_max_abs_error": map_err,
            "clean_E_cached": float(cached_metrics["E"]),
            "clean_E_attack_route": attack_E,
            "clean_E_parity_abs_error": E_err,
            "clean_mu_w": float(cached_metrics["mu_w"]),
            "clean_PG": float(cached_metrics["PG"]),
            "input_grad_abs_max": grad_abs_max,
            "input_grad_abs_mean": grad_abs_mean,
            "epsilon_physical": EPSILON_PHYSICAL,
            "alpha_physical": ALPHA_PHYSICAL,
            "realised_linf_physical": linf,
            "candidate_score_reference": adv_score,
            "candidate_score_margin": adv_score - threshold,
            "classification_preserved": cls_preserved,
            "candidate_E_reference": float(adv_metrics["E"]),
            "candidate_mu_w_reference": float(adv_metrics["mu_w"]),
            "candidate_PG_reference": float(adv_metrics["PG"]),
            "E_drop_clean_minus_candidate": float(cached_metrics["E"] - adv_metrics["E"]),
            "attention_modules_observed": chunk_info["modules_observed"],
            "total_query_chunks": chunk_info["total_query_chunks"],
            "max_chunks_in_one_attention": chunk_info["max_chunks_in_one_attention"],
            "max_score_elems_actual": chunk_info["max_score_elems_actual"],
            "clipped_rectangles": int(clipped),
            "elapsed_seconds": elapsed,
            **{f"before_{k}": v for k, v in before.items()},
            **{f"forward_{k}": v for k, v in forward_mem.items()},
            **{f"backward_{k}": v for k, v in backward_mem.items()},
            **{f"reference_{k}": v for k, v in ref_mem.items()},
        }
    except Exception as exc:
        if is_oom(exc):
            raise PilotOOM(phase, exc) from None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-score-elems", type=int, default=DEFAULT_MAX_SCORE_ELEMS)
    parser.add_argument("--dncnn-segments", type=int, default=DEFAULT_DNCNN_SEGMENTS)
    parser.add_argument("--stop-on-oom", action="store_true")
    args = parser.parse_args()

    require_default_allocator()
    configure_reproducible_float32()
    stage15 = validate_stage15()
    _, threshold = load_frozen_threshold()

    if not SELECTION.is_file():
        raise RuntimeError(
            f"Missing canonical six-image Stage-12 selection: {SELECTION}"
        )
    selection = pd.read_csv(SELECTION, keep_default_na=False)
    if len(selection) != 6:
        raise RuntimeError(f"Expected six pilot images, got {len(selection)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selection_copy = OUT_ROOT / "stage16_pilot_selection.csv"
    selection.to_csv(selection_copy, index=False)

    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("Stage 16 requires CUDA")

    print("LABPC STAGE 16 — CHECKPOINTED / QUERY-CHUNKED NATIVE VRAM PILOT")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"total VRAM: {torch.cuda.get_device_properties(device).total_memory / 2**30:.2f} GiB")
    print("allocator: default")
    print(f"max_score_elems: {args.max_score_elems}")
    print(f"dncnn_segments: {args.dncnn_segments}")
    print(f"frozen threshold: {threshold:.9f}")
    print("full native images; no tiling/resize/crop/re-JPEG")
    print()

    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    print("CUDA trivial allocation PASS:", probe)
    del probe

    print("\nLoading separate unmodified reference TruFor...")
    ref_model, ref_ckpt, _ = load_trufor_model(device)
    freeze_all_parameters(ref_model)

    print("Loading memory-efficient attack TruFor...")
    atk_model, atk_ckpt, _ = load_trufor_model(device)
    route_info = enable_memory_efficient_attack_route(
        atk_model,
        max_score_elems=args.max_score_elems,
        dncnn_segments=args.dncnn_segments,
    )
    print("attack route:", json.dumps(route_info, sort_keys=True))

    regions = load_regions()
    rows = []
    partial = OUT_ROOT / "stage16_results.partial.csv"

    for _, row in selection.sort_values("pilot_order").iterrows():
        print(
            f"\n[{int(row['pilot_order'])}/6] {row['variant']} | "
            f"{int(row['native_width'])}x{int(row['native_height'])} | "
            f"{row['image_path']}"
        )
        gc.collect()
        torch.cuda.empty_cache()
        try:
            result = run_one(
                atk_model,
                ref_model,
                row,
                regions,
                device,
                threshold,
            )
            rows.append(result)
            print(
                "  PASS | "
                f"peak_reserved(backward)={result['backward_peak_reserved_gib']:.2f} GiB | "
                f"peak_allocated(backward)={result['backward_peak_allocated_gib']:.2f} GiB | "
                f"chunks={result['total_query_chunks']} | "
                f"E {result['clean_E_cached']:.5f}->{result['candidate_E_reference']:.5f} | "
                f"score {result['clean_score_cached']:.5f}->{result['candidate_score_reference']:.5f} | "
                f"cls_preserved={result['classification_preserved']}"
            )
        except PilotOOM as exc:
            snap = cuda_snapshot(device)
            rows.append(
                {
                    "status": "OOM",
                    "oom_phase": exc.phase,
                    "oom_message": exc.original_message,
                    "pilot_order": int(row["pilot_order"]),
                    "pilot_role": str(row["pilot_role"]),
                    "eval_split": str(row["eval_split"]),
                    "variant": str(row["variant"]),
                    "hardware_source": str(row["hardware_source"]),
                    "file_stem": str(row["file_stem"]),
                    "image_path": str(row["image_path"]),
                    "native_width": int(row["native_width"]),
                    "native_height": int(row["native_height"]),
                    "native_pixels": int(row["native_pixels"]),
                    **{f"oom_{k}": v for k, v in snap.items()},
                }
            )
            print(f"  OOM during {exc.phase}")
            print(" ", exc.original_message.splitlines()[0])
            gc.collect()
            torch.cuda.empty_cache()
            if args.stop_on_oom:
                pd.DataFrame(rows).to_csv(partial, index=False)
                print("  --stop-on-oom requested; stopping.")
                break
        pd.DataFrame(rows).to_csv(partial, index=False)

    df = pd.DataFrame(rows)
    final = OUT_ROOT / "stage16_results.csv"
    df.to_csv(final, index=False)
    if partial.exists():
        partial.unlink()

    n_pass = int((df["status"].astype(str) == "PASS").sum()) if len(df) else 0
    n_oom = int((df["status"].astype(str) == "OOM").sum()) if len(df) else 0

    columns = [
        "pilot_order", "variant", "native_width", "native_height",
        "status", "oom_phase",
    ]
    if n_pass:
        for c in [
            "backward_peak_allocated_gib",
            "backward_peak_reserved_gib",
            "total_query_chunks",
            "classification_preserved",
            "E_drop_clean_minus_candidate",
        ]:
            if c in df.columns:
                columns.append(c)

    report_lines = [
        "LABPC STAGE 16 — MEMORY-EFFICIENT NATIVE TRUFOR VRAM PILOT",
        "",
        f"attempted: {len(df)}",
        f"PASS: {n_pass}",
        f"OOM: {n_oom}",
        "",
        df[columns].to_string(index=False, float_format=lambda x: f"{x:.6g}")
        if len(df) else "(no rows)",
        "",
    ]

    if len(df) == 6 and n_pass == 6:
        report_lines += [
            "MEMORY GATE: PASS",
            "All six worst-case native images completed full differentiable",
            "forward + first-order input-gradient backward + one projected step.",
            "",
            "STOP HERE.",
            "Do not freeze the multi-step optimiser in this chat. Start the",
            "dedicated adversarial-attack chat with Stage-15/16 evidence.",
        ]
        status = "PASS"
    elif n_oom:
        report_lines += [
            "MEMORY GATE: FAIL / NEEDS REVIEW",
            "At least one worst-case image still OOMed with exact query chunking",
            "and activation recomputation. Do not start the full attack.",
        ]
        status = "FAIL"
    else:
        report_lines += [
            "MEMORY GATE: INCOMPLETE",
            "The six-image gate did not complete.",
        ]
        status = "INCOMPLETE"

    report = "\n".join(report_lines) + "\n"
    report_path = OUT_ROOT / "stage16_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": status,
        "stage": "16_labpc_checkpointed_query_chunked_vram_pilot",
        "stage15_provenance_sha256": sha256_file(STAGE15_PROV),
        "selection_sha256": sha256_file(selection_copy),
        "route_info": route_info,
        "reference_route": "official_unmodified_no_grad",
        "gpu": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "frozen_threshold": threshold,
        "epsilon_physical": EPSILON_PHYSICAL,
        "alpha_physical": ALPHA_PHYSICAL,
        "n_attempted": int(len(df)),
        "n_pass": n_pass,
        "n_oom": n_oom,
        "results": str(final.relative_to(ROOT)),
        "results_sha256": sha256_file(final),
        "scientific_contract": [
            "Attack route sees the complete native image.",
            "No input tiling/resizing/cropping/re-JPEG.",
            "Only query rows inside self-attention are chunked.",
            "Activation checkpointing/recomputation changes memory graph only.",
            "All TruFor weights remain frozen.",
            "create_graph=False and retain_graph=False.",
            "Separate unmodified TruFor instance is authoritative for candidate evaluation.",
        ],
    }
    write_json(OUT_ROOT / "stage16_provenance.json", provenance)

    print("\n" + report)
    print(f"results: {final}")
    print(f"report:  {report_path}")


if __name__ == "__main__":
    main()
