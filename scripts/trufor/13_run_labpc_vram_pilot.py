#!/usr/bin/env python3
"""LABPC Stage 13: one-backward-pass native TruFor VRAM/gradient pilot.

This is the exact same scientific gate as on HOME, but now against fresh LAB-generated
clean maps and LAB Stage-11 localisation metrics.

The script also writes a concise summary report automatically, so no extra stage is needed.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from trufor_common import ROOT, load_trufor_model, resolve_device, sha256_file, write_json
from trufor_labpc_attack_pilot_common import (
    ALPHA_PHYSICAL,
    EPSILON_PHYSICAL,
    MAP_PARITY_ATOL,
    METRIC_PARITY_ATOL,
    SCORE_PARITY_ATOL,
    PilotOOM,
    anomaly_from_logits,
    build_union_mask,
    canonical_inference_from_tensor,
    canonical_model_tensor_from_uint8,
    cuda_snapshot,
    differentiable_E,
    differentiable_forward,
    environment_record,
    freeze_all_model_parameters,
    is_cuda_oom,
    load_frozen_threshold,
    load_regions,
    load_rgb_uint8,
    localisation_values_np,
    physical_linf,
    pilot_root,
    project_one_step_minimise_E,
    reset_cuda_peaks,
    safe_run_tag,
)


def run_one(model, device: torch.device, row: pd.Series, regions: pd.DataFrame, threshold: float) -> Dict[str, object]:
    phase = "setup"
    started = time.time()
    image_path = str(row["image_path"])
    source = ROOT / str(row["cache_path"])
    map_path = ROOT / str(row["map_path"])
    try:
        phase = "load_input"
        rgb = load_rgb_uint8(source)
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if h != int(row["native_height"]) or w != int(row["native_width"]):
            raise RuntimeError(
                f"Native geometry changed for {image_path}: selection={row['native_width']}x{row['native_height']} runtime={w}x{h}"
            )
        mask_np, clipped_count = build_union_mask(regions, image_path, h, w)
        with np.load(map_path, allow_pickle=False) as cached:
            cached_map = np.asarray(cached["map"], dtype=np.float32)
            cached_score = float(np.asarray(cached["score"]).item())
            cached_hw = tuple(int(x) for x in np.asarray(cached["imgsize"]).tolist())
        if cached_hw != (h, w) or cached_map.shape != (h, w):
            raise RuntimeError(f"LAB Stage-09 NPZ/native geometry mismatch: {image_path}")

        clean_x = canonical_model_tensor_from_uint8(rgb, device)
        clean_x.requires_grad_(True)
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(device)

        phase = "gradient_forward"
        if device.type == "cuda":
            torch.cuda.empty_cache()
            reset_cuda_peaks(device)
        before_forward = cuda_snapshot(device)

        out, conf, det = differentiable_forward(model, clean_x)
        anomaly = anomaly_from_logits(out)
        score_t = torch.sigmoid(det).reshape(-1)
        E_t = differentiable_E(anomaly, mask_t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_peak = cuda_snapshot(device)

        clean_score = float(score_t.detach().cpu().item())
        grad_map_cpu = anomaly.detach().squeeze(0).float().cpu().numpy()
        map_error = float(np.max(np.abs(grad_map_cpu - cached_map)))
        score_error = abs(clean_score - cached_score)
        if map_error > MAP_PARITY_ATOL:
            raise RuntimeError(
                f"Gradient-forward map parity failed for {image_path}: {map_error:.9g} > {MAP_PARITY_ATOL}"
            )
        if score_error > SCORE_PARITY_ATOL:
            raise RuntimeError(
                f"Gradient-forward score parity failed for {image_path}: {score_error:.9g} > {SCORE_PARITY_ATOL}"
            )

        clean_metrics = localisation_values_np(grad_map_cpu, mask_np)
        metric_targets = {
            "A": float(row["stage11_A_union"]),
            "E": float(row["stage11_E_union"]),
            "mu_w": float(row["stage11_mu_union"]),
            "PG": float(row["stage11_PG_union"]),
        }
        metric_errors = {k: abs(clean_metrics[k] - metric_targets[k]) for k in metric_targets}
        for k, err in metric_errors.items():
            if err > METRIC_PARITY_ATOL:
                raise RuntimeError(
                    f"Stage-11 localisation parity failed ({k}) for {image_path}: error={err:.9g} > {METRIC_PARITY_ATOL}"
                )

        phase = "backward"
        if device.type == "cuda":
            reset_cuda_peaks(device)
        E_t.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_peak = cuda_snapshot(device)

        if clean_x.grad is None:
            raise RuntimeError("Input gradient is None")
        if not torch.isfinite(clean_x.grad).all():
            raise RuntimeError("Input gradient contains NaN/Inf")
        grad_abs_max = float(clean_x.grad.detach().abs().amax().cpu().item())
        grad_abs_mean = float(clean_x.grad.detach().abs().mean().cpu().item())
        if grad_abs_max == 0.0:
            raise RuntimeError("Input gradient is identically zero")

        phase = "project_one_step"
        candidate = project_one_step_minimise_E(clean_x.detach(), clean_x.grad.detach())
        linf = physical_linf(clean_x.detach(), candidate)
        if linf > EPSILON_PHYSICAL + 1e-7:
            raise RuntimeError(f"Realised physical L_inf exceeds epsilon: {linf} > {EPSILON_PHYSICAL}")

        clean_x.grad = None
        del E_t, anomaly, out, conf, det, score_t
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            reset_cuda_peaks(device)

        phase = "candidate_inference"
        adv_map_t, adv_score_t = canonical_inference_from_tensor(model, candidate)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_peak = cuda_snapshot(device)

        adv_score = float(adv_score_t.detach().cpu().item())
        adv_map = adv_map_t.detach().squeeze(0).float().cpu().numpy()
        adv_metrics = localisation_values_np(adv_map, mask_np)
        classification_preserved = bool(adv_score >= threshold)
        objective_nonincrease = bool(adv_metrics["E"] <= clean_metrics["E"] + 1e-8)
        accepted_under_future_constraints = classification_preserved and objective_nonincrease
        elapsed = time.time() - started

        result: Dict[str, object] = {
            "status": "PASS",
            "oom_phase": "",
            "pilot_order": int(row["pilot_order"]),
            "pilot_role": str(row["pilot_role"]),
            "eval_split": str(row["eval_split"]),
            "variant": str(row["variant"]),
            "hardware_source": str(row["hardware_source"]),
            "file_stem": str(row["file_stem"]),
            "image_path": image_path,
            "native_width": w,
            "native_height": h,
            "native_pixels": h * w,
            "frozen_threshold": threshold,
            "clean_score": clean_score,
            "clean_score_margin": clean_score - threshold,
            "clean_A": clean_metrics["A"],
            "clean_E": clean_metrics["E"],
            "clean_mu_w": clean_metrics["mu_w"],
            "clean_PG": clean_metrics["PG"],
            "gradient_map_parity_max_abs_error": map_error,
            "gradient_score_parity_abs_error": score_error,
            "stage11_A_abs_error": metric_errors["A"],
            "stage11_E_abs_error": metric_errors["E"],
            "stage11_mu_abs_error": metric_errors["mu_w"],
            "stage11_PG_abs_error": metric_errors["PG"],
            "input_grad_abs_max": grad_abs_max,
            "input_grad_abs_mean": grad_abs_mean,
            "epsilon_physical": EPSILON_PHYSICAL,
            "alpha_physical": ALPHA_PHYSICAL,
            "realised_linf_physical": linf,
            "candidate_score": adv_score,
            "candidate_score_margin": adv_score - threshold,
            "classification_preserved": classification_preserved,
            "candidate_E": adv_metrics["E"],
            "candidate_mu_w": adv_metrics["mu_w"],
            "candidate_PG": adv_metrics["PG"],
            "E_change_clean_minus_candidate": clean_metrics["E"] - adv_metrics["E"],
            "objective_nonincrease": objective_nonincrease,
            "accepted_under_future_constraints": accepted_under_future_constraints,
            "clipped_rectangles": clipped_count,
            "elapsed_seconds": elapsed,
        }
        for prefix, snap in [
            ("before_forward", before_forward),
            ("gradient_forward_peak", forward_peak),
            ("backward_peak", backward_peak),
            ("candidate_inference_peak", candidate_peak),
        ]:
            for key, value in snap.items():
                result[f"{prefix}_{key}"] = value
        result["autograd_peak_allocated_gib"] = max(
            float(forward_peak.get("cuda_peak_allocated_gib", 0.0)),
            float(backward_peak.get("cuda_peak_allocated_gib", 0.0)),
        )
        result["autograd_peak_reserved_gib"] = max(
            float(forward_peak.get("cuda_peak_reserved_gib", 0.0)),
            float(backward_peak.get("cuda_peak_reserved_gib", 0.0)),
        )
        del adv_map, adv_map_t, adv_score_t, candidate, clean_x, mask_t
        return result
    except Exception as exc:
        if is_cuda_oom(exc):
            raise PilotOOM(phase, exc) from None
        raise


def summarize_results(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.sort_values("pilot_order").iterrows():
        status = str(r.get("status", ""))
        rows.append({
            "order": int(r["pilot_order"]),
            "family": str(r["variant"]),
            "WxH": f"{int(r['native_width'])}x{int(r['native_height'])}",
            "MP": int(r["native_pixels"]) / 1e6,
            "status": status,
            "oom_phase": str(r.get("oom_phase", "")),
            "peak_reserved_GiB": float(r["autograd_peak_reserved_gib"]) if status == "PASS" and str(r.get("autograd_peak_reserved_gib", "")) != "" else np.nan,
            "peak_allocated_GiB": float(r["autograd_peak_allocated_gib"]) if status == "PASS" and str(r.get("autograd_peak_allocated_gib", "")) != "" else np.nan,
            "cls_preserved": str(r.get("classification_preserved", "")) if status == "PASS" else "",
            "E_drop": float(r["E_change_clean_minus_candidate"]) if status == "PASS" and str(r.get("E_change_clean_minus_candidate", "")) != "" else np.nan,
        })
    summary = pd.DataFrame(rows)
    pass_df = df.loc[df["status"].astype(str) == "PASS"].copy()
    oom_df = df.loc[df["status"].astype(str) == "OOM"].copy()
    lines = [
        "TRUFOR LABPC NATIVE VRAM / INPUT-GRADIENT PILOT SUMMARY",
        "",
        summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
        "",
        f"attempted: {len(df)}",
        f"PASS:      {len(pass_df)}",
        f"OOM:       {len(oom_df)}",
    ]
    if len(pass_df):
        max_reserved = pd.to_numeric(pass_df["autograd_peak_reserved_gib"], errors="coerce").max()
        max_alloc = pd.to_numeric(pass_df["autograd_peak_allocated_gib"], errors="coerce").max()
        lines += [
            f"max autograd peak reserved:  {max_reserved:.3f} GiB",
            f"max autograd peak allocated: {max_alloc:.3f} GiB",
            "max clean gradient-forward map parity error: "
            f"{pd.to_numeric(pass_df['gradient_map_parity_max_abs_error'], errors='coerce').max():.9g}",
            "max clean gradient-forward score parity error: "
            f"{pd.to_numeric(pass_df['gradient_score_parity_abs_error'], errors='coerce').max():.9g}",
        ]
    if len(oom_df):
        lines += [
            "",
            "MEMORY GATE:",
            "At least one legitimate frozen clean-correct image OOMed even on LABPC.",
            "Do NOT freeze a full multi-step native attack yet.",
        ]
    elif len(df) == 6 and len(pass_df) == 6:
        lines += [
            "",
            "MEMORY GATE:",
            "All six worst-case native-resolution images completed gradient forward + backward + one projected step on LABPC.",
            "This clears the machine-feasibility gate for designing the full multi-step attack.",
            "Still STOP HERE before freezing the optimiser hyperparameters.",
        ]
    else:
        lines += [
            "",
            "MEMORY GATE:",
            "The six-image gate is incomplete. Do not decide full-attack execution yet.",
        ]
    lines += [
        "",
        "STOP HERE.",
        "Review this summary before designing/running the full multi-step classification-preserving TruFor attack.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stop-on-oom", action="store_true")
    args = parser.parse_args()

    tag = safe_run_tag(args.run_tag)
    out_root = pilot_root(tag)
    selection_path = out_root / "pilot_selection.csv"
    config_path = out_root / "pilot_config.json"
    if not selection_path.is_file() or not config_path.is_file():
        raise RuntimeError("Run Stage 12 first")

    _, threshold = load_frozen_threshold()
    selection = pd.read_csv(selection_path, keep_default_na=False)
    if len(selection) != 6:
        raise RuntimeError(f"Expected 6 pilot rows, got {len(selection)}")
    config = json.loads(config_path.read_text())
    if config.get("pilot_selection_sha256") != sha256_file(selection_path):
        raise RuntimeError("Pilot selection changed after Stage 12")

    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("This is a CUDA VRAM pilot; use --gpu >= 0")

    print("LABPC STAGE 13 — ONE-BACKWARD-PASS VRAM/GRADIENT PILOT")
    print(f"run tag: {tag}")
    print(f"device: {device}")
    print(f"gpu: {torch.cuda.get_device_name(args.gpu)}")
    print(f"total VRAM: {torch.cuda.get_device_properties(device).total_memory / 2**30:.2f} GiB")
    print(f"frozen threshold: {threshold:.9f}")
    print(f"epsilon physical: {EPSILON_PHYSICAL:.9f} (=1/255)")
    print(f"alpha physical:   {ALPHA_PHYSICAL:.9f} (=0.25/255)")
    print("No resize/crop/re-JPEG. One backward pass + one projected step per image.")
    print()

    model, _, _ = load_trufor_model(device)
    freeze_all_model_parameters(model)
    regions = load_regions()

    results = []
    partial_path = out_root / "vram_pilot_results.partial.csv"
    final_path = out_root / "vram_pilot_results.csv"

    for _, row in selection.sort_values("pilot_order").iterrows():
        print(
            f"[{int(row['pilot_order'])}/6] {row['variant']} | {int(row['native_width'])}x{int(row['native_height'])} | "
            f"score={float(row['trufor_score']):.6f} | {row['image_path']}"
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            result = run_one(model, device, row, regions, threshold)
            results.append(result)
            print(
                "  PASS | "
                f"peak_reserved={result['autograd_peak_reserved_gib']:.2f} GiB | "
                f"peak_allocated={result['autograd_peak_allocated_gib']:.2f} GiB | "
                f"Linf={result['realised_linf_physical']:.7f} | "
                f"E {result['clean_E']:.5f}->{result['candidate_E']:.5f} | "
                f"score {result['clean_score']:.5f}->{result['candidate_score']:.5f} | "
                f"cls_preserved={result['classification_preserved']}"
            )
        except PilotOOM as exc:
            snap = cuda_snapshot(device)
            oom_row: Dict[str, object] = {
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
                "frozen_threshold": threshold,
                "clean_score": float(row["trufor_score"]),
            }
            for key, value in snap.items():
                oom_row[f"oom_{key}"] = value
            results.append(oom_row)
            print(f"  OOM during {exc.phase}")
            print(f"  {exc.original_message.splitlines()[0]}")
            gc.collect()
            torch.cuda.empty_cache()
            if args.stop_on_oom:
                pd.DataFrame(results).to_csv(partial_path, index=False)
                print("  --stop-on-oom requested; stopping after first OOM.")
                break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        pd.DataFrame(results).to_csv(partial_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(final_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    provenance = {
        "status": "PASS" if len(results_df) else "EMPTY",
        "stage": "13_run_labpc_vram_pilot",
        "run_tag": tag,
        "environment": environment_record(device),
        "frozen_threshold": threshold,
        "epsilon_physical_rgb01": EPSILON_PHYSICAL,
        "alpha_physical_rgb01": ALPHA_PHYSICAL,
        "n_requested": 6,
        "n_attempted": int(len(results_df)),
        "n_pass": int((results_df['status'] == 'PASS').sum()) if len(results_df) else 0,
        "n_oom": int((results_df['status'] == 'OOM').sum()) if len(results_df) else 0,
        "results": str(final_path.relative_to(ROOT)),
        "results_sha256": sha256_file(final_path),
        "scientific_note": "This stage tests native-resolution input-gradient/VRAM feasibility only. It does not constitute the final multi-step adversarial attack.",
    }
    write_json(out_root / "vram_pilot_provenance.json", provenance)

    report = summarize_results(results_df)
    report_path = out_root / "vram_pilot_summary.txt"
    report_path.write_text(report)

    print()
    print("STAGE 13 COMPLETE")
    print(f"results: {final_path}")
    print(f"summary: {report_path}")
    print(f"attempted={provenance['n_attempted']} PASS={provenance['n_pass']} OOM={provenance['n_oom']}")
    print()
    print(report)


if __name__ == "__main__":
    main()
