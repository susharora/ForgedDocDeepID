#!/usr/bin/env python3
"""LABPC Stage 14: validate CPU-first TruFor loading and clean-output parity.

This is a loader/allocator diagnostic only. It does NOT run an adversarial
backward pass and does NOT change any scientific protocol parameter.

Checks:
1. CUDA context supports a trivial allocation.
2. Frozen TruFor provenance still passes.
3. Checkpoint epoch is exactly 81.
4. Returned checkpoint state tensors remain on CPU.
5. Strictly-loaded model resides on requested GPU.
6. Native clean map/confidence/score reproduce the already-frozen LAB Stage-08
   direct-wrapper smoke result.

Run this once with the default allocator. If the default Stage-13 probe still
OOMs, run it again with PYTORCH_ALLOC_CONF=expandable_segments:True before
retrying Stage 13 under that allocator.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import socket
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from trufor_common import (
    ROOT,
    EXPECTED_CHECKPOINT_SHA256,
    infer_one,
    load_trufor_model,
    project_git_head,
    resolve_device,
    sha256_file,
    verify_trufor_provenance,
    write_json,
)

LAB_ROOT = ROOT / "output" / "LABPC" / "trufor_pretrained_policy_c_native"
STAGE08_PROVENANCE = LAB_ROOT / "stage08_provenance.json"
OUT_ROOT = ROOT / "output" / "LABPC" / "trufor_cpu_first_loader_validation"


def safe_tag(tag: str) -> str:
    tag = str(tag).strip()
    if not tag or not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise RuntimeError("run-tag must contain only letters, numbers, '.', '_' or '-'")
    return tag


def max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a - b)))


def cuda_mem(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "free_gib": free_b / 2**30,
        "total_gib": total_b / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--atol-map", type=float, default=2e-6)
    parser.add_argument("--atol-score", type=float, default=2e-6)
    args = parser.parse_args()

    tag = safe_tag(args.run_tag)
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if not STAGE08_PROVENANCE.is_file():
        raise RuntimeError(f"Missing Stage-08 provenance: {STAGE08_PROVENANCE}")
    stage08 = json.loads(STAGE08_PROVENANCE.read_text())
    if stage08.get("status") != "PASS":
        raise RuntimeError("LAB Stage-08 provenance is not PASS")
    if stage08.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Stage-08 checkpoint SHA does not match frozen checkpoint")

    smoke_path = ROOT / str(stage08["smoke_cache_path"])
    reference_npz = ROOT / str(stage08["direct_npz"])
    if not smoke_path.is_file():
        raise RuntimeError(f"Missing frozen smoke input: {smoke_path}")
    if not reference_npz.is_file():
        raise RuntimeError(f"Missing frozen Stage-08 direct-wrapper NPZ: {reference_npz}")

    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("This LAB memory validation requires CUDA")

    print("LABPC STAGE 14 — CPU-FIRST TRUFOR LOADER VALIDATION")
    print(f"run tag: {tag}")
    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF', '')!r}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}")

    # Diagnostic 1: CUDA context itself is alive.
    probe = torch.zeros(1, device=device)
    probe = probe + 1
    torch.cuda.synchronize(device)
    print(f"CUDA trivial allocation PASS: {float(probe.item()):.1f}")
    del probe
    torch.cuda.empty_cache()

    provenance = verify_trufor_provenance(check_archive_member=False)
    mem_before = cuda_mem(device)

    model, checkpoint, _ = load_trufor_model(device)
    torch.cuda.synchronize(device)
    mem_after_load = cuda_mem(device)

    epoch = checkpoint.get("epoch", None)
    if epoch is None or int(epoch) != 81:
        raise RuntimeError(f"Unexpected checkpoint epoch: {epoch!r}")

    state_tensor = next(
        (value for value in checkpoint["state_dict"].values() if torch.is_tensor(value)),
        None,
    )
    if state_tensor is None:
        raise RuntimeError("Checkpoint state_dict contains no tensors")
    if state_tensor.device.type != "cpu":
        raise RuntimeError(
            f"CPU-first invariant failed: checkpoint tensor is on {state_tensor.device}"
        )

    model_param = next(model.parameters())
    if model_param.device != device:
        raise RuntimeError(
            f"Model device invariant failed: expected {device}, got {model_param.device}"
        )

    with np.load(reference_npz, allow_pickle=False) as ref:
        ref_map = np.asarray(ref["map"], dtype=np.float32)
        ref_conf = np.asarray(ref["conf"], dtype=np.float32)
        ref_score = float(np.asarray(ref["score"]).item())
        ref_hw = tuple(int(x) for x in np.asarray(ref["imgsize"]).tolist())

    result = infer_one(model, smoke_path, device, include_conf=True)
    torch.cuda.synchronize(device)
    mem_after_inference = cuda_mem(device)

    err_map = max_abs(ref_map, result["map"])
    err_conf = max_abs(ref_conf, result["conf"])
    err_score = abs(ref_score - float(result["score"]))
    hw = tuple(int(x) for x in result["imgsize"])

    print("\nCPU-FIRST / CLEAN PARITY")
    print(f"  checkpoint epoch:            {int(epoch)}")
    print(f"  checkpoint tensor device:    {state_tensor.device}")
    print(f"  model parameter device:      {model_param.device}")
    print(f"  map max abs error:           {err_map:.9g}")
    print(f"  conf max abs error:          {err_conf:.9g}")
    print(f"  score abs error:             {err_score:.9g}")
    print(f"  native HxW:                  {hw}")

    if hw != ref_hw:
        raise RuntimeError(f"Native geometry parity failed: {hw} != {ref_hw}")
    if err_map > args.atol_map or err_conf > args.atol_map:
        raise RuntimeError(
            f"Clean map/conf parity failed: map={err_map}, conf={err_conf}, atol={args.atol_map}"
        )
    if err_score > args.atol_score:
        raise RuntimeError(
            f"Clean score parity failed: {err_score} > {args.atol_score}"
        )

    payload = {
        "status": "PASS",
        "stage": "14_labpc_validate_cpu_first_loader",
        "run_tag": tag,
        "allocator_env": {
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", ""),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "project_git_head": project_git_head(),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_epoch": int(epoch),
        "checkpoint_first_tensor_device": str(state_tensor.device),
        "model_first_parameter_device": str(model_param.device),
        "smoke_cache_path": str(stage08["smoke_cache_path"]),
        "reference_npz": str(stage08["direct_npz"]),
        "reference_npz_sha256": sha256_file(reference_npz),
        "native_hw": list(hw),
        "parity_max_abs_map": err_map,
        "parity_max_abs_conf": err_conf,
        "parity_abs_score": err_score,
        "parity_atol_map": args.atol_map,
        "parity_atol_score": args.atol_score,
        "cuda_memory_before_model_load": mem_before,
        "cuda_memory_after_model_load": mem_after_load,
        "cuda_memory_after_clean_inference": mem_after_inference,
        "scientific_note": (
            "Loader-only validation. Checkpoint bytes, architecture, preprocessing, "
            "clean inference arithmetic, threshold and attack protocol are unchanged."
        ),
    }
    provenance.update({"cpu_first_loader_validation": payload})
    out_json = out_dir / "stage14_cpu_first_loader_validation.json"
    write_json(out_json, payload)

    report = "\n".join(
        [
            "LABPC STAGE 14 — CPU-FIRST TRUFOR LOADER VALIDATION",
            "",
            f"status: PASS",
            f"run tag: {tag}",
            f"allocator: {payload['allocator_env']}",
            f"checkpoint epoch: {int(epoch)}",
            f"checkpoint tensor device: {state_tensor.device}",
            f"model parameter device: {model_param.device}",
            f"map max abs error: {err_map:.9g}",
            f"conf max abs error: {err_conf:.9g}",
            f"score abs error: {err_score:.9g}",
            f"native HxW: {hw}",
            "",
            "Interpretation: CPU-first loading is numerically clean relative to the frozen LAB Stage-08 wrapper output.",
        ]
    ) + "\n"
    out_txt = out_dir / "stage14_report.txt"
    out_txt.write_text(report)

    print("\nSTAGE 14 PASS")
    print(f"report: {out_txt}")
    print(f"json:   {out_json}")

    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
