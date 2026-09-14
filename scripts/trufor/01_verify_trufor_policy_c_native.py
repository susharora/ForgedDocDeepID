#!/usr/bin/env python3
"""Stage 1: freeze provenance and reproduce canonical upstream TruFor inference.

This stage does NOT clone, download, extract, modify, or patch TruFor.
It uses the existing external/TruFor checkout and final official checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch

from trufor_common import (
    OUT_ROOT,
    ROOT,
    TRUFOR_CHECKPOINT,
    TRUFOR_CODE,
    infer_one,
    load_policy_c_population,
    load_trufor_model,
    relative_to_root,
    resolve_device,
    sha256_file,
    verify_trufor_provenance,
    write_json,
)


def max_abs(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise RuntimeError(f"Parity shape mismatch: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU")
    parser.add_argument(
        "--atol-map",
        type=float,
        default=2e-6,
        help="absolute parity tolerance for anomaly/confidence maps",
    )
    parser.add_argument(
        "--atol-score",
        type=float,
        default=2e-6,
        help="absolute parity tolerance for detector score",
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("TRUFOR STAGE 1 — PROVENANCE + CANONICAL INFERENCE PARITY")
    print(f"project root: {ROOT}")

    provenance = verify_trufor_provenance(check_archive_member=True)
    population = load_policy_c_population()

    # Deterministic manipulated held-out example. The exact image is selected
    # from the frozen dev manifest rather than hard-coded to a filesystem guess.
    smoke = (
        population.loc[
            (population["eval_split"] == "dev_val")
            & (population["label"].astype(int) == 1)
        ]
        .sort_values(["variant", "image_path"], kind="stable")
        .iloc[0]
    )
    smoke_path = ROOT / str(smoke["cache_path"])
    if not smoke_path.is_file():
        raise RuntimeError(f"Smoke Policy-C image missing: {smoke_path}")

    expected_cache_sha = str(smoke["cache_sha256"])
    actual_cache_sha = sha256_file(smoke_path)
    if actual_cache_sha != expected_cache_sha:
        raise RuntimeError(
            "Frozen Policy-C smoke image SHA256 mismatch:\n"
            f"image    {smoke['image_path']}\n"
            f"expected {expected_cache_sha}\n"
            f"actual   {actual_cache_sha}"
        )

    smoke_dir = OUT_ROOT / "stage01_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    canonical_npz = smoke_dir / "canonical_upstream.npz"
    direct_npz = smoke_dir / "direct_wrapper.npz"
    for path in (canonical_npz, direct_npz):
        if path.exists():
            path.unlink()

    # First reproduce the unmodified upstream CLI itself.
    cmd = [
        sys.executable,
        "test.py",
        "-g",
        str(args.gpu),
        "-in",
        str(smoke_path),
        "-out",
        str(canonical_npz),
        "-exp",
        "trufor_ph3",
        "TEST.MODEL_FILE",
        str(TRUFOR_CHECKPOINT),
    ]

    print("\nRunning canonical upstream test.py on one frozen Policy-C dev attack...")
    print("command:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=TRUFOR_CODE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Canonical upstream test.py exited with code {result.returncode}"
        )
    if not canonical_npz.is_file():
        # Upstream test.py catches per-image exceptions, so absence of output is
        # a hard failure even if the subprocess exit status is zero.
        raise RuntimeError(
            "Canonical upstream test.py produced no NPZ. Inspect the output above."
        )

    canonical = np.load(canonical_npz, allow_pickle=False)
    expected_keys = {"map", "conf", "score", "imgsize"}
    missing = expected_keys - set(canonical.files)
    if missing:
        raise RuntimeError(f"Canonical NPZ missing keys: {sorted(missing)}")

    # Then reproduce those exact semantics through the reusable wrapper that
    # subsequent FantasyID inference will call.
    device = resolve_device(args.gpu)
    model, checkpoint, cfg = load_trufor_model(device)
    direct = infer_one(model, smoke_path, device, include_conf=True)

    np.savez_compressed(
        direct_npz,
        map=np.asarray(direct["map"], dtype=np.float32),
        conf=np.asarray(direct["conf"], dtype=np.float32),
        score=np.float32(direct["score"]),
        imgsize=np.asarray(direct["imgsize"], dtype=np.int32),
    )

    err_map = max_abs(canonical["map"], direct["map"])
    err_conf = max_abs(canonical["conf"], direct["conf"])
    err_score = abs(float(np.asarray(canonical["score"])) - float(direct["score"]))

    canonical_imgsize = tuple(int(x) for x in np.asarray(canonical["imgsize"]).tolist())
    direct_imgsize = tuple(int(x) for x in direct["imgsize"])
    if canonical_imgsize != direct_imgsize:
        raise RuntimeError(
            f"Native image-size parity failed: {canonical_imgsize} != {direct_imgsize}"
        )

    print("\nPARITY")
    print(f"  map max abs error:   {err_map:.9g}")
    print(f"  conf max abs error:  {err_conf:.9g}")
    print(f"  score abs error:     {err_score:.9g}")
    print(f"  native HxW:          {direct_imgsize}")

    if err_map > args.atol_map or err_conf > args.atol_map:
        raise RuntimeError(
            "Reusable TruFor wrapper does not reproduce canonical map output "
            f"within atol={args.atol_map}"
        )
    if err_score > args.atol_score:
        raise RuntimeError(
            "Reusable TruFor wrapper does not reproduce canonical detector score "
            f"within atol={args.atol_score}"
        )

    provenance.update(
        {
            "stage": "01_verify_trufor_policy_c_native",
            "smoke_image_path": str(smoke["image_path"]),
            "smoke_cache_path": str(smoke["cache_path"]),
            "smoke_cache_sha256": actual_cache_sha,
            "smoke_variant": str(smoke["variant"]),
            "smoke_hardware": str(smoke["hardware_source"]),
            "smoke_native_hw": list(direct_imgsize),
            "canonical_npz": relative_to_root(canonical_npz),
            "direct_npz": relative_to_root(direct_npz),
            "parity_max_abs_map": err_map,
            "parity_max_abs_conf": err_conf,
            "parity_abs_score": err_score,
            "parity_atol_map": args.atol_map,
            "parity_atol_score": args.atol_score,
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(args.gpu)
                if device.type == "cuda"
                else "cpu"
            ),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        }
    )

    out = OUT_ROOT / "stage01_provenance.json"
    write_json(out, provenance)

    print("\nSTAGE 1 PASS")
    print(f"provenance: {out}")
    print("Canonical native-resolution inference is reproduced.")
    print("Proceed to Stage 2 only after this PASS.")


if __name__ == "__main__":
    main()
