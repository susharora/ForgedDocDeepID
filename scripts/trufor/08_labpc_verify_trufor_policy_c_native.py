#!/usr/bin/env python3
"""LABPC Stage 08: fresh TruFor provenance + canonical parity.

This is a LAB-specific reproduction of the already-frozen clean pipeline.
It does NOT alter any shared/Home output. All generated artifacts go under:

    output/LABPC/trufor_pretrained_policy_c_native/

Inputs remain the exact shared/frozen Policy-C JPEGs, manifests, TruFor source,
and official checkpoint.

No threshold fitting occurs here.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch

from trufor_common import (
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

RUN_TAG = "LABPC"
LAB_STAGE2_ROOT = ROOT / "output" / RUN_TAG / "trufor_pretrained_policy_c_native"


def max_abs(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise RuntimeError(f"Parity shape mismatch: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--atol-map", type=float, default=2e-6)
    parser.add_argument("--atol-score", type=float, default=2e-6)
    args = parser.parse_args()

    LAB_STAGE2_ROOT.mkdir(parents=True, exist_ok=True)

    print("LABPC STAGE 08 — TRUFOR PROVENANCE + CANONICAL PARITY")
    print(f"project root: {ROOT}")
    print(f"output root:  {LAB_STAGE2_ROOT}")

    provenance = verify_trufor_provenance(check_archive_member=True)
    population = load_policy_c_population()

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
            "Frozen Policy-C smoke SHA mismatch:\n"
            f"image    {smoke['image_path']}\n"
            f"expected {expected_cache_sha}\n"
            f"actual   {actual_cache_sha}"
        )

    smoke_dir = LAB_STAGE2_ROOT / "stage08_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    canonical_npz = smoke_dir / "canonical_upstream.npz"
    direct_npz = smoke_dir / "direct_wrapper.npz"
    for p in (canonical_npz, direct_npz):
        if p.exists():
            p.unlink()

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

    print("\nRunning unmodified upstream test.py on one frozen Policy-C dev attack...")
    print("command:", " ".join(cmd))

    # Compatibility only: upstream test.py calls torch.load without explicitly
    # setting weights_only. This does not alter model/data/preprocessing.
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    result = subprocess.run(
        cmd,
        cwd=TRUFOR_CODE,
        env=env,
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
        raise RuntimeError("Canonical upstream test.py produced no NPZ")

    with np.load(canonical_npz, allow_pickle=False) as canonical:
        required = {"map", "conf", "score", "imgsize"}
        missing = required - set(canonical.files)
        if missing:
            raise RuntimeError(f"Canonical NPZ missing keys: {sorted(missing)}")
        canonical_map = np.asarray(canonical["map"], dtype=np.float32)
        canonical_conf = np.asarray(canonical["conf"], dtype=np.float32)
        canonical_score = float(np.asarray(canonical["score"]).item())
        canonical_hw = tuple(
            int(x) for x in np.asarray(canonical["imgsize"]).tolist()
        )

    device = resolve_device(args.gpu)
    model, checkpoint, _ = load_trufor_model(device)
    direct = infer_one(model, smoke_path, device, include_conf=True)

    np.savez_compressed(
        direct_npz,
        map=np.asarray(direct["map"], dtype=np.float32),
        conf=np.asarray(direct["conf"], dtype=np.float32),
        score=np.float32(direct["score"]),
        imgsize=np.asarray(direct["imgsize"], dtype=np.int32),
    )

    err_map = max_abs(canonical_map, direct["map"])
    err_conf = max_abs(canonical_conf, direct["conf"])
    err_score = abs(canonical_score - float(direct["score"]))
    direct_hw = tuple(int(x) for x in direct["imgsize"])

    if canonical_hw != direct_hw:
        raise RuntimeError(
            f"Native geometry parity failed: {canonical_hw} != {direct_hw}"
        )

    print("\nPARITY")
    print(f"  map max abs error:   {err_map:.9g}")
    print(f"  conf max abs error:  {err_conf:.9g}")
    print(f"  score abs error:     {err_score:.9g}")
    print(f"  native HxW:          {direct_hw}")

    if err_map > args.atol_map or err_conf > args.atol_map:
        raise RuntimeError(
            "Wrapper does not reproduce canonical map/conf within "
            f"atol={args.atol_map}"
        )
    if err_score > args.atol_score:
        raise RuntimeError(
            "Wrapper does not reproduce canonical detector score within "
            f"atol={args.atol_score}"
        )

    provenance.update(
        {
            "status": "PASS",
            "run_tag": RUN_TAG,
            "stage": "08_labpc_verify_trufor_policy_c_native",
            "scientific_role": "fresh LABPC clean-pipeline reproduction",
            "smoke_image_path": str(smoke["image_path"]),
            "smoke_cache_path": str(smoke["cache_path"]),
            "smoke_cache_sha256": actual_cache_sha,
            "smoke_variant": str(smoke["variant"]),
            "smoke_hardware": str(smoke["hardware_source"]),
            "smoke_native_hw": list(direct_hw),
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

    out = LAB_STAGE2_ROOT / "stage08_provenance.json"
    write_json(out, provenance)

    print("\nLABPC STAGE 08 PASS")
    print(f"provenance: {out}")
    print("Proceed to LABPC Stage 09 only after this PASS.")


if __name__ == "__main__":
    main()
