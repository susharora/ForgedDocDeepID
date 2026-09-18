#!/usr/bin/env python3
"""LABPC Stage 09: fresh native-resolution TruFor inference on all 1,844 images.

All outputs are LAB-specific:

    output/LABPC/trufor_pretrained_policy_c_native/

No Home output is read for inference and no Home output is overwritten.

Population:
- dev_val:      459
- official_test: 1385

Preprocessing remains canonical:
- exact frozen Policy-C JPEG bytes
- native HxW
- RGB float32 / 256.0
- no resize/crop/padding/re-JPEG

The script is safe to resume: validated LABPC NPZ files are reused.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    ROOT,
    atomic_save_npz,
    infer_one,
    load_policy_c_population,
    load_trufor_model,
    project_git_head,
    relative_to_root,
    resolve_device,
    sha256_file,
    verify_trufor_provenance,
    write_json,
)

RUN_TAG = "LABPC"
LAB_ROOT = ROOT / "output" / RUN_TAG / "trufor_pretrained_policy_c_native"


def require_stage08() -> dict:
    path = LAB_ROOT / "stage08_provenance.json"
    if not path.is_file():
        raise RuntimeError(
            f"LABPC Stage-08 provenance missing:\n{path}\n"
            "Run 08_labpc_verify_trufor_policy_c_native.py first."
        )
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError("LABPC Stage 08 does not record PASS")
    if payload.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("LABPC Stage-08 checkpoint SHA mismatch")
    return payload


def map_output_path(eval_split: str, image_path: str) -> Path:
    return LAB_ROOT / "maps" / eval_split / Path(image_path + ".npz")


def load_existing_map(path: Path, expected_image_path: str):
    with np.load(path, allow_pickle=False) as data:
        required = {"map", "score", "imgsize", "image_path", "checkpoint_sha256"}
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(
                f"Existing LABPC map is incomplete {path}: missing {sorted(missing)}"
            )
        stored_image = str(np.asarray(data["image_path"]).item())
        stored_ckpt = str(np.asarray(data["checkpoint_sha256"]).item())
        if stored_image != expected_image_path:
            raise RuntimeError(
                f"LABPC NPZ image-path mismatch: {path}\n"
                f"expected {expected_image_path}\nstored   {stored_image}"
            )
        if stored_ckpt != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"LABPC NPZ checkpoint mismatch: {path}")
        amap = np.asarray(data["map"], dtype=np.float32)
        score = float(np.asarray(data["score"]).item())
        hw = tuple(int(x) for x in np.asarray(data["imgsize"]).tolist())
    if amap.shape != hw:
        raise RuntimeError(
            f"LABPC NPZ map shape mismatch: {path}: {amap.shape} vs {hw}"
        )
    return amap, score, hw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-cache-sha", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    LAB_ROOT.mkdir(parents=True, exist_ok=True)
    require_stage08()
    provenance = verify_trufor_provenance(check_archive_member=False)
    population = load_policy_c_population().reset_index(drop=True)
    device = resolve_device(args.gpu)

    print("LABPC STAGE 09 — POLICY-C NATIVE TRUFOR INFERENCE")
    print(f"output root: {LAB_ROOT}")
    print("condition: Policy-C only")
    print("preprocessing: native HxW RGB -> float32 / 256.0")
    print("NO resize, crop, padding, or extra JPEG stage")
    print(f"images: {len(population)}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(args.gpu)}")

    model, checkpoint, _ = load_trufor_model(device)

    rows = []
    started = time.time()
    partial_path = LAB_ROOT / "inference_manifest.partial.csv"

    for i, row in population.iterrows():
        source = ROOT / str(row["cache_path"])
        if not source.is_file():
            raise RuntimeError(f"Policy-C cache image missing: {source}")

        if not args.skip_cache_sha:
            actual_sha = sha256_file(source)
            expected_sha = str(row["cache_sha256"])
            if actual_sha != expected_sha:
                raise RuntimeError(
                    "Frozen Policy-C cache SHA256 mismatch:\n"
                    f"image    {row['image_path']}\n"
                    f"expected {expected_sha}\nactual   {actual_sha}"
                )
        else:
            actual_sha = "NOT_RECHECKED"

        output = map_output_path(str(row["eval_split"]), str(row["image_path"]))

        if output.is_file() and not args.overwrite:
            amap, score, hw = load_existing_map(output, str(row["image_path"]))
            status = "resumed"
        else:
            with Image.open(source) as image:
                width, height = image.size
            try:
                result = infer_one(model, source, device, include_conf=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during LABPC native TruFor inference. "
                        "No resize was applied.\n"
                        f"image: {row['image_path']}\n"
                        f"native WxH: {width}x{height}"
                    ) from exc
                raise

            amap = np.asarray(result["map"], dtype=np.float32)
            score = float(result["score"])
            hw = tuple(int(x) for x in result["imgsize"])

            atomic_save_npz(
                output,
                map=amap,
                score=np.float32(score),
                imgsize=np.asarray(hw, dtype=np.int32),
                image_path=np.asarray(str(row["image_path"])),
                cache_path=np.asarray(str(row["cache_path"])),
                checkpoint_sha256=np.asarray(EXPECTED_CHECKPOINT_SHA256),
                preprocessing=np.asarray("native_RGB_float32_div256"),
                condition=np.asarray("policy_c_native"),
                run_tag=np.asarray(RUN_TAG),
            )
            status = "computed"

        rows.append(
            {
                "eval_split": str(row["eval_split"]),
                "image_path": str(row["image_path"]),
                "cache_path": str(row["cache_path"]),
                "file_stem": str(row["file_stem"]),
                "traffic_type": str(row["traffic_type"]),
                "variant": str(row["variant"]),
                "hardware_source": str(row["hardware_source"]),
                "label": int(row["label"]),
                "assigned_q": int(row["assigned_q"]),
                "cache_sha256": str(row["cache_sha256"]),
                "cache_sha256_runtime": actual_sha,
                "trufor_score": score,
                "native_height": int(hw[0]),
                "native_width": int(hw[1]),
                "map_min": float(np.min(amap)),
                "map_max": float(np.max(amap)),
                "map_mean": float(np.mean(amap)),
                "map_path": relative_to_root(output),
                "inference_status": status,
                "condition": "policy_c_native",
                "preprocessing": "native_RGB_float32_div256",
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "run_tag": RUN_TAG,
            }
        )

        done = i + 1
        if done % args.progress_every == 0 or done == len(population):
            elapsed = time.time() - started
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(
                f"processed {done}/{len(population)} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"last={row['eval_split']}:{row['variant']}:{row['hardware_source']}"
            )

    manifest = pd.DataFrame(rows)
    if len(manifest) != 1844:
        raise RuntimeError(f"LABPC manifest incomplete: {len(manifest)} != 1844")
    if manifest["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in LABPC manifest")

    expected_split_counts = {"official_test": 1385, "dev_val": 459}
    actual_split_counts = manifest["eval_split"].value_counts().to_dict()
    if actual_split_counts != expected_split_counts:
        raise RuntimeError(
            f"Unexpected LABPC completed split counts: {actual_split_counts}"
        )

    manifest_path = LAB_ROOT / "inference_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    provenance.update(
        {
            "status": "PASS",
            "run_tag": RUN_TAG,
            "stage": "09_labpc_infer_policy_c_native",
            "scientific_role": "fresh LABPC clean-pipeline reproduction",
            "n_images": int(len(manifest)),
            "split_counts": actual_split_counts,
            "cache_sha_rechecked": not args.skip_cache_sha,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(args.gpu)
                if device.type == "cuda"
                else "cpu"
            ),
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "manifest": relative_to_root(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "elapsed_seconds": time.time() - started,
            "native_height_min": int(manifest["native_height"].min()),
            "native_height_max": int(manifest["native_height"].max()),
            "native_width_min": int(manifest["native_width"].min()),
            "native_width_max": int(manifest["native_width"].max()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "project_git_head": project_git_head(),
        }
    )
    write_json(LAB_ROOT / "stage09_inference_provenance.json", provenance)

    print("\nLABPC STAGE 09 PASS")
    print(f"manifest: {manifest_path}")
    print(f"maps:     {LAB_ROOT / 'maps'}")
    print("\nDetection-score quick audit (not final metrics):")
    print(
        manifest.groupby(["eval_split", "traffic_type"])["trufor_score"]
        .agg(["count", "mean", "median", "min", "max"])
        .to_string(float_format=lambda x: f"{x:.6f}")
    )
    print("\nProceed to LABPC Stage 10 for cross-machine reproducibility gate.")
    print("Do NOT recalibrate the threshold.")


if __name__ == "__main__":
    main()
