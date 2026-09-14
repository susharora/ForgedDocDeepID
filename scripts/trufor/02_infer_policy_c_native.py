#!/usr/bin/env python3
"""Stage 2: pretrained TruFor inference on ONLY frozen Policy-C native images.

Population
----------
- dev_val: 459 images (306 attacks, 153 bona-fide)
- official_test: 1385 images (1085 attacks, 300 bona-fide)

No raw-native condition is created or evaluated.
No resize/crop/padding is applied. If native inference OOMs, the script stops
rather than silently changing preprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    OUT_ROOT,
    ROOT,
    atomic_save_npz,
    infer_one,
    load_policy_c_population,
    load_trufor_model,
    map_output_path,
    relative_to_root,
    resolve_device,
    sha256_file,
    verify_trufor_provenance,
    write_json,
)


def require_stage1() -> dict:
    path = OUT_ROOT / "stage01_provenance.json"
    if not path.is_file():
        raise RuntimeError(
            f"Stage 1 provenance missing: {path}\n"
            "Run 01_verify_trufor_policy_c_native.py first."
        )
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError("Stage 1 provenance does not record PASS")
    if payload.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Stage 1 checkpoint SHA no longer matches frozen TruFor")
    return payload


def load_existing_map(path: Path, expected_image_path: str):
    with np.load(path, allow_pickle=False) as data:
        required = {"map", "score", "imgsize", "image_path", "checkpoint_sha256"}
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"Existing map is incomplete {path}: missing {sorted(missing)}")
        stored_image = str(np.asarray(data["image_path"]).item())
        stored_ckpt = str(np.asarray(data["checkpoint_sha256"]).item())
        if stored_image != expected_image_path:
            raise RuntimeError(
                f"Existing NPZ image-path mismatch: {path}\n"
                f"expected {expected_image_path}\nstored   {stored_image}"
            )
        if stored_ckpt != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"Existing NPZ checkpoint mismatch: {path}")
        amap = np.asarray(data["map"], dtype=np.float32)
        score = float(np.asarray(data["score"]).item())
        hw = tuple(int(x) for x in np.asarray(data["imgsize"]).tolist())
    if amap.shape != hw:
        raise RuntimeError(f"Existing NPZ map shape mismatch: {path}: {amap.shape} vs {hw}")
    return amap, score, hw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute maps even when a validated NPZ already exists",
    )
    parser.add_argument(
        "--skip-cache-sha",
        action="store_true",
        help="skip per-image Policy-C cache SHA256 verification (not recommended)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print/write partial manifest every N images",
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    require_stage1()
    provenance = verify_trufor_provenance(check_archive_member=False)
    population = load_policy_c_population().reset_index(drop=True)
    device = resolve_device(args.gpu)

    print("TRUFOR STAGE 2 — POLICY-C NATIVE INFERENCE")
    print("condition: Policy-C only")
    print("preprocessing: native HxW RGB -> float32 / 256.0")
    print("NO resize, crop, padding, or extra JPEG stage")
    print(f"images: {len(population)}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(args.gpu)}")

    model, checkpoint, cfg = load_trufor_model(device)

    rows = []
    started = time.time()
    partial_path = OUT_ROOT / "inference_manifest.partial.csv"

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
            # Read dimensions before inference so an OOM message records the
            # scientifically relevant native geometry.
            with Image.open(source) as image:
                width, height = image.size
            try:
                result = infer_one(model, source, device, include_conf=False)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during NATIVE TruFor inference. No resize was "
                        "applied. Stop here rather than changing preprocessing.\n"
                        f"image: {row['image_path']}\n"
                        f"native WxH: {width}x{height}\n"
                        "Paste this log before deciding whether a documented "
                        "memory-control strategy is scientifically acceptable."
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
            }
        )

        done = i + 1
        if done % args.progress_every == 0 or done == len(population):
            elapsed = time.time() - started
            frame = pd.DataFrame(rows)
            frame.to_csv(partial_path, index=False)
            print(
                f"processed {done}/{len(population)} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"last={row['eval_split']}:{row['variant']}:{row['hardware_source']}"
            )

    manifest = pd.DataFrame(rows)
    if len(manifest) != 1844:
        raise RuntimeError(f"Inference manifest incomplete: {len(manifest)} != 1844")
    if manifest["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in inference manifest")

    expected_split_counts = {"official_test": 1385, "dev_val": 459}
    actual_split_counts = manifest["eval_split"].value_counts().to_dict()
    if actual_split_counts != expected_split_counts:
        raise RuntimeError(f"Unexpected completed split counts: {actual_split_counts}")

    manifest_path = OUT_ROOT / "inference_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    provenance.update(
        {
            "stage": "02_infer_policy_c_native",
            "condition": "policy_c_native",
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
            "elapsed_seconds": time.time() - started,
            "native_height_min": int(manifest["native_height"].min()),
            "native_height_max": int(manifest["native_height"].max()),
            "native_width_min": int(manifest["native_width"].min()),
            "native_width_max": int(manifest["native_width"].max()),
        }
    )
    write_json(OUT_ROOT / "stage02_inference_provenance.json", provenance)

    print("\nSTAGE 2 PASS")
    print(f"manifest: {manifest_path}")
    print(f"maps:     {OUT_ROOT / 'maps'}")
    print("\nDetection-score quick audit (not final metrics):")
    print(
        manifest.groupby(["eval_split", "traffic_type"])["trufor_score"]
        .agg(["count", "mean", "median", "min", "max"])
        .to_string(float_format=lambda x: f"{x:.4f}")
    )
    print("\nProceed to Stage 3 for frozen detection + localisation evaluation.")


if __name__ == "__main__":
    main()
