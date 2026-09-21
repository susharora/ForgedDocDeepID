#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = "resnet_exact_archive_v1"


def _safe_component(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    value = value.strip("._-")
    return value or "na"


def _rect_mask(rectangles, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for rect in rectangles:
        x0, y0, x1, y1 = [int(v) for v in rect]
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return mask


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_numpy(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} is not float32: {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy().copy()


def save_exact_bundle(
    *,
    repo_root,
    archive_root,
    row,
    info,
    clean_input,
    adv_input,
    clean_cam,
    adv_cam,
    clean_probability,
    adv_probability,
    clean_margin,
    adv_margin,
    attack_metadata,
):
    repo_root = Path(repo_root).resolve()
    archive_root = Path(archive_root).resolve()

    clean_input_np = _float32_numpy(clean_input, "clean_input")
    adv_input_np = _float32_numpy(adv_input, "adv_input")
    clean_cam_np = _float32_numpy(clean_cam, "clean_cam")
    adv_cam_np = _float32_numpy(adv_cam, "adv_cam")

    if clean_input_np.ndim != 3:
        raise RuntimeError("clean_input must have shape [3,H,W]")
    if adv_input_np.shape != clean_input_np.shape:
        raise RuntimeError("clean/adv input shape mismatch")
    if clean_cam_np.ndim != 2:
        raise RuntimeError("clean_cam must have shape [H,W]")
    if adv_cam_np.shape != clean_cam_np.shape:
        raise RuntimeError("clean/adv CAM shape mismatch")

    height, width = clean_cam_np.shape

    if clean_input_np.shape[-2:] != (height, width):
        raise RuntimeError(
            f"input/CAM raster mismatch: input={clean_input_np.shape}, "
            f"cam={clean_cam_np.shape}"
        )

    content = np.zeros((height, width), dtype=np.uint8)
    pad_left = int(info["pad_left"])
    content_width = int(info["content_width"])
    content[:, pad_left:pad_left + content_width] = 1

    face = _rect_mask(info.get("face_rects", []), height, width)
    text = _rect_mask(info.get("text_rects", []), height, width)
    union = np.maximum(face, text)

    if union.sum() <= 0:
        raise RuntimeError("Exact archive has empty altered-union mask")
    if np.any(union > content):
        raise RuntimeError("GT union escapes document content support")

    image_path = str(row["image_path"])
    file_stem = str(row["file_stem"])
    evaluation_split = str(row["evaluation_split"])
    variant = str(row["variant"])
    hardware = str(row["hardware_source"])

    unique = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:12]

    directory = (
        archive_root
        / _safe_component(evaluation_split)
        / _safe_component(variant)
        / _safe_component(hardware)
    )
    directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        directory
        / f"{_safe_component(file_stem)}__{unique}__exact.npz"
    )
    temp_path = output_path.with_suffix(".npz.tmp")

    metadata_json = json.dumps(
        attack_metadata,
        sort_keys=True,
        separators=(",", ":"),
    )

    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SCHEMA_VERSION),
            image_path=np.asarray(image_path),
            file_stem=np.asarray(file_stem),
            evaluation_split=np.asarray(evaluation_split),
            variant=np.asarray(variant),
            hardware_source=np.asarray(hardware),
            clean_probability_attack=np.asarray(clean_probability, dtype=np.float64),
            adv_probability_attack=np.asarray(adv_probability, dtype=np.float64),
            clean_margin=np.asarray(clean_margin, dtype=np.float64),
            adv_margin=np.asarray(adv_margin, dtype=np.float64),
            attack_metadata_json=np.asarray(metadata_json),
            clean_input_normalized=clean_input_np,
            adv_input_normalized=adv_input_np,
            clean_cam_full=clean_cam_np,
            adv_cam_full=adv_cam_np,
            content_mask=content,
            face_mask=face,
            text_mask=text,
            union_mask=union,
        )

    temp_path.replace(output_path)

    try:
        relative_path = str(output_path.relative_to(repo_root))
    except ValueError:
        relative_path = str(output_path)

    return {
        "exact_bundle_path": relative_path,
        "exact_bundle_sha256": _sha256_file(output_path),
        "exact_bundle_bytes": int(output_path.stat().st_size),
        "exact_input_dtype": str(clean_input_np.dtype),
        "exact_cam_dtype": str(clean_cam_np.dtype),
        "exact_cam_height": int(height),
        "exact_cam_width": int(width),
        "exact_archive_schema": SCHEMA_VERSION,
    }
