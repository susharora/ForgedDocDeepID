#!/usr/bin/env python3
"""Stage 3: clean pretrained TruFor detection + native-map localisation.

Primary scientific condition: Policy-C native only.

Localisation metrics on attack images with frozen altered rectangles:
  A      = altered union area / native image area
  E      = TruFor anomaly-map mass inside altered union / total map mass
  mu_w   = E / A
  PG     = 1[argmax anomaly map lies in altered union]
  E - A  = excess relevance mass above a spatially uniform map

The same metrics are also computed for altered face and altered text rectangles
where present. CIs use stem-cluster bootstrap resampling.

No thresholded segmentation metric is used as the primary common metric because
FantasyID ground truth is a coarse rectangular region inventory, not a
pixel-perfect forged-pixel mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    INVENTORY,
    OUT_ROOT,
    ROOT,
    relative_to_root,
    stable_token,
    verify_inventory,
)


N_BOOT_DEFAULT = 5000
BOOT_SEED = 10
CONTOUR_QUANTILE = 0.90

COLOR_FACE = (0, 255, 255)
COLOR_TEXT = (255, 0, 255)
COLOR_CAM = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


def require_stage2() -> Tuple[pd.DataFrame, dict]:
    manifest_path = OUT_ROOT / "inference_manifest.csv"
    provenance_path = OUT_ROOT / "stage02_inference_provenance.json"
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(
            "Stage 2 outputs missing. Run 02_infer_policy_c_native.py first."
        )

    provenance = json.loads(provenance_path.read_text())
    if provenance.get("status") != "PASS":
        raise RuntimeError("Stage 2 provenance does not record PASS")
    if provenance.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Stage 2 checkpoint SHA mismatch")
    if provenance.get("condition") != "policy_c_native":
        raise RuntimeError("Stage 2 condition is not policy_c_native")

    frame = pd.read_csv(manifest_path, keep_default_na=False)
    if len(frame) != 1844:
        raise RuntimeError(f"Stage 2 manifest incomplete: {len(frame)} != 1844")
    if frame["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in Stage 2 manifest")
    if set(frame["condition"].unique()) != {"policy_c_native"}:
        raise RuntimeError("Manifest contains a non-Policy-C condition")
    if set(frame["checkpoint_sha256"].unique()) != {EXPECTED_CHECKPOINT_SHA256}:
        raise RuntimeError("Manifest checkpoint SHA mismatch")

    return frame, provenance


def load_regions() -> pd.DataFrame:
    verify_inventory()
    regions = pd.read_excel(INVENTORY, sheet_name="Regions")
    required = {
        "image_path",
        "field_name",
        "region_provenance_raw",
        "x",
        "y",
        "width",
        "height",
    }
    missing = required - set(regions.columns)
    if missing:
        raise RuntimeError(f"Frozen Regions schema mismatch: {sorted(missing)}")

    regions = regions.copy()
    regions["image_path"] = regions["image_path"].astype(str)
    regions["field_name"] = regions["field_name"].astype(str).str.strip().str.lower()
    regions["region_provenance_raw"] = (
        regions["region_provenance_raw"].astype(str).str.strip().str.lower()
    )
    return regions


def binary_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y = frame["label"].astype(int).to_numpy()
    p = frame["trufor_score"].astype(float).to_numpy()
    pred = (p >= 0.5).astype(int)
    attack = y == 1
    bona = y == 0

    result = {
        "n": int(len(frame)),
        "n_attack": int(attack.sum()),
        "n_bonafide": int(bona.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred))
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "attack_recall": float(pred[attack].mean()) if attack.any() else np.nan,
        "bonafide_specificity": (
            float((pred[bona] == 0).mean()) if bona.any() else np.nan
        ),
        "mean_attack_score": float(p[attack].mean()) if attack.any() else np.nan,
        "mean_bonafide_score": float(p[bona].mean()) if bona.any() else np.nan,
    }
    return result


def detection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for split in ["dev_val", "official_test"]:
        subset = frame.loc[frame["eval_split"] == split]
        rows.append(
            {
                "eval_split": split,
                "group_type": "overall",
                "group": "all",
                **binary_metrics(subset),
            }
        )

        families = sorted(
            subset.loc[subset["label"].astype(int) == 1, "variant"].unique()
        )
        for family in families:
            family_subset = subset.loc[
                (subset["label"].astype(int) == 0)
                | (subset["variant"] == family)
            ]
            rows.append(
                {
                    "eval_split": split,
                    "group_type": "attack_family_vs_bonafide",
                    "group": family,
                    **binary_metrics(family_subset),
                }
            )

        for hardware in sorted(subset["hardware_source"].unique()):
            hw_subset = subset.loc[subset["hardware_source"] == hardware]
            rows.append(
                {
                    "eval_split": split,
                    "group_type": "hardware",
                    "group": hardware,
                    **binary_metrics(hw_subset),
                }
            )

    return pd.DataFrame(rows)


def round_box(row: pd.Series) -> Tuple[int, int, int, int]:
    x = int(round(float(row["x"])))
    y = int(round(float(row["y"])))
    w = int(round(float(row["width"])))
    h = int(round(float(row["height"])))
    return x, y, x + w, y + h


def clip_box(
    box: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    x0, y0, x1, y1 = box
    clipped = (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )
    changed = clipped != box
    cx0, cy0, cx1, cy1 = clipped
    if cx1 <= cx0 or cy1 <= cy0:
        return None, changed
    return clipped, changed


def make_mask(
    boxes: Sequence[Tuple[int, int, int, int]], height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    return mask


def localisation_values(amap: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    n = int(mask.size)
    m = int(mask.sum())
    if n <= 0 or m <= 0:
        return {"A": np.nan, "E": np.nan, "mu": np.nan, "PG": np.nan, "E_minus_A": np.nan}

    A = float(m / n)
    total = float(np.asarray(amap, dtype=np.float64).sum())
    if total <= 0.0 or not np.isfinite(total):
        return {"A": A, "E": np.nan, "mu": np.nan, "PG": np.nan, "E_minus_A": np.nan}

    E = float(np.asarray(amap[mask], dtype=np.float64).sum() / total)
    mu = float(E / A) if A > 0 else np.nan
    max_index = int(np.nanargmax(amap))
    PG = float(mask.reshape(-1)[max_index])
    return {
        "A": A,
        "E": E,
        "mu": mu,
        "PG": PG,
        "E_minus_A": float(E - A),
    }


def bootstrap_cluster_mean_ci(
    frame: pd.DataFrame,
    column: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    """Stem-cluster bootstrap CI for the image-level mean.

    This is vectorised over cluster sums/counts, so 5000 replicates remain
    cheap even for the full official test. A sampled stem contributes all of
    its hardware/image rows each time it is drawn.
    """
    work = frame[["file_stem", column]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[np.isfinite(work[column].to_numpy())]
    if work.empty:
        return np.nan, np.nan

    grouped = (
        work.groupby("file_stem", sort=True)[column]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(sums)
    if n_clusters == 1:
        stat = float(sums[0] / counts[0])
        return stat, stat

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot_sum = sums[draw].sum(axis=1)
    boot_count = counts[draw].sum(axis=1)
    stats = boot_sum / boot_count
    low, high = np.quantile(stats, [0.025, 0.975])
    return float(low), float(high)

def metric_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return BOOT_SEED + int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def summarise_localisation(frame: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    groups: List[Tuple[str, str, pd.DataFrame]] = []
    for split in ["dev_val", "official_test"]:
        split_frame = frame.loc[frame["eval_split"] == split]
        groups.append((split, "all_attacks", split_frame))
        for family in sorted(split_frame["variant"].unique()):
            groups.append(
                (
                    split,
                    family,
                    split_frame.loc[split_frame["variant"] == family],
                )
            )

    scope_prefix = {
        "union": "union",
        "face": "face",
        "text": "text",
    }

    for split, group_name, group in groups:
        for scope, prefix in scope_prefix.items():
            A_col = f"A_{prefix}"
            eligible = group.loc[np.isfinite(pd.to_numeric(group[A_col], errors="coerce"))].copy()
            if eligible.empty:
                continue

            row: Dict[str, object] = {
                "eval_split": split,
                "group": group_name,
                "scope": scope,
                "n_images": int(len(eligible)),
                "n_stems": int(eligible["file_stem"].nunique()),
            }

            for short, suffix in [
                ("A", "A"),
                ("E", "E"),
                ("mu_w", "mu"),
                ("PG", "PG"),
                ("E_minus_A", "E_minus_A"),
            ]:
                col = f"{suffix}_{prefix}"
                values = pd.to_numeric(eligible[col], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    row[f"mean_{short}"] = np.nan
                    row[f"median_{short}"] = np.nan
                    row[f"mean_{short}_ci_low"] = np.nan
                    row[f"mean_{short}_ci_high"] = np.nan
                    continue

                row[f"mean_{short}"] = float(np.mean(values))
                row[f"median_{short}"] = float(np.median(values))
                lo, hi = bootstrap_cluster_mean_ci(
                    eligible,
                    col,
                    n_boot,
                    metric_seed(split, group_name, scope, short, "mean"),
                )
                row[f"mean_{short}_ci_low"] = lo
                row[f"mean_{short}_ci_high"] = hi

            rows.append(row)

    return pd.DataFrame(rows)


def mask_boundary(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
        eroded[1:-1, 1:-1] = inner
        eroded[0, :] = False
        eroded[-1, :] = False
        eroded[:, 0] = False
        eroded[:, -1] = False
    boundary = mask & ~eroded

    for _ in range(max(0, thickness - 1)):
        dil = boundary.copy()
        dil[1:, :] |= boundary[:-1, :]
        dil[:-1, :] |= boundary[1:, :]
        dil[:, 1:] |= boundary[:, :-1]
        dil[:, :-1] |= boundary[:, 1:]
        boundary = dil
    return boundary


def heatmap_rgb(amap: np.ndarray) -> np.ndarray:
    from matplotlib import cm

    values = np.asarray(amap, dtype=np.float32)
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.995))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo + 1e-12:
        norm = np.zeros_like(values, dtype=np.float32)
    else:
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    rgba = cm.get_cmap("inferno")(norm)
    rgb = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return rgb


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=color, width=2)
    text = label[:24]
    bbox = draw.textbbox((x0, y0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ty = max(0, y0 - th - 2)
    draw.rectangle([x0, ty, x0 + tw + 3, ty + th + 2], fill=COLOR_BLACK)
    draw.text((x0 + 1, ty), text, fill=color, font=font)


def render_visuals(
    rgb_path: Path,
    amap: np.ndarray,
    score: float,
    region_rows: Sequence[Dict[str, object]],
    output_stem: Path,
) -> None:
    with Image.open(rgb_path) as image:
        base = np.array(image.convert("RGB"), dtype=np.uint8)

    if base.shape[:2] != amap.shape:
        raise RuntimeError(
            f"Visual image/map shape mismatch for {rgb_path}: {base.shape[:2]} vs {amap.shape}"
        )

    heat = heatmap_rgb(amap)
    heat_image = Image.fromarray(heat, mode="RGB")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    heat_image.save(str(output_stem) + "__heatmap.png", "PNG", compress_level=3)

    values = np.asarray(amap, dtype=np.float32)
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.995))
    if hi <= lo + 1e-12:
        alpha = np.zeros_like(values, dtype=np.float32)
    else:
        alpha = 0.58 * np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    overlay = (
        base.astype(np.float32) * (1.0 - alpha[..., None])
        + heat.astype(np.float32) * alpha[..., None]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    q = float(np.quantile(values, CONTOUR_QUANTILE))
    if np.isfinite(q) and float(np.max(values)) > float(np.min(values)) + 1e-12:
        top = values >= q
        boundary = mask_boundary(top, thickness=2)
        overlay[boundary] = np.asarray(COLOR_CAM, dtype=np.uint8)

    canvas = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for item in region_rows:
        box = item["box"]
        kind = str(item["kind"])
        label = str(item["field_name"])
        color = COLOR_FACE if kind == "face" else COLOR_TEXT
        draw_labeled_box(draw, box, label, color, font)

    # Compact legend matching the ResNet visual convention where possible.
    legend_lines = [
        (COLOR_WHITE, "TruFor top-10% contour"),
        (COLOR_FACE, "altered face GT"),
        (COLOR_TEXT, "altered text GT"),
    ]
    line_h = 13
    legend_w = 205
    legend_h = 8 + line_h * len(legend_lines) + 16
    draw.rectangle([0, 0, legend_w, legend_h], fill=COLOR_BLACK)
    y = 4
    for color, text in legend_lines:
        draw.line([(7, y + 5), (26, y + 5)], fill=color, width=2)
        draw.text((31, y), text, fill=COLOR_WHITE, font=font)
        y += line_h
    draw.text((7, y + 1), f"det score={score:.4f}", fill=COLOR_WHITE, font=font)

    canvas.save(str(output_stem) + "__overlay.png", "PNG", compress_level=3)


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="skip PNG heatmaps/overlays; default is to render all attack images",
    )
    parser.add_argument(
        "--max-visuals",
        type=int,
        default=0,
        help="debug cap; 0 means all attack images",
    )
    args = parser.parse_args()

    inference, stage2_provenance = require_stage2()
    regions = load_regions()

    detection = detection_summary(inference)
    detection_path = OUT_ROOT / "detection_summary.csv"
    detection.to_csv(detection_path, index=False)

    altered = regions.loc[regions["region_provenance_raw"] == "altered"].copy()
    region_groups = {
        str(image_path): group.copy()
        for image_path, group in altered.groupby("image_path", sort=False)
    }

    attacks = inference.loc[inference["label"].astype(int) == 1].copy()
    expected_attack_count = 306 + 1085
    if len(attacks) != expected_attack_count:
        raise RuntimeError(f"Expected {expected_attack_count} attacks, got {len(attacks)}")

    per_image_rows: List[Dict[str, object]] = []
    region_audit_rows: List[Dict[str, object]] = []
    exclusion_rows: List[Dict[str, object]] = []
    visuals_done = 0
    clipped_rectangles = 0

    visual_root = OUT_ROOT / "visuals"

    for n, (_, row) in enumerate(attacks.iterrows(), start=1):
        map_path = ROOT / str(row["map_path"])
        if not map_path.is_file():
            raise RuntimeError(f"Missing Stage 2 map: {map_path}")
        with np.load(map_path, allow_pickle=False) as data:
            amap = np.asarray(data["map"], dtype=np.float32)
            score = float(np.asarray(data["score"]).item())
            hw = tuple(int(x) for x in np.asarray(data["imgsize"]).tolist())

        height, width = hw
        if amap.shape != (height, width):
            raise RuntimeError(f"Map/native size mismatch: {row['image_path']}")

        image_regions = region_groups.get(str(row["image_path"]))
        clipped_items: List[Dict[str, object]] = []

        if image_regions is not None:
            for region_index, (_, reg) in enumerate(image_regions.iterrows()):
                original = round_box(reg)
                box, changed = clip_box(original, width, height)
                if changed:
                    clipped_rectangles += 1
                kind = "face" if str(reg["field_name"]).strip().lower() == "face" else "text"

                audit = {
                    "eval_split": str(row["eval_split"]),
                    "variant": str(row["variant"]),
                    "image_path": str(row["image_path"]),
                    "file_stem": str(row["file_stem"]),
                    "hardware_source": str(row["hardware_source"]),
                    "region_index_within_image": region_index,
                    "field_name": str(reg["field_name"]),
                    "kind": kind,
                    "original_x0": original[0],
                    "original_y0": original[1],
                    "original_x1": original[2],
                    "original_y1": original[3],
                    "clipped": bool(changed),
                    "native_width": width,
                    "native_height": height,
                    "valid_after_clip": box is not None,
                }
                if box is not None:
                    audit.update(
                        {
                            "x0": box[0],
                            "y0": box[1],
                            "x1": box[2],
                            "y1": box[3],
                        }
                    )
                    clipped_items.append(
                        {
                            "box": box,
                            "kind": kind,
                            "field_name": str(reg["field_name"]),
                        }
                    )
                else:
                    audit.update({"x0": np.nan, "y0": np.nan, "x1": np.nan, "y1": np.nan})
                    exclusion_rows.append(
                        {
                            "eval_split": str(row["eval_split"]),
                            "variant": str(row["variant"]),
                            "image_path": str(row["image_path"]),
                            "file_stem": str(row["file_stem"]),
                            "reason": "altered_rectangle_invalid_after_clip",
                            "field_name": str(reg["field_name"]),
                        }
                    )
                region_audit_rows.append(audit)

        union_boxes = [item["box"] for item in clipped_items]
        face_boxes = [item["box"] for item in clipped_items if item["kind"] == "face"]
        text_boxes = [item["box"] for item in clipped_items if item["kind"] == "text"]

        record: Dict[str, object] = {
            "eval_split": str(row["eval_split"]),
            "variant": str(row["variant"]),
            "image_path": str(row["image_path"]),
            "cache_path": str(row["cache_path"]),
            "map_path": str(row["map_path"]),
            "file_stem": str(row["file_stem"]),
            "hardware_source": str(row["hardware_source"]),
            "assigned_q": int(row["assigned_q"]),
            "trufor_score": score,
            "clean_correct_at_0p5": bool(score >= 0.5),
            "native_height": height,
            "native_width": width,
            "n_altered_rectangles": len(union_boxes),
            "n_face_rectangles": len(face_boxes),
            "n_text_rectangles": len(text_boxes),
        }

        if union_boxes:
            union_mask = make_mask(union_boxes, height, width)
            vals = localisation_values(amap, union_mask)
            for key, value in vals.items():
                record[f"{key}_union"] = value
        else:
            for key in ["A", "E", "mu", "PG", "E_minus_A"]:
                record[f"{key}_union"] = np.nan
            exclusion_rows.append(
                {
                    "eval_split": str(row["eval_split"]),
                    "variant": str(row["variant"]),
                    "image_path": str(row["image_path"]),
                    "file_stem": str(row["file_stem"]),
                    "reason": "no_valid_altered_rectangle",
                    "field_name": "",
                }
            )

        for scope, boxes in [("face", face_boxes), ("text", text_boxes)]:
            if boxes:
                mask = make_mask(boxes, height, width)
                vals = localisation_values(amap, mask)
                for key, value in vals.items():
                    record[f"{key}_{scope}"] = value
            else:
                for key in ["A", "E", "mu", "PG", "E_minus_A"]:
                    record[f"{key}_{scope}"] = np.nan

        per_image_rows.append(record)

        render_this = not args.no_visuals and (
            args.max_visuals <= 0 or visuals_done < args.max_visuals
        )
        if render_this:
            visual_stem = (
                visual_root
                / safe_component(str(row["eval_split"]))
                / safe_component(str(row["variant"]))
                / safe_component(str(row["hardware_source"]))
                / (
                    safe_component(str(row["file_stem"]))
                    + "__"
                    + stable_token(str(row["image_path"]), 8)
                )
            )
            render_visuals(
                ROOT / str(row["cache_path"]),
                amap,
                score,
                clipped_items,
                visual_stem,
            )
            visuals_done += 1

        if n % 50 == 0 or n == len(attacks):
            print(
                f"localisation/visuals processed {n}/{len(attacks)} | "
                f"visuals={visuals_done}"
            )

    per_image = pd.DataFrame(per_image_rows)
    if len(per_image) != expected_attack_count:
        raise RuntimeError("Per-image localisation table lost attack rows")

    per_image_path = OUT_ROOT / "localisation_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    region_audit = pd.DataFrame(region_audit_rows)
    region_audit_path = OUT_ROOT / "localisation_regions_audit.csv"
    region_audit.to_csv(region_audit_path, index=False)

    exclusions = pd.DataFrame(exclusion_rows)
    exclusion_path = OUT_ROOT / "localisation_exclusions.csv"
    exclusions.to_csv(exclusion_path, index=False)

    local_summary = summarise_localisation(per_image, args.n_bootstrap)
    local_summary_path = OUT_ROOT / "localisation_summary.csv"
    local_summary.to_csv(local_summary_path, index=False)

    annotation_coverage = (
        per_image.assign(annotated=np.isfinite(pd.to_numeric(per_image["A_union"], errors="coerce")))
        .groupby(["eval_split", "variant"], as_index=False)
        .agg(
            n_attacks=("image_path", "size"),
            n_annotated=("annotated", "sum"),
            n_stems=("file_stem", "nunique"),
        )
    )
    annotation_coverage["coverage"] = (
        annotation_coverage["n_annotated"] / annotation_coverage["n_attacks"]
    )
    annotation_coverage_path = OUT_ROOT / "annotation_coverage.csv"
    annotation_coverage.to_csv(annotation_coverage_path, index=False)

    # Human-readable concise report for the next scientific decision.
    det_primary = detection.loc[
        detection["group_type"].isin(["overall", "attack_family_vs_bonafide"])
    ].copy()

    loc_primary = local_summary.loc[local_summary["scope"] == "union"].copy()
    loc_primary = loc_primary.loc[loc_primary["group"] != "all_attacks"]

    report_lines = [
        "PRETRAINED TRUFOR — FANTASYID CLEAN BASELINE",
        "",
        "Condition: policy_c_native ONLY",
        "Preprocessing: frozen Policy C JPEG -> native HxW TruFor RGB / 256.0",
        "No raw-native baseline; no resize/crop/padding; no adaptation.",
        f"Checkpoint SHA256: {EXPECTED_CHECKPOINT_SHA256}",
        f"Inventory SHA256: {verify_inventory()}",
        "",
        "DETECTION (threshold 0.5 for accuracy/recall/specificity):",
        det_primary[
            [
                "eval_split",
                "group_type",
                "group",
                "n",
                "n_attack",
                "n_bonafide",
                "auroc",
                "balanced_accuracy",
                "attack_recall",
                "bonafide_specificity",
                "mean_attack_score",
                "mean_bonafide_score",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "ANNOTATION COVERAGE:",
        annotation_coverage.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "NATIVE LOCALISATION — altered-union primary metrics:",
        (
            loc_primary[
                [
                    "eval_split",
                    "group",
                    "n_images",
                    "n_stems",
                    "mean_A",
                    "mean_E",
                    "mean_mu_w",
                    "mean_PG",
                    "mean_E_minus_A",
                    "mean_E_ci_low",
                    "mean_E_ci_high",
                    "mean_mu_w_ci_low",
                    "mean_mu_w_ci_high",
                    "mean_PG_ci_low",
                    "mean_PG_ci_high",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            if not loc_primary.empty
            else "NO ELIGIBLE LOCALISATION ROWS"
        ),
        "",
        "INTERPRETATION:",
        "  A    = fraction of native image covered by altered GT rectangles.",
        "  E    = fraction of total TruFor anomaly-map mass inside those rectangles.",
        "  mu_w = E/A; 1.0 is spatially uniform map mass.",
        "  PG   = fraction whose native-map maximum lies inside altered GT.",
        "  CIs  = 95% stem-cluster bootstrap intervals.",
        "  Face/text scope rows are in localisation_summary.csv.",
        "  Localisation uses ALL annotated attacks; it is NOT conditioned on correct detection.",
        "  Rectangles are coarse region annotations, not pixel-perfect segmentation masks.",
        "",
        f"clipped altered rectangles: {clipped_rectangles}",
        f"attack heatmap/overlay pairs rendered: {visuals_done}",
        "",
        "STOP HERE before TruFor adaptation or adversarial attacks.",
        "Review detection, annotation coverage, localisation RMA/enrichment/PG, and PNGs first.",
    ]

    report = "\n".join(report_lines) + "\n"
    report_path = OUT_ROOT / "baseline_report.txt"
    report_path.write_text(report)

    final_provenance = dict(stage2_provenance)
    final_provenance.update(
        {
            "status": "PASS",
            "stage": "03_eval_policy_c_native",
            "condition": "policy_c_native",
            "n_attacks_total": int(len(per_image)),
            "n_localisation_eligible": int(np.isfinite(per_image["A_union"]).sum()),
            "clipped_altered_rectangles": int(clipped_rectangles),
            "bootstrap_replicates": int(args.n_bootstrap),
            "bootstrap_cluster": "file_stem",
            "visuals_rendered": int(visuals_done),
            "detection_summary": relative_to_root(detection_path),
            "localisation_per_image": relative_to_root(per_image_path),
            "localisation_summary": relative_to_root(local_summary_path),
            "regions_audit": relative_to_root(region_audit_path),
            "annotation_coverage": relative_to_root(annotation_coverage_path),
            "baseline_report": relative_to_root(report_path),
        }
    )
    (OUT_ROOT / "stage03_evaluation_provenance.json").write_text(
        json.dumps(final_provenance, indent=2, sort_keys=True) + "\n"
    )

    print("\n" + report)
    print("STAGE 3 PASS")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
