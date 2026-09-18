#!/usr/bin/env python3
"""Stage 4: final clean evaluation using the dev-frozen TruFor threshold.

This stage NEVER optimises a threshold. It validates Stage 3's frozen threshold,
then evaluates the unchanged pretrained Policy-C-native TruFor pipeline.

Outputs
-------
1. Detection metrics at both:
     - upstream reference threshold 0.5
     - frozen dev-calibrated threshold (operational)
2. Frozen clean-correct populations for later adversarial experiments.
3. Native localisation A/E/mu_w/PG on:
     - all attacks (threshold-independent clean localisation baseline)
     - each model's clean-correct attack subset (future adversarial baseline)
4. Visuals for ALL attacks from cached Stage-2 anomaly maps.
5. Visuals for ALL bona-fides. TruFor is rerun only to obtain its confidence
   map; anomaly map and scalar score are checked against frozen Stage-2 outputs.

Bona-fide anomaly/confidence maps are diagnostic. A/E/mu_w/PG are undefined for
bona-fides because there is no manipulated ground-truth region.
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
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    INVENTORY,
    OUT_ROOT as STAGE2_ROOT,
    ROOT,
    infer_one,
    load_trufor_model,
    project_git_head,
    relative_to_root,
    resolve_device,
    sha256_file,
    stable_token,
    verify_inventory,
    verify_trufor_provenance,
    write_json,
)

PROTOCOL_ROOT = ROOT / "output" / "trufor_policy_c_frozen_protocol"
CALIB_ROOT = PROTOCOL_ROOT / "stage03_dev_calibration"
THRESHOLD_JSON = CALIB_ROOT / "frozen_threshold.json"
EVAL_ROOT = PROTOCOL_ROOT / "stage04_clean_evaluation"

UPSTREAM_THRESHOLD = 0.5
N_BOOT_DEFAULT = 5000
BOOT_SEED = 2410
CONTOUR_QUANTILE = 0.90
PARITY_MAP_ATOL = 5e-6
PARITY_SCORE_ATOL = 5e-7

COLOR_FACE = (0, 255, 255)
COLOR_TEXT = (255, 0, 255)
COLOR_CONTOUR = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


def require_inputs() -> Tuple[pd.DataFrame, dict, dict, Path]:
    manifest_path = STAGE2_ROOT / "inference_manifest.csv"
    stage2_path = STAGE2_ROOT / "stage02_inference_provenance.json"
    if not manifest_path.is_file() or not stage2_path.is_file():
        raise RuntimeError("Stage-2 inference outputs missing")
    if not THRESHOLD_JSON.is_file():
        raise RuntimeError(
            f"Frozen Stage-3 threshold missing:\n{THRESHOLD_JSON}\n"
            "Run 03_calibrate_dev_threshold.py first."
        )

    inference = pd.read_csv(manifest_path, keep_default_na=False)
    stage2 = json.loads(stage2_path.read_text())
    frozen = json.loads(THRESHOLD_JSON.read_text())

    if len(inference) != 1844:
        raise RuntimeError(f"Stage-2 manifest incomplete: {len(inference)} != 1844")
    if stage2.get("status") != "PASS":
        raise RuntimeError("Stage-2 provenance is not PASS")
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("Stage-3 threshold is not FROZEN")
    if frozen.get("calibration_split") != "dev_val":
        raise RuntimeError("Threshold was not calibrated on dev_val")
    if frozen.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen threshold checkpoint SHA mismatch")
    if frozen.get("decision_rule") != "attack iff trufor_score >= threshold":
        raise RuntimeError("Unexpected frozen decision rule")

    actual_manifest_sha = sha256_file(manifest_path)
    if frozen.get("stage2_manifest_sha256") != actual_manifest_sha:
        raise RuntimeError(
            "Stage-2 inference manifest changed after threshold freeze:\n"
            f"frozen {frozen.get('stage2_manifest_sha256')}\n"
            f"actual {actual_manifest_sha}"
        )

    threshold = float(frozen["frozen_threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("Frozen threshold is non-finite")

    return inference, stage2, frozen, manifest_path


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> Dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["trufor_score"].to_numpy(dtype=float)
    pred = (scores >= float(threshold)).astype(int)
    attack = labels == 1
    bona = labels == 0

    result: Dict[str, float] = {
        "n": int(len(frame)),
        "n_attack": int(attack.sum()),
        "n_bonafide": int(bona.sum()),
        "auroc": (
            float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan
        ),
        "accuracy": float(np.mean(pred == labels)),
        "attack_recall": float(np.mean(pred[attack] == 1)) if attack.any() else np.nan,
        "bonafide_specificity": float(np.mean(pred[bona] == 0)) if bona.any() else np.nan,
        "mean_attack_score": float(np.mean(scores[attack])) if attack.any() else np.nan,
        "mean_bonafide_score": float(np.mean(scores[bona])) if bona.any() else np.nan,
    }
    if attack.any() and bona.any():
        result["balanced_accuracy"] = 0.5 * (
            result["attack_recall"] + result["bonafide_specificity"]
        )
    else:
        result["balanced_accuracy"] = np.nan
    return result


def detection_summary(frame: pd.DataFrame, threshold: float, name: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(split: str, group_type: str, group: str, subset: pd.DataFrame) -> None:
        rows.append(
            {
                "threshold_name": name,
                "threshold": float(threshold),
                "eval_split": split,
                "group_type": group_type,
                "group": group,
                **threshold_metrics(subset, threshold),
            }
        )

    for split in ["dev_val", "official_test"]:
        split_frame = frame.loc[frame["eval_split"].astype(str) == split].copy()
        add(split, "overall", "all", split_frame)

        bona = split_frame["label"].astype(int) == 0
        for family in sorted(split_frame.loc[~bona, "variant"].astype(str).unique()):
            add(
                split,
                "attack_family_vs_bonafide",
                family,
                split_frame.loc[
                    bona | (split_frame["variant"].astype(str) == family)
                ],
            )

        for hardware in sorted(split_frame["hardware_source"].astype(str).unique()):
            add(
                split,
                "hardware",
                hardware,
                split_frame.loc[
                    split_frame["hardware_source"].astype(str) == hardware
                ],
            )

    return pd.DataFrame(rows)


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
    area = int(mask.sum())
    if area <= 0 or mask.size <= 0:
        return {
            "A": np.nan,
            "E": np.nan,
            "mu": np.nan,
            "PG": np.nan,
            "E_minus_A": np.nan,
        }

    A = float(area / mask.size)
    map64 = np.asarray(amap, dtype=np.float64)
    total = float(map64.sum())
    if total <= 0.0 or not np.isfinite(total):
        return {"A": A, "E": np.nan, "mu": np.nan, "PG": np.nan, "E_minus_A": np.nan}

    E = float(map64[mask].sum() / total)
    max_index = int(np.nanargmax(amap))
    PG = float(mask.reshape(-1)[max_index])
    return {
        "A": A,
        "E": E,
        "mu": float(E / A),
        "PG": PG,
        "E_minus_A": float(E - A),
    }


def metric_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return BOOT_SEED + int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def bootstrap_cluster_mean_ci(
    frame: pd.DataFrame,
    column: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    work = frame[["file_stem", column]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[np.isfinite(work[column].to_numpy())]
    if work.empty:
        return np.nan, np.nan

    grouped = work.groupby("file_stem", sort=True)[column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(sums)
    if n_clusters == 1:
        stat = float(sums[0] / counts[0])
        return stat, stat

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    stats = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def summarise_localisation(per_image: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for split in ["dev_val", "official_test"]:
        split_frame = per_image.loc[per_image["eval_split"].astype(str) == split]
        for family in sorted(split_frame["variant"].astype(str).unique()):
            family_frame = split_frame.loc[split_frame["variant"].astype(str) == family]

            populations = [
                ("all_attacks", family_frame),
                (
                    "clean_correct_attacks",
                    family_frame.loc[family_frame["clean_correct_at_frozen_threshold"]],
                ),
            ]

            for population_name, population in populations:
                if population.empty:
                    continue
                for scope in ["union", "face", "text"]:
                    A_col = f"A_{scope}"
                    eligible = population.loc[
                        np.isfinite(pd.to_numeric(population[A_col], errors="coerce"))
                    ].copy()
                    if eligible.empty:
                        continue

                    record: Dict[str, object] = {
                        "eval_split": split,
                        "variant": family,
                        "population": population_name,
                        "scope": scope,
                        "n_images": int(len(eligible)),
                        "n_stems": int(eligible["file_stem"].nunique()),
                    }
                    for short, col in [
                        ("A", f"A_{scope}"),
                        ("E", f"E_{scope}"),
                        ("mu_w", f"mu_{scope}"),
                        ("PG", f"PG_{scope}"),
                        ("E_minus_A", f"E_minus_A_{scope}"),
                    ]:
                        vals = pd.to_numeric(eligible[col], errors="coerce").to_numpy(float)
                        vals = vals[np.isfinite(vals)]
                        record[f"mean_{short}"] = float(np.mean(vals)) if vals.size else np.nan
                        record[f"median_{short}"] = float(np.median(vals)) if vals.size else np.nan
                        lo, hi = bootstrap_cluster_mean_ci(
                            eligible,
                            col,
                            n_boot,
                            metric_seed(split, family, population_name, scope, short),
                        )
                        record[f"mean_{short}_ci_low"] = lo
                        record[f"mean_{short}_ci_high"] = hi
                    rows.append(record)

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


def colourmap_rgb(
    values: np.ndarray,
    cmap_name: str,
    qlo: float = 0.05,
    qhi: float = 0.995,
    fixed_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    from matplotlib import colormaps

    x = np.asarray(values, dtype=np.float32)
    if fixed_range is not None:
        lo, hi = map(float, fixed_range)
    else:
        lo = float(np.quantile(x, qlo))
        hi = float(np.quantile(x, qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
            lo = float(np.min(x))
            hi = float(np.max(x))
    norm = (
        np.zeros_like(x, dtype=np.float32)
        if hi <= lo + 1e-12
        else np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    )
    rgba = colormaps[cmap_name](norm)
    return np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)


def anomaly_overlay(base: np.ndarray, amap: np.ndarray) -> np.ndarray:
    # TruFor's anomaly map is already a class-1 softmax probability map. Use
    # an absolute 0..1 visual scale so a bona-fide map with only weak anomaly
    # response is not made to look artificially strong by per-image stretching.
    heat = colourmap_rgb(amap, "inferno", fixed_range=(0.0, 1.0))
    values = np.asarray(amap, dtype=np.float32)
    alpha = 0.58 * np.clip(values, 0.0, 1.0)
    overlay = (
        base.astype(np.float32) * (1.0 - alpha[..., None])
        + heat.astype(np.float32) * alpha[..., None]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    q = float(np.quantile(values, CONTOUR_QUANTILE))
    if np.isfinite(q) and float(values.max()) > float(values.min()) + 1e-12:
        boundary = mask_boundary(values >= q, thickness=2)
        overlay[boundary] = np.asarray(COLOR_CONTOUR, dtype=np.uint8)
    return overlay


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=color, width=2)
    text = str(label)[:24]
    bbox = draw.textbbox((x0, y0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ty = max(0, y0 - th - 2)
    draw.rectangle([x0, ty, x0 + tw + 3, ty + th + 2], fill=COLOR_BLACK)
    draw.text((x0 + 1, ty), text, fill=color, font=font)


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "none"


def render_attack_visuals(
    rgb_path: Path,
    amap: np.ndarray,
    score: float,
    threshold: float,
    clean_correct: bool,
    region_rows: Sequence[Dict[str, object]],
    output_stem: Path,
) -> None:
    with Image.open(rgb_path) as image:
        base = np.array(image.convert("RGB"), dtype=np.uint8)
    if base.shape[:2] != amap.shape:
        raise RuntimeError("attack visual image/map shape mismatch")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colourmap_rgb(amap, "inferno", fixed_range=(0.0, 1.0))).save(
        str(output_stem) + "__anomaly_heatmap.png", "PNG", compress_level=3
    )

    overlay = anomaly_overlay(base, amap)
    canvas = Image.fromarray(overlay)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for item in region_rows:
        color = COLOR_FACE if item["kind"] == "face" else COLOR_TEXT
        draw_labeled_box(draw, item["box"], str(item["field_name"]), color, font)

    legend = [
        (COLOR_WHITE, "TruFor top-10% anomaly contour"),
        (COLOR_FACE, "altered face GT"),
        (COLOR_TEXT, "altered text GT"),
    ]
    draw.rectangle([0, 0, 255, 64], fill=COLOR_BLACK)
    y = 3
    for color, text in legend:
        draw.line([(6, y + 5), (24, y + 5)], fill=color, width=2)
        draw.text((29, y), text, fill=COLOR_WHITE, font=font)
        y += 13
    draw.text(
        (6, y),
        f"score={score:.4f} t*={threshold:.4f} correct={int(clean_correct)}",
        fill=COLOR_WHITE,
        font=font,
    )
    canvas.save(str(output_stem) + "__anomaly_overlay.png", "PNG", compress_level=3)


def add_top_banner(image: Image.Image, lines: Sequence[str], height: int = 42) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + height), COLOR_BLACK)
    out.paste(image, (0, height))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    y = 3
    for line in lines[:3]:
        draw.text((6, y), str(line), fill=COLOR_WHITE, font=font)
        y += 12
    return out


def fit_height(image: Image.Image, target_h: int) -> Image.Image:
    if image.height <= target_h:
        return image.copy()
    scale = target_h / image.height
    width = max(1, int(round(image.width * scale)))
    return image.resize((width, target_h), Image.Resampling.BILINEAR)


def render_bonafide_visuals(
    rgb_path: Path,
    amap: np.ndarray,
    conf: np.ndarray,
    score: float,
    threshold: float,
    eval_split: str,
    file_stem: str,
    hardware: str,
    output_stem: Path,
) -> Dict[str, str]:
    with Image.open(rgb_path) as image:
        base = np.array(image.convert("RGB"), dtype=np.uint8)
    if base.shape[:2] != amap.shape or conf.shape != amap.shape:
        raise RuntimeError("bona-fide visual image/map/confidence shape mismatch")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    anomaly_heat = colourmap_rgb(amap, "inferno", fixed_range=(0.0, 1.0))
    conf_heat = colourmap_rgb(conf, "viridis", fixed_range=(0.0, 1.0))
    overlay = anomaly_overlay(base, amap)

    anomaly_heat_path = Path(str(output_stem) + "__anomaly_heatmap.png")
    anomaly_overlay_path = Path(str(output_stem) + "__anomaly_overlay.png")
    confidence_path = Path(str(output_stem) + "__confidence.png")
    panel_path = Path(str(output_stem) + "__panel.png")

    Image.fromarray(anomaly_heat).save(anomaly_heat_path, "PNG", compress_level=3)

    overlay_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_img)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, 265, 30], fill=COLOR_BLACK)
    draw.text((6, 3), "WHITE = top-10% anomaly contour", fill=COLOR_WHITE, font=font)
    draw.text((6, 16), f"score={score:.4f}  t*={threshold:.4f}", fill=COLOR_WHITE, font=font)
    overlay_img.save(anomaly_overlay_path, "PNG", compress_level=3)

    Image.fromarray(conf_heat).save(confidence_path, "PNG", compress_level=3)

    # Reviewer panel is intentionally downscaled; the individual PNGs above stay native.
    source_im = Image.fromarray(base)
    anomaly_im = Image.fromarray(overlay)
    conf_im = Image.fromarray(conf_heat)
    target_h = 640
    thumbs = [fit_height(x, target_h) for x in [source_im, anomaly_im, conf_im]]
    widths = [x.width for x in thumbs]
    panel = Image.new("RGB", (sum(widths), max(x.height for x in thumbs)), COLOR_BLACK)
    x = 0
    for thumb in thumbs:
        panel.paste(thumb, (x, 0))
        x += thumb.width
    pred = "attack" if score >= threshold else "bonafide"
    correct = pred == "bonafide"
    banner = add_top_banner(
        panel,
        [
            f"{eval_split} | {file_stem} | {hardware}",
            f"SOURCE | ANOMALY OVERLAY | CONFIDENCE    score={score:.4f} t*={threshold:.4f}",
            f"prediction={pred}  clean_correct={correct}",
        ],
        height=42,
    )
    banner.save(panel_path, "PNG", compress_level=3)

    return {
        "anomaly_heatmap": relative_to_root(anomaly_heat_path),
        "anomaly_overlay": relative_to_root(anomaly_overlay_path),
        "confidence": relative_to_root(confidence_path),
        "panel": relative_to_root(panel_path),
    }


def make_contact_sheet(entries: pd.DataFrame, destination: Path, title: str, n: int = 12) -> None:
    subset = entries.head(n)
    if subset.empty:
        return
    thumbs: List[Image.Image] = []
    for _, row in subset.iterrows():
        path = ROOT / str(row["panel_path"])
        if not path.is_file():
            continue
        with Image.open(path) as image:
            im = image.convert("RGB")
            im.thumbnail((720, 280), Image.Resampling.LANCZOS)
            thumbs.append(im.copy())
    if not thumbs:
        return

    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    cell_w = max(x.width for x in thumbs)
    cell_h = max(x.height for x in thumbs)
    header_h = 26
    sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), COLOR_BLACK)
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill=COLOR_WHITE, font=ImageFont.load_default())
    for i, im in enumerate(thumbs):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = header_h + row * cell_h
        sheet.paste(im, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "PNG", compress_level=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU for bona-fide confidence maps; -1 CPU")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument(
        "--skip-attack-visuals",
        action="store_true",
        help="skip regeneration of attack anomaly heatmap/overlay PNGs",
    )
    parser.add_argument(
        "--skip-bonafide-visuals",
        action="store_true",
        help="skip bona-fide confidence rerun and visualisation",
    )
    parser.add_argument(
        "--max-bonafide-visuals",
        type=int,
        default=0,
        help="debug cap; 0 means all 453 bona-fides",
    )
    args = parser.parse_args()

    if args.n_bootstrap < 100:
        raise RuntimeError("Use at least 100 bootstrap replicates")

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    verify_trufor_provenance(check_archive_member=False)
    inference, stage2, frozen, manifest_path = require_inputs()
    threshold = float(frozen["frozen_threshold"])

    print("TRUFOR STAGE 4 — FROZEN-THRESHOLD CLEAN EVALUATION")
    print("weights: pretrained / unchanged")
    print("condition: policy_c_native")
    print(f"operational threshold frozen on dev: {threshold:.9f}")
    print("official test is evaluation-only in this stage")

    # ------------------------------------------------------------------
    # Detection and clean-correct population freeze
    # ------------------------------------------------------------------
    detection = pd.concat(
        [
            detection_summary(inference, UPSTREAM_THRESHOLD, "upstream_0p5"),
            detection_summary(inference, threshold, "dev_calibrated_frozen"),
        ],
        ignore_index=True,
    )
    detection_path = EVAL_ROOT / "detection_summary.csv"
    detection.to_csv(detection_path, index=False)

    population = inference.copy()
    population["frozen_threshold"] = threshold
    population["clean_prediction"] = (
        population["trufor_score"].to_numpy(dtype=float) >= threshold
    ).astype(int)
    population["clean_correct"] = (
        population["clean_prediction"].to_numpy(dtype=int)
        == population["label"].to_numpy(dtype=int)
    )
    population_path = EVAL_ROOT / "clean_decision_population.csv"
    population.to_csv(population_path, index=False)

    clean_correct = population.loc[population["clean_correct"]].copy()
    clean_correct_path = EVAL_ROOT / "clean_correct_population.csv"
    clean_correct.to_csv(clean_correct_path, index=False)
    clean_correct_attacks_path = EVAL_ROOT / "clean_correct_attacks.csv"
    clean_correct_bonafides_path = EVAL_ROOT / "clean_correct_bonafides.csv"
    clean_correct.loc[clean_correct["label"].astype(int) == 1].to_csv(
        clean_correct_attacks_path, index=False
    )
    clean_correct.loc[clean_correct["label"].astype(int) == 0].to_csv(
        clean_correct_bonafides_path, index=False
    )

    attack_summary = (
        population.loc[population["label"].astype(int) == 1]
        .groupby(["eval_split", "variant"], as_index=False)
        .agg(
            n_total=("image_path", "size"),
            n_clean_correct=("clean_correct", "sum"),
            n_stems=("file_stem", "nunique"),
        )
    )
    attack_correct_stems = (
        population.loc[(population["label"].astype(int) == 1) & population["clean_correct"]]
        .groupby(["eval_split", "variant"])["file_stem"]
        .nunique()
        .rename("n_clean_correct_stems")
        .reset_index()
    )
    attack_summary = attack_summary.merge(
        attack_correct_stems, on=["eval_split", "variant"], how="left"
    )
    attack_summary["n_clean_correct_stems"] = (
        attack_summary["n_clean_correct_stems"].fillna(0).astype(int)
    )
    attack_summary["clean_correct_rate"] = (
        attack_summary["n_clean_correct"] / attack_summary["n_total"]
    )

    bona_summary = (
        population.loc[population["label"].astype(int) == 0]
        .groupby(["eval_split"], as_index=False)
        .agg(
            n_total=("image_path", "size"),
            n_clean_correct=("clean_correct", "sum"),
            n_stems=("file_stem", "nunique"),
        )
    )
    bona_correct_stems = (
        population.loc[(population["label"].astype(int) == 0) & population["clean_correct"]]
        .groupby(["eval_split"])["file_stem"]
        .nunique()
        .rename("n_clean_correct_stems")
        .reset_index()
    )
    bona_summary = bona_summary.merge(bona_correct_stems, on=["eval_split"], how="left")
    bona_summary["n_clean_correct_stems"] = (
        bona_summary["n_clean_correct_stems"].fillna(0).astype(int)
    )
    bona_summary["clean_correct_rate"] = (
        bona_summary["n_clean_correct"] / bona_summary["n_total"]
    )
    attack_summary.to_csv(EVAL_ROOT / "clean_correct_attack_summary.csv", index=False)
    bona_summary.to_csv(EVAL_ROOT / "clean_correct_bonafide_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Attack localisation from frozen Stage-2 maps
    # ------------------------------------------------------------------
    regions = load_regions()
    altered = regions.loc[regions["region_provenance_raw"] == "altered"].copy()
    region_groups = {
        str(image_path): group.copy()
        for image_path, group in altered.groupby("image_path", sort=False)
    }

    attacks = population.loc[population["label"].astype(int) == 1].copy()
    if len(attacks) != 1391:
        raise RuntimeError(f"Expected 1391 attacks, got {len(attacks)}")

    per_image_rows: List[Dict[str, object]] = []
    region_audit_rows: List[Dict[str, object]] = []
    exclusion_rows: List[Dict[str, object]] = []
    clipped_rectangles = 0
    attack_visuals = 0
    attack_visual_root = EVAL_ROOT / "visuals" / "attacks"

    for n, (_, row) in enumerate(attacks.iterrows(), start=1):
        map_path = ROOT / str(row["map_path"])
        if not map_path.is_file():
            raise RuntimeError(f"Missing Stage-2 map: {map_path}")
        with np.load(map_path, allow_pickle=False) as data:
            amap = np.asarray(data["map"], dtype=np.float32)
            score = float(np.asarray(data["score"]).item())
            hw = tuple(int(x) for x in np.asarray(data["imgsize"]).tolist())
        height, width = hw
        if amap.shape != (height, width):
            raise RuntimeError(f"Map/native shape mismatch: {row['image_path']}")

        clipped_items: List[Dict[str, object]] = []
        image_regions = region_groups.get(str(row["image_path"]))
        if image_regions is not None:
            for region_index, (_, reg) in enumerate(image_regions.iterrows()):
                original = round_box(reg)
                box, changed = clip_box(original, width, height)
                if changed:
                    clipped_rectangles += 1
                kind = "face" if str(reg["field_name"]).lower() == "face" else "text"
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
                    audit.update({"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
                    clipped_items.append(
                        {"box": box, "kind": kind, "field_name": str(reg["field_name"])}
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

        union_boxes = [x["box"] for x in clipped_items]
        face_boxes = [x["box"] for x in clipped_items if x["kind"] == "face"]
        text_boxes = [x["box"] for x in clipped_items if x["kind"] == "text"]

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
            "frozen_threshold": threshold,
            "clean_correct_at_frozen_threshold": bool(row["clean_correct"]),
            "native_height": height,
            "native_width": width,
            "n_altered_rectangles": len(union_boxes),
            "n_face_rectangles": len(face_boxes),
            "n_text_rectangles": len(text_boxes),
        }

        for scope, boxes in [("union", union_boxes), ("face", face_boxes), ("text", text_boxes)]:
            if boxes:
                vals = localisation_values(amap, make_mask(boxes, height, width))
            else:
                vals = {"A": np.nan, "E": np.nan, "mu": np.nan, "PG": np.nan, "E_minus_A": np.nan}
            for key, value in vals.items():
                record[f"{key}_{scope}"] = value

        if not union_boxes:
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

        per_image_rows.append(record)

        if not args.skip_attack_visuals:
            visual_stem = (
                attack_visual_root
                / safe_component(str(row["eval_split"]))
                / safe_component(str(row["variant"]))
                / safe_component(str(row["hardware_source"]))
                / (safe_component(str(row["file_stem"])) + "__" + stable_token(str(row["image_path"]), 8))
            )
            render_attack_visuals(
                ROOT / str(row["cache_path"]),
                amap,
                score,
                threshold,
                bool(row["clean_correct"]),
                clipped_items,
                visual_stem,
            )
            attack_visuals += 1

        if n % 100 == 0 or n == len(attacks):
            print(f"attack localisation {n}/{len(attacks)} | visuals={attack_visuals}")

    per_image = pd.DataFrame(per_image_rows)
    per_image_path = EVAL_ROOT / "localisation_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    region_audit = pd.DataFrame(region_audit_rows)
    region_audit.to_csv(EVAL_ROOT / "localisation_regions_audit.csv", index=False)
    pd.DataFrame(exclusion_rows).to_csv(EVAL_ROOT / "localisation_exclusions.csv", index=False)

    local_summary = summarise_localisation(per_image, args.n_bootstrap)
    local_summary_path = EVAL_ROOT / "localisation_summary.csv"
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
    annotation_coverage["coverage"] = annotation_coverage["n_annotated"] / annotation_coverage["n_attacks"]
    annotation_coverage.to_csv(EVAL_ROOT / "annotation_coverage.csv", index=False)

    # ------------------------------------------------------------------
    # Bona-fide anomaly + confidence visualisation
    # ------------------------------------------------------------------
    bona_visual_rows: List[Dict[str, object]] = []
    bona_visuals = 0
    max_map_parity = 0.0
    max_score_parity = 0.0

    bona = population.loc[population["label"].astype(int) == 0].copy()
    if len(bona) != 453:
        raise RuntimeError(f"Expected 453 bona-fides, got {len(bona)}")

    if not args.skip_bonafide_visuals:
        device = resolve_device(args.gpu)
        model, _, _ = load_trufor_model(device)
        print(f"bona-fide confidence rerun device: {device}")
        if device.type == "cuda":
            print(f"gpu: {torch.cuda.get_device_name(args.gpu)}")

        bona_visual_root = EVAL_ROOT / "visuals" / "bonafide"
        for n, (_, row) in enumerate(bona.iterrows(), start=1):
            if args.max_bonafide_visuals > 0 and bona_visuals >= args.max_bonafide_visuals:
                break

            source = ROOT / str(row["cache_path"])
            result = infer_one(model, source, device, include_conf=True)
            rerun_map = np.asarray(result["map"], dtype=np.float32)
            conf = np.asarray(result["conf"], dtype=np.float32)
            rerun_score = float(result["score"])

            cached_path = ROOT / str(row["map_path"])
            with np.load(cached_path, allow_pickle=False) as data:
                cached_map = np.asarray(data["map"], dtype=np.float32)
                cached_score = float(np.asarray(data["score"]).item())

            if cached_map.shape != rerun_map.shape:
                raise RuntimeError(f"Bona-fide rerun map shape mismatch: {row['image_path']}")
            map_error = float(np.max(np.abs(cached_map - rerun_map)))
            score_error = abs(cached_score - rerun_score)
            max_map_parity = max(max_map_parity, map_error)
            max_score_parity = max(max_score_parity, score_error)
            if map_error > PARITY_MAP_ATOL or score_error > PARITY_SCORE_ATOL:
                raise RuntimeError(
                    "Bona-fide confidence rerun changed frozen Stage-2 output:\n"
                    f"image {row['image_path']}\n"
                    f"map max abs error={map_error:.9g} (tol={PARITY_MAP_ATOL})\n"
                    f"score abs error={score_error:.9g} (tol={PARITY_SCORE_ATOL})"
                )

            visual_stem = (
                bona_visual_root
                / safe_component(str(row["eval_split"]))
                / safe_component(str(row["hardware_source"]))
                / (safe_component(str(row["file_stem"])) + "__" + stable_token(str(row["image_path"]), 8))
            )
            paths = render_bonafide_visuals(
                source,
                cached_map,
                conf,
                cached_score,
                threshold,
                str(row["eval_split"]),
                str(row["file_stem"]),
                str(row["hardware_source"]),
                visual_stem,
            )

            pred = int(cached_score >= threshold)
            bona_visual_rows.append(
                {
                    "eval_split": str(row["eval_split"]),
                    "image_path": str(row["image_path"]),
                    "file_stem": str(row["file_stem"]),
                    "hardware_source": str(row["hardware_source"]),
                    "trufor_score": cached_score,
                    "frozen_threshold": threshold,
                    "clean_prediction": pred,
                    "clean_correct": pred == 0,
                    "map_mean": float(cached_map.mean()),
                    "map_max": float(cached_map.max()),
                    "confidence_mean": float(conf.mean()),
                    "confidence_max": float(conf.max()),
                    "map_rerun_max_abs_error": map_error,
                    "score_rerun_abs_error": score_error,
                    "panel_path": paths["panel"],
                    "anomaly_overlay_path": paths["anomaly_overlay"],
                    "anomaly_heatmap_path": paths["anomaly_heatmap"],
                    "confidence_path": paths["confidence"],
                }
            )
            bona_visuals += 1
            if device.type == "cuda":
                # Memory hygiene only; no numerical/spatial preprocessing change.
                torch.cuda.empty_cache()

            if n % 25 == 0 or n == len(bona):
                print(f"bona-fide confidence/visuals {n}/{len(bona)}")

        bona_index = pd.DataFrame(bona_visual_rows)
        bona_index_path = EVAL_ROOT / "bonafide_visual_index.csv"
        bona_index.to_csv(bona_index_path, index=False)

        if args.max_bonafide_visuals <= 0 and len(bona_index) != 453:
            raise RuntimeError(f"Bona-fide visual audit incomplete: {len(bona_index)} != 453")

        # Review sheets explicitly contrast the most suspicious false positives
        # with correctly accepted bona-fides close to the decision boundary.
        fp = bona_index.loc[~bona_index["clean_correct"]].sort_values(
            "trufor_score", ascending=False
        )
        tn = bona_index.loc[bona_index["clean_correct"]].copy()
        tn["distance_below_threshold"] = threshold - tn["trufor_score"]
        tn = tn.sort_values("distance_below_threshold", ascending=True)
        make_contact_sheet(
            fp,
            EVAL_ROOT / "review_bonafide_high_score_false_positives.png",
            "Highest-score bona-fide false positives under frozen t*",
            n=12,
        )
        make_contact_sheet(
            tn,
            EVAL_ROOT / "review_bonafide_correct_near_threshold.png",
            "Correct bona-fides closest below frozen t*",
            n=12,
        )
    else:
        bona_index_path = None

    # ------------------------------------------------------------------
    # Report and provenance
    # ------------------------------------------------------------------
    primary_det = detection.loc[
        (detection["threshold_name"] == "dev_calibrated_frozen")
        & detection["group_type"].isin(["overall", "attack_family_vs_bonafide"])
    ].copy()
    primary_loc = local_summary.loc[
        (local_summary["scope"] == "union")
    ].copy()

    report_lines = [
        "PRETRAINED TRUFOR — FROZEN POLICY-C PROTOCOL",
        "",
        "Weights: unchanged pretrained TruFor",
        "Input: frozen Policy C -> canonical native-resolution TruFor",
        f"Checkpoint SHA256: {EXPECTED_CHECKPOINT_SHA256}",
        f"Frozen dev threshold: {threshold:.9f}",
        "Decision rule: attack iff score >= frozen threshold",
        "",
        "DETECTION AT FROZEN THRESHOLD",
        primary_det[
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
        "CLEAN-CORRECT ATTACK POPULATIONS (future localisation attack eligibility)",
        attack_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "CLEAN-CORRECT BONA-FIDE POPULATIONS",
        bona_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "NATIVE LOCALISATION (union rectangles)",
        primary_loc[
            [
                "eval_split",
                "variant",
                "population",
                "n_images",
                "n_stems",
                "mean_A",
                "mean_E",
                "mean_mu_w",
                "mean_PG",
                "mean_E_ci_low",
                "mean_E_ci_high",
                "mean_mu_w_ci_low",
                "mean_mu_w_ci_high",
                "mean_PG_ci_low",
                "mean_PG_ci_high",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "LOCALISATION DEFINITIONS",
        "  A    = altered GT union area / native image area",
        "  E    = anomaly-map mass inside altered union / total anomaly-map mass",
        "  mu_w = E/A; 1.0 means spatially uniform anomaly mass",
        "  PG   = native anomaly-map maximum inside altered union",
        "  CIs  = 95% stem-cluster bootstrap intervals",
        "  Bona-fides have anomaly/confidence maps but no A/E/mu_w/PG because there is no forged GT.",
        "",
        f"clipped altered rectangles: {clipped_rectangles}",
        f"attack visual pairs rendered: {attack_visuals}",
        f"bona-fide visual sets rendered: {bona_visuals}",
        f"bona-fide rerun max map abs error: {max_map_parity:.9g}",
        f"bona-fide rerun max score abs error: {max_score_parity:.9g}",
        "",
        "STOP HERE before adversarial attacks.",
        "Review the frozen threshold, clean-correct population sizes, localisation baselines,",
        "and bona-fide anomaly/confidence reviewer sheets before defining the attack population.",
    ]
    report = "\n".join(report_lines) + "\n"
    report_path = EVAL_ROOT / "clean_protocol_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": "PASS",
        "stage": "04_eval_frozen_threshold",
        "project_git_head": project_git_head(),
        "condition": "policy_c_native",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "inventory_sha256": verify_inventory(),
        "stage2_manifest": relative_to_root(manifest_path),
        "stage2_manifest_sha256": sha256_file(manifest_path),
        "frozen_threshold_json": relative_to_root(THRESHOLD_JSON),
        "frozen_threshold_json_sha256": sha256_file(THRESHOLD_JSON),
        "frozen_threshold": threshold,
        "threshold_calibration_split": "dev_val",
        "official_test_threshold_fitting": False,
        "clean_decision_population": relative_to_root(population_path),
        "clean_correct_population": relative_to_root(clean_correct_path),
        "clean_correct_attacks": relative_to_root(clean_correct_attacks_path),
        "clean_correct_attacks_sha256": sha256_file(clean_correct_attacks_path),
        "clean_correct_bonafides": relative_to_root(clean_correct_bonafides_path),
        "clean_correct_bonafides_sha256": sha256_file(clean_correct_bonafides_path),
        "detection_summary": relative_to_root(detection_path),
        "localisation_per_image": relative_to_root(per_image_path),
        "localisation_summary": relative_to_root(local_summary_path),
        "bona_fide_visual_index": (
            relative_to_root(bona_index_path) if bona_index_path is not None else None
        ),
        "attack_visuals_rendered": int(attack_visuals),
        "bonafide_visuals_rendered": int(bona_visuals),
        "bonafide_map_rerun_max_abs_error": float(max_map_parity),
        "bonafide_score_rerun_max_abs_error": float(max_score_parity),
        "bootstrap_replicates": int(args.n_bootstrap),
        "bootstrap_cluster": "file_stem",
        "report": relative_to_root(report_path),
    }
    write_json(EVAL_ROOT / "stage04_provenance.json", provenance)

    print("\n" + report)
    print("STAGE 4 PASS")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
