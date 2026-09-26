#!/usr/bin/env python3
"""
Stage 54 — dissertation-ready frozen clean TruFor single-example portrait pages.

This is a report-only packaging stage. It performs NO model inference.

Goal
----
Replace the dense Stage-53 multi-example pages with a more legible portrait
layout: one example per page, two image columns, dissertation-friendly text,
legend and colour bar. The page shows:

  - bona-fide counterpart
  - forged Policy-C input + GT
  - frozen TruFor anomaly overlay + GT
  - enlarged manipulated-region crop from the frozen TruFor overlay

The example set remains deterministic and scientifically grounded:
  - exactly five examples, one per manipulation family
  - restricted to clean-correct attacks
  - exact same-stem / same-hardware bona-fide counterpart required
  - chosen as the robust family medoid over A_union, E_union, mu_union,
    and clean TruFor score

Outputs
-------
output/trufor_policy_c_frozen_protocol/stage54_clean_single_example_panels/
analysis_transfer_bundles/trufor_frozen_clean_single_example_panels.tar
analysis_transfer_bundles/trufor_frozen_clean_single_example_panels.tar.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

FROZEN_ROOT = ROOT / "output" / "trufor_policy_c_frozen_protocol" / "stage04_clean_evaluation_accuracy"
LOCALISATION = FROZEN_ROOT / "localisation_per_image.csv"
LOCALISATION_SUMMARY = FROZEN_ROOT / "localisation_summary.csv"
STAGE4_PROVENANCE = FROZEN_ROOT / "stage04_provenance.json"
INVENTORY = ROOT / "output" / "fantasyid_inventory_2026-09-05_023126.xlsx"
PAIR_INDEXES = [
    ROOT / "output" / "policy_c_cache_index.csv",
    ROOT / "output" / "fantasyid_official_test_policy_c_index.csv",
]
LAB_MAP_ROOT = ROOT / "output" / "LABPC" / "trufor_pretrained_policy_c_native" / "maps"
OUT_ROOT = ROOT / "output" / "trufor_policy_c_frozen_protocol" / "stage54_clean_single_example_panels"
TRANSFER_ROOT = ROOT / "analysis_transfer_bundles"
TAR_PATH = TRANSFER_ROOT / "trufor_frozen_clean_single_example_panels.tar"
TAR_SHA_PATH = Path(str(TAR_PATH) + ".sha256")

EXPECTED_VARIANTS = ["digital_1", "digital_2", "digital_3", "facedancer", "textdiffuserft_bfei"]
EXPECTED_SPLIT_BY_VARIANT = {
    "digital_1": "dev_val",
    "digital_2": "dev_val",
    "digital_3": "official_test",
    "facedancer": "official_test",
    "textdiffuserft_bfei": "official_test",
}

FACE_COLOR = "#00FFFF"
TEXT_COLOR = "#FF00FF"
CONTOUR_COLOR = "#FFFFFF"
CONTOUR_QUANTILE = 0.90
OVERLAY_ALPHA_SCALE = 0.58
METRIC_PARITY_ATOL = 2e-5
SCORE_PARITY_ATOL = 2e-5
CROP_MARGIN_FRACTION = 0.18
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69
DPI = 300


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file missing:\n{path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def resolve_rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")[:120] or "item"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def split_label(value: str) -> str:
    return "DEV" if str(value) == "dev_val" else "OFFICIAL TEST"


def load_frozen_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    for path in [LOCALISATION, LOCALISATION_SUMMARY, STAGE4_PROVENANCE, INVENTORY, *PAIR_INDEXES]:
        require_file(path)

    loc = pd.read_csv(LOCALISATION, keep_default_na=False)
    summary = pd.read_csv(LOCALISATION_SUMMARY, keep_default_na=False)
    provenance = json.loads(STAGE4_PROVENANCE.read_text())

    if provenance.get("status") != "PASS":
        raise RuntimeError("Frozen Stage-4 provenance does not record PASS")
    if len(loc) != 1391:
        raise RuntimeError(f"Expected 1391 Stage-4 attack rows, found {len(loc)}")
    if loc["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in Stage-4 localisation table")

    return loc, summary, provenance


def load_pair_index() -> pd.DataFrame:
    frames = []
    for path in PAIR_INDEXES:
        frame = pd.read_csv(path, keep_default_na=False).copy()
        if "split" not in frame.columns:
            frame["split"] = "official_test"
        required = {"image_path", "cache_path", "file_stem", "traffic_type", "hardware_source", "label", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Pair index schema mismatch {path}: {sorted(missing)}")
        frame["source_index"] = resolve_rel(path)
        bona = (pd.to_numeric(frame["label"], errors="coerce") == 0) | (
            frame["traffic_type"].astype(str).str.strip().str.lower() == "bonafide"
        )
        frame = frame.loc[bona].copy()
        frame["pair_eval_split"] = np.where(frame["split"].astype(str) == "dev_val", "dev_val", "official_test")
        frames.append(frame[["image_path", "cache_path", "file_stem", "hardware_source", "pair_eval_split", "source_index"]].copy())

    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.loc[pairs["cache_path"].astype(str).map(lambda x: (ROOT / x).is_file())].copy()
    if pairs.empty:
        raise RuntimeError("No existing bona-fide Policy-C cache files were found")
    return pairs


def attach_exact_bonafide_pairs(attacks: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    groups: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for key, group in pairs.groupby(["file_stem", "hardware_source", "pair_eval_split"], sort=False):
        groups[(str(key[0]), str(key[1]), str(key[2]))] = group.copy()

    records = []
    for _, row in attacks.iterrows():
        key = (str(row["file_stem"]), str(row["hardware_source"]), str(row["eval_split"]))
        candidates = groups.get(key)
        rec = row.to_dict()
        if candidates is None or candidates.empty:
            rec["has_exact_bonafide_pair"] = False
            rec["bonafide_image_path"] = ""
            rec["bonafide_cache_path"] = ""
            rec["bonafide_pair_source_index"] = ""
        else:
            chosen = candidates.sort_values(["cache_path", "image_path"], kind="stable").iloc[0]
            rec["has_exact_bonafide_pair"] = True
            rec["bonafide_image_path"] = str(chosen["image_path"])
            rec["bonafide_cache_path"] = str(chosen["cache_path"])
            rec["bonafide_pair_source_index"] = str(chosen["source_index"])
        records.append(rec)
    return pd.DataFrame(records)


def robust_medoid_score(frame: pd.DataFrame) -> pd.Series:
    metrics = ["A_union", "E_union", "mu_union", "trufor_score"]
    score = pd.Series(np.zeros(len(frame), dtype=np.float64), index=frame.index)
    for col in metrics:
        x = pd.to_numeric(frame[col], errors="coerce")
        if x.isna().any():
            raise RuntimeError(f"Selection metric {col} contains NaN")
        median = float(x.median())
        q25 = float(x.quantile(0.25))
        q75 = float(x.quantile(0.75))
        scale = q75 - q25
        if not math.isfinite(scale) or scale <= 1e-12:
            mad = float((x - median).abs().median())
            scale = mad if mad > 1e-12 else 1.0
        score = score + (x - median).abs() / scale
    return score


def choose_examples(loc: pd.DataFrame, summary: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    work = loc.copy()
    if "clean_correct_at_frozen_threshold" not in work.columns:
        raise RuntimeError("Stage-4 localisation table lacks clean-correct flag")
    work = work.loc[as_bool(work["clean_correct_at_frozen_threshold"])].copy()
    work = attach_exact_bonafide_pairs(work, pairs)
    work = work.loc[as_bool(work["has_exact_bonafide_pair"])].copy()

    frozen_summary = summary.loc[(summary["population"].astype(str) == "clean_correct_attacks") & (summary["scope"].astype(str) == "union")].copy()
    selections = []
    for variant in EXPECTED_VARIANTS:
        expected_split = EXPECTED_SPLIT_BY_VARIANT[variant]
        group = work.loc[(work["variant"].astype(str) == variant) & (work["eval_split"].astype(str) == expected_split)].copy()
        if group.empty:
            raise RuntimeError(f"No clean-correct exact-paired candidates for {variant}")
        group["_medoid_score"] = robust_medoid_score(group)
        group = group.sort_values(["_medoid_score", "image_path"], kind="stable")
        chosen = group.iloc[0].copy()
        fam = frozen_summary.loc[(frozen_summary["variant"].astype(str) == variant) & (frozen_summary["eval_split"].astype(str) == expected_split)]
        if len(fam) != 1:
            raise RuntimeError(f"Expected one frozen union summary row for {variant}; found {len(fam)}")
        fam = fam.iloc[0]
        chosen["family_mean_A"] = float(fam["mean_A"])
        chosen["family_mean_E"] = float(fam["mean_E"])
        chosen["family_mean_mu_w"] = float(fam["mean_mu_w"])
        chosen["family_mean_PG"] = float(fam["mean_PG"])
        chosen["selection_reason"] = (
            "clean-correct exact-paired family medoid: minimum robust IQR-scaled distance to family medians "
            "of A_union, E_union, mu_union and clean TruFor score"
        )
        selections.append(chosen)

    out = pd.DataFrame(selections).reset_index(drop=True)
    out.insert(0, "example_number", np.arange(1, len(out) + 1))
    if len(out) != 5 or out["image_path"].duplicated().any():
        raise RuntimeError("Stage-54 deterministic selection failed")
    return out


def load_regions() -> pd.DataFrame:
    frame = pd.read_excel(INVENTORY, sheet_name="Regions")
    required = {"image_path", "field_name", "region_provenance_raw", "x", "y", "width", "height"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen Regions schema mismatch: {sorted(missing)}")
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str)
    frame["field_name"] = frame["field_name"].astype(str).str.strip().str.lower()
    frame["region_provenance_raw"] = frame["region_provenance_raw"].astype(str).str.strip().str.lower()
    return frame


def boxes_for_image(regions: pd.DataFrame, image_path: str, width: int, height: int) -> List[Dict[str, object]]:
    rows = regions.loc[(regions["image_path"].astype(str) == str(image_path)) & (regions["region_provenance_raw"].astype(str) == "altered")]
    if rows.empty:
        raise RuntimeError(f"No altered Regions rows for {image_path}")
    items = []
    for _, row in rows.iterrows():
        x0 = int(round(float(row["x"])))
        y0 = int(round(float(row["y"])))
        x1 = x0 + int(round(float(row["width"])))
        y1 = y0 + int(round(float(row["height"])))
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        kind = "face" if str(row["field_name"]).lower() == "face" else "text"
        items.append({"box": (x0, y0, x1, y1), "kind": kind, "field_name": str(row["field_name"])})
    if not items:
        raise RuntimeError(f"No valid altered rectangles after clipping for {image_path}")
    return items


def union_mask(boxes: Sequence[Dict[str, object]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        mask[y0:y1, x0:x1] = True
    return mask


def frozen_metrics(amap: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    area = int(mask.sum())
    if area <= 0:
        raise RuntimeError("Empty GT mask while recomputing display parity")
    A = float(area / mask.size)
    values = np.asarray(amap, dtype=np.float64)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("Invalid TruFor anomaly-map mass")
    E = float(values[mask].sum() / total)
    mu = float(E / A)
    max_index = int(np.nanargmax(amap))
    PG = float(mask.reshape(-1)[max_index])
    return {"A": A, "E": E, "mu_w": mu, "PG": PG}


def resolve_map_path(row: pd.Series, map_source: str) -> Tuple[Path, str]:
    stage04 = ROOT / str(row["map_path"])
    lab = LAB_MAP_ROOT / str(row["eval_split"]) / Path(str(row["image_path"]) + ".npz")
    if map_source == "stage04":
        candidates = [(stage04, "stage04_canonical")]
    elif map_source == "labpc":
        candidates = [(lab, "labpc_reproduction")]
    else:
        candidates = [(stage04, "stage04_canonical"), (lab, "labpc_reproduction")]
    for path, label in candidates:
        if path.is_file():
            return path, label
    attempted = "\n".join(f"  {p}" for p, _ in candidates)
    raise RuntimeError(
        "No frozen clean TruFor map could be resolved for "
        f"{row['image_path']}.\nAttempted:\n{attempted}\n"
        "Restore the Stage-4 cache or the LABPC clean-map cache."
    )


def load_map(row: pd.Series, map_source: str) -> Tuple[np.ndarray, float, Path, str]:
    path, source_label = resolve_map_path(row, map_source)
    with np.load(path, allow_pickle=False) as data:
        if "map" not in data.files:
            raise RuntimeError(f"NPZ missing map: {path}")
        amap = np.asarray(data["map"], dtype=np.float32)
        score = float(np.asarray(data["score"]).item()) if "score" in data.files else float(row["trufor_score"])
    return amap, score, path, source_label


def inferno_rgb(amap: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(amap, dtype=np.float32), 0.0, 1.0)
    rgba = plt.get_cmap("inferno")(x)
    return np.rint(rgba[..., :3] * 255.0).astype(np.uint8)


def mask_boundary(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
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


def anomaly_overlay(rgb: np.ndarray, amap: np.ndarray) -> np.ndarray:
    heat = inferno_rgb(amap)
    values = np.clip(np.asarray(amap, dtype=np.float32), 0.0, 1.0)
    alpha = OVERLAY_ALPHA_SCALE * values
    overlay = rgb.astype(np.float32) * (1.0 - alpha[..., None]) + heat.astype(np.float32) * alpha[..., None]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    q = float(np.quantile(values, CONTOUR_QUANTILE))
    if np.isfinite(q) and float(values.max()) > float(values.min()) + 1e-12:
        contour = mask_boundary(values >= q, thickness=2)
        overlay[contour] = np.asarray([255, 255, 255], dtype=np.uint8)
    return overlay


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def draw_gt(ax, boxes: Sequence[Dict[str, object]]) -> None:
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        color = FACE_COLOR if item["kind"] == "face" else TEXT_COLOR
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.4, zorder=10))


def crop_window_from_boxes(boxes: Sequence[Dict[str, object]], width: int, height: int) -> Tuple[int, int, int, int]:
    xs0 = [b["box"][0] for b in boxes]
    ys0 = [b["box"][1] for b in boxes]
    xs1 = [b["box"][2] for b in boxes]
    ys1 = [b["box"][3] for b in boxes]
    x0 = min(xs0)
    y0 = min(ys0)
    x1 = max(xs1)
    y1 = max(ys1)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    margin = int(round(CROP_MARGIN_FRACTION * max(bw, bh)))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return x0, y0, x1, y1


def remap_boxes_to_crop(boxes: Sequence[Dict[str, object]], crop: Tuple[int, int, int, int]) -> List[Dict[str, object]]:
    cx0, cy0, _, _ = crop
    out = []
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        out.append({"box": (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), "kind": item["kind"], "field_name": item["field_name"]})
    return out


def prepare_example(row: pd.Series, regions: pd.DataFrame, map_source: str) -> Dict[str, object]:
    attack_path = ROOT / str(row["cache_path"])
    bona_path = ROOT / str(row["bonafide_cache_path"])
    require_file(attack_path)
    require_file(bona_path)

    attack_rgb = read_rgb(attack_path)
    bona_rgb = read_rgb(bona_path)
    amap, map_score, resolved_map_path, source_label = load_map(row, map_source)

    height, width = attack_rgb.shape[:2]
    if amap.shape != (height, width):
        raise RuntimeError(f"Map / Policy-C geometry mismatch:\n{row['image_path']}\nimage={attack_rgb.shape[:2]} map={amap.shape}")

    boxes = boxes_for_image(regions, str(row["image_path"]), width, height)
    mask = union_mask(boxes, height, width)
    measured = frozen_metrics(amap, mask)
    expected = {"A": float(row["A_union"]), "E": float(row["E_union"]), "mu_w": float(row["mu_union"]), "PG": float(row["PG_union"])}
    errors = {key: abs(measured[key] - expected[key]) for key in expected}
    if max(errors.values()) > METRIC_PARITY_ATOL:
        raise RuntimeError(f"Frozen localisation metric parity failed for {row['image_path']}: errors={errors}")
    score_error = abs(map_score - float(row["trufor_score"]))
    if score_error > SCORE_PARITY_ATOL:
        raise RuntimeError(f"Frozen score parity failed for {row['image_path']}: {score_error}")

    overlay = anomaly_overlay(attack_rgb, amap)
    crop = crop_window_from_boxes(boxes, width, height)
    cx0, cy0, cx1, cy1 = crop
    overlay_crop = overlay[cy0:cy1, cx0:cx1].copy()
    crop_boxes = remap_boxes_to_crop(boxes, crop)

    return {
        "attack_rgb": attack_rgb,
        "bona_rgb": bona_rgb,
        "overlay_rgb": overlay,
        "overlay_crop": overlay_crop,
        "boxes": boxes,
        "crop_boxes": crop_boxes,
        "map_path": resolved_map_path,
        "map_source": source_label,
        "metric_errors": errors,
        "score_error": score_error,
        "crop_window": crop,
    }


def write_panel_image(ax, image: np.ndarray, title: str, boxes: Sequence[Dict[str, object]] | None = None) -> None:
    ax.imshow(image)
    if boxes:
        draw_gt(ax, boxes)
    ax.set_title(title, fontsize=11.2, pad=7)
    ax.axis("off")


def save_individual_panels(row: pd.Series, prepared: Dict[str, object], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"example_{int(row['example_number']):02d}_{safe_name(str(row['variant']))}_{safe_name(str(row['hardware_source']))}_{safe_name(str(row['file_stem']))}"

    specs = [
        (prepared["bona_rgb"], "bona_fide_counterpart", None, "Bona fide counterpart"),
        (prepared["attack_rgb"], "forged_input_gt", prepared["boxes"], "Forged Policy-C input + GT"),
        (prepared["overlay_rgb"], "trufor_overlay_gt", prepared["boxes"], "Frozen TruFor anomaly overlay + GT"),
        (prepared["overlay_crop"], "trufor_overlay_crop_gt", prepared["crop_boxes"], "Manipulated-region crop (overlay + GT)"),
    ]
    paths = []
    for image, suffix, boxes, title in specs:
        fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor="white")
        write_panel_image(ax, image, title, boxes)
        out_path = out_dir / f"{prefix}_{suffix}.png"
        fig.savefig(out_path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        paths.append(out_path)
    return paths


def render_page(row: pd.Series, prepared: Dict[str, object], page_path: Path, page_index: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), facecolor="white")
    gs = fig.add_gridspec(
        4, 2,
        left=0.05, right=0.97, top=0.965, bottom=0.045,
        hspace=0.35, wspace=0.08,
        height_ratios=[0.18, 1.0, 1.0, 0.28],
    )

    split = split_label(row["eval_split"])
    variant = str(row["variant"])
    hardware = str(row["hardware_source"])
    stem = str(row["file_stem"])
    title = f"Frozen clean TruFor qualitative example — page {page_index}/{total_pages}"
    subtitle1 = f"{split} • {variant} • {hardware} • {stem}"
    subtitle2 = (
        f"score={float(row['trufor_score']):.3f}    A={float(row['A_union']):.3f}    "
        f"E={float(row['E_union']):.3f}    μw={float(row['mu_union']):.2f}    PG={float(row['PG_union']):.2f}"
    )
    subtitle3 = (
        f"family mean: E={float(row['family_mean_E']):.3f}, μw={float(row['family_mean_mu_w']):.2f}, "
        f"PG={float(row['family_mean_PG']):.2f}"
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(0.5, 0.88, title, ha="center", va="center", fontsize=16, fontweight="bold")
    header_ax.text(0.5, 0.50, subtitle1, ha="center", va="center", fontsize=12.2)
    header_ax.text(0.5, 0.19, subtitle2 + "    |    " + subtitle3, ha="center", va="center", fontsize=9.6)

    ax = fig.add_subplot(gs[1, 0])
    write_panel_image(ax, prepared["bona_rgb"], "Bona fide counterpart")

    ax = fig.add_subplot(gs[1, 1])
    write_panel_image(ax, prepared["attack_rgb"], "Forged Policy-C input + GT", prepared["boxes"])

    ax = fig.add_subplot(gs[2, 0])
    write_panel_image(ax, prepared["overlay_rgb"], "Frozen TruFor anomaly overlay + GT", prepared["boxes"])

    ax = fig.add_subplot(gs[2, 1])
    write_panel_image(ax, prepared["overlay_crop"], "Manipulated-region crop (overlay + GT)", prepared["crop_boxes"])

    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis("off")

    cax = legend_ax.inset_axes([0.02, 0.28, 0.32, 0.23])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="inferno")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0 low", "0.5", "1 high"])
    cbar.ax.tick_params(labelsize=8.2)
    cbar.set_label("TruFor manipulated-class probability (fixed 0–1 scale)", fontsize=8.4, labelpad=2)

    handles = [
        Line2D([0], [0], color=FACE_COLOR, linewidth=4.0, label="Cyan = altered face GT"),
        Line2D([0], [0], color=TEXT_COLOR, linewidth=4.0, label="Magenta = altered text GT"),
        Line2D([0], [0], color="white", linewidth=3.0, marker="s", markerfacecolor="white", markeredgecolor="black", markersize=6, label="White = top-10% TruFor anomaly contour"),
    ]
    legend = legend_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.37, 0.46), ncol=1, fontsize=8.6, frameon=True, handlelength=2.8)
    legend.get_frame().set_facecolor("#222222")
    legend.get_frame().set_edgecolor("#666666")
    for text in legend.get_texts():
        text.set_color("white")

    note = (
        "Panels shown: bona-fide counterpart | forged input + GT | frozen TruFor overlay + GT | manipulated-region crop.\n"
        "Selection is deterministic: clean-correct attack with exact same-stem/same-hardware bona-fide pair, chosen as the family medoid."
    )
    legend_ax.text(0.69, 0.46, note, ha="left", va="center", fontsize=8.1, wrap=True)

    page_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(page_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def write_readme(selection: pd.DataFrame, page_paths: List[Path], map_sources: Dict[str, str]) -> None:
    lines = [
        "# Stage 54 — Frozen clean TruFor single-example portrait pages",
        "",
        "This stage replaces the dense Stage-53 multi-example pages with a more legible layout:",
        "one selected example per portrait page, using a two-column image layout.",
        "",
        "Each page contains:",
        "",
        "1. bona-fide counterpart (same file_stem + hardware_source)",
        "2. forged Policy-C input + GT",
        "3. frozen TruFor anomaly overlay + GT",
        "4. manipulated-region crop from the frozen TruFor overlay",
        "",
        "Selection is deterministic and not visually cherry-picked.",
        "Allowed population: clean-correct attacks with exact bona-fide pair.",
        "Within each family, the example is the robust family medoid over A_union, E_union, mu_union and TruFor score.",
        "",
        "Generated pages:",
        "",
    ]
    for p in page_paths:
        lines.append(f"- {p.relative_to(OUT_ROOT)}")
    lines += ["", "Map sources used:"]
    for k, v in sorted(map_sources.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "Selected rows:", ""]
    for _, row in selection.iterrows():
        lines.append(
            f"- Example {int(row['example_number'])}: {split_label(row['eval_split'])} / {row['variant']} / {row['hardware_source']} / {row['file_stem']} — "
            f"score={float(row['trufor_score']):.3f}, E={float(row['E_union']):.3f}, μw={float(row['mu_union']):.2f}, PG={float(row['PG_union']):.0f}"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def copy_code_snapshot() -> Path:
    dst = OUT_ROOT / "code_snapshot" / Path(__file__).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), dst)
    return dst


def write_sha256sums() -> Path:
    path = OUT_ROOT / "SHA256SUMS.txt"
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p != path)
    lines = [f"{sha256_file(p)}  {p.relative_to(OUT_ROOT)}" for p in files]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_tar_bundle() -> Tuple[Path, str]:
    TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)
    if TAR_PATH.exists():
        TAR_PATH.unlink()
    if TAR_SHA_PATH.exists():
        TAR_SHA_PATH.unlink()
    with tarfile.open(TAR_PATH, mode="w") as tar:
        tar.add(OUT_ROOT, arcname=OUT_ROOT.name)
    digest = sha256_file(TAR_PATH)
    TAR_SHA_PATH.write_text(f"{digest}  {TAR_PATH.name}\n")
    return TAR_PATH, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild Stage-54 output if it already exists.")
    parser.add_argument("--map-source", choices=["auto", "stage04", "labpc"], default="auto", help="Which frozen clean map cache to use. auto prefers Stage-4 then LABPC.")
    args = parser.parse_args()

    loc, summary, provenance = load_frozen_inputs()
    if OUT_ROOT.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists:\n{OUT_ROOT}\nUse --overwrite only for an intentional rebuild.")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pairs = load_pair_index()
    selection = choose_examples(loc, summary, pairs)
    selection.to_csv(OUT_ROOT / "selected_examples.csv", index=False)
    regions = load_regions()

    print("STAGE 54 — FROZEN CLEAN TRUFOR SINGLE-EXAMPLE PANELS")
    print()
    print(selection[["example_number", "eval_split", "variant", "hardware_source", "file_stem", "trufor_score", "A_union", "E_union", "mu_union", "PG_union", "image_path", "bonafide_image_path"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    prepared: Dict[str, Dict[str, object]] = {}
    map_sources: Dict[str, str] = {}
    audit_rows = []
    max_metric_error = 0.0
    max_score_error = 0.0

    for _, row in selection.iterrows():
        key = str(row["image_path"])
        item = prepare_example(row, regions, args.map_source)
        prepared[key] = item
        map_sources[key] = item["map_source"]
        max_metric_error = max(max_metric_error, max(item["metric_errors"].values()))
        max_score_error = max(max_score_error, float(item["score_error"]))
        audit_rows.append({
            "example_number": int(row["example_number"]),
            "image_path": key,
            "resolved_map_path": resolve_rel(item["map_path"]),
            "map_source": item["map_source"],
            "A_abs_error": item["metric_errors"]["A"],
            "E_abs_error": item["metric_errors"]["E"],
            "mu_abs_error": item["metric_errors"]["mu_w"],
            "PG_abs_error": item["metric_errors"]["PG"],
            "score_abs_error": item["score_error"],
        })

    pd.DataFrame(audit_rows).to_csv(OUT_ROOT / "render_parity_audit.csv", index=False)

    pages_dir = OUT_ROOT / "pages"
    page_paths = []
    individual_panel_paths = []
    total_pages = len(selection)
    individual_dir = OUT_ROOT / "individual_panels"
    for idx, (_, row) in enumerate(selection.iterrows(), start=1):
        page_name = f"page_{idx:02d}_{safe_name(str(row['variant']))}_{safe_name(str(row['hardware_source']))}_{safe_name(str(row['file_stem']))}.png"
        page_path = pages_dir / page_name
        prep = prepared[str(row["image_path"])]
        render_page(row, prep, page_path, idx, total_pages)
        page_paths.append(page_path)
        individual_panel_paths.extend(save_individual_panels(row, prep, individual_dir))

    write_readme(selection, page_paths, map_sources)
    code_snapshot = copy_code_snapshot()
    prov = {
        "status": "PASS",
        "stage": "54_build_frozen_clean_single_example_panels",
        "purpose": "dissertation-ready frozen clean TruFor single-example portrait pages; no model inference",
        "project_git_head": git_head(),
        "source_stage4_provenance": resolve_rel(STAGE4_PROVENANCE),
        "source_stage4_provenance_sha256": sha256_file(STAGE4_PROVENANCE),
        "source_localisation": resolve_rel(LOCALISATION),
        "source_localisation_sha256": sha256_file(LOCALISATION),
        "source_localisation_summary": resolve_rel(LOCALISATION_SUMMARY),
        "source_localisation_summary_sha256": sha256_file(LOCALISATION_SUMMARY),
        "inventory": resolve_rel(INVENTORY),
        "inventory_sha256": sha256_file(INVENTORY),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "frozen_threshold": provenance.get("frozen_threshold"),
        "selection_population": "clean-correct attacks with exact same-stem/same-hardware bona-fide counterpart",
        "selection_rule": "one per family; robust IQR-scaled medoid over A_union, E_union, mu_union, trufor_score",
        "selected_n": int(len(selection)),
        "selected_by_split": selection["eval_split"].value_counts().sort_index().to_dict(),
        "selected_by_variant": selection["variant"].value_counts().sort_index().to_dict(),
        "map_source_requested": args.map_source,
        "map_sources_actual": map_sources,
        "max_metric_parity_abs_error": max_metric_error,
        "max_score_parity_abs_error": max_score_error,
        "page_paths": [resolve_rel(p) for p in page_paths],
        "individual_panel_paths": [resolve_rel(p) for p in individual_panel_paths],
        "visual_convention": {
            "face_gt": FACE_COLOR,
            "text_gt": TEXT_COLOR,
            "top10_anomaly_contour": CONTOUR_COLOR,
            "anomaly_colormap": "inferno",
            "anomaly_display_scale": [0.0, 1.0],
            "overlay_alpha": "0.58 * clipped anomaly probability",
        },
        "code_snapshot": resolve_rel(code_snapshot),
        "scientific_notes": [
            "No visual/manual cherry-picking is used.",
            "No new TruFor inference is performed.",
            "Rendered figures are qualitative only; quantitative values remain the frozen Stage-4 metrics.",
            "Bona-fide counterparts are exact matches on file_stem and hardware_source.",
            "This stage supersedes Stage-53 for dissertation legibility.",
        ],
    }
    (OUT_ROOT / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")

    checksum_path = write_sha256sums()
    tar_path, tar_digest = write_tar_bundle()

    print("STAGE 54 PASS")
    print(f"output root: {OUT_ROOT}")
    print(f"pages:       {len(page_paths)}")
    print(f"individuals: {len(individual_panel_paths)}")
    print(f"selection:   {OUT_ROOT / 'selected_examples.csv'}")
    print(f"checksums:   {checksum_path}")
    print(f"tar:         {tar_path}")
    print(f"tar sha256:  {tar_digest}")
    print(f"sha sidecar: {TAR_SHA_PATH}")


if __name__ == "__main__":
    main()
