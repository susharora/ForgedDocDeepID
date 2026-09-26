#!/usr/bin/env python3
"""Stage 53 — dissertation-ready frozen clean TruFor qualitative portrait panels.

No model inference is performed. The script packages already-frozen clean
TruFor outputs into report-ready portrait figures.

Five deterministic examples are selected, one per family:
  DEV: digital_1, digital_2
  OFFICIAL TEST: digital_3, facedancer, textdiffuserft_bfei

Each example occupies one row with three columns:
  1) exact bona-fide counterpart (same file_stem + hardware_source)
  2) frozen Policy-C forged image + GT rectangles only
  3) frozen TruFor anomaly overlay + same GT rectangles

Visual convention matches the ResNet/adversarial figures:
  CYAN    = altered face GT
  MAGENTA = altered text GT
  WHITE   = top-10% TruFor anomaly contour
  INFERNO = manipulated-class probability on a fixed 0..1 scale

Two portrait alternatives are generated:
  - 3 rows x 3 columns = 9 image cells/page
  - 2 rows x 3 columns = 6 image cells/page

Outputs include a TAR bundle and SHA256 sidecar.
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

FROZEN_ROOT = (
    ROOT / "output" / "trufor_policy_c_frozen_protocol"
    / "stage04_clean_evaluation_accuracy"
)
LOCALISATION = FROZEN_ROOT / "localisation_per_image.csv"
LOCALISATION_SUMMARY = FROZEN_ROOT / "localisation_summary.csv"
STAGE4_PROVENANCE = FROZEN_ROOT / "stage04_provenance.json"
INVENTORY = ROOT / "output" / "fantasyid_inventory_2026-09-05_023126.xlsx"
PAIR_INDEXES = [
    ROOT / "output" / "policy_c_cache_index.csv",
    ROOT / "output" / "fantasyid_official_test_policy_c_index.csv",
]
LAB_MAP_ROOT = (
    ROOT / "output" / "LABPC" / "trufor_pretrained_policy_c_native" / "maps"
)
OUT_ROOT = (
    ROOT / "output" / "trufor_policy_c_frozen_protocol"
    / "stage53_clean_qualitative_portrait_panels"
)
TRANSFER_ROOT = ROOT / "analysis_transfer_bundles"
TAR_PATH = TRANSFER_ROOT / "trufor_frozen_clean_qualitative_portrait_panels.tar"
TAR_SHA_PATH = Path(str(TAR_PATH) + ".sha256")

EXPECTED_VARIANTS = [
    "digital_1",
    "digital_2",
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
]
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
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69
DPI = 300


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def require(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file missing:\n{path}")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_frozen_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    for path in [LOCALISATION, LOCALISATION_SUMMARY, STAGE4_PROVENANCE, INVENTORY, *PAIR_INDEXES]:
        require(path)

    loc = pd.read_csv(LOCALISATION, keep_default_na=False)
    summary = pd.read_csv(LOCALISATION_SUMMARY, keep_default_na=False)
    provenance = json.loads(STAGE4_PROVENANCE.read_text())

    if provenance.get("status") != "PASS":
        raise RuntimeError("Stage-4 provenance is not PASS")
    if len(loc) != 1391:
        raise RuntimeError(f"Expected 1391 attack rows, got {len(loc)}")
    if loc["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in localisation table")

    return loc, summary, provenance


def load_pair_index() -> pd.DataFrame:
    frames = []
    for path in PAIR_INDEXES:
        frame = pd.read_csv(path, keep_default_na=False).copy()
        if "split" not in frame.columns:
            frame["split"] = "official_test"

        bona = (
            (pd.to_numeric(frame["label"], errors="coerce") == 0)
            | (frame["traffic_type"].astype(str).str.strip().str.lower() == "bonafide")
        )
        frame = frame.loc[bona].copy()
        frame["pair_eval_split"] = np.where(
            frame["split"].astype(str) == "dev_val", "dev_val", "official_test"
        )
        frame["source_index"] = rel(path)
        frames.append(frame[[
            "image_path", "cache_path", "file_stem", "hardware_source",
            "pair_eval_split", "source_index",
        ]])

    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.loc[pairs["cache_path"].astype(str).map(lambda x: (ROOT / x).is_file())].copy()
    if pairs.empty:
        raise RuntimeError("No existing bona-fide Policy-C cache files found")
    return pairs


def attach_pairs(attacks: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    groups: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for key, group in pairs.groupby(["file_stem", "hardware_source", "pair_eval_split"], sort=False):
        groups[(str(key[0]), str(key[1]), str(key[2]))] = group.copy()

    records = []
    for _, row in attacks.iterrows():
        key = (str(row["file_stem"]), str(row["hardware_source"]), str(row["eval_split"]))
        g = groups.get(key)
        rec = row.to_dict()
        if g is None or g.empty:
            rec.update(
                has_exact_bonafide_pair=False,
                bonafide_image_path="",
                bonafide_cache_path="",
                bonafide_pair_source_index="",
            )
        else:
            chosen = g.sort_values(["cache_path", "image_path"], kind="stable").iloc[0]
            rec.update(
                has_exact_bonafide_pair=True,
                bonafide_image_path=str(chosen["image_path"]),
                bonafide_cache_path=str(chosen["cache_path"]),
                bonafide_pair_source_index=str(chosen["source_index"]),
            )
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
        scale = float(x.quantile(0.75) - x.quantile(0.25))
        if not math.isfinite(scale) or scale <= 1e-12:
            mad = float((x - median).abs().median())
            scale = mad if mad > 1e-12 else 1.0
        score = score + (x - median).abs() / scale
    return score


def choose_examples(loc: pd.DataFrame, summary: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    work = loc.loc[as_bool(loc["clean_correct_at_frozen_threshold"])].copy()
    work = attach_pairs(work, pairs)
    work = work.loc[as_bool(work["has_exact_bonafide_pair"])].copy()

    fam_summary = summary.loc[
        (summary["population"].astype(str) == "clean_correct_attacks")
        & (summary["scope"].astype(str) == "union")
    ].copy()

    chosen_rows = []
    for variant in EXPECTED_VARIANTS:
        split = EXPECTED_SPLIT_BY_VARIANT[variant]
        g = work.loc[
            (work["variant"].astype(str) == variant)
            & (work["eval_split"].astype(str) == split)
        ].copy()
        if g.empty:
            raise RuntimeError(f"No clean-correct exact-paired candidates for {variant}")

        g["_medoid_score"] = robust_medoid_score(g)
        chosen = g.sort_values(["_medoid_score", "image_path"], kind="stable").iloc[0].copy()

        s = fam_summary.loc[
            (fam_summary["variant"].astype(str) == variant)
            & (fam_summary["eval_split"].astype(str) == split)
        ]
        if len(s) != 1:
            raise RuntimeError(f"Expected one frozen family summary row for {variant}; got {len(s)}")
        s = s.iloc[0]
        chosen["family_mean_A"] = float(s["mean_A"])
        chosen["family_mean_E"] = float(s["mean_E"])
        chosen["family_mean_mu_w"] = float(s["mean_mu_w"])
        chosen["family_mean_PG"] = float(s["mean_PG"])
        chosen["selection_reason"] = (
            "clean-correct exact-paired family medoid: minimum robust IQR-scaled "
            "distance to family medians of A_union, E_union, mu_union and TruFor score"
        )
        chosen_rows.append(chosen)

    out = pd.DataFrame(chosen_rows).reset_index(drop=True)
    out.insert(0, "example_number", np.arange(1, len(out) + 1))
    if len(out) != 5 or out["image_path"].duplicated().any():
        raise RuntimeError("Five unique family representatives were not produced")
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
    rows = regions.loc[
        (regions["image_path"] == str(image_path))
        & (regions["region_provenance_raw"] == "altered")
    ]
    if rows.empty:
        raise RuntimeError(f"No altered Regions rows for {image_path}")

    out = []
    for _, row in rows.iterrows():
        x0 = int(round(float(row["x"])))
        y0 = int(round(float(row["y"])))
        x1 = x0 + int(round(float(row["width"])))
        y1 = y0 + int(round(float(row["height"])))
        x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
        y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        kind = "face" if str(row["field_name"]).lower() == "face" else "text"
        out.append({"box": (x0, y0, x1, y1), "kind": kind})

    if not out:
        raise RuntimeError(f"No valid altered rectangles after clipping: {image_path}")
    return out


def union_mask(boxes: Sequence[Dict[str, object]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        mask[y0:y1, x0:x1] = True
    return mask


def metrics_from_map(amap: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    area = int(mask.sum())
    if area <= 0:
        raise RuntimeError("Empty GT mask")
    A = float(area / mask.size)
    values = np.asarray(amap, dtype=np.float64)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("Invalid anomaly-map mass")
    E = float(values[mask].sum() / total)
    mu = float(E / A)
    PG = float(mask.reshape(-1)[int(np.nanargmax(amap))])
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
        f"No frozen clean TruFor map found for {row['image_path']}.\nAttempted:\n{attempted}\n"
        "Restore the Stage-4 map cache or the LABPC clean-map cache archived to OneDrive."
    )


def load_map(row: pd.Series, map_source: str) -> Tuple[np.ndarray, float, Path, str]:
    path, label = resolve_map_path(row, map_source)
    with np.load(path, allow_pickle=False) as data:
        amap = np.asarray(data["map"], dtype=np.float32)
        score = float(np.asarray(data["score"]).item()) if "score" in data.files else float(row["trufor_score"])
    return amap, score, path, label


def inferno_rgb(amap: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(amap, dtype=np.float32), 0.0, 1.0)
    return np.rint(plt.get_cmap("inferno")(x)[..., :3] * 255.0).astype(np.uint8)


def mask_boundary(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
            & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
        eroded[1:-1, 1:-1] = inner
        eroded[0, :] = eroded[-1, :] = False
        eroded[:, 0] = eroded[:, -1] = False
    boundary = mask & ~eroded
    for _ in range(max(0, thickness - 1)):
        d = boundary.copy()
        d[1:, :] |= boundary[:-1, :]
        d[:-1, :] |= boundary[1:, :]
        d[:, 1:] |= boundary[:, :-1]
        d[:, :-1] |= boundary[:, 1:]
        boundary = d
    return boundary


def anomaly_overlay(rgb: np.ndarray, amap: np.ndarray) -> np.ndarray:
    heat = inferno_rgb(amap)
    values = np.clip(np.asarray(amap, dtype=np.float32), 0.0, 1.0)
    alpha = OVERLAY_ALPHA_SCALE * values
    overlay = (
        rgb.astype(np.float32) * (1.0 - alpha[..., None])
        + heat.astype(np.float32) * alpha[..., None]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    q = float(np.quantile(values, CONTOUR_QUANTILE))
    if np.isfinite(q) and float(values.max()) > float(values.min()) + 1e-12:
        overlay[mask_boundary(values >= q, thickness=2)] = np.array([255, 255, 255], dtype=np.uint8)
    return overlay


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def draw_gt(ax, boxes: Sequence[Dict[str, object]]) -> None:
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        color = FACE_COLOR if item["kind"] == "face" else TEXT_COLOR
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=2.5, zorder=10,
        ))


def prepare_example(row: pd.Series, regions: pd.DataFrame, map_source: str) -> Dict[str, object]:
    attack_path = ROOT / str(row["cache_path"])
    bona_path = ROOT / str(row["bonafide_cache_path"])
    require(attack_path)
    require(bona_path)

    attack_rgb = read_rgb(attack_path)
    bona_rgb = read_rgb(bona_path)
    amap, map_score, map_path, source_label = load_map(row, map_source)
    height, width = attack_rgb.shape[:2]
    if amap.shape != (height, width):
        raise RuntimeError(
            f"Map/image geometry mismatch for {row['image_path']}: image={attack_rgb.shape[:2]} map={amap.shape}"
        )

    boxes = boxes_for_image(regions, str(row["image_path"]), width, height)
    mask = union_mask(boxes, height, width)
    got = metrics_from_map(amap, mask)
    expected = {
        "A": float(row["A_union"]),
        "E": float(row["E_union"]),
        "mu_w": float(row["mu_union"]),
        "PG": float(row["PG_union"]),
    }
    errors = {k: abs(got[k] - expected[k]) for k in expected}
    if max(errors.values()) > METRIC_PARITY_ATOL:
        raise RuntimeError(
            f"Frozen metric parity failed for {row['image_path']}: source={source_label} errors={errors}"
        )
    score_error = abs(map_score - float(row["trufor_score"]))
    if score_error > SCORE_PARITY_ATOL:
        raise RuntimeError(f"Frozen score parity failed: {row['image_path']} error={score_error}")

    return {
        "bona_rgb": bona_rgb,
        "attack_rgb": attack_rgb,
        "overlay_rgb": anomaly_overlay(attack_rgb, amap),
        "boxes": boxes,
        "map_path": map_path,
        "map_source": source_label,
        "metric_errors": errors,
        "score_error": score_error,
    }


def split_label(value: str) -> str:
    return "DEV" if str(value) == "dev_val" else "OFFICIAL TEST"


def render_page(rows: pd.DataFrame, prepared: Dict[str, Dict[str, object]], destination: Path,
                rows_per_page: int, page_number: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(A4_WIDTH_IN, A4_HEIGHT_IN), facecolor="white")
    gs = fig.add_gridspec(
        rows_per_page + 1, 3,
        left=0.035, right=0.985, top=0.945, bottom=0.035,
        hspace=0.34, wspace=0.055,
        height_ratios=[1.0] * rows_per_page + [0.20],
    )
    fig.suptitle(
        f"Frozen clean TruFor qualitative examples — page {page_number}/{total_pages}",
        fontsize=12.5, fontweight="bold", y=0.982,
    )

    for rpos in range(rows_per_page):
        if rpos >= len(rows):
            for c in range(3):
                ax = fig.add_subplot(gs[rpos, c]); ax.axis("off")
            continue

        row = rows.iloc[rpos]
        item = prepared[str(row["image_path"])]
        split = split_label(row["eval_split"])
        variant = str(row["variant"])
        hardware = str(row["hardware_source"])
        metrics = (
            f"A={float(row['A_union']):.3f}  E={float(row['E_union']):.3f}  "
            f"μw={float(row['mu_union']):.2f}  PG={int(round(float(row['PG_union'])))}"
        )
        fam = (
            f"family E={float(row['family_mean_E']):.3f}, "
            f"μw={float(row['family_mean_mu_w']):.2f}, PG={float(row['family_mean_PG']):.2f}"
        )

        ax = fig.add_subplot(gs[rpos, 0])
        ax.imshow(item["bona_rgb"]); ax.axis("off")
        ax.set_title(f"{split} • {variant} • {hardware}\nBona fide counterpart", fontsize=8.3, pad=4)

        ax = fig.add_subplot(gs[rpos, 1])
        ax.imshow(item["attack_rgb"]); draw_gt(ax, item["boxes"]); ax.axis("off")
        ax.set_title("Forged Policy-C input + GT\n" + metrics, fontsize=8.0, pad=4)

        ax = fig.add_subplot(gs[rpos, 2])
        ax.imshow(item["overlay_rgb"]); draw_gt(ax, item["boxes"]); ax.axis("off")
        ax.set_title(
            f"Frozen TruFor overlay + GT\nscore={float(row['trufor_score']):.3f}  {fam}",
            fontsize=7.5, pad=4,
        )

    legend_ax = fig.add_subplot(gs[rows_per_page, :]); legend_ax.axis("off")
    cax = legend_ax.inset_axes([0.02, 0.30, 0.35, 0.36])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="inferno"); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0]); cbar.set_ticklabels(["0 low", "0.5", "1 high"])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("TruFor manipulated-class probability (fixed 0–1 scale)", fontsize=7.5, labelpad=1)

    handles = [
        Line2D([0], [0], color=FACE_COLOR, linewidth=3.5, label="Cyan = altered face GT"),
        Line2D([0], [0], color=TEXT_COLOR, linewidth=3.5, label="Magenta = altered text GT"),
        Line2D([0], [0], color="white", linewidth=3.0, marker="s", markerfacecolor="white",
               markeredgecolor="black", markersize=5, label="White = top-10% TruFor anomaly contour"),
    ]
    legend = legend_ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.40, 0.50),
        ncol=1, fontsize=7.4, frameon=True, handlelength=2.6,
    )
    legend.get_frame().set_facecolor("#222222"); legend.get_frame().set_edgecolor("#666666")
    for text in legend.get_texts(): text.set_color("white")

    legend_ax.text(
        0.72, 0.58,
        "Columns: exact bona fide counterpart | forged + GT | frozen TruFor overlay + GT",
        fontsize=7.1, va="center", ha="left", transform=legend_ax.transAxes,
    )
    legend_ax.text(
        0.72, 0.25,
        "Deterministic clean-correct family medoids with exact same-stem/same-hardware bona-fide pairs; no visual cherry-picking.",
        fontsize=6.7, va="center", ha="left", wrap=True, transform=legend_ax.transAxes,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=DPI, facecolor="white")
    plt.close(fig)


def render_layout(selection: pd.DataFrame, prepared: Dict[str, Dict[str, object]], rows_per_page: int,
                  out_dir: Path, prefix: str) -> List[Path]:
    total_pages = int(math.ceil(len(selection) / rows_per_page))
    paths = []
    for p in range(total_pages):
        subset = selection.iloc[p * rows_per_page:(p + 1) * rows_per_page].reset_index(drop=True)
        dest = out_dir / f"{prefix}_page_{p + 1:02d}.png"
        render_page(subset, prepared, dest, rows_per_page, p + 1, total_pages)
        paths.append(dest)
    return paths


def write_readme(selection: pd.DataFrame, layout3: List[Path], layout2: List[Path]) -> None:
    lines = [
        "# Frozen clean TruFor dissertation qualitative panels", "",
        "Five deterministic examples are included:", "",
        "- DEV: digital_1, digital_2",
        "- OFFICIAL TEST: digital_3, facedancer, textdiffuserft_bfei", "",
        "Each row is: exact bona fide counterpart | forged + GT | frozen TruFor overlay + GT.", "",
        "Visual convention:", "",
        "- CYAN = altered face GT",
        "- MAGENTA = altered text GT",
        "- WHITE = top-10% TruFor anomaly contour",
        "- Inferno = manipulated-class probability, fixed 0–1 scale", "",
        "Selection is deterministic: clean-correct attacks only, exact same-stem/same-hardware bona-fide pair required, then family medoid over A, E, mu_w and clean score.", "",
        f"3x3 portrait pages: {len(layout3)}",
        f"2x3 portrait pages: {len(layout2)}", "",
        "Selected examples:", "",
    ]
    for _, row in selection.iterrows():
        lines.append(
            f"- {int(row['example_number'])}: {split_label(row['eval_split'])} / {row['variant']} / "
            f"{row['hardware_source']} / {row['file_stem']} — E={float(row['E_union']):.3f}, "
            f"mu_w={float(row['mu_union']):.2f}, PG={float(row['PG_union']):.0f}"
        )
    lines += [
        "", "No new model inference is performed. Cached map/score/localisation parity is checked before rendering.",
        "The PNGs are qualitative report figures; quantitative science remains the frozen Stage-4 outputs.", "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def copy_code_snapshot() -> Path:
    dest = OUT_ROOT / "code_snapshot" / Path(__file__).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), dest)
    return dest


def write_sha256sums() -> Path:
    out = OUT_ROOT / "SHA256SUMS.txt"
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p != out)
    out.write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(OUT_ROOT)}" for p in files) + "\n")
    return out


def package_tar() -> Tuple[Path, str]:
    TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)
    for p in [TAR_PATH, TAR_SHA_PATH]:
        if p.exists(): p.unlink()
    with tarfile.open(TAR_PATH, "w") as tar:
        tar.add(OUT_ROOT, arcname=OUT_ROOT.name)
    digest = sha256_file(TAR_PATH)
    TAR_SHA_PATH.write_text(f"{digest}  {TAR_PATH.name}\n")
    return TAR_PATH, digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--map-source", choices=["auto", "stage04", "labpc"], default="auto")
    args = ap.parse_args()

    loc, summary, stage4 = load_frozen_inputs()
    if OUT_ROOT.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists:\n{OUT_ROOT}\nUse --overwrite to rebuild intentionally.")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    selection = choose_examples(loc, summary, load_pair_index())
    selection_path = OUT_ROOT / "selected_examples.csv"
    selection.to_csv(selection_path, index=False)

    regions = load_regions()
    prepared: Dict[str, Dict[str, object]] = {}
    audit = []
    map_sources: Dict[str, str] = {}

    print("STAGE 53 — FROZEN CLEAN TRUFOR QUALITATIVE PORTRAIT PANELS")
    print(selection[[
        "example_number", "eval_split", "variant", "hardware_source", "file_stem",
        "trufor_score", "A_union", "E_union", "mu_union", "PG_union", "image_path",
        "bonafide_image_path",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    for _, row in selection.iterrows():
        item = prepare_example(row, regions, args.map_source)
        key = str(row["image_path"])
        prepared[key] = item
        map_sources[key] = item["map_source"]
        audit.append({
            "example_number": int(row["example_number"]),
            "image_path": key,
            "resolved_map_path": rel(item["map_path"]),
            "map_source": item["map_source"],
            "A_abs_error": item["metric_errors"]["A"],
            "E_abs_error": item["metric_errors"]["E"],
            "mu_abs_error": item["metric_errors"]["mu_w"],
            "PG_abs_error": item["metric_errors"]["PG"],
            "score_abs_error": item["score_error"],
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT_ROOT / "render_parity_audit.csv", index=False)

    layout3 = render_layout(
        selection, prepared, rows_per_page=3,
        out_dir=OUT_ROOT / "portrait_3x3", prefix="frozen_clean_trufor_3x3",
    )
    layout2 = render_layout(
        selection, prepared, rows_per_page=2,
        out_dir=OUT_ROOT / "portrait_2x3", prefix="frozen_clean_trufor_2x3",
    )

    write_readme(selection, layout3, layout2)
    code_snapshot = copy_code_snapshot()

    provenance = {
        "status": "PASS",
        "stage": "53_build_frozen_clean_qualitative_portrait_panels",
        "project_git_head": git_head(),
        "purpose": "dissertation-ready frozen clean TruFor qualitative panels; no new inference",
        "source_stage4_provenance": rel(STAGE4_PROVENANCE),
        "source_stage4_provenance_sha256": sha256_file(STAGE4_PROVENANCE),
        "source_localisation": rel(LOCALISATION),
        "source_localisation_sha256": sha256_file(LOCALISATION),
        "source_localisation_summary": rel(LOCALISATION_SUMMARY),
        "source_localisation_summary_sha256": sha256_file(LOCALISATION_SUMMARY),
        "inventory": rel(INVENTORY),
        "inventory_sha256": sha256_file(INVENTORY),
        "checkpoint_sha256": stage4.get("checkpoint_sha256"),
        "frozen_threshold": stage4.get("frozen_threshold"),
        "selection_population": "clean-correct attacks with exact same-stem/same-hardware bona-fide counterpart",
        "selection_rule": "one per family; robust IQR-scaled medoid over A_union, E_union, mu_union and trufor_score",
        "selected_n": int(len(selection)),
        "selected_by_split": selection["eval_split"].value_counts().sort_index().to_dict(),
        "selected_by_variant": selection["variant"].value_counts().sort_index().to_dict(),
        "map_source_requested": args.map_source,
        "map_sources_actual": map_sources,
        "max_metric_parity_abs_error": float(audit_df[["A_abs_error", "E_abs_error", "mu_abs_error", "PG_abs_error"]].to_numpy().max()),
        "max_score_parity_abs_error": float(audit_df["score_abs_error"].max()),
        "visual_convention": {
            "face_gt": FACE_COLOR,
            "text_gt": TEXT_COLOR,
            "top10_anomaly_contour": CONTOUR_COLOR,
            "anomaly_colormap": "inferno",
            "anomaly_display_scale": [0.0, 1.0],
            "overlay_alpha": "0.58 * anomaly probability",
            "padding_boundary": "not applicable: TruFor clean inference is native-resolution with no artificial padding",
        },
        "layouts": {
            "portrait_3x3": [rel(p) for p in layout3],
            "portrait_2x3": [rel(p) for p in layout2],
        },
        "code_snapshot": rel(code_snapshot),
        "scientific_notes": [
            "No visual/manual cherry-picking.",
            "No new TruFor inference.",
            "Bona-fide pair is exact on file_stem and hardware_source.",
            "Anomaly maps use a fixed 0..1 display scale.",
            "Rendered figures are qualitative only; frozen Stage-4 metrics remain authoritative.",
        ],
    }
    (OUT_ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum_path = write_sha256sums()
    tar_path, tar_digest = package_tar()

    print("STAGE 53 PASS")
    print(f"output root: {OUT_ROOT}")
    print(f"3x3 pages:   {len(layout3)}")
    print(f"2x3 pages:   {len(layout2)}")
    print(f"selection:   {selection_path}")
    print(f"checksums:   {checksum_path}")
    print(f"tar:         {tar_path}")
    print(f"tar sha256:  {tar_digest}")
    print(f"sha sidecar: {TAR_SHA_PATH}")


if __name__ == "__main__":
    main()
