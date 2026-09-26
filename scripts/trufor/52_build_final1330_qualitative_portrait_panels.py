#!/usr/bin/env python3
"""
Stage 52 — rebuild final TruFor qualitative examples as portrait 3x2 panels.

Purpose
-------
Create dissertation-ready, portrait-oriented qualitative panels from the final
1330-image TruFor adversarial-localisation results.

Each selected example is presented as THREE ROWS x TWO COLUMNS:

    Row 1: bona fide counterpart | clean forged/attack input
    Row 2: clean TruFor anomaly + GT | adversarial TruFor anomaly + GT
    Row 3: adversarial image preview | 100x absolute perturbation

Changes relative to Stage 50
----------------------------
- portrait 3x2 layout rather than 2x3 landscape;
- requires an exact bona fide counterpart matched by file_stem + hardware;
- uses a thicker cyan GT contour for visibility;
- includes an in-figure legend:
    * fixed 0..1 Inferno colour bar for TruFor anomaly score;
    * cyan line = manipulated-region ground truth;
    * note that perturbation is amplified 100x for display;
- anomaly maps are NEVER independently renormalised: clean and adversarial
  maps use the same fixed 0..1 colour scale.

Selection
---------
Five deterministic manipulation-family representatives:
    digital_1, digital_2, digital_3, facedancer, textdiffuserft_bfei

Candidates must have an exact bona fide counterpart with the same file_stem
and hardware_source. If a family contains primary DCEC examples, the
representative is selected from its DCEC subset; otherwise the paired full
family is used. Selection is based only on clean-state median proximity using
A, E_clean, RRA_clean and clean_score.

A sixth example is an explicitly labelled residual-evidence stress case,
selected among paired primary DCEC/DCEW examples as the row with the largest:

    max(E_adv / tau_E, RRA_adv / tau_RRA)

The stress example is NOT described as representative.

Scientific note
---------------
The adversarial PNG is only an 8-bit display preview reconstructed from the
exact retained float `adv_x_model`. All quantitative science continues to use
the retained NPZ tensors/maps.
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
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "final_1330"
)

AVAILABLE = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseAB_available"
)

ATTACK_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_attack"
)

EVIDENCE = (
    ANALYSIS
    / "46_final1330_evidence_per_image.csv"
)

MASTER = (
    AVAILABLE
    / "44_available_master_manifest.csv"
)

FULL_SELECTION = (
    ATTACK_ROOT
    / "full_population"
    / "full_population_selection.csv"
)

RUNTIME = (
    ANALYSIS
    / "47_final1330_runtime_per_image.csv"
)

PAIR_INDEX_CANDIDATES = [
    ROOT / "output" / "policy_c_cache_index.csv",
    ROOT / "output" / "fantasyid_official_test_policy_c_index.csv",
]

OUT_ROOT = (
    ANALYSIS
    / "52_qualitative_portrait_panels"
)

TRANSFER_ROOT = (
    ROOT
    / "analysis_transfer_bundles"
)

TAR_PATH = (
    TRANSFER_ROOT
    / "IMTA135_trufor_final1330_qualitative_portrait_panels.tar"
)

EXPECTED_VARIANTS = [
    "digital_1",
    "digital_2",
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
]

TAU_E = 0.5
TAU_RRA = 0.5

PROTOCOL_SHA = (
    "5840aa9bee076b493c6a645e140edaad"
    "892373433c7e919e95ed4e60e44158d5"
)

CHECKPOINT_SHA = (
    "ac1d90e329a72e0d66e8665e123a19e"
    "94bfae3209c3ef8a4f9ca3b91578c7844"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return text.strip("_")[:100] or "item"


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    lookup = {
        str(c).strip().lower(): str(c)
        for c in frame.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    if required:
        raise RuntimeError(
            "Could not resolve required column. "
            f"Candidates={list(candidates)}, "
            f"available={list(frame.columns)}"
        )

    return None


def is_bonafide_frame(frame: pd.DataFrame) -> pd.Series:
    class_col = find_column(
        frame,
        [
            "class_name",
            "class",
            "label_name",
            "category",
        ],
        required=False,
    )

    label_col = find_column(
        frame,
        [
            "label",
            "target",
            "y",
        ],
        required=False,
    )

    masks: List[pd.Series] = []

    if class_col is not None:
        masks.append(
            frame[class_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "bonafide",
                    "bona_fide",
                    "bona fide",
                    "genuine",
                    "real",
                ]
            )
        )

    if label_col is not None:
        numeric = pd.to_numeric(
            frame[label_col],
            errors="coerce",
        )
        masks.append(
            numeric.eq(0)
        )

    # Path-based fallback is safe as an additional positive signal.
    image_col = find_column(
        frame,
        [
            "image_path",
            "relative_path",
            "path",
        ],
        required=False,
    )

    if image_col is not None:
        masks.append(
            frame[image_col]
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.lower()
            .str.contains(
                "/bonafide/",
                regex=False,
            )
        )

    if not masks:
        raise RuntimeError(
            "Could not identify bona fide rows in pairing index: "
            f"{list(frame.columns)}"
        )

    out = masks[0].copy()

    for mask in masks[1:]:
        out = out | mask

    return out


def normalise_pair_index(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        keep_default_na=False,
    )

    stem_col = find_column(
        frame,
        [
            "file_stem",
            "stem",
            "document_stem",
        ],
    )

    hardware_col = find_column(
        frame,
        [
            "hardware_source",
            "hardware",
            "acquisition",
            "capture_source",
        ],
    )

    cache_col = find_column(
        frame,
        [
            "cache_path",
            "processed_path",
            "model_path",
        ],
    )

    image_col = find_column(
        frame,
        [
            "image_path",
            "relative_path",
            "path",
        ],
        required=False,
    )

    split_col = find_column(
        frame,
        [
            "eval_split",
            "evaluation_split",
            "split",
            "dataset_split",
        ],
        required=False,
    )

    bona = is_bonafide_frame(frame)

    out = pd.DataFrame(
        {
            "pair_file_stem":
                frame[stem_col]
                .astype(str)
                .str.strip(),

            "pair_hardware_source":
                frame[hardware_col]
                .astype(str)
                .str.strip(),

            "pair_cache_path":
                frame[cache_col]
                .astype(str)
                .str.strip(),

            "pair_image_path":
                (
                    frame[image_col]
                    .astype(str)
                    .str.strip()
                    if image_col is not None
                    else ""
                ),

            "pair_split":
                (
                    frame[split_col]
                    .astype(str)
                    .str.strip()
                    if split_col is not None
                    else ""
                ),

            "pair_source_index":
                str(
                    path.relative_to(ROOT)
                ),

            "is_bonafide":
                bona.to_numpy(dtype=bool),
        }
    )

    out = out.loc[
        out["is_bonafide"]
    ].copy()

    out = out.loc[
        out["pair_file_stem"].str.len() > 0
    ].copy()

    out = out.loc[
        out["pair_hardware_source"].str.len() > 0
    ].copy()

    out = out.loc[
        out["pair_cache_path"].str.len() > 0
    ].copy()

    return out.reset_index(drop=True)


def load_pair_index() -> pd.DataFrame:
    frames = []

    for path in PAIR_INDEX_CANDIDATES:
        if not path.is_file():
            continue

        f = normalise_pair_index(
            path
        )

        frames.append(
            f
        )

    if not frames:
        raise RuntimeError(
            "No pairing index could be loaded. "
            f"Expected one or more of: {PAIR_INDEX_CANDIDATES}"
        )

    pairs = pd.concat(
        frames,
        ignore_index=True,
    )

    # Keep only paths that physically exist in the project tree.
    existence = []

    for value in pairs[
        "pair_cache_path"
    ].astype(str):
        existence.append(
            (ROOT / value).is_file()
        )

    pairs[
        "pair_cache_exists"
    ] = existence

    pairs = pairs.loc[
        pairs[
            "pair_cache_exists"
        ]
    ].copy()

    if len(pairs) == 0:
        raise RuntimeError(
            "Pairing indices contained no existing bona fide cache images."
        )

    return pairs.reset_index(drop=True)


def resolve_exact_bonafide_pairs(
    evidence: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add an exact same-stem + same-hardware bona fide match to each attack row.

    If several exact matches exist, prefer a split match where possible, then
    deterministic lexicographic ordering by cache path.
    """
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}

    for key, g in pairs.groupby(
        [
            "pair_file_stem",
            "pair_hardware_source",
        ],
        sort=False,
    ):
        groups[
            (
                str(key[0]),
                str(key[1]),
            )
        ] = g.copy()

    records = []

    for _, row in evidence.iterrows():
        stem = str(
            row["file_stem"]
        ).strip()

        hardware = str(
            row["hardware_source"]
        ).strip()

        split = str(
            row["eval_split"]
        ).strip()

        g = groups.get(
            (
                stem,
                hardware,
            )
        )

        r = row.to_dict()

        if g is None or len(g) == 0:
            r["has_exact_bonafide_pair"] = False
            r["bonafide_cache_path"] = ""
            r["bonafide_image_path"] = ""
            r["bonafide_pair_source_index"] = ""
            r["bonafide_pair_split"] = ""
            records.append(r)
            continue

        candidates = g.copy()

        same_split = candidates.loc[
            candidates[
                "pair_split"
            ].astype(str)
            == split
        ]

        if len(same_split):
            candidates = same_split

        candidates = candidates.sort_values(
            [
                "pair_cache_path",
                "pair_image_path",
            ],
            ascending=True,
        )

        chosen = candidates.iloc[0]

        r["has_exact_bonafide_pair"] = True
        r["bonafide_cache_path"] = str(
            chosen[
                "pair_cache_path"
            ]
        )
        r["bonafide_image_path"] = str(
            chosen[
                "pair_image_path"
            ]
        )
        r["bonafide_pair_source_index"] = str(
            chosen[
                "pair_source_index"
            ]
        )
        r["bonafide_pair_split"] = str(
            chosen[
                "pair_split"
            ]
        )

        records.append(r)

    return pd.DataFrame(
        records
    )


def robust_clean_medoid_score(
    frame: pd.DataFrame,
) -> pd.Series:
    metrics = [
        "A_union",
        "E_clean",
        "RRA_clean",
        "clean_score",
    ]

    score = pd.Series(
        np.zeros(
            len(frame),
            dtype=np.float64,
        ),
        index=frame.index,
    )

    for col in metrics:
        x = pd.to_numeric(
            frame[col],
            errors="coerce",
        )

        if x.isna().any():
            raise RuntimeError(
                f"Selection metric {col} contains NaN"
            )

        median = float(
            x.median()
        )

        q25 = float(
            x.quantile(0.25)
        )

        q75 = float(
            x.quantile(0.75)
        )

        iqr = q75 - q25

        if (
            not math.isfinite(iqr)
            or iqr <= 1e-12
        ):
            mad = float(
                (
                    x - median
                )
                .abs()
                .median()
            )
            scale = (
                mad
                if mad > 1e-12
                else 1.0
            )
        else:
            scale = iqr

        score = (
            score
            + (
                x - median
            ).abs()
            / scale
        )

    return score


def choose_examples(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    paired = frame.loc[
        as_bool(
            frame[
                "has_exact_bonafide_pair"
            ]
        )
    ].copy()

    if len(paired) == 0:
        raise RuntimeError(
            "No exact bona fide pairs were resolved."
        )

    print(
        "exact paired attack rows:",
        len(paired),
        "/",
        len(frame),
    )

    print()

    print(
        "paired rows by variant:"
    )

    print(
        paired[
            "variant"
        ]
        .value_counts()
        .to_string()
    )

    selections = []
    used = set()

    for variant in EXPECTED_VARIANTS:
        g = paired.loc[
            paired[
                "variant"
            ].astype(str)
            == variant
        ].copy()

        if len(g) == 0:
            raise RuntimeError(
                f"No exact-paired candidates available for variant {variant}"
            )

        dcec = as_bool(
            g[
                "DCEC"
            ]
        )

        if dcec.any():
            candidates = (
                g.loc[
                    dcec
                ]
                .copy()
            )
            scope = (
                "exact_pair_primary_DCEC_subset"
            )
        else:
            candidates = g.copy()
            scope = (
                "exact_pair_full_variant_no_primary_DCEC"
            )

        candidates[
            "_selection_score"
        ] = robust_clean_medoid_score(
            candidates
        )

        candidates = candidates.sort_values(
            [
                "_selection_score",
                "image_path",
            ],
            ascending=[
                True,
                True,
            ],
        )

        chosen = (
            candidates
            .iloc[0]
            .copy()
        )

        used.add(
            str(
                chosen[
                    "image_path"
                ]
            )
        )

        chosen[
            "_selection_role"
        ] = "variant_representative"

        chosen[
            "_selection_scope"
        ] = scope

        chosen[
            "_selection_note"
        ] = (
            "Deterministic clean-state median-proximity selection "
            "restricted to exact same-stem/same-hardware bona fide pairs."
        )

        selections.append(
            chosen
        )

    stress = paired.loc[
        as_bool(
            paired[
                "DCEC"
            ]
        )
        & as_bool(
            paired[
                "DCEW"
            ]
        )
    ].copy()

    stress = stress.loc[
        ~stress[
            "image_path"
        ]
        .astype(str)
        .isin(
            used
        )
    ].copy()

    if len(stress) == 0:
        raise RuntimeError(
            "No paired non-duplicate primary DCEC/DCEW stress candidate."
        )

    stress[
        "_residual_fraction"
    ] = np.maximum(
        pd.to_numeric(
            stress[
                "E_adv"
            ]
        )
        / TAU_E,
        pd.to_numeric(
            stress[
                "RRA_adv"
            ]
        )
        / TAU_RRA,
    )

    stress = stress.sort_values(
        [
            "_residual_fraction",
            "image_path",
        ],
        ascending=[
            False,
            True,
        ],
    )

    chosen = (
        stress
        .iloc[0]
        .copy()
    )

    chosen[
        "_selection_score"
    ] = np.nan

    chosen[
        "_selection_role"
    ] = (
        "largest_residual_evidence_DCEC"
    )

    chosen[
        "_selection_scope"
    ] = (
        "exact_pair_primary_DCEC_excluding_family_representatives"
    )

    chosen[
        "_selection_note"
    ] = (
        "Explicit stress case: paired primary DCEC/DCEW example "
        "with largest remaining evidence fraction relative to tau_E/tau_RRA."
    )

    selections.append(
        chosen
    )

    out = pd.DataFrame(
        selections
    ).reset_index(
        drop=True
    )

    out.insert(
        0,
        "example_number",
        np.arange(
            1,
            len(out) + 1,
        ),
    )

    if out[
        "image_path"
    ].duplicated().any():
        raise RuntimeError(
            "Selected example images are not unique."
        )

    return out


def map_to_rgb(
    values: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=np.float32,
    )

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2-D map, got {x.shape}"
        )

    # Fixed scientific display scale: no per-image renormalisation.
    x = np.clip(
        x,
        0.0,
        1.0,
    )

    rgba = plt.get_cmap(
        "inferno"
    )(x)

    return np.rint(
        rgba[
            :,
            :,
            :3,
        ]
        * 255.0
    ).astype(
        np.uint8
    )


def one_step_dilate(
    mask: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        mask,
        dtype=bool,
    )

    p = np.pad(
        x,
        1,
        mode="constant",
        constant_values=False,
    )

    out = np.zeros_like(
        x,
        dtype=bool,
    )

    for dy in range(3):
        for dx in range(3):
            out |= p[
                dy:dy + x.shape[0],
                dx:dx + x.shape[1],
            ]

    return out


def mask_boundary(
    mask: np.ndarray,
) -> np.ndarray:
    m = np.asarray(
        mask,
        dtype=bool,
    )

    if m.ndim != 2:
        raise ValueError(
            f"Expected 2-D mask, got {m.shape}"
        )

    # Four-neighbour erosion for the base inner boundary.
    up = np.zeros_like(m)
    down = np.zeros_like(m)
    left = np.zeros_like(m)
    right = np.zeros_like(m)

    up[1:, :] = m[:-1, :]
    down[:-1, :] = m[1:, :]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]

    eroded = (
        m
        & up
        & down
        & left
        & right
    )

    return (
        m
        & ~eroded
    )


def thick_mask_boundary(
    mask: np.ndarray,
    thickness_px: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    m = np.asarray(
        mask,
        dtype=bool,
    )

    h, w = m.shape

    if thickness_px is None:
        # Scales to document resolution while remaining clearly visible.
        thickness_px = max(
            5,
            int(
                round(
                    min(
                        h,
                        w,
                    )
                    * 0.004
                )
            ),
        )

    boundary = mask_boundary(
        m
    )

    thick = boundary.copy()

    # Dilation radius is roughly thickness_px/2 on each side.
    radius = max(
        1,
        int(
            math.ceil(
                thickness_px / 2
            )
        ),
    )

    for _ in range(radius):
        thick = one_step_dilate(
            thick
        )

    return (
        thick,
        thickness_px,
    )


def paint_gt_contour(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    out = np.asarray(
        rgb,
        dtype=np.uint8,
    ).copy()

    boundary, thickness = (
        thick_mask_boundary(
            mask
        )
    )

    # High-visibility cyan/light-blue GT contour.
    out[
        boundary,
        0,
    ] = 0

    out[
        boundary,
        1,
    ] = 255

    out[
        boundary,
        2,
    ] = 255

    return (
        out,
        thickness,
    )


def overlay_heat_gt(
    base_rgb: np.ndarray,
    amap: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, int]:
    base = np.asarray(
        base_rgb,
        dtype=np.uint8,
    )

    heat = map_to_rgb(
        amap
    )

    if heat.shape != base.shape:
        raise ValueError(
            f"Heat/base geometry mismatch: {heat.shape} vs {base.shape}"
        )

    out = np.rint(
        (
            1.0 - alpha
        )
        * base.astype(
            np.float32
        )
        + alpha
        * heat.astype(
            np.float32
        )
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    return paint_gt_contour(
        out,
        mask,
    )


def resize_to_match(
    image: np.ndarray,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    rgb = np.asarray(
        image,
        dtype=np.uint8,
    )

    target_h, target_w = (
        int(target_hw[0]),
        int(target_hw[1]),
    )

    if rgb.shape[:2] == (
        target_h,
        target_w,
    ):
        return rgb

    return np.asarray(
        Image.fromarray(
            rgb
        ).resize(
            (
                target_w,
                target_h,
            ),
            Image.Resampling.LANCZOS,
        )
    )


def save_portrait_panel(
    path: Path,
    bonafide_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    clean_overlay_rgb: np.ndarray,
    adv_overlay_rgb: np.ndarray,
    adv_rgb: np.ndarray,
    perturbation_rgb: np.ndarray,
    row: pd.Series,
    gt_thickness_px: int,
) -> None:
    # Portrait-oriented A4-like aspect.
    fig = plt.figure(
        figsize=(
            8.27,
            11.69,
        )
    )

    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[
            1.0,
            1.0,
            1.0,
            0.18,
        ],
        hspace=0.24,
        wspace=0.08,
        left=0.04,
        right=0.96,
        top=0.91,
        bottom=0.055,
    )

    panels = [
        (
            bonafide_rgb,
            "Bona fide counterpart",
        ),
        (
            clean_rgb,
            "Clean forged / attack input",
        ),
        (
            clean_overlay_rgb,
            "Clean TruFor anomaly + GT",
        ),
        (
            adv_overlay_rgb,
            "Adversarial TruFor anomaly + GT",
        ),
        (
            adv_rgb,
            "Adversarial image preview",
        ),
        (
            perturbation_rgb,
            "100× absolute perturbation",
        ),
    ]

    for idx, (
        image,
        title,
    ) in enumerate(
        panels
    ):
        row_idx = (
            idx // 2
        )

        col_idx = (
            idx % 2
        )

        ax = fig.add_subplot(
            gs[
                row_idx,
                col_idx,
            ]
        )

        ax.imshow(
            image
        )

        ax.set_title(
            title,
            fontsize=10.5,
            pad=5,
        )

        ax.axis(
            "off"
        )

    title = (
        f"{row['variant']} | {row['hardware_source']} | "
        f"{row['_selection_role']}"
    )

    subtitle = (
        f"E {float(row['E_clean']):.4f} → {float(row['E_adv']):.4f}   |   "
        f"RRA {float(row['RRA_clean']):.4f} → {float(row['RRA_adv']):.4f}   |   "
        f"L∞ {float(row['physical_linf_recomputed']):.6f}"
    )

    fig.suptitle(
        title + "\n" + subtitle,
        fontsize=12,
        y=0.977,
    )

    # --------------------------------------------------
    # Bottom in-figure legend
    # --------------------------------------------------

    legend_ax = fig.add_subplot(
        gs[
            3,
            :,
        ]
    )

    legend_ax.axis(
        "off"
    )

    # Fixed anomaly-score colour bar.
    cax = legend_ax.inset_axes(
        [
            0.04,
            0.43,
            0.42,
            0.24,
        ]
    )

    sm = ScalarMappable(
        norm=Normalize(
            vmin=0.0,
            vmax=1.0,
        ),
        cmap="inferno",
    )

    sm.set_array(
        []
    )

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_ticks(
        [
            0.0,
            0.5,
            1.0,
        ]
    )

    cbar.set_ticklabels(
        [
            "0 low",
            "0.5",
            "1 high",
        ]
    )

    cbar.ax.tick_params(
        labelsize=8,
    )

    cbar.set_label(
        "TruFor anomaly score (fixed 0–1 scale)",
        fontsize=8.5,
        labelpad=2,
    )

    cyan_line = Line2D(
        [0],
        [0],
        color="#00FFFF",
        linewidth=5,
        label=(
            "Cyan contour = manipulated-region GT "
            f"(display thickness ≈ {gt_thickness_px}px)"
        ),
    )

    perturb_line = Line2D(
        [0],
        [0],
        color="black",
        linewidth=1,
        label=(
            "Perturbation panel is amplified 100× for visibility"
        ),
    )

    legend_ax.legend(
        handles=[
            cyan_line,
            perturb_line,
        ],
        loc="center left",
        bbox_to_anchor=(
            0.52,
            0.5,
        ),
        frameon=True,
        fontsize=8.5,
        handlelength=3.0,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and rebuild the Stage-52 output directory if it exists."
        ),
    )

    args = parser.parse_args()

    required = [
        EVIDENCE,
        MASTER,
        FULL_SELECTION,
        RUNTIME,
    ]

    for path in required:
        if not path.is_file():
            raise RuntimeError(
                f"Missing required Stage-52 input: {path}"
            )

    if OUT_ROOT.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Stage-52 output already exists:\n{OUT_ROOT}\n"
                "Use --overwrite only if intentionally rebuilding it."
            )

        shutil.rmtree(
            OUT_ROOT
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence = pd.read_csv(
        EVIDENCE,
        keep_default_na=False,
    )

    master = pd.read_csv(
        MASTER,
        keep_default_na=False,
    )

    selection = pd.read_csv(
        FULL_SELECTION,
        keep_default_na=False,
    )

    runtime = pd.read_csv(
        RUNTIME,
        keep_default_na=False,
    )

    if len(evidence) != 1330:
        raise RuntimeError(
            f"Expected 1330 evidence rows, found {len(evidence)}"
        )

    for name, frame in [
        (
            "evidence",
            evidence,
        ),
        (
            "master",
            master,
        ),
        (
            "selection",
            selection,
        ),
    ]:
        if frame[
            "image_path"
        ].duplicated().any():
            raise RuntimeError(
                f"Duplicate image_path in {name}"
            )

    runtime_cols = [
        "image_path",
        "active_wall_seconds",
        "active_wall_minutes",
        "megapixels",
    ]

    evidence = evidence.merge(
        runtime[
            runtime_cols
        ],
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    if evidence[
        "active_wall_seconds"
    ].isna().any():
        raise RuntimeError(
            "Runtime merge left missing values"
        )

    pair_index = load_pair_index()

    paired_evidence = resolve_exact_bonafide_pairs(
        evidence,
        pair_index,
    )

    selected = choose_examples(
        paired_evidence
    )

    master_by = master.set_index(
        "image_path",
        drop=False,
    )

    selection_by = selection.set_index(
        "image_path",
        drop=False,
    )

    selected_rows = []

    print()
    print("=" * 72)
    print("TRUFOR STAGE 52 — PORTRAIT QUALITATIVE PANELS")
    print("=" * 72)
    print("selected examples:", len(selected))
    print()

    for _, row in selected.iterrows():
        example_number = int(
            row[
                "example_number"
            ]
        )

        image_path = str(
            row[
                "image_path"
            ]
        )

        variant = str(
            row[
                "variant"
            ]
        )

        if image_path not in master_by.index:
            raise RuntimeError(
                f"Selected image missing from master: {image_path}"
            )

        if image_path not in selection_by.index:
            raise RuntimeError(
                f"Selected image missing from frozen selection: {image_path}"
            )

        mrow = master_by.loc[
            image_path
        ]

        srow = selection_by.loc[
            image_path
        ]

        npz_path = Path(
            str(
                mrow[
                    "central_npz_path"
                ]
            )
        )

        result_json = Path(
            str(
                mrow[
                    "central_result_json"
                ]
            )
        )

        trace_csv = Path(
            str(
                mrow[
                    "central_trace_csv"
                ]
            )
        )

        result_dir = Path(
            str(
                mrow[
                    "central_result_dir"
                ]
            )
        )

        complete_json = (
            result_dir
            / "COMPLETE.json"
        )

        clean_attack_path = (
            ROOT
            / str(
                srow[
                    "cache_path"
                ]
            )
        )

        bonafide_path = (
            ROOT
            / str(
                row[
                    "bonafide_cache_path"
                ]
            )
        )

        for p in [
            npz_path,
            result_json,
            trace_csv,
            complete_json,
            clean_attack_path,
            bonafide_path,
        ]:
            if not p.is_file():
                raise RuntimeError(
                    f"Missing selected-example artifact: {p}"
                )

        example_name = (
            f"{example_number:02d}_"
            f"{safe_name(variant)}_"
            f"{safe_name(Path(image_path).stem)}"
        )

        example_dir = (
            OUT_ROOT
            / "examples"
            / example_name
        )

        raw_dir = (
            example_dir
            / "raw"
        )

        display_dir = (
            example_dir
            / "display"
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        display_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            npz_path,
            raw_dir
            / "adversarial_result.npz",
        )

        shutil.copy2(
            result_json,
            raw_dir
            / "result.json",
        )

        shutil.copy2(
            trace_csv,
            raw_dir
            / "trace.csv",
        )

        shutil.copy2(
            complete_json,
            raw_dir
            / "COMPLETE.json",
        )

        attack_suffix = (
            clean_attack_path
            .suffix
            .lower()
        )

        bonafide_suffix = (
            bonafide_path
            .suffix
            .lower()
        )

        shutil.copy2(
            clean_attack_path,
            raw_dir
            / (
                "clean_forged_input_original"
                + attack_suffix
            ),
        )

        shutil.copy2(
            bonafide_path,
            raw_dir
            / (
                "bonafide_counterpart_original"
                + bonafide_suffix
            ),
        )

        with Image.open(
            clean_attack_path
        ) as im:
            clean_rgb = np.asarray(
                im.convert(
                    "RGB"
                ),
                dtype=np.uint8,
            )

        with Image.open(
            bonafide_path
        ) as im:
            bonafide_rgb = np.asarray(
                im.convert(
                    "RGB"
                ),
                dtype=np.uint8,
            )

        with np.load(
            npz_path,
            allow_pickle=False,
        ) as z:
            expected_keys = {
                "adv_x_model",
                "clean_anomaly_map",
                "adv_anomaly_map",
                "altered_union_mask",
            }

            if set(
                z.files
            ) != expected_keys:
                raise RuntimeError(
                    f"Unexpected NPZ keys for {image_path}: {set(z.files)}"
                )

            adv_x_model = np.asarray(
                z[
                    "adv_x_model"
                ],
                dtype=np.float32,
            )

            clean_map = np.asarray(
                z[
                    "clean_anomaly_map"
                ],
                dtype=np.float32,
            )

            adv_map = np.asarray(
                z[
                    "adv_anomaly_map"
                ],
                dtype=np.float32,
            )

            mask = np.asarray(
                z[
                    "altered_union_mask"
                ],
                dtype=np.uint8,
            )

        h, w = clean_rgb.shape[
            :2
        ]

        if adv_x_model.shape != (
            3,
            h,
            w,
        ):
            raise RuntimeError(
                f"adv_x geometry mismatch for {image_path}: "
                f"{adv_x_model.shape} vs (3,{h},{w})"
            )

        for name, arr in [
            (
                "clean_map",
                clean_map,
            ),
            (
                "adv_map",
                adv_map,
            ),
            (
                "mask",
                mask,
            ),
        ]:
            if arr.shape != (
                h,
                w,
            ):
                raise RuntimeError(
                    f"{name} geometry mismatch for {image_path}: "
                    f"{arr.shape} vs {(h,w)}"
                )

        # Resize the bona fide counterpart for side-by-side display only.
        bonafide_display = resize_to_match(
            bonafide_rgb,
            (
                h,
                w,
            ),
        )

        # Exact model tensor -> physical-space display preview.
        adv_physical = np.clip(
            adv_x_model.astype(
                np.float64
            )
            * (
                256.0
                / 255.0
            ),
            0.0,
            1.0,
        )

        adv_rgb = np.rint(
            adv_physical.transpose(
                1,
                2,
                0,
            )
            * 255.0
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        clean_physical = (
            clean_rgb.astype(
                np.float64
            )
            / 255.0
        )

        delta_abs = np.abs(
            adv_physical.transpose(
                1,
                2,
                0,
            )
            - clean_physical
        )

        perturbation_x100 = np.rint(
            np.clip(
                delta_abs
                * 100.0,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(
            np.uint8
        )

        clean_overlay_rgb, gt_thickness = (
            overlay_heat_gt(
                clean_rgb,
                clean_map,
                mask,
            )
        )

        adv_overlay_rgb, gt_thickness_adv = (
            overlay_heat_gt(
                adv_rgb,
                adv_map,
                mask,
            )
        )

        if gt_thickness_adv != gt_thickness:
            raise RuntimeError(
                "GT display thickness inconsistent between clean/adv panels"
            )

        clean_gt_rgb, _ = paint_gt_contour(
            clean_rgb,
            mask,
        )

        # Individual display assets.
        Image.fromarray(
            bonafide_display
        ).save(
            display_dir
            / "bonafide_counterpart_display.png"
        )

        Image.fromarray(
            clean_rgb
        ).save(
            display_dir
            / "clean_forged_input_display.png"
        )

        Image.fromarray(
            clean_overlay_rgb
        ).save(
            display_dir
            / "clean_TruFor_anomaly_GT_overlay.png"
        )

        Image.fromarray(
            adv_overlay_rgb
        ).save(
            display_dir
            / "adversarial_TruFor_anomaly_GT_overlay.png"
        )

        Image.fromarray(
            adv_rgb
        ).save(
            display_dir
            / "adversarial_image_preview.png"
        )

        Image.fromarray(
            perturbation_x100
        ).save(
            display_dir
            / "perturbation_abs_x100.png"
        )

        Image.fromarray(
            clean_gt_rgb
        ).save(
            display_dir
            / "clean_forged_GT_contour.png"
        )

        # Raw fixed-scale anomaly maps retained as separate display assets.
        Image.fromarray(
            map_to_rgb(
                clean_map
            )
        ).save(
            display_dir
            / "clean_TruFor_anomaly_fixed_0_1.png"
        )

        Image.fromarray(
            map_to_rgb(
                adv_map
            )
        ).save(
            display_dir
            / "adversarial_TruFor_anomaly_fixed_0_1.png"
        )

        save_portrait_panel(
            display_dir
            / "qualitative_panel_portrait_3x2.png",
            bonafide_display,
            clean_rgb,
            clean_overlay_rgb,
            adv_overlay_rgb,
            adv_rgb,
            perturbation_x100,
            row,
            gt_thickness,
        )

        metadata = {
            "example_number":
                example_number,

            "selection_role":
                str(
                    row[
                        "_selection_role"
                    ]
                ),

            "selection_scope":
                str(
                    row[
                        "_selection_scope"
                    ]
                ),

            "selection_note":
                str(
                    row[
                        "_selection_note"
                    ]
                ),

            "image_path":
                image_path,

            "file_stem":
                str(
                    row[
                        "file_stem"
                    ]
                ),

            "variant":
                variant,

            "hardware_source":
                str(
                    row[
                        "hardware_source"
                    ]
                ),

            "eval_split":
                str(
                    row[
                        "eval_split"
                    ]
                ),

            "bonafide_image_path":
                str(
                    row[
                        "bonafide_image_path"
                    ]
                ),

            "bonafide_cache_path":
                str(
                    row[
                        "bonafide_cache_path"
                    ]
                ),

            "bonafide_pair_source_index":
                str(
                    row[
                        "bonafide_pair_source_index"
                    ]
                ),

            "pairing_rule":
                "exact file_stem + hardware_source; split preferred when available",

            "A_union":
                float(
                    row[
                        "A_union"
                    ]
                ),

            "E_clean":
                float(
                    row[
                        "E_clean"
                    ]
                ),

            "E_adv":
                float(
                    row[
                        "E_adv"
                    ]
                ),

            "RRA_clean":
                float(
                    row[
                        "RRA_clean"
                    ]
                ),

            "RRA_adv":
                float(
                    row[
                        "RRA_adv"
                    ]
                ),

            "DCEC":
                bool(
                    str(
                        row[
                            "DCEC"
                        ]
                    )
                    .strip()
                    .lower()
                    in [
                        "true",
                        "1",
                        "yes",
                    ]
                ),

            "DCEW":
                bool(
                    str(
                        row[
                            "DCEW"
                        ]
                    )
                    .strip()
                    .lower()
                    in [
                        "true",
                        "1",
                        "yes",
                    ]
                ),

            "clean_score":
                float(
                    row[
                        "clean_score"
                    ]
                ),

            "adv_score":
                float(
                    row[
                        "adv_score"
                    ]
                ),

            "physical_linf":
                float(
                    row[
                        "physical_linf_recomputed"
                    ]
                ),

            "active_wall_seconds":
                float(
                    row[
                        "active_wall_seconds"
                    ]
                ),

            "native_height":
                int(
                    row[
                        "native_height"
                    ]
                ),

            "native_width":
                int(
                    row[
                        "native_width"
                    ]
                ),

            "GT_display_colour":
                "cyan #00FFFF",

            "GT_display_thickness_px":
                int(
                    gt_thickness
                ),

            "TruFor_map_display_scale":
                "fixed 0.0 to 1.0 Inferno colormap; no per-image renormalisation",

            "perturbation_display":
                "100x absolute physical-space perturbation; display only",

            "adversarial_preview_note":
                (
                    "8-bit display preview reconstructed from exact float "
                    "adv_x_model; scientific computation uses the retained NPZ."
                ),

            "protocol_sha256":
                PROTOCOL_SHA,

            "checkpoint_sha256":
                CHECKPOINT_SHA,
        }

        (
            example_dir
            / "EXAMPLE_METADATA.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        selected_rows.append(
            {
                "example_number":
                    example_number,

                "example_directory":
                    str(
                        example_dir.relative_to(
                            OUT_ROOT
                        )
                    ),

                "selection_role":
                    metadata[
                        "selection_role"
                    ],

                "image_path":
                    image_path,

                "bonafide_image_path":
                    metadata[
                        "bonafide_image_path"
                    ],

                "variant":
                    variant,

                "hardware_source":
                    metadata[
                        "hardware_source"
                    ],

                "eval_split":
                    metadata[
                        "eval_split"
                    ],

                "A_union":
                    metadata[
                        "A_union"
                    ],

                "E_clean":
                    metadata[
                        "E_clean"
                    ],

                "E_adv":
                    metadata[
                        "E_adv"
                    ],

                "RRA_clean":
                    metadata[
                        "RRA_clean"
                    ],

                "RRA_adv":
                    metadata[
                        "RRA_adv"
                    ],

                "DCEC":
                    metadata[
                        "DCEC"
                    ],

                "DCEW":
                    metadata[
                        "DCEW"
                    ],

                "GT_display_thickness_px":
                    gt_thickness,

                "portrait_panel":
                    str(
                        (
                            display_dir
                            / "qualitative_panel_portrait_3x2.png"
                        ).relative_to(
                            OUT_ROOT
                        )
                    ),
            }
        )

        print(
            f"[{example_number}/6] "
            f"{metadata['selection_role']} | "
            f"{variant} | "
            f"{metadata['hardware_source']} | "
            f"GT thickness={gt_thickness}px"
        )

    selected_csv = (
        OUT_ROOT
        / "SELECTED_EXAMPLES.csv"
    )

    pd.DataFrame(
        selected_rows
    ).to_csv(
        selected_csv,
        index=False,
    )

    method = """# Stage-52 portrait qualitative figure conventions

Each example is a portrait-oriented 3-row x 2-column figure:

Row 1:
- bona fide counterpart
- clean forged / attack input

Row 2:
- clean TruFor anomaly map overlaid on the forged input
- adversarial TruFor anomaly map overlaid on the adversarial input

Row 3:
- adversarial image preview
- 100x absolute perturbation

The bona fide counterpart is matched exactly by:
- file_stem
- hardware_source

When split metadata are present, a split match is preferred.

The cyan GT contour is deliberately thicker than the previous Stage-50 display
so that it remains visible when the figure is reduced to a portrait
dissertation page.

The legend is embedded inside every panel:
- Inferno colour bar: TruFor anomaly score, fixed 0..1 scale
- dark/purple = lower anomaly evidence
- yellow/white = higher anomaly evidence
- cyan contour = manipulated-region ground truth
- perturbation is magnified 100x for visibility

Clean and adversarial anomaly maps are not independently renormalised.

The five family examples are selected automatically using clean-state metrics,
restricted to images with an exact bona fide pair. The sixth panel is an
explicit residual-evidence stress case and must not be described as
representative.
"""

    (
        OUT_ROOT
        / "FIGURE_CONVENTIONS.md"
    ).write_text(
        method
    )

    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).strip()

    provenance = {
        "status":
            "PASS",

        "population":
            1330,

        "examples":
            len(
                selected_rows
            ),

        "git_commit":
            commit,

        "protocol_sha256":
            PROTOCOL_SHA,

        "checkpoint_sha256":
            CHECKPOINT_SHA,

        "tau_E":
            TAU_E,

        "tau_RRA":
            TAU_RRA,

        "pair_indices_used":
            [
                str(
                    p.relative_to(
                        ROOT
                    )
                )
                for p in PAIR_INDEX_CANDIDATES
                if p.is_file()
            ],

        "pair_rule":
            "exact file_stem + hardware_source; split preferred",

        "evidence_table_sha256":
            sha256_file(
                EVIDENCE
            ),

        "master_manifest_sha256":
            sha256_file(
                MASTER
            ),
    }

    (
        OUT_ROOT
        / "PACKAGE_PROVENANCE.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    readme = """# Final TruFor portrait qualitative panels

This package supersedes the earlier landscape Stage-50 display panels for
dissertation figure use.

Each example has:

- `display/qualitative_panel_portrait_3x2.png`
- individual display PNGs
- exact `raw/adversarial_result.npz`
- result.json
- trace.csv
- COMPLETE.json
- original clean forged input
- exact matched bona fide counterpart
- EXAMPLE_METADATA.json

The figure itself contains the heatmap/GT legend.

Use the portrait panel for the dissertation. Use the individual PNGs only if
the writer needs to recompose a figure manually.
"""

    (
        OUT_ROOT
        / "README.md"
    ).write_text(
        readme
    )

    sums_path = (
        OUT_ROOT
        / "SHA256SUMS.txt"
    )

    files_to_hash = sorted(
        p
        for p in OUT_ROOT.rglob(
            "*"
        )
        if p.is_file()
        and p != sums_path
    )

    sums_path.write_text(
        "".join(
            f"{sha256_file(p)}  "
            f"{p.relative_to(OUT_ROOT)}\n"
            for p in files_to_hash
        )
    )

    if TAR_PATH.exists():
        TAR_PATH.unlink()

    with tarfile.open(
        TAR_PATH,
        "w",
    ) as tf:
        tf.add(
            OUT_ROOT,
            arcname=OUT_ROOT.name,
        )

    tar_sha = sha256_file(
        TAR_PATH
    )

    sidecar = Path(
        str(
            TAR_PATH
        )
        + ".sha256"
    )

    sidecar.write_text(
        f"{tar_sha}  "
        f"{TAR_PATH.name}\n"
    )

    print()
    print("=" * 72)
    print("STAGE 52 PORTRAIT QUALITATIVE PACKAGE RESULT")
    print("=" * 72)
    print("population       : 1330")
    print("examples         :", len(selected_rows))
    print("package          :", OUT_ROOT)
    print("transfer TAR     :", TAR_PATH)
    print(
        "TAR size MiB     :",
        f"{TAR_PATH.stat().st_size / 1024**2:.2f}",
    )
    print("TAR SHA256       :", tar_sha)
    print()
    print("STAGE 52 PASS")


if __name__ == "__main__":
    main()
