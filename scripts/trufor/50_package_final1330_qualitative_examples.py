#!/usr/bin/env python3
"""
Stage 50 — create a small, deterministic qualitative-example package for the
final 1330-image TruFor adversarial-localisation experiment.

Selection is deliberately metric-driven rather than visual/cherry-picked:

1. One clean-state representative example per manipulation family:
   digital_1, digital_2, digital_3, facedancer, textdiffuserft_bfei.
   - If a family contains primary DCEC images, select from its DCEC subset.
   - Otherwise (currently facedancer), select from the full family.
   - Selection uses ONLY clean-state metrics (A, E_clean, RRA_clean,
     clean_score) and chooses the robust-medoid-like row closest to the
     candidate-set medians.

2. One additional "residual-evidence stress case":
   among primary DCEC images not already selected, choose the image with the
   largest max(E_adv/tau_E, RRA_adv/tau_RRA). This is an explicit stress
   example, not a representative example.

For every selected image the package retains:
- exact adversarial_result.npz
- result.json
- trace.csv
- COMPLETE.json
- original cached clean image
- full-resolution display PNGs
- clean/adversarial anomaly maps with a fixed 0..1 colour scale
- clean/adversarial anomaly overlays with GT contour
- GT union mask
- 100x absolute perturbation display
- publication-ready 2x3 qualitative panel
- metrics/provenance JSON

The display PNG of the adversarial image is reconstructed from the exact
adv_x_model tensor and quantised to uint8 ONLY for visualisation. Scientific
measurements continue to use the retained float NPZ tensor.

Large all-population arrays are not duplicated; only the selected examples are
included.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

PRIMARY_SUMMARY = (
    ANALYSIS
    / "46_final1330_primary_summary.json"
)

RUNTIME = (
    ANALYSIS
    / "47_final1330_runtime_per_image.csv"
)

PROTOCOL = (
    ROOT
    / "scripts"
    / "trufor"
    / "evidence_evaluation_protocol_v1.json"
)

OUT_ROOT = (
    ANALYSIS
    / "50_qualitative_examples"
)

TRANSFER_ROOT = (
    ROOT
    / "analysis_transfer_bundles"
)

TAR_PATH = (
    TRANSFER_ROOT
    / "IMTA135_trufor_final1330_qualitative_examples.tar"
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


def robust_clean_medoid_score(frame: pd.DataFrame) -> pd.Series:
    """
    Deterministic clean-state representativeness score.

    Lower is more representative. Uses clean information only, avoiding
    selection for visually dramatic adversarial outcomes.
    """
    metrics = [
        "A_union",
        "E_clean",
        "RRA_clean",
        "clean_score",
    ]

    score = pd.Series(
        np.zeros(len(frame), dtype=np.float64),
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

        med = float(x.median())
        q25 = float(x.quantile(0.25))
        q75 = float(x.quantile(0.75))
        iqr = q75 - q25

        if not math.isfinite(iqr) or iqr <= 1e-12:
            mad = float((x - med).abs().median())
            scale = mad if mad > 1e-12 else 1.0
        else:
            scale = iqr

        score = score + (x - med).abs() / scale

    return score


def choose_examples(df: pd.DataFrame) -> pd.DataFrame:
    selections = []
    used = set()

    dcec_all = as_bool(df["DCEC"])
    dcew_all = as_bool(df["DCEW"])

    for variant in EXPECTED_VARIANTS:
        g = df.loc[
            df["variant"].astype(str) == variant
        ].copy()

        if len(g) == 0:
            raise RuntimeError(
                f"Expected variant not found: {variant}"
            )

        dcec = as_bool(g["DCEC"])

        if dcec.any():
            candidates = g.loc[dcec].copy()
            candidate_scope = "primary_DCEC_subset"
        else:
            candidates = g.copy()
            candidate_scope = "full_variant_no_primary_DCEC"

        candidates["_selection_score"] = robust_clean_medoid_score(
            candidates
        )

        candidates = candidates.sort_values(
            ["_selection_score", "image_path"],
            ascending=[True, True],
        )

        chosen = candidates.iloc[0].copy()

        used.add(str(chosen["image_path"]))

        chosen["_selection_role"] = "variant_representative"
        chosen["_selection_scope"] = candidate_scope
        chosen["_selection_note"] = (
            "Deterministic clean-state median-proximity selection; "
            "not visually selected."
        )

        selections.append(chosen)

    stress = df.loc[
        dcec_all & dcew_all
    ].copy()

    stress = stress.loc[
        ~stress["image_path"].astype(str).isin(used)
    ].copy()

    if len(stress) == 0:
        raise RuntimeError(
            "No non-duplicate DCEC/DCEW row available for stress example"
        )

    stress["_residual_fraction"] = np.maximum(
        pd.to_numeric(stress["E_adv"]) / TAU_E,
        pd.to_numeric(stress["RRA_adv"]) / TAU_RRA,
    )

    stress = stress.sort_values(
        ["_residual_fraction", "image_path"],
        ascending=[False, True],
    )

    chosen = stress.iloc[0].copy()
    chosen["_selection_score"] = np.nan
    chosen["_selection_role"] = "largest_residual_evidence_DCEC"
    chosen["_selection_scope"] = "all_primary_DCEC_excluding_variant_representatives"
    chosen["_selection_note"] = (
        "Explicit stress case: largest remaining post-attack evidence "
        "fraction relative to the primary E/RRA thresholds."
    )

    selections.append(chosen)

    out = pd.DataFrame(selections).reset_index(drop=True)
    out.insert(0, "example_number", np.arange(1, len(out) + 1))

    if out["image_path"].duplicated().any():
        raise RuntimeError(
            "Qualitative selection unexpectedly contains duplicate image_path"
        )

    return out


def map_to_rgb(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(
            f"Expected 2-D map, got {x.shape}"
        )
    x = np.clip(x, 0.0, 1.0)
    rgba = plt.get_cmap("inferno")(x)
    return np.rint(
        rgba[:, :, :3] * 255.0
    ).astype(np.uint8)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(
            f"Expected 2-D mask, got {m.shape}"
        )

    interior = m.copy()

    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False

    interior[1:-1, 1:-1] &= (
        m[:-2, 1:-1]
        & m[2:, 1:-1]
        & m[1:-1, :-2]
        & m[1:-1, 2:]
    )

    return m & ~interior


def overlay_heat_gt(
    base_rgb: np.ndarray,
    amap: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.uint8)
    heat = map_to_rgb(amap)

    if heat.shape != base.shape:
        raise ValueError(
            f"Heat/base mismatch: {heat.shape} vs {base.shape}"
        )

    out = np.rint(
        (1.0 - alpha) * base.astype(np.float32)
        + alpha * heat.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    boundary = mask_boundary(mask)

    # Cyan GT contour for display only.
    out[boundary, 0] = 0
    out[boundary, 1] = 255
    out[boundary, 2] = 255

    return out


def gt_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    out = np.asarray(base_rgb, dtype=np.uint8).copy()
    boundary = mask_boundary(mask)

    out[boundary, 0] = 0
    out[boundary, 1] = 255
    out[boundary, 2] = 255

    return out


def resize_for_panel(
    image: np.ndarray,
    max_side: int = 1500,
) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]

    scale = min(
        1.0,
        float(max_side) / max(h, w),
    )

    if scale >= 1.0:
        return rgb

    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    return np.asarray(
        Image.fromarray(rgb).resize(
            (nw, nh),
            Image.Resampling.LANCZOS,
        )
    )


def save_panel(
    path: Path,
    clean_rgb: np.ndarray,
    adv_rgb: np.ndarray,
    clean_overlay_rgb: np.ndarray,
    adv_overlay_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    perturbation_rgb: np.ndarray,
    row: pd.Series,
) -> None:
    images = [
        resize_for_panel(clean_rgb),
        resize_for_panel(clean_overlay_rgb),
        resize_for_panel(adv_rgb),
        resize_for_panel(adv_overlay_rgb),
        resize_for_panel(gt_rgb),
        resize_for_panel(perturbation_rgb),
    ]

    titles = [
        "Clean image",
        "Clean anomaly + GT",
        "Adversarial preview",
        "Adversarial anomaly + GT",
        "Union GT contour",
        "100× |perturbation|",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    for ax, im, title in zip(
        axes.flat,
        images,
        titles,
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    subtitle = (
        f"{row['variant']} | {row['hardware_source']} | "
        f"E {float(row['E_clean']):.4f}→{float(row['E_adv']):.4f} | "
        f"RRA {float(row['RRA_clean']):.4f}→{float(row['RRA_adv']):.4f} | "
        f"L∞ {float(row['physical_linf_recomputed']):.6f}"
    )

    fig.suptitle(
        subtitle,
        fontsize=12,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95)
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the qualitative package if it already exists.",
    )

    args = parser.parse_args()

    required = [
        EVIDENCE,
        MASTER,
        FULL_SELECTION,
        PRIMARY_SUMMARY,
        RUNTIME,
        PROTOCOL,
    ]

    for p in required:
        if not p.is_file():
            raise RuntimeError(
                f"Missing required Stage-50 input: {p}"
            )

    if OUT_ROOT.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Output already exists: {OUT_ROOT}\n"
                "Use --overwrite only if intentionally rebuilding."
            )
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
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

    if len(df) != 1330:
        raise RuntimeError(
            f"Expected 1330 evidence rows, found {len(df)}"
        )

    if len(master) != 1330:
        raise RuntimeError(
            f"Expected 1330 master rows, found {len(master)}"
        )

    if df["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in evidence table"
        )

    if master["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in master table"
        )

    if selection["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in frozen selection"
        )

    # Add runtime to the evidence table for writer metadata.
    runtime_cols = [
        "image_path",
        "active_wall_seconds",
        "active_wall_minutes",
        "megapixels",
    ]

    df = df.merge(
        runtime[runtime_cols],
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    if df["active_wall_seconds"].isna().any():
        raise RuntimeError(
            "Runtime merge left missing values"
        )

    selected = choose_examples(df)

    master_by = master.set_index(
        "image_path",
        drop=False,
    )

    selection_by = selection.set_index(
        "image_path",
        drop=False,
    )

    selected_rows = []

    print("=" * 72)
    print("TRUFOR STAGE 50 — FINAL QUALITATIVE EXAMPLES")
    print("=" * 72)
    print("selected examples:", len(selected))
    print()

    for _, row in selected.iterrows():
        number = int(row["example_number"])
        variant = str(row["variant"])
        image_path = str(row["image_path"])

        if image_path not in master_by.index:
            raise RuntimeError(
                f"Selected image missing from Stage-44 master: {image_path}"
            )

        if image_path not in selection_by.index:
            raise RuntimeError(
                f"Selected image missing from frozen selection: {image_path}"
            )

        mrow = master_by.loc[image_path]
        srow = selection_by.loc[image_path]

        npz_path = Path(
            str(mrow["central_npz_path"])
        )

        result_json = Path(
            str(mrow["central_result_json"])
        )

        trace_csv = Path(
            str(mrow["central_trace_csv"])
        )

        result_dir = Path(
            str(mrow["central_result_dir"])
        )

        complete_json = (
            result_dir
            / "COMPLETE.json"
        )

        cache_path = (
            ROOT
            / str(srow["cache_path"])
        )

        for p in [
            npz_path,
            result_json,
            trace_csv,
            complete_json,
            cache_path,
        ]:
            if not p.is_file():
                raise RuntimeError(
                    f"Missing selected-example artifact: {p}"
                )

        example_name = (
            f"{number:02d}_"
            f"{safe_name(variant)}_"
            f"{safe_name(Path(image_path).stem)}"
        )

        ex_dir = (
            OUT_ROOT
            / "examples"
            / example_name
        )

        raw_dir = ex_dir / "raw"
        display_dir = ex_dir / "display"

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        display_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Copy exact scientific/provenance artifacts.
        shutil.copy2(
            npz_path,
            raw_dir / "adversarial_result.npz",
        )

        shutil.copy2(
            result_json,
            raw_dir / "result.json",
        )

        shutil.copy2(
            trace_csv,
            raw_dir / "trace.csv",
        )

        shutil.copy2(
            complete_json,
            raw_dir / "COMPLETE.json",
        )

        clean_original_name = (
            "clean_source_original"
            + cache_path.suffix.lower()
        )

        shutil.copy2(
            cache_path,
            raw_dir / clean_original_name,
        )

        with Image.open(cache_path) as im:
            clean_rgb = np.asarray(
                im.convert("RGB"),
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

            if set(z.files) != expected_keys:
                raise RuntimeError(
                    f"Unexpected NPZ keys for {image_path}: {set(z.files)}"
                )

            adv_x_model = np.asarray(
                z["adv_x_model"],
                dtype=np.float32,
            )

            clean_map = np.asarray(
                z["clean_anomaly_map"],
                dtype=np.float32,
            )

            adv_map = np.asarray(
                z["adv_anomaly_map"],
                dtype=np.float32,
            )

            mask = np.asarray(
                z["altered_union_mask"],
                dtype=np.uint8,
            )

        h, w = clean_rgb.shape[:2]

        if adv_x_model.shape != (3, h, w):
            raise RuntimeError(
                f"adv_x geometry mismatch for {image_path}: "
                f"{adv_x_model.shape} vs (3,{h},{w})"
            )

        for name, arr in [
            ("clean_map", clean_map),
            ("adv_map", adv_map),
            ("mask", mask),
        ]:
            if arr.shape != (h, w):
                raise RuntimeError(
                    f"{name} geometry mismatch for {image_path}: "
                    f"{arr.shape} vs {(h,w)}"
                )

        # Exact physical-space reconstruction used only to make display images.
        adv_physical = np.clip(
            adv_x_model.astype(np.float64)
            * (256.0 / 255.0),
            0.0,
            1.0,
        )

        adv_rgb = np.rint(
            adv_physical.transpose(1, 2, 0)
            * 255.0
        ).clip(
            0,
            255,
        ).astype(np.uint8)

        clean_physical = (
            clean_rgb.astype(np.float64)
            / 255.0
        )

        delta_abs = np.abs(
            adv_physical.transpose(1, 2, 0)
            - clean_physical
        )

        perturbation_x100 = np.rint(
            np.clip(
                delta_abs * 100.0,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

        clean_map_rgb = map_to_rgb(clean_map)
        adv_map_rgb = map_to_rgb(adv_map)

        clean_overlay_rgb = overlay_heat_gt(
            clean_rgb,
            clean_map,
            mask,
        )

        adv_overlay_rgb = overlay_heat_gt(
            adv_rgb,
            adv_map,
            mask,
        )

        gt_rgb = gt_overlay(
            clean_rgb,
            mask,
        )

        gt_mask_rgb = (
            np.asarray(mask, dtype=np.uint8)
            * 255
        )

        Image.fromarray(
            clean_rgb
        ).save(
            display_dir / "clean_display.png"
        )

        Image.fromarray(
            adv_rgb
        ).save(
            display_dir / "adversarial_display_preview.png"
        )

        Image.fromarray(
            clean_map_rgb
        ).save(
            display_dir / "clean_anomaly_fixed_0_1.png"
        )

        Image.fromarray(
            adv_map_rgb
        ).save(
            display_dir / "adversarial_anomaly_fixed_0_1.png"
        )

        Image.fromarray(
            clean_overlay_rgb
        ).save(
            display_dir / "clean_anomaly_GT_overlay.png"
        )

        Image.fromarray(
            adv_overlay_rgb
        ).save(
            display_dir / "adversarial_anomaly_GT_overlay.png"
        )

        Image.fromarray(
            gt_rgb
        ).save(
            display_dir / "clean_GT_contour.png"
        )

        Image.fromarray(
            gt_mask_rgb
        ).save(
            display_dir / "union_GT_mask.png"
        )

        Image.fromarray(
            perturbation_x100
        ).save(
            display_dir / "perturbation_abs_x100.png"
        )

        save_panel(
            display_dir / "qualitative_panel.png",
            clean_rgb,
            adv_rgb,
            clean_overlay_rgb,
            adv_overlay_rgb,
            gt_rgb,
            perturbation_x100,
            row,
        )

        metadata = {
            "example_number":
                number,

            "selection_role":
                str(row["_selection_role"]),

            "selection_scope":
                str(row["_selection_scope"]),

            "selection_note":
                str(row["_selection_note"]),

            "selection_score":
                (
                    None
                    if pd.isna(
                        row["_selection_score"]
                    )
                    else float(
                        row["_selection_score"]
                    )
                ),

            "image_path":
                image_path,

            "file_stem":
                str(row["file_stem"]),

            "variant":
                variant,

            "hardware_source":
                str(row["hardware_source"]),

            "eval_split":
                str(row["eval_split"]),

            "A_union":
                float(row["A_union"]),

            "E_clean":
                float(row["E_clean"]),

            "E_adv":
                float(row["E_adv"]),

            "relative_E_degradation":
                float(row["relative_E_degradation"]),

            "RRA_clean":
                float(row["RRA_clean"]),

            "RRA_adv":
                float(row["RRA_adv"]),

            "relative_RRA_degradation":
                (
                    None
                    if pd.isna(
                        pd.to_numeric(
                            pd.Series(
                                [row["relative_RRA_degradation"]]
                            ),
                            errors="coerce",
                        ).iloc[0]
                    )
                    else float(
                        row["relative_RRA_degradation"]
                    )
                ),

            "DCEC":
                bool(
                    str(row["DCEC"]).strip().lower()
                    in ["true", "1", "yes"]
                ),

            "DCEW":
                bool(
                    str(row["DCEW"]).strip().lower()
                    in ["true", "1", "yes"]
                ),

            "clean_score":
                float(row["clean_score"]),

            "adv_score":
                float(row["adv_score"]),

            "physical_linf":
                float(row["physical_linf_recomputed"]),

            "active_wall_seconds":
                float(row["active_wall_seconds"]),

            "native_height":
                int(row["native_height"]),

            "native_width":
                int(row["native_width"]),

            "source_host":
                str(row["source_host"]),

            "protocol_sha256":
                PROTOCOL_SHA,

            "checkpoint_sha256":
                CHECKPOINT_SHA,

            "display_note":
                (
                    "adversarial_display_preview.png is an 8-bit visualisation "
                    "reconstructed from the exact float adv_x_model tensor. "
                    "Use raw/adversarial_result.npz for scientific computation."
                ),

            "map_display_note":
                (
                    "Anomaly heatmaps use the same fixed 0..1 colour scale "
                    "for clean and adversarial maps; they are not independently "
                    "renormalised."
                ),

            "raw_artifact_sha256": {
                "adversarial_result.npz":
                    sha256_file(
                        raw_dir
                        / "adversarial_result.npz"
                    ),

                "result.json":
                    sha256_file(
                        raw_dir
                        / "result.json"
                    ),

                "trace.csv":
                    sha256_file(
                        raw_dir
                        / "trace.csv"
                    ),

                "COMPLETE.json":
                    sha256_file(
                        raw_dir
                        / "COMPLETE.json"
                    ),

                clean_original_name:
                    sha256_file(
                        raw_dir
                        / clean_original_name
                    ),
            },
        }

        (
            ex_dir
            / "EXAMPLE_METADATA.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        selected_row = {
            "example_number":
                number,

            "example_directory":
                str(
                    ex_dir.relative_to(
                        OUT_ROOT
                    )
                ),

            "selection_role":
                metadata[
                    "selection_role"
                ],

            "selection_scope":
                metadata[
                    "selection_scope"
                ],

            "image_path":
                image_path,

            "file_stem":
                metadata[
                    "file_stem"
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

            "physical_linf":
                metadata[
                    "physical_linf"
                ],

            "active_wall_seconds":
                metadata[
                    "active_wall_seconds"
                ],

            "qualitative_panel":
                str(
                    (
                        display_dir
                        / "qualitative_panel.png"
                    ).relative_to(
                        OUT_ROOT
                    )
                ),
        }

        selected_rows.append(
            selected_row
        )

        print(
            f"[{number}/6] "
            f"{metadata['selection_role']} | "
            f"{variant} | "
            f"{metadata['hardware_source']} | "
            f"{image_path}"
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

    selection_method = """# Qualitative-example selection method

The examples in this package were selected automatically from the final
1330-image analysis table before any visual inspection.

## Family representatives

One example is selected for each manipulation family:

- digital_1
- digital_2
- digital_3
- facedancer
- textdiffuserft_bfei

If the family contains primary DCEC images, selection is restricted to its
DCEC subset. If it has no primary DCEC images (currently facedancer), the
complete family is used.

Within the candidate set, a robust clean-state median-proximity score is
computed from:

- GT area A
- clean E/RMA
- clean RRA
- clean image-level attack score

Each variable is centred by its candidate-set median and scaled by its IQR
(with a MAD fallback for degenerate IQRs). The row with the smallest summed
absolute scaled distance is selected. Ties are broken by image_path.

This method uses clean-state metrics only; it does not select visually dramatic
adversarial outcomes.

## Residual-evidence stress case

A sixth example is deliberately a stress case, not a representative case.

Among primary DCEC/DCEW images not already selected, it chooses the image with
the largest:

    max(E_adv / tau_E, RRA_adv / tau_RRA)

using tau_E = tau_RRA = 0.5.

This shows the DCEW example retaining the most post-attack spatial evidence
relative to the primary threshold.

## Display conventions

- Clean/adversarial anomaly heatmaps use the same fixed 0..1 colour scale.
- They are not independently renormalised.
- Cyan contours mark the union manipulation GT boundary.
- The adversarial PNG is an 8-bit display preview reconstructed from the exact
  float model tensor; scientific analysis must use adversarial_result.npz.
- The perturbation panel shows 100x absolute physical-space perturbation for
  visibility and must not be interpreted at native visual amplitude.
"""

    (
        OUT_ROOT
        / "SELECTION_METHOD.md"
    ).write_text(
        selection_method
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

        "evidence_table":
            str(
                EVIDENCE.relative_to(
                    ROOT
                )
            ),

        "evidence_table_sha256":
            sha256_file(
                EVIDENCE
            ),

        "master_manifest":
            str(
                MASTER.relative_to(
                    ROOT
                )
            ),

        "master_manifest_sha256":
            sha256_file(
                MASTER
            ),

        "selection_method":
            (
                "Five clean-state family representatives "
                "+ one explicit largest-residual-evidence stress case"
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

    readme = """# Final TruFor qualitative-example package

This package accompanies the final 1330-image quantitative Stage-49 report.

Start with:

- `SELECTED_EXAMPLES.csv`
- `SELECTION_METHOD.md`

Each example directory contains:

- `display/qualitative_panel.png` — ready-to-review 2x3 panel
- display component PNGs for flexible dissertation layout
- `raw/adversarial_result.npz` — exact retained scientific arrays
- `raw/result.json`
- `raw/trace.csv`
- `raw/COMPLETE.json`
- original cached clean image
- `EXAMPLE_METADATA.json`

The five family examples are deterministic clean-state representatives.
The sixth example is explicitly labelled as a residual-evidence stress case.

Do not describe the stress case as representative.

The main quantitative source remains the Stage-49 final report package.
"""

    (
        OUT_ROOT
        / "README.md"
    ).write_text(
        readme
    )

    # Internal package hashes.
    sums = (
        OUT_ROOT
        / "SHA256SUMS.txt"
    )

    files_to_hash = sorted(
        p
        for p in OUT_ROOT.rglob("*")
        if p.is_file()
        and p != sums
    )

    sums.write_text(
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
        str(TAR_PATH)
        + ".sha256"
    )

    sidecar.write_text(
        f"{tar_sha}  {TAR_PATH.name}\n"
    )

    print()
    print("=" * 72)
    print("STAGE 50 QUALITATIVE PACKAGE RESULT")
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
    print("STAGE 50 PASS")


if __name__ == "__main__":
    main()
