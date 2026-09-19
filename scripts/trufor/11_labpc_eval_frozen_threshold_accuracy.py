#!/usr/bin/env python3
"""LABPC Stage 11: clean evaluation, localisation freeze, and PNG audit.

Inputs:
- LABPC Stage-09 inference manifest + maps
- canonical frozen threshold from HOME dev calibration (not recalibrated)
- frozen Regions inventory

Outputs (under output/LABPC/trufor_policy_c_frozen_protocol/stage11_clean_evaluation_accuracy/):
- detection_summary.csv
- detection_diagnostics.csv
- clean_correct_attacks.csv
- clean_correct_bonafides.csv
- localisation_per_image.csv
- localisation_summary.csv
- localisation_summary_clean_correct.csv
- PNG visual audit for attacks and selected bona-fides
- stage11_report.txt
- stage11_provenance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from trufor_common import ROOT, infer_one, load_trufor_model, resolve_device, sha256_file, write_json
from trufor_labpc_attack_pilot_common import (
    EXPECTED_OBJECTIVE,
    LAB_DECISION_POPULATION,
    LAB_REPRO_ROOT,
    LAB_STAGE2_ROOT,
    RUN_TAG,
    STAGE11_ROOT,
    THRESHOLD_JSON,
    build_union_mask,
    load_frozen_threshold,
    load_regions,
    localisation_values_np,
)

# alias because README refers to these stable filenames
LAB_STAGE11_CLEAN_ATTACKS = ROOT / "output" / RUN_TAG / "trufor_policy_c_frozen_protocol" / "stage11_clean_evaluation_accuracy" / "clean_correct_attacks.csv"
LAB_STAGE11_CLEAN_BONAFIDES = ROOT / "output" / RUN_TAG / "trufor_policy_c_frozen_protocol" / "stage11_clean_evaluation_accuracy" / "clean_correct_bonafides.csv"


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    manifest_path = LAB_STAGE2_ROOT / "inference_manifest.csv"
    prov_path = LAB_STAGE2_ROOT / "stage09_inference_provenance.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing LAB Stage-09 manifest: {manifest_path}")
    if not prov_path.is_file():
        raise RuntimeError(f"Missing LAB Stage-09 provenance: {prov_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if len(manifest) != 1844:
        raise RuntimeError(f"Unexpected LAB Stage-09 manifest size: {len(manifest)}")
    if manifest["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in LAB Stage-09 manifest")
    provenance = json.loads(prov_path.read_text())
    if provenance.get("status") != "PASS":
        raise RuntimeError("LAB Stage-09 provenance is not PASS")
    if not LAB_DECISION_POPULATION.is_file():
        raise RuntimeError(
            f"Missing Stage-10 LAB decision population: {LAB_DECISION_POPULATION}"
        )
    population = pd.read_csv(LAB_DECISION_POPULATION, keep_default_na=False)
    if len(population) != 1844:
        raise RuntimeError(f"Unexpected LAB decision population size: {len(population)}")
    threshold_payload, threshold = load_frozen_threshold()
    return manifest, population, threshold, threshold_payload


def bootstrap_ci(df: pd.DataFrame, value_col: str, stem_col: str = "file_stem", n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    stems = pd.Index(df[stem_col].astype(str).unique())
    rng = np.random.default_rng(seed)
    values = []
    by_stem = {stem: df.loc[df[stem_col].astype(str) == stem, value_col].to_numpy(dtype=float) for stem in stems}
    for _ in range(n_boot):
        sample_stems = rng.choice(stems.to_numpy(), size=len(stems), replace=True)
        sample_vals = []
        for stem in sample_stems:
            sample_vals.extend(by_stem[str(stem)].tolist())
        values.append(float(np.mean(sample_vals)))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def summarise_detection(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["label"] = work["label"].astype(int)
    work["pred"] = (work["trufor_score"].astype(float) >= threshold).astype(int)
    work["correct"] = (work["pred"] == work["label"]).astype(int)

    rows = []

    def metric_row(sub: pd.DataFrame, eval_split: str, group_type: str, group: str):
        labels = sub["label"].to_numpy(dtype=int)
        scores = sub["trufor_score"].to_numpy(dtype=float)
        attack = labels == 1
        bona = labels == 0
        recall = float(np.mean(sub.loc[attack, "pred"] == 1)) if attack.any() else np.nan
        spec = float(np.mean(sub.loc[bona, "pred"] == 0)) if bona.any() else np.nan
        bacc = float(0.5 * (recall + spec)) if attack.any() and bona.any() else np.nan
        auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan
        rows.append({
            "eval_split": eval_split,
            "group_type": group_type,
            "group": group,
            "n": int(len(sub)),
            "n_attack": int(attack.sum()),
            "n_bonafide": int(bona.sum()),
            "auroc": auroc,
            "accuracy": float(np.mean(sub["correct"])),
            "balanced_accuracy": bacc,
            "attack_recall": recall,
            "bonafide_specificity": spec,
            "mean_attack_score": float(sub.loc[attack, "trufor_score"].astype(float).mean()) if attack.any() else np.nan,
            "mean_bonafide_score": float(sub.loc[bona, "trufor_score"].astype(float).mean()) if bona.any() else np.nan,
        })

    for split, sub in work.groupby("eval_split", sort=True):
        metric_row(sub, str(split), "overall", "all")
        bons = sub.loc[sub["label"] == 0]
        for family, fam_sub in sub.loc[sub["label"] == 1].groupby("variant", sort=True):
            joined = pd.concat([fam_sub, bons], ignore_index=True)
            metric_row(joined, str(split), "attack_family_vs_bonafide", str(family))
        for hw, hw_sub in sub.groupby("hardware_source", sort=True):
            metric_row(hw_sub, str(split), "hardware", str(hw))
    summary = pd.DataFrame(rows)

    diag = work.groupby(["eval_split", "hardware_source", "traffic_type"], as_index=False)["trufor_score"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    return summary, diag


def draw_boxes_on_image(rgb: np.ndarray, regions: pd.DataFrame, image_path: str) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    rows = regions.loc[(regions["image_path"] == str(image_path)) & (regions["region_provenance_raw"] == "altered")]
    for _, row in rows.iterrows():
        x0, y0, x1, y1 = int(round(float(row["x"]))), int(round(float(row["y"]))), int(round(float(row["x"]+row["width"]))), int(round(float(row["y"]+row["height"])))
        draw.rectangle([x0, y0, x1, y1], outline=(0,255,255), width=4)
    return np.array(image)


def heatmap_rgb(amap: np.ndarray) -> np.ndarray:
    cmap = matplotlib.colormaps["inferno"]
    norm = Normalize(vmin=float(np.min(amap)), vmax=float(np.max(amap)))
    rgba = cmap(norm(amap))
    return np.asarray((rgba[:, :, :3] * 255).astype(np.uint8))


def overlay_heat(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    x = rgb.astype(np.float32)
    y = heat.astype(np.float32)
    z = (1.0 - alpha) * x + alpha * y
    return np.clip(z, 0, 255).astype(np.uint8)


def save_attack_panel(rgb: np.ndarray, amap: np.ndarray, image_path: str, out_path: Path, regions: pd.DataFrame, title: str = "") -> None:
    boxed = draw_boxes_on_image(rgb, regions, image_path)
    heat = heatmap_rgb(amap)
    over = overlay_heat(boxed, heat)
    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(1,3,i+1) for i in range(3)]
    axes[0].imshow(boxed); axes[0].set_title("RGB + GT altered boxes")
    axes[1].imshow(heat); axes[1].set_title("TruFor anomaly heatmap")
    axes[2].imshow(over); axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_bonafide_panel(rgb: np.ndarray, amap: np.ndarray, conf: np.ndarray, out_path: Path, title: str = "") -> None:
    heat_a = heatmap_rgb(amap)
    over_a = overlay_heat(rgb, heat_a)
    heat_c = heatmap_rgb(conf)
    over_c = overlay_heat(rgb, heat_c)
    fig = plt.figure(figsize=(16, 4))
    axes = [fig.add_subplot(1,5,i+1) for i in range(5)]
    imgs = [rgb, heat_a, over_a, heat_c, over_c]
    titles = ["RGB", "Anomaly heatmap", "Anomaly overlay", "Confidence heatmap", "Confidence overlay"]
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="used only for selected bona-fide confidence reruns")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bonafide-panels", type=int, default=60)
    args = parser.parse_args()

    STAGE11_ROOT.mkdir(parents=True, exist_ok=True)
    attack_vis_root = STAGE11_ROOT / "pngs" / "attacks"
    bona_vis_root = STAGE11_ROOT / "pngs" / "bonafides"

    manifest, population, threshold, threshold_payload = require_inputs()
    regions = load_regions()

    work = manifest.merge(
        population[["image_path", "clean_prediction", "clean_correct", "frozen_threshold"]],
        on="image_path",
        how="left",
        validate="one_to_one",
    )
    if work["clean_prediction"].isna().any() or work["clean_correct"].isna().any():
        raise RuntimeError("Stage-10 clean decision columns failed to merge")
    if not np.allclose(work["frozen_threshold"].astype(float), threshold):
        raise RuntimeError("Merged frozen threshold does not match canonical threshold")

    det_summary, det_diag = summarise_detection(work, threshold)
    det_summary.to_csv(STAGE11_ROOT / "detection_summary.csv", index=False)
    det_diag.to_csv(STAGE11_ROOT / "detection_diagnostics.csv", index=False)

    # Clean-correct populations from LAB decision population, re-emitted here for stage-local provenance.
    clean_correct_attacks = work.loc[(work["label"].astype(int) == 1) & (work["clean_correct"].astype(bool))].copy()
    clean_correct_bonafides = work.loc[(work["label"].astype(int) == 0) & (work["clean_correct"].astype(bool))].copy()
    clean_correct_attacks.to_csv(LAB_STAGE11_CLEAN_ATTACKS, index=False)
    clean_correct_bonafides.to_csv(LAB_STAGE11_CLEAN_BONAFIDES, index=False)

    # Per-image localisation on ALL attacks.
    per_rows = []
    coverage_rows = []
    attack_count = 0
    clipped_total = 0
    for _, row in work.loc[work["label"].astype(int) == 1].iterrows():
        image_path = str(row["image_path"])
        map_path = ROOT / str(row["map_path"])
        cache_path = ROOT / str(row["cache_path"])
        if not map_path.is_file():
            raise RuntimeError(f"Missing LAB anomaly map: {map_path}")
        if not cache_path.is_file():
            raise RuntimeError(f"Missing Policy-C JPEG: {cache_path}")
        with np.load(map_path, allow_pickle=False) as data:
            amap = np.asarray(data["map"], dtype=np.float32)
            score = float(np.asarray(data["score"]).item())
            hw = tuple(int(x) for x in np.asarray(data["imgsize"]).tolist())
        h, w0 = hw
        altered_mask, clipped = build_union_mask(regions, image_path, h, w0)
        clipped_total += int(clipped)
        union = localisation_values_np(amap, altered_mask)

        # face/text breakdown where available.
        face_rows = regions.loc[(regions["image_path"] == image_path) & (regions["region_provenance_raw"] == "altered") & (regions["field_name"].str.contains("face", regex=False))]
        text_rows = regions.loc[(regions["image_path"] == image_path) & (regions["region_provenance_raw"] == "altered") & (regions["field_name"].str.contains("text", regex=False))]

        def mask_from_rows(sub: pd.DataFrame):
            if sub.empty:
                return None
            mask = np.zeros((h, w0), dtype=bool)
            ok = 0
            for _, rr in sub.iterrows():
                # use same clipping logic
                from trufor_labpc_attack_pilot_common import round_box, clip_box
                box, _ = clip_box(round_box(rr), w0, h)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                mask[y0:y1, x0:x1] = True
                ok += 1
            return mask if ok and int(mask.sum()) else None

        face_mask = mask_from_rows(face_rows)
        text_mask = mask_from_rows(text_rows)
        face_metrics = localisation_values_np(amap, face_mask) if face_mask is not None else {"A": np.nan, "E": np.nan, "mu_w": np.nan, "PG": np.nan}
        text_metrics = localisation_values_np(amap, text_mask) if text_mask is not None else {"A": np.nan, "E": np.nan, "mu_w": np.nan, "PG": np.nan}

        per_rows.append({
            "image_path": image_path,
            "cache_path": str(row["cache_path"]),
            "map_path": str(row["map_path"]),
            "eval_split": str(row["eval_split"]),
            "variant": str(row["variant"]),
            "hardware_source": str(row["hardware_source"]),
            "file_stem": str(row["file_stem"]),
            "trufor_score": score,
            "clean_prediction": int(row["clean_prediction"]),
            "clean_correct": bool(row["clean_correct"]),
            "native_height": h,
            "native_width": w0,
            "A_union": union["A"],
            "E_union": union["E"],
            "mu_union": union["mu_w"],
            "PG_union": union["PG"],
            "A_face": face_metrics["A"],
            "E_face": face_metrics["E"],
            "mu_face": face_metrics["mu_w"],
            "PG_face": face_metrics["PG"],
            "A_text": text_metrics["A"],
            "E_text": text_metrics["E"],
            "mu_text": text_metrics["mu_w"],
            "PG_text": text_metrics["PG"],
        })
        coverage_rows.append({
            "eval_split": str(row["eval_split"]),
            "variant": str(row["variant"]),
            "file_stem": str(row["file_stem"]),
            "image_path": image_path,
        })

        # attack visual audit for all annotated attacks
        rgb = np.asarray(Image.open(cache_path).convert("RGB"), dtype=np.uint8)
        stemname = Path(image_path).with_suffix("").name
        out_path = attack_vis_root / str(row["eval_split"]) / str(row["variant"]) / str(row["hardware_source"]) / f"{stemname}.png"
        title = f"{row['eval_split']} | {row['variant']} | {row['hardware_source']} | score={score:.6f} | clean_correct={bool(row['clean_correct'])}"
        save_attack_panel(rgb, amap, image_path, out_path, regions, title=title)
        attack_count += 1

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(STAGE11_ROOT / "localisation_per_image.csv", index=False)

    cov_df = pd.DataFrame(coverage_rows)
    cov_summary = (
        cov_df.groupby(["eval_split", "variant"], as_index=False)
        .agg(n_attacks=("image_path", "size"), n_annotated=("image_path", "size"), n_stems=("file_stem", "nunique"))
    )
    cov_summary["coverage"] = cov_summary["n_annotated"] / cov_summary["n_attacks"]
    cov_summary.to_csv(STAGE11_ROOT / "annotation_coverage.csv", index=False)

    def localisation_summary(source_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        rows = []
        for (eval_split, variant), sub in source_df.groupby(["eval_split", "variant"], sort=True):
            e_lo, e_hi = bootstrap_ci(sub, "E_union", n_boot=args.bootstrap_reps, seed=0)
            m_lo, m_hi = bootstrap_ci(sub, "mu_union", n_boot=args.bootstrap_reps, seed=1)
            pg_lo, pg_hi = bootstrap_ci(sub, "PG_union", n_boot=args.bootstrap_reps, seed=2)
            rows.append({
                "eval_split": str(eval_split),
                "group": str(variant),
                "n_images": int(len(sub)),
                "n_stems": int(sub["file_stem"].astype(str).nunique()),
                "mean_A": float(sub["A_union"].astype(float).mean()),
                "mean_E": float(sub["E_union"].astype(float).mean()),
                "mean_mu_w": float(sub["mu_union"].astype(float).mean()),
                "mean_PG": float(sub["PG_union"].astype(float).mean()),
                "mean_E_minus_A": float((sub["E_union"].astype(float) - sub["A_union"].astype(float)).mean()),
                "mean_E_ci_low": e_lo,
                "mean_E_ci_high": e_hi,
                "mean_mu_w_ci_low": m_lo,
                "mean_mu_w_ci_high": m_hi,
                "mean_PG_ci_low": pg_lo,
                "mean_PG_ci_high": pg_hi,
            })
        out = pd.DataFrame(rows)
        out.to_csv(STAGE11_ROOT / f"localisation_summary{suffix}.csv", index=False)
        return out

    all_loc_summary = localisation_summary(per_df, "")
    cc_loc_summary = localisation_summary(per_df.loc[per_df["clean_correct"].astype(bool)], "_clean_correct")

    # Selected bona-fide panels: top scoring bona-fides regardless of correctness.
    bona_top = (
        work.loc[work["label"].astype(int) == 0]
        .sort_values(["trufor_score", "image_path"], ascending=[False, True], kind="stable")
        .head(args.bonafide_panels)
        .copy()
    )
    bona_top.to_csv(STAGE11_ROOT / "selected_bonafide_panels.csv", index=False)

    # We need confidence maps only for these selected bona-fides.
    device = resolve_device(args.gpu)
    model, _, _ = load_trufor_model(device)
    bona_meta_rows = []
    for _, row in bona_top.iterrows():
        source = ROOT / str(row["cache_path"])
        result = infer_one(model, source, device, include_conf=True)
        amap = np.asarray(result["map"], dtype=np.float32)
        conf = np.asarray(result["conf"], dtype=np.float32)
        rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        stemname = Path(str(row["image_path"])).with_suffix("").name
        out_path = bona_vis_root / str(row["eval_split"]) / str(row["hardware_source"]) / f"{stemname}.png"
        title = (
            f"{row['eval_split']} | {row['hardware_source']} | {row['image_path']} | "
            f"score={float(row['trufor_score']):.6f} | pred={int(row['clean_prediction'])} | correct={bool(row['clean_correct'])}"
        )
        save_bonafide_panel(rgb, amap, conf, out_path, title=title)
        bona_meta_rows.append({
            "image_path": str(row["image_path"]),
            "panel_path": str(out_path.relative_to(ROOT)),
            "trufor_score": float(row["trufor_score"]),
            "clean_prediction": int(row["clean_prediction"]),
            "clean_correct": bool(row["clean_correct"]),
            "eval_split": str(row["eval_split"]),
            "hardware_source": str(row["hardware_source"]),
        })
    pd.DataFrame(bona_meta_rows).to_csv(STAGE11_ROOT / "bonafide_panel_manifest.csv", index=False)

    report_lines = [
        "LABPC PRETRAINED TRUFOR — FROZEN-THRESHOLD CLEAN EVALUATION",
        "",
        "Condition: policy_c_native ONLY (LAB-generated clean maps)",
        "Frozen threshold reused unchanged from HOME dev calibration",
        f"Frozen threshold: {threshold:.9f}",
        f"Threshold objective origin: {EXPECTED_OBJECTIVE}",
        "",
        "DETECTION (threshold frozen; no recalibration):",
        det_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "ANNOTATION COVERAGE:",
        cov_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "NATIVE LOCALISATION — altered-union primary metrics (ALL annotated attacks):",
        all_loc_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "NATIVE LOCALISATION — altered-union primary metrics (LAB clean-correct attacks only):",
        cc_loc_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "CLEAN-CORRECT ATTACK COUNTS (expected identical to HOME because Stage-10 had 0 flips):",
        clean_correct_attacks.groupby(["eval_split", "variant"], as_index=False).size().to_string(index=False),
        "",
        f"attack PNG panels rendered: {attack_count}",
        f"selected bona-fide anomaly+confidence panels rendered: {len(bona_top)}",
        f"clipped altered rectangles total: {clipped_total}",
        "",
        "STOP HERE before freezing the full multi-step optimiser.",
        "Proceed to the LAB six-image native input-gradient / VRAM pilot.",
    ]
    report = "\n".join(report_lines) + "\n"
    report_path = STAGE11_ROOT / "stage11_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": "PASS",
        "stage": "11_labpc_eval_frozen_threshold_accuracy",
        "run_tag": RUN_TAG,
        "frozen_threshold": threshold,
        "threshold_json": str(THRESHOLD_JSON.relative_to(ROOT)),
        "threshold_json_sha256": sha256_file(THRESHOLD_JSON),
        "lab_manifest": str((LAB_STAGE2_ROOT / 'inference_manifest.csv').relative_to(ROOT)),
        "lab_manifest_sha256": sha256_file(LAB_STAGE2_ROOT / 'inference_manifest.csv'),
        "n_images_total": 1844,
        "n_attack_images": int((work['label'].astype(int) == 1).sum()),
        "n_clean_correct_attacks": int(len(clean_correct_attacks)),
        "n_clean_correct_bonafides": int(len(clean_correct_bonafides)),
        "attack_png_panels": attack_count,
        "bonafide_panels": int(len(bona_top)),
        "report": str(report_path.relative_to(ROOT)),
        "scientific_contract": [
            "No threshold was fit or refit on LABPC.",
            "Localisation was measured from fresh LAB-generated clean anomaly maps.",
            "Attack visuals were rendered for all annotated attacks.",
            "Selected bona-fide panels were rerun with confidence output only for visual inspection.",
        ],
    }
    write_json(STAGE11_ROOT / "stage11_provenance.json", provenance)

    print(report)
    print(f"report: {report_path}")
    print(f"attack png root: {attack_vis_root}")
    print(f"bonafide png root: {bona_vis_root}")


if __name__ == "__main__":
    main()
