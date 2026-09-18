#!/usr/bin/env python3
"""LABPC Stage 10: cross-machine clean reproducibility + frozen-threshold gate.

No model inference occurs here.

This stage compares:
- canonical committed HOME Stage-2 inference manifest
- fresh LABPC Stage-09 inference manifest

Hard integrity requirements:
- same 1,844 image paths
- same labels/families/hardware
- same frozen Policy-C cache SHA256
- same native geometry
- same checkpoint SHA256
- canonical threshold JSON must still be anchored to the exact committed HOME
  Stage-2 manifest used for calibration

The threshold is NEVER refit. The already-frozen HOME dev calibration is
referenced unchanged and applied to LABPC scores:

    attack iff score >= t*

Outputs:
- per-image HOME-vs-LAB score/map-summary drift
- grouped drift diagnostics
- LABPC clean decision population at frozen t*
- LABPC clean-correct attack/bona-fide populations
- concise scientific gate report

Stop after this stage for review before regenerating LAB localisation / running
the six-image gradient pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    ROOT,
    sha256_file,
    write_json,
)

RUN_TAG = "LABPC"

HOME_ROOT = ROOT / "output" / "trufor_pretrained_policy_c_native"
HOME_MANIFEST = HOME_ROOT / "inference_manifest.csv"

LAB_ROOT = ROOT / "output" / RUN_TAG / "trufor_pretrained_policy_c_native"
LAB_MANIFEST = LAB_ROOT / "inference_manifest.csv"
LAB_PROVENANCE = LAB_ROOT / "stage09_inference_provenance.json"

THRESHOLD_JSON = (
    ROOT
    / "output"
    / "trufor_policy_c_frozen_protocol"
    / "stage03_dev_calibration_accuracy"
    / "frozen_threshold.json"
)

OUT_ROOT = ROOT / "output" / RUN_TAG / "trufor_cross_machine_reproducibility"

EXPECTED_OBJECTIVE = "maximize pooled ordinary image-level accuracy on dev_val"


def load_threshold() -> tuple[dict, float]:
    if not THRESHOLD_JSON.is_file():
        raise RuntimeError(f"Missing canonical frozen threshold: {THRESHOLD_JSON}")
    payload = json.loads(THRESHOLD_JSON.read_text())

    if payload.get("status") != "FROZEN":
        raise RuntimeError("Canonical threshold is not FROZEN")
    if payload.get("calibration_split") != "dev_val":
        raise RuntimeError("Canonical threshold was not calibrated on dev_val")
    if payload.get("selection_objective") != EXPECTED_OBJECTIVE:
        raise RuntimeError(
            "Unexpected canonical threshold objective:\n"
            f"{payload.get('selection_objective')!r}"
        )
    if payload.get("decision_rule") != "attack iff trufor_score >= threshold":
        raise RuntimeError("Unexpected frozen decision rule")
    if payload.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen threshold checkpoint SHA mismatch")

    threshold = float(payload["frozen_threshold"])
    if not np.isfinite(threshold):
        raise RuntimeError("Frozen threshold is non-finite")

    return payload, threshold


def require_manifests(threshold_payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for p in (HOME_MANIFEST, LAB_MANIFEST, LAB_PROVENANCE):
        if not p.is_file():
            raise RuntimeError(f"Required file missing: {p}")

    # Critical calibration provenance anchor: the HOME manifest currently in the
    # repo must be byte-identical to the manifest used to fit t*.
    expected_home_sha = str(threshold_payload.get("stage2_manifest_sha256", ""))
    actual_home_sha = sha256_file(HOME_MANIFEST)
    if expected_home_sha != actual_home_sha:
        raise RuntimeError(
            "Committed HOME Stage-2 manifest no longer matches the manifest used "
            "to calibrate the frozen threshold:\n"
            f"threshold JSON: {expected_home_sha}\n"
            f"HOME manifest:  {actual_home_sha}"
        )

    lab_prov = json.loads(LAB_PROVENANCE.read_text())
    if lab_prov.get("status") != "PASS":
        raise RuntimeError("LABPC Stage-09 provenance is not PASS")
    if lab_prov.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("LABPC Stage-09 checkpoint SHA mismatch")
    if lab_prov.get("run_tag") != RUN_TAG:
        raise RuntimeError("LABPC Stage-09 run_tag mismatch")

    home = pd.read_csv(HOME_MANIFEST, keep_default_na=False)
    lab = pd.read_csv(LAB_MANIFEST, keep_default_na=False)

    for name, frame in [("HOME", home), ("LAB", lab)]:
        if len(frame) != 1844:
            raise RuntimeError(f"{name} manifest length {len(frame)} != 1844")
        if frame["image_path"].duplicated().any():
            raise RuntimeError(f"{name} manifest has duplicate image_path")
        if set(frame["checkpoint_sha256"].astype(str).unique()) != {
            EXPECTED_CHECKPOINT_SHA256
        }:
            raise RuntimeError(f"{name} checkpoint SHA column mismatch")

    return home, lab, lab_prov


def hard_integrity_join(home: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    core = [
        "image_path",
        "eval_split",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
        "assigned_q",
        "cache_sha256",
        "native_height",
        "native_width",
        "trufor_score",
        "map_min",
        "map_max",
        "map_mean",
        "checkpoint_sha256",
    ]

    missing_home = [c for c in core if c not in home.columns]
    missing_lab = [c for c in core if c not in lab.columns]
    if missing_home or missing_lab:
        raise RuntimeError(
            f"Manifest schema mismatch. HOME missing={missing_home}, LAB missing={missing_lab}"
        )

    h = home[core].copy().add_suffix("_home")
    h = h.rename(columns={"image_path_home": "image_path"})
    l = lab[core].copy().add_suffix("_lab")
    l = l.rename(columns={"image_path_lab": "image_path"})

    merged = h.merge(l, on="image_path", how="outer", validate="one_to_one", indicator=True)
    if set(merged["_merge"].unique()) != {"both"}:
        bad = merged.loc[merged["_merge"] != "both", ["image_path", "_merge"]].head(20)
        raise RuntimeError(
            "HOME/LAB image-path population mismatch:\n"
            + bad.to_string(index=False)
        )
    merged = merged.drop(columns="_merge")

    exact_fields = [
        "eval_split",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
        "assigned_q",
        "cache_sha256",
        "native_height",
        "native_width",
        "checkpoint_sha256",
    ]

    for field in exact_fields:
        a = merged[f"{field}_home"].astype(str)
        b = merged[f"{field}_lab"].astype(str)
        bad = a != b
        if bad.any():
            sample = merged.loc[
                bad,
                ["image_path", f"{field}_home", f"{field}_lab"],
            ].head(20)
            raise RuntimeError(
                f"Hard HOME/LAB integrity mismatch in {field}:\n"
                + sample.to_string(index=False)
            )

    return merged


def detection_metrics(frame: pd.DataFrame, score_col: str, threshold: float) -> Dict[str, float]:
    labels = frame["label_home"].to_numpy(dtype=int)
    scores = frame[score_col].to_numpy(dtype=float)
    pred = (scores >= threshold).astype(int)
    attack = labels == 1
    bona = labels == 0

    recall = float(np.mean(pred[attack] == 1)) if attack.any() else np.nan
    spec = float(np.mean(pred[bona] == 0)) if bona.any() else np.nan
    return {
        "n": int(len(frame)),
        "accuracy": float(np.mean(pred == labels)),
        "balanced_accuracy": (
            float(0.5 * (recall + spec))
            if attack.any() and bona.any()
            else np.nan
        ),
        "attack_recall": recall,
        "bonafide_specificity": spec,
        "auroc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2
            else np.nan
        ),
    }


def grouped_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(scope: str, group: str, sub: pd.DataFrame) -> None:
        hd = detection_metrics(sub, "trufor_score_home", threshold)
        ld = detection_metrics(sub, "trufor_score_lab", threshold)
        absdiff = sub["score_abs_diff"].to_numpy(dtype=float)

        rows.append(
            {
                "scope": scope,
                "group": group,
                "n": int(len(sub)),
                "score_abs_diff_mean": float(np.mean(absdiff)),
                "score_abs_diff_median": float(np.median(absdiff)),
                "score_abs_diff_p95": float(np.quantile(absdiff, 0.95)),
                "score_abs_diff_max": float(np.max(absdiff)),
                "decision_flips": int(sub["decision_flip"].sum()),
                "home_accuracy": hd["accuracy"],
                "lab_accuracy": ld["accuracy"],
                "home_attack_recall": hd["attack_recall"],
                "lab_attack_recall": ld["attack_recall"],
                "home_bonafide_specificity": hd["bonafide_specificity"],
                "lab_bonafide_specificity": ld["bonafide_specificity"],
                "home_auroc": hd["auroc"],
                "lab_auroc": ld["auroc"],
            }
        )

    add("overall", "all", df)

    for split in sorted(df["eval_split_home"].astype(str).unique()):
        add(
            "eval_split",
            split,
            df.loc[df["eval_split_home"].astype(str) == split],
        )

    for family in sorted(
        df.loc[df["label_home"].astype(int) == 1, "variant_home"]
        .astype(str)
        .unique()
    ):
        add(
            "attack_family",
            family,
            df.loc[
                (df["label_home"].astype(int) == 1)
                & (df["variant_home"].astype(str) == family)
            ],
        )

    for hardware in sorted(df["hardware_source_home"].astype(str).unique()):
        add(
            "hardware",
            hardware,
            df.loc[df["hardware_source_home"].astype(str) == hardware],
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    threshold_payload, threshold = load_threshold()
    home, lab, lab_prov = require_manifests(threshold_payload)
    merged = hard_integrity_join(home, lab)

    merged["score_signed_diff_lab_minus_home"] = (
        merged["trufor_score_lab"].astype(float)
        - merged["trufor_score_home"].astype(float)
    )
    merged["score_abs_diff"] = merged["score_signed_diff_lab_minus_home"].abs()

    for name in ("map_min", "map_max", "map_mean"):
        merged[f"{name}_signed_diff_lab_minus_home"] = (
            merged[f"{name}_lab"].astype(float)
            - merged[f"{name}_home"].astype(float)
        )
        merged[f"{name}_abs_diff"] = (
            merged[f"{name}_signed_diff_lab_minus_home"].abs()
        )

    merged["home_prediction"] = (
        merged["trufor_score_home"].astype(float) >= threshold
    ).astype(int)
    merged["lab_prediction"] = (
        merged["trufor_score_lab"].astype(float) >= threshold
    ).astype(int)
    merged["decision_flip"] = (
        merged["home_prediction"] != merged["lab_prediction"]
    )
    merged["home_correct"] = (
        merged["home_prediction"] == merged["label_home"].astype(int)
    )
    merged["lab_correct"] = (
        merged["lab_prediction"] == merged["label_home"].astype(int)
    )

    merged["home_distance_to_threshold"] = (
        merged["trufor_score_home"].astype(float) - threshold
    )
    merged["lab_distance_to_threshold"] = (
        merged["trufor_score_lab"].astype(float) - threshold
    )

    per_image_path = OUT_ROOT / "home_vs_lab_per_image.csv"
    merged.to_csv(per_image_path, index=False)

    grouped = grouped_summary(merged, threshold)
    grouped_path = OUT_ROOT / "home_vs_lab_grouped.csv"
    grouped.to_csv(grouped_path, index=False)

    # Freeze LAB clean decision populations at the already-canonical threshold.
    lab_population = lab.copy()
    lab_population["frozen_threshold"] = threshold
    lab_population["clean_prediction"] = (
        lab_population["trufor_score"].astype(float) >= threshold
    ).astype(int)
    lab_population["clean_correct"] = (
        lab_population["clean_prediction"].astype(int)
        == lab_population["label"].astype(int)
    )

    population_path = OUT_ROOT / "lab_clean_decision_population.csv"
    lab_population.to_csv(population_path, index=False)

    lab_correct = lab_population.loc[lab_population["clean_correct"]].copy()
    attacks_path = OUT_ROOT / "lab_clean_correct_attacks.csv"
    bona_path = OUT_ROOT / "lab_clean_correct_bonafides.csv"
    lab_correct.loc[lab_correct["label"].astype(int) == 1].to_csv(
        attacks_path, index=False
    )
    lab_correct.loc[lab_correct["label"].astype(int) == 0].to_csv(
        bona_path, index=False
    )

    attack_counts_home = (
        merged.loc[merged["label_home"].astype(int) == 1]
        .groupby(["eval_split_home", "variant_home"], as_index=False)
        .agg(
            n=("image_path", "size"),
            home_clean_correct=("home_correct", "sum"),
            lab_clean_correct=("lab_correct", "sum"),
            decision_flips=("decision_flip", "sum"),
        )
    )
    attack_counts_path = OUT_ROOT / "attack_clean_correct_comparison.csv"
    attack_counts_home.to_csv(attack_counts_path, index=False)

    overall = grouped.loc[
        (grouped["scope"] == "overall") & (grouped["group"] == "all")
    ].iloc[0]

    flips = merged.loc[merged["decision_flip"]].copy()
    flips = flips.sort_values(
        ["score_abs_diff", "image_path"],
        ascending=[False, True],
    )
    flips_path = OUT_ROOT / "decision_flips.csv"
    flips.to_csv(flips_path, index=False)

    max_map_mean_diff = float(merged["map_mean_abs_diff"].max())
    max_score_diff = float(merged["score_abs_diff"].max())
    med_score_diff = float(merged["score_abs_diff"].median())
    p95_score_diff = float(merged["score_abs_diff"].quantile(0.95))
    n_flips = int(merged["decision_flip"].sum())

    report_lines = [
        "TRUFOR LABPC CROSS-MACHINE CLEAN REPRODUCIBILITY GATE",
        "",
        f"Frozen threshold (NOT refit): {threshold:.9f}",
        f"Threshold objective origin: {EXPECTED_OBJECTIVE}",
        f"HOME manifest SHA256: {sha256_file(HOME_MANIFEST)}",
        f"LAB manifest SHA256:  {sha256_file(LAB_MANIFEST)}",
        f"LAB GPU: {lab_prov.get('gpu')}",
        f"LAB torch: {lab_prov.get('torch')}",
        f"LAB CUDA: {lab_prov.get('torch_cuda')}",
        "",
        "HARD INPUT/POPULATION INTEGRITY",
        "  image paths: PASS",
        "  labels/families/hardware: PASS",
        "  Policy-C cache SHA256 values: PASS",
        "  native geometry: PASS",
        "  TruFor checkpoint SHA256: PASS",
        "",
        "NUMERICAL DRIFT",
        f"  score abs diff median: {med_score_diff:.9g}",
        f"  score abs diff p95:    {p95_score_diff:.9g}",
        f"  score abs diff max:    {max_score_diff:.9g}",
        f"  map-mean abs diff max: {max_map_mean_diff:.9g}",
        f"  frozen-threshold decision flips: {n_flips}/1844",
        "",
        "OVERALL DETECTION AT THE SAME FROZEN THRESHOLD",
        (
            f"  HOME accuracy={overall['home_accuracy']:.6f} "
            f"attack_recall={overall['home_attack_recall']:.6f} "
            f"BF_specificity={overall['home_bonafide_specificity']:.6f}"
        ),
        (
            f"  LAB  accuracy={overall['lab_accuracy']:.6f} "
            f"attack_recall={overall['lab_attack_recall']:.6f} "
            f"BF_specificity={overall['lab_bonafide_specificity']:.6f}"
        ),
        "",
        "CLEAN-CORRECT ATTACK COUNTS",
        attack_counts_home.to_string(index=False),
        "",
    ]

    if n_flips == 0:
        report_lines += [
            "SCIENTIFIC GATE",
            "No clean decision membership changed at the frozen threshold.",
            "This is strong evidence that machine-level floating-point drift does not",
            "alter the clean-correct attack population used by the next LAB stages.",
        ]
    else:
        report_lines += [
            "SCIENTIFIC GATE",
            f"{n_flips} clean decisions changed between HOME and LAB at the frozen threshold.",
            "Do NOT recalibrate. Inspect decision_flips.csv and the score drift before",
            "freezing LAB localisation/adversarial populations.",
        ]

    report_lines += [
        "",
        "STOP HERE.",
        "Do not run LAB localisation or the six-image gradient pilot until this",
        "cross-machine report has been reviewed.",
    ]

    report = "\n".join(report_lines) + "\n"
    report_path = OUT_ROOT / "reproducibility_report.txt"
    report_path.write_text(report)

    provenance = {
        "status": "PASS_INTEGRITY_REVIEW_NUMERICS",
        "stage": "10_labpc_cross_machine_reproducibility",
        "run_tag": RUN_TAG,
        "frozen_threshold": threshold,
        "threshold_json": str(THRESHOLD_JSON.relative_to(ROOT)),
        "threshold_json_sha256": sha256_file(THRESHOLD_JSON),
        "threshold_origin_home_manifest_sha256": sha256_file(HOME_MANIFEST),
        "lab_manifest": str(LAB_MANIFEST.relative_to(ROOT)),
        "lab_manifest_sha256": sha256_file(LAB_MANIFEST),
        "n_images": 1844,
        "n_decision_flips": n_flips,
        "score_abs_diff_median": med_score_diff,
        "score_abs_diff_p95": p95_score_diff,
        "score_abs_diff_max": max_score_diff,
        "map_mean_abs_diff_max": max_map_mean_diff,
        "lab_clean_correct_attacks": str(attacks_path.relative_to(ROOT)),
        "lab_clean_correct_attacks_sha256": sha256_file(attacks_path),
        "lab_clean_correct_bonafides": str(bona_path.relative_to(ROOT)),
        "lab_clean_correct_bonafides_sha256": sha256_file(bona_path),
        "report": str(report_path.relative_to(ROOT)),
        "scientific_contract": [
            "No threshold was fit or refit on LABPC.",
            "The canonical HOME dev-calibrated threshold was applied unchanged.",
            "Exact Policy-C cache hashes and native geometries were required to match.",
            "Numerical drift is reported rather than silently normalised away.",
        ],
    }
    write_json(OUT_ROOT / "stage10_reproducibility_provenance.json", provenance)

    print(report)
    print(f"report: {report_path}")
    print(f"per-image: {per_image_path}")
    print(f"grouped: {grouped_path}")
    print(f"decision flips: {flips_path}")


if __name__ == "__main__":
    main()
