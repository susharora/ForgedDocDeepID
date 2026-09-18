#!/usr/bin/env python3
"""Replacement Stage 3: calibrate ONE TruFor decision threshold on dev_val.

This supersedes the discarded balanced-accuracy calibration while preserving it
for provenance. It reads ONLY the frozen Stage-2 Policy-C-native inference
manifest and ONLY dev_val rows. Official-test scores are never used to choose
the threshold.

Scientific contract
-------------------
- pretrained TruFor weights remain unchanged;
- Policy-C native inference remains unchanged;
- the only fitted quantity is one scalar detector operating threshold;
- objective: pooled image-level ACCURACY on all 459 dev images;
- no family-specific or hardware-specific thresholds;
- bootstrap is diagnostic only and never changes the point estimate.

Decision rule: attack iff trufor_score >= threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    OUT_ROOT as STAGE2_ROOT,
    ROOT,
    project_git_head,
    relative_to_root,
    sha256_file,
    verify_trufor_provenance,
    write_json,
)

PROTOCOL_ROOT = ROOT / "output" / "trufor_policy_c_frozen_protocol"
CALIB_ROOT = PROTOCOL_ROOT / "stage03_dev_calibration_accuracy"
THRESHOLD_JSON = CALIB_ROOT / "frozen_threshold.json"
DISCARDED_BALANCED_JSON = (
    PROTOCOL_ROOT / "stage03_dev_calibration" / "frozen_threshold.json"
)

N_BOOT_DEFAULT = 5000
BOOT_SEED = 1701
UPSTREAM_THRESHOLD = 0.5
TIE_TOL = 1e-12


def require_stage2_manifest() -> Tuple[pd.DataFrame, Path, dict]:
    manifest_path = STAGE2_ROOT / "inference_manifest.csv"
    provenance_path = STAGE2_ROOT / "stage02_inference_provenance.json"

    if not manifest_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(
            "Frozen Stage-2 outputs are missing. Required:\n"
            f"  {manifest_path}\n  {provenance_path}"
        )

    provenance = json.loads(provenance_path.read_text())
    if provenance.get("status") != "PASS":
        raise RuntimeError("Stage-2 provenance does not record PASS")
    if provenance.get("condition") != "policy_c_native":
        raise RuntimeError("Stage-2 condition is not policy_c_native")
    if provenance.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Stage-2 checkpoint SHA mismatch")

    frame = pd.read_csv(manifest_path, keep_default_na=False)
    if len(frame) != 1844:
        raise RuntimeError(f"Stage-2 manifest incomplete: {len(frame)} != 1844")
    if frame["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in Stage-2 manifest")
    if set(frame["checkpoint_sha256"].astype(str).unique()) != {
        EXPECTED_CHECKPOINT_SHA256
    }:
        raise RuntimeError("Stage-2 manifest checkpoint SHA mismatch")
    if set(frame["condition"].astype(str).unique()) != {"policy_c_native"}:
        raise RuntimeError("Stage-2 manifest contains a non-Policy-C condition")

    split_counts = frame["eval_split"].value_counts().to_dict()
    if split_counts != {"official_test": 1385, "dev_val": 459}:
        raise RuntimeError(f"Unexpected Stage-2 split counts: {split_counts}")

    return frame, manifest_path, provenance


def threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if scores.shape != labels.shape:
        raise RuntimeError("score/label shape mismatch")
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite detector score")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise RuntimeError("labels must be binary 0/1")

    pred = (scores >= float(threshold)).astype(np.int64)
    attack = labels == 1
    bona = labels == 0

    tp = int(np.sum(pred[attack] == 1)) if attack.any() else 0
    fn = int(np.sum(pred[attack] == 0)) if attack.any() else 0
    tn = int(np.sum(pred[bona] == 0)) if bona.any() else 0
    fp = int(np.sum(pred[bona] == 1)) if bona.any() else 0

    accuracy = float(np.mean(pred == labels))
    recall = float(tp / (tp + fn)) if attack.any() else np.nan
    specificity = float(tn / (tn + fp)) if bona.any() else np.nan
    bacc = (
        float(0.5 * (recall + specificity))
        if attack.any() and bona.any()
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": bacc,
        "attack_recall": recall,
        "bonafide_specificity": specificity,
        "n_correct": int(np.sum(pred == labels)),
        "n": int(len(labels)),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def all_partition_thresholds(scores: np.ndarray) -> np.ndarray:
    """One threshold for every distinct binary partition induced by dev scores."""
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if unique.size < 2:
        raise RuntimeError("Need at least two distinct dev detector scores")

    internal = (unique[:-1] + unique[1:]) / 2.0
    low = np.nextafter(unique[0], -np.inf)
    high = np.nextafter(unique[-1], np.inf)
    return np.concatenate(([low], internal, [high]))


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    candidates = all_partition_thresholds(scores)
    curve = pd.DataFrame(
        [threshold_metrics(scores, labels, float(t)) for t in candidates]
    )

    # PRIMARY OBJECTIVE: ordinary pooled image-level accuracy.
    best = float(curve["accuracy"].max())
    tied = curve.loc[
        np.abs(curve["accuracy"].to_numpy(dtype=float) - best) <= TIE_TOL
    ].copy()

    # Tie-break is fixed in advance and does not inspect official test.
    tied["distance_to_upstream_0p5"] = np.abs(
        tied["threshold"].to_numpy(dtype=float) - UPSTREAM_THRESHOLD
    )
    tied = tied.sort_values(
        ["distance_to_upstream_0p5", "threshold"],
        ascending=[True, True],
        kind="mergesort",
    )
    selected = float(tied.iloc[0]["threshold"])

    curve["selected"] = np.isclose(
        curve["threshold"].to_numpy(dtype=float),
        selected,
        rtol=0.0,
        atol=0.0,
    )
    return selected, curve


def frame_metrics(frame: pd.DataFrame, threshold: float) -> Dict[str, float]:
    scores = frame["trufor_score"].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=int)
    out = threshold_metrics(scores, labels, threshold)
    out["auroc"] = (
        float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else np.nan
    )
    return out


def grouped_diagnostics(
    dev: pd.DataFrame,
    threshold: float,
    threshold_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(scope: str, group: str, subset: pd.DataFrame) -> None:
        m = frame_metrics(subset, threshold)
        rows.append(
            {
                "threshold_name": threshold_name,
                "threshold": float(threshold),
                "scope": scope,
                "group": group,
                "n": int(len(subset)),
                "n_stems": int(subset["file_stem"].nunique()),
                "n_attack": int((subset["label"].astype(int) == 1).sum()),
                "n_bonafide": int((subset["label"].astype(int) == 0).sum()),
                **{k: v for k, v in m.items() if k != "threshold"},
            }
        )

    add("overall", "all", dev)

    # Attack-family recall: no bona-fides mixed into these rows.
    attacks = dev.loc[dev["label"].astype(int) == 1]
    for family in sorted(attacks["variant"].astype(str).unique()):
        add(
            "attack_family_only",
            family,
            attacks.loc[attacks["variant"].astype(str) == family],
        )

    # Family-vs-bona rows retain a conventional binary metric view.
    bona_mask = dev["label"].astype(int) == 0
    for family in sorted(attacks["variant"].astype(str).unique()):
        add(
            "attack_family_vs_bonafide",
            family,
            dev.loc[bona_mask | (dev["variant"].astype(str) == family)],
        )

    for hardware in sorted(dev["hardware_source"].astype(str).unique()):
        add(
            "hardware",
            hardware,
            dev.loc[dev["hardware_source"].astype(str) == hardware],
        )

    return pd.DataFrame(rows)


def stem_bootstrap(
    dev: pd.DataFrame,
    frozen_threshold: float,
    n_boot: int,
) -> pd.DataFrame:
    stems = sorted(dev["file_stem"].astype(str).unique())
    if len(stems) != 51:
        raise RuntimeError(f"Expected 51 dev stems, got {len(stems)}")

    clusters = {
        stem: dev.loc[dev["file_stem"].astype(str) == stem].copy()
        for stem in stems
    }
    rng = np.random.default_rng(BOOT_SEED)
    rows: List[Dict[str, object]] = []

    for b in range(n_boot):
        sampled = rng.choice(stems, size=len(stems), replace=True)
        boot = pd.concat([clusters[str(s)] for s in sampled], ignore_index=True)
        scores = boot["trufor_score"].to_numpy(dtype=float)
        labels = boot["label"].to_numpy(dtype=int)

        selected, _ = select_threshold(scores, labels)
        fixed = threshold_metrics(scores, labels, frozen_threshold)
        chosen = threshold_metrics(scores, labels, selected)

        rows.append(
            {
                "replicate": b,
                "selected_threshold": float(selected),
                "selected_accuracy": chosen["accuracy"],
                "selected_balanced_accuracy": chosen["balanced_accuracy"],
                "fixed_threshold": float(frozen_threshold),
                "fixed_accuracy": fixed["accuracy"],
                "fixed_balanced_accuracy": fixed["balanced_accuracy"],
                "fixed_attack_recall": fixed["attack_recall"],
                "fixed_bonafide_specificity": fixed["bonafide_specificity"],
            }
        )

    return pd.DataFrame(rows)


def quantile_ci(values: Iterable[float]) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(arr))
    low, high = np.quantile(arr, [0.025, 0.975])
    return median, float(low), float(high)


def family_recall(diagnostics: pd.DataFrame, threshold_name: str, family: str) -> float:
    row = diagnostics.loc[
        (diagnostics["threshold_name"] == threshold_name)
        & (diagnostics["scope"] == "attack_family_only")
        & (diagnostics["group"] == family)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique family diagnostic: {threshold_name} {family}")
    return float(row.iloc[0]["attack_recall"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument(
        "--overwrite-frozen-threshold",
        action="store_true",
        help=(
            "allow replacing an existing ACCURACY-calibrated threshold. Do not "
            "use after the corresponding Stage 4 has begun."
        ),
    )
    args = parser.parse_args()

    if args.n_bootstrap < 100:
        raise RuntimeError("Use at least 100 bootstrap replicates")

    CALIB_ROOT.mkdir(parents=True, exist_ok=True)
    if THRESHOLD_JSON.exists() and not args.overwrite_frozen_threshold:
        raise RuntimeError(
            f"Accuracy-calibrated frozen threshold already exists:\n{THRESHOLD_JSON}\n"
            "Refusing to refit it."
        )

    verify_trufor_provenance(check_archive_member=False)
    inference, manifest_path, _ = require_stage2_manifest()

    # SCIENTIFIC FIREWALL: threshold fitting sees only dev_val rows.
    dev = inference.loc[inference["eval_split"].astype(str) == "dev_val"].copy()
    if len(dev) != 459:
        raise RuntimeError(f"Expected 459 dev rows, got {len(dev)}")
    if dev["label"].astype(int).value_counts().to_dict() != {1: 306, 0: 153}:
        raise RuntimeError("Unexpected dev class counts")
    if dev["file_stem"].nunique() != 51:
        raise RuntimeError("Expected 51 dev stems")

    scores = dev["trufor_score"].to_numpy(dtype=float)
    labels = dev["label"].to_numpy(dtype=int)

    threshold, curve = select_threshold(scores, labels)
    selected_metrics = threshold_metrics(scores, labels, threshold)
    upstream_metrics = threshold_metrics(scores, labels, UPSTREAM_THRESHOLD)
    dev_auc = float(roc_auc_score(labels, scores))

    curve_path = CALIB_ROOT / "threshold_curve.csv"
    curve.to_csv(curve_path, index=False)

    diagnostics = pd.concat(
        [
            grouped_diagnostics(dev, UPSTREAM_THRESHOLD, "upstream_0p5"),
            grouped_diagnostics(dev, threshold, "dev_accuracy_calibrated"),
        ],
        ignore_index=True,
    )
    diagnostics_path = CALIB_ROOT / "dev_detection_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    d1_05 = family_recall(diagnostics, "upstream_0p5", "digital_1")
    d2_05 = family_recall(diagnostics, "upstream_0p5", "digital_2")
    d1_cal = family_recall(diagnostics, "dev_accuracy_calibrated", "digital_1")
    d2_cal = family_recall(diagnostics, "dev_accuracy_calibrated", "digital_2")

    print("TRUFOR STAGE 3A — DEV THRESHOLD CALIBRATION BY ORDINARY ACCURACY")
    print("model weights: unchanged")
    print("condition: policy_c_native")
    print("selection data: dev_val ONLY (459 images / 51 stems)")
    print("objective: maximum pooled IMAGE-LEVEL ACCURACY")
    print("tie-break: nearest to upstream 0.5, then lower threshold")
    print(f"upstream threshold: {UPSTREAM_THRESHOLD:.9f}")
    print(f"frozen candidate:   {threshold:.9f}")
    print(f"dev AUROC:          {dev_auc:.6f}")
    print(
        "dev @0.5:        "
        f"acc={upstream_metrics['accuracy']:.4f} "
        f"bACC={upstream_metrics['balanced_accuracy']:.4f} "
        f"recall={upstream_metrics['attack_recall']:.4f} "
        f"specificity={upstream_metrics['bonafide_specificity']:.4f} "
        f"d1={d1_05:.4f} d2={d2_05:.4f} "
        f"correct={upstream_metrics['n_correct']}/{upstream_metrics['n']}"
    )
    print(
        "dev @calibrated: "
        f"acc={selected_metrics['accuracy']:.4f} "
        f"bACC={selected_metrics['balanced_accuracy']:.4f} "
        f"recall={selected_metrics['attack_recall']:.4f} "
        f"specificity={selected_metrics['bonafide_specificity']:.4f} "
        f"d1={d1_cal:.4f} d2={d2_cal:.4f} "
        f"correct={selected_metrics['n_correct']}/{selected_metrics['n']}"
    )

    bootstrap = stem_bootstrap(dev, threshold, args.n_bootstrap)
    bootstrap_path = CALIB_ROOT / "stem_bootstrap.csv"
    bootstrap.to_csv(bootstrap_path, index=False)

    t_med, t_low, t_high = quantile_ci(bootstrap["selected_threshold"])
    a_med, a_low, a_high = quantile_ci(bootstrap["fixed_accuracy"])
    b_med, b_low, b_high = quantile_ci(bootstrap["fixed_balanced_accuracy"])
    r_med, r_low, r_high = quantile_ci(bootstrap["fixed_attack_recall"])
    s_med, s_low, s_high = quantile_ci(bootstrap["fixed_bonafide_specificity"])

    discarded = None
    if DISCARDED_BALANCED_JSON.is_file():
        discarded = {
            "path": relative_to_root(DISCARDED_BALANCED_JSON),
            "sha256": sha256_file(DISCARDED_BALANCED_JSON),
            "status": "scientifically_discarded_before_final_test_protocol",
            "reason": "balanced-accuracy objective gave equal total class weight and sacrificed too many attacks",
        }

    payload = {
        "status": "FROZEN",
        "stage": "03_calibrate_dev_threshold_accuracy",
        "project_git_head": project_git_head(),
        "calibration_script_sha256": sha256_file(Path(__file__).resolve()),
        "condition": "policy_c_native",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "stage2_manifest": relative_to_root(manifest_path),
        "stage2_manifest_sha256": sha256_file(manifest_path),
        "calibration_split": "dev_val",
        "calibration_n_images": 459,
        "calibration_n_stems": 51,
        "calibration_n_attack": 306,
        "calibration_n_bonafide": 153,
        "upstream_reference_threshold": UPSTREAM_THRESHOLD,
        "selection_objective": "maximize pooled ordinary image-level accuracy on dev_val",
        "threshold_candidate_rule": "midpoints between adjacent distinct dev scores plus endpoint partitions",
        "tie_break": "nearest to 0.5; residual tie -> lower threshold",
        "decision_rule": "attack iff trufor_score >= threshold",
        "frozen_threshold": float(threshold),
        "dev_auroc": dev_auc,
        "dev_at_upstream_0p5": upstream_metrics,
        "dev_at_frozen_threshold": selected_metrics,
        "dev_attack_family_recall": {
            "upstream_0p5": {"digital_1": d1_05, "digital_2": d2_05},
            "frozen_threshold": {"digital_1": d1_cal, "digital_2": d2_cal},
        },
        "discarded_previous_balanced_accuracy_calibration": discarded,
        "bootstrap": {
            "cluster": "file_stem",
            "replicates": int(args.n_bootstrap),
            "seed": BOOT_SEED,
            "selected_threshold_median": t_med,
            "selected_threshold_ci_low": t_low,
            "selected_threshold_ci_high": t_high,
            "fixed_threshold_accuracy_median": a_med,
            "fixed_threshold_accuracy_ci_low": a_low,
            "fixed_threshold_accuracy_ci_high": a_high,
            "fixed_threshold_bacc_median": b_med,
            "fixed_threshold_bacc_ci_low": b_low,
            "fixed_threshold_bacc_ci_high": b_high,
            "fixed_threshold_attack_recall_median": r_med,
            "fixed_threshold_attack_recall_ci_low": r_low,
            "fixed_threshold_attack_recall_ci_high": r_high,
            "fixed_threshold_bonafide_specificity_median": s_med,
            "fixed_threshold_bonafide_specificity_ci_low": s_low,
            "fixed_threshold_bonafide_specificity_ci_high": s_high,
        },
        "scientific_contract": [
            "No TruFor weights were adapted to FantasyID.",
            "No official-test score was used to choose the threshold.",
            "Exactly one global threshold is frozen for all families and hardware.",
            "Every dev image has equal weight in the calibration objective.",
            "Bootstrap is diagnostic and does not alter the frozen point estimate.",
            "The earlier balanced-accuracy calibration is preserved only as discarded provenance.",
        ],
    }
    write_json(THRESHOLD_JSON, payload)

    report = "\n".join(
        [
            "TRUFOR DEV THRESHOLD FREEZE — ORDINARY ACCURACY",
            "",
            "Pretrained TruFor weights: unchanged",
            "Condition: policy_c_native",
            "Calibration set: dev_val only (459 images, 51 stems)",
            "Objective: maximum pooled image-level accuracy",
            "Every dev image has equal objective weight.",
            "Tie-break: closest to upstream 0.5; then lower threshold",
            "",
            f"Dev AUROC: {dev_auc:.6f}",
            f"Upstream threshold: {UPSTREAM_THRESHOLD:.9f}",
            f"Frozen threshold:   {threshold:.9f}",
            "",
            "DEV OPERATING POINTS",
            (
                f"0.5        acc={upstream_metrics['accuracy']:.4f} "
                f"bACC={upstream_metrics['balanced_accuracy']:.4f} "
                f"attack_recall={upstream_metrics['attack_recall']:.4f} "
                f"bonafide_specificity={upstream_metrics['bonafide_specificity']:.4f} "
                f"d1_recall={d1_05:.4f} d2_recall={d2_05:.4f} "
                f"correct={upstream_metrics['n_correct']}/{upstream_metrics['n']}"
            ),
            (
                f"calibrated acc={selected_metrics['accuracy']:.4f} "
                f"bACC={selected_metrics['balanced_accuracy']:.4f} "
                f"attack_recall={selected_metrics['attack_recall']:.4f} "
                f"bonafide_specificity={selected_metrics['bonafide_specificity']:.4f} "
                f"d1_recall={d1_cal:.4f} d2_recall={d2_cal:.4f} "
                f"correct={selected_metrics['n_correct']}/{selected_metrics['n']}"
            ),
            "",
            "STEM-CLUSTER BOOTSTRAP DIAGNOSTIC",
            f"selected threshold median={t_med:.6f}, 95% CI=[{t_low:.6f}, {t_high:.6f}]",
            f"fixed-threshold accuracy median={a_med:.4f}, 95% CI=[{a_low:.4f}, {a_high:.4f}]",
            f"fixed-threshold bACC median={b_med:.4f}, 95% CI=[{b_low:.4f}, {b_high:.4f}]",
            "",
            "SCIENTIFIC GATE",
            "Inspect the selected threshold, bootstrap stability, overall attack recall,",
            "d1/d2 recall, bona-fide specificity, and hardware diagnostics BEFORE Stage 4.",
            "Do not run Stage 4 if calibration achieves accuracy by an unacceptable family collapse.",
        ]
    ) + "\n"

    report_path = CALIB_ROOT / "calibration_report.txt"
    report_path.write_text(report)

    print("\n" + report)
    print("STAGE 3A PASS — ACCURACY THRESHOLD WRITTEN")
    print(f"threshold:   {THRESHOLD_JSON}")
    print(f"diagnostics: {diagnostics_path}")
    print(f"report:      {report_path}")
    print("STOP HERE FOR SCIENTIFIC REVIEW BEFORE STAGE 4.")


if __name__ == "__main__":
    main()
