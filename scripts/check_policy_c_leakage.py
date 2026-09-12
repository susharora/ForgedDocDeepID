#!/usr/bin/env python3
"""Disambiguate residual Policy C JPEG leakage from blockiness signal."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "output/compression_leakage.csv"
OUTPUT = ROOT / "output/compression_leakage_policy_c_summary.csv"

SEED = 10
N_BOOT = 5000


TABLE_FEATURES = [
    "jpeg_quality",
    "q_luma_mean",
    "q_chroma_mean",
]

STRICT_HISTORY_FEATURES = [
    *TABLE_FEATURES,
    "lattice_q75_mean",
    "lattice_q75_p90",
    "lattice_current_mean",
    "lattice_current_p90",
    "recompress_current_mae",
]

BLOCKINESS_FEATURES = [
    "blockiness_h",
    "blockiness_v",
]

FEATURE_SETS = {
    "final_table_only": TABLE_FEATURES,
    "strict_jpeg_history": STRICT_HISTORY_FEATURES,
    "blockiness_only": BLOCKINESS_FEATURES,
    "all_compression_features": (
        STRICT_HISTORY_FEATURES
        + BLOCKINESS_FEATURES
    ),
}


def load_policy_c():
    df = pd.read_csv(INPUT)

    required = {
        "split",
        "file_stem",
        "traffic_type",
        "label",
        "policy",
        *STRICT_HISTORY_FEATURES,
        *BLOCKINESS_FEATURES,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        df.policy == "C_q75_then_q50_90"
    ].copy()

    train = df[
        df.split == "project_train"
    ].copy()

    dev = df[
        df.split == "dev_val"
    ].copy()

    if len(train) != 1440:
        raise RuntimeError(
            f"Expected 1440 train rows, got {len(train)}"
        )

    if len(dev) != 459:
        raise RuntimeError(
            f"Expected 459 dev rows, got {len(dev)}"
        )

    if train.file_stem.nunique() != 160:
        raise RuntimeError(
            "Expected 160 train card identities"
        )

    if dev.file_stem.nunique() != 51:
        raise RuntimeError(
            "Expected 51 dev card identities"
        )

    overlap = (
        set(train.file_stem)
        & set(dev.file_stem)
    )

    if overlap:
        raise RuntimeError(
            f"Identity leakage: {len(overlap)} stems"
        )

    if set(df.label.unique()) != {0, 1}:
        raise RuntimeError(
            "Unexpected label values"
        )

    if not np.array_equal(
        df.label.to_numpy(),
        (df.traffic_type == "attack")
        .astype(int)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Label polarity mismatch"
        )

    return train, dev


def fit_model(train, features):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
    )

    model.fit(
        train[features],
        train.label,
    )

    return model


def stem_bootstrap_auc(dev, scores):
    """
    Cluster bootstrap by card identity.

    All nine images belonging to a sampled card stem move together.
    This avoids pretending the 459 dev images are independent.
    """

    work = dev[
        [
            "file_stem",
            "label",
        ]
    ].copy()

    work["score"] = scores

    stems = np.array(
        sorted(work.file_stem.unique())
    )

    groups = {
        stem: work[
            work.file_stem == stem
        ][["label", "score"]].to_numpy()
        for stem in stems
    }

    rng = np.random.default_rng(SEED)

    aucs = np.empty(
        N_BOOT,
        dtype=float,
    )

    for i in range(N_BOOT):
        sampled = rng.choice(
            stems,
            size=len(stems),
            replace=True,
        )

        boot = np.concatenate(
            [groups[s] for s in sampled],
            axis=0,
        )

        aucs[i] = roc_auc_score(
            boot[:, 0],
            boot[:, 1],
        )

    return (
        float(np.quantile(aucs, 0.025)),
        float(np.quantile(aucs, 0.975)),
        float(np.mean(aucs)),
    )


def evaluate(train, dev):
    rows = []

    print(
        "Policy C residual leakage analysis\n"
    )

    print(
        f"train: {len(train)} images / "
        f"{train.file_stem.nunique()} stems"
    )

    print(
        f"dev:   {len(dev)} images / "
        f"{dev.file_stem.nunique()} stems\n"
    )

    for name, features in FEATURE_SETS.items():
        model = fit_model(
            train,
            features,
        )

        train_score = model.predict_proba(
            train[features]
        )[:, 1]

        dev_score = model.predict_proba(
            dev[features]
        )[:, 1]

        train_auc = roc_auc_score(
            train.label,
            train_score,
        )

        dev_auc = roc_auc_score(
            dev.label,
            dev_score,
        )

        ci_low, ci_high, boot_mean = (
            stem_bootstrap_auc(
                dev,
                dev_score,
            )
        )

        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "train_auc": train_auc,
                "dev_auc": dev_auc,
                "bootstrap_mean_auc": boot_mean,
                "stem_bootstrap_ci_low": ci_low,
                "stem_bootstrap_ci_high": ci_high,
                "features": "|".join(features),
            }
        )

        print(
            f"{name:25s} "
            f"train={train_auc:.4f}  "
            f"dev={dev_auc:.4f}  "
            f"stem-95%CI="
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )

    return pd.DataFrame(rows)


def matched_final_q_check(dev):
    """
    Policy C deliberately gives attack and bonafide images from the
    same card+hardware pair the same final JPEG quality.

    Confirm the intervention itself did not introduce a label/Q mismatch.
    """

    if "assigned_q" not in dev.columns:
        return

    grouped = (
        dev.groupby(
            [
                "file_stem",
                "hardware_source",
            ]
        )
        .assigned_q
        .nunique()
    )

    if grouped.max() != 1:
        raise RuntimeError(
            "Final Q differs within a "
            "card/hardware matched group"
        )

    print(
        "\nFinal-Q matching check: PASS"
    )


def main():
    train, dev = load_policy_c()

    matched_final_q_check(dev)

    summary = evaluate(
        train,
        dev,
    )

    summary.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"\nWrote {OUTPUT}"
    )

    print(
        "\nStop here. Do not train ResNet "
        "until Policy C is frozen."
    )


if __name__ == "__main__":
    main()