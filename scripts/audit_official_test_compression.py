#!/usr/bin/env python3
"""Compression-only audit of the official test after frozen Policy C."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compression_leakage import compression_features


ROOT = Path(__file__).resolve().parents[1]

TRAIN_FEATURES = (
    ROOT
    / "output"
    / "compression_leakage.csv"
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

OUT_FEATURES = (
    ROOT
    / "output"
    / "official_test_policy_c_compression_features.csv"
)

OUT_SUMMARY = (
    ROOT
    / "output"
    / "official_test_policy_c_compression_leakage.csv"
)


TABLE_FEATURES = [
    "jpeg_quality",
    "q_luma_mean",
    "q_chroma_mean",
]

STRICT_HISTORY = [
    *TABLE_FEATURES,
    "lattice_q75_mean",
    "lattice_q75_p90",
    "lattice_current_mean",
    "lattice_current_p90",
    "recompress_current_mae",
]

BLOCKINESS = [
    "blockiness_h",
    "blockiness_v",
]

FEATURE_SETS = {
    "final_table_only":
        TABLE_FEATURES,

    "strict_jpeg_history":
        STRICT_HISTORY,

    "blockiness_only":
        BLOCKINESS,

    "all_compression_features":
        STRICT_HISTORY
        + BLOCKINESS,
}


def load_train():
    df = pd.read_csv(
        TRAIN_FEATURES
    )

    df = df[
        (
            df.policy
            == "C_q75_then_q50_90"
        )
        & (
            df.split
            == "project_train"
        )
    ].copy()

    if len(df) != 1440:
        raise RuntimeError(
            f"Expected 1440 Policy-C "
            f"train rows, got {len(df)}"
        )

    return df


def extract_test_features():
    index = pd.read_csv(
        TEST_INDEX
    )

    if len(index) != 1385:
        raise RuntimeError(
            f"Expected 1385 test rows, "
            f"got {len(index)}"
        )

    rows = []

    for i, row in enumerate(
        index.itertuples(
            index=False
        ),
        start=1,
    ):
        path = (
            ROOT
            / row.cache_path
        )

        features = (
            compression_features(
                path.read_bytes()
            )
        )

        rows.append(
            {
                "image_path":
                    row.image_path,

                "file_stem":
                    row.file_stem,

                "traffic_type":
                    row.traffic_type,

                "variant":
                    (
                        ""
                        if pd.isna(
                            row.variant
                        )
                        else row.variant
                    ),

                "hardware_source":
                    row.hardware_source,

                "label":
                    int(row.label),

                "assigned_q":
                    int(row.assigned_q),

                **features,
            }
        )

        if (
            i % 100 == 0
            or i == len(index)
        ):
            print(
                f"processed "
                f"{i}/{len(index)}"
            )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT_FEATURES,
        index=False,
    )

    return df


def fit_and_score(
    train,
    test,
    features,
):
    model = make_pipeline(
        StandardScaler(),

        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=10,
        ),
    )

    model.fit(
        train[features],
        train.label,
    )

    return (
        model.predict_proba(
            test[features]
        )[:, 1]
    )


def family_metrics(
    frame,
    score,
):
    work = frame.copy()

    work["score"] = score

    rows = []

    bona = (
        work.traffic_type
        == "bonafide"
    )

    for family in [
        "digital_3",
        "facedancer",
        "textdiffuserft_bfei",
    ]:
        subset = work[
            bona
            | (
                work.variant
                == family
            )
        ]

        y = subset.label.to_numpy()
        p = subset.score.to_numpy()

        rows.append(
            {
                "family":
                    family,

                "n_attack":
                    int(
                        y.sum()
                    ),

                "n_bonafide":
                    int(
                        (y == 0).sum()
                    ),

                "auroc":
                    float(
                        roc_auc_score(
                            y,
                            p,
                        )
                    ),

                "mean_attack_score":
                    float(
                        p[
                            y == 1
                        ].mean()
                    ),

                "mean_bonafide_score":
                    float(
                        p[
                            y == 0
                        ].mean()
                    ),
            }
        )

    return rows


def main():
    train = load_train()

    test = (
        extract_test_features()
    )

    rows = []

    print(
        "\nOfficial-test compression "
        "leakage after Policy C:"
    )

    for name, features in (
        FEATURE_SETS.items()
    ):
        score = fit_and_score(
            train,
            test,
            features,
        )

        family_rows = (
            family_metrics(
                test,
                score,
            )
        )

        for result in family_rows:
            rows.append(
                {
                    "feature_set":
                        name,

                    **result,
                }
            )

        print(
            f"\n{name}"
        )

        print(
            pd.DataFrame(
                family_rows
            ).to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print(
        f"\nFeatures: {OUT_FEATURES}"
        f"\nSummary:  {OUT_SUMMARY}"
    )

    print(
        "\nStop interpretation:"
        "\nIf strict_jpeg_history remains "
        "near chance for all families, "
        "Policy C is suppressing the known "
        "JPEG history reasonably well even "
        "under test-family shift."
    )


if __name__ == "__main__":
    main()