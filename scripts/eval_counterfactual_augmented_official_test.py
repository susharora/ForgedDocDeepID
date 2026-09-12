#!/usr/bin/env python3
"""Post-hoc official-test evaluation of the counterfactual-augmented ResNet.

Uses exactly the already-frozen Policy-C official-test cache.

Compares:
    compression-controlled baseline
    vs
    counterfactual-augmented compression-controlled model

Primary comparisons:
    overall
    digital_3
    facedancer
    textdiffuserft_bfei

IMPORTANT:
The official test was already inspected earlier in B_viability and during
the compression-controlled experiment. This is therefore post-hoc
diagnostic evidence, not pristine untouched-test model selection.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision.models import resnet18
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT = (
    "POST_HOC_COUNTERFACTUAL_AUGMENTED_"
    "COMPRESSION_CONTROLLED_EXPERIMENT"
)

EVALUATION = (
    "POST_HOC_DIAGNOSTIC_OFFICIAL_TEST"
)

SEED = 10

CONTENT_H = 512
CANVAS_W = 864

BATCH_SIZE = 8

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

CHECKPOINT = (
    ROOT
    / "runs"
    / "post_hoc_counterfactual_augmented_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

CHECKPOINT_HASH = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_checkpoint.sha256"
)

BASELINE_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_predictions.csv"
)

BASELINE_METRICS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_metrics.csv"
)

OUT_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_official_test_predictions.csv"
)

OUT_METRICS = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_official_test_metrics.csv"
)

OUT_COMPARISON = (
    ROOT
    / "output"
    / "official_test_augmented_vs_controlled.csv"
)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


class TestDataset(Dataset):
    def __init__(self, frame):
        self.frame = (
            frame
            .reset_index(drop=True)
            .copy()
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]

        path = (
            ROOT
            / row.cache_path
        )

        with Image.open(path) as im:
            im = im.convert("RGB")

            width, height = im.size

            new_width = int(
                round(
                    width
                    * CONTENT_H
                    / height
                )
            )

            if new_width > CANVAS_W:
                raise RuntimeError(
                    "Image exceeds frozen r512 canvas: "
                    f"{row.image_path}"
                )

            im = TF.resize(
                im,
                [CONTENT_H, new_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            x = TF.to_tensor(im)

        x = TF.normalize(
            x,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        canvas = torch.zeros(
            (
                3,
                CONTENT_H,
                CANVAS_W,
            ),
            dtype=x.dtype,
        )

        x0 = (
            CANVAS_W
            - new_width
        ) // 2

        canvas[
            :,
            :,
            x0:x0 + new_width,
        ] = x

        return (
            canvas,
            int(row.label),
            i,
        )


def validate_checkpoint():
    if not CHECKPOINT_HASH.is_file():
        raise RuntimeError(
            "Augmented checkpoint hash missing"
        )

    expected_sha = (
        CHECKPOINT_HASH
        .read_text()
        .strip()
        .split()[0]
    )

    actual_sha = sha256_file(
        CHECKPOINT
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Augmented checkpoint SHA mismatch:\n"
            f"expected {expected_sha}\n"
            f"actual   {actual_sha}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
        )

    expected_metadata = {
        "experiment":
            EXPERIMENT,

        "seed":
            SEED,

        "architecture":
            "resnet18",

        "resolution":
            "r512",

        "stage":
            "full",

        "epoch":
            2,
    }

    for key, expected in (
        expected_metadata.items()
    ):
        actual = checkpoint.get(
            key
        )

        if actual != expected:
            raise RuntimeError(
                "Unexpected checkpoint metadata: "
                f"{key}={actual!r}, "
                f"expected {expected!r}"
            )

    if not np.isclose(
        float(
            checkpoint["dev_auc"]
        ),
        1.0,
    ):
        raise RuntimeError(
            "Expected selected dev AUROC 1.0"
        )

    if checkpoint.get(
        "policy"
    ) != (
        "Q75 -> matched deterministic Q50-90"
    ):
        raise RuntimeError(
            "Unexpected checkpoint compression policy"
        )

    augmentation = str(
        checkpoint.get(
            "augmentation",
            "",
        )
    )

    for term in [
        "face_restored",
        "text_restored",
        "both_restored",
    ]:
        if term not in augmentation:
            raise RuntimeError(
                "Checkpoint augmentation metadata "
                f"missing {term}"
            )

    return (
        actual_sha,
        checkpoint,
    )


def validate_test_index():
    df = pd.read_csv(
        TEST_INDEX
    )

    if len(df) != 1385:
        raise RuntimeError(
            f"Expected 1385 test rows, "
            f"got {len(df)}"
        )

    class_counts = (
        df.traffic_type
        .value_counts()
        .to_dict()
    )

    if class_counts != {
        "attack": 1085,
        "bonafide": 300,
    }:
        raise RuntimeError(
            f"Unexpected test class counts: "
            f"{class_counts}"
        )

    family_counts = (
        df.loc[
            df.label == 1,
            "variant",
        ]
        .value_counts()
        .to_dict()
    )

    expected_families = {
        "digital_3": 786,
        "facedancer": 150,
        "textdiffuserft_bfei": 149,
    }

    if (
        family_counts
        != expected_families
    ):
        raise RuntimeError(
            "Unexpected attack-family counts: "
            f"{family_counts}"
        )

    if not np.array_equal(
        df.label.to_numpy(),
        (
            df.traffic_type
            == "attack"
        )
        .astype(int)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Class polarity mismatch"
        )

    if df.image_path.duplicated().any():
        raise RuntimeError(
            "Duplicate official-test image path"
        )

    missing = [
        path
        for path in df.cache_path
        if not (
            ROOT
            / path
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} cached "
            "official-test images missing"
        )

    print(
        "Official-test hardware:",
        sorted(
            df.hardware_source
            .dropna()
            .unique()
        ),
    )

    return df


def load_model(
    checkpoint,
    device,
):
    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict(
    model,
    loader,
    device,
):
    labels = []
    scores = []
    indices = []

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=(
                torch.float16
                if device.type
                == "cuda"
                else torch.bfloat16
            ),
            enabled=(
                device.type
                == "cuda"
            ),
        ):
            logits = model(x)

        score = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .float()
            .cpu()
            .numpy()
        )

        labels.append(
            y.numpy()
        )

        scores.append(
            score
        )

        indices.append(
            idx.numpy()
        )

    return (
        np.concatenate(
            labels
        ),
        np.concatenate(
            scores
        ),
        np.concatenate(
            indices
        ),
    )


def binary_metrics(frame):
    y = (
        frame.label
        .to_numpy()
    )

    p = (
        frame.attack_probability
        .to_numpy()
    )

    pred = (
        p >= 0.5
    ).astype(int)

    attack = (
        y == 1
    )

    bona = (
        y == 0
    )

    return {
        "n":
            len(frame),

        "n_attack":
            int(
                attack.sum()
            ),

        "n_bonafide":
            int(
                bona.sum()
            ),

        "auroc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "attack_recall":
            float(
                pred[
                    attack
                ].mean()
            ),

        "bonafide_specificity":
            float(
                (
                    pred[
                        bona
                    ]
                    == 0
                ).mean()
            ),

        "mean_attack_probability":
            float(
                p[
                    attack
                ].mean()
            ),

        "median_attack_probability":
            float(
                np.median(
                    p[
                        attack
                    ]
                )
            ),

        "mean_bonafide_probability":
            float(
                p[
                    bona
                ].mean()
            ),

        "median_bonafide_probability":
            float(
                np.median(
                    p[
                        bona
                    ]
                )
            ),
    }


def metric_row(
    frame,
    group_type,
    group,
):
    return {
        "experiment":
            EXPERIMENT,

        "evaluation":
            EVALUATION,

        "seed":
            SEED,

        "group_type":
            group_type,

        "group":
            group,

        **binary_metrics(
            frame
        ),
    }


def build_metrics(df):
    rows = [
        metric_row(
            df,
            "overall",
            "all",
        )
    ]

    bonafide = (
        df.traffic_type
        == "bonafide"
    )

    for family in [
        "digital_3",
        "facedancer",
        "textdiffuserft_bfei",
    ]:
        subset = df[
            bonafide
            | (
                df.variant
                == family
            )
        ]

        rows.append(
            metric_row(
                subset,
                "attack_family_vs_bonafide",
                family,
            )
        )

    return pd.DataFrame(
        rows
    )


def validate_same_test_population(
    augmented,
):
    baseline = pd.read_csv(
        BASELINE_PREDICTIONS
    )

    if len(baseline) != 1385:
        raise RuntimeError(
            "Baseline official-test predictions "
            "do not contain 1385 rows"
        )

    baseline_paths = set(
        baseline.image_path
    )

    augmented_paths = set(
        augmented.image_path
    )

    if (
        baseline_paths
        != augmented_paths
    ):
        raise RuntimeError(
            "Baseline and augmented models "
            "were not evaluated on identical "
            "official-test image populations"
        )

    return baseline


def build_comparison(
    augmented_predictions,
    augmented_metrics,
):
    baseline_predictions = (
        validate_same_test_population(
            augmented_predictions
        )
    )

    baseline_metrics = pd.read_csv(
        BASELINE_METRICS
    )

    wanted = {
        ("overall", "all"),
        (
            "attack_family_vs_bonafide",
            "digital_3",
        ),
        (
            "attack_family_vs_bonafide",
            "facedancer",
        ),
        (
            "attack_family_vs_bonafide",
            "textdiffuserft_bfei",
        ),
    }

    baseline_metrics = (
        baseline_metrics[
            baseline_metrics.apply(
                lambda row:
                    (
                        row.group_type,
                        row.group,
                    )
                    in wanted,
                axis=1,
            )
        ]
        .copy()
    )

    baseline_metrics = (
        baseline_metrics[
            [
                "group_type",
                "group",
                "auroc",
                "balanced_accuracy",
                "attack_recall",
                "bonafide_specificity",
                "mean_attack_probability",
                "mean_bonafide_probability",
            ]
        ]
        .rename(
            columns={
                "auroc":
                    "baseline_auroc",

                "balanced_accuracy":
                    "baseline_balanced_accuracy",

                "attack_recall":
                    "baseline_attack_recall",

                "bonafide_specificity":
                    "baseline_bonafide_specificity",

                "mean_attack_probability":
                    "baseline_mean_attack_probability",

                "mean_bonafide_probability":
                    "baseline_mean_bonafide_probability",
            }
        )
    )

    augmented_metrics = (
        augmented_metrics[
            [
                "group_type",
                "group",
                "auroc",
                "balanced_accuracy",
                "attack_recall",
                "bonafide_specificity",
                "mean_attack_probability",
                "mean_bonafide_probability",
            ]
        ]
        .rename(
            columns={
                "auroc":
                    "augmented_auroc",

                "balanced_accuracy":
                    "augmented_balanced_accuracy",

                "attack_recall":
                    "augmented_attack_recall",

                "bonafide_specificity":
                    "augmented_bonafide_specificity",

                "mean_attack_probability":
                    "augmented_mean_attack_probability",

                "mean_bonafide_probability":
                    "augmented_mean_bonafide_probability",
            }
        )
    )

    comparison = baseline_metrics.merge(
        augmented_metrics,
        on=[
            "group_type",
            "group",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != 4:
        raise RuntimeError(
            f"Expected four comparison rows, "
            f"got {len(comparison)}"
        )

    comparison[
        "delta_auroc"
    ] = (
        comparison.augmented_auroc
        - comparison.baseline_auroc
    )

    comparison[
        "delta_balanced_accuracy"
    ] = (
        comparison.augmented_balanced_accuracy
        - comparison.baseline_balanced_accuracy
    )

    comparison[
        "delta_attack_recall"
    ] = (
        comparison.augmented_attack_recall
        - comparison.baseline_attack_recall
    )

    comparison[
        "delta_bonafide_specificity"
    ] = (
        comparison.augmented_bonafide_specificity
        - comparison.baseline_bonafide_specificity
    )

    comparison[
        "delta_mean_attack_probability"
    ] = (
        comparison.augmented_mean_attack_probability
        - comparison.baseline_mean_attack_probability
    )

    # --------------------------------------------------
    # Paired per-image attack-score changes.
    # --------------------------------------------------

    pair = (
        baseline_predictions[
            [
                "image_path",
                "traffic_type",
                "variant",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    "baseline_probability",
            }
        )
        .merge(
            augmented_predictions[
                [
                    "image_path",
                    "attack_probability",
                ]
            ].rename(
                columns={
                    "attack_probability":
                        "augmented_probability",
                }
            ),
            on="image_path",
            how="inner",
            validate="one_to_one",
        )
    )

    pair[
        "score_change"
    ] = (
        pair.augmented_probability
        - pair.baseline_probability
    )

    paired_rows = []

    for group_type, group in [
        (
            "overall",
            "all",
        ),
        (
            "attack_family_vs_bonafide",
            "digital_3",
        ),
        (
            "attack_family_vs_bonafide",
            "facedancer",
        ),
        (
            "attack_family_vs_bonafide",
            "textdiffuserft_bfei",
        ),
    ]:
        if group == "all":
            attacks = pair[
                pair.traffic_type
                == "attack"
            ]
        else:
            attacks = pair[
                pair.variant
                == group
            ]

        paired_rows.append(
            {
                "group_type":
                    group_type,

                "group":
                    group,

                "mean_paired_attack_score_change":
                    attacks.score_change.mean(),

                "median_paired_attack_score_change":
                    attacks.score_change.median(),

                "fraction_attack_score_increased":
                    (
                        attacks.score_change
                        > 0
                    ).mean(),
            }
        )

    paired = pd.DataFrame(
        paired_rows
    )

    comparison = comparison.merge(
        paired,
        on=[
            "group_type",
            "group",
        ],
        how="left",
        validate="one_to_one",
    )

    return comparison


def main():
    checkpoint_sha, checkpoint = (
        validate_checkpoint()
    )

    test = validate_test_index()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"experiment:  {EXPERIMENT}"
        f"\nevaluation:  {EVALUATION}"
        f"\nseed:        {SEED}"
        f"\ndevice:      {device}"
        f"\ntest images: {len(test)}"
        f"\ncheckpoint:  {checkpoint_sha}"
    )

    if device.type == "cuda":
        print(
            "gpu:         "
            + torch.cuda
            .get_device_name(0)
        )

    print(
        "\nFrozen augmented checkpoint:"
        f"\n  stage:   {checkpoint['stage']}"
        f"\n  epoch:   {checkpoint['epoch']}"
        f"\n  dev_auc: {checkpoint['dev_auc']:.4f}"
    )

    model = load_model(
        checkpoint,
        device,
    )

    dataset = TestDataset(
        test
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    (
        labels,
        scores,
        indices,
    ) = predict(
        model,
        loader,
        device,
    )

    if not np.array_equal(
        labels,
        test.label.to_numpy(),
    ):
        raise RuntimeError(
            "Prediction-label alignment failed"
        )

    test = test.copy()

    test[
        "attack_probability"
    ] = np.nan

    test.loc[
        indices,
        "attack_probability",
    ] = scores

    if (
        test.attack_probability
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Prediction alignment failed"
        )

    test.insert(
        0,
        "experiment",
        EXPERIMENT,
    )

    test.insert(
        1,
        "evaluation",
        EVALUATION,
    )

    test.insert(
        2,
        "seed",
        SEED,
    )

    test.insert(
        3,
        "checkpoint_sha256",
        checkpoint_sha,
    )

    metrics = build_metrics(
        test
    )

    comparison = build_comparison(
        test,
        metrics,
    )

    OUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    comparison.to_csv(
        OUT_COMPARISON,
        index=False,
    )

    metric_columns = [
        "group_type",
        "group",
        "n",
        "n_attack",
        "n_bonafide",
        "auroc",
        "balanced_accuracy",
        "attack_recall",
        "bonafide_specificity",
        "mean_attack_probability",
        "mean_bonafide_probability",
    ]

    print(
        "\nAUGMENTED MODEL — "
        "post-hoc official-test metrics:"
    )

    print(
        metrics[
            metric_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    comparison_columns = [
        "group",
        "baseline_auroc",
        "augmented_auroc",
        "delta_auroc",
        "baseline_attack_recall",
        "augmented_attack_recall",
        "delta_attack_recall",
        "baseline_mean_attack_probability",
        "augmented_mean_attack_probability",
        "delta_mean_attack_probability",
        "mean_paired_attack_score_change",
        "median_paired_attack_score_change",
        "fraction_attack_score_increased",
    ]

    print(
        "\nCONTROLLED BASELINE "
        "vs COUNTERFACTUAL-AUGMENTED:"
    )

    print(
        comparison[
            comparison_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\nPredictions: {OUT_PREDICTIONS}"
        f"\nMetrics:     {OUT_METRICS}"
        f"\nComparison:  {OUT_COMPARISON}"
    )

    print(
        "\nInterpretation warning:"
        "\nThe official test was already inspected "
        "before this experiment."
        "\nThese results are post-hoc diagnostic "
        "evidence, not untouched model-selection evidence."
    )


if __name__ == "__main__":
    main()