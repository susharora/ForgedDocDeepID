#!/usr/bin/env python3
"""Official-test JPEG nuisance marginalisation.

For each raw official-test image:

    raw
    -> common Q75
    -> five independent deterministic Q50-90 draws
    -> score both frozen ResNet models

Reports:
    metrics for each JPEG draw
    metrics after averaging the five probabilities

This is an additional post-hoc test-time nuisance experiment.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)

from eval_counterfactual_restoration import (
    ROOT,
    DATA,
    encode_quality,
    jpeg_decode_rgb,
    tensor_from_policy_bytes,
)


TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

CONTROLLED_CHECKPOINT = (
    ROOT
    / "runs"
    / "post_hoc_compression_controlled_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

CONTROLLED_HASH = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_checkpoint.sha256"
)

AUGMENTED_CHECKPOINT = (
    ROOT
    / "runs"
    / "post_hoc_counterfactual_augmented_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

AUGMENTED_HASH = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_checkpoint.sha256"
)

OUT_PREDICTIONS = (
    ROOT
    / "output"
    / "official_test_multijpeg_predictions.csv"
)

OUT_METRICS = (
    ROOT
    / "output"
    / "official_test_multijpeg_metrics.csv"
)

N_DRAWS = 5

# Fixed experiment-level draw seeds.
DRAW_SEEDS = [
    101,
    211,
    307,
    401,
    503,
]


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def final_q_draw(
    draw_seed,
    file_stem,
    hardware,
):
    """
    Same stem+hardware gets same Q
    within each draw, independent of label/family.
    """

    key = (
        f"{draw_seed}|"
        f"{file_stem}|"
        f"{hardware}"
    ).encode()

    n = int.from_bytes(
        hashlib.sha256(
            key
        ).digest()[:8],
        "big",
    )

    return (
        50
        + n % 41
    )


def load_checkpoint(
    path,
    hash_file,
    device,
):
    expected = (
        hash_file
        .read_text()
        .strip()
        .split()[0]
    )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: "
            f"{path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

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
def score(
    model,
    batch,
    device,
):
    batch = batch.to(
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
        logits = model(
            batch
        )

    return (
        torch.softmax(
            logits,
            dim=1,
        )[:, 1]
        .float()
        .cpu()
        .numpy()
    )


def metric_row(
    frame,
    probability_column,
    model_name,
    draw_name,
    group,
):
    if group == "all":
        subset = frame

    else:
        subset = frame[
            (
                frame.traffic_type
                == "bonafide"
            )
            | (
                frame.variant
                == group
            )
        ]

    y = subset.label.to_numpy()

    p = (
        subset[
            probability_column
        ]
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
        "model":
            model_name,

        "draw":
            draw_name,

        "group":
            group,

        "n":
            len(subset),

        "auroc":
            float(
                roc_auc_score(
                    y,
                    p,
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

        "mean_bonafide_probability":
            float(
                p[
                    bona
                ].mean()
            ),
    }


def main():
    if len(
        DRAW_SEEDS
    ) != N_DRAWS:
        raise RuntimeError(
            "DRAW_SEEDS mismatch"
        )

    index = pd.read_csv(
        TEST_INDEX
    )

    if len(index) != 1385:
        raise RuntimeError(
            "Expected 1385 test rows"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    controlled = (
        load_checkpoint(
            CONTROLLED_CHECKPOINT,
            CONTROLLED_HASH,
            device,
        )
    )

    augmented = (
        load_checkpoint(
            AUGMENTED_CHECKPOINT,
            AUGMENTED_HASH,
            device,
        )
    )

    print(
        f"device: {device}"
        f"\ntest images: {len(index)}"
        f"\nJPEG draws: {N_DRAWS}"
        "\nrange: Q50-90"
    )

    rows = []

    for i, row in enumerate(
        index.itertuples(
            index=False
        ),
        start=1,
    ):
        raw = (
            DATA
            / row.image_path
        ).read_bytes()

        rgb = jpeg_decode_rgb(
            raw
        )

        # Common Q75 only once.
        q75 = encode_quality(
            rgb,
            75,
        )

        q75_rgb = (
            jpeg_decode_rgb(
                q75
            )
        )

        tensors = []
        qualities = []

        for draw_seed in (
            DRAW_SEEDS
        ):
            q = final_q_draw(
                draw_seed,
                row.file_stem,
                row.hardware_source,
            )

            final = encode_quality(
                q75_rgb,
                q,
            )

            tensors.append(
                tensor_from_policy_bytes(
                    final
                )
            )

            qualities.append(
                q
            )

        batch = torch.stack(
            tensors
        )

        controlled_scores = (
            score(
                controlled,
                batch,
                device,
            )
        )

        augmented_scores = (
            score(
                augmented,
                batch,
                device,
            )
        )

        for draw_idx, (
            q,
            c_score,
            a_score,
        ) in enumerate(
            zip(
                qualities,
                controlled_scores,
                augmented_scores,
            )
        ):
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

                    "draw":
                        draw_idx,

                    "draw_seed":
                        DRAW_SEEDS[
                            draw_idx
                        ],

                    "final_q":
                        q,

                    "controlled_probability":
                        float(
                            c_score
                        ),

                    "augmented_probability":
                        float(
                            a_score
                        ),
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

    predictions = pd.DataFrame(
        rows
    )

    predictions.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    # --------------------------------------------------
    # Per-draw metrics
    # --------------------------------------------------

    metric_rows = []

    groups = [
        "all",
        "digital_3",
        "facedancer",
        "textdiffuserft_bfei",
    ]

    for draw in range(
        N_DRAWS
    ):
        subset = predictions[
            predictions.draw
            == draw
        ].copy()

        for model_name, column in [
            (
                "controlled",
                "controlled_probability",
            ),
            (
                "augmented",
                "augmented_probability",
            ),
        ]:
            for group in groups:
                metric_rows.append(
                    metric_row(
                        subset,
                        column,
                        model_name,
                        f"draw_{draw}",
                        group,
                    )
                )

    # --------------------------------------------------
    # TTA: average probability across JPEG draws
    # --------------------------------------------------

    identity_columns = [
        "image_path",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
    ]

    ensemble = (
        predictions.groupby(
            identity_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            controlled_probability=(
                "controlled_probability",
                "mean",
            ),

            augmented_probability=(
                "augmented_probability",
                "mean",
            ),

            q_min=(
                "final_q",
                "min",
            ),

            q_max=(
                "final_q",
                "max",
            ),

            q_mean=(
                "final_q",
                "mean",
            ),
        )
    )

    if len(ensemble) != 1385:
        raise RuntimeError(
            "JPEG-draw averaging "
            "did not recover 1385 images"
        )

    for model_name, column in [
        (
            "controlled",
            "controlled_probability",
        ),
        (
            "augmented",
            "augmented_probability",
        ),
    ]:
        for group in groups:
            metric_rows.append(
                metric_row(
                    ensemble,
                    column,
                    model_name,
                    "mean_5_draws",
                    group,
                )
            )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    # Compact printout: ensemble first.
    ensemble_metrics = metrics[
        metrics.draw
        == "mean_5_draws"
    ]

    print(
        "\nFIVE-DRAW JPEG-MARGINALISED "
        "OFFICIAL TEST:"
    )

    print(
        ensemble_metrics[
            [
                "model",
                "group",
                "auroc",
                "balanced_accuracy",
                "attack_recall",
                "bonafide_specificity",
                "mean_attack_probability",
                "mean_bonafide_probability",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nPer-draw AUROC range:"
    )

    per_draw = metrics[
        metrics.draw
        != "mean_5_draws"
    ]

    ranges = (
        per_draw.groupby(
            [
                "model",
                "group",
            ]
        )
        .auroc
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )

    print(
        ranges.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print(
        f"\nPredictions: "
        f"{OUT_PREDICTIONS}"
        f"\nMetrics:     "
        f"{OUT_METRICS}"
    )

    print(
        "\nInterpretation:"
        "\nIf digital_3 remains catastrophically "
        "low across all five Q draws, the failure "
        "is not an accident of the single final "
        "JPEG-quality assignment."
    )


if __name__ == "__main__":
    main()