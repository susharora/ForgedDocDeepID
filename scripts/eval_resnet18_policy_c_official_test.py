#!/usr/bin/env python3
"""Diagnostic official-test evaluation of the frozen Policy-C ResNet-18."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torchvision.models import (
    resnet18,
)
from torchvision.transforms import (
    functional as TF,
)
from torchvision.transforms.functional import (
    InterpolationMode,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT = (
    "POST_HOC_COMPRESSION_CONTROLLED_EXPERIMENT"
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

INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

CHECKPOINT = (
    ROOT
    / "runs"
    / "post_hoc_compression_controlled_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

CHECKPOINT_HASH = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_checkpoint.sha256"
)

PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_predictions.csv"
)

METRICS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_metrics.csv"
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

        path = ROOT / row.cache_path

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
                    "Image exceeds r512 canvas: "
                    f"{row.image_path} -> "
                    f"{new_width}px"
                )

            im = TF.resize(
                im,
                [CONTENT_H, new_width],
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            )

            x = TF.to_tensor(im)

        x = TF.normalize(
            x,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        # Same ImageNet-mean padding used in training.
        canvas = torch.zeros(
            (3, CONTENT_H, CANVAS_W),
            dtype=x.dtype,
        )

        x0 = (
            CANVAS_W - new_width
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


def validate_frozen_checkpoint():
    if not CHECKPOINT_HASH.is_file():
        raise RuntimeError(
            "Checkpoint freeze hash missing. "
            "Freeze the checkpoint before official-test evaluation."
        )

    expected = (
        CHECKPOINT_HASH
        .read_text()
        .strip()
        .split()[0]
    )

    actual = sha256_file(
        CHECKPOINT
    )

    if actual != expected:
        raise RuntimeError(
            "Frozen checkpoint hash mismatch:\n"
            f"expected {expected}\n"
            f"actual   {actual}"
        )

    return actual


def validate_test_index():
    df = pd.read_csv(
        INDEX
    )

    if len(df) != 1385:
        raise RuntimeError(
            f"Expected 1385 rows, got {len(df)}"
        )

    counts = (
        df.traffic_type
        .value_counts()
        .to_dict()
    )

    if counts != {
        "attack": 1085,
        "bonafide": 300,
    }:
        raise RuntimeError(
            f"Unexpected class counts: {counts}"
        )

    families = (
        df.loc[
            df.label == 1,
            "variant",
        ]
        .value_counts()
        .to_dict()
    )

    expected = {
        "digital_3": 786,
        "facedancer": 150,
        "textdiffuserft_bfei": 149,
    }

    if families != expected:
        raise RuntimeError(
            f"Unexpected attack composition: {families}"
        )


    hardware = sorted(
    df.hardware_source
    .dropna()
    .unique()
    )

    if not hardware:
        raise RuntimeError(
            "No hardware labels found"
    )

    print(
        "Official-test hardware:",
        hardware,
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

    missing = [
        p
        for p in df.cache_path
        if not (ROOT / p).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} Policy-C cached images missing"
        )

    return df


def load_frozen_model(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
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
            12,
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

    if abs(
        float(checkpoint["dev_auc"])
        - 1.0
    ) > 1e-12:
        raise RuntimeError(
            "Expected frozen dev AUROC 1.0"
        )

    if checkpoint.get(
        "class_mapping"
    ) != {
        "bonafide": 0,
        "attack": 1,
    }:
        raise RuntimeError(
            "Checkpoint class mapping mismatch"
        )

    if checkpoint.get(
        "policy"
    ) != (
        "Q75 -> matched deterministic Q50-90"
    ):
        raise RuntimeError(
            "Checkpoint Policy-C metadata mismatch"
        )

    if (
        float(
            checkpoint["backbone_lr"]
        )
        != 1e-4
        or float(
            checkpoint["classifier_lr"]
        )
        != 1e-3
    ):
        raise RuntimeError(
            "Checkpoint learning-rate metadata mismatch"
        )

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def predict(
    model,
    loader,
    device,
):
    labels = []
    scores = []
    indices = []

    use_amp = (
        device.type == "cuda"
    )

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp,
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
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(indices),
    )


def binary_metrics(frame):
    y = frame.label.to_numpy()

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

    bonafide = (
        y == 0
    )

    return {
        "n": len(frame),
        "n_attack":
            int(attack.sum()),
        "n_bonafide":
            int(bonafide.sum()),
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
                pred[attack].mean()
            ),
        "bonafide_specificity":
            float(
                (pred[bonafide] == 0)
                .mean()
            ),
        "mean_attack_probability":
            float(
                p[attack].mean()
            ),
        "mean_bonafide_probability":
            float(
                p[bonafide].mean()
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
        **binary_metrics(frame),
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

    for hardware in sorted(
        df.hardware_source.unique()
    ):
        subset = df[
            df.hardware_source
            == hardware
        ]

        rows.append(
            metric_row(
                subset,
                "hardware",
                hardware,
            )
        )

    return pd.DataFrame(
        rows
    )


def main():
    checkpoint_sha = (
        validate_frozen_checkpoint()
    )

    df = validate_test_index()

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
        f"\ntest images: {len(df)}"
        f"\ncheckpoint:  {checkpoint_sha}"
    )

    if device.type == "cuda":
        print(
            "gpu:         "
            + torch.cuda.get_device_name(0)
        )

    model, checkpoint = (
        load_frozen_model(
            device
        )
    )

    print(
        "\nFrozen checkpoint:"
        f"\n  stage:   {checkpoint['stage']}"
        f"\n  epoch:   {checkpoint['epoch']}"
        f"\n  dev_auc: {checkpoint['dev_auc']:.4f}"
    )

    dataset = TestDataset(
        df
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

    labels, scores, indices = (
        predict(
            model,
            loader,
            device,
        )
    )

    if not np.array_equal(
        labels,
        df.label.to_numpy(),
    ):
        raise RuntimeError(
            "Prediction label alignment failed"
        )

    df = df.copy()

    df["attack_probability"] = (
        np.nan
    )

    df.loc[
        indices,
        "attack_probability",
    ] = scores

    if (
        df.attack_probability
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Prediction alignment failed"
        )

    df.insert(
        0,
        "experiment",
        EXPERIMENT,
    )

    df.insert(
        1,
        "evaluation",
        EVALUATION,
    )

    df.insert(
        2,
        "seed",
        SEED,
    )

    df.insert(
        3,
        "checkpoint_sha256",
        checkpoint_sha,
    )

    metrics = build_metrics(
        df
    )

    df.to_csv(
        PREDICTIONS,
        index=False,
    )

    metrics.to_csv(
        METRICS,
        index=False,
    )

    columns = [
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
        "\nPOST-HOC diagnostic official-test metrics:"
    )

    print(
        metrics[columns]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\nPredictions: {PREDICTIONS}"
        f"\nMetrics:     {METRICS}"
        "\n\nInterpretation warning:"
        "\nThe official test had already been inspected "
        "in B_viability."
        "\nThese results are post-hoc diagnostic evidence, "
        "not untouched model-selection evidence."
    )


if __name__ == "__main__":
    main()