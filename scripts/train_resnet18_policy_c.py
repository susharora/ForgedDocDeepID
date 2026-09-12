#!/usr/bin/env python3
"""One-seed compression-controlled ResNet-18 viability experiment."""

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.transforms import functional as TF
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

SEED = 10

CONTENT_H = 512
CANVAS_W = 864

BATCH_SIZE = 8
NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)

HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 12

BACKBONE_LR = 1e-4
CLASSIFIER_LR = 1e-3

# Explicit PyTorch AdamW default.
WEIGHT_DECAY = 1e-2

INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

TRAIN_MANIFEST = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_project_train.csv"
)

DEV_MANIFEST = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_dev_val.csv"
)

EXPECTED_MANIFEST_SHA = {
    TRAIN_MANIFEST:
        "6307b1516f0e8a6db661077fedad704bb"
        "d13f6242c9795ca11310ac86d650cc8",

    DEV_MANIFEST:
        "46953b0e474fb52be7a94c2555d0453a"
        "57123d8412bd2e21780986155a9e250d",
}

RUN = (
    ROOT
    / "runs"
    / "post_hoc_compression_controlled_resnet18_seed10"
)

CHECKPOINT = (
    RUN
    / "checkpoints"
    / "best.pt"
)

PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_predictions.csv"
)

METRICS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_metrics.csv"
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is more important here
    # than squeezing out maximum throughput.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FantasyPolicyCDataset(Dataset):
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
                    f"Image exceeds r512 canvas: "
                    f"{row.image_path} -> "
                    f"{new_width} px"
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

        # Zero after ImageNet normalisation is equivalent
        # to padding with ImageNet mean RGB.
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


def validate_inputs():
    for path, expected in (
        EXPECTED_MANIFEST_SHA.items()
    ):
        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError(
                f"Frozen manifest changed: "
                f"{path}"
            )

    if not INDEX.exists():
        raise RuntimeError(
            "Policy-C cache index missing. "
            "Run prepare_policy_c_cache.py first."
        )

    df = pd.read_csv(INDEX)

    train = df[
        df.split == "project_train"
    ].copy()

    dev = df[
        df.split == "dev_val"
    ].copy()

    if (
        len(train) != 1440
        or len(dev) != 459
    ):
        raise RuntimeError(
            "Unexpected train/dev counts"
        )

    if (
        train.file_stem.nunique() != 160
        or dev.file_stem.nunique() != 51
    ):
        raise RuntimeError(
            "Unexpected train/dev identity counts"
        )

    if (
        set(train.file_stem)
        & set(dev.file_stem)
    ):
        raise RuntimeError(
            "Train/dev identity leakage"
        )

    if set(df.label.unique()) != {0, 1}:
        raise RuntimeError(
            "Unexpected class labels"
        )

    expected_label = (
        df.traffic_type == "attack"
    ).astype(int)

    if not np.array_equal(
        df.label.to_numpy(),
        expected_label.to_numpy(),
    ):
        raise RuntimeError(
            "Class polarity mismatch"
        )

    if not df.assigned_q.between(
        50,
        90,
    ).all():
        raise RuntimeError(
            "Policy-C final Q outside Q50-90"
        )

    q_counts = (
        df.groupby(
            [
                "file_stem",
                "hardware_source",
            ]
        )
        .assigned_q
        .nunique()
    )

    if q_counts.max() != 1:
        raise RuntimeError(
            "Matched final-Q invariant failed"
        )

    missing = [
        p
        for p in df.cache_path
        if not (ROOT / p).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} cached images missing"
        )

    return (
        train.reset_index(drop=True),
        dev.reset_index(drop=True),
    )


def make_loaders(train, dev, device):
    train_ds = FantasyPolicyCDataset(train)
    dev_ds = FantasyPolicyCDataset(dev)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **common,
    )

    train_eval_loader = DataLoader(
        train_ds,
        shuffle=False,
        **common,
    )

    dev_loader = DataLoader(
        dev_ds,
        shuffle=False,
        **common,
    )

    return (
        train_loader,
        train_eval_loader,
        dev_loader,
    )


def class_weights(train):
    counts = (
        train.label
        .value_counts()
        .sort_index()
    )

    if (
        counts.to_dict()
        != {0: 480, 1: 960}
    ):
        raise RuntimeError(
            f"Unexpected train class counts: "
            f"{counts.to_dict()}"
        )

    n = len(train)
    k = 2

    # Standard inverse-frequency balanced weighting:
    # bonafide = 1.5
    # attack   = 0.75
    weights = np.array(
        [
            n / (k * counts[0]),
            n / (k * counts[1]),
        ],
        dtype=np.float32,
    )

    return torch.tensor(weights)


def build_model():
    model = resnet18(
        weights=(
            ResNet18_Weights.IMAGENET1K_V1
        )
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2,
    )

    return model


def set_head_only(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def set_full_finetune(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
):
    model.train()

    total_loss = 0.0
    total_n = 0

    use_amp = (
        device.type == "cuda"
    )

    for x, y, _ in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp,
        ):
            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        batch_n = len(y)

        total_loss += (
            float(loss.item())
            * batch_n
        )

        total_n += batch_n

    return total_loss / total_n


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    use_amp = (
        device.type == "cuda"
    )

    labels = []
    scores = []
    indices = []

    total_loss = 0.0
    total_n = 0

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        y_device = y.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp,
        ):
            logits = model(x)

            loss = criterion(
                logits,
                y_device,
            )

        probability = (
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
            probability
        )

        indices.append(
            idx.numpy()
        )

        total_loss += (
            float(loss.item())
            * len(y)
        )

        total_n += len(y)

    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    indices = np.concatenate(indices)

    auc = roc_auc_score(
        labels,
        scores,
    )

    return {
        "loss": total_loss / total_n,
        "auc": float(auc),
        "labels": labels,
        "scores": scores,
        "indices": indices,
    }


def save_if_best(
    model,
    stage,
    epoch,
    dev_auc,
    best_auc,
    class_weight_values,
):
    if dev_auc <= best_auc:
        return best_auc, False

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "experiment": EXPERIMENT,
            "seed": SEED,
            "architecture": "resnet18",
            "weights":
                "IMAGENET1K_V1",
            "resolution": "r512",
            "canvas": [512, 864],
            "policy":
                "Q75 -> matched deterministic Q50-90",
            "class_mapping": {
                "bonafide": 0,
                "attack": 1,
            },
            "class_weights":
                class_weight_values,
            "backbone_lr": BACKBONE_LR,
            "classifier_lr":
                CLASSIFIER_LR,
            "weight_decay":
                WEIGHT_DECAY,
            "stage": stage,
            "epoch": epoch,
            "dev_auc": dev_auc,
            "model_state":
                model.state_dict(),
        },
        CHECKPOINT,
    )

    return dev_auc, True


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
        "n_attack": int(
            attack.sum()
        ),
        "n_bonafide": int(
            bonafide.sum()
        ),
        "auroc": float(
            roc_auc_score(y, p)
        ),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                pred,
            )
        ),
        "attack_recall": float(
            pred[attack].mean()
        ),
        "bonafide_specificity": float(
            (pred[bonafide] == 0).mean()
        ),
        "mean_attack_probability": float(
            p[attack].mean()
        ),
        "mean_bonafide_probability": float(
            p[bonafide].mean()
        ),
    }


def prediction_frame(
    frame,
    result,
):
    out = frame.copy()

    out[
        "attack_probability"
    ] = np.nan

    out.loc[
        result["indices"],
        "attack_probability",
    ] = result["scores"]

    if out.attack_probability.isna().any():
        raise RuntimeError(
            "Prediction alignment failed"
        )

    return out


def metric_row(
    frame,
    split,
    group_type,
    group,
):
    m = binary_metrics(frame)

    return {
        "experiment": EXPERIMENT,
        "seed": SEED,
        "split": split,
        "group_type": group_type,
        "group": group,
        **m,
    }


def summarise(
    train_pred,
    dev_pred,
):
    rows = []

    rows.append(
        metric_row(
            train_pred,
            "project_train",
            "overall",
            "all",
        )
    )

    rows.append(
        metric_row(
            dev_pred,
            "dev_val",
            "overall",
            "all",
        )
    )

    # Each attack family is compared against
    # all dev bonafides so AUROC is defined.
    bonafide = (
        dev_pred.traffic_type
        == "bonafide"
    )

    families = sorted(
        dev_pred.loc[
            dev_pred.label == 1,
            "variant",
        ]
        .dropna()
        .unique()
    )

    for family in families:
        subset = dev_pred[
            bonafide
            | (
                dev_pred.variant
                == family
            )
        ]

        rows.append(
            metric_row(
                subset,
                "dev_val",
                "attack_family_vs_bonafide",
                family,
            )
        )

    for hardware in sorted(
        dev_pred.hardware_source.unique()
    ):
        subset = dev_pred[
            dev_pred.hardware_source
            == hardware
        ]

        rows.append(
            metric_row(
                subset,
                "dev_val",
                "hardware",
                hardware,
            )
        )

    return pd.DataFrame(rows)


def print_metrics(metrics):
    print(
        "\nFrozen best-checkpoint metrics:"
    )

    columns = [
        "split",
        "group_type",
        "group",
        "n",
        "auroc",
        "balanced_accuracy",
        "attack_recall",
        "bonafide_specificity",
    ]

    print(
        metrics[columns]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


def main():
    seed_everything(SEED)

    train, dev = validate_inputs()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"experiment: {EXPERIMENT}"
        f"\nseed:       {SEED}"
        f"\ndevice:     {device}"
        f"\ntrain:      {len(train)}"
        f"\ndev:        {len(dev)}"
    )

    if device.type == "cuda":
        print(
            "gpu:        "
            + torch.cuda.get_device_name(0)
        )
    else:
        print(
            "WARNING: CUDA unavailable; "
            "r512 training will be slow."
        )

    weights = class_weights(train)

    print(
        "class weights "
        f"[bonafide, attack]: "
        f"{weights.tolist()}"
    )

    (
        train_loader,
        train_eval_loader,
        dev_loader,
    ) = make_loaders(
        train,
        dev,
        device,
    )

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    use_amp = (
        device.type == "cuda"
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    best_auc = -np.inf

    # --------------------------------------------------
    # Stage 1: classifier head only
    # --------------------------------------------------
    set_head_only(model)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=CLASSIFIER_LR,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "\nStage 1: classifier head only"
    )

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )

        dev_result = evaluate(
            model,
            dev_loader,
            criterion,
            device,
        )

        best_auc, saved = save_if_best(
            model,
            "head",
            epoch,
            dev_result["auc"],
            best_auc,
            weights.tolist(),
        )

        print(
            f"head {epoch:02d}/{HEAD_EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_result['loss']:.4f} "
            f"dev_auc={dev_result['auc']:.4f} "
            f"{'*' if saved else ''}"
        )

    # --------------------------------------------------
    # Stage 2: full fine-tuning
    # --------------------------------------------------
    set_full_finetune(model)

    backbone_parameters = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("fc.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_parameters,
                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    model.fc.parameters(),
                "lr":
                    CLASSIFIER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "\nStage 2: full fine-tuning"
    )

    for epoch in range(
        1,
        FINETUNE_EPOCHS + 1,
    ):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )

        dev_result = evaluate(
            model,
            dev_loader,
            criterion,
            device,
        )

        best_auc, saved = save_if_best(
            model,
            "full",
            epoch,
            dev_result["auc"],
            best_auc,
            weights.tolist(),
        )

        print(
            f"full {epoch:02d}/{FINETUNE_EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_result['loss']:.4f} "
            f"dev_auc={dev_result['auc']:.4f} "
            f"{'*' if saved else ''}"
        )

    # --------------------------------------------------
    # Freeze the best dev checkpoint and evaluate it.
    # --------------------------------------------------
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print(
        "\nBest checkpoint:"
        f"\n  stage:   {checkpoint['stage']}"
        f"\n  epoch:   {checkpoint['epoch']}"
        f"\n  dev_auc: {checkpoint['dev_auc']:.4f}"
    )

    train_result = evaluate(
        model,
        train_eval_loader,
        criterion,
        device,
    )

    dev_result = evaluate(
        model,
        dev_loader,
        criterion,
        device,
    )

    train_pred = prediction_frame(
        train,
        train_result,
    )

    dev_pred = prediction_frame(
        dev,
        dev_result,
    )

    predictions = pd.concat(
        [
            train_pred,
            dev_pred,
        ],
        ignore_index=True,
    )

    predictions.insert(
        0,
        "experiment",
        EXPERIMENT,
    )

    predictions.insert(
        1,
        "seed",
        SEED,
    )

    metrics = summarise(
        train_pred,
        dev_pred,
    )

    PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS,
        index=False,
    )

    metrics.to_csv(
        METRICS,
        index=False,
    )

    print_metrics(metrics)

    print(
        f"\ncheckpoint:  {CHECKPOINT}"
        f"\npredictions: {PREDICTIONS}"
        f"\nmetrics:     {METRICS}"
        "\n\nStop here. Do not inspect the "
        "official test set yet."
    )


if __name__ == "__main__":
    main()