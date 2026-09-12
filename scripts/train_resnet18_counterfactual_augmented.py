#!/usr/bin/env python3
"""One-seed counterfactual-augmented Policy-C ResNet experiment."""

import hashlib
import os
import random
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
    ResNet18_Weights,
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
    "POST_HOC_COUNTERFACTUAL_AUGMENTED_"
    "COMPRESSION_CONTROLLED_EXPERIMENT"
)

SEED = 10

CONTENT_H = 512
CANVAS_W = 864

BATCH_SIZE = 8

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)

# Augmented dataset is exactly 3x the size
# of the original project-train set.
#
# These epoch counts preserve EXACTLY the
# baseline optimizer-step budget:
#
# baseline:
#   head: 3 * 1440 / 8 = 540 steps
#   full: 12 * 1440 / 8 = 2160 steps
#
# augmented:
#   head: 1 * 4320 / 8 = 540 steps
#   full: 4 * 4320 / 8 = 2160 steps
HEAD_EPOCHS = 1
FINETUNE_EPOCHS = 4

BACKBONE_LR = 1e-4
CLASSIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-2

TRAIN_INDEX = (
    ROOT
    / "output/"
    "counterfactual_augmented_train_index.csv"
)

POLICY_INDEX = (
    ROOT
    / "output/"
    "policy_c_cache_index.csv"
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

EXPECTED_SHA = {
    TRAIN_MANIFEST:
        "6307b1516f0e8a6db661077fedad704bb"
        "d13f6242c9795ca11310ac86d650cc8",

    DEV_MANIFEST:
        "46953b0e474fb52be7a94c2555d0453a"
        "57123d8412bd2e21780986155a9e250d",
}

RUN = (
    ROOT
    / "runs/"
    "post_hoc_counterfactual_augmented_resnet18_seed10"
)

CHECKPOINT = (
    RUN
    / "checkpoints/best.pt"
)

PREDICTIONS = (
    ROOT
    / "output/"
    "resnet18_counterfactual_augmented_seed10_predictions.csv"
)

METRICS = (
    ROOT
    / "output/"
    "resnet18_counterfactual_augmented_seed10_metrics.csv"
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


def sha_file(path):
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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Dataset512(Dataset):
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

            w, h = im.size

            new_w = int(
                round(
                    w
                    * CONTENT_H
                    / h
                )
            )

            if new_w > CANVAS_W:
                raise RuntimeError(
                    "Image exceeds r512 "
                    f"canvas: {path}"
                )

            im = TF.resize(
                im,
                [CONTENT_H, new_w],
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

        canvas = torch.zeros(
            (
                3,
                CONTENT_H,
                CANVAS_W,
            ),
            dtype=x.dtype,
        )

        x0 = (
            CANVAS_W - new_w
        ) // 2

        canvas[
            :,
            :,
            x0:x0 + new_w,
        ] = x

        return (
            canvas,
            int(row.label),
            i,
        )


def validate_inputs():
    for path, expected in (
        EXPECTED_SHA.items()
    ):
        if sha_file(path) != expected:
            raise RuntimeError(
                f"Frozen manifest changed: "
                f"{path}"
            )

    train = pd.read_csv(
        TRAIN_INDEX
    )

    policy = pd.read_csv(
        POLICY_INDEX
    )

    dev = policy[
        policy.split
        == "dev_val"
    ].copy()

    if len(train) != 4320:
        raise RuntimeError(
            "Expected 4320 augmented "
            "train rows"
        )

    if len(dev) != 459:
        raise RuntimeError(
            "Expected 459 untouched "
            "natural dev rows"
        )

    counts = (
        train.label
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if counts != {
        0: 1440,
        1: 2880,
    }:
        raise RuntimeError(
            f"Unexpected train counts: "
            f"{counts}"
        )

    source_counts = (
        train.source_type
        .value_counts()
        .to_dict()
    )

    expected_sources = {
        "natural_bonafide": 480,
        "original_attack": 960,
        "face_restored": 960,
        "text_restored": 960,
        "both_restored": 960,
    }

    if (
        source_counts
        != expected_sources
    ):
        raise RuntimeError(
            "Unexpected augmented "
            f"sources: {source_counts}"
        )

    if (
        train.file_stem.nunique()
        != 160
    ):
        raise RuntimeError(
            "Expected 160 train stems"
        )

    if (
        dev.file_stem.nunique()
        != 51
    ):
        raise RuntimeError(
            "Expected 51 dev stems"
        )

    if (
        set(train.file_stem)
        & set(dev.file_stem)
    ):
        raise RuntimeError(
            "Train/dev identity leakage"
        )

    # Critical: dev is natural Policy-C only.
    if "source_type" in dev.columns:
        raise RuntimeError(
            "Unexpected augmented dev data"
        )

    missing = [
        p
        for p in train.cache_path
        if not (
            ROOT / p
        ).is_file()
    ]

    missing += [
        p
        for p in dev.cache_path
        if not (
            ROOT / p
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} cached "
            "images missing"
        )

    return (
        train.reset_index(
            drop=True
        ),
        dev.reset_index(
            drop=True
        ),
    )


def loaders(
    train,
    dev,
    device,
):
    train_ds = Dataset512(
        train
    )

    dev_ds = Dataset512(
        dev
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            == "cuda"
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

    train_eval = DataLoader(
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
        train_eval,
        dev_loader,
    )


def class_weights(train):
    counts = (
        train.label
        .value_counts()
        .sort_index()
    )

    n = len(train)

    weights = np.array(
        [
            n
            / (
                2 * counts[0]
            ),

            n
            / (
                2 * counts[1]
            ),
        ],
        dtype=np.float32,
    )

    # Because augmentation preserves 2:1,
    # these must remain identical to baseline.
    if not np.allclose(
        weights,
        [1.5, 0.75],
    ):
        raise RuntimeError(
            f"Unexpected class weights: "
            f"{weights}"
        )

    return torch.tensor(
        weights
    )


def build_model():
    model = resnet18(
        weights=(
            ResNet18_Weights
            .IMAGENET1K_V1
        )
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    return model


def set_head_only(model):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True


def set_full(model):
    for p in model.parameters():
        p.requires_grad = True


def amp_context(device):
    return torch.autocast(
        device_type=device.type,
        dtype=(
            torch.float16
            if device.type
            == "cuda"
            else torch.bfloat16
        ),
        enabled=(
            device.type == "cuda"
        ),
    )


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
            set_to_none=True
        )

        with amp_context(
            device
        ):
            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            float(loss.item())
            * len(y)
        )

        total_n += len(y)

    return (
        total_loss
        / total_n
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    labels = []
    scores = []
    indices = []

    loss_sum = 0.0
    n = 0

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        yd = y.to(
            device,
            non_blocking=True,
        )

        with amp_context(
            device
        ):
            logits = model(x)

            loss = criterion(
                logits,
                yd,
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

        loss_sum += (
            float(loss.item())
            * len(y)
        )

        n += len(y)

    labels = np.concatenate(
        labels
    )

    scores = np.concatenate(
        scores
    )

    indices = np.concatenate(
        indices
    )

    return {
        "loss":
            loss_sum / n,

        "auc":
            float(
                roc_auc_score(
                    labels,
                    scores,
                )
            ),

        "labels":
            labels,

        "scores":
            scores,

        "indices":
            indices,
    }


def save_best(
    model,
    stage,
    epoch,
    dev_auc,
    best_auc,
    weights,
):
    if dev_auc <= best_auc:
        return (
            best_auc,
            False,
        )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "experiment":
                EXPERIMENT,

            "seed":
                SEED,

            "architecture":
                "resnet18",

            "weights":
                "IMAGENET1K_V1",

            "resolution":
                "r512",

            "canvas":
                [512, 864],

            "policy":
                "Q75 -> matched "
                "deterministic Q50-90",

            "augmentation":
                (
                    "original + "
                    "face_restored(label=1) + "
                    "text_restored(label=1) + "
                    "both_restored(label=0)"
                ),

            "optimization_budget":
                (
                    "matched baseline steps: "
                    "540 head + 2160 full"
                ),

            "class_mapping": {
                "bonafide": 0,
                "attack": 1,
            },

            "class_weights":
                weights.tolist(),

            "backbone_lr":
                BACKBONE_LR,

            "classifier_lr":
                CLASSIFIER_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "stage":
                stage,

            "epoch":
                epoch,

            "dev_auc":
                dev_auc,

            "model_state":
                model.state_dict(),
        },
        CHECKPOINT,
    )

    return (
        dev_auc,
        True,
    )


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

    if (
        out.attack_probability
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Prediction alignment failed"
        )

    return out


def metrics(frame):
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

        "mean_bonafide_probability":
            float(
                p[
                    bona
                ].mean()
            ),
    }


def dev_summary(dev):
    rows = []

    rows.append(
        {
            "group_type":
                "overall",

            "group":
                "all",

            **metrics(dev),
        }
    )

    bona = (
        dev.traffic_type
        == "bonafide"
    )

    for family in [
        "digital_1",
        "digital_2",
    ]:
        subset = dev[
            bona
            | (
                dev.variant
                == family
            )
        ]

        rows.append(
            {
                "group_type":
                    "attack_family_vs_bonafide",

                "group":
                    family,

                **metrics(
                    subset
                ),
            }
        )

    for hardware in sorted(
        dev.hardware_source.unique()
    ):
        subset = dev[
            dev.hardware_source
            == hardware
        ]

        rows.append(
            {
                "group_type":
                    "hardware",

                "group":
                    hardware,

                **metrics(
                    subset
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    seed_everything(
        SEED
    )

    train, dev = (
        validate_inputs()
    )

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
        f"\nhead steps: "
        f"{HEAD_EPOCHS * len(train) // BATCH_SIZE}"
        f"\nfull steps: "
        f"{FINETUNE_EPOCHS * len(train) // BATCH_SIZE}"
    )

    if device.type == "cuda":
        print(
            "gpu:        "
            + torch.cuda
            .get_device_name(0)
        )

    weights = class_weights(
        train
    )

    print(
        "class weights "
        f"[bonafide, attack]: "
        f"{weights.tolist()}"
    )

    (
        train_loader,
        train_eval_loader,
        dev_loader,
    ) = loaders(
        train,
        dev,
        device,
    )

    model = build_model().to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=weights.to(
                device
            )
        )
    )

    scaler = (
        torch.cuda.amp
        .GradScaler(
            enabled=(
                device.type
                == "cuda"
            )
        )
    )

    best_auc = -np.inf

    # --------------------------------------------------
    # Stage 1: head-only.
    # Same 540 optimizer steps as baseline.
    # --------------------------------------------------

    set_head_only(
        model
    )

    optimizer = (
        torch.optim.AdamW(
            model.fc.parameters(),
            lr=CLASSIFIER_LR,
            weight_decay=WEIGHT_DECAY,
        )
    )

    print(
        "\nStage 1: classifier "
        "head only"
    )

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):
        train_loss = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        )

        dev_result = evaluate(
            model,
            dev_loader,
            criterion,
            device,
        )

        best_auc, saved = (
            save_best(
                model,
                "head",
                epoch,
                dev_result["auc"],
                best_auc,
                weights,
            )
        )

        print(
            f"head {epoch:02d}/"
            f"{HEAD_EPOCHS} "
            f"train_loss="
            f"{train_loss:.4f} "
            f"dev_loss="
            f"{dev_result['loss']:.4f} "
            f"dev_auc="
            f"{dev_result['auc']:.4f} "
            f"{'*' if saved else ''}"
        )

    # --------------------------------------------------
    # Stage 2: full fine-tuning.
    # Same 2160 optimizer steps as baseline.
    # --------------------------------------------------

    set_full(
        model
    )

    backbone = [
        p
        for name, p
        in model.named_parameters()
        if not name.startswith(
            "fc."
        )
    ]

    optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params":
                        backbone,

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
    )

    print(
        "\nStage 2: full fine-tuning"
    )

    for epoch in range(
        1,
        FINETUNE_EPOCHS + 1,
    ):
        train_loss = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        )

        dev_result = evaluate(
            model,
            dev_loader,
            criterion,
            device,
        )

        best_auc, saved = (
            save_best(
                model,
                "full",
                epoch,
                dev_result["auc"],
                best_auc,
                weights,
            )
        )

        print(
            f"full {epoch:02d}/"
            f"{FINETUNE_EPOCHS} "
            f"train_loss="
            f"{train_loss:.4f} "
            f"dev_loss="
            f"{dev_result['loss']:.4f} "
            f"dev_auc="
            f"{dev_result['auc']:.4f} "
            f"{'*' if saved else ''}"
        )

    # --------------------------------------------------
    # Freeze best natural-dev checkpoint.
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    print(
        "\nBest checkpoint:"
        f"\n  stage:   "
        f"{checkpoint['stage']}"
        f"\n  epoch:   "
        f"{checkpoint['epoch']}"
        f"\n  dev_auc: "
        f"{checkpoint['dev_auc']:.4f}"
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

    train_pred = (
        prediction_frame(
            train,
            train_result,
        )
    )

    dev_pred = (
        prediction_frame(
            dev,
            dev_result,
        )
    )

    train_pred.insert(
        0,
        "split",
        "augmented_project_train",
    )

    # Dev already carries split from
    # Policy-C index.
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

    summary = dev_summary(
        dev_pred
    )

    summary.insert(
        0,
        "experiment",
        EXPERIMENT,
    )

    summary.insert(
        1,
        "seed",
        SEED,
    )

    PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS,
        index=False,
    )

    summary.to_csv(
        METRICS,
        index=False,
    )

    columns = [
        "group_type",
        "group",
        "n",
        "auroc",
        "balanced_accuracy",
        "attack_recall",
        "bonafide_specificity",
        "mean_attack_probability",
        "mean_bonafide_probability",
    ]

    print(
        "\nUntouched natural DEV metrics:"
    )

    print(
        summary[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\ncheckpoint:  {CHECKPOINT}"
        f"\npredictions: {PREDICTIONS}"
        f"\nmetrics:     {METRICS}"
    )

    print(
        "\nStop here."
        "\nDo NOT inspect the official test yet."
        "\nNext decision: does the "
        "counterfactual-augmented model "
        "retain natural dev viability?"
    )


if __name__ == "__main__":
    main()