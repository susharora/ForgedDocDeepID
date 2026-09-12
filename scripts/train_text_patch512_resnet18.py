#!/usr/bin/env python3
"""
Native 512x512 text-patch ResNet diagnostic.

Train:
    digital_1 / digital_2 project_train text patches only.

Checkpoint selection:
    card-disjoint digital_1 / digital_2 dev patches only.

Post-selection diagnostic:
    same-source Digital-3 project-dev text patches.

No whole-document resizing is performed.
No Digital-3 sample contributes to training or checkpoint selection.
"""

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

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

INDEX = (
    ROOT
    / "output"
    / "text_patch512_index.csv"
)

RUN = (
    ROOT
    / "runs"
    / "text_patch512_resnet18_seed10"
)

CHECKPOINT = (
    RUN
    / "checkpoints"
    / "best.pt"
)

OUT_PREDICTIONS = (
    ROOT
    / "output"
    / "text_patch512_resnet18_seed10_predictions.csv"
)

OUT_METRICS = (
    ROOT
    / "output"
    / "text_patch512_resnet18_seed10_metrics.csv"
)

EXPERIMENT = (
    "POST_HOC_NATIVE_TEXT_PATCH512_DIAGNOSTIC"
)

SEED = 10

PATCH = 512

BATCH_SIZE = 16

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)

HEAD_EPOCHS = 1
FINETUNE_EPOCHS = 6

BACKBONE_LR = 1e-4
CLASSIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-2

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PatchDataset(Dataset):
    def __init__(
        self,
        frame,
    ):
        self.frame = (
            frame
            .reset_index(
                drop=True
            )
            .copy()
        )

    def __len__(self):
        return len(
            self.frame
        )

    def __getitem__(
        self,
        index,
    ):
        row = (
            self.frame.iloc[
                index
            ]
        )

        path = (
            ROOT
            / row.patch_path
        )

        with Image.open(
            path
        ) as im:
            im = im.convert(
                "RGB"
            )

            if im.size != (
                PATCH,
                PATCH,
            ):
                raise RuntimeError(
                    "Patch is not "
                    f"{PATCH}x{PATCH}: "
                    f"{path}"
                )

            tensor = (
                TF.to_tensor(
                    im
                )
            )

        tensor = TF.normalize(
            tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        return (
            tensor,
            int(
                row.label
            ),
            index,
        )


def make_loader(
    frame,
    shuffle,
    device,
    generator=None,
):
    return DataLoader(
        PatchDataset(
            frame
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
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
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            False
        )

    for parameter in (
        model.fc.parameters()
    ):
        parameter.requires_grad = (
            True
        )


def set_full(model):
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            True
        )


def amp_context(device):
    return torch.autocast(
        device_type=(
            device.type
        ),
        dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.bfloat16
        ),
        enabled=(
            device.type
            == "cuda"
        ),
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):
    model.train()

    loss_total = 0.0
    count = 0

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

        loss_total += (
            float(
                loss.item()
            )
            * len(y)
        )

        count += len(y)

    return (
        loss_total
        / count
    )


@torch.no_grad()
def predict(
    model,
    loader,
    device,
):
    model.eval()

    scores = []
    labels = []
    indices = []

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        with amp_context(
            device
        ):
            logits = model(x)

        probability = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .float()
            .cpu()
            .numpy()
        )

        scores.append(
            probability
        )

        labels.append(
            y.numpy()
        )

        indices.append(
            idx.numpy()
        )

    return {
        "score":
            np.concatenate(
                scores
            ),

        "label":
            np.concatenate(
                labels
            ),

        "index":
            np.concatenate(
                indices
            ),
    }


def attach_predictions(
    frame,
    result,
):
    out = (
        frame
        .reset_index(
            drop=True
        )
        .copy()
    )

    out[
        "attack_probability"
    ] = np.nan

    out.loc[
        result[
            "index"
        ],
        "attack_probability",
    ] = result[
        "score"
    ]

    if (
        out[
            "attack_probability"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Prediction alignment "
            "failure"
        )

    return out


def patch_metrics(frame):
    y = (
        frame[
            "label"
        ]
        .to_numpy()
    )

    p = (
        frame[
            "attack_probability"
        ]
        .to_numpy()
    )

    pred = (
        p >= 0.5
    ).astype(
        int
    )

    return {
        "patch_auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "patch_balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),
    }


def document_scores(
    frame,
    aggregation,
):
    """
    Aggregate multiple field patches into one score per document.

    Parents are intentionally retained separately for each attack variant:
    the comparison remains exactly variant-matched.
    """

    if aggregation == "max":
        aggregate = "max"

    elif aggregation == "mean":
        aggregate = "mean"

    else:
        raise ValueError(
            aggregation
        )

    document = (
        frame.groupby(
            [
                "variant",
                "role",
                "label",
                "file_stem",
                "hardware_source",
                "attack_image_path",
                "parent_image_path",
            ],
            as_index=False,
        )
        .agg(
            attack_probability=(
                "attack_probability",
                aggregate,
            )
        )
    )

    return document


def document_auc(
    frame,
    aggregation="max",
):
    document = (
        document_scores(
            frame,
            aggregation,
        )
    )

    return float(
        roc_auc_score(
            document[
                "label"
            ],
            document[
                "attack_probability"
            ],
        )
    )


def metrics_by_variant(
    frame,
    population,
):
    rows = []

    variants = sorted(
        frame[
            "variant"
        ].unique()
    )

    for variant in [
        "all",
        *variants,
    ]:
        if variant == "all":
            subset = frame

        else:
            subset = frame[
                frame[
                    "variant"
                ]
                == variant
            ]

        patch = patch_metrics(
            subset
        )

        max_auc = document_auc(
            subset,
            "max",
        )

        mean_auc = document_auc(
            subset,
            "mean",
        )

        rows.append(
            {
                "population":
                    population,

                "variant":
                    variant,

                "n_patch_rows":
                    len(
                        subset
                    ),

                "n_patch_pairs":
                    subset[
                        "pair_id"
                    ].nunique(),

                "n_stems":
                    subset[
                        "file_stem"
                    ].nunique(),

                **patch,

                "document_auc_max":
                    max_auc,

                "document_auc_mean":
                    mean_auc,
            }
        )

    return rows


def same_source_deltas(
    frame,
):
    """
    Exact attack-vs-parent comparison for each field pair.
    """

    attack = (
        frame[
            frame[
                "role"
            ]
            == "attack"
        ][
            [
                "pair_id",
                "variant",
                "file_stem",
                "hardware_source",
                "field_name",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    "attack_score",
            }
        )
    )

    parent = (
        frame[
            frame[
                "role"
            ]
            == "bonafide"
        ][
            [
                "pair_id",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    "parent_score",
            }
        )
    )

    paired = attack.merge(
        parent,
        on="pair_id",
        validate="one_to_one",
    )

    paired[
        "delta"
    ] = (
        paired[
            "attack_score"
        ]
        -
        paired[
            "parent_score"
        ]
    )

    return paired


def d3_field_metrics(
    frame,
):
    rows = []

    for field in [
        "all",
        *sorted(
            frame[
                "field_name"
            ].unique()
        ),
    ]:
        if field == "all":
            subset = frame

        else:
            subset = frame[
                frame[
                    "field_name"
                ]
                == field
            ]

        if (
            subset[
                "label"
            ]
            .nunique()
            != 2
        ):
            continue

        paired = (
            same_source_deltas(
                subset
            )
        )

        rows.append(
            {
                "population":
                    "digital3_dev",

                "field_name":
                    field,

                "n_patch_pairs":
                    subset[
                        "pair_id"
                    ].nunique(),

                "n_stems":
                    subset[
                        "file_stem"
                    ].nunique(),

                "patch_auc":
                    float(
                        roc_auc_score(
                            subset[
                                "label"
                            ],
                            subset[
                                "attack_probability"
                            ],
                        )
                    ),

                "mean_parent_score":
                    paired[
                        "parent_score"
                    ].mean(),

                "mean_attack_score":
                    paired[
                        "attack_score"
                    ].mean(),

                "mean_delta":
                    paired[
                        "delta"
                    ].mean(),

                "median_delta":
                    paired[
                        "delta"
                    ].median(),

                "fraction_score_increased":
                    (
                        paired[
                            "delta"
                        ]
                        > 0
                    ).mean(),
            }
        )

    return rows


def validate_index(index):
    train = index[
        index[
            "split"
        ]
        == "project_train"
    ]

    dev = index[
        index[
            "split"
        ]
        == "dev_val"
    ]

    d3 = index[
        index[
            "split"
        ]
        == "digital3_dev"
    ]

    if (
        len(train) == 0
        or len(dev) == 0
        or len(d3) == 0
    ):
        raise RuntimeError(
            "Patch index incomplete"
        )

    train_stems = set(
        train[
            "file_stem"
        ]
    )

    dev_stems = set(
        dev[
            "file_stem"
        ]
    )

    d3_stems = set(
        d3[
            "file_stem"
        ]
    )

    if (
        train_stems
        & dev_stems
    ):
        raise RuntimeError(
            "Project train/dev "
            "identity overlap"
        )

    if (
        d3_stems
        != dev_stems
    ):
        raise RuntimeError(
            "Digital-3 diagnostic "
            "population is not "
            "the dev card population"
        )

    for name, frame in [
        (
            "train",
            train,
        ),
        (
            "dev",
            dev,
        ),
        (
            "digital3",
            d3,
        ),
    ]:
        counts = (
            frame[
                "label"
            ]
            .value_counts()
            .to_dict()
        )

        if (
            counts.get(
                0,
                0,
            )
            !=
            counts.get(
                1,
                0,
            )
        ):
            raise RuntimeError(
                f"{name} classes "
                "are not balanced"
            )

        if (
            frame[
                "face_overlap_pixels"
            ]
            .max()
            != 0
        ):
            raise RuntimeError(
                f"{name} contains "
                "face overlap"
            )

    return (
        train,
        dev,
        d3,
    )


def save_checkpoint(
    model,
    stage,
    epoch,
    dev_auc,
):
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

            "input":
                "native_policy_c_text_patch",

            "patch_size":
                512,

            "whole_document_resize":
                False,

            "face_overlap":
                0,

            "checkpoint_selection":
                "dev_document_auc_max",

            "stage":
                stage,

            "epoch":
                epoch,

            "dev_document_auc_max":
                dev_auc,

            "model_state":
                model.state_dict(),
        },
        CHECKPOINT,
    )


def main():
    seed_everything(
        SEED
    )

    index = pd.read_csv(
        INDEX
    )

    (
        train,
        dev,
        d3,
    ) = validate_index(
        index
    )

    train = train.reset_index(
        drop=True
    )

    dev = dev.reset_index(
        drop=True
    )

    d3 = d3.reset_index(
        drop=True
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
        f"\ntrain rows: {len(train)}"
        f"\ndev rows:   {len(dev)}"
        f"\nd3 rows:    {len(d3)}"
        "\ninput:      native 512x512"
        "\nresize:     NONE"
        "\nface overlap: ZERO"
    )

    if device.type == "cuda":
        print(
            "gpu:        "
            + torch.cuda
            .get_device_name(0)
        )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    train_loader = make_loader(
        train,
        shuffle=True,
        device=device,
        generator=generator,
    )

    dev_loader = make_loader(
        dev,
        shuffle=False,
        device=device,
    )

    d3_loader = make_loader(
        d3,
        shuffle=False,
        device=device,
    )

    model = build_model().to(
        device
    )

    # Exactly balanced paired patches.
    criterion = (
        nn.CrossEntropyLoss()
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
    # Head-only
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
        "\nStage 1: head only"
    )

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        result = predict(
            model,
            dev_loader,
            device,
        )

        dev_pred = (
            attach_predictions(
                dev,
                result,
            )
        )

        dev_auc = (
            document_auc(
                dev_pred,
                "max",
            )
        )

        patch_auc = float(
            roc_auc_score(
                dev_pred[
                    "label"
                ],
                dev_pred[
                    "attack_probability"
                ],
            )
        )

        saved = False

        if dev_auc > best_auc:
            best_auc = (
                dev_auc
            )

            save_checkpoint(
                model,
                "head",
                epoch,
                dev_auc,
            )

            saved = True

        print(
            f"head {epoch:02d}/"
            f"{HEAD_EPOCHS} "
            f"loss={loss:.4f} "
            f"dev_patch_auc="
            f"{patch_auc:.4f} "
            f"dev_doc_auc="
            f"{dev_auc:.4f} "
            f"{'*' if saved else ''}"
        )

    # --------------------------------------------------
    # Full fine-tuning
    # --------------------------------------------------

    set_full(
        model
    )

    backbone = [
        parameter
        for name, parameter
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
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        result = predict(
            model,
            dev_loader,
            device,
        )

        dev_pred = (
            attach_predictions(
                dev,
                result,
            )
        )

        dev_auc = (
            document_auc(
                dev_pred,
                "max",
            )
        )

        patch_auc = float(
            roc_auc_score(
                dev_pred[
                    "label"
                ],
                dev_pred[
                    "attack_probability"
                ],
            )
        )

        saved = False

        if dev_auc > best_auc:
            best_auc = (
                dev_auc
            )

            save_checkpoint(
                model,
                "full",
                epoch,
                dev_auc,
            )

            saved = True

        print(
            f"full {epoch:02d}/"
            f"{FINETUNE_EPOCHS} "
            f"loss={loss:.4f} "
            f"dev_patch_auc="
            f"{patch_auc:.4f} "
            f"dev_doc_auc="
            f"{dev_auc:.4f} "
            f"{'*' if saved else ''}"
        )

    # --------------------------------------------------
    # Freeze best checkpoint
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
        f"\n  stage: "
        f"{checkpoint['stage']}"
        f"\n  epoch: "
        f"{checkpoint['epoch']}"
        f"\n  dev document AUROC: "
        f"{checkpoint['dev_document_auc_max']:.4f}"
    )

    # --------------------------------------------------
    # Final dev predictions
    # --------------------------------------------------

    dev_result = predict(
        model,
        dev_loader,
        device,
    )

    dev_pred = (
        attach_predictions(
            dev,
            dev_result,
        )
    )

    # --------------------------------------------------
    # Digital-3 diagnostic AFTER checkpoint is frozen
    # --------------------------------------------------

    d3_result = predict(
        model,
        d3_loader,
        device,
    )

    d3_pred = (
        attach_predictions(
            d3,
            d3_result,
        )
    )

    predictions = pd.concat(
        [
            dev_pred,
            d3_pred,
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

    predictions.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metric_rows = []

    metric_rows.extend(
        metrics_by_variant(
            dev_pred,
            "dev_val",
        )
    )

    metric_rows.extend(
        metrics_by_variant(
            d3_pred,
            "digital3_dev",
        )
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    field_metrics = pd.DataFrame(
        d3_field_metrics(
            d3_pred
        )
    )

    # Store field-specific rows in the same CSV with distinguishing columns.
    field_metrics[
        "variant"
    ] = "digital_3"

    field_metrics[
        "metric_scope"
    ] = "field"

    metrics[
        "metric_scope"
    ] = "population"

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    # Separate field file is easiest to inspect.
    field_path = (
        ROOT
        / "output"
        / "text_patch512_resnet18_seed10_digital3_fields.csv"
    )

    field_metrics.to_csv(
        field_path,
        index=False,
    )

    # --------------------------------------------------
    # Same-source image-level Digital-3 deltas
    # --------------------------------------------------

    d3_max = (
        document_scores(
            d3_pred,
            "max",
        )
    )

    attack_doc = (
        d3_max[
            d3_max[
                "role"
            ]
            == "attack"
        ][
            [
                "file_stem",
                "hardware_source",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    "attack_max",
            }
        )
    )

    parent_doc = (
        d3_max[
            d3_max[
                "role"
            ]
            == "bonafide"
        ][
            [
                "file_stem",
                "hardware_source",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    "parent_max",
            }
        )
    )

    d3_pair = attack_doc.merge(
        parent_doc,
        on=[
            "file_stem",
            "hardware_source",
        ],
        validate="one_to_one",
    )

    d3_pair[
        "delta"
    ] = (
        d3_pair[
            "attack_max"
        ]
        -
        d3_pair[
            "parent_max"
        ]
    )

    print(
        "\nHELD-OUT D1/D2 "
        "TEXT-PATCH PERFORMANCE:"
    )

    print(
        metrics[
            metrics[
                "population"
            ]
            == "dev_val"
        ][
            [
                "variant",
                "n_patch_pairs",
                "n_stems",
                "patch_auc",
                "patch_balanced_accuracy",
                "document_auc_max",
                "document_auc_mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nDIGITAL-3 SAME-SOURCE "
        "TEXT-PATCH DIAGNOSTIC:"
    )

    print(
        metrics[
            metrics[
                "population"
            ]
            == "digital3_dev"
        ][
            [
                "variant",
                "n_patch_pairs",
                "n_stems",
                "patch_auc",
                "patch_balanced_accuracy",
                "document_auc_max",
                "document_auc_mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nDIGITAL-3 BY ALTERED FIELD:"
    )

    print(
        field_metrics[
            [
                "field_name",
                "n_patch_pairs",
                "n_stems",
                "patch_auc",
                "mean_parent_score",
                "mean_attack_score",
                "mean_delta",
                "median_delta",
                "fraction_score_increased",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nDIGITAL-3 IMAGE-LEVEL "
        "MAX-PATCH PAIRED RESPONSE:"
        f"\n  n images: "
        f"{len(d3_pair)}"
        f"\n  mean parent max: "
        f"{d3_pair['parent_max'].mean():.4f}"
        f"\n  mean attack max: "
        f"{d3_pair['attack_max'].mean():.4f}"
        f"\n  mean delta: "
        f"{d3_pair['delta'].mean():.4f}"
        f"\n  median delta: "
        f"{d3_pair['delta'].median():.4f}"
        f"\n  fraction increased: "
        f"{(d3_pair['delta'] > 0).mean():.4f}"
    )

    print(
        f"\nCheckpoint:  {CHECKPOINT}"
        f"\nPredictions: {OUT_PREDICTIONS}"
        f"\nMetrics:     {OUT_METRICS}"
        f"\nFields:      {field_path}"
    )

    print(
        "\nInterpretation:"
        "\n- Strong D1/D2 dev + weak Digital-3 "
        "=> manipulation-method shift."
        "\n- Strong D1/D2 dev + strong Digital-3 "
        "=> whole-document spatial dilution was "
        "a major cause of the previous failure."
        "\n- Weak D1/D2 dev itself "
        "=> this simple patch classifier is not "
        "a sufficient local-forensics representation."
    )


if __name__ == "__main__":
    main()