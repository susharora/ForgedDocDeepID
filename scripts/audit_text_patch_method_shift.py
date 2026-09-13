#!/usr/bin/env python3
"""
Frozen native-text-patch representation audit.

No training and no checkpoint selection.

For every exact attack/parent patch pair, measure:

1. Pixel-space manipulation magnitude.
2. ResNet penultimate-feature shift:
       dz = z_attack - z_parent
3. Projection of dz onto the learned binary attack axis.
4. Cosine alignment of dz with that axis.
5. Similarity to the typical project-train digital_1 / digital_2
   manipulation directions.

Purpose
-------
Distinguish:

A. Digital-3 is simply a much weaker pixel manipulation.

B. Digital-3 produces a feature change, but the backbone encodes it weakly.

C. Digital-3 produces a substantial feature change, but in a direction
   poorly aligned with the learned d1/d2 forgery representation.

The frozen Digital-3 population remains diagnostic only.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

from scipy.ndimage import laplace
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

INDEX = (
    ROOT
    / "output"
    / "text_patch512_index.csv"
)

CHECKPOINT = (
    ROOT
    / "runs"
    / "text_patch512_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

CHECKPOINT_HASH = (
    ROOT
    / "output"
    / "text_patch512_resnet18_seed10_checkpoint.sha256"
)

OUT_PAIRS = (
    ROOT
    / "output"
    / "text_patch512_method_shift_pairs.csv"
)

OUT_SUMMARY = (
    ROOT
    / "output"
    / "text_patch512_method_shift_summary.csv"
)

OUT_FIELDS = (
    ROOT
    / "output"
    / "text_patch512_method_shift_fields.csv"
)

OUT_CORRELATION = (
    ROOT
    / "output"
    / "text_patch512_method_shift_correlations.csv"
)


BATCH_SIZE = 32
NUM_WORKERS = 4

MEAN = (
    0.485,
    0.456,
    0.406,
)

STD = (
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


def verify_checkpoint():
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
            "Patch checkpoint SHA mismatch"
        )

    return actual


class PatchDataset(Dataset):
    def __init__(self, frame):
        self.frame = (
            frame
            .reset_index(drop=True)
            .copy()
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = (
            ROOT
            / self.frame.iloc[
                index
            ].patch_path
        )

        with Image.open(path) as im:
            im = im.convert("RGB")

            if im.size != (
                512,
                512,
            ):
                raise RuntimeError(
                    f"Unexpected patch size: {path}"
                )

            x = TF.to_tensor(im)

        x = TF.normalize(
            x,
            mean=MEAN,
            std=STD,
        )

        return x, index


def build_model(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
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

    return model, checkpoint


@torch.no_grad()
def extract_features(
    model,
    frame,
    device,
):
    loader = DataLoader(
        PatchDataset(frame),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    # Everything before the final linear layer.
    backbone = nn.Sequential(
        *list(
            model.children()
        )[:-1]
    )

    backbone.eval()

    n = len(frame)

    features = np.empty(
        (
            n,
            model.fc.in_features,
        ),
        dtype=np.float32,
    )

    for x, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        z = (
            backbone(x)
            .flatten(1)
            .float()
            .cpu()
            .numpy()
        )

        features[
            idx.numpy()
        ] = z

    return features


def gray(rgb):
    rgb = rgb.astype(
        np.float32
    )

    return (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )


def load_rgb(path):
    with Image.open(
        ROOT / path
    ) as im:
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def annotation_mask(row):
    """
    Annotation rectangle intersected with the 512x512 patch.
    """

    x0 = int(
        round(
            row.annotation_x0
            - row.crop_x0
        )
    )

    y0 = int(
        round(
            row.annotation_y0
            - row.crop_y0
        )
    )

    x1 = int(
        round(
            row.annotation_x1
            - row.crop_x0
        )
    )

    y1 = int(
        round(
            row.annotation_y1
            - row.crop_y0
        )
    )

    cx0 = max(
        0,
        min(
            512,
            x0,
        ),
    )

    cy0 = max(
        0,
        min(
            512,
            y0,
        ),
    )

    cx1 = max(
        0,
        min(
            512,
            x1,
        ),
    )

    cy1 = max(
        0,
        min(
            512,
            y1,
        ),
    )

    if (
        cx1 <= cx0
        or cy1 <= cy0
    ):
        raise RuntimeError(
            f"Annotation outside patch: "
            f"{row.pair_id}"
        )

    mask = np.zeros(
        (
            512,
            512,
        ),
        dtype=bool,
    )

    mask[
        cy0:cy1,
        cx0:cx1,
    ] = True

    annotation_area = max(
        1,
        (
            int(row.annotation_x1)
            - int(row.annotation_x0)
        )
        *
        (
            int(row.annotation_y1)
            - int(row.annotation_y0)
        ),
    )

    visible_area = (
        (cx1 - cx0)
        *
        (cy1 - cy0)
    )

    coverage = (
        visible_area
        / annotation_area
    )

    return (
        mask,
        coverage,
    )


def pixel_metrics(
    attack_row,
    parent_row,
):
    attack = load_rgb(
        attack_row.patch_path
    )

    parent = load_rgb(
        parent_row.patch_path
    )

    if (
        attack.shape
        != parent.shape
    ):
        raise RuntimeError(
            "Patch shape mismatch"
        )

    mask, coverage = (
        annotation_mask(
            attack_row
        )
    )

    attack_g = gray(
        attack
    )

    parent_g = gray(
        parent
    )

    diff = np.abs(
        attack_g
        - parent_g
    )

    lap_diff = np.abs(
        laplace(
            attack_g,
            mode="reflect",
        )
        -
        laplace(
            parent_g,
            mode="reflect",
        )
    )

    inside = diff[
        mask
    ]

    whole = diff.ravel()

    return {
        "gray_mad_inside":
            float(
                inside.mean()
            ),

        "gray_mad_patch":
            float(
                whole.mean()
            ),

        "lap_mad_inside":
            float(
                lap_diff[
                    mask
                ].mean()
            ),

        "changed_inside_ge2":
            float(
                (
                    inside >= 2
                ).mean()
            ),

        "changed_inside_ge5":
            float(
                (
                    inside >= 5
                ).mean()
            ),

        "annotation_coverage":
            float(
                coverage
            ),
    }


def sigmoid(x):
    return (
        1.0
        /
        (
            1.0
            + np.exp(
                -np.clip(
                    x,
                    -50,
                    50,
                )
            )
        )
    )


def unit(v):
    norm = np.linalg.norm(
        v
    )

    if norm <= 1e-12:
        return np.zeros_like(
            v
        )

    return (
        v / norm
    )


def mean_unit_direction(
    delta_matrix,
):
    norms = np.linalg.norm(
        delta_matrix,
        axis=1,
    )

    valid = (
        norms > 1e-12
    )

    normalized = (
        delta_matrix[
            valid
        ]
        /
        norms[
            valid,
            None,
        ]
    )

    return unit(
        normalized.mean(
            axis=0
        )
    )


def cosine_to(
    delta,
    direction,
):
    delta_norm = (
        np.linalg.norm(
            delta
        )
    )

    if (
        delta_norm <= 1e-12
        or
        np.linalg.norm(
            direction
        )
        <= 1e-12
    ):
        return np.nan

    return float(
        np.dot(
            delta,
            direction,
        )
        /
        delta_norm
    )


def build_pair_table(
    frame,
    features,
    attack_axis,
    attack_bias,
):
    records = []
    deltas = []

    axis_norm = float(
        np.linalg.norm(
            attack_axis
        )
    )

    if axis_norm <= 0:
        raise RuntimeError(
            "Degenerate classifier axis"
        )

    grouped = frame.groupby(
        "pair_id",
        sort=False,
    )

    for pair_number, (
        pair_id,
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):
        if len(group) != 2:
            raise RuntimeError(
                f"{pair_id}: "
                "expected two rows"
            )

        attack_group = group[
            group.role
            == "attack"
        ]

        parent_group = group[
            group.role
            == "bonafide"
        ]

        if (
            len(attack_group) != 1
            or
            len(parent_group) != 1
        ):
            raise RuntimeError(
                f"{pair_id}: invalid roles"
            )

        attack_row = (
            attack_group.iloc[0]
        )

        parent_row = (
            parent_group.iloc[0]
        )

        attack_idx = int(
            attack_row.row_id
        )

        parent_idx = int(
            parent_row.row_id
        )

        za = features[
            attack_idx
        ]

        zp = features[
            parent_idx
        ]

        delta = (
            za - zp
        )

        shift_norm = float(
            np.linalg.norm(
                delta
            )
        )

        parent_margin = float(
            np.dot(
                attack_axis,
                zp,
            )
            + attack_bias
        )

        attack_margin = float(
            np.dot(
                attack_axis,
                za,
            )
            + attack_bias
        )

        delta_margin = (
            attack_margin
            - parent_margin
        )

        axis_projection = (
            delta_margin
            / axis_norm
        )

        axis_cosine = (
            delta_margin
            /
            (
                axis_norm
                * shift_norm
            )
            if shift_norm > 1e-12
            else np.nan
        )

        parent_prob = float(
            sigmoid(
                parent_margin
            )
        )

        attack_prob = float(
            sigmoid(
                attack_margin
            )
        )

        pixels = pixel_metrics(
            attack_row,
            parent_row,
        )

        records.append(
            {
                "pair_id":
                    pair_id,

                "split":
                    attack_row.split,

                "variant":
                    attack_row.variant,

                "file_stem":
                    attack_row.file_stem,

                "hardware_source":
                    attack_row.hardware_source,

                "field_name":
                    attack_row.field_name,

                "assigned_q":
                    attack_row.assigned_q,

                "parent_probability":
                    parent_prob,

                "attack_probability":
                    attack_prob,

                "probability_delta":
                    (
                        attack_prob
                        - parent_prob
                    ),

                "parent_margin":
                    parent_margin,

                "attack_margin":
                    attack_margin,

                "margin_delta":
                    delta_margin,

                "feature_shift_norm":
                    shift_norm,

                "attack_axis_projection":
                    axis_projection,

                "attack_axis_cosine":
                    axis_cosine,

                **pixels,
            }
        )

        deltas.append(
            delta.astype(
                np.float32
            )
        )

        if (
            pair_number % 500 == 0
            or
            pair_number
            == grouped.ngroups
        ):
            print(
                "paired "
                f"{pair_number}/"
                f"{grouped.ngroups}"
            )

    return (
        pd.DataFrame(
            records
        ),
        np.stack(
            deltas
        ),
    )


def add_reference_alignment(
    pairs,
    deltas,
):
    train = (
        pairs.split
        == "project_train"
    )

    d1_train = (
        train
        &
        (
            pairs.variant
            == "digital_1"
        )
    )

    d2_train = (
        train
        &
        (
            pairs.variant
            == "digital_2"
        )
    )

    global_direction = (
        mean_unit_direction(
            deltas[
                train.to_numpy()
            ]
        )
    )

    d1_direction = (
        mean_unit_direction(
            deltas[
                d1_train.to_numpy()
            ]
        )
    )

    d2_direction = (
        mean_unit_direction(
            deltas[
                d2_train.to_numpy()
            ]
        )
    )

    field_directions = {}

    for field_name, group in (
        pairs[
            train
        ]
        .groupby(
            "field_name"
        )
    ):
        indices = (
            group.index
            .to_numpy()
        )

        if len(indices) >= 30:
            field_directions[
                field_name
            ] = (
                mean_unit_direction(
                    deltas[
                        indices
                    ]
                )
            )

    cos_global = []
    cos_d1 = []
    cos_d2 = []
    cos_field = []

    for i, row in (
        pairs.iterrows()
    ):
        delta = deltas[
            i
        ]

        cos_global.append(
            cosine_to(
                delta,
                global_direction,
            )
        )

        cos_d1.append(
            cosine_to(
                delta,
                d1_direction,
            )
        )

        cos_d2.append(
            cosine_to(
                delta,
                d2_direction,
            )
        )

        direction = (
            field_directions.get(
                row.field_name
            )
        )

        cos_field.append(
            (
                cosine_to(
                    delta,
                    direction,
                )
                if direction
                is not None
                else np.nan
            )
        )

    pairs[
        "cos_to_train_global"
    ] = cos_global

    pairs[
        "cos_to_train_d1"
    ] = cos_d1

    pairs[
        "cos_to_train_d2"
    ] = cos_d2

    pairs[
        "cos_to_train_same_field"
    ] = cos_field

    return pairs


def same_source_auc(frame):
    y = np.concatenate(
        [
            np.zeros(
                len(frame),
                dtype=int,
            ),
            np.ones(
                len(frame),
                dtype=int,
            ),
        ]
    )

    p = np.concatenate(
        [
            frame[
                "parent_probability"
            ].to_numpy(),

            frame[
                "attack_probability"
            ].to_numpy(),
        ]
    )

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def summarize(
    pairs,
):
    rows = []

    populations = [
        (
            "dev_digital_1",
            pairs[
                (
                    pairs.split
                    == "dev_val"
                )
                &
                (
                    pairs.variant
                    == "digital_1"
                )
            ],
        ),
        (
            "dev_digital_2",
            pairs[
                (
                    pairs.split
                    == "dev_val"
                )
                &
                (
                    pairs.variant
                    == "digital_2"
                )
            ],
        ),
        (
            "dev_digital_3",
            pairs[
                pairs.split
                == "digital3_dev"
            ],
        ),
    ]

    for name, frame in populations:
        rows.append(
            {
                "population":
                    name,

                "n_pairs":
                    len(frame),

                "n_stems":
                    frame[
                        "file_stem"
                    ].nunique(),

                "same_source_auc":
                    same_source_auc(
                        frame
                    ),

                "mean_parent_probability":
                    frame[
                        "parent_probability"
                    ].mean(),

                "mean_attack_probability":
                    frame[
                        "attack_probability"
                    ].mean(),

                "mean_probability_delta":
                    frame[
                        "probability_delta"
                    ].mean(),

                "median_probability_delta":
                    frame[
                        "probability_delta"
                    ].median(),

                "mean_feature_shift_norm":
                    frame[
                        "feature_shift_norm"
                    ].mean(),

                "median_feature_shift_norm":
                    frame[
                        "feature_shift_norm"
                    ].median(),

                "mean_attack_axis_projection":
                    frame[
                        "attack_axis_projection"
                    ].mean(),

                "median_attack_axis_projection":
                    frame[
                        "attack_axis_projection"
                    ].median(),

                "fraction_positive_projection":
                    (
                        frame[
                            "attack_axis_projection"
                        ]
                        > 0
                    ).mean(),

                "mean_attack_axis_cosine":
                    frame[
                        "attack_axis_cosine"
                    ].mean(),

                "mean_cos_train_global":
                    frame[
                        "cos_to_train_global"
                    ].mean(),

                "mean_cos_train_d1":
                    frame[
                        "cos_to_train_d1"
                    ].mean(),

                "mean_cos_train_d2":
                    frame[
                        "cos_to_train_d2"
                    ].mean(),

                "mean_cos_train_same_field":
                    frame[
                        "cos_to_train_same_field"
                    ].mean(),

                "mean_gray_mad_inside":
                    frame[
                        "gray_mad_inside"
                    ].mean(),

                "mean_lap_mad_inside":
                    frame[
                        "lap_mad_inside"
                    ].mean(),

                "mean_changed_inside_ge5":
                    frame[
                        "changed_inside_ge5"
                    ].mean(),

                "mean_annotation_coverage":
                    frame[
                        "annotation_coverage"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_fields(
    pairs,
):
    rows = []

    for (
        split,
        variant,
        field_name,
    ), frame in (
        pairs.groupby(
            [
                "split",
                "variant",
                "field_name",
            ]
        )
    ):
        rows.append(
            {
                "split":
                    split,

                "variant":
                    variant,

                "field_name":
                    field_name,

                "n_pairs":
                    len(frame),

                "n_stems":
                    frame[
                        "file_stem"
                    ].nunique(),

                "same_source_auc":
                    same_source_auc(
                        frame
                    ),

                "mean_probability_delta":
                    frame[
                        "probability_delta"
                    ].mean(),

                "median_probability_delta":
                    frame[
                        "probability_delta"
                    ].median(),

                "mean_feature_shift_norm":
                    frame[
                        "feature_shift_norm"
                    ].mean(),

                "mean_axis_projection":
                    frame[
                        "attack_axis_projection"
                    ].mean(),

                "mean_axis_cosine":
                    frame[
                        "attack_axis_cosine"
                    ].mean(),

                "fraction_positive_projection":
                    (
                        frame[
                            "attack_axis_projection"
                        ]
                        > 0
                    ).mean(),

                "mean_cos_train_global":
                    frame[
                        "cos_to_train_global"
                    ].mean(),

                "mean_cos_train_same_field":
                    frame[
                        "cos_to_train_same_field"
                    ].mean(),

                "mean_gray_mad_inside":
                    frame[
                        "gray_mad_inside"
                    ].mean(),

                "mean_lap_mad_inside":
                    frame[
                        "lap_mad_inside"
                    ].mean(),

                "mean_changed_inside_ge5":
                    frame[
                        "changed_inside_ge5"
                    ].mean(),

                "mean_annotation_coverage":
                    frame[
                        "annotation_coverage"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def correlations(
    pairs,
):
    rows = []

    features = [
        "gray_mad_inside",
        "lap_mad_inside",
        "changed_inside_ge5",
        "feature_shift_norm",
        "attack_axis_cosine",
        "cos_to_train_global",
        "cos_to_train_same_field",
    ]

    targets = [
        "probability_delta",
        "attack_axis_projection",
    ]

    populations = [
        (
            "dev_digital_1",
            pairs[
                (
                    pairs.split
                    == "dev_val"
                )
                &
                (
                    pairs.variant
                    == "digital_1"
                )
            ],
        ),
        (
            "dev_digital_2",
            pairs[
                (
                    pairs.split
                    == "dev_val"
                )
                &
                (
                    pairs.variant
                    == "digital_2"
                )
            ],
        ),
        (
            "dev_digital_3",
            pairs[
                pairs.split
                == "digital3_dev"
            ],
        ),
    ]

    for population, frame in populations:
        for feature in features:
            for target in targets:
                valid = (
                    frame[
                        [
                            feature,
                            target,
                        ]
                    ]
                    .dropna()
                )

                if len(valid) < 3:
                    continue

                result = spearmanr(
                    valid[
                        feature
                    ],
                    valid[
                        target
                    ],
                )

                rows.append(
                    {
                        "population":
                            population,

                        "feature":
                            feature,

                        "target":
                            target,

                        "spearman_r":
                            float(
                                result.statistic
                            ),

                        "p_value":
                            float(
                                result.pvalue
                            ),

                        "n":
                            len(valid),
                    }
                )

    return pd.DataFrame(
        rows
    )


def main():
    checkpoint_sha = (
        verify_checkpoint()
    )

    frame = pd.read_csv(
        INDEX
    ).reset_index(
        drop=True
    )

    frame[
        "row_id"
    ] = np.arange(
        len(frame)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint = (
        build_model(
            device
        )
    )

    print(
        "Frozen text-patch "
        "representation audit:"
        f"\n  checkpoint SHA: "
        f"{checkpoint_sha}"
        f"\n  selected stage: "
        f"{checkpoint['stage']}"
        f"\n  selected epoch: "
        f"{checkpoint['epoch']}"
        f"\n  rows:           "
        f"{len(frame)}"
        f"\n  pairs:          "
        f"{frame['pair_id'].nunique()}"
        f"\n  device:         "
        f"{device}"
    )

    features = (
        extract_features(
            model,
            frame,
            device,
        )
    )

    # Binary softmax decision direction:
    #
    # margin = logit_attack - logit_bonafide
    #
    attack_axis = (
        model.fc.weight[
            1
        ]
        -
        model.fc.weight[
            0
        ]
    ).detach().cpu().numpy()

    attack_bias = float(
        (
            model.fc.bias[
                1
            ]
            -
            model.fc.bias[
                0
            ]
        )
        .detach()
        .cpu()
        .item()
    )

    pairs, deltas = (
        build_pair_table(
            frame,
            features,
            attack_axis,
            attack_bias,
        )
    )

    pairs = (
        add_reference_alignment(
            pairs,
            deltas,
        )
    )

    summary = summarize(
        pairs
    )

    fields = summarize_fields(
        pairs
    )

    correlation = correlations(
        pairs
    )

    pairs.to_csv(
        OUT_PAIRS,
        index=False,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    fields.to_csv(
        OUT_FIELDS,
        index=False,
    )

    correlation.to_csv(
        OUT_CORRELATION,
        index=False,
    )

    print(
        "\nREPRESENTATION SHIFT "
        "— HELD-OUT METHODS:"
    )

    display = [
        "population",
        "n_pairs",
        "n_stems",
        "same_source_auc",
        "mean_feature_shift_norm",
        "mean_attack_axis_projection",
        "median_attack_axis_projection",
        "fraction_positive_projection",
        "mean_attack_axis_cosine",
        "mean_cos_train_global",
        "mean_cos_train_d1",
        "mean_cos_train_d2",
        "mean_cos_train_same_field",
        "mean_gray_mad_inside",
        "mean_changed_inside_ge5",
    ]

    print(
        summary[
            display
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nDIGITAL-3 FIELD "
        "REPRESENTATION BREAKDOWN:"
    )

    d3_fields = fields[
        fields[
            "split"
        ]
        == "digital3_dev"
    ].sort_values(
        "n_pairs",
        ascending=False,
    )

    print(
        d3_fields[
            [
                "field_name",
                "n_pairs",
                "n_stems",
                "same_source_auc",
                "mean_probability_delta",
                "mean_feature_shift_norm",
                "mean_axis_projection",
                "mean_axis_cosine",
                "fraction_positive_projection",
                "mean_cos_train_global",
                "mean_cos_train_same_field",
                "mean_gray_mad_inside",
                "mean_changed_inside_ge5",
                "mean_annotation_coverage",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nDIGITAL-3 CORRELATIONS "
        "WITH CLASSIFIER RESPONSE:"
    )

    d3_corr = correlation[
        correlation[
            "population"
        ]
        == "dev_digital_3"
    ]

    print(
        d3_corr.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\nPairs:       {OUT_PAIRS}"
        f"\nSummary:     {OUT_SUMMARY}"
        f"\nFields:      {OUT_FIELDS}"
        f"\nCorrelations:{OUT_CORRELATION}"
    )

    print(
        "\nDecision guide:"
        "\n"
        "\nA) D3 pixel magnitude << D1/D2"
        "\n   => D3 itself is substantially subtler."
        "\n"
        "\nB) D3 pixel magnitude similar, but "
        "feature_shift_norm << D1/D2"
        "\n   => backbone suppresses / fails to encode "
        "the D3 forensic evidence."
        "\n"
        "\nC) D3 feature_shift_norm substantial, but "
        "attack_axis_projection/cosine << D1/D2"
        "\n   => D3 is represented, but mostly outside "
        "the learned forgery direction: strong evidence "
        "for manipulation-method-specific representation."
        "\n"
        "\nD) D3 name has much stronger axis alignment "
        "than dob/doe despite comparable pixel magnitude"
        "\n   => the name/date gap is representational, "
        "not merely caused by more changed pixels."
    )


if __name__ == "__main__":
    main()