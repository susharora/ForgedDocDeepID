#!/usr/bin/env python3
"""
Final clean Grad-CAM / Relevance-Mass localisation baseline for the frozen
compression-controlled FantasyID ResNet18.

Evaluated populations
---------------------
1. project_train
       digital_1: all 480 attacks
       digital_2: all 480 attacks

   Role:
       training diagnostic only

2. dev_val
       digital_1: all 153 attacks
       digital_2: all 153 attacks

   Role:
       PRIMARY held-out clean localisation baseline

3. official_test
       facedancer:
           ALL 150 attacks

           Role:
               primary unseen-family clean localisation baseline

       textdiffuserft_bfei:
           ONLY images correctly classified as attacks by the frozen clean
           ResNet at the already-fixed threshold p_attack >= 0.5

           Role:
               exploratory clean-correct localisation subset

       digital_3:
           ONLY images correctly classified as attacks by the frozen clean
           ResNet at p_attack >= 0.5

           Role:
               exploratory clean-correct localisation subset

No threshold is selected using localisation results.
No official-test score is used for model fitting.

Frozen model
------------
runs/post_hoc_compression_controlled_resnet18_seed10/checkpoints/best.pt

Grad-CAM
--------
Target layer:

    model.layer4[-1]

Target scalar:

    attack_logit - bonafide_logit

Primary area-aware localisation metrics
---------------------------------------
Let Ω_i be the actual resized document-content support, excluding horizontal
padding.

Let M_i be the UNION of all altered annotation rectangles for image i.

A_i = |M_i| / |Ω_i|

E_i = sum_{p in M_i} CAM_i(p)
      --------------------------------
      sum_{p in Ω_i} CAM_i(p)

mu_w,i = E_i / A_i

PG_i = 1[
    argmax_{p in Ω_i} CAM_i(p) in M_i
]

Interpretation:

    A      GT area fraction
    E      Relevance Mass Accuracy / Energy-Based Pointing Game
    mu_w   area-normalised relevance enrichment
           1.0 = spatially uniform CAM
           >1  = preferential localisation in GT
    PG     conventional pointing game

Also retained:

    E - A
    face/text-specific A/E/mu_w/PG
    global padding energy
    global maximum in padding
    CAM-zero fraction

Dataset reporting
-----------------
For A, E, mu_w and PG:

    mean
    median
    stem-cluster bootstrap 95% CI for mean
    stem-cluster bootstrap 95% CI for median

Area stratification
-------------------
Small / medium / large thresholds are frozen from PROJECT-TRAIN altered-union
area A:

    small:
        A <= train 33.333 percentile

    medium:
        q33 < A <= train 66.667 percentile

    large:
        A > q67

The training-derived thresholds are then applied unchanged to dev and official
test.

Visualisation
-------------
For every evaluated image:

    *__heatmap.png
    *__overlay.png

WHITE:
    visual-only contour around top 10% of positive CAM activation within
    document content

CYAN:
    altered face ground-truth rectangle

MAGENTA:
    altered text ground-truth rectangle

GREY:
    document / artificial-padding boundary

The white contour is NOT used by any quantitative localisation metric.

Ground-truth rectangles come from the frozen FantasyID Regions inventory,
which is the canonical extraction of the accompanying JSON annotations.

Important interpretation
------------------------
TextDiffuserFT and Digital-3 official-test localisation results are conditional
on CLEAN CORRECT CLASSIFICATION.

They are NOT estimates of localisation performance over those complete attack
families.

FaceDancer and held-out d1/d2 do not have that selection caveat.
"""

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
)

import torch
from torch import nn
import torch.nn.functional as F

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


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

PROJECT_INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

OFFICIAL_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

PROJECT_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_predictions.csv"
)

OFFICIAL_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_predictions.csv"
)

INVENTORY = (
    ROOT
    / "output"
    / "fantasyid_inventory_2026-09-05_023126.xlsx"
)

INVENTORY_SHA256 = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
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


# =====================================================================
# Outputs
# =====================================================================

OUT_ROOT = (
    ROOT
    / "output"
    / "resnet18_gradcam_clean_rma"
)

OUT_METRICS = (
    OUT_ROOT
    / "gradcam_rma_metrics.csv"
)

OUT_SUMMARY = (
    OUT_ROOT
    / "gradcam_rma_summary.csv"
)

OUT_AREA_THRESHOLDS = (
    OUT_ROOT
    / "gradcam_rma_area_thresholds.csv"
)

OUT_GENERALISATION = (
    OUT_ROOT
    / "gradcam_rma_train_dev_generalisation.csv"
)

OUT_REGIONS = (
    OUT_ROOT
    / "gradcam_rma_regions.csv"
)

OUT_SELECTION = (
    OUT_ROOT
    / "gradcam_rma_official_selection.csv"
)

OUT_CAMS = (
    OUT_ROOT
    / "gradcam_rma_layer4_maps.npz"
)

VISUAL_ROOT = (
    OUT_ROOT
    / "visuals"
)

REVIEW_ROOT = (
    OUT_ROOT
    / "review_panels"
)


# =====================================================================
# Constants
# =====================================================================

CONTENT_H = 512
CANVAS_W = 864

BATCH_SIZE = 4

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)

SEED = 10
N_BOOT = 5000

CAM_CONTOUR_QUANTILE = 0.90

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


# =====================================================================
# Colours
# =====================================================================

COLOR_FACE = (
    0,
    255,
    255,
)

COLOR_TEXT = (
    255,
    0,
    255,
)

COLOR_CAM = (
    255,
    255,
    255,
)

COLOR_BOUNDARY = (
    160,
    160,
    160,
)

COLOR_BLACK = (
    0,
    0,
    0,
)

COLOR_WHITE = (
    255,
    255,
    255,
)


# =====================================================================
# Generic helpers
# =====================================================================

def require_file(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required file missing:\n{path}"
        )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1 << 20),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def short_hash(text):
    return hashlib.sha256(
        text.encode()
    ).hexdigest()[:8]


def safe_name(value):
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )

    return value.strip("_")


def verify_checkpoint():
    require_file(
        CHECKPOINT
    )

    require_file(
        CHECKPOINT_HASH
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
            "Checkpoint SHA mismatch"
            f"\nexpected: {expected}"
            f"\nactual:   {actual}"
        )

    return actual


def verify_inventory():
    require_file(
        INVENTORY
    )

    actual = sha256_file(
        INVENTORY
    )

    if actual != INVENTORY_SHA256:
        raise RuntimeError(
            "Frozen inventory SHA mismatch"
            f"\nexpected: {INVENTORY_SHA256}"
            f"\nactual:   {actual}"
        )


# =====================================================================
# Frozen preprocessing dataset
# =====================================================================

class PolicyCDataset(Dataset):
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
            / row.cache_path
        )

        with Image.open(
            path
        ) as image:
            image = image.convert(
                "RGB"
            )

            width, height = (
                image.size
            )

            new_width = int(
                round(
                    width
                    * CONTENT_H
                    / height
                )
            )

            if new_width > CANVAS_W:
                raise RuntimeError(
                    "Image exceeds frozen "
                    "512x864 canvas:"
                    f"\n{row.image_path}"
                    f"\nresized width={new_width}"
                )

            image = TF.resize(
                image,
                [
                    CONTENT_H,
                    new_width,
                ],
                interpolation=(
                    InterpolationMode
                    .BILINEAR
                ),
                antialias=True,
            )

            tensor = TF.to_tensor(
                image
            )

        tensor = TF.normalize(
            tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        canvas = torch.zeros(
            (
                3,
                CONTENT_H,
                CANVAS_W,
            ),
            dtype=tensor.dtype,
        )

        left = (
            CANVAS_W
            - new_width
        ) // 2

        canvas[
            :,
            :,
            left:
            left + new_width,
        ] = tensor

        return (
            canvas,
            int(
                row.label
            ),
            index,
        )


# =====================================================================
# Model
# =====================================================================

def load_model(device):
    try:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT,
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

    return (
        model,
        checkpoint,
    )


# =====================================================================
# Grad-CAM
# =====================================================================

class GradCAM:
    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model

        self.activations = None
        self.gradients = None

        self.handle = (
            target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

    def _save_gradient(
        self,
        gradient,
    ):
        self.gradients = gradient

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

        output.register_hook(
            self._save_gradient
        )

    def remove(self):
        self.handle.remove()

    def generate(
        self,
        x,
    ):
        self.activations = None
        self.gradients = None

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            x
        )

        margin = (
            logits[:, 1]
            -
            logits[:, 0]
        )

        margin.sum().backward()

        if (
            self.activations
            is None
            or self.gradients
            is None
        ):
            raise RuntimeError(
                "Grad-CAM hook failed"
            )

        weights = (
            self.gradients.mean(
                dim=(
                    2,
                    3,
                ),
                keepdim=True,
            )
        )

        low_cam = (
            weights
            * self.activations
        ).sum(
            dim=1
        )

        low_cam = torch.relu(
            low_cam
        )

        maximum = (
            low_cam
            .flatten(1)
            .amax(
                dim=1
            )
            .view(
                -1,
                1,
                1,
            )
        )

        low_cam = torch.where(
            maximum > 1e-12,
            low_cam
            /
            torch.clamp(
                maximum,
                min=1e-12,
            ),
            torch.zeros_like(
                low_cam
            ),
        )

        full_cam = (
            F.interpolate(
                low_cam.unsqueeze(1),
                size=(
                    CONTENT_H,
                    CANVAS_W,
                ),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
        )

        full_cam = torch.clamp(
            full_cam,
            min=0.0,
        )

        probability = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        )

        return {
            "probability":
                probability.detach(),

            "margin":
                margin.detach(),

            "low_cam":
                low_cam.detach(),

            "full_cam":
                full_cam.detach(),
        }


# =====================================================================
# Evaluation population
# =====================================================================

def load_prediction_lookup(
    path,
):
    require_file(
        path
    )

    frame = pd.read_csv(
        path,
        keep_default_na=False,
    )

    if (
        frame[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Duplicate prediction paths: "
            f"{path}"
        )

    return frame[
        [
            "image_path",
            "attack_probability",
        ]
    ].copy()


def load_evaluation_population():
    require_file(
        PROJECT_INDEX
    )

    require_file(
        OFFICIAL_INDEX
    )

    project = pd.read_csv(
        PROJECT_INDEX,
        keep_default_na=False,
    )

    official = pd.read_csv(
        OFFICIAL_INDEX,
        keep_default_na=False,
    )

    # --------------------------------------------------
    # Project train + dev attacks
    # --------------------------------------------------

    project_attacks = project[
        (
            project[
                "traffic_type"
            ]
            == "attack"
        )
        &
        (
            project[
                "split"
            ].isin(
                [
                    "project_train",
                    "dev_val",
                ]
            )
        )
    ].copy()

    project_attacks[
        "evaluation_split"
    ] = project_attacks[
        "split"
    ]

    project_attacks[
        "selection_scope"
    ] = "all_attacks"

    project_attacks[
        "evidence_role"
    ] = np.where(
        project_attacks[
            "split"
        ]
        == "project_train",
        "training_diagnostic",
        "heldout_primary",
    )

    project_predictions = (
        load_prediction_lookup(
            PROJECT_PREDICTIONS
        )
    )

    project_attacks = (
        project_attacks.merge(
            project_predictions,
            on="image_path",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "attack_probability":
                    "saved_attack_probability",
            }
        )
    )

    if (
        project_attacks[
            "saved_attack_probability"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Project prediction join failed"
        )

    project_counts = (
        project_attacks.groupby(
            [
                "evaluation_split",
                "variant",
            ]
        )
        .size()
        .to_dict()
    )

    expected_project = {
        (
            "project_train",
            "digital_1",
        ):
            480,

        (
            "project_train",
            "digital_2",
        ):
            480,

        (
            "dev_val",
            "digital_1",
        ):
            153,

        (
            "dev_val",
            "digital_2",
        ):
            153,
    }

    if (
        project_counts
        != expected_project
    ):
        raise RuntimeError(
            "Unexpected project population"
            f"\nexpected={expected_project}"
            f"\nactual={project_counts}"
        )

    # --------------------------------------------------
    # Official test attacks
    # --------------------------------------------------

    target_variants = {
        "facedancer",
        "textdiffuserft_bfei",
        "digital_3",
    }

    official_attacks = official[
        (
            official[
                "traffic_type"
            ]
            == "attack"
        )
        &
        (
            official[
                "variant"
            ].isin(
                target_variants
            )
        )
    ].copy()

    official_predictions = (
        load_prediction_lookup(
            OFFICIAL_PREDICTIONS
        )
    )

    official_attacks = (
        official_attacks.merge(
            official_predictions,
            on="image_path",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "attack_probability":
                    "saved_attack_probability",
            }
        )
    )

    if (
        official_attacks[
            "saved_attack_probability"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Official prediction join failed"
        )

    expected_official = {
        "digital_3":
            786,

        "facedancer":
            150,

        "textdiffuserft_bfei":
            149,
    }

    actual_official = (
        official_attacks[
            "variant"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        actual_official
        != expected_official
    ):
        raise RuntimeError(
            "Official family counts changed"
            f"\nexpected={expected_official}"
            f"\nactual={actual_official}"
        )

    official_attacks[
        "clean_correct_at_0_5"
    ] = (
        official_attacks[
            "saved_attack_probability"
        ]
        >= 0.5
    )

    selected = (
        (
            official_attacks[
                "variant"
            ]
            == "facedancer"
        )
        |
        (
            (
                official_attacks[
                    "variant"
                ]
                .isin(
                    [
                        "digital_3",
                        "textdiffuserft_bfei",
                    ]
                )
            )
            &
            (
                official_attacks[
                    "clean_correct_at_0_5"
                ]
            )
        )
    )

    official_attacks[
        "selected_for_gradcam"
    ] = selected

    official_attacks[
        "selection_scope"
    ] = np.where(
        official_attacks[
            "variant"
        ]
        == "facedancer",
        "all_family",
        "clean_correct_at_0.5",
    )

    official_attacks[
        "evidence_role"
    ] = np.where(
        official_attacks[
            "variant"
        ]
        == "facedancer",
        "unseen_family_primary",
        "exploratory_clean_correct",
    )

    selection_manifest = (
        official_attacks[
            [
                "image_path",
                "file_stem",
                "variant",
                "hardware_source",
                "label",
                "saved_attack_probability",
                "clean_correct_at_0_5",
                "selected_for_gradcam",
                "selection_scope",
                "evidence_role",
            ]
        ]
        .copy()
    )

    official_selected = (
        official_attacks[
            official_attacks[
                "selected_for_gradcam"
            ]
        ]
        .copy()
    )

    official_selected[
        "evaluation_split"
    ] = "official_test"

    # --------------------------------------------------
    # Align columns
    # --------------------------------------------------

    shared_columns = [
        "image_path",
        "cache_path",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
        "assigned_q",
        "evaluation_split",
        "selection_scope",
        "evidence_role",
        "saved_attack_probability",
    ]

    project_selected = (
        project_attacks[
            shared_columns
        ]
        .copy()
    )

    official_selected = (
        official_selected[
            shared_columns
        ]
        .copy()
    )

    combined = pd.concat(
        [
            project_selected,
            official_selected,
        ],
        ignore_index=True,
    )

    combined = (
        combined.sort_values(
            [
                "evaluation_split",
                "variant",
                "file_stem",
                "hardware_source",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        combined,
        selection_manifest,
    )


# =====================================================================
# Regions
# =====================================================================

def load_regions():
    verify_inventory()

    regions = pd.read_excel(
        INVENTORY,
        sheet_name="Regions",
    )

    required = {
        "image_path",
        "field_name",
        "region_provenance_raw",
        "x",
        "y",
        "width",
        "height",
    }

    missing = (
        required
        - set(
            regions.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Regions schema changed: "
            f"{sorted(missing)}"
        )

    regions = (
        regions.reset_index()
        .rename(
            columns={
                "index":
                    "inventory_row",
            }
        )
    )

    regions[
        "provenance_norm"
    ] = (
        regions[
            "region_provenance_raw"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    regions[
        "field_norm"
    ] = (
        regions[
            "field_name"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return regions


def native_box(
    row,
):
    x0 = int(
        round(
            float(row.x)
        )
    )

    y0 = int(
        round(
            float(row.y)
        )
    )

    width = int(
        round(
            float(row.width)
        )
    )

    height = int(
        round(
            float(row.height)
        )
    )

    if (
        width <= 0
        or height <= 0
    ):
        raise RuntimeError(
            "Invalid rectangle"
        )

    return (
        x0,
        y0,
        x0 + width,
        y0 + height,
    )


def project_box(
    box,
    native_width,
    native_height,
    content_width,
    pad_left,
):
    x0, y0, x1, y1 = box

    cx0 = max(
        0,
        min(
            native_width,
            x0,
        ),
    )

    cy0 = max(
        0,
        min(
            native_height,
            y0,
        ),
    )

    cx1 = max(
        0,
        min(
            native_width,
            x1,
        ),
    )

    cy1 = max(
        0,
        min(
            native_height,
            y1,
        ),
    )

    clipped_native = (
        cx0,
        cy0,
        cx1,
        cy1,
    )

    clipped = (
        clipped_native
        != box
    )

    if (
        cx1 <= cx0
        or cy1 <= cy0
    ):
        raise RuntimeError(
            "GT box outside image"
        )

    scale_x = (
        content_width
        / native_width
    )

    scale_y = (
        CONTENT_H
        / native_height
    )

    if (
        abs(
            scale_x
            - scale_y
        )
        > 0.002
    ):
        raise RuntimeError(
            "Unexpected anisotropic resize"
        )

    mx0 = (
        pad_left
        +
        int(
            np.floor(
                cx0
                * scale_x
            )
        )
    )

    my0 = int(
        np.floor(
            cy0
            * scale_y
        )
    )
    
    mx1 = (
        pad_left
        +
        int(
            np.ceil(
                cx1
                * scale_x
            )
        )
    )

    #fix_m0
    content_x0 = pad_left
    content_x1 = pad_left + content_width

    mx0 = max(content_x0, min(content_x1, mx0))
    mx1 = max(content_x0, min(content_x1, mx1))


    my1 = int(
        np.ceil(
            cy1
            * scale_y
        )
    )

    mx0 = max(
        0,
        min(
            CANVAS_W,
            mx0,
        ),
    )

    mx1 = max(
        0,
        min(
            CANVAS_W,
            mx1,
        ),
    )

    my0 = max(
        0,
        min(
            CONTENT_H,
            my0,
        ),
    )

    my1 = max(
        0,
        min(
            CONTENT_H,
            my1,
        ),
    )

    model_box = (
        mx0,
        my0,
        mx1,
        my1,
    )

    if (
        mx1 <= mx0
        or my1 <= my0
    ):
        raise RuntimeError(
            "Projected GT box empty"
        )

    return (
        clipped_native,
        model_box,
        clipped,
    )


def prepare_annotation_geometry(
    frame,
    regions,
):
    altered = (
        regions[
            regions[
                "provenance_norm"
            ]
            == "altered"
        ]
        .copy()
    )

    grouped = {
        image_path:
            group.copy()
        for image_path, group
        in altered.groupby(
            "image_path"
        )
    }

    image_infos = []

    region_records = []

    clipped_count = 0

    for row in frame.itertuples(
        index=False
    ):
        if (
            row.image_path
            not in grouped
        ):
            raise RuntimeError(
                "No altered annotation:"
                f"\n{row.image_path}"
            )

        cache_path = (
            ROOT
            / row.cache_path
        )

        with Image.open(
            cache_path
        ) as image:
            native_width, native_height = (
                image.size
            )

        content_width = int(
            round(
                native_width
                * CONTENT_H
                / native_height
            )
        )

        if (
            content_width
            > CANVAS_W
        ):
            raise RuntimeError(
                "Image exceeds canvas"
            )

        pad_left = (
            CANVAS_W
            - content_width
        ) // 2

        pad_right = (
            CANVAS_W
            - pad_left
            - content_width
        )

        face_rects = []
        text_rects = []
        visual_regions = []

        for region in (
            grouped[
                row.image_path
            ]
            .itertuples(
                index=False
            )
        ):
            original = native_box(
                region
            )

            (
                clipped_native,
                model_box,
                clipped,
            ) = project_box(
                original,
                native_width,
                native_height,
                content_width,
                pad_left,
            )

            clipped_count += int(
                clipped
            )

            semantic_type = (
                "face"
                if (
                    region.field_norm
                    == "face"
                )
                else "text"
            )

            if (
                semantic_type
                == "face"
            ):
                face_rects.append(
                    model_box
                )

            else:
                text_rects.append(
                    model_box
                )

            visual_regions.append(
                {
                    "field_name":
                        str(
                            region.field_name
                        ),

                    "semantic_type":
                        semantic_type,

                    "model_box":
                        model_box,

                    "clipped":
                        bool(
                            clipped
                        ),
                }
            )

            region_records.append(
                {
                    "image_path":
                        row.image_path,

                    "evaluation_split":
                        row.evaluation_split,

                    "selection_scope":
                        row.selection_scope,

                    "variant":
                        row.variant,

                    "file_stem":
                        row.file_stem,

                    "hardware_source":
                        row.hardware_source,

                    "inventory_row":
                        int(
                            region.inventory_row
                        ),

                    "field_name":
                        str(
                            region.field_name
                        ),

                    "semantic_type":
                        semantic_type,

                    "region_provenance":
                        "altered",

                    "native_width":
                        native_width,

                    "native_height":
                        native_height,

                    "original_x0":
                        original[0],

                    "original_y0":
                        original[1],

                    "original_x1":
                        original[2],

                    "original_y1":
                        original[3],

                    "clipped_x0":
                        clipped_native[0],

                    "clipped_y0":
                        clipped_native[1],

                    "clipped_x1":
                        clipped_native[2],

                    "clipped_y1":
                        clipped_native[3],

                    "annotation_clipped":
                        int(
                            clipped
                        ),

                    "content_width_r512":
                        content_width,

                    "pad_left_r512":
                        pad_left,

                    "pad_right_r512":
                        pad_right,

                    "model_x0":
                        model_box[0],

                    "model_y0":
                        model_box[1],

                    "model_x1":
                        model_box[2],

                    "model_y1":
                        model_box[3],
                }
            )

        if (
            not face_rects
            and not text_rects
        ):
            raise RuntimeError(
                "Image has no usable altered "
                f"regions: {row.image_path}"
            )

        image_infos.append(
            {
                "native_width":
                    native_width,

                "native_height":
                    native_height,

                "content_width":
                    content_width,

                "pad_left":
                    pad_left,

                "pad_right":
                    pad_right,

                "face_rects":
                    face_rects,

                "text_rects":
                    text_rects,

                "visual_regions":
                    visual_regions,
            }
        )

    return (
        image_infos,
        pd.DataFrame(
            region_records
        ),
        clipped_count,
    )


# =====================================================================
# Masks
# =====================================================================

def mask_from_rectangles(
    rectangles,
):
    mask = np.zeros(
        (
            CONTENT_H,
            CANVAS_W,
        ),
        dtype=bool,
    )

    for (
        x0,
        y0,
        x1,
        y1,
    ) in rectangles:
        mask[
            y0:y1,
            x0:x1,
        ] = True

    return mask


def content_mask(
    info,
):
    mask = np.zeros(
        (
            CONTENT_H,
            CANVAS_W,
        ),
        dtype=bool,
    )

    x0 = (
        info[
            "pad_left"
        ]
    )

    x1 = (
        x0
        +
        info[
            "content_width"
        ]
    )

    mask[
        :,
        x0:x1,
    ] = True

    return mask


# =====================================================================
# RMA / area-aware metrics
# =====================================================================

def semantic_metrics(
    cam,
    mask,
    content,
    content_energy,
    content_max_position,
):
    if (
        mask.sum()
        == 0
    ):
        return {
            "A":
                np.nan,

            "E":
                np.nan,

            "mu_w":
                np.nan,

            "PG":
                np.nan,
        }

    area = float(
        mask.sum()
        / content.sum()
    )

    if (
        content_energy
        <= 1e-12
    ):
        energy = 0.0

    else:
        energy = float(
            cam[
                mask
            ].sum()
            / content_energy
        )

    mu_w = (
        energy
        / area
        if area > 0
        else np.nan
    )

    if (
        content_max_position
        is None
    ):
        pointing = 0.0

    else:
        max_y, max_x = (
            content_max_position
        )

        pointing = float(
            mask[
                max_y,
                max_x,
            ]
        )

    return {
        "A":
            area,

        "E":
            energy,

        "mu_w":
            mu_w,

        "PG":
            pointing,
    }


def compute_rma_metrics(
    cam,
    info,
):
    face = mask_from_rectangles(
        info[
            "face_rects"
        ]
    )

    text = mask_from_rectangles(
        info[
            "text_rects"
        ]
    )

    union = (
        face
        | text
    )

    content = content_mask(
        info
    )

    padding = (
        ~content
    )

    if (
        union.sum()
        == 0
    ):
        raise RuntimeError(
            "Empty GT union"
        )

    content_energy = float(
        cam[
            content
        ].sum()
    )

    total_energy = float(
        cam.sum()
    )

    if (
        content_energy
        <= 1e-12
    ):
        content_max_position = None

    else:
        content_only = np.where(
            content,
            cam,
            -np.inf,
        )

        flat_index = int(
            np.argmax(
                content_only
            )
        )

        content_max_position = (
            np.unravel_index(
                flat_index,
                cam.shape,
            )
        )

    union_metrics = (
        semantic_metrics(
            cam,
            union,
            content,
            content_energy,
            content_max_position,
        )
    )

    face_metrics = (
        semantic_metrics(
            cam,
            face,
            content,
            content_energy,
            content_max_position,
        )
    )

    text_metrics = (
        semantic_metrics(
            cam,
            text,
            content,
            content_energy,
            content_max_position,
        )
    )

    # --------------------------------------------------
    # Full canvas diagnostic
    # --------------------------------------------------

    if (
        total_energy
        <= 1e-12
    ):
        padding_energy = 0.0
        pointing_padding = 0.0
        global_x = -1
        global_y = -1

    else:
        padding_energy = float(
            cam[
                padding
            ].sum()
            / total_energy
        )

        global_flat = int(
            np.argmax(
                cam
            )
        )

        global_y, global_x = (
            np.unravel_index(
                global_flat,
                cam.shape,
            )
        )

        pointing_padding = float(
            padding[
                global_y,
                global_x,
            ]
        )

    face_absolute = float(
        cam[
            face
        ].sum()
    )

    text_absolute = float(
        cam[
            text
        ].sum()
    )

    altered_absolute = (
        face_absolute
        +
        text_absolute
    )

    if (
        altered_absolute
        > 1e-12
    ):
        face_share = (
            face_absolute
            / altered_absolute
        )

        text_share = (
            text_absolute
            / altered_absolute
        )

    else:
        face_share = np.nan
        text_share = np.nan

    return {
        # Primary quartet.
        "rma_A":
            union_metrics[
                "A"
            ],

        "rma_E":
            union_metrics[
                "E"
            ],

        "rma_mu_w":
            union_metrics[
                "mu_w"
            ],

        "rma_PG":
            union_metrics[
                "PG"
            ],

        "rma_energy_gain":
            (
                union_metrics[
                    "E"
                ]
                -
                union_metrics[
                    "A"
                ]
            ),

        # Face.
        "face_A":
            face_metrics[
                "A"
            ],

        "face_E":
            face_metrics[
                "E"
            ],

        "face_mu_w":
            face_metrics[
                "mu_w"
            ],

        "face_PG":
            face_metrics[
                "PG"
            ],

        # Text.
        "text_A":
            text_metrics[
                "A"
            ],

        "text_E":
            text_metrics[
                "E"
            ],

        "text_mu_w":
            text_metrics[
                "mu_w"
            ],

        "text_PG":
            text_metrics[
                "PG"
            ],

        # Cue balance.
        "face_share_altered_energy":
            face_share,

        "text_share_altered_energy":
            text_share,

        # Padding / full canvas.
        "padding_area_canvas":
            float(
                padding.mean()
            ),

        "padding_energy_canvas":
            padding_energy,

        "pointing_padding_canvas":
            pointing_padding,

        "content_energy_fraction_canvas":
            (
                float(
                    content_energy
                    / total_energy
                )
                if total_energy > 1e-12
                else 0.0
            ),

        "cam_zero_content":
            int(
                content_energy
                <= 1e-12
            ),

        "content_max_x":
            (
                int(
                    content_max_position[
                        1
                    ]
                )
                if (
                    content_max_position
                    is not None
                )
                else -1
            ),

        "content_max_y":
            (
                int(
                    content_max_position[
                        0
                    ]
                )
                if (
                    content_max_position
                    is not None
                )
                else -1
            ),

        "global_max_x":
            int(
                global_x
            ),

        "global_max_y":
            int(
                global_y
            ),
    }


# =====================================================================
# Visual contour
# =====================================================================

def make_cam_contour(
    cam,
    info,
):
    content = content_mask(
        info
    )

    positive = (
        content
        &
        (
            cam > 0
        )
    )

    values = cam[
        positive
    ]

    if len(values) == 0:
        return (
            np.zeros_like(
                content,
                dtype=bool,
            ),
            np.nan,
            0.0,
        )

    threshold = float(
        np.quantile(
            values,
            CAM_CONTOUR_QUANTILE,
        )
    )

    active = (
        content
        &
        (
            cam >= threshold
        )
        &
        (
            cam > 0
        )
    )

    eroded = binary_erosion(
        active,
        iterations=1,
        border_value=0,
    )

    contour = (
        active
        &
        ~eroded
    )

    contour = binary_dilation(
        contour,
        iterations=1,
    )

    return (
        contour,
        threshold,
        float(
            active.sum()
            / content.sum()
        ),
    )


# =====================================================================
# Rendering
# =====================================================================

def canvas_to_rgb(
    tensor,
):
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
    ).view(
        3,
        1,
        1,
    )

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
    ).view(
        3,
        1,
        1,
    )

    rgb = (
        tensor.cpu()
        * std
        + mean
    )

    rgb = (
        rgb.clamp(
            0.0,
            1.0,
        )
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )

    return (
        rgb
        * 255.0
    ).round().astype(
        np.uint8
    )


def heat_rgb(
    cam,
):
    value = np.clip(
        cam,
        0.0,
        1.0,
    )

    red = (
        255.0
        * np.sqrt(
            value
        )
    )

    green = (
        255.0
        * value
    )

    blue = np.zeros_like(
        value
    )

    return np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=2,
    ).round().astype(
        np.uint8
    )


def blend_overlay(
    base,
    cam,
):
    heat = heat_rgb(
        cam
    ).astype(
        np.float32
    )

    base = base.astype(
        np.float32
    )

    alpha = (
        0.62
        * np.clip(
            cam,
            0.0,
            1.0,
        )[
            :,
            :,
            None,
        ]
    )

    result = (
        base
        * (
            1.0
            - alpha
        )
        +
        heat
        * alpha
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


def apply_contour(
    image,
    contour,
):
    result = image.copy()

    result[
        contour
    ] = np.asarray(
        COLOR_CAM,
        dtype=np.uint8,
    )

    return result


def default_font():
    return ImageFont.load_default()


def draw_label(
    draw,
    x,
    y,
    text,
    color,
):
    font = default_font()

    bbox = draw.textbbox(
        (
            x,
            y,
        ),
        str(text),
        font=font,
    )

    draw.rectangle(
        (
            bbox[0] - 2,
            bbox[1] - 1,
            bbox[2] + 2,
            bbox[3] + 1,
        ),
        fill=COLOR_BLACK,
    )

    draw.text(
        (
            x,
            y,
        ),
        str(text),
        fill=color,
        font=font,
    )


def draw_gt(
    image,
    info,
):
    draw = ImageDraw.Draw(
        image
    )

    left = (
        info[
            "pad_left"
        ]
    )

    right = (
        left
        +
        info[
            "content_width"
        ]
        - 1
    )

    draw.line(
        (
            left,
            0,
            left,
            CONTENT_H - 1,
        ),
        fill=COLOR_BOUNDARY,
        width=1,
    )

    draw.line(
        (
            right,
            0,
            right,
            CONTENT_H - 1,
        ),
        fill=COLOR_BOUNDARY,
        width=1,
    )

    for region in (
        info[
            "visual_regions"
        ]
    ):
        x0, y0, x1, y1 = (
            region[
                "model_box"
            ]
        )

        color = (
            COLOR_FACE
            if (
                region[
                    "semantic_type"
                ]
                == "face"
            )
            else COLOR_TEXT
        )

        draw.rectangle(
            (
                x0,
                y0,
                max(
                    x0,
                    x1 - 1,
                ),
                max(
                    y0,
                    y1 - 1,
                ),
            ),
            outline=color,
            width=3,
        )

        label = (
            region[
                "field_name"
            ]
        )

        if (
            region[
                "clipped"
            ]
        ):
            label += " *"

        draw_label(
            draw,
            x0 + 2,
            max(
                1,
                y0 - 13,
            ),
            label,
            color,
        )

    return image


def draw_legend(
    image,
):
    draw = ImageDraw.Draw(
        image
    )

    font = default_font()

    entries = [
        (
            COLOR_CAM,
            "CAM top-positive contour",
        ),
        (
            COLOR_FACE,
            "altered face GT",
        ),
        (
            COLOR_TEXT,
            "altered text GT",
        ),
        (
            COLOR_BOUNDARY,
            "document boundary",
        ),
    ]

    x = 8
    y = 8

    line_h = 15

    draw.rectangle(
        (
            4,
            4,
            198,
            8
            +
            line_h
            * len(
                entries
            ),
        ),
        fill=COLOR_BLACK,
    )

    for index, (
        color,
        label,
    ) in enumerate(
        entries
    ):
        yy = (
            y
            +
            index
            * line_h
        )

        draw.line(
            (
                x,
                yy + 6,
                x + 20,
                yy + 6,
            ),
            fill=color,
            width=3,
        )

        draw.text(
            (
                x + 26,
                yy,
            ),
            label,
            fill=COLOR_WHITE,
            font=font,
        )

    return image


def render_images(
    tensor,
    cam,
    info,
):
    base = canvas_to_rgb(
        tensor
    )

    (
        contour,
        threshold,
        contour_area,
    ) = make_cam_contour(
        cam,
        info,
    )

    heat = apply_contour(
        heat_rgb(
            cam
        ),
        contour,
    )

    overlay = apply_contour(
        blend_overlay(
            base,
            cam,
        ),
        contour,
    )

    input_gt = draw_gt(
        Image.fromarray(
            base.copy()
        ),
        info,
    )

    heatmap = draw_gt(
        Image.fromarray(
            heat
        ),
        info,
    )

    overlay = draw_gt(
        Image.fromarray(
            overlay
        ),
        info,
    )

    input_gt = draw_legend(
        input_gt
    )

    heatmap = draw_legend(
        heatmap
    )

    overlay = draw_legend(
        overlay
    )

    return {
        "input_gt":
            input_gt,

        "heatmap":
            heatmap,

        "overlay":
            overlay,

        "contour_threshold":
            threshold,

        "contour_area":
            contour_area,
    }


def add_panel_header(
    image,
    title,
):
    header = 24

    output = Image.new(
        "RGB",
        (
            image.width,
            image.height + header,
        ),
        COLOR_BLACK,
    )

    output.paste(
        image,
        (
            0,
            header,
        ),
    )

    draw = ImageDraw.Draw(
        output
    )

    draw.text(
        (
            8,
            6,
        ),
        title,
        fill=COLOR_WHITE,
        font=default_font(),
    )

    return output


def review_panel(
    input_gt,
    heatmap,
    overlay,
):
    panels = [
        add_panel_header(
            input_gt,
            "INPUT + GT",
        ),
        add_panel_header(
            heatmap,
            "GRAD-CAM + GT",
        ),
        add_panel_header(
            overlay,
            "OVERLAY",
        ),
    ]

    width = sum(
        panel.width
        for panel
        in panels
    )

    height = max(
        panel.height
        for panel
        in panels
    )

    output = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        COLOR_BLACK,
    )

    x = 0

    for panel in panels:
        output.paste(
            panel,
            (
                x,
                0,
            ),
        )

        x += panel.width

    return output


# =====================================================================
# Reviewer subset
# =====================================================================

def choose_review_paths(
    frame,
):
    selected = set()

    for (
        evaluation_split,
        variant,
        selection_scope,
        hardware,
    ), group in (
        frame.groupby(
            [
                "evaluation_split",
                "variant",
                "selection_scope",
                "hardware_source",
            ]
        )
    ):
        subset = (
            group.sort_values(
                [
                    "file_stem",
                    "image_path",
                ]
            )
            .head(2)
        )

        selected.update(
            subset[
                "image_path"
            ]
        )

    return selected


# =====================================================================
# Cluster bootstrap
# =====================================================================

def cluster_groups(
    frame,
    column,
):
    subset = (
        frame[
            [
                "file_stem",
                column,
            ]
        ]
        .dropna()
    )

    return {
        stem:
            group[
                column
            ]
            .to_numpy(
                dtype=float
            )

        for stem, group
        in subset.groupby(
            "file_stem"
        )
    }


def bootstrap_stat_ci(
    frame,
    column,
    statistic,
):
    groups = cluster_groups(
        frame,
        column,
    )

    if not groups:
        return (
            np.nan,
            np.nan,
        )

    stems = np.array(
        list(
            groups.keys()
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    values = np.empty(
        N_BOOT,
        dtype=float,
    )

    for bootstrap_index in range(
        N_BOOT
    ):
        sampled = rng.choice(
            stems,
            size=len(stems),
            replace=True,
        )

        data = np.concatenate(
            [
                groups[
                    stem
                ]
                for stem
                in sampled
            ]
        )

        values[
            bootstrap_index
        ] = statistic(
            data
        )

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
            )
        ),
    )


def metric_summary(
    frame,
    column,
    prefix,
):
    values = (
        frame[
            column
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(values) == 0:
        return {
            f"{prefix}_mean":
                np.nan,

            f"{prefix}_median":
                np.nan,

            f"{prefix}_mean_ci_low":
                np.nan,

            f"{prefix}_mean_ci_high":
                np.nan,

            f"{prefix}_median_ci_low":
                np.nan,

            f"{prefix}_median_ci_high":
                np.nan,
        }

    mean_ci = (
        bootstrap_stat_ci(
            frame,
            column,
            np.mean,
        )
    )

    median_ci = (
        bootstrap_stat_ci(
            frame,
            column,
            np.median,
        )
    )

    return {
        f"{prefix}_mean":
            float(
                np.mean(
                    values
                )
            ),

        f"{prefix}_median":
            float(
                np.median(
                    values
                )
            ),

        f"{prefix}_mean_ci_low":
            mean_ci[0],

        f"{prefix}_mean_ci_high":
            mean_ci[1],

        f"{prefix}_median_ci_low":
            median_ci[0],

        f"{prefix}_median_ci_high":
            median_ci[1],
    }


def bootstrap_difference_ci(
    train,
    dev,
    column,
):
    train_groups = (
        cluster_groups(
            train,
            column,
        )
    )

    dev_groups = (
        cluster_groups(
            dev,
            column,
        )
    )

    train_stems = np.array(
        list(
            train_groups.keys()
        )
    )

    dev_stems = np.array(
        list(
            dev_groups.keys()
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    values = np.empty(
        N_BOOT,
        dtype=float,
    )

    for index in range(
        N_BOOT
    ):
        train_sample = (
            rng.choice(
                train_stems,
                size=len(
                    train_stems
                ),
                replace=True,
            )
        )

        dev_sample = (
            rng.choice(
                dev_stems,
                size=len(
                    dev_stems
                ),
                replace=True,
            )
        )

        train_values = np.concatenate(
            [
                train_groups[
                    stem
                ]
                for stem
                in train_sample
            ]
        )

        dev_values = np.concatenate(
            [
                dev_groups[
                    stem
                ]
                for stem
                in dev_sample
            ]
        )

        values[
            index
        ] = (
            dev_values.mean()
            -
            train_values.mean()
        )

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
            )
        ),
    )


# =====================================================================
# Area bins
# =====================================================================

def assign_area_bins(
    metrics,
):
    train = metrics[
        metrics[
            "evaluation_split"
        ]
        == "project_train"
    ]

    if len(train) != 960:
        raise RuntimeError(
            "Area-bin source must contain "
            "960 project-train attacks"
        )

    q33, q67 = np.quantile(
        train[
            "rma_A"
        ],
        [
            1.0 / 3.0,
            2.0 / 3.0,
        ],
    )

    def classify(value):
        if value <= q33:
            return "small"

        if value <= q67:
            return "medium"

        return "large"

    metrics = metrics.copy()

    metrics[
        "area_bin"
    ] = (
        metrics[
            "rma_A"
        ]
        .map(
            classify
        )
    )

    thresholds = pd.DataFrame(
        [
            {
                "source":
                    "project_train_d1_d2",

                "n_images":
                    len(train),

                "metric":
                    "rma_A",

                "small_max":
                    float(q33),

                "medium_max":
                    float(q67),

                "large_rule":
                    "A > medium_max",
            }
        ]
    )

    return (
        metrics,
        thresholds,
    )


# =====================================================================
# Summary tables
# =====================================================================

def evidence_role_for_group(
    frame,
):
    values = (
        frame[
            "evidence_role"
        ]
        .unique()
    )

    if len(values) != 1:
        return "mixed"

    return values[0]


def summary_row(
    frame,
    area_bin,
):
    row = {
        "evaluation_split":
            frame[
                "evaluation_split"
            ].iloc[0],

        "variant":
            frame[
                "variant"
            ].iloc[0],

        "selection_scope":
            frame[
                "selection_scope"
            ].iloc[0],

        "evidence_role":
            evidence_role_for_group(
                frame
            ),

        "area_bin":
            area_bin,

        "n_images":
            len(frame),

        "n_stems":
            frame[
                "file_stem"
            ].nunique(),

        "attack_probability_mean":
            frame[
                "attack_probability"
            ].mean(),

        "attack_probability_median":
            frame[
                "attack_probability"
            ].median(),

        "attack_recall_at_0_5":
            frame[
                "predicted_attack"
            ].mean(),

        "zero_cam_fraction":
            frame[
                "cam_zero_content"
            ].mean(),

        "padding_energy_mean":
            frame[
                "padding_energy_canvas"
            ].mean(),

        "pointing_padding":
            frame[
                "pointing_padding_canvas"
            ].mean(),

        "face_share_altered_energy_mean":
            frame[
                "face_share_altered_energy"
            ].mean(),

        "text_share_altered_energy_mean":
            frame[
                "text_share_altered_energy"
            ].mean(),
    }

    row.update(
        metric_summary(
            frame,
            "rma_A",
            "A",
        )
    )

    row.update(
        metric_summary(
            frame,
            "rma_E",
            "E",
        )
    )

    row.update(
        metric_summary(
            frame,
            "rma_mu_w",
            "mu_w",
        )
    )

    row.update(
        metric_summary(
            frame,
            "rma_PG",
            "PG",
        )
    )

    row.update(
        metric_summary(
            frame,
            "rma_energy_gain",
            "energy_gain",
        )
    )

    # Semantic-specific reporting.
    for semantic in [
        "face",
        "text",
    ]:
        for metric in [
            "A",
            "E",
            "mu_w",
            "PG",
        ]:
            column = (
                f"{semantic}_"
                f"{metric}"
            )

            row.update(
                metric_summary(
                    frame,
                    column,
                    column,
                )
            )

    return row


def make_summary(
    metrics,
):
    rows = []

    population_columns = [
        "evaluation_split",
        "variant",
        "selection_scope",
    ]

    for _, population in (
        metrics.groupby(
            population_columns,
            sort=False,
        )
    ):
        rows.append(
            summary_row(
                population,
                "all",
            )
        )

        for area_bin in [
            "small",
            "medium",
            "large",
        ]:
            subset = population[
                population[
                    "area_bin"
                ]
                == area_bin
            ]

            if len(subset) == 0:
                continue

            rows.append(
                summary_row(
                    subset,
                    area_bin,
                )
            )

    return pd.DataFrame(
        rows
    )


def make_train_dev_generalisation(
    metrics,
):
    rows = []

    for variant in [
        "digital_1",
        "digital_2",
    ]:
        train = metrics[
            (
                metrics[
                    "evaluation_split"
                ]
                == "project_train"
            )
            &
            (
                metrics[
                    "variant"
                ]
                == variant
            )
        ]

        dev = metrics[
            (
                metrics[
                    "evaluation_split"
                ]
                == "dev_val"
            )
            &
            (
                metrics[
                    "variant"
                ]
                == variant
            )
        ]

        row = {
            "variant":
                variant,

            "n_train":
                len(train),

            "n_dev":
                len(dev),

            "train_stems":
                train[
                    "file_stem"
                ].nunique(),

            "dev_stems":
                dev[
                    "file_stem"
                ].nunique(),
        }

        for column in [
            "rma_A",
            "rma_E",
            "rma_mu_w",
            "rma_PG",
            "rma_energy_gain",
            "face_share_altered_energy",
            "padding_energy_canvas",
        ]:
            train_mean = float(
                train[
                    column
                ]
                .dropna()
                .mean()
            )

            dev_mean = float(
                dev[
                    column
                ]
                .dropna()
                .mean()
            )

            ci_low, ci_high = (
                bootstrap_difference_ci(
                    train,
                    dev,
                    column,
                )
            )

            row[
                f"train_{column}"
            ] = train_mean

            row[
                f"dev_{column}"
            ] = dev_mean

            row[
                f"delta_dev_minus_train_{column}"
            ] = (
                dev_mean
                - train_mean
            )

            row[
                f"delta_ci_low_{column}"
            ] = ci_low

            row[
                f"delta_ci_high_{column}"
            ] = ci_high

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Reviewer subset
# =====================================================================

def choose_review_paths(
    frame,
):
    selected = set()

    for _, group in frame.groupby(
        [
            "evaluation_split",
            "variant",
            "selection_scope",
            "hardware_source",
        ]
    ):
        subset = (
            group.sort_values(
                [
                    "file_stem",
                    "image_path",
                ]
            )
            .head(2)
        )

        selected.update(
            subset[
                "image_path"
            ]
        )

    return selected


# =====================================================================
# Main
# =====================================================================

def main():
    checkpoint_sha = (
        verify_checkpoint()
    )

    (
        evaluation,
        selection_manifest,
    ) = load_evaluation_population()

    regions = load_regions()

    (
        annotation_info,
        region_table,
        clipped_count,
    ) = prepare_annotation_geometry(
        evaluation,
        regions,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISUAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_manifest.to_csv(
        OUT_SELECTION,
        index=False,
    )

    region_table.to_csv(
        OUT_REGIONS,
        index=False,
    )

    # --------------------------------------------------
    # Selection audit
    # --------------------------------------------------

    official_selection_summary = (
        selection_manifest.groupby(
            "variant"
        )
        .agg(
            total_attacks=(
                "image_path",
                "size",
            ),

            clean_correct=(
                "clean_correct_at_0_5",
                "sum",
            ),

            selected=(
                "selected_for_gradcam",
                "sum",
            ),

            n_stems=(
                "file_stem",
                "nunique",
            ),
        )
    )

    print(
        "FINAL CLEAN RESNET18 "
        "GRAD-CAM / RMA EVALUATION"
    )

    print(
        f"\ncheckpoint SHA: "
        f"{checkpoint_sha}"
        f"\ntotal evaluated images: "
        f"{len(evaluation)}"
        f"\nclipped GT rectangles: "
        f"{clipped_count}"
    )

    print(
        "\nOFFICIAL TEST SELECTION:"
    )

    print(
        official_selection_summary
        .to_string()
    )

    print(
        "\nSelection rule:"
        "\n  FaceDancer: all family"
        "\n  TextDiffuserFT: clean-correct only"
        "\n  Digital-3: clean-correct only"
        "\n  clean-correct threshold: p_attack >= 0.5"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint = (
        load_model(
            device
        )
    )

    gradcam = GradCAM(
        model,
        model.layer4[-1],
    )

    dataset = PolicyCDataset(
        evaluation
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    reviewer_paths = (
        choose_review_paths(
            evaluation
        )
    )

    print(
        f"\ntarget layer: "
        f"layer4[-1]"
        f"\ntarget scalar: "
        f"attack_logit - bonafide_logit"
        f"\ndevice: "
        f"{device}"
    )

    if (
        device.type
        == "cuda"
    ):
        print(
            "gpu: "
            + torch.cuda
            .get_device_name(0)
        )

    records = []

    low_cams = []

    max_probability_error = 0.0

    processed = 0

    heatmap_count = 0
    overlay_count = 0
    reviewer_count = 0

    try:
        for (
            x,
            labels,
            indices,
        ) in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            result = gradcam.generate(
                x
            )

            probabilities = (
                result[
                    "probability"
                ]
                .cpu()
                .numpy()
            )

            margins = (
                result[
                    "margin"
                ]
                .cpu()
                .numpy()
            )

            low_batch = (
                result[
                    "low_cam"
                ]
                .cpu()
                .numpy()
            )

            full_batch = (
                result[
                    "full_cam"
                ]
                .cpu()
                .numpy()
            )

            for (
                batch_position,
                frame_index,
            ) in enumerate(
                indices.numpy()
            ):
                row = evaluation.iloc[
                    frame_index
                ]

                info = (
                    annotation_info[
                        frame_index
                    ]
                )

                probability = float(
                    probabilities[
                        batch_position
                    ]
                )

                saved_probability = float(
                    row[
                        "saved_attack_probability"
                    ]
                )

                probability_error = abs(
                    probability
                    - saved_probability
                )

                max_probability_error = max(
                    max_probability_error,
                    probability_error,
                )

                cam = (
                    full_batch[
                        batch_position
                    ]
                )

                metrics = (
                    compute_rma_metrics(
                        cam,
                        info,
                    )
                )

                rendered = (
                    render_images(
                        x[
                            batch_position
                        ]
                        .detach()
                        .cpu(),
                        cam,
                        info,
                    )
                )

                visual_dir = (
                    VISUAL_ROOT
                    / str(
                        row[
                            "evaluation_split"
                        ]
                    )
                    / str(
                        row[
                            "selection_scope"
                        ]
                    )
                    / str(
                        row[
                            "variant"
                        ]
                    )
                    / str(
                        row[
                            "hardware_source"
                        ]
                    )
                )

                visual_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                identifier = (
                    safe_name(
                        row[
                            "file_stem"
                        ]
                    )
                    + "__"
                    + short_hash(
                        row[
                            "image_path"
                        ]
                    )
                )

                heatmap_path = (
                    visual_dir
                    / (
                        identifier
                        + "__heatmap.png"
                    )
                )

                overlay_path = (
                    visual_dir
                    / (
                        identifier
                        + "__overlay.png"
                    )
                )

                rendered[
                    "heatmap"
                ].save(
                    heatmap_path,
                    "PNG",
                    compress_level=3,
                )

                heatmap_count += 1

                rendered[
                    "overlay"
                ].save(
                    overlay_path,
                    "PNG",
                    compress_level=3,
                )

                overlay_count += 1

                if (
                    row[
                        "image_path"
                    ]
                    in reviewer_paths
                ):
                    panel = review_panel(
                        rendered[
                            "input_gt"
                        ],
                        rendered[
                            "heatmap"
                        ],
                        rendered[
                            "overlay"
                        ],
                    )

                    panel_name = (
                        safe_name(
                            row[
                                "evaluation_split"
                            ]
                        )
                        + "__"
                        + safe_name(
                            row[
                                "selection_scope"
                            ]
                        )
                        + "__"
                        + safe_name(
                            row[
                                "variant"
                            ]
                        )
                        + "__"
                        + safe_name(
                            row[
                                "hardware_source"
                            ]
                        )
                        + "__"
                        + identifier
                        + ".png"
                    )

                    panel.save(
                        REVIEW_ROOT
                        / panel_name,
                        "PNG",
                        compress_level=3,
                    )

                    reviewer_count += 1

                records.append(
                    {
                        "image_path":
                            row[
                                "image_path"
                            ],

                        "evaluation_split":
                            row[
                                "evaluation_split"
                            ],

                        "selection_scope":
                            row[
                                "selection_scope"
                            ],

                        "evidence_role":
                            row[
                                "evidence_role"
                            ],

                        "variant":
                            row[
                                "variant"
                            ],

                        "file_stem":
                            row[
                                "file_stem"
                            ],

                        "hardware_source":
                            row[
                                "hardware_source"
                            ],

                        "label":
                            int(
                                row[
                                    "label"
                                ]
                            ),

                        "assigned_q":
                            int(
                                row[
                                    "assigned_q"
                                ]
                            ),

                        "attack_probability":
                            probability,

                        "saved_attack_probability":
                            saved_probability,

                        "probability_abs_error":
                            probability_error,

                        "attack_margin":
                            float(
                                margins[
                                    batch_position
                                ]
                            ),

                        "predicted_attack":
                            int(
                                probability >= 0.5
                            ),

                        "native_width":
                            info[
                                "native_width"
                            ],

                        "native_height":
                            info[
                                "native_height"
                            ],

                        "content_width":
                            info[
                                "content_width"
                            ],

                        "pad_left":
                            info[
                                "pad_left"
                            ],

                        "pad_right":
                            info[
                                "pad_right"
                            ],

                        "visual_contour_threshold":
                            rendered[
                                "contour_threshold"
                            ],

                        "visual_contour_area_content":
                            rendered[
                                "contour_area"
                            ],

                        "heatmap_path":
                            str(
                                heatmap_path.relative_to(
                                    ROOT
                                )
                            ),

                        "overlay_path":
                            str(
                                overlay_path.relative_to(
                                    ROOT
                                )
                            ),

                        **metrics,
                    }
                )

                low_cams.append(
                    low_batch[
                        batch_position
                    ].astype(
                        np.float16
                    )
                )

            processed += len(
                indices
            )

            if (
                processed % 50
                < BATCH_SIZE
                or processed
                == len(
                    evaluation
                )
            ):
                print(
                    f"processed "
                    f"{processed}/"
                    f"{len(evaluation)}"
                )

    finally:
        gradcam.remove()

    metrics = pd.DataFrame(
        records
    )

    if (
        len(metrics)
        != len(evaluation)
    ):
        raise RuntimeError(
            "Result count mismatch"
        )

    if (
        max_probability_error
        > 0.01
    ):
        raise RuntimeError(
            "Inference consistency failed:"
            f"\nmax probability error="
            f"{max_probability_error:.8f}"
        )

    # --------------------------------------------------
    # Area bins frozen from project-train
    # --------------------------------------------------

    (
        metrics,
        area_thresholds,
    ) = assign_area_bins(
        metrics
    )

    area_thresholds.to_csv(
        OUT_AREA_THRESHOLDS,
        index=False,
    )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = make_summary(
        metrics
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    generalisation = (
        make_train_dev_generalisation(
            metrics
        )
    )

    generalisation.to_csv(
        OUT_GENERALISATION,
        index=False,
    )

    # --------------------------------------------------
    # Frozen layer4 maps
    # --------------------------------------------------

    low_cams = np.stack(
        low_cams
    )

    np.savez_compressed(
        OUT_CAMS,
        cams=low_cams,

        image_paths=(
            metrics[
                "image_path"
            ]
            .to_numpy(
                dtype=str
            )
        ),

        evaluation_splits=(
            metrics[
                "evaluation_split"
            ]
            .to_numpy(
                dtype=str
            )
        ),

        selection_scopes=(
            metrics[
                "selection_scope"
            ]
            .to_numpy(
                dtype=str
            )
        ),

        variants=(
            metrics[
                "variant"
            ]
            .to_numpy(
                dtype=str
            )
        ),
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print(
        "\nINFERENCE CONSISTENCY:"
        f"\n  max probability error: "
        f"{max_probability_error:.8f}"
        f"\n  evaluated images:      "
        f"{len(metrics)}"
        f"\n  zero-content CAMs:     "
        f"{metrics['cam_zero_content'].sum()}"
    )

    print(
        "\nAREA STRATA — "
        "FROZEN FROM PROJECT TRAIN:"
    )

    print(
        area_thresholds.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    primary_rows = (
        summary[
            summary[
                "area_bin"
            ]
            == "all"
        ]
        .copy()
    )

    display = [
        "evaluation_split",
        "variant",
        "selection_scope",
        "evidence_role",
        "n_images",
        "n_stems",

        "A_mean",
        "A_median",

        "E_mean",
        "E_median",
        "E_mean_ci_low",
        "E_mean_ci_high",

        "mu_w_mean",
        "mu_w_median",
        "mu_w_mean_ci_low",
        "mu_w_mean_ci_high",

        "PG_mean",
        "PG_median",
        "PG_mean_ci_low",
        "PG_mean_ci_high",

        "energy_gain_mean",
        "padding_energy_mean",
        "pointing_padding",

        "face_E_mean",
        "face_mu_w_mean",
        "face_PG_mean",

        "text_E_mean",
        "text_mu_w_mean",
        "text_PG_mean",
    ]

    print(
        "\nPRIMARY AREA-AWARE "
        "LOCALISATION RESULTS:"
    )

    print(
        primary_rows[
            display
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    area_rows = (
        summary[
            summary[
                "area_bin"
            ]
            != "all"
        ]
    )

    print(
        "\nAREA-STRATIFIED "
        "LOCALISATION:"
    )

    print(
        area_rows[
            [
                "evaluation_split",
                "variant",
                "selection_scope",
                "area_bin",
                "n_images",
                "n_stems",
                "A_mean",
                "E_mean",
                "mu_w_mean",
                "PG_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nTRAIN -> DEV "
        "LOCALISATION GENERALISATION:"
    )

    print(
        generalisation[
            [
                "variant",

                "train_rma_E",
                "dev_rma_E",
                "delta_dev_minus_train_rma_E",
                "delta_ci_low_rma_E",
                "delta_ci_high_rma_E",

                "train_rma_mu_w",
                "dev_rma_mu_w",
                "delta_dev_minus_train_rma_mu_w",
                "delta_ci_low_rma_mu_w",
                "delta_ci_high_rma_mu_w",

                "train_rma_PG",
                "dev_rma_PG",
                "delta_dev_minus_train_rma_PG",
                "delta_ci_low_rma_PG",
                "delta_ci_high_rma_PG",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nOFFICIAL TEST INTERPRETATION:"
    )

    for row in (
        official_selection_summary
        .reset_index()
        .itertuples(
            index=False
        )
    ):
        print(
            f"  {row.variant}: "
            f"total={row.total_attacks}, "
            f"clean_correct={int(row.clean_correct)}, "
            f"evaluated={int(row.selected)}"
        )

    print(
        "\n  FaceDancer results describe "
        "the complete attack family."
        "\n  TextDiffuserFT and Digital-3 "
        "results describe CLEAN-CORRECT "
        "subsets only."
    )

    print(
        "\nVISUAL OUTPUT AUDIT:"
        f"\n  heatmaps:       "
        f"{heatmap_count}"
        f"\n  overlays:       "
        f"{overlay_count}"
        f"\n  reviewer panels:"
        f" {reviewer_count}"
    )

    print(
        "\nOUTPUTS:"
        f"\n  per-image metrics: "
        f"{OUT_METRICS}"
        f"\n  summary:           "
        f"{OUT_SUMMARY}"
        f"\n  area thresholds:   "
        f"{OUT_AREA_THRESHOLDS}"
        f"\n  train/dev delta:   "
        f"{OUT_GENERALISATION}"
        f"\n  GT region audit:   "
        f"{OUT_REGIONS}"
        f"\n  test selection:    "
        f"{OUT_SELECTION}"
        f"\n  layer4 CAM maps:   "
        f"{OUT_CAMS}"
        f"\n  all visuals:       "
        f"{VISUAL_ROOT}"
        f"\n  review panels:     "
        f"{REVIEW_ROOT}"
    )

    print(
        "\nPRIMARY METRIC DEFINITIONS:"
        "\n"
        "\n  A = GT union area / document-content area"
        "\n"
        "\n  E = CAM relevance mass inside GT union"
        "\n      / total CAM relevance mass inside "
        "document content"
        "\n"
        "\n  mu_w = E / A"
        "\n"
        "\n  PG = 1 if strongest content CAM point "
        "falls in GT union"
        "\n"
        "\n  mu_w = 1 corresponds to spatially "
        "uniform CAM mass."
        "\n"
        "\nThe artificial horizontal padding is excluded "
        "from A/E/mu_w/PG and audited separately."
        "\n"
        "\nStop here before adversarial attacks."
    )


if __name__ == "__main__":
    main()