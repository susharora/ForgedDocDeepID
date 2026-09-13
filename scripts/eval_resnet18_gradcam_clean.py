#!/usr/bin/env python3
"""
Clean Grad-CAM localisation baseline for the frozen compression-controlled
FantasyID ResNet18.

Populations
-----------
project_train attacks:
    digital_1: 480
    digital_2: 480
    total:     960

held-out dev attacks:
    digital_1: 153
    digital_2: 153
    total:     306

Train and dev are NEVER pooled into the headline localisation result.
Training localisation is a diagnostic of what the fitted model learned.
Held-out dev localisation is the primary clean generalisation result.

Frozen model
------------
runs/post_hoc_compression_controlled_resnet18_seed10/checkpoints/best.pt

Frozen preprocessing
--------------------
Policy-C native cached image
    -> aspect-preserving resize to height 512
    -> horizontal centering in 512 x 864 canvas
    -> ImageNet normalization
    -> ImageNet-mean padding (zero after normalization)

Grad-CAM
--------
Target layer:
    model.layer4[-1]

Target scalar:
    attack_logit - bonafide_logit

The resulting ReLU Grad-CAM therefore represents positive spatial evidence
moving the classifier toward "attack".

Ground truth
------------
Uses ONLY FantasyID Regions rows with:

    region_provenance_raw == "altered"

The Regions sheet is the frozen canonical extraction of the accompanying
FantasyID JSON rectangle annotations.

    field_name == "face" -> face
    every other altered field -> text

The original native annotation is preserved in the region audit CSV.
If an annotation extends outside the decoded image, only the visible
intersection can be drawn/evaluated; the original box is still recorded.

Metrics
-------
Primary threshold-free metrics:

    CAM energy fraction inside altered union
    CAM enrichment over annotation-area baseline
    CAM global-maximum pointing accuracy

Face and text are reported separately.

Two normalisations are retained:

    canvas:
        denominator includes document + padding

    content:
        denominator is document content only

Padding energy is reported independently.

Visualisation
-------------
For EVERY attack image:

    *__heatmap.png
        pure Grad-CAM heatmap + CAM contour + GT rectangles

    *__overlay.png
        actual 512x864 model input + Grad-CAM + contour + GT rectangles

Visual convention:

    WHITE   = strongest 10% of positive CAM activation contour
              within document content
    CYAN    = altered face ground-truth rectangle
    MAGENTA = altered text ground-truth rectangle
    GREY    = document/padding boundary

A deterministic subset also receives 3-panel reviewer figures:

    input + GT | Grad-CAM + GT | overlay

Important
---------
The top-10% contour is FOR VISUALISATION ONLY.

All primary localisation metrics remain threshold-free.

The annotations are coarse rectangles, not pixel-perfect manipulation masks.
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


# ---------------------------------------------------------------------
# Frozen paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
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

SAVED_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_predictions.csv"
)


# ---------------------------------------------------------------------
# Output tree
# ---------------------------------------------------------------------

OUT_ROOT = (
    ROOT
    / "output"
    / "resnet18_gradcam_clean"
)

OUT_METRICS = (
    OUT_ROOT
    / "gradcam_metrics.csv"
)

OUT_SUMMARY = (
    OUT_ROOT
    / "gradcam_summary.csv"
)

OUT_GENERALISATION = (
    OUT_ROOT
    / "gradcam_train_dev_generalisation.csv"
)

OUT_REGIONS = (
    OUT_ROOT
    / "gradcam_regions.csv"
)

OUT_CAMS = (
    OUT_ROOT
    / "gradcam_layer4_maps.npz"
)

VISUAL_ROOT = (
    OUT_ROOT
    / "visuals"
)

REVIEW_ROOT = (
    OUT_ROOT
    / "review_panels"
)


# ---------------------------------------------------------------------
# Frozen preprocessing
# ---------------------------------------------------------------------

CONTENT_H = 512
CANVAS_W = 864

BATCH_SIZE = 4

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)

SEED = 10

N_BOOT = 5000

# Visualisation only.
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


# ---------------------------------------------------------------------
# Visual colours
# ---------------------------------------------------------------------

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

COLOR_CAM_CONTOUR = (
    255,
    255,
    255,
)

COLOR_CONTENT_BOUNDARY = (
    160,
    160,
    160,
)

COLOR_LABEL_BACKGROUND = (
    0,
    0,
    0,
)

COLOR_LABEL_TEXT = (
    255,
    255,
    255,
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def require_file(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required file missing:\n{path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def short_hash(text):
    return (
        hashlib.sha256(
            text.encode()
        )
        .hexdigest()[:8]
    )


def safe_name(text):
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )

    return value.strip(
        "_"
    )


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
        )


# ---------------------------------------------------------------------
# Dataset — exact frozen r512 preprocessing
# ---------------------------------------------------------------------

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

            if (
                new_width
                > CANVAS_W
            ):
                raise RuntimeError(
                    "Image exceeds frozen "
                    "r512 canvas:"
                    f"\n{row.image_path}"
                    f"\nresized width="
                    f"{new_width}"
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

        # Zero after ImageNet normalization
        # == ImageNet mean RGB before normalization.
        canvas = torch.zeros(
            (
                3,
                CONTENT_H,
                CANVAS_W,
            ),
            dtype=tensor.dtype,
        )

        pad_left = (
            CANVAS_W
            - new_width
        ) // 2

        canvas[
            :,
            :,
            pad_left:
            pad_left + new_width,
        ] = tensor

        return (
            canvas,
            int(
                row.label
            ),
            index,
        )


# ---------------------------------------------------------------------
# Frozen ResNet
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------

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
        self.gradients = (
            gradient
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = (
            output
        )

        output.register_hook(
            self._save_gradient
        )

    def remove(self):
        self.handle.remove()

    def generate(
        self,
        x,
    ):
        """
        Generate CAM for:

            attack_logit - bonafide_logit

        rather than for whichever class
        happens to win the prediction.
        """

        self.activations = None
        self.gradients = None

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            x
        )

        attack_margin = (
            logits[:, 1]
            -
            logits[:, 0]
        )

        attack_margin.sum().backward()

        if (
            self.activations
            is None
            or self.gradients
            is None
        ):
            raise RuntimeError(
                "Grad-CAM hook failed"
            )

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        weights = (
            gradients.mean(
                dim=(
                    2,
                    3,
                ),
                keepdim=True,
            )
        )

        low_cam = (
            weights
            * activations
        ).sum(
            dim=1
        )

        low_cam = torch.relu(
            low_cam
        )

        # Normalize independently.
        # Energy fractions are unchanged
        # by this positive scalar.
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

        normalized = torch.where(
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
                normalized.unsqueeze(1),
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
            "logits":
                logits.detach(),

            "attack_margin":
                attack_margin.detach(),

            "attack_probability":
                probability.detach(),

            "low_cam":
                normalized.detach(),

            "full_cam":
                full_cam.detach(),
        }


# ---------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------

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
        regions
        .reset_index()
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
            float(
                row.x
            )
        )
    )

    y0 = int(
        round(
            float(
                row.y
            )
        )
    )

    width = int(
        round(
            float(
                row.width
            )
        )
    )

    height = int(
        round(
            float(
                row.height
            )
        )
    )

    if (
        width <= 0
        or height <= 0
    ):
        raise RuntimeError(
            "Invalid annotation rectangle"
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
    """
    Preserve BOTH:

        original native rectangle
        clipped visible rectangle

    Then project the clipped rectangle to
    the exact 512x864 model canvas.
    """

    x0, y0, x1, y1 = (
        box
    )

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

    clipped_box = (
        cx0,
        cy0,
        cx1,
        cy1,
    )

    clipped = (
        clipped_box
        != box
    )

    if (
        cx1 <= cx0
        or cy1 <= cy0
    ):
        raise RuntimeError(
            "Altered annotation does not "
            "intersect decoded image"
        )

    scale_x = (
        content_width
        / native_width
    )

    scale_y = (
        CONTENT_H
        / native_height
    )

    # Rounding of resized width means
    # they are nearly, not perfectly,
    # identical.
    if (
        abs(
            scale_x
            - scale_y
        )
        > 0.002
    ):
        raise RuntimeError(
            "Unexpected anisotropic "
            "preprocessing geometry"
        )

    model_x0 = (
        pad_left
        +
        int(
            np.floor(
                cx0
                * scale_x
            )
        )
    )

    model_y0 = int(
        np.floor(
            cy0
            * scale_y
        )
    )

    model_x1 = (
        pad_left
        +
        int(
            np.ceil(
                cx1
                * scale_x
            )
        )
    )

    model_y1 = int(
        np.ceil(
            cy1
            * scale_y
        )
    )

    model_x0 = max(
        0,
        min(
            CANVAS_W,
            model_x0,
        ),
    )

    model_x1 = max(
        0,
        min(
            CANVAS_W,
            model_x1,
        ),
    )

    model_y0 = max(
        0,
        min(
            CONTENT_H,
            model_y0,
        ),
    )

    model_y1 = max(
        0,
        min(
            CONTENT_H,
            model_y1,
        ),
    )

    model_box = (
        model_x0,
        model_y0,
        model_x1,
        model_y1,
    )

    if (
        model_x1 <= model_x0
        or model_y1 <= model_y0
    ):
        raise RuntimeError(
            "Projected altered rectangle "
            "is empty"
        )

    return (
        clipped_box,
        model_box,
        clipped,
    )


def prepare_annotation_geometry(
    frame,
    regions,
):
    altered = regions[
        regions[
            "provenance_norm"
        ]
        == "altered"
    ].copy()

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

    clipped_boxes = 0

    for row in frame.itertuples(
        index=False
    ):
        if (
            row.image_path
            not in grouped
        ):
            raise RuntimeError(
                "No altered Regions rows:"
                f"\n{row.image_path}"
            )

        path = (
            ROOT
            / row.cache_path
        )

        with Image.open(
            path
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
                "Image exceeds frozen canvas"
            )

        pad_left = (
            CANVAS_W
            - content_width
        ) // 2

        pad_right = (
            CANVAS_W
            -
            pad_left
            -
            content_width
        )

        face_rects = []
        text_rects = []

        visual_regions = []

        image_group = (
            grouped[
                row.image_path
            ]
        )

        for region in (
            image_group.itertuples(
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

            clipped_boxes += int(
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

            field_name = str(
                region.field_name
            )

            if semantic_type == "face":
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
                        field_name,

                    "field_norm":
                        region.field_norm,

                    "semantic_type":
                        semantic_type,

                    "model_box":
                        model_box,

                    "annotation_clipped":
                        clipped,
                }
            )

            region_records.append(
                {
                    "image_path":
                        row.image_path,

                    "split":
                        row.split,

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
                        field_name,

                    "field_norm":
                        region.field_norm,

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

        if not face_rects:
            raise RuntimeError(
                "Attack has no altered "
                "face rectangle:"
                f"\n{row.image_path}"
            )

        if not text_rects:
            raise RuntimeError(
                "Attack has no altered "
                "text rectangle:"
                f"\n{row.image_path}"
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
        clipped_boxes,
    )


# ---------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------

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


def content_mask_from_info(
    info,
):
    content = np.zeros(
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

    content[
        :,
        x0:x1,
    ] = True

    return content


# ---------------------------------------------------------------------
# Grad-CAM metrics
# ---------------------------------------------------------------------

def safe_enrichment(
    energy_fraction,
    area_fraction,
):
    if (
        area_fraction
        <= 1e-12
    ):
        return np.nan

    return (
        energy_fraction
        / area_fraction
    )


def compute_cam_metrics(
    cam,
    info,
):
    """
    Threshold-free quantitative metrics.

    `cam` is 512x864 and non-negative.
    """

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

    face_text_overlap = (
        face
        & text
    )

    content = (
        content_mask_from_info(
            info
        )
    )

    padding = (
        ~content
    )

    canvas_pixels = (
        CONTENT_H
        * CANVAS_W
    )

    content_pixels = int(
        content.sum()
    )

    # --------------------------------------------------
    # Area baselines
    # --------------------------------------------------

    area_union_canvas = (
        union.sum()
        / canvas_pixels
    )

    area_face_canvas = (
        face.sum()
        / canvas_pixels
    )

    area_text_canvas = (
        text.sum()
        / canvas_pixels
    )

    area_padding_canvas = (
        padding.sum()
        / canvas_pixels
    )

    area_union_content = (
        union.sum()
        / content_pixels
    )

    area_face_content = (
        face.sum()
        / content_pixels
    )

    area_text_content = (
        text.sum()
        / content_pixels
    )

    overlap_canvas = (
        face_text_overlap.sum()
        / canvas_pixels
    )

    # --------------------------------------------------
    # CAM energy
    # --------------------------------------------------

    total_energy = float(
        cam.sum()
    )

    content_energy_absolute = float(
        cam[
            content
        ].sum()
    )

    zero_cam = (
        total_energy
        <= 1e-12
    )

    if zero_cam:
        energy_union_canvas = 0.0
        energy_face_canvas = 0.0
        energy_text_canvas = 0.0
        energy_padding_canvas = 0.0

        energy_union_content = 0.0
        energy_face_content = 0.0
        energy_text_content = 0.0

        point_union = 0.0
        point_face = 0.0
        point_text = 0.0
        point_padding = 0.0

        max_x = -1
        max_y = -1

    else:
        energy_union_canvas = float(
            cam[
                union
            ].sum()
            / total_energy
        )

        energy_face_canvas = float(
            cam[
                face
            ].sum()
            / total_energy
        )

        energy_text_canvas = float(
            cam[
                text
            ].sum()
            / total_energy
        )

        energy_padding_canvas = float(
            cam[
                padding
            ].sum()
            / total_energy
        )

        if (
            content_energy_absolute
            > 1e-12
        ):
            energy_union_content = float(
                cam[
                    union
                ].sum()
                / content_energy_absolute
            )

            energy_face_content = float(
                cam[
                    face
                ].sum()
                / content_energy_absolute
            )

            energy_text_content = float(
                cam[
                    text
                ].sum()
                / content_energy_absolute
            )

        else:
            energy_union_content = 0.0
            energy_face_content = 0.0
            energy_text_content = 0.0

        flat_index = int(
            np.argmax(
                cam
            )
        )

        max_y, max_x = (
            np.unravel_index(
                flat_index,
                cam.shape,
            )
        )

        point_union = float(
            union[
                max_y,
                max_x,
            ]
        )

        point_face = float(
            face[
                max_y,
                max_x,
            ]
        )

        point_text = float(
            text[
                max_y,
                max_x,
            ]
        )

        point_padding = float(
            padding[
                max_y,
                max_x,
            ]
        )

    enrichment_union_canvas = (
        safe_enrichment(
            energy_union_canvas,
            area_union_canvas,
        )
    )

    enrichment_face_canvas = (
        safe_enrichment(
            energy_face_canvas,
            area_face_canvas,
        )
    )

    enrichment_text_canvas = (
        safe_enrichment(
            energy_text_canvas,
            area_text_canvas,
        )
    )

    enrichment_padding_canvas = (
        safe_enrichment(
            energy_padding_canvas,
            area_padding_canvas,
        )
    )

    enrichment_union_content = (
        safe_enrichment(
            energy_union_content,
            area_union_content,
        )
    )

    enrichment_face_content = (
        safe_enrichment(
            energy_face_content,
            area_face_content,
        )
    )

    enrichment_text_content = (
        safe_enrichment(
            energy_text_content,
            area_text_content,
        )
    )

    altered_energy_sum = (
        energy_face_content
        +
        energy_text_content
    )

    if (
        altered_energy_sum
        > 1e-12
    ):
        face_share = (
            energy_face_content
            / altered_energy_sum
        )

        text_share = (
            energy_text_content
            / altered_energy_sum
        )

    else:
        face_share = 0.0
        text_share = 0.0

    return {
        "area_union_canvas":
            float(
                area_union_canvas
            ),

        "area_face_canvas":
            float(
                area_face_canvas
            ),

        "area_text_canvas":
            float(
                area_text_canvas
            ),

        "area_padding_canvas":
            float(
                area_padding_canvas
            ),

        "area_union_content":
            float(
                area_union_content
            ),

        "area_face_content":
            float(
                area_face_content
            ),

        "area_text_content":
            float(
                area_text_content
            ),

        "area_face_text_overlap_canvas":
            float(
                overlap_canvas
            ),

        "energy_union_canvas":
            energy_union_canvas,

        "energy_face_canvas":
            energy_face_canvas,

        "energy_text_canvas":
            energy_text_canvas,

        "energy_padding_canvas":
            energy_padding_canvas,

        "energy_union_content":
            energy_union_content,

        "energy_face_content":
            energy_face_content,

        "energy_text_content":
            energy_text_content,

        "enrichment_union_canvas":
            enrichment_union_canvas,

        "enrichment_face_canvas":
            enrichment_face_canvas,

        "enrichment_text_canvas":
            enrichment_text_canvas,

        "enrichment_padding_canvas":
            enrichment_padding_canvas,

        "enrichment_union_content":
            enrichment_union_content,

        "enrichment_face_content":
            enrichment_face_content,

        "enrichment_text_content":
            enrichment_text_content,

        # Primary threshold-free effect.
        "energy_gain_union_content":
            (
                energy_union_content
                -
                area_union_content
            ),

        "point_union":
            point_union,

        "point_face":
            point_face,

        "point_text":
            point_text,

        "point_padding":
            point_padding,

        # Global max is free to fall in padding,
        # so full-canvas area is the fair random baseline.
        "point_gain_union_canvas":
            (
                point_union
                -
                area_union_canvas
            ),

        "face_share_altered_energy":
            float(
                face_share
            ),

        "text_share_altered_energy":
            float(
                text_share
            ),

        "cam_zero":
            int(
                zero_cam
            ),

        "cam_max_x":
            int(
                max_x
            ),

        "cam_max_y":
            int(
                max_y
            ),
    }


# ---------------------------------------------------------------------
# Visual-only CAM contour
# ---------------------------------------------------------------------

def make_cam_contour(
    cam,
    info,
):
    """
    White contour around the strongest 10%
    of POSITIVE CAM values within document content.

    Visualisation only.
    """

    content = (
        content_mask_from_info(
            info
        )
    )

    positive_mask = (
        content
        &
        (
            cam > 0
        )
    )

    values = cam[
        positive_mask
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

    # Boundary of active component(s).
    eroded = binary_erosion(
        active,
        iterations=1,
        border_value=0,
    )

    boundary = (
        active
        &
        ~eroded
    )

    # Two-pixel-ish white outline
    # so it remains visible over heat.
    boundary = binary_dilation(
        boundary,
        iterations=1,
    )

    area_fraction = float(
        active.sum()
        / content.sum()
    )

    return (
        boundary,
        threshold,
        area_fraction,
    )


# ---------------------------------------------------------------------
# Visual rendering
# ---------------------------------------------------------------------

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
        rgb
        .clamp(
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
    """
    Simple deterministic red -> yellow heat map.
    No plotting-library dependency.
    """

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

    return (
        np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=2,
        )
        .round()
        .astype(
            np.uint8
        )
    )


def blend_overlay(
    base,
    cam,
):
    heat = (
        heat_rgb(
            cam
        )
        .astype(
            np.float32
        )
    )

    base_float = (
        base.astype(
            np.float32
        )
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

    output = (
        base_float
        * (
            1.0
            - alpha
        )
        +
        heat
        * alpha
    )

    return (
        np.clip(
            output,
            0,
            255,
        )
        .astype(
            np.uint8
        )
    )


def default_font():
    return ImageFont.load_default()


def draw_text_label(
    draw,
    position,
    text,
    color,
):
    font = default_font()

    x, y = position

    text = str(
        text
    )

    bbox = draw.textbbox(
        (
            x,
            y,
        ),
        text,
        font=font,
    )

    x0, y0, x1, y1 = bbox

    draw.rectangle(
        (
            x0 - 2,
            y0 - 1,
            x1 + 2,
            y1 + 1,
        ),
        fill=COLOR_LABEL_BACKGROUND,
    )

    draw.text(
        (
            x,
            y,
        ),
        text,
        fill=color,
        font=font,
    )


def draw_regions_and_boundaries(
    image,
    info,
):
    draw = ImageDraw.Draw(
        image
    )

    # Document/padding boundaries.
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
        fill=(
            COLOR_CONTENT_BOUNDARY
        ),
        width=1,
    )

    draw.line(
        (
            right,
            0,
            right,
            CONTENT_H - 1,
        ),
        fill=(
            COLOR_CONTENT_BOUNDARY
        ),
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

        field_label = str(
            region[
                "field_name"
            ]
        )

        if (
            region[
                "annotation_clipped"
            ]
        ):
            field_label += " *"

        label_y = max(
            1,
            y0 - 13,
        )

        draw_text_label(
            draw,
            (
                x0 + 2,
                label_y,
            ),
            field_label,
            color,
        )

    return image


def apply_contour(
    image_array,
    contour,
):
    output = image_array.copy()

    output[
        contour
    ] = np.asarray(
        COLOR_CAM_CONTOUR,
        dtype=np.uint8,
    )

    return output


def draw_legend(
    image,
):
    draw = ImageDraw.Draw(
        image
    )

    font = default_font()

    entries = [
        (
            COLOR_CAM_CONTOUR,
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
            COLOR_CONTENT_BOUNDARY,
            "document boundary",
        ),
    ]

    x0 = 8
    y0 = 8

    line_height = 15

    width = 190

    height = (
        8
        +
        line_height
        * len(entries)
    )

    draw.rectangle(
        (
            x0 - 4,
            y0 - 4,
            x0 + width,
            y0 + height,
        ),
        fill=(
            0,
            0,
            0,
        ),
    )

    for index, (
        color,
        label,
    ) in enumerate(
        entries
    ):
        y = (
            y0
            +
            index
            * line_height
        )

        draw.line(
            (
                x0,
                y + 6,
                x0 + 20,
                y + 6,
            ),
            fill=color,
            width=3,
        )

        draw.text(
            (
                x0 + 26,
                y,
            ),
            label,
            fill=COLOR_LABEL_TEXT,
            font=font,
        )

    return image


def render_images(
    canvas_tensor,
    cam,
    info,
):
    base = canvas_to_rgb(
        canvas_tensor
    )

    contour, threshold, contour_area = (
        make_cam_contour(
            cam,
            info,
        )
    )

    # --------------------------------------------------
    # Input + GT only
    # --------------------------------------------------

    input_gt = Image.fromarray(
        base.copy()
    )

    input_gt = (
        draw_regions_and_boundaries(
            input_gt,
            info,
        )
    )

    input_gt = draw_legend(
        input_gt
    )

    # --------------------------------------------------
    # Pure heatmap + GT
    # --------------------------------------------------

    heat = heat_rgb(
        cam
    )

    heat = apply_contour(
        heat,
        contour,
    )

    heatmap = Image.fromarray(
        heat
    )

    heatmap = (
        draw_regions_and_boundaries(
            heatmap,
            info,
        )
    )

    heatmap = draw_legend(
        heatmap
    )

    # --------------------------------------------------
    # Model input + heatmap + GT
    # --------------------------------------------------

    overlay_array = (
        blend_overlay(
            base,
            cam,
        )
    )

    overlay_array = (
        apply_contour(
            overlay_array,
            contour,
        )
    )

    overlay = Image.fromarray(
        overlay_array
    )

    overlay = (
        draw_regions_and_boundaries(
            overlay,
            info,
        )
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

        "contour_area_content":
            contour_area,
    }


def add_panel_header(
    image,
    title,
):
    header_h = 24

    panel = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + header_h,
        ),
        (
            0,
            0,
            0,
        ),
    )

    panel.paste(
        image,
        (
            0,
            header_h,
        ),
    )

    draw = ImageDraw.Draw(
        panel
    )

    draw.text(
        (
            8,
            6,
        ),
        title,
        fill=(
            255,
            255,
            255,
        ),
        font=default_font(),
    )

    return panel


def make_review_panel(
    input_gt,
    heatmap,
    overlay,
):
    panels = [
        add_panel_header(
            input_gt,
            "MODEL INPUT + GROUND TRUTH",
        ),
        add_panel_header(
            heatmap,
            "GRAD-CAM + CONTOUR + GROUND TRUTH",
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
        (
            0,
            0,
            0,
        ),
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


# ---------------------------------------------------------------------
# Reviewer sample
# ---------------------------------------------------------------------

def choose_review_paths(
    frame,
):
    """
    Fixed, non-CAM-selected reviewer subset.

    2 alphabetical samples for every:
        split x variant x hardware

    2 splits x 2 variants x 3 hardware x 2
    = 24 reviewer panels.
    """

    selected = set()

    for (
        split,
        variant,
        hardware,
    ), group in (
        frame.groupby(
            [
                "split",
                "variant",
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


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def stem_groups(
    frame,
    column,
):
    return {
        stem:
            frame[
                frame[
                    "file_stem"
                ]
                == stem
            ][
                column
            ]
            .to_numpy(
                dtype=float
            )
        for stem
        in sorted(
            frame[
                "file_stem"
            ].unique()
        )
    }


def bootstrap_mean_ci(
    frame,
    column,
):
    groups = stem_groups(
        frame,
        column,
    )

    stems = np.array(
        list(
            groups.keys()
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    results = np.empty(
        N_BOOT,
        dtype=float,
    )

    for index in range(
        N_BOOT
    ):
        sampled = rng.choice(
            stems,
            size=len(stems),
            replace=True,
        )

        values = np.concatenate(
            [
                groups[
                    stem
                ]
                for stem
                in sampled
            ]
        )

        results[index] = (
            np.mean(
                values
            )
        )

    return (
        float(
            np.quantile(
                results,
                0.025,
            )
        ),
        float(
            np.quantile(
                results,
                0.975,
            )
        ),
    )


def bootstrap_difference_ci(
    train,
    dev,
    column,
):
    """
    Independent stem-cluster bootstrap.

    Returns CI for:

        DEV mean - TRAIN mean
    """

    train_groups = (
        stem_groups(
            train,
            column,
        )
    )

    dev_groups = (
        stem_groups(
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

    difference = np.empty(
        N_BOOT,
        dtype=float,
    )

    for index in range(
        N_BOOT
    ):
        sampled_train = (
            rng.choice(
                train_stems,
                size=len(
                    train_stems
                ),
                replace=True,
            )
        )

        sampled_dev = (
            rng.choice(
                dev_stems,
                size=len(
                    dev_stems
                ),
                replace=True,
            )
        )

        train_values = (
            np.concatenate(
                [
                    train_groups[
                        stem
                    ]
                    for stem
                    in sampled_train
                ]
            )
        )

        dev_values = (
            np.concatenate(
                [
                    dev_groups[
                        stem
                    ]
                    for stem
                    in sampled_dev
                ]
            )
        )

        difference[
            index
        ] = (
            dev_values.mean()
            -
            train_values.mean()
        )

    return (
        float(
            np.quantile(
                difference,
                0.025,
            )
        ),
        float(
            np.quantile(
                difference,
                0.975,
            )
        ),
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summary_row(
    frame,
    group_type,
    group,
):
    energy_ci = (
        bootstrap_mean_ci(
            frame,
            "energy_gain_union_content",
        )
    )

    pointing_ci = (
        bootstrap_mean_ci(
            frame,
            "point_gain_union_canvas",
        )
    )

    return {
        "group_type":
            group_type,

        "group":
            group,

        "n_images":
            len(frame),

        "n_stems":
            frame[
                "file_stem"
            ].nunique(),

        "mean_attack_probability":
            frame[
                "attack_probability"
            ].mean(),

        "attack_recall_at_0_5":
            frame[
                "predicted_attack"
            ].mean(),

        # Primary union metrics.
        "mean_union_area_content":
            frame[
                "area_union_content"
            ].mean(),

        "mean_union_energy_content":
            frame[
                "energy_union_content"
            ].mean(),

        "mean_union_enrichment_content":
            frame[
                "enrichment_union_content"
            ].mean(),

        "mean_energy_gain_union_content":
            frame[
                "energy_gain_union_content"
            ].mean(),

        "energy_gain_ci_low":
            energy_ci[0],

        "energy_gain_ci_high":
            energy_ci[1],

        "pointing_union":
            frame[
                "point_union"
            ].mean(),

        "mean_union_area_canvas":
            frame[
                "area_union_canvas"
            ].mean(),

        "mean_point_gain_union_canvas":
            frame[
                "point_gain_union_canvas"
            ].mean(),

        "point_gain_ci_low":
            pointing_ci[0],

        "point_gain_ci_high":
            pointing_ci[1],

        # Face.
        "mean_face_area_content":
            frame[
                "area_face_content"
            ].mean(),

        "mean_face_energy_content":
            frame[
                "energy_face_content"
            ].mean(),

        "mean_face_enrichment_content":
            frame[
                "enrichment_face_content"
            ].mean(),

        "pointing_face":
            frame[
                "point_face"
            ].mean(),

        # Text.
        "mean_text_area_content":
            frame[
                "area_text_content"
            ].mean(),

        "mean_text_energy_content":
            frame[
                "energy_text_content"
            ].mean(),

        "mean_text_enrichment_content":
            frame[
                "enrichment_text_content"
            ].mean(),

        "pointing_text":
            frame[
                "point_text"
            ].mean(),

        # Cue balance.
        "mean_face_share_altered_energy":
            frame[
                "face_share_altered_energy"
            ].mean(),

        "mean_text_share_altered_energy":
            frame[
                "text_share_altered_energy"
            ].mean(),

        # Padding sanity check.
        "mean_padding_area_canvas":
            frame[
                "area_padding_canvas"
            ].mean(),

        "mean_padding_energy_canvas":
            frame[
                "energy_padding_canvas"
            ].mean(),

        "mean_padding_enrichment_canvas":
            frame[
                "enrichment_padding_canvas"
            ].mean(),

        "pointing_padding":
            frame[
                "point_padding"
            ].mean(),

        # Misc.
        "zero_cam_fraction":
            frame[
                "cam_zero"
            ].mean(),

        "mean_face_text_overlap_canvas":
            frame[
                "area_face_text_overlap_canvas"
            ].mean(),

        "mean_visual_contour_area_content":
            frame[
                "visual_contour_area_content"
            ].mean(),
    }


def make_summary(
    metrics,
):
    rows = []

    # Split-level only.
    for split, group in (
        metrics.groupby(
            "split"
        )
    ):
        rows.append(
            summary_row(
                group,
                "split",
                split,
            )
        )

    # Split + manipulation method.
    for (
        split,
        variant,
    ), group in (
        metrics.groupby(
            [
                "split",
                "variant",
            ]
        )
    ):
        rows.append(
            summary_row(
                group,
                "split_variant",
                (
                    f"{split}|"
                    f"{variant}"
                ),
            )
        )

    # Full hardware detail.
    for (
        split,
        variant,
        hardware,
    ), group in (
        metrics.groupby(
            [
                "split",
                "variant",
                "hardware_source",
            ]
        )
    ):
        rows.append(
            summary_row(
                group,
                "split_variant_hardware",
                (
                    f"{split}|"
                    f"{variant}|"
                    f"{hardware}"
                ),
            )
        )

    return pd.DataFrame(
        rows
    )


def make_generalisation_table(
    metrics,
):
    rows = []

    columns = [
        "energy_gain_union_content",
        "enrichment_union_content",
        "point_union",
        "point_gain_union_canvas",
        "face_share_altered_energy",
        "text_share_altered_energy",
        "energy_padding_canvas",
    ]

    for variant in [
        "digital_1",
        "digital_2",
    ]:
        train = metrics[
            (
                metrics[
                    "split"
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
                    "split"
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

        for column in columns:
            train_mean = float(
                train[
                    column
                ].mean()
            )

            dev_mean = float(
                dev[
                    column
                ].mean()
            )

            (
                ci_low,
                ci_high,
            ) = bootstrap_difference_ci(
                train,
                dev,
                column,
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


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def load_attacks():
    require_file(
        INDEX
    )

    frame = pd.read_csv(
        INDEX,
        keep_default_na=False,
    )

    attacks = frame[
        frame[
            "traffic_type"
        ]
        == "attack"
    ].copy()

    attacks = attacks[
        attacks[
            "split"
        ].isin(
            [
                "project_train",
                "dev_val",
            ]
        )
    ].copy()

    attacks[
        "variant"
    ] = (
        attacks[
            "variant"
        ]
        .astype(str)
    )

    attacks = (
        attacks.sort_values(
            [
                "split",
                "variant",
                "file_stem",
                "hardware_source",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    counts = (
        attacks.groupby(
            [
                "split",
                "variant",
            ]
        )
        .size()
        .to_dict()
    )

    expected = {
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

    if counts != expected:
        raise RuntimeError(
            "Unexpected attack population"
            f"\nexpected: {expected}"
            f"\nactual:   {counts}"
        )

    if (
        attacks[
            attacks[
                "split"
            ]
            == "project_train"
        ][
            "file_stem"
        ].nunique()
        != 160
    ):
        raise RuntimeError(
            "Expected 160 training stems"
        )

    if (
        attacks[
            attacks[
                "split"
            ]
            == "dev_val"
        ][
            "file_stem"
        ].nunique()
        != 51
    ):
        raise RuntimeError(
            "Expected 51 dev stems"
        )

    return attacks


def attach_saved_predictions(
    attacks,
):
    require_file(
        SAVED_PREDICTIONS
    )

    saved = pd.read_csv(
        SAVED_PREDICTIONS,
        keep_default_na=False,
    )

    saved = saved[
        (
            saved[
                "label"
            ]
            == 1
        )
        &
        (
            saved[
                "split"
            ]
            .isin(
                [
                    "project_train",
                    "dev_val",
                ]
            )
        )
    ][
        [
            "image_path",
            "attack_probability",
        ]
    ].copy()

    if len(saved) != 1266:
        raise RuntimeError(
            "Expected 1266 saved attack "
            f"predictions, got {len(saved)}"
        )

    merged = attacks.merge(
        saved,
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    if (
        merged[
            "attack_probability"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Saved prediction join failed"
        )

    return merged.rename(
        columns={
            "attack_probability":
                "saved_attack_probability",
        }
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    checkpoint_sha = (
        verify_checkpoint()
    )

    attacks = load_attacks()

    attacks = (
        attach_saved_predictions(
            attacks
        )
    )

    regions = load_regions()

    (
        annotation_info,
        region_table,
        clipped_boxes,
    ) = prepare_annotation_geometry(
        attacks,
        regions,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    region_table.to_csv(
        OUT_REGIONS,
        index=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        model,
        checkpoint,
    ) = load_model(
        device
    )

    gradcam = GradCAM(
        model,
        model.layer4[-1],
    )

    dataset = PolicyCDataset(
        attacks
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
            attacks
        )
    )

    VISUAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "CLEAN RESNET18 GRAD-CAM "
        "LOCALISATION"
    )

    print(
        f"\ncheckpoint SHA:        "
        f"{checkpoint_sha}"
        f"\nselected stage:        "
        f"{checkpoint['stage']}"
        f"\nselected epoch:        "
        f"{checkpoint['epoch']}"
        f"\nproject-train attacks: "
        "960"
        f"\nheld-out dev attacks:  "
        "306"
        f"\ntotal attacks:         "
        f"{len(attacks)}"
        f"\ntarget layer:          "
        "layer4[-1]"
        f"\ntarget scalar:         "
        "attack_logit - bonafide_logit"
        f"\nclipped GT boxes:      "
        f"{clipped_boxes}"
        f"\ndevice:                "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            "gpu:                   "
            + torch.cuda
            .get_device_name(0)
        )

    records = []

    low_cams = []

    max_probability_error = (
        0.0
    )

    processed = 0

    try:
        for (
            x,
            y,
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
                    "attack_probability"
                ]
                .float()
                .cpu()
                .numpy()
            )

            margins = (
                result[
                    "attack_margin"
                ]
                .float()
                .cpu()
                .numpy()
            )

            low_batch = (
                result[
                    "low_cam"
                ]
                .float()
                .cpu()
                .numpy()
            )

            full_batch = (
                result[
                    "full_cam"
                ]
                .float()
                .cpu()
                .numpy()
            )

            for (
                batch_position,
                frame_index,
            ) in enumerate(
                indices.numpy()
            ):
                row = attacks.iloc[
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

                quantitative = (
                    compute_cam_metrics(
                        cam,
                        info,
                    )
                )

                rendered = render_images(
                    x[
                        batch_position
                    ]
                    .detach()
                    .cpu(),
                    cam,
                    info,
                )

                visual_dir = (
                    VISUAL_ROOT
                    / str(
                        row[
                            "split"
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

                rendered[
                    "overlay"
                ].save(
                    overlay_path,
                    "PNG",
                    compress_level=3,
                )

                if (
                    row[
                        "image_path"
                    ]
                    in reviewer_paths
                ):
                    panel = (
                        make_review_panel(
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
                    )

                    panel_name = (
                        safe_name(
                            row[
                                "split"
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

                records.append(
                    {
                        "image_path":
                            row[
                                "image_path"
                            ],

                        "split":
                            row[
                                "split"
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
                                probability
                                >= 0.5
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
                                "contour_area_content"
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

                        **quantitative,
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
                == len(attacks)
            ):
                print(
                    f"processed "
                    f"{processed}/"
                    f"{len(attacks)}"
                )

    finally:
        gradcam.remove()

    metrics = pd.DataFrame(
        records
    )

    if len(metrics) != 1266:
        raise RuntimeError(
            "Grad-CAM result count mismatch"
        )

    if (
        max_probability_error
        > 0.01
    ):
        raise RuntimeError(
            "Frozen Grad-CAM inference "
            "does not reproduce saved "
            "model probabilities."
            f"\nmax absolute error="
            f"{max_probability_error:.8f}"
        )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    summary = make_summary(
        metrics
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    generalisation = (
        make_generalisation_table(
            metrics
        )
    )

    generalisation.to_csv(
        OUT_GENERALISATION,
        index=False,
    )

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

        splits=(
            metrics[
                "split"
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

        hardware_sources=(
            metrics[
                "hardware_source"
            ]
            .to_numpy(
                dtype=str
            )
        ),
    )

    # -----------------------------------------------------------------
    # Inference consistency
    # -----------------------------------------------------------------

    print(
        "\nINFERENCE CONSISTENCY:"
    )

    print(
        f"  max probability error: "
        f"{max_probability_error:.8f}"
    )

    for split in [
        "project_train",
        "dev_val",
    ]:
        subset = metrics[
            metrics[
                "split"
            ]
            == split
        ]

        print(
            f"  {split}: "
            f"n={len(subset)}, "
            f"recall@0.5="
            f"{subset['predicted_attack'].mean():.4f}, "
            f"zero_CAM="
            f"{subset['cam_zero'].mean():.4f}"
        )

    # -----------------------------------------------------------------
    # Train diagnostic
    # -----------------------------------------------------------------

    train_summary = summary[
        (
            summary[
                "group_type"
            ]
            == "split_variant"
        )
        &
        (
            summary[
                "group"
            ]
            .str.startswith(
                "project_train|"
            )
        )
    ]

    print(
        "\nPROJECT-TRAIN GRAD-CAM "
        "DIAGNOSTIC:"
    )

    diagnostic_columns = [
        "group",
        "n_images",
        "n_stems",
        "mean_attack_probability",
        "mean_union_area_content",
        "mean_union_energy_content",
        "mean_union_enrichment_content",
        "mean_energy_gain_union_content",
        "energy_gain_ci_low",
        "energy_gain_ci_high",
        "pointing_union",
        "mean_point_gain_union_canvas",
        "point_gain_ci_low",
        "point_gain_ci_high",
        "mean_face_enrichment_content",
        "mean_text_enrichment_content",
        "mean_face_share_altered_energy",
        "mean_padding_energy_canvas",
        "pointing_padding",
    ]

    print(
        train_summary[
            diagnostic_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Held-out dev primary result
    # -----------------------------------------------------------------

    dev_summary = summary[
        (
            summary[
                "group_type"
            ]
            == "split_variant"
        )
        &
        (
            summary[
                "group"
            ]
            .str.startswith(
                "dev_val|"
            )
        )
    ]

    print(
        "\nHELD-OUT DEV GRAD-CAM "
        "LOCALISATION — PRIMARY:"
    )

    print(
        dev_summary[
            diagnostic_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Generalisation
    # -----------------------------------------------------------------

    print(
        "\nTRAIN -> DEV LOCALISATION "
        "GENERALISATION:"
    )

    print(
        generalisation[
            [
                "variant",

                "train_energy_gain_union_content",
                "dev_energy_gain_union_content",
                "delta_dev_minus_train_energy_gain_union_content",
                "delta_ci_low_energy_gain_union_content",
                "delta_ci_high_energy_gain_union_content",

                "train_enrichment_union_content",
                "dev_enrichment_union_content",
                "delta_dev_minus_train_enrichment_union_content",

                "train_point_union",
                "dev_point_union",
                "delta_dev_minus_train_point_union",
                "delta_ci_low_point_union",
                "delta_ci_high_point_union",

                "train_face_share_altered_energy",
                "dev_face_share_altered_energy",
                "delta_dev_minus_train_face_share_altered_energy",

                "train_energy_padding_canvas",
                "dev_energy_padding_canvas",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Visual output audit
    # -----------------------------------------------------------------

    expected_visual_files = (
        len(metrics)
        * 2
    )

    actual_heatmaps = len(
        list(
            VISUAL_ROOT.rglob(
                "*__heatmap.png"
            )
        )
    )

    actual_overlays = len(
        list(
            VISUAL_ROOT.rglob(
                "*__overlay.png"
            )
        )
    )

    reviewer_panels = len(
        list(
            REVIEW_ROOT.glob(
                "*.png"
            )
        )
    )

    print(
        "\nVISUAL OUTPUT AUDIT:"
        f"\n  expected attack images: "
        f"{len(metrics)}"
        f"\n  heatmap PNGs:           "
        f"{actual_heatmaps}"
        f"\n  overlay PNGs:           "
        f"{actual_overlays}"
        f"\n  reviewer panels:        "
        f"{reviewer_panels}"
    )

    if (
        actual_heatmaps
        != len(metrics)
        or actual_overlays
        != len(metrics)
    ):
        raise RuntimeError(
            "Visual output count mismatch"
        )

    print(
        "\nOUTPUTS:"
        f"\n  per-image metrics: "
        f"{OUT_METRICS}"
        f"\n  summary:           "
        f"{OUT_SUMMARY}"
        f"\n  train/dev delta:   "
        f"{OUT_GENERALISATION}"
        f"\n  GT region audit:   "
        f"{OUT_REGIONS}"
        f"\n  layer4 CAMs:       "
        f"{OUT_CAMS}"
        f"\n  all PNGs:          "
        f"{VISUAL_ROOT}"
        f"\n  reviewer panels:   "
        f"{REVIEW_ROOT}"
    )

    print(
        "\nVISUAL CONVENTION:"
        "\n  WHITE   = top-positive CAM contour"
        "\n  CYAN    = altered face GT rectangle"
        "\n  MAGENTA = altered text GT rectangle"
        "\n  GREY    = document/padding boundary"
        "\n  * after label = original GT box was "
        "clipped to decoded image bounds"
    )

    print(
        "\nCLEAN LOCALISATION GATE:"
        "\n"
        "\nHeld-out DEV is the primary evidence."
        "\nTraining maps are diagnostic only."
        "\n"
        "\nA useful clean Grad-CAM baseline should show:"
        "\n"
        "\n  dev energy_gain_union_content > 0"
        "\n  with stem-bootstrap CI entirely > 0"
        "\n"
        "\nand preferably:"
        "\n"
        "\n  dev point_gain_union_canvas > 0"
        "\n  with stem-bootstrap CI entirely > 0."
        "\n"
        "\nEnrichment > 1 means altered rectangles "
        "receive more CAM energy than expected "
        "from their area inside the document."
        "\n"
        "\nDo NOT treat the rectangle masks as "
        "pixel-perfect segmentation ground truth."
        "\n"
        "\nStop here before adversarial attacks."
    )


if __name__ == "__main__":
    main()