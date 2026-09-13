#!/usr/bin/env python3
"""
Evaluate the frozen native text-patch ResNet on the official FantasyID test.

IMPORTANT
---------
The text-patch cache safely covers:

    1085 / 1085 attacks
     299 /  300 official bona-fides

Therefore all comparisons involving the text-patch model use the exact same
1384-image COMMON SUBSET:

    1085 attacks + 299 bona-fides

The one uncovered bona-fide is NOT silently dropped from the historical
whole-image benchmark. For reference we also report the original whole-image
ResNet metrics on the complete 1385-image official test:

    1085 attacks + 300 bona-fides

Patch localization policy
-------------------------
Attack images:
    self ORIGINAL text annotations only.

Bona-fide images:
    ORIGINAL text coordinates transferred from an exact
    file_stem + hardware_source attack counterpart with identical dimensions.

NEVER USED:
    altered annotations
    manipulation masks
    cross-hardware transfer
    coordinate resizing / warping
    official-test tuning

Primary text score:
    maximum patch attack probability per document.

Fixed hybrid:
    max(whole_document_score, text_patch_max_score)

The max fusion is fixed a priori. No fusion weights or thresholds are selected
from the official test.

This remains an annotation-assisted diagnostic, not an end-to-end deployable
detector.
"""

import hashlib
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

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PATCH_INDEX = (
    ROOT
    / "output"
    / "text_patch512_official_index.csv"
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

CONTROLLED = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_predictions.csv"
)

AUGMENTED = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_official_test_predictions.csv"
)

OUT_PATCH = (
    ROOT
    / "output"
    / "text_patch512_official_patch_predictions.csv"
)

OUT_IMAGE = (
    ROOT
    / "output"
    / "text_patch512_official_image_predictions.csv"
)

OUT_METRICS = (
    ROOT
    / "output"
    / "text_patch512_official_hybrid_metrics.csv"
)

OUT_COMPARISON = (
    ROOT
    / "output"
    / "text_patch512_official_hybrid_comparison.csv"
)

OUT_FULL_REFERENCE = (
    ROOT
    / "output"
    / "text_patch512_official_whole_reference_1385.csv"
)

OUT_EXCLUDED = (
    ROOT
    / "output"
    / "text_patch512_official_common_subset_excluded.csv"
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 4

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

GROUPS = [
    "all",
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
]

EXPECTED_FULL = {
    "n": 1385,
    "attack": 1085,
    "bonafide": 300,
    "digital_3": 786,
    "facedancer": 150,
    "textdiffuserft_bfei": 149,
}

EXPECTED_COMMON = {
    "n": 1384,
    "attack": 1085,
    "bonafide": 299,
    "digital_3": 786,
    "facedancer": 150,
    "textdiffuserft_bfei": 149,
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require_file(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required file missing:\n{path}"
        )


def normalize_metadata(frame):
    frame = frame.copy()

    if "variant" in frame.columns:
        frame[
            "variant"
        ] = (
            frame[
                "variant"
            ]
            .fillna("")
            .astype(str)
        )

    if (
        "traffic_type"
        in frame.columns
    ):
        frame[
            "traffic_type"
        ] = (
            frame[
                "traffic_type"
            ]
            .astype(str)
        )

    if (
        "hardware_source"
        in frame.columns
    ):
        frame[
            "hardware_source"
        ] = (
            frame[
                "hardware_source"
            ]
            .astype(str)
        )

    if (
        "file_stem"
        in frame.columns
    ):
        frame[
            "file_stem"
        ] = (
            frame[
                "file_stem"
            ]
            .astype(str)
        )

    return frame


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

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
            "Patch checkpoint SHA mismatch"
            f"\nexpected: {expected}"
            f"\nactual:   {actual}"
        )

    return actual


# ---------------------------------------------------------------------
# Patch dataset
# ---------------------------------------------------------------------

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
        path = (
            ROOT
            / self.frame.iloc[
                index
            ].patch_path
        )

        with Image.open(
            path
        ) as im:
            im = im.convert(
                "RGB"
            )

            if im.size != (
                512,
                512,
            ):
                raise RuntimeError(
                    "Unexpected patch size "
                    f"{im.size}: {path}"
                )

            x = TF.to_tensor(
                im
            )

        x = TF.normalize(
            x,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        return (
            x,
            index,
        )


def build_model(device):
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


@torch.no_grad()
def predict_patches(
    model,
    frame,
    device,
):
    loader = DataLoader(
        PatchDataset(
            frame
        ),
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

    scores = np.empty(
        len(frame),
        dtype=np.float32,
    )

    for x, idx in loader:
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
            logits = model(
                x
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

        scores[
            idx.numpy()
        ] = probability

    return scores


# ---------------------------------------------------------------------
# Population validation
# ---------------------------------------------------------------------

def population_counts(frame):
    return {
        "n":
            len(frame),

        "attack":
            int(
                (
                    frame[
                        "label"
                    ]
                    == 1
                ).sum()
            ),

        "bonafide":
            int(
                (
                    frame[
                        "label"
                    ]
                    == 0
                ).sum()
            ),

        "digital_3":
            int(
                (
                    frame[
                        "variant"
                    ]
                    == "digital_3"
                ).sum()
            ),

        "facedancer":
            int(
                (
                    frame[
                        "variant"
                    ]
                    == "facedancer"
                ).sum()
            ),

        "textdiffuserft_bfei":
            int(
                (
                    frame[
                        "variant"
                    ]
                    == "textdiffuserft_bfei"
                ).sum()
            ),
    }


def validate_population(
    frame,
    expected,
    name,
):
    counts = population_counts(
        frame
    )

    if counts != expected:
        raise RuntimeError(
            f"{name} population mismatch"
            f"\nexpected: {expected}"
            f"\nactual:   {counts}"
        )

    return counts


# ---------------------------------------------------------------------
# Validate patch index
# ---------------------------------------------------------------------

def load_patch_index():
    require_file(
        PATCH_INDEX
    )

    patches = pd.read_csv(
        PATCH_INDEX,
        keep_default_na=False,
    )

    patches = normalize_metadata(
        patches
    )

    required = {
        "patch_id",
        "image_path",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
        "patch_path",
        "annotation_source",
        "annotation_provenance",
        "face_overlap_pixels",
    }

    missing = (
        required
        - set(
            patches.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Official patch index "
            "schema incomplete: "
            f"{sorted(missing)}"
        )

    if (
        patches[
            "patch_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate patch IDs"
        )

    # --------------------------------------------------
    # Hard safety assertions
    # --------------------------------------------------

    if set(
        patches[
            "annotation_provenance"
        ].unique()
    ) != {
        "original_only"
    }:
        raise RuntimeError(
            "Patch cache contains "
            "non-original annotations"
        )

    allowed_sources = {
        "self_original",
        "matched_attack_original",
    }

    observed_sources = set(
        patches[
            "annotation_source"
        ].unique()
    )

    if not (
        observed_sources
        <= allowed_sources
    ):
        raise RuntimeError(
            "Unexpected annotation source: "
            f"{observed_sources}"
        )

    if (
        patches[
            "face_overlap_pixels"
        ]
        .astype(float)
        .max()
        != 0
    ):
        raise RuntimeError(
            "Face content leaked into "
            "official text patches"
        )

    attack_rows = patches[
        patches[
            "traffic_type"
        ]
        == "attack"
    ]

    bona_rows = patches[
        patches[
            "traffic_type"
        ]
        == "bonafide"
    ]

    if set(
        attack_rows[
            "annotation_source"
        ].unique()
    ) != {
        "self_original"
    }:
        raise RuntimeError(
            "Official attacks are not "
            "self-annotated exclusively"
        )

    if set(
        bona_rows[
            "annotation_source"
        ].unique()
    ) != {
        "matched_attack_original"
    }:
        raise RuntimeError(
            "Official bona-fides are not "
            "using matched attack ORIGINAL "
            "annotations exclusively"
        )

    # --------------------------------------------------
    # Metadata must be constant within image
    # --------------------------------------------------

    metadata = [
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
        "annotation_source",
    ]

    consistency = (
        patches.groupby(
            "image_path"
        )[metadata]
        .nunique(
            dropna=False
        )
    )

    if (
        consistency
        > 1
    ).any().any():
        raise RuntimeError(
            "Patch-level metadata "
            "inconsistent within image"
        )

    n_images = (
        patches[
            "image_path"
        ].nunique()
    )

    if n_images != 1384:
        raise RuntimeError(
            "Expected exactly 1384 "
            "covered official images, got "
            f"{n_images}"
        )

    # --------------------------------------------------
    # Image-level population from patch cache
    # --------------------------------------------------

    image_meta = (
        patches[
            [
                "image_path",
                "file_stem",
                "traffic_type",
                "variant",
                "hardware_source",
                "label",
            ]
        ]
        .drop_duplicates()
    )

    if len(
        image_meta
    ) != 1384:
        raise RuntimeError(
            "Expected one metadata row "
            "per covered image"
        )

    validate_population(
        image_meta,
        EXPECTED_COMMON,
        "patch common subset",
    )

    # Every covered image must have at least one patch.
    patch_counts = (
        patches.groupby(
            "image_path"
        )
        .size()
    )

    if (
        patch_counts.min()
        < 1
    ):
        raise RuntimeError(
            "Covered image without patch"
        )

    # --------------------------------------------------
    # Annotation source image-level audit
    # --------------------------------------------------

    source_images = (
        patches[
            [
                "image_path",
                "traffic_type",
                "annotation_source",
                "annotation_source_variant",
            ]
        ]
        .drop_duplicates()
    )

    if (
        source_images[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Image has multiple annotation "
            "source identities"
        )

    source_counts = (
        source_images[
            "annotation_source"
        ]
        .value_counts()
        .to_dict()
    )

    expected_sources = {
        "self_original": 1085,
        "matched_attack_original": 299,
    }

    if (
        source_counts
        != expected_sources
    ):
        raise RuntimeError(
            "Annotation source count mismatch"
            f"\nexpected: {expected_sources}"
            f"\nactual:   {source_counts}"
        )

    bona_sources = source_images[
        source_images[
            "traffic_type"
        ]
        == "bonafide"
    ]

    donor_counts = (
        bona_sources[
            "annotation_source_variant"
        ]
        .value_counts()
        .to_dict()
    )

    expected_donors = {
        "facedancer": 150,
        "textdiffuserft_bfei": 149,
    }

    if (
        donor_counts
        != expected_donors
    ):
        raise RuntimeError(
            "Bona-fide donor-family "
            "count mismatch"
            f"\nexpected: {expected_donors}"
            f"\nactual:   {donor_counts}"
        )

    return (
        patches,
        source_counts,
        donor_counts,
    )


# ---------------------------------------------------------------------
# Whole-document predictions
# ---------------------------------------------------------------------

def load_whole_predictions(
    path,
    score_name,
):
    require_file(
        path
    )

    frame = pd.read_csv(
        path,
        keep_default_na=False,
    )

    frame = normalize_metadata(
        frame
    )

    required = {
        "image_path",
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
        "attack_probability",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Prediction schema incomplete "
            f"for {path}: "
            f"{sorted(missing)}"
        )

    if len(frame) != 1385:
        raise RuntimeError(
            f"Expected 1385 predictions "
            f"in {path}, got {len(frame)}"
        )

    if (
        frame[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Duplicate prediction paths "
            f"in {path}"
        )

    validate_population(
        frame,
        EXPECTED_FULL,
        f"whole prediction {score_name}",
    )

    selected = (
        frame[
            [
                "image_path",
                "file_stem",
                "traffic_type",
                "variant",
                "hardware_source",
                "label",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    score_name,
            }
        )
    )

    return selected


def build_full_whole_reference():
    controlled = (
        load_whole_predictions(
            CONTROLLED,
            "controlled_score",
        )
    )

    augmented = (
        load_whole_predictions(
            AUGMENTED,
            "augmented_score",
        )
    )

    # Validate that both model files describe the exact same test set.
    metadata = [
        "file_stem",
        "traffic_type",
        "variant",
        "hardware_source",
        "label",
    ]

    merged = controlled.merge(
        augmented[
            [
                "image_path",
                *metadata,
                "augmented_score",
            ]
        ],
        on="image_path",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_aug",
        ),
    )

    if len(merged) != 1385:
        raise RuntimeError(
            "Controlled/augmented "
            "official prediction sets differ"
        )

    for column in metadata:
        other = (
            f"{column}_aug"
        )

        if not (
            merged[
                column
            ]
            .astype(str)
            ==
            merged[
                other
            ]
            .astype(str)
        ).all():
            raise RuntimeError(
                "Controlled/augmented "
                "metadata mismatch: "
                f"{column}"
            )

        merged = merged.drop(
            columns=[
                other
            ]
        )

    validate_population(
        merged,
        EXPECTED_FULL,
        "full 1385 whole reference",
    )

    return merged


# ---------------------------------------------------------------------
# Aggregate patch predictions to image level
# ---------------------------------------------------------------------

def aggregate_patch_predictions(
    patches,
):
    image = (
        patches.groupby(
            "image_path",
            as_index=False,
        )
        .agg(
            file_stem=(
                "file_stem",
                "first",
            ),

            traffic_type=(
                "traffic_type",
                "first",
            ),

            variant=(
                "variant",
                "first",
            ),

            hardware_source=(
                "hardware_source",
                "first",
            ),

            label=(
                "label",
                "first",
            ),

            text_patch_max=(
                "patch_attack_probability",
                "max",
            ),

            text_patch_mean=(
                "patch_attack_probability",
                "mean",
            ),

            text_patch_median=(
                "patch_attack_probability",
                "median",
            ),

            n_text_patches=(
                "patch_id",
                "size",
            ),

            n_text_fields=(
                "field_name",
                "nunique",
            ),
        )
    )

    if len(image) != 1384:
        raise RuntimeError(
            "Patch aggregation did not "
            "produce 1384 documents"
        )

    validate_population(
        image,
        EXPECTED_COMMON,
        "aggregated patch common subset",
    )

    return image


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def metric_row(
    frame,
    score_column,
    detector,
    group,
    evaluation_population,
):
    if group == "all":
        subset = frame

    else:
        subset = frame[
            (
                frame[
                    "traffic_type"
                ]
                == "bonafide"
            )
            |
            (
                frame[
                    "variant"
                ]
                == group
            )
        ]

    y = (
        subset[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    p = (
        subset[
            score_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.isnan(p)
        .any()
    ):
        raise RuntimeError(
            f"NaN score in "
            f"{detector}/{group}"
        )

    if (
        len(
            np.unique(y)
        )
        != 2
    ):
        raise RuntimeError(
            f"Metric subset lacks both "
            f"classes: {detector}/{group}"
        )

    pred = (
        p >= 0.5
    ).astype(
        int
    )

    attack = (
        y == 1
    )

    bona = (
        y == 0
    )

    return {
        "evaluation_population":
            evaluation_population,

        "detector":
            detector,

        "group":
            group,

        "n":
            len(subset),

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


def evaluate_detectors(
    frame,
    detector_columns,
    evaluation_population,
):
    rows = []

    for (
        detector,
        column,
    ) in detector_columns:
        for group in GROUPS:
            rows.append(
                metric_row(
                    frame,
                    column,
                    detector,
                    group,
                    evaluation_population,
                )
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    checkpoint_sha = (
        verify_checkpoint()
    )

    (
        patches,
        annotation_source_counts,
        donor_family_counts,
    ) = load_patch_index()

    full = (
        build_full_whole_reference()
    )

    # --------------------------------------------------
    # Determine the exact common subset BEFORE inference
    # --------------------------------------------------

    common_paths = set(
        patches[
            "image_path"
        ].unique()
    )

    excluded = full[
        ~full[
            "image_path"
        ].isin(
            common_paths
        )
    ].copy()

    if len(excluded) != 1:
        raise RuntimeError(
            "Expected exactly one image "
            "outside the patch common subset, "
            f"got {len(excluded)}"
        )

    excluded_row = (
        excluded.iloc[0]
    )

    if (
        excluded_row[
            "traffic_type"
        ]
        != "bonafide"
        or int(
            excluded_row[
                "label"
            ]
        )
        != 0
    ):
        raise RuntimeError(
            "The single common-subset "
            "exclusion is not bona-fide"
        )

    excluded.to_csv(
        OUT_EXCLUDED,
        index=False,
    )

    # --------------------------------------------------
    # Load frozen patch model
    # --------------------------------------------------

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
        "OFFICIAL TEXT-PATCH + HYBRID "
        "EVALUATION"
    )

    print(
        f"\nPatch checkpoint SHA: "
        f"{checkpoint_sha}"
        f"\nselected stage:       "
        f"{checkpoint['stage']}"
        f"\nselected epoch:       "
        f"{checkpoint['epoch']}"
        f"\ndevice:               "
        f"{device}"
        f"\npatch rows:           "
        f"{len(patches)}"
        f"\ncommon images:        "
        f"{len(common_paths)}"
        "\ncommon composition:   "
        "1085 attack + 299 bona-fide"
        "\nfull reference:       "
        "1085 attack + 300 bona-fide"
        "\nfusion:               "
        "fixed max"
    )

    if device.type == "cuda":
        print(
            "gpu:                  "
            + torch.cuda
            .get_device_name(0)
        )

    # --------------------------------------------------
    # Annotation-assistance safety audit
    # --------------------------------------------------

    print(
        "\nANNOTATION ASSISTANCE AUDIT:"
    )

    print(
        "  provenance:          "
        "ORIGINAL ONLY"
        "\n  altered boxes used:  NO"
        "\n  face overlap:        ZERO"
        "\n  attack source:       self original"
        "\n  bona-fide source:    "
        "exact same-stem/hardware attack original"
    )

    print(
        "\n  annotation source image counts:"
    )

    for key, value in sorted(
        annotation_source_counts.items()
    ):
        print(
            f"    {key}: {value}"
        )

    print(
        "\n  bona-fide donor-family counts:"
    )

    for key, value in sorted(
        donor_family_counts.items()
    ):
        print(
            f"    {key}: {value}"
        )

    print(
        "\nCOMMON-SUBSET EXCLUDED IMAGE:"
    )

    print(
        excluded[
            [
                "image_path",
                "file_stem",
                "hardware_source",
                "traffic_type",
                "variant",
                "label",
                "controlled_score",
                "augmented_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # Patch inference
    # --------------------------------------------------

    print(
        "\nScoring frozen text patches..."
    )

    patches[
        "patch_attack_probability"
    ] = predict_patches(
        model,
        patches,
        device,
    )

    patches.to_csv(
        OUT_PATCH,
        index=False,
    )

    image = (
        aggregate_patch_predictions(
            patches
        )
    )

    # --------------------------------------------------
    # Join whole-image model scores on the common subset
    # --------------------------------------------------

    whole_common = full[
        full[
            "image_path"
        ].isin(
            common_paths
        )
    ].copy()

    validate_population(
        whole_common,
        EXPECTED_COMMON,
        "whole common subset",
    )

    image = image.merge(
        whole_common[
            [
                "image_path",
                "controlled_score",
                "augmented_score",
            ]
        ],
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    if image[
        [
            "controlled_score",
            "augmented_score",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Whole-image score merge "
            "failed on common subset"
        )

    # --------------------------------------------------
    # Fixed parameter-free fusion
    # --------------------------------------------------

    image[
        "controlled_hybrid_max"
    ] = np.maximum(
        image[
            "controlled_score"
        ],
        image[
            "text_patch_max"
        ],
    )

    image[
        "augmented_hybrid_max"
    ] = np.maximum(
        image[
            "augmented_score"
        ],
        image[
            "text_patch_max"
        ],
    )

    image.to_csv(
        OUT_IMAGE,
        index=False,
    )

    # --------------------------------------------------
    # Original 1385 whole-image reference
    # --------------------------------------------------

    full_reference_metrics = (
        evaluate_detectors(
            full,
            [
                (
                    "controlled",
                    "controlled_score",
                ),
                (
                    "augmented",
                    "augmented_score",
                ),
            ],
            "full_1385_reference",
        )
    )

    full_reference_metrics.to_csv(
        OUT_FULL_REFERENCE,
        index=False,
    )

    # --------------------------------------------------
    # Fair 1384 common-subset comparison
    # --------------------------------------------------

    common_metrics = (
        evaluate_detectors(
            image,
            [
                (
                    "text_patch_max",
                    "text_patch_max",
                ),
                (
                    "text_patch_mean",
                    "text_patch_mean",
                ),
                (
                    "controlled",
                    "controlled_score",
                ),
                (
                    "controlled+text_max",
                    "controlled_hybrid_max",
                ),
                (
                    "augmented",
                    "augmented_score",
                ),
                (
                    "augmented+text_max",
                    "augmented_hybrid_max",
                ),
            ],
            "common_1384",
        )
    )

    common_metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    # --------------------------------------------------
    # Before / after hybrid table
    # --------------------------------------------------

    comparison_rows = []

    for base, hybrid in [
        (
            "controlled",
            "controlled+text_max",
        ),
        (
            "augmented",
            "augmented+text_max",
        ),
    ]:
        for group in GROUPS:
            before = (
                common_metrics[
                    (
                        common_metrics[
                            "detector"
                        ]
                        == base
                    )
                    &
                    (
                        common_metrics[
                            "group"
                        ]
                        == group
                    )
                ]
                .iloc[0]
            )

            after = (
                common_metrics[
                    (
                        common_metrics[
                            "detector"
                        ]
                        == hybrid
                    )
                    &
                    (
                        common_metrics[
                            "group"
                        ]
                        == group
                    )
                ]
                .iloc[0]
            )

            comparison_rows.append(
                {
                    "base":
                        base,

                    "hybrid":
                        hybrid,

                    "group":
                        group,

                    "base_auroc":
                        before[
                            "auroc"
                        ],

                    "hybrid_auroc":
                        after[
                            "auroc"
                        ],

                    "delta_auroc":
                        (
                            after[
                                "auroc"
                            ]
                            -
                            before[
                                "auroc"
                            ]
                        ),

                    "base_balanced_accuracy":
                        before[
                            "balanced_accuracy"
                        ],

                    "hybrid_balanced_accuracy":
                        after[
                            "balanced_accuracy"
                        ],

                    "delta_balanced_accuracy":
                        (
                            after[
                                "balanced_accuracy"
                            ]
                            -
                            before[
                                "balanced_accuracy"
                            ]
                        ),

                    "base_attack_recall":
                        before[
                            "attack_recall"
                        ],

                    "hybrid_attack_recall":
                        after[
                            "attack_recall"
                        ],

                    "delta_attack_recall":
                        (
                            after[
                                "attack_recall"
                            ]
                            -
                            before[
                                "attack_recall"
                            ]
                        ),

                    "base_bonafide_specificity":
                        before[
                            "bonafide_specificity"
                        ],

                    "hybrid_bonafide_specificity":
                        after[
                            "bonafide_specificity"
                        ],

                    "delta_bonafide_specificity":
                        (
                            after[
                                "bonafide_specificity"
                            ]
                            -
                            before[
                                "bonafide_specificity"
                            ]
                        ),

                    "base_mean_attack_probability":
                        before[
                            "mean_attack_probability"
                        ],

                    "hybrid_mean_attack_probability":
                        after[
                            "mean_attack_probability"
                        ],

                    "base_mean_bonafide_probability":
                        before[
                            "mean_bonafide_probability"
                        ],

                    "hybrid_mean_bonafide_probability":
                        after[
                            "mean_bonafide_probability"
                        ],
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        OUT_COMPARISON,
        index=False,
    )

    # --------------------------------------------------
    # Patch-score descriptive statistics
    # --------------------------------------------------

    population = image.copy()

    population[
        "population"
    ] = population[
        "variant"
    ]

    population.loc[
        population[
            "traffic_type"
        ]
        == "bonafide",
        "population",
    ] = "bonafide"

    patch_population_summary = (
        population.groupby(
            "population"
        )
        .agg(
            n_images=(
                "image_path",
                "size",
            ),

            n_stems=(
                "file_stem",
                "nunique",
            ),

            mean_text_max=(
                "text_patch_max",
                "mean",
            ),

            median_text_max=(
                "text_patch_max",
                "median",
            ),

            mean_text_mean=(
                "text_patch_mean",
                "mean",
            ),

            median_text_mean=(
                "text_patch_mean",
                "median",
            ),

            mean_n_patches=(
                "n_text_patches",
                "mean",
            ),

            min_n_patches=(
                "n_text_patches",
                "min",
            ),

            max_n_patches=(
                "n_text_patches",
                "max",
            ),
        )
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    display_columns = [
        "detector",
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
        "\nORIGINAL 1385 WHOLE-IMAGE "
        "REFERENCE:"
    )

    print(
        full_reference_metrics[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nCOMMON 1384 — TEXT PATCH MODEL:"
    )

    print(
        common_metrics[
            common_metrics[
                "detector"
            ].isin(
                [
                    "text_patch_max",
                    "text_patch_mean",
                ]
            )
        ][
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nCOMMON 1384 — "
        "WHOLE VS FIXED HYBRID:"
    )

    print(
        common_metrics[
            common_metrics[
                "detector"
            ].isin(
                [
                    "controlled",
                    "controlled+text_max",
                    "augmented",
                    "augmented+text_max",
                ]
            )
        ][
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nHYBRID CHANGE "
        "ON COMMON 1384:"
    )

    print(
        comparison[
            [
                "base",
                "hybrid",
                "group",
                "base_auroc",
                "hybrid_auroc",
                "delta_auroc",
                "base_balanced_accuracy",
                "hybrid_balanced_accuracy",
                "delta_balanced_accuracy",
                "base_attack_recall",
                "hybrid_attack_recall",
                "delta_attack_recall",
                "base_bonafide_specificity",
                "hybrid_bonafide_specificity",
                "delta_bonafide_specificity",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nTEXT-PATCH SCORE "
        "BY POPULATION:"
    )

    print(
        patch_population_summary
        .to_string(
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nOUTPUTS:"
        f"\n  patch predictions: "
        f"{OUT_PATCH}"
        f"\n  image predictions: "
        f"{OUT_IMAGE}"
        f"\n  common metrics:    "
        f"{OUT_METRICS}"
        f"\n  hybrid comparison: "
        f"{OUT_COMPARISON}"
        f"\n  full reference:    "
        f"{OUT_FULL_REFERENCE}"
        f"\n  excluded image:    "
        f"{OUT_EXCLUDED}"
    )

    print(
        "\nINTERPRETATION RULE:"
        "\n"
        "\n1. Compare text_patch_max with whole models "
        "on Digital-3 and TextDiffuserFT."
        "\n"
        "\n2. Compare controlled+text_max and "
        "augmented+text_max with their base models "
        "ONLY on common_1384."
        "\n"
        "\n3. Facedancer should remain primarily a "
        "whole-image/face-model success. The text branch "
        "should not need to detect untouched text."
        "\n"
        "\n4. Watch bona-fide specificity. A hybrid AUROC "
        "gain bought by a large bona-fide false-positive "
        "increase is not a clean overall improvement."
        "\n"
        "\n5. Do NOT tune fusion weights or thresholds "
        "using these official-test results."
        "\n"
        "\nStop here after printing results."
    )


if __name__ == "__main__":
    main()