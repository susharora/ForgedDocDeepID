#!/usr/bin/env python3
"""
Evaluate the frozen native-text-patch ResNet on the complete official test.

No training.
No test tuning.
No altered-region annotations.

Primary patch document score:
    maximum attack probability over all eligible text patches.

Fixed hybrid:
    max(whole_document_score, text_patch_max_score)

This is an annotation-localized diagnostic because field boxes come from
ORIGINAL region metadata. It is not yet a fully automatic inference system.
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

GROUPS = [
    "all",
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
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


def verify_checkpoint():
    if not CHECKPOINT_HASH.is_file():
        raise RuntimeError(
            "Checkpoint SHA file missing. "
            "Freeze the patch checkpoint first."
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
        )

    return actual


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
                    f"Bad patch size: "
                    f"{path}"
                )

            x = TF.to_tensor(
                im
            )

        x = TF.normalize(
            x,
            mean=MEAN,
            std=STD,
        )

        return (
            x,
            index,
        )


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

    return (
        model,
        checkpoint,
    )


@torch.no_grad()
def predict(
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
            dtype=torch.float16,
            enabled=(
                device.type
                == "cuda"
            ),
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

        scores[
            idx.numpy()
        ] = probability

    return scores


def load_whole_predictions(
    path,
    score_name,
):
    frame = pd.read_csv(
        path
    )

    required = {
        "image_path",
        "attack_probability",
    }

    if not required.issubset(
        frame.columns
    ):
        raise RuntimeError(
            f"Invalid prediction file: "
            f"{path}"
        )

    if (
        frame[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Duplicate predictions: "
            f"{path}"
        )

    if len(frame) != 1385:
        raise RuntimeError(
            f"Expected 1385 predictions "
            f"in {path}, got {len(frame)}"
        )

    return (
        frame[
            [
                "image_path",
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


def metric_row(
    frame,
    score_col,
    detector,
    group,
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
        .to_numpy()
    )

    p = (
        subset[
            score_col
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

        "mean_bonafide_probability":
            float(
                p[
                    bona
                ].mean()
            ),
    }


def main():
    checkpoint_sha = (
        verify_checkpoint()
    )

    patches = pd.read_csv(
        PATCH_INDEX
    )

    if len(
        patches[
            "image_path"
        ].unique()
    ) != 1385:
        raise RuntimeError(
            "Patch index does not "
            "cover all 1385 images"
        )

    if (
        patches[
            "face_overlap_pixels"
        ].max()
        != 0
    ):
        raise RuntimeError(
            "Face content in patch index"
        )

    if set(
        patches[
            "annotation_source"
        ].unique()
    ) != {
        "original_only"
    }:
        raise RuntimeError(
            "Unexpected annotation source"
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
        "Official text-patch "
        "evaluation:"
        f"\n  checkpoint SHA: "
        f"{checkpoint_sha}"
        f"\n  selected stage: "
        f"{checkpoint['stage']}"
        f"\n  selected epoch: "
        f"{checkpoint['epoch']}"
        f"\n  patches:        "
        f"{len(patches)}"
        f"\n  images:         "
        f"{patches['image_path'].nunique()}"
        f"\n  device:         "
        f"{device}"
        "\n  crop metadata:  "
        "ORIGINAL fields only"
        "\n  altered masks:  NEVER USED"
        "\n  fusion:         fixed MAX"
    )

    scores = predict(
        model,
        patches,
        device,
    )

    patches[
        "patch_attack_probability"
    ] = scores

    patches.to_csv(
        OUT_PATCH,
        index=False,
    )

    # --------------------------------------------------
    # Aggregate text evidence per complete document
    # --------------------------------------------------

    image = (
        patches.groupby(
            [
                "image_path",
                "file_stem",
                "traffic_type",
                "variant",
                "hardware_source",
                "label",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            text_patch_max=(
                "patch_attack_probability",
                "max",
            ),

            text_patch_mean=(
                "patch_attack_probability",
                "mean",
            ),

            n_text_patches=(
                "patch_id",
                "size",
            ),
        )
    )

    if len(image) != 1385:
        raise RuntimeError(
            f"Expected 1385 image rows, "
            f"got {len(image)}"
        )

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

    image = image.merge(
        controlled,
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    image = image.merge(
        augmented,
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
            "Whole-image prediction "
            "merge failed"
        )

    # --------------------------------------------------
    # Fixed, parameter-free fusion.
    # No official-test tuning.
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

    detectors = [
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
    ]

    rows = []

    for detector, column in detectors:
        for group in GROUPS:
            rows.append(
                metric_row(
                    image,
                    column,
                    detector,
                    group,
                )
            )

    metrics = pd.DataFrame(
        rows
    )

    metrics.to_csv(
        OUT_METRICS,
        index=False,
    )

    # --------------------------------------------------
    # Explicit before/after hybrid comparison.
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
            before = metrics[
                (
                    metrics[
                        "detector"
                    ]
                    == base
                )
                &
                (
                    metrics[
                        "group"
                    ]
                    == group
                )
            ].iloc[0]

            after = metrics[
                (
                    metrics[
                        "detector"
                    ]
                    == hybrid
                )
                &
                (
                    metrics[
                        "group"
                    ]
                    == group
                )
            ].iloc[0]

            comparison_rows.append(
                {
                    "base":
                        base,

                    "hybrid":
                        hybrid,

                    "group":
                        group,

                    "base_auroc":
                        before.auroc,

                    "hybrid_auroc":
                        after.auroc,

                    "delta_auroc":
                        (
                            after.auroc
                            - before.auroc
                        ),

                    "base_balanced_accuracy":
                        before.balanced_accuracy,

                    "hybrid_balanced_accuracy":
                        after.balanced_accuracy,

                    "delta_balanced_accuracy":
                        (
                            after.balanced_accuracy
                            - before.balanced_accuracy
                        ),

                    "base_attack_recall":
                        before.attack_recall,

                    "hybrid_attack_recall":
                        after.attack_recall,

                    "delta_attack_recall":
                        (
                            after.attack_recall
                            - before.attack_recall
                        ),

                    "base_bonafide_specificity":
                        before.bonafide_specificity,

                    "hybrid_bonafide_specificity":
                        after.bonafide_specificity,

                    "delta_bonafide_specificity":
                        (
                            after.bonafide_specificity
                            - before.bonafide_specificity
                        ),
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
    # Print key results
    # --------------------------------------------------

    print(
        "\nOFFICIAL TEST — "
        "TEXT PATCH MODEL:"
    )

    print(
        metrics[
            metrics[
                "detector"
            ]
            .isin(
                [
                    "text_patch_max",
                    "text_patch_mean",
                ]
            )
        ][
            [
                "detector",
                "group",
                "auroc",
                "balanced_accuracy",
                "attack_recall",
                "bonafide_specificity",
                "mean_attack_probability",
                "mean_bonafide_probability",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nOFFICIAL TEST — "
        "WHOLE VS FIXED HYBRID:"
    )

    print(
        metrics[
            metrics[
                "detector"
            ]
            .isin(
                [
                    "controlled",
                    "controlled+text_max",
                    "augmented",
                    "augmented+text_max",
                ]
            )
        ][
            [
                "detector",
                "group",
                "auroc",
                "balanced_accuracy",
                "attack_recall",
                "bonafide_specificity",
                "mean_attack_probability",
                "mean_bonafide_probability",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nHYBRID CHANGE:"
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nTEXT-PATCH SCORE BY POPULATION:"
    )

    population = (
        image.copy()
    )

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

    print(
        population.groupby(
            "population"
        )
        .agg(
            n=(
                "image_path",
                "size",
            ),

            mean_text_max=(
                "text_patch_max",
                "mean",
            ),

            median_text_max=(
                "text_patch_max",
                "median",
            ),

            mean_n_patches=(
                "n_text_patches",
                "mean",
            ),
        )
        .to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print(
        f"\nPatch predictions: "
        f"{OUT_PATCH}"
        f"\nImage predictions: "
        f"{OUT_IMAGE}"
        f"\nMetrics:           "
        f"{OUT_METRICS}"
        f"\nComparison:        "
        f"{OUT_COMPARISON}"
    )

    print(
        "\nStop here."
        "\nDo not choose a fusion weight "
        "from the official test."
    )


if __name__ == "__main__":
    main()