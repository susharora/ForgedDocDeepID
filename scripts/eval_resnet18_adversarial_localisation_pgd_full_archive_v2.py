#!/usr/bin/env python3
"""
Full ResNet18 adversarial localisation robustness evaluation.

Frozen model
------------
runs/post_hoc_compression_controlled_resnet18_seed10/checkpoints/best.pt

SHA256
------
25ad8b1482be20e9d5e450770d6970b820ebc9470c9da3c4ec46db2558009402

Primary populations
-------------------
dev_val / digital_1          all 153
dev_val / digital_2          all 153
official_test / facedancer   all 150

Total: 456 attack images.

PGD baseline
------------
Targeted L_inf PGD toward bonafide (class 0), applied only to true
document-content pixels.

epsilon = 1/255 RGB
alpha   = 0.25/255 RGB
steps   = 10
random start
horizontal padding frozen exactly

Why 1/255
---------
The pilot tested 4/255, 2/255 and 1/255. All three produced 100% attack
success among clean-correct pilot samples. We therefore freeze the smallest
standard 8-bit-scale point tested, 1/255, rather than tuning below it to find
a model-specific transition threshold.

Grad-CAM
--------
Target layer:
    layer4[-1]

Target scalar:
    attack_logit - bonafide_logit

Localisation metrics
--------------------
A, E, mu_w and PG are computed using the exact frozen clean-RMA
compute_rma_metrics() implementation.

For E and mu_w, both signed change and positive degradation are stored:

    delta = adversarial - clean

    absolute_degradation = clean - adversarial

    relative_degradation = (clean - adversarial) / clean

PG degradation is:
    PG_clean - PG_adv

CAM-map change
--------------
Over document-content pixels:
    cosine similarity
    total variation between content-normalised CAM distributions

If either content CAM has zero mass, cosine and TV are NaN. Zero-CAM state is
reported explicitly and is part of the robustness result.

Uncertainty
-----------
Primary aggregate uncertainty uses card/file-stem cluster bootstrap 95% CIs.
Attack success is defined only among samples classified correctly as attack
before perturbation.

This script does not run an epsilon sweep.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_resnet18_gradcam_clean_rma as clean_rma
from resnet_exact_archive import save_exact_bundle


# =====================================================================
# Frozen experiment configuration
# =====================================================================

SEED = 10

EXPECTED_CHECKPOINT_SHA256 = (
    "25ad8b1482be20e9d5e450770d6970b820ebc9470c9da3c4ec46db2558009402"
)

EPSILON_PIXEL = 1.0 / 255.0
ALPHA_PIXEL = 0.25 / 255.0
PGD_STEPS = 10

ATTACK_THRESHOLD = 0.5
BATCH_SIZE = 4
N_BOOT = 5000
ZERO_CAM_EPS = 1e-12


# =====================================================================
# Outputs
# =====================================================================

OUT_ROOT = (
    ROOT
    / "output"
    / "resnet18_adversarial_localisation_pgd_eps1_full_archive_v2"
)

OUT_SELECTION = OUT_ROOT / "pgd_eps1_selection.csv"
OUT_PER_IMAGE = OUT_ROOT / "pgd_eps1_per_image.csv"
OUT_SUMMARY = OUT_ROOT / "pgd_eps1_summary.csv"
OUT_OUTCOME_SUMMARY = OUT_ROOT / "pgd_eps1_outcome_summary.csv"
OUT_CAMS = OUT_ROOT / "pgd_eps1_layer4_maps.npz"
EXACT_ARCHIVE_ROOT = OUT_ROOT / "exact_archive"
OUT_CONFIG = OUT_ROOT / "pgd_eps1_config.json"


# =====================================================================
# Normalisation tensors
# =====================================================================

IMAGENET_MEAN = torch.tensor(
    clean_rma.IMAGENET_MEAN,
    dtype=torch.float32,
).view(
    1,
    3,
    1,
    1,
)

IMAGENET_STD = torch.tensor(
    clean_rma.IMAGENET_STD,
    dtype=torch.float32,
).view(
    1,
    3,
    1,
    1,
)


# =====================================================================
# Population
# =====================================================================

def load_primary_population():
    """
    Select exactly the three frozen primary adversarial populations.
    """
    evaluation, _ = (
        clean_rma.load_evaluation_population()
    )

    keep = (
        (
            (
                evaluation["evaluation_split"]
                == "dev_val"
            )
            &
            (
                evaluation["variant"]
                .isin(
                    [
                        "digital_1",
                        "digital_2",
                    ]
                )
            )
        )
        |
        (
            (
                evaluation["evaluation_split"]
                == "official_test"
            )
            &
            (
                evaluation["variant"]
                == "facedancer"
            )
        )
    )

    frame = (
        evaluation[
            keep
        ]
        .copy()
        .sort_values(
            [
                "evaluation_split",
                "variant",
                "file_stem",
                "hardware_source",
                "image_path",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_counts = {
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

        (
            "official_test",
            "facedancer",
        ):
            150,
    }

    actual_counts = (
        frame.groupby(
            [
                "evaluation_split",
                "variant",
            ],
            sort=False,
        )
        .size()
        .to_dict()
    )

    if (
        actual_counts
        != expected_counts
    ):
        raise RuntimeError(
            "Primary population audit failed"
            f"\nexpected={expected_counts}"
            f"\nactual={actual_counts}"
        )

    if len(frame) != 456:
        raise RuntimeError(
            "Expected 456 primary images, "
            f"got {len(frame)}"
        )

    if (
        frame[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate image paths in "
            "primary population"
        )

    return frame


# =====================================================================
# Frozen annotation geometry
# =====================================================================

def prepare_geometry(frame):
    regions = (
        clean_rma.load_regions()
    )

    (
        annotation_info,
        region_table,
        clipped_count,
    ) = (
        clean_rma.prepare_annotation_geometry(
            frame,
            regions,
        )
    )

    if (
        len(annotation_info)
        != len(frame)
    ):
        raise RuntimeError(
            "Annotation geometry is not "
            "aligned with evaluation frame"
        )

    return (
        annotation_info,
        region_table,
        clipped_count,
    )


def build_content_masks(
    annotation_info,
    device,
):
    """
    Exact document-content support from the frozen clean evaluator.
    """
    masks = []

    for info in annotation_info:
        mask = (
            clean_rma.content_mask(
                info
            )
        )

        if (
            mask.shape
            != (
                clean_rma.CONTENT_H,
                clean_rma.CANVAS_W,
            )
        ):
            raise RuntimeError(
                "Unexpected content mask shape: "
                f"{mask.shape}"
            )

        masks.append(
            torch.from_numpy(
                mask.astype(
                    np.float32,
                    copy=False,
                )
            )
            .unsqueeze(0)
        )

    result = (
        torch.stack(
            masks,
            dim=0,
        )
        .to(device)
    )

    expected_shape = (
        len(annotation_info),
        1,
        clean_rma.CONTENT_H,
        clean_rma.CANVAS_W,
    )

    if (
        tuple(result.shape)
        != expected_shape
    ):
        raise RuntimeError(
            "Unexpected content-mask tensor shape"
            f"\nexpected={expected_shape}"
            f"\nactual={tuple(result.shape)}"
        )

    return result


# =====================================================================
# Perturbation audits
# =====================================================================

def verify_padding_frozen(
    clean_x,
    adv_x,
    content_mask,
):
    padding_mask = (
        1.0
        - content_mask
    )

    error = (
        (
            adv_x
            - clean_x
        )
        .abs()
        .mul(
            padding_mask
        )
        .amax()
        .item()
    )

    if error != 0.0:
        raise RuntimeError(
            "Artificial padding changed "
            "during PGD: "
            f"max_abs_error={error:.12g}"
        )

    return error


def pixel_linf(
    clean_x,
    adv_x,
    content_mask,
):
    """
    Per-image L_inf in RGB pixel units.
    """
    std = (
        IMAGENET_STD.to(
            device=clean_x.device,
            dtype=clean_x.dtype,
        )
    )

    delta_pixel = (
        (
            adv_x
            - clean_x
        )
        * std
        * content_mask
    )

    return (
        delta_pixel
        .abs()
        .flatten(1)
        .amax(
            dim=1
        )
    )


# =====================================================================
# Targeted content-only PGD
# =====================================================================

def pgd_target_bonafide(
    model,
    clean_x,
    content_mask,
    *,
    epsilon_pixel=EPSILON_PIXEL,
    alpha_pixel=ALPHA_PIXEL,
    steps=PGD_STEPS,
):
    """
    Targeted L_inf PGD toward bonafide (class 0).

    The L_inf constraint is defined in resized RGB pixel space. The
    implementation works in ImageNet-normalised coordinates using
    per-channel standard deviations.

    Artificial horizontal padding is never perturbed.
    """
    device = clean_x.device
    dtype = clean_x.dtype

    mean = (
        IMAGENET_MEAN.to(
            device=device,
            dtype=dtype,
        )
    )

    std = (
        IMAGENET_STD.to(
            device=device,
            dtype=dtype,
        )
    )

    eps_norm = (
        epsilon_pixel
        / std
    )

    alpha_norm = (
        alpha_pixel
        / std
    )

    lower_valid = (
        (
            0.0
            - mean
        )
        / std
    )

    upper_valid = (
        (
            1.0
            - mean
        )
        / std
    )

    # --------------------------------------------------------------
    # Random start inside RGB L_inf ball
    # --------------------------------------------------------------

    random_delta_pixel = (
        torch.empty_like(
            clean_x
        )
        .uniform_(
            -epsilon_pixel,
            epsilon_pixel,
        )
    )

    adv = (
        clean_x
        +
        (
            random_delta_pixel
            / std
        )
        * content_mask
    )

    # Valid RGB bounds on document content.
    clipped = torch.maximum(
        torch.minimum(
            adv,
            upper_valid,
        ),
        lower_valid,
    )

    adv = (
        clipped
        * content_mask
        +
        clean_x
        * (
            1.0
            - content_mask
        )
    )

    # Initial epsilon projection.
    delta = (
        adv
        - clean_x
    )

    delta = torch.maximum(
        torch.minimum(
            delta,
            eps_norm,
        ),
        -eps_norm,
    )

    adv = (
        clean_x
        +
        delta
        * content_mask
    )

    adv = (
        adv
        * content_mask
        +
        clean_x
        * (
            1.0
            - content_mask
        )
    )

    target = torch.zeros(
        clean_x.shape[0],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------------
    # PGD
    # --------------------------------------------------------------

    for _ in range(
        steps
    ):
        adv = (
            adv.detach()
            .requires_grad_(
                True
            )
        )

        logits = model(
            adv
        )

        # Targeted attack:
        # minimise CE to bonafide.
        loss = (
            F.cross_entropy(
                logits,
                target,
                reduction="sum",
            )
        )

        gradient = (
            torch.autograd.grad(
                loss,
                adv,
                only_inputs=True,
            )[0]
        )

        adv_next = (
            adv
            -
            alpha_norm
            * gradient.sign()
            * content_mask
        )

        # Project to clean-centred L_inf ball.
        delta = (
            adv_next
            - clean_x
        )

        delta = torch.maximum(
            torch.minimum(
                delta,
                eps_norm,
            ),
            -eps_norm,
        )

        adv_next = (
            clean_x
            +
            delta
            * content_mask
        )

        # Clamp document pixels to valid RGB.
        clipped = torch.maximum(
            torch.minimum(
                adv_next,
                upper_valid,
            ),
            lower_valid,
        )

        adv_next = (
            clipped
            * content_mask
            +
            clean_x
            * (
                1.0
                - content_mask
            )
        )

        # Reproject after valid-RGB clamp.
        delta = (
            adv_next
            - clean_x
        )

        delta = torch.maximum(
            torch.minimum(
                delta,
                eps_norm,
            ),
            -eps_norm,
        )

        adv = (
            clean_x
            +
            delta
            * content_mask
        )

        # Exact padding restoration every step.
        adv = (
            adv
            * content_mask
            +
            clean_x
            * (
                1.0
                - content_mask
            )
        )

    adv = (
        adv.detach()
    )

    verify_padding_frozen(
        clean_x,
        adv,
        content_mask,
    )

    linf = pixel_linf(
        clean_x,
        adv,
        content_mask,
    )

    if (
        linf
        >
        epsilon_pixel
        + 1e-6
    ).any():
        raise RuntimeError(
            "PGD escaped pixel-space L_inf bound: "
            f"max={linf.max().item():.8f}, "
            f"epsilon={epsilon_pixel:.8f}"
        )

    return adv


# =====================================================================
# Frozen localisation metrics
# =====================================================================

def frozen_localisation_metrics(
    cam,
    infos,
):
    """
    Run the exact clean-RMA metric implementation.
    """
    cam_np = (
        cam.detach()
        .cpu()
        .numpy()
    )

    if (
        len(cam_np)
        != len(infos)
    ):
        raise RuntimeError(
            "CAM / annotation-info "
            "batch mismatch"
        )

    rows = []

    for index, info in enumerate(
        infos
    ):
        metrics = (
            clean_rma.compute_rma_metrics(
                cam_np[
                    index
                ],
                info,
            )
        )

        rows.append(
            {
                "A":
                    float(
                        metrics[
                            "rma_A"
                        ]
                    ),

                "E":
                    float(
                        metrics[
                            "rma_E"
                        ]
                    ),

                "mu_w":
                    float(
                        metrics[
                            "rma_mu_w"
                        ]
                    ),

                "PG":
                    float(
                        metrics[
                            "rma_PG"
                        ]
                    ),
            }
        )

    return {
        key:
            torch.tensor(
                [
                    row[
                        key
                    ]
                    for row
                    in rows
                ],
                device=cam.device,
                dtype=cam.dtype,
            )
        for key
        in (
            "A",
            "E",
            "mu_w",
            "PG",
        )
    }


# =====================================================================
# CAM-map change
# =====================================================================

def cam_change_metrics(
    clean_cam,
    adv_cam,
    content_mask,
):
    """
    Compare clean/adversarial CAMs inside document content.

    Cosine and TV are undefined if either content CAM has zero mass.
    """
    mask = (
        content_mask[
            :,
            0,
        ]
    )

    clean_vector = (
        clean_cam
        * mask
    ).flatten(
        1
    )

    adv_vector = (
        adv_cam
        * mask
    ).flatten(
        1
    )

    clean_mass = (
        clean_vector.sum(
            dim=1
        )
    )

    adv_mass = (
        adv_vector.sum(
            dim=1
        )
    )

    clean_zero = (
        clean_mass
        <= ZERO_CAM_EPS
    )

    adv_zero = (
        adv_mass
        <= ZERO_CAM_EPS
    )

    valid = (
        (~clean_zero)
        &
        (~adv_zero)
    )

    cosine = torch.full(
        (
            clean_cam.shape[0],
        ),
        float("nan"),
        device=clean_cam.device,
        dtype=clean_cam.dtype,
    )

    tv = torch.full_like(
        cosine,
        float("nan"),
    )

    if valid.any():
        cosine[
            valid
        ] = (
            F.cosine_similarity(
                clean_vector[
                    valid
                ],
                adv_vector[
                    valid
                ],
                dim=1,
                eps=ZERO_CAM_EPS,
            )
        )

        clean_distribution = (
            clean_vector[
                valid
            ]
            /
            clean_mass[
                valid
            ][
                :,
                None,
            ]
        )

        adv_distribution = (
            adv_vector[
                valid
            ]
            /
            adv_mass[
                valid
            ][
                :,
                None,
            ]
        )

        tv[
            valid
        ] = (
            0.5
            *
            (
                clean_distribution
                -
                adv_distribution
            )
            .abs()
            .sum(
                dim=1
            )
        )

    return {
        "cosine":
            cosine,

        "tv":
            tv,

        "clean_zero":
            clean_zero,

        "adv_zero":
            adv_zero,

        "clean_mass":
            clean_mass,

        "adv_mass":
            adv_mass,
    }


# =====================================================================
# Scalar helpers
# =====================================================================

def safe_relative_change(
    clean_value,
    adv_value,
):
    if (
        not np.isfinite(
            clean_value
        )
        or
        abs(
            clean_value
        )
        <= ZERO_CAM_EPS
    ):
        return np.nan

    return (
        (
            adv_value
            - clean_value
        )
        /
        clean_value
    )


def safe_relative_degradation(
    clean_value,
    adv_value,
):
    if (
        not np.isfinite(
            clean_value
        )
        or
        abs(
            clean_value
        )
        <= ZERO_CAM_EPS
    ):
        return np.nan

    return (
        (
            clean_value
            - adv_value
        )
        /
        clean_value
    )


# =====================================================================
# Stem-cluster bootstrap
# =====================================================================

def stem_cluster_bootstrap_mean_ci(
    frame,
    column,
    *,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Cluster bootstrap by file_stem.

    Each sampled stem contributes all rows belonging to that stem.
    Non-finite values are excluded from the statistic.
    """
    if frame.empty:
        return (
            np.nan,
            np.nan,
        )

    groups = {
        str(stem):
            group[
                column
            ]
            .to_numpy(
                dtype=float
            )
        for stem, group
        in frame.groupby(
            "file_stem",
            sort=False,
        )
    }

    stems = np.array(
        list(
            groups.keys()
        ),
        dtype=object,
    )

    if len(stems) == 0:
        return (
            np.nan,
            np.nan,
        )

    seed_material = (
        f"{seed}|{column}|"
        f"{'|'.join(sorted(stems.astype(str)))}"
    )

    local_seed = int(
        hashlib.sha256(
            seed_material.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :16
        ],
        16,
    ) % (
        2 ** 32
    )

    rng = (
        np.random.default_rng(
            local_seed
        )
    )

    bootstrap_values = []

    for _ in range(
        n_boot
    ):
        sampled_stems = (
            rng.choice(
                stems,
                size=len(
                    stems
                ),
                replace=True,
            )
        )

        values = (
            np.concatenate(
                [
                    groups[
                        str(stem)
                    ]
                    for stem
                    in sampled_stems
                ]
            )
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values):
            bootstrap_values.append(
                float(
                    values.mean()
                )
            )

    if not bootstrap_values:
        return (
            np.nan,
            np.nan,
        )

    low, high = (
        np.quantile(
            np.asarray(
                bootstrap_values,
                dtype=float,
            ),
            [
                0.025,
                0.975,
            ],
        )
    )

    return (
        float(
            low
        ),
        float(
            high
        ),
    )


def add_mean_and_ci(
    destination,
    frame,
    column,
    output_prefix,
):
    values = (
        frame[
            column
        ]
        .to_numpy(
            dtype=float
        )
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    destination[
        f"{output_prefix}_mean"
    ] = (
        float(
            finite.mean()
        )
        if len(finite)
        else np.nan
    )

    low, high = (
        stem_cluster_bootstrap_mean_ci(
            frame,
            column,
        )
    )

    destination[
        f"{output_prefix}_ci_low"
    ] = low

    destination[
        f"{output_prefix}_ci_high"
    ] = high


# =====================================================================
# Aggregation
# =====================================================================

def make_group_summary(
    per_image,
):
    rows = []

    for (
        split,
        variant,
    ), group in (
        per_image.groupby(
            [
                "evaluation_split",
                "variant",
            ],
            sort=False,
        )
    ):
        clean_correct = (
            group[
                group[
                    "clean_correct"
                ]
            ]
            .copy()
        )

        row = {
            "evaluation_split":
                split,

            "variant":
                variant,

            "n_images":
                int(
                    len(
                        group
                    )
                ),

            "n_stems":
                int(
                    group[
                        "file_stem"
                    ]
                    .nunique()
                ),

            "n_clean_correct":
                int(
                    group[
                        "clean_correct"
                    ]
                    .sum()
                ),

            "n_adv_correct":
                int(
                    group[
                        "adv_correct"
                    ]
                    .sum()
                ),

            "n_attack_success":
                int(
                    group[
                        "attack_success"
                    ]
                    .sum()
                ),

            "clean_attack_accuracy":
                float(
                    group[
                        "clean_correct"
                    ]
                    .mean()
                ),

            "adv_attack_accuracy":
                float(
                    group[
                        "adv_correct"
                    ]
                    .mean()
                ),

            "attack_success_rate_clean_correct":
                (
                    float(
                        clean_correct[
                            "attack_success"
                        ]
                        .mean()
                    )
                    if len(
                        clean_correct
                    )
                    else np.nan
                ),

            "A_mean":
                float(
                    group[
                        "A"
                    ]
                    .mean()
                ),

            "clean_zero_cam_fraction":
                float(
                    group[
                        "clean_zero_cam"
                    ]
                    .mean()
                ),

            "adv_zero_cam_fraction":
                float(
                    group[
                        "adv_zero_cam"
                    ]
                    .mean()
                ),

            "cam_change_defined_fraction":
                float(
                    group[
                        "cam_change_defined"
                    ]
                    .mean()
                ),

            "pixel_linf_mean":
                float(
                    group[
                        "pixel_linf"
                    ]
                    .mean()
                ),

            "pixel_linf_max":
                float(
                    group[
                        "pixel_linf"
                    ]
                    .max()
                ),

            "padding_max_abs_error":
                float(
                    group[
                        "padding_max_abs_error"
                    ]
                    .max()
                ),

            "max_saved_probability_abs_error":
                float(
                    group[
                        "saved_probability_abs_error"
                    ]
                    .max()
                ),
        }

        # Attack-success CI is conditional on clean-correct samples.
        if len(
            clean_correct
        ):
            asr_low, asr_high = (
                stem_cluster_bootstrap_mean_ci(
                    clean_correct,
                    "attack_success",
                )
            )
        else:
            asr_low, asr_high = (
                np.nan,
                np.nan,
            )

        row[
            "attack_success_ci_low"
        ] = asr_low

        row[
            "attack_success_ci_high"
        ] = asr_high

        # Primary classification outputs.
        for column, prefix in [
            (
                "clean_probability_attack",
                "clean_p_attack",
            ),
            (
                "adv_probability_attack",
                "adv_p_attack",
            ),
            (
                "clean_margin",
                "clean_margin",
            ),
            (
                "adv_margin",
                "adv_margin",
            ),
        ]:
            add_mean_and_ci(
                row,
                group,
                column,
                prefix,
            )

        # Clean / adversarial localisation.
        for column, prefix in [
            (
                "E_clean",
                "E_clean",
            ),
            (
                "E_adv",
                "E_adv",
            ),
            (
                "mu_w_clean",
                "mu_w_clean",
            ),
            (
                "mu_w_adv",
                "mu_w_adv",
            ),
            (
                "PG_clean",
                "PG_clean",
            ),
            (
                "PG_adv",
                "PG_adv",
            ),
        ]:
            add_mean_and_ci(
                row,
                group,
                column,
                prefix,
            )

        # Change and degradation.
        for column, prefix in [
            (
                "delta_E",
                "delta_E",
            ),
            (
                "absolute_E_degradation",
                "absolute_E_degradation",
            ),
            (
                "relative_E_change",
                "relative_E_change",
            ),
            (
                "relative_E_degradation",
                "relative_E_degradation",
            ),
            (
                "delta_mu_w",
                "delta_mu_w",
            ),
            (
                "absolute_mu_w_degradation",
                "absolute_mu_w_degradation",
            ),
            (
                "relative_mu_w_change",
                "relative_mu_w_change",
            ),
            (
                "relative_mu_w_degradation",
                "relative_mu_w_degradation",
            ),
            (
                "delta_PG",
                "delta_PG",
            ),
            (
                "absolute_PG_degradation",
                "absolute_PG_degradation",
            ),
        ]:
            add_mean_and_ci(
                row,
                group,
                column,
                prefix,
            )

        # CAM change is averaged only over rows where it is defined.
        for column, prefix in [
            (
                "cam_cosine",
                "cam_cosine",
            ),
            (
                "cam_tv",
                "cam_tv",
            ),
        ]:
            add_mean_and_ci(
                row,
                group,
                column,
                prefix,
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def make_outcome_summary(
    per_image,
):
    """
    Diagnostic split among clean-correct cases:
        classification attack succeeded
        classification attack failed

    With a saturated attack, the failure stratum may be empty.
    """
    base = (
        per_image[
            per_image[
                "clean_correct"
            ]
        ]
        .copy()
    )

    if base.empty:
        return pd.DataFrame()

    rows = []

    for (
        split,
        variant,
        attack_success,
    ), group in (
        base.groupby(
            [
                "evaluation_split",
                "variant",
                "attack_success",
            ],
            sort=False,
        )
    ):
        row = {
            "evaluation_split":
                split,

            "variant":
                variant,

            "attack_success":
                bool(
                    attack_success
                ),

            "n_images":
                int(
                    len(
                        group
                    )
                ),

            "n_stems":
                int(
                    group[
                        "file_stem"
                    ]
                    .nunique()
                ),

            "adv_zero_cam_fraction":
                float(
                    group[
                        "adv_zero_cam"
                    ]
                    .mean()
                ),

            "cam_change_defined_fraction":
                float(
                    group[
                        "cam_change_defined"
                    ]
                    .mean()
                ),
        }

        for column, prefix in [
            (
                "adv_probability_attack",
                "adv_p_attack",
            ),
            (
                "absolute_E_degradation",
                "absolute_E_degradation",
            ),
            (
                "relative_E_degradation",
                "relative_E_degradation",
            ),
            (
                "absolute_mu_w_degradation",
                "absolute_mu_w_degradation",
            ),
            (
                "relative_mu_w_degradation",
                "relative_mu_w_degradation",
            ),
            (
                "absolute_PG_degradation",
                "absolute_PG_degradation",
            ),
            (
                "cam_cosine",
                "cam_cosine",
            ),
            (
                "cam_tv",
                "cam_tv",
            ),
        ]:
            add_mean_and_ci(
                row,
                group,
                column,
                prefix,
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Main
# =====================================================================

def main():
    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_sha = (
        clean_rma.verify_checkpoint()
    )

    if (
        checkpoint_sha
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Wrong frozen ResNet checkpoint"
            f"\nexpected: {EXPECTED_CHECKPOINT_SHA256}"
            f"\nactual:   {checkpoint_sha}"
        )

    clean_rma.verify_inventory()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, _ = (
        clean_rma.load_model(
            device
        )
    )

    frame = (
        load_primary_population()
    )

    (
        annotation_info,
        region_table,
        clipped_count,
    ) = (
        prepare_geometry(
            frame
        )
    )

    frame.to_csv(
        OUT_SELECTION,
        index=False,
    )

    config = {
        "checkpoint_sha256":
            checkpoint_sha,

        "seed":
            SEED,

        "epsilon_pixel":
            EPSILON_PIXEL,

        "epsilon_255":
            EPSILON_PIXEL
            * 255.0,

        "alpha_pixel":
            ALPHA_PIXEL,

        "alpha_255":
            ALPHA_PIXEL
            * 255.0,

        "pgd_steps":
            PGD_STEPS,

        "random_start":
            True,

        "attack_target":
            "bonafide_class_0",

        "attack_support":
            "document_content_only",

        "padding_frozen":
            True,

        "gradcam_layer":
            "layer4[-1]",

        "gradcam_target":
            "attack_logit - bonafide_logit",

        "n_images":
            int(
                len(
                    frame
                )
            ),

        "n_boot":
            N_BOOT,
    }

    OUT_CONFIG.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    dataset = (
        clean_rma.PolicyCDataset(
            frame
        )
    )

    loader = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
    )

    all_content_masks = (
        build_content_masks(
            annotation_info,
            device,
        )
    )

    gradcam = (
        clean_rma.GradCAM(
            model,
            model.layer4[-1],
        )
    )

    records = []

    cam_keys = []
    clean_low_cams = []
    adv_low_cams = []

    processed = 0

    try:
        for (
            batch_x,
            labels,
            indices,
        ) in loader:
            batch_x = (
                batch_x.to(
                    device
                )
            )

            labels = (
                labels.to(
                    device
                )
            )

            batch_indices = (
                indices.tolist()
            )

            if not torch.all(
                labels
                == 1
            ):
                raise RuntimeError(
                    "Primary population contains "
                    "a non-attack sample"
                )

            index_tensor = (
                indices.to(
                    device
                )
            )

            content_mask = (
                all_content_masks[
                    index_tensor
                ]
            )

            batch_infos = [
                annotation_info[
                    index
                ]
                for index
                in batch_indices
            ]

            # ------------------------------------------------------
            # Clean state
            # ------------------------------------------------------

            clean_result = (
                gradcam.generate(
                    batch_x
                )
            )

            clean_probability = (
                clean_result[
                    "probability"
                ]
            )

            clean_margin = (
                clean_result[
                    "margin"
                ]
            )

            clean_cam = (
                clean_result[
                    "full_cam"
                ]
            )

            clean_metrics = (
                frozen_localisation_metrics(
                    clean_cam,
                    batch_infos,
                )
            )

            # ------------------------------------------------------
            # Adversarial example
            # ------------------------------------------------------

            adv_x = (
                pgd_target_bonafide(
                    model,
                    batch_x,
                    content_mask,
                )
            )

            padding_error = (
                verify_padding_frozen(
                    batch_x,
                    adv_x,
                    content_mask,
                )
            )

            linf = (
                pixel_linf(
                    batch_x,
                    adv_x,
                    content_mask,
                )
            )

            # ------------------------------------------------------
            # Adversarial state
            # ------------------------------------------------------

            adv_result = (
                gradcam.generate(
                    adv_x
                )
            )

            adv_probability = (
                adv_result[
                    "probability"
                ]
            )

            adv_margin = (
                adv_result[
                    "margin"
                ]
            )

            adv_cam = (
                adv_result[
                    "full_cam"
                ]
            )

            adv_metrics = (
                frozen_localisation_metrics(
                    adv_cam,
                    batch_infos,
                )
            )

            # A must be invariant because geometry did not change.
            A_error = (
                (
                    clean_metrics[
                        "A"
                    ]
                    -
                    adv_metrics[
                        "A"
                    ]
                )
                .abs()
                .max()
                .item()
            )

            if (
                A_error
                > 1e-7
            ):
                raise RuntimeError(
                    "Clean/adversarial A mismatch: "
                    f"max_abs_error={A_error:.12g}"
                )

            cam_change = (
                cam_change_metrics(
                    clean_cam,
                    adv_cam,
                    content_mask,
                )
            )

            # ------------------------------------------------------
            # Per-image records
            # ------------------------------------------------------

            for (
                j,
                global_index,
            ) in enumerate(
                batch_indices
            ):
                row = (
                    frame.iloc[
                        global_index
                    ]
                )

                clean_p = (
                    clean_probability[
                        j
                    ]
                    .item()
                )

                adv_p = (
                    adv_probability[
                        j
                    ]
                    .item()
                )

                clean_correct = bool(
                    clean_p
                    >= ATTACK_THRESHOLD
                )

                adv_correct = bool(
                    adv_p
                    >= ATTACK_THRESHOLD
                )

                attack_success = bool(
                    clean_correct
                    and
                    not adv_correct
                )

                A = (
                    clean_metrics[
                        "A"
                    ][
                        j
                    ]
                    .item()
                )

                E_clean = (
                    clean_metrics[
                        "E"
                    ][
                        j
                    ]
                    .item()
                )

                E_adv = (
                    adv_metrics[
                        "E"
                    ][
                        j
                    ]
                    .item()
                )

                mu_clean = (
                    clean_metrics[
                        "mu_w"
                    ][
                        j
                    ]
                    .item()
                )

                mu_adv = (
                    adv_metrics[
                        "mu_w"
                    ][
                        j
                    ]
                    .item()
                )

                PG_clean = (
                    clean_metrics[
                        "PG"
                    ][
                        j
                    ]
                    .item()
                )

                PG_adv = (
                    adv_metrics[
                        "PG"
                    ][
                        j
                    ]
                    .item()
                )

                saved_probability = float(
                    row[
                        "saved_attack_probability"
                    ]
                )

                clean_zero = bool(
                    cam_change[
                        "clean_zero"
                    ][
                        j
                    ]
                    .item()
                )

                adv_zero = bool(
                    cam_change[
                        "adv_zero"
                    ][
                        j
                    ]
                    .item()
                )

                exact_archive = save_exact_bundle(
                    repo_root=ROOT,
                    archive_root=EXACT_ARCHIVE_ROOT,
                    row=row,
                    info=batch_infos[j],
                    clean_input=batch_x[j],
                    adv_input=adv_x[j],
                    clean_cam=clean_result["full_cam"][j],
                    adv_cam=adv_result["full_cam"][j],
                    clean_probability=clean_p,
                    adv_probability=adv_p,
                    clean_margin=clean_margin[j].item(),
                    adv_margin=adv_margin[j].item(),
                    attack_metadata={
                        "attack_type": "targeted_bonafide_pgd",
                        "epsilon_pixel": EPSILON_PIXEL,
                        "epsilon_255": EPSILON_PIXEL * 255.0,
                        "alpha_pixel": ALPHA_PIXEL,
                        "alpha_255": ALPHA_PIXEL * 255.0,
                        "steps": PGD_STEPS,
                        "random_start": True,
                        "attack_target": "bonafide_class_0",
                        "attack_support": "document_content_only",
                        "padding_frozen": True,
                    },
                )

                record = {
                    "evaluation_split":
                        row[
                            "evaluation_split"
                        ],

                    "variant":
                        row[
                            "variant"
                        ],

                    "selection_scope":
                        row[
                            "selection_scope"
                        ],

                    "evidence_role":
                        row[
                            "evidence_role"
                        ],

                    "file_stem":
                        row[
                            "file_stem"
                        ],

                    "hardware_source":
                        row[
                            "hardware_source"
                        ],

                    "image_path":
                        row[
                            "image_path"
                        ],

                    "label":
                        int(
                            labels[
                                j
                            ]
                            .item()
                        ),

                    "epsilon_pixel":
                        EPSILON_PIXEL,

                    "epsilon_255":
                        EPSILON_PIXEL
                        * 255.0,

                    "alpha_pixel":
                        ALPHA_PIXEL,

                    "alpha_255":
                        ALPHA_PIXEL
                        * 255.0,

                    "pgd_steps":
                        PGD_STEPS,

                    "pixel_linf":
                        linf[
                            j
                        ]
                        .item(),

                    "padding_max_abs_error":
                        padding_error,

                    "saved_clean_probability_attack":
                        saved_probability,

                    "clean_probability_attack":
                        clean_p,

                    "saved_probability_abs_error":
                        abs(
                            clean_p
                            - saved_probability
                        ),

                    "adv_probability_attack":
                        adv_p,

                    "clean_margin":
                        clean_margin[
                            j
                        ]
                        .item(),

                    "adv_margin":
                        adv_margin[
                            j
                        ]
                        .item(),

                    "clean_correct":
                        clean_correct,

                    "adv_correct":
                        adv_correct,

                    "attack_success":
                        attack_success,

                    "A":
                        A,

                    # ----------------------------------------------
                    # E
                    # ----------------------------------------------

                    "E_clean":
                        E_clean,

                    "E_adv":
                        E_adv,

                    "delta_E":
                        (
                            E_adv
                            - E_clean
                        ),

                    "absolute_E_degradation":
                        (
                            E_clean
                            - E_adv
                        ),

                    "relative_E_change":
                        safe_relative_change(
                            E_clean,
                            E_adv,
                        ),

                    "relative_E_degradation":
                        safe_relative_degradation(
                            E_clean,
                            E_adv,
                        ),

                    # ----------------------------------------------
                    # mu_w
                    # ----------------------------------------------

                    "mu_w_clean":
                        mu_clean,

                    "mu_w_adv":
                        mu_adv,

                    "delta_mu_w":
                        (
                            mu_adv
                            - mu_clean
                        ),

                    "absolute_mu_w_degradation":
                        (
                            mu_clean
                            - mu_adv
                        ),

                    "relative_mu_w_change":
                        safe_relative_change(
                            mu_clean,
                            mu_adv,
                        ),

                    "relative_mu_w_degradation":
                        safe_relative_degradation(
                            mu_clean,
                            mu_adv,
                        ),

                    # ----------------------------------------------
                    # PG
                    # ----------------------------------------------

                    "PG_clean":
                        PG_clean,

                    "PG_adv":
                        PG_adv,

                    "delta_PG":
                        (
                            PG_adv
                            - PG_clean
                        ),

                    "absolute_PG_degradation":
                        (
                            PG_clean
                            - PG_adv
                        ),

                    # ----------------------------------------------
                    # CAM map
                    # ----------------------------------------------

                    "cam_cosine":
                        cam_change[
                            "cosine"
                        ][
                            j
                        ]
                        .item(),

                    "cam_tv":
                        cam_change[
                            "tv"
                        ][
                            j
                        ]
                        .item(),

                    "clean_zero_cam":
                        clean_zero,

                    "adv_zero_cam":
                        adv_zero,

                    "cam_change_defined":
                        bool(
                            (not clean_zero)
                            and
                            (not adv_zero)
                        ),

                    "clean_content_cam_mass":
                        cam_change[
                            "clean_mass"
                        ][
                            j
                        ]
                        .item(),

                    "adv_content_cam_mass":
                        cam_change[
                            "adv_mass"
                        ][
                            j
                        ]
                        .item(),

                    "exact_bundle_path": exact_archive["exact_bundle_path"],
                    "exact_bundle_sha256": exact_archive["exact_bundle_sha256"],
                    "exact_bundle_bytes": exact_archive["exact_bundle_bytes"],
                    "exact_input_dtype": exact_archive["exact_input_dtype"],
                    "exact_cam_dtype": exact_archive["exact_cam_dtype"],
                    "exact_cam_height": exact_archive["exact_cam_height"],
                    "exact_cam_width": exact_archive["exact_cam_width"],
                    "exact_archive_schema": exact_archive["exact_archive_schema"],
                }

                records.append(
                    record
                )

                cam_keys.append(
                    str(
                        row[
                            "image_path"
                        ]
                    )
                )

                clean_low_cams.append(
                    clean_result[
                        "low_cam"
                    ][
                        j
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float16
                    )
                )

                adv_low_cams.append(
                    adv_result[
                        "low_cam"
                    ][
                        j
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float16
                    )
                )

            processed += len(
                batch_indices
            )

            print(
                f"processed "
                f"{processed}/"
                f"{len(frame)}",
                flush=True,
            )

    finally:
        gradcam.remove()

    # =================================================================
    # Save per-image outputs
    # =================================================================

    per_image = pd.DataFrame(
        records
    )

    if (
        len(per_image)
        != len(frame)
    ):
        raise RuntimeError(
            "Per-image output count mismatch"
            f"\nexpected={len(frame)}"
            f"\nactual={len(per_image)}"
        )

    per_image.to_csv(
        OUT_PER_IMAGE,
        index=False,
    )

    summary = (
        make_group_summary(
            per_image
        )
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    outcome_summary = (
        make_outcome_summary(
            per_image
        )
    )

    outcome_summary.to_csv(
        OUT_OUTCOME_SUMMARY,
        index=False,
    )

    np.savez_compressed(
        OUT_CAMS,
        keys=np.array(
            cam_keys,
            dtype=object,
        ),
        clean_maps=np.stack(
            clean_low_cams
        ),
        adv_maps=np.stack(
            adv_low_cams
        ),
    )

    # =================================================================
    # Final audits
    # =================================================================

    max_linf = float(
        per_image[
            "pixel_linf"
        ]
        .max()
    )

    if (
        max_linf
        >
        EPSILON_PIXEL
        + 1e-6
    ):
        raise RuntimeError(
            "Final L_inf audit failed: "
            f"{max_linf:.8f}"
        )

    max_padding_error = float(
        per_image[
            "padding_max_abs_error"
        ]
        .max()
    )

    if (
        max_padding_error
        != 0.0
    ):
        raise RuntimeError(
            "Final padding audit failed: "
            f"{max_padding_error}"
        )

    # =================================================================
    # Terminal report
    # =================================================================

    print()
    print(
        "FULL RESNET18 ADVERSARIAL "
        "LOCALISATION EVALUATION"
    )
    print()

    print(
        f"checkpoint SHA: "
        f"{checkpoint_sha}"
    )

    print(
        f"device:         "
        f"{device}"
    )

    if (
        device.type
        == "cuda"
    ):
        print(
            f"gpu:            "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"evaluated images: "
        f"{len(frame)}"
    )

    print(
        f"clipped GT rectangles: "
        f"{clipped_count}"
    )

    print(
        "attack: targeted bonafide "
        "pixel-space L_inf PGD"
    )

    print(
        f"epsilon: "
        f"{EPSILON_PIXEL * 255.0:.2f}/255"
    )

    print(
        f"alpha:   "
        f"{ALPHA_PIXEL * 255.0:.2f}/255"
    )

    print(
        f"steps:   "
        f"{PGD_STEPS}"
    )

    print(
        "random start: yes"
    )

    print(
        "attack support: "
        "document content only"
    )

    print(
        "padding: frozen exactly"
    )

    print(
        "localisation evaluator: "
        "frozen clean_rma.compute_rma_metrics"
    )

    print(
        "uncertainty: "
        "file-stem cluster bootstrap 95% CI"
    )

    print(
        f"max pixel L_inf: "
        f"{max_linf:.8f}"
    )

    print(
        f"max padding change: "
        f"{max_padding_error:.12g}"
    )

    print(
        "max clean-vs-saved "
        "probability error: "
        f"{per_image['saved_probability_abs_error'].max():.8f}"
    )

    print()
    print(
        "PRIMARY FULL SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "CLEAN-CORRECT OUTCOME DIAGNOSTIC:"
    )

    if outcome_summary.empty:
        print(
            "(no clean-correct images)"
        )
    else:
        print(
            outcome_summary.to_string(
                index=False
            )
        )

    print()
    print(
        "OUTPUTS:"
    )

    print(
        f"  config:          "
        f"{OUT_CONFIG}"
    )

    print(
        f"  selection:       "
        f"{OUT_SELECTION}"
    )

    print(
        f"  per image:       "
        f"{OUT_PER_IMAGE}"
    )

    print(
        f"  summary:         "
        f"{OUT_SUMMARY}"
    )

    print(
        f"  outcome summary: "
        f"{OUT_OUTCOME_SUMMARY}"
    )

    print(
        f"  CAMs:            "
        f"{OUT_CAMS}"
    )

    print()
    print(
        "STOP HERE."
    )

    print(
        "Next experiment is the "
        "classification-preserving "
        "localisation attack."
    )


if __name__ == "__main__":
    main()
