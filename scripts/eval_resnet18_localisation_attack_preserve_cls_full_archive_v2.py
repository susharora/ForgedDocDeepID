#!/usr/bin/env python3
"""
ResNet18 classification-preserving Grad-CAM localisation attack — full run.

Purpose
-------
The frozen classification-breaking PGD baseline at epsilon=1/255 succeeds on
every clean-correct primary sample. This second attack isolates explanation
robustness by enforcing correct ATTACK classification while directly degrading
Grad-CAM localisation.

Evaluation population
---------------------
Clean-correct primary attack samples only:

    dev_val / digital_1          153
    dev_val / digital_2          153
    official_test / facedancer   144

Total: 450 images.

Perturbation budget
-------------------
Same frozen budget as the primary PGD baseline:

    L_inf epsilon = 1/255 in resized RGB pixel space
    step size      = 0.25/255
    steps          = 10

Only document-content pixels may change. Artificial horizontal padding is
frozen exactly.

Classification constraint
-------------------------
The attack starts from the clean image and preserves

    attack_logit - bonafide_logit >= 0

at every accepted iterate.

A candidate that crosses the classification boundary is pulled back along the
step direction by feasibility bisection. A candidate is accepted only when:

    1. classification remains ATTACK, and
    2. the Grad-CAM relevance-mass objective E does not increase.

Localisation objective
----------------------
Directly minimise differentiable Grad-CAM relevance mass in the altered union:

    E = CAM mass inside altered union / CAM mass inside document content

Grad-CAM:
    target layer  = layer4[-1]
    target scalar = attack_logit - bonafide_logit

Optimising E differentiates through Grad-CAM gradient weights, so this is a
second-order white-box explanation attack.

Final evaluation
----------------
Clean/adversarial A, E, mu_w and PG are evaluated using the exact frozen
clean_rma.compute_rma_metrics() implementation. The frozen clean Grad-CAM hook
is scoped only to evaluation calls and removed before no-grad feasibility
checks.

Also report:
    classification preservation
    clean/adversarial attack probability and margin
    absolute / relative E degradation
    absolute / relative mu_w degradation
    PG degradation
    CAM cosine similarity
    CAM total variation
    zero-CAM frequency
    realised L_inf
    accepted-step count
    classification-boundary pull-backs

Uncertainty
-----------
Final population summaries use file-stem cluster bootstrap 95% confidence
intervals for primary means/fractions.

This script does not tune epsilon or attack hyperparameters.
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
ATTACK_STEPS = 10

CLASSIFICATION_MARGIN_FLOOR = 0.0
BISECTION_STEPS = 8

N_BOOT = 5000
BATCH_SIZE = 1

ATTACK_THRESHOLD = 0.5
ZERO_CAM_EPS = 1e-12
OBJECTIVE_TOL = 1e-8


# =====================================================================
# Outputs
# =====================================================================

OUT_ROOT = (
    ROOT
    / "output"
    / "resnet18_localisation_attack_preserve_cls_full_archive_v2"
)

OUT_CONFIG = OUT_ROOT / "localisation_attack_full_config.json"
OUT_SELECTION = OUT_ROOT / "localisation_attack_full_selection.csv"
OUT_PER_IMAGE = OUT_ROOT / "localisation_attack_full_per_image.csv"
OUT_SUMMARY = OUT_ROOT / "localisation_attack_full_summary.csv"
OUT_CAMS = OUT_ROOT / "localisation_attack_full_layer4_maps.npz"
EXACT_ARCHIVE_ROOT = OUT_ROOT / "exact_archive"


# =====================================================================
# Normalisation tensors
# =====================================================================

IMAGENET_MEAN = torch.tensor(
    clean_rma.IMAGENET_MEAN,
    dtype=torch.float32,
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    clean_rma.IMAGENET_STD,
    dtype=torch.float32,
).view(1, 3, 1, 1)


# =====================================================================
# Full clean-correct primary population
# =====================================================================

def load_clean_correct_population():
    """
    Use every clean-correct attack sample in the three primary populations.

    Clean-incorrect images are excluded by definition: an explanation attack
    cannot be called classification-preserving when the unperturbed detector
    is already wrong.
    """
    evaluation, _ = (
        clean_rma.load_evaluation_population()
    )

    specs = [
        (
            "dev_val",
            "digital_1",
            153,
        ),
        (
            "dev_val",
            "digital_2",
            153,
        ),
        (
            "official_test",
            "facedancer",
            144,
        ),
    ]

    groups = []

    for (
        split,
        variant,
        expected_clean_correct,
    ) in specs:
        group = evaluation[
            (
                evaluation[
                    "evaluation_split"
                ]
                == split
            )
            &
            (
                evaluation[
                    "variant"
                ]
                == variant
            )
            &
            (
                evaluation[
                    "saved_attack_probability"
                ]
                >= ATTACK_THRESHOLD
            )
        ].copy()

        if (
            len(group)
            != expected_clean_correct
        ):
            raise RuntimeError(
                "Unexpected clean-correct population "
                f"for {split}/{variant}: "
                f"expected={expected_clean_correct}, "
                f"actual={len(group)}"
            )

        groups.append(
            group
        )

    frame = (
        pd.concat(
            groups,
            ignore_index=True,
        )
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
            144,
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
            "Full clean-correct population audit failed"
            f"\nexpected={expected_counts}"
            f"\nactual={actual_counts}"
        )

    if len(frame) != 450:
        raise RuntimeError(
            "Expected 450 clean-correct primary samples, "
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
            "Duplicate image paths in full population"
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
            "aligned with pilot dataframe"
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
    masks = []

    for info in annotation_info:
        mask = (
            clean_rma.content_mask(
                info
            )
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
            "Unexpected content-mask shape "
            f"{tuple(result.shape)}"
        )

    return result


def build_altered_union_masks(
    annotation_info,
    device,
):
    masks = []

    for info in annotation_info:
        face = (
            clean_rma.mask_from_rectangles(
                info[
                    "face_rects"
                ]
            )
        )

        text = (
            clean_rma.mask_from_rectangles(
                info[
                    "text_rects"
                ]
            )
        )

        union = (
            face
            | text
        )

        content = (
            clean_rma.content_mask(
                info
            )
        )

        union = (
            union
            & content
        )

        if (
            union.sum()
            == 0
        ):
            raise RuntimeError(
                "Empty altered union"
            )

        masks.append(
            torch.from_numpy(
                union.astype(
                    np.float32,
                    copy=False,
                )
            )
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
        clean_rma.CONTENT_H,
        clean_rma.CANVAS_W,
    )

    if (
        tuple(result.shape)
        != expected_shape
    ):
        raise RuntimeError(
            "Unexpected altered-mask shape "
            f"{tuple(result.shape)}"
        )

    return result


# =====================================================================
# Pixel-space projection / audits
# =====================================================================

def verify_padding_frozen(
    clean_x,
    adv_x,
    content_mask,
):
    error = (
        (
            adv_x
            - clean_x
        )
        .abs()
        .mul(
            1.0
            - content_mask
        )
        .amax()
        .item()
    )

    if error != 0.0:
        raise RuntimeError(
            "Artificial padding changed: "
            f"max_abs_error={error:.12g}"
        )

    return error


def pixel_linf(
    clean_x,
    adv_x,
    content_mask,
):
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


def project_to_feasible_pixel_ball(
    candidate,
    clean_x,
    content_mask,
    *,
    epsilon_pixel=EPSILON_PIXEL,
):
    """
    Project onto:
        - valid RGB [0,1]
        - clean-centred pixel L_inf ball
        - frozen padding
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

    delta = (
        candidate
        - clean_x
    )

    delta = torch.maximum(
        torch.minimum(
            delta,
            eps_norm,
        ),
        -eps_norm,
    )

    projected = (
        clean_x
        +
        delta
        * content_mask
    )

    clipped = torch.maximum(
        torch.minimum(
            projected,
            upper_valid,
        ),
        lower_valid,
    )

    projected = (
        clipped
        * content_mask
        +
        clean_x
        * (
            1.0
            - content_mask
        )
    )

    # Re-project after RGB clipping.
    delta = (
        projected
        - clean_x
    )

    delta = torch.maximum(
        torch.minimum(
            delta,
            eps_norm,
        ),
        -eps_norm,
    )

    projected = (
        clean_x
        +
        delta
        * content_mask
    )

    projected = (
        projected
        * content_mask
        +
        clean_x
        * (
            1.0
            - content_mask
        )
    )

    return projected


# =====================================================================
# Classification helpers
# =====================================================================

def model_margin(
    model,
    x,
):
    logits = model(
        x
    )

    return (
        logits[
            :,
            1,
        ]
        -
        logits[
            :,
            0,
        ]
    )


def classification_feasible(
    model,
    x,
):
    with torch.no_grad():
        margin = model_margin(
            model,
            x,
        )

    return (
        margin
        >= CLASSIFICATION_MARGIN_FLOOR
    ), margin


def pull_back_to_classification_boundary(
    model,
    current,
    candidate,
    clean_x,
    content_mask,
):
    """
    Return a classification-feasible interpolation between current and
    candidate.

    current must already be feasible.

    For samples where candidate is feasible, candidate is retained.
    For infeasible samples, keep the largest feasible point found by bisection
    on the segment current -> candidate.

    We do not assume global monotonicity of margin on the segment. The lower
    endpoint is explicitly maintained as feasible at every iteration, so the
    returned point is always feasible.
    """
    candidate = (
        project_to_feasible_pixel_ball(
            candidate,
            clean_x,
            content_mask,
        )
    )

    feasible, _ = (
        classification_feasible(
            model,
            candidate,
        )
    )

    if feasible.all():
        return (
            candidate,
            feasible,
        )

    low = current.clone()
    high = candidate.clone()

    needs_search = (
        ~feasible
    )

    for _ in range(
        BISECTION_STEPS
    ):
        mid = (
            0.5
            * (
                low
                + high
            )
        )

        mid = (
            project_to_feasible_pixel_ball(
                mid,
                clean_x,
                content_mask,
            )
        )

        mid_feasible, _ = (
            classification_feasible(
                model,
                mid,
            )
        )

        accept_mid = (
            needs_search
            &
            mid_feasible
        )

        reject_mid = (
            needs_search
            &
            (~mid_feasible)
        )

        if accept_mid.any():
            low[
                accept_mid
            ] = (
                mid[
                    accept_mid
                ]
            )

        if reject_mid.any():
            high[
                reject_mid
            ] = (
                mid[
                    reject_mid
                ]
            )

    result = candidate.clone()

    if needs_search.any():
        result[
            needs_search
        ] = (
            low[
                needs_search
            ]
        )

    final_feasible, _ = (
        classification_feasible(
            model,
            result,
        )
    )

    if not final_feasible.all():
        raise RuntimeError(
            "Classification pull-back failed"
        )

    return (
        result,
        final_feasible,
    )


# =====================================================================
# Differentiable Grad-CAM objective
# =====================================================================

class DifferentiableGradCAM:
    """
    Grad-CAM implementation that permits second-order differentiation from the
    relevance map back to the input image.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.activations = None

        self.handle = (
            target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def remove(self):
        self.handle.remove()

    def relevance(
        self,
        x,
        content_mask,
        altered_union_mask,
        *,
        create_graph,
    ):
        """
        Return differentiable E and supporting state.

        The usual per-image CAM max-normalisation is intentionally omitted here
        because E is invariant to multiplication by a positive scalar. Final
        evaluation still uses the exact frozen Grad-CAM implementation.
        """
        self.activations = None

        logits = self.model(
            x
        )

        margin = (
            logits[
                :,
                1,
            ]
            -
            logits[
                :,
                0,
            ]
        )

        if (
            self.activations
            is None
        ):
            raise RuntimeError(
                "Differentiable Grad-CAM "
                "forward hook failed"
            )

        activation_gradient = (
            torch.autograd.grad(
                margin.sum(),
                self.activations,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        )

        weights = (
            activation_gradient.mean(
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

        full_cam = (
            F.interpolate(
                low_cam.unsqueeze(
                    1
                ),
                size=(
                    clean_rma.CONTENT_H,
                    clean_rma.CANVAS_W,
                ),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(
                1
            )
        )

        full_cam = torch.clamp(
            full_cam,
            min=0.0,
        )

        content = (
            content_mask[
                :,
                0,
            ]
        )

        inside_energy = (
            (
                full_cam
                * altered_union_mask
            )
            .flatten(1)
            .sum(
                dim=1
            )
        )

        content_energy = (
            (
                full_cam
                * content
            )
            .flatten(1)
            .sum(
                dim=1
            )
        )

        E = torch.where(
            content_energy
            > ZERO_CAM_EPS,
            inside_energy
            /
            torch.clamp(
                content_energy,
                min=ZERO_CAM_EPS,
            ),
            torch.zeros_like(
                content_energy
            ),
        )

        probability = (
            torch.softmax(
                logits,
                dim=1,
            )[
                :,
                1,
            ]
        )

        return {
            "E":
                E,

            "margin":
                margin,

            "probability":
                probability,

            "full_cam":
                full_cam,

            "content_energy":
                content_energy,
        }


# =====================================================================
# Direct classification-preserving localisation attack
# =====================================================================

def localisation_attack(
    model,
    differentiable_cam,
    clean_x,
    content_mask,
    altered_union_mask,
):
    """
    Monotone E-minimisation under a hard classification constraint.

    Each accepted step must:
        - stay inside epsilon,
        - leave padding unchanged,
        - keep margin >= 0,
        - not increase E.
    """
    clean_feasible, clean_margin = (
        classification_feasible(
            model,
            clean_x,
        )
    )

    if not clean_feasible.all():
        raise RuntimeError(
            "Classification-preserving attack "
            "received a clean-incorrect sample"
        )

    current = (
        clean_x.detach()
        .clone()
    )

    accepted_steps = torch.zeros(
        clean_x.shape[0],
        dtype=torch.long,
        device=clean_x.device,
    )

    boundary_pullbacks = torch.zeros_like(
        accepted_steps
    )

    # Initial objective.
    current_for_eval = (
        current.detach()
        .requires_grad_(
            True
        )
    )

    initial_state = (
        differentiable_cam.relevance(
            current_for_eval,
            content_mask,
            altered_union_mask,
            create_graph=False,
        )
    )

    initial_E = (
        initial_state[
            "E"
        ]
        .detach()
    )

    current_E = (
        initial_E.clone()
    )

    std = (
        IMAGENET_STD.to(
            device=clean_x.device,
            dtype=clean_x.dtype,
        )
    )

    alpha_norm = (
        ALPHA_PIXEL
        / std
    )

    for _ in range(
        ATTACK_STEPS
    ):
        x = (
            current.detach()
            .requires_grad_(
                True
            )
        )

        state = (
            differentiable_cam.relevance(
                x,
                content_mask,
                altered_union_mask,
                create_graph=True,
            )
        )

        objective = (
            state[
                "E"
            ]
            .sum()
        )

        gradient = (
            torch.autograd.grad(
                objective,
                x,
                only_inputs=True,
            )[0]
        )

        candidate = (
            x
            -
            alpha_norm
            * gradient.sign()
            * content_mask
        )

        candidate = (
            project_to_feasible_pixel_ball(
                candidate.detach(),
                clean_x,
                content_mask,
            )
        )

        direct_feasible, _ = (
            classification_feasible(
                model,
                candidate,
            )
        )

        boundary_pullbacks += (
            ~direct_feasible
        ).long()

        candidate, _ = (
            pull_back_to_classification_boundary(
                model,
                current,
                candidate,
                clean_x,
                content_mask,
            )
        )

        # Evaluate E at the feasible candidate.
        candidate_eval = (
            candidate.detach()
            .requires_grad_(
                True
            )
        )

        candidate_state = (
            differentiable_cam.relevance(
                candidate_eval,
                content_mask,
                altered_union_mask,
                create_graph=False,
            )
        )

        candidate_E = (
            candidate_state[
                "E"
            ]
            .detach()
        )

        improvement = (
            candidate_E
            <= current_E
            + OBJECTIVE_TOL
        )

        if improvement.any():
            mask = (
                improvement.view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            current = torch.where(
                mask,
                candidate.detach(),
                current,
            )

            current_E = torch.where(
                improvement,
                candidate_E,
                current_E,
            )

            accepted_steps += (
                improvement.long()
            )

    final_x = (
        current.detach()
    )

    verify_padding_frozen(
        clean_x,
        final_x,
        content_mask,
    )

    linf = (
        pixel_linf(
            clean_x,
            final_x,
            content_mask,
        )
    )

    if (
        linf
        >
        EPSILON_PIXEL
        + 1e-6
    ).any():
        raise RuntimeError(
            "Localisation attack escaped "
            "pixel-space L_inf bound"
        )

    final_feasible, final_margin = (
        classification_feasible(
            model,
            final_x,
        )
    )

    if not final_feasible.all():
        raise RuntimeError(
            "Final localisation attack "
            "broke classification"
        )

    return {
        "adv_x":
            final_x,

        "initial_objective_E":
            initial_E,

        "final_objective_E":
            current_E,

        "accepted_steps":
            accepted_steps,

        "boundary_pullbacks":
            boundary_pullbacks,

        "clean_margin":
            clean_margin,

        "final_margin":
            final_margin,

        "linf":
            linf,
    }


# =====================================================================
# Frozen final localisation evaluation
# =====================================================================

def frozen_localisation_metrics(
    cam,
    infos,
):
    cam_np = (
        cam.detach()
        .cpu()
        .numpy()
    )

    rows = []

    for (
        index,
        info,
    ) in enumerate(
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


def cam_change_metrics(
    clean_cam,
    adv_cam,
    content_mask,
):
    content = (
        content_mask[
            :,
            0,
        ]
    )

    clean_vector = (
        clean_cam
        * content
    ).flatten(
        1
    )

    adv_vector = (
        adv_cam
        * content
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
        float(
            "nan"
        ),
        device=clean_cam.device,
        dtype=clean_cam.dtype,
    )

    tv = torch.full_like(
        cosine,
        float(
            "nan"
        ),
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
# Scoped frozen Grad-CAM evaluation
# =====================================================================

def frozen_gradcam_generate(
    model,
    x,
):
    """
    Run the exact frozen clean_rma.GradCAM implementation without leaving its
    forward hook registered during the optimisation.

    Why this is necessary
    ---------------------
    clean_rma.GradCAM._forward_hook() unconditionally calls
    output.register_hook(...). Classification-feasibility checks in this
    attack intentionally run under torch.no_grad(), so keeping a frozen
    GradCAM instance registered during those checks would make PyTorch raise:

        RuntimeError:
        cannot register a hook on a tensor that doesn't require gradient

    Therefore the frozen evaluation hook exists only for the duration of this
    function call and is removed immediately afterwards.

    The differentiable attack hook may remain registered: its forward hook only
    stores activations and does not call Tensor.register_hook().
    """
    evaluator = (
        clean_rma.GradCAM(
            model,
            model.layer4[-1],
        )
    )

    try:
        result = (
            evaluator.generate(
                x
            )
        )
    finally:
        evaluator.remove()

    return result


# =====================================================================
# Scalar helpers / stem-cluster bootstrap / summary
# =====================================================================

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


def stem_cluster_bootstrap_mean_ci(
    frame,
    column,
    *,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    File-stem cluster bootstrap 95% CI for a mean/fraction.

    Each resampled stem contributes all rows belonging to that stem.
    Non-finite values are excluded from each bootstrap statistic.
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

    local_seed = (
        int(
            hashlib.sha256(
                seed_material.encode(
                    "utf-8"
                )
            )
            .hexdigest()[
                :16
            ],
            16,
        )
        %
        (
            2 ** 32
        )
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

        values = np.concatenate(
            [
                groups[
                    str(stem)
                ]
                for stem
                in sampled_stems
            ]
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

    low, high = np.quantile(
        np.asarray(
            bootstrap_values,
            dtype=float,
        ),
        [
            0.025,
            0.975,
        ],
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
    prefix,
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
        f"{prefix}_mean"
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
        f"{prefix}_ci_low"
    ] = low

    destination[
        f"{prefix}_ci_high"
    ] = high


def make_summary(
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

            "classification_preserved":
                int(
                    group[
                        "classification_preserved"
                    ]
                    .sum()
                ),

            "classification_preservation_rate":
                float(
                    group[
                        "classification_preserved"
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

            "accepted_steps_mean":
                float(
                    group[
                        "accepted_steps"
                    ]
                    .mean()
                ),

            "boundary_pullbacks_mean":
                float(
                    group[
                        "boundary_pullbacks"
                    ]
                    .mean()
                ),

            "max_saved_probability_abs_error":
                float(
                    group[
                        "saved_probability_abs_error"
                    ]
                    .max()
                ),
        }

        # Primary fractions.
        for column, prefix in [
            (
                "classification_preserved",
                "classification_preservation_rate",
            ),
            (
                "E_strictly_improved",
                "E_strictly_improved_fraction",
            ),
            (
                "clean_zero_cam",
                "clean_zero_cam_fraction_boot",
            ),
            (
                "adv_zero_cam",
                "adv_zero_cam_fraction_boot",
            ),
            (
                "cam_change_defined",
                "cam_change_defined_fraction_boot",
            ),
        ]:
            add_mean_and_ci(
                row,
                group,
                column,
                prefix,
            )

        # Classification state.
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

        # Geometry/localisation.
        for column, prefix in [
            (
                "A",
                "A",
            ),
            (
                "E_clean",
                "E_clean",
            ),
            (
                "E_adv",
                "E_adv",
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
                "mu_w_clean",
                "mu_w_clean",
            ),
            (
                "mu_w_adv",
                "mu_w_adv",
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
                "PG_clean",
                "PG_clean",
            ),
            (
                "PG_adv",
                "PG_adv",
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

        # Map change.
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
            "Wrong frozen checkpoint"
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
        load_clean_correct_population()
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

        "epsilon_255":
            EPSILON_PIXEL
            * 255.0,

        "alpha_255":
            ALPHA_PIXEL
            * 255.0,

        "steps":
            ATTACK_STEPS,

        "classification_margin_floor":
            CLASSIFICATION_MARGIN_FLOOR,

        "bisection_steps":
            BISECTION_STEPS,

        "n_images":
            int(
                len(
                    frame
                )
            ),

        "n_boot":
            N_BOOT,

        "attack_support":
            "document_content_only",

        "padding_frozen":
            True,

        "objective":
            "minimise_gradcam_E",

        "gradcam_layer":
            "layer4[-1]",

        "gradcam_target":
            "attack_logit - bonafide_logit",

        "second_order_attack":
            True,
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

    all_altered_masks = (
        build_altered_union_masks(
            annotation_info,
            device,
        )
    )

    attack_cam = (
        DifferentiableGradCAM(
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
                    "Full population contains "
                    "non-attack sample"
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

            altered_mask = (
                all_altered_masks[
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
            # Frozen clean evaluation
            # ------------------------------------------------------

            clean_eval = (
                frozen_gradcam_generate(
                    model,
                    batch_x,
                )
            )

            clean_probability = (
                clean_eval[
                    "probability"
                ]
            )

            clean_margin = (
                clean_eval[
                    "margin"
                ]
            )

            if (
                clean_probability
                < ATTACK_THRESHOLD
            ).any():
                raise RuntimeError(
                    "Selected clean-correct sample "
                    "is not clean-correct under "
                    "fresh inference"
                )

            clean_metrics = (
                frozen_localisation_metrics(
                    clean_eval[
                        "full_cam"
                    ],
                    batch_infos,
                )
            )

            # ------------------------------------------------------
            # Direct localisation attack
            # ------------------------------------------------------

            attack_result = (
                localisation_attack(
                    model,
                    attack_cam,
                    batch_x,
                    content_mask,
                    altered_mask,
                )
            )

            adv_x = (
                attack_result[
                    "adv_x"
                ]
            )

            # ------------------------------------------------------
            # Frozen adversarial evaluation
            # ------------------------------------------------------

            adv_eval = (
                frozen_gradcam_generate(
                    model,
                    adv_x,
                )
            )

            adv_probability = (
                adv_eval[
                    "probability"
                ]
            )

            adv_margin = (
                adv_eval[
                    "margin"
                ]
            )

            adv_metrics = (
                frozen_localisation_metrics(
                    adv_eval[
                        "full_cam"
                    ],
                    batch_infos,
                )
            )

            if (
                adv_probability
                < ATTACK_THRESHOLD
            ).any():
                raise RuntimeError(
                    "Classification-preserving "
                    "attack returned an incorrect "
                    "final prediction"
                )

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
                    "Clean/adversarial A mismatch"
                )

            cam_change = (
                cam_change_metrics(
                    clean_eval[
                        "full_cam"
                    ],
                    adv_eval[
                        "full_cam"
                    ],
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

                saved_probability = float(
                    row[
                        "saved_attack_probability"
                    ]
                )

                exact_archive = save_exact_bundle(
                    repo_root=ROOT,
                    archive_root=EXACT_ARCHIVE_ROOT,
                    row=row,
                    info=batch_infos[j],
                    clean_input=batch_x[j],
                    adv_input=adv_x[j],
                    clean_cam=clean_eval["full_cam"][j],
                    adv_cam=adv_eval["full_cam"][j],
                    clean_probability=clean_p,
                    adv_probability=adv_p,
                    clean_margin=clean_margin[j].item(),
                    adv_margin=adv_margin[j].item(),
                    attack_metadata={
                        "attack_type": "classification_preserving_gradcam_E",
                        "epsilon_pixel": EPSILON_PIXEL,
                        "epsilon_255": EPSILON_PIXEL * 255.0,
                        "alpha_pixel": ALPHA_PIXEL,
                        "alpha_255": ALPHA_PIXEL * 255.0,
                        "steps": ATTACK_STEPS,
                        "bisection_steps": BISECTION_STEPS,
                        "classification_margin_floor": CLASSIFICATION_MARGIN_FLOOR,
                        "objective_tol": OBJECTIVE_TOL,
                        "initialisation": "clean_image",
                        "random_restart": False,
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

                    "classification_preserved":
                        bool(
                            adv_p
                            >= ATTACK_THRESHOLD
                        ),

                    "epsilon_255":
                        EPSILON_PIXEL
                        * 255.0,

                    "alpha_255":
                        ALPHA_PIXEL
                        * 255.0,

                    "steps":
                        ATTACK_STEPS,

                    "pixel_linf":
                        attack_result[
                            "linf"
                        ][
                            j
                        ]
                        .item(),

                    "padding_max_abs_error":
                        padding_error,

                    "accepted_steps":
                        int(
                            attack_result[
                                "accepted_steps"
                            ][
                                j
                            ]
                            .item()
                        ),

                    "boundary_pullbacks":
                        int(
                            attack_result[
                                "boundary_pullbacks"
                            ][
                                j
                            ]
                            .item()
                        ),

                    "objective_E_initial":
                        attack_result[
                            "initial_objective_E"
                        ][
                            j
                        ]
                        .item(),

                    "objective_E_final":
                        attack_result[
                            "final_objective_E"
                        ][
                            j
                        ]
                        .item(),

                    "A":
                        clean_metrics[
                            "A"
                        ][
                            j
                        ]
                        .item(),

                    "E_clean":
                        E_clean,

                    "E_adv":
                        E_adv,

                    "absolute_E_degradation":
                        (
                            E_clean
                            - E_adv
                        ),

                    "relative_E_degradation":
                        safe_relative_degradation(
                            E_clean,
                            E_adv,
                        ),

                    "E_strictly_improved":
                        bool(
                            E_adv
                            <
                            E_clean
                            - 1e-6
                        ),

                    "mu_w_clean":
                        mu_clean,

                    "mu_w_adv":
                        mu_adv,

                    "absolute_mu_w_degradation":
                        (
                            mu_clean
                            - mu_adv
                        ),

                    "relative_mu_w_degradation":
                        safe_relative_degradation(
                            mu_clean,
                            mu_adv,
                        ),

                    "PG_clean":
                        PG_clean,

                    "PG_adv":
                        PG_adv,

                    "absolute_PG_degradation":
                        (
                            PG_clean
                            - PG_adv
                        ),

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
                    clean_eval[
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
                    adv_eval[
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
        attack_cam.remove()

    # =================================================================
    # Save / summarise
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
        )

    if not (
        per_image[
            "classification_preserved"
        ]
        .all()
    ):
        raise RuntimeError(
            "At least one final sample "
            "did not preserve classification"
        )

    per_image.to_csv(
        OUT_PER_IMAGE,
        index=False,
    )

    summary = (
        make_summary(
            per_image
        )
    )

    summary.to_csv(
        OUT_SUMMARY,
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
    # Terminal report
    # =================================================================

    print()
    print(
        "FULL RESNET18 CLASSIFICATION-PRESERVING "
        "LOCALISATION ATTACK EVALUATION"
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
        "population: clean-correct only"
    )

    print(
        "objective: minimise Grad-CAM E"
    )

    print(
        "constraint: "
        "attack_logit - bonafide_logit >= 0"
    )

    print(
        "optimizer: projected sign-gradient "
        "with classification-boundary pull-back"
    )

    print(
        "second-order Grad-CAM attack: yes"
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
        f"{ATTACK_STEPS}"
    )

    print(
        f"bisection steps: "
        f"{BISECTION_STEPS}"
    )

    print(
        "attack support: document content only"
    )

    print(
        "padding: frozen exactly"
    )

    print(
        "final evaluator: "
        "frozen clean_rma.compute_rma_metrics"
    )

    print(
        "uncertainty: file-stem cluster "
        "bootstrap 95% CI"
    )

    print(
        f"max pixel L_inf: "
        f"{per_image['pixel_linf'].max():.8f}"
    )

    print(
        f"max padding change: "
        f"{per_image['padding_max_abs_error'].max():.12g}"
    )

    print(
        "max clean-vs-saved probability error: "
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
        "OUTPUTS:"
    )

    print(
        f"  config:    "
        f"{OUT_CONFIG}"
    )

    print(
        f"  selection: "
        f"{OUT_SELECTION}"
    )

    print(
        f"  per image: "
        f"{OUT_PER_IMAGE}"
    )

    print(
        f"  summary:   "
        f"{OUT_SUMMARY}"
    )

    print(
        f"  CAMs:      "
        f"{OUT_CAMS}"
    )

    print()
    print(
        "STOP HERE."
    )

    print(
        "Classification-preserving localisation "
        "attack evaluation complete."
    )


if __name__ == "__main__":
    main()
