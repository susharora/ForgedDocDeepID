#!/usr/bin/env python3
"""
ResNet18 classification-preserving Grad-CAM localisation attack — pilot.

Purpose
-------
The standard targeted PGD baseline at epsilon=1/255 flips every clean-correct
sample in the full primary evaluation. That experiment therefore measures
classification failure and explanation collapse together.

This second attack asks a different question:

    Can Grad-CAM localisation be degraded while the detector still predicts
    ATTACK correctly?

Attack population
-----------------
Clean-correct subsets only:

    dev_val / digital_1
    dev_val / digital_2
    official_test / facedancer

Pilot:
    24 deterministic clean-correct samples per population = 72 total.

Perturbation budget
-------------------
Same conservative budget as the frozen classification-breaking baseline:

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

A candidate step that crosses the classification boundary is pulled back along
the step direction using a short feasibility bisection. A step is accepted only
when:

    1. classification remains attack, and
    2. the Grad-CAM relevance-mass objective E does not increase.

Thus the final adversarial image is classification-preserving by construction.

Localisation objective
----------------------
Directly minimise differentiable Grad-CAM relevance mass inside the altered
union:

    E = CAM mass inside altered union / CAM mass inside document content

Grad-CAM target:
    layer4[-1]
    attack_logit - bonafide_logit

Optimising E requires differentiating through the Grad-CAM gradient weights,
so this is a second-order white-box explanation attack.

Evaluation
----------
Final clean/adversarial A, E, mu_w and PG are evaluated with the exact frozen
clean_rma.compute_rma_metrics() implementation. The frozen clean Grad-CAM
forward hook is scoped only to those evaluation calls and is removed before
classification-feasibility checks.

Also report:
    classification preservation
    attack probability and margin
    absolute / relative E degradation
    absolute / relative mu_w degradation
    PG degradation
    CAM cosine similarity
    CAM total variation
    zero-CAM frequency
    realised L_inf
    accepted-step count

Stop after this 72-image pilot. Do not run the full population until the pilot
has been inspected.
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

PILOT_N_PER_GROUP = 24
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
    / "resnet18_localisation_attack_preserve_cls_pilot"
)

OUT_CONFIG = OUT_ROOT / "localisation_attack_config.json"
OUT_SELECTION = OUT_ROOT / "localisation_attack_selection.csv"
OUT_PER_IMAGE = OUT_ROOT / "localisation_attack_per_image.csv"
OUT_SUMMARY = OUT_ROOT / "localisation_attack_summary.csv"
OUT_CAMS = OUT_ROOT / "localisation_attack_layer4_maps.npz"


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
# Stable pilot population
# =====================================================================

def stable_hash(value):
    return int(
        hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()[:16],
        16,
    )


def deterministic_take(frame, n):
    frame = frame.copy()

    frame["_pilot_hash"] = (
        frame["image_path"]
        .astype(str)
        .map(stable_hash)
    )

    return (
        frame
        .sort_values(
            [
                "_pilot_hash",
                "image_path",
            ]
        )
        .head(n)
        .drop(
            columns="_pilot_hash"
        )
        .reset_index(
            drop=True
        )
    )


def load_clean_correct_pilot():
    """
    Use only clean-correct attack samples.

    For this experiment a clean misclassification cannot be called
    classification-preserving, so it is outside the estimand by definition.
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
        sub = evaluation[
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
            len(sub)
            != expected_clean_correct
        ):
            raise RuntimeError(
                "Unexpected clean-correct population "
                f"for {split}/{variant}: "
                f"expected={expected_clean_correct}, "
                f"actual={len(sub)}"
            )

        pilot = deterministic_take(
            sub,
            PILOT_N_PER_GROUP,
        )

        if (
            len(pilot)
            != PILOT_N_PER_GROUP
        ):
            raise RuntimeError(
                "Pilot selection failed "
                f"for {split}/{variant}"
            )

        groups.append(
            pilot
        )

    frame = pd.concat(
        groups,
        ignore_index=True,
    )

    if (
        len(frame)
        != 3
        * PILOT_N_PER_GROUP
    ):
        raise RuntimeError(
            "Expected 72 pilot samples, "
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
            "Duplicate image paths in pilot"
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
# Scalar helpers / summary
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
        rows.append(
            {
                "evaluation_split":
                    split,

                "variant":
                    variant,

                "n":
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

                "clean_p_attack_mean":
                    float(
                        group[
                            "clean_probability_attack"
                        ]
                        .mean()
                    ),

                "adv_p_attack_mean":
                    float(
                        group[
                            "adv_probability_attack"
                        ]
                        .mean()
                    ),

                "clean_margin_mean":
                    float(
                        group[
                            "clean_margin"
                        ]
                        .mean()
                    ),

                "adv_margin_mean":
                    float(
                        group[
                            "adv_margin"
                        ]
                        .mean()
                    ),

                "A_mean":
                    float(
                        group[
                            "A"
                        ]
                        .mean()
                    ),

                "E_clean_mean":
                    float(
                        group[
                            "E_clean"
                        ]
                        .mean()
                    ),

                "E_adv_mean":
                    float(
                        group[
                            "E_adv"
                        ]
                        .mean()
                    ),

                "absolute_E_degradation_mean":
                    float(
                        group[
                            "absolute_E_degradation"
                        ]
                        .mean()
                    ),

                "relative_E_degradation_mean":
                    float(
                        group[
                            "relative_E_degradation"
                        ]
                        .mean()
                    ),

                "E_strictly_improved_fraction":
                    float(
                        group[
                            "E_strictly_improved"
                        ]
                        .mean()
                    ),

                "mu_w_clean_mean":
                    float(
                        group[
                            "mu_w_clean"
                        ]
                        .mean()
                    ),

                "mu_w_adv_mean":
                    float(
                        group[
                            "mu_w_adv"
                        ]
                        .mean()
                    ),

                "absolute_mu_w_degradation_mean":
                    float(
                        group[
                            "absolute_mu_w_degradation"
                        ]
                        .mean()
                    ),

                "relative_mu_w_degradation_mean":
                    float(
                        group[
                            "relative_mu_w_degradation"
                        ]
                        .mean()
                    ),

                "PG_clean":
                    float(
                        group[
                            "PG_clean"
                        ]
                        .mean()
                    ),

                "PG_adv":
                    float(
                        group[
                            "PG_adv"
                        ]
                        .mean()
                    ),

                "absolute_PG_degradation":
                    float(
                        group[
                            "absolute_PG_degradation"
                        ]
                        .mean()
                    ),

                "cam_cosine_mean":
                    float(
                        group[
                            "cam_cosine"
                        ]
                        .mean()
                    ),

                "cam_tv_mean":
                    float(
                        group[
                            "cam_tv"
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
        load_clean_correct_pilot()
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

        "pilot_n_per_group":
            PILOT_N_PER_GROUP,

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
                    "Pilot contains "
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
        "RESNET18 CLASSIFICATION-PRESERVING "
        "LOCALISATION ATTACK PILOT"
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
        f"pilot images:   "
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
        "PRIMARY PILOT SUMMARY:"
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
        "Inspect classification preservation "
        "and localisation degradation before "
        "running a full-population attack."
    )


if __name__ == "__main__":
    main()
