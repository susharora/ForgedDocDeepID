#!/usr/bin/env python3
"""
ResNet18 adversarial localisation robustness — PGD pilot.

Pilot only. Do not perform an epsilon sweep here.

Attack:
    targeted L_inf PGD -> bonafide
    epsilon = 4/255 pixel RGB
    alpha   = 1/255 pixel RGB
    steps   = 10
    random start
    document-content pixels only
    frozen artificial horizontal padding

Populations:
    dev_val / digital_1       24
    dev_val / digital_2       24
    official_test / facedancer 24

Classification attack success for forged images:
    clean prediction attack AND adversarial prediction bonafide

Primary localisation:
    A
    E
    mu_w = E / A
    PG

Change metrics:
    delta_E       = E_adv - E_clean
    rel_E         = (E_adv - E_clean) / E_clean

    delta_mu_w    = mu_adv - mu_clean
    rel_mu_w      = (mu_adv - mu_clean) / mu_clean

    delta_PG      = PG_adv - PG_clean

CAM-map change:
    CAM cosine similarity over document content
    CAM relevance-distribution TV distance over document content

Stop after pilot summary. The results determine the next epsilon decision.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]

# Import frozen machinery rather than duplicating preprocessing / GT logic.
import sys
sys.path.insert(0, str(ROOT / "scripts"))

import eval_resnet18_gradcam_clean_rma as clean_rma


# ---------------------------------------------------------------------
# Frozen experiment definition
# ---------------------------------------------------------------------

SEED = 10

EPSILON_PIXEL = 4.0 / 255.0
ALPHA_PIXEL = 1.0 / 255.0
PGD_STEPS = 10

PILOT_N_PER_GROUP = 24

ATTACK_THRESHOLD = 0.5

OUT_ROOT = (
    ROOT
    / "output"
    / "resnet18_adversarial_localisation_pilot"
)

OUT_PER_IMAGE = OUT_ROOT / "pgd_pilot_per_image.csv"
OUT_SUMMARY = OUT_ROOT / "pgd_pilot_summary.csv"
OUT_CAMS = OUT_ROOT / "pgd_pilot_cams.npz"


IMAGENET_MEAN = torch.tensor(
    clean_rma.IMAGENET_MEAN,
    dtype=torch.float32,
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    clean_rma.IMAGENET_STD,
    dtype=torch.float32,
).view(1, 3, 1, 1)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def stable_hash(value):
    return int(
        hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()[:16],
        16,
    )


def deterministic_take(frame, n):
    """
    Stable selection independent of dataframe ordering.
    """
    frame = frame.copy()

    frame["_pilot_hash"] = (
        frame["image_path"]
        .astype(str)
        .map(stable_hash)
    )

    frame = (
        frame
        .sort_values(
            ["_pilot_hash", "image_path"]
        )
        .head(n)
        .drop(columns="_pilot_hash")
        .reset_index(drop=True)
    )

    return frame


def content_mask_from_rows(frame, device):
    """
    Reconstruct exact frozen document support.

    Frozen preprocessing:
        resize image to H=512 preserving aspect ratio,
        horizontally centre on W=864,
        normalised tensor outside content == exactly zero.
    """
    masks = []

    for row in frame.itertuples():
        # Use cached image dimensions if available.
        # Otherwise open cache image exactly as frozen dataset does.
        path = ROOT / row.cache_path

        from PIL import Image

        with Image.open(path) as im:
            width, height = im.size

        new_width = int(
            round(
                width
                * clean_rma.CONTENT_H
                / height
            )
        )

        if new_width > clean_rma.CANVAS_W:
            raise RuntimeError(
                f"Unexpected content width {new_width}: "
                f"{row.image_path}"
            )

        left = (
            clean_rma.CANVAS_W
            - new_width
        ) // 2

        mask = torch.zeros(
            (
                1,
                clean_rma.CONTENT_H,
                clean_rma.CANVAS_W,
            ),
            dtype=torch.float32,
        )

        mask[
            :,
            :,
            left:left + new_width,
        ] = 1.0

        masks.append(mask)

    return torch.stack(
        masks,
        dim=0,
    ).to(device)


def verify_padding_frozen(
    clean_x,
    adv_x,
    content_mask,
):
    padding_mask = 1.0 - content_mask

    error = (
        (
            adv_x - clean_x
        ).abs()
        * padding_mask
    ).amax().item()

    if error != 0.0:
        raise RuntimeError(
            "Artificial padding changed during attack: "
            f"max_abs_error={error:.12g}"
        )

    return error


def pixel_linf(
    clean_x,
    adv_x,
    content_mask,
):
    """
    Convert normalised-space change back to pixel RGB units.
    """
    std = IMAGENET_STD.to(
        device=clean_x.device,
        dtype=clean_x.dtype,
    )

    delta_pixel = (
        adv_x - clean_x
    ) * std

    delta_pixel = (
        delta_pixel
        * content_mask
    )

    return (
        delta_pixel
        .abs()
        .flatten(1)
        .amax(dim=1)
    )


# ---------------------------------------------------------------------
# Targeted content-only PGD
# ---------------------------------------------------------------------

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
    Targeted PGD in RGB pixel space.

    clean_x is already ImageNet-normalised and padded.

    We implement the pixel-space L_inf ball exactly in normalised
    coordinates using channel-specific std.

    Crucially:
        perturbation *= content mask
        padding is restored exactly after random start AND every step.
    """

    device = clean_x.device
    dtype = clean_x.dtype

    mean = IMAGENET_MEAN.to(
        device=device,
        dtype=dtype,
    )

    std = IMAGENET_STD.to(
        device=device,
        dtype=dtype,
    )

    eps_norm = (
        epsilon_pixel
        / std
    )

    alpha_norm = (
        alpha_pixel
        / std
    )

    # Valid RGB [0,1] represented in normalised coordinates.
    lower_valid = (
        0.0 - mean
    ) / std

    upper_valid = (
        1.0 - mean
    ) / std

    # Random start uniformly inside pixel-space Linf ball.
    random_delta_pixel = torch.empty_like(
        clean_x
    ).uniform_(
        -epsilon_pixel,
        epsilon_pixel,
    )

    random_delta_norm = (
        random_delta_pixel
        / std
    )

    random_delta_norm = (
        random_delta_norm
        * content_mask
    )

    adv = (
        clean_x
        + random_delta_norm
    )

    # Pixel validity.
    adv = torch.maximum(
        torch.minimum(
            adv,
            upper_valid,
        ),
        lower_valid,
    )

    # Project into clean-centred epsilon ball.
    delta = adv - clean_x

    delta = torch.maximum(
        torch.minimum(
            delta,
            eps_norm,
        ),
        -eps_norm,
    )

    delta = (
        delta
        * content_mask
    )

    adv = clean_x + delta

    # Exact clean padding.
    adv = (
        adv * content_mask
        + clean_x * (
            1.0 - content_mask
        )
    )

    target = torch.zeros(
        clean_x.shape[0],
        dtype=torch.long,
        device=device,
    )

    for _ in range(steps):
        adv = (
            adv.detach()
            .requires_grad_(True)
        )

        logits = model(adv)

        # Targeted attack:
        # minimise CE toward class 0 = bonafide.
        loss = F.cross_entropy(
            logits,
            target,
            reduction="sum",
        )

        grad = torch.autograd.grad(
            loss,
            adv,
            only_inputs=True,
        )[0]

        # Targeted PGD = gradient DESCENT on target CE.
        adv_next = (
            adv
            - alpha_norm
            * grad.sign()
            * content_mask
        )

        # Epsilon projection.
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

        delta = (
            delta
            * content_mask
        )

        adv_next = clean_x + delta

        # Valid underlying RGB range.
        clipped = torch.maximum(
            torch.minimum(
                adv_next,
                upper_valid,
            ),
            lower_valid,
        )

        # RGB clipping applies to document content only.
        adv_next = (
            clipped * content_mask
            + clean_x * (
                1.0 - content_mask
            )
        )

        # Re-project in case RGB clipping changed numerical edge cases.
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

        delta = (
            delta
            * content_mask
        )

        adv = (
            clean_x
            + delta
        )

        # Final exact restoration of padding.
        adv = (
            adv * content_mask
            + clean_x * (
                1.0 - content_mask
            )
        )

    adv = adv.detach()

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
        epsilon_pixel + 1e-6
    ).any():
        raise RuntimeError(
            "PGD escaped pixel-space Linf bound: "
            f"{linf.max().item():.8f}"
        )

    return adv


# ---------------------------------------------------------------------
# CAM comparison
# ---------------------------------------------------------------------

def cam_distribution(
    cam,
    content_mask,
):
    """
    Convert non-negative CAM to a probability distribution over
    document-content pixels.
    """
    x = (
        cam
        * content_mask[:, 0]
    )

    total = (
        x
        .flatten(1)
        .sum(dim=1)
        .clamp_min(1e-12)
    )

    return (
        x.flatten(1)
        / total[:, None]
    )


def cam_change_metrics(
    clean_cam,
    adv_cam,
    content_mask,
):
    mask = content_mask[:, 0]

    clean_vec = (
        clean_cam
        * mask
    ).flatten(1)

    adv_vec = (
        adv_cam
        * mask
    ).flatten(1)

    cosine = F.cosine_similarity(
        clean_vec,
        adv_vec,
        dim=1,
        eps=1e-12,
    )

    p = cam_distribution(
        clean_cam,
        content_mask,
    )

    q = cam_distribution(
        adv_cam,
        content_mask,
    )

    # Total variation distance between relevance distributions.
    # 0 = identical spatial relevance distribution
    # 1 = disjoint distributions
    tv = (
        0.5
        * (
            p - q
        ).abs().sum(dim=1)
    )

    return cosine, tv


# ---------------------------------------------------------------------
# Area-aware localisation
# ---------------------------------------------------------------------

def localisation_metrics(
    cam,
    content_mask,
    altered_union_mask,
):
    """
    A is independent of CAM and therefore identical clean/adv.
    """
    content = (
        content_mask[:, 0]
        > 0.5
    )

    altered = (
        altered_union_mask
        > 0.5
    ) & content

    content_area = (
        content
        .flatten(1)
        .sum(dim=1)
        .float()
    )

    altered_area = (
        altered
        .flatten(1)
        .sum(dim=1)
        .float()
    )

    A = (
        altered_area
        / content_area
    )

    content_cam = (
        cam
        * content.float()
    )

    altered_cam = (
        cam
        * altered.float()
    )

    content_energy = (
        content_cam
        .flatten(1)
        .sum(dim=1)
    )

    altered_energy = (
        altered_cam
        .flatten(1)
        .sum(dim=1)
    )

    E = torch.where(
        content_energy > 1e-12,
        altered_energy
        / content_energy,
        torch.zeros_like(
            content_energy
        ),
    )

    mu_w = torch.where(
        A > 0,
        E / A,
        torch.zeros_like(E),
    )

    # Content-space maximum only.
    masked_cam = cam.clone()

    masked_cam[
        ~content
    ] = -1.0

    max_index = (
        masked_cam
        .flatten(1)
        .argmax(dim=1)
    )

    altered_flat = (
        altered
        .flatten(1)
    )

    PG = altered_flat.gather(
        1,
        max_index[:, None],
    )[:, 0].float()

    return {
        "A": A,
        "E": E,
        "mu_w": mu_w,
        "PG": PG,
    }


# ---------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------

def make_pilot_frame():
    """
    Use the frozen clean RMA script's canonical population construction.

    IMPORTANT:
    Replace the call below with the actual population-builder function
    already present in eval_resnet18_gradcam_clean_rma.py if its local
    function name differs.

    The resulting frame must contain exactly the same rows/annotations
    used by the frozen clean evaluation.
    """

    # This deliberately avoids re-specifying inventory parsing here.
    #
    # In your frozen script, identify the dataframe immediately before
    # PolicyCDataset(...) is instantiated. Extract that construction into:
    #
    #     clean_rma.build_evaluation_frame()
    #
    # returning all frozen evaluation rows.
    #
    # That is the ONLY small refactor needed in the clean file.
    frame = clean_rma.build_evaluation_frame()

    groups = []

    specs = [
        (
            "dev_val",
            "digital_1",
        ),
        (
            "dev_val",
            "digital_2",
        ),
        (
            "official_test",
            "facedancer",
        ),
    ]

    for split, variant in specs:
        sub = frame[
            (
                frame["evaluation_split"]
                == split
            )
            &
            (
                frame["variant"]
                == variant
            )
        ].copy()

        if len(sub) < PILOT_N_PER_GROUP:
            raise RuntimeError(
                f"Too few rows for {split}/{variant}: "
                f"{len(sub)}"
            )

        sub = deterministic_take(
            sub,
            PILOT_N_PER_GROUP,
        )

        groups.append(sub)

    result = pd.concat(
        groups,
        ignore_index=True,
    )

    if len(result) != 72:
        raise RuntimeError(
            f"Expected 72 pilot images, got {len(result)}"
        )

    return result


# ---------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------

def build_altered_union_masks(
    frame,
    device,
):
    """
    Do NOT reconstruct annotation geometry independently.

    Reuse the exact resized union-mask function from the frozen clean RMA
    script.

    Make that helper public as:

        clean_rma.build_union_mask_for_row(row)

    returning a [512,864] bool/float numpy array or torch tensor.
    """
    masks = []

    for row in frame.itertuples():
        mask = (
            clean_rma
            .build_union_mask_for_row(row)
        )

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32,
        )

        if tuple(mask.shape) != (
            clean_rma.CONTENT_H,
            clean_rma.CANVAS_W,
        ):
            raise RuntimeError(
                "Unexpected GT mask shape: "
                f"{tuple(mask.shape)}"
            )

        masks.append(mask)

    return torch.stack(
        masks,
        dim=0,
    ).to(device)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_sha = (
        clean_rma.verify_checkpoint()
    )

    expected_sha = (
        "25ad8b1482be20e9d5e450770d6970b820ebc9470c9da3c4ec46db2558009402"
    )

    if checkpoint_sha != expected_sha:
        raise RuntimeError(
            "Wrong frozen ResNet checkpoint"
        )

    clean_rma.verify_inventory()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint = (
        clean_rma.load_model(device)
    )

    frame = make_pilot_frame()

    dataset = clean_rma.PolicyCDataset(
        frame
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    all_content_masks = (
        content_mask_from_rows(
            frame,
            device,
        )
    )

    all_union_masks = (
        build_altered_union_masks(
            frame,
            device,
        )
    )

    gradcam = clean_rma.GradCAM(
        model,
        model.layer4[-1],
    )

    records = []

    clean_low_cams = {}
    adv_low_cams = {}

    try:
        for batch_x, labels, indices in loader:
            batch_x = batch_x.to(device)
            labels = labels.to(device)
            indices = indices.to(device)

            if not torch.all(
                labels == 1
            ):
                raise RuntimeError(
                    "Pilot must contain attack samples only"
                )

            content_mask = (
                all_content_masks[
                    indices
                ]
            )

            union_mask = (
                all_union_masks[
                    indices
                ]
            )

            # ---------------------------------------------------------
            # Clean state
            # ---------------------------------------------------------

            clean_result = (
                gradcam.generate(
                    batch_x
                )
            )

            clean_prob = (
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
                localisation_metrics(
                    clean_cam,
                    content_mask,
                    union_mask,
                )
            )

            # ---------------------------------------------------------
            # PGD
            # ---------------------------------------------------------

            adv_x = (
                pgd_target_bonafide(
                    model,
                    batch_x,
                    content_mask,
                )
            )

            linf = pixel_linf(
                batch_x,
                adv_x,
                content_mask,
            )

            # ---------------------------------------------------------
            # Adversarial Grad-CAM
            # ---------------------------------------------------------

            adv_result = (
                gradcam.generate(
                    adv_x
                )
            )

            adv_prob = (
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
                localisation_metrics(
                    adv_cam,
                    content_mask,
                    union_mask,
                )
            )

            cam_cosine, cam_tv = (
                cam_change_metrics(
                    clean_cam,
                    adv_cam,
                    content_mask,
                )
            )

            # ---------------------------------------------------------
            # Per image
            # ---------------------------------------------------------

            for j, global_index in enumerate(
                indices.tolist()
            ):
                row = frame.iloc[
                    global_index
                ]

                clean_correct = bool(
                    clean_prob[j].item()
                    >= ATTACK_THRESHOLD
                )

                adv_correct = bool(
                    adv_prob[j].item()
                    >= ATTACK_THRESHOLD
                )

                attack_success = bool(
                    clean_correct
                    and not adv_correct
                )

                A = (
                    clean_metrics["A"][j]
                    .item()
                )

                E0 = (
                    clean_metrics["E"][j]
                    .item()
                )

                E1 = (
                    adv_metrics["E"][j]
                    .item()
                )

                mu0 = (
                    clean_metrics["mu_w"][j]
                    .item()
                )

                mu1 = (
                    adv_metrics["mu_w"][j]
                    .item()
                )

                pg0 = (
                    clean_metrics["PG"][j]
                    .item()
                )

                pg1 = (
                    adv_metrics["PG"][j]
                    .item()
                )

                record = {
                    "evaluation_split":
                        row["evaluation_split"],

                    "variant":
                        row["variant"],

                    "image_path":
                        row["image_path"],

                    "label":
                        int(labels[j].item()),

                    "epsilon_pixel":
                        EPSILON_PIXEL,

                    "epsilon_255":
                        EPSILON_PIXEL * 255.0,

                    "alpha_255":
                        ALPHA_PIXEL * 255.0,

                    "pgd_steps":
                        PGD_STEPS,

                    "pixel_linf":
                        linf[j].item(),

                    "clean_probability_attack":
                        clean_prob[j].item(),

                    "adv_probability_attack":
                        adv_prob[j].item(),

                    "clean_margin":
                        clean_margin[j].item(),

                    "adv_margin":
                        adv_margin[j].item(),

                    "clean_correct":
                        clean_correct,

                    "adv_correct":
                        adv_correct,

                    "attack_success":
                        attack_success,

                    "A":
                        A,

                    "E_clean":
                        E0,

                    "E_adv":
                        E1,

                    "delta_E":
                        E1 - E0,

                    "relative_E_change":
                        (
                            (E1 - E0) / E0
                            if E0 > 1e-12
                            else np.nan
                        ),

                    "mu_w_clean":
                        mu0,

                    "mu_w_adv":
                        mu1,

                    "delta_mu_w":
                        mu1 - mu0,

                    "relative_mu_w_change":
                        (
                            (mu1 - mu0) / mu0
                            if mu0 > 1e-12
                            else np.nan
                        ),

                    "PG_clean":
                        pg0,

                    "PG_adv":
                        pg1,

                    "delta_PG":
                        pg1 - pg0,

                    "cam_cosine":
                        cam_cosine[j].item(),

                    "cam_tv":
                        cam_tv[j].item(),
                }

                records.append(record)

                key = str(
                    row["image_path"]
                )

                clean_low_cams[key] = (
                    clean_result[
                        "low_cam"
                    ][j]
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )

                adv_low_cams[key] = (
                    adv_result[
                        "low_cam"
                    ][j]
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )

    finally:
        gradcam.remove()

    per_image = pd.DataFrame(
        records
    )

    per_image.to_csv(
        OUT_PER_IMAGE,
        index=False,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary_rows = []

    for (
        split,
        variant,
    ), group in per_image.groupby(
        [
            "evaluation_split",
            "variant",
        ],
        sort=False,
    ):
        clean_correct = group[
            group["clean_correct"]
        ]

        attack_success_rate = (
            clean_correct[
                "attack_success"
            ].mean()
            if len(clean_correct)
            else np.nan
        )

        summary_rows.append(
            {
                "evaluation_split":
                    split,

                "variant":
                    variant,

                "n":
                    len(group),

                "n_clean_correct":
                    int(
                        group[
                            "clean_correct"
                        ].sum()
                    ),

                "clean_attack_accuracy":
                    group[
                        "clean_correct"
                    ].mean(),

                "adv_attack_accuracy":
                    group[
                        "adv_correct"
                    ].mean(),

                # Conditional on samples that PGD can actually attack
                # from a correct clean state.
                "attack_success_rate_clean_correct":
                    attack_success_rate,

                "clean_p_attack_mean":
                    group[
                        "clean_probability_attack"
                    ].mean(),

                "adv_p_attack_mean":
                    group[
                        "adv_probability_attack"
                    ].mean(),

                "E_clean_mean":
                    group[
                        "E_clean"
                    ].mean(),

                "E_adv_mean":
                    group[
                        "E_adv"
                    ].mean(),

                "delta_E_mean":
                    group[
                        "delta_E"
                    ].mean(),

                "relative_E_change_mean":
                    group[
                        "relative_E_change"
                    ].mean(),

                "mu_w_clean_mean":
                    group[
                        "mu_w_clean"
                    ].mean(),

                "mu_w_adv_mean":
                    group[
                        "mu_w_adv"
                    ].mean(),

                "delta_mu_w_mean":
                    group[
                        "delta_mu_w"
                    ].mean(),

                "relative_mu_w_change_mean":
                    group[
                        "relative_mu_w_change"
                    ].mean(),

                "PG_clean":
                    group[
                        "PG_clean"
                    ].mean(),

                "PG_adv":
                    group[
                        "PG_adv"
                    ].mean(),

                "delta_PG":
                    group[
                        "PG_adv"
                    ].mean()
                    -
                    group[
                        "PG_clean"
                    ].mean(),

                "cam_cosine_mean":
                    group[
                        "cam_cosine"
                    ].mean(),

                "cam_tv_mean":
                    group[
                        "cam_tv"
                    ].mean(),

                "pixel_linf_max":
                    group[
                        "pixel_linf"
                    ].max(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    np.savez_compressed(
        OUT_CAMS,
        clean_keys=np.array(
            list(clean_low_cams.keys()),
            dtype=object,
        ),
        clean_maps=np.stack(
            list(clean_low_cams.values())
        ),
        adv_keys=np.array(
            list(adv_low_cams.keys()),
            dtype=object,
        ),
        adv_maps=np.stack(
            list(adv_low_cams.values())
        ),
    )

    # -------------------------------------------------------------
    # Mandatory audit
    # -------------------------------------------------------------

    print()
    print(
        "RESNET18 ADVERSARIAL LOCALISATION PILOT"
    )
    print()

    print(
        f"checkpoint SHA: {checkpoint_sha}"
    )

    print(
        "attack: targeted bonafide "
        "pixel-space L_inf PGD"
    )

    print(
        f"epsilon: {EPSILON_PIXEL * 255:.1f}/255"
    )

    print(
        f"alpha:   {ALPHA_PIXEL * 255:.1f}/255"
    )

    print(
        f"steps:   {PGD_STEPS}"
    )

    print(
        "random start: yes"
    )

    print(
        "attack support: document content only"
    )

    print(
        "padding: frozen exactly"
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        "OUTPUTS:"
    )
    print(
        f"  per image: {OUT_PER_IMAGE}"
    )
    print(
        f"  summary:   {OUT_SUMMARY}"
    )
    print(
        f"  CAMs:      {OUT_CAMS}"
    )

    print()
    print(
        "STOP HERE."
    )
    print(
        "Do not run another epsilon until this pilot "
        "has been inspected."
    )


if __name__ == "__main__":
    main()