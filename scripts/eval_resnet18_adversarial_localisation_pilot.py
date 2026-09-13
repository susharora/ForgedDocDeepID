#!/usr/bin/env python3
"""
ResNet18 adversarial localisation robustness — PGD pilot.

Pilot only. Do not perform an epsilon sweep here.

Frozen checkpoint:
    runs/post_hoc_compression_controlled_resnet18_seed10/checkpoints/best.pt
SHA256:
    25ad8b1482be20e9d5e450770d6970b820ebc9470c9da3c4ec46db2558009402

Attack:
    targeted L_inf PGD -> bonafide
    epsilon = 4/255 in resized RGB pixel space
    alpha   = 1/255
    steps   = 10
    random start
    document-content pixels only
    artificial horizontal padding frozen exactly

Pilot:
    dev_val/digital_1        24
    dev_val/digital_2        24
    official_test/facedancer 24

Localisation:
    A, E, mu_w, PG are computed by the exact frozen clean-RMA
    compute_rma_metrics() implementation.

CAM-map change:
    cosine similarity over document content
    total-variation distance between content-normalised CAM distributions

If either content CAM has zero mass, cosine/TV are NaN and zero-CAM flags are
reported separately.

STOP after this pilot.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_resnet18_gradcam_clean_rma as clean_rma


SEED = 10
EXPECTED_CHECKPOINT_SHA256 = (
    "25ad8b1482be20e9d5e450770d6970b820ebc9470c9da3c4ec46db2558009402"
)

EPSILON_PIXEL = 1.0 / 255.0
ALPHA_PIXEL = 0.25 / 255.0
PGD_STEPS = 10
PILOT_N_PER_GROUP = 24
BATCH_SIZE = 4
ATTACK_THRESHOLD = 0.5
ZERO_CAM_EPS = 1e-12

OUT_ROOT = ROOT / "output" / "resnet18_adversarial_localisation_pilot_eps1"
OUT_PER_IMAGE = OUT_ROOT / "pgd_pilot_per_image.csv"
OUT_SUMMARY = OUT_ROOT / "pgd_pilot_summary.csv"
OUT_OUTCOME_SUMMARY = OUT_ROOT / "pgd_pilot_outcome_summary.csv"
OUT_SELECTION = OUT_ROOT / "pgd_pilot_selection.csv"
OUT_CAMS = OUT_ROOT / "pgd_pilot_cams.npz"

IMAGENET_MEAN = torch.tensor(clean_rma.IMAGENET_MEAN, dtype=torch.float32).view(
    1, 3, 1, 1
)
IMAGENET_STD = torch.tensor(clean_rma.IMAGENET_STD, dtype=torch.float32).view(
    1, 3, 1, 1
)


def stable_hash(value):
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16)


def deterministic_take(frame, n):
    frame = frame.copy()
    frame["_pilot_hash"] = frame["image_path"].astype(str).map(stable_hash)
    return (
        frame.sort_values(["_pilot_hash", "image_path"])
        .head(n)
        .drop(columns="_pilot_hash")
        .reset_index(drop=True)
    )


def make_pilot_frame():
    """Build the pilot from the frozen clean-RMA evaluation population."""
    evaluation, _ = clean_rma.load_evaluation_population()

    specs = [
        ("dev_val", "digital_1", 153),
        ("dev_val", "digital_2", 153),
        ("official_test", "facedancer", 150),
    ]

    groups = []
    for split, variant, expected_available in specs:
        sub = evaluation[
            (evaluation["evaluation_split"] == split)
            & (evaluation["variant"] == variant)
        ].copy()

        if len(sub) != expected_available:
            raise RuntimeError(
                f"Unexpected frozen population for {split}/{variant}: "
                f"expected={expected_available}, actual={len(sub)}"
            )

        sub = deterministic_take(sub, PILOT_N_PER_GROUP)
        if len(sub) != PILOT_N_PER_GROUP:
            raise RuntimeError(
                f"Pilot selection failed for {split}/{variant}: "
                f"expected={PILOT_N_PER_GROUP}, actual={len(sub)}"
            )
        groups.append(sub)

    frame = pd.concat(groups, ignore_index=True)
    expected_total = len(specs) * PILOT_N_PER_GROUP

    if len(frame) != expected_total:
        raise RuntimeError(
            f"Unexpected pilot size: expected={expected_total}, actual={len(frame)}"
        )

    if frame["image_path"].duplicated().any():
        dupes = frame.loc[
            frame["image_path"].duplicated(keep=False), "image_path"
        ].tolist()
        raise RuntimeError(f"Duplicate pilot image paths: {dupes}")

    return frame


def prepare_pilot_geometry(frame):
    """Reuse the exact frozen annotation projection/clipping code."""
    regions = clean_rma.load_regions()
    annotation_info, region_table, clipped_count = clean_rma.prepare_annotation_geometry(
        frame, regions
    )

    if len(annotation_info) != len(frame):
        raise RuntimeError(
            "Annotation geometry misalignment: "
            f"infos={len(annotation_info)}, rows={len(frame)}"
        )

    return annotation_info, region_table, clipped_count


def build_content_masks(annotation_info, device):
    """Build document-content masks through the frozen clean helper."""
    masks = []

    for info in annotation_info:
        mask_np = clean_rma.content_mask(info)

        if mask_np.shape != (clean_rma.CONTENT_H, clean_rma.CANVAS_W):
            raise RuntimeError(
                f"Unexpected frozen content-mask shape: {mask_np.shape}"
            )

        masks.append(
            torch.from_numpy(mask_np.astype(np.float32, copy=False)).unsqueeze(0)
        )

    result = torch.stack(masks, dim=0).to(device)
    expected = (
        len(annotation_info),
        1,
        clean_rma.CONTENT_H,
        clean_rma.CANVAS_W,
    )

    if tuple(result.shape) != expected:
        raise RuntimeError(
            f"Unexpected content-mask tensor shape: "
            f"expected={expected}, actual={tuple(result.shape)}"
        )

    return result


def verify_padding_frozen(clean_x, adv_x, content_mask):
    padding_mask = 1.0 - content_mask
    max_error = (((adv_x - clean_x).abs() * padding_mask).amax().item())

    if max_error != 0.0:
        raise RuntimeError(
            "Artificial padding changed during attack: "
            f"max_abs_error={max_error:.12g}"
        )

    return max_error


def pixel_linf(clean_x, adv_x, content_mask):
    """Return per-image L_inf in RGB pixel units over document content."""
    std = IMAGENET_STD.to(device=clean_x.device, dtype=clean_x.dtype)
    delta_pixel = (adv_x - clean_x) * std * content_mask
    return delta_pixel.abs().flatten(1).amax(dim=1)


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
    Targeted PGD toward class 0 (bonafide).

    The attack is defined in resized RGB pixel space before ImageNet
    normalisation, implemented equivalently in normalised coordinates.
    """
    device = clean_x.device
    dtype = clean_x.dtype

    mean = IMAGENET_MEAN.to(device=device, dtype=dtype)
    std = IMAGENET_STD.to(device=device, dtype=dtype)

    eps_norm = epsilon_pixel / std
    alpha_norm = alpha_pixel / std
    lower_valid = (0.0 - mean) / std
    upper_valid = (1.0 - mean) / std

    # Random start inside the RGB pixel-space L_inf ball.
    random_delta_pixel = torch.empty_like(clean_x).uniform_(
        -epsilon_pixel, epsilon_pixel
    )
    adv = clean_x + (random_delta_pixel / std) * content_mask

    # Valid RGB range on document content only.
    clipped = torch.maximum(torch.minimum(adv, upper_valid), lower_valid)
    adv = clipped * content_mask + clean_x * (1.0 - content_mask)

    # Initial epsilon projection.
    delta = adv - clean_x
    delta = torch.maximum(torch.minimum(delta, eps_norm), -eps_norm)
    adv = clean_x + delta * content_mask
    adv = adv * content_mask + clean_x * (1.0 - content_mask)

    target = torch.zeros(clean_x.shape[0], dtype=torch.long, device=device)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        logits = model(adv)

        # Targeted attack = gradient descent on CE for bonafide target.
        loss = F.cross_entropy(logits, target, reduction="sum")
        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]

        adv_next = adv - alpha_norm * grad.sign() * content_mask

        delta = adv_next - clean_x
        delta = torch.maximum(torch.minimum(delta, eps_norm), -eps_norm)
        adv_next = clean_x + delta * content_mask

        clipped = torch.maximum(
            torch.minimum(adv_next, upper_valid),
            lower_valid,
        )
        adv_next = clipped * content_mask + clean_x * (1.0 - content_mask)

        delta = adv_next - clean_x
        delta = torch.maximum(torch.minimum(delta, eps_norm), -eps_norm)
        adv = clean_x + delta * content_mask

        # Padding must be exact after every step.
        adv = adv * content_mask + clean_x * (1.0 - content_mask)

    adv = adv.detach()

    verify_padding_frozen(clean_x, adv, content_mask)

    linf = pixel_linf(clean_x, adv, content_mask)
    if (linf > epsilon_pixel + 1e-6).any():
        raise RuntimeError(
            "PGD escaped pixel-space L_inf bound: "
            f"max={linf.max().item():.8f}, epsilon={epsilon_pixel:.8f}"
        )

    return adv


def frozen_localisation_metrics(cam, infos):
    """Compute A/E/mu_w/PG with frozen clean_rma.compute_rma_metrics()."""
    cam_np = cam.detach().cpu().numpy()

    if len(cam_np) != len(infos):
        raise RuntimeError(
            f"CAM/info batch mismatch: cams={len(cam_np)}, infos={len(infos)}"
        )

    rows = []
    for index, info in enumerate(infos):
        metrics = clean_rma.compute_rma_metrics(cam_np[index], info)
        rows.append(
            {
                "A": float(metrics["rma_A"]),
                "E": float(metrics["rma_E"]),
                "mu_w": float(metrics["rma_mu_w"]),
                "PG": float(metrics["rma_PG"]),
            }
        )

    return {
        key: torch.tensor(
            [row[key] for row in rows],
            device=cam.device,
            dtype=cam.dtype,
        )
        for key in ("A", "E", "mu_w", "PG")
    }


def cam_change_metrics(clean_cam, adv_cam, content_mask):
    """
    Compare clean/adversarial CAM maps over document content.

    Cosine and TV are NaN if either map has zero content mass.
    """
    mask = content_mask[:, 0]
    clean_vec = (clean_cam * mask).flatten(1)
    adv_vec = (adv_cam * mask).flatten(1)

    clean_mass = clean_vec.sum(dim=1)
    adv_mass = adv_vec.sum(dim=1)

    clean_zero = clean_mass <= ZERO_CAM_EPS
    adv_zero = adv_mass <= ZERO_CAM_EPS
    valid = (~clean_zero) & (~adv_zero)

    cosine = torch.full(
        (clean_cam.shape[0],),
        float("nan"),
        device=clean_cam.device,
        dtype=clean_cam.dtype,
    )
    tv = torch.full_like(cosine, float("nan"))

    if valid.any():
        cosine[valid] = F.cosine_similarity(
            clean_vec[valid],
            adv_vec[valid],
            dim=1,
            eps=ZERO_CAM_EPS,
        )

        p = clean_vec[valid] / clean_mass[valid][:, None]
        q = adv_vec[valid] / adv_mass[valid][:, None]
        tv[valid] = 0.5 * (p - q).abs().sum(dim=1)

    return {
        "cosine": cosine,
        "tv": tv,
        "clean_zero": clean_zero,
        "adv_zero": adv_zero,
        "clean_mass": clean_mass,
        "adv_mass": adv_mass,
    }


def safe_relative_change(before, after):
    if not np.isfinite(before) or abs(before) <= ZERO_CAM_EPS:
        return np.nan
    return (after - before) / before


def make_group_summary(per_image):
    rows = []

    for (split, variant), group in per_image.groupby(
        ["evaluation_split", "variant"],
        sort=False,
    ):
        clean_correct_group = group[group["clean_correct"]]
        attack_success_rate = (
            clean_correct_group["attack_success"].mean()
            if len(clean_correct_group)
            else np.nan
        )

        rows.append(
            {
                "evaluation_split": split,
                "variant": variant,
                "n": len(group),
                "n_clean_correct": int(group["clean_correct"].sum()),
                "n_adv_correct": int(group["adv_correct"].sum()),
                "n_attack_success": int(group["attack_success"].sum()),
                "clean_attack_accuracy": group["clean_correct"].mean(),
                "adv_attack_accuracy": group["adv_correct"].mean(),
                "attack_success_rate_clean_correct": attack_success_rate,
                "clean_p_attack_mean": group["clean_probability_attack"].mean(),
                "adv_p_attack_mean": group["adv_probability_attack"].mean(),
                "clean_margin_mean": group["clean_margin"].mean(),
                "adv_margin_mean": group["adv_margin"].mean(),
                "A_mean": group["A"].mean(),
                "E_clean_mean": group["E_clean"].mean(),
                "E_adv_mean": group["E_adv"].mean(),
                "delta_E_mean": group["delta_E"].mean(),
                "relative_E_change_mean": group["relative_E_change"].mean(),
                "mu_w_clean_mean": group["mu_w_clean"].mean(),
                "mu_w_adv_mean": group["mu_w_adv"].mean(),
                "delta_mu_w_mean": group["delta_mu_w"].mean(),
                "relative_mu_w_change_mean": group[
                    "relative_mu_w_change"
                ].mean(),
                "PG_clean": group["PG_clean"].mean(),
                "PG_adv": group["PG_adv"].mean(),
                "delta_PG": group["delta_PG"].mean(),
                "cam_cosine_mean": group["cam_cosine"].mean(),
                "cam_tv_mean": group["cam_tv"].mean(),
                "clean_zero_cam_fraction": group["clean_zero_cam"].mean(),
                "adv_zero_cam_fraction": group["adv_zero_cam"].mean(),
                "pixel_linf_mean": group["pixel_linf"].mean(),
                "pixel_linf_max": group["pixel_linf"].max(),
                "padding_max_abs_error": group["padding_max_abs_error"].max(),
                "max_saved_probability_abs_error": group[
                    "saved_probability_abs_error"
                ].max(),
            }
        )

    return pd.DataFrame(rows)


def make_outcome_summary(per_image):
    """
    Split clean-correct images by whether the classifier attack succeeded.
    """
    base = per_image[per_image["clean_correct"]].copy()
    if base.empty:
        return pd.DataFrame()

    rows = []

    for (split, variant, attack_success), group in base.groupby(
        ["evaluation_split", "variant", "attack_success"],
        sort=False,
    ):
        rows.append(
            {
                "evaluation_split": split,
                "variant": variant,
                "attack_success": bool(attack_success),
                "n": len(group),
                "delta_E_mean": group["delta_E"].mean(),
                "delta_E_median": group["delta_E"].median(),
                "relative_E_change_mean": group["relative_E_change"].mean(),
                "relative_E_change_median": group[
                    "relative_E_change"
                ].median(),
                "delta_mu_w_mean": group["delta_mu_w"].mean(),
                "delta_mu_w_median": group["delta_mu_w"].median(),
                "relative_mu_w_change_mean": group[
                    "relative_mu_w_change"
                ].mean(),
                "relative_mu_w_change_median": group[
                    "relative_mu_w_change"
                ].median(),
                "delta_PG_mean": group["delta_PG"].mean(),
                "cam_cosine_mean": group["cam_cosine"].mean(),
                "cam_tv_mean": group["cam_tv"].mean(),
                "adv_zero_cam_fraction": group["adv_zero_cam"].mean(),
                "adv_probability_attack_mean": group[
                    "adv_probability_attack"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    checkpoint_sha = clean_rma.verify_checkpoint()
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Wrong frozen ResNet checkpoint"
            f"\nexpected: {EXPECTED_CHECKPOINT_SHA256}"
            f"\nactual:   {checkpoint_sha}"
        )

    clean_rma.verify_inventory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clean_rma.load_model(device)

    frame = make_pilot_frame()
    annotation_info, region_table, clipped_count = prepare_pilot_geometry(frame)

    frame.to_csv(OUT_SELECTION, index=False)

    dataset = clean_rma.PolicyCDataset(frame)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_content_masks = build_content_masks(annotation_info, device)

    gradcam = clean_rma.GradCAM(model, model.layer4[-1])

    records = []
    cam_keys = []
    clean_low_cams = []
    adv_low_cams = []
    processed = 0

    try:
        for batch_x, labels, indices in loader:
            batch_x = batch_x.to(device)
            labels = labels.to(device)
            batch_indices = indices.tolist()

            if not torch.all(labels == 1):
                raise RuntimeError("Pilot contains a non-attack sample")

            index_tensor = indices.to(device)
            content_mask = all_content_masks[index_tensor]
            batch_infos = [annotation_info[i] for i in batch_indices]

            # Clean state.
            clean_result = gradcam.generate(batch_x)
            clean_prob = clean_result["probability"]
            clean_margin = clean_result["margin"]
            clean_cam = clean_result["full_cam"]
            clean_metrics = frozen_localisation_metrics(clean_cam, batch_infos)

            # PGD.
            adv_x = pgd_target_bonafide(
                model,
                batch_x,
                content_mask,
            )

            padding_error = verify_padding_frozen(
                batch_x,
                adv_x,
                content_mask,
            )
            linf = pixel_linf(
                batch_x,
                adv_x,
                content_mask,
            )

            # Adversarial state.
            adv_result = gradcam.generate(adv_x)
            adv_prob = adv_result["probability"]
            adv_margin = adv_result["margin"]
            adv_cam = adv_result["full_cam"]
            adv_metrics = frozen_localisation_metrics(adv_cam, batch_infos)

            # A is geometry-only.
            a_error = (
                clean_metrics["A"] - adv_metrics["A"]
            ).abs().max().item()

            if a_error > 1e-7:
                raise RuntimeError(
                    "Clean/adversarial A mismatch: "
                    f"max_abs_error={a_error:.12g}"
                )

            cam_change = cam_change_metrics(
                clean_cam,
                adv_cam,
                content_mask,
            )

            for j, global_index in enumerate(batch_indices):
                row = frame.iloc[global_index]

                clean_probability = clean_prob[j].item()
                adversarial_probability = adv_prob[j].item()

                clean_correct = bool(
                    clean_probability >= ATTACK_THRESHOLD
                )
                adv_correct = bool(
                    adversarial_probability >= ATTACK_THRESHOLD
                )
                attack_success = bool(
                    clean_correct and not adv_correct
                )

                A = clean_metrics["A"][j].item()
                E0 = clean_metrics["E"][j].item()
                E1 = adv_metrics["E"][j].item()
                mu0 = clean_metrics["mu_w"][j].item()
                mu1 = adv_metrics["mu_w"][j].item()
                pg0 = clean_metrics["PG"][j].item()
                pg1 = adv_metrics["PG"][j].item()

                saved_probability = float(
                    row["saved_attack_probability"]
                )

                records.append(
                    {
                        "evaluation_split": row["evaluation_split"],
                        "variant": row["variant"],
                        "selection_scope": row["selection_scope"],
                        "evidence_role": row["evidence_role"],
                        "file_stem": row["file_stem"],
                        "hardware_source": row["hardware_source"],
                        "image_path": row["image_path"],
                        "label": int(labels[j].item()),
                        "epsilon_pixel": EPSILON_PIXEL,
                        "epsilon_255": EPSILON_PIXEL * 255.0,
                        "alpha_pixel": ALPHA_PIXEL,
                        "alpha_255": ALPHA_PIXEL * 255.0,
                        "pgd_steps": PGD_STEPS,
                        "pixel_linf": linf[j].item(),
                        "padding_max_abs_error": padding_error,
                        "saved_clean_probability_attack": saved_probability,
                        "clean_probability_attack": clean_probability,
                        "saved_probability_abs_error": abs(
                            clean_probability - saved_probability
                        ),
                        "adv_probability_attack": adversarial_probability,
                        "clean_margin": clean_margin[j].item(),
                        "adv_margin": adv_margin[j].item(),
                        "clean_correct": clean_correct,
                        "adv_correct": adv_correct,
                        "attack_success": attack_success,
                        "A": A,
                        "E_clean": E0,
                        "E_adv": E1,
                        "delta_E": E1 - E0,
                        "relative_E_change": safe_relative_change(E0, E1),
                        "mu_w_clean": mu0,
                        "mu_w_adv": mu1,
                        "delta_mu_w": mu1 - mu0,
                        "relative_mu_w_change": safe_relative_change(mu0, mu1),
                        "PG_clean": pg0,
                        "PG_adv": pg1,
                        "delta_PG": pg1 - pg0,
                        "cam_cosine": cam_change["cosine"][j].item(),
                        "cam_tv": cam_change["tv"][j].item(),
                        "clean_zero_cam": bool(
                            cam_change["clean_zero"][j].item()
                        ),
                        "adv_zero_cam": bool(
                            cam_change["adv_zero"][j].item()
                        ),
                        "clean_content_cam_mass": cam_change[
                            "clean_mass"
                        ][j].item(),
                        "adv_content_cam_mass": cam_change[
                            "adv_mass"
                        ][j].item(),
                    }
                )

                cam_keys.append(str(row["image_path"]))
                clean_low_cams.append(
                    clean_result["low_cam"][j]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                adv_low_cams.append(
                    adv_result["low_cam"][j]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )

            processed += len(batch_indices)
            print(f"processed {processed}/{len(frame)}", flush=True)

    finally:
        gradcam.remove()

    per_image = pd.DataFrame(records)

    if len(per_image) != len(frame):
        raise RuntimeError(
            "Per-image output count mismatch: "
            f"expected={len(frame)}, actual={len(per_image)}"
        )

    per_image.to_csv(OUT_PER_IMAGE, index=False)

    summary = make_group_summary(per_image)
    summary.to_csv(OUT_SUMMARY, index=False)

    outcome_summary = make_outcome_summary(per_image)
    outcome_summary.to_csv(OUT_OUTCOME_SUMMARY, index=False)

    np.savez_compressed(
        OUT_CAMS,
        keys=np.array(cam_keys, dtype=object),
        clean_maps=np.stack(clean_low_cams),
        adv_maps=np.stack(adv_low_cams),
    )

    # Mandatory audit.
    group_counts = (
        frame.groupby(["evaluation_split", "variant"], sort=False)
        .size()
        .to_dict()
    )

    expected_group_counts = {
        ("dev_val", "digital_1"): PILOT_N_PER_GROUP,
        ("dev_val", "digital_2"): PILOT_N_PER_GROUP,
        ("official_test", "facedancer"): PILOT_N_PER_GROUP,
    }

    if group_counts != expected_group_counts:
        raise RuntimeError(
            "Pilot group-count audit failed"
            f"\nexpected={expected_group_counts}"
            f"\nactual={group_counts}"
        )

    max_linf = per_image["pixel_linf"].max()
    if max_linf > EPSILON_PIXEL + 1e-6:
        raise RuntimeError(
            f"Final L_inf audit failed: max={max_linf:.8f}"
        )

    max_padding_error = per_image["padding_max_abs_error"].max()
    if max_padding_error != 0.0:
        raise RuntimeError(
            f"Final padding audit failed: {max_padding_error}"
        )

    print()
    print("RESNET18 ADVERSARIAL LOCALISATION PILOT")
    print()
    print(f"checkpoint SHA: {checkpoint_sha}")
    print(f"device:         {device}")

    if device.type == "cuda":
        print(f"gpu:            {torch.cuda.get_device_name(device)}")

    print(f"pilot images:   {len(frame)}")
    print(f"clipped GT rectangles in pilot: {clipped_count}")
    print("attack: targeted bonafide pixel-space L_inf PGD")
    print(f"epsilon: {EPSILON_PIXEL * 255.0:.1f}/255")
    print(f"alpha:   {ALPHA_PIXEL * 255.0:.1f}/255")
    print(f"steps:   {PGD_STEPS}")
    print("random start: yes")
    print("attack support: document content only")
    print("padding: frozen exactly")
    print("localisation evaluator: frozen clean_rma.compute_rma_metrics")
    print(f"max pixel L_inf: {max_linf:.8f}")
    print(f"max padding change: {max_padding_error:.12g}")
    print(
        "max clean-vs-saved probability error: "
        f"{per_image['saved_probability_abs_error'].max():.8f}"
    )

    print()
    print("PRIMARY PILOT SUMMARY:")
    print(summary.to_string(index=False))

    print()
    print("CLEAN-CORRECT OUTCOME DIAGNOSTIC:")
    if outcome_summary.empty:
        print("(no clean-correct images)")
    else:
        print(outcome_summary.to_string(index=False))

    print()
    print("OUTPUTS:")
    print(f"  selection:       {OUT_SELECTION}")
    print(f"  per image:       {OUT_PER_IMAGE}")
    print(f"  summary:         {OUT_SUMMARY}")
    print(f"  outcome summary: {OUT_OUTCOME_SUMMARY}")
    print(f"  CAMs:            {OUT_CAMS}")

    print()
    print("STOP HERE.")
    print("Do not run another epsilon until this pilot has been inspected.")


if __name__ == "__main__":
    main()