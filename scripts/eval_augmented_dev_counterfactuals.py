#!/usr/bin/env python3
"""Evaluate the counterfactual-augmented ResNet on held-out dev interventions.

Uses the already-audited counterfactual construction implementation from
eval_counterfactual_restoration.py.

Population:
    digital_1 dev: 153 images / 51 stems
    digital_2 dev: 150 images / 50 stems
        (chinese-001_03 digital_2 excluded for the known geometry mismatch)

Modes:
    original
    face_restored
    text_restored
    both_restored

The matched bona-fide floor is taken from THIS augmented model's own
natural-dev predictions, not from the earlier baseline model.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

from eval_counterfactual_restoration import (
    ROOT,
    DATA,
    load_valid_pairs,
    load_manifests,
    load_region_boxes,
    jpeg_decode_rgb,
    jpeg_parameters,
    recompress_bonafide_like_attack,
    mask_from_boxes,
    policy_c_bytes,
    tensor_from_policy_bytes,
    score_batch,
    sha256_bytes,
)


EXPERIMENT = (
    "POST_HOC_COUNTERFACTUAL_AUGMENTED_"
    "COMPRESSION_CONTROLLED_EXPERIMENT"
)

CHECKPOINT = (
    ROOT
    / "runs"
    / "post_hoc_counterfactual_augmented_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

CHECKPOINT_HASH = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_checkpoint.sha256"
)

NATURAL_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_predictions.csv"
)

BASELINE_CF_SUMMARY = (
    ROOT
    / "output"
    / "counterfactual_policy_c_seed10_summary.csv"
)

OUT_PREDICTIONS = (
    ROOT
    / "output"
    / "counterfactual_augmented_seed10_dev_predictions.csv"
)

OUT_SUMMARY = (
    ROOT
    / "output"
    / "counterfactual_augmented_seed10_dev_summary.csv"
)

OUT_COMPARISON = (
    ROOT
    / "output"
    / "counterfactual_augmented_vs_baseline_dev.csv"
)

SEED = 10
N_BOOT = 5000

MODES = [
    "original",
    "face_restored",
    "text_restored",
    "both_restored",
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


def load_augmented_model(device):
    if not CHECKPOINT_HASH.is_file():
        raise RuntimeError(
            "Augmented checkpoint hash missing. "
            "Freeze the checkpoint before evaluation."
        )

    expected_sha = (
        CHECKPOINT_HASH
        .read_text()
        .strip()
        .split()[0]
    )

    actual_sha = sha256_file(
        CHECKPOINT
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Augmented checkpoint SHA mismatch:\n"
            f"expected {expected_sha}\n"
            f"actual   {actual_sha}"
        )

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

    expected_metadata = {
        "experiment":
            EXPERIMENT,
        "seed":
            10,
        "architecture":
            "resnet18",
        "resolution":
            "r512",
        "stage":
            "full",
        "epoch":
            2,
    }

    for key, expected in (
        expected_metadata.items()
    ):
        actual = checkpoint.get(
            key
        )

        if actual != expected:
            raise RuntimeError(
                "Unexpected augmented checkpoint "
                f"metadata: {key}={actual!r}, "
                f"expected {expected!r}"
            )

    if not np.isclose(
        float(
            checkpoint["dev_auc"]
        ),
        1.0,
    ):
        raise RuntimeError(
            "Expected selected dev AUROC 1.0"
        )

    if checkpoint.get(
        "policy"
    ) != (
        "Q75 -> matched deterministic Q50-90"
    ):
        raise RuntimeError(
            "Unexpected compression policy"
        )

    augmentation = str(
        checkpoint.get(
            "augmentation",
            "",
        )
    )

    required_terms = [
        "face_restored",
        "text_restored",
        "both_restored",
    ]

    if not all(
        term in augmentation
        for term in required_terms
    ):
        raise RuntimeError(
            "Checkpoint does not record the "
            "expected counterfactual augmentation"
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
        actual_sha,
        checkpoint,
    )


def load_natural_dev_predictions():
    df = pd.read_csv(
        NATURAL_PREDICTIONS
    )

    required = {
        "split",
        "image_path",
        "file_stem",
        "traffic_type",
        "hardware_source",
        "label",
        "attack_probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Natural prediction columns missing: "
            f"{sorted(missing)}"
        )

    dev = df[
        df["split"]
        == "dev_val"
    ].copy()

    if len(dev) != 459:
        raise RuntimeError(
            f"Expected 459 natural dev rows, "
            f"got {len(dev)}"
        )

    counts = (
        dev.traffic_type
        .value_counts()
        .to_dict()
    )

    if counts != {
        "attack": 306,
        "bonafide": 153,
    }:
        raise RuntimeError(
            f"Unexpected dev population: "
            f"{counts}"
        )

    if (
        dev.file_stem.nunique()
        != 51
    ):
        raise RuntimeError(
            "Expected 51 natural dev stems"
        )

    if dev.image_path.duplicated().any():
        raise RuntimeError(
            "Duplicate natural-dev paths"
        )

    attack_lookup = (
        dev[
            dev.traffic_type
            == "attack"
        ]
        .set_index(
            "image_path"
        )[
            "attack_probability"
        ]
    )

    bonafide = dev[
        dev.traffic_type
        == "bonafide"
    ].copy()

    key = [
        "file_stem",
        "hardware_source",
    ]

    if bonafide.duplicated(
        key
    ).any():
        raise RuntimeError(
            "Duplicate matched bona-fide keys"
        )

    bona_lookup = (
        bonafide
        .set_index(key)
        ["attack_probability"]
    )

    return (
        attack_lookup,
        bona_lookup,
    )


def stem_bootstrap(group):
    """Cluster bootstrap by card identity."""

    stems = np.array(
        sorted(
            group.file_stem.unique()
        )
    )

    groups = {
        stem: (
            group[
                group.file_stem
                == stem
            ][
                [
                    "delta_vs_original",
                    "delta_vs_bonafide",
                    "attack_probability",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )
        for stem in stems
    }

    rng = np.random.default_rng(
        SEED
    )

    boot = np.empty(
        (
            N_BOOT,
            6,
        ),
        dtype=float,
    )

    for i in range(
        N_BOOT
    ):
        sampled = rng.choice(
            stems,
            size=len(stems),
            replace=True,
        )

        values = np.concatenate(
            [
                groups[stem]
                for stem in sampled
            ],
            axis=0,
        )

        d_original = values[:, 0]
        d_bona = values[:, 1]
        probability = values[:, 2]

        boot[i] = [
            d_original.mean(),
            np.median(
                d_original
            ),
            (
                d_original < 0
            ).mean(),
            d_bona.mean(),
            np.median(
                d_bona
            ),
            (
                probability >= 0.5
            ).mean(),
        ]

    low = np.quantile(
        boot,
        0.025,
        axis=0,
    )

    high = np.quantile(
        boot,
        0.975,
        axis=0,
    )

    return {
        "mean_delta_original_ci_low":
            low[0],
        "mean_delta_original_ci_high":
            high[0],

        "median_delta_original_ci_low":
            low[1],
        "median_delta_original_ci_high":
            high[1],

        "fraction_decreased_ci_low":
            low[2],
        "fraction_decreased_ci_high":
            high[2],

        "mean_delta_bonafide_ci_low":
            low[3],
        "mean_delta_bonafide_ci_high":
            high[3],

        "median_delta_bonafide_ci_low":
            low[4],
        "median_delta_bonafide_ci_high":
            high[4],

        "attack_recall_ci_low":
            low[5],
        "attack_recall_ci_high":
            high[5],
    }


def build_summary(predictions):
    df = predictions.copy()

    original = (
        df[
            df["mode"]
            == "original"
        ]
        .set_index(
            "attack_path"
        )[
            "attack_probability"
        ]
    )

    if len(original) != 303:
        raise RuntimeError(
            f"Expected 303 dev original scores, "
            f"got {len(original)}"
        )

    df[
        "original_probability"
    ] = (
        df.attack_path.map(
            original
        )
    )

    if (
        df.original_probability
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Original-score join failed"
        )

    df[
        "delta_vs_original"
    ] = (
        df.attack_probability
        - df.original_probability
    )

    df[
        "delta_vs_bonafide"
    ] = (
        df.attack_probability
        - df.bona_fide_probability
    )

    rows = []

    for variant in [
        "digital_1",
        "digital_2",
    ]:
        for mode in MODES:
            group = df[
                (
                    df["variant"]
                    == variant
                )
                & (
                    df["mode"]
                    == mode
                )
            ].copy()

            rows.append(
                {
                    "variant":
                        variant,

                    "mode":
                        mode,

                    "n_images":
                        len(group),

                    "n_stems":
                        group.file_stem.nunique(),

                    "mean_attack_probability":
                        group.attack_probability.mean(),

                    "median_attack_probability":
                        group.attack_probability.median(),

                    "attack_recall_at_0_5":
                        (
                            group.attack_probability
                            >= 0.5
                        ).mean(),

                    "mean_bonafide_probability":
                        group.bona_fide_probability.mean(),

                    "median_bonafide_probability":
                        group.bona_fide_probability.median(),

                    "mean_delta_vs_original":
                        group.delta_vs_original.mean(),

                    "median_delta_vs_original":
                        group.delta_vs_original.median(),

                    "fraction_score_decreased":
                        (
                            group.delta_vs_original
                            < 0
                        ).mean(),

                    "mean_delta_vs_bonafide":
                        group.delta_vs_bonafide.mean(),

                    "median_delta_vs_bonafide":
                        group.delta_vs_bonafide.median(),

                    "mean_restored_fraction":
                        group.restored_fraction.mean(),

                    "median_restored_fraction":
                        group.restored_fraction.median(),

                    **stem_bootstrap(
                        group
                    ),
                }
            )

    summary = pd.DataFrame(
        rows
    )

    expected = {
        "digital_1":
            (153, 51),

        "digital_2":
            (150, 50),
    }

    for variant, (
        n_images,
        n_stems,
    ) in expected.items():
        check = summary[
            summary.variant
            == variant
        ]

        if len(check) != 4:
            raise RuntimeError(
                f"Missing modes for "
                f"{variant}"
            )

        if not (
            check.n_images
            == n_images
        ).all():
            raise RuntimeError(
                f"Unexpected image count "
                f"for {variant}"
            )

        if not (
            check.n_stems
            == n_stems
        ).all():
            raise RuntimeError(
                f"Unexpected stem count "
                f"for {variant}"
            )

    return (
        df,
        summary,
    )


def compare_with_baseline(
    augmented_summary,
):
    baseline = pd.read_csv(
        BASELINE_CF_SUMMARY
    )

    baseline = baseline[
        baseline["split"]
        == "dev_val"
    ][
        [
            "variant",
            "mode",
            "mean_attack_probability",
            "median_attack_probability",
            "attack_recall_at_0_5",
            "mean_delta_vs_bonafide",
        ]
    ].copy()

    baseline = baseline.rename(
        columns={
            "mean_attack_probability":
                "baseline_mean_probability",

            "median_attack_probability":
                "baseline_median_probability",

            "attack_recall_at_0_5":
                "baseline_attack_recall",

            "mean_delta_vs_bonafide":
                "baseline_delta_vs_bonafide",
        }
    )

    augmented = augmented_summary[
        [
            "variant",
            "mode",
            "mean_attack_probability",
            "median_attack_probability",
            "attack_recall_at_0_5",
            "mean_delta_vs_bonafide",
        ]
    ].copy()

    augmented = augmented.rename(
        columns={
            "mean_attack_probability":
                "augmented_mean_probability",

            "median_attack_probability":
                "augmented_median_probability",

            "attack_recall_at_0_5":
                "augmented_attack_recall",

            "mean_delta_vs_bonafide":
                "augmented_delta_vs_bonafide",
        }
    )

    comparison = baseline.merge(
        augmented,
        on=[
            "variant",
            "mode",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != 8:
        raise RuntimeError(
            "Baseline/augmented comparison "
            "did not produce eight rows"
        )

    comparison[
        "change_mean_probability"
    ] = (
        comparison.augmented_mean_probability
        - comparison.baseline_mean_probability
    )

    comparison[
        "change_attack_recall"
    ] = (
        comparison.augmented_attack_recall
        - comparison.baseline_attack_recall
    )

    return comparison


def main():
    # --------------------------------------------------
    # Only valid held-out dev attacks.
    # --------------------------------------------------

    valid = load_valid_pairs()

    valid = valid[
        valid["split"]
        == "dev_val"
    ].copy()

    counts = (
        valid.variant
        .value_counts()
        .to_dict()
    )

    if counts != {
        "digital_1": 153,
        "digital_2": 150,
    }:
        raise RuntimeError(
            f"Unexpected valid dev attacks: "
            f"{counts}"
        )

    print(
        "Held-out counterfactual population:"
        f"\n  digital_1: 153 images / "
        f"{valid.loc[valid.variant == 'digital_1', 'file_stem'].nunique()} stems"
        f"\n  digital_2: 150 images / "
        f"{valid.loc[valid.variant == 'digital_2', 'file_stem'].nunique()} stems"
        "\n  project_train counterfactuals: NOT evaluated"
    )

    manifest = load_manifests()

    valid_paths = set(
        valid.attack_path
    )

    region_boxes = load_region_boxes(
        valid_paths
    )

    (
        natural_attack_lookup,
        bona_lookup,
    ) = load_natural_dev_predictions()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        model,
        checkpoint_sha,
        checkpoint,
    ) = load_augmented_model(
        device
    )

    print(
        "\nAugmented frozen checkpoint:"
        f"\n  SHA256:  {checkpoint_sha}"
        f"\n  stage:   {checkpoint['stage']}"
        f"\n  epoch:   {checkpoint['epoch']}"
        f"\n  dev_auc: {checkpoint['dev_auc']:.4f}"
        f"\n  device:  {device}"
    )

    rows = []

    clipped_boxes = 0

    valid = valid.sort_values(
        [
            "variant",
            "file_stem",
            "hardware_source",
        ]
    )

    for i, pair in enumerate(
        valid.itertuples(
            index=False
        ),
        start=1,
    ):
        attack_path = (
            pair.attack_path
        )

        bona_path = (
            pair.bonafide_path
        )

        attack_manifest = (
            manifest.loc[
                attack_path
            ]
        )

        bona_manifest = (
            manifest.loc[
                bona_path
            ]
        )

        attack_raw = (
            DATA
            / attack_path
        ).read_bytes()

        bona_raw = (
            DATA
            / bona_path
        ).read_bytes()

        if (
            sha256_bytes(
                attack_raw
            )
            != attack_manifest.image_sha256
        ):
            raise RuntimeError(
                f"Attack SHA mismatch: "
                f"{attack_path}"
            )

        if (
            sha256_bytes(
                bona_raw
            )
            != bona_manifest.image_sha256
        ):
            raise RuntimeError(
                f"Bona-fide SHA mismatch: "
                f"{bona_path}"
            )

        attack_rgb = jpeg_decode_rgb(
            attack_raw
        )

        bona_rgb = jpeg_decode_rgb(
            bona_raw
        )

        if (
            attack_rgb.shape
            != bona_rgb.shape
        ):
            raise RuntimeError(
                "Known geometry exclusion "
                "failed to remove mismatch: "
                f"{attack_path}"
            )

        tables, sampling = (
            jpeg_parameters(
                attack_raw
            )
        )

        source_rgb = (
            recompress_bonafide_like_attack(
                bona_rgb,
                tables,
                sampling,
            )
        )

        h, w = (
            attack_rgb.shape[:2]
        )

        annotation = (
            region_boxes[
                attack_path
            ]
        )

        face_mask, face_clipped = (
            mask_from_boxes(
                h,
                w,
                annotation["face"],
            )
        )

        text_mask, text_clipped = (
            mask_from_boxes(
                h,
                w,
                annotation["text"],
            )
        )

        clipped_boxes += (
            len(face_clipped)
            + len(text_clipped)
        )

        both_mask = (
            face_mask
            | text_mask
        )

        masks = {
            "face_restored":
                face_mask,

            "text_restored":
                text_mask,

            "both_restored":
                both_mask,
        }

        # Original + three restored variants
        # in the SAME model batch.
        tensors = [
            tensor_from_policy_bytes(
                policy_c_bytes(
                    attack_rgb,
                    pair.file_stem,
                    pair.hardware_source,
                )
            )
        ]

        restored_fractions = {}

        for mode in [
            "face_restored",
            "text_restored",
            "both_restored",
        ]:
            restored = (
                attack_rgb.copy()
            )

            mask = masks[
                mode
            ]

            restored[
                mask
            ] = source_rgb[
                mask
            ]

            restored_fractions[
                mode
            ] = float(
                mask.mean()
            )

            tensors.append(
                tensor_from_policy_bytes(
                    policy_c_bytes(
                        restored,
                        pair.file_stem,
                        pair.hardware_source,
                    )
                )
            )

        scores = score_batch(
            model,
            tensors,
            device,
        )

        original_score = float(
            scores[0]
        )

        cached_original = float(
            natural_attack_lookup.loc[
                attack_path
            ]
        )

        pipeline_error = abs(
            original_score
            - cached_original
        )

        if pipeline_error > 1e-3:
            raise RuntimeError(
                "Augmented model in-process "
                "original differs from frozen "
                f"natural prediction: "
                f"{attack_path}, "
                f"error={pipeline_error:.8g}"
            )

        bona_key = (
            pair.file_stem,
            pair.hardware_source,
        )

        try:
            bona_score = float(
                bona_lookup.loc[
                    bona_key
                ]
            )
        except KeyError as exc:
            raise RuntimeError(
                "Missing augmented-model "
                "matched bona-fide score: "
                f"{bona_key}"
            ) from exc

        common = {
            "split":
                "dev_val",

            "variant":
                pair.variant,

            "hardware_source":
                pair.hardware_source,

            "file_stem":
                pair.file_stem,

            "attack_path":
                attack_path,

            "bonafide_path":
                bona_path,

            "checkpoint_sha256":
                checkpoint_sha,

            "bona_fide_probability":
                bona_score,

            "cached_original_probability":
                cached_original,

            "original_pipeline_abs_error":
                pipeline_error,
        }

        rows.append(
            {
                **common,

                "mode":
                    "original",

                "attack_probability":
                    original_score,

                "restored_fraction":
                    0.0,
            }
        )

        for mode, score in zip(
            [
                "face_restored",
                "text_restored",
                "both_restored",
            ],
            scores[1:],
        ):
            rows.append(
                {
                    **common,

                    "mode":
                        mode,

                    "attack_probability":
                        float(score),

                    "restored_fraction":
                        restored_fractions[
                            mode
                        ],
                }
            )

        if (
            i % 50 == 0
            or i == len(valid)
        ):
            print(
                f"processed "
                f"{i}/{len(valid)}"
            )

    predictions = pd.DataFrame(
        rows
    )

    # Save raw result BEFORE summary.
    OUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    print(
        "\nRaw predictions saved."
        f"\nMaximum original pipeline error: "
        f"{predictions.original_pipeline_abs_error.max():.8g}"
        f"\nClipped altered boxes in dev: "
        f"{clipped_boxes}"
    )

    enriched, summary = (
        build_summary(
            predictions
        )
    )

    enriched.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    comparison = (
        compare_with_baseline(
            summary
        )
    )

    comparison.to_csv(
        OUT_COMPARISON,
        index=False,
    )

    display = [
        "variant",
        "mode",
        "n_images",
        "n_stems",
        "mean_attack_probability",
        "median_attack_probability",
        "attack_recall_at_0_5",
        "mean_bonafide_probability",
        "mean_delta_vs_original",
        "median_delta_vs_original",
        "fraction_score_decreased",
        "mean_delta_vs_bonafide",
        "median_delta_vs_bonafide",
        "mean_restored_fraction",
    ]

    print(
        "\nAUGMENTED MODEL — "
        "held-out dev counterfactuals:"
    )

    print(
        summary[
            display
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    compare_display = [
        "variant",
        "mode",
        "baseline_mean_probability",
        "augmented_mean_probability",
        "change_mean_probability",
        "baseline_attack_recall",
        "augmented_attack_recall",
        "change_attack_recall",
        "augmented_delta_vs_bonafide",
    ]

    print(
        "\nCHANGE FROM ORIGINAL "
        "COMPRESSION-CONTROLLED MODEL:"
    )

    print(
        comparison[
            compare_display
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\nPredictions: {OUT_PREDICTIONS}"
        f"\nSummary:     {OUT_SUMMARY}"
        f"\nComparison:  {OUT_COMPARISON}"
    )

    print(
        "\nStop here."
        "\nDo NOT evaluate the official test yet."
    )


if __name__ == "__main__":
    main()