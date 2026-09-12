#!/usr/bin/env python3
"""
Held-out Digital-3 forensic-signal retention audit.

No training. No CNN inference.

For each SAME-SOURCE project-dev pair:

    matched bona-fide parent
    vs
    Digital-3 edited image

measure the amount of local image difference inside the frozen altered-text
annotations under four processing conditions:

    1. native_raw
    2. native_policy_c
    3. r512_raw
    4. r512_policy_c

This separates:

    JPEG intervention effect
    from
    whole-document downscaling effect.

Primary population:
    153 images
    51 held-out card stems
    3 hardware sources

The horizontal ImageNet-mean padding used by the ResNet is not included
in these measurements because it is identical in parent and attack and
therefore contributes exactly zero pairwise difference.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from scipy.ndimage import laplace
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Frozen paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "FantasyID"
)

INVENTORY = (
    ROOT
    / "output"
    / "fantasyid_inventory_2026-09-05_023126.xlsx"
)

PAIRS = (
    ROOT
    / "output"
    / "digital3_same_source_pairs.csv"
)

NATURAL_INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

OUT_DETAIL = (
    ROOT
    / "output"
    / "digital3_signal_retention_dev.csv"
)

OUT_SUMMARY = (
    ROOT
    / "output"
    / "digital3_signal_retention_dev_summary.csv"
)

OUT_HARDWARE = (
    ROOT
    / "output"
    / "digital3_signal_retention_dev_hardware.csv"
)

OUT_CORRELATION = (
    ROOT
    / "output"
    / "digital3_signal_retention_model_correlation.csv"
)

OUT_GEOMETRY_FAILURES = (
    ROOT
    / "output"
    / "digital3_signal_retention_geometry_failures.csv"
)


INVENTORY_SHA256 = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
)

SEED = 10
N_BOOT = 5000

CONTENT_H = 512

STAGES = [
    "native_raw",
    "native_policy_c",
    "r512_raw",
    "r512_policy_c",
]


# ---------------------------------------------------------------------
# Hash / JPEG helpers
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


def sha256_bytes(data):
    return hashlib.sha256(
        data
    ).hexdigest()


def decode_rgb(data):
    from io import BytesIO

    with Image.open(
        BytesIO(data)
    ) as im:
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def encode_jpeg(
    rgb,
    quality,
):
    from io import BytesIO

    buf = BytesIO()

    Image.fromarray(
        rgb
    ).save(
        buf,
        "JPEG",
        quality=int(quality),
        subsampling=2,
        optimize=False,
        progressive=False,
    )

    return buf.getvalue()


def final_q(
    file_stem,
    hardware_source,
):
    """
    Exact frozen Policy-C Q assignment.
    """

    key = (
        f"{file_stem}|"
        f"{hardware_source}"
    ).encode()

    value = int.from_bytes(
        hashlib.sha256(
            key
        ).digest()[:8],
        "big",
    )

    return (
        50
        + value % 41
    )


def policy_c_rgb(
    rgb,
    file_stem,
    hardware_source,
):
    """
    Frozen Policy C:

        RGB
        -> Q75 4:2:0
        -> decode
        -> deterministic Q50-90 4:2:0
        -> decode
    """

    q75 = encode_jpeg(
        rgb,
        75,
    )

    q75_rgb = decode_rgb(
        q75
    )

    q = final_q(
        file_stem,
        hardware_source,
    )

    final = encode_jpeg(
        q75_rgb,
        q,
    )

    return (
        decode_rgb(
            final
        ),
        q,
    )


# ---------------------------------------------------------------------
# Resize exactly like model content resize
# ---------------------------------------------------------------------

def resize_r512(rgb):
    h, w = (
        rgb.shape[:2]
    )

    new_w = int(
        round(
            w
            * CONTENT_H
            / h
        )
    )

    resized = (
        Image.fromarray(
            rgb
        )
        .resize(
            (
                new_w,
                CONTENT_H,
            ),
            Image.Resampling.BILINEAR,
        )
    )

    return np.asarray(
        resized,
        dtype=np.uint8,
    )


def resize_mask_r512(mask):
    h, w = (
        mask.shape
    )

    new_w = int(
        round(
            w
            * CONTENT_H
            / h
        )
    )

    image = Image.fromarray(
        (
            mask.astype(
                np.uint8
            )
            * 255
        )
    )

    image = image.resize(
        (
            new_w,
            CONTENT_H,
        ),
        Image.Resampling.NEAREST,
    )

    return (
        np.asarray(
            image
        )
        > 0
    )


# ---------------------------------------------------------------------
# Frozen altered-region annotations
# ---------------------------------------------------------------------

def load_altered_boxes(
    attack_paths,
):
    if (
        sha256_file(
            INVENTORY
        )
        != INVENTORY_SHA256
    ):
        raise RuntimeError(
            "Inventory SHA mismatch"
        )

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

    subset = regions[
        regions[
            "image_path"
        ].isin(
            attack_paths
        )
    ].copy()

    subset = subset[
        subset[
            "region_provenance_raw"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("altered")
    ]

    boxes = {}

    fields = {}

    for path, group in (
        subset.groupby(
            "image_path"
        )
    ):
        image_boxes = []
        image_fields = []

        for _, row in (
            group.iterrows()
        ):
            x = int(
                round(
                    float(
                        row["x"]
                    )
                )
            )

            y = int(
                round(
                    float(
                        row["y"]
                    )
                )
            )

            w = int(
                round(
                    float(
                        row["width"]
                    )
                )
            )

            h = int(
                round(
                    float(
                        row["height"]
                    )
                )
            )

            if (
                w <= 0
                or h <= 0
            ):
                raise RuntimeError(
                    "Invalid altered box: "
                    f"{path}"
                )

            image_boxes.append(
                (
                    x,
                    y,
                    x + w,
                    y + h,
                )
            )

            image_fields.append(
                str(
                    row[
                        "field_name"
                    ]
                )
                .strip()
                .lower()
            )

        boxes[path] = (
            image_boxes
        )

        fields[path] = (
            image_fields
        )

    missing_paths = (
        set(
            attack_paths
        )
        - set(
            boxes
        )
    )

    if missing_paths:
        raise RuntimeError(
            f"{len(missing_paths)} "
            "Digital-3 dev images "
            "lack altered annotations"
        )

    return (
        boxes,
        fields,
    )


def build_mask(
    height,
    width,
    boxes,
):
    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=bool,
    )

    clipped = 0

    for (
        x0,
        y0,
        x1,
        y1,
    ) in boxes:
        cx0 = max(
            0,
            min(
                width,
                x0,
            ),
        )

        cy0 = max(
            0,
            min(
                height,
                y0,
            ),
        )

        cx1 = max(
            0,
            min(
                width,
                x1,
            ),
        )

        cy1 = max(
            0,
            min(
                height,
                y1,
            ),
        )

        if (
            cx1 <= cx0
            or cy1 <= cy0
        ):
            raise RuntimeError(
                "Altered annotation "
                "does not intersect image"
            )

        if (
            cx0,
            cy0,
            cx1,
            cy1,
        ) != (
            x0,
            y0,
            x1,
            y1,
        ):
            clipped += 1

        mask[
            cy0:cy1,
            cx0:cx1,
        ] = True

    if not mask.any():
        raise RuntimeError(
            "Empty altered mask"
        )

    return (
        mask,
        clipped,
    )


# ---------------------------------------------------------------------
# Forensic-difference metrics
# ---------------------------------------------------------------------

def rgb_to_gray(
    rgb,
):
    rgb = rgb.astype(
        np.float32
    )

    return (
        0.299
        * rgb[:, :, 0]
        +
        0.587
        * rgb[:, :, 1]
        +
        0.114
        * rgb[:, :, 2]
    )


def compute_metrics(
    parent_rgb,
    attack_rgb,
    altered_mask,
):
    if (
        parent_rgb.shape
        != attack_rgb.shape
    ):
        raise RuntimeError(
            "Stage image shape mismatch"
        )

    if (
        altered_mask.shape
        != parent_rgb.shape[:2]
    ):
        raise RuntimeError(
            "Mask/image shape mismatch"
        )

    parent_gray = (
        rgb_to_gray(
            parent_rgb
        )
    )

    attack_gray = (
        rgb_to_gray(
            attack_rgb
        )
    )

    diff = np.abs(
        attack_gray
        - parent_gray
    )

    parent_lap = laplace(
        parent_gray,
        mode="reflect",
    )

    attack_lap = laplace(
        attack_gray,
        mode="reflect",
    )

    lap_diff = np.abs(
        attack_lap
        - parent_lap
    )

    inside = (
        altered_mask
    )

    outside = (
        ~altered_mask
    )

    if (
        not inside.any()
        or not outside.any()
    ):
        raise RuntimeError(
            "Invalid inside/outside masks"
        )

    gray_inside = float(
        diff[
            inside
        ].mean()
    )

    gray_outside = float(
        diff[
            outside
        ].mean()
    )

    lap_inside = float(
        lap_diff[
            inside
        ].mean()
    )

    lap_outside = float(
        lap_diff[
            outside
        ].mean()
    )

    eps = 1e-6

    return {
        "altered_fraction":
            float(
                inside.mean()
            ),

        "gray_mad_inside":
            gray_inside,

        "gray_mad_outside":
            gray_outside,

        "gray_inside_minus_outside":
            (
                gray_inside
                - gray_outside
            ),

        "gray_inside_outside_ratio":
            (
                gray_inside
                /
                (
                    gray_outside
                    + eps
                )
            ),

        "lap_mad_inside":
            lap_inside,

        "lap_mad_outside":
            lap_outside,

        "lap_inside_minus_outside":
            (
                lap_inside
                - lap_outside
            ),

        "lap_inside_outside_ratio":
            (
                lap_inside
                /
                (
                    lap_outside
                    + eps
                )
            ),

        "changed_frac_inside_ge2":
            float(
                (
                    diff[
                        inside
                    ]
                    >= 2.0
                ).mean()
            ),

        "changed_frac_inside_ge5":
            float(
                (
                    diff[
                        inside
                    ]
                    >= 5.0
                ).mean()
            ),

        "changed_frac_outside_ge2":
            float(
                (
                    diff[
                        outside
                    ]
                    >= 2.0
                ).mean()
            ),
    }


# ---------------------------------------------------------------------
# Input population / hashes
# ---------------------------------------------------------------------

def load_population():
    for path in [
        PAIRS,
        NATURAL_INDEX,
        TEST_INDEX,
    ]:
        if not path.is_file():
            raise RuntimeError(
                f"Missing required file: "
                f"{path}"
            )

    pairs = pd.read_csv(
        PAIRS
    )

    dev = pairs[
        pairs[
            "split"
        ]
        == "dev_val"
    ].copy()

    if len(dev) != 153:
        raise RuntimeError(
            f"Expected 153 dev pairs, "
            f"got {len(dev)}"
        )

    if (
        dev[
            "file_stem"
        ].nunique()
        != 51
    ):
        raise RuntimeError(
            "Expected 51 held-out stems"
        )

    if (
        dev[
            "hardware_source"
        ]
        .value_counts()
        .to_dict()
        != {
            "huawei": 51,
            "iphone15pro": 51,
            "scan": 51,
        }
    ):
        raise RuntimeError(
            "Unexpected dev hardware "
            "composition"
        )

    natural = pd.read_csv(
        NATURAL_INDEX
    )

    test = pd.read_csv(
        TEST_INDEX
    )

    parent_hash = (
        natural.set_index(
            "image_path"
        )[
            "source_sha256"
        ]
    )

    test_hash = (
        test.set_index(
            "image_path"
        )[
            "source_sha256"
        ]
    )

    return (
        dev,
        parent_hash,
        test_hash,
    )


# ---------------------------------------------------------------------
# Bootstrap by stem
# ---------------------------------------------------------------------

def bootstrap_mean(
    frame,
    column,
):
    stems = np.array(
        sorted(
            frame[
                "file_stem"
            ].unique()
        )
    )

    groups = {
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
        for stem in stems
    }

    rng = np.random.default_rng(
        SEED
    )

    values = np.empty(
        N_BOOT,
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

        boot = np.concatenate(
            [
                groups[
                    stem
                ]
                for stem
                in sampled
            ]
        )

        values[i] = (
            boot.mean()
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


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def make_summary(
    detail,
):
    rows = []

    for stage in STAGES:
        group = detail[
            detail[
                "stage"
            ]
            == stage
        ].copy()

        if len(group) != 153:
            raise RuntimeError(
                f"Expected 153 rows at "
                f"{stage}, got {len(group)}"
            )

        gray_ci = bootstrap_mean(
            group,
            "gray_retention_vs_native",
        )

        lap_ci = bootstrap_mean(
            group,
            "lap_retention_vs_native",
        )

        rows.append(
            {
                "stage":
                    stage,

                "n_pairs":
                    len(group),

                "n_stems":
                    group[
                        "file_stem"
                    ].nunique(),

                "mean_altered_fraction":
                    group[
                        "altered_fraction"
                    ].mean(),

                "mean_gray_mad_inside":
                    group[
                        "gray_mad_inside"
                    ].mean(),

                "median_gray_mad_inside":
                    group[
                        "gray_mad_inside"
                    ].median(),

                "mean_gray_mad_outside":
                    group[
                        "gray_mad_outside"
                    ].mean(),

                "mean_gray_signal_ratio":
                    group[
                        "gray_inside_outside_ratio"
                    ].mean(),

                "mean_lap_mad_inside":
                    group[
                        "lap_mad_inside"
                    ].mean(),

                "median_lap_mad_inside":
                    group[
                        "lap_mad_inside"
                    ].median(),

                "mean_lap_mad_outside":
                    group[
                        "lap_mad_outside"
                    ].mean(),

                "mean_lap_signal_ratio":
                    group[
                        "lap_inside_outside_ratio"
                    ].mean(),

                "mean_gray_retention_vs_native":
                    group[
                        "gray_retention_vs_native"
                    ].mean(),

                "median_gray_retention_vs_native":
                    group[
                        "gray_retention_vs_native"
                    ].median(),

                "gray_retention_ci_low":
                    gray_ci[0],

                "gray_retention_ci_high":
                    gray_ci[1],

                "mean_lap_retention_vs_native":
                    group[
                        "lap_retention_vs_native"
                    ].mean(),

                "median_lap_retention_vs_native":
                    group[
                        "lap_retention_vs_native"
                    ].median(),

                "lap_retention_ci_low":
                    lap_ci[0],

                "lap_retention_ci_high":
                    lap_ci[1],

                "mean_changed_inside_ge2":
                    group[
                        "changed_frac_inside_ge2"
                    ].mean(),

                "mean_changed_inside_ge5":
                    group[
                        "changed_frac_inside_ge5"
                    ].mean(),

                "mean_changed_outside_ge2":
                    group[
                        "changed_frac_outside_ge2"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_hardware_summary(
    detail,
):
    rows = []

    for stage in STAGES:
        stage_frame = detail[
            detail[
                "stage"
            ]
            == stage
        ]

        for hardware in [
            "huawei",
            "iphone15pro",
            "scan",
        ]:
            group = stage_frame[
                stage_frame[
                    "hardware_source"
                ]
                == hardware
            ]

            rows.append(
                {
                    "stage":
                        stage,

                    "hardware_source":
                        hardware,

                    "n_pairs":
                        len(group),

                    "mean_gray_mad_inside":
                        group[
                            "gray_mad_inside"
                        ].mean(),

                    "mean_lap_mad_inside":
                        group[
                            "lap_mad_inside"
                        ].mean(),

                    "mean_gray_retention_vs_native":
                        group[
                            "gray_retention_vs_native"
                        ].mean(),

                    "mean_lap_retention_vs_native":
                        group[
                            "lap_retention_vs_native"
                        ].mean(),

                    "mean_gray_signal_ratio":
                        group[
                            "gray_inside_outside_ratio"
                        ].mean(),

                    "mean_lap_signal_ratio":
                        group[
                            "lap_inside_outside_ratio"
                        ].mean(),
                }
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Relationship with actual CNN response
# ---------------------------------------------------------------------

def make_correlations(
    detail,
    pairs,
):
    delta_cols = [
        "controlled_delta",
        "augmented_delta",
    ]

    pair_delta = pairs[
        [
            "digital3_image_path",
            *delta_cols,
        ]
    ].copy()

    merged = detail.merge(
        pair_delta,
        on="digital3_image_path",
        how="left",
        validate="many_to_one",
    )

    rows = []

    feature_cols = [
        "gray_mad_inside",
        "lap_mad_inside",
        "gray_inside_outside_ratio",
        "lap_inside_outside_ratio",
    ]

    for stage in STAGES:
        group = merged[
            merged[
                "stage"
            ]
            == stage
        ]

        for model_delta in (
            delta_cols
        ):
            for feature in (
                feature_cols
            ):
                result = spearmanr(
                    group[
                        feature
                    ],
                    group[
                        model_delta
                    ],
                    nan_policy="omit",
                )

                rows.append(
                    {
                        "stage":
                            stage,

                        "model_delta":
                            model_delta,

                        "feature":
                            feature,

                        "spearman_r":
                            float(
                                result.statistic
                            ),

                        "p_value":
                            float(
                                result.pvalue
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    (
        pairs,
        parent_hash_lookup,
        d3_hash_lookup,
    ) = load_population()

    attack_paths = set(
        pairs[
            "digital3_image_path"
        ]
    )

    boxes, fields = (
        load_altered_boxes(
            attack_paths
        )
    )

    print(
        "Held-out Digital-3 "
        "signal-retention audit:"
        f"\n  pairs:       {len(pairs)}"
        f"\n  stems:       "
        f"{pairs['file_stem'].nunique()}"
        "\n  stages:"
        "\n    native_raw"
        "\n    native_policy_c"
        "\n    r512_raw"
        "\n    r512_policy_c"
    )

    records = []
    failures = []

    clipped_total = 0

    pairs = pairs.sort_values(
        [
            "file_stem",
            "hardware_source",
        ]
    )

    for i, pair in enumerate(
        pairs.itertuples(
            index=False
        ),
        start=1,
    ):
        parent_path = (
            pair.parent_image_path
        )

        attack_path = (
            pair.digital3_image_path
        )

        parent_bytes = (
            DATA
            / parent_path
        ).read_bytes()

        attack_bytes = (
            DATA
            / attack_path
        ).read_bytes()

        if (
            sha256_bytes(
                parent_bytes
            )
            != parent_hash_lookup.loc[
                parent_path
            ]
        ):
            raise RuntimeError(
                "Parent source SHA mismatch: "
                f"{parent_path}"
            )

        if (
            sha256_bytes(
                attack_bytes
            )
            != d3_hash_lookup.loc[
                attack_path
            ]
        ):
            raise RuntimeError(
                "Digital-3 source SHA mismatch: "
                f"{attack_path}"
            )

        parent_raw = decode_rgb(
            parent_bytes
        )

        attack_raw = decode_rgb(
            attack_bytes
        )

        if (
            parent_raw.shape
            != attack_raw.shape
        ):
            failures.append(
                {
                    "file_stem":
                        pair.file_stem,

                    "hardware_source":
                        pair.hardware_source,

                    "parent_image_path":
                        parent_path,

                    "digital3_image_path":
                        attack_path,

                    "parent_height":
                        parent_raw.shape[0],

                    "parent_width":
                        parent_raw.shape[1],

                    "attack_height":
                        attack_raw.shape[0],

                    "attack_width":
                        attack_raw.shape[1],
                }
            )

            continue

        h, w = (
            parent_raw.shape[:2]
        )

        mask_native, clipped = (
            build_mask(
                h,
                w,
                boxes[
                    attack_path
                ],
            )
        )

        clipped_total += clipped

        assigned_q = final_q(
            pair.file_stem,
            pair.hardware_source,
        )

        if int(
            pair.parent_assigned_q
        ) != assigned_q:
            raise RuntimeError(
                "Parent assigned-Q "
                "does not match frozen policy"
            )

        if int(
            pair.digital3_assigned_q
        ) != assigned_q:
            raise RuntimeError(
                "Digital-3 assigned-Q "
                "does not match frozen policy"
            )

        parent_policy, _ = (
            policy_c_rgb(
                parent_raw,
                pair.file_stem,
                pair.hardware_source,
            )
        )

        attack_policy, _ = (
            policy_c_rgb(
                attack_raw,
                pair.file_stem,
                pair.hardware_source,
            )
        )

        parent_r512_raw = (
            resize_r512(
                parent_raw
            )
        )

        attack_r512_raw = (
            resize_r512(
                attack_raw
            )
        )

        parent_r512_policy = (
            resize_r512(
                parent_policy
            )
        )

        attack_r512_policy = (
            resize_r512(
                attack_policy
            )
        )

        mask_r512 = (
            resize_mask_r512(
                mask_native
            )
        )

        stage_inputs = {
            "native_raw":
                (
                    parent_raw,
                    attack_raw,
                    mask_native,
                ),

            "native_policy_c":
                (
                    parent_policy,
                    attack_policy,
                    mask_native,
                ),

            "r512_raw":
                (
                    parent_r512_raw,
                    attack_r512_raw,
                    mask_r512,
                ),

            "r512_policy_c":
                (
                    parent_r512_policy,
                    attack_r512_policy,
                    mask_r512,
                ),
        }

        for stage, (
            parent_stage,
            attack_stage,
            mask_stage,
        ) in stage_inputs.items():
            metrics = (
                compute_metrics(
                    parent_stage,
                    attack_stage,
                    mask_stage,
                )
            )

            records.append(
                {
                    "file_stem":
                        pair.file_stem,

                    "hardware_source":
                        pair.hardware_source,

                    "parent_image_path":
                        parent_path,

                    "digital3_image_path":
                        attack_path,

                    "assigned_q":
                        assigned_q,

                    "fields":
                        "|".join(
                            sorted(
                                fields[
                                    attack_path
                                ]
                            )
                        ),

                    "stage":
                        stage,

                    "native_height":
                        h,

                    "native_width":
                        w,

                    "clipped_box_count":
                        clipped,

                    **metrics,
                }
            )

        if (
            i % 25 == 0
            or i == len(pairs)
        ):
            print(
                f"processed "
                f"{i}/{len(pairs)}"
            )

    failures = pd.DataFrame(
        failures
    )

    if len(failures):
        failures.to_csv(
            OUT_GEOMETRY_FAILURES,
            index=False,
        )

        raise RuntimeError(
            f"{len(failures)} held-out "
            "Digital-3 pairs have "
            "different image geometry. "
            f"See {OUT_GEOMETRY_FAILURES}"
        )

    detail = pd.DataFrame(
        records
    )

    if len(detail) != (
        153
        * len(STAGES)
    ):
        raise RuntimeError(
            "Unexpected detailed row count: "
            f"{len(detail)}"
        )

    # --------------------------------------------------
    # Retention relative to each pair's native-raw signal
    # --------------------------------------------------

    baseline = (
        detail[
            detail[
                "stage"
            ]
            == "native_raw"
        ][
            [
                "digital3_image_path",
                "gray_mad_inside",
                "lap_mad_inside",
            ]
        ]
        .rename(
            columns={
                "gray_mad_inside":
                    "native_gray_mad_inside",

                "lap_mad_inside":
                    "native_lap_mad_inside",
            }
        )
    )

    detail = detail.merge(
        baseline,
        on="digital3_image_path",
        how="left",
        validate="many_to_one",
    )

    eps = 1e-8

    detail[
        "gray_retention_vs_native"
    ] = (
        detail[
            "gray_mad_inside"
        ]
        /
        (
            detail[
                "native_gray_mad_inside"
            ]
            + eps
        )
    )

    detail[
        "lap_retention_vs_native"
    ] = (
        detail[
            "lap_mad_inside"
        ]
        /
        (
            detail[
                "native_lap_mad_inside"
            ]
            + eps
        )
    )

    detail.to_csv(
        OUT_DETAIL,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = make_summary(
        detail
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    hardware = (
        make_hardware_summary(
            detail
        )
    )

    hardware.to_csv(
        OUT_HARDWARE,
        index=False,
    )

    correlations = (
        make_correlations(
            detail,
            pairs,
        )
    )

    correlations.to_csv(
        OUT_CORRELATION,
        index=False,
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    display = [
        "stage",
        "n_pairs",
        "n_stems",
        "mean_gray_mad_inside",
        "mean_gray_mad_outside",
        "mean_gray_signal_ratio",
        "mean_lap_mad_inside",
        "mean_lap_mad_outside",
        "mean_lap_signal_ratio",
        "mean_gray_retention_vs_native",
        "median_gray_retention_vs_native",
        "gray_retention_ci_low",
        "gray_retention_ci_high",
        "mean_lap_retention_vs_native",
        "median_lap_retention_vs_native",
        "lap_retention_ci_low",
        "lap_retention_ci_high",
        "mean_changed_inside_ge2",
        "mean_changed_inside_ge5",
        "mean_changed_outside_ge2",
    ]

    print(
        "\nHELD-OUT DEV — "
        "DIGITAL-3 SIGNAL RETENTION:"
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

    print(
        "\nHARDWARE BREAKDOWN:"
    )

    print(
        hardware.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nCORRELATION WITH CNN "
        "PAIRED SCORE RESPONSE:"
    )

    print(
        correlations.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\nClipped altered boxes: "
        f"{clipped_total}"
        f"\nDetail:      {OUT_DETAIL}"
        f"\nSummary:     {OUT_SUMMARY}"
        f"\nHardware:    {OUT_HARDWARE}"
        f"\nCorrelation: {OUT_CORRELATION}"
    )

    print(
        "\nInterpretation guide:"
        "\n"
        "\n1. native_policy_c << native_raw"
        "\n   => compression control removes substantial local evidence."
        "\n"
        "\n2. r512_raw << native_raw"
        "\n   => whole-document downscaling removes substantial evidence."
        "\n"
        "\n3. native_policy_c remains high but r512_policy_c collapses"
        "\n   => resizing is the dominant information bottleneck."
        "\n"
        "\n4. both retention measures remain high at r512_policy_c"
        "\n   => the evidence still exists in pixel space and the main "
        "problem is more likely representation/supervision rather "
        "than preprocessing information loss."
    )


if __name__ == "__main__":
    main()