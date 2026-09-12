#!/usr/bin/env python3
"""
Prepare native-resolution 512x512 text-region patches.

Training / validation:
    digital_1 + digital_2 altered text patches -> label 1
    exact same-coordinate matched bona-fide patches -> label 0

Diagnostic:
    Digital-3 project-dev altered text patches -> label 1
    exact same-coordinate source bona-fide patches -> label 0

All source images are the already-created full-resolution Policy-C images.

Therefore:

    raw full image
      -> Q75
      -> matched deterministic Q50-90
      -> decode at native resolution
      -> 512x512 crop

There is NO whole-document resize before patch extraction and NO new JPEG
encoding after cropping. Patches are saved losslessly as PNG.

Face regions are forbidden from intersecting the selected patch. Crop position
is shifted deterministically around the altered text-region centre when needed.

Digital-3 is NEVER placed in project_train or dev_val.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

INVENTORY = (
    ROOT
    / "output"
    / "fantasyid_inventory_2026-09-05_023126.xlsx"
)

POLICY_INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

D3_PAIRS = (
    ROOT
    / "output"
    / "digital3_same_source_pairs.csv"
)

CACHE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "text_patch512"
)

OUT_INDEX = (
    ROOT
    / "output"
    / "text_patch512_index.csv"
)

OUT_EXCLUSIONS = (
    ROOT
    / "output"
    / "text_patch512_exclusions.csv"
)

INVENTORY_SHA256 = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
)

PATCH = 512


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def token(text):
    return hashlib.sha256(
        text.encode()
    ).hexdigest()[:20]


def rect_area(rect):
    x0, y0, x1, y1 = rect

    return max(
        0,
        x1 - x0,
    ) * max(
        0,
        y1 - y0,
    )


def intersection_area(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    x0 = max(
        ax0,
        bx0,
    )

    y0 = max(
        ay0,
        by0,
    )

    x1 = min(
        ax1,
        bx1,
    )

    y1 = min(
        ay1,
        by1,
    )

    return rect_area(
        (
            x0,
            y0,
            x1,
            y1,
        )
    )


def clip(value, low, high):
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def choose_face_free_crop(
    image_width,
    image_height,
    text_box,
    face_boxes,
):
    """
    Choose the 512x512 crop closest to a field-centred crop while requiring
    zero intersection with every annotated face region.

    The text-box centre must remain inside the patch.

    We do not resize the patch or source box.
    """

    if (
        image_width < PATCH
        or image_height < PATCH
    ):
        return None

    tx0, ty0, tx1, ty1 = text_box

    cx = (
        tx0 + tx1
    ) / 2.0

    cy = (
        ty0 + ty1
    ) / 2.0

    ideal_x = int(
        round(
            cx
            - PATCH / 2
        )
    )

    ideal_y = int(
        round(
            cy
            - PATCH / 2
        )
    )

    max_x = (
        image_width
        - PATCH
    )

    max_y = (
        image_height
        - PATCH
    )

    offsets = [
        0,
        -32,
        32,
        -64,
        64,
        -96,
        96,
        -128,
        128,
        -160,
        160,
        -192,
        192,
        -224,
        224,
        -256,
        256,
    ]

    x_candidates = {
        clip(
            ideal_x + d,
            0,
            max_x,
        )
        for d in offsets
    }

    y_candidates = {
        clip(
            ideal_y + d,
            0,
            max_y,
        )
        for d in offsets
    }

    # Useful alternatives around the annotation edges.
    x_candidates.update(
        {
            clip(
                tx0,
                0,
                max_x,
            ),
            clip(
                tx1 - PATCH,
                0,
                max_x,
            ),
        }
    )

    y_candidates.update(
        {
            clip(
                ty0,
                0,
                max_y,
            ),
            clip(
                ty1 - PATCH,
                0,
                max_y,
            ),
        }
    )

    candidates = []

    for x0 in x_candidates:
        for y0 in y_candidates:
            crop = (
                x0,
                y0,
                x0 + PATCH,
                y0 + PATCH,
            )

            # Text-region centre must stay visible.
            if not (
                x0 <= cx < x0 + PATCH
                and
                y0 <= cy < y0 + PATCH
            ):
                continue

            face_overlap = sum(
                intersection_area(
                    crop,
                    face,
                )
                for face
                in face_boxes
            )

            distance = (
                (
                    x0
                    - ideal_x
                ) ** 2
                +
                (
                    y0
                    - ideal_y
                ) ** 2
            )

            candidates.append(
                (
                    face_overlap,
                    distance,
                    crop,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            (
                item[0],
                item[1],
            )
    )

    face_overlap, _, crop = (
        candidates[0]
    )

    if face_overlap != 0:
        return None

    return crop


def load_regions():
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
            "Regions schema mismatch: "
            f"{sorted(missing)}"
        )

    return regions


def row_box(row):
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
            "Invalid annotation size"
        )

    return (
        x,
        y,
        x + w,
        y + h,
    )


def annotation_lookup(
    regions,
    paths,
):
    subset = regions[
        regions[
            "image_path"
        ].isin(
            paths
        )
    ].copy()

    output = {}

    for image_path, group in (
        subset.groupby(
            "image_path"
        )
    ):
        text = []

        faces = []

        for region_index, (
            _,
            row,
        ) in enumerate(
            group.iterrows()
        ):
            field = (
                str(
                    row[
                        "field_name"
                    ]
                )
                .strip()
                .lower()
            )

            provenance = (
                str(
                    row[
                        "region_provenance_raw"
                    ]
                )
                .strip()
                .lower()
            )

            box = row_box(
                row
            )

            # Exclude every face rectangle, independent of provenance.
            if field == "face":
                faces.append(
                    box
                )

            elif provenance == "altered":
                text.append(
                    {
                        "region_index":
                            region_index,

                        "field_name":
                            field,

                        "box":
                            box,
                    }
                )

        output[
            image_path
        ] = {
            "text":
                text,

            "faces":
                faces,
        }

    return output


def save_patch(
    image,
    crop,
    destination,
):
    x0, y0, x1, y1 = (
        crop
    )

    patch = image.crop(
        (
            x0,
            y0,
            x1,
            y1,
        )
    )

    if patch.size != (
        PATCH,
        PATCH,
    ):
        raise RuntimeError(
            "Patch geometry changed"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Lossless. Do not introduce another JPEG stage.
    patch.save(
        destination,
        "PNG",
        compress_level=1,
    )


def make_pair_rows(
    split,
    variant,
    attack_image_path,
    parent_image_path,
    attack_cache_path,
    parent_cache_path,
    file_stem,
    hardware_source,
    assigned_q,
    annotations,
    exclusions,
):
    attack_full = (
        ROOT
        / attack_cache_path
    )

    parent_full = (
        ROOT
        / parent_cache_path
    )

    with Image.open(
        attack_full
    ) as im:
        attack_image = (
            im.convert(
                "RGB"
            )
        )

        attack_size = (
            attack_image.size
        )

    with Image.open(
        parent_full
    ) as im:
        parent_image = (
            im.convert(
                "RGB"
            )
        )

        parent_size = (
            parent_image.size
        )

    if (
        attack_size
        != parent_size
    ):
        exclusions.append(
            {
                "split":
                    split,

                "variant":
                    variant,

                "file_stem":
                    file_stem,

                "hardware_source":
                    hardware_source,

                "field_name":
                    "",

                "reason":
                    "image_dimension_mismatch",

                "attack_image_path":
                    attack_image_path,

                "parent_image_path":
                    parent_image_path,
            }
        )

        return []

    width, height = (
        attack_size
    )

    rows = []

    for item in (
        annotations["text"]
    ):
        field = (
            item["field_name"]
        )

        box = item["box"]

        crop = (
            choose_face_free_crop(
                width,
                height,
                box,
                annotations[
                    "faces"
                ],
            )
        )

        if crop is None:
            exclusions.append(
                {
                    "split":
                        split,

                    "variant":
                        variant,

                    "file_stem":
                        file_stem,

                    "hardware_source":
                        hardware_source,

                    "field_name":
                        field,

                    "reason":
                        "no_face_free_512_crop",

                    "attack_image_path":
                        attack_image_path,

                    "parent_image_path":
                        parent_image_path,
                }
            )

            continue

        region_id = (
            f"{attack_image_path}|"
            f"{item['region_index']}|"
            f"{field}|"
            f"{crop}"
        )

        pair_id = token(
            region_id
        )

        attack_patch = (
            CACHE_ROOT
            / split
            / variant
            / "attack"
            / f"{pair_id}.png"
        )

        parent_patch = (
            CACHE_ROOT
            / split
            / variant
            / "bonafide"
            / f"{pair_id}.png"
        )

        save_patch(
            attack_image,
            crop,
            attack_patch,
        )

        save_patch(
            parent_image,
            crop,
            parent_patch,
        )

        x0, y0, x1, y1 = (
            crop
        )

        common = {
            "split":
                split,

            "variant":
                variant,

            "pair_id":
                pair_id,

            "file_stem":
                file_stem,

            "hardware_source":
                hardware_source,

            "field_name":
                field,

            "assigned_q":
                int(
                    assigned_q
                ),

            "attack_image_path":
                attack_image_path,

            "parent_image_path":
                parent_image_path,

            "crop_x0":
                x0,

            "crop_y0":
                y0,

            "crop_x1":
                x1,

            "crop_y1":
                y1,

            "annotation_x0":
                box[0],

            "annotation_y0":
                box[1],

            "annotation_x1":
                box[2],

            "annotation_y1":
                box[3],

            "face_overlap_pixels":
                0,
        }

        rows.append(
            {
                **common,

                "role":
                    "attack",

                "label":
                    1,

                "patch_path":
                    str(
                        attack_patch.relative_to(
                            ROOT
                        )
                    ),
            }
        )

        rows.append(
            {
                **common,

                "role":
                    "bonafide",

                "label":
                    0,

                "patch_path":
                    str(
                        parent_patch.relative_to(
                            ROOT
                        )
                    ),
            }
        )

    return rows


def prepare_d1_d2(
    policy_index,
    regions,
):
    rows = []

    exclusions = []

    for split in [
        "project_train",
        "dev_val",
    ]:
        frame = policy_index[
            policy_index[
                "split"
            ]
            == split
        ].copy()

        attacks = frame[
            frame[
                "traffic_type"
            ]
            == "attack"
        ].copy()

        bonafides = frame[
            frame[
                "traffic_type"
            ]
            == "bonafide"
        ].copy()

        bona_lookup = (
            bonafides.set_index(
                [
                    "file_stem",
                    "hardware_source",
                ]
            )
        )

        attack_paths = set(
            attacks[
                "image_path"
            ]
        )

        annotation = (
            annotation_lookup(
                regions,
                attack_paths,
            )
        )

        for attack in (
            attacks.itertuples(
                index=False
            )
        ):
            variant = str(
                attack.variant
            )

            if variant not in {
                "digital_1",
                "digital_2",
            }:
                raise RuntimeError(
                    "Unexpected train/dev "
                    f"attack variant: {variant}"
                )

            key = (
                attack.file_stem,
                attack.hardware_source,
            )

            try:
                parent = (
                    bona_lookup.loc[
                        key
                    ]
                )
            except KeyError as exc:
                raise RuntimeError(
                    "Missing matched "
                    f"bona-fide: {key}"
                ) from exc

            if (
                attack.image_path
                not in annotation
            ):
                raise RuntimeError(
                    "No Regions rows for "
                    f"{attack.image_path}"
                )

            rows.extend(
                make_pair_rows(
                    split=split,
                    variant=variant,
                    attack_image_path=(
                        attack.image_path
                    ),
                    parent_image_path=(
                        parent.image_path
                    ),
                    attack_cache_path=(
                        attack.cache_path
                    ),
                    parent_cache_path=(
                        parent.cache_path
                    ),
                    file_stem=(
                        attack.file_stem
                    ),
                    hardware_source=(
                        attack.hardware_source
                    ),
                    assigned_q=(
                        attack.assigned_q
                    ),
                    annotations=annotation[
                        attack.image_path
                    ],
                    exclusions=exclusions,
                )
            )

    return (
        rows,
        exclusions,
    )


def prepare_digital3_dev(
    natural_index,
    test_index,
    regions,
):
    """
    Only project-dev parents.

    This is diagnostic test data and is never included in model training
    or model selection.
    """

    pairs = pd.read_csv(
        D3_PAIRS
    )

    pairs = pairs[
        pairs[
            "split"
        ]
        == "dev_val"
    ].copy()

    if (
        len(pairs) != 153
        or
        pairs[
            "file_stem"
        ].nunique()
        != 51
    ):
        raise RuntimeError(
            "Unexpected Digital-3 "
            "held-out population"
        )

    natural_lookup = (
        natural_index.set_index(
            "image_path"
        )
    )

    test_lookup = (
        test_index.set_index(
            "image_path"
        )
    )

    attack_paths = set(
        pairs[
            "digital3_image_path"
        ]
    )

    annotation = (
        annotation_lookup(
            regions,
            attack_paths,
        )
    )

    rows = []

    exclusions = []

    for pair in (
        pairs.itertuples(
            index=False
        )
    ):
        attack_path = (
            pair.digital3_image_path
        )

        parent_path = (
            pair.parent_image_path
        )

        attack = (
            test_lookup.loc[
                attack_path
            ]
        )

        parent = (
            natural_lookup.loc[
                parent_path
            ]
        )

        if int(
            attack.assigned_q
        ) != int(
            parent.assigned_q
        ):
            raise RuntimeError(
                "Digital-3 final-Q "
                "pair mismatch"
            )

        if (
            attack_path
            not in annotation
        ):
            raise RuntimeError(
                "Digital-3 Regions "
                f"missing: {attack_path}"
            )

        rows.extend(
            make_pair_rows(
                split="digital3_dev",
                variant="digital_3",
                attack_image_path=(
                    attack_path
                ),
                parent_image_path=(
                    parent_path
                ),
                attack_cache_path=(
                    attack.cache_path
                ),
                parent_cache_path=(
                    parent.cache_path
                ),
                file_stem=(
                    pair.file_stem
                ),
                hardware_source=(
                    pair.hardware_source
                ),
                assigned_q=(
                    attack.assigned_q
                ),
                annotations=annotation[
                    attack_path
                ],
                exclusions=exclusions,
            )
        )

    return (
        rows,
        exclusions,
    )


def validate_index(index):
    if index[
        "pair_id"
    ].isna().any():
        raise RuntimeError(
            "Missing pair IDs"
        )

    pair_counts = (
        index.groupby(
            "pair_id"
        )
        .agg(
            n=(
                "label",
                "size",
            ),
            n_labels=(
                "label",
                "nunique",
            ),
        )
    )

    if not (
        (
            pair_counts["n"]
            == 2
        )
        &
        (
            pair_counts[
                "n_labels"
            ]
            == 2
        )
    ).all():
        raise RuntimeError(
            "Every patch pair must "
            "contain exactly one attack "
            "and one bona-fide"
        )

    for split in [
        "project_train",
        "dev_val",
        "digital3_dev",
    ]:
        frame = index[
            index[
                "split"
            ]
            == split
        ]

        if len(frame) == 0:
            raise RuntimeError(
                f"No rows for {split}"
            )

        counts = (
            frame[
                "label"
            ]
            .value_counts()
            .to_dict()
        )

        if (
            counts.get(
                0,
                0,
            )
            !=
            counts.get(
                1,
                0,
            )
        ):
            raise RuntimeError(
                f"{split} patch "
                "classes not balanced"
            )

    train_stems = set(
        index.loc[
            index[
                "split"
            ]
            == "project_train",
            "file_stem",
        ]
    )

    dev_stems = set(
        index.loc[
            index[
                "split"
            ]
            == "dev_val",
            "file_stem",
        ]
    )

    d3_stems = set(
        index.loc[
            index[
                "split"
            ]
            == "digital3_dev",
            "file_stem",
        ]
    )

    if (
        train_stems
        & dev_stems
    ):
        raise RuntimeError(
            "Train/dev card leakage"
        )

    # Digital-3 diagnostic should be exactly the project-dev card population.
    if d3_stems != dev_stems:
        raise RuntimeError(
            "Digital-3 diagnostic stems "
            "do not match project dev"
        )


def main():
    for path in [
        POLICY_INDEX,
        TEST_INDEX,
        D3_PAIRS,
        INVENTORY,
    ]:
        if not path.is_file():
            raise RuntimeError(
                f"Missing required file: "
                f"{path}"
            )

    policy_index = (
        pd.read_csv(
            POLICY_INDEX
        )
    )

    test_index = (
        pd.read_csv(
            TEST_INDEX
        )
    )

    regions = load_regions()

    train_dev_rows, exclusions_1 = (
        prepare_d1_d2(
            policy_index,
            regions,
        )
    )

    d3_rows, exclusions_2 = (
        prepare_digital3_dev(
            policy_index,
            test_index,
            regions,
        )
    )

    index = pd.DataFrame(
        train_dev_rows
        + d3_rows
    )

    exclusions = pd.DataFrame(
        exclusions_1
        + exclusions_2
    )

    validate_index(
        index
    )

    OUT_INDEX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.to_csv(
        OUT_INDEX,
        index=False,
    )

    exclusions.to_csv(
        OUT_EXCLUSIONS,
        index=False,
    )

    print(
        "Native 512x512 text "
        "patch cache complete."
    )

    print(
        "\nPatch rows:"
    )

    print(
        index.groupby(
            [
                "split",
                "variant",
                "role",
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nPatch pairs:"
    )

    print(
        index[
            [
                "split",
                "variant",
                "pair_id",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "split",
                "variant",
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nField counts "
        "(attack side only):"
    )

    print(
        index[
            index[
                "role"
            ]
            == "attack"
        ]
        .groupby(
            [
                "split",
                "variant",
                "field_name",
            ]
        )
        .size()
        .to_string()
    )

    print(
        f"\nFace-overlap exclusions: "
        f"{len(exclusions)}"
    )

    if len(exclusions):
        print(
            exclusions[
                [
                    "split",
                    "variant",
                    "field_name",
                    "reason",
                ]
            ]
            .value_counts()
            .to_string()
        )

    print(
        f"\nIndex:      {OUT_INDEX}"
        f"\nExclusions: {OUT_EXCLUSIONS}"
        f"\nCache:      {CACHE_ROOT}"
    )

    print(
        "\nDigital-3 patches were "
        "prepared for diagnostic "
        "evaluation only."
    )


if __name__ == "__main__":
    main()