#!/usr/bin/env python3
"""
Prepare annotation-localized native 512x512 text patches for the official
FantasyID test set.

Annotation policy
-----------------
Attack images:
    use ONLY their own region_provenance == "original" annotations.

Official bona-fide images:
    the inventory currently provides no directly usable Regions rows.
    Recover text/face coordinates only from an official attack with the exact
    same:

        file_stem + hardware_source

    and require identical decoded image dimensions.

We NEVER:
    - use altered-region annotations,
    - use knowledge of which field was manipulated,
    - resize/warp/register annotation coordinates,
    - transfer coordinates across hardware,
    - infer boxes from another card.

Candidate text field names are frozen from the project_train native-patch
experiment.

Input images are the frozen native-resolution Policy-C official-test cache.
There is no whole-document resize and no additional JPEG encoding after
Policy C. Patches are saved losslessly as PNG.

This remains an annotation-assisted diagnostic, not an end-to-end deployable
text-region detector.
"""

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

INVENTORY = (
    ROOT
    / "output"
    / "fantasyid_inventory_2026-09-05_023126.xlsx"
)

INVENTORY_SHA256 = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

TRAIN_PATCH_INDEX = (
    ROOT
    / "output"
    / "text_patch512_index.csv"
)

CACHE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "text_patch512_official"
)

OUT_INDEX = (
    ROOT
    / "output"
    / "text_patch512_official_index.csv"
)

OUT_EXCLUSIONS = (
    ROOT
    / "output"
    / "text_patch512_official_exclusions.csv"
)

OUT_MISSING = (
    ROOT
    / "output"
    / "text_patch512_official_missing_images.csv"
)

OUT_TRANSFER_AUDIT = (
    ROOT
    / "output"
    / "text_patch512_official_annotation_transfer.csv"
)

PATCH = 512


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


def token(text):
    return hashlib.sha256(
        text.encode()
    ).hexdigest()[:24]


def norm_field(value):
    return (
        str(value)
        .strip()
        .lower()
    )


def norm_variant(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def rect_area(rect):
    x0, y0, x1, y1 = rect

    return (
        max(0, x1 - x0)
        *
        max(0, y1 - y0)
    )


def intersection_area(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    return rect_area(
        (
            max(ax0, bx0),
            max(ay0, by0),
            min(ax1, bx1),
            min(ay1, by1),
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


def row_box(row):
    x = int(
        round(
            float(row.x)
        )
    )

    y = int(
        round(
            float(row.y)
        )
    )

    w = int(
        round(
            float(row.width)
        )
    )

    h = int(
        round(
            float(row.height)
        )
    )

    if (
        w <= 0
        or h <= 0
    ):
        raise RuntimeError(
            "Invalid annotation rectangle"
        )

    return (
        x,
        y,
        x + w,
        y + h,
    )


def image_size(cache_path):
    with Image.open(
        ROOT / cache_path
    ) as im:
        return im.size


# ---------------------------------------------------------------------
# 512 crop policy
# ---------------------------------------------------------------------

def choose_face_free_crop(
    image_width,
    image_height,
    text_box,
    face_boxes,
):
    """
    Find a native 512x512 crop containing the centre of the text field
    while intersecting zero annotated face pixels.

    This is exactly a crop selection problem:
        no resize
        no warp
        no annotation snapping
    """

    if (
        image_width < PATCH
        or image_height < PATCH
    ):
        return None

    tx0, ty0, tx1, ty1 = (
        text_box
    )

    cx = (
        tx0 + tx1
    ) / 2.0

    cy = (
        ty0 + ty1
    ) / 2.0

    ideal_x = int(
        round(
            cx - PATCH / 2
        )
    )

    ideal_y = int(
        round(
            cy - PATCH / 2
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
            ideal_x + offset,
            0,
            max_x,
        )
        for offset
        in offsets
    }

    y_candidates = {
        clip(
            ideal_y + offset,
            0,
            max_y,
        )
        for offset
        in offsets
    }

    # Also allow crops aligned to either edge of the field.
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

            # Text-field centre must remain visible.
            if not (
                x0 <= cx < x0 + PATCH
                and
                y0 <= cy < y0 + PATCH
            ):
                continue

            face_overlap = sum(
                intersection_area(
                    crop,
                    face_box,
                )
                for face_box
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
        key=lambda item: (
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


# ---------------------------------------------------------------------
# Frozen field vocabulary
# ---------------------------------------------------------------------

def load_frozen_fields():
    frame = pd.read_csv(
        TRAIN_PATCH_INDEX
    )

    selected = frame[
        (
            frame["split"]
            == "project_train"
        )
        &
        (
            frame["role"]
            == "attack"
        )
    ]

    fields = sorted(
        {
            norm_field(value)
            for value
            in selected[
                "field_name"
            ].dropna()
        }
    )

    if not fields:
        raise RuntimeError(
            "No frozen training text fields"
        )

    return fields


# ---------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------

def load_regions():
    actual_sha = sha256_file(
        INVENTORY
    )

    if (
        actual_sha
        != INVENTORY_SHA256
    ):
        raise RuntimeError(
            "Inventory SHA mismatch"
            f"\nexpected: {INVENTORY_SHA256}"
            f"\nactual:   {actual_sha}"
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

    regions = regions.copy()

    regions[
        "field_norm"
    ] = (
        regions[
            "field_name"
        ]
        .map(
            norm_field
        )
    )

    regions[
        "provenance_norm"
    ] = (
        regions[
            "region_provenance_raw"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Hard safety gate: all annotation bundles created below come only
    # from rows explicitly marked original.
    return regions[
        regions[
            "provenance_norm"
        ]
        == "original"
    ].copy()


def make_annotation_bundle(
    group,
    frozen_fields,
):
    """
    Convert ORIGINAL Regions rows for one image into a canonical bundle.

    Returns:
        text: list of {field_name, box}
        faces: list of boxes
        signature: canonical geometry signature used to detect ambiguous
                   same-key attack donors
    """

    face_rows = group[
        group[
            "field_norm"
        ]
        == "face"
    ]

    face_boxes = sorted(
        {
            row_box(row)
            for row
            in face_rows.itertuples(
                index=False
            )
        }
    )

    text_rows = group[
        group[
            "field_norm"
        ]
        .isin(
            frozen_fields
        )
    ].copy()

    text_items = []

    seen = set()

    for row in text_rows.itertuples(
        index=False
    ):
        field_name = (
            row.field_norm
        )

        box = row_box(
            row
        )

        key = (
            field_name,
            *box,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        text_items.append(
            {
                "field_name":
                    field_name,

                "box":
                    box,
            }
        )

    text_items.sort(
        key=lambda item: (
            item[
                "field_name"
            ],
            item[
                "box"
            ],
        )
    )

    signature = (
        tuple(
            (
                item[
                    "field_name"
                ],
                *item[
                    "box"
                ],
            )
            for item
            in text_items
        ),
        tuple(
            face_boxes
        ),
    )

    return {
        "text":
            text_items,

        "faces":
            face_boxes,

        "signature":
            signature,
    }


def build_direct_annotation_lookup(
    original_regions,
    test_paths,
    frozen_fields,
):
    subset = original_regions[
        original_regions[
            "image_path"
        ]
        .isin(
            test_paths
        )
    ]

    lookup = {}

    for image_path, group in (
        subset.groupby(
            "image_path"
        )
    ):
        bundle = (
            make_annotation_bundle(
                group,
                frozen_fields,
            )
        )

        # An image with only face rows is not useful as a text donor.
        if not bundle[
            "text"
        ]:
            continue

        lookup[
            image_path
        ] = bundle

    return lookup


# ---------------------------------------------------------------------
# Exact attack donors for official bona-fides
# ---------------------------------------------------------------------

def build_attack_donor_lookup(
    test,
    direct_annotations,
):
    """
    Index annotated official attacks by:

        file_stem + hardware_source

    We may theoretically encounter >1 attack donor for a key. Such donors
    are accepted only when their ORIGINAL text+face geometry signatures are
    identical. Otherwise the key is marked ambiguous and cannot be used.
    """

    donor_lookup = {}

    attack_rows = test[
        test[
            "traffic_type"
        ]
        == "attack"
    ]

    for row in attack_rows.itertuples(
        index=False
    ):
        if (
            row.image_path
            not in direct_annotations
        ):
            continue

        key = (
            str(
                row.file_stem
            ),
            str(
                row.hardware_source
            ),
        )

        candidate = {
            "image_path":
                row.image_path,

            "cache_path":
                row.cache_path,

            "variant":
                norm_variant(
                    row.variant
                ),

            "bundle":
                direct_annotations[
                    row.image_path
                ],
        }

        donor_lookup.setdefault(
            key,
            [],
        ).append(
            candidate
        )

    return donor_lookup


def resolve_attack_donor(
    candidates,
):
    if not candidates:
        return (
            None,
            "no_exact_attack_annotation_source",
        )

    signatures = {
        candidate[
            "bundle"
        ][
            "signature"
        ]
        for candidate
        in candidates
    }

    if len(signatures) != 1:
        return (
            None,
            "ambiguous_attack_annotation_geometry",
        )

    # Deterministic selection if more than one perfectly agreeing donor exists.
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate[
                "variant"
            ],
            candidate[
                "image_path"
            ],
        )
    )

    return (
        candidates[0],
        None,
    )


# ---------------------------------------------------------------------
# Patch output
# ---------------------------------------------------------------------

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
            "Patch geometry failure"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Lossless: do not add another JPEG history stage.
    patch.save(
        destination,
        "PNG",
        compress_level=1,
    )


def population_name(
    traffic_type,
    variant,
):
    if (
        traffic_type
        == "bonafide"
    ):
        return "bonafide"

    return variant


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    for path in [
        INVENTORY,
        TEST_INDEX,
        TRAIN_PATCH_INDEX,
    ]:
        if not path.is_file():
            raise RuntimeError(
                f"Missing required file: "
                f"{path}"
            )

    test = pd.read_csv(
        TEST_INDEX
    )

    if len(test) != 1385:
        raise RuntimeError(
            f"Expected 1385 official images, "
            f"got {len(test)}"
        )

    if (
        test[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate official image paths"
        )

    expected_counts = {
        "attack": 1085,
        "bonafide": 300,
    }

    actual_counts = (
        test[
            "traffic_type"
        ]
        .value_counts()
        .to_dict()
    )

    if actual_counts != expected_counts:
        raise RuntimeError(
            "Unexpected official class counts: "
            f"{actual_counts}"
        )

    frozen_fields = (
        load_frozen_fields()
    )

    print(
        "Frozen eligible text fields:"
    )

    for field_name in (
        frozen_fields
    ):
        print(
            f"  {field_name}"
        )

    original_regions = (
        load_regions()
    )

    test_paths = set(
        test[
            "image_path"
        ]
    )

    direct_annotations = (
        build_direct_annotation_lookup(
            original_regions,
            test_paths,
            frozen_fields,
        )
    )

    donor_lookup = (
        build_attack_donor_lookup(
            test,
            direct_annotations,
        )
    )

    rows = []
    exclusions = []
    missing_images = []
    transfer_audit = []

    direct_image_count = 0
    transferred_image_count = 0

    for i, item in enumerate(
        test.itertuples(
            index=False
        ),
        start=1,
    ):
        image_path = (
            item.image_path
        )

        traffic_type = str(
            item.traffic_type
        )

        variant = (
            norm_variant(
                item.variant
            )
        )

        target_size = image_size(
            item.cache_path
        )

        annotation_source = None
        annotation_source_image = None
        annotation_source_variant = None
        bundle = None

        # --------------------------------------------------------------
        # Case 1: image owns ORIGINAL annotation rows.
        # --------------------------------------------------------------

        if (
            image_path
            in direct_annotations
        ):
            bundle = (
                direct_annotations[
                    image_path
                ]
            )

            annotation_source = (
                "self_original"
            )

            annotation_source_image = (
                image_path
            )

            annotation_source_variant = (
                variant
            )

            direct_image_count += 1

        # --------------------------------------------------------------
        # Case 2: official bona-fide without Regions rows.
        # Exact same stem + hardware attack donor only.
        # --------------------------------------------------------------

        elif (
            traffic_type
            == "bonafide"
        ):
            key = (
                str(
                    item.file_stem
                ),
                str(
                    item.hardware_source
                ),
            )

            donor, error = (
                resolve_attack_donor(
                    donor_lookup.get(
                        key,
                        [],
                    )
                )
            )

            if donor is None:
                missing_images.append(
                    {
                        "image_path":
                            image_path,

                        "file_stem":
                            item.file_stem,

                        "hardware_source":
                            item.hardware_source,

                        "traffic_type":
                            traffic_type,

                        "variant":
                            variant,

                        "reason":
                            error,

                        "annotation_source_image_path":
                            "",

                        "annotation_source_variant":
                            "",
                    }
                )

                continue

            donor_size = image_size(
                donor[
                    "cache_path"
                ]
            )

            if (
                donor_size
                != target_size
            ):
                missing_images.append(
                    {
                        "image_path":
                            image_path,

                        "file_stem":
                            item.file_stem,

                        "hardware_source":
                            item.hardware_source,

                        "traffic_type":
                            traffic_type,

                        "variant":
                            variant,

                        "reason":
                            "exact_key_dimension_mismatch",

                        "annotation_source_image_path":
                            donor[
                                "image_path"
                            ],

                        "annotation_source_variant":
                            donor[
                                "variant"
                            ],

                        "target_width":
                            target_size[0],

                        "target_height":
                            target_size[1],

                        "source_width":
                            donor_size[0],

                        "source_height":
                            donor_size[1],
                    }
                )

                continue

            bundle = donor[
                "bundle"
            ]

            annotation_source = (
                "matched_attack_original"
            )

            annotation_source_image = (
                donor[
                    "image_path"
                ]
            )

            annotation_source_variant = (
                donor[
                    "variant"
                ]
            )

            transferred_image_count += 1

            transfer_audit.append(
                {
                    "bonafide_image_path":
                        image_path,

                    "file_stem":
                        item.file_stem,

                    "hardware_source":
                        item.hardware_source,

                    "annotation_source_image_path":
                        donor[
                            "image_path"
                        ],

                    "annotation_source_variant":
                        donor[
                            "variant"
                        ],

                    "target_width":
                        target_size[0],

                    "target_height":
                        target_size[1],

                    "source_width":
                        donor_size[0],

                    "source_height":
                        donor_size[1],

                    "same_file_stem":
                        True,

                    "same_hardware_source":
                        True,

                    "same_dimensions":
                        True,

                    "annotation_provenance":
                        "original_only",
                }
            )

        # --------------------------------------------------------------
        # An official attack without its own ORIGINAL annotations is
        # not allowed to borrow another image's annotations.
        # --------------------------------------------------------------

        else:
            missing_images.append(
                {
                    "image_path":
                        image_path,

                    "file_stem":
                        item.file_stem,

                    "hardware_source":
                        item.hardware_source,

                    "traffic_type":
                        traffic_type,

                    "variant":
                        variant,

                    "reason":
                        "attack_missing_self_original_annotations",

                    "annotation_source_image_path":
                        "",

                    "annotation_source_variant":
                        "",
                }
            )

            continue

        if (
            bundle is None
            or not bundle[
                "text"
            ]
        ):
            missing_images.append(
                {
                    "image_path":
                        image_path,

                    "file_stem":
                        item.file_stem,

                    "hardware_source":
                        item.hardware_source,

                    "traffic_type":
                        traffic_type,

                    "variant":
                        variant,

                    "reason":
                        "no_eligible_original_text_annotations",

                    "annotation_source_image_path":
                        (
                            annotation_source_image
                            or ""
                        ),

                    "annotation_source_variant":
                        (
                            annotation_source_variant
                            or ""
                        ),
                }
            )

            continue

        cache_path = (
            ROOT
            / item.cache_path
        )

        image_rows_before = len(
            rows
        )

        with Image.open(
            cache_path
        ) as im:
            image = im.convert(
                "RGB"
            )

            width, height = (
                image.size
            )

            if (
                width,
                height,
            ) != target_size:
                raise RuntimeError(
                    "Decoded target geometry "
                    "changed unexpectedly"
                )

            for region_number, text_item in enumerate(
                bundle[
                    "text"
                ]
            ):
                field_name = (
                    text_item[
                        "field_name"
                    ]
                )

                box = (
                    text_item[
                        "box"
                    ]
                )

                crop = (
                    choose_face_free_crop(
                        width,
                        height,
                        box,
                        bundle[
                            "faces"
                        ],
                    )
                )

                if crop is None:
                    exclusions.append(
                        {
                            "image_path":
                                image_path,

                            "file_stem":
                                item.file_stem,

                            "traffic_type":
                                traffic_type,

                            "variant":
                                variant,

                            "hardware_source":
                                item.hardware_source,

                            "field_name":
                                field_name,

                            "reason":
                                "no_face_free_512_crop",

                            "annotation_source":
                                annotation_source,

                            "annotation_source_image_path":
                                annotation_source_image,

                            "annotation_source_variant":
                                annotation_source_variant,
                        }
                    )

                    continue

                patch_id = token(
                    f"{image_path}|"
                    f"{field_name}|"
                    f"{box}|"
                    f"{crop}"
                )

                population = (
                    population_name(
                        traffic_type,
                        variant,
                    )
                )

                destination = (
                    CACHE_ROOT
                    / population
                    / str(
                        item.hardware_source
                    )
                    / f"{patch_id}.png"
                )

                save_patch(
                    image,
                    crop,
                    destination,
                )

                x0, y0, x1, y1 = (
                    crop
                )

                rows.append(
                    {
                        "patch_id":
                            patch_id,

                        "image_path":
                            image_path,

                        "file_stem":
                            item.file_stem,

                        "traffic_type":
                            traffic_type,

                        "variant":
                            variant,

                        "hardware_source":
                            item.hardware_source,

                        "label":
                            int(
                                item.label
                            ),

                        "assigned_q":
                            int(
                                item.assigned_q
                            ),

                        "field_name":
                            field_name,

                        "patch_path":
                            str(
                                destination.relative_to(
                                    ROOT
                                )
                            ),

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

                        "annotation_source":
                            annotation_source,

                        "annotation_source_image_path":
                            annotation_source_image,

                        "annotation_source_variant":
                            annotation_source_variant,

                        "annotation_provenance":
                            "original_only",

                        "face_overlap_pixels":
                            0,
                    }
                )

        # Annotation bundle existed, but all candidate crops were rejected.
        if (
            len(rows)
            == image_rows_before
        ):
            missing_images.append(
                {
                    "image_path":
                        image_path,

                    "file_stem":
                        item.file_stem,

                    "hardware_source":
                        item.hardware_source,

                    "traffic_type":
                        traffic_type,

                    "variant":
                        variant,

                    "reason":
                        "all_text_regions_excluded",

                    "annotation_source_image_path":
                        annotation_source_image,

                    "annotation_source_variant":
                        annotation_source_variant,
                }
            )

        if (
            i % 100 == 0
            or i == len(test)
        ):
            print(
                f"processed "
                f"{i}/{len(test)}"
            )

    # -----------------------------------------------------------------
    # Freeze outputs
    # -----------------------------------------------------------------

    index = pd.DataFrame(
        rows
    )

    exclusions = pd.DataFrame(
        exclusions
    )

    missing = pd.DataFrame(
        missing_images
    )

    transfer = pd.DataFrame(
        transfer_audit
    )

    if len(index) == 0:
        raise RuntimeError(
            "No official text patches created"
        )

    if (
        index[
            "patch_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate patch IDs"
        )

    if (
        index[
            "face_overlap_pixels"
        ].max()
        != 0
    ):
        raise RuntimeError(
            "Face content leaked into "
            "text-patch index"
        )

    if set(
        index[
            "annotation_provenance"
        ].unique()
    ) != {
        "original_only"
    }:
        raise RuntimeError(
            "Non-original annotation "
            "provenance reached patch index"
        )

    covered_paths = set(
        index[
            "image_path"
        ]
    )

    # Every attack must remain covered.
    attack_paths = set(
        test.loc[
            test[
                "traffic_type"
            ]
            == "attack",
            "image_path",
        ]
    )

    missing_attacks = (
        attack_paths
        - covered_paths
    )

    if missing_attacks:
        raise RuntimeError(
            f"{len(missing_attacks)} "
            "official attacks have no "
            "usable text patches"
        )

    covered_bona = (
        index.loc[
            index[
                "traffic_type"
            ]
            == "bonafide",
            "image_path",
        ]
        .nunique()
    )

    # With 150 facedancer + 149 textdiffuser official attacks,
    # one of the 300 bona-fide captures may legitimately have no
    # exact same-key annotation donor. Anything worse requires review.
    if covered_bona < 299:
        raise RuntimeError(
            "Too many official bona-fides "
            "lack exact annotation donors: "
            f"covered {covered_bona}/300"
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

    missing.to_csv(
        OUT_MISSING,
        index=False,
    )

    transfer.to_csv(
        OUT_TRANSFER_AUDIT,
        index=False,
    )

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    covered_total = (
        index[
            "image_path"
        ].nunique()
    )

    attack_covered = (
        index.loc[
            index[
                "traffic_type"
            ]
            == "attack",
            "image_path",
        ]
        .nunique()
    )

    bona_covered = (
        index.loc[
            index[
                "traffic_type"
            ]
            == "bonafide",
            "image_path",
        ]
        .nunique()
    )

    print(
        "\nOFFICIAL TEXT-PATCH CACHE "
        "WITH BONA-FIDE TRANSFER:"
    )

    print(
        f"  images total:             "
        f"{len(test)}"
        f"\n  images covered:           "
        f"{covered_total}"
        f"\n  attacks covered:          "
        f"{attack_covered}/1085"
        f"\n  bona-fides covered:       "
        f"{bona_covered}/300"
        f"\n  images missing:           "
        f"{len(test) - covered_total}"
        f"\n  patches:                  "
        f"{len(index)}"
        f"\n  crop exclusions:          "
        f"{len(exclusions)}"
        f"\n  self-annotated images:    "
        f"{direct_image_count}"
        f"\n  transferred bona-fides:   "
        f"{transferred_image_count}"
    )

    temp = index.copy()

    temp[
        "population"
    ] = temp[
        "variant"
    ]

    temp.loc[
        temp[
            "traffic_type"
        ]
        == "bonafide",
        "population",
    ] = "bonafide"

    print(
        "\nPatch counts by population:"
    )

    print(
        temp.groupby(
            "population"
        )
        .size()
        .to_string()
    )

    print(
        "\nImages covered by population:"
    )

    print(
        temp.groupby(
            "population"
        )[
            "image_path"
        ]
        .nunique()
        .to_string()
    )

    per_image = (
        index.groupby(
            "image_path"
        )
        .size()
    )

    print(
        "\nPatches per covered image:"
        f"\n  min:    "
        f"{per_image.min()}"
        f"\n  median: "
        f"{per_image.median():.1f}"
        f"\n  mean:   "
        f"{per_image.mean():.2f}"
        f"\n  max:    "
        f"{per_image.max()}"
    )

    print(
        "\nAnnotation source counts:"
    )

    print(
        index[
            [
                "image_path",
                "annotation_source",
            ]
        ]
        .drop_duplicates()
        [
            "annotation_source"
        ]
        .value_counts()
        .to_string()
    )

    if len(transfer):
        print(
            "\nTransferred bona-fide "
            "donor families:"
        )

        print(
            transfer[
                "annotation_source_variant"
            ]
            .value_counts()
            .to_string()
        )

    if len(exclusions):
        print(
            "\nCrop exclusions:"
        )

        print(
            exclusions.groupby(
                [
                    "traffic_type",
                    "variant",
                    "field_name",
                    "reason",
                ],
                dropna=False,
            )
            .size()
            .to_string()
        )

    if len(missing):
        print(
            "\nMISSING IMAGES:"
        )

        columns = [
            "image_path",
            "file_stem",
            "hardware_source",
            "traffic_type",
            "variant",
            "reason",
            "annotation_source_image_path",
            "annotation_source_variant",
        ]

        columns = [
            column
            for column
            in columns
            if column
            in missing.columns
        ]

        print(
            missing[
                columns
            ]
            .to_string(
                index=False
            )
        )

    print(
        f"\nIndex:          "
        f"{OUT_INDEX}"
        f"\nExclusions:     "
        f"{OUT_EXCLUSIONS}"
        f"\nMissing:        "
        f"{OUT_MISSING}"
        f"\nTransfer audit: "
        f"{OUT_TRANSFER_AUDIT}"
        f"\nCache:          "
        f"{CACHE_ROOT}"
    )

    print(
        "\nSAFETY AUDIT:"
        "\n  altered annotations used:      NO"
        "\n  cross-hardware transfer:       NO"
        "\n  coordinate resizing/warping:   NO"
        "\n  exact donor dimensions:        REQUIRED"
        "\n  face overlap in saved patches: ZERO"
    )

    if (
        attack_covered == 1085
        and bona_covered == 300
    ):
        print(
            "\nPASS: all 1385 official images "
            "have valid text patches."
        )

    elif (
        attack_covered == 1085
        and bona_covered == 299
    ):
        print(
            "\nPARTIAL PASS: all attacks and "
            "299/300 bona-fides are covered."
            "\nOne bona-fide has no safe exact-key "
            "annotation donor."
            "\nDo NOT fabricate its coordinates."
            "\nEvaluate all detectors on the same "
            "1384-image common subset."
        )

    else:
        raise RuntimeError(
            "Unexpected official patch "
            "coverage state"
        )


if __name__ == "__main__":
    main()