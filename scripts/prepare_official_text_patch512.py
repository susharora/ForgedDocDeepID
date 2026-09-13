#!/usr/bin/env python3
"""
Prepare annotation-localized native 512x512 text patches for the complete
FantasyID official test set.

IMPORTANT:
    - Uses ONLY region_provenance == "original".
    - Never uses altered-region annotations.
    - Never uses attack type to decide where to crop.
    - Candidate field names are frozen from the project_train attack-side
      patch experiment.
    - Same extraction policy is applied to:
          bona-fide
          digital_3
          facedancer
          textdiffuserft_bfei

This is therefore a region-annotation-assisted diagnostic, not a final
end-to-end detector.

Input images are the frozen native-resolution Policy-C official-test cache.
No whole-document resize and no additional JPEG compression are performed.
Patches are saved losslessly as PNG.
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
    ).hexdigest()[:24]


def norm_field(value):
    return (
        str(value)
        .strip()
        .lower()
    )


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
        min(high, value),
    )


def row_box(row):
    x = int(
        round(float(row.x))
    )

    y = int(
        round(float(row.y))
    )

    w = int(
        round(float(row.width))
    )

    h = int(
        round(float(row.height))
    )

    if w <= 0 or h <= 0:
        raise RuntimeError(
            "Invalid annotation box"
        )

    return (
        x,
        y,
        x + w,
        y + h,
    )


def choose_face_free_crop(
    image_width,
    image_height,
    text_box,
    face_boxes,
):
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
        -32, 32,
        -64, 64,
        -96, 96,
        -128, 128,
        -160, 160,
        -192, 192,
        -224, 224,
        -256, 256,
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

            # Field centre must stay in crop.
            if not (
                x0 <= cx < x0 + PATCH
                and
                y0 <= cy < y0 + PATCH
            ):
                continue

            overlap = sum(
                intersection_area(
                    crop,
                    face,
                )
                for face
                in face_boxes
            )

            distance = (
                (x0 - ideal_x) ** 2
                +
                (y0 - ideal_y) ** 2
            )

            candidates.append(
                (
                    overlap,
                    distance,
                    crop,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    overlap, _, crop = (
        candidates[0]
    )

    if overlap != 0:
        return None

    return crop


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
            norm_field(x)
            for x
            in selected[
                "field_name"
            ]
            .dropna()
        }
    )

    if not fields:
        raise RuntimeError(
            "No frozen training fields"
        )

    return fields


def load_regions():
    if (
        sha256_file(INVENTORY)
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
        - set(regions.columns)
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
        .map(norm_field)
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

    return regions


def save_patch(
    image,
    crop,
    destination,
):
    x0, y0, x1, y1 = crop

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

    patch.save(
        destination,
        "PNG",
        compress_level=1,
    )


def main():
    for path in [
        INVENTORY,
        TEST_INDEX,
        TRAIN_PATCH_INDEX,
    ]:
        if not path.is_file():
            raise RuntimeError(
                f"Missing: {path}"
            )

    test = pd.read_csv(
        TEST_INDEX
    )

    if len(test) != 1385:
        raise RuntimeError(
            f"Expected 1385 official "
            f"images, got {len(test)}"
        )

    if (
        test["image_path"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate official paths"
        )

    fields = load_frozen_fields()

    print(
        "Frozen eligible text fields:"
    )

    for field in fields:
        print(
            f"  {field}"
        )

    regions = load_regions()

    # Crucially, extraction uses ORIGINAL text annotations only.
    original = regions[
        regions[
            "provenance_norm"
        ]
        == "original"
    ].copy()

    original = original[
        original[
            "image_path"
        ].isin(
            set(
                test[
                    "image_path"
                ]
            )
        )
    ]

    grouped = {
        image_path:
            group.copy()
        for image_path, group
        in original.groupby(
            "image_path"
        )
    }

    rows = []
    exclusions = []

    for i, item in enumerate(
        test.itertuples(
            index=False
        ),
        start=1,
    ):
        image_path = (
            item.image_path
        )

        if (
            image_path
            not in grouped
        ):
            continue

        group = grouped[
            image_path
        ]

        # Face boxes are used only to stop text crops leaking face content.
        face_rows = group[
            group[
                "field_norm"
            ]
            == "face"
        ]

        face_boxes = list(
            {
                row_box(row)
                for row
                in face_rows.itertuples(
                    index=False
                )
            }
        )

        # Fixed field set learned from TRAIN metadata.
        text_rows = group[
            group[
                "field_norm"
            ]
            .isin(fields)
        ].copy()

        # Remove annotation duplicates but retain genuinely separate boxes.
        text_rows[
            "box_key"
        ] = [
            (
                row.field_norm,
                *row_box(row),
            )
            for row in text_rows.itertuples(
                index=False
            )
        ]

        text_rows = (
            text_rows
            .drop_duplicates(
                "box_key"
            )
        )

        cache_path = (
            ROOT
            / item.cache_path
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

            for region_number, row in enumerate(
                text_rows.itertuples(
                    index=False
                )
            ):
                box = row_box(
                    row
                )

                crop = (
                    choose_face_free_crop(
                        width,
                        height,
                        box,
                        face_boxes,
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
                                item.traffic_type,

                            "variant":
                                (
                                    ""
                                    if pd.isna(
                                        item.variant
                                    )
                                    else item.variant
                                ),

                            "hardware_source":
                                item.hardware_source,

                            "field_name":
                                row.field_norm,

                            "reason":
                                "no_face_free_512_crop",
                        }
                    )

                    continue

                patch_id = token(
                    f"{image_path}|"
                    f"{row.field_norm}|"
                    f"{box}|"
                    f"{crop}"
                )

                family = (
                    "bonafide"
                    if item.traffic_type
                    == "bonafide"
                    else str(
                        item.variant
                    )
                )

                destination = (
                    CACHE_ROOT
                    / family
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
                            item.traffic_type,

                        "variant":
                            (
                                ""
                                if pd.isna(
                                    item.variant
                                )
                                else item.variant
                            ),

                        "hardware_source":
                            item.hardware_source,

                        "label":
                            int(item.label),

                        "assigned_q":
                            int(item.assigned_q),

                        "field_name":
                            row.field_norm,

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
                            "original_only",

                        "face_overlap_pixels":
                            0,
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

    index = pd.DataFrame(
        rows
    )

    exclusions = pd.DataFrame(
        exclusions
    )

    if len(index) == 0:
        raise RuntimeError(
            "No patches created"
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
            "Face leaked into "
            "official text patches"
        )

    covered = set(
        index[
            "image_path"
        ]
    )

    missing = test[
        ~test[
            "image_path"
        ].isin(
            covered
        )
    ].copy()

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

    print(
        "\nOFFICIAL TEXT-PATCH CACHE:"
    )

    print(
        f"  images total:   {len(test)}"
        f"\n  images covered: {len(covered)}"
        f"\n  images missing: {len(missing)}"
        f"\n  patches:        {len(index)}"
        f"\n  exclusions:     {len(exclusions)}"
    )

    print(
        "\nPatch counts by population:"
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
        temp.groupby(
            "population"
        )
        .size()
        .to_string()
    )

    print(
        "\nPatches per image:"
    )

    per_image = (
        index.groupby(
            "image_path"
        )
        .size()
    )

    print(
        f"  min:    "
        f"{per_image.min()}"
        f"\n  median: "
        f"{per_image.median():.1f}"
        f"\n  mean:   "
        f"{per_image.mean():.2f}"
        f"\n  max:    "
        f"{per_image.max()}"
    )

    if len(exclusions):
        print(
            "\nExclusions:"
        )

        print(
            exclusions.groupby(
                [
                    "variant",
                    "field_name",
                    "reason",
                ],
                dropna=False,
            )
            .size()
            .to_string()
        )

    print(
        f"\nIndex:      {OUT_INDEX}"
        f"\nExclusions: {OUT_EXCLUSIONS}"
        f"\nMissing:    {OUT_MISSING}"
        f"\nCache:      {CACHE_ROOT}"
    )

    if len(missing):
        raise RuntimeError(
            "Some official images have "
            "zero eligible text patches. "
            "Inspect missing CSV before "
            "running evaluation."
        )

    print(
        "\nPASS: every official image has "
        "at least one annotation-localized "
        "text patch."
    )


if __name__ == "__main__":
    main()