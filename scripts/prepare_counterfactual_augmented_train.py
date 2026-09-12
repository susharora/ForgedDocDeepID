#!/usr/bin/env python3
"""Build train-only counterfactual augmentation under frozen Policy C."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "FantasyID"

INVENTORY = (
    ROOT
    / "output/fantasyid_inventory_2026-09-05_023126.xlsx"
)

TRAIN_MANIFEST = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_project_train.csv"
)

POLICY_INDEX = (
    ROOT
    / "output/policy_c_cache_index.csv"
)

CACHE_ROOT = (
    ROOT
    / "data/processed/counterfactual_augmented_train"
)

OUT = (
    ROOT
    / "output/counterfactual_augmented_train_index.csv"
)

INVENTORY_SHA = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
)

TRAIN_SHA = (
    "6307b1516f0e8a6db661077fedad704bbd13f6242c9795ca11310ac86d650cc8"
)


def sha_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def decode_rgb(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def jpeg_params(data):
    with Image.open(io.BytesIO(data)) as im:
        q = im.quantization

        return (
            {
                0: [int(v) for v in q[0]],
                1: [int(v) for v in q[1]],
            },
            int(JpegImagePlugin.get_sampling(im)),
        )


def encode_quality(rgb, quality):
    buf = io.BytesIO()

    Image.fromarray(rgb).save(
        buf,
        "JPEG",
        quality=int(quality),
        subsampling=2,
        optimize=False,
        progressive=False,
    )

    return buf.getvalue()


def final_q(file_stem, hardware):
    key = (
        f"{file_stem}|{hardware}"
    ).encode()

    n = int.from_bytes(
        hashlib.sha256(key).digest()[:8],
        "big",
    )

    return 50 + n % 41


def policy_c(rgb, file_stem, hardware):
    q75 = encode_quality(
        rgb,
        75,
    )

    q75_rgb = decode_rgb(
        q75
    )

    q = final_q(
        file_stem,
        hardware,
    )

    return (
        encode_quality(
            q75_rgb,
            q,
        ),
        q,
    )


def attack_matched_source(
    bonafide_rgb,
    attack_tables,
    attack_sampling,
):
    buf = io.BytesIO()

    Image.fromarray(
        bonafide_rgb
    ).save(
        buf,
        "JPEG",
        qtables=attack_tables,
        subsampling=attack_sampling,
        optimize=False,
        progressive=False,
    )

    encoded = buf.getvalue()

    tables, sampling = (
        jpeg_params(encoded)
    )

    if tables != attack_tables:
        raise RuntimeError(
            "Attack qtables not preserved"
        )

    if sampling != attack_sampling:
        raise RuntimeError(
            "Attack subsampling not preserved"
        )

    return decode_rgb(
        encoded
    )


def mask_from_boxes(
    height,
    width,
    boxes,
):
    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    clipped = 0

    for x0, y0, x1, y1 in boxes:
        cx0 = max(
            0,
            min(width, x0),
        )

        cy0 = max(
            0,
            min(height, y0),
        )

        cx1 = max(
            0,
            min(width, x1),
        )

        cy1 = max(
            0,
            min(height, y1),
        )

        if (
            cx1 <= cx0
            or cy1 <= cy0
        ):
            raise RuntimeError(
                "Annotation has no "
                "image intersection"
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

    return mask, clipped


def load_boxes(attack_paths):
    if sha_file(INVENTORY) != INVENTORY_SHA:
        raise RuntimeError(
            "Inventory SHA mismatch"
        )

    r = pd.read_excel(
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

    missing = required - set(
        r.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing Regions columns: "
            f"{sorted(missing)}"
        )

    r = r[
        r.image_path.isin(
            attack_paths
        )
    ].copy()

    r = r[
        r.region_provenance_raw
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("altered")
    ]

    boxes = {}

    for path, group in r.groupby(
        "image_path"
    ):
        face = []
        text = []

        for _, row in group.iterrows():
            x = int(row.x)
            y = int(row.y)
            w = int(row.width)
            h = int(row.height)

            box = (
                x,
                y,
                x + w,
                y + h,
            )

            if (
                str(row.field_name)
                .strip()
                .lower()
                == "face"
            ):
                face.append(box)
            else:
                text.append(box)

        boxes[path] = {
            "face": face,
            "text": text,
        }

    missing = (
        set(attack_paths)
        - set(boxes)
    )

    if missing:
        raise RuntimeError(
            f"{len(missing)} attacks "
            "missing altered boxes"
        )

    for path, value in boxes.items():
        if (
            not value["face"]
            or not value["text"]
        ):
            raise RuntimeError(
                f"Missing face/text "
                f"intervention: {path}"
            )

    return boxes


def main():
    if sha_file(TRAIN_MANIFEST) != TRAIN_SHA:
        raise RuntimeError(
            "Project-train manifest SHA mismatch"
        )

    train = pd.read_csv(
        TRAIN_MANIFEST
    )

    if len(train) != 1440:
        raise RuntimeError(
            "Expected 1440 project-train rows"
        )

    if (
        train.file_stem.nunique()
        != 160
    ):
        raise RuntimeError(
            "Expected 160 train identities"
        )

    attacks = train[
        train.traffic_type
        == "attack"
    ].copy()

    bona = train[
        train.traffic_type
        == "bonafide"
    ].copy()

    if len(attacks) != 960:
        raise RuntimeError(
            "Expected 960 train attacks"
        )

    if len(bona) != 480:
        raise RuntimeError(
            "Expected 480 train bona-fides"
        )

    keys = [
        "file_stem",
        "hardware_source",
    ]

    bona_lookup = (
        bona.set_index(keys)
    )

    if len(bona_lookup) != 480:
        raise RuntimeError(
            "Unexpected bona-fide pairing"
        )

    # No dimension-mismatch exclusions should
    # exist in project_train.
    for row in attacks.itertuples(
        index=False
    ):
        if (
            row.file_stem,
            row.hardware_source,
        ) not in bona_lookup.index:
            raise RuntimeError(
                "Missing matched bona-fide: "
                f"{row.image_path}"
            )

    boxes = load_boxes(
        set(attacks.image_path)
    )

    policy_index = pd.read_csv(
        POLICY_INDEX
    )

    natural = policy_index[
        policy_index.split
        == "project_train"
    ].copy()

    if len(natural) != 1440:
        raise RuntimeError(
            "Expected 1440 natural "
            "Policy-C cache rows"
        )

    rows = []

    # --------------------------------------------------
    # Natural images already cached under Policy C.
    # --------------------------------------------------

    for row in natural.itertuples(
        index=False
    ):
        rows.append(
            {
                "source_type":
                    (
                        "original_attack"
                        if row.label == 1
                        else "natural_bonafide"
                    ),

                "image_path":
                    row.image_path,

                "cache_path":
                    row.cache_path,

                "file_stem":
                    row.file_stem,

                "traffic_type":
                    row.traffic_type,

                "variant":
                    (
                        ""
                        if pd.isna(row.variant)
                        else row.variant
                    ),

                "hardware_source":
                    row.hardware_source,

                "label":
                    int(row.label),

                "assigned_q":
                    int(row.assigned_q),

                "cache_sha256":
                    row.cache_sha256,
            }
        )

    clipped_boxes = 0

    # --------------------------------------------------
    # Synthetic training interventions.
    # --------------------------------------------------

    for i, attack in enumerate(
        attacks.itertuples(
            index=False
        ),
        start=1,
    ):
        key = (
            attack.file_stem,
            attack.hardware_source,
        )

        bona_row = bona_lookup.loc[
            key
        ]

        attack_raw = (
            DATA
            / attack.image_path
        ).read_bytes()

        bona_raw = (
            DATA
            / bona_row.image_path
        ).read_bytes()

        if (
            sha_bytes(attack_raw)
            != attack.image_sha256
        ):
            raise RuntimeError(
                "Attack SHA mismatch: "
                f"{attack.image_path}"
            )

        if (
            sha_bytes(bona_raw)
            != bona_row.image_sha256
        ):
            raise RuntimeError(
                "Bona-fide SHA mismatch: "
                f"{bona_row.image_path}"
            )

        attack_rgb = decode_rgb(
            attack_raw
        )

        bona_rgb = decode_rgb(
            bona_raw
        )

        if (
            attack_rgb.shape
            != bona_rgb.shape
        ):
            raise RuntimeError(
                "Project-train pair "
                "dimension mismatch: "
                f"{attack.image_path}"
            )

        tables, sampling = (
            jpeg_params(
                attack_raw
            )
        )

        source_rgb = (
            attack_matched_source(
                bona_rgb,
                tables,
                sampling,
            )
        )

        h, w = (
            attack_rgb.shape[:2]
        )

        annotation = boxes[
            attack.image_path
        ]

        face_mask, face_clip = (
            mask_from_boxes(
                h,
                w,
                annotation["face"],
            )
        )

        text_mask, text_clip = (
            mask_from_boxes(
                h,
                w,
                annotation["text"],
            )
        )

        clipped_boxes += (
            face_clip
            + text_clip
        )

        modes = {
            "face_restored":
                (
                    face_mask,
                    1,
                ),

            "text_restored":
                (
                    text_mask,
                    1,
                ),

            "both_restored":
                (
                    face_mask
                    | text_mask,
                    0,
                ),
        }

        for mode, (
            mask,
            label,
        ) in modes.items():
            restored = (
                attack_rgb.copy()
            )

            restored[
                mask
            ] = source_rgb[
                mask
            ]

            encoded, assigned_q = (
                policy_c(
                    restored,
                    attack.file_stem,
                    attack.hardware_source,
                )
            )

            destination = (
                CACHE_ROOT
                / mode
                / attack.image_path
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination.write_bytes(
                encoded
            )

            rows.append(
                {
                    "source_type":
                        mode,

                    "image_path":
                        attack.image_path,

                    "cache_path":
                        str(
                            destination.relative_to(
                                ROOT
                            )
                        ),

                    "file_stem":
                        attack.file_stem,

                    "traffic_type":
                        (
                            "attack"
                            if label == 1
                            else "bonafide"
                        ),

                    "variant":
                        attack.variant,

                    "hardware_source":
                        attack.hardware_source,

                    "label":
                        label,

                    "assigned_q":
                        assigned_q,

                    "cache_sha256":
                        sha_bytes(encoded),
                }
            )

        if (
            i % 100 == 0
            or i == len(attacks)
        ):
            print(
                f"processed "
                f"{i}/{len(attacks)} attacks"
            )

    index = pd.DataFrame(
        rows
    )

    if len(index) != 4320:
        raise RuntimeError(
            f"Expected 4320 training rows, "
            f"got {len(index)}"
        )

    counts = (
        index.label
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if counts != {
        0: 1440,
        1: 2880,
    }:
        raise RuntimeError(
            f"Unexpected class counts: "
            f"{counts}"
        )

    expected_types = {
        "natural_bonafide": 480,
        "original_attack": 960,
        "face_restored": 960,
        "text_restored": 960,
        "both_restored": 960,
    }

    types = (
        index.source_type
        .value_counts()
        .to_dict()
    )

    if types != expected_types:
        raise RuntimeError(
            f"Unexpected source counts: "
            f"{types}"
        )

    # Every representation of a card/hardware
    # pair must receive the same final JPEG Q.
    q_counts = (
        index.groupby(
            [
                "file_stem",
                "hardware_source",
            ]
        )
        .assigned_q
        .nunique()
    )

    if q_counts.max() != 1:
        raise RuntimeError(
            "Final-Q matching failed"
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.to_csv(
        OUT,
        index=False,
    )

    print(
        "\nCounterfactual-augmented "
        "train cache complete."
    )

    print(
        "\nSource counts:"
    )

    print(
        index.source_type
        .value_counts()
        .to_string()
    )

    print(
        "\nClass counts:"
    )

    print(
        index.label
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nClipped altered boxes: "
        f"{clipped_boxes}"
    )

    print(
        f"\nWrote {OUT}"
        f"\nCache: {CACHE_ROOT}"
    )

    print(
        "\nNo dev images or dev "
        "counterfactuals were used."
    )


if __name__ == "__main__":
    main()