#!/usr/bin/env python3
"""
Compression-controlled counterfactual restoration evaluation.

Evaluates the frozen seed-10 Policy-C ResNet on:

    original
    face_restored
    text_restored
    both_restored

Scientific constraints
----------------------
- Frozen project_train/dev_val manifests only.
- digital_1 and digital_2 only.
- Match attack -> bona-fide by file_stem + hardware_source.
- Identity-coordinate restoration only.
- NO semantic source-box resizing.
- NO geometric registration/warping.
- NO MCU snapping.
- Three dimension-mismatched digital_2/chinese-001_03 pairs excluded.
- Out-of-image annotations are clipped to decoded image bounds.
- Bona-fide restoration source is re-encoded using the attack JPEG
  quantisation tables and subsampling before pixel replacement.
- Composite then passes through the exact frozen Policy C:
      common Q75 -> matched deterministic Q50-90.
- Original attack is re-scored in the SAME four-image batch as the
  three counterfactuals and checked against the frozen cached score.
- Matched same-card/same-hardware bona-fide score is attached as the
  counterfactual floor.
- Train and dev are summarised separately.
- Uncertainty uses card-stem cluster bootstrap.
"""

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


# ---------------------------------------------------------------------
# Paths / frozen constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "FantasyID"

INVENTORY = (
    ROOT
    / "output"
    / "fantasyid_inventory_2026-09-05_023126.xlsx"
)

ALIGNMENT = (
    ROOT
    / "output"
    / "counterfactual_alignment_audit.csv"
)

TRAIN_MANIFEST = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_project_train.csv"
)

DEV_MANIFEST = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_dev_val.csv"
)

POLICY_INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

FROZEN_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_predictions.csv"
)

CHECKPOINT = (
    ROOT
    / "runs"
    / "post_hoc_compression_controlled_resnet18_seed10"
    / "checkpoints"
    / "best.pt"
)

CHECKPOINT_HASH = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_checkpoint.sha256"
)

OUT_PREDICTIONS = (
    ROOT
    / "output"
    / "counterfactual_policy_c_seed10_predictions.csv"
)

OUT_SUMMARY = (
    ROOT
    / "output"
    / "counterfactual_policy_c_seed10_summary.csv"
)

INVENTORY_SHA256 = (
    "54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"
)

TRAIN_SHA256 = (
    "6307b1516f0e8a6db661077fedad704bbd13f6242c9795ca11310ac86d650cc8"
)

DEV_SHA256 = (
    "46953b0e474fb52be7a94c2555d0453a57123d8412bd2e21780986155a9e250d"
)

SEED = 10
N_BOOT = 5000

CONTENT_H = 512
CANVAS_W = 864

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

MODES = [
    "original",
    "face_restored",
    "text_restored",
    "both_restored",
]


# ---------------------------------------------------------------------
# Basic helpers
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
    return hashlib.sha256(data).hexdigest()


def jpeg_decode_rgb(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def encode_quality(rgb, quality):
    buf = io.BytesIO()

    Image.fromarray(rgb).save(
        buf,
        "JPEG",
        quality=int(quality),
        subsampling=2,       # 4:2:0
        optimize=False,
        progressive=False,
    )

    return buf.getvalue()


def jpeg_parameters(data):
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "JPEG":
            raise RuntimeError("Expected JPEG")

        q = im.quantization

        if 0 not in q or 1 not in q:
            raise RuntimeError(
                "Missing JPEG quantisation tables"
            )

        tables = {
            0: [int(v) for v in q[0]],
            1: [int(v) for v in q[1]],
        }

        sampling = int(
            JpegImagePlugin.get_sampling(im)
        )

    return tables, sampling


def final_q(file_stem, hardware_source):
    """
    Exact frozen Policy-C final-Q assignment.
    """

    key = (
        f"{file_stem}|{hardware_source}"
    ).encode()

    n = int.from_bytes(
        hashlib.sha256(key).digest()[:8],
        "big",
    )

    return 50 + n % 41


def policy_c_bytes(
    rgb,
    file_stem,
    hardware_source,
):
    """
    Frozen Policy C:

        RGB
        -> common Q75 / 4:2:0
        -> decode
        -> matched deterministic Q50-90 / 4:2:0
    """

    q75 = encode_quality(
        rgb,
        75,
    )

    q75_rgb = jpeg_decode_rgb(
        q75
    )

    quality = final_q(
        file_stem,
        hardware_source,
    )

    return encode_quality(
        q75_rgb,
        quality,
    )


def recompress_bonafide_like_attack(
    bonafide_rgb,
    attack_qtables,
    attack_sampling,
):
    """
    Produce restoration source using the attack JPEG parameters.
    """

    buf = io.BytesIO()

    Image.fromarray(
        bonafide_rgb
    ).save(
        buf,
        "JPEG",
        qtables=attack_qtables,
        subsampling=attack_sampling,
        optimize=False,
        progressive=False,
    )

    encoded = buf.getvalue()

    tables, sampling = jpeg_parameters(
        encoded
    )

    if tables != attack_qtables:
        raise RuntimeError(
            "Attack-matched quantisation tables "
            "were not preserved"
        )

    if sampling != attack_sampling:
        raise RuntimeError(
            "Attack-matched subsampling "
            "was not preserved"
        )

    return jpeg_decode_rgb(
        encoded
    )


# ---------------------------------------------------------------------
# Model preprocessing
# ---------------------------------------------------------------------

def tensor_from_policy_bytes(data):
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")

        width, height = im.size

        new_width = int(
            round(
                width
                * CONTENT_H
                / height
            )
        )

        if new_width > CANVAS_W:
            raise RuntimeError(
                "Image exceeds frozen r512 canvas: "
                f"{width}x{height} -> "
                f"{new_width}px"
            )

        im = TF.resize(
            im,
            [CONTENT_H, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        x = TF.to_tensor(im)

    x = TF.normalize(
        x,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    canvas = torch.zeros(
        (3, CONTENT_H, CANVAS_W),
        dtype=x.dtype,
    )

    x0 = (
        CANVAS_W - new_width
    ) // 2

    canvas[
        :,
        :,
        x0:x0 + new_width,
    ] = x

    return canvas


# ---------------------------------------------------------------------
# Frozen population
# ---------------------------------------------------------------------

def load_valid_pairs():
    df = pd.read_csv(
        ALIGNMENT
    )

    if len(df) != 1266:
        raise RuntimeError(
            f"Expected 1266 alignment rows, "
            f"got {len(df)}"
        )

    bad = df[
        ~df.same_dimensions
    ]

    expected_bad = {
        (
            "digital_2",
            "chinese-001_03",
            "huawei",
        ),
        (
            "digital_2",
            "chinese-001_03",
            "iphone15pro",
        ),
        (
            "digital_2",
            "chinese-001_03",
            "scan",
        ),
    }

    actual_bad = set(
        zip(
            bad.variant,
            bad.file_stem,
            bad.hardware_source,
        )
    )

    if actual_bad != expected_bad:
        raise RuntimeError(
            "Unexpected dimension mismatches:\n"
            f"{actual_bad}"
        )

    valid = df[
        df.same_dimensions
    ].copy()

    shifted = valid[
        (valid.phase_dy_small != 0)
        | (valid.phase_dx_small != 0)
    ]

    if len(shifted):
        raise RuntimeError(
            "Found dimension-matched "
            "non-zero registration cases"
        )

    if len(valid) != 1263:
        raise RuntimeError(
            f"Expected 1263 valid pairs, "
            f"got {len(valid)}"
        )

    print(
        "Alignment population:"
        "\n  total attacks:             1266"
        "\n  excluded dimension cases:    3"
        "\n  identity-coordinate pairs: 1263"
        "\n  geometric resizing:        OFF"
        "\n  registration warp:         OFF"
        "\n  MCU snapping:              OFF"
    )

    return valid


def load_manifests():
    if (
        sha256_file(TRAIN_MANIFEST)
        != TRAIN_SHA256
    ):
        raise RuntimeError(
            "Project-train manifest SHA mismatch"
        )

    if (
        sha256_file(DEV_MANIFEST)
        != DEV_SHA256
    ):
        raise RuntimeError(
            "Dev manifest SHA mismatch"
        )

    train = pd.read_csv(
        TRAIN_MANIFEST
    )

    dev = pd.read_csv(
        DEV_MANIFEST
    )

    train["split"] = "project_train"
    dev["split"] = "dev_val"

    frame = pd.concat(
        [train, dev],
        ignore_index=True,
    )

    if len(frame) != 1899:
        raise RuntimeError(
            "Unexpected manifest population"
        )

    if frame.image_path.duplicated().any():
        raise RuntimeError(
            "Duplicate manifest image paths"
        )

    return frame.set_index(
        "image_path"
    )


# ---------------------------------------------------------------------
# Regions schema / boxes
# ---------------------------------------------------------------------

def normalise_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def named_column(frame, candidates):
    lookup = {
        normalise_name(c): c
        for c in frame.columns
    }

    for candidate in candidates:
        key = normalise_name(
            candidate
        )

        if key in lookup:
            return lookup[key]

    return None


def infer_semantic_column(frame):
    for column in frame.columns:
        values = set(
            frame[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        if (
            "face" in values
            and (
                "dob" in values
                or "doe" in values
            )
        ):
            return column

    return None


def infer_status_column(frame):
    for column in frame.columns:
        values = set(
            frame[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        if (
            "altered" in values
            and "original" in values
        ):
            return column

    return None


def resolve_regions_schema(regions):
    schema = {
        "image_path":
            named_column(
                regions,
                [
                    "image_path",
                    "relative_path",
                    "filepath",
                    "path",
                ],
            ),

        "semantic":
            named_column(
                regions,
                [
                    "field_name",
                    "semantic_field",
                    "semantic",
                    "field",
                ],
            ),

        "status":
            named_column(
                regions,
                [
                    "region_provenance_raw",
                    "status",
                    "region_status",
                    "alteration_status",
                ],
            ),

        "x":
            named_column(
                regions,
                ["x", "bbox_x", "left", "x0"],
            ),

        "y":
            named_column(
                regions,
                ["y", "bbox_y", "top", "y0"],
            ),

        "width":
            named_column(
                regions,
                ["width", "bbox_width", "w"],
            ),

        "height":
            named_column(
                regions,
                ["height", "bbox_height", "h"],
            ),
    }

    if schema["semantic"] is None:
        schema["semantic"] = (
            infer_semantic_column(
                regions
            )
        )

    if schema["status"] is None:
        schema["status"] = (
            infer_status_column(
                regions
            )
        )

    missing = [
        key
        for key, value
        in schema.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            "Could not resolve Regions schema. "
            f"Missing: {missing}\n"
            f"Columns: {list(regions.columns)}"
        )

    print(
        "\nResolved Regions schema:"
    )

    for key, value in schema.items():
        print(
            f"  {key:10s} -> {value}"
        )

    return schema


def load_region_boxes(valid_paths):
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

    schema = resolve_regions_schema(
        regions
    )

    subset = regions[
        regions[
            schema["image_path"]
        ].isin(valid_paths)
    ].copy()

    status = (
        subset[
            schema["status"]
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    subset = subset[
        status == "altered"
    ].copy()

    boxes = {}

    for image_path, group in subset.groupby(
        schema["image_path"]
    ):
        face = []
        text = []

        for _, row in group.iterrows():
            values = [
                pd.to_numeric(
                    row[schema["x"]],
                    errors="coerce",
                ),
                pd.to_numeric(
                    row[schema["y"]],
                    errors="coerce",
                ),
                pd.to_numeric(
                    row[schema["width"]],
                    errors="coerce",
                ),
                pd.to_numeric(
                    row[schema["height"]],
                    errors="coerce",
                ),
            ]

            if any(
                pd.isna(v)
                for v in values
            ):
                raise RuntimeError(
                    "Non-numeric altered region: "
                    f"{image_path}"
                )

            x, y, w, h = [
                int(round(float(v)))
                for v in values
            ]

            if (
                w <= 0
                or h <= 0
            ):
                raise RuntimeError(
                    "Non-positive altered box: "
                    f"{image_path}"
                )

            box = (
                x,
                y,
                x + w,
                y + h,
            )

            semantic = (
                str(
                    row[
                        schema["semantic"]
                    ]
                )
                .strip()
                .lower()
            )

            if semantic == "face":
                face.append(box)
            else:
                text.append(box)

        boxes[image_path] = {
            "face": face,
            "text": text,
        }

    missing = (
        set(valid_paths)
        - set(boxes)
    )

    if missing:
        raise RuntimeError(
            f"{len(missing)} attacks lack "
            "altered-region annotations"
        )

    no_face = [
        path
        for path, value in boxes.items()
        if not value["face"]
    ]

    no_text = [
        path
        for path, value in boxes.items()
        if not value["text"]
    ]

    if no_face or no_text:
        raise RuntimeError(
            "Expected every attack to have "
            "altered face AND altered text"
        )

    print(
        "\nAltered-region validation:"
        f"\n  attack images: {len(boxes)}"
        f"\n  missing face:  {len(no_face)}"
        f"\n  missing text:  {len(no_text)}"
    )

    return boxes


# ---------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------

def mask_from_boxes(
    height,
    width,
    boxes,
):
    """
    Intersect frozen annotation rectangles with actual decoded image.
    No resizing or coordinate transformation.
    """

    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    clipped = []

    for raw_box in boxes:
        x0, y0, x1, y1 = raw_box

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
                "Altered box has no intersection "
                "with decoded image: "
                f"box={raw_box}, "
                f"image={width}x{height}"
            )

        clipped_box = (
            cx0,
            cy0,
            cx1,
            cy1,
        )

        if clipped_box != raw_box:
            clipped.append(
                (
                    raw_box,
                    clipped_box,
                )
            )

        mask[
            cy0:cy1,
            cx0:cx1,
        ] = True

    return mask, clipped


def dilated_mask_from_boxes(
    height,
    width,
    boxes,
    pad=16,
):
    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    for x0, y0, x1, y1 in boxes:
        x0 = max(
            0,
            min(width, x0 - pad),
        )

        y0 = max(
            0,
            min(height, y0 - pad),
        )

        x1 = max(
            0,
            min(width, x1 + pad),
        )

        y1 = max(
            0,
            min(height, y1 + pad),
        )

        if (
            x1 > x0
            and y1 > y0
        ):
            mask[
                y0:y1,
                x0:x1,
            ] = True

    return mask


# ---------------------------------------------------------------------
# Frozen predictions / Policy-C index
# ---------------------------------------------------------------------

def load_frozen_predictions():
    df = pd.read_csv(
        FROZEN_PREDICTIONS
    )

    required = {
        "split",
        "image_path",
        "file_stem",
        "traffic_type",
        "hardware_source",
        "attack_probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Frozen prediction columns missing: "
            f"{sorted(missing)}"
        )

    return df


def build_bonafide_lookup(
    frozen_predictions,
):
    bona = frozen_predictions[
        frozen_predictions.traffic_type
        == "bonafide"
    ].copy()

    key = [
        "split",
        "file_stem",
        "hardware_source",
    ]

    if bona.duplicated(
        key
    ).any():
        raise RuntimeError(
            "Duplicate bona-fide matching keys"
        )

    return (
        bona.set_index(key)
        ["attack_probability"]
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(device):
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
            "Frozen checkpoint SHA mismatch"
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

    checks = {
        "seed": 10,
        "stage": "full",
        "epoch": 12,
    }

    for key, expected in checks.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"Unexpected checkpoint "
                f"{key}: {checkpoint.get(key)}"
            )

    if not np.isclose(
        float(checkpoint["dev_auc"]),
        1.0,
    ):
        raise RuntimeError(
            "Unexpected checkpoint dev AUROC"
        )

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)
    model.eval()

    return model, actual_sha


@torch.no_grad()
def score_batch(
    model,
    tensors,
    device,
):
    batch = torch.stack(
        tensors
    ).to(
        device,
        non_blocking=True,
    )

    with torch.autocast(
        device_type=device.type,
        dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.bfloat16
        ),
        enabled=(
            device.type == "cuda"
        ),
    ):
        logits = model(
            batch
        )

    return (
        torch.softmax(
            logits,
            dim=1,
        )[:, 1]
        .float()
        .cpu()
        .numpy()
    )


# ---------------------------------------------------------------------
# One-off locality check
# ---------------------------------------------------------------------

def audit_reencode_locality(
    valid,
    manifest,
    region_boxes,
    policy_cache,
):
    """
    One representative pair per variant x hardware.

    After Policy-C re-encoding, restoration should not alter decoded
    pixels beyond a 16-pixel dilation of the intervention rectangles.
    """

    checked = set()

    for pair in valid.itertuples(
        index=False
    ):
        key = (
            pair.variant,
            pair.hardware_source,
        )

        if key in checked:
            continue

        attack_path = (
            pair.attack_path
        )

        bona_path = (
            pair.bonafide_path
        )

        attack_raw = (
            DATA / attack_path
        ).read_bytes()

        bona_raw = (
            DATA / bona_path
        ).read_bytes()

        if (
            sha256_bytes(
                attack_raw
            )
            != manifest.loc[
                attack_path,
                "image_sha256",
            ]
        ):
            raise RuntimeError(
                "Attack SHA mismatch during "
                f"locality audit: {attack_path}"
            )

        attack_rgb = jpeg_decode_rgb(
            attack_raw
        )

        bona_rgb = jpeg_decode_rgb(
            bona_raw
        )

        qtables, sampling = (
            jpeg_parameters(
                attack_raw
            )
        )

        source_rgb = (
            recompress_bonafide_like_attack(
                bona_rgb,
                qtables,
                sampling,
            )
        )

        h, w = (
            attack_rgb.shape[:2]
        )

        annotations = (
            region_boxes[
                attack_path
            ]
        )

        face_mask, _ = (
            mask_from_boxes(
                h,
                w,
                annotations["face"],
            )
        )

        text_mask, _ = (
            mask_from_boxes(
                h,
                w,
                annotations["text"],
            )
        )

        both_mask = (
            face_mask
            | text_mask
        )

        composite = (
            attack_rgb.copy()
        )

        composite[
            both_mask
        ] = source_rgb[
            both_mask
        ]

        original_bytes = (
            policy_c_bytes(
                attack_rgb,
                pair.file_stem,
                pair.hardware_source,
            )
        )

        expected_sha = (
            policy_cache.loc[
                attack_path,
                "cache_sha256",
            ]
        )

        if (
            sha256_bytes(
                original_bytes
            )
            != expected_sha
        ):
            raise RuntimeError(
                "Locality-audit original does "
                "not reproduce frozen cache"
            )

        restored_bytes = (
            policy_c_bytes(
                composite,
                pair.file_stem,
                pair.hardware_source,
            )
        )

        original_final = (
            jpeg_decode_rgb(
                original_bytes
            )
        )

        restored_final = (
            jpeg_decode_rgb(
                restored_bytes
            )
        )

        diff = np.abs(
            original_final.astype(
                np.int16
            )
            - restored_final.astype(
                np.int16
            )
        )

        changed = np.any(
            diff > 0,
            axis=2,
        )

        print(
            f"\nPolicy-C locality {key}:"
        )

        for pad in [
            16,
            32,
            64,
        ]:
            guard = (
                dilated_mask_from_boxes(
                    h,
                    w,
                    (
                        annotations["face"]
                        + annotations["text"]
                    ),
                    pad=pad,
                )
            )

            outside = (
                ~guard
            )

            changed_outside = int(
                (
                    changed
                    & outside
                ).sum()
            )

            outside_pixels = int(
                outside.sum()
            )

            fraction = (
                changed_outside
                / outside_pixels
                if outside_pixels
                else 0.0
            )

            if outside_pixels:
                max_abs = int(
                    diff[
                        outside
                    ].max()
                )

                mean_abs = float(
                    diff[
                        outside
                    ].mean()
                )
            else:
                max_abs = 0
                mean_abs = 0.0

            print(
                f"  outside {pad:2d}px: "
                f"changed={changed_outside}, "
                f"fraction={fraction:.8f}, "
                f"max_abs={max_abs}, "
                f"mean_abs={mean_abs:.8f}"
            )

        checked.add(
            key
        )

    expected = {
        ("digital_1", "huawei"),
        ("digital_1", "iphone15pro"),
        ("digital_1", "scan"),
        ("digital_2", "huawei"),
        ("digital_2", "iphone15pro"),
        ("digital_2", "scan"),
    }

    if checked != expected:
        raise RuntimeError(
            "Locality audit did not cover "
            "all six variant/hardware cells"
        )

    print(
    "\nPolicy-C locality audit complete "
    "(diagnostic only; non-blocking)."
    )


# ---------------------------------------------------------------------
# Stem-level bootstrap
# ---------------------------------------------------------------------

def stem_bootstrap(
    group,
):
    """
    Cluster bootstrap entire file_stem groups, keeping the three
    hardware observations for a card together.
    """

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
            5,
        ),
        dtype=float,
    )

    for i in range(N_BOOT):
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
    }


# ---------------------------------------------------------------------
# Split-aware summary
# ---------------------------------------------------------------------

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

    if len(original) != 1263:
        raise RuntimeError(
            "Expected 1263 original scores"
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
            "Original score mapping failed"
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

    for split in [
        "dev_val",
        "project_train",
    ]:
        for variant in [
            "digital_1",
            "digital_2",
        ]:
            for mode in MODES:
                group = df[
                    (
                        df["split"]
                        == split
                    )
                    & (
                        df["variant"]
                        == variant
                    )
                    & (
                        df["mode"]
                        == mode
                    )
                ].copy()

                if len(group) == 0:
                    continue

                rows.append(
                    {
                        "split":
                            split,

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
        (
            "project_train",
            "digital_1",
        ): (480, 160),

        (
            "project_train",
            "digital_2",
        ): (480, 160),

        (
            "dev_val",
            "digital_1",
        ): (153, 51),

        (
            "dev_val",
            "digital_2",
        ): (150, 50),
    }

    for (
        split,
        variant,
    ), (
        expected_images,
        expected_stems,
    ) in expected.items():
        check = summary[
            (
                summary["split"]
                == split
            )
            & (
                summary["variant"]
                == variant
            )
        ]

        if len(check) != 4:
            raise RuntimeError(
                "Missing counterfactual modes for "
                f"{split}/{variant}"
            )

        if not (
            check.n_images
            == expected_images
        ).all():
            raise RuntimeError(
                "Unexpected image count for "
                f"{split}/{variant}"
            )

        if not (
            check.n_stems
            == expected_stems
        ).all():
            raise RuntimeError(
                "Unexpected stem count for "
                f"{split}/{variant}"
            )

    return df, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    valid = load_valid_pairs()

    manifest = load_manifests()

    valid_paths = set(
        valid.attack_path
    )

    region_boxes = (
        load_region_boxes(
            valid_paths
        )
    )

    frozen_predictions = (
        load_frozen_predictions()
    )

    frozen_attack = (
        frozen_predictions[
            frozen_predictions.image_path.isin(
                valid_paths
            )
        ]
        .set_index(
            "image_path"
        )
    )

    if len(frozen_attack) != 1263:
        raise RuntimeError(
            "Expected 1263 frozen attack scores"
        )

    bona_lookup = (
        build_bonafide_lookup(
            frozen_predictions
        )
    )

    policy_index = pd.read_csv(
        POLICY_INDEX
    )

    if policy_index.image_path.duplicated().any():
        raise RuntimeError(
            "Duplicate Policy-C index paths"
        )

    policy_cache = (
        policy_index
        .set_index(
            "image_path"
        )
    )

    # --------------------------------------------------------------
    # One-off locality check before expensive inference
    # --------------------------------------------------------------

    audit_reencode_locality(
        valid,
        manifest,
        region_boxes,
        policy_cache,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint_sha = (
        load_model(
            device
        )
    )

    print(
        "\nCounterfactual evaluation:"
        f"\n  device:       {device}"
        f"\n  checkpoint:   {checkpoint_sha}"
        "\n  source:"
        "\n    bona-fide -> attack JPEG parameters -> decode"
        "\n  restoration:"
        "\n    identity coordinates"
        "\n    annotation ∩ decoded-image bounds"
        "\n  model preprocessing:"
        "\n    frozen Policy C"
    )

    rows = []

    total_clipped_boxes = 0

    valid = valid.sort_values(
        [
            "split",
            "variant",
            "file_stem",
            "hardware_source",
        ]
    )

    for i, pair in enumerate(
        valid.itertuples(index=False),
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

        attack_rgb = (
            jpeg_decode_rgb(
                attack_raw
            )
        )

        bona_rgb = (
            jpeg_decode_rgb(
                bona_raw
            )
        )

        if (
            attack_rgb.shape
            != bona_rgb.shape
        ):
            raise RuntimeError(
                "Dimension mismatch reached "
                f"restoration stage: {attack_path}"
            )

        attack_qtables, attack_sampling = (
            jpeg_parameters(
                attack_raw
            )
        )

        source_rgb = (
            recompress_bonafide_like_attack(
                bona_rgb,
                attack_qtables,
                attack_sampling,
            )
        )

        if (
            source_rgb.shape
            != attack_rgb.shape
        ):
            raise RuntimeError(
                "Attack-matched restoration source "
                "changed image geometry"
            )

        height, width = (
            attack_rgb.shape[:2]
        )

        annotations = (
            region_boxes[
                attack_path
            ]
        )

        face_mask, face_clipped = (
            mask_from_boxes(
                height,
                width,
                annotations["face"],
            )
        )

        text_mask, text_clipped = (
            mask_from_boxes(
                height,
                width,
                annotations["text"],
            )
        )

        clipped_boxes = (
            face_clipped
            + text_clipped
        )

        total_clipped_boxes += len(
            clipped_boxes
        )

        both_mask = (
            face_mask
            | text_mask
        )

        overlap_fraction = float(
            (
                face_mask
                & text_mask
            ).mean()
        )

        # ----------------------------------------------------------
        # Original attack:
        # recompute exact Policy C in-process.
        # ----------------------------------------------------------

        original_policy_bytes = (
            policy_c_bytes(
                attack_rgb,
                pair.file_stem,
                pair.hardware_source,
            )
        )

        expected_cache_sha = (
            policy_cache.loc[
                attack_path,
                "cache_sha256",
            ]
        )

        actual_cache_sha = (
            sha256_bytes(
                original_policy_bytes
            )
        )

        if (
            actual_cache_sha
            != expected_cache_sha
        ):
            raise RuntimeError(
                "Policy-C bytes differ from "
                f"frozen cache: {attack_path}"
            )

        tensors = [
            tensor_from_policy_bytes(
                original_policy_bytes
            )
        ]

        # ----------------------------------------------------------
        # Three counterfactual modes
        # ----------------------------------------------------------

        restored_fractions = {}

        masks = {
            "face_restored":
                face_mask,

            "text_restored":
                text_mask,

            "both_restored":
                both_mask,
        }

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

            encoded = (
                policy_c_bytes(
                    restored,
                    pair.file_stem,
                    pair.hardware_source,
                )
            )

            tensors.append(
                tensor_from_policy_bytes(
                    encoded
                )
            )

        # Score original + three counterfactuals
        # in exactly the same model call.
        scores = score_batch(
            model,
            tensors,
            device,
        )

        original_score = float(
            scores[0]
        )

        restored_scores = (
            scores[1:]
        )

        cached_original_score = float(
            frozen_attack.loc[
                attack_path,
                "attack_probability",
            ]
        )

        pipeline_error = abs(
            original_score
            - cached_original_score
        )

        if pipeline_error > 1e-3:
            raise RuntimeError(
                "In-process original differs from "
                "frozen prediction: "
                f"{attack_path}, "
                f"error={pipeline_error:.8f}"
            )

        # Matched bona-fide floor.
        bona_key = (
            pair.split,
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
                "Missing matched bona-fide score: "
                f"{bona_key}"
            ) from exc

        common = {
            "split":
                pair.split,

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
                cached_original_score,

            "inprocess_original_probability":
                original_score,

            "original_pipeline_abs_error":
                pipeline_error,

            "face_box_count":
                len(
                    annotations["face"]
                ),

            "text_box_count":
                len(
                    annotations["text"]
                ),

            "face_fraction":
                float(
                    face_mask.mean()
                ),

            "text_fraction":
                float(
                    text_mask.mean()
                ),

            "both_fraction":
                float(
                    both_mask.mean()
                ),

            "face_text_overlap_fraction":
                overlap_fraction,

            "clipped_box_count":
                len(
                    clipped_boxes
                ),

            "has_clipped_box":
                bool(
                    clipped_boxes
                ),
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
            restored_scores,
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

    # -----------------------------------------------------------------
    # CRITICAL: save raw inference BEFORE summarisation.
    # -----------------------------------------------------------------

    predictions = pd.DataFrame(
        rows
    )

    OUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    max_pipeline_error = float(
        predictions[
            "original_pipeline_abs_error"
        ].max()
    )

    originals = predictions[
        predictions["mode"]
        == "original"
    ]

    observed_clipped_boxes = int(
        originals[
            "clipped_box_count"
        ].sum()
    )

    observed_clipped_images = int(
        originals[
            "has_clipped_box"
        ].sum()
    )

    print(
        "\nRaw counterfactual predictions saved:"
        f"\n  {OUT_PREDICTIONS}"
        f"\nMaximum original pipeline error: "
        f"{max_pipeline_error:.8g}"
        f"\nClipped altered boxes: "
        f"{observed_clipped_boxes}"
        f"\nImages with clipped boxes: "
        f"{observed_clipped_images}"
    )

    # We independently audited 84 such frozen annotations.
    if observed_clipped_boxes != 84:
        raise RuntimeError(
            "Expected 84 clipped altered boxes "
            f"from prior audit, got "
            f"{observed_clipped_boxes}"
        )

    # -----------------------------------------------------------------
    # Split-aware paired summary.
    # -----------------------------------------------------------------

    predictions_with_deltas, summary = (
        build_summary(
            predictions
        )
    )

    # Save enriched raw data too.
    predictions_with_deltas.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    display = [
        "split",
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
        "\nDEV counterfactual evidence:"
    )

    print(
        summary[
            summary["split"]
            == "dev_val"
        ][display]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nTRAIN counterfactual comparison:"
    )

    print(
        summary[
            summary["split"]
            == "project_train"
        ][display]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nArea diagnostics "
        "(one row per original attack):"
    )

    area = (
        originals[
            [
                "variant",
                "face_fraction",
                "text_fraction",
                "both_fraction",
                "face_text_overlap_fraction",
            ]
        ]
        .groupby(
            "variant"
        )
        .agg(
            ["mean", "median", "max"]
        )
    )

    print(
        area.to_string(
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        f"\nSummary: {OUT_SUMMARY}"
        "\n\nPrimary scientific evidence is "
        "DEV only."
        "\nTrain is retained only as a "
        "memorisation/control comparison."
        "\nThese restored images are "
        "counterfactual composites, not "
        "naturally generated face-only or "
        "text-only attacks."
    )


if __name__ == "__main__":
    main()