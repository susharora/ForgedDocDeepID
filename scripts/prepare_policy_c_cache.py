#!/usr/bin/env python3
"""Materialise the frozen Policy-C JPEG intervention for train/dev only."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "FantasyID"
CACHE = ROOT / "data" / "processed" / "policy_c"

TRAIN = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_project_train.csv"
)
DEV = (
    ROOT
    / "output/splits/"
    "fantasyid_project_split_2026-09-05_091937_dev_val.csv"
)

OUT = ROOT / "output" / "policy_c_cache_index.csv"

EXPECTED = {
    TRAIN: (
        "6307b1516f0e8a6db661077fedad704bb"
        "d13f6242c9795ca11310ac86d650cc8",
        1440,
    ),
    DEV: (
        "46953b0e474fb52be7a94c2555d0453a"
        "57123d8412bd2e21780986155a9e250d",
        459,
    ),
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def encode_jpeg(rgb, quality):
    buf = io.BytesIO()

    rgb.save(
        buf,
        "JPEG",
        quality=int(quality),
        subsampling=2,       # 4:2:0
        optimize=False,
        progressive=False,
    )

    return buf.getvalue()


def read_jpeg_params(data):
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "JPEG":
            raise RuntimeError("Expected JPEG")

        q = im.quantization

        if 0 not in q or 1 not in q:
            raise RuntimeError(
                "Missing JPEG quantisation tables"
            )

        return (
            np.asarray(q[0]),
            np.asarray(q[1]),
            int(JpegImagePlugin.get_sampling(im)),
        )


# Standard tables produced by the same Pillow/libjpeg installation.
_DUMMY = Image.new(
    "RGB",
    (16, 16),
    (128, 128, 128),
)

STANDARD = {
    q: read_jpeg_params(
        encode_jpeg(_DUMMY, q)
    )[:2]
    for q in range(50, 91)
}


def final_q(row):
    """
    Exact audited Policy-C assignment.

    All attack/bonafide versions sharing card stem + hardware
    receive the same final JPEG quality.
    """
    key = (
        f"{row.file_stem}|{row.hardware_source}"
    ).encode()

    n = int.from_bytes(
        hashlib.sha256(key).digest()[:8],
        "big",
    )

    return 50 + n % 41


def load_manifest(path, split):
    expected_sha, expected_rows = EXPECTED[path]

    actual_sha = sha256_file(path)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Manifest SHA mismatch:\n"
            f"{path}\n"
            f"expected {expected_sha}\n"
            f"actual   {actual_sha}"
        )

    df = pd.read_csv(path)

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: {path}"
        )

    df["split"] = split

    return df


def validate_output(data, quality):
    q0, q1, sampling = read_jpeg_params(data)

    ref0, ref1 = STANDARD[quality]

    if sampling != 2:
        raise RuntimeError(
            f"Expected 4:2:0, got sampling={sampling}"
        )

    if (
        not np.array_equal(q0, ref0)
        or not np.array_equal(q1, ref1)
    ):
        raise RuntimeError(
            f"Output quantisation table "
            f"does not match Q{quality}"
        )


def main():
    train = load_manifest(
        TRAIN,
        "project_train",
    )

    dev = load_manifest(
        DEV,
        "dev_val",
    )

    overlap = (
        set(train.file_stem)
        & set(dev.file_stem)
    )

    if overlap:
        raise RuntimeError(
            f"Train/dev identity leakage: "
            f"{len(overlap)} stems"
        )

    df = pd.concat(
        [train, dev],
        ignore_index=True,
    )

    if not DATA.is_dir():
        raise RuntimeError(
            f"Missing FantasyID root: {DATA}"
        )

    rows = []

    print(
        "Preparing frozen Policy C:"
        "\n  raw"
        "\n  -> JPEG Q75 / 4:2:0"
        "\n  -> deterministic matched Q50-90 / 4:2:0"
        f"\n\nimages: {len(df)}"
    )

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        source = DATA / row.image_path

        raw = source.read_bytes()

        if sha256_bytes(raw) != row.image_sha256:
            raise RuntimeError(
                f"Source SHA mismatch: "
                f"{row.image_path}"
            )

        # First common equalisation stage.
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert("RGB")

        q75 = encode_jpeg(
            rgb,
            75,
        )

        # Decode Q75 result, then apply matched final Q.
        with Image.open(io.BytesIO(q75)) as im:
            q75_rgb = im.convert("RGB")

        quality = final_q(row)

        final = encode_jpeg(
            q75_rgb,
            quality,
        )

        validate_output(
            final,
            quality,
        )

        destination = (
            CACHE
            / row.image_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(final)

        rows.append(
            {
                "split": row.split,
                "image_path": row.image_path,
                "cache_path": str(
                    destination.relative_to(ROOT)
                ),
                "file_stem": row.file_stem,
                "traffic_type": row.traffic_type,
                "variant": (
                    ""
                    if pd.isna(row.variant)
                    else row.variant
                ),
                "hardware_source":
                    row.hardware_source,
                "label": int(
                    row.traffic_type == "attack"
                ),
                "assigned_q": quality,
                "source_sha256":
                    row.image_sha256,
                "cache_sha256":
                    sha256_bytes(final),
            }
        )

        if (
            i % 100 == 0
            or i == len(df)
        ):
            print(
                f"processed {i}/{len(df)}"
            )

    index = pd.DataFrame(rows)

    # Same card + hardware must have exactly one final Q
    # across bonafide/digital_1/digital_2.
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

    if (
        len(index[index.split == "project_train"])
        != 1440
        or
        len(index[index.split == "dev_val"])
        != 459
    ):
        raise RuntimeError(
            "Unexpected cache index counts"
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
        "\nPolicy-C cache complete."
        f"\nindex: {OUT}"
        f"\ncache: {CACHE}"
        "\nFinal-Q matching: PASS"
    )


if __name__ == "__main__":
    main()