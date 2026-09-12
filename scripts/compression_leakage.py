#!/usr/bin/env python3
"""Model-free JPEG leakage audit on the frozen FantasyID train/dev manifests."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "FantasyID"

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

OUT = ROOT / "output/compression_leakage.csv"

EXPECTED = {
    TRAIN: (
        "6307b1516f0e8a6db661077fedad704bbd13f6242c9795ca11310ac86d650cc8",
        1440,
        960,
        480,
    ),
    DEV: (
        "46953b0e474fb52be7a94c2555d0453a57123d8412bd2e21780986155a9e250d",
        459,
        306,
        153,
    ),
}

FEATURES = [
    "jpeg_quality",
    "q_luma_mean",
    "q_chroma_mean",
    "sampling",
    "lattice_q75_mean",
    "lattice_q75_p90",
    "lattice_current_mean",
    "lattice_current_p90",
    "blockiness_h",
    "blockiness_v",
    "recompress_current_mae",
]


def file_sha(path):
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


def qtables(jpeg_bytes):
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        q = im.quantization

        return (
            np.asarray(q[0]).reshape(8, 8),
            np.asarray(q[1]).reshape(8, 8),
        )


# Generate the exact standard JPEG tables used by the local Pillow/libjpeg.
_DUMMY = Image.new("RGB", (8, 8), (128, 128, 128))

STANDARD = {
    q: qtables(encode_jpeg(_DUMMY, q))
    for q in range(1, 101)
}


def exact_quality(q0, q1):
    for quality, (s0, s1) in STANDARD.items():
        if np.array_equal(q0, s0) and np.array_equal(q1, s1):
            return quality

    return -1


def validate_manifests():
    frames = []

    for path, (sha, n, n_attack, n_bonafide) in EXPECTED.items():

        if file_sha(path) != sha:
            raise RuntimeError(
                f"Manifest SHA mismatch: {path}"
            )

        df = pd.read_csv(path)

        counts = df.traffic_type.value_counts()

        if (
            len(df) != n
            or counts.get("attack") != n_attack
            or counts.get("bonafide") != n_bonafide
        ):
            raise RuntimeError(
                f"Unexpected manifest counts: {path}"
            )

        if df.image_path.duplicated().any():
            raise RuntimeError(
                f"Duplicate image_path: {path}"
            )

        frames.append(df)

    train, dev = frames

    overlap = set(train.file_stem) & set(dev.file_stem)

    if overlap:
        raise RuntimeError(
            f"Train/dev identity leakage: "
            f"{len(overlap)} overlapping stems"
        )

    if not DATA.is_dir():
        raise RuntimeError(
            f"Missing FantasyID root: {DATA}"
        )

    return train, dev


def final_q(row):
    """
    Deterministic pseudo-random Q50-90.

    Matching attack/bonafide versions of the same card + hardware
    receive the same final quality. This prevents final-Q assignment
    itself becoming a label cue in the leakage audit.
    """

    key = (
        f"{row.file_stem}|{row.hardware_source}"
    ).encode()

    n = int.from_bytes(
        hashlib.sha256(key).digest()[:8],
        "big",
    )

    return 50 + n % 41


def sampled_dct(y, grid=24):
    """
    Sample aligned 8x8 blocks throughout the original image.

    No resizing is performed because that would destroy the JPEG grid.
    """

    h_blocks = y.shape[0] // 8
    w_blocks = y.shape[1] // 8

    rows = np.linspace(
        0,
        h_blocks - 1,
        min(grid, h_blocks),
        dtype=int,
    )

    cols = np.linspace(
        0,
        w_blocks - 1,
        min(grid, w_blocks),
        dtype=int,
    )

    blocks = np.stack(
        [
            y[
                r * 8:(r + 1) * 8,
                c * 8:(c + 1) * 8,
            ]
            for r in rows
            for c in cols
        ]
    ).astype(np.float32)

    blocks -= 128.0

    return dctn(
        blocks,
        axes=(1, 2),
        norm="ortho",
    )


def lattice(coeff, table):
    """
    Distance of sampled DCT coefficients from a JPEG quantisation lattice.
    """

    residual = np.abs(
        coeff / table[None]
        - np.rint(coeff / table[None])
    )

    # Flatten the 8x8 coefficients and remove DC only.
    residual = (
        residual
        .reshape(len(residual), -1)[:, 1:]
        .ravel()
    )

    return (
        float(residual.mean()),
        float(np.quantile(residual, 0.90)),
    )


def blockiness(y):
    """
    Difference between changes across 8-pixel JPEG boundaries and
    immediately adjacent non-boundary changes.
    """

    xs = np.arange(8, y.shape[1], 8)
    ys = np.arange(8, y.shape[0], 8)

    horizontal = (
        np.abs(y[:, xs] - y[:, xs - 1]).mean()
        - np.abs(y[:, xs - 1] - y[:, xs - 2]).mean()
    )

    vertical = (
        np.abs(y[ys, :] - y[ys - 1, :]).mean()
        - np.abs(y[ys - 1, :] - y[ys - 2, :]).mean()
    )

    return float(horizontal), float(vertical)


def recompress_mae(rgb, quality):
    """
    Recompression stability on an MCU-aligned centre patch.

    The 512x512 crop makes this cheap while preserving JPEG grid phase.
    """

    h, w = rgb.shape[:2]

    size = min(
        512,
        (h // 16) * 16,
        (w // 16) * 16,
    )

    y0 = ((h - size) // 2 // 16) * 16
    x0 = ((w - size) // 2 // 16) * 16

    patch = rgb[
        y0:y0 + size,
        x0:x0 + size,
    ]

    encoded = encode_jpeg(
        Image.fromarray(patch),
        quality,
    )

    with Image.open(io.BytesIO(encoded)) as im:
        other = np.asarray(
            im.convert("RGB"),
            dtype=np.float32,
        )

    return float(
        np.abs(
            patch.astype(np.float32)
            - other
        ).mean()
    )


def compression_features(jpeg_bytes):

    with Image.open(io.BytesIO(jpeg_bytes)) as im:

        if im.format != "JPEG":
            raise RuntimeError(
                "Non-JPEG input encountered"
            )

        q = im.quantization

        if 0 not in q or 1 not in q:
            raise RuntimeError(
                "Missing JPEG quantisation tables"
            )

        q0 = np.asarray(q[0]).reshape(8, 8)
        q1 = np.asarray(q[1]).reshape(8, 8)

        sampling = int(
            JpegImagePlugin.get_sampling(im)
        )

        rgb = np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )

    quality = exact_quality(q0, q1)

    if quality < 0:
        raise RuntimeError(
            "Non-standard JPEG quantisation table encountered"
        )

    y = np.asarray(
        Image.fromarray(rgb).convert("YCbCr"),
        dtype=np.float32,
    )[:, :, 0]

    coeff = sampled_dct(y)

    q75_mean, q75_p90 = lattice(
        coeff,
        STANDARD[75][0],
    )

    current_mean, current_p90 = lattice(
        coeff,
        q0,
    )

    block_h, block_v = blockiness(y)

    return {
        "jpeg_quality": quality,
        "q_luma_mean": float(q0.mean()),
        "q_chroma_mean": float(q1.mean()),
        "sampling": sampling,

        "lattice_q75_mean": q75_mean,
        "lattice_q75_p90": q75_p90,

        "lattice_current_mean": current_mean,
        "lattice_current_p90": current_p90,

        "blockiness_h": block_h,
        "blockiness_v": block_v,

        "recompress_current_mae": recompress_mae(
            rgb,
            quality,
        ),
    }


def process(row, split):

    path = DATA / row.image_path

    raw = path.read_bytes()

    actual_sha = hashlib.sha256(raw).hexdigest()

    if actual_sha != row.image_sha256:
        raise RuntimeError(
            f"Image SHA mismatch: {row.image_path}"
        )

    # Policy A
    raw_bytes = raw

    # Policy B:
    # raw -> common JPEG Q75
    with Image.open(io.BytesIO(raw)) as im:
        raw_rgb = im.convert("RGB")

    q75_bytes = encode_jpeg(
        raw_rgb,
        75,
    )

    # Policy C:
    # raw -> common Q75 -> deterministic paired Q50-90
    with Image.open(io.BytesIO(q75_bytes)) as im:
        q75_rgb = im.convert("RGB")

    q = final_q(row)

    random_final_bytes = encode_jpeg(
        q75_rgb,
        q,
    )

    common = {
        "split": split,
        "image_path": row.image_path,
        "file_stem": row.file_stem,
        "traffic_type": row.traffic_type,

        # Frozen polarity:
        # bonafide=0, attack=1
        "label": int(
            row.traffic_type == "attack"
        ),

        "variant": (
            ""
            if pd.isna(row.variant)
            else row.variant
        ),

        "hardware_source": row.hardware_source,
    }

    outputs = []

    policies = [
        (
            "A_raw",
            raw_bytes,
            np.nan,
        ),
        (
            "B_common_q75",
            q75_bytes,
            75,
        ),
        (
            "C_q75_then_q50_90",
            random_final_bytes,
            q,
        ),
    ]

    for policy, data, assigned_q in policies:

        outputs.append(
            {
                **common,
                "policy": policy,
                "assigned_q": assigned_q,
                **compression_features(data),
            }
        )

    return outputs


def check_interventions(df):

    raw = df[
        df.policy == "A_raw"
    ]

    print("\nRaw JPEG evidence:")

    print(
        pd.crosstab(
            raw.traffic_type,
            raw.jpeg_quality,
        ).to_string()
    )

    print(
        "sampling:",
        sorted(raw.sampling.unique()),
    )

    attack_q = set(
        raw.loc[
            raw.label == 1,
            "jpeg_quality",
        ]
    )

    bonafide_q = set(
        raw.loc[
            raw.label == 0,
            "jpeg_quality",
        ]
    )

    if (
        attack_q != {75}
        or bonafide_q != {95}
        or set(raw.sampling) != {2}
    ):
        raise RuntimeError(
            "Raw JPEG confound does not match "
            "the frozen B_viability finding"
        )

    policy_b = df[
        df.policy == "B_common_q75"
    ]

    if (
        set(policy_b.jpeg_quality) != {75}
        or set(policy_b.sampling) != {2}
    ):
        raise RuntimeError(
            "Policy B is not uniformly "
            "Q75 / 4:2:0"
        )

    policy_c = df[
        df.policy == "C_q75_then_q50_90"
    ]

    if (
        not policy_c.jpeg_quality
        .between(50, 90)
        .all()
        or set(policy_c.sampling) != {2}
    ):
        raise RuntimeError(
            "Policy C is not Q50-90 / 4:2:0"
        )

    encoded_q = (
        policy_c.jpeg_quality
        .to_numpy()
    )

    assigned_q = (
        policy_c.assigned_q
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        encoded_q,
        assigned_q,
    ):
        raise RuntimeError(
            "Policy C assigned Q does not "
            "match encoded Q"
        )


def evaluate(df):

    df = df.copy()

    df["model_score"] = np.nan

    print(
        "\nCompression-only leakage model:"
    )

    policies = [
        "A_raw",
        "B_common_q75",
        "C_q75_then_q50_90",
    ]

    for policy in policies:

        train = df[
            (df.policy == policy)
            & (df.split == "project_train")
        ]

        dev = df[
            (df.policy == policy)
            & (df.split == "dev_val")
        ]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=10,
            ),
        )

        model.fit(
            train[FEATURES],
            train.label,
        )

        train_score = model.predict_proba(
            train[FEATURES]
        )[:, 1]

        dev_score = model.predict_proba(
            dev[FEATURES]
        )[:, 1]

        df.loc[
            train.index,
            "model_score",
        ] = train_score

        df.loc[
            dev.index,
            "model_score",
        ] = dev_score

        train_auc = roc_auc_score(
            train.label,
            train_score,
        )

        dev_auc = roc_auc_score(
            dev.label,
            dev_score,
        )

        univariate = []

        for feature in FEATURES:

            if dev[feature].nunique() < 2:
                auc = 0.5

            else:
                auc = roc_auc_score(
                    dev.label,
                    dev[feature],
                )

            # Leakage can have either polarity.
            leakage_auc = max(
                auc,
                1.0 - auc,
            )

            univariate.append(
                (
                    feature,
                    leakage_auc,
                )
            )

        univariate.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        strongest = ", ".join(
            f"{name}={auc:.3f}"
            for name, auc
            in univariate[:4]
        )

        print(
            f"{policy:20s} "
            f"train={train_auc:.4f}  "
            f"dev={dev_auc:.4f}"
        )

        print(
            "  strongest dev features: "
            f"{strongest}"
        )

    return df


def main():

    train, dev = validate_manifests()

    print(
        f"Data root: {DATA}"
    )

    print(
        f"Frozen rows: "
        f"train={len(train)}, "
        f"dev={len(dev)}"
    )

    rows = []

    total = len(train) + len(dev)
    done = 0

    for split, frame in [
        ("project_train", train),
        ("dev_val", dev),
    ]:

        for row in frame.itertuples(
            index=False
        ):

            rows.extend(
                process(
                    row,
                    split,
                )
            )

            done += 1

            if (
                done % 100 == 0
                or done == total
            ):
                print(
                    f"processed "
                    f"{done}/{total}"
                )

    df = pd.DataFrame(rows)

    check_interventions(df)

    df = evaluate(df)

    df.to_csv(
        OUT,
        index=False,
    )

    print(
        f"\nWrote {OUT}"
    )

    print(
        "Stop here: choose B vs C "
        "only after inspecting dev leakage."
    )


if __name__ == "__main__":
    main()