#!/usr/bin/env python3
"""Audit attack↔bonafide registration before identity-coordinate restoration.

Uses only the frozen project-train + dev manifests. No new split/discovery.
Each digital_1/digital_2 attack is paired to the bonafide image with the
same file_stem and hardware_source.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


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

OUT = (
    ROOT
    / "output"
    / "counterfactual_alignment_audit.csv"
)

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

# Registration is estimated on a reduced image only for speed.
# Reported shifts are also converted back to approximate original pixels.
MAX_SIDE = 768


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


def load_manifest(path, split):
    expected_sha, expected_rows = EXPECTED[path]

    actual_sha = sha256_file(path)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Manifest SHA mismatch: {path}\n"
            f"expected {expected_sha}\n"
            f"actual   {actual_sha}"
        )

    df = pd.read_csv(path)

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count for {path}: "
            f"{len(df)}"
        )

    df["split"] = split

    return df


def load_gray_small(image_path, expected_sha):
    path = DATA / image_path

    if not path.is_file():
        raise RuntimeError(
            f"Missing image: {path}"
        )

    data = path.read_bytes()

    actual_sha = sha256_bytes(data)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Image SHA mismatch: {image_path}"
        )

    with Image.open(io.BytesIO(data)) as im:
        if im.format != "JPEG":
            raise RuntimeError(
                f"Non-JPEG input: {image_path}"
            )

        gray = im.convert("L")

        width, height = gray.size

        scale = min(
            1.0,
            MAX_SIDE / max(width, height),
        )

        small_width = max(
            8,
            int(round(width * scale)),
        )

        small_height = max(
            8,
            int(round(height * scale)),
        )

        if (
            small_width != width
            or small_height != height
        ):
            gray = gray.resize(
                (small_width, small_height),
                Image.Resampling.BILINEAR,
            )

        array = np.asarray(
            gray,
            dtype=np.float32,
        )

    return {
        "width": width,
        "height": height,
        "small_width": small_width,
        "small_height": small_height,
        "scale": scale,
        "array": array,
    }


def phase_correlation_shift(a, b):
    """
    Integer phase-correlation shift at the reduced audit resolution.

    Local face/text manipulations occupy only a minority of the document,
    so the unchanged document structure should dominate registration.
    """

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}"
        )

    a = a.astype(
        np.float32,
        copy=False,
    )

    b = b.astype(
        np.float32,
        copy=False,
    )

    a = (
        a - a.mean()
    ) / (
        a.std() + 1e-6
    )

    b = (
        b - b.mean()
    ) / (
        b.std() + 1e-6
    )

    height, width = a.shape

    # Reduce boundary dominance.
    window = np.outer(
        np.hanning(height),
        np.hanning(width),
    ).astype(np.float32)

    fa = np.fft.rfft2(
        a * window
    )

    fb = np.fft.rfft2(
        b * window
    )

    cross_power = (
        fa * np.conj(fb)
    )

    magnitude = np.abs(
        cross_power
    )

    cross_power /= np.maximum(
        magnitude,
        1e-12,
    )

    correlation = np.fft.irfft2(
        cross_power,
        s=a.shape,
    )

    peak_y, peak_x = np.unravel_index(
        np.argmax(
            np.abs(correlation)
        ),
        correlation.shape,
    )

    dy = int(peak_y)
    dx = int(peak_x)

    if dy > height // 2:
        dy -= height

    if dx > width // 2:
        dx -= width

    peak = float(
        np.abs(
            correlation[
                peak_y,
                peak_x,
            ]
        )
    )

    return dy, dx, peak


def image_difference(a, b):
    if a.shape != b.shape:
        return np.nan, np.nan, np.nan

    diff = np.abs(
        a.astype(np.float32)
        - b.astype(np.float32)
    )

    mean_abs = float(
        diff.mean()
    )

    median_abs = float(
        np.median(diff)
    )

    corr = float(
        np.corrcoef(
            a.ravel(),
            b.ravel(),
        )[0, 1]
    )

    return (
        mean_abs,
        median_abs,
        corr,
    )


def build_population():
    train = load_manifest(
        TRAIN,
        "project_train",
    )

    dev = load_manifest(
        DEV,
        "dev_val",
    )

    df = pd.concat(
        [train, dev],
        ignore_index=True,
    )

    counts = (
        df.traffic_type
        .value_counts()
        .to_dict()
    )

    if counts != {
        "attack": 1266,
        "bonafide": 633,
    }:
        raise RuntimeError(
            f"Unexpected training population: {counts}"
        )

    attacks = df[
        df.traffic_type == "attack"
    ].copy()

    bonafides = df[
        df.traffic_type == "bonafide"
    ].copy()

    variants = (
        attacks.variant
        .value_counts()
        .to_dict()
    )

    if variants != {
        "digital_1": 633,
        "digital_2": 633,
    }:
        raise RuntimeError(
            f"Unexpected attack variants: {variants}"
        )

    key = [
        "file_stem",
        "hardware_source",
    ]

    bona_counts = (
        bonafides.groupby(key)
        .size()
    )

    attack_counts = (
        attacks.groupby(key)
        .size()
    )

    if (
        len(bona_counts) != 633
        or not (
            bona_counts == 1
        ).all()
    ):
        raise RuntimeError(
            "Expected exactly one bonafide "
            "per stem+hardware key"
        )

    if (
        len(attack_counts) != 633
        or not (
            attack_counts == 2
        ).all()
    ):
        raise RuntimeError(
            "Expected exactly two attacks "
            "per stem+hardware key"
        )

    if (
        set(bona_counts.index)
        != set(attack_counts.index)
    ):
        raise RuntimeError(
            "Attack/bonafide pairing keys differ"
        )

    bona = (
        bonafides
        .set_index(key)
    )

    return attacks, bona


def main():
    attacks, bona = (
        build_population()
    )

    print(
        "Counterfactual registration audit"
        f"\nattack pairs:       {len(attacks)}"
        f"\nbonafide sources:   {len(bona)}"
        "\nexpected pairing:   "
        "same file_stem + hardware_source"
    )

    rows = []

    # Cache each bonafide once; each source is reused by
    # digital_1 and digital_2.
    bona_cache = {}

    attacks = attacks.sort_values(
        [
            "file_stem",
            "hardware_source",
            "variant",
        ]
    )

    for i, attack in enumerate(
        attacks.itertuples(index=False),
        start=1,
    ):
        key = (
            attack.file_stem,
            attack.hardware_source,
        )

        bona_row = bona.loc[key]

        if attack.split != bona_row["split"]:
            raise RuntimeError(
                "Attack/bonafide split mismatch: "
                f"{key}"
            )

        if key not in bona_cache:
            bona_cache[key] = (
                load_gray_small(
                    bona_row["image_path"],
                    bona_row["image_sha256"],
                )
            )

        genuine = bona_cache[key]

        forged = load_gray_small(
            attack.image_path,
            attack.image_sha256,
        )

        same_dimensions = (
            forged["width"]
            == genuine["width"]
            and forged["height"]
            == genuine["height"]
        )

        if same_dimensions:
            if (
                forged["array"].shape
                != genuine["array"].shape
            ):
                raise RuntimeError(
                    "Reduced image shape mismatch "
                    "despite equal native dimensions"
                )

            dy, dx, peak = (
                phase_correlation_shift(
                    forged["array"],
                    genuine["array"],
                )
            )

            (
                mean_abs,
                median_abs,
                pixel_corr,
            ) = image_difference(
                forged["array"],
                genuine["array"],
            )

            scale = forged["scale"]

            approx_dy_native = (
                dy / scale
                if scale > 0
                else np.nan
            )

            approx_dx_native = (
                dx / scale
                if scale > 0
                else np.nan
            )

        else:
            dy = np.nan
            dx = np.nan
            peak = np.nan
            mean_abs = np.nan
            median_abs = np.nan
            pixel_corr = np.nan
            scale = np.nan
            approx_dy_native = np.nan
            approx_dx_native = np.nan

        rows.append(
            {
                "split":
                    attack.split,
                "file_stem":
                    attack.file_stem,
                "hardware_source":
                    attack.hardware_source,
                "variant":
                    attack.variant,
                "attack_path":
                    attack.image_path,
                "bonafide_path":
                    bona_row["image_path"],
                "width":
                    forged["width"],
                "height":
                    forged["height"],
                "bonafide_width":
                    genuine["width"],
                "bonafide_height":
                    genuine["height"],
                "same_dimensions":
                    same_dimensions,
                "audit_scale":
                    scale,
                "phase_dy_small":
                    dy,
                "phase_dx_small":
                    dx,
                "phase_abs_max_small":
                    (
                        max(abs(dy), abs(dx))
                        if same_dimensions
                        else np.nan
                    ),
                "approx_phase_dy_native":
                    approx_dy_native,
                "approx_phase_dx_native":
                    approx_dx_native,
                "phase_peak":
                    peak,
                "mean_abs_gray_diff":
                    mean_abs,
                "median_abs_gray_diff":
                    median_abs,
                "gray_pixel_correlation":
                    pixel_corr,
            }
        )

        if (
            i % 100 == 0
            or i == len(attacks)
        ):
            print(
                f"processed {i}/{len(attacks)}"
            )

    result = pd.DataFrame(
        rows
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT,
        index=False,
    )

    shape_failures = (
        ~result.same_dimensions
    ).sum()

    aligned_zero = (
        (
            result.phase_dy_small == 0
        )
        & (
            result.phase_dx_small == 0
        )
    ).sum()

    within_one = (
        result.phase_abs_max_small
        <= 1
    ).sum()

    print(
        "\nRegistration summary:"
        f"\n  pairs:                 {len(result)}"
        f"\n  dimension mismatches:  {shape_failures}"
        f"\n  exact (0,0) shift:     "
        f"{aligned_zero}/{len(result)}"
        f"\n  within 1 audit pixel:  "
        f"{within_one}/{len(result)}"
    )

    if shape_failures == 0:
        print(
            "\nMaximum phase shift "
            "at audit resolution:"
        )

        print(
            result[
                [
                    "phase_dy_small",
                    "phase_dx_small",
                ]
            ]
            .abs()
            .max()
            .to_string()
        )

        print(
            "\nApproximate maximum shift "
            "in native-image pixels:"
        )

        print(
            result[
                [
                    "approx_phase_dy_native",
                    "approx_phase_dx_native",
                ]
            ]
            .abs()
            .max()
            .to_string()
        )

        print(
            "\nImage-difference diagnostics:"
        )

        print(
            result[
                [
                    "mean_abs_gray_diff",
                    "median_abs_gray_diff",
                    "gray_pixel_correlation",
                ]
            ]
            .describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        print(
            "\nExact-zero registration by variant:"
        )

        temp = result.copy()

        temp["zero_shift"] = (
            (
                temp.phase_dy_small == 0
            )
            & (
                temp.phase_dx_small == 0
            )
        )

        print(
            temp.groupby(
                "variant"
            )
            .zero_shift
            .agg(
                ["sum", "count", "mean"]
            )
            .to_string()
        )

        print(
            "\nExact-zero registration by hardware:"
        )

        print(
            temp.groupby(
                "hardware_source"
            )
            .zero_shift
            .agg(
                ["sum", "count", "mean"]
            )
            .to_string()
        )

    nonzero = result[
        (
            result.phase_dy_small != 0
        )
        | (
            result.phase_dx_small != 0
        )
        | (
            ~result.same_dimensions
        )
    ]

    if len(nonzero):
        print(
            "\nLargest/non-zero registration cases:"
        )

        print(
            nonzero.sort_values(
                "phase_abs_max_small",
                ascending=False,
            )[
                [
                    "split",
                    "variant",
                    "hardware_source",
                    "file_stem",
                    "same_dimensions",
                    "phase_dy_small",
                    "phase_dx_small",
                    "approx_phase_dy_native",
                    "approx_phase_dx_native",
                ]
            ]    
            .head(20)
            .to_string(index=False)
        )

    print(
        f"\nWrote {OUT}"
        "\n\nStop here. Do not generate "
        "counterfactual images until this "
        "registration audit is inspected."
    )


if __name__ == "__main__":
    main()