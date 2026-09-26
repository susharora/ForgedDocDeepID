#!/usr/bin/env python3
"""
Stage 47 — final 1330-image reporting + runtime audit.

Uses:
- Stage 46 final evidence table
- Stage 44 canonical 1330-image master manifest

Produces:
- formal RRA tie-robustness certificate
- continuous clean/adversarial metric summaries
- degradation-rate summaries
- active attack runtime per image
- runtime summaries overall / variant / hardware / split
- portable compute-time estimates

Calendar shard/log duration is deliberately NOT used for scientific
compute-cost reporting because it can contain host sleep, WSL restart,
manual recovery, etc.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "final_1330"
)

EVIDENCE = (
    ANALYSIS
    / "46_final1330_evidence_per_image.csv"
)

MASTER = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseAB_available"
    / "44_available_master_manifest.csv"
)

TAU_E = 0.5
TAU_RRA = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:

    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def describe(series: pd.Series) -> dict:

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return {
        "n":
            int(len(x)),

        "mean":
            float(x.mean()),

        "median":
            float(x.median()),

        "q05":
            float(x.quantile(0.05)),

        "q25":
            float(x.quantile(0.25)),

        "q75":
            float(x.quantile(0.75)),

        "q95":
            float(x.quantile(0.95)),

        "min":
            float(x.min()),

        "max":
            float(x.max()),
    }


def runtime_summary(
    frame: pd.DataFrame,
    group: str,
    value: str,
) -> dict:

    sec = pd.to_numeric(
        frame[
            "active_wall_seconds"
        ],
        errors="coerce",
    )

    if sec.isna().any():
        raise RuntimeError(
            f"Missing runtime in {group}={value}"
        )

    hours = float(
        sec.sum() / 3600.0
    )

    mp = (
        pd.to_numeric(
            frame[
                "native_pixels"
            ],
            errors="coerce",
        )
        / 1e6
    )

    return {
        "group":
            group,

        "value":
            value,

        "n":
            int(len(frame)),

        "total_active_hours":
            hours,

        "mean_minutes_per_image":
            float(
                sec.mean() / 60.0
            ),

        "median_minutes_per_image":
            float(
                sec.median() / 60.0
            ),

        "q05_minutes":
            float(
                sec.quantile(0.05)
                / 60.0
            ),

        "q25_minutes":
            float(
                sec.quantile(0.25)
                / 60.0
            ),

        "q75_minutes":
            float(
                sec.quantile(0.75)
                / 60.0
            ),

        "q95_minutes":
            float(
                sec.quantile(0.95)
                / 60.0
            ),

        "max_minutes":
            float(
                sec.max()
                / 60.0
            ),

        "images_per_active_hour":
            (
                float(
                    len(frame)
                    / hours
                )
                if hours > 0
                else np.nan
            ),

        "mean_megapixels":
            float(
                mp.mean()
            ),
    }


def main():

    for p in [
        EVIDENCE,
        MASTER,
    ]:
        if not p.is_file():
            raise RuntimeError(
                f"Missing required file: {p}"
            )


    df = pd.read_csv(
        EVIDENCE,
        keep_default_na=False,
    )

    master = pd.read_csv(
        MASTER,
        keep_default_na=False,
    )


    if len(df) != 1330:
        raise RuntimeError(
            f"Expected 1330 evidence rows; "
            f"found {len(df)}"
        )

    if len(master) != 1330:
        raise RuntimeError(
            f"Expected 1330 master rows; "
            f"found {len(master)}"
        )

    if df[
        "image_path"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate evidence image_path"
        )

    if master[
        "image_path"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate master image_path"
        )


    # ======================================================
    # TIE ROBUSTNESS
    # ======================================================

    dc_clean = as_bool(
        df[
            "DC_clean"
        ]
    )

    dc_adv = as_bool(
        df[
            "DC_adv"
        ]
    )


    primary_dcec = (
        dc_clean
        & (
            df[
                "E_clean"
            ] >= TAU_E
        )
        & (
            df[
                "RRA_clean"
            ] >= TAU_RRA
        )
    )


    dcec_min = (
        dc_clean
        & (
            df[
                "E_clean"
            ] >= TAU_E
        )
        & (
            df[
                "RRA_clean_tie_min"
            ] >= TAU_RRA
        )
    )


    dcec_max = (
        dc_clean
        & (
            df[
                "E_clean"
            ] >= TAU_E
        )
        & (
            df[
                "RRA_clean_tie_max"
            ] >= TAU_RRA
        )
    )


    primary_dcew = (
        primary_dcec
        & dc_adv
        & (
            (
                df[
                    "E_adv"
                ] < TAU_E
            )
            | (
                df[
                    "RRA_adv"
                ] < TAU_RRA
            )
        )
    )


    dcew_adv_min = (
        primary_dcec
        & dc_adv
        & (
            (
                df[
                    "E_adv"
                ] < TAU_E
            )
            | (
                df[
                    "RRA_adv_tie_min"
                ] < TAU_RRA
            )
        )
    )


    dcew_adv_max = (
        primary_dcec
        & dc_adv
        & (
            (
                df[
                    "E_adv"
                ] < TAU_E
            )
            | (
                df[
                    "RRA_adv_tie_max"
                ] < TAU_RRA
            )
        )
    )


    tie_summary = {
        "primary_DCEC":
            int(
                primary_dcec.sum()
            ),

        "DCEC_tie_min":
            int(
                dcec_min.sum()
            ),

        "DCEC_tie_max":
            int(
                dcec_max.sum()
            ),

        "clean_DCEC_label_changes":
            int(
                (
                    dcec_min
                    != dcec_max
                ).sum()
            ),

        "primary_DCEW":
            int(
                primary_dcew.sum()
            ),

        "DCEW_adv_tie_min":
            int(
                dcew_adv_min.sum()
            ),

        "DCEW_adv_tie_max":
            int(
                dcew_adv_max.sum()
            ),

        "adv_DCEW_label_changes":
            int(
                (
                    dcew_adv_min
                    != dcew_adv_max
                ).sum()
            ),

        "clean_cutoff_ties":
            int(
                as_bool(
                    df[
                        "RRA_clean_cutoff_tie"
                    ]
                ).sum()
            ),

        "adv_cutoff_ties":
            int(
                as_bool(
                    df[
                        "RRA_adv_cutoff_tie"
                    ]
                ).sum()
            ),

        "max_clean_tie_width":
            float(
                pd.to_numeric(
                    df[
                        "RRA_clean_tie_width"
                    ]
                ).max()
            ),

        "max_adv_tie_width":
            float(
                pd.to_numeric(
                    df[
                        "RRA_adv_tie_width"
                    ]
                ).max()
            ),
    }


    if (
        tie_summary[
            "clean_DCEC_label_changes"
        ] != 0
        or tie_summary[
            "adv_DCEW_label_changes"
        ] != 0
    ):
        raise RuntimeError(
            "Primary labels are not robust "
            "to RRA tie resolution"
        )


    # ======================================================
    # CONTINUOUS SCIENTIFIC METRICS
    # ======================================================

    metrics = [
        "A_union",

        "E_clean",
        "E_adv",
        "relative_E_degradation",

        "RRA_clean",
        "RRA_adv",
        "relative_RRA_degradation",

        "mu_clean",
        "mu_adv",

        "RRA_lift_clean",
        "RRA_lift_adv",

        "physical_linf_recomputed",
        "physical_l2_recomputed",
        "physical_rms_recomputed",
    ]


    continuous_rows = []


    groups = [
        (
            "overall",
            "all",
            df,
        )
    ]


    for col in [
        "variant",
        "hardware_source",
        "eval_split",
    ]:

        for value, g in df.groupby(
            col,
            dropna=False,
        ):

            groups.append(
                (
                    col,
                    str(value),
                    g,
                )
            )


    for group, value, frame in groups:

        for metric in metrics:

            stats = describe(
                frame[
                    metric
                ]
            )

            continuous_rows.append(
                {
                    "group":
                        group,

                    "value":
                        value,

                    "metric":
                        metric,

                    **stats,
                }
            )


    continuous = pd.DataFrame(
        continuous_rows
    )


    continuous_path = (
        ANALYSIS
        / "47_final1330_continuous_metrics.csv"
    )


    continuous.to_csv(
        continuous_path,
        index=False,
    )


    # ======================================================
    # DEGRADATION THRESHOLDS
    # ======================================================

    degradation_rows = []


    populations = {
        "all_1330":
            df,

        "clean_DCEC":
            df.loc[
                primary_dcec
            ].copy(),
    }


    for population, frame in (
        populations.items()
    ):

        for threshold in [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]:

            e = (
                pd.to_numeric(
                    frame[
                        "relative_E_degradation"
                    ]
                )
                >= threshold
            )

            rra = (
                pd.to_numeric(
                    frame[
                        "relative_RRA_degradation"
                    ]
                )
                >= threshold
            )


            degradation_rows.append(
                {
                    "population":
                        population,

                    "threshold":
                        threshold,

                    "n":
                        int(
                            len(frame)
                        ),

                    "E_n":
                        int(
                            e.sum()
                        ),

                    "E_rate":
                        float(
                            e.mean()
                        ),

                    "RRA_n":
                        int(
                            rra.sum()
                        ),

                    "RRA_rate":
                        float(
                            rra.mean()
                        ),

                    "both_n":
                        int(
                            (
                                e & rra
                            ).sum()
                        ),

                    "both_rate":
                        float(
                            (
                                e & rra
                            ).mean()
                        ),
                }
            )


    degradation_path = (
        ANALYSIS
        / "47_final1330_degradation_thresholds.csv"
    )


    pd.DataFrame(
        degradation_rows
    ).to_csv(
        degradation_path,
        index=False,
    )


    # ======================================================
    # RUNTIME FROM RESULT.JSON
    # ======================================================

    master_lookup = (
        master[
            [
                "image_path",
                "central_result_json",
            ]
        ]
        .copy()
    )


    merged = df.merge(
        master_lookup,
        on="image_path",
        how="left",
        validate="one_to_one",
    )


    runtime_values = []


    for n, (_, row) in enumerate(
        merged.iterrows(),
        start=1,
    ):

        result_path = Path(
            str(
                row[
                    "central_result_json"
                ]
            )
        )


        if not result_path.is_file():

            raise RuntimeError(
                f"Missing result.json: "
                f"{result_path}"
            )


        result = json.loads(
            result_path.read_text()
        )


        timing = result.get(
            "timing",
            {}
        )


        sec = timing.get(
            "image_wall_seconds_before_artifact_hashing"
        )


        if sec is None:

            raise RuntimeError(
                "Missing image wall timing for "
                f"{row['image_path']}"
            )


        sec = float(
            sec
        )


        if (
            not np.isfinite(sec)
            or sec <= 0
        ):
            raise RuntimeError(
                "Invalid runtime for "
                f"{row['image_path']}: {sec}"
            )


        runtime_values.append(
            sec
        )


        if (
            n % 250 == 0
            or n == len(
                merged
            )
        ):

            print(
                f"timing "
                f"{n}/{len(merged)}"
            )


    merged[
        "active_wall_seconds"
    ] = runtime_values


    merged[
        "active_wall_minutes"
    ] = (
        merged[
            "active_wall_seconds"
        ]
        / 60.0
    )


    merged[
        "megapixels"
    ] = (
        pd.to_numeric(
            merged[
                "native_pixels"
            ]
        )
        / 1e6
    )


    runtime_per_image_path = (
        ANALYSIS
        / "47_final1330_runtime_per_image.csv"
    )


    merged[
        [
            "image_path",
            "file_stem",
            "variant",
            "hardware_source",
            "eval_split",
            "native_height",
            "native_width",
            "native_pixels",
            "megapixels",
            "active_wall_seconds",
            "active_wall_minutes",
            "physical_linf_recomputed",
        ]
    ].to_csv(
        runtime_per_image_path,
        index=False,
    )


    runtime_rows = [
        runtime_summary(
            merged,
            "overall",
            "all",
        )
    ]


    for col in [
        "variant",
        "hardware_source",
        "eval_split",
    ]:

        for value, g in merged.groupby(
            col,
            dropna=False,
        ):

            runtime_rows.append(
                runtime_summary(
                    g,
                    col,
                    str(value),
                )
            )


    runtime_summary_df = (
        pd.DataFrame(
            runtime_rows
        )
    )


    runtime_summary_path = (
        ANALYSIS
        / "47_final1330_runtime_summary.csv"
    )


    runtime_summary_df.to_csv(
        runtime_summary_path,
        index=False,
    )


    total_seconds = float(
        merged[
            "active_wall_seconds"
        ].sum()
    )


    total_hours = (
        total_seconds
        / 3600.0
    )


    spearman_mp_runtime = float(
        merged[
            [
                "megapixels",
                "active_wall_seconds",
            ]
        ].corr(
            method="spearman"
        ).iloc[
            0,
            1
        ]
    )


    runtime_science = {
        "n_images":
            1330,

        "timing_definition":
            (
                "image_wall_seconds_before_artifact_hashing "
                "from per-image result.json"
            ),

        "excludes":
            (
                "external host sleep, WSL restart, "
                "manual recovery and post-result "
                "artifact hashing"
            ),

        "aggregate_active_gpu_hours":
            total_hours,

        "sequential_equivalent_days":
            float(
                total_hours
                / 24.0
            ),

        "mean_minutes_per_image":
            float(
                merged[
                    "active_wall_minutes"
                ].mean()
            ),

        "median_minutes_per_image":
            float(
                merged[
                    "active_wall_minutes"
                ].median()
            ),

        "q25_minutes":
            float(
                merged[
                    "active_wall_minutes"
                ].quantile(
                    0.25
                )
            ),

        "q75_minutes":
            float(
                merged[
                    "active_wall_minutes"
                ].quantile(
                    0.75
                )
            ),

        "q95_minutes":
            float(
                merged[
                    "active_wall_minutes"
                ].quantile(
                    0.95
                )
            ),

        "images_per_active_hour":
            float(
                1330
                / total_hours
            ),

        "ideal_two_gpu_active_hours":
            float(
                total_hours
                / 2.0
            ),

        "ideal_four_gpu_active_hours":
            float(
                total_hours
                / 4.0
            ),

        "spearman_megapixels_vs_active_runtime":
            spearman_mp_runtime,

        "gpu":
            "NVIDIA RTX PRO 5000 Blackwell",
    }


    runtime_json_path = (
        ANALYSIS
        / "47_final1330_runtime_science.json"
    )


    runtime_json_path.write_text(
        json.dumps(
            runtime_science,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    # ======================================================
    # FINAL STAGE SUMMARY
    # ======================================================

    summary = {
        "status":
            "PASS",

        "population":
            1330,

        "classification_preserved":
            int(
                dc_adv.sum()
            ),

        "DCEC":
            int(
                primary_dcec.sum()
            ),

        "DCEC_rate":
            float(
                primary_dcec.mean()
            ),

        "DCEW":
            int(
                primary_dcew.sum()
            ),

        "DCEW_rate_given_DCEC":
            float(
                primary_dcew.sum()
                / primary_dcec.sum()
            ),

        "tie_robustness":
            tie_summary,

        "runtime":
            runtime_science,

        "stage46_input_sha256":
            sha256_file(
                EVIDENCE
            ),

        "stage44_master_sha256":
            sha256_file(
                MASTER
            ),
    }


    summary_path = (
        ANALYSIS
        / "47_final1330_reporting_summary.json"
    )


    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    hash_targets = [
        continuous_path,
        degradation_path,
        runtime_per_image_path,
        runtime_summary_path,
        runtime_json_path,
        summary_path,
    ]


    sums = (
        ANALYSIS
        / "47_SHA256SUMS.txt"
    )


    sums.write_text(
        "".join(
            f"{sha256_file(p)}  {p.name}\n"
            for p in hash_targets
        )
    )


    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 47 FINAL REPORTING RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "population                 :",
        1330,
    )

    print(
        "classification preserved   :",
        int(
            dc_adv.sum()
        ),
    )

    print(
        "DCEC                       :",
        int(
            primary_dcec.sum()
        ),
    )

    print(
        "DCEW                       :",
        int(
            primary_dcew.sum()
        ),
    )

    print(
        "tie label changes clean     :",
        tie_summary[
            "clean_DCEC_label_changes"
        ],
    )

    print(
        "tie label changes adv       :",
        tie_summary[
            "adv_DCEW_label_changes"
        ],
    )

    print()

    print(
        "aggregate active GPU hours  :",
        f"{total_hours:.3f}",
    )

    print(
        "mean min/image              :",
        f"{runtime_science['mean_minutes_per_image']:.3f}",
    )

    print(
        "median min/image            :",
        f"{runtime_science['median_minutes_per_image']:.3f}",
    )

    print(
        "Q25 / Q75 min/image         :",
        (
            f"{runtime_science['q25_minutes']:.3f}"
            " / "
            f"{runtime_science['q75_minutes']:.3f}"
        ),
    )

    print(
        "Q95 min/image               :",
        f"{runtime_science['q95_minutes']:.3f}",
    )

    print(
        "active throughput images/h  :",
        f"{runtime_science['images_per_active_hour']:.3f}",
    )

    print(
        "Spearman MP vs runtime      :",
        f"{spearman_mp_runtime:.4f}",
    )

    print()

    print(
        "STAGE 47 PASS"
    )


if __name__ == "__main__":
    main()
