#!/usr/bin/env python3
"""
Stage 41 — document-stem clustered bootstrap for Phase-A TruFor results.

Primary purpose:
- report-ready descriptive statistics
- clustered percentile CIs accounting for repeated observations
  associated with the same underlying document stem

IMPORTANT:
- Phase A is a deterministic 966/1330 completed subset, not a random
  sample of the frozen 1330 population.
- These intervals describe stability under document-stem cluster
  resampling of Phase A; they must not be presented as proving that
  Phase A is representative of the missing 364.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ROOT = (
    ROOT
    / "output/LABPC/"
      "trufor_adversarial_localisation_analysis/"
      "phaseA_966"
)

INPUT = (
    ANALYSIS_ROOT
    / "39_phaseA_evidence_per_image.csv"
)

N_BOOT = 10000
BOOTSTRAP_SEED = 20260926
CI_LOW = 0.025
CI_HIGH = 0.975


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def as_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy(
            dtype=bool
        )

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            ["true", "1", "yes"]
        )
        .to_numpy(
            dtype=bool
        )
    )


def metric_values(
    frame: pd.DataFrame,
) -> dict[str, float]:

    dcec = as_bool(
        frame["DCEC"]
    )

    dcew = as_bool(
        frame["DCEW"]
    )

    dc_adv = as_bool(
        frame["DC_adv"]
    )

    n_dcec = int(
        dcec.sum()
    )

    result = {
        "n":
            float(
                len(frame)
            ),

        "classification_preservation_rate":
            float(
                dc_adv.mean()
            ),

        "DCEC_rate":
            float(
                dcec.mean()
            ),

        "DCEW_rate_given_DCEC":
            (
                float(
                    dcew.sum()
                    / n_dcec
                )
                if n_dcec
                else np.nan
            ),
    }


    continuous = [
        "E_clean",
        "E_adv",
        "relative_E_degradation",

        "RRA_clean",
        "RRA_adv",
        "relative_RRA_degradation",

        "A_union",

        "mu_clean",
        "mu_adv",

        "RRA_lift_clean",
        "RRA_lift_adv",

        "physical_linf_recomputed",
        "physical_l2_recomputed",
    ]


    for col in continuous:

        if col not in frame.columns:
            continue

        x = pd.to_numeric(
            frame[col],
            errors="coerce",
        ).dropna()

        if len(x) == 0:
            result[
                f"{col}_mean"
            ] = np.nan

            result[
                f"{col}_median"
            ] = np.nan

            continue

        result[
            f"{col}_mean"
        ] = float(
            x.mean()
        )

        result[
            f"{col}_median"
        ] = float(
            x.median()
        )


    return result


def cluster_bootstrap(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
]:

    if "file_stem" not in frame.columns:
        raise RuntimeError(
            "file_stem missing; clustered "
            "bootstrap cannot be performed"
        )


    clusters = {
        stem:
            g.index.to_numpy()
        for stem, g in frame.groupby(
            "file_stem",
            sort=False,
        )
    }


    names = list(
        clusters
    )

    if not names:
        raise RuntimeError(
            "No bootstrap clusters"
        )


    point = metric_values(
        frame
    )


    boot_rows = []


    for b in range(
        N_BOOT
    ):

        sampled = rng.choice(
            names,
            size=len(names),
            replace=True,
        )


        indices = np.concatenate(
            [
                clusters[name]
                for name in sampled
            ]
        )


        sample = frame.loc[
            indices
        ]


        values = metric_values(
            sample
        )

        values[
            "bootstrap_iteration"
        ] = b

        boot_rows.append(
            values
        )


        if (
            (b + 1) % 1000 == 0
            or b + 1 == N_BOOT
        ):
            print(
                f"  bootstrap "
                f"{b+1}/{N_BOOT}"
            )


    return (
        point,
        pd.DataFrame(
            boot_rows
        ),
    )


def summarise_bootstrap(
    point: dict[str, float],
    boot: pd.DataFrame,
    group: str,
    value: str,
    n_clusters: int,
):

    rows = []


    for metric, estimate in (
        point.items()
    ):

        if metric == "n":
            continue


        x = pd.to_numeric(
            boot[metric],
            errors="coerce",
        ).dropna()


        if len(x) == 0:
            low = np.nan
            high = np.nan
            boot_sd = np.nan
            n_valid = 0

        else:
            low = float(
                x.quantile(
                    CI_LOW
                )
            )

            high = float(
                x.quantile(
                    CI_HIGH
                )
            )

            boot_sd = float(
                x.std(
                    ddof=1
                )
            )

            n_valid = int(
                len(x)
            )


        rows.append(
            {
                "group":
                    group,

                "value":
                    value,

                "metric":
                    metric,

                "estimate":
                    estimate,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "bootstrap_sd":
                    boot_sd,

                "n_rows":
                    int(
                        point["n"]
                    ),

                "n_clusters":
                    n_clusters,

                "bootstrap_valid":
                    n_valid,

                "bootstrap_replicates":
                    N_BOOT,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,

                "method":
                    (
                        "document-stem "
                        "cluster percentile "
                        "bootstrap"
                    ),
            }
        )


    return rows


def run_group(
    frame: pd.DataFrame,
    group: str,
    value: str,
    rng: np.random.Generator,
):

    n_clusters = int(
        frame[
            "file_stem"
        ].nunique()
    )


    print()
    print(
        f"[{group}={value}] "
        f"n={len(frame)}, "
        f"clusters={n_clusters}"
    )


    point, boot = cluster_bootstrap(
        frame,
        rng,
    )


    return summarise_bootstrap(
        point,
        boot,
        group,
        value,
        n_clusters,
    )


def main():

    if not INPUT.is_file():
        raise RuntimeError(
            f"Missing Stage-39 input: "
            f"{INPUT}"
        )


    df = pd.read_csv(
        INPUT,
        keep_default_na=False,
    )


    if len(df) != 966:
        raise RuntimeError(
            f"Expected 966 rows; "
            f"found {len(df)}"
        )


    if df[
        "image_path"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path"
        )


    if (
        df[
            "file_stem"
        ]
        .astype(str)
        .str.len()
        .eq(0)
        .any()
    ):
        raise RuntimeError(
            "Blank file_stem"
        )


    print(
        "=" * 72
    )

    print(
        "TRUFOR STAGE 41 — "
        "PHASE-A CLUSTER BOOTSTRAP"
    )

    print(
        "=" * 72
    )

    print(
        "images     :",
        len(df),
    )

    print(
        "stems      :",
        df[
            "file_stem"
        ].nunique(),
    )

    print(
        "replicates :",
        N_BOOT,
    )

    print(
        "seed       :",
        BOOTSTRAP_SEED,
    )


    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )


    rows = []


    rows.extend(
        run_group(
            df,
            "overall",
            "all",
            rng,
        )
    )


    for column in [
        "variant",
        "hardware_source",
        "eval_split",
    ]:

        for value, group_df in df.groupby(
            column,
            dropna=False,
        ):

            rows.extend(
                run_group(
                    group_df,
                    column,
                    str(value),
                    rng,
                )
            )


    result = pd.DataFrame(
        rows
    )


    out_csv = (
        ANALYSIS_ROOT
        / "41_phaseA_cluster_bootstrap_CI.csv"
    )


    result.to_csv(
        out_csv,
        index=False,
    )


    # Compact primary report table.
    primary_metrics = [
        "classification_preservation_rate",
        "DCEC_rate",
        "DCEW_rate_given_DCEC",

        "E_clean_mean",
        "E_adv_mean",
        "relative_E_degradation_median",

        "RRA_clean_mean",
        "RRA_adv_mean",
        "relative_RRA_degradation_median",

        "mu_clean_median",
        "mu_adv_median",

        "RRA_lift_clean_median",
        "RRA_lift_adv_median",

        "physical_linf_recomputed_mean",
    ]


    primary = result.loc[
        (
            result[
                "group"
            ]
            == "overall"
        )
        & result[
            "metric"
        ].isin(
            primary_metrics
        )
    ].copy()


    primary_out = (
        ANALYSIS_ROOT
        / "41_phaseA_primary_report_table.csv"
    )


    primary.to_csv(
        primary_out,
        index=False,
    )


    summary = {
        "status":
            "PASS",

        "phaseA_n":
            966,

        "unique_file_stems":
            int(
                df[
                    "file_stem"
                ].nunique()
            ),

        "bootstrap_replicates":
            N_BOOT,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "ci":
            [
                CI_LOW,
                CI_HIGH,
            ],

        "method":
            (
                "document-stem cluster "
                "percentile bootstrap"
            ),

        "interpretation_warning":
            (
                "Phase A is a deterministic "
                "cost-balanced completed subset. "
                "Cluster-bootstrap intervals "
                "quantify resampling stability "
                "within Phase A and do not establish "
                "representativeness of the missing "
                "364 images."
            ),

        "stage39_input_sha256":
            sha256_file(
                INPUT
            ),
    }


    summary_path = (
        ANALYSIS_ROOT
        / "41_phaseA_cluster_bootstrap_summary.json"
    )


    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    sums_path = (
        ANALYSIS_ROOT
        / "41_SHA256SUMS.txt"
    )


    hash_targets = [
        out_csv,
        primary_out,
        summary_path,
    ]


    sums_path.write_text(
        "".join(
            (
                f"{sha256_file(p)}  "
                f"{p.name}\n"
            )
            for p in hash_targets
        )
    )


    print()
    print(
        "=" * 72
    )

    print(
        "OVERALL PRIMARY RESULTS"
    )

    print(
        "=" * 72
    )


    show = primary[
        [
            "metric",
            "estimate",
            "ci95_low",
            "ci95_high",
        ]
    ]


    print(
        show.to_string(
            index=False
        )
    )


    print()
    print(
        "STAGE 41 PASS"
    )

    print(
        "outputs:",
        ANALYSIS_ROOT,
    )


if __name__ == "__main__":
    main()
