#!/usr/bin/env python3
"""
Stage 40 — Phase-A reporting / representativeness audit.

Does not change any scientific definition.

Produces:
1. Phase-A 966 vs frozen 1330 composition audit.
2. Phase-A vs missing-364 numeric balance diagnostics.
3. Continuous localisation degradation summaries.
4. Degradation-threshold rates.
5. DCEC coverage by variant/hardware/split.
6. Provenance summary + SHA256 manifest.

These outputs are descriptive. They do not declare the
cost-balanced Phase-A subset to be a random sample.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ATTACK_ROOT = (
    ROOT
    / "output/LABPC/"
      "trufor_adversarial_localisation_attack"
)

ANALYSIS_ROOT = (
    ROOT
    / "output/LABPC/"
      "trufor_adversarial_localisation_analysis/"
      "phaseA_966"
)

FULL_PATH = (
    ATTACK_ROOT
    / "full_population/full_population_selection.csv"
)

EVIDENCE_PATH = (
    ANALYSIS_ROOT
    / "39_phaseA_evidence_per_image.csv"
)


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
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def descriptive(x: pd.Series) -> dict:
    x = pd.to_numeric(
        x,
        errors="coerce",
    ).dropna()

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "q05": float(x.quantile(0.05)),
        "q25": float(x.quantile(0.25)),
        "q75": float(x.quantile(0.75)),
        "q95": float(x.quantile(0.95)),
    }


def pooled_smd(
    a: pd.Series,
    b: pd.Series,
) -> float:

    a = pd.to_numeric(
        a,
        errors="coerce",
    ).dropna()

    b = pd.to_numeric(
        b,
        errors="coerce",
    ).dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = float(
        a.var(ddof=1)
    )

    vb = float(
        b.var(ddof=1)
    )

    denom_n = (
        len(a)
        + len(b)
        - 2
    )

    pooled_var = (
        (
            (len(a) - 1) * va
            + (len(b) - 1) * vb
        )
        / denom_n
    )

    if pooled_var <= 0:
        return 0.0

    return float(
        (
            a.mean()
            - b.mean()
        )
        / math.sqrt(
            pooled_var
        )
    )


def first_existing(
    frame: pd.DataFrame,
    names: list[str],
):
    for name in names:
        if name in frame.columns:
            return name

    return None


def main():

    if not FULL_PATH.is_file():
        raise RuntimeError(
            f"Missing full selection: {FULL_PATH}"
        )

    if not EVIDENCE_PATH.is_file():
        raise RuntimeError(
            f"Missing Stage-39 output: {EVIDENCE_PATH}"
        )

    full = pd.read_csv(
        FULL_PATH,
        keep_default_na=False,
    )

    ev = pd.read_csv(
        EVIDENCE_PATH,
        keep_default_na=False,
    )

    if len(full) != 1330:
        raise RuntimeError(
            f"Expected 1330 full rows; found {len(full)}"
        )

    if len(ev) != 966:
        raise RuntimeError(
            f"Expected 966 Phase-A rows; found {len(ev)}"
        )

    if full["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in full selection"
        )

    if ev["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in Stage 39"
        )

    phase_ids = set(
        ev["image_path"].astype(str)
    )

    full["_phaseA"] = (
        full["image_path"]
        .astype(str)
        .isin(phase_ids)
    )

    if int(full["_phaseA"].sum()) != 966:
        raise RuntimeError(
            "Phase-A membership mismatch"
        )

    missing = full.loc[
        ~full["_phaseA"]
    ].copy()

    if len(missing) != 364:
        raise RuntimeError(
            f"Expected 364 missing; found {len(missing)}"
        )


    print("=" * 72)
    print("TRUFOR STAGE 40 — PHASE-A REPORTING AUDIT")
    print("=" * 72)
    print("full population :", len(full))
    print("Phase A         :", len(ev))
    print("missing         :", len(missing))
    print()


    # -------------------------------------------------------
    # CATEGORICAL COMPOSITION
    # -------------------------------------------------------

    cat_rows = []
    tvd_summary = {}

    cat_cols = [
        "variant",
        "hardware_source",
        "eval_split",
    ]

    for col in cat_cols:

        if col not in full.columns:
            continue

        levels = sorted(
            set(
                full[col]
                .astype(str)
            )
        )

        p_full = {}
        p_phase = {}
        p_missing = {}

        for level in levels:

            nf = int(
                (
                    full[col].astype(str)
                    == level
                ).sum()
            )

            na = int(
                (
                    full.loc[
                        full["_phaseA"],
                        col,
                    ].astype(str)
                    == level
                ).sum()
            )

            nm = int(
                (
                    missing[col].astype(str)
                    == level
                ).sum()
            )

            pf = nf / len(full)
            pa = na / 966
            pm = nm / 364

            p_full[level] = pf
            p_phase[level] = pa
            p_missing[level] = pm

            cat_rows.append(
                {
                    "variable": col,
                    "level": level,
                    "n_full": nf,
                    "prop_full": pf,
                    "n_phaseA": na,
                    "prop_phaseA": pa,
                    "n_missing": nm,
                    "prop_missing": pm,
                    "phaseA_minus_full_pp":
                        100.0 * (pa - pf),
                    "phaseA_minus_missing_pp":
                        100.0 * (pa - pm),
                }
            )

        tvd_full = (
            0.5
            * sum(
                abs(
                    p_phase[x]
                    - p_full[x]
                )
                for x in levels
            )
        )

        tvd_missing = (
            0.5
            * sum(
                abs(
                    p_phase[x]
                    - p_missing[x]
                )
                for x in levels
            )
        )

        tvd_summary[col] = {
            "phaseA_vs_full_TVD":
                float(tvd_full),

            "phaseA_vs_missing_TVD":
                float(tvd_missing),
        }


    categorical = pd.DataFrame(
        cat_rows
    )

    categorical_path = (
        ANALYSIS_ROOT
        / "40_phaseA_categorical_composition.csv"
    )

    categorical.to_csv(
        categorical_path,
        index=False,
    )


    # -------------------------------------------------------
    # NUMERIC BALANCE AGAINST MISSING 364
    # -------------------------------------------------------

    if (
        "native_height" in full.columns
        and "native_width" in full.columns
    ):
        full["_native_pixels"] = (
            pd.to_numeric(
                full["native_height"],
                errors="coerce",
            )
            * pd.to_numeric(
                full["native_width"],
                errors="coerce",
            )
        )


    numeric_specs = {
        "A_union": [
            "A_union_stage4",
            "A_union",
        ],

        "E_clean": [
            "E_union_stage4",
            "E_union",
        ],

        "mu_clean": [
            "mu_union_stage4",
            "mu_union",
            "mu_w",
        ],

        "PG_clean": [
            "PG_union_stage4",
            "PG_union",
            "PG",
        ],

        "native_pixels": [
            "_native_pixels",
            "native_pixels",
        ],
    }


    numeric_rows = []


    for metric, candidates in (
        numeric_specs.items()
    ):

        col = first_existing(
            full,
            candidates,
        )

        if col is None:
            continue

        all_values = pd.to_numeric(
            full[col],
            errors="coerce",
        )

        phase_values = all_values[
            full["_phaseA"]
        ]

        missing_values = all_values[
            ~full["_phaseA"]
        ]

        sf = descriptive(
            all_values
        )

        sa = descriptive(
            phase_values
        )

        sm = descriptive(
            missing_values
        )

        numeric_rows.append(
            {
                "metric": metric,
                "source_column": col,

                **{
                    f"full_{k}": v
                    for k, v
                    in sf.items()
                },

                **{
                    f"phaseA_{k}": v
                    for k, v
                    in sa.items()
                },

                **{
                    f"missing_{k}": v
                    for k, v
                    in sm.items()
                },

                "phaseA_vs_missing_SMD":
                    pooled_smd(
                        phase_values,
                        missing_values,
                    ),
            }
        )


    numeric = pd.DataFrame(
        numeric_rows
    )

    numeric_path = (
        ANALYSIS_ROOT
        / "40_phaseA_numeric_balance.csv"
    )

    numeric.to_csv(
        numeric_path,
        index=False,
    )


    # -------------------------------------------------------
    # CONTINUOUS ATTACK RESULTS
    # -------------------------------------------------------

    metric_cols = [
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


    continuous_rows = []

    groups = [
        ("overall", "all", ev),
    ]

    for col in [
        "variant",
        "hardware_source",
        "eval_split",
    ]:
        if col in ev.columns:
            for value, g in ev.groupby(
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

        for metric in metric_cols:

            if metric not in frame.columns:
                continue

            stats = descriptive(
                frame[metric]
            )

            continuous_rows.append(
                {
                    "group": group,
                    "value": value,
                    "metric": metric,
                    **stats,
                }
            )


    continuous = pd.DataFrame(
        continuous_rows
    )

    continuous_path = (
        ANALYSIS_ROOT
        / "40_phaseA_continuous_metrics.csv"
    )

    continuous.to_csv(
        continuous_path,
        index=False,
    )


    # -------------------------------------------------------
    # DEGRADATION THRESHOLD SUMMARIES
    # -------------------------------------------------------

    dcec = as_bool(
        ev["DCEC"]
    )

    degradation_rows = []

    populations = {
        "all_phaseA": ev,
        "clean_DCEC": ev.loc[
            dcec
        ].copy(),
    }


    thresholds = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]


    for population, frame in (
        populations.items()
    ):

        for threshold in thresholds:

            e = (
                pd.to_numeric(
                    frame[
                        "relative_E_degradation"
                    ],
                    errors="coerce",
                )
                >= threshold
            )

            r = (
                pd.to_numeric(
                    frame[
                        "relative_RRA_degradation"
                    ],
                    errors="coerce",
                )
                >= threshold
            )

            degradation_rows.append(
                {
                    "population":
                        population,

                    "relative_degradation_threshold":
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
                            r.sum()
                        ),

                    "RRA_rate":
                        float(
                            r.mean()
                        ),

                    "both_n":
                        int(
                            (
                                e & r
                            ).sum()
                        ),

                    "both_rate":
                        float(
                            (
                                e & r
                            ).mean()
                        ),
                }
            )


    degradation = pd.DataFrame(
        degradation_rows
    )

    degradation_path = (
        ANALYSIS_ROOT
        / "40_phaseA_degradation_thresholds.csv"
    )

    degradation.to_csv(
        degradation_path,
        index=False,
    )


    # -------------------------------------------------------
    # DCEC / DCEW COVERAGE TABLE
    # -------------------------------------------------------

    coverage_rows = []


    for group_col in [
        "variant",
        "hardware_source",
        "eval_split",
    ]:

        if group_col not in ev.columns:
            continue

        for value, frame in ev.groupby(
            group_col,
            dropna=False,
        ):

            gdcec = as_bool(
                frame[
                    "DCEC"
                ]
            )

            gdcew = as_bool(
                frame[
                    "DCEW"
                ]
            )

            gdcadv = as_bool(
                frame[
                    "DC_adv"
                ]
            )

            coverage_rows.append(
                {
                    "group":
                        group_col,

                    "value":
                        str(value),

                    "n":
                        int(
                            len(frame)
                        ),

                    "classification_preserved_n":
                        int(
                            gdcadv.sum()
                        ),

                    "classification_preserved_rate":
                        float(
                            gdcadv.mean()
                        ),

                    "DCEC_n":
                        int(
                            gdcec.sum()
                        ),

                    "DCEC_rate":
                        float(
                            gdcec.mean()
                        ),

                    "DCEW_n":
                        int(
                            gdcew.sum()
                        ),

                    "DCEW_rate_given_DCEC":
                        (
                            float(
                                gdcew.sum()
                                / gdcec.sum()
                            )
                            if gdcec.sum()
                            else np.nan
                        ),
                }
            )


    coverage = pd.DataFrame(
        coverage_rows
    )

    coverage_path = (
        ANALYSIS_ROOT
        / "40_phaseA_DCEC_DCEW_coverage.csv"
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )


    # -------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------

    summary = {
        "status":
            "PASS",

        "full_population_n":
            1330,

        "phaseA_n":
            966,

        "phaseA_fraction":
            float(
                966 / 1330
            ),

        "missing_n":
            364,

        "classification_preserved":
            int(
                as_bool(
                    ev["DC_adv"]
                ).sum()
            ),

        "primary_DCEC_n":
            int(
                dcec.sum()
            ),

        "primary_DCEW_n":
            int(
                as_bool(
                    ev["DCEW"]
                ).sum()
            ),

        "categorical_total_variation":
            tvd_summary,

        "numeric_metrics_audited":
            list(
                numeric[
                    "metric"
                ]
            )
            if len(numeric)
            else [],

        "phaseA_unique_file_stems":
            (
                int(
                    ev[
                        "file_stem"
                    ].nunique()
                )
                if "file_stem"
                in ev.columns
                else None
            ),

        "full_unique_file_stems":
            (
                int(
                    full[
                        "file_stem"
                    ].nunique()
                )
                if "file_stem"
                in full.columns
                else None
            ),

        "interpretation": {
            "sampling":
                (
                    "Phase A is a deterministic "
                    "cost-balanced shard subset, "
                    "not a random sample."
                ),

            "DCEC_threshold":
                (
                    "0.5/0.5 is the frozen "
                    "study-defined majority "
                    "criterion, not a universal "
                    "literature cutoff."
                ),

            "human_review":
                (
                    "sidecar analysis only; "
                    "not used for threshold "
                    "selection"
                ),
        },
    }


    summary_path = (
        ANALYSIS_ROOT
        / "40_phaseA_reporting_audit_summary.json"
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
        categorical_path,
        numeric_path,
        continuous_path,
        degradation_path,
        coverage_path,
        summary_path,
    ]


    sums_path = (
        ANALYSIS_ROOT
        / "40_SHA256SUMS.txt"
    )

    sums_path.write_text(
        "".join(
            (
                f"{sha256_file(p)}  "
                f"{p.name}\n"
            )
            for p in hash_targets
        )
    )


    print("Phase A coverage :", f"{966/1330:.3%}")
    print(
        "primary DCEC    :",
        int(
            dcec.sum()
        ),
    )
    print(
        "primary DCEW    :",
        int(
            as_bool(
                ev["DCEW"]
            ).sum()
        ),
    )

    print()
    print("CATEGORICAL TOTAL-VARIATION DISTANCES")

    for key, value in tvd_summary.items():
        print(
            key,
            value,
        )

    print()
    print("NUMERIC BALANCE")

    if len(numeric):
        print(
            numeric[
                [
                    "metric",
                    "phaseA_mean",
                    "missing_mean",
                    "phaseA_vs_missing_SMD",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("STAGE 40 PASS")
    print("outputs:", ANALYSIS_ROOT)


if __name__ == "__main__":
    main()
