#!/usr/bin/env python3
"""
Stage 46 — Final-1330 RRA + DC/DCEC/DCEW evaluation.

IMPORTANT:
The evidence-evaluation protocol must be committed before this
script is executed on adversarial results.

Primary:
    tau_E   = 0.50
    tau_RRA = 0.50

These are study-defined majority thresholds, not literature-
mandated RMA/RRA cutoffs.

Outputs:
- per-image clean/adversarial RRA
- cutoff-tie diagnostics
- DC / DCEC / DCEW
- classification flips separately
- continuous E/RRA degradation
- area-normalised localisation context
- threshold-sensitivity grid
- subgroup summaries
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0,
    str(HERE),
)

from trufor_evidence_metrics import (  # noqa: E402
    relevance_rank_accuracy,
)


ANALYSIS_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "final_1330"
)

STAGE38 = (
    ANALYSIS_ROOT
    / "45_final1330_scientific_state_audit.csv"
)

PROTOCOL_PATH = (
    HERE
    / "evidence_evaluation_protocol_v1.json"
)

CLASS_THRESHOLD_PATH = (
    ROOT
    / "output"
    / "trufor_policy_c_frozen_protocol"
    / "stage03_dev_calibration_accuracy"
    / "frozen_threshold.json"
)

EXPECTED_CLASS_THRESHOLD = (
    0.532955974340439
)


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def find_threshold(
    obj,
    expected: float,
) -> float:
    """
    Resolve the already-frozen image-level classification threshold.

    The calibration JSON may legitimately contain other threshold-like
    numeric values (for example 0.5).  Do not infer semantics from a
    recursive uniqueness assumption.  Instead require the artifact to
    contain the threshold fixed by the execution protocol.
    """

    preferred = {
        "threshold",
        "selected_threshold",
        "frozen_threshold",
    }

    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():

                if (
                    str(k).lower()
                    in preferred
                    and isinstance(
                        v,
                        (int, float),
                    )
                ):
                    found.append(
                        float(v)
                    )

                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    unique = []

    for value in found:
        if not any(
            math.isclose(
                value,
                old,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for old in unique
        ):
            unique.append(value)

    matches = [
        value
        for value in unique
        if math.isclose(
            value,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Frozen classification threshold "
            "could not be resolved against the "
            "execution-protocol expectation. "
            f"Expected={expected}, "
            f"candidates={unique}, "
            f"matches={matches}"
        )

    return matches[0]


def safe_rate(
    numerator: int,
    denominator: int,
):
    if denominator == 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


def evaluate_threshold_pair(
    frame: pd.DataFrame,
    tau_E: float,
    tau_RRA: float,
):

    dc_clean = (
        frame[
            "DC_clean"
        ].astype(bool)
    )

    dc_adv = (
        frame[
            "DC_adv"
        ].astype(bool)
    )

    dcec = (
        dc_clean
        & (
            frame[
                "E_clean"
            ]
            >= tau_E
        )
        & (
            frame[
                "RRA_clean"
            ]
            >= tau_RRA
        )
    )

    e_fail = (
        frame[
            "E_adv"
        ]
        < tau_E
    )

    rra_fail = (
        frame[
            "RRA_adv"
        ]
        < tau_RRA
    )

    dcew = (
        dcec
        & dc_adv
        & (
            e_fail
            | rra_fail
        )
    )

    flips = (
        dcec
        & ~dc_adv
    )

    e_only = (
        dcew
        & e_fail
        & ~rra_fail
    )

    rra_only = (
        dcew
        & ~e_fail
        & rra_fail
    )

    both = (
        dcew
        & e_fail
        & rra_fail
    )

    n_dcec = int(
        dcec.sum()
    )

    n_dcew = int(
        dcew.sum()
    )

    n_flips = int(
        flips.sum()
    )

    return {
        "tau_E":
            float(
                tau_E
            ),

        "tau_RRA":
            float(
                tau_RRA
            ),

        "n_total":
            int(
                len(frame)
            ),

        "n_DCEC":
            n_dcec,

        "DCEC_rate_all":
            safe_rate(
                n_dcec,
                len(frame),
            ),

        "n_DCEW":
            n_dcew,

        "DCEW_rate_given_DCEC":
            safe_rate(
                n_dcew,
                n_dcec,
            ),

        "n_classification_flips_given_DCEC":
            n_flips,

        "classification_flip_rate_given_DCEC":
            safe_rate(
                n_flips,
                n_dcec,
            ),

        "n_DCEW_E_only":
            int(
                e_only.sum()
            ),

        "n_DCEW_RRA_only":
            int(
                rra_only.sum()
            ),

        "n_DCEW_both":
            int(
                both.sum()
            ),
    }


def group_summary(
    frame: pd.DataFrame,
    group_name: str,
    group_value: str,
):

    n = len(
        frame
    )

    dcec = frame[
        "DCEC"
    ].astype(bool)

    dcew = frame[
        "DCEW"
    ].astype(bool)

    flips = frame[
        "classification_flip"
    ].astype(bool)

    n_dcec = int(
        dcec.sum()
    )

    n_dcew = int(
        dcew.sum()
    )

    return {
        "group":
            group_name,

        "value":
            group_value,

        "n":
            int(n),

        "DC_adv_rate":
            float(
                frame[
                    "DC_adv"
                ].mean()
            ),

        "DCEC_n":
            n_dcec,

        "DCEC_rate_all":
            safe_rate(
                n_dcec,
                n,
            ),

        "DCEW_n":
            n_dcew,

        "DCEW_rate_given_DCEC":
            safe_rate(
                n_dcew,
                n_dcec,
            ),

        "classification_flip_n":
            int(
                flips.sum()
            ),

        "classification_flip_rate_given_DCEC":
            safe_rate(
                int(
                    flips.sum()
                ),
                n_dcec,
            ),

        "E_clean_mean":
            float(
                frame[
                    "E_clean"
                ].mean()
            ),

        "E_clean_median":
            float(
                frame[
                    "E_clean"
                ].median()
            ),

        "E_adv_mean":
            float(
                frame[
                    "E_adv"
                ].mean()
            ),

        "E_adv_median":
            float(
                frame[
                    "E_adv"
                ].median()
            ),

        "relative_E_degradation_median":
            float(
                frame[
                    "relative_E_degradation"
                ].median()
            ),

        "RRA_clean_mean":
            float(
                frame[
                    "RRA_clean"
                ].mean()
            ),

        "RRA_clean_median":
            float(
                frame[
                    "RRA_clean"
                ].median()
            ),

        "RRA_adv_mean":
            float(
                frame[
                    "RRA_adv"
                ].mean()
            ),

        "RRA_adv_median":
            float(
                frame[
                    "RRA_adv"
                ].median()
            ),

        "delta_RRA_median":
            float(
                frame[
                    "delta_RRA_adv_minus_clean"
                ].median()
            ),

        "A_median":
            float(
                frame[
                    "A_union"
                ].median()
            ),

        "mu_clean_median":
            float(
                frame[
                    "mu_clean"
                ].median()
            ),

        "mu_adv_median":
            float(
                frame[
                    "mu_adv"
                ].median()
            ),

        "RRA_lift_clean_median":
            float(
                frame[
                    "RRA_lift_clean"
                ].median()
            ),

        "RRA_lift_adv_median":
            float(
                frame[
                    "RRA_lift_adv"
                ].median()
            ),
    }


def main():

    if not STAGE38.is_file():
        raise RuntimeError(
            f"Missing Stage-38 CSV: "
            f"{STAGE38}"
        )

    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(
            f"Missing evidence protocol: "
            f"{PROTOCOL_PATH}"
        )

    if not CLASS_THRESHOLD_PATH.is_file():
        raise RuntimeError(
            "Missing frozen classification "
            f"threshold: {CLASS_THRESHOLD_PATH}"
        )


    protocol = json.loads(
        PROTOCOL_PATH.read_text()
    )


    tau_E = float(
        protocol[
            "primary_operational_evidence_thresholds"
        ][
            "tau_E"
        ]
    )

    tau_RRA = float(
        protocol[
            "primary_operational_evidence_thresholds"
        ][
            "tau_RRA"
        ]
    )


    threshold_json = json.loads(
        CLASS_THRESHOLD_PATH.read_text()
    )

    class_threshold = find_threshold(
        threshold_json,
        EXPECTED_CLASS_THRESHOLD,
    )


    if not math.isclose(
        class_threshold,
        EXPECTED_CLASS_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Frozen classification threshold "
            "does not match execution protocol: "
            f"{class_threshold}"
        )


    src = pd.read_csv(
        STAGE38,
        keep_default_na=False,
    )


    if len(src) != 1330:
        raise RuntimeError(
            f"Expected 1330 rows, got "
            f"{len(src)}"
        )


    if src[
        "image_path"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in Stage 38"
        )


    print(
        "=" * 72
    )

    print(
        "TRUFOR STAGE 46 — "
        "FINAL-1330 RRA / DCEC / DCEW"
    )

    print(
        "=" * 72
    )

    print(
        "images              :",
        len(src),
    )

    print(
        "class threshold     :",
        class_threshold,
    )

    print(
        "tau_E               :",
        tau_E,
    )

    print(
        "tau_RRA             :",
        tau_RRA,
    )

    print(
        "protocol SHA256     :",
        sha256_file(
            PROTOCOL_PATH
        ),
    )

    print()


    out_rows = []


    for n, (_, row) in enumerate(
        src.iterrows(),
        start=1,
    ):

        npz_path = Path(
            str(
                row[
                    "central_npz_path"
                ]
            )
        )


        with np.load(
            npz_path,
            allow_pickle=False,
        ) as z:

            clean_map = np.asarray(
                z[
                    "clean_anomaly_map"
                ]
            )

            adv_map = np.asarray(
                z[
                    "adv_anomaly_map"
                ]
            )

            gt = np.asarray(
                z[
                    "altered_union_mask"
                ],
                dtype=bool,
            )


        clean_rra = (
            relevance_rank_accuracy(
                clean_map,
                gt,
            )
        )

        adv_rra = (
            relevance_rank_accuracy(
                adv_map,
                gt,
            )
        )


        A = float(
            row[
                "A_union"
            ]
        )

        E_clean = float(
            row[
                "E_clean_recomputed"
            ]
        )

        E_adv = float(
            row[
                "E_adv_recomputed"
            ]
        )

        clean_score = float(
            row[
                "clean_score"
            ]
        )

        adv_score = float(
            row[
                "adv_score"
            ]
        )


        DC_clean = bool(
            clean_score
            >= class_threshold
        )

        DC_adv = bool(
            adv_score
            >= class_threshold
        )


        DCEC = bool(
            DC_clean
            and E_clean
                >= tau_E
            and clean_rra.rra
                >= tau_RRA
        )


        E_adv_fail = bool(
            E_adv
            < tau_E
        )

        RRA_adv_fail = bool(
            adv_rra.rra
            < tau_RRA
        )


        DCEW = bool(
            DCEC
            and DC_adv
            and (
                E_adv_fail
                or RRA_adv_fail
            )
        )


        classification_flip = bool(
            DCEC
            and not DC_adv
        )


        r = row.to_dict()


        r.update(
            {
                "classification_threshold":
                    class_threshold,

                "tau_E":
                    tau_E,

                "tau_RRA":
                    tau_RRA,

                "DC_clean":
                    DC_clean,

                "DC_adv":
                    DC_adv,

                "E_clean":
                    E_clean,

                "E_adv":
                    E_adv,

                "RRA_clean":
                    clean_rra.rra,

                "RRA_adv":
                    adv_rra.rra,

                "delta_RRA_adv_minus_clean":
                    (
                        adv_rra.rra
                        - clean_rra.rra
                    ),

                "relative_RRA_degradation":
                    (
                        (
                            clean_rra.rra
                            - adv_rra.rra
                        )
                        / clean_rra.rra
                        if clean_rra.rra
                        > 0
                        else np.nan
                    ),

                "RRA_clean_tie_expected":
                    clean_rra.rra_tie_expected,

                "RRA_clean_tie_min":
                    clean_rra.rra_tie_min,

                "RRA_clean_tie_max":
                    clean_rra.rra_tie_max,

                "RRA_clean_tie_width":
                    (
                        clean_rra.rra_tie_max
                        - clean_rra.rra_tie_min
                    ),

                "RRA_clean_cutoff":
                    clean_rra.cutoff,

                "RRA_clean_cutoff_tie":
                    clean_rra.cutoff_tie_crosses_boundary,

                "RRA_adv_tie_expected":
                    adv_rra.rra_tie_expected,

                "RRA_adv_tie_min":
                    adv_rra.rra_tie_min,

                "RRA_adv_tie_max":
                    adv_rra.rra_tie_max,

                "RRA_adv_tie_width":
                    (
                        adv_rra.rra_tie_max
                        - adv_rra.rra_tie_min
                    ),

                "RRA_adv_cutoff":
                    adv_rra.cutoff,

                "RRA_adv_cutoff_tie":
                    adv_rra.cutoff_tie_crosses_boundary,

                "RRA_K":
                    clean_rra.k,

                "RRA_N_valid":
                    clean_rra.n_valid,

                "RRA_lift_clean":
                    (
                        clean_rra.rra / A
                        if A > 0
                        else np.nan
                    ),

                "RRA_lift_adv":
                    (
                        adv_rra.rra / A
                        if A > 0
                        else np.nan
                    ),

                "RRA_excess_over_area_clean":
                    clean_rra.rra - A,

                "RRA_excess_over_area_adv":
                    adv_rra.rra - A,

                "E_excess_over_area_clean":
                    E_clean - A,

                "E_excess_over_area_adv":
                    E_adv - A,

                "DCEC":
                    DCEC,

                "E_adv_below_tau":
                    E_adv_fail,

                "RRA_adv_below_tau":
                    RRA_adv_fail,

                "DCEW":
                    DCEW,

                "classification_flip":
                    classification_flip,

                "DCEW_E_only":
                    bool(
                        DCEW
                        and E_adv_fail
                        and not RRA_adv_fail
                    ),

                "DCEW_RRA_only":
                    bool(
                        DCEW
                        and not E_adv_fail
                        and RRA_adv_fail
                    ),

                "DCEW_both":
                    bool(
                        DCEW
                        and E_adv_fail
                        and RRA_adv_fail
                    ),
            }
        )


        out_rows.append(
            r
        )


        if (
            n % 25 == 0
            or n == len(src)
        ):

            print(
                f"processed "
                f"{n}/{len(src)}"
            )


    out = pd.DataFrame(
        out_rows
    )


    if not out[
        "DC_clean"
    ].all():

        bad = out.loc[
            ~out[
                "DC_clean"
            ],
            [
                "image_path",
                "clean_score",
            ],
        ]

        raise RuntimeError(
            "Phase-A selection contains "
            "clean decision failures:\n"
            + bad.head(
                20
            ).to_string(
                index=False
            )
        )


    # Primary results.
    per_image_path = (
        ANALYSIS_ROOT
        / "46_final1330_evidence_per_image.csv"
    )

    out.to_csv(
        per_image_path,
        index=False,
    )


    primary = (
        evaluate_threshold_pair(
            out,
            tau_E,
            tau_RRA,
        )
    )


    primary.update(
        {
            "classification_threshold":
                class_threshold,

            "protocol_sha256":
                sha256_file(
                    PROTOCOL_PATH
                ),

            "n_DC_clean":
                int(
                    out[
                        "DC_clean"
                    ].sum()
                ),

            "n_DC_adv":
                int(
                    out[
                        "DC_adv"
                    ].sum()
                ),

            "classification_preservation_rate_all":
                float(
                    out[
                        "DC_adv"
                    ].mean()
                ),

            "E_clean_mean":
                float(
                    out[
                        "E_clean"
                    ].mean()
                ),

            "E_clean_median":
                float(
                    out[
                        "E_clean"
                    ].median()
                ),

            "E_adv_mean":
                float(
                    out[
                        "E_adv"
                    ].mean()
                ),

            "E_adv_median":
                float(
                    out[
                        "E_adv"
                    ].median()
                ),

            "relative_E_degradation_mean":
                float(
                    out[
                        "relative_E_degradation"
                    ].mean()
                ),

            "relative_E_degradation_median":
                float(
                    out[
                        "relative_E_degradation"
                    ].median()
                ),

            "RRA_clean_mean":
                float(
                    out[
                        "RRA_clean"
                    ].mean()
                ),

            "RRA_clean_median":
                float(
                    out[
                        "RRA_clean"
                    ].median()
                ),

            "RRA_adv_mean":
                float(
                    out[
                        "RRA_adv"
                    ].mean()
                ),

            "RRA_adv_median":
                float(
                    out[
                        "RRA_adv"
                    ].median()
                ),

            "relative_RRA_degradation_mean":
                float(
                    out[
                        "relative_RRA_degradation"
                    ].mean()
                ),

            "relative_RRA_degradation_median":
                float(
                    out[
                        "relative_RRA_degradation"
                    ].median()
                ),

            "clean_cutoff_ties":
                int(
                    out[
                        "RRA_clean_cutoff_tie"
                    ].sum()
                ),

            "adv_cutoff_ties":
                int(
                    out[
                        "RRA_adv_cutoff_tie"
                    ].sum()
                ),

            "max_clean_tie_width":
                float(
                    out[
                        "RRA_clean_tie_width"
                    ].max()
                ),

            "max_adv_tie_width":
                float(
                    out[
                        "RRA_adv_tie_width"
                    ].max()
                ),

            "max_physical_linf":
                float(
                    out[
                        "physical_linf_recomputed"
                    ].max()
                ),
        }
    )


    summary_json = (
        ANALYSIS_ROOT
        / "46_final1330_primary_summary.json"
    )

    summary_json.write_text(
        json.dumps(
            primary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    pd.DataFrame(
        [
            primary
        ]
    ).to_csv(
        ANALYSIS_ROOT
        / "46_final1330_primary_summary.csv",
        index=False,
    )


    # Sensitivity grid.
    sens = []

    for te in protocol[
        "threshold_sensitivity"
    ][
        "tau_E_values"
    ]:

        for tr in protocol[
            "threshold_sensitivity"
        ][
            "tau_RRA_values"
        ]:

            sens.append(
                evaluate_threshold_pair(
                    out,
                    float(te),
                    float(tr),
                )
            )


    pd.DataFrame(
        sens
    ).to_csv(
        ANALYSIS_ROOT
        / "46_final1330_threshold_sensitivity.csv",
        index=False,
    )


    # Subgroup summaries.
    grouping_specs = [
        (
            "variant",
            "variant",
        ),
        (
            "hardware_source",
            "hardware_source",
        ),
        (
            "eval_split",
            "eval_split",
        ),
    ]


    subgroup_rows = []


    for group_name, column in (
        grouping_specs
    ):

        for value, g in out.groupby(
            column,
            dropna=False,
        ):

            subgroup_rows.append(
                group_summary(
                    g,
                    group_name,
                    str(value),
                )
            )


    subgroup = pd.DataFrame(
        subgroup_rows
    )


    subgroup.to_csv(
        ANALYSIS_ROOT
        / "46_final1330_subgroup_summary.csv",
        index=False,
    )


    tie_diag = out[
        [
            "image_path",
            "variant",
            "hardware_source",
            "A_union",
            "RRA_K",
            "RRA_clean",
            "RRA_clean_cutoff_tie",
            "RRA_clean_tie_min",
            "RRA_clean_tie_max",
            "RRA_clean_tie_width",
            "RRA_adv",
            "RRA_adv_cutoff_tie",
            "RRA_adv_tie_min",
            "RRA_adv_tie_max",
            "RRA_adv_tie_width",
        ]
    ].copy()


    tie_diag.to_csv(
        ANALYSIS_ROOT
        / "46_final1330_RRA_tie_diagnostics.csv",
        index=False,
    )


    protocol_snapshot = {
        "protocol_file":
            str(
                PROTOCOL_PATH.relative_to(
                    ROOT
                )
            ),

        "protocol_sha256":
            sha256_file(
                PROTOCOL_PATH
            ),

        "stage38_input":
            str(
                STAGE38.relative_to(
                    ROOT
                )
            ),

        "stage38_sha256":
            sha256_file(
                STAGE38
            ),

        "classification_threshold_file":
            str(
                CLASS_THRESHOLD_PATH.relative_to(
                    ROOT
                )
            ),

        "classification_threshold_file_sha256":
            sha256_file(
                CLASS_THRESHOLD_PATH
            ),

        "classification_threshold":
            class_threshold,
    }


    snapshot_path = (
        ANALYSIS_ROOT
        / "46_analysis_protocol_snapshot.json"
    )

    snapshot_path.write_text(
        json.dumps(
            protocol_snapshot,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    hash_targets = [
        per_image_path,
        summary_json,
        ANALYSIS_ROOT
        / "46_final1330_primary_summary.csv",
        ANALYSIS_ROOT
        / "46_final1330_threshold_sensitivity.csv",
        ANALYSIS_ROOT
        / "46_final1330_subgroup_summary.csv",
        ANALYSIS_ROOT
        / "46_final1330_RRA_tie_diagnostics.csv",
        snapshot_path,
    ]


    sums = (
        ANALYSIS_ROOT
        / "46_SHA256SUMS.txt"
    )


    sums.write_text(
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
        "STAGE 46 FINAL PRIMARY RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "n total                    :",
        primary[
            "n_total"
        ],
    )

    print(
        "classification preserved   :",
        (
            f"{primary['n_DC_adv']}"
            f"/{primary['n_total']}"
        ),
    )

    print(
        "clean DCEC                 :",
        primary[
            "n_DCEC"
        ],
    )

    print(
        "DCEW                       :",
        primary[
            "n_DCEW"
        ],
    )

    print(
        "DCEW rate | DCEC           :",
        primary[
            "DCEW_rate_given_DCEC"
        ],
    )

    print(
        "classification flips | DCEC:",
        primary[
            "n_classification_flips_given_DCEC"
        ],
    )

    print(
        "E-only DCEW                :",
        primary[
            "n_DCEW_E_only"
        ],
    )

    print(
        "RRA-only DCEW              :",
        primary[
            "n_DCEW_RRA_only"
        ],
    )

    print(
        "both E+RRA DCEW             :",
        primary[
            "n_DCEW_both"
        ],
    )

    print(
        "clean cutoff ties           :",
        primary[
            "clean_cutoff_ties"
        ],
    )

    print(
        "adv cutoff ties             :",
        primary[
            "adv_cutoff_ties"
        ],
    )

    print(
        "max clean tie width         :",
        primary[
            "max_clean_tie_width"
        ],
    )

    print(
        "max adv tie width           :",
        primary[
            "max_adv_tie_width"
        ],
    )

    print()

    print(
        "STAGE 46 PASS"
    )

    print(
        "outputs:",
        ANALYSIS_ROOT,
    )


if __name__ == "__main__":
    main()
