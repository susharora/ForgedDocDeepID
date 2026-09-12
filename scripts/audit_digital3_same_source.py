#!/usr/bin/env python3
"""
Same-source Digital-3 score audit.

Question
--------
Does applying the Digital-3 text manipulation raise or lower the model's
attack score relative to the SAME card captured with the SAME hardware?

This is distinct from the published official-test AUROC, where Digital-3
attacks are compared against a different 300-image bona-fide population.

Uses saved predictions only:
    - no image processing
    - no model inference
    - no new split construction
    - no test-set relabeling

Matching key
------------
    file_stem + hardware_source

Because Policy C final Q is keyed by exactly the same pair, the matched
Digital-3 and bona-fide images must also have the same assigned final Q.

Primary evidence
----------------
project dev:
    51 held-out stems x 3 hardware = 153 matched pairs

Secondary:
    project train:
    160 stems x 3 = 480 pairs

Expected unmatched Digital-3:
    51 stems x 3 = 153 images
These belong to the original train-val population but are not represented
in our frozen project train/dev split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

SEED = 10
N_BOOT = 5000


# ---------------------------------------------------------------------
# Frozen Policy-C population indices
# ---------------------------------------------------------------------

NATURAL_INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)


# ---------------------------------------------------------------------
# Controlled-model saved predictions
# ---------------------------------------------------------------------

CONTROLLED_NATURAL = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_predictions.csv"
)

CONTROLLED_TEST = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_predictions.csv"
)


# ---------------------------------------------------------------------
# Counterfactual-augmented-model saved predictions
# ---------------------------------------------------------------------

AUGMENTED_NATURAL = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_predictions.csv"
)

AUGMENTED_TEST = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_official_test_predictions.csv"
)


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

OUT_PAIRS = (
    ROOT
    / "output"
    / "digital3_same_source_pairs.csv"
)

OUT_SUMMARY = (
    ROOT
    / "output"
    / "digital3_same_source_summary.csv"
)

OUT_HARDWARE = (
    ROOT
    / "output"
    / "digital3_same_source_hardware.csv"
)

OUT_UNMATCHED = (
    ROOT
    / "output"
    / "digital3_unmatched_project_cards.csv"
)


ANCHOR_STEMS = [
    "arabic-003_03",
    "arabic-005_03",
    "arabic-024_03",
    "arabic-112_03",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_file(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required file missing:\n{path}"
        )


def same_source_auc(
    parent_scores,
    attack_scores,
):
    """
    Standard AUROC over a matched source population:

        bona-fide parents -> label 0
        Digital-3 edits  -> label 1

    Unlike official AUROC, both score populations originate from
    exactly the same card/hardware keys.
    """

    parent_scores = np.asarray(
        parent_scores,
        dtype=float,
    )

    attack_scores = np.asarray(
        attack_scores,
        dtype=float,
    )

    if (
        len(parent_scores)
        != len(attack_scores)
    ):
        raise RuntimeError(
            "Parent/attack score counts differ"
        )

    y = np.concatenate(
        [
            np.zeros(
                len(parent_scores),
                dtype=int,
            ),
            np.ones(
                len(attack_scores),
                dtype=int,
            ),
        ]
    )

    score = np.concatenate(
        [
            parent_scores,
            attack_scores,
        ]
    )

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def official_auc(
    test_predictions,
):
    """
    Reproduce the published-style Digital-3 comparator:
    Digital-3 attacks vs the separate official bona-fide population.
    """

    d3 = test_predictions[
        test_predictions["variant"]
        == "digital_3"
    ]

    bona = test_predictions[
        test_predictions["traffic_type"]
        == "bonafide"
    ]

    if (
        len(d3) != 786
        or len(bona) != 300
    ):
        raise RuntimeError(
            "Unexpected official-test population: "
            f"digital3={len(d3)}, "
            f"bonafide={len(bona)}"
        )

    y = np.concatenate(
        [
            np.zeros(
                len(bona),
                dtype=int,
            ),
            np.ones(
                len(d3),
                dtype=int,
            ),
        ]
    )

    score = np.concatenate(
        [
            bona[
                "attack_probability"
            ].to_numpy(),
            d3[
                "attack_probability"
            ].to_numpy(),
        ]
    )

    return {
        "auroc":
            float(
                roc_auc_score(
                    y,
                    score,
                )
            ),

        "mean_attack_probability":
            float(
                d3[
                    "attack_probability"
                ].mean()
            ),

        "median_attack_probability":
            float(
                d3[
                    "attack_probability"
                ].median()
            ),

        "mean_bonafide_probability":
            float(
                bona[
                    "attack_probability"
                ].mean()
            ),

        "median_bonafide_probability":
            float(
                bona[
                    "attack_probability"
                ].median()
            ),

        "attack_recall_at_0_5":
            float(
                (
                    d3[
                        "attack_probability"
                    ]
                    >= 0.5
                ).mean()
            ),

        "bonafide_specificity_at_0_5":
            float(
                (
                    bona[
                        "attack_probability"
                    ]
                    < 0.5
                ).mean()
            ),
    }


# ---------------------------------------------------------------------
# Build authoritative pairing map
# ---------------------------------------------------------------------

def build_pair_map():
    require_file(
        NATURAL_INDEX
    )

    require_file(
        TEST_INDEX
    )

    natural = pd.read_csv(
        NATURAL_INDEX
    )

    test = pd.read_csv(
        TEST_INDEX
    )

    # Our frozen project population:
    # 480 train bona-fides + 153 dev bona-fides.
    parent = natural[
        natural["traffic_type"]
        == "bonafide"
    ].copy()

    if len(parent) != 633:
        raise RuntimeError(
            f"Expected 633 project bona-fides, "
            f"got {len(parent)}"
        )

    if (
        parent["file_stem"]
        .nunique()
        != 211
    ):
        raise RuntimeError(
            "Expected 211 project card stems"
        )

    split_counts = (
        parent["split"]
        .value_counts()
        .to_dict()
    )

    if split_counts != {
        "project_train": 480,
        "dev_val": 153,
    }:
        raise RuntimeError(
            "Unexpected project parent split counts: "
            f"{split_counts}"
        )

    d3 = test[
        test["variant"]
        == "digital_3"
    ].copy()

    if len(d3) != 786:
        raise RuntimeError(
            f"Expected 786 Digital-3 images, "
            f"got {len(d3)}"
        )

    if (
        d3["file_stem"]
        .nunique()
        != 262
    ):
        raise RuntimeError(
            "Expected 262 Digital-3 card stems"
        )

    key = [
        "file_stem",
        "hardware_source",
    ]

    if parent.duplicated(
        key
    ).any():
        raise RuntimeError(
            "Duplicate project parent keys"
        )

    if d3.duplicated(
        key
    ).any():
        raise RuntimeError(
            "Duplicate Digital-3 keys"
        )

    parent = parent.rename(
        columns={
            "image_path":
                "parent_image_path",

            "assigned_q":
                "parent_assigned_q",

            "cache_sha256":
                "parent_cache_sha256",
        }
    )

    d3 = d3.rename(
        columns={
            "image_path":
                "digital3_image_path",

            "assigned_q":
                "digital3_assigned_q",

            "cache_sha256":
                "digital3_cache_sha256",
        }
    )

    pairs = d3.merge(
        parent[
            [
                "file_stem",
                "hardware_source",
                "split",
                "parent_image_path",
                "parent_assigned_q",
                "parent_cache_sha256",
            ]
        ],
        on=key,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    pairs["parent_membership"] = (
        pairs["_merge"]
        .map(
            {
                "both":
                    "matched_project",

                "left_only":
                    "outside_frozen_project_split",

                "right_only":
                    "unexpected",
            }
        )
        .astype(str)
    )

    matched = pairs[
        pairs["_merge"]
        == "both"
    ].copy()

    unmatched = pairs[
        pairs["_merge"]
        == "left_only"
    ].copy()

    if len(matched) != 633:
        raise RuntimeError(
            f"Expected 633 matched Digital-3 "
            f"images, got {len(matched)}"
        )

    if len(unmatched) != 153:
        raise RuntimeError(
            f"Expected 153 unmatched Digital-3 "
            f"images, got {len(unmatched)}"
        )

    if (
        matched["file_stem"]
        .nunique()
        != 211
    ):
        raise RuntimeError(
            "Expected 211 matched stems"
        )

    if (
        unmatched["file_stem"]
        .nunique()
        != 51
    ):
        raise RuntimeError(
            "Expected 51 unmatched stems"
        )

    matched_split_counts = (
        matched["split"]
        .value_counts()
        .to_dict()
    )

    if matched_split_counts != {
        "project_train": 480,
        "dev_val": 153,
    }:
        raise RuntimeError(
            "Unexpected matched split counts: "
            f"{matched_split_counts}"
        )

    # Crucial compression-control invariant:
    # the same stem + hardware receives the same
    # deterministic final Policy-C quality.
    q_match = (
        matched[
            "parent_assigned_q"
        ].astype(int)
        ==
        matched[
            "digital3_assigned_q"
        ].astype(int)
    )

    if not q_match.all():
        failed = matched[
            ~q_match
        ]

        raise RuntimeError(
            "Matched parent/Digital-3 final-Q "
            "assignment differs for "
            f"{len(failed)} rows"
        )

    print(
        "Digital-3 pairing map:"
        f"\n  total Digital-3:      {len(d3)}"
        f"\n  total stems:          "
        f"{d3['file_stem'].nunique()}"
        f"\n  matched project:      {len(matched)}"
        f"\n    project_train:      "
        f"{(matched['split'] == 'project_train').sum()}"
        f"\n    dev_val:            "
        f"{(matched['split'] == 'dev_val').sum()}"
        f"\n  outside project:      {len(unmatched)}"
        f"\n  outside-project stems:"
        f" {unmatched['file_stem'].nunique()}"
        "\n  matched final-Q:      PASS"
    )

    return (
        pairs.drop(
            columns=["_merge"]
        ),
        matched.drop(
            columns=["_merge"]
        ),
        unmatched.drop(
            columns=["_merge"]
        ),
    )


# ---------------------------------------------------------------------
# Attach one model's predictions
# ---------------------------------------------------------------------

def attach_model_scores(
    matched,
    natural_prediction_path,
    test_prediction_path,
    model_name,
):
    require_file(
        natural_prediction_path
    )

    require_file(
        test_prediction_path
    )

    natural = pd.read_csv(
        natural_prediction_path
    )

    test = pd.read_csv(
        test_prediction_path
    )

    for name, frame in [
        ("natural", natural),
        ("test", test),
    ]:
        if (
            "image_path"
            not in frame.columns
            or "attack_probability"
            not in frame.columns
        ):
            raise RuntimeError(
                f"{model_name} {name} predictions "
                "lack required columns"
            )

    # Synthetic augmented training examples never use bona-fide
    # image paths, so parent paths are still unique even though the
    # augmented prediction file contains the 4320-row augmented set.
    natural_for_parents = natural[
        natural["image_path"]
        .isin(
            set(
                matched[
                    "parent_image_path"
                ]
            )
        )
    ].copy()

    if (
        natural_for_parents[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        duplicated = (
            natural_for_parents[
                natural_for_parents[
                    "image_path"
                ].duplicated(
                    keep=False
                )
            ]
        )

        raise RuntimeError(
            f"{model_name}: duplicate parent "
            "prediction rows:\n"
            f"{duplicated[['image_path']].head()}"
        )

    if (
        len(
            natural_for_parents
        )
        != 633
    ):
        raise RuntimeError(
            f"{model_name}: expected 633 "
            "matched parent predictions, got "
            f"{len(natural_for_parents)}"
        )

    test_d3 = test[
        test["image_path"]
        .isin(
            set(
                matched[
                    "digital3_image_path"
                ]
            )
        )
    ].copy()

    if (
        test_d3[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"{model_name}: duplicate "
            "Digital-3 prediction paths"
        )

    if len(test_d3) != 633:
        raise RuntimeError(
            f"{model_name}: expected 633 "
            f"matched Digital-3 predictions, "
            f"got {len(test_d3)}"
        )

    parent_scores = (
        natural_for_parents[
            [
                "image_path",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "image_path":
                    "parent_image_path",

                "attack_probability":
                    f"{model_name}_parent_score",
            }
        )
    )

    d3_scores = (
        test_d3[
            [
                "image_path",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "image_path":
                    "digital3_image_path",

                "attack_probability":
                    f"{model_name}_digital3_score",
            }
        )
    )

    out = matched.merge(
        parent_scores,
        on="parent_image_path",
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        d3_scores,
        on="digital3_image_path",
        how="left",
        validate="one_to_one",
    )

    if (
        out[
            f"{model_name}_parent_score"
        ].isna().any()
        or
        out[
            f"{model_name}_digital3_score"
        ].isna().any()
    ):
        raise RuntimeError(
            f"{model_name}: missing scores "
            "after pairing"
        )

    out[
        f"{model_name}_delta"
    ] = (
        out[
            f"{model_name}_digital3_score"
        ]
        -
        out[
            f"{model_name}_parent_score"
        ]
    )

    return out


# ---------------------------------------------------------------------
# Stem-cluster bootstrap
# ---------------------------------------------------------------------

def bootstrap_matched(
    frame,
    parent_col,
    attack_col,
):
    """
    Resample complete card stems.

    All three hardware observations for a sampled stem stay together.
    """

    stems = np.array(
        sorted(
            frame[
                "file_stem"
            ].unique()
        )
    )

    groups = {
        stem:
            frame[
                frame[
                    "file_stem"
                ]
                == stem
            ][
                [
                    parent_col,
                    attack_col,
                ]
            ]
            .to_numpy(
                dtype=float
            )
        for stem in stems
    }

    rng = np.random.default_rng(
        SEED
    )

    boot = np.empty(
        (
            N_BOOT,
            4,
        ),
        dtype=float,
    )

    for i in range(
        N_BOOT
    ):
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

        parent = values[:, 0]
        attack = values[:, 1]

        delta = (
            attack
            - parent
        )

        boot[
            i
        ] = [
            delta.mean(),

            np.median(
                delta
            ),

            (
                delta > 0
            ).mean(),

            same_source_auc(
                parent,
                attack,
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
        "mean_delta_ci_low":
            low[0],

        "mean_delta_ci_high":
            high[0],

        "median_delta_ci_low":
            low[1],

        "median_delta_ci_high":
            high[1],

        "fraction_increased_ci_low":
            low[2],

        "fraction_increased_ci_high":
            high[2],

        "same_source_auc_ci_low":
            low[3],

        "same_source_auc_ci_high":
            high[3],
    }


# ---------------------------------------------------------------------
# Matched-population summary
# ---------------------------------------------------------------------

def summarize_matched(
    frame,
    model_name,
    population,
):
    parent_col = (
        f"{model_name}_parent_score"
    )

    attack_col = (
        f"{model_name}_digital3_score"
    )

    delta = (
        frame[
            attack_col
        ]
        -
        frame[
            parent_col
        ]
    )

    result = {
        "model":
            model_name,

        "population":
            population,

        "n_pairs":
            len(frame),

        "n_stems":
            frame[
                "file_stem"
            ].nunique(),

        "mean_parent_score":
            frame[
                parent_col
            ].mean(),

        "median_parent_score":
            frame[
                parent_col
            ].median(),

        "mean_digital3_score":
            frame[
                attack_col
            ].mean(),

        "median_digital3_score":
            frame[
                attack_col
            ].median(),

        "mean_delta":
            delta.mean(),

        "median_delta":
            delta.median(),

        "fraction_score_increased":
            (
                delta > 0
            ).mean(),

        "fraction_score_decreased":
            (
                delta < 0
            ).mean(),

        "digital3_recall_at_0_5":
            (
                frame[
                    attack_col
                ]
                >= 0.5
            ).mean(),

        "parent_false_positive_rate_at_0_5":
            (
                frame[
                    parent_col
                ]
                >= 0.5
            ).mean(),

        "same_source_auc":
            same_source_auc(
                frame[
                    parent_col
                ],
                frame[
                    attack_col
                ],
            ),
    }

    result.update(
        bootstrap_matched(
            frame,
            parent_col,
            attack_col,
        )
    )

    return result


# ---------------------------------------------------------------------
# Hardware breakdown
# ---------------------------------------------------------------------

def hardware_summary(
    matched,
):
    rows = []

    for model_name in [
        "controlled",
        "augmented",
    ]:
        parent_col = (
            f"{model_name}_parent_score"
        )

        attack_col = (
            f"{model_name}_digital3_score"
        )

        for split_name, split_frame in [
            (
                "matched_all_project",
                matched,
            ),
            (
                "matched_dev_val",
                matched[
                    matched["split"]
                    == "dev_val"
                ],
            ),
        ]:
            for hardware in sorted(
                split_frame[
                    "hardware_source"
                ].unique()
            ):
                frame = split_frame[
                    split_frame[
                        "hardware_source"
                    ]
                    == hardware
                ]

                delta = (
                    frame[
                        attack_col
                    ]
                    -
                    frame[
                        parent_col
                    ]
                )

                rows.append(
                    {
                        "model":
                            model_name,

                        "population":
                            split_name,

                        "hardware_source":
                            hardware,

                        "n_pairs":
                            len(frame),

                        "n_stems":
                            frame[
                                "file_stem"
                            ].nunique(),

                        "mean_parent_score":
                            frame[
                                parent_col
                            ].mean(),

                        "mean_digital3_score":
                            frame[
                                attack_col
                            ].mean(),

                        "mean_delta":
                            delta.mean(),

                        "median_delta":
                            delta.median(),

                        "fraction_score_increased":
                            (
                                delta > 0
                            ).mean(),

                        "same_source_auc":
                            same_source_auc(
                                frame[
                                    parent_col
                                ],
                                frame[
                                    attack_col
                                ],
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Unmatched Digital-3 summary
# ---------------------------------------------------------------------

def attach_unmatched_scores(
    unmatched,
):
    out = unmatched.copy()

    for model_name, path in [
        (
            "controlled",
            CONTROLLED_TEST,
        ),
        (
            "augmented",
            AUGMENTED_TEST,
        ),
    ]:
        require_file(
            path
        )

        predictions = pd.read_csv(
            path
        )

        selected = (
            predictions[
                predictions[
                    "image_path"
                ]
                .isin(
                    set(
                        unmatched[
                            "digital3_image_path"
                        ]
                    )
                )
            ][
                [
                    "image_path",
                    "attack_probability",
                ]
            ]
            .rename(
                columns={
                    "image_path":
                        "digital3_image_path",

                    "attack_probability":
                        f"{model_name}_digital3_score",
                }
            )
        )

        if len(selected) != 153:
            raise RuntimeError(
                f"{model_name}: expected "
                f"153 unmatched Digital-3 "
                f"predictions, got "
                f"{len(selected)}"
            )

        out = out.merge(
            selected,
            on="digital3_image_path",
            how="left",
            validate="one_to_one",
        )

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    for path in [
        CONTROLLED_NATURAL,
        CONTROLLED_TEST,
        AUGMENTED_NATURAL,
        AUGMENTED_TEST,
    ]:
        require_file(
            path
        )

    (
        all_d3,
        matched,
        unmatched,
    ) = build_pair_map()

    # --------------------------------------------------
    # Controlled model
    # --------------------------------------------------

    matched = attach_model_scores(
        matched,
        CONTROLLED_NATURAL,
        CONTROLLED_TEST,
        "controlled",
    )

    # --------------------------------------------------
    # Counterfactual-augmented model
    # --------------------------------------------------

    matched = attach_model_scores(
        matched,
        AUGMENTED_NATURAL,
        AUGMENTED_TEST,
        "augmented",
    )

    # --------------------------------------------------
    # Save detailed pair table first
    # --------------------------------------------------

    OUT_PAIRS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched.to_csv(
        OUT_PAIRS,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    rows = []

    populations = [
        (
            "matched_dev_val",
            matched[
                matched["split"]
                == "dev_val"
            ],
        ),
        (
            "matched_project_train",
            matched[
                matched["split"]
                == "project_train"
            ],
        ),
        (
            "matched_all_project",
            matched,
        ),
    ]

    for model_name in [
        "controlled",
        "augmented",
    ]:
        for (
            population_name,
            frame,
        ) in populations:
            rows.append(
                summarize_matched(
                    frame,
                    model_name,
                    population_name,
                )
            )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------
    # Hardware breakdown
    # --------------------------------------------------

    hardware = hardware_summary(
        matched
    )

    hardware.to_csv(
        OUT_HARDWARE,
        index=False,
    )

    # --------------------------------------------------
    # Unmatched 51-card population
    # --------------------------------------------------

    unmatched = attach_unmatched_scores(
        unmatched
    )

    unmatched.to_csv(
        OUT_UNMATCHED,
        index=False,
    )

    # --------------------------------------------------
    # Official comparator for reference
    # --------------------------------------------------

    official_rows = []

    for model_name, path in [
        (
            "controlled",
            CONTROLLED_TEST,
        ),
        (
            "augmented",
            AUGMENTED_TEST,
        ),
    ]:
        predictions = pd.read_csv(
            path
        )

        result = official_auc(
            predictions
        )

        official_rows.append(
            {
                "model":
                    model_name,

                **result,
            }
        )

    official = pd.DataFrame(
        official_rows
    )

    # --------------------------------------------------
    # Print primary evidence: DEV FIRST
    # --------------------------------------------------

    display = [
        "model",
        "population",
        "n_pairs",
        "n_stems",
        "mean_parent_score",
        "mean_digital3_score",
        "mean_delta",
        "median_delta",
        "fraction_score_increased",
        "digital3_recall_at_0_5",
        "same_source_auc",
        "same_source_auc_ci_low",
        "same_source_auc_ci_high",
    ]

    print(
        "\nSAME-SOURCE DIGITAL-3 AUDIT:"
    )

    print(
        summary[
            display
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nPUBLISHED-STYLE OFFICIAL COMPARATOR:"
    )

    print(
        official.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # Direct comparison of official AUROC vs
    # same-source dev/all AUROC
    # --------------------------------------------------

    comparison_rows = []

    for model_name in [
        "controlled",
        "augmented",
    ]:
        official_auc_value = float(
            official.loc[
                official["model"]
                == model_name,
                "auroc",
            ].iloc[0]
        )

        for population in [
            "matched_dev_val",
            "matched_all_project",
        ]:
            row = summary[
                (
                    summary["model"]
                    == model_name
                )
                &
                (
                    summary["population"]
                    == population
                )
            ].iloc[0]

            comparison_rows.append(
                {
                    "model":
                        model_name,

                    "population":
                        population,

                    "official_digital3_auc":
                        official_auc_value,

                    "same_source_auc":
                        row[
                            "same_source_auc"
                        ],

                    "auc_difference":
                        (
                            row[
                                "same_source_auc"
                            ]
                            -
                            official_auc_value
                        ),

                    "mean_paired_delta":
                        row[
                            "mean_delta"
                        ],

                    "fraction_edit_increased_score":
                        row[
                            "fraction_score_increased"
                        ],
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    print(
        "\nOFFICIAL VS SAME-SOURCE COMPARISON:"
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # Hardware breakdown, dev only
    # --------------------------------------------------

    print(
        "\nHELD-OUT DEV — HARDWARE BREAKDOWN:"
    )

    dev_hardware = hardware[
        hardware["population"]
        == "matched_dev_val"
    ]

    print(
        dev_hardware.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # Unmatched population
    # --------------------------------------------------

    print(
        "\nDIGITAL-3 OUTSIDE FROZEN PROJECT SPLIT:"
        f"\n  images: {len(unmatched)}"
        f"\n  stems:  "
        f"{unmatched['file_stem'].nunique()}"
    )

    for model_name in [
        "controlled",
        "augmented",
    ]:
        col = (
            f"{model_name}_digital3_score"
        )

        print(
            f"\n{model_name}:"
            f"\n  mean score:   "
            f"{unmatched[col].mean():.4f}"
            f"\n  median score: "
            f"{unmatched[col].median():.4f}"
            f"\n  recall@0.5:   "
            f"{(unmatched[col] >= 0.5).mean():.4f}"
        )

    # --------------------------------------------------
    # Four user-inspected anchor examples
    # --------------------------------------------------

    anchors = matched[
        matched[
            "file_stem"
        ].isin(
            ANCHOR_STEMS
        )
        &
        (
            matched[
                "hardware_source"
            ]
            == "huawei"
        )
    ].copy()

    anchor_columns = [
        "file_stem",
        "split",
        "hardware_source",
        "parent_assigned_q",
        "digital3_assigned_q",
        "controlled_parent_score",
        "controlled_digital3_score",
        "controlled_delta",
        "augmented_parent_score",
        "augmented_digital3_score",
        "augmented_delta",
    ]

    print(
        "\nFOUR INSPECTED HUAWEI EXAMPLES:"
    )

    print(
        anchors[
            anchor_columns
        ]
        .sort_values(
            "file_stem"
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print(
        f"\nPairs:      {OUT_PAIRS}"
        f"\nSummary:    {OUT_SUMMARY}"
        f"\nHardware:   {OUT_HARDWARE}"
        f"\nUnmatched:  {OUT_UNMATCHED}"
    )

    print(
        "\nInterpretation rule:"
        "\n- Treat matched_dev_val as the primary evidence."
        "\n- fraction_score_increased answers whether Digital-3 "
        "editing moves the model in the attack direction."
        "\n- same_source_auc asks whether the edited versions rank "
        "above their corresponding source population."
        "\n- Compare that with the official Digital-3 AUROC, which "
        "uses the separate 300-image bona-fide population."
        "\n- project_train is secondary because those parent cards "
        "were used for model fitting."
    )


if __name__ == "__main__":
    main()