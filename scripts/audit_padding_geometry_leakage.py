#!/usr/bin/env python3
"""
Audit geometry / padding leakage in the FantasyID ResNet preprocessing.

NO CNN inference.
NO training of a new image model.
NO test-set model selection.

Current preprocessing
---------------------
native image
    -> preserve aspect ratio
    -> resize content height to 512
    -> centre horizontally in 512 x 864 canvas
    -> ImageNet-mean padding

The network can therefore directly observe:
    - content aspect ratio
    - resized content width
    - left/right padding width
    - document boundary position

It cannot directly observe the original native W/H after resizing, but native
dimensions are also audited because they can identify capture/device
populations and may correlate with resampling characteristics.

Questions
---------
1. Does input geometry predict attack/bona label in project_train -> dev?
2. Are bona / d1 / d2 geometries actually matched within stem+hardware?
3. Does geometry distinguish official attack families from official bona-fide?
4. Does that distinction survive restricting to hardware shared by both classes?
5. Are frozen ResNet scores correlated with padding width?
6. For matched Digital-3 parent pairs, is padding itself unchanged?

Important
---------
Official-family classifiers below are POST-HOC DIAGNOSTIC cross-validation.
They quantify how much population information exists in geometry.
They do NOT establish that the CNN uses that information.

A causal padding perturbation should only be run if this audit gives a reason.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAVE_STRATIFIED_GROUP = True

except ImportError:
    from sklearn.model_selection import GroupKFold

    HAVE_STRATIFIED_GROUP = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TRAIN_INDEX = (
    ROOT
    / "output"
    / "policy_c_cache_index.csv"
)

TEST_INDEX = (
    ROOT
    / "output"
    / "fantasyid_official_test_policy_c_index.csv"
)

CONTROLLED_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_policy_c_seed10_official_test_predictions.csv"
)

AUGMENTED_PREDICTIONS = (
    ROOT
    / "output"
    / "resnet18_counterfactual_augmented_seed10_official_test_predictions.csv"
)

OUT_FEATURES = (
    ROOT
    / "output"
    / "padding_geometry_features.csv"
)

OUT_LABEL_LEAKAGE = (
    ROOT
    / "output"
    / "padding_geometry_train_dev_leakage.csv"
)

OUT_OFFICIAL = (
    ROOT
    / "output"
    / "padding_geometry_official_discrimination.csv"
)

OUT_POPULATIONS = (
    ROOT
    / "output"
    / "padding_geometry_population_summary.csv"
)

OUT_CORRELATIONS = (
    ROOT
    / "output"
    / "padding_geometry_model_score_correlations.csv"
)

OUT_INCONSISTENCIES = (
    ROOT
    / "output"
    / "padding_geometry_matched_inconsistencies.csv"
)

OUT_D3_MATCHED = (
    ROOT
    / "output"
    / "padding_geometry_digital3_same_source.csv"
)


# ---------------------------------------------------------------------
# Frozen preprocessing geometry
# ---------------------------------------------------------------------

CONTENT_HEIGHT = 512
CANVAS_WIDTH = 864

SEED = 10
N_BOOT = 2000

FAMILIES = [
    "digital_3",
    "facedancer",
    "textdiffuserft_bfei",
]

FEATURE_SETS = {
    # This is the cue most directly visible from our fixed canvas.
    "padding_only": [
        "pad_total_r512",
    ],

    # Equivalent information in continuous native geometry.
    "aspect_ratio_only": [
        "aspect_ratio",
    ],

    # Not directly visible after resizing, but useful to expose
    # source/capture population differences.
    "native_size_only": [
        "log_native_width",
        "log_native_height",
    ],

    # Broad geometry diagnostic.
    "all_geometry": [
        "aspect_ratio",
        "log_native_width",
        "log_native_height",
    ],
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def require_file(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required file missing:\n{path}"
        )


def normalize_metadata(frame):
    frame = frame.copy()

    for column in [
        "image_path",
        "cache_path",
        "file_stem",
        "traffic_type",
        "hardware_source",
    ]:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
            )

    if "variant" in frame.columns:
        frame["variant"] = (
            frame["variant"]
            .fillna("")
            .astype(str)
        )

    frame["label"] = (
        frame["label"]
        .astype(int)
    )

    return frame


def population_name(row):
    if (
        row.traffic_type
        == "bonafide"
    ):
        return "bonafide"

    return str(
        row.variant
    )


# ---------------------------------------------------------------------
# Extract exact geometry seen by preprocessing
# ---------------------------------------------------------------------

def extract_geometry(
    frame,
    source_dataset,
):
    records = []

    for i, row in enumerate(
        frame.itertuples(
            index=False
        ),
        start=1,
    ):
        path = (
            ROOT
            / row.cache_path
        )

        with Image.open(path) as image:
            width, height = (
                image.size
            )

        if (
            width <= 0
            or height <= 0
        ):
            raise RuntimeError(
                f"Invalid image geometry: {path}"
            )

        aspect = (
            width
            / height
        )

        content_width = int(
            round(
                width
                * CONTENT_HEIGHT
                / height
            )
        )

        if (
            content_width
            > CANVAS_WIDTH
        ):
            raise RuntimeError(
                "Image would not fit frozen "
                "512x864 preprocessing:"
                f"\n{row.image_path}"
                f"\nresized width={content_width}"
            )

        total_pad = (
            CANVAS_WIDTH
            - content_width
        )

        left_pad = (
            total_pad
            // 2
        )

        right_pad = (
            total_pad
            - left_pad
        )

        records.append(
            {
                "source_dataset":
                    source_dataset,

                "split":
                    getattr(
                        row,
                        "split",
                        "",
                    ),

                "image_path":
                    row.image_path,

                "file_stem":
                    row.file_stem,

                "traffic_type":
                    row.traffic_type,

                "variant":
                    (
                        ""
                        if pd.isna(
                            row.variant
                        )
                        else str(
                            row.variant
                        )
                    ),

                "hardware_source":
                    row.hardware_source,

                "label":
                    int(
                        row.label
                    ),

                "native_width":
                    width,

                "native_height":
                    height,

                "aspect_ratio":
                    aspect,

                "log_native_width":
                    float(
                        np.log(
                            width
                        )
                    ),

                "log_native_height":
                    float(
                        np.log(
                            height
                        )
                    ),

                "log_native_area":
                    float(
                        np.log(
                            width
                            * height
                        )
                    ),

                "content_width_r512":
                    content_width,

                "pad_total_r512":
                    total_pad,

                "pad_left_r512":
                    left_pad,

                "pad_right_r512":
                    right_pad,

                "content_fraction":
                    (
                        content_width
                        / CANVAS_WIDTH
                    ),
            }
        )

        if (
            i % 500 == 0
            or i == len(frame)
        ):
            print(
                f"{source_dataset}: "
                f"geometry {i}/{len(frame)}"
            )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Logistic geometry probe
# ---------------------------------------------------------------------

def build_probe():
    return make_pipeline(
        StandardScaler(),

        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
    )


def bootstrap_auc_by_stem(
    frame,
    score_column,
):
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
                    "label",
                    score_column,
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

    aucs = []

    for _ in range(
        N_BOOT
    ):
        sampled = rng.choice(
            stems,
            size=len(stems),
            replace=True,
        )

        values = np.concatenate(
            [
                groups[
                    stem
                ]
                for stem
                in sampled
            ],
            axis=0,
        )

        y = values[:, 0]
        p = values[:, 1]

        if len(
            np.unique(y)
        ) != 2:
            continue

        aucs.append(
            roc_auc_score(
                y,
                p,
            )
        )

    if not aucs:
        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            np.quantile(
                aucs,
                0.025,
            )
        ),
        float(
            np.quantile(
                aucs,
                0.975,
            )
        ),
    )


def train_dev_probe(
    train,
    dev,
    feature_set_name,
    features,
):
    model = build_probe()

    model.fit(
        train[
            features
        ],
        train[
            "label"
        ],
    )

    score = model.predict_proba(
        dev[
            features
        ]
    )[:, 1]

    scored = dev.copy()

    scored[
        "geometry_score"
    ] = score

    auc = float(
        roc_auc_score(
            scored[
                "label"
            ],
            scored[
                "geometry_score"
            ],
        )
    )

    ci_low, ci_high = (
        bootstrap_auc_by_stem(
            scored,
            "geometry_score",
        )
    )

    return {
        "evaluation":
            "project_train_to_dev",

        "feature_set":
            feature_set_name,

        "n_train":
            len(train),

        "n_eval":
            len(dev),

        "n_eval_stems":
            dev[
                "file_stem"
            ].nunique(),

        "auroc":
            auc,

        "auc_ci_low":
            ci_low,

        "auc_ci_high":
            ci_high,
    }


# ---------------------------------------------------------------------
# Grouped out-of-fold official diagnostic
# ---------------------------------------------------------------------

def grouped_oof_scores(
    frame,
    features,
):
    frame = (
        frame
        .reset_index(
            drop=True
        )
        .copy()
    )

    y = (
        frame[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    groups = (
        frame[
            "file_stem"
        ]
        .to_numpy()
    )

    x = (
        frame[
            features
        ]
        .to_numpy(
            dtype=float
        )
    )

    if HAVE_STRATIFIED_GROUP:
        splitter = (
            StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=SEED,
            )
        )

        folds = splitter.split(
            x,
            y,
            groups,
        )

    else:
        splitter = (
            GroupKFold(
                n_splits=5
            )
        )

        folds = splitter.split(
            x,
            y,
            groups,
        )

    scores = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    for train_index, test_index in folds:
        model = build_probe()

        model.fit(
            x[
                train_index
            ],
            y[
                train_index
            ],
        )

        scores[
            test_index
        ] = (
            model.predict_proba(
                x[
                    test_index
                ]
            )[:, 1]
        )

    if np.isnan(
        scores
    ).any():
        raise RuntimeError(
            "OOF geometry predictions "
            "incomplete"
        )

    return scores


def official_probe(
    frame,
    family,
    scope,
    feature_set_name,
    features,
):
    bona = frame[
        frame[
            "traffic_type"
        ]
        == "bonafide"
    ].copy()

    attack = frame[
        frame[
            "variant"
        ]
        == family
    ].copy()

    if scope == "shared_hardware":
        shared = sorted(
            set(
                bona[
                    "hardware_source"
                ]
            )
            &
            set(
                attack[
                    "hardware_source"
                ]
            )
        )

        bona = bona[
            bona[
                "hardware_source"
            ].isin(
                shared
            )
        ]

        attack = attack[
            attack[
                "hardware_source"
            ].isin(
                shared
            )
        ]

    elif scope == "all_hardware":
        shared = sorted(
            set(
                bona[
                    "hardware_source"
                ]
            )
            &
            set(
                attack[
                    "hardware_source"
                ]
            )
        )

    else:
        raise ValueError(
            scope
        )

    subset = pd.concat(
        [
            bona,
            attack,
        ],
        ignore_index=True,
    )

    if (
        subset[
            "label"
        ]
        .nunique()
        != 2
    ):
        raise RuntimeError(
            f"{family}/{scope}: "
            "missing one class"
        )

    score = grouped_oof_scores(
        subset,
        features,
    )

    subset[
        "geometry_score"
    ] = score

    auc = float(
        roc_auc_score(
            subset[
                "label"
            ],
            subset[
                "geometry_score"
            ],
        )
    )

    ci_low, ci_high = (
        bootstrap_auc_by_stem(
            subset,
            "geometry_score",
        )
    )

    return {
        "family":
            family,

        "scope":
            scope,

        "shared_hardware":
            "|".join(
                shared
            ),

        "feature_set":
            feature_set_name,

        "n_attack":
            len(attack),

        "n_bonafide":
            len(bona),

        "n_stems":
            subset[
                "file_stem"
            ].nunique(),

        "auroc":
            auc,

        "auc_ci_low":
            ci_low,

        "auc_ci_high":
            ci_high,
    }


# ---------------------------------------------------------------------
# Matched train/dev geometry audit
# ---------------------------------------------------------------------

def matched_geometry_audit(
    train_geometry,
):
    grouped = (
        train_geometry.groupby(
            [
                "split",
                "file_stem",
                "hardware_source",
            ],
            dropna=False,
        )
        .agg(
            n_rows=(
                "image_path",
                "size",
            ),

            n_native_width=(
                "native_width",
                "nunique",
            ),

            n_native_height=(
                "native_height",
                "nunique",
            ),

            n_content_width=(
                "content_width_r512",
                "nunique",
            ),

            n_pad=(
                "pad_total_r512",
                "nunique",
            ),
        )
        .reset_index()
    )

    grouped[
        "native_dimension_mismatch"
    ] = (
        (
            grouped[
                "n_native_width"
            ]
            > 1
        )
        |
        (
            grouped[
                "n_native_height"
            ]
            > 1
        )
    )

    grouped[
        "r512_padding_mismatch"
    ] = (
        grouped[
            "n_pad"
        ]
        > 1
    )

    bad = grouped[
        (
            grouped[
                "native_dimension_mismatch"
            ]
        )
        |
        (
            grouped[
                "r512_padding_mismatch"
            ]
        )
        |
        (
            grouped[
                "n_rows"
            ]
            != 3
        )
    ].copy()

    return (
        grouped,
        bad,
    )


# ---------------------------------------------------------------------
# Digital-3 same-source geometry
# ---------------------------------------------------------------------

def digital3_same_source_geometry(
    train_geometry,
    official_geometry,
):
    parent = train_geometry[
        train_geometry[
            "traffic_type"
        ]
        == "bonafide"
    ][
        [
            "split",
            "file_stem",
            "hardware_source",
            "native_width",
            "native_height",
            "content_width_r512",
            "pad_total_r512",
        ]
    ].copy()

    parent = parent.rename(
        columns={
            "native_width":
                "parent_width",

            "native_height":
                "parent_height",

            "content_width_r512":
                "parent_content_width",

            "pad_total_r512":
                "parent_pad",
        }
    )

    d3 = official_geometry[
        official_geometry[
            "variant"
        ]
        == "digital_3"
    ][
        [
            "image_path",
            "file_stem",
            "hardware_source",
            "native_width",
            "native_height",
            "content_width_r512",
            "pad_total_r512",
        ]
    ].copy()

    d3 = d3.rename(
        columns={
            "native_width":
                "digital3_width",

            "native_height":
                "digital3_height",

            "content_width_r512":
                "digital3_content_width",

            "pad_total_r512":
                "digital3_pad",
        }
    )

    matched = d3.merge(
        parent,
        on=[
            "file_stem",
            "hardware_source",
        ],
        how="inner",
        validate="one_to_one",
    )

    matched[
        "native_dimensions_match"
    ] = (
        (
            matched[
                "parent_width"
            ]
            ==
            matched[
                "digital3_width"
            ]
        )
        &
        (
            matched[
                "parent_height"
            ]
            ==
            matched[
                "digital3_height"
            ]
        )
    )

    matched[
        "r512_content_width_match"
    ] = (
        matched[
            "parent_content_width"
        ]
        ==
        matched[
            "digital3_content_width"
        ]
    )

    matched[
        "padding_match"
    ] = (
        matched[
            "parent_pad"
        ]
        ==
        matched[
            "digital3_pad"
        ]
    )

    return matched


# ---------------------------------------------------------------------
# Population summaries
# ---------------------------------------------------------------------

def population_summary(
    geometry,
):
    frame = geometry.copy()

    frame[
        "population"
    ] = [
        population_name(
            row
        )
        for row
        in frame.itertuples(
            index=False
        )
    ]

    return (
        frame.groupby(
            [
                "source_dataset",
                "population",
            ],
            as_index=False,
        )
        .agg(
            n_images=(
                "image_path",
                "size",
            ),

            n_stems=(
                "file_stem",
                "nunique",
            ),

            mean_aspect_ratio=(
                "aspect_ratio",
                "mean",
            ),

            std_aspect_ratio=(
                "aspect_ratio",
                "std",
            ),

            mean_content_width=(
                "content_width_r512",
                "mean",
            ),

            min_content_width=(
                "content_width_r512",
                "min",
            ),

            max_content_width=(
                "content_width_r512",
                "max",
            ),

            mean_total_pad=(
                "pad_total_r512",
                "mean",
            ),

            min_total_pad=(
                "pad_total_r512",
                "min",
            ),

            max_total_pad=(
                "pad_total_r512",
                "max",
            ),

            mean_native_width=(
                "native_width",
                "mean",
            ),

            mean_native_height=(
                "native_height",
                "mean",
            ),
        )
    )


# ---------------------------------------------------------------------
# ResNet-score correlation with geometry
# ---------------------------------------------------------------------

def load_prediction(
    path,
    score_name,
):
    require_file(
        path
    )

    frame = pd.read_csv(
        path,
        keep_default_na=False,
    )

    if (
        len(frame) != 1385
    ):
        raise RuntimeError(
            f"Expected 1385 rows: {path}"
        )

    if (
        frame[
            "image_path"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Duplicate prediction paths: "
            f"{path}"
        )

    return (
        frame[
            [
                "image_path",
                "attack_probability",
            ]
        ]
        .rename(
            columns={
                "attack_probability":
                    score_name,
            }
        )
    )


def score_correlations(
    official_geometry,
):
    controlled = (
        load_prediction(
            CONTROLLED_PREDICTIONS,
            "controlled_score",
        )
    )

    augmented = (
        load_prediction(
            AUGMENTED_PREDICTIONS,
            "augmented_score",
        )
    )

    frame = (
        official_geometry
        .merge(
            controlled,
            on="image_path",
            validate="one_to_one",
        )
        .merge(
            augmented,
            on="image_path",
            validate="one_to_one",
        )
    )

    frame[
        "population"
    ] = [
        population_name(
            row
        )
        for row
        in frame.itertuples(
            index=False
        )
    ]

    rows = []

    for model_name, score_column in [
        (
            "controlled",
            "controlled_score",
        ),
        (
            "augmented",
            "augmented_score",
        ),
    ]:
        for population in [
            "bonafide",
            "digital_3",
            "facedancer",
            "textdiffuserft_bfei",
        ]:
            subset = frame[
                frame[
                    "population"
                ]
                == population
            ]

            for feature in [
                "pad_total_r512",
                "aspect_ratio",
                "log_native_area",
            ]:
                result = spearmanr(
                    subset[
                        feature
                    ],
                    subset[
                        score_column
                    ],
                )

                rows.append(
                    {
                        "model":
                            model_name,

                        "population":
                            population,

                        "feature":
                            feature,

                        "n":
                            len(subset),

                        "n_stems":
                            subset[
                                "file_stem"
                            ].nunique(),

                        "spearman_r":
                            float(
                                result.statistic
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Hardware composition
# ---------------------------------------------------------------------

def hardware_table(
    official_geometry,
):
    frame = (
        official_geometry.copy()
    )

    frame[
        "population"
    ] = [
        population_name(
            row
        )
        for row
        in frame.itertuples(
            index=False
        )
    ]

    return (
        frame.groupby(
            [
                "population",
                "hardware_source",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    for path in [
        TRAIN_INDEX,
        TEST_INDEX,
        CONTROLLED_PREDICTIONS,
        AUGMENTED_PREDICTIONS,
    ]:
        require_file(
            path
        )

    train_index = (
        normalize_metadata(
            pd.read_csv(
                TRAIN_INDEX,
                keep_default_na=False,
            )
        )
    )

    test_index = (
        normalize_metadata(
            pd.read_csv(
                TEST_INDEX,
                keep_default_na=False,
            )
        )
    )

    if len(
        train_index
    ) != 1899:
        raise RuntimeError(
            "Expected 1899 project "
            "train+dev images"
        )

    if len(
        test_index
    ) != 1385:
        raise RuntimeError(
            "Expected 1385 official "
            "test images"
        )

    print(
        "PADDING / GEOMETRY "
        "LEAKAGE AUDIT"
    )

    print(
        "\nFrozen preprocessing:"
        "\n  content height: 512"
        "\n  canvas:         512 x 864"
        "\n  aspect ratio:   preserved"
        "\n  horizontal:     centred"
        "\n  pad value:      ImageNet mean"
    )

    # --------------------------------------------------
    # Extract geometry
    # --------------------------------------------------

    train_geometry = (
        extract_geometry(
            train_index,
            "project",
        )
    )

    official_geometry = (
        extract_geometry(
            test_index,
            "official",
        )
    )

    all_geometry = pd.concat(
        [
            train_geometry,
            official_geometry,
        ],
        ignore_index=True,
    )

    all_geometry.to_csv(
        OUT_FEATURES,
        index=False,
    )

    # --------------------------------------------------
    # 1. Project train -> dev geometry leakage
    # --------------------------------------------------

    project_train = (
        train_geometry[
            train_geometry[
                "split"
            ]
            == "project_train"
        ]
    )

    dev = (
        train_geometry[
            train_geometry[
                "split"
            ]
            == "dev_val"
        ]
    )

    if (
        len(project_train) != 1440
        or len(dev) != 459
    ):
        raise RuntimeError(
            "Frozen split counts changed"
        )

    leakage_rows = []

    for (
        feature_set_name,
        features,
    ) in FEATURE_SETS.items():
        leakage_rows.append(
            train_dev_probe(
                project_train,
                dev,
                feature_set_name,
                features,
            )
        )

    leakage = pd.DataFrame(
        leakage_rows
    )

    leakage.to_csv(
        OUT_LABEL_LEAKAGE,
        index=False,
    )

    # --------------------------------------------------
    # 2. Exact matched-geometry consistency
    # --------------------------------------------------

    (
        matched_groups,
        inconsistencies,
    ) = matched_geometry_audit(
        train_geometry
    )

    inconsistencies.to_csv(
        OUT_INCONSISTENCIES,
        index=False,
    )

    # --------------------------------------------------
    # 3. Digital-3 same-source geometry
    # --------------------------------------------------

    d3_matched = (
        digital3_same_source_geometry(
            train_geometry,
            official_geometry,
        )
    )

    d3_matched.to_csv(
        OUT_D3_MATCHED,
        index=False,
    )

    # --------------------------------------------------
    # 4. Official population geometry discrimination
    # --------------------------------------------------

    official_rows = []

    for family in FAMILIES:
        for scope in [
            "all_hardware",
            "shared_hardware",
        ]:
            for (
                feature_set_name,
                features,
            ) in FEATURE_SETS.items():
                official_rows.append(
                    official_probe(
                        official_geometry,
                        family,
                        scope,
                        feature_set_name,
                        features,
                    )
                )

    official_summary = pd.DataFrame(
        official_rows
    )

    official_summary.to_csv(
        OUT_OFFICIAL,
        index=False,
    )

    # --------------------------------------------------
    # 5. Population descriptive geometry
    # --------------------------------------------------

    populations = (
        population_summary(
            all_geometry
        )
    )

    populations.to_csv(
        OUT_POPULATIONS,
        index=False,
    )

    # --------------------------------------------------
    # 6. Correlation with frozen CNN scores
    # --------------------------------------------------

    correlations = (
        score_correlations(
            official_geometry
        )
    )

    correlations.to_csv(
        OUT_CORRELATIONS,
        index=False,
    )

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    print(
        "\nMATCHED PROJECT GEOMETRY:"
    )

    print(
        f"  stem+hardware groups:       "
        f"{len(matched_groups)}"
        f"\n  groups with !=3 rows:       "
        f"{(matched_groups['n_rows'] != 3).sum()}"
        f"\n  native dimension mismatch:  "
        f"{matched_groups['native_dimension_mismatch'].sum()}"
        f"\n  r512 padding mismatch:       "
        f"{matched_groups['r512_padding_mismatch'].sum()}"
    )

    if len(
        inconsistencies
    ):
        print(
            "\nMatched inconsistencies:"
        )

        print(
            inconsistencies.to_string(
                index=False
            )
        )

    print(
        "\nPROJECT TRAIN -> HELD-OUT DEV "
        "GEOMETRY LABEL LEAKAGE:"
    )

    print(
        leakage[
            [
                "feature_set",
                "n_train",
                "n_eval",
                "n_eval_stems",
                "auroc",
                "auc_ci_low",
                "auc_ci_high",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nDIGITAL-3 SAME-SOURCE "
        "GEOMETRY:"
    )

    print(
        f"  matched images:              "
        f"{len(d3_matched)}"
        f"\n  matched stems:               "
        f"{d3_matched['file_stem'].nunique()}"
        f"\n  native dimensions identical: "
        f"{d3_matched['native_dimensions_match'].mean():.4f}"
        f"\n  r512 width identical:        "
        f"{d3_matched['r512_content_width_match'].mean():.4f}"
        f"\n  padding identical:           "
        f"{d3_matched['padding_match'].mean():.4f}"
    )

    print(
        "\nOFFICIAL POPULATION GEOMETRY:"
    )

    print(
        populations[
            populations[
                "source_dataset"
            ]
            == "official"
        ][
            [
                "population",
                "n_images",
                "n_stems",
                "mean_aspect_ratio",
                "mean_content_width",
                "min_content_width",
                "max_content_width",
                "mean_total_pad",
                "min_total_pad",
                "max_total_pad",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nOFFICIAL HARDWARE COMPOSITION:"
    )

    print(
        hardware_table(
            official_geometry
        ).to_string()
    )

    print(
        "\nOFFICIAL GEOMETRY-ONLY "
        "DISCRIMINATION:"
    )

    print(
        official_summary[
            [
                "family",
                "scope",
                "shared_hardware",
                "feature_set",
                "n_attack",
                "n_bonafide",
                "n_stems",
                "auroc",
                "auc_ci_low",
                "auc_ci_high",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nFROZEN CNN SCORE "
        "CORRELATION WITH GEOMETRY:"
    )

    print(
        correlations.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nOUTPUTS:"
        f"\n  features:        {OUT_FEATURES}"
        f"\n  train/dev:       {OUT_LABEL_LEAKAGE}"
        f"\n  official probe:  {OUT_OFFICIAL}"
        f"\n  populations:     {OUT_POPULATIONS}"
        f"\n  correlations:    {OUT_CORRELATIONS}"
        f"\n  inconsistencies: {OUT_INCONSISTENCIES}"
        f"\n  D3 matched:      {OUT_D3_MATCHED}"
    )

    print(
        "\nINTERPRETATION GUIDE:"
        "\n"
        "\n1. project train->dev padding_only ~0.50"
        "\n   => padding width was not a useful attack-label shortcut "
        "inside the training distribution."
        "\n"
        "\n2. Digital-3 same-source padding_match == 1.0"
        "\n   => padding cannot explain the paired Digital-3 score "
        "increase/decrease relative to its own parent."
        "\n"
        "\n3. Official Digital-3 vs bona padding_only substantially "
        ">0.50"
        "\n   => padding encodes card-population/domain information and "
        "could contribute to the catastrophic official inversion."
        "\n"
        "\n4. If that remains high under shared_hardware, the effect "
        "is not merely iphone15pro vs iphone15 composition."
        "\n"
        "\n5. Large |Spearman rho| between pad_total_r512 and frozen "
        "CNN scores would make a causal padding perturbation worth "
        "running next."
        "\n"
        "\n6. Geometry discrimination alone does NOT prove the CNN "
        "uses padding. Only a controlled padding intervention can "
        "establish that."
        "\n"
        "\nStop here before changing preprocessing or retraining."
    )


if __name__ == "__main__":
    main()