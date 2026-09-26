#!/usr/bin/env python3
"""
Stage 45 — Final-1330 scientific-state audit.

Input:
  Stage-44 canonical master manifest (1330 images)

Checks / recomputes:
  - NPZ keys
  - dtypes / shapes
  - finite values
  - union-mask geometry and binary state
  - A_union
  - E_clean and E_adv directly from transferred maps
  - agreement with result.json
  - exact final adversarial tensor perturbation:
      physical Linf
      physical L2
      physical RMS
  - agreement of recomputed Linf with result.json
  - provisional Pointing-Game diagnostics

Does NOT calculate final RRA/DCEC/DCEW.
Those require the exact frozen RRA implementation / thresholds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "final_1330"
)

MASTER = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseAB_available"
    / "44_available_master_manifest.csv"
)

FULL_SELECTION = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_attack"
    / "full_population"
    / "full_population_selection.csv"
)

OUT_CSV = (
    ANALYSIS_ROOT
    / "45_final1330_scientific_state_audit.csv"
)

OUT_JSON = (
    ANALYSIS_ROOT
    / "45_final1330_scientific_state_summary.json"
)


EXPECTED_KEYS = {
    "adv_x_model",
    "clean_anomaly_map",
    "adv_anomaly_map",
    "altered_union_mask",
}

A_ATOL = 1e-12
E_ATOL = 5e-6
LINF_ATOL = 2e-7


def finite(name, x):
    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{name} contains non-finite values"
        )


def metric_E(S, M):
    denominator = float(
        np.sum(
            S,
            dtype=np.float64,
        )
    )

    if (
        not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise RuntimeError(
            f"Invalid map mass: {denominator}"
        )

    numerator = float(
        np.sum(
            S * M,
            dtype=np.float64,
        )
    )

    return numerator / denominator


def pointing_any_max(S, M):
    """
    Tie-safe diagnostic:
    PASS if at least one global-max pixel lies inside M.
    """

    mx = float(
        np.max(S)
    )

    inside = S[
        M > 0.5
    ]

    if inside.size == 0:
        return False

    return bool(
        np.max(inside) == mx
    )


def main():

    if not MASTER.is_file():
        raise RuntimeError(
            f"Missing master manifest: {MASTER}"
        )

    if not FULL_SELECTION.is_file():
        raise RuntimeError(
            f"Missing selection: {FULL_SELECTION}"
        )


    master = pd.read_csv(
        MASTER,
        keep_default_na=False,
    )

    selection = pd.read_csv(
        FULL_SELECTION,
        keep_default_na=False,
    )


    if len(master) != 1330:
        raise RuntimeError(
            f"Expected final 1330 rows, "
            f"found {len(master)}"
        )

    if master[
        "image_path"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path in Phase-A master"
        )


    selection_by = (
        selection
        .set_index(
            "image_path",
            drop=False,
        )
    )


    rows = []

    failures = []


    print(
        "TRUFOR STAGE 45 — "
        "FINAL-1330 SCIENTIFIC-STATE AUDIT"
    )

    print(
        "images:",
        len(master),
    )


    for n, (_, mrow) in enumerate(
        master.iterrows(),
        start=1,
    ):

        image_path = str(
            mrow["image_path"]
        )

        try:

            if image_path not in selection_by.index:
                raise RuntimeError(
                    "image missing from frozen selection"
                )

            sel = selection_by.loc[
                image_path
            ]


            npz_path = Path(
                str(
                    mrow[
                        "central_npz_path"
                    ]
                )
            )

            result_path = Path(
                str(
                    mrow[
                        "central_result_json"
                    ]
                )
            )


            if not npz_path.is_file():
                raise RuntimeError(
                    f"NPZ missing: {npz_path}"
                )

            if not result_path.is_file():
                raise RuntimeError(
                    f"result.json missing: "
                    f"{result_path}"
                )


            result = json.loads(
                result_path.read_text()
            )


            with np.load(
                npz_path,
                allow_pickle=False,
            ) as z:

                keys = set(
                    z.files
                )

                if keys != EXPECTED_KEYS:
                    raise RuntimeError(
                        f"NPZ keys mismatch: {keys}"
                    )


                adv_x = np.asarray(
                    z[
                        "adv_x_model"
                    ]
                )

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

                mask = np.asarray(
                    z[
                        "altered_union_mask"
                    ]
                )


                finite(
                    "adv_x_model",
                    adv_x,
                )

                finite(
                    "clean_anomaly_map",
                    clean_map,
                )

                finite(
                    "adv_anomaly_map",
                    adv_map,
                )

                finite(
                    "altered_union_mask",
                    mask,
                )


                if (
                    adv_x.ndim != 3
                    or adv_x.shape[0] != 3
                ):
                    raise RuntimeError(
                        f"bad adv_x shape: "
                        f"{adv_x.shape}"
                    )


                H = int(
                    sel[
                        "native_height"
                    ]
                )

                W = int(
                    sel[
                        "native_width"
                    ]
                )


                if (
                    adv_x.shape
                    != (3, H, W)
                ):
                    raise RuntimeError(
                        f"adv_x shape mismatch: "
                        f"{adv_x.shape} "
                        f"vs (3,{H},{W})"
                    )


                if (
                    clean_map.shape
                    != (H, W)
                    or adv_map.shape
                    != (H, W)
                    or mask.shape
                    != (H, W)
                ):
                    raise RuntimeError(
                        "map/mask geometry mismatch: "
                        f"clean={clean_map.shape}, "
                        f"adv={adv_map.shape}, "
                        f"mask={mask.shape}, "
                        f"expected={(H,W)}"
                    )


                mask_min = float(
                    np.min(mask)
                )

                mask_max = float(
                    np.max(mask)
                )


                nonbinary = int(
                    np.count_nonzero(
                        (mask != 0)
                        & (mask != 1)
                    )
                )


                if nonbinary:
                    raise RuntimeError(
                        f"union mask has "
                        f"{nonbinary} "
                        f"non-binary pixels"
                    )


                A = float(
                    np.mean(
                        mask,
                        dtype=np.float64,
                    )
                )


                frozen_A = float(
                    sel[
                        "A_union_stage4"
                    ]
                )


                if (
                    abs(
                        A
                        - frozen_A
                    )
                    > A_ATOL
                ):
                    raise RuntimeError(
                        "A_union mismatch: "
                        f"runtime={A}, "
                        f"frozen={frozen_A}"
                    )


                E_clean = metric_E(
                    clean_map,
                    mask,
                )

                E_adv = metric_E(
                    adv_map,
                    mask,
                )


                result_E_clean = float(
                    result[
                        "clean"
                    ][
                        "E"
                    ]
                )

                result_E_adv = float(
                    result[
                        "adversarial"
                    ][
                        "E"
                    ]
                )


                clean_E_abs_error = abs(
                    E_clean
                    - result_E_clean
                )

                adv_E_abs_error = abs(
                    E_adv
                    - result_E_adv
                )


                if (
                    clean_E_abs_error
                    > E_ATOL
                ):
                    raise RuntimeError(
                        "clean E recomputation mismatch: "
                        f"{clean_E_abs_error}"
                    )


                if (
                    adv_E_abs_error
                    > E_ATOL
                ):
                    raise RuntimeError(
                        "adv E recomputation mismatch: "
                        f"{adv_E_abs_error}"
                    )


                cache_path = (
                    ROOT
                    / str(
                        sel[
                            "cache_path"
                        ]
                    )
                )


                if not cache_path.is_file():
                    raise RuntimeError(
                        f"cache image missing: "
                        f"{cache_path}"
                    )


                with Image.open(
                    cache_path
                ) as im:

                    rgb = np.asarray(
                        im.convert(
                            "RGB"
                        ),
                        dtype=np.uint8,
                    )


                if (
                    rgb.shape[:2]
                    != (H, W)
                ):
                    raise RuntimeError(
                        "cache image geometry mismatch"
                    )


                clean_x = (
                    rgb
                    .astype(
                        np.float32
                    )
                    .transpose(
                        2,
                        0,
                        1,
                    )
                    / 256.0
                )


                delta_model = (
                    adv_x.astype(
                        np.float64
                    )
                    - clean_x.astype(
                        np.float64
                    )
                )


                # TruFor coordinates:
                # x = z * 255/256
                # therefore:
                # delta_z = delta_x * 256/255

                delta_physical = (
                    delta_model
                    * (256.0 / 255.0)
                )


                linf = float(
                    np.max(
                        np.abs(
                            delta_physical
                        )
                    )
                )


                l2 = float(
                    np.sqrt(
                        np.sum(
                            delta_physical
                            * delta_physical,
                            dtype=np.float64,
                        )
                    )
                )


                rms = float(
                    np.sqrt(
                        np.mean(
                            delta_physical
                            * delta_physical,
                            dtype=np.float64,
                        )
                    )
                )


                result_linf = float(
                    result[
                        "adversarial"
                    ][
                        "physical_linf"
                    ]
                )


                linf_abs_error = abs(
                    linf
                    - result_linf
                )


                if (
                    linf_abs_error
                    > LINF_ATOL
                ):
                    raise RuntimeError(
                        "Linf recomputation mismatch: "
                        f"{linf_abs_error}"
                    )


                K = int(
                    np.count_nonzero(
                        mask
                    )
                )


                row = {
                    "image_path":
                        image_path,

                    "source_host":
                        mrow[
                            "source_host"
                        ],

                    "execution_generation":
                        mrow[
                            "execution_generation"
                        ],

                    "execution_shard_id":
                        mrow[
                            "execution_shard_id"
                        ],

                    "eval_split":
                        str(
                            sel[
                                "eval_split"
                            ]
                        ),

                    "variant":
                        str(
                            sel[
                                "variant"
                            ]
                        ),

                    "hardware_source":
                        str(
                            sel[
                                "hardware_source"
                            ]
                        ),

                    "file_stem":
                        str(
                            sel[
                                "file_stem"
                            ]
                        ),

                    "native_height":
                        H,

                    "native_width":
                        W,

                    "native_pixels":
                        H * W,

                    "mask_pixels_K":
                        K,

                    "A_union":
                        A,

                    "E_clean_recomputed":
                        E_clean,

                    "E_adv_recomputed":
                        E_adv,

                    "delta_E_adv_minus_clean":
                        E_adv
                        - E_clean,

                    "relative_E_degradation":
                        (
                            (
                                E_clean
                                - E_adv
                            )
                            / E_clean
                            if E_clean != 0
                            else np.nan
                        ),

                    "mu_clean":
                        (
                            E_clean / A
                            if A > 0
                            else np.nan
                        ),

                    "mu_adv":
                        (
                            E_adv / A
                            if A > 0
                            else np.nan
                        ),

                    "clean_E_result":
                        result_E_clean,

                    "adv_E_result":
                        result_E_adv,

                    "clean_E_abs_error":
                        clean_E_abs_error,

                    "adv_E_abs_error":
                        adv_E_abs_error,

                    "clean_map_dtype":
                        str(
                            clean_map.dtype
                        ),

                    "adv_map_dtype":
                        str(
                            adv_map.dtype
                        ),

                    "adv_x_dtype":
                        str(
                            adv_x.dtype
                        ),

                    "mask_dtype":
                        str(
                            mask.dtype
                        ),

                    "clean_map_min":
                        float(
                            np.min(
                                clean_map
                            )
                        ),

                    "clean_map_max":
                        float(
                            np.max(
                                clean_map
                            )
                        ),

                    "adv_map_min":
                        float(
                            np.min(
                                adv_map
                            )
                        ),

                    "adv_map_max":
                        float(
                            np.max(
                                adv_map
                            )
                        ),

                    "mask_min":
                        mask_min,

                    "mask_max":
                        mask_max,

                    # Provisional Pointing Game.
                    # Final report definition can be
                    # frozen separately.
                    "PG_clean_anymax":
                        pointing_any_max(
                            clean_map,
                            mask,
                        ),

                    "PG_adv_anymax":
                        pointing_any_max(
                            adv_map,
                            mask,
                        ),

                    "adv_score":
                        float(
                            result[
                                "adversarial"
                            ][
                                "score"
                            ]
                        ),

                    "clean_score":
                        float(
                            result[
                                "clean"
                            ][
                                "score"
                            ]
                        ),

                    "physical_linf_recomputed":
                        linf,

                    "physical_linf_result":
                        result_linf,

                    "physical_linf_abs_error":
                        linf_abs_error,

                    "physical_l2_recomputed":
                        l2,

                    "physical_rms_recomputed":
                        rms,

                    "central_npz_path":
                        str(
                            npz_path
                        ),
                }


                rows.append(
                    row
                )


        except Exception as exc:

            failures.append(
                {
                    "image_path":
                        image_path,

                    "exception_type":
                        type(
                            exc
                        ).__name__,

                    "exception":
                        str(
                            exc
                        ),
                }
            )


        if (
            n % 25 == 0
            or n == len(master)
        ):
            print(
                f"audited "
                f"{n}/{len(master)}"
            )


    audit = pd.DataFrame(
        rows
    )


    audit.to_csv(
        OUT_CSV,
        index=False,
    )


    summary = {
        "status":
            (
                "PASS"
                if not failures
                else "FAIL"
            ),

        "expected_images":
            1330,

        "audited_images":
            int(
                len(audit)
            ),

        "failures":
            failures,

        "map_dtype_counts_clean":
            (
                audit[
                    "clean_map_dtype"
                ]
                .value_counts()
                .to_dict()
                if len(audit)
                else {}
            ),

        "map_dtype_counts_adv":
            (
                audit[
                    "adv_map_dtype"
                ]
                .value_counts()
                .to_dict()
                if len(audit)
                else {}
            ),

        "adv_x_dtype_counts":
            (
                audit[
                    "adv_x_dtype"
                ]
                .value_counts()
                .to_dict()
                if len(audit)
                else {}
            ),

        "max_clean_E_abs_error":
            (
                float(
                    audit[
                        "clean_E_abs_error"
                    ].max()
                )
                if len(audit)
                else None
            ),

        "max_adv_E_abs_error":
            (
                float(
                    audit[
                        "adv_E_abs_error"
                    ].max()
                )
                if len(audit)
                else None
            ),

        "max_linf_abs_error":
            (
                float(
                    audit[
                        "physical_linf_abs_error"
                    ].max()
                )
                if len(audit)
                else None
            ),

        "max_physical_linf":
            (
                float(
                    audit[
                        "physical_linf_recomputed"
                    ].max()
                )
                if len(audit)
                else None
            ),

        "mean_physical_l2":
            (
                float(
                    audit[
                        "physical_l2_recomputed"
                    ].mean()
                )
                if len(audit)
                else None
            ),

        "mean_E_clean":
            (
                float(
                    audit[
                        "E_clean_recomputed"
                    ].mean()
                )
                if len(audit)
                else None
            ),

        "mean_E_adv":
            (
                float(
                    audit[
                        "E_adv_recomputed"
                    ].mean()
                )
                if len(audit)
                else None
            ),

        "median_relative_E_degradation":
            (
                float(
                    audit[
                        "relative_E_degradation"
                    ].median()
                )
                if len(audit)
                else None
            ),
    }


    OUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()

    print(
        "===== RESULT ====="
    )

    print(
        "expected:",
        1330,
    )

    print(
        "audited :",
        len(audit),
    )

    print(
        "failures:",
        len(failures),
    )


    if len(audit):

        print(
            "max clean E error:",
            audit[
                "clean_E_abs_error"
            ].max(),
        )

        print(
            "max adv E error:",
            audit[
                "adv_E_abs_error"
            ].max(),
        )

        print(
            "max Linf error:",
            audit[
                "physical_linf_abs_error"
            ].max(),
        )

        print(
            "max physical Linf:",
            audit[
                "physical_linf_recomputed"
            ].max(),
        )


    print(
        "audit CSV:",
        OUT_CSV,
    )

    print(
        "summary:",
        OUT_JSON,
    )


    if failures:

        print()

        print(
            "FIRST FAILURES"
        )

        for f in failures[:20]:
            print(
                f
            )

        raise SystemExit(
            "STAGE 45 FAIL"
        )


    print()

    print(
        "STAGE 45 PASS"
    )


if __name__ == "__main__":
    main()
