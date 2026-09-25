from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RRAResult:
    rra: float
    rra_tie_expected: float
    rra_tie_min: float
    rra_tie_max: float
    cutoff: float
    k: int
    n_valid: int
    n_strictly_above: int
    n_equal_cutoff: int
    n_needed_from_tie: int
    cutoff_tie_crosses_boundary: bool


def relevance_rank_accuracy(
    relevance: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> RRAResult:
    """
    CLEVR-XAI-style Relevance Rank Accuracy.

    K = number of GT pixels in valid content.
    Select the K highest-relevance valid pixels.
    RRA = selected GT pixels / K.

    Cutoff ties:
      primary = ascending flattened valid-pixel index;
      additionally return expected/min/max tie-resolved RRA.
    """

    r = np.asarray(relevance)
    gt = np.asarray(gt_mask, dtype=bool)

    if r.ndim != 2:
        raise ValueError(
            f"relevance must be 2-D, got {r.shape}"
        )

    if gt.shape != r.shape:
        raise ValueError(
            f"GT shape {gt.shape} != relevance {r.shape}"
        )

    if valid_mask is None:
        valid = np.ones(
            r.shape,
            dtype=bool,
        )
    else:
        valid = np.asarray(
            valid_mask,
            dtype=bool,
        )

        if valid.shape != r.shape:
            raise ValueError(
                f"valid shape {valid.shape} != relevance {r.shape}"
            )

    if not np.isfinite(r[valid]).all():
        raise ValueError(
            "non-finite relevance values"
        )

    gt = gt & valid

    K = int(
        np.count_nonzero(gt)
    )

    N = int(
        np.count_nonzero(valid)
    )

    if K <= 0:
        raise ValueError(
            "GT mask contains zero valid pixels"
        )

    if K > N:
        raise ValueError(
            f"K={K} > N={N}"
        )

    values = r[valid].astype(
        np.float64,
        copy=False,
    )

    gt_valid = gt[valid]

    # K-th largest value:
    # ascending partition position N-K.
    partition_index = N - K

    cutoff = float(
        np.partition(
            values,
            partition_index,
        )[partition_index]
    )

    above = values > cutoff
    equal = values == cutoff

    n_above = int(
        np.count_nonzero(above)
    )

    n_equal = int(
        np.count_nonzero(equal)
    )

    need = K - n_above

    if need < 0 or need > n_equal:
        raise RuntimeError(
            "internal top-K accounting failure: "
            f"K={K}, above={n_above}, "
            f"equal={n_equal}, need={need}"
        )

    inside_above = int(
        np.count_nonzero(
            gt_valid & above
        )
    )

    equal_indices = np.flatnonzero(
        equal
    )

    # Deterministic primary tie rule:
    # np.flatnonzero is ascending in flattened-valid order.
    chosen_equal = (
        equal_indices[:need]
        if need
        else equal_indices[:0]
    )

    inside_chosen_equal = int(
        np.count_nonzero(
            gt_valid[
                chosen_equal
            ]
        )
    )

    primary_inside = (
        inside_above
        + inside_chosen_equal
    )

    inside_equal = int(
        np.count_nonzero(
            gt_valid & equal
        )
    )

    outside_equal = (
        n_equal
        - inside_equal
    )

    min_inside_from_tie = max(
        0,
        need - outside_equal,
    )

    max_inside_from_tie = min(
        need,
        inside_equal,
    )

    expected_inside_from_tie = (
        need
        * inside_equal
        / n_equal
        if n_equal
        else 0.0
    )

    tie_crosses = bool(
        need > 0
        and n_equal > need
    )

    return RRAResult(
        rra=float(
            primary_inside / K
        ),
        rra_tie_expected=float(
            (
                inside_above
                + expected_inside_from_tie
            )
            / K
        ),
        rra_tie_min=float(
            (
                inside_above
                + min_inside_from_tie
            )
            / K
        ),
        rra_tie_max=float(
            (
                inside_above
                + max_inside_from_tie
            )
            / K
        ),
        cutoff=cutoff,
        k=K,
        n_valid=N,
        n_strictly_above=n_above,
        n_equal_cutoff=n_equal,
        n_needed_from_tie=need,
        cutoff_tie_crosses_boundary=tie_crosses,
    )
