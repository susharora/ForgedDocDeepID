#!/usr/bin/env python3
"""Stage 07: concise memory/gradient-gate summary for the TruFor pilot.

This stage makes no new model calls. It reports the empirical evidence needed
to decide whether the full attack should remain on Home PC or move to Lab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trufor_attack_pilot_common import pilot_root, safe_run_tag


def fmt(x, digits=2):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="HOMEPC")
    args = parser.parse_args()

    tag = safe_run_tag(args.run_tag)
    out_root = pilot_root(tag)
    results_path = out_root / "vram_pilot_results.csv"
    if not results_path.is_file():
        raise RuntimeError(f"Missing Stage-06 results: {results_path}")

    df = pd.read_csv(results_path, keep_default_na=False)
    if df.empty:
        raise RuntimeError("Stage-06 results are empty")

    rows = []
    for _, r in df.sort_values("pilot_order").iterrows():
        status = str(r.get("status", ""))
        rows.append(
            {
                "order": int(r["pilot_order"]),
                "family": str(r["variant"]),
                "WxH": f"{int(r['native_width'])}x{int(r['native_height'])}",
                "MP": int(r["native_pixels"]) / 1e6,
                "status": status,
                "oom_phase": str(r.get("oom_phase", "")),
                "peak_reserved_GiB": (
                    float(r["autograd_peak_reserved_gib"])
                    if status == "PASS" and str(r.get("autograd_peak_reserved_gib", "")) != ""
                    else np.nan
                ),
                "peak_allocated_GiB": (
                    float(r["autograd_peak_allocated_gib"])
                    if status == "PASS" and str(r.get("autograd_peak_allocated_gib", "")) != ""
                    else np.nan
                ),
                "cls_preserved": (
                    str(r.get("classification_preserved", ""))
                    if status == "PASS"
                    else ""
                ),
                "E_drop": (
                    float(r["E_change_clean_minus_candidate"])
                    if status == "PASS" and str(r.get("E_change_clean_minus_candidate", "")) != ""
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    pass_df = df.loc[df["status"].astype(str) == "PASS"].copy()
    oom_df = df.loc[df["status"].astype(str) == "OOM"].copy()

    lines = [
        "TRUFOR NATIVE VRAM / INPUT-GRADIENT PILOT SUMMARY",
        "",
        summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
        "",
        f"attempted: {len(df)}",
        f"PASS:      {len(pass_df)}",
        f"OOM:       {len(oom_df)}",
    ]

    if len(pass_df):
        max_reserved = pd.to_numeric(
            pass_df["autograd_peak_reserved_gib"], errors="coerce"
        ).max()
        max_alloc = pd.to_numeric(
            pass_df["autograd_peak_allocated_gib"], errors="coerce"
        ).max()
        lines += [
            f"max autograd peak reserved:  {max_reserved:.3f} GiB",
            f"max autograd peak allocated: {max_alloc:.3f} GiB",
        ]

        if "gradient_map_parity_max_abs_error" in pass_df:
            lines.append(
                "max clean gradient-forward map parity error: "
                f"{pd.to_numeric(pass_df['gradient_map_parity_max_abs_error'], errors='coerce').max():.9g}"
            )
        if "gradient_score_parity_abs_error" in pass_df:
            lines.append(
                "max clean gradient-forward score parity error: "
                f"{pd.to_numeric(pass_df['gradient_score_parity_abs_error'], errors='coerce').max():.9g}"
            )

    if len(oom_df):
        lines += [
            "",
            "MEMORY GATE:",
            "At least one legitimate frozen clean-correct image OOMed.",
            "Do NOT resize/crop/omit it to keep the Home run alive.",
            "This is evidence to move the complete canonical attack experiment to the 48-GB Lab GPU.",
        ]
    elif len(df) == 6 and len(pass_df) == 6:
        lines += [
            "",
            "MEMORY GATE:",
            "All six worst-case native-resolution images completed a full input-gradient backward pass.",
            "Home PC is empirically memory-feasible for the next multi-step attack pilot.",
            "This does NOT yet freeze the final optimiser; inspect peak headroom and one-step behaviour first.",
        ]
    else:
        lines += [
            "",
            "MEMORY GATE:",
            "The six-image gate is incomplete. Do not decide Home vs Lab yet.",
        ]

    lines += [
        "",
        "STOP HERE.",
        "Paste this summary before implementing/running the multi-step classification-preserving TruFor attack.",
    ]

    report = "\n".join(lines) + "\n"
    report_path = out_root / "vram_pilot_summary.txt"
    report_path.write_text(report)
    print(report)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
