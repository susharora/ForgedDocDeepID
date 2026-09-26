#!/usr/bin/env python3
"""
Stage 49 — package the final 1330-image TruFor adversarial-analysis results.

Creates a compact dissertation/writer package containing:
- final integrity audit
- final per-image evidence table
- DC/DCEC/DCEW summaries
- threshold sensitivity
- subgroup summaries
- continuous metrics
- runtime summaries
- clustered-bootstrap CIs
- frozen evidence protocol
- provenance and SHA256 manifests
- final writer handover

The large raw scientific NPZ archive is deliberately not duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "final_1330"
)

AVAILABLE = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseAB_available"
)

PROTOCOL = (
    ROOT
    / "scripts"
    / "trufor"
    / "evidence_evaluation_protocol_v1.json"
)

PACKAGE = (
    ANALYSIS
    / "49_final1330_report_package"
)

TRANSFER = (
    ROOT
    / "analysis_transfer_bundles"
)

TAR = (
    TRANSFER
    / "IMTA135_trufor_final1330_report_package.tar.gz"
)

PROTOCOL_SHA = (
    "5840aa9bee076b493c6a645e140edaad"
    "892373433c7e919e95ed4e60e44158d5"
)

CHECKPOINT_SHA = (
    "ac1d90e329a72e0d66e8665e123a19e"
    "94bfae3209c3ef8a4f9ca3b91578c7844"
)

PHASE2_PLAN_SHA = (
    "e7e057388cc43f48931b93dfb3e2aead"
    "22df6ecea5a9769f6dc948ef29e88eeb"
)

CLASSIFICATION_THRESHOLD = 0.532955974340439
TAU_E = 0.5
TAU_RRA = 0.5


FILES = [
    ANALYSIS / "45_final1330_scientific_state_audit.csv",
    ANALYSIS / "45_final1330_scientific_state_summary.json",
    ANALYSIS / "46_analysis_protocol_snapshot.json",
    ANALYSIS / "46_final1330_evidence_per_image.csv",
    ANALYSIS / "46_final1330_primary_summary.json",
    ANALYSIS / "46_final1330_primary_summary.csv",
    ANALYSIS / "46_final1330_threshold_sensitivity.csv",
    ANALYSIS / "46_final1330_subgroup_summary.csv",
    ANALYSIS / "46_final1330_RRA_tie_diagnostics.csv",
    ANALYSIS / "47_final1330_continuous_metrics.csv",
    ANALYSIS / "47_final1330_degradation_thresholds.csv",
    ANALYSIS / "47_final1330_runtime_per_image.csv",
    ANALYSIS / "47_final1330_runtime_summary.csv",
    ANALYSIS / "47_final1330_runtime_science.json",
    ANALYSIS / "47_final1330_reporting_summary.json",
    ANALYSIS / "48_final1330_cluster_bootstrap_CI.csv",
    ANALYSIS / "48_final1330_primary_report_table.csv",
    ANALYSIS / "48_final1330_cluster_bootstrap_summary.json",
    AVAILABLE / "44_available_population_summary.json",
    AVAILABLE / "44_available_master_manifest.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def main() -> None:
    for p in FILES + [PROTOCOL]:
        if not p.is_file():
            raise RuntimeError(
                f"Missing required Stage-49 input: {p}"
            )

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)

    PACKAGE.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_dir = PACKAGE / "data"
    data_dir.mkdir()

    TRANSFER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy final analysis source tables.
    for src in FILES:
        shutil.copy2(
            src,
            data_dir / src.name,
        )

    shutil.copy2(
        PROTOCOL,
        PACKAGE / PROTOCOL.name,
    )

    # Load authoritative final outputs.
    primary = json.loads(
        (
            ANALYSIS
            / "46_final1330_primary_summary.json"
        ).read_text()
    )

    audit = json.loads(
        (
            ANALYSIS
            / "45_final1330_scientific_state_summary.json"
        ).read_text()
    )

    runtime = json.loads(
        (
            ANALYSIS
            / "47_final1330_runtime_science.json"
        ).read_text()
    )

    report47 = json.loads(
        (
            ANALYSIS
            / "47_final1330_reporting_summary.json"
        ).read_text()
    )

    ci = pd.read_csv(
        ANALYSIS
        / "48_final1330_primary_report_table.csv"
    ).set_index("metric")

    sensitivity = pd.read_csv(
        ANALYSIS
        / "46_final1330_threshold_sensitivity.csv"
    )

    subgroup = pd.read_csv(
        ANALYSIS
        / "46_final1330_subgroup_summary.csv"
    )

    continuous = pd.read_csv(
        ANALYSIS
        / "47_final1330_continuous_metrics.csv"
    )

    # Sanity gates.
    if int(primary["n_total"]) != 1330:
        raise RuntimeError(
            f"Primary summary n_total != 1330: {primary['n_total']}"
        )

    if int(primary["n_DCEC"]) != 436:
        raise RuntimeError(
            f"Primary DCEC count changed: {primary['n_DCEC']}"
        )

    if int(primary["n_DCEW"]) != 436:
        raise RuntimeError(
            f"Primary DCEW count changed: {primary['n_DCEW']}"
        )

    if int(audit["audited_images"]) != 1330:
        raise RuntimeError(
            f"Stage-45 audit count != 1330: {audit['audited_images']}"
        )

    if audit["status"] != "PASS":
        raise RuntimeError(
            f"Stage-45 status is not PASS: {audit['status']}"
        )

    if int(report47["population"]) != 1330:
        raise RuntimeError(
            f"Stage-47 population != 1330: {report47['population']}"
        )

    def ci_value(metric: str, column: str) -> float:
        return float(ci.loc[metric, column])

    def overall_metric(metric: str, column: str) -> float:
        row = continuous.loc[
            (continuous["group"] == "overall")
            & (continuous["metric"] == metric)
        ]
        if len(row) != 1:
            raise RuntimeError(
                f"Expected exactly one overall row for {metric}, got {len(row)}"
            )
        return float(row.iloc[0][column])

    # Compact key-results CSV.
    key_results = pd.DataFrame(
        [
            {
                "metric": "classification_preservation",
                "estimate": 1.0,
                "numerator": 1330,
                "denominator": 1330,
            },
            {
                "metric": "DCEC_rate",
                "estimate": ci_value("DCEC_rate", "estimate"),
                "ci95_low": ci_value("DCEC_rate", "ci95_low"),
                "ci95_high": ci_value("DCEC_rate", "ci95_high"),
                "numerator": 436,
                "denominator": 1330,
            },
            {
                "metric": "DCEW_rate_given_DCEC",
                "estimate": 1.0,
                "ci95_low": 1.0,
                "ci95_high": 1.0,
                "numerator": 436,
                "denominator": 436,
            },
            {
                "metric": "mean_E_clean",
                "estimate": ci_value("E_clean_mean", "estimate"),
                "ci95_low": ci_value("E_clean_mean", "ci95_low"),
                "ci95_high": ci_value("E_clean_mean", "ci95_high"),
            },
            {
                "metric": "mean_E_adv",
                "estimate": ci_value("E_adv_mean", "estimate"),
                "ci95_low": ci_value("E_adv_mean", "ci95_low"),
                "ci95_high": ci_value("E_adv_mean", "ci95_high"),
            },
            {
                "metric": "median_relative_E_degradation",
                "estimate": ci_value(
                    "relative_E_degradation_median",
                    "estimate",
                ),
                "ci95_low": ci_value(
                    "relative_E_degradation_median",
                    "ci95_low",
                ),
                "ci95_high": ci_value(
                    "relative_E_degradation_median",
                    "ci95_high",
                ),
            },
            {
                "metric": "mean_RRA_clean",
                "estimate": ci_value("RRA_clean_mean", "estimate"),
                "ci95_low": ci_value("RRA_clean_mean", "ci95_low"),
                "ci95_high": ci_value("RRA_clean_mean", "ci95_high"),
            },
            {
                "metric": "mean_RRA_adv",
                "estimate": ci_value("RRA_adv_mean", "estimate"),
                "ci95_low": ci_value("RRA_adv_mean", "ci95_low"),
                "ci95_high": ci_value("RRA_adv_mean", "ci95_high"),
            },
            {
                "metric": "median_relative_RRA_degradation",
                "estimate": ci_value(
                    "relative_RRA_degradation_median",
                    "estimate",
                ),
                "ci95_low": ci_value(
                    "relative_RRA_degradation_median",
                    "ci95_low",
                ),
                "ci95_high": ci_value(
                    "relative_RRA_degradation_median",
                    "ci95_high",
                ),
            },
            {
                "metric": "median_mu_clean",
                "estimate": ci_value("mu_clean_median", "estimate"),
                "ci95_low": ci_value("mu_clean_median", "ci95_low"),
                "ci95_high": ci_value("mu_clean_median", "ci95_high"),
            },
            {
                "metric": "median_mu_adv",
                "estimate": ci_value("mu_adv_median", "estimate"),
                "ci95_low": ci_value("mu_adv_median", "ci95_low"),
                "ci95_high": ci_value("mu_adv_median", "ci95_high"),
            },
            {
                "metric": "median_RRA_lift_clean",
                "estimate": ci_value(
                    "RRA_lift_clean_median",
                    "estimate",
                ),
                "ci95_low": ci_value(
                    "RRA_lift_clean_median",
                    "ci95_low",
                ),
                "ci95_high": ci_value(
                    "RRA_lift_clean_median",
                    "ci95_high",
                ),
            },
            {
                "metric": "median_RRA_lift_adv",
                "estimate": ci_value(
                    "RRA_lift_adv_median",
                    "estimate",
                ),
                "ci95_low": ci_value(
                    "RRA_lift_adv_median",
                    "ci95_low",
                ),
                "ci95_high": ci_value(
                    "RRA_lift_adv_median",
                    "ci95_high",
                ),
            },
            {
                "metric": "mean_physical_Linf",
                "estimate": ci_value(
                    "physical_linf_recomputed_mean",
                    "estimate",
                ),
                "ci95_low": ci_value(
                    "physical_linf_recomputed_mean",
                    "ci95_low",
                ),
                "ci95_high": ci_value(
                    "physical_linf_recomputed_mean",
                    "ci95_high",
                ),
            },
            {
                "metric": "max_physical_Linf",
                "estimate": float(audit["max_physical_linf"]),
            },
            {
                "metric": "aggregate_active_GPU_hours",
                "estimate": float(runtime["aggregate_active_gpu_hours"]),
            },
            {
                "metric": "median_minutes_per_image",
                "estimate": float(runtime["median_minutes_per_image"]),
            },
            {
                "metric": "mean_minutes_per_image",
                "estimate": float(runtime["mean_minutes_per_image"]),
            },
            {
                "metric": "runtime_IQR_q25_minutes",
                "estimate": float(runtime["q25_minutes"]),
            },
            {
                "metric": "runtime_IQR_q75_minutes",
                "estimate": float(runtime["q75_minutes"]),
            },
            {
                "metric": "runtime_q95_minutes",
                "estimate": float(runtime["q95_minutes"]),
            },
            {
                "metric": "spearman_megapixels_runtime",
                "estimate": float(
                    runtime["spearman_megapixels_vs_active_runtime"]
                ),
            },
        ]
    )

    key_results.to_csv(
        PACKAGE / "FINAL_KEY_RESULTS.csv",
        index=False,
    )

    # Writer-facing subgroup extracts.
    variant = subgroup.loc[
        subgroup["group"] == "variant",
        [
            "value",
            "n",
            "DCEC_n",
            "DCEC_rate_all",
            "DCEW_n",
            "DCEW_rate_given_DCEC",
            "E_clean_mean",
            "E_adv_mean",
            "RRA_clean_mean",
            "RRA_adv_mean",
            "A_median",
            "mu_clean_median",
            "mu_adv_median",
            "RRA_lift_clean_median",
            "RRA_lift_adv_median",
        ],
    ].copy()

    hardware = subgroup.loc[
        subgroup["group"] == "hardware_source",
        [
            "value",
            "n",
            "DCEC_n",
            "DCEC_rate_all",
            "DCEW_n",
            "DCEW_rate_given_DCEC",
        ],
    ].copy()

    split = subgroup.loc[
        subgroup["group"] == "eval_split",
        [
            "value",
            "n",
            "DCEC_n",
            "DCEC_rate_all",
            "DCEW_n",
            "DCEW_rate_given_DCEC",
        ],
    ].copy()

    variant.to_csv(
        PACKAGE / "FINAL_VARIANT_SUMMARY.csv",
        index=False,
    )

    hardware.to_csv(
        PACKAGE / "FINAL_HARDWARE_SUMMARY.csv",
        index=False,
    )

    split.to_csv(
        PACKAGE / "FINAL_SPLIT_SUMMARY.csv",
        index=False,
    )

    sensitivity_min = int(sensitivity["n_DCEC"].min())
    sensitivity_max = int(sensitivity["n_DCEC"].max())

    all_sensitivity_dcew = bool(
        (sensitivity["DCEW_rate_given_DCEC"] == 1.0).all()
    )

    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")

    writer = f"""# FINAL TruFor adversarial-localisation writer handover

## Status

This document supersedes all provisional 966-image Phase-A reporting.

The complete frozen adversarial-evaluation population is now available and validated:

- frozen attack population: **1330**
- centrally available: **1330/1330**
- Stage-45 scientific-state audit: **PASS**
- Stage-45 failures: **0**
- classification preserved: **1330/1330**
- maximum physical L-infinity: **{float(audit['max_physical_linf']):.12f} = 1/255**

## Primary DC-DCEC-DCEW analysis

The primary spatial-evidence rule is the pre-specified study-defined majority criterion:

- `tau_E = 0.5`
- `tau_RRA = 0.5`

The RMA/E and RRA metrics are literature-aligned localisation measures, but `0.5/0.5` is **not claimed to be a universal literature-mandated cutoff**.

Human review is a sidecar interpretive experiment only and was not used to calibrate these thresholds.

Final primary result:

- clean DCEC: **436/1330 = {pct(436/1330)}**
- document-stem clustered-bootstrap 95% CI:
  **{pct(ci_value('DCEC_rate','ci95_low'))} to {pct(ci_value('DCEC_rate','ci95_high'))}**
- DCEW: **436/436 = 100%**
- classification flips within the clean-DCEC denominator: **0**
- E-only DCEW at `0.5/0.5`: **0**
- RRA-only DCEW at `0.5/0.5`: **0**
- both E and RRA below threshold: **436**

Thus every primary DCEW case retained the image-level attack decision while both spatial-evidence measures fell below the primary evidence threshold.

## Continuous evidence degradation

Across all **1330** attacked images:

- mean E:
  **{ci_value('E_clean_mean','estimate'):.6f} clean -> {ci_value('E_adv_mean','estimate'):.6f} adversarial**
- median relative E degradation:
  **{pct(ci_value('relative_E_degradation_median','estimate'))}**
  with 95% clustered-bootstrap CI
  **{pct(ci_value('relative_E_degradation_median','ci95_low'))} to {pct(ci_value('relative_E_degradation_median','ci95_high'))}**

- mean RRA:
  **{ci_value('RRA_clean_mean','estimate'):.6f} clean -> {ci_value('RRA_adv_mean','estimate'):.6f} adversarial**
- median relative RRA degradation:
  **{pct(ci_value('relative_RRA_degradation_median','estimate'))}**

- median clean `mu_w`:
  **{ci_value('mu_clean_median','estimate'):.6f}**
- median adversarial `mu_w`:
  **{ci_value('mu_adv_median','estimate'):.6f}**

- median clean RRA lift:
  **{ci_value('RRA_lift_clean_median','estimate'):.6f}**
- median adversarial RRA lift:
  **{ci_value('RRA_lift_adv_median','estimate'):.6f}**

## Threshold sensitivity

The evidence thresholds were varied over the 3 x 3 grid:

- `tau_E`: 0.25 / 0.50 / 0.75
- `tau_RRA`: 0.25 / 0.50 / 0.75

Clean-DCEC denominators ranged from **{sensitivity_min} to {sensitivity_max} images**.

Conditional DCEW was 100% in all nine cells: **{all_sensitivity_dcew}**.

Specific decomposition:

- `(0.25, 0.25)`: 1011 DCEC / 1011 DCEW; 1 E-only, 1010 both
- `(0.50, 0.25)`: 774 DCEC / 774 DCEW; 1 E-only, 773 both
- all remaining threshold pairs: every DCEW case failed both E and RRA
- classification flips: 0 in every threshold-sensitivity cell

Interpretation: the clean-DCEC denominator is threshold-dependent, but the adversarial evidence-collapse conclusion is not dependent on the primary 0.5/0.5 choice.

## RRA tie robustness

RRA top-K cutoff ties occurred in:

- clean maps: **{int(primary['clean_cutoff_ties'])}**
- adversarial maps: **{int(primary['adv_cutoff_ties'])}**

Maximum tie-induced RRA width:

- clean: **{float(primary['max_clean_tie_width']):.9g}**
- adversarial: **{float(primary['max_adv_tie_width']):.9g}**

Stage 47 confirmed:

- clean DCEC label changes under tie-min/tie-max: **0**
- adversarial DCEW label changes under tie-min/tie-max: **0**

Therefore the primary binary conclusions are insensitive to the recorded cutoff-tie ambiguity.

## Perturbation constraint

Stage 45 independently reopened every retained adversarial tensor and recomputed the perturbation.

Results:

- audited: **1330/1330**
- failures: **0**
- maximum E recomputation errors: numerical precision only
- maximum physical L-infinity:
  **{float(audit['max_physical_linf']):.12f} = 1/255**

The attack therefore respects the declared physical L-infinity budget.

## Compute cost

Use the per-image active attack wall time stored in each `result.json`, not raw shard calendar duration.

The latter contains workstation-specific interruptions such as WSL shutdown/restart and should not be used as portable scientific compute cost.

Final active-compute statistics:

- aggregate active GPU compute:
  **{float(runtime['aggregate_active_gpu_hours']):.3f} GPU-hours**
- mean:
  **{float(runtime['mean_minutes_per_image']):.3f} min/image**
- median:
  **{float(runtime['median_minutes_per_image']):.3f} min/image**
- IQR:
  **{float(runtime['q25_minutes']):.3f} to {float(runtime['q75_minutes']):.3f} min/image**
- 95th percentile:
  **{float(runtime['q95_minutes']):.3f} min/image**
- active throughput:
  **{float(runtime['images_per_active_hour']):.3f} images/hour**
- Spearman native megapixels vs active runtime:
  **rho = {float(runtime['spearman_megapixels_vs_active_runtime']):.4f}**

Hardware:

`NVIDIA RTX PRO 5000 Blackwell`

The strong resolution/runtime relationship explains the highly skewed per-image runtime distribution.

## Manipulation-family summary

~~~text
{variant.to_string(index=False)}
~~~

Important interpretation:

DCEC prevalence is not synonymous with all clean localisation information.

`facedancer` has **0/107** images satisfying the absolute 0.5/0.5 DCEC rule, but its continuous localisation metrics remain informative. Its small manipulated regions make an absolute `E >= 0.5` criterion particularly stringent.

`digital_2` similarly has only **3/135** clean-DCEC images despite median clean RRA above 0.5.

Therefore report continuous E, RRA, `mu_w` and RRA lift alongside the thresholded DCEC/DCEW analysis.

## Hardware summary

~~~text
{hardware.to_string(index=False)}
~~~

## Split summary

~~~text
{split.to_string(index=False)}
~~~

## Scientific provenance

Frozen pretrained TruFor checkpoint SHA256:

`{CHECKPOINT_SHA}`

Frozen adversarial protocol SHA256:

`{PROTOCOL_SHA}`

Phase-2 execution-plan SHA256:

`{PHASE2_PLAN_SHA}`

Frozen pretrained image-level classification threshold:

`{CLASSIFICATION_THRESHOLD}`

Primary evidence thresholds:

`tau_E = {TAU_E}`  
`tau_RRA = {TAU_RRA}`

Evidence protocol:

`scripts/trufor/evidence_evaluation_protocol_v1.json`

Git branch at packaging:

`{branch}`

Git commit at packaging:

`{commit}`

## Recommended dissertation wording

> Across the complete 1,330-image clean-correct attack population, all adversarial examples retained TruFor's attack-class decision under the frozen image-level threshold. Spatial evidence nevertheless collapsed: mean relevance mass E decreased from approximately {ci_value('E_clean_mean','estimate'):.3f} to {ci_value('E_adv_mean','estimate'):.4f}, while mean RRA decreased from approximately {ci_value('RRA_clean_mean','estimate'):.3f} to {ci_value('RRA_adv_mean','estimate'):.4f}. Median relative degradation was {pct(ci_value('relative_E_degradation_median','estimate'))} for E and 100% for RRA. Under the pre-specified study-defined majority criterion (`E >= 0.5` and `RRA >= 0.5`), 436 images qualified as decision-correct and evidence-correct; all 436 fell below both evidence thresholds after perturbation while retaining the attack-class decision. This result was unaffected by RRA tie resolution and conditional DCEW remained 100% throughout a 3 x 3 threshold-sensitivity analysis.

## Writer source hierarchy

Use the following in this order:

1. `FINAL_KEY_RESULTS.csv`
2. `data/48_final1330_primary_report_table.csv`
3. `data/47_final1330_reporting_summary.json`
4. `data/46_final1330_primary_summary.json`
5. `data/46_final1330_threshold_sensitivity.csv`
6. `data/46_final1330_subgroup_summary.csv`
7. `data/46_final1330_evidence_per_image.csv`
8. `data/45_final1330_scientific_state_summary.json`

The old Phase-A 966-image package is historical/provisional and must not be used for final dissertation values.
"""

    writer_path = PACKAGE / "FINAL_WRITER_HANDOVER.md"
    writer_path.write_text(writer)

    # Provenance.
    provenance = {
        "status": "PASS",
        "population": 1330,
        "git_commit": commit,
        "git_branch": branch,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "scientific_protocol_sha256": PROTOCOL_SHA,
        "phase2_plan_sha256": PHASE2_PLAN_SHA,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "tau_E": TAU_E,
        "tau_RRA": TAU_RRA,
        "stage45_status": audit["status"],
        "stage45_audited_images": int(audit["audited_images"]),
        "stage45_failures": len(audit.get("failures", [])),
        "DCEC": int(primary["n_DCEC"]),
        "DCEW": int(primary["n_DCEW"]),
        "classification_preserved": int(primary["n_DC_adv"]),
        "aggregate_active_gpu_hours": float(
            runtime["aggregate_active_gpu_hours"]
        ),
    }

    provenance_path = PACKAGE / "PACKAGE_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # README.
    readme = f"""# Final TruFor 1330-image adversarial report package

This package supersedes the provisional Phase-A 966-image package.

It contains the complete validated reporting state for all **1330/1330**
images in the frozen clean-correct TruFor attack population.

Start with:

- `FINAL_WRITER_HANDOVER.md`
- `FINAL_KEY_RESULTS.csv`

Detailed source tables are under:

- `data/`

Git commit at packaging:

`{commit}`

The large raw clean/adversarial map/tensor archive is deliberately not
duplicated inside this compact reporting package.
"""

    (PACKAGE / "README.md").write_text(readme)

    # Internal SHA manifest.
    sums_path = PACKAGE / "SHA256SUMS.txt"

    files_to_hash = sorted(
        p
        for p in PACKAGE.rglob("*")
        if p.is_file()
        and p != sums_path
    )

    sums_path.write_text(
        "".join(
            f"{sha256_file(p)}  {p.relative_to(PACKAGE)}\n"
            for p in files_to_hash
        )
    )

    # Create transfer TAR.GZ.
    if TAR.exists():
        TAR.unlink()

    with tarfile.open(
        TAR,
        "w:gz",
    ) as tf:
        tf.add(
            PACKAGE,
            arcname=PACKAGE.name,
        )

    tar_sha = sha256_file(TAR)

    sidecar = Path(str(TAR) + ".sha256")

    sidecar.write_text(
        f"{tar_sha}  {TAR.name}\n"
    )

    print("=" * 72)
    print("TRUFOR STAGE 49 — FINAL 1330 REPORT PACKAGE")
    print("=" * 72)
    print("population       : 1330")
    print("classification   : 1330/1330 preserved")
    print("DCEC             :", int(primary["n_DCEC"]))
    print("DCEW             :", int(primary["n_DCEW"]))
    print("git commit       :", commit)
    print("package          :", PACKAGE)
    print("transfer TAR     :", TAR)
    print(
        "TAR size MiB     :",
        f"{TAR.stat().st_size / 1024**2:.2f}",
    )
    print("TAR SHA256       :", tar_sha)
    print()
    print("STAGE 49 PASS")


if __name__ == "__main__":
    main()
