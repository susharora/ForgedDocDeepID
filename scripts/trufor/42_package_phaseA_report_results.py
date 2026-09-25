#!/usr/bin/env python3
"""
Stage 42 — Package the validated TruFor Phase-A reporting state.

Creates a small, self-contained reporting bundle containing:
- evaluation protocol
- integrity/audit summaries
- per-image analysis table
- primary DC/DCEC/DCEW results
- threshold sensitivity
- subgroup results
- representativeness diagnostics
- clustered-bootstrap confidence intervals
- provenance/hashes
- writer handover

Does not copy the ~44 GiB scientific NPZ archive.
"""

from __future__ import annotations

import csv
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
    / "phaseA_966"
)

PROTOCOL = (
    ROOT
    / "scripts"
    / "trufor"
    / "evidence_evaluation_protocol_v1.json"
)

PACKAGE = (
    ANALYSIS
    / "42_phaseA_report_package"
)

TRANSFER = (
    ROOT
    / "analysis_transfer_bundles"
)

TAR = (
    TRANSFER
    / "IMTA135_trufor_phaseA_966_report_package.tar.gz"
)


FILES = [
    "phaseA_ingest_summary.json",

    "38_phaseA_scientific_state_summary.json",
    "38_phaseA_scientific_state_audit.csv",

    "39_analysis_protocol_snapshot.json",
    "39_phaseA_primary_summary.json",
    "39_phaseA_primary_summary.csv",
    "39_phaseA_evidence_per_image.csv",
    "39_phaseA_threshold_sensitivity.csv",
    "39_phaseA_subgroup_summary.csv",
    "39_phaseA_RRA_tie_diagnostics.csv",

    "40_phaseA_reporting_audit_summary.json",
    "40_phaseA_categorical_composition.csv",
    "40_phaseA_numeric_balance.csv",
    "40_phaseA_continuous_metrics.csv",
    "40_phaseA_degradation_thresholds.csv",
    "40_phaseA_DCEC_DCEW_coverage.csv",

    "41_phaseA_cluster_bootstrap_CI.csv",
    "41_phaseA_primary_report_table.csv",
    "41_phaseA_cluster_bootstrap_summary.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git(*args) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def pct(x: float) -> str:
    return f"{100.0*x:.2f}%"


def main():

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)

    PACKAGE.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFER.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_dir = PACKAGE / "data"
    data_dir.mkdir()


    # ------------------------------------------------------
    # COPY VALIDATED REPORT INPUTS / OUTPUTS
    # ------------------------------------------------------

    copied = []

    for name in FILES:
        src = ANALYSIS / name

        if not src.is_file():
            raise RuntimeError(
                f"Missing required Stage 42 input: {src}"
            )

        dst = data_dir / name

        shutil.copy2(
            src,
            dst,
        )

        copied.append(dst)


    protocol_dst = (
        PACKAGE
        / PROTOCOL.name
    )

    if not PROTOCOL.is_file():
        raise RuntimeError(
            f"Missing protocol: {PROTOCOL}"
        )

    shutil.copy2(
        PROTOCOL,
        protocol_dst,
    )

    copied.append(
        protocol_dst
    )


    # ------------------------------------------------------
    # LOAD KEY RESULTS
    # ------------------------------------------------------

    primary = json.loads(
        (
            ANALYSIS
            / "39_phaseA_primary_summary.json"
        ).read_text()
    )

    stage38 = json.loads(
        (
            ANALYSIS
            / "38_phaseA_scientific_state_summary.json"
        ).read_text()
    )

    stage40 = json.loads(
        (
            ANALYSIS
            / "40_phaseA_reporting_audit_summary.json"
        ).read_text()
    )

    ci = pd.read_csv(
        ANALYSIS
        / "41_phaseA_primary_report_table.csv"
    )

    sensitivity = pd.read_csv(
        ANALYSIS
        / "39_phaseA_threshold_sensitivity.csv"
    )

    subgroup = pd.read_csv(
        ANALYSIS
        / "39_phaseA_subgroup_summary.csv"
    )

    numeric_balance = pd.read_csv(
        ANALYSIS
        / "40_phaseA_numeric_balance.csv"
    )


    ci_by_metric = (
        ci
        .set_index("metric")
    )


    def ci_value(
        metric: str,
        column: str,
    ):
        return float(
            ci_by_metric.loc[
                metric,
                column,
            ]
        )


    # ------------------------------------------------------
    # KEY RESULT TABLE
    # ------------------------------------------------------

    key_rows = [
        {
            "metric":
                "Phase-A coverage",
            "estimate":
                966 / 1330,
            "ci95_low":
                "",
            "ci95_high":
                "",
            "denominator":
                1330,
        },
        {
            "metric":
                "Classification preservation",
            "estimate":
                ci_value(
                    "classification_preservation_rate",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "classification_preservation_rate",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "classification_preservation_rate",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Clean DCEC prevalence",
            "estimate":
                ci_value(
                    "DCEC_rate",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "DCEC_rate",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "DCEC_rate",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "DCEW conditional on DCEC",
            "estimate":
                ci_value(
                    "DCEW_rate_given_DCEC",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "DCEW_rate_given_DCEC",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "DCEW_rate_given_DCEC",
                    "ci95_high",
                ),
            "denominator":
                int(
                    primary["n_DCEC"]
                ),
        },
        {
            "metric":
                "Mean clean E",
            "estimate":
                ci_value(
                    "E_clean_mean",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "E_clean_mean",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "E_clean_mean",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Mean adversarial E",
            "estimate":
                ci_value(
                    "E_adv_mean",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "E_adv_mean",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "E_adv_mean",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Median relative E degradation",
            "estimate":
                ci_value(
                    "relative_E_degradation_median",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "relative_E_degradation_median",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "relative_E_degradation_median",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Mean clean RRA",
            "estimate":
                ci_value(
                    "RRA_clean_mean",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "RRA_clean_mean",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "RRA_clean_mean",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Mean adversarial RRA",
            "estimate":
                ci_value(
                    "RRA_adv_mean",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "RRA_adv_mean",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "RRA_adv_mean",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Median relative RRA degradation",
            "estimate":
                ci_value(
                    "relative_RRA_degradation_median",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "relative_RRA_degradation_median",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "relative_RRA_degradation_median",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Mean physical Linf",
            "estimate":
                ci_value(
                    "physical_linf_recomputed_mean",
                    "estimate",
                ),
            "ci95_low":
                ci_value(
                    "physical_linf_recomputed_mean",
                    "ci95_low",
                ),
            "ci95_high":
                ci_value(
                    "physical_linf_recomputed_mean",
                    "ci95_high",
                ),
            "denominator":
                966,
        },
        {
            "metric":
                "Maximum physical Linf",
            "estimate":
                float(
                    stage38[
                        "max_physical_linf"
                    ]
                ),
            "ci95_low":
                "",
            "ci95_high":
                "",
            "denominator":
                966,
        },
    ]


    key_path = (
        PACKAGE
        / "phaseA_key_results.csv"
    )

    pd.DataFrame(
        key_rows
    ).to_csv(
        key_path,
        index=False,
    )

    copied.append(
        key_path
    )


    # ------------------------------------------------------
    # WRITER HANDOVER
    # ------------------------------------------------------

    dcec_n = int(
        primary["n_DCEC"]
    )

    dcew_n = int(
        primary["n_DCEW"]
    )

    writer = f"""# TruFor Phase-A results handover

## Status

This is the **provisional Phase-A completed subset**, not the final frozen-population analysis.

- Frozen full attack population: **1330**
- Centrally validated Phase-A images: **966**
- Coverage: **{pct(966/1330)}**
- Images still outside this analysis: **364**
- Phase-A sampling mechanism: deterministic cost-balanced shard completion, **not random sampling**

Do not describe the 966 images as a representative sample of the full 1330 population.

## Integrity

Central ingest, artifact hashes, scientific-state reconstruction and perturbation recomputation passed for all **966/966** images.

- Stage 38 failures: **0**
- Maximum recomputation error for clean E: **{stage38['max_clean_E_abs_error']:.3e}**
- Maximum recomputation error for adversarial E: **{stage38['max_adv_E_abs_error']:.3e}**
- Maximum recomputation error for physical Linf: **{stage38['max_linf_abs_error']:.3e}**
- Maximum observed physical Linf: **{stage38['max_physical_linf']:.12f}**, equal to 1/255 within numerical precision

## Primary DC-DCEC-DCEW operational analysis

The evidence metrics are RMA/E and RRA.

The primary study-defined evidence criterion is:

- tau_E = **0.5**
- tau_RRA = **0.5**

These are a pre-specified majority criterion, **not a universal literature-mandated RMA/RRA cutoff**.

Human review is a sidecar interpretive experiment and is not used to calibrate these thresholds.

Results:

- Classification preserved: **966/966**
- Clean DCEC: **{dcec_n}/966 = {pct(dcec_n/966)}**
- Cluster-bootstrap 95% CI for clean DCEC prevalence:
  **{pct(ci_value('DCEC_rate','ci95_low'))} to {pct(ci_value('DCEC_rate','ci95_high'))}**
- DCEW: **{dcew_n}/{dcec_n} clean-DCEC images**
- Conditional DCEW rate: **{pct(dcew_n/dcec_n)}**
- Classification flips within the clean-DCEC denominator: **0**
- DCEW due to E only: **{primary['n_DCEW_E_only']}**
- DCEW due to RRA only: **{primary['n_DCEW_RRA_only']}**
- DCEW due to both E and RRA: **{primary['n_DCEW_both']}**

Every primary DCEW case therefore failed **both** spatial-evidence criteria while retaining the document-level attack decision.

## Continuous localisation degradation

Across all 966 Phase-A images:

- Mean E: **{ci_value('E_clean_mean','estimate'):.6f} clean -> {ci_value('E_adv_mean','estimate'):.6f} adversarial**
- Median relative E degradation:
  **{pct(ci_value('relative_E_degradation_median','estimate'))}**
  with clustered-bootstrap 95% CI
  **{pct(ci_value('relative_E_degradation_median','ci95_low'))} to {pct(ci_value('relative_E_degradation_median','ci95_high'))}**

- Mean RRA: **{ci_value('RRA_clean_mean','estimate'):.6f} clean -> {ci_value('RRA_adv_mean','estimate'):.6f} adversarial**
- Median relative RRA degradation:
  **{pct(ci_value('relative_RRA_degradation_median','estimate'))}**

- Median clean mu_w:
  **{ci_value('mu_clean_median','estimate'):.6f}**
- Median adversarial mu_w:
  **{ci_value('mu_adv_median','estimate'):.6f}**

- Median clean RRA lift:
  **{ci_value('RRA_lift_clean_median','estimate'):.6f}**
- Median adversarial RRA lift:
  **{ci_value('RRA_lift_adv_median','estimate'):.6f}**

## RRA tie sensitivity

Top-K cutoff ties occurred numerically, especially in adversarial maps, but:

- primary clean DCEC = 314
- tie-min clean DCEC = 314
- tie-max clean DCEC = 314
- primary DCEW = 314
- adversarial tie-min DCEW = 314
- adversarial tie-max DCEW = 314

Therefore no binary DCEC/DCEW label changes under the recorded tie uncertainty.

## Threshold sensitivity

The 3 x 3 tau_E/tau_RRA sensitivity grid uses:

- 0.25
- 0.50
- 0.75

All nine cells show a conditional DCEW rate of **100%** among the images satisfying the corresponding clean DCEC criterion.

The clean denominator changes substantially (21 to 716 images), so this should be reported as evidence that the adversarial conclusion is robust to the operational threshold choice while the DCEC prevalence itself is threshold-dependent.

See `data/39_phaseA_threshold_sensitivity.csv`.

## Phase-A population limitation

Stage 40 found that the completed 966-image subset differs from the missing 364.

Important numeric differences include:

- mean manipulated-region area A:
  Phase A **0.0622**, missing **0.0344**, SMD about **+0.607**
- clean mu_w:
  Phase A **15.08**, missing **20.13**, SMD about **-0.419**
- native pixel count SMD about **+0.254**

Therefore Phase-A results must not be extrapolated as final 1330-image population estimates.

The final scripts should be rerun after the missing 364 attacks are centrally ingested.

## Interpretation guidance

A careful provisional statement is:

> In the 966-image Phase-A completed subset, all adversarial examples retained TruFor's attack-class decision. Under the pre-specified 0.5/0.5 clean evidence criterion, 314 images qualified as decision-correct and evidence-correct (DCEC); all 314 subsequently fell below both RMA and RRA evidence thresholds while retaining the attack decision. Continuous localisation measures showed the same collapse across the broader 966-image subset, with median relative E degradation of approximately 99.99% and median relative RRA degradation of 100%. These Phase-A estimates remain provisional because the completed shards are not a random sample of the frozen 1330-image population.

Do not write:

> The attack succeeds on 100% of all 1330 images.

That statement is not supported until the final 364 images are completed and analysed.

## Files

The `data/` directory contains the underlying Stage 38-41 tables needed to verify all numbers in this handover.
"""


    writer_path = (
        PACKAGE
        / "PHASE_A_WRITER_HANDOVER.md"
    )

    writer_path.write_text(
        writer
    )

    copied.append(
        writer_path
    )


    # ------------------------------------------------------
    # README / PROVENANCE
    # ------------------------------------------------------

    commit = git(
        "rev-parse",
        "HEAD",
    )

    branch = git(
        "branch",
        "--show-current",
    )

    status = git(
        "status",
        "--porcelain",
    )


    provenance = {
        "package_status":
            "PASS",

        "git_commit":
            commit,

        "git_branch":
            branch,

        "git_worktree_clean_at_packaging":
            status == "",

        "phaseA_images":
            966,

        "full_population":
            1330,

        "missing":
            364,

        "primary_DCEC":
            dcec_n,

        "primary_DCEW":
            dcew_n,

        "stage38_status":
            stage38[
                "status"
            ],

        "stage40_status":
            stage40[
                "status"
            ],

        "evidence_protocol_sha256":
            sha256(
                PROTOCOL
            ),
    }


    provenance_path = (
        PACKAGE
        / "PACKAGE_PROVENANCE.json"
    )

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    copied.append(
        provenance_path
    )


    readme = f"""# TruFor Phase-A 966-image report package

This package records the validated provisional analysis state for the
966 completed TruFor adversarial-localisation attacks.

It does not contain the ~44 GiB clean/adversarial map/tensor archive.
Those scientific arrays remain in the validated central ingest on IMTA135.

Full frozen population: 1330
Phase-A analysed: 966
Missing at packaging: 364

Git commit at packaging:
{commit}

Primary evaluation protocol:
`evidence_evaluation_protocol_v1.json`

Writer-oriented interpretation:
`PHASE_A_WRITER_HANDOVER.md`

Compact key metrics:
`phaseA_key_results.csv`

Full source tables:
`data/`

This package is provisional. Regenerate the final report package after
all 1330 images have been centrally reconciled and analysed.
"""


    readme_path = (
        PACKAGE
        / "README.md"
    )

    readme_path.write_text(
        readme
    )

    copied.append(
        readme_path
    )


    # ------------------------------------------------------
    # INTERNAL HASH MANIFEST
    # ------------------------------------------------------

    manifest_path = (
        PACKAGE
        / "SHA256SUMS.txt"
    )


    files = sorted(
        p
        for p in PACKAGE.rglob("*")
        if p.is_file()
        and p != manifest_path
    )


    manifest_path.write_text(
        "".join(
            (
                f"{sha256(p)}  "
                f"{p.relative_to(PACKAGE)}\n"
            )
            for p in files
        )
    )


    # ------------------------------------------------------
    # CREATE SMALL TRANSFER TAR.GZ
    # ------------------------------------------------------

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


    tar_sha = sha256(
        TAR
    )


    sidecar = Path(
        str(TAR)
        + ".sha256"
    )

    sidecar.write_text(
        f"{tar_sha}  {TAR.name}\n"
    )


    print("=" * 72)
    print("TRUFOR STAGE 42 — PHASE-A REPORT PACKAGE")
    print("=" * 72)
    print("Phase-A images :", 966)
    print("full population:", 1330)
    print("DCEC           :", dcec_n)
    print("DCEW           :", dcew_n)
    print("git commit     :", commit)
    print("package        :", PACKAGE)
    print("transfer TAR   :", TAR)
    print(
        "TAR size MiB   :",
        f"{TAR.stat().st_size / 1024**2:.2f}",
    )
    print("TAR SHA256     :", tar_sha)
    print()
    print("STAGE 42 PASS")


if __name__ == "__main__":
    main()
