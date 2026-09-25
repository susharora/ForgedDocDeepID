#!/usr/bin/env python3
"""
Stage 37: central ingest of compact TruFor Phase-A analysis bundles.

Read-only with respect to worker/scientific outputs.

Actions:
- verify four downloaded TAR SHA256 sidecars;
- safely extract each host into its own isolated directory;
- verify source manifest hash;
- verify every transferred COMPLETE/result/trace/NPZ artifact hash;
- reject duplicates across machines;
- reconcile against frozen 1330-image selection;
- prove Phase-2 missing-image/shard accounting;
- write one canonical Phase-A master manifest.

Does NOT yet calculate DCEC/DCEW metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


EXPECTED_HOSTS = [
    "IMTA134",
    "IMTA135",
    "IMTA136",
    "IMTA138",
]


ATTACK_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_attack"
)


FULL_SELECTION = (
    ATTACK_ROOT
    / "full_population"
    / "full_population_selection.csv"
)


PHASE2_ROOT = (
    ATTACK_ROOT
    / "phase2_multigpu_sharding"
)


PHASE2_ASSIGNMENT = (
    PHASE2_ROOT
    / "shard_assignment.csv"
)


PHASE2_COMPLETED = (
    PHASE2_ROOT
    / "completed_snapshot.csv"
)


ANALYSIS_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseA_966"
)


PROTOCOL_SHA = (
    "5840aa9bee076b493c6a645e140edaad"
    "892373433c7e919e95ed4e60e44158d5"
)


PHASE2_PLAN_SHA = (
    "e7e057388cc43f48931b93dfb3e2aead"
    "22df6ecea5a9769f6dc948ef29e88eeb"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def safe_extract(
    tar_path: Path,
    destination: Path,
) -> None:

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = destination.resolve()

    with tarfile.open(
        tar_path,
        "r",
    ) as tf:

        members = tf.getmembers()

        for member in members:

            target = (
                destination
                / member.name
            ).resolve()

            if (
                target != base
                and base
                not in target.parents
            ):
                raise RuntimeError(
                    "Unsafe TAR member path: "
                    f"{member.name}"
                )

        tf.extractall(
            destination
        )


def parse_sidecar(
    sidecar: Path,
) -> tuple[str, str]:

    text = (
        sidecar
        .read_text()
        .strip()
    )

    parts = text.split()

    if len(parts) < 2:
        raise RuntimeError(
            f"Malformed SHA sidecar: "
            f"{sidecar}"
        )

    return (
        parts[0],
        Path(
            parts[-1]
        ).name,
    )


def classify_execution(
    result_dir: str,
) -> tuple[str, int | None]:

    text = str(
        result_dir
    ).replace(
        "\\",
        "/",
    )


    if (
        "/full_population/images/"
        in f"/{text}"
    ):
        return (
            "legacy_stage24",
            None,
        )


    m = re.search(
        r"/full_population_shards/"
        r"shard_(\d+)/",
        f"/{text}",
    )

    if m:
        return (
            "phase1_3shard",
            int(
                m.group(1)
            ),
        )


    m = re.search(
        r"/phase2_multigpu_outputs/"
        r"shard_(\d+)/",
        f"/{text}",
    )

    if m:
        return (
            "phase2_10shard",
            int(
                m.group(1)
            ),
        )


    return (
        "unknown",
        None,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bundle-dir",
        required=True,
        help=(
            "Directory containing the four TARs "
            "and their .sha256 sidecars."
        ),
    )

    parser.add_argument(
        "--ingest-root",
        default=str(
            Path.home()
            / "trufor_central_ingest"
            / "phaseA_966"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()


    bundle_dir = Path(
        args.bundle_dir
    ).resolve()

    ingest_root = Path(
        args.ingest_root
    ).resolve()


    if not bundle_dir.is_dir():
        raise RuntimeError(
            f"Bundle directory missing: "
            f"{bundle_dir}"
        )


    if (
        ingest_root.exists()
        and any(
            ingest_root.iterdir()
        )
    ):

        if not args.overwrite:
            raise RuntimeError(
                f"Ingest directory already "
                f"contains files:\n"
                f"{ingest_root}\n"
                f"Use --overwrite only if "
                f"you deliberately want to "
                f"rebuild Phase A."
            )

        shutil.rmtree(
            ingest_root
        )


    ingest_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not FULL_SELECTION.is_file():
        raise RuntimeError(
            f"Missing frozen selection: "
            f"{FULL_SELECTION}"
        )


    if not PHASE2_ASSIGNMENT.is_file():
        raise RuntimeError(
            f"Missing Phase-2 assignment: "
            f"{PHASE2_ASSIGNMENT}"
        )


    if not PHASE2_COMPLETED.is_file():
        raise RuntimeError(
            f"Missing Phase-2 completed "
            f"snapshot: {PHASE2_COMPLETED}"
        )


    print(
        "=" * 72
    )

    print(
        "TRUFOR STAGE 37 — "
        "CENTRAL PHASE-A INGEST"
    )

    print(
        "=" * 72
    )


    print(
        "bundle dir :",
        bundle_dir,
    )

    print(
        "ingest root:",
        ingest_root,
    )

    print()


    combined_frames = []
    source_rows = []


    for host in EXPECTED_HOSTS:

        tar_name = (
            f"{host}_"
            f"trufor_analysis_phaseA.tar"
        )

        tar_path = (
            bundle_dir
            / tar_name
        )

        sha_path = Path(
            str(
                tar_path
            )
            + ".sha256"
        )


        if not tar_path.is_file():
            raise RuntimeError(
                f"Missing TAR for "
                f"{host}: {tar_path}"
            )


        if not sha_path.is_file():
            raise RuntimeError(
                f"Missing SHA sidecar "
                f"for {host}: "
                f"{sha_path}"
            )


        expected_tar_sha, expected_name = (
            parse_sidecar(
                sha_path
            )
        )


        if expected_name != tar_name:
            raise RuntimeError(
                f"{host}: SHA sidecar "
                f"filename mismatch: "
                f"{expected_name}"
            )


        print(
            f"[{host}] "
            f"verifying TAR SHA256..."
        )


        actual_tar_sha = sha256_file(
            tar_path
        )


        if (
            actual_tar_sha
            != expected_tar_sha
        ):
            raise RuntimeError(
                f"{host}: downloaded "
                f"TAR SHA mismatch\n"
                f"expected="
                f"{expected_tar_sha}\n"
                f"actual  ="
                f"{actual_tar_sha}"
            )


        host_root = (
            ingest_root
            / host
        )


        print(
            f"[{host}] "
            f"extracting..."
        )


        safe_extract(
            tar_path,
            host_root,
        )


        control_root = (
            host_root
            / "execution_fingerprints"
            / "analysis_transfer"
            / host
        )


        metadata_path = (
            control_root
            / (
                f"{host}_"
                f"analysis_export_metadata.json"
            )
        )


        manifest_path = (
            control_root
            / (
                f"{host}_"
                f"analysis_manifest.csv"
            )
        )


        if not metadata_path.is_file():
            raise RuntimeError(
                f"{host}: metadata "
                f"not found after extraction"
            )


        if not manifest_path.is_file():
            raise RuntimeError(
                f"{host}: manifest "
                f"not found after extraction"
            )


        metadata = json.loads(
            metadata_path.read_text()
        )


        if (
            metadata.get(
                "status"
            )
            != "PASS"
        ):
            raise RuntimeError(
                f"{host}: export "
                f"metadata not PASS"
            )


        if (
            metadata.get(
                "hostname"
            )
            != host
        ):
            raise RuntimeError(
                f"{host}: hostname "
                f"metadata mismatch"
            )


        if (
            metadata.get(
                "protocol_sha256"
            )
            != PROTOCOL_SHA
        ):
            raise RuntimeError(
                f"{host}: protocol "
                f"SHA mismatch"
            )


        actual_manifest_sha = (
            sha256_file(
                manifest_path
            )
        )


        if (
            actual_manifest_sha
            != metadata.get(
                "manifest_sha256"
            )
        ):
            raise RuntimeError(
                f"{host}: manifest "
                f"SHA mismatch"
            )


        manifest = pd.read_csv(
            manifest_path,
            keep_default_na=False,
        )


        expected_rows = int(
            metadata[
                "unique_completed_images"
            ]
        )


        if len(
            manifest
        ) != expected_rows:
            raise RuntimeError(
                f"{host}: manifest "
                f"row mismatch: "
                f"{len(manifest)} "
                f"!= {expected_rows}"
            )


        print(
            f"[{host}] "
            f"verifying "
            f"{len(manifest)} "
            f"scientific records..."
        )


        verified_rows = []


        for idx, row in (
            manifest.iterrows()
        ):

            result_dir = (
                host_root
                / str(
                    row[
                        "result_dir"
                    ]
                )
            )


            artifacts = {
                "complete":
                    result_dir
                    / "COMPLETE.json",

                "result":
                    result_dir
                    / "result.json",

                "trace":
                    result_dir
                    / "trace.csv",

                "npz":
                    result_dir
                    / "adversarial_result.npz",
            }


            for key, path in (
                artifacts.items()
            ):

                if not path.is_file():
                    raise RuntimeError(
                        f"{host}: missing "
                        f"{key} for "
                        f"{row['image_path']}\n"
                        f"{path}"
                    )


            checks = {
                "complete":
                    str(
                        row[
                            "complete_sha256"
                        ]
                    ),

                "result":
                    str(
                        row[
                            "result_sha256"
                        ]
                    ),

                "trace":
                    str(
                        row[
                            "trace_sha256"
                        ]
                    ),

                "npz":
                    str(
                        row[
                            "npz_sha256"
                        ]
                    ),
            }


            for key, expected in (
                checks.items()
            ):

                actual = sha256_file(
                    artifacts[key]
                )

                if actual != expected:
                    raise RuntimeError(
                        f"{host}: {key} "
                        f"SHA mismatch for "
                        f"{row['image_path']}"
                    )


            result = json.loads(
                artifacts[
                    "result"
                ].read_text()
            )


            if (
                str(
                    result.get(
                        "image_path"
                    )
                )
                != str(
                    row[
                        "image_path"
                    ]
                )
            ):
                raise RuntimeError(
                    f"{host}: result "
                    f"image_path mismatch"
                )


            if (
                result.get(
                    "protocol_sha256"
                )
                != PROTOCOL_SHA
            ):
                raise RuntimeError(
                    f"{host}: result "
                    f"protocol SHA mismatch"
                )


            generation, shard_id = (
                classify_execution(
                    str(
                        row[
                            "result_dir"
                        ]
                    )
                )
            )


            verified = row.to_dict()


            verified[
                "source_host"
            ] = host


            verified[
                "execution_generation"
            ] = generation


            verified[
                "execution_shard_id"
            ] = shard_id


            verified[
                "central_result_dir"
            ] = str(
                result_dir
            )


            verified[
                "central_npz_path"
            ] = str(
                artifacts["npz"]
            )


            verified[
                "central_result_json"
            ] = str(
                artifacts["result"]
            )


            verified[
                "central_trace_csv"
            ] = str(
                artifacts["trace"]
            )


            verified_rows.append(
                verified
            )


            if (
                (idx + 1) % 50
                == 0
                or idx + 1
                == len(manifest)
            ):

                print(
                    f"  {host}: "
                    f"{idx+1}/"
                    f"{len(manifest)} "
                    f"verified"
                )


        verified_df = (
            pd.DataFrame(
                verified_rows
            )
        )


        combined_frames.append(
            verified_df
        )


        source_rows.append(
            {
                "host":
                    host,

                "images":
                    len(
                        verified_df
                    ),

                "tar_sha256":
                    actual_tar_sha,

                "manifest_sha256":
                    actual_manifest_sha,

                "tar_bytes":
                    tar_path.stat().st_size,
            }
        )


    print()

    print(
        "===== GLOBAL RECONCILIATION ====="
    )


    master = pd.concat(
        combined_frames,
        ignore_index=True,
    )


    if (
        master[
            "image_path"
        ]
        .duplicated()
        .any()
    ):

        dup = (
            master.loc[
                master[
                    "image_path"
                ].duplicated(
                    keep=False
                ),
                [
                    "image_path",
                    "source_host",
                    "result_dir",
                ],
            ]
            .sort_values(
                "image_path"
            )
        )

        raise RuntimeError(
            "Duplicate image_path "
            "across Phase-A bundles:\n"
            + dup.head(
                30
            ).to_string(
                index=False
            )
        )


    full = pd.read_csv(
        FULL_SELECTION,
        keep_default_na=False,
    )


    expected_images = set(
        full[
            "image_path"
        ].astype(str)
    )


    actual_images = set(
        master[
            "image_path"
        ].astype(str)
    )


    outside = (
        actual_images
        - expected_images
    )


    if outside:
        raise RuntimeError(
            "Phase-A bundle contains "
            "images outside frozen "
            "selection:\n"
            + "\n".join(
                sorted(
                    outside
                )[:20]
            )
        )


    missing = (
        full.loc[
            ~full[
                "image_path"
            ]
            .astype(str)
            .isin(
                actual_images
            )
        ]
        .copy()
    )


    phase2 = pd.read_csv(
        PHASE2_ASSIGNMENT,
        keep_default_na=False,
    )


    phase2_completed = (
        pd.read_csv(
            PHASE2_COMPLETED,
            keep_default_na=False,
        )
    )


    prephase2_expected = set(
        phase2_completed[
            "image_path"
        ].astype(str)
    )


    prephase2_missing = (
        prephase2_expected
        - actual_images
    )


    if prephase2_missing:
        raise RuntimeError(
            "Phase-A transfer is "
            "missing images that were "
            "already COMPLETE before "
            "Phase 2:\n"
            + "\n".join(
                sorted(
                    prephase2_missing
                )[:20]
            )
        )


    phase2_present = (
        actual_images
        & set(
            phase2[
                "image_path"
            ].astype(str)
        )
    )


    phase2_missing = (
        phase2.loc[
            ~phase2[
                "image_path"
            ]
            .astype(str)
            .isin(
                actual_images
            )
        ]
        .copy()
    )


    missing_by_shard = (
        phase2_missing
        .groupby(
            "shard_id",
            as_index=False,
        )
        .agg(
            missing_images=(
                "image_path",
                "size",
            ),
            estimated_hours=(
                "estimated_seconds",
                lambda s:
                    float(
                        s.sum()
                        / 3600.0
                    ),
            ),
        )
        .sort_values(
            "shard_id"
        )
    )


    ANALYSIS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    master = (
        master
        .sort_values(
            "pilot_order"
        )
        .reset_index(
            drop=True
        )
    )


    master_path = (
        ANALYSIS_ROOT
        / "phaseA_master_manifest.csv"
    )


    missing_path = (
        ANALYSIS_ROOT
        / "phaseA_missing_images.csv"
    )


    missing_shard_path = (
        ANALYSIS_ROOT
        / (
            "phaseA_missing_by_"
            "phase2_shard.csv"
        )
    )


    sources_path = (
        ANALYSIS_ROOT
        / "phaseA_source_bundles.csv"
    )


    master.to_csv(
        master_path,
        index=False,
    )


    missing.to_csv(
        missing_path,
        index=False,
    )


    missing_by_shard.to_csv(
        missing_shard_path,
        index=False,
    )


    pd.DataFrame(
        source_rows
    ).to_csv(
        sources_path,
        index=False,
    )


    summary = {
        "status":
            "PASS",

        "expected_population":
            int(
                len(full)
            ),

        "phaseA_unique_images":
            int(
                len(master)
            ),

        "phaseA_fraction":
            float(
                len(master)
                / len(full)
            ),

        "missing_images":
            int(
                len(missing)
            ),

        "prephase2_expected":
            int(
                len(
                    prephase2_expected
                )
            ),

        "prephase2_present":
            int(
                len(
                    prephase2_expected
                    & actual_images
                )
            ),

        "phase2_population":
            int(
                len(phase2)
            ),

        "phase2_present":
            int(
                len(
                    phase2_present
                )
            ),

        "phase2_missing":
            int(
                len(
                    phase2_missing
                )
            ),

        "protocol_sha256":
            PROTOCOL_SHA,

        "phase2_plan_sha256":
            PHASE2_PLAN_SHA,

        "source_hosts":
            EXPECTED_HOSTS,

        "missing_by_phase2_shard":
            (
                missing_by_shard
                .to_dict(
                    orient="records"
                )
            ),
    }


    summary_path = (
        ANALYSIS_ROOT
        / "phaseA_ingest_summary.json"
    )


    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    files_to_hash = [
        master_path,
        missing_path,
        missing_shard_path,
        sources_path,
        summary_path,
    ]


    sums_path = (
        ANALYSIS_ROOT
        / "SHA256SUMS.txt"
    )


    sums_path.write_text(
        "".join(
            (
                f"{sha256_file(p)}  "
                f"{p.name}\n"
            )
            for p in files_to_hash
        )
    )


    print(
        "expected population :",
        len(full),
    )

    print(
        "Phase-A unique      :",
        len(master),
    )

    print(
        "coverage            :",
        f"{100*len(master)/len(full):.2f}%",
    )

    print(
        "missing             :",
        len(missing),
    )

    print(
        "pre-Phase2 present  :",
        (
            f"{len(prephase2_expected & actual_images)}"
            f"/{len(prephase2_expected)}"
        ),
    )

    print(
        "Phase2 present      :",
        (
            f"{len(phase2_present)}"
            f"/{len(phase2)}"
        ),
    )


    print()

    print(
        "MISSING BY PHASE2 SHARD"
    )

    print(
        missing_by_shard.to_string(
            index=False
        )
    )


    print()

    print(
        "CENTRAL PHASE-A INGEST: PASS"
    )

    print(
        "analysis root:",
        ANALYSIS_ROOT,
    )


if __name__ == "__main__":
    main()
