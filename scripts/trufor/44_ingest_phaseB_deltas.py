#!/usr/bin/env python3
"""
Stage 44 — incrementally ingest available Phase-B shard delta bundles.

Starts from the validated 966-image Phase-A master and adds any available
Phase-B delta TARs.

Validates:
- TAR SHA sidecars
- Phase-2 plan SHA
- protocol SHA
- export manifest SHA
- exact shard membership
- artifact existence
- per-artifact SHA256
- COMPLETE directory validity
- no overlap with Phase A
- no duplicate Phase-B image
- global membership in frozen 1330 selection

Can be rerun later when shard 01 arrives. Existing extracted bundles are
reused only if their saved TAR SHA agrees.

Does not compute scientific metrics; it only establishes the canonical
available scientific population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0,
    str(HERE),
)

import trufor_multigpu_common as mg  # noqa: E402


PROTOCOL_SHA = (
    "5840aa9bee076b493c6a645e140edaad"
    "892373433c7e919e95ed4e60e44158d5"
)

PHASE2_PLAN_SHA = (
    "e7e057388cc43f48931b93dfb3e2aead"
    "22df6ecea5a9769f6dc948ef29e88eeb"
)


ATTACK_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_attack"
)

PHASE2_ROOT = (
    ATTACK_ROOT
    / "phase2_multigpu_sharding"
)

PHASE2_PLAN = (
    PHASE2_ROOT
    / "phase2_plan_freeze.json"
)

PHASE2_ASSIGNMENT = (
    PHASE2_ROOT
    / "shard_assignment.csv"
)

FULL_SELECTION = (
    ATTACK_ROOT
    / "full_population"
    / "full_population_selection.csv"
)

PHASEA_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseA_966"
)

PHASEA_MASTER = (
    PHASEA_ROOT
    / "phaseA_master_manifest.csv"
)

OUT_ROOT = (
    ROOT
    / "output"
    / "LABPC"
    / "trufor_adversarial_localisation_analysis"
    / "phaseAB_available"
)


BUNDLE_RE = re.compile(
    r"^(?P<host>IMTA\d+)_"
    r"trufor_phaseB_"
    r"shard_(?P<shard>\d{2})\.tar$"
)


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def parse_sidecar(
    path: Path,
) -> tuple[str, str]:

    parts = (
        path.read_text()
        .strip()
        .split()
    )

    if len(parts) < 2:
        raise RuntimeError(
            f"Malformed SHA sidecar: {path}"
        )

    return (
        parts[0],
        Path(
            parts[-1]
        ).name,
    )


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
                and base not in target.parents
            ):
                raise RuntimeError(
                    "Unsafe TAR path: "
                    f"{member.name}"
                )

        tf.extractall(
            destination
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bundle-dir",
        default=str(
            ROOT
            / "analysis_transfer_bundles"
        ),
    )

    parser.add_argument(
        "--ingest-root",
        default=str(
            Path.home()
            / "trufor_central_ingest"
            / "phaseB_deltas"
        ),
    )

    args = parser.parse_args()

    bundle_dir = Path(
        args.bundle_dir
    ).resolve()

    ingest_root = Path(
        args.ingest_root
    ).resolve()


    for p in [
        bundle_dir,
        PHASEA_ROOT,
    ]:
        if not p.is_dir():
            raise RuntimeError(
                f"Missing directory: {p}"
            )


    for p in [
        PHASE2_PLAN,
        PHASE2_ASSIGNMENT,
        FULL_SELECTION,
        PHASEA_MASTER,
    ]:
        if not p.is_file():
            raise RuntimeError(
                f"Missing required file: {p}"
            )


    actual_plan_sha = sha256_file(
        PHASE2_PLAN
    )

    if (
        actual_plan_sha
        != PHASE2_PLAN_SHA
    ):
        raise RuntimeError(
            "Local Phase-2 plan SHA mismatch\n"
            f"expected={PHASE2_PLAN_SHA}\n"
            f"actual  ={actual_plan_sha}"
        )


    phaseA = pd.read_csv(
        PHASEA_MASTER,
        keep_default_na=False,
    )

    if len(phaseA) != 966:
        raise RuntimeError(
            f"Expected Phase A 966 rows; "
            f"found {len(phaseA)}"
        )

    if phaseA[
        "image_path"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image_path "
            "in Phase-A master"
        )


    phaseA_images = set(
        phaseA[
            "image_path"
        ].astype(str)
    )


    full = pd.read_csv(
        FULL_SELECTION,
        keep_default_na=False,
    )

    if len(full) != 1330:
        raise RuntimeError(
            f"Expected full population "
            f"1330; found {len(full)}"
        )


    full_images = set(
        full[
            "image_path"
        ].astype(str)
    )


    assignment = pd.read_csv(
        PHASE2_ASSIGNMENT,
        keep_default_na=False,
    )


    tar_paths = sorted(
        bundle_dir.glob(
            "IMTA*_trufor_phaseB_shard_*.tar"
        )
    )


    if not tar_paths:
        raise RuntimeError(
            f"No Phase-B TARs found in "
            f"{bundle_dir}"
        )


    print(
        "=" * 72
    )

    print(
        "TRUFOR STAGE 44 — "
        "PHASE-B CENTRAL DELTA INGEST"
    )

    print(
        "=" * 72
    )

    print(
        "Phase-A base:",
        len(phaseA),
    )

    print(
        "delta TARs :",
        len(tar_paths),
    )

    print()


    ingest_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    delta_frames = []
    bundle_rows = []

    seen_delta_images = set()
    seen_shards = set()


    for tar_path in tar_paths:

        match = BUNDLE_RE.match(
            tar_path.name
        )

        if not match:
            continue


        host = match.group(
            "host"
        )

        shard_id = int(
            match.group(
                "shard"
            )
        )

        sid = (
            f"{shard_id:02d}"
        )


        key = (
            host,
            shard_id,
        )


        if key in seen_shards:
            raise RuntimeError(
                f"Duplicate bundle key: "
                f"{key}"
            )


        seen_shards.add(
            key
        )


        sidecar = Path(
            str(
                tar_path
            )
            + ".sha256"
        )


        if not sidecar.is_file():
            raise RuntimeError(
                f"Missing SHA sidecar: "
                f"{sidecar}"
            )


        expected_sha, expected_name = (
            parse_sidecar(
                sidecar
            )
        )


        if (
            expected_name
            != tar_path.name
        ):
            raise RuntimeError(
                f"{tar_path.name}: "
                f"sidecar filename mismatch "
                f"{expected_name}"
            )


        print(
            f"[{host} shard {sid}] "
            f"verifying TAR..."
        )


        actual_sha = sha256_file(
            tar_path
        )


        if (
            actual_sha
            != expected_sha
        ):
            raise RuntimeError(
                f"{tar_path.name}: "
                f"TAR SHA mismatch"
            )


        host_root = (
            ingest_root
            / (
                f"{host}_"
                f"shard_{sid}"
            )
        )


        fingerprint = (
            host_root
            / "BUNDLE_SHA256.txt"
        )


        if host_root.exists():

            if not fingerprint.is_file():
                raise RuntimeError(
                    f"Existing ingest directory "
                    f"has no bundle fingerprint: "
                    f"{host_root}"
                )

            saved_sha = (
                fingerprint
                .read_text()
                .strip()
            )

            if (
                saved_sha
                != actual_sha
            ):
                raise RuntimeError(
                    f"Existing extraction SHA "
                    f"does not match current TAR: "
                    f"{host_root}"
                )

            print(
                f"[{host} shard {sid}] "
                f"reusing existing extraction"
            )

        else:

            print(
                f"[{host} shard {sid}] "
                f"extracting..."
            )

            safe_extract(
                tar_path,
                host_root,
            )

            fingerprint.write_text(
                actual_sha
                + "\n"
            )


        control = (
            host_root
            / "execution_fingerprints"
            / "phaseB_delta_exports"
            / f"{host}_shard_{sid}"
        )


        metadata_path = (
            control
            / (
                f"{host}_phaseB_"
                f"shard_{sid}_metadata.json"
            )
        )

        manifest_path = (
            control
            / (
                f"{host}_phaseB_"
                f"shard_{sid}_manifest.csv"
            )
        )


        for p in [
            metadata_path,
            manifest_path,
        ]:
            if not p.is_file():
                raise RuntimeError(
                    f"Missing extracted control "
                    f"file: {p}"
                )


        metadata = json.loads(
            metadata_path.read_text()
        )


        required_meta = {
            "status":
                "PASS",

            "hostname":
                host,

            "shard_id":
                shard_id,

            "protocol_sha256":
                PROTOCOL_SHA,

            "phase2_plan_sha256":
                PHASE2_PLAN_SHA,
        }


        for name, expected in (
            required_meta.items()
        ):

            actual = metadata.get(
                name
            )

            if actual != expected:
                raise RuntimeError(
                    f"{host} shard {sid}: "
                    f"metadata {name} mismatch\n"
                    f"expected={expected!r}\n"
                    f"actual  ={actual!r}"
                )


        if (
            sha256_file(
                manifest_path
            )
            != metadata[
                "export_manifest_sha256"
            ]
        ):
            raise RuntimeError(
                f"{host} shard {sid}: "
                f"export manifest SHA mismatch"
            )


        manifest = pd.read_csv(
            manifest_path,
            keep_default_na=False,
        )


        assigned = assignment.loc[
            assignment[
                "shard_id"
            ].astype(int)
            == shard_id
        ].copy()


        expected_images = set(
            assigned[
                "image_path"
            ].astype(str)
        )


        actual_images = set(
            manifest[
                "image_path"
            ].astype(str)
        )


        if (
            actual_images
            != expected_images
        ):

            raise RuntimeError(
                f"{host} shard {sid}: "
                f"membership mismatch\n"
                f"missing="
                f"{sorted(expected_images-actual_images)[:10]}\n"
                f"extra="
                f"{sorted(actual_images-expected_images)[:10]}"
            )


        if (
            len(manifest)
            != int(
                metadata[
                    "assigned"
                ]
            )
            or len(manifest)
            != int(
                metadata[
                    "completed"
                ]
            )
        ):
            raise RuntimeError(
                f"{host} shard {sid}: "
                f"row/count mismatch"
            )


        overlap_phaseA = (
            actual_images
            & phaseA_images
        )


        if overlap_phaseA:
            raise RuntimeError(
                f"{host} shard {sid}: "
                f"overlaps Phase A, e.g. "
                f"{sorted(overlap_phaseA)[:5]}"
            )


        overlap_delta = (
            actual_images
            & seen_delta_images
        )


        if overlap_delta:
            raise RuntimeError(
                f"{host} shard {sid}: "
                f"duplicates another delta, e.g. "
                f"{sorted(overlap_delta)[:5]}"
            )


        seen_delta_images.update(
            actual_images
        )


        verified_rows = []


        print(
            f"[{host} shard {sid}] "
            f"verifying "
            f"{len(manifest)} images..."
        )


        for n, (_, row) in enumerate(
            manifest.iterrows(),
            start=1,
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
                "COMPLETE.json":
                    result_dir
                    / "COMPLETE.json",

                "result.json":
                    result_dir
                    / "result.json",

                "trace.csv":
                    result_dir
                    / "trace.csv",

                "adversarial_result.npz":
                    result_dir
                    / "adversarial_result.npz",
            }


            sha_columns = {
                "COMPLETE.json":
                    "complete_sha256",

                "result.json":
                    "result_sha256",

                "trace.csv":
                    "trace_sha256",

                "adversarial_result.npz":
                    "npz_sha256",
            }


            for name, path in (
                artifacts.items()
            ):

                if not path.is_file():
                    raise RuntimeError(
                        f"{host} shard {sid}: "
                        f"missing {name}: "
                        f"{path}"
                    )


                expected_artifact_sha = str(
                    row[
                        sha_columns[
                            name
                        ]
                    ]
                )


                actual_artifact_sha = (
                    sha256_file(
                        path
                    )
                )


                if (
                    actual_artifact_sha
                    != expected_artifact_sha
                ):
                    raise RuntimeError(
                        f"{host} shard {sid}: "
                        f"{name} SHA mismatch "
                        f"for "
                        f"{row['image_path']}"
                    )


            rec = mg.validate_complete_dir(
                result_dir,
                PROTOCOL_SHA,
                deep_npz=False,
            )


            if (
                str(
                    rec[
                        "image_path"
                    ]
                )
                != str(
                    row[
                        "image_path"
                    ]
                )
            ):
                raise RuntimeError(
                    f"{host} shard {sid}: "
                    f"COMPLETE image_path mismatch"
                )


            r = row.to_dict()

            r[
                "source_host"
            ] = host

            r[
                "execution_generation"
            ] = "phase2_10shard"

            r[
                "execution_shard_id"
            ] = shard_id

            r[
                "central_result_dir"
            ] = str(
                result_dir
            )

            r[
                "central_npz_path"
            ] = str(
                artifacts[
                    "adversarial_result.npz"
                ]
            )

            r[
                "central_result_json"
            ] = str(
                artifacts[
                    "result.json"
                ]
            )

            r[
                "central_trace_csv"
            ] = str(
                artifacts[
                    "trace.csv"
                ]
            )


            verified_rows.append(
                r
            )


            if (
                n % 25 == 0
                or n == len(
                    manifest
                )
            ):
                print(
                    f"  verified "
                    f"{n}/{len(manifest)}"
                )


        delta_frames.append(
            pd.DataFrame(
                verified_rows
            )
        )


        bundle_rows.append(
            {
                "host":
                    host,

                "shard_id":
                    shard_id,

                "images":
                    len(
                        verified_rows
                    ),

                "tar_sha256":
                    actual_sha,

                "tar_bytes":
                    tar_path.stat().st_size,

                "export_manifest_sha256":
                    sha256_file(
                        manifest_path
                    ),
            }
        )


    delta = pd.concat(
        delta_frames,
        ignore_index=True,
        sort=False,
    )


    # Standardise Phase-A identity names where needed.
    combined = pd.concat(
        [
            phaseA,
            delta,
        ],
        ignore_index=True,
        sort=False,
    )


    if combined[
        "image_path"
    ].duplicated().any():

        dup = combined.loc[
            combined[
                "image_path"
            ].duplicated(
                keep=False
            ),
            [
                "image_path",
                "source_host",
                "execution_generation",
                "execution_shard_id",
            ],
        ]

        raise RuntimeError(
            "Duplicate image_path in "
            "combined available population:\n"
            + dup.head(
                30
            ).to_string(
                index=False
            )
        )


    combined_images = set(
        combined[
            "image_path"
        ].astype(str)
    )


    outside = (
        combined_images
        - full_images
    )


    if outside:
        raise RuntimeError(
            "Combined population has "
            "images outside frozen 1330:\n"
            + "\n".join(
                sorted(
                    outside
                )[:20]
            )
        )


    missing = full.loc[
        ~full[
            "image_path"
        ]
        .astype(str)
        .isin(
            combined_images
        )
    ].copy()


    missing_assignment = (
        assignment.loc[
            assignment[
                "image_path"
            ]
            .astype(str)
            .isin(
                set(
                    missing[
                        "image_path"
                    ].astype(str)
                )
            )
        ]
        .copy()
    )


    missing_by_shard = (
        missing_assignment
        .groupby(
            "shard_id",
            as_index=False,
        )
        .agg(
            missing_images=(
                "image_path",
                "size",
            )
        )
        .sort_values(
            "shard_id"
        )
    )


    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    master_path = (
        OUT_ROOT
        / "44_available_master_manifest.csv"
    )

    delta_path = (
        OUT_ROOT
        / "44_phaseB_delta_manifest.csv"
    )

    missing_path = (
        OUT_ROOT
        / "44_remaining_missing_images.csv"
    )

    missing_shard_path = (
        OUT_ROOT
        / "44_remaining_by_phase2_shard.csv"
    )

    bundles_path = (
        OUT_ROOT
        / "44_ingested_bundles.csv"
    )


    combined.to_csv(
        master_path,
        index=False,
    )

    delta.to_csv(
        delta_path,
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
        bundle_rows
    ).to_csv(
        bundles_path,
        index=False,
    )


    summary = {
        "status":
            "PASS",

        "full_population":
            1330,

        "phaseA_images":
            966,

        "phaseB_delta_images":
            int(
                len(
                    delta
                )
            ),

        "available_unique_images":
            int(
                len(
                    combined
                )
            ),

        "remaining_images":
            int(
                len(
                    missing
                )
            ),

        "ingested_shards":
            sorted(
                {
                    int(x)
                    for x in delta[
                        "execution_shard_id"
                    ]
                }
            ),

        "protocol_sha256":
            PROTOCOL_SHA,

        "phase2_plan_sha256":
            PHASE2_PLAN_SHA,

        "remaining_by_phase2_shard":
            missing_by_shard.to_dict(
                orient="records"
            ),
    }


    summary_path = (
        OUT_ROOT
        / "44_available_population_summary.json"
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
        master_path,
        delta_path,
        missing_path,
        missing_shard_path,
        bundles_path,
        summary_path,
    ]


    sums_path = (
        OUT_ROOT
        / "44_SHA256SUMS.txt"
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


    print()

    print(
        "=" * 72
    )

    print(
        "STAGE 44 AVAILABLE POPULATION"
    )

    print(
        "=" * 72
    )

    print(
        "Phase A             :",
        966,
    )

    print(
        "new Phase-B images  :",
        len(
            delta
        ),
    )

    print(
        "available unique    :",
        len(
            combined
        ),
        "/ 1330",
    )

    print(
        "remaining           :",
        len(
            missing
        ),
    )

    print(
        "ingested shards     :",
        summary[
            "ingested_shards"
        ],
    )

    print()

    print(
        "REMAINING BY SHARD"
    )

    if len(
        missing_by_shard
    ):
        print(
            missing_by_shard.to_string(
                index=False
            )
        )
    else:
        print(
            "NONE — FULL 1330 AVAILABLE"
        )

    print()

    print(
        "STAGE 44 PASS"
    )

    print(
        "outputs:",
        OUT_ROOT,
    )


if __name__ == "__main__":
    main()
