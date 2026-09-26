#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(0, str(HERE))

import trufor_multigpu_common as mg


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
    / "output/LABPC/"
      "trufor_adversarial_localisation_attack"
)

PHASE2_ROOT = (
    ATTACK_ROOT
    / "phase2_multigpu_sharding"
)

OUTPUTS_ROOT = (
    ATTACK_ROOT
    / "phase2_multigpu_outputs"
)

EXPORT_ROOT = (
    ROOT
    / "analysis_transfer_bundles"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shard-id",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    shard_id = args.shard_id
    sid = f"{shard_id:02d}"
    host = socket.gethostname()

    shard_csv = (
        PHASE2_ROOT
        / "shards"
        / f"shard_{sid}.csv"
    )

    phase2_plan = (
        PHASE2_ROOT
        / "phase2_plan_freeze.json"
    )

    shard_root = (
        OUTPUTS_ROOT
        / f"shard_{sid}"
    )

    images_root = (
        shard_root
        / "images"
    )

    if not shard_csv.is_file():
        raise RuntimeError(
            f"Missing shard CSV: {shard_csv}"
        )

    if not phase2_plan.is_file():
        raise RuntimeError(
            f"Missing Phase-2 plan: {phase2_plan}"
        )

    actual_plan_sha = sha256_file(
        phase2_plan
    )

    if actual_plan_sha != PHASE2_PLAN_SHA:
        raise RuntimeError(
            "Phase-2 plan SHA mismatch:\n"
            f"expected={PHASE2_PLAN_SHA}\n"
            f"actual  ={actual_plan_sha}"
        )

    import pandas as pd

    shard = pd.read_csv(
        shard_csv,
        keep_default_na=False,
    )

    assigned = len(shard)

    markers = sorted(
        images_root.rglob(
            "COMPLETE.json"
        )
    )

    if len(markers) != assigned:
        raise RuntimeError(
            f"Shard {sid} not complete: "
            f"{len(markers)}/{assigned}"
        )

    rows = []
    tar_files = []

    for marker in markers:

        rec = mg.validate_complete_dir(
            marker.parent,
            PROTOCOL_SHA,
            deep_npz=False,
        )

        image_path = str(
            rec["image_path"]
        )

        required = {
            "COMPLETE.json":
                marker.parent
                / "COMPLETE.json",

            "result.json":
                marker.parent
                / "result.json",

            "trace.csv":
                marker.parent
                / "trace.csv",

            "adversarial_result.npz":
                marker.parent
                / "adversarial_result.npz",
        }

        for name, path in required.items():

            if not path.is_file():
                raise RuntimeError(
                    f"Missing {name}: {path}"
                )

        result = json.loads(
            required[
                "result.json"
            ].read_text()
        )

        rows.append(
            {
                "hostname":
                    host,

                "shard_id":
                    shard_id,

                "image_path":
                    image_path,

                "result_dir":
                    str(
                        marker.parent
                        .relative_to(ROOT)
                    ),

                "complete_sha256":
                    sha256_file(
                        required[
                            "COMPLETE.json"
                        ]
                    ),

                "result_sha256":
                    sha256_file(
                        required[
                            "result.json"
                        ]
                    ),

                "trace_sha256":
                    sha256_file(
                        required[
                            "trace.csv"
                        ]
                    ),

                "npz_sha256":
                    sha256_file(
                        required[
                            "adversarial_result.npz"
                        ]
                    ),

                "npz_bytes":
                    required[
                        "adversarial_result.npz"
                    ].stat().st_size,

                "variant":
                    result.get(
                        "variant"
                    ),

                "hardware_source":
                    result.get(
                        "hardware_source"
                    ),

                "eval_split":
                    result.get(
                        "eval_split"
                    ),

                "global_order":
                    result.get(
                        "pilot_order"
                    ),

                "protocol_sha256":
                    result.get(
                        "protocol_sha256"
                    ),
            }
        )

        for path in required.values():
            tar_files.append(
                str(
                    path.relative_to(
                        ROOT
                    )
                )
            )

    expected = set(
        shard[
            "image_path"
        ].astype(str)
    )

    actual = {
        row[
            "image_path"
        ]
        for row in rows
    }

    if actual != expected:

        missing = sorted(
            expected - actual
        )

        extra = sorted(
            actual - expected
        )

        raise RuntimeError(
            "Shard membership mismatch\n"
            f"missing={missing[:10]}\n"
            f"extra={extra[:10]}"
        )


    # Include worker-level provenance/status.
    for name in [
        "worker_status.json",
        "worker_provenance.json",
    ]:

        p = (
            shard_root
            / name
        )

        if p.is_file():
            tar_files.append(
                str(
                    p.relative_to(
                        ROOT
                    )
                )
            )


    control = (
        ROOT
        / "execution_fingerprints"
        / "phaseB_delta_exports"
        / f"{host}_shard_{sid}"
    )

    control.mkdir(
        parents=True,
        exist_ok=True,
    )


    manifest = (
        control
        / (
            f"{host}_phaseB_"
            f"shard_{sid}_manifest.csv"
        )
    )

    with manifest.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


    metadata = {
        "status":
            "PASS",

        "hostname":
            host,

        "shard_id":
            shard_id,

        "assigned":
            assigned,

        "completed":
            len(rows),

        "protocol_sha256":
            PROTOCOL_SHA,

        "phase2_plan_sha256":
            PHASE2_PLAN_SHA,

        "shard_manifest_sha256":
            sha256_file(
                shard_csv
            ),

        "export_manifest_sha256":
            sha256_file(
                manifest
            ),

        "total_npz_bytes":
            sum(
                int(
                    x["npz_bytes"]
                )
                for x in rows
            ),
    }


    metadata_path = (
        control
        / (
            f"{host}_phaseB_"
            f"shard_{sid}_metadata.json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    tar_files.extend(
        [
            str(
                manifest.relative_to(
                    ROOT
                )
            ),

            str(
                metadata_path.relative_to(
                    ROOT
                )
            ),

            str(
                shard_csv.relative_to(
                    ROOT
                )
            ),

            str(
                phase2_plan.relative_to(
                    ROOT
                )
            ),
        ]
    )


    tar_files = sorted(
        set(
            tar_files
        )
    )


    list_file = (
        control
        / "tar_file_list.txt"
    )

    list_file.write_text(
        "\n".join(
            tar_files
        )
        + "\n"
    )


    EXPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    tar_path = (
        EXPORT_ROOT
        / (
            f"{host}_trufor_phaseB_"
            f"shard_{sid}.tar"
        )
    )


    print(
        "=" * 72
    )

    print(
        "TRUFOR PHASE-B DELTA EXPORT"
    )

    print(
        "=" * 72
    )

    print(
        "host        :",
        host,
    )

    print(
        "shard       :",
        sid,
    )

    print(
        "completed   :",
        f"{len(rows)}/{assigned}",
    )

    print(
        "NPZ GiB     :",
        f"{metadata['total_npz_bytes']/1024**3:.2f}",
    )

    print(
        "creating TAR:",
        tar_path,
    )


    subprocess.run(
        [
            "tar",
            "-cf",
            str(
                tar_path
            ),
            "-T",
            str(
                list_file
            ),
        ],
        cwd=ROOT,
        check=True,
    )


    tar_sha = sha256_file(
        tar_path
    )


    sidecar = Path(
        str(
            tar_path
        )
        + ".sha256"
    )

    sidecar.write_text(
        f"{tar_sha}  "
        f"{tar_path.name}\n"
    )


    print()
    print(
        "TAR GiB    :",
        f"{tar_path.stat().st_size/1024**3:.2f}",
    )

    print(
        "TAR SHA256 :",
        tar_sha,
    )

    print(
        "sidecar    :",
        sidecar,
    )

    print()
    print(
        "PHASE-B DELTA EXPORT PASS"
    )


if __name__ == "__main__":
    main()
