#!/usr/bin/env python3
from __future__ import annotations

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

ATTACK_ROOT = (
    ROOT
    / "output/LABPC/"
      "trufor_adversarial_localisation_attack"
)

EXPORT_ROOT = (
    ROOT
    / "analysis_transfer_bundles"
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


host = socket.gethostname()

control_dir = (
    ROOT
    / "execution_fingerprints"
    / "analysis_transfer"
    / host
)

control_dir.mkdir(
    parents=True,
    exist_ok=True,
)

EXPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# Primary scientific result locations only.
# Deliberately EXCLUDE phase2 archive directories.
roots = [
    ATTACK_ROOT
    / "full_population/images",

    ATTACK_ROOT
    / "full_population_shards",

    ATTACK_ROOT
    / "phase2_multigpu_outputs",
]


seen = {}
rows = []
tar_files = []


for root in roots:

    if not root.is_dir():
        continue

    for marker in sorted(
        root.rglob("COMPLETE.json")
    ):

        rec = mg.validate_complete_dir(
            marker.parent,
            PROTOCOL_SHA,
            deep_npz=False,
        )

        image = str(
            rec["image_path"]
        )

        if image in seen:
            raise RuntimeError(
                "Duplicate primary result "
                f"on {host}: {image}\n"
                f"{seen[image]}\n"
                f"{marker.parent}"
            )

        seen[image] = marker.parent

        required = {
            "complete":
                marker.parent
                / "COMPLETE.json",

            "result":
                marker.parent
                / "result.json",

            "trace":
                marker.parent
                / "trace.csv",

            "npz":
                marker.parent
                / "adversarial_result.npz",
        }

        for name, path in required.items():
            if not path.is_file():
                raise RuntimeError(
                    f"Missing {name}: {path}"
                )

        result = json.loads(
            required["result"].read_text()
        )

        row = {
            "hostname":
                host,

            "image_path":
                image,

            "pilot_order":
                result.get(
                    "pilot_order"
                ),

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

            "result_dir":
                str(
                    marker.parent
                    .relative_to(ROOT)
                ),

            "complete_sha256":
                sha256_file(
                    required["complete"]
                ),

            "result_sha256":
                sha256_file(
                    required["result"]
                ),

            "trace_sha256":
                sha256_file(
                    required["trace"]
                ),

            "npz_sha256":
                sha256_file(
                    required["npz"]
                ),

            "npz_bytes":
                required[
                    "npz"
                ].stat().st_size,

            "image_wall_seconds":
                (
                    result.get(
                        "timing",
                        {}
                    )
                    .get(
                        "image_wall_seconds_before_artifact_hashing"
                    )
                ),
        }

        rows.append(row)

        for path in required.values():
            tar_files.append(
                str(
                    path.relative_to(
                        ROOT
                    )
                )
            )


rows = sorted(
    rows,
    key=lambda x: (
        int(
            x["pilot_order"]
            if x["pilot_order"] is not None
            else 10**9
        ),
        x["image_path"],
    ),
)


manifest_csv = (
    control_dir
    / f"{host}_analysis_manifest.csv"
)

with manifest_csv.open(
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

    "protocol_sha256":
        PROTOCOL_SHA,

    "unique_completed_images":
        len(rows),

    "artifact_set_per_image": [
        "COMPLETE.json",
        "result.json",
        "trace.csv",
        "adversarial_result.npz",
    ],

    "archive_directories_included":
        False,

    "manifest_file":
        str(
            manifest_csv
            .relative_to(ROOT)
        ),

    "manifest_sha256":
        sha256_file(
            manifest_csv
        ),

    "total_npz_bytes":
        sum(
            int(
                x["npz_bytes"]
            )
            for x in rows
        ),
}

metadata_json = (
    control_dir
    / f"{host}_analysis_export_metadata.json"
)

metadata_json.write_text(
    json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


file_list = (
    control_dir
    / f"{host}_tar_file_list.txt"
)

extra_files = [
    str(
        manifest_csv.relative_to(
            ROOT
        )
    ),
    str(
        metadata_json.relative_to(
            ROOT
        )
    ),
]

all_files = sorted(
    set(
        tar_files
        + extra_files
    )
)

file_list.write_text(
    "\n".join(
        all_files
    )
    + "\n"
)


tar_path = (
    EXPORT_ROOT
    / f"{host}_trufor_analysis_phaseA.tar"
)


print(
    "===== BUNDLE SUMMARY ====="
)

print(
    "host             :",
    host,
)

print(
    "unique images    :",
    len(rows),
)

print(
    "NPZ GiB          :",
    f"{metadata['total_npz_bytes']/1024**3:.2f}",
)

print(
    "manifest SHA256  :",
    metadata[
        "manifest_sha256"
    ],
)

print(
    "creating TAR     :",
    tar_path,
)


subprocess.run(
    [
        "tar",
        "-cf",
        str(tar_path),
        "-T",
        str(file_list),
    ],
    cwd=ROOT,
    check=True,
)


tar_sha = sha256_file(
    tar_path
)


sha_path = Path(
    str(tar_path)
    + ".sha256"
)

sha_path.write_text(
    f"{tar_sha}  "
    f"{tar_path.name}\n"
)


print()
print(
    "BUNDLE PASS"
)

print(
    "tar:",
    tar_path,
)

print(
    "tar SHA256:",
    tar_sha,
)

print(
    "sha sidecar:",
    sha_path,
)

print(
    "tar GiB:",
    f"{tar_path.stat().st_size/1024**3:.2f}",
)
