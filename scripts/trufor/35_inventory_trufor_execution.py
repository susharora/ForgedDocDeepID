#!/usr/bin/env python3
"""
Read-only forensic inventory of distributed TruFor adversarial execution.

Does NOT modify, merge, move, or delete attack outputs.

Discovers:
- legacy Stage-24 COMPLETE outputs
- Phase-1 / original 3-shard COMPLETE outputs
- Phase-2 10-shard COMPLETE outputs
- archived Phase-2 output trees
- worker status/provenance files
- Stage-26 / Phase-2 logs
- frozen plan hashes

Produces small CSV/JSON inventory files suitable for central collation.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import trufor_multigpu_common as mg  # noqa: E402
from trufor_common import sha256_file  # noqa: E402


RUN_TAG = "LABPC"

PROTOCOL_SHA = (
    "5840aa9bee076b493c6a645e140edaad"
    "892373433c7e919e95ed4e60e44158d5"
)

OLD_PLAN_SHA = (
    "672fb656729a3a23210df373c19956db"
    "966dc6ef9470afc21483d5917a6bccf8"
)

PHASE2_PLAN_SHA = (
    "e7e057388cc43f48931b93dfb3e2aead"
    "22df6ecea5a9769f6dc948ef29e88eeb"
)

ATTACK_ROOT = (
    ROOT
    / "output"
    / RUN_TAG
    / "trufor_adversarial_localisation_attack"
)

LOG_ROOT = (
    ROOT
    / "logs"
    / RUN_TAG
)


def classify_complete(marker: Path):
    rel = marker.relative_to(ATTACK_ROOT)
    parts = rel.parts
    text = rel.as_posix()

    if text.startswith(
        "full_population/images/"
    ):
        return "legacy_stage24", None

    if text.startswith(
        "full_population_shards/"
    ):
        for part in parts:
            if part.startswith("shard_"):
                try:
                    return (
                        "phase1_3shard",
                        int(part.split("_")[1]),
                    )
                except Exception:
                    pass

        return "phase1_3shard", None

    if text.startswith(
        "phase2_multigpu_outputs/"
    ):
        for part in parts:
            if part.startswith("shard_"):
                try:
                    return (
                        "phase2_10shard",
                        int(part.split("_")[1]),
                    )
                except Exception:
                    pass

        return "phase2_10shard", None

    if (
        "phase2_multigpu_outputs_shard_"
        in text
        and "_archive" in text
    ):
        try:
            token = text.split(
                "phase2_multigpu_outputs_shard_",
                1,
            )[1].split("_", 1)[0]

            return (
                "phase2_archive",
                int(token),
            )
        except Exception:
            return "phase2_archive", None

    return "other", None


def nested(d, *keys, default=None):
    x = d

    for k in keys:
        if not isinstance(x, dict):
            return default

        if k not in x:
            return default

        x = x[k]

    return x


def inventory_complete_results():
    markers = []

    roots = [
        ATTACK_ROOT
        / "full_population"
        / "images",

        ATTACK_ROOT
        / "full_population_shards",

        ATTACK_ROOT
        / "phase2_multigpu_outputs",
    ]

    roots.extend(
        sorted(
            ATTACK_ROOT.glob(
                "phase2_multigpu_outputs_shard_*_archive"
            )
        )
    )

    for root in roots:
        if not root.is_dir():
            continue

        markers.extend(
            root.rglob("COMPLETE.json")
        )

    rows = []
    invalid = []

    for marker in sorted(
        set(markers)
    ):
        generation, shard_id = (
            classify_complete(marker)
        )

        try:
            rec = mg.validate_complete_dir(
                marker.parent,
                PROTOCOL_SHA,
                deep_npz=False,
            )

            result = rec["result"]

            rows.append(
                {
                    "hostname":
                        socket.gethostname(),

                    "generation":
                        generation,

                    "shard_id":
                        shard_id,

                    "image_path":
                        rec["image_path"],

                    "output_dir":
                        str(
                            marker.parent.relative_to(
                                ROOT
                            )
                        ),

                    "complete_marker_sha256":
                        rec[
                            "marker_sha256"
                        ],

                    "result_json_sha256":
                        sha256_file(
                            marker.parent
                            / "result.json"
                        ),

                    "global_or_pilot_order":
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

                    "image_wall_seconds":
                        nested(
                            result,
                            "timing",
                            "image_wall_seconds_before_artifact_hashing",
                        ),

                    "clean_E":
                        nested(
                            result,
                            "clean",
                            "E",
                        ),

                    "adv_E":
                        nested(
                            result,
                            "adversarial",
                            "E",
                        ),

                    "adv_score":
                        nested(
                            result,
                            "adversarial",
                            "score",
                        ),

                    "physical_linf":
                        nested(
                            result,
                            "adversarial",
                            "physical_linf",
                        ),

                    "relative_E_degradation":
                        nested(
                            result,
                            "degradation",
                            "relative_E",
                        ),

                    "protocol_sha256":
                        result.get(
                            "protocol_sha256"
                        ),

                    "complete_mtime":
                        datetime.fromtimestamp(
                            marker.stat().st_mtime
                        ).astimezone().isoformat(),
                }
            )

        except Exception as exc:
            invalid.append(
                {
                    "marker":
                        str(marker),

                    "error_type":
                        type(exc).__name__,

                    "error":
                        str(exc),
                }
            )

    return rows, invalid


def inventory_workers():
    rows = []

    if not ATTACK_ROOT.is_dir():
        return rows

    for status_path in sorted(
        ATTACK_ROOT.rglob(
            "worker_status.json"
        )
    ):
        try:
            x = json.loads(
                status_path.read_text()
            )

            rows.append(
                {
                    "hostname":
                        socket.gethostname(),

                    "path":
                        str(
                            status_path.relative_to(
                                ROOT
                            )
                        ),

                    "status":
                        x.get("status"),

                    "recorded_hostname":
                        x.get("hostname"),

                    "shard_id":
                        x.get("shard_id"),

                    "n_assigned":
                        x.get("n_assigned"),

                    "n_complete_or_skip":
                        x.get(
                            "n_complete_or_skip"
                        ),

                    "n_failures":
                        x.get("n_failures"),

                    "protocol_sha256":
                        x.get(
                            "protocol_sha256"
                        ),

                    "plan_freeze_sha256":
                        x.get(
                            "plan_freeze_sha256"
                        ),

                    "sha256":
                        sha256_file(
                            status_path
                        ),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "hostname":
                        socket.gethostname(),

                    "path":
                        str(
                            status_path.relative_to(
                                ROOT
                            )
                        ),

                    "status":
                        "READ_ERROR",

                    "error":
                        str(exc),
                }
            )

    return rows


def inventory_logs():
    rows = []

    if not LOG_ROOT.is_dir():
        return rows

    patterns = [
        "trufor_stage26*.log",
        "trufor_phase2*.log",
        "trufor_stage24*.log",
        "trufor_stage25*.log",
    ]

    seen = set()

    for pattern in patterns:
        for path in sorted(
            LOG_ROOT.glob(pattern)
        ):
            if path in seen:
                continue

            seen.add(path)

            rows.append(
                {
                    "hostname":
                        socket.gethostname(),

                    "filename":
                        path.name,

                    "size_bytes":
                        path.stat().st_size,

                    "mtime":
                        datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).astimezone().isoformat(),

                    "sha256":
                        sha256_file(path),
                }
            )

    return rows


def plan_info():
    info = {}

    candidates = {
        "phase1_plan":
            ATTACK_ROOT
            / "multigpu_sharding"
            / "multigpu_plan_freeze.json",

        "phase2_plan":
            ATTACK_ROOT
            / "phase2_multigpu_sharding"
            / "phase2_plan_freeze.json",
    }

    for name, path in candidates.items():
        if path.is_file():
            info[name] = {
                "path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "sha256":
                    sha256_file(path),
            }
        else:
            info[name] = None

    return info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        default=str(
            ROOT
            / "execution_fingerprints"
            / "execution_audit"
        ),
    )

    args = parser.parse_args()

    host = socket.gethostname()

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete, invalid = (
        inventory_complete_results()
    )

    workers = inventory_workers()
    logs = inventory_logs()
    plans = plan_info()

    complete_df = pd.DataFrame(
        complete
    )

    worker_df = pd.DataFrame(
        workers
    )

    log_df = pd.DataFrame(
        logs
    )

    complete_csv = (
        out_dir
        / f"{host}_complete_inventory.csv"
    )

    worker_csv = (
        out_dir
        / f"{host}_worker_status_inventory.csv"
    )

    log_csv = (
        out_dir
        / f"{host}_log_inventory.csv"
    )

    summary_json = (
        out_dir
        / f"{host}_execution_summary.json"
    )

    complete_df.to_csv(
        complete_csv,
        index=False,
    )

    worker_df.to_csv(
        worker_csv,
        index=False,
    )

    log_df.to_csv(
        log_csv,
        index=False,
    )

    generation_counts = Counter(
        r["generation"]
        for r in complete
    )

    shard_counts = Counter(
        (
            r["generation"],
            r["shard_id"],
        )
        for r in complete
    )

    image_counts = Counter(
        r["image_path"]
        for r in complete
    )

    local_duplicates = sorted(
        image
        for image, n
        in image_counts.items()
        if n > 1
    )

    summary = {
        "hostname":
            host,

        "protocol_sha256":
            PROTOCOL_SHA,

        "expected_phase1_plan_sha256":
            OLD_PLAN_SHA,

        "expected_phase2_plan_sha256":
            PHASE2_PLAN_SHA,

        "plans":
            plans,

        "complete_markers":
            len(complete),

        "unique_image_paths":
            len(image_counts),

        "local_duplicate_image_paths":
            local_duplicates,

        "invalid_complete_markers":
            invalid,

        "generation_counts":
            dict(
                generation_counts
            ),

        "shard_counts":
            {
                f"{g}:shard_{s}":
                    n
                for (g, s), n
                in sorted(
                    shard_counts.items(),
                    key=lambda x: (
                        str(x[0][0]),
                        -1
                        if x[0][1] is None
                        else x[0][1],
                    ),
                )
            },

        "worker_status_files":
            len(workers),

        "log_files":
            len(logs),

        "files": {
            "complete_inventory":
                complete_csv.name,

            "worker_status_inventory":
                worker_csv.name,

            "log_inventory":
                log_csv.name,
        },
    }

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "=" * 68
    )

    print(
        "TRUFOR EXECUTION INVENTORY"
    )

    print(
        "=" * 68
    )

    print(
        "HOST:",
        host,
    )

    print(
        "COMPLETE markers:",
        len(complete),
    )

    print(
        "UNIQUE image paths:",
        len(image_counts),
    )

    print(
        "INVALID markers:",
        len(invalid),
    )

    print(
        "LOCAL duplicates:",
        len(local_duplicates),
    )

    print()

    print(
        "BY GENERATION"
    )

    for k, v in sorted(
        generation_counts.items()
    ):
        print(
            f"  {k:24s} {v}"
        )

    print()

    print(
        "BY GENERATION / SHARD"
    )

    for (g, s), n in sorted(
        shard_counts.items(),
        key=lambda x: (
            str(x[0][0]),
            -1
            if x[0][1] is None
            else x[0][1],
        ),
    ):
        print(
            f"  {g:24s} "
            f"shard={str(s):>4s} "
            f"{n}"
        )

    print()

    print(
        "PLAN FILES"
    )

    for name, value in plans.items():
        print(
            f"  {name}:",
            (
                value["sha256"]
                if value
                else "ABSENT"
            ),
        )

    print()

    print(
        "OUTPUT DIRECTORY:",
        out_dir,
    )

    print(
        "INVENTORY PASS"
        if not invalid
        else "INVENTORY COMPLETED WITH ERRORS"
    )


if __name__ == "__main__":
    main()
