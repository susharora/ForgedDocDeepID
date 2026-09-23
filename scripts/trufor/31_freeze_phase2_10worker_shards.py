#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd

import trufor_multigpu_common as mg
import trufor_adversarial_attack_common as attack
from trufor_common import sha256_file


RUN_TAG = "LABPC"
N_SHARDS = 10

OLD_PLAN_SHA = (
    "672fb656729a3a23210df373c19956db966dc6ef9470afc21483d5917a6bccf8"
)

METADATA_FILES = [
    "/tmp/IMTA135_completed_metadata.json",
    "/tmp/IMTA134_completed_metadata.json",
    "/tmp/IMTA136_completed_metadata.json",
]


def phase2_root():
    return (
        attack.attack_root(RUN_TAG)
        / "phase2_multigpu_sharding"
    )


def main():

    root = phase2_root()

    if root.exists():
        raise RuntimeError(
            f"Existing Phase-2 directory exists:\n{root}"
        )

    root.mkdir(parents=True)
    (root / "shards").mkdir()

    print("===== BUILD FULL SELECTION =====")

    selection = mg.build_full_selection()

    print("full population:", len(selection))


    print()
    print("===== LOAD COMPLETION METADATA =====")

    completed_records = []

    for f in METADATA_FILES:

        p = Path(f)

        if not p.exists():
            raise RuntimeError(
                f"Missing metadata file: {p}"
            )

        rows = json.loads(
            p.read_text()
        )

        print(
            p.name,
            "records:",
            len(rows)
        )

        completed_records.extend(rows)


    print()
    print(
        "COMPLETE METADATA RECORDS:",
        len(completed_records)
    )


    completed_df = pd.DataFrame(
        completed_records
    )


    if completed_df["image_path"].duplicated().any():

        dup = (
            completed_df
            [completed_df["image_path"].duplicated(
                keep=False
            )]
            ["image_path"]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate completed images:\n"
            + "\n".join(dup[:20])
        )


    print(
        "UNIQUE COMPLETED:",
        len(completed_df)
    )


    print()
    print("===== RECONSTRUCT COST DATA =====")


    selection_idx = selection.set_index(
        "image_path"
    )


    completed_rows = []

    for _, row in completed_df.iterrows():

        image = row["image_path"]

        if image not in selection_idx.index:
            raise RuntimeError(
                f"Completed image outside population: {image}"
            )

        sel = selection_idx.loc[image]

        completed_rows.append(
            {
                "image_path": image,
                "hardware_source":
                    str(sel["hardware_source"]),
                "native_pixels":
                    int(sel["native_pixels"]),
                "wall_seconds":
                    float(row["wall_seconds"]),
            }
        )


    completed_cost_df = pd.DataFrame(
        completed_rows
    )


    remaining = selection.loc[
        ~selection["image_path"].isin(
            completed_cost_df["image_path"]
        )
    ].copy()


    print(
        "completed:",
        len(completed_cost_df)
    )

    print(
        "remaining:",
        len(remaining)
    )


    if len(completed_cost_df)+len(remaining)!=1330:
        raise RuntimeError(
            "Population accounting failure"
        )


    print()
    print("===== COST ESTIMATION =====")

    remaining["estimated_seconds"] = (
        mg.estimate_remaining_costs(
            remaining,
            completed_cost_df,
        )
    )


    assigned = mg.greedy_lpt_assign(
        remaining,
        N_SHARDS,
    )


    print()
    print("===== WRITE SHARDS =====")


    attack.atomic_write_csv(
        root / "completed_snapshot.csv",
        completed_cost_df,
    )

    attack.atomic_write_csv(
        root / "remaining_selection.csv",
        remaining,
    )

    attack.atomic_write_csv(
        root / "shard_assignment.csv",
        assigned.sort_values(
            "global_order"
        ),
    )


    stats=[]


    for shard_id in range(N_SHARDS):

        shard = (
            assigned[
                assigned["shard_id"] == shard_id
            ]
            .copy()
            .sort_values(
                [
                    "estimated_seconds",
                    "global_order"
                ],
                ascending=[
                    False,
                    True
                ]
            )
        )

        shard.insert(
            1,
            "shard_order",
            range(1,len(shard)+1)
        )

        attack.atomic_write_csv(
            root / "shards" /
            f"shard_{shard_id:02d}.csv",
            shard,
        )

        stats.append(
            {
                "shard_id": shard_id,
                "n_images": int(len(shard)),
                "estimated_hours":
                    float(
                        shard["estimated_seconds"]
                        .sum()/3600
                    ),
            }
        )


    freeze = {

        "status": "FROZEN",

        "stage":
            "31_freeze_phase2_10worker_shards",

        "scientific_protocol_changed":
            False,

        "old_plan_sha256":
            OLD_PLAN_SHA,

        "protocol_sha256":
            mg.EXPECTED_PROTOCOL_SHA256,

        "n_total":
            1330,

        "completed_before_phase2":
            int(len(completed_cost_df)),

        "remaining":
            int(len(remaining)),

        "n_shards":
            N_SHARDS,

        "stats":
            stats,
    }


    attack.atomic_write_json(
        root / "phase2_plan_freeze.json",
        freeze,
    )


    print()
    print("===== PHASE2 PLAN FROZEN =====")

    print(
        "plan SHA256:",
        sha256_file(
            root / "phase2_plan_freeze.json"
        )
    )

    print(
        pd.DataFrame(stats)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
