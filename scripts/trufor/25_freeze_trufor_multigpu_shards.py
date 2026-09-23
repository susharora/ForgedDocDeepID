#!/usr/bin/env python3
"""Stage 25: freeze deterministic cost-balanced multi-GPU shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import trufor_adversarial_attack_common as attack
import trufor_multigpu_common as mg
from trufor_common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--n-shards", type=int, default=mg.DEFAULT_N_SHARDS)
    args = parser.parse_args()

    conflicts = mg.conflicting_unsharded_stage24_processes()
    if conflicts:
        raise RuntimeError(
            "Unsharded Stage 24 is still running. Stop it before Stage 25:\n"
            + "\n".join(conflicts)
        )

    workers = mg.active_shard_workers()
    if workers:
        raise RuntimeError(
            "Shard workers already running; refusing to re-plan:\n"
            + "\n".join(workers)
        )

    plan_root = mg.multigpu_root(args.run_tag)
    freeze_path = mg.plan_freeze_path(args.run_tag)

    if freeze_path.exists():
        raise RuntimeError(
            f"Frozen plan already exists:\n{freeze_path}\n"
            "Do not re-shard after workers start."
        )

    protocol_cfg = attack.protocol_config_path(args.run_tag)
    if not protocol_cfg.is_file():
        raise RuntimeError(f"Missing frozen protocol config: {protocol_cfg}")

    protocol_sha = sha256_file(protocol_cfg)
    if protocol_sha != mg.EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Protocol SHA mismatch:\n"
            f"expected={mg.EXPECTED_PROTOCOL_SHA256}\nactual={protocol_sha}"
        )

    selection = mg.build_full_selection()
    completed = mg.completed_snapshot_frame(
        args.run_tag,
        selection,
        protocol_sha,
    )

    completed_paths = (
        set(completed["image_path"].tolist()) if len(completed) else set()
    )
    remaining = selection.loc[
        ~selection["image_path"].isin(completed_paths)
    ].copy()

    if len(completed) + len(remaining) != mg.EXPECTED_N:
        raise RuntimeError("Completed + remaining != 1330")

    remaining["estimated_seconds"] = mg.estimate_remaining_costs(
        remaining,
        completed,
    )
    assigned = mg.greedy_lpt_assign(remaining, args.n_shards)

    plan_root.mkdir(parents=True, exist_ok=True)
    (plan_root / "shards").mkdir(parents=True, exist_ok=True)

    full_path = plan_root / "full_selection.csv"
    completed_path = plan_root / "completed_before_sharding.csv"
    remaining_path = plan_root / "remaining_selection.csv"
    assignment_path = plan_root / "shard_assignment.csv"
    cfg_snapshot_path = plan_root / "frozen_attack_protocol_config.json"

    attack.atomic_write_csv(full_path, selection)
    attack.atomic_write_csv(completed_path, completed)
    attack.atomic_write_csv(remaining_path, remaining)
    attack.atomic_write_csv(
        assignment_path,
        assigned.sort_values("global_order").reset_index(drop=True),
    )
    mg.atomic_copy_bytes(protocol_cfg, cfg_snapshot_path)

    shard_stats = []
    shard_files = []

    for shard_id in range(args.n_shards):
        shard = (
            assigned.loc[assigned["shard_id"] == shard_id]
            .copy()
            .sort_values(
                ["estimated_seconds", "global_order"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )
        shard.insert(1, "shard_order", range(1, len(shard) + 1))

        path = mg.shard_csv_path(args.run_tag, shard_id)
        attack.atomic_write_csv(path, shard)
        shard_files.append(path)

        shard_stats.append(
            {
                "shard_id": shard_id,
                "n_images": int(len(shard)),
                "estimated_hours": float(
                    shard["estimated_seconds"].sum() / 3600.0
                ),
                "huawei": int((shard["hardware_source"] == "huawei").sum()),
                "iphone15pro": int(
                    (shard["hardware_source"] == "iphone15pro").sum()
                ),
                "scan": int((shard["hardware_source"] == "scan").sum()),
            }
        )

    manifest_files = [
        full_path,
        completed_path,
        remaining_path,
        assignment_path,
        cfg_snapshot_path,
        *shard_files,
    ]

    manifest_sha = {
        str(p.relative_to(plan_root)): sha256_file(p)
        for p in manifest_files
    }

    freeze = {
        "status": "FROZEN",
        "stage": "25_freeze_trufor_multigpu_sharding",
        "run_tag": args.run_tag,
        "protocol_sha256": protocol_sha,
        "scientific_protocol_changed": False,
        "attack_common_sha256": sha256_file(
            Path(attack.__file__).resolve()
        ),
        "n_total": int(len(selection)),
        "n_completed_before_sharding": int(len(completed)),
        "n_remaining": int(len(remaining)),
        "n_shards": int(args.n_shards),
        "shard_stats": shard_stats,
        "manifest_sha256": manifest_sha,
        "cost_model": (
            "median wall_seconds of up to 25 nearest completed images on "
            "same hardware by native pixel count; greedy LPT balancing"
        ),
    }
    attack.atomic_write_json(freeze_path, freeze)

    print("TRUFOR STAGE 25 — MULTI-GPU SHARD PLAN FROZEN")
    print(f"protocol SHA:             {protocol_sha}")
    print(f"legacy COMPLETE snapshot: {len(completed)}/{mg.EXPECTED_N}")
    print(f"remaining:                {len(remaining)}")
    print(f"shards:                   {args.n_shards}")
    print(f"plan SHA256:              {sha256_file(freeze_path)}")
    print()
    print(pd.DataFrame(shard_stats).to_string(index=False))
    print()
    print("STAGE 25 PASS")
    print("Do not restart the old unsharded Stage-24 process.")


if __name__ == "__main__":
    main()
