#!/usr/bin/env python3

from pathlib import Path
import argparse
import gc
import json
import socket
import time
import traceback

import torch
import pandas as pd

import trufor_multigpu_common as mg
import trufor_adversarial_attack_common as attack


RUN_TAG = "LABPC"

PHASE2_DIR = (
    attack.attack_root(RUN_TAG)
    / "phase2_multigpu_sharding"
)

PHASE2_OUTPUT = (
    attack.attack_root(RUN_TAG)
    / "phase2_multigpu_outputs"
)


def phase2_validate():

    freeze_path = (
        PHASE2_DIR
        / "phase2_plan_freeze.json"
    )

    if not freeze_path.is_file():
        raise RuntimeError(
            f"Missing Phase-2 freeze: {freeze_path}"
        )

    freeze = json.loads(
        freeze_path.read_text()
    )

    if freeze["status"] != "FROZEN":
        raise RuntimeError(
            "Phase-2 plan is not frozen"
        )

    return freeze


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-tag",
        default="LABPC",
    )

    parser.add_argument(
        "--shard-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    args = parser.parse_args()


    freeze = phase2_validate()

    n_shards = int(
        freeze["n_shards"]
    )

    if args.shard_id < 0 or args.shard_id >= n_shards:
        raise RuntimeError(
            f"shard-id must be 0..{n_shards-1}, got {args.shard_id}"
        )


    shard_csv = (
        PHASE2_DIR
        / "shards"
        / f"shard_{args.shard_id:02d}.csv"
    )

    if not shard_csv.is_file():
        raise RuntimeError(
            f"Missing shard CSV: {shard_csv}"
        )


    shard = pd.read_csv(
        shard_csv,
        keep_default_na=False
    )


    print("="*60)
    print("TRUFOR PHASE-2 SHARD WORKER")
    print("="*60)

    print("hostname:",
          socket.gethostname())

    print("shard:",
          args.shard_id)

    print("assigned images:",
          len(shard))

    print("plan:",
          mg.sha256_file(
              PHASE2_DIR /
              "phase2_plan_freeze.json"
          ))

    print("GPU:",
          torch.cuda.get_device_name(args.gpu)
          if torch.cuda.is_available()
          else "NO CUDA")

    print("="*60)


    # Reuse the validated Stage-26 execution path
    # but inject the Phase-2 shard dataframe.

    statuses=[]
    failures=[]


    output_root = (
        PHASE2_OUTPUT
        / f"shard_{args.shard_id:02d}"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )


    for _, row in shard.iterrows():

        t0=time.perf_counter()

        try:

            result = attack.run_one_pilot_image(
                run_tag=RUN_TAG,
                selection_row=row,
                protocol_sha=freeze["protocol_sha256"],
                attack_model=None,
                reference_model=None,
                route_metadata={
                    "phase2": True,
                    "shard_id": args.shard_id,
                },
                threshold=float(
                    row["frozen_threshold"]
                ),
                device=torch.device(
                    f"cuda:{args.gpu}"
                ),
            )

            statuses.append(
                {
                    "image_path":
                        row["image_path"],
                    "status":
                        result["status"],
                    "seconds":
                        time.perf_counter()-t0,
                }
            )


        except Exception as exc:

            failures.append(
                {
                    "image_path":
                        row["image_path"],
                    "exception":
                        str(exc),
                    "traceback":
                        traceback.format_exc(),
                }
            )


        finally:
            gc.collect()
            torch.cuda.empty_cache()


    status = {
        "status":
            "PASS_ALL_ASSIGNED"
            if not failures
            else "INCOMPLETE_FAILURES",

        "hostname":
            socket.gethostname(),

        "shard_id":
            args.shard_id,

        "n_assigned":
            len(shard),

        "completed":
            len(statuses),

        "failures":
            len(failures),

        "rows":
            statuses + failures,
    }


    out = (
        output_root
        / "worker_status.json"
    )

    out.write_text(
        json.dumps(
            status,
            indent=2
        )
    )

    print(json.dumps(
        status,
        indent=2
    ))


if __name__ == "__main__":
    main()
