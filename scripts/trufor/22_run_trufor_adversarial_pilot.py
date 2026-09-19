#!/usr/bin/env python3
"""
Stage 22: execute the frozen six-image x 10-step TruFor adversarial pilot.

Prerequisites:
- Stage-19 protocol freeze;
- corrected Stage-20 v2 largest-image memory gate PASS;
- Stage-21 exact execution implementation freeze.

Operational resilience:
- one image is the atomic unit;
- completed/hash-valid images are skipped on restart;
- interrupted/incomplete image restarts from clean;
- no mid-image checkpointing;
- by default an image failure is recorded and the runner continues to the
  remaining pilot images, maximizing unattended diagnostic value;
- --stop-on-error is available for interactive debugging.

Scientific attack configuration is not tunable from this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from trufor_attack_pilot_common import load_frozen_threshold
from trufor_adversarial_attack_common import (
    attack_root,
    atomic_write_json,
    environment_record,
    image_output_dir,
    load_two_models_cpu_first,
    patch_memory_efficient_attack_model,
    pilot_root,
    resolve_cuda_device,
    run_one_pilot_image,
    validate_backend_record,
    verify_protocol_freeze,
)
import importlib.util

_STAGE21_PATH = Path(__file__).resolve().parent / "21_freeze_trufor_adversarial_pilot_execution.py"
_STAGE21_SPEC = importlib.util.spec_from_file_location(
    "trufor_stage21_pilot_freeze",
    _STAGE21_PATH,
)
if _STAGE21_SPEC is None or _STAGE21_SPEC.loader is None:
    raise RuntimeError(f"Cannot import Stage 21 helper: {_STAGE21_PATH}")
_STAGE21 = importlib.util.module_from_spec(_STAGE21_SPEC)
_STAGE21_SPEC.loader.exec_module(_STAGE21)
validate_execution_freeze = _STAGE21.validate_execution_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on the first image failure. Default: continue.",
    )
    args = parser.parse_args()

    backend = validate_backend_record()
    cfg, protocol_sha, selection, stage19 = verify_protocol_freeze(args.run_tag)
    execution_freeze, execution_freeze_sha = validate_execution_freeze(
        args.run_tag
    )
    _, threshold = load_frozen_threshold()

    device = resolve_cuda_device(args.gpu)
    torch.cuda.set_device(device)

    print("TRUFOR STAGE 22 — SIX-IMAGE x 10-STEP ADVERSARIAL PILOT")
    print(f"run tag:              {args.run_tag}")
    print(f"device:               {device}")
    print(f"GPU:                  {torch.cuda.get_device_name(device)}")
    print(f"protocol SHA:         {protocol_sha}")
    print(f"execution freeze SHA: {execution_freeze_sha}")
    print(f"threshold:            {threshold:.15f}")
    print("memory gate:          validated v2 prerequisite PASS")
    print("resume:               per-image atomic")
    print(
        "on image error:       "
        + ("STOP" if args.stop_on_error else "record failure and CONTINUE")
    )
    print()

    t_load = time.perf_counter()
    attack_model, reference_model, load_meta = load_two_models_cpu_first(device)
    route_meta = patch_memory_efficient_attack_model(attack_model)
    load_seconds = time.perf_counter() - t_load

    if route_meta.get("attention_chunk_activation_checkpointing") is not True:
        raise RuntimeError("Validated v2 attention checkpoint route is not active")
    if route_meta.get("attention_modules") != 32:
        raise RuntimeError("Unexpected attention-module count")
    if route_meta.get("mlp_modules") != 32:
        raise RuntimeError("Unexpected MLP-module count")

    print("CPU-first dual-model load PASS")
    print(json.dumps(load_meta, indent=2, sort_keys=True))
    print("Validated memory route:")
    print(json.dumps(route_meta, indent=2, sort_keys=True))
    print(f"model load/patch wall time: {load_seconds:.1f}s")
    print()

    root = pilot_root(args.run_tag)
    root.mkdir(parents=True, exist_ok=True)

    status_path = root / "stage22_run_status.json"
    rows = selection.sort_values("pilot_order").reset_index(drop=True)

    statuses = []
    failures = []

    for _, row in rows.iterrows():
        order = int(row["pilot_order"])
        print(
            f"[{order}/6] {row['variant']} | "
            f"{int(row['native_width'])}x{int(row['native_height'])} | "
            f"clean score={float(row['clean_score_stage4']):.6f} | "
            f"{row['image_path']}",
            flush=True,
        )

        image_t0 = time.perf_counter()
        try:
            state = run_one_pilot_image(
                run_tag=args.run_tag,
                selection_row=row,
                protocol_sha=protocol_sha,
                attack_model=attack_model,
                reference_model=reference_model,
                route_metadata=route_meta,
                threshold=threshold,
                device=device,
            )

            if state["status"] == "SKIP_COMPLETE":
                statuses.append(
                    {
                        "pilot_order": order,
                        "image_path": str(row["image_path"]),
                        "status": "SKIP_COMPLETE",
                        "wall_seconds_this_invocation": float(
                            time.perf_counter() - image_t0
                        ),
                    }
                )
                print("  hash-valid COMPLETE already exists — skipped.", flush=True)
            else:
                r = state["result"]
                statuses.append(
                    {
                        "pilot_order": order,
                        "image_path": str(row["image_path"]),
                        "status": "PASS",
                        "wall_seconds_this_invocation": float(
                            time.perf_counter() - image_t0
                        ),
                        "clean_E": float(r["clean"]["E"]),
                        "adv_E": float(r["adversarial"]["E"]),
                        "relative_E_degradation": float(
                            r["degradation"]["relative_E"]
                        ),
                        "adv_score": float(r["adversarial"]["score"]),
                        "adv_score_margin": float(
                            r["adversarial"]["score_margin"]
                        ),
                        "physical_linf": float(
                            r["adversarial"]["physical_linf"]
                        ),
                        "accepted_steps": int(
                            r["optimisation"]["accepted_steps"]
                        ),
                        "boundary_pullbacks": int(
                            r["optimisation"]["boundary_pullbacks"]
                        ),
                    }
                )
                print(
                    "  COMPLETE | "
                    f"E {r['clean']['E']:.6f} -> {r['adversarial']['E']:.6f} | "
                    f"relΔE={r['degradation']['relative_E']:.4f} | "
                    f"score={r['adversarial']['score']:.6f} | "
                    f"margin={r['adversarial']['score_margin']:.6f} | "
                    f"Linf={r['adversarial']['physical_linf']:.8f} | "
                    f"accepted={r['optimisation']['accepted_steps']}/10 | "
                    f"pullbacks={r['optimisation']['boundary_pullbacks']}",
                    flush=True,
                )

        except Exception as exc:
            failure = {
                "pilot_order": order,
                "image_path": str(row["image_path"]),
                "status": "FAILED_INCOMPLETE",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds_this_invocation": float(
                    time.perf_counter() - image_t0
                ),
            }
            failures.append(failure)
            statuses.append(failure)
            print(
                f"  FAILED_INCOMPLETE | {type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.stop_on_error:
                atomic_write_json(
                    status_path,
                    {
                        "status": "INCOMPLETE_FAILURE",
                        "protocol_sha256": protocol_sha,
                        "execution_freeze_sha256": execution_freeze_sha,
                        "rows": statuses,
                    },
                )
                raise

        finally:
            gc.collect()
            torch.cuda.empty_cache()

        atomic_write_json(
            status_path,
            {
                "status": (
                    "RUNNING_WITH_FAILURES" if failures else "RUNNING"
                ),
                "protocol_sha256": protocol_sha,
                "execution_freeze_sha256": execution_freeze_sha,
                "rows": statuses,
            },
        )
        print()

    n_pass_or_skip = sum(
        s["status"] in {"PASS", "SKIP_COMPLETE"} for s in statuses
    )

    final_status = {
        "status": "PASS_ALL_SIX" if not failures and n_pass_or_skip == 6 else "INCOMPLETE_FAILURES",
        "stage": "22_run_trufor_adversarial_pilot",
        "run_tag": args.run_tag,
        "protocol_sha256": protocol_sha,
        "execution_freeze_sha256": execution_freeze_sha,
        "environment": environment_record(device),
        "model_load": load_meta,
        "memory_route": route_meta,
        "n_images_expected": 6,
        "n_pass_or_skip": int(n_pass_or_skip),
        "n_failures": int(len(failures)),
        "rows": statuses,
    }
    atomic_write_json(status_path, final_status)

    print("STAGE 22 FINISHED")
    print(f"complete/skip: {n_pass_or_skip}/6")
    print(f"failures:      {len(failures)}")
    print(f"status file:   {status_path}")
    print()

    if failures:
        print(
            "One or more images failed, but independent later images were still "
            "attempted. Rerunning Stage 22 restarts only incomplete images."
        )
        raise RuntimeError(
            f"Stage 22 finished with {len(failures)} incomplete pilot image(s)"
        )

    print("STAGE 22 PASS — ALL SIX PILOT IMAGES COMPLETE")
    print("Next: Stage 23 validation/summary. Do not start the 1330-image run.")


if __name__ == "__main__":
    main()
