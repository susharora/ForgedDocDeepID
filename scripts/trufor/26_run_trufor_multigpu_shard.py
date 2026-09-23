#!/usr/bin/env python3
"""Stage 26: run one frozen multi-GPU TruFor shard."""

from __future__ import annotations

import argparse
import gc
import json
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import trufor_adversarial_attack_common as attack
import trufor_multigpu_common as mg
from trufor_common import ROOT, sha256_file


def patch_shard_output_dir(run_tag: str, shard_id: int):
    root = mg.shard_root(run_tag, shard_id)

    def _image_output_dir(
        run_tag_arg: str,
        pilot_order: int,
        image_path: str,
    ) -> Path:
        if run_tag_arg != run_tag:
            raise RuntimeError(f"Unexpected run_tag: {run_tag_arg}")
        token = attack.stable_token(image_path, 12)
        return root / "images" / f"{int(pilot_order):04d}_{token}"

    attack.image_output_dir = _image_output_dir
    return root


def one_step_preflight(
    row: pd.Series,
    attack_model,
    reference_model,
    threshold: float,
    device: torch.device,
) -> dict:
    cache_path = ROOT / str(row["cache_path"])
    rgb = attack.load_rgb_uint8(cache_path)
    H, W = int(rgb.shape[0]), int(rgb.shape[1])

    regions = attack.load_regions()
    union_mask_np, _ = attack.build_union_mask(
        regions,
        str(row["image_path"]),
        H,
        W,
    )
    altered_mask = torch.from_numpy(
        union_mask_np.astype(np.float32, copy=False)
    ).unsqueeze(0).to(device=device)

    clean_x = attack.canonical_model_tensor_from_uint8(rgb, device)
    evaluator = attack.ReferenceEvaluator(
        reference_model,
        altered_mask,
        threshold,
        device,
    )
    clean_eval = evaluator.evaluate(clean_x, return_map=False)

    if not clean_eval["feasible"]:
        raise RuntimeError("Preflight image is not clean-correct")

    grad_state = attack.gradient_step(
        attack_model,
        clean_x,
        altered_mask,
        device,
    )

    gap = abs(float(grad_state["attack_E"]) - float(clean_eval["E"]))
    if gap > attack.ATTACK_REFERENCE_E_PARITY_ATOL:
        raise RuntimeError(f"Attack/reference E parity failed: {gap}")

    grad = grad_state.pop("grad")
    proposal = attack.propose_sign_step(clean_x, clean_x, grad)
    audit = attack.physical_bounds_audit(clean_x, proposal)

    if abs(
        float(audit["physical_linf"]) - attack.ALPHA_PHYSICAL
    ) > 2e-7:
        raise RuntimeError("Preflight alpha/L_inf regression failed")

    proposal_eval = evaluator.evaluate(proposal, return_map=False)

    result = {
        "status": "PASS",
        "image_path": str(row["image_path"]),
        "native_MP": H * W / 1e6,
        "clean_score": float(clean_eval["score"]),
        "clean_E": float(clean_eval["E"]),
        "attack_E": float(grad_state["attack_E"]),
        "attack_reference_E_abs_gap": float(gap),
        "proposal_score": float(proposal_eval["score"]),
        "proposal_E": float(proposal_eval["E"]),
        "proposal_classification_feasible": bool(
            proposal_eval["feasible"]
        ),
        "physical_linf": float(audit["physical_linf"]),
        "gradient_seconds": float(grad_state["gradient_seconds"]),
        "gradient_peak_allocated_gib": float(
            grad_state["gradient_peak_allocated_gib"]
        ),
        "total_query_chunks": int(grad_state["total_query_chunks"]),
        "split_attention_modules": int(
            grad_state["split_attention_modules"]
        ),
        "max_chunks_one_attention": int(
            grad_state["max_chunks_one_attention"]
        ),
    }

    del grad, proposal, altered_mask, clean_x
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    freeze, freeze_sha = mg.validate_plan(args.run_tag)
    n_shards = int(freeze["n_shards"])

    if not (0 <= args.shard_id < n_shards):
        raise RuntimeError(
            f"shard-id must be 0..{n_shards - 1}, got {args.shard_id}"
        )

    shard_csv = mg.shard_csv_path(args.run_tag, args.shard_id)
    shard = pd.read_csv(shard_csv, keep_default_na=False)

    cfg = json.loads(
        (
            mg.multigpu_root(args.run_tag)
            / "frozen_attack_protocol_config.json"
        ).read_text()
    )
    protocol_sha = str(freeze["protocol_sha256"])
    threshold = float(
        cfg["classification_constraint"]["threshold"]
    )

    attack.validate_backend_record()
    device = attack.resolve_cuda_device(args.gpu)
    torch.cuda.set_device(device)

    output_root = patch_shard_output_dir(
        args.run_tag,
        args.shard_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    status_path = output_root / "worker_status.json"
    provenance_path = output_root / "worker_provenance.json"

    print("TRUFOR STAGE 26 — MULTI-GPU SHARD WORKER")
    print(f"hostname:           {socket.gethostname()}")
    print(f"shard:              {args.shard_id:02d}")
    print(f"assigned:           {len(shard)}")
    print(f"GPU:                {torch.cuda.get_device_name(device)}")
    print(f"protocol SHA:       {protocol_sha}")
    print(f"plan freeze SHA:    {freeze_sha}")
    print(f"shard manifest SHA: {sha256_file(shard_csv)}")
    print(f"output root:        {output_root}")
    print()

    visible_complete = mg.existing_complete_image_paths(
        args.run_tag,
        protocol_sha,
    )
    own_paths = set(shard["image_path"].astype(str))

    foreign = {
        p: d
        for p, d in visible_complete.items()
        if p in own_paths
        and not str(d).startswith(str(output_root / "images"))
    }
    if foreign:
        raise RuntimeError(
            "Assigned images already COMPLETE outside own shard. "
            f"Examples: {list(foreign.items())[:10]}"
        )

    t_load = time.perf_counter()
    attack_model, reference_model, load_meta = (
        attack.load_two_models_cpu_first(device)
    )
    route_meta = attack.patch_memory_efficient_attack_model(
        attack_model
    )
    load_seconds = time.perf_counter() - t_load

    if route_meta.get(
        "attention_chunk_activation_checkpointing"
    ) is not True:
        raise RuntimeError("Validated v2 route inactive")

    preflight = None
    if not args.skip_preflight and len(shard):
        preflight_row = (
            shard.sort_values(
                ["native_pixels", "global_order"],
                ascending=[False, True],
            )
            .iloc[0]
        )
        print(
            "preflight largest assigned image: "
            f"{preflight_row['image_path']}"
        )
        preflight = one_step_preflight(
            preflight_row,
            attack_model,
            reference_model,
            threshold,
            device,
        )
        print(
            "preflight PASS | "
            f"{preflight['native_MP']:.3f} MP | "
            f"grad={preflight['gradient_seconds']:.1f}s | "
            f"Linf={preflight['physical_linf']:.12f}"
        )
        print()

    provenance = {
        "status": "STARTED",
        "stage": "26_run_trufor_multigpu_shard",
        "hostname": socket.gethostname(),
        "shard_id": int(args.shard_id),
        "protocol_sha256": protocol_sha,
        "plan_freeze_sha256": freeze_sha,
        "shard_manifest_sha256": sha256_file(shard_csv),
        "attack_common_sha256": sha256_file(
            Path(attack.__file__).resolve()
        ),
        "environment": attack.environment_record(device),
        "model_load": load_meta,
        "memory_route": route_meta,
        "preflight": preflight,
        "model_load_seconds": float(load_seconds),
    }
    attack.atomic_write_json(provenance_path, provenance)

    statuses = []
    failures = []

    for _, row in shard.iterrows():
        global_order = int(row["global_order"])
        shard_order = int(row["shard_order"])

        print(
            f"[shard {args.shard_id:02d} {shard_order}/{len(shard)} | "
            f"global {global_order}/1330] "
            f"{row['variant']} {row['hardware_source']} | "
            f"{int(row['native_width'])}x{int(row['native_height'])} | "
            f"{row['image_path']}",
            flush=True,
        )

        t0 = time.perf_counter()
        try:
            state = attack.run_one_pilot_image(
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
                entry = {
                    "global_order": global_order,
                    "shard_order": shard_order,
                    "image_path": str(row["image_path"]),
                    "status": "SKIP_COMPLETE",
                    "wall_seconds_this_invocation": float(
                        time.perf_counter() - t0
                    ),
                }
                print("  hash-valid COMPLETE already exists — skipped.")
            else:
                r = state["result"]
                entry = {
                    "global_order": global_order,
                    "shard_order": shard_order,
                    "image_path": str(row["image_path"]),
                    "status": "PASS",
                    "wall_seconds_this_invocation": float(
                        time.perf_counter() - t0
                    ),
                    "clean_E": float(r["clean"]["E"]),
                    "adv_E": float(r["adversarial"]["E"]),
                    "adv_score": float(r["adversarial"]["score"]),
                    "physical_linf": float(
                        r["adversarial"]["physical_linf"]
                    ),
                }
                print(
                    "  COMPLETE | "
                    f"E {r['clean']['E']:.6f}->{r['adversarial']['E']:.6f} | "
                    f"score={r['adversarial']['score']:.6f} | "
                    f"Linf={r['adversarial']['physical_linf']:.8f}"
                )

            statuses.append(entry)

        except Exception as exc:
            failure = {
                "global_order": global_order,
                "shard_order": shard_order,
                "image_path": str(row["image_path"]),
                "status": "FAILED_INCOMPLETE",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds_this_invocation": float(
                    time.perf_counter() - t0
                ),
            }
            statuses.append(failure)
            failures.append(failure)
            print(
                f"  FAILED_INCOMPLETE | {type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.stop_on_error:
                attack.atomic_write_json(
                    status_path,
                    {
                        "status": "INCOMPLETE_FAILURE",
                        "rows": statuses,
                    },
                )
                raise

        finally:
            gc.collect()
            torch.cuda.empty_cache()

        attack.atomic_write_json(
            status_path,
            {
                "status": (
                    "RUNNING_WITH_FAILURES"
                    if failures
                    else "RUNNING"
                ),
                "hostname": socket.gethostname(),
                "shard_id": int(args.shard_id),
                "protocol_sha256": protocol_sha,
                "plan_freeze_sha256": freeze_sha,
                "rows": statuses,
            },
        )
        print()

    n_ok = sum(
        row["status"] in {"PASS", "SKIP_COMPLETE"}
        for row in statuses
    )

    final_status = {
        "status": (
            "PASS_ALL_ASSIGNED"
            if n_ok == len(shard) and not failures
            else "INCOMPLETE_FAILURES"
        ),
        "hostname": socket.gethostname(),
        "shard_id": int(args.shard_id),
        "n_assigned": int(len(shard)),
        "n_complete_or_skip": int(n_ok),
        "n_failures": int(len(failures)),
        "protocol_sha256": protocol_sha,
        "plan_freeze_sha256": freeze_sha,
        "rows": statuses,
    }
    attack.atomic_write_json(status_path, final_status)

    provenance["status"] = final_status["status"]
    provenance["worker_status_sha256"] = sha256_file(status_path)
    attack.atomic_write_json(provenance_path, provenance)

    print("STAGE 26 SHARD FINISHED")
    print(f"shard:         {args.shard_id:02d}")
    print(f"complete/skip: {n_ok}/{len(shard)}")
    print(f"failures:      {len(failures)}")

    if failures:
        raise RuntimeError(
            f"Shard {args.shard_id:02d} has {len(failures)} incomplete image(s). "
            "Rerun identical command; only incomplete images restart."
        )


if __name__ == "__main__":
    main()
