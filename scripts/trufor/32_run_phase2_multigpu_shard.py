#!/usr/bin/env python3
"""Stage 32: run one frozen Phase-2 (10-worker) TruFor shard.

This is a port of the validated Stage 26 execution path onto the Phase-2
plan frozen by Stage 31. The per-image work is identical
(attack.run_one_pilot_image); only the plan location, output location and
plan validation differ:

  plan:     <attack_root>/phase2_multigpu_sharding/phase2_plan_freeze.json
  shards:   <attack_root>/phase2_multigpu_sharding/shards/shard_XX.csv
  outputs:  <attack_root>/phase2_multigpu_outputs/shard_XX/

Plan validation. Stage 31's freeze carries no manifest_sha256 /
attack_common_sha256, so the chain is verified through the committed
Stage 25 freeze instead:

  phase2 freeze.old_plan_sha256  ==  sha256(Stage 25 freeze)
  Stage 25 freeze                 passes mg.validate_plan (protocol SHA,
                                  attack_common SHA, manifest SHAs)
  phase2 freeze.protocol_sha256  ==  Stage 25 freeze.protocol_sha256
  phase2 freeze.scientific_protocol_changed is False

and the shard CSVs are integrity-checked as a set (all present, row
total == freeze.remaining, no duplicate image_path, own shard_id column
consistent). Every worker records the SHA-256 of the freeze, its own
shard CSV, all shard CSVs and attack_common in worker_provenance.json,
so the collection machine can verify that all ten hosts ran against
identical inputs.

Deployment model: one host per shard, single GPU, outputs collected by
hand onto one machine afterwards. Rerunning the identical command after a
crash restarts only images without a hash-valid COMPLETE.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import trufor_adversarial_attack_common as attack
import trufor_multigpu_common as mg
from trufor_common import ROOT, sha256_file


STAGE = "32_run_phase2_multigpu_shard"
THRESHOLD_ATOL = 1e-12


# --------------------------------------------------------------------------
# Phase-2 paths
# --------------------------------------------------------------------------


def phase2_root(run_tag: str) -> Path:
    return attack.attack_root(run_tag) / "phase2_multigpu_sharding"


def phase2_outputs_root(run_tag: str) -> Path:
    return attack.attack_root(run_tag) / "phase2_multigpu_outputs"


def phase2_freeze_path(run_tag: str) -> Path:
    return phase2_root(run_tag) / "phase2_plan_freeze.json"


def phase2_shard_csv_path(run_tag: str, shard_id: int) -> Path:
    return phase2_root(run_tag) / "shards" / f"shard_{shard_id:02d}.csv"


def phase2_shard_root(run_tag: str, shard_id: int) -> Path:
    return phase2_outputs_root(run_tag) / f"shard_{shard_id:02d}"


# --------------------------------------------------------------------------
# Plan validation
# --------------------------------------------------------------------------


def validate_phase2_plan(run_tag: str) -> Tuple[dict, str, dict, str]:
    """Return (phase2_freeze, phase2_freeze_sha, stage25_freeze, stage25_sha)."""
    freeze_path = phase2_freeze_path(run_tag)
    if not freeze_path.is_file():
        raise RuntimeError(
            f"Missing frozen Phase-2 plan:\n{freeze_path}\nRun Stage 31 first."
        )
    freeze = json.loads(freeze_path.read_text())
    freeze_sha = sha256_file(freeze_path)

    if freeze.get("status") != "FROZEN":
        raise RuntimeError("Phase-2 plan is not FROZEN")
    if freeze.get("stage") != "31_freeze_phase2_10worker_shards":
        raise RuntimeError(
            f"Unexpected Phase-2 freeze stage: {freeze.get('stage')}"
        )
    if freeze.get("scientific_protocol_changed") is not False:
        raise RuntimeError(
            "Phase-2 freeze does not assert scientific_protocol_changed=False"
        )
    if freeze.get("protocol_sha256") != mg.EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Phase-2 freeze protocol SHA does not match the frozen protocol"
        )

    # Chain through the committed Stage 25 plan: this validates the protocol
    # snapshot, attack_common SHA and Stage 25 manifest exactly as Stage 26
    # would, and pins the Phase-2 freeze to that specific Stage 25 artifact.
    stage25_freeze, stage25_sha = mg.validate_plan(run_tag)
    if freeze.get("old_plan_sha256") != stage25_sha:
        raise RuntimeError(
            "Phase-2 freeze old_plan_sha256 does not match the committed "
            f"Stage 25 freeze:\n  freeze:  {freeze.get('old_plan_sha256')}\n"
            f"  on disk: {stage25_sha}"
        )
    if stage25_freeze.get("protocol_sha256") != freeze.get("protocol_sha256"):
        raise RuntimeError("Stage 25 / Phase-2 protocol SHA mismatch")

    n_shards = int(freeze["n_shards"])
    if n_shards < 1:
        raise RuntimeError(f"Invalid n_shards in Phase-2 freeze: {n_shards}")

    return freeze, freeze_sha, stage25_freeze, stage25_sha


def validate_shard_set(
    run_tag: str,
    freeze: dict,
) -> Dict[str, str]:
    """Integrity-check all shard CSVs as a set; return {rel_path: sha256}."""
    n_shards = int(freeze["n_shards"])
    expected_total = int(freeze["remaining"])

    manifest: Dict[str, str] = {}
    frames: List[pd.DataFrame] = []
    for shard_id in range(n_shards):
        p = phase2_shard_csv_path(run_tag, shard_id)
        if not p.is_file():
            raise RuntimeError(f"Missing Phase-2 shard CSV: {p}")
        manifest[f"shards/shard_{shard_id:02d}.csv"] = sha256_file(p)
        f = pd.read_csv(p, keep_default_na=False)
        if "shard_id" not in f.columns or "image_path" not in f.columns:
            raise RuntimeError(f"Shard CSV missing required columns: {p}")
        bad = f.loc[f["shard_id"].astype(int) != shard_id]
        if len(bad):
            raise RuntimeError(
                f"shard_{shard_id:02d}.csv contains rows with shard_id != "
                f"{shard_id} ({len(bad)} rows)"
            )
        frames.append(f)

    union = pd.concat(frames, ignore_index=True)
    if len(union) != expected_total:
        raise RuntimeError(
            f"Shard CSV rows ({len(union)}) != freeze.remaining "
            f"({expected_total})"
        )
    dup = union["image_path"].astype(str).duplicated()
    if dup.any():
        raise RuntimeError(
            "Duplicate image_path across Phase-2 shards, e.g. "
            f"{union.loc[dup, 'image_path'].head(5).tolist()}"
        )
    return manifest


# --------------------------------------------------------------------------
# Output routing (identical shape to Stage 26)
# --------------------------------------------------------------------------


def patch_shard_output_dir(run_tag: str, shard_id: int) -> Path:
    root = phase2_shard_root(run_tag, shard_id)

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


def foreign_complete_image_paths(
    run_tag: str,
    protocol_sha: str,
    own_shard_id: int,
) -> Dict[str, str]:
    """COMPLETE images visible on this host outside this shard's own root.

    Covers the unsharded Stage 24 root, the retired 3-shard roots and any
    *other* Phase-2 shard root present on this host (relevant on the
    collection machine). The own shard root is deliberately not scanned:
    resume of own images is handled per-image by run_one_pilot_image.
    """
    roots: List[Tuple[str, Path]] = [
        ("legacy_stage24", mg.legacy_images_root(run_tag))
    ]
    shard_base = mg.shard_outputs_root(run_tag)
    if shard_base.is_dir():
        for p in sorted(shard_base.glob("shard_*/images")):
            roots.append((f"stage26_{p.parent.name}", p))
    p2_base = phase2_outputs_root(run_tag)
    if p2_base.is_dir():
        own = phase2_shard_root(run_tag, own_shard_id)
        for p in sorted(p2_base.glob("shard_*/images")):
            if p.parent.resolve() == own.resolve():
                continue
            roots.append((f"phase2_{p.parent.name}", p))

    records = mg.scan_complete_records(roots, protocol_sha, deep_npz=False)
    result: Dict[str, str] = {}
    for rec in records:
        image_path = rec["image_path"]
        if image_path in result:
            raise RuntimeError(
                f"Duplicate COMPLETE image already exists: {image_path}\n"
                f"{result[image_path]}\n{rec['image_dir']}"
            )
        result[image_path] = rec["image_dir"]
    return result


GPU_WORKER_SCRIPTS = (
    "24_run_trufor_adversarial_full.py",
    "26_run_trufor_multigpu_shard.py",
    "32_run_phase2_multigpu_shard.py",
)


def active_phase2_workers_on_this_host() -> List[str]:
    """Other TruFor attack workers on this host (one host runs one shard).

    Matches retired Stage 24/26 workers as well as Stage 32: any of them
    left running holds GPU memory that this shard needs.
    """
    try:
        p = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        lines = p.stdout.splitlines()
    except Exception:
        return []
    me = str(os.getpid())
    out = []
    for line in lines:
        s = line.strip()
        if not any(name in s for name in GPU_WORKER_SCRIPTS):
            continue
        if s.split(None, 1)[0] == me:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Preflight (verbatim from Stage 26)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--allow-concurrent",
        action="store_true",
        help="Permit another Stage 32 worker on this host (not the default: "
        "one host runs one shard).",
    )
    args = parser.parse_args()

    # ---- plan + shard set --------------------------------------------------
    freeze, freeze_sha, stage25_freeze, stage25_sha = validate_phase2_plan(
        args.run_tag
    )
    n_shards = int(freeze["n_shards"])
    n_total = int(freeze["n_total"])

    if not (0 <= args.shard_id < n_shards):
        raise RuntimeError(
            f"shard-id must be 0..{n_shards - 1}, got {args.shard_id}"
        )

    shard_manifest = validate_shard_set(args.run_tag, freeze)
    shard_csv = phase2_shard_csv_path(args.run_tag, args.shard_id)
    shard = pd.read_csv(shard_csv, keep_default_na=False)

    others = active_phase2_workers_on_this_host()
    if others and not args.allow_concurrent:
        raise RuntimeError(
            "Another Stage 32 worker is already running on this host "
            "(one host runs one shard; pass --allow-concurrent to override):\n"
            + "\n".join(others)
        )

    # ---- threshold: Stage 25 protocol snapshot, cross-checked --------------
    cfg_path = (
        mg.multigpu_root(args.run_tag) / "frozen_attack_protocol_config.json"
    )
    cfg = json.loads(cfg_path.read_text())
    protocol_sha = str(freeze["protocol_sha256"])
    threshold = float(cfg["classification_constraint"]["threshold"])

    _, calib_threshold = attack.load_frozen_threshold()
    if abs(threshold - float(calib_threshold)) > THRESHOLD_ATOL:
        raise RuntimeError(
            "Threshold mismatch between frozen protocol snapshot and "
            f"calibration source: protocol={threshold!r} "
            f"calibration={calib_threshold!r}"
        )

    # ---- backend / device / outputs ----------------------------------------
    attack.validate_backend_record()
    device = attack.resolve_cuda_device(args.gpu)
    torch.cuda.set_device(device)

    output_root = patch_shard_output_dir(args.run_tag, args.shard_id)
    output_root.mkdir(parents=True, exist_ok=True)

    status_path = output_root / "worker_status.json"
    provenance_path = output_root / "worker_provenance.json"

    print("TRUFOR STAGE 32 — PHASE-2 MULTI-GPU SHARD WORKER")
    print(f"hostname:             {socket.gethostname()}")
    print(f"shard:                {args.shard_id:02d} of {n_shards}")
    print(f"assigned:             {len(shard)}")
    print(f"GPU:                  {torch.cuda.get_device_name(device)}")
    print(f"protocol SHA:         {protocol_sha}")
    print(f"phase2 freeze SHA:    {freeze_sha}")
    print(f"stage25 freeze SHA:   {stage25_sha}")
    print(f"shard manifest SHA:   {sha256_file(shard_csv)}")
    print(f"threshold:            {threshold!r}")
    print(f"output root:          {output_root}")
    print()

    # ---- foreign-completion guard ------------------------------------------
    foreign_visible = foreign_complete_image_paths(
        args.run_tag,
        protocol_sha,
        args.shard_id,
    )
    own_paths = set(shard["image_path"].astype(str))
    foreign = {
        p: d for p, d in foreign_visible.items() if p in own_paths
    }
    if foreign:
        raise RuntimeError(
            "Assigned images already COMPLETE outside own shard "
            "(3-shard leftover or duplicate shard on this host). "
            f"Examples: {list(foreign.items())[:10]}"
        )

    # ---- models + validated memory route -----------------------------------
    t_load = time.perf_counter()
    attack_model, reference_model, load_meta = (
        attack.load_two_models_cpu_first(device)
    )
    route_meta = attack.patch_memory_efficient_attack_model(attack_model)
    load_seconds = time.perf_counter() - t_load

    if route_meta.get("attention_chunk_activation_checkpointing") is not True:
        raise RuntimeError("Validated v2 route inactive")

    route_meta = {
        **route_meta,
        "phase2": True,
        "shard_id": int(args.shard_id),
        "stage": STAGE,
    }

    # ---- preflight on largest assigned image -------------------------------
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

    # ---- provenance --------------------------------------------------------
    provenance = {
        "status": "STARTED",
        "stage": STAGE,
        "hostname": socket.gethostname(),
        "shard_id": int(args.shard_id),
        "n_shards": n_shards,
        "protocol_sha256": protocol_sha,
        "phase2_plan_freeze_sha256": freeze_sha,
        "phase2_old_plan_sha256": str(freeze.get("old_plan_sha256")),
        "stage25_plan_freeze_sha256": stage25_sha,
        "shard_manifest_sha256": sha256_file(shard_csv),
        "phase2_shard_set_sha256": shard_manifest,
        "frozen_attack_protocol_config_sha256": sha256_file(cfg_path),
        "attack_common_sha256": sha256_file(
            Path(attack.__file__).resolve()
        ),
        "threshold": threshold,
        "environment": attack.environment_record(device),
        "model_load": load_meta,
        "memory_route": route_meta,
        "preflight": preflight,
        "model_load_seconds": float(load_seconds),
    }
    attack.atomic_write_json(provenance_path, provenance)

    # ---- per-image loop (identical to Stage 26) ----------------------------
    statuses = []
    failures = []

    for _, row in shard.iterrows():
        global_order = int(row["global_order"])
        shard_order = int(row["shard_order"])

        print(
            f"[shard {args.shard_id:02d} {shard_order}/{len(shard)} | "
            f"global {global_order}/{n_total}] "
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
            try:
                torch.cuda.empty_cache()
            except Exception as cache_exc:
                # An async CUDA error (e.g. cudaErrorMemoryAllocation) can be
                # re-reported here. Only swallow it when this image is already
                # recorded as FAILED_INCOMPLETE: then the status write below
                # still happens and the next image gets its chance. In any
                # other state the device is unknown — fail loudly.
                already_failed = bool(
                    failures and failures[-1]["global_order"] == global_order
                )
                print(
                    f"  empty_cache raised {type(cache_exc).__name__}: "
                    f"{cache_exc}",
                    flush=True,
                )
                if not already_failed:
                    raise
                failures[-1]["empty_cache_exception"] = (
                    f"{type(cache_exc).__name__}: {cache_exc}"
                )

        attack.atomic_write_json(
            status_path,
            {
                "status": (
                    "RUNNING_WITH_FAILURES" if failures else "RUNNING"
                ),
                "hostname": socket.gethostname(),
                "shard_id": int(args.shard_id),
                "protocol_sha256": protocol_sha,
                "phase2_plan_freeze_sha256": freeze_sha,
                "rows": statuses,
            },
        )
        print()

    # ---- epilogue ----------------------------------------------------------
    n_ok = sum(
        row["status"] in {"PASS", "SKIP_COMPLETE"} for row in statuses
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
        "phase2_plan_freeze_sha256": freeze_sha,
        "rows": statuses,
    }
    attack.atomic_write_json(status_path, final_status)

    provenance["status"] = final_status["status"]
    provenance["worker_status_sha256"] = sha256_file(status_path)
    attack.atomic_write_json(provenance_path, provenance)

    print("STAGE 32 SHARD FINISHED")
    print(f"shard:         {args.shard_id:02d}")
    print(f"complete/skip: {n_ok}/{len(shard)}")
    print(f"failures:      {len(failures)}")

    if failures:
        raise RuntimeError(
            f"Shard {args.shard_id:02d} has {len(failures)} incomplete "
            "image(s). Rerun identical command; only incomplete images "
            "restart."
        )


if __name__ == "__main__":
    main()
