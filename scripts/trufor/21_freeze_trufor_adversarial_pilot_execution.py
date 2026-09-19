#!/usr/bin/env python3
"""
Stage 21: freeze the six-image TruFor adversarial-pilot execution implementation.

This stage performs NO attack inference.

It refuses to freeze unless:
- the Stage-19 scientific protocol is hash-valid;
- the corrected Stage-20 largest-image v2 memory gate is PASS;
- that memory gate used exactly the same Stage-19 protocol SHA;
- the memory gate reproduced the Stage-18 query-chunk regression and
  classification-preserving one-step result.

It then freezes hashes of the exact common module + Stages 21-23 before any
multi-step pilot result is observed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trufor_common import ROOT, sha256_file
from trufor_adversarial_attack_common import (
    ALPHA_PHYSICAL,
    EPSILON_PHYSICAL,
    attack_root,
    atomic_write_json,
    protocol_config_path,
    stage19_provenance_path,
    verify_protocol_freeze,
)

ENGINEERING_REVISION = "v2_attention_chunk_activation_checkpoint"
EXECUTION_FILES = (
    "trufor_adversarial_attack_common.py",
    "21_freeze_trufor_adversarial_pilot_execution.py",
    "22_run_trufor_adversarial_pilot.py",
    "23_validate_trufor_adversarial_pilot.py",
)

EXPECTED_CHUNK_REGRESSION = {
    "attention_modules_observed": 32,
    "total_query_chunks": 148,
    "split_attention_modules": 32,
    "max_chunks_one_attention": 10,
}

MEMORY_GATE_STAGE18_REGRESSION_ATOL = 2.0e-3
ALPHA_LINF_ATOL = 2.0e-7


def memory_gate_path(run_tag: str) -> Path:
    return (
        attack_root(run_tag)
        / "memory_gate"
        / "stage20_memory_gate_result.json"
    )


def execution_freeze_path(run_tag: str) -> Path:
    return (
        attack_root(run_tag)
        / "pilot_execution_freeze.json"
    )


def execution_file_hashes() -> dict:
    here = Path(__file__).resolve().parent
    hashes = {}
    for name in EXECUTION_FILES:
        path = here / name
        if not path.is_file():
            raise RuntimeError(f"Missing pilot execution file: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def validate_passed_memory_gate(run_tag: str, protocol_sha: str) -> tuple[dict, str]:
    path = memory_gate_path(run_tag)
    if not path.is_file():
        raise RuntimeError(
            f"Missing corrected Stage-20 memory gate result:\n{path}\n"
            "Run the v2 one-step memory gate first."
        )

    payload = json.loads(path.read_text())
    actual_sha = sha256_file(path)

    if payload.get("status") != "PASS":
        raise RuntimeError("Corrected Stage-20 memory gate is not PASS")
    if payload.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(
            "Memory gate protocol SHA differs from frozen Stage-19 protocol"
        )
    if payload.get("scientific_protocol_changed") is not False:
        raise RuntimeError("Memory gate unexpectedly reports scientific protocol change")
    if payload.get("engineering_revision") != ENGINEERING_REVISION:
        raise RuntimeError(
            "Memory gate did not use the validated v2 attention checkpoint route"
        )

    gradient = payload.get("gradient", {})
    if gradient.get("chunk_actual") != EXPECTED_CHUNK_REGRESSION:
        raise RuntimeError(
            "Memory-gate query chunk regression no longer matches Stage 18:\n"
            f"actual={gradient.get('chunk_actual')}\n"
            f"expected={EXPECTED_CHUNK_REGRESSION}"
        )

    one = payload.get("one_projected_step", {})
    if one.get("classification_preserved") is not True:
        raise RuntimeError("Memory-gate projected step did not preserve classification")

    linf = float(one.get("physical_linf", float("nan")))
    if abs(linf - ALPHA_PHYSICAL) > ALPHA_LINF_ATOL:
        raise RuntimeError(
            f"Memory-gate one-step L_inf differs from alpha: {linf} vs {ALPHA_PHYSICAL}"
        )

    if float(one.get("stage18_E_abs_error", float("inf"))) > MEMORY_GATE_STAGE18_REGRESSION_ATOL:
        raise RuntimeError("Memory-gate Stage-18 E regression tolerance failed")
    if float(one.get("stage18_score_abs_error", float("inf"))) > MEMORY_GATE_STAGE18_REGRESSION_ATOL:
        raise RuntimeError("Memory-gate Stage-18 score regression tolerance failed")

    return payload, actual_sha


def validate_execution_freeze(run_tag: str) -> tuple[dict, str]:
    cfg, protocol_sha, selection, stage19 = verify_protocol_freeze(run_tag)
    freeze_path = execution_freeze_path(run_tag)
    if not freeze_path.is_file():
        raise RuntimeError(
            f"Missing Stage-21 pilot execution freeze:\n{freeze_path}\n"
            "Run Stage 21 first."
        )

    freeze = json.loads(freeze_path.read_text())
    freeze_sha = sha256_file(freeze_path)

    if freeze.get("status") != "FROZEN":
        raise RuntimeError("Pilot execution freeze is not FROZEN")
    if freeze.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Pilot execution freeze protocol SHA mismatch")

    actual_hashes = execution_file_hashes()
    if freeze.get("execution_file_sha256") != actual_hashes:
        raise RuntimeError(
            "Pilot execution files changed after Stage 21 freeze.\n"
            "Do not continue under the old implementation freeze."
        )

    _, memory_gate_sha = validate_passed_memory_gate(run_tag, protocol_sha)
    if freeze.get("memory_gate_result_sha256") != memory_gate_sha:
        raise RuntimeError("Passed Stage-20 memory-gate result changed after Stage 21")

    return freeze, freeze_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="LABPC")
    args = parser.parse_args()

    cfg, protocol_sha, selection, stage19 = verify_protocol_freeze(args.run_tag)
    memory_gate, memory_gate_sha = validate_passed_memory_gate(
        args.run_tag,
        protocol_sha,
    )

    hashes = execution_file_hashes()

    payload = {
        "status": "FROZEN",
        "stage": "21_freeze_trufor_adversarial_pilot_execution",
        "run_tag": args.run_tag,
        "protocol_sha256": protocol_sha,
        "scientific_protocol_changed": False,
        "engineering_revision": ENGINEERING_REVISION,
        "n_pilot_images": 6,
        "attack_steps_per_image": int(cfg["optimisation"]["steps"]),
        "epsilon_physical_rgb01": float(
            cfg["optimisation"]["epsilon_physical_rgb01"]
        ),
        "alpha_physical_rgb01": float(
            cfg["optimisation"]["alpha_physical_rgb01"]
        ),
        "classification_threshold": float(
            cfg["classification_constraint"]["threshold"]
        ),
        "bisection_steps": int(
            cfg["classification_constraint"]["bisection_steps"]
        ),
        "objective": cfg["objective"]["primary"],
        "memory_gate_result": str(memory_gate_path(args.run_tag).relative_to(ROOT)),
        "memory_gate_result_sha256": memory_gate_sha,
        "memory_gate_chunk_regression": EXPECTED_CHUNK_REGRESSION,
        "execution_file_sha256": hashes,
        "stage19_provenance": str(
            stage19_provenance_path(args.run_tag).relative_to(ROOT)
        ),
        "stage19_provenance_sha256": sha256_file(
            stage19_provenance_path(args.run_tag)
        ),
        "protocol_config": str(
            protocol_config_path(args.run_tag).relative_to(ROOT)
        ),
        "protocol_config_sha256": protocol_sha,
        "pilot_selection_sha256": stage19["pilot_selection_sha256"],
        "resume_policy": (
            "one image atomic; hash-valid COMPLETE images skipped; "
            "incomplete image restarted from clean"
        ),
        "scientific_note": (
            "The six-image multi-step execution implementation is frozen before "
            "observing any corrected-route multi-step pilot result."
        ),
    }

    out_path = execution_freeze_path(args.run_tag)
    atomic_write_json(out_path, payload)

    print("TRUFOR STAGE 21 — SIX-IMAGE PILOT EXECUTION FREEZE")
    print(f"run tag:              {args.run_tag}")
    print(f"scientific protocol:  {protocol_sha}")
    print(f"memory gate SHA256:   {memory_gate_sha}")
    print(f"execution freeze:     {out_path}")
    print(f"execution freeze SHA: {sha256_file(out_path)}")
    print()
    print("memory gate prerequisite: PASS")
    print("execution file hashes frozen: PASS")
    print("scientific protocol changed: NO")
    print()
    print("STAGE 21 PASS")
    print("Next: Stage 22 six-image x 10-step pilot.")


if __name__ == "__main__":
    main()
