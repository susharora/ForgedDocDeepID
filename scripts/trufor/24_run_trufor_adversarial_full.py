#!/usr/bin/env python3
"""Stage 24: resumable full 1,330-image TruFor adversarial attack."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import trufor_adversarial_attack_common as attack
from trufor_attack_pilot_common import (
    CLEAN_CORRECT_ATTACKS,
    LOCALISATION_PER_IMAGE,
    load_frozen_threshold,
    load_regions,
    validate_stage4_population,
)
from trufor_common import ROOT, project_git_head, sha256_file

EXPECTED_N = 1330

_STAGE21_PATH = (
    Path(__file__).resolve().parent
    / "21_freeze_trufor_adversarial_pilot_execution.py"
)
_STAGE21_SPEC = importlib.util.spec_from_file_location(
    "stage21", _STAGE21_PATH
)
if _STAGE21_SPEC is None or _STAGE21_SPEC.loader is None:
    raise RuntimeError(f"Cannot import Stage 21 helper: {_STAGE21_PATH}")

_STAGE21 = importlib.util.module_from_spec(_STAGE21_SPEC)
_STAGE21_SPEC.loader.exec_module(_STAGE21)
validate_execution_freeze = _STAGE21.validate_execution_freeze


def root_for(tag: str) -> Path:
    return attack.attack_root(tag) / "full_population"


def selection_file(tag: str) -> Path:
    return root_for(tag) / "full_population_selection.csv"


def freeze_file(tag: str) -> Path:
    return root_for(tag) / "full_population_selection_freeze.json"


def status_file(tag: str) -> Path:
    return root_for(tag) / "stage24_run_status.json"


def build_selection():
    clean, loc, stage4 = validate_stage4_population()

    loc = loc[
        ["image_path", "A_union", "E_union", "mu_union", "PG_union"]
    ]

    frame = clean.merge(
        loc,
        on="image_path",
        how="left",
        validate="one_to_one",
    )

    if len(frame) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} clean-correct attacks; "
            f"got {len(frame)}"
        )

    if frame[
        ["A_union", "E_union", "mu_union", "PG_union"]
    ].isna().any().any():
        raise RuntimeError("Missing frozen localisation values")

    if frame["image_path"].duplicated().any():
        raise RuntimeError("Duplicate full-population image_path")

    frame = frame.reset_index(drop=True)

    # run_one_pilot_image() uses these generic names.
    frame["pilot_order"] = np.arange(
        1, EXPECTED_N + 1, dtype=np.int64
    )
    frame["pilot_role"] = "full_population"
    frame["clean_score_stage4"] = frame["trufor_score"].astype(float)
    frame["A_union_stage4"] = frame["A_union"].astype(float)
    frame["E_union_stage4"] = frame["E_union"].astype(float)
    frame["mu_union_stage4"] = frame["mu_union"].astype(float)
    frame["PG_union_stage4"] = frame["PG_union"].astype(float)

    return frame, stage4


def freeze_or_validate_selection(
    tag: str,
    protocol_sha: str,
    execution_sha: str,
):
    out = root_for(tag)
    out.mkdir(parents=True, exist_ok=True)

    sel = selection_file(tag)
    frz = freeze_file(tag)
    runner_sha = sha256_file(Path(__file__))

    if sel.exists() or frz.exists():
        if not (sel.exists() and frz.exists()):
            raise RuntimeError("Incomplete full-selection freeze")

        meta = json.loads(frz.read_text())
        selection_sha = sha256_file(sel)

        checks = {
            "status": meta.get("status") == "FROZEN",
            "n": meta.get("n_images") == EXPECTED_N,
            "protocol":
                meta.get("protocol_sha256") == protocol_sha,
            "execution":
                meta.get("execution_freeze_sha256") == execution_sha,
            "selection":
                meta.get("selection_sha256") == selection_sha,
            "runner":
                meta.get("runner_sha256") == runner_sha,
            "clean_source":
                meta.get("clean_correct_attacks_sha256")
                == sha256_file(CLEAN_CORRECT_ATTACKS),
            "loc_source":
                meta.get("localisation_per_image_sha256")
                == sha256_file(LOCALISATION_PER_IMAGE),
        }

        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise RuntimeError(
                f"Frozen full selection validation failed: {bad}"
            )

        frame = pd.read_csv(sel, keep_default_na=False)

        if len(frame) != EXPECTED_N:
            raise RuntimeError(
                "Frozen full selection row count mismatch"
            )

        return frame, selection_sha

    frame, stage4 = build_selection()

    attack.atomic_write_csv(sel, frame)
    selection_sha = sha256_file(sel)

    attack.atomic_write_json(
        frz,
        {
            "status": "FROZEN",
            "stage": "24_run_trufor_adversarial_full",
            "n_images": EXPECTED_N,
            "protocol_sha256": protocol_sha,
            "execution_freeze_sha256": execution_sha,
            "selection": str(sel.relative_to(ROOT)),
            "selection_sha256": selection_sha,
            "clean_correct_attacks_sha256":
                sha256_file(CLEAN_CORRECT_ATTACKS),
            "localisation_per_image_sha256":
                sha256_file(LOCALISATION_PER_IMAGE),
            "stage4_status": stage4.get("status"),
            "runner_sha256": runner_sha,
            "project_git_head_at_freeze": project_git_head(),
        },
    )

    return frame, selection_sha


def pilot_gate(
    tag: str,
    protocol_sha: str,
    execution_sha: str,
    override: bool,
):
    path = (
        attack.attack_root(tag)
        / "pilot_validated_protocol_freeze.json"
    )

    if path.exists():
        meta = json.loads(path.read_text())

        if (
            meta.get("status")
            != "PILOT_VALIDATED_TECHNICALLY"
            or meta.get("protocol_sha256") != protocol_sha
            or meta.get("execution_freeze_sha256")
            != execution_sha
        ):
            raise RuntimeError(
                "Stage-23 validation does not match frozen protocol"
            )

        return {
            "mode": "STAGE23_VALIDATED",
            "sha256": sha256_file(path),
        }

    if not override:
        raise RuntimeError(
            "Stage-23 is absent. Use "
            "--executive-override-no-stage23 only for an explicit "
            "decision to launch now."
        )

    markers = list(
        (attack.attack_root(tag) / "pilot")
        .rglob("COMPLETE.json")
    )

    return {
        "mode": "EXECUTIVE_OVERRIDE_NO_STAGE23",
        "pilot_complete_markers_observed": len(markers),
    }


def acquire_lock(tag: str):
    path = root_for(tag) / ".stage24.lock"
    path.parent.mkdir(parents=True, exist_ok=True)

    fh = path.open("a+")

    try:
        fcntl.flock(
            fh.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as exc:
        fh.seek(0)
        raise RuntimeError(
            "Another Stage-24 runner appears active: "
            + fh.read().strip()
        ) from exc

    fh.seek(0)
    fh.truncate()
    fh.write(
        f"pid={os.getpid()} host={os.uname().nodename}\n"
    )
    fh.flush()

    return fh


def write_status(
    tag,
    state,
    protocol_sha,
    execution_sha,
    selection_sha,
    complete,
    failures,
    order,
    image,
    started,
    recent,
    gate,
):
    attack.atomic_write_json(
        status_file(tag),
        {
            "status": state,
            "stage": "24_run_trufor_adversarial_full",
            "run_tag": tag,
            "protocol_sha256": protocol_sha,
            "execution_freeze_sha256": execution_sha,
            "selection_sha256": selection_sha,
            "n_expected": EXPECTED_N,
            "n_complete_or_skip_this_scan": int(complete),
            "n_failures_this_invocation": int(failures),
            "current_order": order,
            "current_image_path": image,
            "elapsed_hours_this_invocation":
                (time.time() - started) / 3600.0,
            "pilot_gate": gate,
            "recent_rows": recent[-20:],
        },
    )


def show_status(tag: str):
    out = root_for(tag)
    images = out / "images"

    n_complete = (
        len(list(images.glob("*/COMPLETE.json")))
        if images.is_dir()
        else 0
    )

    n_failed = (
        len(list(images.glob("*/failure.json")))
        if images.is_dir()
        else 0
    )

    print(f"COMPLETE markers: {n_complete}/{EXPECTED_N}")
    print(f"failure files:    {n_failed}")

    if status_file(tag).exists():
        print(status_file(tag).read_text())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-tag", default="LABPC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--executive-override-no-stage23",
        action="store_true",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.status_only:
        show_status(args.run_tag)
        return

    # Held for process lifetime; prevents accidental double launch.
    lock_handle = acquire_lock(args.run_tag)
    _ = lock_handle

    attack.validate_backend_record()

    _, protocol_sha, _, _ = (
        attack.verify_protocol_freeze(args.run_tag)
    )

    _, execution_sha = validate_execution_freeze(
        args.run_tag
    )

    _, threshold = load_frozen_threshold()

    gate = pilot_gate(
        args.run_tag,
        protocol_sha,
        execution_sha,
        args.executive_override_no_stage23,
    )

    rows, selection_sha = freeze_or_validate_selection(
        args.run_tag,
        protocol_sha,
        execution_sha,
    )

    rows = (
        rows.sort_values("pilot_order")
        .reset_index(drop=True)
    )

    # Reuse the frozen, validated atomic image runner, but redirect
    # outputs from pilot/ into full_population/.
    attack.pilot_root = root_for

    # Immutable workbook: read once instead of 1,330 times.
    regions = load_regions()
    attack.load_regions = lambda: regions

    device = attack.resolve_cuda_device(args.gpu)
    torch.cuda.set_device(device)

    print(
        "TRUFOR STAGE 24 — "
        "FULL 1330-IMAGE ADVERSARIAL ATTACK"
    )
    print(f"run tag:              {args.run_tag}")
    print(f"device:               {device}")
    print(
        f"GPU:                  "
        f"{torch.cuda.get_device_name(device)}"
    )
    print(f"protocol SHA:         {protocol_sha}")
    print(f"execution freeze SHA: {execution_sha}")
    print(f"selection SHA:        {selection_sha}")
    print(f"threshold:            {threshold:.15f}")
    print(f"pilot gate:           {gate['mode']}")
    print(
        "checkpointing:        "
        "per-image atomic + artifact hashes"
    )
    print(
        "resume:               "
        "hash-valid COMPLETE images skipped"
    )
    print()

    t0 = time.perf_counter()

    attack_model, reference_model, load_meta = (
        attack.load_two_models_cpu_first(device)
    )

    route = attack.patch_memory_efficient_attack_model(
        attack_model
    )

    if (
        route.get(
            "attention_chunk_activation_checkpointing"
        ) is not True
        or route.get("attention_modules") != 32
        or route.get("mlp_modules") != 32
    ):
        raise RuntimeError(
            f"Validated memory route mismatch: {route}"
        )

    print("CPU-first dual-model load PASS")
    print(json.dumps(load_meta, indent=2, sort_keys=True))
    print(
        "model load/patch wall time: "
        f"{time.perf_counter() - t0:.1f}s"
    )
    print()

    started = time.time()
    complete = 0
    failures = []
    recent = []

    for _, row in rows.iterrows():
        order = int(row["pilot_order"])
        image = str(row["image_path"])

        print(
            f"[{order}/{EXPECTED_N}] "
            f"{row['variant']} | "
            f"{int(row['native_width'])}x"
            f"{int(row['native_height'])} | "
            f"clean score="
            f"{float(row['clean_score_stage4']):.6f} | "
            f"{image}",
            flush=True,
        )

        write_status(
            args.run_tag,
            "RUNNING",
            protocol_sha,
            execution_sha,
            selection_sha,
            complete,
            len(failures),
            order,
            image,
            started,
            recent,
            gate,
        )

        image_t0 = time.perf_counter()

        try:
            state = attack.run_one_pilot_image(
                run_tag=args.run_tag,
                selection_row=row,
                protocol_sha=protocol_sha,
                attack_model=attack_model,
                reference_model=reference_model,
                route_metadata=route,
                threshold=threshold,
                device=device,
            )

            complete += 1

            if state["status"] == "SKIP_COMPLETE":
                item = {
                    "full_order": order,
                    "image_path": image,
                    "status": "SKIP_COMPLETE",
                    "wall_seconds_this_invocation":
                        time.perf_counter() - image_t0,
                }

                print(
                    "  hash-valid COMPLETE already exists "
                    "— skipped.",
                    flush=True,
                )

            else:
                r = state["result"]

                item = {
                    "full_order": order,
                    "image_path": image,
                    "status": "PASS",
                    "wall_seconds_this_invocation":
                        time.perf_counter() - image_t0,
                    "clean_E": float(r["clean"]["E"]),
                    "adv_E": float(r["adversarial"]["E"]),
                    "relative_E_degradation":
                        float(
                            r["degradation"]["relative_E"]
                        ),
                    "adv_score":
                        float(r["adversarial"]["score"]),
                    "physical_linf":
                        float(
                            r["adversarial"][
                                "physical_linf"
                            ]
                        ),
                }

                print(
                    "  COMPLETE | "
                    f"E {r['clean']['E']:.6f} -> "
                    f"{r['adversarial']['E']:.6f} | "
                    f"relΔE="
                    f"{r['degradation']['relative_E']:.4f} | "
                    f"score="
                    f"{r['adversarial']['score']:.6f} | "
                    f"Linf="
                    f"{r['adversarial']['physical_linf']:.8f}",
                    flush=True,
                )

            recent.append(item)

        except Exception as exc:
            failure = {
                "full_order": order,
                "image_path": image,
                "status": "FAILED_INCOMPLETE",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds_this_invocation":
                    time.perf_counter() - image_t0,
            }

            failures.append(failure)
            recent.append(failure)

            print(
                "  FAILED_INCOMPLETE | "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            if args.stop_on_error:
                write_status(
                    args.run_tag,
                    "INCOMPLETE_FAILURE",
                    protocol_sha,
                    execution_sha,
                    selection_sha,
                    complete,
                    len(failures),
                    order,
                    image,
                    started,
                    recent,
                    gate,
                )
                raise

        finally:
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        write_status(
            args.run_tag,
            (
                "RUNNING_WITH_FAILURES"
                if failures
                else "RUNNING"
            ),
            protocol_sha,
            execution_sha,
            selection_sha,
            complete,
            len(failures),
            order,
            image,
            started,
            recent,
            gate,
        )

        print()

    final = (
        "PASS_ALL_1330"
        if not failures and complete == EXPECTED_N
        else "INCOMPLETE_FAILURES"
    )

    write_status(
        args.run_tag,
        final,
        protocol_sha,
        execution_sha,
        selection_sha,
        complete,
        len(failures),
        None,
        None,
        started,
        recent,
        gate,
    )

    print("STAGE 24 FINISHED")
    print(
        f"complete/skip: {complete}/{EXPECTED_N}"
    )
    print(f"failures:      {len(failures)}")
    print(
        f"status file:   {status_file(args.run_tag)}"
    )

    if failures:
        raise RuntimeError(
            "Stage 24 has incomplete images. "
            "Re-run the identical command; hash-valid "
            "COMPLETE images will be skipped."
        )

    print(
        "STAGE 24 PASS — ALL 1330 IMAGES COMPLETE"
    )


if __name__ == "__main__":
    main()
