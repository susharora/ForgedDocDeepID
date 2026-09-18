#!/usr/bin/env python3
"""Stage 4 launcher for the accuracy-calibrated TruFor protocol.

This intentionally reuses the already-committed 04_eval_frozen_threshold.py
implementation, but redirects it to the replacement Stage-3 accuracy threshold
and to a separate Stage-4 output directory. The old balanced-accuracy artifacts
remain untouched for provenance.

Run ONLY after scientific review of Stage 3A.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

EXPECTED_OBJECTIVE = "maximize pooled ordinary image-level accuracy on dev_val"

HERE = Path(__file__).resolve().parent
BASE_STAGE4 = HERE / "04_eval_frozen_threshold.py"
ROOT = HERE.parents[1]
PROTOCOL_ROOT = ROOT / "output" / "trufor_policy_c_frozen_protocol"
CALIB_ROOT = PROTOCOL_ROOT / "stage03_dev_calibration_accuracy"
THRESHOLD_JSON = CALIB_ROOT / "frozen_threshold.json"
EVAL_ROOT = PROTOCOL_ROOT / "stage04_clean_evaluation_accuracy"


def load_base_module():
    if not BASE_STAGE4.is_file():
        raise RuntimeError(
            "Required committed base Stage-4 script is missing:\n"
            f"{BASE_STAGE4}\n"
            "Restore scripts/trufor/04_eval_frozen_threshold.py from git."
        )

    spec = importlib.util.spec_from_file_location(
        "trufor_stage4_base",
        BASE_STAGE4,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load base Stage-4 module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_accuracy_threshold() -> dict:
    if not THRESHOLD_JSON.is_file():
        raise RuntimeError(
            "Accuracy-calibrated threshold is missing:\n"
            f"{THRESHOLD_JSON}\n"
            "Run 03_calibrate_dev_threshold_accuracy.py first."
        )

    payload = json.loads(THRESHOLD_JSON.read_text())
    if payload.get("status") != "FROZEN":
        raise RuntimeError("Accuracy-calibrated threshold is not FROZEN")
    if payload.get("calibration_split") != "dev_val":
        raise RuntimeError("Threshold was not calibrated on dev_val")
    if payload.get("selection_objective") != EXPECTED_OBJECTIVE:
        raise RuntimeError(
            "Refusing to run Stage 4 with the wrong calibration objective:\n"
            f"found: {payload.get('selection_objective')!r}\n"
            f"expected: {EXPECTED_OBJECTIVE!r}"
        )
    if payload.get("decision_rule") != "attack iff trufor_score >= threshold":
        raise RuntimeError("Unexpected decision rule")
    return payload


def main() -> None:
    frozen = validate_accuracy_threshold()
    module = load_base_module()

    # Redirect ONLY path globals. All clean-evaluation mathematics and visual
    # generation remain the committed Stage-4 implementation.
    module.PROTOCOL_ROOT = PROTOCOL_ROOT
    module.CALIB_ROOT = CALIB_ROOT
    module.THRESHOLD_JSON = THRESHOLD_JSON
    module.EVAL_ROOT = EVAL_ROOT

    print("TRUFOR STAGE 4 — ACCURACY-CALIBRATED CLEAN EVALUATION")
    print(f"frozen threshold: {float(frozen['frozen_threshold']):.9f}")
    print(f"calibration objective: {EXPECTED_OBJECTIVE}")
    print(f"output root: {EVAL_ROOT}")
    print("base evaluator: scripts/trufor/04_eval_frozen_threshold.py")
    print()

    module.main()


if __name__ == "__main__":
    main()
