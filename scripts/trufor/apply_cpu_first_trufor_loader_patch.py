#!/usr/bin/env python3
"""Apply the minimal CPU-first TruFor checkpoint-loading patch.

This patch changes only checkpoint/model materialisation order:

    disk -> CPU checkpoint -> CPU model strict load -> model.to(device)

It does not alter checkpoint bytes, architecture, preprocessing, inference
arithmetic, threshold, localisation metric, or attack threat model.

The script is intentionally defensive: it patches only the exact loader blocks
used by this project and creates a backup before writing.
"""

from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "scripts" / "trufor" / "trufor_common.py"
BACKUP = TARGET.with_name("trufor_common.py.pre_cpu_first_loader.bak")

OLD_SAFE_LOAD = '''def _safe_torch_load(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
'''

NEW_SAFE_LOAD = '''def _safe_torch_load(path: Path, device):
    """Deserialize TruFor checkpoint storage on CPU first.

    The ``device`` argument is retained for call-site compatibility, but the
    checkpoint is deliberately materialised on CPU. The fully constructed and
    strictly loaded model is moved to ``device`` afterwards in
    ``load_trufor_model``.
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
'''

OLD_MODEL_LOAD = '''    checkpoint = _safe_torch_load(TRUFOR_CHECKPOINT, device)
    if "state_dict" not in checkpoint:
        raise RuntimeError("TruFor checkpoint does not contain state_dict")

    model = get_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, cfg
'''

NEW_MODEL_LOAD = '''    checkpoint = _safe_torch_load(TRUFOR_CHECKPOINT, device)
    if "state_dict" not in checkpoint:
        raise RuntimeError("TruFor checkpoint does not contain state_dict")

    epoch = checkpoint.get("epoch", None)
    if epoch is None or int(epoch) != 81:
        raise RuntimeError(
            f"Unexpected TruFor checkpoint epoch: expected 81, got {epoch!r}"
        )

    # CPU-first loading is deliberate. This avoids materialising serialized
    # checkpoint storage directly on CUDA before the model exists.
    first_state_tensor = next(
        (value for value in checkpoint["state_dict"].values() if torch.is_tensor(value)),
        None,
    )
    if first_state_tensor is None:
        raise RuntimeError("TruFor checkpoint state_dict contains no tensors")
    if first_state_tensor.device.type != "cpu":
        raise RuntimeError(
            "CPU-first checkpoint invariant failed: state_dict tensor is on "
            f"{first_state_tensor.device}"
        )

    model = get_model(cfg)
    first_model_param = next(model.parameters(), None)
    if first_model_param is None:
        raise RuntimeError("Constructed TruFor model has no parameters")
    if first_model_param.device.type != "cpu":
        raise RuntimeError(
            "Expected TruFor model construction on CPU before state_dict load; "
            f"got {first_model_param.device}"
        )

    load_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict TruFor state_dict load mismatch:\n"
            f"missing={load_result.missing_keys}\n"
            f"unexpected={load_result.unexpected_keys}"
        )

    model = model.to(device)
    model.eval()

    first_model_param = next(model.parameters())
    if first_model_param.device != device:
        raise RuntimeError(
            "TruFor model did not move to requested device: "
            f"expected {device}, got {first_model_param.device}"
        )

    return model, checkpoint, cfg
'''


def patched_text(original: str) -> str:
    safe_count = original.count(OLD_SAFE_LOAD)
    model_count = original.count(OLD_MODEL_LOAD)

    # Idempotence: recognise an already-applied patch.
    if safe_count == 0 and NEW_SAFE_LOAD in original and NEW_MODEL_LOAD in original:
        return original

    if safe_count != 1:
        raise RuntimeError(
            "Could not identify the expected _safe_torch_load block exactly once. "
            f"Found {safe_count}. Refusing to patch automatically."
        )
    if model_count != 1:
        raise RuntimeError(
            "Could not identify the expected load_trufor_model block exactly once. "
            f"Found {model_count}. Refusing to patch automatically."
        )

    out = original.replace(OLD_SAFE_LOAD, NEW_SAFE_LOAD, 1)
    out = out.replace(OLD_MODEL_LOAD, NEW_MODEL_LOAD, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the patch; without this flag only print the proposed diff",
    )
    args = parser.parse_args()

    if not TARGET.is_file():
        raise RuntimeError(f"Missing target: {TARGET}")

    original = TARGET.read_text()
    updated = patched_text(original)

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(TARGET),
            tofile=str(TARGET) + " (CPU-first)",
        )
    )

    if not diff:
        print("CPU-first loader patch is already applied; no changes needed.")
        return

    print(diff)
    if not args.apply:
        print("\nDRY RUN ONLY. Re-run with --apply after reviewing the diff.")
        return

    if not BACKUP.exists():
        BACKUP.write_text(original)
        print(f"backup: {BACKUP}")
    else:
        print(f"backup already exists: {BACKUP}")

    tmp = TARGET.with_name(TARGET.name + ".tmp_cpu_first")
    tmp.write_text(updated)
    os.replace(tmp, TARGET)
    print(f"patched: {TARGET}")
    print("CPU-FIRST TRUFOR LOADER PATCH APPLIED")


if __name__ == "__main__":
    main()
