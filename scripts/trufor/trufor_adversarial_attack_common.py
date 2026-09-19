#!/usr/bin/env python3
"""
Shared implementation for the frozen TruFor classification-preserving
adversarial localisation pilot.

Scientific contract
-------------------
- attack only the already-frozen TruFor own clean-correct population;
- primary objective is altered-union anomaly-map mass E;
- physical RGB L_inf epsilon = 1/255;
- physical alpha = 0.25/255;
- 10 projected sign-gradient steps;
- hard classification constraint: authoritative TruFor score >= frozen threshold;
- 8-step classification-boundary feasibility bisection;
- candidate is accepted only if authoritative E does not increase
  (same acceptance concept as the frozen ResNet comparator);
- no random start and no restarts;
- gradient instance uses the validated exact full-native memory route:
  CPU-first load, frozen parameters, DnCNN checkpointed in 4 sequential
  segments, 32 SegFormer MLPs checkpointed, and exact self-attention
  query-row chunking;
- every authoritative score/map/E is produced by a separate, unmodified
  reference TruFor instance;
- final adversarial input remains continuous float32. No uint8 rounding,
  JPEG, Policy-C rerun, resize, crop, or padding is introduced.

The attack is GT-aware/oracle because the altered-union GT is used directly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from trufor_common import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TRUFOR_COMMIT,
    ROOT,
    TRUFOR_CHECKPOINT,
    TRUFOR_CODE,
    TRUFOR_CONFIG,
    project_git_head,
    relative_to_root,
    sha256_file,
    stable_token,
    verify_trufor_provenance,
)
from trufor_attack_pilot_common import (
    ALPHA_MODEL,
    ALPHA_PHYSICAL,
    CLEAN_CORRECT_ATTACKS,
    EPSILON_MODEL,
    EPSILON_PHYSICAL,
    LOCALISATION_PER_IMAGE,
    MODEL_MAX,
    PILOT_IMAGE_PATHS,
    PILOT_ROLE,
    STAGE4_PROVENANCE,
    build_union_mask,
    canonical_model_tensor_from_uint8,
    load_frozen_threshold,
    load_regions,
    load_rgb_uint8,
    localisation_values_np,
    physical_linf,
    safe_run_tag,
    validate_stage4_population,
)

ATTACK_STEPS = 10
BISECTION_STEPS = 8
OBJECTIVE_TOL = 1e-8

# Frozen, validated Stage-18 route.
MAX_SCORE_ELEMS = 300_000_000
DNCNN_CHECKPOINT_SEGMENTS = 4
EXPECTED_ATTENTION_MODULES = 32
EXPECTED_MLP_MODULES = 32
EXPECTED_CHECKPOINT_EPOCH = 81

# Integrity tolerances. Cross-machine clean scores were already observed to
# differ by at most 1.13e-5 without threshold flips.
CLEAN_SCORE_PARITY_ATOL = 2.0e-5
CLEAN_E_PARITY_ATOL = 2.0e-5
ATTACK_REFERENCE_E_PARITY_ATOL = 2.0e-5
L_INF_AUDIT_ATOL = 2.0e-7

ALLOCATOR_ENV_VARS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
)

IMPLEMENTATION_FILENAMES = (
    "trufor_adversarial_attack_common.py",
    "19_freeze_trufor_adversarial_protocol.py",
    "20_run_trufor_adversarial_pilot.py",
)


def attack_root(run_tag: str) -> Path:
    return (
        ROOT
        / "output"
        / safe_run_tag(run_tag)
        / "trufor_adversarial_localisation_attack"
    )


def pilot_root(run_tag: str) -> Path:
    return attack_root(run_tag) / "pilot"


def protocol_config_path(run_tag: str) -> Path:
    return attack_root(run_tag) / "attack_protocol_config.json"


def selection_path(run_tag: str) -> Path:
    return attack_root(run_tag) / "pilot_selection.csv"


def stage19_provenance_path(run_tag: str) -> Path:
    return attack_root(run_tag) / "stage19_protocol_provenance.json"


def implementation_hashes() -> Dict[str, str]:
    here = Path(__file__).resolve().parent
    result = {}
    for name in IMPLEMENTATION_FILENAMES:
        path = here / name
        if not path.is_file():
            raise RuntimeError(f"Missing bundled implementation file: {path}")
        result[name] = sha256_file(path)
    return result


def canonical_json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(str(tmp), **arrays)
    os.replace(tmp, path)


def protocol_config(threshold: float) -> dict:
    return {
        "protocol_id": "trufor_adv_localisation_v1",
        "scientific_role": (
            "GT-aware white-box localisation robustness stress test conditional "
            "on clean-correct detection"
        ),
        "population": {
            "selection": "TruFor own frozen clean-correct attack population",
            "selection_time": "before attack",
            "post_attack_reselection": False,
            "pilot_images": 6,
        },
        "objective": {
            "primary": "minimise altered-union anomaly-map mass E",
            "formula": "E = anomaly mass inside frozen altered union / total anomaly mass",
            "gt_aware": True,
            "candidate_acceptance": (
                "accept classification-feasible candidate only if "
                "E_candidate <= E_current + 1e-8"
            ),
            "objective_tolerance": OBJECTIVE_TOL,
        },
        "classification_constraint": {
            "type": "hard",
            "decision_rule": "attack iff score >= threshold",
            "threshold": float(threshold),
            "bisection_steps": BISECTION_STEPS,
            "bisection_role": (
                "feasibility pull-back only; E is not used to choose a point "
                "inside the bisection"
            ),
        },
        "optimisation": {
            "method": "projected sign-gradient descent on E",
            "steps": ATTACK_STEPS,
            "random_start": False,
            "restarts": 0,
            "epsilon_physical_rgb01": EPSILON_PHYSICAL,
            "alpha_physical_rgb01": ALPHA_PHYSICAL,
            "epsilon_model_x": EPSILON_MODEL,
            "alpha_model_x": ALPHA_MODEL,
            "model_coordinate_relation": "x = z * 255/256",
            "model_valid_range": [0.0, MODEL_MAX],
            "support": (
                "native image pixels only; TruFor has no artificial padding, "
                "so all native pixels are document-content support"
            ),
        },
        "candidate_evaluation": {
            "gradient_model": (
                "memory-efficient exact full-native TruFor attack instance"
            ),
            "authoritative_model": (
                "separate completely unmodified reference TruFor"
            ),
            "reported_scores_maps_metrics_from": "authoritative reference only",
        },
        "memory_route": {
            "engineering_revision": "v2_attention_chunk_activation_checkpoint",
            "cpu_first_checkpoint_load": True,
            "weights_frozen": True,
            "input_gradients_only": True,
            "dncnn_checkpoint_segments": DNCNN_CHECKPOINT_SEGMENTS,
            "segformer_mlp_checkpoint_count": EXPECTED_MLP_MODULES,
            "segformer_attention_query_chunk_count": EXPECTED_ATTENTION_MODULES,
            "max_score_elems": MAX_SCORE_ELEMS,
            "query_chunk_rule": (
                "max(1024, MAX_SCORE_ELEMS // (n_keys * n_heads))"
            ),
            "attention_chunk_activation_checkpointing": True,
            "attention_chunk_checkpoint_use_reentrant": False,
            "attention_chunk_semantics": (
                "exact q/k/v + exact softmax; only large score/softmax activation "
                "lifetime changes and is recomputed during backward"
            ),
            "full_native_image": True,
            "image_tiling": False,
            "attention_context_approximation": False,
            "create_graph": False,
            "retain_graph": False,
        },
        "backend": {
            "CUBLAS_WORKSPACE_CONFIG": "unset",
            "PYTORCH_ALLOC_CONF": "unset",
            "PYTORCH_CUDA_ALLOC_CONF": "unset",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "cudnn_enabled": False,
            "deterministic_algorithms": False,
            "float32_matmul_precision": "highest",
        },
        "final_adversarial_representation": {
            "authoritative": "continuous float32 TruFor model input x",
            "physical_recovery": "z = x * 256/255",
            "requantise_uint8": False,
            "re_jpeg": False,
            "rerun_policy_c": False,
            "resize": False,
            "crop": False,
            "pad": False,
        },
        "resume": {
            "atomic_unit": "one image",
            "mid_image_checkpointing": False,
            "restart_policy": (
                "skip only hash-validated completed images; rerun incomplete image"
            ),
        },
        "frozen_upstream": {
            "trufor_commit": EXPECTED_TRUFOR_COMMIT,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        },
    }


def verify_protocol_freeze(run_tag: str) -> Tuple[dict, str, pd.DataFrame, dict]:
    cfg_path = protocol_config_path(run_tag)
    sel_path = selection_path(run_tag)
    prov_path = stage19_provenance_path(run_tag)

    for path in (cfg_path, sel_path, prov_path):
        if not path.is_file():
            raise RuntimeError(
                f"Missing Stage-19 frozen protocol artifact: {path}\n"
                "Run Stage 19 first."
            )

    provenance = json.loads(prov_path.read_text())
    if provenance.get("status") != "FROZEN":
        raise RuntimeError("Stage-19 provenance is not FROZEN")

    actual_cfg_sha = sha256_file(cfg_path)
    if actual_cfg_sha != provenance.get("attack_protocol_config_sha256"):
        raise RuntimeError("Frozen attack protocol config SHA256 mismatch")

    actual_sel_sha = sha256_file(sel_path)
    if actual_sel_sha != provenance.get("pilot_selection_sha256"):
        raise RuntimeError("Frozen pilot selection SHA256 mismatch")

    actual_impl = implementation_hashes()
    if actual_impl != provenance.get("implementation_sha256"):
        raise RuntimeError(
            "Implementation files changed after Stage 19.\n"
            "Do not continue under the old frozen hash. Review changes and "
            "rerun Stage 19 deliberately if the protocol is being superseded."
        )

    cfg = json.loads(cfg_path.read_text())
    _, threshold = load_frozen_threshold()
    expected_cfg = protocol_config(threshold)
    if cfg != expected_cfg:
        raise RuntimeError(
            "Frozen config content does not match the implementation's exact "
            "protocol definition"
        )

    selection = pd.read_csv(sel_path, keep_default_na=False)
    if len(selection) != 6:
        raise RuntimeError(f"Expected six pilot rows, found {len(selection)}")

    expected_paths = list(PILOT_IMAGE_PATHS)
    actual_paths = selection.sort_values("pilot_order")["image_path"].tolist()
    if actual_paths != expected_paths:
        raise RuntimeError("Pilot selection/order differs from predeclared six images")

    return cfg, actual_cfg_sha, selection, provenance


def assert_allocator_environment_unset() -> None:
    present = {name: os.environ.get(name) for name in ALLOCATOR_ENV_VARS if os.environ.get(name)}
    if present:
        raise RuntimeError(
            "Frozen backend requires allocator/CUBLAS environment variables to "
            f"be unset, but found: {present}\n"
            "Start a fresh shell with these variables unset. In particular, do "
            "not use expandable_segments on this machine."
        )


def enforce_backend_contract() -> None:
    assert_allocator_environment_unset()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")


def backend_record() -> dict:
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def validate_backend_record() -> dict:
    enforce_backend_contract()
    record = backend_record()
    expected = {
        "CUBLAS_WORKSPACE_CONFIG": None,
        "PYTORCH_ALLOC_CONF": None,
        "PYTORCH_CUDA_ALLOC_CONF": None,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "cudnn_enabled": False,
        "deterministic_algorithms": False,
        "float32_matmul_precision": "highest",
    }
    if record != expected:
        raise RuntimeError(f"Backend contract mismatch:\nactual={record}\nexpected={expected}")
    return record


def resolve_cuda_device(gpu: int) -> torch.device:
    if gpu < 0:
        raise RuntimeError(
            "The frozen pilot is a CUDA memory/attack experiment; CPU execution "
            "is not an equivalent route."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU {gpu} requested but only {torch.cuda.device_count()} devices exist"
        )
    return torch.device(f"cuda:{gpu}")


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _prepare_upstream_imports():
    code = str(TRUFOR_CODE)
    if code not in sys.path:
        sys.path.insert(0, code)

    if "lib" in sys.modules:
        lib_file = getattr(sys.modules["lib"], "__file__", None)
        if lib_file is not None and not str(Path(lib_file).resolve()).startswith(
            str(TRUFOR_CODE.resolve())
        ):
            raise RuntimeError(
                "A non-TruFor module named 'lib' was imported before the frozen "
                f"TruFor source tree: {lib_file}"
            )

    from lib.config import config as base_config  # type: ignore
    from lib.utils import get_model  # type: ignore
    import lib  # type: ignore

    lib_file = Path(lib.__file__).resolve()
    if not str(lib_file).startswith(str(TRUFOR_CODE.resolve())):
        raise RuntimeError(f"TruFor import escaped canonical source tree: {lib_file}")

    cfg = base_config.clone()
    cfg.defrost()
    cfg.merge_from_file(str(TRUFOR_CONFIG))
    cfg.TEST.MODEL_FILE = str(TRUFOR_CHECKPOINT)
    cfg.freeze()
    return get_model, cfg


def _all_checkpoint_tensors_cpu(state_dict: dict) -> bool:
    tensors = [v for v in state_dict.values() if torch.is_tensor(v)]
    return bool(tensors) and all(v.device.type == "cpu" for v in tensors)


def freeze_all_parameters(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


def load_two_models_cpu_first(device: torch.device):
    """
    Exactly preserve the validated CPU-first load:
      checkpoint -> CPU
      model -> CPU
      strict state_dict load
      model -> CUDA

    Returns attack model, unmodified reference model, and load metadata.
    """
    verify_trufor_provenance(check_archive_member=False)
    validate_backend_record()

    checkpoint = _torch_load_cpu(TRUFOR_CHECKPOINT)
    if "state_dict" not in checkpoint:
        raise RuntimeError("TruFor checkpoint has no state_dict")

    epoch = int(checkpoint.get("epoch", -1))
    if epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise RuntimeError(
            f"Checkpoint epoch mismatch: expected {EXPECTED_CHECKPOINT_EPOCH}, got {epoch}"
        )

    state_dict = checkpoint["state_dict"]
    if not _all_checkpoint_tensors_cpu(state_dict):
        raise RuntimeError("CPU-first load failed: checkpoint tensors are not all on CPU")

    get_model, cfg = _prepare_upstream_imports()

    def build_one():
        model = get_model(cfg)  # CPU construction
        incompat = model.load_state_dict(state_dict, strict=True)
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(
                f"Strict state_dict load mismatch: missing={missing}, unexpected={unexpected}"
            )
        freeze_all_parameters(model)
        model = model.to(device)
        freeze_all_parameters(model)
        return model, missing, unexpected

    attack_model, missing_a, unexpected_a = build_one()
    reference_model, missing_r, unexpected_r = build_one()

    # Release the CPU checkpoint only after both strict loads completed.
    del state_dict
    del checkpoint
    gc.collect()

    validate_backend_record()

    if next(attack_model.parameters()).device != device:
        raise RuntimeError("Attack model did not move to requested CUDA device")
    if next(reference_model.parameters()).device != device:
        raise RuntimeError("Reference model did not move to requested CUDA device")

    metadata = {
        "checkpoint_epoch": epoch,
        "checkpoint_tensor_device": "cpu",
        "attack_model_parameter_device": str(next(attack_model.parameters()).device),
        "reference_model_parameter_device": str(next(reference_model.parameters()).device),
        "strict_attack_missing": len(missing_a),
        "strict_attack_unexpected": len(unexpected_a),
        "strict_reference_missing": len(missing_r),
        "strict_reference_unexpected": len(unexpected_r),
    }
    return attack_model, reference_model, metadata


def disable_inplace_activations(model) -> int:
    changed = 0
    for module in model.modules():
        if hasattr(module, "inplace") and getattr(module, "inplace") is True:
            try:
                module.inplace = False
                changed += 1
            except Exception:
                pass
    return changed


def patch_memory_efficient_attack_model(model) -> dict:
    """
    Instance-only monkey patches. The reference model remains untouched.

    Attention chunking is exact because each query row is independent once K/V
    are fixed; keys, values, softmax dimension, projections, and full image
    context remain unchanged.
    """
    from lib.models.cmx.encoders.dual_segformer import Attention, Mlp  # type: ignore

    inplace_changed = disable_inplace_activations(model)

    attention_modules = []
    mlp_modules = []

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            attention_modules.append((name, module))
        if isinstance(module, Mlp):
            mlp_modules.append((name, module))

    if len(attention_modules) != EXPECTED_ATTENTION_MODULES:
        raise RuntimeError(
            f"Expected {EXPECTED_ATTENTION_MODULES} SegFormer Attention modules, "
            f"found {len(attention_modules)}"
        )
    if len(mlp_modules) != EXPECTED_MLP_MODULES:
        raise RuntimeError(
            f"Expected {EXPECTED_MLP_MODULES} SegFormer MLP modules, "
            f"found {len(mlp_modules)}"
        )
    if not isinstance(model.dncnn, torch.nn.Sequential):
        raise RuntimeError(
            "Expected model.dncnn to be nn.Sequential for the validated "
            "checkpoint_sequential route"
        )

    query_stats = {}

    for name, module in attention_modules:
        query_stats[name] = {
            "calls": 0,
            "total_chunks": 0,
            "max_chunks": 0,
            "split_calls": 0,
            "last_n_queries": 0,
            "last_n_keys": 0,
            "last_heads": 0,
            "last_chunk_rows": 0,
        }

        def chunked_forward(self, x, H, W, _name=name):
            B, N, C = x.shape
            q = (
                self.q(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            k, v = kv[0], kv[1]

            n_keys = int(k.shape[-2])
            denom = n_keys * int(self.num_heads)
            if denom <= 0:
                raise RuntimeError("Invalid attention key/head count")

            chunk_rows = max(1024, MAX_SCORE_ELEMS // denom)
            n_chunks = int(math.ceil(N / float(chunk_rows)))

            stat = query_stats[_name]
            stat["calls"] += 1
            stat["total_chunks"] += n_chunks
            stat["max_chunks"] = max(stat["max_chunks"], n_chunks)
            stat["split_calls"] += int(n_chunks > 1)
            stat["last_n_queries"] = int(N)
            stat["last_n_keys"] = n_keys
            stat["last_heads"] = int(self.num_heads)
            stat["last_chunk_rows"] = int(chunk_rows)

            # IMPORTANT MEMORY INVARIANT:
            # Each query chunk's O(Q_chunk * K) attention-score/softmax tensor is
            # activation-checkpointed. Therefore it is discarded after the
            # chunk's forward computation and recomputed during backward rather
            # than retained for all chunks / all attention blocks.
            #
            # This does NOT tile the image and does NOT approximate attention:
            # q, k, v, scaling, softmax dimension and full K/V context are exact.
            def attend_one_chunk(q_chunk, k_full, v_full):
                scores = (q_chunk @ k_full.transpose(-2, -1)) * self.scale
                probs = scores.softmax(dim=-1)
                probs = self.attn_drop(probs)
                return probs @ v_full

            pieces = []
            for start in range(0, N, chunk_rows):
                stop = min(N, start + chunk_rows)
                q_chunk = q[:, :, start:stop, :]

                if (
                    torch.is_grad_enabled()
                    and (
                        q_chunk.requires_grad
                        or k.requires_grad
                        or v.requires_grad
                    )
                ):
                    chunk_out = checkpoint(
                        attend_one_chunk,
                        q_chunk,
                        k,
                        v,
                        use_reentrant=False,
                    )
                else:
                    chunk_out = attend_one_chunk(q_chunk, k, v)

                pieces.append(chunk_out)

            x_out = torch.cat(pieces, dim=2)
            x_out = x_out.transpose(1, 2).reshape(B, N, C)
            x_out = self.proj(x_out)
            x_out = self.proj_drop(x_out)
            return x_out

        module.forward = types.MethodType(chunked_forward, module)

    for _, module in mlp_modules:
        original_forward = module.forward

        def checkpointed_mlp_forward(self, x, H, W, _orig=original_forward):
            if torch.is_grad_enabled() and x.requires_grad:
                return checkpoint(
                    lambda y: _orig(y, H, W),
                    x,
                    use_reentrant=False,
                )
            return _orig(x, H, W)

        module.forward = types.MethodType(checkpointed_mlp_forward, module)

    model._trufor_attack_query_stats = query_stats
    model._trufor_attack_route = {
        "attention_modules": len(attention_modules),
        "mlp_modules": len(mlp_modules),
        "dncnn_segments": DNCNN_CHECKPOINT_SEGMENTS,
        "max_score_elems": MAX_SCORE_ELEMS,
        "attention_chunk_activation_checkpointing": True,
        "attention_chunk_checkpoint_use_reentrant": False,
        "inplace_activations_disabled": inplace_changed,
    }
    return dict(model._trufor_attack_route)


def attention_chunk_checkpoint_equivalence_self_test() -> dict:
    """
    CPU unit test for the exact engineering primitive used in the patched
    attention modules: full attention vs query-chunked + activation-checkpointed
    attention. This is not a TruFor empirical test; it guards against changing
    the attention mathematics while fixing activation lifetime.
    """
    torch.manual_seed(1701)

    B, H, NQ, NK, D = 1, 3, 41, 17, 8
    scale = D ** -0.5
    chunk_rows = 11

    q0 = torch.randn(B, H, NQ, D, dtype=torch.float32, requires_grad=True)
    k0 = torch.randn(B, H, NK, D, dtype=torch.float32, requires_grad=True)
    v0 = torch.randn(B, H, NK, D, dtype=torch.float32, requires_grad=True)

    def full_attention(q, k, v):
        scores = (q @ k.transpose(-2, -1)) * scale
        probs = scores.softmax(dim=-1)
        return probs @ v

    full_out = full_attention(q0, k0, v0)
    full_loss = (full_out.square()).sum()
    full_grads = torch.autograd.grad(full_loss, (q0, k0, v0))

    q1 = q0.detach().clone().requires_grad_(True)
    k1 = k0.detach().clone().requires_grad_(True)
    v1 = v0.detach().clone().requires_grad_(True)

    def attend_one_chunk(q_chunk, k_full, v_full):
        scores = (q_chunk @ k_full.transpose(-2, -1)) * scale
        probs = scores.softmax(dim=-1)
        return probs @ v_full

    pieces = []
    for start in range(0, NQ, chunk_rows):
        stop = min(NQ, start + chunk_rows)
        pieces.append(
            checkpoint(
                attend_one_chunk,
                q1[:, :, start:stop, :],
                k1,
                v1,
                use_reentrant=False,
            )
        )

    chunk_out = torch.cat(pieces, dim=2)
    chunk_loss = (chunk_out.square()).sum()
    chunk_grads = torch.autograd.grad(chunk_loss, (q1, k1, v1))

    out_max_abs = float((full_out - chunk_out).abs().max().item())
    grad_max_abs = [
        float((a - b).abs().max().item())
        for a, b in zip(full_grads, chunk_grads)
    ]

    if out_max_abs > 5e-6 or max(grad_max_abs) > 5e-5:
        raise RuntimeError(
            "Attention chunk activation-checkpoint equivalence self-test failed: "
            f"out={out_max_abs}, grads={grad_max_abs}"
        )

    return {
        "status": "PASS",
        "output_max_abs_error": out_max_abs,
        "q_grad_max_abs_error": grad_max_abs[0],
        "k_grad_max_abs_error": grad_max_abs[1],
        "v_grad_max_abs_error": grad_max_abs[2],
        "chunk_rows": chunk_rows,
        "use_reentrant": False,
    }


def reset_query_stats(model) -> None:
    stats = getattr(model, "_trufor_attack_query_stats", None)
    if stats is None:
        raise RuntimeError("Attack model has not been memory-route patched")
    for stat in stats.values():
        stat.update(
            calls=0,
            total_chunks=0,
            max_chunks=0,
            split_calls=0,
            last_n_queries=0,
            last_n_keys=0,
            last_heads=0,
            last_chunk_rows=0,
        )


def query_stats_summary(model) -> dict:
    stats = getattr(model, "_trufor_attack_query_stats", None)
    if stats is None:
        raise RuntimeError("Attack model has not been memory-route patched")

    observed = sum(int(v["calls"] > 0) for v in stats.values())
    total_chunks = sum(int(v["total_chunks"]) for v in stats.values())
    split_modules = sum(int(v["split_calls"] > 0) for v in stats.values())
    max_chunks = max((int(v["max_chunks"]) for v in stats.values()), default=0)
    return {
        "attention_modules_observed": observed,
        "total_query_chunks": total_chunks,
        "split_attention_modules": split_modules,
        "max_chunks_one_attention": max_chunks,
    }


def memory_efficient_differentiable_forward(model, rgb: torch.Tensor):
    """
    Full native TruFor forward with the exact Stage-18 memory route.
    No image tiling, resize, crop, or attention approximation is performed.
    """
    from lib.models.cmx.layer_utils import weighted_statistics_pooling  # type: ignore

    modal_x = None
    if "NP++" in model.mods:
        modal_x = checkpoint_sequential(
            model.dncnn,
            DNCNN_CHECKPOINT_SEGMENTS,
            rgb,
            use_reentrant=False,
        )
        if model.np_out_ch == 1:
            modal_x = torch.tile(modal_x, (3, 1, 1))
        elif model.np_out_ch != 3:
            raise RuntimeError(f"Unexpected NP++ channels: {model.np_out_ch}")

    rgb_branch = rgb
    if "RGB" not in model.mods:
        rgb_branch = None
    elif model.prepro is not None:
        rgb_branch = model.prepro(rgb_branch)

    orisize = rgb_branch.shape if rgb_branch is not None else modal_x.shape

    features = model.backbone(rgb_branch, modal_x)

    out = model.decode_head(features)
    out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)

    if model.decode_head_conf is None:
        raise RuntimeError("TruFor confidence head missing")
    conf = model.decode_head_conf(features)
    conf = F.interpolate(conf, size=orisize[2:], mode="bilinear", align_corners=False)

    if model.detection is None or model.conf_detection != "confpool":
        raise RuntimeError("Unexpected TruFor detection configuration")

    f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
    f2 = weighted_statistics_pooling(
        out[:, 1:2, :, :] - out[:, 0:1, :, :],
        F.logsigmoid(conf),
    ).view(out.shape[0], -1)
    det = model.detection(torch.cat((f1, f2), dim=-1))
    return out, conf, det


def anomaly_from_logits(out: torch.Tensor) -> torch.Tensor:
    return F.softmax(out, dim=1)[:, 1, :, :]


def differentiable_E(anomaly: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if anomaly.ndim != 3 or mask.ndim != 3:
        raise RuntimeError(
            f"Expected [B,H,W] anomaly/mask, got {anomaly.shape}/{mask.shape}"
        )
    denom = anomaly.sum(dim=(1, 2))
    numer = (anomaly * mask).sum(dim=(1, 2))
    if torch.any(~torch.isfinite(denom)) or torch.any(denom <= 0):
        raise RuntimeError("Invalid anomaly-map mass denominator")
    return numer / denom


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gib(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0 ** 3)


def cuda_memory_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "allocated_gib": gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": gib(torch.cuda.memory_reserved(device)),
        "free_gib": gib(free_b),
        "total_gib": gib(total_b),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }


def gradient_step(
    attack_model,
    current_x: torch.Tensor,
    altered_mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Return dE/dx for exactly one first-order step. No model parameter gradients
    are created.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    reset_query_stats(attack_model)

    x = current_x.detach().clone().requires_grad_(True)
    sync_if_cuda(device)
    memory_before = cuda_memory_snapshot(device)

    t_forward = time.perf_counter()
    out, conf, det = memory_efficient_differentiable_forward(attack_model, x)
    anomaly = anomaly_from_logits(out)
    E = differentiable_E(anomaly, altered_mask).sum()
    sync_if_cuda(device)
    forward_seconds = time.perf_counter() - t_forward
    memory_after_forward = cuda_memory_snapshot(device)

    # Query statistics are measured after the forward. Chunk checkpoint
    # recomputation happens inside backward and must not change the forward
    # chunk-count regression values.
    stats = query_stats_summary(attack_model)

    t_backward = time.perf_counter()
    grad = torch.autograd.grad(
        E,
        x,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    sync_if_cuda(device)
    backward_seconds = time.perf_counter() - t_backward
    memory_after_backward = cuda_memory_snapshot(device)
    seconds = forward_seconds + backward_seconds

    if not torch.isfinite(E):
        raise RuntimeError("Non-finite differentiable E")
    if not torch.isfinite(grad).all():
        raise RuntimeError("Non-finite input gradient")

    if stats["attention_modules_observed"] != EXPECTED_ATTENTION_MODULES:
        raise RuntimeError(
            "Not all 32 SegFormer self-attention modules were observed in "
            f"gradient forward: {stats}"
        )

    peak_alloc = gib(torch.cuda.max_memory_allocated(device))
    peak_reserved = gib(torch.cuda.max_memory_reserved(device))

    result = {
        "attack_E": float(E.detach().cpu().item()),
        "grad": grad.detach(),
        "gradient_seconds": float(seconds),
        "gradient_forward_seconds": float(forward_seconds),
        "gradient_backward_seconds": float(backward_seconds),
        "gradient_peak_allocated_gib": peak_alloc,
        "gradient_peak_reserved_gib": peak_reserved,
        "memory_before_gradient": memory_before,
        "memory_after_forward": memory_after_forward,
        "memory_after_backward": memory_after_backward,
        **stats,
    }

    del out, conf, det, anomaly, E, x, grad
    return result


def project_candidate(
    candidate: torch.Tensor,
    clean_x: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        lower = torch.clamp(
            clean_x - EPSILON_MODEL,
            min=0.0,
            max=MODEL_MAX,
        )
        upper = torch.clamp(
            clean_x + EPSILON_MODEL,
            min=0.0,
            max=MODEL_MAX,
        )
        candidate = torch.maximum(torch.minimum(candidate, upper), lower)
        candidate = torch.clamp(candidate, min=0.0, max=MODEL_MAX)
    return candidate.detach()


def propose_sign_step(
    current_x: torch.Tensor,
    clean_x: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        proposal = current_x - ALPHA_MODEL * grad.sign()
    return project_candidate(proposal, clean_x)


class ReferenceEvaluator:
    """
    Authoritative, completely unmodified TruFor evaluation.
    """

    def __init__(
        self,
        model,
        altered_mask: torch.Tensor,
        threshold: float,
        device: torch.device,
    ):
        self.model = model
        self.altered_mask = altered_mask
        self.threshold = float(threshold)
        self.device = device
        self.forward_count = 0
        self.total_seconds = 0.0

    def evaluate(self, x: torch.Tensor, return_map: bool = False) -> dict:
        sync_if_cuda(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out, conf, det, _ = self.model(x, save_np=False)
            anomaly = anomaly_from_logits(out)
            E = differentiable_E(anomaly, self.altered_mask)[0]
            score = torch.sigmoid(det).reshape(-1)[0]

        sync_if_cuda(self.device)
        elapsed = time.perf_counter() - t0
        self.forward_count += 1
        self.total_seconds += elapsed

        score_value = float(score.detach().cpu().item())
        E_value = float(E.detach().cpu().item())
        result = {
            "score": score_value,
            "E": E_value,
            "feasible": bool(score_value >= self.threshold),
            "seconds": float(elapsed),
        }

        if return_map:
            amap = anomaly[0].detach().cpu().numpy().astype(np.float32, copy=True)
            if not np.isfinite(amap).all():
                raise RuntimeError("Reference anomaly map contains non-finite values")
            if amap.min() < -1e-6 or amap.max() > 1.0 + 1e-6:
                raise RuntimeError(
                    f"Reference anomaly map outside [0,1]: {amap.min()}..{amap.max()}"
                )
            result["map"] = amap

        del out, conf, det, anomaly, E, score
        return result


def feasibility_pullback(
    evaluator: ReferenceEvaluator,
    current_x: torch.Tensor,
    current_eval: dict,
    proposal_x: torch.Tensor,
    clean_x: torch.Tensor,
) -> dict:
    """
    Mirror the frozen ResNet feasibility concept.

    E is not used to steer bisection. The lower endpoint is explicitly kept
    classification-feasible. Because score need not be globally monotone on
    the segment, this is a deterministic feasibility-backtracking heuristic,
    not a proof of the globally furthest feasible point.
    """
    proposal_eval = evaluator.evaluate(proposal_x, return_map=False)
    if proposal_eval["feasible"]:
        return {
            "candidate_x": proposal_x.detach(),
            "candidate_eval": proposal_eval,
            "boundary_pullback": False,
            "bisection_evals": 0,
            "accepted_lambda": 1.0,
            "direct_proposal_score": proposal_eval["score"],
            "direct_proposal_E": proposal_eval["E"],
        }

    low = current_x.detach().clone()
    high = proposal_x.detach().clone()
    low_eval = dict(current_eval)
    low_lambda = 0.0
    high_lambda = 1.0

    for _ in range(BISECTION_STEPS):
        mid = project_candidate(0.5 * (low + high), clean_x)
        mid_lambda = 0.5 * (low_lambda + high_lambda)
        mid_eval = evaluator.evaluate(mid, return_map=False)
        if mid_eval["feasible"]:
            low = mid
            low_eval = mid_eval
            low_lambda = mid_lambda
        else:
            high = mid
            high_lambda = mid_lambda

    if not low_eval["feasible"]:
        raise RuntimeError("Classification pull-back lost its feasible lower endpoint")

    return {
        "candidate_x": low.detach(),
        "candidate_eval": low_eval,
        "boundary_pullback": True,
        "bisection_evals": BISECTION_STEPS,
        "accepted_lambda": float(low_lambda),
        "direct_proposal_score": proposal_eval["score"],
        "direct_proposal_E": proposal_eval["E"],
    }


def physical_bounds_audit(clean_x: torch.Tensor, adv_x: torch.Tensor) -> dict:
    linf = physical_linf(clean_x, adv_x)
    if linf > EPSILON_PHYSICAL + L_INF_AUDIT_ATOL:
        raise RuntimeError(
            "Adversarial tensor escaped physical L_inf budget: "
            f"{linf} > {EPSILON_PHYSICAL}"
        )

    if float(adv_x.min().item()) < -1e-7:
        raise RuntimeError("Adversarial x fell below model valid range")
    if float(adv_x.max().item()) > MODEL_MAX + 1e-7:
        raise RuntimeError("Adversarial x exceeded model valid range")

    z = adv_x * (256.0 / 255.0)
    z_min = float(z.min().item())
    z_max = float(z.max().item())
    if z_min < -1e-6 or z_max > 1.0 + 1e-6:
        raise RuntimeError(f"Recovered physical z outside [0,1]: {z_min}..{z_max}")

    return {
        "physical_linf": float(linf),
        "adv_x_min": float(adv_x.min().item()),
        "adv_x_max": float(adv_x.max().item()),
        "adv_z_min": z_min,
        "adv_z_max": z_max,
    }


def image_output_dir(run_tag: str, pilot_order: int, image_path: str) -> Path:
    token = stable_token(image_path, 12)
    return pilot_root(run_tag) / "images" / f"{int(pilot_order):02d}_{token}"


def completion_marker_path(image_dir: Path) -> Path:
    return image_dir / "COMPLETE.json"


def validate_completion_marker(
    image_dir: Path,
    expected_protocol_sha: str,
    expected_image_path: str,
) -> bool:
    marker_path = completion_marker_path(image_dir)
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "COMPLETE":
            return False
        if marker.get("protocol_sha256") != expected_protocol_sha:
            return False
        if marker.get("image_path") != expected_image_path:
            return False
        for filename, expected_sha in marker.get("artifact_sha256", {}).items():
            path = image_dir / filename
            if not path.is_file() or sha256_file(path) != expected_sha:
                return False
        return set(marker.get("artifact_sha256", {}).keys()) == {
            "result.json",
            "trace.csv",
            "adversarial_result.npz",
        }
    except Exception:
        return False


def clear_incomplete_image_outputs(image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    if completion_marker_path(image_dir).exists():
        completion_marker_path(image_dir).unlink()
    for name in (
        "result.json",
        "trace.csv",
        "adversarial_result.npz",
        "failure.json",
    ):
        path = image_dir / name
        if path.exists():
            path.unlink()
    for path in image_dir.glob("*.tmp*"):
        path.unlink()


def environment_record(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": str(device),
        "gpu_name": props.name,
        "gpu_total_memory_gib": gib(props.total_memory),
        "project_git_head": project_git_head(),
        "backend": validate_backend_record(),
    }


def family_display(variant: str) -> str:
    mapping = {
        "digital_1": "d1",
        "digital_2": "d2",
        "digital_3": "d3",
        "facedancer": "FaceDancer",
        "textdiffuserft_bfei": "TextDiffuser",
    }
    return mapping.get(str(variant), str(variant))


def run_one_pilot_image(
    *,
    run_tag: str,
    selection_row: pd.Series,
    protocol_sha: str,
    attack_model,
    reference_model,
    route_metadata: dict,
    threshold: float,
    device: torch.device,
) -> dict:
    order = int(selection_row["pilot_order"])
    image_path = str(selection_row["image_path"])
    image_dir = image_output_dir(run_tag, order, image_path)

    if validate_completion_marker(image_dir, protocol_sha, image_path):
        return {"status": "SKIP_COMPLETE", "image_dir": str(image_dir)}

    clear_incomplete_image_outputs(image_dir)

    t_image0 = time.perf_counter()

    try:
        cache_path = ROOT / str(selection_row["cache_path"])
        if not cache_path.is_file():
            raise RuntimeError(f"Missing Policy-C cache image: {cache_path}")

        expected_cache_sha = str(selection_row["cache_sha256"])
        actual_cache_sha = sha256_file(cache_path)
        if actual_cache_sha != expected_cache_sha:
            raise RuntimeError(
                f"Policy-C cache SHA mismatch for {image_path}\n"
                f"expected={expected_cache_sha}\nactual={actual_cache_sha}"
            )

        rgb = load_rgb_uint8(cache_path)
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        if H != int(selection_row["native_height"]) or W != int(selection_row["native_width"]):
            raise RuntimeError(
                f"Native geometry mismatch: selection={selection_row['native_width']}x"
                f"{selection_row['native_height']}, runtime={W}x{H}"
            )

        regions = load_regions()
        union_mask_np, clipped_count = build_union_mask(
            regions,
            image_path,
            H,
            W,
        )
        A = float(union_mask_np.mean())
        if abs(A - float(selection_row["A_union_stage4"])) > 1e-12:
            raise RuntimeError(
                f"Altered-union A mismatch: runtime={A}, "
                f"stage4={selection_row['A_union_stage4']}"
            )

        clean_x = canonical_model_tensor_from_uint8(rgb, device)
        altered_mask = torch.from_numpy(
            union_mask_np.astype(np.float32, copy=False)
        ).unsqueeze(0).to(device=device)

        evaluator = ReferenceEvaluator(
            reference_model,
            altered_mask,
            threshold,
            device,
        )

        # --------------------------------------------------------------
        # Fresh authoritative clean baseline.
        # --------------------------------------------------------------
        clean_eval = evaluator.evaluate(clean_x, return_map=True)
        if not clean_eval["feasible"]:
            raise RuntimeError(
                "Frozen clean-correct pilot image is no longer clean-correct "
                "under fresh authoritative reference inference"
            )

        saved_score = float(selection_row["clean_score_stage4"])
        score_abs_error = abs(clean_eval["score"] - saved_score)
        if score_abs_error > CLEAN_SCORE_PARITY_ATOL:
            raise RuntimeError(
                "Fresh reference clean score differs from frozen Stage-4 score "
                f"by {score_abs_error:.9g} > {CLEAN_SCORE_PARITY_ATOL}"
            )

        clean_metrics = localisation_values_np(clean_eval["map"], union_mask_np)
        clean_E_abs_error = abs(
            clean_metrics["E"] - float(selection_row["E_union_stage4"])
        )
        if clean_E_abs_error > CLEAN_E_PARITY_ATOL:
            raise RuntimeError(
                "Fresh reference clean E differs from frozen Stage-4 E "
                f"by {clean_E_abs_error:.9g} > {CLEAN_E_PARITY_ATOL}"
            )

        current_x = clean_x.detach().clone()
        current_eval = {
            "score": clean_eval["score"],
            "E": clean_eval["E"],
            "feasible": True,
        }

        traces = []
        accepted_steps = 0
        boundary_pullbacks = 0
        grad_seconds_total = 0.0

        # --------------------------------------------------------------
        # Frozen ten-step optimisation.
        # --------------------------------------------------------------
        for step in range(1, ATTACK_STEPS + 1):
            step_t0 = time.perf_counter()
            reference_E_before = float(current_eval["E"])
            reference_score_before = float(current_eval["score"])

            grad_state = gradient_step(
                attack_model,
                current_x,
                altered_mask,
                device,
            )
            grad_seconds_total += grad_state["gradient_seconds"]

            attack_ref_E_gap = abs(
                grad_state["attack_E"] - float(current_eval["E"])
            )
            if attack_ref_E_gap > ATTACK_REFERENCE_E_PARITY_ATOL:
                raise RuntimeError(
                    "Modified attack instance E diverged from authoritative "
                    f"reference at step {step}: gap={attack_ref_E_gap}"
                )

            proposal_x = propose_sign_step(
                current_x,
                clean_x,
                grad_state["grad"],
            )

            pullback = feasibility_pullback(
                evaluator,
                current_x,
                current_eval,
                proposal_x,
                clean_x,
            )

            candidate_x = pullback["candidate_x"]
            candidate_eval = pullback["candidate_eval"]
            candidate_linf = physical_linf(clean_x, candidate_x)

            improvement = bool(
                candidate_eval["E"]
                <= float(current_eval["E"]) + OBJECTIVE_TOL
            )

            if improvement:
                current_x = candidate_x.detach()
                current_eval = {
                    "score": candidate_eval["score"],
                    "E": candidate_eval["E"],
                    "feasible": True,
                }
                accepted_steps += 1

            if pullback["boundary_pullback"]:
                boundary_pullbacks += 1

            current_audit = physical_bounds_audit(clean_x, current_x)

            step_seconds = time.perf_counter() - step_t0
            trace = {
                "step": step,
                "attack_E_at_current": grad_state["attack_E"],
                "reference_E_at_current_before": reference_E_before,
                "reference_score_at_current_before": reference_score_before,
                "attack_reference_E_abs_gap": attack_ref_E_gap,
                "direct_proposal_score": pullback["direct_proposal_score"],
                "direct_proposal_E": pullback["direct_proposal_E"],
                "classification_pullback": bool(pullback["boundary_pullback"]),
                "bisection_evals": int(pullback["bisection_evals"]),
                "accepted_lambda": float(pullback["accepted_lambda"]),
                "candidate_score": float(candidate_eval["score"]),
                "candidate_E": float(candidate_eval["E"]),
                "candidate_physical_linf": float(candidate_linf),
                "objective_nonincrease": improvement,
                "step_accepted": improvement,
                "current_score_after": float(current_eval["score"]),
                "current_E_after": float(current_eval["E"]),
                "current_physical_linf_after": float(current_audit["physical_linf"]),
                "gradient_seconds": grad_state["gradient_seconds"],
                "step_wall_seconds": float(step_seconds),
                "gradient_peak_allocated_gib": grad_state[
                    "gradient_peak_allocated_gib"
                ],
                "gradient_peak_reserved_gib": grad_state[
                    "gradient_peak_reserved_gib"
                ],
                "attention_modules_observed": grad_state[
                    "attention_modules_observed"
                ],
                "total_query_chunks": grad_state["total_query_chunks"],
                "split_attention_modules": grad_state[
                    "split_attention_modules"
                ],
                "max_chunks_one_attention": grad_state[
                    "max_chunks_one_attention"
                ],
            }
            traces.append(trace)

            print(
                f"    step {step:02d}/{ATTACK_STEPS} | "
                f"E={trace['current_E_after']:.6f} | "
                f"score={trace['current_score_after']:.6f} | "
                f"linf={trace['current_physical_linf_after']:.8f} | "
                f"accepted={int(improvement)} | "
                f"pullback={int(pullback['boundary_pullback'])} "
                f"lambda={pullback['accepted_lambda']:.5f} | "
                f"{step_seconds:.1f}s",
                flush=True,
            )

            del grad_state, proposal_x, candidate_x
            gc.collect()

        # --------------------------------------------------------------
        # Final authoritative evaluation.
        # --------------------------------------------------------------
        final_eval = evaluator.evaluate(current_x, return_map=True)
        if not final_eval["feasible"]:
            raise RuntimeError("Final adversarial image broke frozen classification")

        final_metrics = localisation_values_np(final_eval["map"], union_mask_np)
        final_audit = physical_bounds_audit(clean_x, current_x)

        # The final reference call should agree with the last accepted state.
        if abs(final_eval["E"] - float(current_eval["E"])) > 2e-6:
            raise RuntimeError("Final authoritative E disagrees with final attack state")

        # No effect-size threshold is imposed. Non-increase is an optimiser
        # integrity condition only.
        if final_metrics["E"] > clean_metrics["E"] + OBJECTIVE_TOL + 2e-6:
            raise RuntimeError("Final E increased above clean baseline")

        delta_E = clean_metrics["E"] - final_metrics["E"]
        rel_E = (
            delta_E / clean_metrics["E"]
            if clean_metrics["E"] > 0
            else float("nan")
        )
        delta_mu = clean_metrics["mu_w"] - final_metrics["mu_w"]
        rel_mu = (
            delta_mu / clean_metrics["mu_w"]
            if clean_metrics["mu_w"] > 0
            else float("nan")
        )
        delta_PG = clean_metrics["PG"] - final_metrics["PG"]

        trace_df = pd.DataFrame(traces)
        trace_path = image_dir / "trace.csv"
        atomic_write_csv(trace_path, trace_df)

        npz_path = image_dir / "adversarial_result.npz"
        adv_x_cpu = (
            current_x[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        atomic_save_npz(
            npz_path,
            adv_x_model=adv_x_cpu,
            clean_anomaly_map=clean_eval["map"].astype(np.float32, copy=False),
            adv_anomaly_map=final_eval["map"].astype(np.float32, copy=False),
            altered_union_mask=union_mask_np.astype(np.uint8, copy=False),
        )

        result = {
            "status": "PASS",
            "pilot_order": order,
            "pilot_role": str(selection_row["pilot_role"]),
            "eval_split": str(selection_row["eval_split"]),
            "variant": str(selection_row["variant"]),
            "family_display": family_display(str(selection_row["variant"])),
            "hardware_source": str(selection_row["hardware_source"]),
            "file_stem": str(selection_row["file_stem"]),
            "image_path": image_path,
            "cache_path": str(selection_row["cache_path"]),
            "cache_sha256": actual_cache_sha,
            "native_height": H,
            "native_width": W,
            "native_pixels": int(H * W),
            "clipped_gt_rectangles": int(clipped_count),
            "protocol_sha256": protocol_sha,
            "threshold": float(threshold),
            "clean": {
                "score": float(clean_eval["score"]),
                "score_margin": float(clean_eval["score"] - threshold),
                "A": clean_metrics["A"],
                "E": clean_metrics["E"],
                "mu_w": clean_metrics["mu_w"],
                "PG": clean_metrics["PG"],
                "stage4_score_abs_error": float(score_abs_error),
                "stage4_E_abs_error": float(clean_E_abs_error),
            },
            "adversarial": {
                "score": float(final_eval["score"]),
                "score_margin": float(final_eval["score"] - threshold),
                "classification_preserved": True,
                "A": final_metrics["A"],
                "E": final_metrics["E"],
                "mu_w": final_metrics["mu_w"],
                "PG": final_metrics["PG"],
                **final_audit,
            },
            "degradation": {
                "delta_E": float(delta_E),
                "relative_E": float(rel_E),
                "delta_mu_w": float(delta_mu),
                "relative_mu_w": float(rel_mu),
                "delta_PG": float(delta_PG),
            },
            "optimisation": {
                "steps": ATTACK_STEPS,
                "accepted_steps": int(accepted_steps),
                "boundary_pullbacks": int(boundary_pullbacks),
                "reference_forwards_total": int(evaluator.forward_count),
                "gradient_seconds_total": float(grad_seconds_total),
                "reference_seconds_total": float(evaluator.total_seconds),
                "max_gradient_peak_allocated_gib": float(
                    trace_df["gradient_peak_allocated_gib"].max()
                ),
                "max_gradient_peak_reserved_gib": float(
                    trace_df["gradient_peak_reserved_gib"].max()
                ),
                "min_split_attention_modules": int(
                    trace_df["split_attention_modules"].min()
                ),
                "max_query_chunks_one_attention": int(
                    trace_df["max_chunks_one_attention"].max()
                ),
            },
            "memory_route": route_metadata,
            "authoritative_representation": {
                "npz_key": "adv_x_model",
                "dtype": "float32",
                "shape_chw": list(adv_x_cpu.shape),
                "relation_to_physical": "z = x * 256/255",
                "requantised": False,
                "re_jpeg": False,
                "policy_c_rerun": False,
            },
            "timing": {
                "image_wall_seconds_before_artifact_hashing": float(
                    time.perf_counter() - t_image0
                )
            },
        }

        result_path = image_dir / "result.json"
        atomic_write_json(result_path, result)

        artifact_sha = {
            "result.json": sha256_file(result_path),
            "trace.csv": sha256_file(trace_path),
            "adversarial_result.npz": sha256_file(npz_path),
        }
        marker = {
            "status": "COMPLETE",
            "protocol_sha256": protocol_sha,
            "image_path": image_path,
            "artifact_sha256": artifact_sha,
        }
        atomic_write_json(completion_marker_path(image_dir), marker)

        return {
            "status": "PASS",
            "image_dir": str(image_dir),
            "result": result,
        }

    except Exception as exc:
        failure = {
            "status": "FAILED_INCOMPLETE",
            "pilot_order": order,
            "image_path": image_path,
            "protocol_sha256": protocol_sha,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        try:
            atomic_write_json(image_dir / "failure.json", failure)
        finally:
            # Deliberately do not create COMPLETE.json.
            pass
        raise


def load_completed_result(
    run_tag: str,
    row: pd.Series,
    protocol_sha: str,
) -> Tuple[dict, pd.DataFrame, Path]:
    image_dir = image_output_dir(
        run_tag,
        int(row["pilot_order"]),
        str(row["image_path"]),
    )
    if not validate_completion_marker(
        image_dir,
        protocol_sha,
        str(row["image_path"]),
    ):
        raise RuntimeError(
            f"Pilot image is not hash-valid COMPLETE: {row['image_path']}"
        )
    result = json.loads((image_dir / "result.json").read_text())
    trace = pd.read_csv(image_dir / "trace.csv", keep_default_na=False)
    return result, trace, image_dir
