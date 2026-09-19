#!/usr/bin/env python3
"""Memory-efficient differentiable TruFor route for native-resolution attacks.

Scientific intent
-----------------
This module does NOT resize, crop, tile, re-JPEG, quantise, retrain, or alter
TruFor weights. It changes only the autograd execution graph used to obtain
input gradients:

1. all model parameters remain frozen;
2. Noiseprint++ / DnCNN is recomputed in four checkpointed sequential segments;
3. every SegFormer MLP submodule is activation-checkpointed;
4. every SegFormer self-attention module keeps the full native-image K/V context
   but evaluates independent QUERY rows in exact chunks;
5. final authoritative evaluation must use a separate, unmodified TruFor model.

The query-chunked attention computes exactly the same mathematical operation as
the upstream Attention.forward, modulo normal floating-point summation/order
effects. Stage 15 validates map/score/objective/input-gradient agreement before
the route is allowed on the large-image VRAM pilot.

This follows the memory strategy described by the project reviewer and is
implemented against the frozen upstream TruFor commit already enforced by
trufor_common.verify_trufor_provenance().
"""

from __future__ import annotations

import math
import types
from typing import Dict, Iterable, List, Tuple

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from trufor_common import TRUFOR_CODE

# Make the frozen upstream TruFor source importable before importing `lib`.
_trufor_code = str(TRUFOR_CODE)
if _trufor_code not in sys.path:
    sys.path.insert(0, _trufor_code)

from lib.models.cmx.layer_utils import weighted_statistics_pooling

# Reviewer's successful B3 route used this order of magnitude.
DEFAULT_MAX_SCORE_ELEMS = 400_000_000
DEFAULT_DNCNN_SEGMENTS = 4

BLOCK_STACK_NAMES = (
    "block1", "block2", "block3", "block4",
    "extra_block1", "extra_block2", "extra_block3", "extra_block4",
)


def require_default_allocator() -> None:
    """Reject allocator experiments for this route.

    On IMTA134, expandable_segments made even torch.empty(1, device='cuda')
    fail with a CUDA driver unknown error. This route is intentionally tested
    only with the healthy default allocator.
    """
    import os
    bad = {}
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        value = os.environ.get(name, "").strip()
        if value:
            bad[name] = value
    if bad:
        raise RuntimeError(
            "Memory-efficient TruFor route must be tested with the default CUDA "
            f"allocator on IMTA134. Unset allocator variables first: {bad}"
        )


def configure_reproducible_float32() -> None:
    """Disable TF32 for validation/evaluation comparability."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()


def disable_inplace_activations(model: nn.Module) -> int:
    """Disable in-place activations needed by checkpoint recomputation.

    ReLU/LeakyReLU(inplace=False) is mathematically identical to inplace=True;
    only storage mutation semantics change.
    """
    changed = 0
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)) and getattr(module, "inplace", False):
            module.inplace = False
            changed += 1
    return changed


def _checkpointed_sequential_all_segments(
    seq: nn.Sequential,
    x: torch.Tensor,
    segments: int,
) -> torch.Tensor:
    """Checkpoint every segment of an nn.Sequential, including the final one."""
    layers = list(seq.children())
    if not layers:
        return x
    segments = max(1, min(int(segments), len(layers)))

    # Nearly equal contiguous partitions.
    bounds = [round(i * len(layers) / segments) for i in range(segments + 1)]
    bounds[0] = 0
    bounds[-1] = len(layers)

    for start, end in zip(bounds[:-1], bounds[1:]):
        if end <= start:
            continue

        def run_segment(inp, _start=start, _end=end):
            out = inp
            for layer in layers[_start:_end]:
                out = layer(out)
            return out

        if torch.is_grad_enabled() and x.requires_grad:
            x = checkpoint(run_segment, x, use_reentrant=False)
        else:
            x = run_segment(x)
    return x


def _make_checkpointed_mlp_forward(original_forward):
    def wrapped(self, x, H, W):
        if torch.is_grad_enabled() and x.requires_grad:
            def fn(inp):
                return original_forward(inp, H, W)
            return checkpoint(fn, x, use_reentrant=False)
        return original_forward(x, H, W)
    return wrapped


def _make_chunked_attention_forward(max_score_elems: int):
    """Return an Attention.forward replacement with exact query chunking."""

    def forward(self, x, H, W):
        B, N, C = x.shape
        if C % self.num_heads != 0:
            raise RuntimeError(
                f"Attention dim/head mismatch: C={C}, heads={self.num_heads}"
            )

        head_dim = C // self.num_heads
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]
        k_len = int(k.shape[-2])

        # Score tensor for a q-chunk is [B, heads, q_chunk, k_len].
        denom = max(1, int(B) * int(self.num_heads) * k_len)
        q_chunk = max(1, int(max_score_elems) // denom)
        q_chunk = min(int(N), q_chunk)
        n_chunks = int(math.ceil(N / q_chunk))

        # Diagnostics are assigned, not accumulated, so checkpoint recomputation
        # during backward cannot double-count them.
        self._trufor_last_q_len = int(N)
        self._trufor_last_k_len = int(k_len)
        self._trufor_last_q_chunk = int(q_chunk)
        self._trufor_last_chunk_count = int(n_chunks)
        self._trufor_last_max_score_elems = int(
            B * self.num_heads * q_chunk * k_len
        )

        parts = []
        kt = k.transpose(-2, -1)

        for start in range(0, N, q_chunk):
            qc = q[:, :, start : start + q_chunk, :]

            def chunk_fn(q_part, k_full, v_full):
                attn = (q_part @ k_full.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                return attn @ v_full

            if torch.is_grad_enabled() and qc.requires_grad:
                part = checkpoint(
                    chunk_fn,
                    qc,
                    k,
                    v,
                    use_reentrant=False,
                )
            else:
                part = chunk_fn(qc, k, v)
            parts.append(part)

        y = torch.cat(parts, dim=2)
        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    return forward


def enable_memory_efficient_attack_route(
    model: nn.Module,
    *,
    max_score_elems: int = DEFAULT_MAX_SCORE_ELEMS,
    dncnn_segments: int = DEFAULT_DNCNN_SEGMENTS,
) -> Dict[str, object]:
    """Patch one loaded TruFor instance in memory for differentiable attack use."""
    if getattr(model, "_trufor_memroute_enabled", False):
        raise RuntimeError("Memory-efficient route is already enabled on this model")

    freeze_all_parameters(model)
    inplace_changed = disable_inplace_activations(model)

    backbone = model.backbone
    missing = [name for name in BLOCK_STACK_NAMES if not hasattr(backbone, name)]
    if missing:
        raise RuntimeError(f"Unexpected TruFor backbone; missing stacks: {missing}")

    attention_count = 0
    mlp_count = 0
    stack_depths = {}

    for stack_name in BLOCK_STACK_NAMES:
        stack = getattr(backbone, stack_name)
        stack_depths[stack_name] = len(stack)
        for block_idx, block in enumerate(stack):
            if not hasattr(block, "attn") or not hasattr(block, "mlp"):
                raise RuntimeError(
                    f"Unexpected block structure: {stack_name}[{block_idx}]"
                )

            attn = block.attn
            if getattr(attn, "_trufor_chunked", False):
                raise RuntimeError(
                    f"Attention already patched: {stack_name}[{block_idx}]"
                )
            attn.forward = types.MethodType(
                _make_chunked_attention_forward(int(max_score_elems)),
                attn,
            )
            attn._trufor_chunked = True
            attn._trufor_stack_name = stack_name
            attn._trufor_block_index = int(block_idx)
            attention_count += 1

            mlp = block.mlp
            if getattr(mlp, "_trufor_checkpointed", False):
                raise RuntimeError(
                    f"MLP already patched: {stack_name}[{block_idx}]"
                )
            original_forward = mlp.forward
            mlp.forward = types.MethodType(
                _make_checkpointed_mlp_forward(original_forward),
                mlp,
            )
            mlp._trufor_checkpointed = True
            mlp_count += 1

    # mit_b2 has 16 blocks per modality = 32 Attention/MLP modules.
    if attention_count != 32 or mlp_count != 32:
        raise RuntimeError(
            "Unexpected mit_b2 block count: "
            f"attention={attention_count}, mlp={mlp_count}, depths={stack_depths}"
        )

    model._trufor_memroute_enabled = True
    model._trufor_memroute_max_score_elems = int(max_score_elems)
    model._trufor_memroute_dncnn_segments = int(dncnn_segments)

    return {
        "route": "encoder_ckpt_exact_query_chunked_attention",
        "attention_modules_chunked": attention_count,
        "mlp_modules_checkpointed": mlp_count,
        "dncnn_segments": int(dncnn_segments),
        "max_score_elems": int(max_score_elems),
        "inplace_activations_disabled": int(inplace_changed),
        "stack_depths": stack_depths,
    }


def _forward_from_modalities(model, rgb_branch, modal_x):
    """Same arithmetic as frozen TruFor encode/decode, without upstream no_grad."""
    orisize = rgb_branch.shape if rgb_branch is not None else modal_x.shape

    features = model.backbone(rgb_branch, modal_x)

    out = model.decode_head(features)
    out = F.interpolate(
        out,
        size=orisize[2:],
        mode="bilinear",
        align_corners=False,
    )

    if model.decode_head_conf is None:
        raise RuntimeError("TruFor confidence head is missing")
    conf = model.decode_head_conf(features)
    conf = F.interpolate(
        conf,
        size=orisize[2:],
        mode="bilinear",
        align_corners=False,
    )

    if model.detection is None or model.conf_detection != "confpool":
        raise RuntimeError("Unexpected TruFor detection configuration")

    f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
    f2 = weighted_statistics_pooling(
        out[:, 1:2, :, :] - out[:, 0:1, :, :],
        F.logsigmoid(conf),
    ).view(out.shape[0], -1)
    det = model.detection(torch.cat((f1, f2), dim=-1))
    return out, conf, det


def plain_differentiable_forward(model: nn.Module, rgb: torch.Tensor):
    """Existing ordinary differentiable route used by the earlier pilot."""
    modal_x = None
    if "NP++" in model.mods:
        modal_x = model.dncnn(rgb)
        if model.np_out_ch == 1:
            modal_x = torch.tile(modal_x, (3, 1, 1))
        elif model.np_out_ch != 3:
            raise RuntimeError(f"Unexpected NP++ output channels: {model.np_out_ch}")

    rgb_branch = rgb
    if "RGB" not in model.mods:
        rgb_branch = None
    elif model.prepro is not None:
        rgb_branch = model.prepro(rgb_branch)

    return _forward_from_modalities(model, rgb_branch, modal_x)


def memory_efficient_differentiable_forward(model: nn.Module, rgb: torch.Tensor):
    """Differentiable full-native-image route with exact recomputation/chunking."""
    if not getattr(model, "_trufor_memroute_enabled", False):
        raise RuntimeError("Call enable_memory_efficient_attack_route(model) first")

    modal_x = None
    if "NP++" in model.mods:
        modal_x = _checkpointed_sequential_all_segments(
            model.dncnn,
            rgb,
            int(model._trufor_memroute_dncnn_segments),
        )
        if model.np_out_ch == 1:
            modal_x = torch.tile(modal_x, (3, 1, 1))
        elif model.np_out_ch != 3:
            raise RuntimeError(f"Unexpected NP++ output channels: {model.np_out_ch}")

    rgb_branch = rgb
    if "RGB" not in model.mods:
        rgb_branch = None
    elif model.prepro is not None:
        rgb_branch = model.prepro(rgb_branch)

    return _forward_from_modalities(model, rgb_branch, modal_x)


def anomaly_from_logits(out: torch.Tensor) -> torch.Tensor:
    return F.softmax(out, dim=1)[:, 1, :, :]


def score_from_det(det: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(det).reshape(-1)


def localisation_E(anomaly: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = anomaly.sum(dim=(1, 2))
    numer = (anomaly * mask).sum(dim=(1, 2))
    if torch.any(denom <= 0):
        raise RuntimeError("Non-positive anomaly-map mass")
    return numer / denom


def attention_chunk_stats(model: nn.Module) -> Dict[str, object]:
    rows = []
    for stack_name in BLOCK_STACK_NAMES:
        stack = getattr(model.backbone, stack_name)
        for i, block in enumerate(stack):
            attn = block.attn
            if not hasattr(attn, "_trufor_last_chunk_count"):
                continue
            rows.append(
                {
                    "stack": stack_name,
                    "block": int(i),
                    "q_len": int(attn._trufor_last_q_len),
                    "k_len": int(attn._trufor_last_k_len),
                    "q_chunk": int(attn._trufor_last_q_chunk),
                    "chunks": int(attn._trufor_last_chunk_count),
                    "max_score_elems_actual": int(attn._trufor_last_max_score_elems),
                }
            )
    if not rows:
        return {
            "modules_observed": 0,
            "total_query_chunks": 0,
            "max_chunks_in_one_attention": 0,
            "max_score_elems_actual": 0,
            "rows": [],
        }
    return {
        "modules_observed": len(rows),
        "total_query_chunks": int(sum(r["chunks"] for r in rows)),
        "max_chunks_in_one_attention": int(max(r["chunks"] for r in rows)),
        "max_score_elems_actual": int(max(r["max_score_elems_actual"] for r in rows)),
        "rows": rows,
    }


def gradient_similarity(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Compare two input gradients on CPU in float64."""
    a = a.detach().cpu().double().reshape(-1)
    b = b.detach().cpu().double().reshape(-1)
    if a.numel() != b.numel():
        raise RuntimeError("Gradient shape mismatch")

    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float((torch.dot(a, b) / denom).item()) if float(denom) > 0 else float("nan")

    sign_all = float((torch.sign(a) == torch.sign(b)).double().mean().item())

    # Also report sign agreement away from near-zero numerical noise.
    active = torch.maximum(a.abs(), b.abs()) > 1e-12
    if bool(active.any()):
        sign_active = float(
            (torch.sign(a[active]) == torch.sign(b[active])).double().mean().item()
        )
        active_fraction = float(active.double().mean().item())
    else:
        sign_active = float("nan")
        active_fraction = 0.0

    return {
        "gradient_cosine": cosine,
        "gradient_sign_agreement_all": sign_all,
        "gradient_sign_agreement_active": sign_active,
        "gradient_active_fraction": active_fraction,
        "gradient_max_abs_diff": float((a - b).abs().max().item()),
        "gradient_mean_abs_diff": float((a - b).abs().mean().item()),
    }
