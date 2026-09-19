#!/usr/bin/env python3
"""Corrected memory-efficient differentiable TruFor route.

This module is intentionally narrow. It uses the reviewer-provided implementation
ONLY as a reference to correct bugs in the current ForgedDocDeepID pilot:

- preserve this project's existing TruFor backend semantics; do not import the
  reviewer's protocol-specific TF32/CUDNN policy;
- use nn.Module wrappers rather than monkey-patching forward methods;
- use the reviewer's query-chunk formula, including a minimum query chunk of 1024;
- checkpoint DnCNN in fixed sequential segments;
- disable in-place activations for checkpoint recomputation;
- keep first-order input gradients only.

It does NOT copy the reviewer's attack parameters/objective/protocol.
The ForgedDocDeepID threat model, frozen threshold, Policy-C inputs and
localisation objective remain unchanged.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from trufor_common import TRUFOR_CODE

# Import the frozen upstream TruFor tree before importing `lib`.
_code = str(TRUFOR_CODE)
if _code not in sys.path:
    sys.path.insert(0, _code)

from lib.models.cmx.layer_utils import weighted_statistics_pooling  # noqa: E402


STACK_NAMES = (
    "block1", "block2", "block3", "block4",
    "extra_block1", "extra_block2", "extra_block3", "extra_block4",
)

DEFAULT_MAX_SCORE_ELEMS = 300_000_000
DEFAULT_DNCNN_SEGMENTS = 4


def require_default_allocator() -> None:
    """The expandable allocator is known-bad on IMTA134."""
    bad = {}
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        value = os.environ.get(name, "").strip()
        if value:
            bad[name] = value
    if bad:
        raise RuntimeError(
            "Unset allocator overrides before running this route. "
            f"Found: {bad}"
        )


def backend_state() -> Dict[str, object]:
    """Observe backend state without changing it."""
    return {
        "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn.enabled": bool(torch.backends.cudnn.enabled),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def assert_project_trufor_backend(cfg) -> Dict[str, object]:
    """Assert only the backend choices already made by this project's loader.

    load_trufor_model() applies cfg.CUDNN.{BENCHMARK,DETERMINISTIC,ENABLED}.
    TF32 and matmul-precision values are RECORDED but not modified here.
    """
    st = backend_state()
    expected = {
        "cudnn.benchmark": bool(cfg.CUDNN.BENCHMARK),
        "cudnn.deterministic": bool(cfg.CUDNN.DETERMINISTIC),
        "cudnn.enabled": bool(cfg.CUDNN.ENABLED),
    }
    for key, want in expected.items():
        if st[key] != want:
            raise RuntimeError(
                f"Project TruFor backend drift: {key}={st[key]} expected={want}"
            )
    if st["PYTORCH_ALLOC_CONF"] or st["PYTORCH_CUDA_ALLOC_CONF"]:
        raise RuntimeError("Allocator override unexpectedly active")
    return st


def freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()


def disable_inplace(module: nn.Module) -> int:
    """ReLU(inplace=True)->False changes aliasing only, not computed values."""
    n = 0
    for m in module.modules():
        if getattr(m, "inplace", False):
            m.inplace = False
            n += 1
    return n


class CkptSub(nn.Module):
    """Checkpoint a Block submodule called as m(x, H, W)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, H, W):
        if not torch.is_grad_enabled() or not x.requires_grad:
            return self.inner(x, H, W)
        return checkpoint(self.inner, x, H, W, use_reentrant=False)


class ChunkedAttention(nn.Module):
    """Exact SegFormer attention by independent query-row chunks.

    The softmax axis is the KEY axis, so splitting only the query axis preserves
    the same mathematical attention. This is NOT image tiling.
    """

    def __init__(self, inner: nn.Module, max_score_elems: int):
        super().__init__()
        self.inner = inner
        self.max_score_elems = int(max_score_elems)

        self.last_q_len = 0
        self.last_k_len = 0
        self.last_query_chunk = 0
        self.last_chunk_count = 0
        self.last_max_score_elems_actual = 0

    def forward(self, x, H, W):
        a = self.inner
        B, N, C = x.shape
        nh = int(a.num_heads)
        hd = C // nh

        q = a.q(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)

        if a.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = a.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = a.norm(x_)
            kv = (
                a.kv(x_)
                .reshape(B, -1, 2, nh, hd)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                a.kv(x)
                .reshape(B, -1, 2, nh, hd)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        def chunk_fn(qc, kk, vv):
            att = (qc @ kk.transpose(-2, -1)) * a.scale
            att = att.softmax(dim=-1)
            att = a.attn_drop(att)
            return att @ vv

        nk = int(k.shape[-2])

        # Match the proven reference structure: max(1024, bound/(keys*heads)).
        chunk = max(
            1024,
            int(self.max_score_elems // max(nk * nh, 1)),
        )
        use_ckpt = torch.is_grad_enabled() and x.requires_grad

        if chunk >= N:
            out = (
                checkpoint(chunk_fn, q, k, v, use_reentrant=False)
                if use_ckpt
                else chunk_fn(q, k, v)
            )
            n_chunks = 1
            actual_q_chunk = N
        else:
            parts = []
            for i in range(0, N, chunk):
                qc = q[:, :, i : i + chunk, :]
                parts.append(
                    checkpoint(
                        chunk_fn,
                        qc,
                        k,
                        v,
                        use_reentrant=False,
                    )
                    if use_ckpt
                    else chunk_fn(qc, k, v)
                )
            out = torch.cat(parts, dim=2)
            n_chunks = int(math.ceil(N / chunk))
            actual_q_chunk = chunk

        # Assign rather than accumulate: checkpoint backward recomputation must
        # not double-count diagnostics.
        self.last_q_len = int(N)
        self.last_k_len = int(nk)
        self.last_query_chunk = int(actual_q_chunk)
        self.last_chunk_count = int(n_chunks)
        self.last_max_score_elems_actual = int(
            B * nh * min(actual_q_chunk, N) * nk
        )

        out = out.transpose(1, 2).reshape(B, N, C)
        out = a.proj(out)
        return a.proj_drop(out)


class CkptSequential(nn.Module):
    """Checkpoint an nn.Sequential trunk in fixed contiguous segments."""

    def __init__(self, inner: nn.Sequential, segments: int = 4):
        super().__init__()
        mods = list(inner)
        size = max(1, (len(mods) + int(segments) - 1) // int(segments))
        self.chunks = nn.ModuleList(
            [
                nn.Sequential(*mods[i : i + size])
                for i in range(0, len(mods), size)
            ]
        )

    def forward(self, x):
        if not torch.is_grad_enabled() or not x.requires_grad:
            for c in self.chunks:
                x = c(x)
            return x

        for c in self.chunks:
            x = checkpoint(c, x, use_reentrant=False)
        return x


def apply_memory_efficient_route(
    model: nn.Module,
    *,
    max_score_elems: int = DEFAULT_MAX_SCORE_ELEMS,
    dncnn_segments: int = DEFAULT_DNCNN_SEGMENTS,
) -> Dict[str, object]:
    """Wrap the already-loaded attack model in memory only."""
    if getattr(model, "_forgeddoc_memroute_v2", False):
        raise RuntimeError("Memory-efficient route already applied")

    freeze_all_parameters(model)
    n_inplace = disable_inplace(model)

    wrapped_attn = 0
    wrapped_mlp = 0
    stack_depths = {}

    backbone = model.backbone
    for stack_name in STACK_NAMES:
        stack = getattr(backbone, stack_name, None)
        if stack is None:
            raise RuntimeError(f"Missing expected backbone stack: {stack_name}")
        stack_depths[stack_name] = len(stack)

        for blk in stack:
            if not isinstance(blk.attn, ChunkedAttention):
                blk.attn = ChunkedAttention(
                    blk.attn,
                    max_score_elems=max_score_elems,
                )
                wrapped_attn += 1

            if not isinstance(blk.mlp, CkptSub):
                blk.mlp = CkptSub(blk.mlp)
                wrapped_mlp += 1

    if wrapped_attn != 32 or wrapped_mlp != 32:
        raise RuntimeError(
            "Unexpected MiT-B2 wrapper count: "
            f"attention={wrapped_attn}, mlp={wrapped_mlp}"
        )

    dncnn_wrapped = False
    if not isinstance(model.dncnn, CkptSequential):
        model.dncnn = CkptSequential(
            model.dncnn,
            segments=dncnn_segments,
        )
        dncnn_wrapped = True

    model._forgeddoc_memroute_v2 = True
    model._forgeddoc_max_score_elems = int(max_score_elems)
    model._forgeddoc_dncnn_segments = int(dncnn_segments)
    model.eval()

    return {
        "route": "full_native_ckpt_exact_query_chunk_v2",
        "attention_wrapped": wrapped_attn,
        "mlp_wrapped": wrapped_mlp,
        "dncnn_wrapped": dncnn_wrapped,
        "dncnn_segments_requested": int(dncnn_segments),
        "dncnn_chunks_actual": len(model.dncnn.chunks),
        "inplace_ops_disabled": n_inplace,
        "max_score_elems": int(max_score_elems),
        "stack_depths": stack_depths,
    }


def _forward_from_modalities(model, rgb_branch, modal_x):
    """Same frozen TruFor arithmetic, bypassing only upstream no_grad wrappers."""
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
        raise RuntimeError("TruFor confidence head missing")
    conf = model.decode_head_conf(features)
    conf = F.interpolate(
        conf,
        size=orisize[2:],
        mode="bilinear",
        align_corners=False,
    )

    if model.detection is None or model.conf_detection != "confpool":
        raise RuntimeError("Unexpected TruFor detector configuration")

    f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
    f2 = weighted_statistics_pooling(
        out[:, 1:2] - out[:, 0:1],
        F.logsigmoid(conf),
    ).view(out.shape[0], -1)
    det = model.detection(torch.cat((f1, f2), dim=-1))
    return out, conf, det


def plain_differentiable_forward(model, rgb):
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


def efficient_differentiable_forward(model, rgb):
    if not getattr(model, "_forgeddoc_memroute_v2", False):
        raise RuntimeError("Memory-efficient route has not been applied")

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


def anomaly_from_logits(out):
    return F.softmax(out, dim=1)[:, 1]


def score_from_det(det):
    return torch.sigmoid(det).reshape(-1)


def localisation_E(amap, mask):
    denom = amap.sum(dim=(1, 2))
    numer = (amap * mask).sum(dim=(1, 2))
    if torch.any(denom <= 0):
        raise RuntimeError("Non-positive anomaly map mass")
    return numer / denom


def chunk_stats(model) -> Dict[str, object]:
    rows = []
    for stack_name in STACK_NAMES:
        stack = getattr(model.backbone, stack_name)
        for i, blk in enumerate(stack):
            attn = blk.attn
            if not isinstance(attn, ChunkedAttention):
                continue
            rows.append(
                {
                    "stack": stack_name,
                    "block": int(i),
                    "q_len": int(attn.last_q_len),
                    "k_len": int(attn.last_k_len),
                    "query_chunk": int(attn.last_query_chunk),
                    "chunks": int(attn.last_chunk_count),
                    "max_score_elems_actual": int(
                        attn.last_max_score_elems_actual
                    ),
                }
            )

    observed = [r for r in rows if r["q_len"] > 0]
    return {
        "modules_observed": len(observed),
        "total_query_chunks": int(sum(r["chunks"] for r in observed)),
        "max_chunks_in_one_attention": (
            int(max(r["chunks"] for r in observed)) if observed else 0
        ),
        "n_attention_modules_actually_split": int(
            sum(r["chunks"] > 1 for r in observed)
        ),
        "max_score_elems_actual": (
            int(max(r["max_score_elems_actual"] for r in observed))
            if observed
            else 0
        ),
        "rows": observed,
    }


def gradient_similarity(a, b) -> Dict[str, float]:
    a = a.detach().cpu().double().reshape(-1)
    b = b.detach().cpu().double().reshape(-1)

    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = (
        float((torch.dot(a, b) / denom).item())
        if float(denom) > 0
        else float("nan")
    )
    sign = float(
        (torch.sign(a) == torch.sign(b)).double().mean().item()
    )
    return {
        "gradient_cosine": cosine,
        "gradient_sign_agreement": sign,
        "gradient_max_abs_diff": float((a - b).abs().max().item()),
        "gradient_mean_abs_diff": float((a - b).abs().mean().item()),
    }
