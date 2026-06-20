"""
Parallax attention integration for nanochat.

Parallax = softmax attention + a second query ``r``:
    s1 = q·k · qk_scale ; s2 = r·k ; p1 = softmax(s1) ; p2 = p1·s2
    out = O1/d1 · (1 + d2/d1) − O2/d1      (O1=p1·v, O2=p2·v, d1=Σp1, d2=Σp2)

This module mirrors ``nanochat.flash_attention``'s ``flash_attn_func`` /
``flash_attn_with_kvcache`` (plus the extra ``r`` arg) so it can be dropped into
``CausalSelfAttention`` behind a config flag.

torch.compile: the TRAINING/prefill forward+backward Triton kernels are wrapped as
torch custom ops (``nanochat::parallax_attn_fwd`` / ``parallax_attn_bwd``) with fake
(meta) impls + ``register_autograd``. The compiled model graph then treats Parallax as
an opaque op — no graph break, correct shape propagation. The autotuned ``@triton.jit``
kernels run eagerly inside the op body (black-box to Inductor).

Inference (decode/prefill) is NEVER compiled (nanochat runs the uncompiled ``orig_model``
through ``Engine`` under ``inference_mode``), so it calls the raw forward-only kernels
directly — crucially NOT the autograd ``parallax_func``, whose ``save_for_backward`` would
raise under ``inference_mode``.

Layout note: nanochat uses BTHD ``(B, T, H, D)``. The Triton training kernel uses
``(B, H, L, D)`` folded to ``(B*H, L, D)``; we transpose then reshape. The decode kernels
use BTHD natively, matching nanochat's KV cache.
"""

import os
from typing import Tuple

import torch

from nanochat.parallax import decode_available as _CUTE_AVAILABLE
from nanochat.parallax.triton.parallax_fwd import parallax_fwd as _raw_fwd
from nanochat.parallax.triton.parallax_bwd import parallax_bwd as _raw_bwd
from nanochat.parallax.triton.parallax_func import parallax_func as _parallax_func  # escape hatch
from nanochat.parallax.triton.parallax_decode import parallax_decode as _triton_decode

if _CUTE_AVAILABLE:
    from nanochat.parallax.cute import parallax_attn_with_kvcache as _cute_decode
else:  # pragma: no cover - depends on hardware/install
    _cute_decode = None

# Escape hatch: bypass the custom op and call the upstream autograd.Function directly.
# Relies on the tolerated graph break (model is compiled with fullgraph=False). Used to
# cross-check that the custom-op path produces identical numerics.
_DIRECT = os.environ.get("NANOCHAT_PARALLAX_DIRECT", "0") == "1"
# Decode kernel selection: "auto" (cute on SM90 else triton), "cute", or "triton".
_DECODE_IMPL = os.environ.get("NANOCHAT_PARALLAX_DECODE", "auto").lower()


# =============================================================================
# Window mapping: nanochat (left, right=0) -> parallax window_size_left
# =============================================================================
def _map_window_left(window_size, t_kv: int) -> int:
    """nanochat passes FA-style ``(left, 0)`` (SDPA includes ``left+1`` keys: cols
    ``[i-left, i]``). Parallax ``window_size_left=W`` keeps exactly ``W`` keys including
    the diagonal. So ``W = left + 1`` preserves the attended set. Full context
    (``left < 0`` or ``left >= t_kv``) maps to ``-1`` (disabled)."""
    left = window_size[0] if isinstance(window_size, (tuple, list)) else window_size
    if left is None or left < 0 or left >= t_kv:
        return -1
    return int(left) + 1


# =============================================================================
# torch custom ops for the (compiled) training/prefill path
# =============================================================================
# Fixed-arity Tuple returns (NOT Tensor[]): a list output is treated as a single
# TensorList, so autograd would hand the backward one *list* of grads; a tuple gives
# distinct outputs with one grad each.
_FwdOut = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_BwdOut = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@torch.library.custom_op("nanochat::parallax_attn_fwd", mutates_args=())
def _plx_fwd_op(q: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                qk_scale: float, n_rep: int, window_size_left: int) -> _FwdOut:
    """Opaque wrapper over the raw Triton forward. Inputs/outputs are folded
    ``(B*H, L, D)``. Returns ``(o, barv, d1, bart, m)`` (the aux tensors are saved for
    the backward)."""
    o, barv, d1, bart, m = _raw_fwd(q, r, k, v, qk_scale, n_rep, window_size_left)
    # Force contiguous so the real outputs match the (contiguous) register_fake under
    # torch.compile regardless of any view-y strides the kernel might return.
    return o.contiguous(), barv.contiguous(), d1.contiguous(), bart.contiguous(), m.contiguous()


@_plx_fwd_op.register_fake
def _(q, r, k, v, qk_scale, n_rep, window_size_left):
    bhq, lq, d = q.shape
    o = q.new_empty((bhq, lq, d))
    barv = q.new_empty((bhq, lq, d))
    d1 = q.new_empty((bhq, lq, 1), dtype=torch.float32)
    bart = q.new_empty((bhq, lq, 1), dtype=torch.float32)
    m = q.new_empty((bhq, lq, 1), dtype=torch.float32)
    return o, barv, d1, bart, m


@torch.library.custom_op("nanochat::parallax_attn_bwd", mutates_args=())
def _plx_bwd_op(q: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                o: torch.Tensor, barv: torch.Tensor, d1: torch.Tensor, bart: torch.Tensor,
                m: torch.Tensor, grad_o: torch.Tensor, qk_scale: float, n_rep: int,
                window_size_left: int) -> _BwdOut:
    dq, dr, dk, dv = _raw_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o,
                              qk_scale, n_rep, window_size_left)
    # Force contiguous to match the (contiguous) register_fake under torch.compile.
    return dq.contiguous(), dr.contiguous(), dk.contiguous(), dv.contiguous()


@_plx_bwd_op.register_fake
def _(q, r, k, v, o, barv, d1, bart, m, grad_o, qk_scale, n_rep, window_size_left):
    # The kernel emits *contiguous* bf16 grads regardless of input dtype/stride. Use
    # new_empty(shape) (contiguous) — NOT empty_like, which would inherit a non-contiguous
    # stride from a transposed/view input and mismatch the real output under torch.compile.
    dq = q.new_empty(q.shape, dtype=torch.bfloat16)
    dr = r.new_empty(r.shape, dtype=torch.bfloat16)
    dk = k.new_empty(k.shape, dtype=torch.bfloat16)
    dv = v.new_empty(v.shape, dtype=torch.bfloat16)
    return dq, dr, dk, dv


def _fwd_setup_context(ctx, inputs, output):
    q, r, k, v, qk_scale, n_rep, window_size_left = inputs
    o, barv, d1, bart, m = output
    ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
    ctx.qk_scale = qk_scale
    ctx.n_rep = n_rep
    ctx.window_size_left = window_size_left


def _fwd_backward(ctx, *grads):
    grad_o = grads[0]  # only the gradient w.r.t. `o` is meaningful; aux outputs are unused
    q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
    dq, dr, dk, dv = torch.ops.nanochat.parallax_attn_bwd(
        q, r, k, v, o, barv, d1, bart, m, grad_o,
        ctx.qk_scale, ctx.n_rep, ctx.window_size_left,
    )
    # Cast back to input dtype (defensive: kernel returns bf16; bf16 training is a no-op).
    return (dq.to(q.dtype), dr.to(r.dtype), dk.to(k.dtype), dv.to(v.dtype),
            None, None, None)


torch.library.register_autograd(
    "nanochat::parallax_attn_fwd", _fwd_backward, setup_context=_fwd_setup_context
)


# =============================================================================
# Public API (mirrors nanochat.flash_attention, plus the extra `r`)
# =============================================================================
def parallax_attn_func(q, r, k, v, causal=True, window_size=(-1, 0)):
    """Training/prefill Parallax (the compiled path). BTHD in/out.

    Args:
        q, r: (B, T, H_q, D)   k, v: (B, T, H_kv, D)
        causal: must be True (Parallax training kernel is causal-only).
        window_size: nanochat (left, right=0) tuple.
    Returns:
        (B, T, H_q, D)
    """
    assert causal, "Parallax training kernel is causal-only"
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    n_rep = Hq // Hkv
    qk_scale = float(D ** -0.5)
    wsl = _map_window_left(window_size, T)

    if _DIRECT:
        # (B,T,H,D) -> (B,H,T,D); parallax_func folds + runs autograd internally.
        o = _parallax_func(q.transpose(1, 2), r.transpose(1, 2),
                           k.transpose(1, 2), v.transpose(1, 2), qk_scale, wsl)
        return o.transpose(1, 2)

    # BTHD -> (B,H,T,D) -> folded (B*H, T, D). transpose-then-reshape forces contiguity.
    qf = q.transpose(1, 2).reshape(B * Hq, T, D)
    rf = r.transpose(1, 2).reshape(B * Hq, T, D)
    kf = k.transpose(1, 2).reshape(B * Hkv, T, D)
    vf = v.transpose(1, 2).reshape(B * Hkv, T, D)
    o = torch.ops.nanochat.parallax_attn_fwd(qf, rf, kf, vf, qk_scale, n_rep, wsl)[0]
    return o.reshape(B, Hq, T, D).transpose(1, 2)


def _use_cute(q, D, n_rep) -> bool:
    if _DECODE_IMPL == "triton":
        return False
    if not (_CUTE_AVAILABLE and _cute_decode is not None and torch.cuda.is_available()):
        return False
    if torch.cuda.get_device_capability(q.device)[0] != 9:  # Hopper SM90 only
        return False
    if D not in (64, 128):
        return False
    if n_rep not in (1, 2, 4, 8):  # cute GQA packing constraint
        return False
    return True


def parallax_attn_with_kvcache(q, r, k_cache, v_cache, k=None, v=None,
                               cache_seqlens=None, causal=True, window_size=(-1, 0)):
    """Inference Parallax against a KV cache (NOT compiled). Mirrors
    ``flash_attn_with_kvcache``: inserts new k,v into the cache in place, then routes:
      * T_new == 1 (decode): CuteDSL (SM90, default) or Triton decode kernel.
      * T_new  > 1 (prefill at pos==0): the raw Triton forward kernel (causal).
    ``r`` is computed fresh per step and never cached.
    """
    B, T_new, Hq, D = q.shape
    Hkv = k_cache.shape[2]
    n_rep = Hq // Hkv
    qk_scale = float(D ** -0.5)
    pos = int(cache_seqlens[0].item())

    # Insert new k, v into the cache in place (matches FA3 / the SDPA fallback).
    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new] = k
        v_cache[:, pos:pos + T_new] = v
    end_pos = pos + T_new

    if T_new == 1:
        # ---- decode ----
        wsl = _map_window_left(window_size, end_pos)
        if _use_cute(q, D, n_rep):
            # Full fixed-shape cache + seqused_k => stable launch shape (no per-step
            # recompile); cache is zero-init beyond end_pos (finite-padding contract).
            seqused = (cache_seqlens + T_new).to(torch.int32)
            win = None if wsl < 0 else wsl
            return _cute_decode(q, r, k_cache, v_cache,
                                seqused_k=seqused, window_size=win, scale=qk_scale)
        # Triton fallback: slice valid region (kernel is do_not_specialize on kv len).
        return _triton_decode(q, r, k_cache[:, :end_pos], v_cache[:, :end_pos],
                              qk_scale=qk_scale, window_size_left=wsl, cache_start=None)

    # ---- prefill (pos==0): standard causal over the new chunk ----
    wsl = _map_window_left(window_size, T_new)
    qf = q.transpose(1, 2).reshape(B * Hq, T_new, D)
    rf = r.transpose(1, 2).reshape(B * Hq, T_new, D)
    kf = k_cache[:, :end_pos].transpose(1, 2).reshape(B * Hkv, end_pos, D)
    vf = v_cache[:, :end_pos].transpose(1, 2).reshape(B * Hkv, end_pos, D)
    o = _raw_fwd(qf, rf, kf, vf, qk_scale, n_rep, wsl)[0]  # forward-only, no autograd
    return o.reshape(B, Hq, T_new, D).transpose(1, 2)
