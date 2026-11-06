# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fused CUDA kernels of the quantized delta dispatch formats (Triton).

Same arithmetic and row layout as the torch reference in ``delta_dispatch.py``
(``encode_delta_reference`` / ``decode_delta_reference``); one program per
(token, group of ``QUANT_GROUP_SIZE`` values).

* ``encode_delta`` (Attention): ``d = h' - S``, the group's ``lo`` / ``step``
  rounded to fp16 (IEEE divisions, like torch), ``q = round_half_even((d -
  lo) / step)`` clamped, the
  packed values and the scales written into the row, and ``S += q * step +
  lo`` in place: one launch per MoE layer.
* ``decode_delta`` (FFN): ``S += q * step + lo`` and ``h_out = S`` in the
  model dtype: one launch per MoE layer.

Both sides dequantize with the same ``_dequantize`` (an explicit fused
multiply-add), so the state the FFN rank rebuilds is bit-identical to the one
the Attention side keeps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from afd_plugin.delta_dispatch import (
    BITS_PER_BYTE,
    FORMAT_BITS,
    MIN_GROUP_RANGE,
    QUANT_GROUP_SIZE,
    row_bytes,
)

HALF_GROUP = QUANT_GROUP_SIZE // 2
NIBBLE_MASK = 0xF


@triton.jit
def _dequantize(q, lo, step):
    return tl.fma(q, step, lo)


@triton.jit
def _step(lo, hi, min_range, levels: tl.constexpr):
    """(max - lo) / levels with an IEEE division, rounded to fp16."""
    span = tl.maximum(hi - lo, min_range)
    step = tl.math.div_rn(span, tl.full(span.shape, levels, tl.float32))
    return step.to(tl.float16).to(tl.float32)


@triton.jit
def _round_half_even(x):
    """torch.round: ties to even. Exact for the small values quantized here
    (|x| <= 2^bits), and runs in Triton's interpreter too."""
    r = tl.floor(x + 0.5)
    tie = (r - x) == 0.5
    return tl.where(tie & ((r % 2) != 0), r - 1.0, r)


@triton.jit
def _quantize(d, lo, step, levels: tl.constexpr):
    safe = tl.where(step > 0, step, 1.0)
    q = _round_half_even(tl.math.div_rn(d - lo, safe))
    q = tl.minimum(tl.maximum(q, 0.0), levels)
    return tl.where(step > 0, q, 0.0)


@triton.jit
def _encode_kernel(
    h_ptr,
    h_stride,
    s_ptr,
    s_stride,
    row_ptr,
    row_stride,
    scale_ptr,
    scale_stride,
    min_range,
    bits: tl.constexpr,
    group: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.program_id(1)
    levels: tl.constexpr = (1 << bits) - 1
    if bits == 4:
        half: tl.constexpr = group // 2
        k = tl.arange(0, half)
        base = g * group
        h_e = tl.load(h_ptr + t * h_stride + base + 2 * k).to(tl.float32)
        h_o = tl.load(h_ptr + t * h_stride + base + 2 * k + 1).to(tl.float32)
        s_e = tl.load(s_ptr + t * s_stride + base + 2 * k)
        s_o = tl.load(s_ptr + t * s_stride + base + 2 * k + 1)
        d_e = h_e - s_e
        d_o = h_o - s_o
        lo = tl.minimum(tl.min(d_e, 0), tl.min(d_o, 0))
        hi = tl.maximum(tl.max(d_e, 0), tl.max(d_o, 0))
        step = _step(lo, hi, min_range, levels)
        lo = lo.to(tl.float16).to(tl.float32)
        q_e = _quantize(d_e, lo, step, levels)
        q_o = _quantize(d_o, lo, step, levels)
        tl.store(s_ptr + t * s_stride + base + 2 * k, s_e + _dequantize(q_e, lo, step))
        tl.store(
            s_ptr + t * s_stride + base + 2 * k + 1, s_o + _dequantize(q_o, lo, step)
        )
        packed = q_e.to(tl.uint8) | (q_o.to(tl.uint8) << 4)
        tl.store(row_ptr + t * row_stride + g * half + k, packed)
    else:
        k = tl.arange(0, group)
        offs = g * group + k
        h = tl.load(h_ptr + t * h_stride + offs).to(tl.float32)
        s = tl.load(s_ptr + t * s_stride + offs)
        d = h - s
        lo = tl.min(d, 0)
        hi = tl.max(d, 0)
        step = _step(lo, hi, min_range, levels)
        lo = lo.to(tl.float16).to(tl.float32)
        q = _quantize(d, lo, step, levels)
        tl.store(s_ptr + t * s_stride + offs, s + _dequantize(q, lo, step))
        tl.store(row_ptr + t * row_stride + offs, q.to(tl.uint8))
    tl.store(scale_ptr + t * scale_stride + 2 * g, lo.to(tl.float16))
    tl.store(scale_ptr + t * scale_stride + 2 * g + 1, step.to(tl.float16))


@triton.jit
def _decode_kernel(
    row_ptr,
    row_stride,
    scale_ptr,
    scale_stride,
    s_ptr,
    s_stride,
    out_ptr,
    out_stride,
    bits: tl.constexpr,
    group: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.program_id(1)
    lo = tl.load(scale_ptr + t * scale_stride + 2 * g).to(tl.float32)
    step = tl.load(scale_ptr + t * scale_stride + 2 * g + 1).to(tl.float32)
    if bits == 4:
        half: tl.constexpr = group // 2
        k = tl.arange(0, half)
        base = g * group
        packed = tl.load(row_ptr + t * row_stride + g * half + k)
        q_e = (packed & 0xF).to(tl.float32)
        q_o = (packed >> 4).to(tl.float32)
        s_e = tl.load(s_ptr + t * s_stride + base + 2 * k) + _dequantize(q_e, lo, step)
        s_o = tl.load(s_ptr + t * s_stride + base + 2 * k + 1) + _dequantize(
            q_o, lo, step
        )
        tl.store(s_ptr + t * s_stride + base + 2 * k, s_e)
        tl.store(s_ptr + t * s_stride + base + 2 * k + 1, s_o)
        out_ty = out_ptr.dtype.element_ty
        tl.store(out_ptr + t * out_stride + base + 2 * k, s_e.to(out_ty))
        tl.store(out_ptr + t * out_stride + base + 2 * k + 1, s_o.to(out_ty))
    else:
        k = tl.arange(0, group)
        offs = g * group + k
        q = tl.load(row_ptr + t * row_stride + offs).to(tl.float32)
        s = tl.load(s_ptr + t * s_stride + offs) + _dequantize(q, lo, step)
        tl.store(s_ptr + t * s_stride + offs, s)
        tl.store(out_ptr + t * out_stride + offs, s.to(out_ptr.dtype.element_ty))


def _scales(rows: torch.Tensor, hidden: int, bits: int) -> torch.Tensor:
    """fp16 view [T, 2 * groups] of the scales at the end of each row."""
    data = hidden * bits // BITS_PER_BYTE
    return rows[:, data:].view(torch.float16)


def encode_delta(fmt: str, h_prime: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Fused Attention kernel: rows of Q(h' - S); ``S += deq`` in place."""
    bits = FORMAT_BITS[fmt]
    tokens, hidden = state.shape
    if h_prime.stride(-1) != 1 or state.stride(-1) != 1:
        raise ValueError("delta dispatch needs row-contiguous h' and state")
    rows = torch.empty(
        (tokens, row_bytes(fmt, hidden)), dtype=torch.uint8, device=state.device
    )
    if tokens:
        scales = _scales(rows, hidden, bits)
        _encode_kernel[(tokens, hidden // QUANT_GROUP_SIZE)](
            h_prime,
            h_prime.stride(0),
            state,
            state.stride(0),
            rows,
            rows.stride(0),
            scales,
            scales.stride(0),
            MIN_GROUP_RANGE,
            bits=bits,
            group=QUANT_GROUP_SIZE,
        )
    return rows


def decode_delta(
    fmt: str, rows: torch.Tensor, state: torch.Tensor, h_out: torch.Tensor
) -> None:
    """Fused FFN kernel: ``S += deq(rows)`` and ``h_out = S`` (h_out's dtype)."""
    bits = FORMAT_BITS[fmt]
    tokens, hidden = state.shape
    if rows.shape != (tokens, row_bytes(fmt, hidden)) or rows.stride(-1) != 1:
        raise ValueError(
            f"{fmt} rows {tuple(rows.shape)} for state {tuple(state.shape)}"
        )
    if tokens:
        scales = _scales(rows, hidden, bits)
        _decode_kernel[(tokens, hidden // QUANT_GROUP_SIZE)](
            rows,
            rows.stride(0),
            scales,
            scales.stride(0),
            state,
            state.stride(0),
            h_out,
            h_out.stride(0),
            bits=bits,
            group=QUANT_GROUP_SIZE,
        )


__all__ = ["decode_delta", "encode_delta"]
