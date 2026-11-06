# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fused CUDA kernels of delta dispatch (Triton).

Same arithmetic and byte layout as the torch reference in ``delta_dispatch.py``
(``encode_rows_reference`` / ``decode_rows_reference``); one program per
(row, group of ``QUANT_GROUP_SIZE`` values), each row in the format of its code
and at its own byte offset, so one launch per side and MoE layer covers rows
of every format.

* ``encode_rows`` (Attention): per row, ``h'`` in bf16 (first visit, the state
  becomes ``h'``) or ``d = h' - S`` in fp32 / int8 / int4 / int3 (the group's
  ``lo`` / ``step`` rounded to fp16, IEEE divisions, ``q = round_half_even((d -
  lo) / step)`` clamped) with ``S += deq`` in place; ``h_int4`` (first visit)
  quantizes ``h'`` with ``S`` taken as zero, so ``S = deq``. Also the row's
  word and the MoE layer of the visit. Without states (``x_int4``, ``x_int8``) ``S`` is
  zero.
* ``decode_rows`` (FFN): per row, the value rebuilt from the bytes: ``S = h'``
  (``h_int4``: ``S = 0 + deq``) or ``S += deq`` and ``h_out = S`` in the model
  dtype (``h_out = deq`` without states).
* ``encode_combine`` (FFN) / ``decode_combine`` (Attention): the compressed
  partial sums of combine compression, every row in one int format: ``v =
  partial (+ e)``, the bytes, ``deq`` and the new residual ``e = v - deq``; and
  ``deq`` back from the bytes.

Both sides dequantize with the same ``_dequantize`` (an explicit fused
multiply-add), so the states the FFN rank rebuilds are bit-identical to the
ones the Attention side keeps.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from afd_plugin.delta_dispatch import (
    CODE_BITS,
    CODE_MASK,
    CODE_SHIFT,
    D_INT4,
    D_INT8,
    MIN_GROUP_RANGE,
    QUANT_GROUP_SIZE,
    RANK_SHIFT,
    DeltaDispatchConfig,
    row_bytes,
)

# The formats the combine kernels handle (their int4 branch is the else).
COMBINE_CODES = (D_INT8, D_INT4)


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


# Runtime integers are not specialized (a value of 1 would become a constant and
# the batch size would select kernel variants).
@triton.jit(do_not_specialize=["num_tokens", "rank_word", "layer_pos"])
def _encode_rows_kernel(
    src_ptr,
    src_stride,
    state_ptr,
    state_stride,
    last_ptr,
    flat_ptr,
    code_ptr,
    off_ptr,
    out_ptr,
    meta_ptr,
    num_tokens,
    rank_word,
    layer_pos,
    min_range,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
    code_shift: tl.constexpr,
):
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    flat = tl.load(flat_ptr + i)
    t = flat % num_tokens
    code = tl.load(code_ptr + i)
    row = out_ptr + tl.load(off_ptr + i)
    base = g * group
    if code == 0:  # h_bf16
        k = tl.arange(0, group)
        h = tl.load(src_ptr + t * src_stride + base + k)
        tl.store(row.to(tl.pointer_type(tl.bfloat16)) + base + k, h)
        if stateful:
            tl.store(state_ptr + flat * state_stride + base + k, h.to(tl.float32))
    elif code == 1:  # d_fp32
        k = tl.arange(0, group)
        h = tl.load(src_ptr + t * src_stride + base + k).to(tl.float32)
        s = tl.load(state_ptr + flat * state_stride + base + k)
        d = h - s
        tl.store(row.to(tl.pointer_type(tl.float32)) + base + k, d)
        tl.store(state_ptr + flat * state_stride + base + k, s + d)
    elif code == 2:  # d_int8
        k = tl.arange(0, group)
        h = tl.load(src_ptr + t * src_stride + base + k).to(tl.float32)
        if stateful:
            s = tl.load(state_ptr + flat * state_stride + base + k)
        else:
            s = tl.zeros([group], dtype=tl.float32)
        d = h - s
        lo = tl.min(d, 0)
        step = _step(lo, tl.max(d, 0), min_range, 255)
        lo = lo.to(tl.float16).to(tl.float32)
        q = _quantize(d, lo, step, 255)
        if stateful:
            tl.store(
                state_ptr + flat * state_stride + base + k, s + _dequantize(q, lo, step)
            )
        tl.store(row + base + k, q.to(tl.uint8))
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 2 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))
    elif (code == 3) | (code == 4):  # d_int4, h_int4
        k = tl.arange(0, half)
        h_e = tl.load(src_ptr + t * src_stride + base + 2 * k).to(tl.float32)
        h_o = tl.load(src_ptr + t * src_stride + base + 2 * k + 1).to(tl.float32)
        if stateful:
            # h_int4 quantizes h' itself and replaces the state (counts as 0).
            diff = code == 3
            s_e = tl.load(
                state_ptr + flat * state_stride + base + 2 * k, mask=diff, other=0.0
            )
            s_o = tl.load(
                state_ptr + flat * state_stride + base + 2 * k + 1,
                mask=diff,
                other=0.0,
            )
        else:
            s_e = tl.zeros([half], dtype=tl.float32)
            s_o = tl.zeros([half], dtype=tl.float32)
        d_e = h_e - s_e
        d_o = h_o - s_o
        lo = tl.minimum(tl.min(d_e, 0), tl.min(d_o, 0))
        hi = tl.maximum(tl.max(d_e, 0), tl.max(d_o, 0))
        step = _step(lo, hi, min_range, 15)
        lo = lo.to(tl.float16).to(tl.float32)
        q_e = _quantize(d_e, lo, step, 15)
        q_o = _quantize(d_o, lo, step, 15)
        if stateful:
            tl.store(
                state_ptr + flat * state_stride + base + 2 * k,
                s_e + _dequantize(q_e, lo, step),
            )
            tl.store(
                state_ptr + flat * state_stride + base + 2 * k + 1,
                s_o + _dequantize(q_o, lo, step),
            )
        tl.store(row + g * half + k, q_e.to(tl.uint8) | (q_o.to(tl.uint8) << 4))
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 4 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))
    elif code == 5:  # d_int3
        # The group as lanes x 8 values; a lane's 8 3-bit values are one
        # little-endian 24-bit word, stored as 3 bytes (of a 4-wide tile).
        lanes: tl.constexpr = group // 8
        r = tl.arange(0, lanes)[:, None]
        c = tl.arange(0, 8)[None, :]
        b = tl.arange(0, 4)[None, :]
        idx = base + 8 * r + c
        h = tl.load(src_ptr + t * src_stride + idx).to(tl.float32)
        if stateful:
            s = tl.load(state_ptr + flat * state_stride + idx)
        else:
            s = tl.zeros([lanes, 8], dtype=tl.float32)
        d = h - s
        lo = tl.min(tl.min(d, 1), 0)
        step = _step(lo, tl.max(tl.max(d, 1), 0), min_range, 7)
        lo = lo.to(tl.float16).to(tl.float32)
        q = _quantize(d, lo, step, 7)
        if stateful:
            tl.store(
                state_ptr + flat * state_stride + idx, s + _dequantize(q, lo, step)
            )
        word = tl.sum(q.to(tl.int32) << (3 * c), 1)
        packed = (word[:, None] >> (8 * b)) & 0xFF
        tl.store(row + g * (3 * lanes) + 3 * r + b, packed.to(tl.uint8), mask=b < 3)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden * 3 // 16 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))
    if g == 0:
        word = t + (code.to(tl.int64) << code_shift) + rank_word
        tl.store(meta_ptr + i, word.to(tl.int32))
        if stateful:
            tl.store(last_ptr + flat, layer_pos)


@triton.jit
def _decode_rows_kernel(
    data_ptr,
    meta_ptr,
    off_ptr,
    rows_ptr,
    state_ptr,
    state_stride,
    out_ptr,
    out_stride,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
    code_shift: tl.constexpr,
    code_mask: tl.constexpr,
):
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    code = (tl.load(meta_ptr + i) >> code_shift) & code_mask
    row = data_ptr + tl.load(off_ptr + i)
    srow = tl.load(rows_ptr + i)
    base = g * group
    out_ty = out_ptr.dtype.element_ty
    if code == 0:  # h_bf16
        k = tl.arange(0, group)
        h = tl.load(row.to(tl.pointer_type(tl.bfloat16)) + base + k).to(tl.float32)
        if stateful:
            tl.store(state_ptr + srow * state_stride + base + k, h)
        tl.store(out_ptr + i * out_stride + base + k, h.to(out_ty))
    elif code == 1:  # d_fp32
        k = tl.arange(0, group)
        d = tl.load(row.to(tl.pointer_type(tl.float32)) + base + k)
        s = tl.load(state_ptr + srow * state_stride + base + k) + d
        tl.store(state_ptr + srow * state_stride + base + k, s)
        tl.store(out_ptr + i * out_stride + base + k, s.to(out_ty))
    elif code == 2:  # d_int8
        k = tl.arange(0, group)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 2 + 2 * g
        lo = tl.load(scales).to(tl.float32)
        step = tl.load(scales + 1).to(tl.float32)
        v = _dequantize(tl.load(row + base + k).to(tl.float32), lo, step)
        if stateful:
            v = tl.load(state_ptr + srow * state_stride + base + k) + v
            tl.store(state_ptr + srow * state_stride + base + k, v)
        tl.store(out_ptr + i * out_stride + base + k, v.to(out_ty))
    elif (code == 3) | (code == 4):  # d_int4, h_int4
        k = tl.arange(0, half)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 4 + 2 * g
        lo = tl.load(scales).to(tl.float32)
        step = tl.load(scales + 1).to(tl.float32)
        packed = tl.load(row + g * half + k)
        v_e = _dequantize((packed & 0xF).to(tl.float32), lo, step)
        v_o = _dequantize((packed >> 4).to(tl.float32), lo, step)
        if stateful:
            # h_int4 replaces the state (counts as 0, as on the Attention side).
            diff = code == 3
            s_row = state_ptr + srow * state_stride + base
            v_e = tl.load(s_row + 2 * k, mask=diff, other=0.0) + v_e
            v_o = tl.load(s_row + 2 * k + 1, mask=diff, other=0.0) + v_o
            tl.store(s_row + 2 * k, v_e)
            tl.store(s_row + 2 * k + 1, v_o)
        tl.store(out_ptr + i * out_stride + base + 2 * k, v_e.to(out_ty))
        tl.store(out_ptr + i * out_stride + base + 2 * k + 1, v_o.to(out_ty))
    elif code == 5:  # d_int3
        lanes: tl.constexpr = group // 8
        r = tl.arange(0, lanes)[:, None]
        c = tl.arange(0, 8)[None, :]
        b = tl.arange(0, 4)[None, :]
        idx = base + 8 * r + c
        scales = row.to(tl.pointer_type(tl.float16)) + hidden * 3 // 16 + 2 * g
        lo = tl.load(scales).to(tl.float32)
        step = tl.load(scales + 1).to(tl.float32)
        packed = tl.load(row + g * (3 * lanes) + 3 * r + b, mask=b < 3, other=0)
        word = tl.sum(packed.to(tl.int32) << (8 * b), 1)
        v = _dequantize(((word[:, None] >> (3 * c)) & 7).to(tl.float32), lo, step)
        if stateful:
            v = tl.load(state_ptr + srow * state_stride + idx) + v
            tl.store(state_ptr + srow * state_stride + idx, v)
        tl.store(out_ptr + i * out_stride + idx, v.to(out_ty))


@triton.jit
def _encode_combine_kernel(
    part_ptr,
    part_stride,
    rows_ptr,
    err_ptr,
    err_stride,
    out_ptr,
    deq_ptr,
    deq_stride,
    row_bytes,
    min_range,
    hidden: tl.constexpr,
    group: tl.constexpr,
    bits: tl.constexpr,
    ef: tl.constexpr,
):
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    row = out_ptr + i * row_bytes
    base = g * group
    if ef:
        erow = err_ptr + tl.load(rows_ptr + i) * err_stride
    if bits == 8:
        k = tl.arange(0, group)
        v = tl.load(part_ptr + i * part_stride + base + k).to(tl.float32)
        if ef:
            v = v + tl.load(erow + base + k)
        lo = tl.min(v, 0)
        step = _step(lo, tl.max(v, 0), min_range, 255)
        lo = lo.to(tl.float16).to(tl.float32)
        q = _quantize(v, lo, step, 255)
        d = _dequantize(q, lo, step)
        tl.store(row + base + k, q.to(tl.uint8))
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 2 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))
        tl.store(deq_ptr + i * deq_stride + base + k, d)
        if ef:
            tl.store(erow + base + k, v - d)
    else:
        k = tl.arange(0, half)
        v_e = tl.load(part_ptr + i * part_stride + base + 2 * k).to(tl.float32)
        v_o = tl.load(part_ptr + i * part_stride + base + 2 * k + 1).to(tl.float32)
        if ef:
            v_e = v_e + tl.load(erow + base + 2 * k)
            v_o = v_o + tl.load(erow + base + 2 * k + 1)
        lo = tl.minimum(tl.min(v_e, 0), tl.min(v_o, 0))
        hi = tl.maximum(tl.max(v_e, 0), tl.max(v_o, 0))
        step = _step(lo, hi, min_range, 15)
        lo = lo.to(tl.float16).to(tl.float32)
        q_e = _quantize(v_e, lo, step, 15)
        q_o = _quantize(v_o, lo, step, 15)
        d_e = _dequantize(q_e, lo, step)
        d_o = _dequantize(q_o, lo, step)
        tl.store(row + g * half + k, q_e.to(tl.uint8) | (q_o.to(tl.uint8) << 4))
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 4 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))
        tl.store(deq_ptr + i * deq_stride + base + 2 * k, d_e)
        tl.store(deq_ptr + i * deq_stride + base + 2 * k + 1, d_o)
        if ef:
            tl.store(erow + base + 2 * k, v_e - d_e)
            tl.store(erow + base + 2 * k + 1, v_o - d_o)


@triton.jit
def _decode_combine_kernel(
    data_ptr,
    row_bytes,
    out_ptr,
    out_stride,
    hidden: tl.constexpr,
    group: tl.constexpr,
    bits: tl.constexpr,
):
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    row = data_ptr + i * row_bytes
    base = g * group
    if bits == 8:
        k = tl.arange(0, group)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 2 + 2 * g
        lo = tl.load(scales).to(tl.float32)
        step = tl.load(scales + 1).to(tl.float32)
        v = _dequantize(tl.load(row + base + k).to(tl.float32), lo, step)
        tl.store(out_ptr + i * out_stride + base + k, v)
    else:
        k = tl.arange(0, half)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 4 + 2 * g
        lo = tl.load(scales).to(tl.float32)
        step = tl.load(scales + 1).to(tl.float32)
        packed = tl.load(row + g * half + k)
        tl.store(
            out_ptr + i * out_stride + base + 2 * k,
            _dequantize((packed & 0xF).to(tl.float32), lo, step),
        )
        tl.store(
            out_ptr + i * out_stride + base + 2 * k + 1,
            _dequantize((packed >> 4).to(tl.float32), lo, step),
        )


def _state_args(state: torch.Tensor | None, like: torch.Tensor) -> tuple[Any, int]:
    if state is None:
        return like, 0
    if state.stride(-1) != 1:
        raise ValueError("delta dispatch states must be row-contiguous")
    return state, state.stride(0)


def encode_rows(
    cfg: DeltaDispatchConfig,
    src: torch.Tensor,
    state: torch.Tensor | None,
    last: torch.Tensor | None,
    layer_pos: int,
    flat: torch.Tensor,
    codes: torch.Tensor,
    offsets: torch.Tensor,
    num_tokens: int,
    rank: int,
    out: torch.Tensor,
    meta: torch.Tensor,
) -> None:
    """Fused Attention kernel (see ``encode_rows_reference``)."""
    rows, hidden = int(flat.shape[0]), src.shape[1]
    if not rows:
        return
    if src.stride(-1) != 1 or out.dim() != 1 or out.dtype != torch.uint8:
        raise ValueError("delta dispatch needs row-contiguous src and flat uint8 out")
    stateful = cfg.stateful
    state_arg, state_stride = _state_args(state if stateful else None, src)
    _encode_rows_kernel[(rows, hidden // QUANT_GROUP_SIZE)](
        src,
        src.stride(0),
        state_arg,
        state_stride,
        last if (stateful and last is not None) else flat,
        flat,
        codes,
        offsets,
        out,
        meta,
        num_tokens,
        rank << RANK_SHIFT,
        layer_pos,
        MIN_GROUP_RANGE,
        hidden=hidden,
        group=QUANT_GROUP_SIZE,
        stateful=stateful,
        code_shift=CODE_SHIFT,
    )


def decode_rows(
    cfg: DeltaDispatchConfig,
    data: torch.Tensor,
    meta: torch.Tensor,
    offsets: torch.Tensor,
    state_rows: torch.Tensor,
    state: torch.Tensor | None,
    h_out: torch.Tensor,
) -> None:
    """Fused FFN kernel (see ``decode_rows_reference``)."""
    rows, hidden = int(meta.shape[0]), h_out.shape[1]
    if not rows:
        return
    if h_out.stride(-1) != 1 or data.dim() != 1 or data.dtype != torch.uint8:
        raise ValueError(
            "delta dispatch needs row-contiguous h_out and flat uint8 data"
        )
    stateful = cfg.stateful
    state_arg, state_stride = _state_args(state if stateful else None, h_out)
    _decode_rows_kernel[(rows, hidden // QUANT_GROUP_SIZE)](
        data,
        meta,
        offsets,
        state_rows,
        state_arg,
        state_stride,
        h_out,
        h_out.stride(0),
        hidden=hidden,
        group=QUANT_GROUP_SIZE,
        stateful=stateful,
        code_shift=CODE_SHIFT,
        code_mask=CODE_MASK,
    )


def encode_combine(
    code: int,
    partial: torch.Tensor,
    rows: torch.Tensor,
    err: torch.Tensor | None,
    out: torch.Tensor,
    deq: torch.Tensor,
) -> None:
    """Fused FFN kernel (see ``encode_combine_reference``)."""
    assert code in COMBINE_CODES, f"combine rows are d_int8 / d_int4, not {code}"
    n, hidden = partial.shape
    if not n:
        return
    if (
        partial.stride(-1) != 1
        or deq.stride(-1) != 1
        or deq.dtype != torch.float32
        or out.dim() != 1
        or out.dtype != torch.uint8
    ):
        raise ValueError(
            "combine compression needs row-contiguous partial and fp32 deq, "
            "flat uint8 out"
        )
    err_arg, err_stride = _state_args(err, deq)
    _encode_combine_kernel[(n, hidden // QUANT_GROUP_SIZE)](
        partial,
        partial.stride(0),
        rows,
        err_arg,
        err_stride,
        out,
        deq,
        deq.stride(0),
        row_bytes(code, hidden),
        MIN_GROUP_RANGE,
        hidden=hidden,
        group=QUANT_GROUP_SIZE,
        bits=CODE_BITS[code],
        ef=err is not None,
    )


def decode_combine(code: int, data: torch.Tensor, out: torch.Tensor) -> None:
    """Fused Attention kernel (see ``decode_combine_reference``)."""
    assert code in COMBINE_CODES, f"combine rows are d_int8 / d_int4, not {code}"
    n, hidden = out.shape
    if not n:
        return
    if out.stride(-1) != 1 or out.dtype != torch.float32 or data.dim() != 1:
        raise ValueError("combine compression needs a row-contiguous fp32 out")
    _decode_combine_kernel[(n, hidden // QUANT_GROUP_SIZE)](
        data,
        row_bytes(code, hidden),
        out,
        out.stride(0),
        hidden=hidden,
        group=QUANT_GROUP_SIZE,
        bits=CODE_BITS[code],
    )


__all__ = ["decode_combine", "decode_rows", "encode_combine", "encode_rows"]
