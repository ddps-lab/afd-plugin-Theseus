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
  lo) / step)`` clamped) with ``S += deq`` in place; ``h_int4`` / ``h_int8``
  (first visit) quantizes ``h'`` with ``S`` taken as zero, so ``S = deq``.
  Also the row's word and the MoE layer of the visit. Without states
  (``x_int4``, ``x_int8``) ``S`` is zero.
* ``decode_rows`` (FFN): per row, the value rebuilt from the bytes: ``S = h'``
  (``h_int4`` / ``h_int8``: ``S = 0 + deq``) or ``S += deq`` and ``h_out = S``
  in the model dtype (``h_out = deq`` without states).
* ``encode_combine`` (FFN) / ``decode_combine`` (Attention): the compressed
  partial sums of combine compression, every row of a layer in one format:
  ``v = partial (+ e)``, the bytes, ``deq`` and the new residual ``e =
  v - deq``; and ``deq`` back from the bytes.

NVFP4 rows (``h_nvfp4`` / ``d_nvfp4``, combine ``nvfp4``) need the row's |max|
before any block is compressed. Two encode structures are kept so they can be
timed against each other (the slower one is to be removed): two passes
(``NVFP4_TWO_PASS``: a row-maximum kernel, then the (row, group) kernel) and
one pass (``NVFP4_ONE_PASS``: one program per row; rows of other formats in
the launch run the group code over their groups). The E4M3 block scales are
rounded in software by default, or by the GPU's conversion with
``hardware_e4m3`` (Ada and later); both give the same bytes.

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
    D_NVFP4,
    E2M1_MAX,
    E4M3_MAX,
    MIN_GROUP_RANGE,
    NVFP4_BLOCK,
    NVFP4_CODES,
    NVFP4_MIN_SCALE,
    NVFP4_ONE_PASS,
    NVFP4_PASSES,
    NVFP4_TWO_PASS,
    QUANT_GROUP_SIZE,
    RANK_SHIFT,
    TOKEN_MASK,
    DeltaDispatchConfig,
    row_bytes,
)

# The formats the combine kernels handle (their int4 branch is the else).
COMBINE_CODES = (D_INT8, D_INT4, D_NVFP4)
# The NVFP4 constants as Triton compile-time values (kernels read module
# globals only as constexpr).
_NVFP4_BLOCK = tl.constexpr(NVFP4_BLOCK)
_E2M1_MAX = tl.constexpr(E2M1_MAX)
_NVFP4_ROW_DIVISOR = tl.constexpr(E4M3_MAX * E2M1_MAX)
_NVFP4_MIN_SCALE = tl.constexpr(NVFP4_MIN_SCALE)


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
def _row_code(
    i,
    flat,
    last_ptr,
    code_ptr,
    first_code,
    rest_code,
    stateful: tl.constexpr,
    codes_in_kernel: tl.constexpr,
):
    """The format code of row ``i`` (state row ``flat``)."""
    if codes_in_kernel:
        # One row size: a first visit (no earlier visit to the FFN rank in this
        # forward) or a later one. Every program of the row reads ``last``
        # before the host updates it after the launch (not here).
        if stateful:
            code = tl.where(tl.load(last_ptr + flat) < 0, first_code, rest_code)
        else:
            code = rest_code + 0 * flat
    else:
        code = tl.load(code_ptr + i)
    return code


@triton.jit
def _encode_group(
    src_row,
    state_row,
    row,
    code,
    g,
    min_range,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
):
    """Group ``g`` of a row in a format other than NVFP4: ``src_row`` / ``state_row``
    point at the row's token in ``src`` and its state, ``row`` at its bytes."""
    half: tl.constexpr = group // 2
    base = g * group
    if code == 0:  # h_bf16
        k = tl.arange(0, group)
        h = tl.load(src_row + base + k)
        tl.store(row.to(tl.pointer_type(tl.bfloat16)) + base + k, h)
        if stateful:
            tl.store(state_row + base + k, h.to(tl.float32))
    elif code == 1:  # d_fp32
        k = tl.arange(0, group)
        h = tl.load(src_row + base + k).to(tl.float32)
        s = tl.load(state_row + base + k)
        d = h - s
        tl.store(row.to(tl.pointer_type(tl.float32)) + base + k, d)
        tl.store(state_row + base + k, s + d)
    elif (code == 2) | (code == 6):  # d_int8, h_int8
        k = tl.arange(0, group)
        h = tl.load(src_row + base + k).to(tl.float32)
        if stateful:
            # h_int8 quantizes h' itself and replaces the state (counts as 0).
            s = tl.load(state_row + base + k, mask=code == 2, other=0.0)
        else:
            s = tl.zeros([group], dtype=tl.float32)
        d = h - s
        lo = tl.min(d, 0)
        step = _step(lo, tl.max(d, 0), min_range, 255)
        lo = lo.to(tl.float16).to(tl.float32)
        q = _quantize(d, lo, step, 255)
        if stateful:
            tl.store(state_row + base + k, s + _dequantize(q, lo, step))
        tl.store(row + base + k, q.to(tl.uint8))
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 2 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))
    elif (code == 3) | (code == 4):  # d_int4, h_int4
        k = tl.arange(0, half)
        h_e = tl.load(src_row + base + 2 * k).to(tl.float32)
        h_o = tl.load(src_row + base + 2 * k + 1).to(tl.float32)
        if stateful:
            # h_int4 quantizes h' itself and replaces the state (counts as 0).
            diff = code == 3
            s_e = tl.load(state_row + base + 2 * k, mask=diff, other=0.0)
            s_o = tl.load(
                state_row + base + 2 * k + 1,
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
                state_row + base + 2 * k,
                s_e + _dequantize(q_e, lo, step),
            )
            tl.store(
                state_row + base + 2 * k + 1,
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
        h = tl.load(src_row + idx).to(tl.float32)
        if stateful:
            s = tl.load(state_row + idx)
        else:
            s = tl.zeros([lanes, 8], dtype=tl.float32)
        d = h - s
        lo = tl.min(tl.min(d, 1), 0)
        step = _step(lo, tl.max(tl.max(d, 1), 0), min_range, 7)
        lo = lo.to(tl.float16).to(tl.float32)
        q = _quantize(d, lo, step, 7)
        if stateful:
            tl.store(state_row + idx, s + _dequantize(q, lo, step))
        word = tl.sum(q.to(tl.int32) << (3 * c), 1)
        packed = (word[:, None] >> (8 * b)) & 0xFF
        tl.store(row + g * (3 * lanes) + 3 * r + b, packed.to(tl.uint8), mask=b < 3)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden * 3 // 16 + 2 * g
        tl.store(scales, lo.to(tl.float16))
        tl.store(scales + 1, step.to(tl.float16))


@triton.jit
def _pow2(e):
    """2^e as fp32 for int32 exponents in fp32's normal range."""
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e4m3(x, hardware: tl.constexpr):
    """fp32 ``x`` in [0, E4M3_MAX] -> (the nearest E4M3 value as fp32, ties to
    even; its byte). ``hardware``: the GPU's fp32 -> E4M3 conversion (Ada and
    later), called directly: Triton's own cast goes through fp16 truncated
    toward zero on sm_89, two roundings that differ just above a midpoint.
    Else from the fp32 bits: E4M3 keeps 3 mantissa bits for exponents -6..8
    and steps of 2^-9 below 2^-6 (subnormals)."""
    if hardware:
        pair = tl.inline_asm_elementwise(
            asm="cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
            constraints="=h,f",
            args=[x],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
        byte = (pair & 0xFF).to(tl.uint8)
        v = _e4m3_value(byte, hardware)
    else:
        e = ((x.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
        e = tl.minimum(tl.maximum(e, -6), 8)
        m = _round_half_even(x * _pow2(3 - e))
        v = m * _pow2(e - 3)
        bits = v.to(tl.int32, bitcast=True)
        # E4M3 exponent field = fp32 exponent field - 127 + 7.
        normal = ((((bits >> 23) & 0xFF) - 120) << 3) | ((bits >> 20) & 7)
        byte = tl.where(v >= 0.015625, normal, m.to(tl.int32)).to(tl.uint8)
    return v, byte


@triton.jit
def _e4m3_value(byte, hardware: tl.constexpr):
    """An E4M3 byte (sign 0) -> fp32."""
    if hardware:
        v = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        b = byte.to(tl.int32)
        field = b >> 3
        mant = b & 7
        normal = (((field + 120) << 23) | (mant << 20)).to(tl.float32, bitcast=True)
        v = tl.where(field == 0, mant.to(tl.float32) * 0.001953125, normal)
    return v


@triton.jit
def _e2m1_code(y):
    """fp32 -> E2M1 code (int32): the sign in bit 3, below it the index of the
    nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6 to |y| (saturating at 6). On a
    midpoint the even index wins: ``>`` at the midpoints above an even index,
    ``>=`` at those above an odd one."""
    a = tl.abs(y)
    index = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return index | ((y < 0).to(tl.int32) << 3)


@triton.jit
def _nvfp4_dequantize(code, scale):
    """E2M1 ``code`` (int32) times its block's ``scale`` (same shape). An
    explicit fused multiply-add with 0: one rounding like the torch
    reference's product, and the compiler does not fuse the product into the
    sum that follows (the state update, the combine residual)."""
    index = (code & 7).to(tl.float32)
    mag = tl.where(index < 4, index * 0.5, tl.where(index < 7, index - 2.0, 6.0))
    value = tl.where((code & 8) != 0, -mag, mag)
    return tl.fma(value, scale, tl.zeros(value.shape, tl.float32))


@triton.jit
def _nvfp4_row_scales(amax, blocks: tl.constexpr):
    """The row scale max(|max| / (E4M3_MAX * E2M1_MAX), NVFP4_MIN_SCALE), as
    ``blocks`` equal fp32 values (IEEE division, like the reference)."""
    span = amax + tl.zeros([blocks], tl.float32)
    gs = tl.math.div_rn(span, tl.full([blocks], _NVFP4_ROW_DIVISOR, tl.float32))
    return tl.maximum(gs, tl.full([blocks], _NVFP4_MIN_SCALE, tl.float32))


@triton.jit
def _nvfp4_tile(d_e, d_o, gs, hardware: tl.constexpr):
    """NVFP4 of whole blocks: ``d_e`` / ``d_o`` fp32 [B, 8] the even / odd
    values of B blocks of NVFP4_BLOCK, ``gs`` the row scale ([B] equal values).
    Returns the data bytes [B, 8] (even value in the low nibble), the blocks'
    E4M3 bytes [B] and the dequantized even / odd values."""
    blocks: tl.constexpr = d_e.shape[0]
    bmax = tl.maximum(tl.max(tl.abs(d_e), 1), tl.max(tl.abs(d_o), 1))
    bs_in = tl.math.div_rn(bmax, tl.full([blocks], _E2M1_MAX, tl.float32))
    bs, bs_byte = _e4m3(tl.math.div_rn(bs_in, gs), hardware)
    scale = tl.broadcast_to((bs * gs)[:, None], d_e.shape)
    floor = tl.full(d_e.shape, _NVFP4_MIN_SCALE, tl.float32)
    safe = tl.where(scale > 0, tl.maximum(scale, floor), 1.0)
    q_e = tl.where(scale > 0, _e2m1_code(tl.math.div_rn(d_e, safe)), 0)
    q_o = tl.where(scale > 0, _e2m1_code(tl.math.div_rn(d_o, safe)), 0)
    data = (q_e | (q_o << 4)).to(tl.uint8)
    return data, bs_byte, _nvfp4_dequantize(q_e, scale), _nvfp4_dequantize(q_o, scale)


@triton.jit
def _encode_nvfp4_group(
    src_row,
    state_row,
    row,
    code,
    g,
    amax,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
    hardware: tl.constexpr,
):
    """Group ``g`` of an h_nvfp4 / d_nvfp4 row (``amax``: the row's |max|):
    its blocks' codes and E4M3 bytes; the group-0 program also writes the
    row scale."""
    blocks: tl.constexpr = group // _NVFP4_BLOCK
    r = tl.arange(0, blocks)[:, None]
    c = tl.arange(0, _NVFP4_BLOCK // 2)[None, :]
    idx = g * group + _NVFP4_BLOCK * r + 2 * c
    h_e = tl.load(src_row + idx).to(tl.float32)
    h_o = tl.load(src_row + idx + 1).to(tl.float32)
    if stateful:
        # h_nvfp4 compresses h' itself and replaces the state (counts as 0).
        s_e = tl.load(state_row + idx, mask=code == 8, other=0.0)
        s_o = tl.load(state_row + idx + 1, mask=code == 8, other=0.0)
    else:
        s_e = tl.zeros([blocks, _NVFP4_BLOCK // 2], dtype=tl.float32)
        s_o = tl.zeros([blocks, _NVFP4_BLOCK // 2], dtype=tl.float32)
    gs = _nvfp4_row_scales(amax, blocks)
    data, bs_byte, v_e, v_o = _nvfp4_tile(h_e - s_e, h_o - s_o, gs, hardware)
    if stateful:
        tl.store(state_row + idx, s_e + v_e)
        tl.store(state_row + idx + 1, s_o + v_o)
    tl.store(row + g * (group // 2) + (_NVFP4_BLOCK // 2) * r + c, data)
    tl.store(row + hidden // 2 + g * blocks + tl.arange(0, blocks), bs_byte)
    if g == 0:
        gs_ptr = (row + hidden // 2 + hidden // _NVFP4_BLOCK).to(
            tl.pointer_type(tl.float32)
        )
        tl.store(gs_ptr, tl.max(gs, 0))


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
    amax_ptr,
    num_tokens,
    rank_word,
    layer_pos,
    min_range,
    first_code,
    rest_code,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
    code_shift: tl.constexpr,
    codes_in_kernel: tl.constexpr,
    nvfp4: tl.constexpr,
    hardware: tl.constexpr,
):
    """One program per (row, group). ``nvfp4``: the mode has NVFP4 rows, whose
    |max| over the row ``_encode_rows_amax_kernel`` wrote to ``amax_ptr``."""
    i = tl.program_id(0)
    g = tl.program_id(1)
    flat = tl.load(flat_ptr + i)
    t = flat % num_tokens
    code = _row_code(
        i, flat, last_ptr, code_ptr, first_code, rest_code, stateful, codes_in_kernel
    )
    row = out_ptr + tl.load(off_ptr + i)
    src_row = src_ptr + t * src_stride
    state_row = state_ptr + flat * state_stride
    if nvfp4:
        if (code == 7) | (code == 8):  # h_nvfp4, d_nvfp4
            amax = tl.load(amax_ptr + i).to(tl.float32)
            _encode_nvfp4_group(
                src_row,
                state_row,
                row,
                code,
                g,
                amax,
                hidden,
                group,
                stateful,
                hardware,
            )
        else:
            _encode_group(
                src_row, state_row, row, code, g, min_range, hidden, group, stateful
            )
    else:
        _encode_group(
            src_row, state_row, row, code, g, min_range, hidden, group, stateful
        )
    if g == 0:
        word = t + (code.to(tl.int64) << code_shift) + rank_word
        tl.store(meta_ptr + i, word.to(tl.int32))
        if stateful and not codes_in_kernel:
            tl.store(last_ptr + flat, layer_pos)


@triton.jit(do_not_specialize=["num_tokens"])
def _encode_rows_amax_kernel(
    src_ptr,
    src_stride,
    state_ptr,
    state_stride,
    last_ptr,
    flat_ptr,
    code_ptr,
    amax_ptr,
    num_tokens,
    first_code,
    rest_code,
    hidden: tl.constexpr,
    hidden_p2: tl.constexpr,
    stateful: tl.constexpr,
    codes_in_kernel: tl.constexpr,
):
    """Two-pass NVFP4, pass 1 (one program per row): the |max| of an h_nvfp4
    (h') / d_nvfp4 (h' - S) row into ``amax_ptr``; other rows write nothing."""
    i = tl.program_id(0)
    flat = tl.load(flat_ptr + i)
    t = flat % num_tokens
    code = _row_code(
        i, flat, last_ptr, code_ptr, first_code, rest_code, stateful, codes_in_kernel
    )
    if (code == 7) | (code == 8):  # h_nvfp4, d_nvfp4
        k = tl.arange(0, hidden_p2)
        inside = k < hidden
        h = tl.load(src_ptr + t * src_stride + k, mask=inside, other=0.0)
        if stateful:
            s = tl.load(
                state_ptr + flat * state_stride + k,
                mask=inside & (code == 8),
                other=0.0,
            )
        else:
            s = tl.zeros([hidden_p2], dtype=tl.float32)
        tl.store(amax_ptr + i, tl.max(tl.abs(h.to(tl.float32) - s), 0))


@triton.jit(do_not_specialize=["num_tokens", "rank_word", "layer_pos"])
def _encode_rows_row_kernel(
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
    first_code,
    rest_code,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
    code_shift: tl.constexpr,
    codes_in_kernel: tl.constexpr,
    blocks_p2: tl.constexpr,
    hardware: tl.constexpr,
):
    """One-pass NVFP4: one program per row. An h_nvfp4 / d_nvfp4 row takes its
    |max| and compresses all its blocks at once; a row of another format runs
    the group code over its groups."""
    i = tl.program_id(0)
    flat = tl.load(flat_ptr + i)
    t = flat % num_tokens
    code = _row_code(
        i, flat, last_ptr, code_ptr, first_code, rest_code, stateful, codes_in_kernel
    )
    row = out_ptr + tl.load(off_ptr + i)
    src_row = src_ptr + t * src_stride
    state_row = state_ptr + flat * state_stride
    if (code == 7) | (code == 8):  # h_nvfp4, d_nvfp4
        blocks: tl.constexpr = hidden // _NVFP4_BLOCK
        r = tl.arange(0, blocks_p2)[:, None]
        c = tl.arange(0, _NVFP4_BLOCK // 2)[None, :]
        inside = (r < blocks) & (c >= 0)
        idx = _NVFP4_BLOCK * r + 2 * c
        h_e = tl.load(src_row + idx, mask=inside, other=0.0).to(tl.float32)
        h_o = tl.load(src_row + idx + 1, mask=inside, other=0.0).to(tl.float32)
        if stateful:
            # h_nvfp4 compresses h' itself and replaces the state (counts as 0).
            diff = inside & (code == 8)
            s_e = tl.load(state_row + idx, mask=diff, other=0.0)
            s_o = tl.load(state_row + idx + 1, mask=diff, other=0.0)
        else:
            s_e = tl.zeros([blocks_p2, _NVFP4_BLOCK // 2], dtype=tl.float32)
            s_o = tl.zeros([blocks_p2, _NVFP4_BLOCK // 2], dtype=tl.float32)
        d_e = h_e - s_e
        d_o = h_o - s_o
        amax = tl.max(tl.maximum(tl.max(tl.abs(d_e), 1), tl.max(tl.abs(d_o), 1)), 0)
        gs = _nvfp4_row_scales(amax, blocks_p2)
        data, bs_byte, v_e, v_o = _nvfp4_tile(d_e, d_o, gs, hardware)
        if stateful:
            tl.store(state_row + idx, s_e + v_e, mask=inside)
            tl.store(state_row + idx + 1, s_o + v_o, mask=inside)
        tl.store(row + (_NVFP4_BLOCK // 2) * r + c, data, mask=inside)
        b = tl.arange(0, blocks_p2)
        tl.store(row + hidden // 2 + b, bs_byte, mask=b < blocks)
        gs_ptr = (row + hidden // 2 + blocks).to(tl.pointer_type(tl.float32))
        tl.store(gs_ptr, tl.max(gs, 0))
    else:
        for g in range(0, hidden // group):
            _encode_group(
                src_row, state_row, row, code, g, min_range, hidden, group, stateful
            )
    word = t + (code.to(tl.int64) << code_shift) + rank_word
    tl.store(meta_ptr + i, word.to(tl.int32))
    if stateful and not codes_in_kernel:
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
    row_size,
    max_tokens,
    hidden: tl.constexpr,
    group: tl.constexpr,
    stateful: tl.constexpr,
    code_shift: tl.constexpr,
    code_mask: tl.constexpr,
    rank_shift: tl.constexpr,
    token_mask: tl.constexpr,
    fixed: tl.constexpr,
    rows_from_meta: tl.constexpr,
    nvfp4: tl.constexpr,
    hardware: tl.constexpr,
):
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    # Not ``word``: the d_int3 branch below has its own (vector) ``word``, and
    # Triton needs one type per name across the branches.
    meta_w = tl.load(meta_ptr + i)
    code = (meta_w >> code_shift) & code_mask
    if fixed:
        row = data_ptr + i.to(tl.int64) * row_size
    else:
        row = data_ptr + tl.load(off_ptr + i)
    if rows_from_meta:
        # The state row from the word (Attention rank * max_tokens + token),
        # written out for the combine of this layer.
        srow = (meta_w >> rank_shift).to(tl.int64) * max_tokens + (meta_w & token_mask)
        if g == 0:
            tl.store(rows_ptr + i, srow.to(rows_ptr.dtype.element_ty))
    else:
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
    elif (code == 2) | (code == 6):  # d_int8, h_int8
        k = tl.arange(0, group)
        scales = row.to(tl.pointer_type(tl.float16)) + hidden // 2 + 2 * g
        lo = tl.load(scales).to(tl.float32)
        step = tl.load(scales + 1).to(tl.float32)
        v = _dequantize(tl.load(row + base + k).to(tl.float32), lo, step)
        if stateful:
            # h_int8 replaces the state (counts as 0, as on the Attention side).
            v = (
                tl.load(
                    state_ptr + srow * state_stride + base + k,
                    mask=code == 2,
                    other=0.0,
                )
                + v
            )
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
    elif nvfp4:  # h_nvfp4, d_nvfp4 (the codes left); names of their own
        nv_blocks: tl.constexpr = group // _NVFP4_BLOCK
        nv_r = tl.arange(0, nv_blocks)[:, None]
        nv_c = tl.arange(0, _NVFP4_BLOCK // 2)[None, :]
        nv_idx = base + _NVFP4_BLOCK * nv_r + 2 * nv_c
        nv_data = tl.load(row + g * half + (_NVFP4_BLOCK // 2) * nv_r + nv_c)
        nv_gs_ptr = (row + hidden // 2 + hidden // _NVFP4_BLOCK).to(
            tl.pointer_type(tl.float32)
        )
        nv_bs = _e4m3_value(
            tl.load(row + hidden // 2 + g * nv_blocks + tl.arange(0, nv_blocks)),
            hardware,
        )
        nv_scale = tl.broadcast_to(
            (nv_bs * tl.load(nv_gs_ptr))[:, None], [nv_blocks, _NVFP4_BLOCK // 2]
        )
        nv_e = _nvfp4_dequantize((nv_data & 0xF).to(tl.int32), nv_scale)
        nv_o = _nvfp4_dequantize((nv_data >> 4).to(tl.int32), nv_scale)
        if stateful:
            # h_nvfp4 replaces the state (counts as 0, as on the Attention side).
            nv_row = state_ptr + srow * state_stride
            nv_e = tl.load(nv_row + nv_idx, mask=code == 8, other=0.0) + nv_e
            nv_o = tl.load(nv_row + nv_idx + 1, mask=code == 8, other=0.0) + nv_o
            tl.store(nv_row + nv_idx, nv_e)
            tl.store(nv_row + nv_idx + 1, nv_o)
        tl.store(out_ptr + i * out_stride + nv_idx, nv_e.to(out_ty))
        tl.store(out_ptr + i * out_stride + nv_idx + 1, nv_o.to(out_ty))


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
    amax_ptr,
    row_bytes,
    min_range,
    hidden: tl.constexpr,
    group: tl.constexpr,
    bits: tl.constexpr,
    ef: tl.constexpr,
    nvfp4: tl.constexpr,
    hardware: tl.constexpr,
):
    """One program per (row, group). ``nvfp4``: the row's |max| from
    ``_combine_amax_kernel`` in ``amax_ptr``."""
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    row = out_ptr + i * row_bytes
    base = g * group
    if ef:
        erow = err_ptr + tl.load(rows_ptr + i) * err_stride
    if nvfp4:
        blocks: tl.constexpr = group // _NVFP4_BLOCK
        r = tl.arange(0, blocks)[:, None]
        c = tl.arange(0, _NVFP4_BLOCK // 2)[None, :]
        idx = base + _NVFP4_BLOCK * r + 2 * c
        v_e = tl.load(part_ptr + i * part_stride + idx).to(tl.float32)
        v_o = tl.load(part_ptr + i * part_stride + idx + 1).to(tl.float32)
        if ef:
            v_e = v_e + tl.load(erow + idx)
            v_o = v_o + tl.load(erow + idx + 1)
        gs = _nvfp4_row_scales(tl.load(amax_ptr + i), blocks)
        data, bs_byte, d_e, d_o = _nvfp4_tile(v_e, v_o, gs, hardware)
        tl.store(row + g * half + (_NVFP4_BLOCK // 2) * r + c, data)
        tl.store(row + hidden // 2 + g * blocks + tl.arange(0, blocks), bs_byte)
        if g == 0:
            gs_ptr = (row + hidden // 2 + hidden // _NVFP4_BLOCK).to(
                tl.pointer_type(tl.float32)
            )
            tl.store(gs_ptr, tl.max(gs, 0))
        tl.store(deq_ptr + i * deq_stride + idx, d_e)
        tl.store(deq_ptr + i * deq_stride + idx + 1, d_o)
        if ef:
            tl.store(erow + idx, v_e - d_e)
            tl.store(erow + idx + 1, v_o - d_o)
    elif bits == 8:
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
def _combine_amax_kernel(
    part_ptr,
    part_stride,
    rows_ptr,
    err_ptr,
    err_stride,
    amax_ptr,
    hidden: tl.constexpr,
    hidden_p2: tl.constexpr,
    ef: tl.constexpr,
):
    """Two-pass NVFP4 combine, pass 1 (one program per row): |partial (+ e)|'s
    max into ``amax_ptr``."""
    i = tl.program_id(0)
    k = tl.arange(0, hidden_p2)
    inside = k < hidden
    v = tl.load(part_ptr + i * part_stride + k, mask=inside, other=0.0).to(tl.float32)
    if ef:
        erow = err_ptr + tl.load(rows_ptr + i) * err_stride
        v = v + tl.load(erow + k, mask=inside, other=0.0)
    tl.store(amax_ptr + i, tl.max(tl.abs(v), 0))


@triton.jit
def _encode_combine_row_kernel(
    part_ptr,
    part_stride,
    rows_ptr,
    err_ptr,
    err_stride,
    out_ptr,
    deq_ptr,
    deq_stride,
    row_bytes,
    hidden: tl.constexpr,
    blocks_p2: tl.constexpr,
    ef: tl.constexpr,
    hardware: tl.constexpr,
):
    """One-pass NVFP4 combine: one program per row (every row NVFP4)."""
    blocks: tl.constexpr = hidden // _NVFP4_BLOCK
    i = tl.program_id(0)
    row = out_ptr + i * row_bytes
    r = tl.arange(0, blocks_p2)[:, None]
    c = tl.arange(0, _NVFP4_BLOCK // 2)[None, :]
    inside = (r < blocks) & (c >= 0)
    idx = _NVFP4_BLOCK * r + 2 * c
    prow = part_ptr + i * part_stride
    v_e = tl.load(prow + idx, mask=inside, other=0.0).to(tl.float32)
    v_o = tl.load(prow + idx + 1, mask=inside, other=0.0).to(tl.float32)
    if ef:
        erow = err_ptr + tl.load(rows_ptr + i) * err_stride
        v_e = v_e + tl.load(erow + idx, mask=inside, other=0.0)
        v_o = v_o + tl.load(erow + idx + 1, mask=inside, other=0.0)
    amax = tl.max(tl.maximum(tl.max(tl.abs(v_e), 1), tl.max(tl.abs(v_o), 1)), 0)
    gs = _nvfp4_row_scales(amax, blocks_p2)
    data, bs_byte, d_e, d_o = _nvfp4_tile(v_e, v_o, gs, hardware)
    tl.store(row + (_NVFP4_BLOCK // 2) * r + c, data, mask=inside)
    b = tl.arange(0, blocks_p2)
    tl.store(row + hidden // 2 + b, bs_byte, mask=b < blocks)
    tl.store(
        (row + hidden // 2 + blocks).to(tl.pointer_type(tl.float32)), tl.max(gs, 0)
    )
    tl.store(deq_ptr + i * deq_stride + idx, d_e, mask=inside)
    tl.store(deq_ptr + i * deq_stride + idx + 1, d_o, mask=inside)
    if ef:
        tl.store(erow + idx, v_e - d_e, mask=inside)
        tl.store(erow + idx + 1, v_o - d_o, mask=inside)


@triton.jit
def _decode_combine_kernel(
    data_ptr,
    row_bytes,
    out_ptr,
    out_stride,
    hidden: tl.constexpr,
    group: tl.constexpr,
    bits: tl.constexpr,
    nvfp4: tl.constexpr,
    hardware: tl.constexpr,
):
    half: tl.constexpr = group // 2
    i = tl.program_id(0)
    g = tl.program_id(1)
    row = data_ptr + i * row_bytes
    base = g * group
    if nvfp4:
        blocks: tl.constexpr = group // _NVFP4_BLOCK
        r = tl.arange(0, blocks)[:, None]
        c = tl.arange(0, _NVFP4_BLOCK // 2)[None, :]
        idx = base + _NVFP4_BLOCK * r + 2 * c
        data = tl.load(row + g * half + (_NVFP4_BLOCK // 2) * r + c)
        gs_ptr = (row + hidden // 2 + hidden // _NVFP4_BLOCK).to(
            tl.pointer_type(tl.float32)
        )
        bs = _e4m3_value(
            tl.load(row + hidden // 2 + g * blocks + tl.arange(0, blocks)), hardware
        )
        scale = tl.broadcast_to(
            (bs * tl.load(gs_ptr))[:, None], [blocks, _NVFP4_BLOCK // 2]
        )
        tl.store(
            out_ptr + i * out_stride + idx,
            _nvfp4_dequantize((data & 0xF).to(tl.int32), scale),
        )
        tl.store(
            out_ptr + i * out_stride + idx + 1,
            _nvfp4_dequantize((data >> 4).to(tl.int32), scale),
        )
    elif bits == 8:
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


def _check_pass(nvfp4_pass: str) -> None:
    if nvfp4_pass not in NVFP4_PASSES:
        raise ValueError(f"nvfp4_pass {nvfp4_pass!r} not in {NVFP4_PASSES}")


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
    nvfp4_pass: str = NVFP4_TWO_PASS,
    hardware_e4m3: bool = False,
) -> None:
    """Fused Attention kernel (see ``encode_rows_reference``)."""
    _check_pass(nvfp4_pass)
    rows, hidden = int(flat.shape[0]), src.shape[1]
    if not rows:
        return
    if src.stride(-1) != 1 or out.dim() != 1 or out.dtype != torch.uint8:
        raise ValueError("delta dispatch needs row-contiguous src and flat uint8 out")
    stateful = cfg.stateful
    state_arg, state_stride = _state_args(state if stateful else None, src)
    in_kernel = codes is None
    first, rest = cfg.first_rest_codes_at(layer_pos) if in_kernel else (0, 0)
    last_arg = last if (stateful and last is not None) else flat
    code_arg = flat if in_kernel else codes
    # Whether rows of this layer can be NVFP4 (an int8 layer of
    # ``int8_layers`` has none: no row maximum, no NVFP4 branches).
    nvfp4 = cfg.layer_uses_nvfp4(layer_pos)
    if nvfp4 and nvfp4_pass == NVFP4_ONE_PASS:
        _encode_rows_row_kernel[(rows,)](
            src,
            src.stride(0),
            state_arg,
            state_stride,
            last_arg,
            flat,
            code_arg,
            offsets,
            out,
            meta,
            num_tokens,
            rank << RANK_SHIFT,
            layer_pos,
            MIN_GROUP_RANGE,
            first,
            rest,
            hidden=hidden,
            group=QUANT_GROUP_SIZE,
            stateful=stateful,
            code_shift=CODE_SHIFT,
            codes_in_kernel=in_kernel,
            blocks_p2=triton.next_power_of_2(hidden // NVFP4_BLOCK),
            hardware=hardware_e4m3,
        )
    else:
        amax = meta  # read only for NVFP4 rows
        if nvfp4:
            amax = torch.empty(rows, dtype=torch.float32, device=src.device)
            _encode_rows_amax_kernel[(rows,)](
                src,
                src.stride(0),
                state_arg,
                state_stride,
                last_arg,
                flat,
                code_arg,
                amax,
                num_tokens,
                first,
                rest,
                hidden=hidden,
                hidden_p2=triton.next_power_of_2(hidden),
                stateful=stateful,
                codes_in_kernel=in_kernel,
            )
        _encode_rows_kernel[(rows, hidden // QUANT_GROUP_SIZE)](
            src,
            src.stride(0),
            state_arg,
            state_stride,
            last_arg,
            flat,
            code_arg,
            offsets,
            out,
            meta,
            amax,
            num_tokens,
            rank << RANK_SHIFT,
            layer_pos,
            MIN_GROUP_RANGE,
            first,
            rest,
            hidden=hidden,
            group=QUANT_GROUP_SIZE,
            stateful=stateful,
            code_shift=CODE_SHIFT,
            codes_in_kernel=in_kernel,
            nvfp4=nvfp4,
            hardware=hardware_e4m3,
        )
    if in_kernel and stateful and last is not None:
        # After the kernel (same stream): every program read the old value.
        last.index_fill_(0, flat, layer_pos)


def decode_rows(
    cfg: DeltaDispatchConfig,
    data: torch.Tensor,
    meta: torch.Tensor,
    offsets: torch.Tensor | None,
    state_rows: torch.Tensor | None,
    state: torch.Tensor | None,
    h_out: torch.Tensor,
    rows_out: torch.Tensor | None = None,
    max_tokens: int = 0,
    row_size: int = 0,
    hardware_e4m3: bool = False,
    nvfp4: bool | None = None,
) -> None:
    """Fused FFN kernel (see ``decode_rows_reference``); ``nvfp4``: whether
    rows can be NVFP4 (default ``cfg.uses_nvfp4``)."""
    rows, hidden = int(meta.shape[0]), h_out.shape[1]
    if not rows:
        return
    if h_out.stride(-1) != 1 or data.dim() != 1 or data.dtype != torch.uint8:
        raise ValueError(
            "delta dispatch needs row-contiguous h_out and flat uint8 data"
        )
    stateful = cfg.stateful
    state_arg, state_stride = _state_args(state if stateful else None, h_out)
    from_meta = state_rows is None
    if from_meta and rows_out is None:
        raise ValueError("decode_rows without state_rows writes them to rows_out")
    if offsets is None and row_size <= 0:
        raise ValueError("decode_rows without offsets needs the row size")
    _decode_rows_kernel[(rows, hidden // QUANT_GROUP_SIZE)](
        data,
        meta,
        meta if offsets is None else offsets,
        rows_out if from_meta else state_rows,
        state_arg,
        state_stride,
        h_out,
        h_out.stride(0),
        row_size,
        max_tokens,
        hidden=hidden,
        group=QUANT_GROUP_SIZE,
        stateful=stateful,
        code_shift=CODE_SHIFT,
        code_mask=CODE_MASK,
        rank_shift=RANK_SHIFT,
        token_mask=TOKEN_MASK,
        fixed=offsets is None,
        rows_from_meta=from_meta,
        nvfp4=cfg.uses_nvfp4 if nvfp4 is None else nvfp4,
        hardware=hardware_e4m3,
    )


def encode_combine(
    code: int,
    partial: torch.Tensor,
    rows: torch.Tensor,
    err: torch.Tensor | None,
    out: torch.Tensor,
    deq: torch.Tensor,
    nvfp4_pass: str = NVFP4_TWO_PASS,
    hardware_e4m3: bool = False,
) -> None:
    """Fused FFN kernel (see ``encode_combine_reference``)."""
    assert code in COMBINE_CODES, f"combine rows are {COMBINE_CODES}, not {code}"
    _check_pass(nvfp4_pass)
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
    nvfp4 = code in NVFP4_CODES
    size = row_bytes(code, hidden)
    if nvfp4 and nvfp4_pass == NVFP4_ONE_PASS:
        _encode_combine_row_kernel[(n,)](
            partial,
            partial.stride(0),
            rows,
            err_arg,
            err_stride,
            out,
            deq,
            deq.stride(0),
            size,
            hidden=hidden,
            blocks_p2=triton.next_power_of_2(hidden // NVFP4_BLOCK),
            ef=err is not None,
            hardware=hardware_e4m3,
        )
        return
    amax = deq  # read only for NVFP4 rows
    if nvfp4:
        amax = torch.empty(n, dtype=torch.float32, device=partial.device)
        _combine_amax_kernel[(n,)](
            partial,
            partial.stride(0),
            rows,
            err_arg,
            err_stride,
            amax,
            hidden=hidden,
            hidden_p2=triton.next_power_of_2(hidden),
            ef=err is not None,
        )
    _encode_combine_kernel[(n, hidden // QUANT_GROUP_SIZE)](
        partial,
        partial.stride(0),
        rows,
        err_arg,
        err_stride,
        out,
        deq,
        deq.stride(0),
        amax,
        size,
        MIN_GROUP_RANGE,
        hidden=hidden,
        group=QUANT_GROUP_SIZE,
        bits=CODE_BITS[code],
        ef=err is not None,
        nvfp4=nvfp4,
        hardware=hardware_e4m3,
    )


def decode_combine(
    code: int, data: torch.Tensor, out: torch.Tensor, hardware_e4m3: bool = False
) -> None:
    """Fused Attention kernel (see ``decode_combine_reference``)."""
    assert code in COMBINE_CODES, f"combine rows are {COMBINE_CODES}, not {code}"
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
        nvfp4=code in NVFP4_CODES,
        hardware=hardware_e4m3,
    )


__all__ = [
    "NVFP4_ONE_PASS",
    "NVFP4_PASSES",
    "NVFP4_TWO_PASS",
    "decode_combine",
    "decode_rows",
    "encode_combine",
    "encode_rows",
]
