# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch for the one-hop routed path (one FFN rank).

Instead of the MoE input ``x = Norm(h')`` the Attention side sends what the
FFN rank does not know yet about ``h'`` (the residual right after attention).
Both sides keep, per stage, the same fp32 state ``S``: the ``h'`` the FFN rank
rebuilt at the previous MoE layer plus ``routed_scaling_factor`` times the
partial sum it returned (added as soon as the sum is known). At a MoE layer

    Attention:  d = h' - S,  send Q(d),  S += deq(Q(d))
    FFN:        S += deq(Q(d)),  x = Norm(S in the model dtype)

so the FFN rank rebuilds exactly what the Attention side predicts and wire
errors do not accumulate (closed loop). The first MoE layer of a forward
sends ``h'`` in full. After the experts / the combine, both sides do
``S += routed_scaling_factor * partial_sum``.

Row formats (one per MoE layer; the receiver checks the size it got):

* ``h_bf16``: ``h'`` in bf16 (2 * hidden bytes).
* ``d_fp32``: ``d`` in fp32 (4 * hidden bytes); rebuilds ``h'`` exactly.
* ``d_int4`` / ``d_int8``: ``d`` quantized per group of ``QUANT_GROUP_SIZE``
  values, asymmetric: per group the minimum ``lo`` and the step
  ``(max - lo) / (2^bits - 1)``, both rounded to fp16, and
  ``q = round((d - lo) / step)`` clamped to ``[0, 2^bits - 1]`` (the same
  arithmetic as the offline experiment's ``compress.py``). A row is the
  packed values (int4: two per byte, even index in the low nibble) followed
  by the ``(lo, step)`` fp16 pairs of its groups: hidden / 2 + 4 * hidden /
  128 bytes for int4, hidden + 4 * hidden / 128 for int8.

Modes (``connector_extra_config = {"delta_dispatch": {"mode": ...}}``):

* ``off``: the existing path (``x`` in the wire dtype), untouched.
* ``full_bf16``: every MoE layer sends ``h_bf16`` (checks the FFN-side norm
  and the byte-row wire).
* ``delta_fp32``: first MoE layer ``h_bf16``, the others ``d_fp32`` (checks
  the states and the closed loop; outputs must equal ``off``).
* ``delta_int4``: first MoE layer ``h_bf16``; the ``early`` first MoE layers
  (the first one included) and the ``late`` last ones ``d_int8``; the others
  ``d_int4``.

The quantized formats run one fused Triton kernel per side and layer on CUDA
tensors (``delta_dispatch_triton.py``) and the torch reference below on other
devices (unit tests). Requirements (checked by the callers): one FFN rank,
bf16 model, Attention TP 1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from afd_plugin.config import connector_extra_config_from_source

MODES = ("off", "full_bf16", "delta_fp32", "delta_int4")
EXTRA_CONFIG_KEY = "delta_dispatch"
QUANT_GROUP_SIZE = 128
# Bytes of the (lo, step) fp16 pair stored per group.
GROUP_SCALE_BYTES = 2 * torch.float16.itemsize
# Below this range a group is constant: every value is lo (step 0).
MIN_GROUP_RANGE = 1e-12
FORMAT_BITS = {"d_int4": 4, "d_int8": 8}
FORMAT_DTYPES = {"h_bf16": torch.bfloat16, "d_fp32": torch.float32}
BITS_PER_BYTE = 8


@dataclass(frozen=True)
class DeltaDispatchConfig:
    mode: str = "off"
    early: int = 0
    late: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeltaDispatchConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("delta_dispatch must be a mapping")
        unknown = set(raw) - {"mode", "early", "late"}
        if unknown:
            raise ValueError(f"unknown delta_dispatch keys: {sorted(unknown)}")
        mode = str(raw.get("mode", "off"))
        if mode not in MODES:
            raise ValueError(f"delta_dispatch mode {mode!r} not in {MODES}")
        early, late = int(raw.get("early", 0)), int(raw.get("late", 0))
        if early < 0 or late < 0:
            raise ValueError("delta_dispatch early and late must be >= 0")
        if (early or late) and mode != "delta_int4":
            raise ValueError("delta_dispatch early/late apply to delta_int4 only")
        return cls(mode=mode, early=early, late=late)

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any] | None) -> DeltaDispatchConfig:
        """From the connector extra config mapping of the AFD config."""
        return cls.from_mapping((extra or {}).get(EXTRA_CONFIG_KEY))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> DeltaDispatchConfig:
        """From ``additional_config["afd"]["connector_extra_config"]`` of
        ``vllm_config`` (the same source on every rank)."""
        additional_config = getattr(vllm_config, "additional_config", None)
        return cls.from_extra_config(
            connector_extra_config_from_source(additional_config)
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def stateful(self) -> bool:
        return self.mode in ("delta_fp32", "delta_int4")

    def row_format(self, layer_pos: int, num_layers: int) -> str:
        """Format of the ``layer_pos``-th of ``num_layers`` MoE layers (0 = first)."""
        if self.mode == "full_bf16" or layer_pos == 0:
            return "h_bf16"
        if self.mode == "delta_fp32":
            return "d_fp32"
        if self.mode == "delta_int4":
            if layer_pos < self.early or layer_pos >= num_layers - self.late:
                return "d_int8"
            return "d_int4"
        raise RuntimeError(f"no row format in mode {self.mode!r}")

    def max_row_bytes(self, hidden: int) -> int:
        fmts = {
            "full_bf16": ("h_bf16",),
            "delta_fp32": ("h_bf16", "d_fp32"),
            "delta_int4": ("h_bf16", "d_int4", "d_int8"),
        }[self.mode]
        return max(row_bytes(f, hidden) for f in fmts)


def row_bytes(fmt: str, hidden: int) -> int:
    if fmt in FORMAT_BITS:
        if hidden % QUANT_GROUP_SIZE:
            raise ValueError(f"hidden {hidden} is not a multiple of {QUANT_GROUP_SIZE}")
        data = hidden * FORMAT_BITS[fmt] // BITS_PER_BYTE
        return data + hidden // QUANT_GROUP_SIZE * GROUP_SCALE_BYTES
    return FORMAT_DTYPES[fmt].itemsize * hidden


def _check_rows(fmt: str, rows: torch.Tensor, hidden: int) -> None:
    expected = row_bytes(fmt, hidden)
    if rows.dtype != torch.uint8 or rows.dim() != 2 or rows.shape[1] != expected:
        raise ValueError(
            f"{fmt} rows must be uint8 [*, {expected}], "
            f"got {rows.dtype} {tuple(rows.shape)}",
        )


def encode_plain(fmt: str, value: torch.Tensor) -> torch.Tensor:
    """``h_bf16`` / ``d_fp32``: ``value`` [T, hidden] -> uint8 rows (a view
    when ``value`` already has the format's dtype and is contiguous)."""
    dtype = FORMAT_DTYPES[fmt]
    v = value if value.dtype == dtype else value.to(dtype)
    return v.contiguous().view(torch.uint8)


def decode_plain(fmt: str, rows: torch.Tensor, hidden: int) -> torch.Tensor:
    """``h_bf16`` / ``d_fp32`` rows -> the value [T, hidden] in fp32."""
    _check_rows(fmt, rows, hidden)
    return rows.contiguous().view(FORMAT_DTYPES[fmt]).float()


# ----------------------------------------------------------- quantized rows
def _quantize_groups(
    d: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """d fp32 [T, G, QUANT_GROUP_SIZE] -> (q uint8, lo fp16, step fp16)."""
    levels = (1 << bits) - 1
    lo = d.amin(-1, keepdim=True)
    hi = d.amax(-1, keepdim=True)
    step = ((hi - lo).clamp_min(MIN_GROUP_RANGE) / levels).half()
    lo = lo.half()
    stepf = step.float()
    q = torch.where(
        stepf > 0,
        torch.round((d - lo.float()) / stepf.clamp_min(MIN_GROUP_RANGE)),
        torch.zeros_like(d),
    )
    return q.clamp(0, levels).to(torch.uint8), lo, step


def _dequantize_groups(
    q: torch.Tensor, lo: torch.Tensor, step: torch.Tensor
) -> torch.Tensor:
    return q.float() * step.float() + lo.float()


def encode_delta_reference(
    fmt: str, h_prime: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    """Torch reference of the fused Attention kernel: rows of Q(h' - S) and
    ``S += deq(Q(h' - S))`` in place. ``state`` fp32 [T, hidden]."""
    bits = FORMAT_BITS[fmt]
    tokens, hidden = state.shape
    groups = hidden // QUANT_GROUP_SIZE
    d = (h_prime.float() - state).view(tokens, groups, QUANT_GROUP_SIZE)
    q, lo, step = _quantize_groups(d, bits)
    state += _dequantize_groups(q, lo, step).view(tokens, hidden)
    q = q.view(tokens, hidden)
    if bits == 4:
        q = q[:, 0::2] | (q[:, 1::2] << 4)
    scales = torch.cat([lo, step], dim=-1).view(tokens, groups * 2)
    return torch.cat([q, scales.view(torch.uint8)], dim=1)


def decode_delta_reference(fmt: str, rows: torch.Tensor, state: torch.Tensor) -> None:
    """Torch reference of the fused FFN kernel: ``S += deq(rows)`` in place."""
    bits = FORMAT_BITS[fmt]
    tokens, hidden = state.shape
    _check_rows(fmt, rows, hidden)
    groups = hidden // QUANT_GROUP_SIZE
    data = hidden * bits // BITS_PER_BYTE
    packed, scales = rows[:, :data], rows[:, data:].contiguous().view(torch.float16)
    if bits == 4:
        q = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(tokens, hidden)
    else:
        q = packed
    scales = scales.view(tokens, groups, 2)
    lo, step = scales[..., :1], scales[..., 1:]
    deq = _dequantize_groups(q.view(tokens, groups, QUANT_GROUP_SIZE), lo, step)
    state += deq.view(tokens, hidden)


def encode_delta(fmt: str, h_prime: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if state.is_cuda:
        # Lazy import: Triton is only needed (and installed) with CUDA.
        from afd_plugin import delta_dispatch_triton

        return delta_dispatch_triton.encode_delta(fmt, h_prime, state)
    return encode_delta_reference(fmt, h_prime, state)


def decode_delta(
    fmt: str, rows: torch.Tensor, state: torch.Tensor, h_out: torch.Tensor
) -> None:
    """``S += deq(rows)`` and ``h_out = S`` in ``h_out``'s dtype."""
    if state.is_cuda:
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.decode_delta(fmt, rows, state, h_out)
        return
    decode_delta_reference(fmt, rows, state)
    h_out.copy_(state)


# ------------------------------------------------------------------ states
class _LayerOrder:
    """MoE layer index -> position in the forward (0 = first MoE layer)."""

    def __init__(self, moe_layers: Sequence[int]) -> None:
        order = sorted(int(i) for i in moe_layers)
        if not order:
            raise ValueError("delta dispatch needs the MoE layer indices")
        self.pos = {idx: p for p, idx in enumerate(order)}
        self.count = len(order)

    def format(self, cfg: DeltaDispatchConfig, layer_idx: int) -> str:
        try:
            pos = self.pos[int(layer_idx)]
        except KeyError:
            raise RuntimeError(f"layer {layer_idx} is not a known MoE layer") from None
        return cfg.row_format(pos, self.count)


class AttentionDeltaState:
    """Attention side: per stage, the state ``S`` of the FFN rank."""

    def __init__(
        self, cfg: DeltaDispatchConfig, moe_layers: Sequence[int], routed_scale: float
    ) -> None:
        self.cfg, self.layers, self.scale = cfg, _LayerOrder(moe_layers), routed_scale
        self._state: dict[int, torch.Tensor] = {}  # stage -> fp32 [T, hidden]

    def encode_dispatch(
        self, layer_idx: int, stage: int, h_prime: torch.Tensor
    ) -> torch.Tensor:
        """Rows to send for ``h_prime`` (the residual after attention, [T, hidden])."""
        fmt = self.layers.format(self.cfg, layer_idx)
        if fmt == "h_bf16":
            if self.cfg.stateful:
                self._state[stage] = h_prime.float()
            return encode_plain(fmt, h_prime)
        state = self._state.get(stage)
        if state is None or state.shape != h_prime.shape:
            raise RuntimeError(f"stage {stage}: no FFN state before layer {layer_idx}")
        if fmt == "d_fp32":
            d = h_prime.float() - state
            state += d
            return encode_plain(fmt, d)
        return encode_delta(fmt, h_prime, state)

    def after_combine(
        self, layer_idx: int, stage: int, routed_output: torch.Tensor
    ) -> None:
        """Add the partial sum the FFN rank returned (before scaling)."""
        if self.cfg.stateful:
            self._state[stage].add_(routed_output, alpha=self.scale)


class FFNDeltaState:
    """FFN side (one rank): per stage, the state ``S``."""

    def __init__(
        self,
        cfg: DeltaDispatchConfig,
        moe_layers: Sequence[int],
        routed_scale: float,
        hidden: int,
    ) -> None:
        self.cfg, self.layers, self.scale = cfg, _LayerOrder(moe_layers), routed_scale
        self.hidden = int(hidden)
        self._state: dict[int, torch.Tensor] = {}
        # stage -> h' in the model dtype, and the buffer handed to the experts.
        self._h: dict[int, torch.Tensor] = {}
        self._x: dict[int, torch.Tensor] = {}

    def expected_row_bytes(self, layer_idx: int) -> int:
        return row_bytes(self.layers.format(self.cfg, layer_idx), self.hidden)

    def _buffer(
        self,
        store: dict[int, torch.Tensor],
        stage: int,
        rows: int,
        like: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = store.get(stage)
        if buf is None or buf.shape[0] < rows or buf.dtype != dtype:
            # Zeros: the launch rows past the received ones stay finite.
            buf = torch.zeros((rows, self.hidden), dtype=dtype, device=like.device)
            store[stage] = buf
        return buf

    def decode_dispatch(
        self,
        layer_idx: int,
        stage: int,
        rows: torch.Tensor,
        norm_into: Callable[[torch.Tensor, torch.Tensor], object],
        out_rows: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Rebuild ``h'`` from ``rows`` (uint8 [n, row_bytes]) and return the
        experts' input ``Norm(h')`` in ``dtype`` with ``out_rows`` >= n rows
        (the experts kernel's row count; rows past n are not real and left as
        they are). ``norm_into(out, h)`` writes ``Norm(h)`` into ``out``."""
        n = int(rows.shape[0])
        fmt = self.layers.format(self.cfg, layer_idx)
        x = self._buffer(self._x, stage, out_rows, rows, dtype)
        if fmt == "h_bf16":
            _check_rows(fmt, rows, self.hidden)
            h = rows.view(torch.bfloat16)
            if self.cfg.stateful:
                self._state[stage] = h.float()
            if dtype != torch.bfloat16:
                h = h.to(dtype)
        else:
            state = self._state.get(stage)
            if state is None or state.shape[0] != n:
                raise RuntimeError(
                    f"stage {stage}: no FFN state before layer {layer_idx}"
                )
            h = self._buffer(self._h, stage, n, rows, dtype)[:n]
            if fmt == "d_fp32":
                state += decode_plain(fmt, rows, self.hidden)
                h.copy_(state)
            else:
                decode_delta(fmt, rows, state, h)
        if n:
            norm_into(x[:n], h)
        return x[:out_rows]

    def after_experts(self, layer_idx: int, stage: int, output: torch.Tensor) -> None:
        """Add the partial sum returned for this layer (its first n rows)."""
        if self.cfg.stateful:
            state = self._state[stage]
            state.add_(output[: state.shape[0]], alpha=self.scale)


__all__ = [
    "EXTRA_CONFIG_KEY",
    "MODES",
    "QUANT_GROUP_SIZE",
    "AttentionDeltaState",
    "DeltaDispatchConfig",
    "FFNDeltaState",
    "decode_delta_reference",
    "decode_plain",
    "encode_delta_reference",
    "encode_plain",
    "row_bytes",
]
