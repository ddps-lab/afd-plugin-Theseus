# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch for the one-hop routed path (one FFN rank).

Instead of the MoE input ``x = Norm(h')`` the Attention side sends what the
FFN rank does not know yet about ``h'`` (the residual right after attention):
the FFN rank keeps, per stage, the ``h'`` it reconstructed at the previous
MoE layer and the partial sum it returned, so it predicts

    pred_l = h'_{l-1} + routed_scaling_factor * out_{l-1}

and the Attention side, which can compute the same prediction, sends
``d_l = h'_l - pred_l`` (closed loop: what the FFN side reconstructs is fed
back into the next prediction, so wire errors do not accumulate). The FFN
side reconstructs ``h'_l = pred_l + d_l`` and applies the layer's
``post_attention_layernorm`` itself. The first MoE layer of a forward sends
``h'`` in full.

Row formats (one per MoE layer, chosen by the mode; the receiver checks the
row size it got against the one it expects):

* ``h_bf16``: ``h'`` in bf16 (2 * hidden bytes), no state needed.
* ``d_fp32``: ``d`` in fp32 (4 * hidden bytes); reconstructs ``h'`` exactly.

Modes (``connector_extra_config = {"delta_dispatch": {"mode": ...}}``):

* ``off``: the existing path (``x`` in the wire dtype), untouched.
* ``full_bf16``: every MoE layer sends ``h_bf16``; checks the FFN-side norm
  and the byte-row wire.
* ``delta_fp32``: first MoE layer ``h_bf16``, the others ``d_fp32``; checks
  the kept states and the closed loop. Outputs must equal ``off``.

Only plain torch is used here so the arithmetic can be unit-tested without
vLLM. Requirements (checked by the callers): one FFN rank, bf16 model,
Attention TP 1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

MODES = ("off", "full_bf16", "delta_fp32")
FORMAT_DTYPES = {"h_bf16": torch.bfloat16, "d_fp32": torch.float32}
EXTRA_CONFIG_KEY = "delta_dispatch"


@dataclass(frozen=True)
class DeltaDispatchConfig:
    mode: str = "off"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeltaDispatchConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("delta_dispatch must be a mapping")
        unknown = set(raw) - {"mode"}
        if unknown:
            raise ValueError(f"unknown delta_dispatch keys: {sorted(unknown)}")
        mode = str(raw.get("mode", "off"))
        if mode not in MODES:
            raise ValueError(f"delta_dispatch mode {mode!r} not in {MODES}")
        return cls(mode=mode)

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any] | None) -> DeltaDispatchConfig:
        """From ``AFDConfig.connector_extra_config`` (the same source on every rank)."""
        return cls.from_mapping((extra or {}).get(EXTRA_CONFIG_KEY))

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def row_format(self, layer_pos: int) -> str:
        """Format of the rows of the ``layer_pos``-th MoE layer (0 = first)."""
        if self.mode == "full_bf16" or layer_pos == 0:
            return "h_bf16"
        if self.mode == "delta_fp32":
            return "d_fp32"
        raise RuntimeError(f"no row format in mode {self.mode!r}")

    def max_row_bytes(self, hidden: int) -> int:
        fmts = ("h_bf16",) if self.mode == "full_bf16" else tuple(FORMAT_DTYPES)
        return max(row_bytes(f, hidden) for f in fmts)


def row_bytes(fmt: str, hidden: int) -> int:
    return FORMAT_DTYPES[fmt].itemsize * hidden


def encode_rows(fmt: str, value: torch.Tensor) -> torch.Tensor:
    """``value`` [T, hidden] -> the wire rows as uint8 [T, row_bytes] (a view
    when ``value`` already has the format's dtype and is contiguous)."""
    dtype = FORMAT_DTYPES[fmt]
    v = value if value.dtype == dtype else value.to(dtype)
    return v.contiguous().view(torch.uint8)


def decode_rows(fmt: str, rows: torch.Tensor, hidden: int) -> torch.Tensor:
    """uint8 rows [T, row_bytes] -> the value [T, hidden] in fp32."""
    expected = row_bytes(fmt, hidden)
    if rows.dtype != torch.uint8 or rows.shape[-1] != expected:
        raise ValueError(
            f"{fmt} rows must be uint8 [*, {expected}], "
            f"got {rows.dtype} {tuple(rows.shape)}",
        )
    return rows.contiguous().view(FORMAT_DTYPES[fmt]).float()


class _LayerOrder:
    """MoE layer index -> position in the forward (0 = first MoE layer)."""

    def __init__(self, moe_layers: Sequence[int]) -> None:
        order = sorted(int(i) for i in moe_layers)
        if not order:
            raise ValueError("delta dispatch needs the MoE layer indices")
        self.pos = {idx: p for p, idx in enumerate(order)}

    def __call__(self, layer_idx: int) -> int:
        try:
            return self.pos[int(layer_idx)]
        except KeyError:
            raise RuntimeError(f"layer {layer_idx} is not a known MoE layer") from None


class AttentionDeltaState:
    """Attention side: per stage, the FFN rank's ``h'`` and its last partial sum."""

    def __init__(
        self, cfg: DeltaDispatchConfig, moe_layers: Sequence[int], routed_scale: float
    ) -> None:
        self.cfg, self.layer_pos, self.scale = (
            cfg,
            _LayerOrder(moe_layers),
            float(routed_scale),
        )
        self._h: dict[int, torch.Tensor] = {}  # stage -> fp32 [T, hidden]
        self._out: dict[
            int, torch.Tensor
        ] = {}  # stage -> partial sum [T, hidden] (model dtype)

    def encode_dispatch(
        self, layer_idx: int, stage: int, h_prime: torch.Tensor
    ) -> torch.Tensor:
        """Rows to send for ``h_prime`` (the residual after attention, [T, hidden])."""
        fmt = self.cfg.row_format(self.layer_pos(layer_idx))
        if fmt == "h_bf16":
            if self.cfg.mode == "delta_fp32":
                self._h[stage] = h_prime.float()
            return encode_rows(fmt, h_prime)
        h, out = self._h.get(stage), self._out.get(stage)
        if h is None or out is None or h.shape != h_prime.shape:
            raise RuntimeError(f"stage {stage}: no FFN state before layer {layer_idx}")
        pred = h + self.scale * out.float()
        d = h_prime.float() - pred
        # What the FFN side reconstructs (same expression on both sides).
        self._h[stage] = pred + d
        return encode_rows(fmt, d)

    def after_combine(
        self, layer_idx: int, stage: int, routed_output: torch.Tensor
    ) -> None:
        """Record the partial sum the FFN rank returned (before scaling)."""
        if self.cfg.mode == "delta_fp32":
            self._out[stage] = routed_output


class FFNDeltaState:
    """FFN side (one rank): per stage, the reconstructed ``h'`` and the partial sum."""

    def __init__(
        self,
        cfg: DeltaDispatchConfig,
        moe_layers: Sequence[int],
        routed_scale: float,
        hidden: int,
    ) -> None:
        self.cfg, self.layer_pos, self.scale = (
            cfg,
            _LayerOrder(moe_layers),
            float(routed_scale),
        )
        self.hidden = int(hidden)
        self._h: dict[int, torch.Tensor] = {}
        self._out: dict[int, torch.Tensor] = {}
        self._x: dict[
            int, torch.Tensor
        ] = {}  # stage -> model-dtype buffer handed to the experts

    def expected_row_bytes(self, layer_idx: int) -> int:
        return row_bytes(self.cfg.row_format(self.layer_pos(layer_idx)), self.hidden)

    def decode_dispatch(
        self,
        layer_idx: int,
        stage: int,
        rows: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
        out_rows: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct ``h'`` from ``rows`` (uint8 [n, row_bytes]) and return
        ``norm(h')`` in ``dtype`` with ``out_rows`` >= n rows (the experts
        kernel's row count; rows past n are not real and left as they are)."""
        n = int(rows.shape[0])
        fmt = self.cfg.row_format(self.layer_pos(layer_idx))
        value = decode_rows(fmt, rows, self.hidden)
        if fmt == "h_bf16":
            h = value
        else:
            h_prev, out = self._h.get(stage), self._out.get(stage)
            if h_prev is None or out is None or h_prev.shape[0] != n:
                raise RuntimeError(
                    f"stage {stage}: no FFN state before layer {layer_idx}"
                )
            h = (h_prev + self.scale * out.float()) + value
        if self.cfg.mode == "delta_fp32":
            self._h[stage] = h
        buf = self._x.get(stage)
        if buf is None or buf.shape[0] < out_rows or buf.dtype != dtype:
            buf = torch.empty((out_rows, self.hidden), dtype=dtype, device=rows.device)
            self._x[stage] = buf
        if n:
            buf[:n] = norm(h.to(dtype))
        return buf[:out_rows]

    def after_experts(self, layer_idx: int, stage: int, output: torch.Tensor) -> None:
        """Keep the partial sum returned for this layer (its first n rows)."""
        if self.cfg.mode == "delta_fp32":
            h = self._h[stage]
            self._out[stage] = output[: h.shape[0]].clone()


__all__ = [
    "EXTRA_CONFIG_KEY",
    "AttentionDeltaState",
    "DeltaDispatchConfig",
    "FFNDeltaState",
    "MODES",
    "decode_rows",
    "encode_rows",
    "row_bytes",
]
