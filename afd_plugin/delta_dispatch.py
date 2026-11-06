# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch for the one-hop routed path (any number of FFN ranks).

Instead of the MoE input ``x = Norm(h')`` the Attention side sends, for every
(token, FFN rank) row of the dispatch, what that FFN rank does not know yet
about ``h'`` (the residual right after attention). Every FFN rank keeps, per
(Attention rank, stage, token), a fp32 state ``S``: the ``h'`` it rebuilt the
last time the token visited it in this forward plus ``routed_scaling_factor``
times the partial sum it returned then. The Attention side keeps a copy of
every FFN rank's states. At a MoE layer, for a row of FFN rank ``f``:

    first visit of the forward:  send h' in bf16,              S_f = h'
    later visit:                 d = h' - S_f, send Q(d),      S_f += deq(Q(d))

the FFN rank normalizes ``S_f`` itself (``x = Norm(S_f in bf16)``), and after
the experts / before the combine sums the partials both sides do
``S_f += routed_scaling_factor * partial sum of f``. The state is rebuilt from
what was sent (closed loop), so wire errors do not accumulate. A token that
does not visit ``f`` at a layer keeps its older ``S_f``; the next visit's
difference also carries what changed in between.

Rows, one per (token, FFN rank) pair in the dispatch order (FFN rank major,
token order inside, as ``P2pAll2AllEngine.begin_plan`` orders them), are
packed back to back at the size of their format code:

* ``0`` ``h_bf16``: ``h'`` in bf16 (2 * hidden bytes).
* ``1`` ``d_fp32``: ``d`` in fp32 (4 * hidden bytes); rebuilds ``h'`` exactly.
* ``2`` ``d_int8`` / ``3`` ``d_int4``: ``d`` quantized per group of
  ``QUANT_GROUP_SIZE`` values, asymmetric: per group the minimum ``lo`` and the
  step ``(max - lo) / (2^bits - 1)``, both rounded to fp16, and
  ``q = round((d - lo) / step)`` clamped to ``[0, 2^bits - 1]`` (the offline
  experiment's ``compress.py`` arithmetic). A row is the packed values (int4:
  two per byte, even index in the low nibble) followed by the ``(lo, step)``
  fp16 pairs of its groups.

Each row also carries a 4-byte word ``token | code << CODE_SHIFT | rank <<
RANK_SHIFT``: the token's index in the Attention rank's batch, the row's format
and the sending Attention rank (so the FFN rank finds the state without the
per-rank row counts on the GPU).

Codes by mode (``connector_extra_config = {"delta_dispatch": {...}}``):

* ``off``: the existing path (``x`` in the wire dtype), untouched.
* ``full_bf16``: every row ``h_bf16``, no state (checks the FFN-side norm and
  the byte rows).
* ``delta_fp32``: first visit ``h_bf16``, then ``d_fp32`` (checks the states
  and the closed loop; outputs must equal ``off``).
* ``delta_int4``: first visit ``h_bf16``; then ``d_int8`` on the ``early``
  first and ``late`` last MoE layers and, with ``revisit: "int8"``, on rows
  whose last visit to the FFN rank was ``rgap`` or more MoE layers before;
  else ``d_int4`` (the rules of the offline experiment's ``sim.FFNSim``).
* ``x_int4``: no state: every row is ``x = Norm(h')`` in int4 and the FFN rank
  uses the dequantized ``x`` as is (plain quantization, the baseline).

The fused Triton kernels (``delta_dispatch_triton.py``) run on CUDA tensors and
the torch reference below elsewhere (unit tests). Requirements (checked by the
callers): bf16 model, Attention TP 1, ``--enforce-eager``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from afd_plugin.config import connector_extra_config_from_source

MODES = ("off", "full_bf16", "delta_fp32", "delta_int4", "x_int4")
REVISIT_FORMATS = ("int4", "int8")
EXTRA_CONFIG_KEY = "delta_dispatch"
H_BF16, D_FP32, D_INT8, D_INT4 = 0, 1, 2, 3
CODE_FORMATS = ("h_bf16", "d_fp32", "d_int8", "d_int4")
CODE_BITS = {D_INT8: 8, D_INT4: 4}
CODE_SHIFT = 24
CODE_MASK = 0x3
RANK_SHIFT = 26
MAX_ATTENTION_RANKS = 1 << (31 - RANK_SHIFT)
TOKEN_MASK = (1 << CODE_SHIFT) - 1
QUANT_GROUP_SIZE = 128
# Bytes of the (lo, step) fp16 pair stored per group.
GROUP_SCALE_BYTES = 2 * torch.float16.itemsize
# Below this range a group is constant: every value is lo (step 0).
MIN_GROUP_RANGE = 1e-12
BITS_PER_BYTE = 8


def row_bytes(code: int, hidden: int) -> int:
    """Bytes of one row of format ``code``."""
    if code == H_BF16:
        return 2 * hidden
    if code == D_FP32:
        return 4 * hidden
    if hidden % QUANT_GROUP_SIZE:
        raise ValueError(f"hidden {hidden} is not a multiple of {QUANT_GROUP_SIZE}")
    data = hidden * CODE_BITS[code] // BITS_PER_BYTE
    return data + hidden // QUANT_GROUP_SIZE * GROUP_SCALE_BYTES


def row_bytes_table(hidden: int, device: torch.device | str) -> torch.Tensor:
    """int64 [4]: bytes of a row per format code."""
    return torch.tensor(
        [row_bytes(code, hidden) for code in range(len(CODE_FORMATS))],
        dtype=torch.int64,
        device=device,
    )


@dataclass(frozen=True)
class DeltaDispatchConfig:
    mode: str = "off"
    early: int = 0
    late: int = 0
    revisit: str = "int4"
    rgap: int = 2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeltaDispatchConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("delta_dispatch must be a mapping")
        unknown = set(raw) - {"mode", "early", "late", "revisit", "rgap"}
        if unknown:
            raise ValueError(f"unknown delta_dispatch keys: {sorted(unknown)}")
        mode = str(raw.get("mode", "off"))
        if mode not in MODES:
            raise ValueError(f"delta_dispatch mode {mode!r} not in {MODES}")
        early, late = int(raw.get("early", 0)), int(raw.get("late", 0))
        revisit, rgap = str(raw.get("revisit", "int4")), int(raw.get("rgap", 2))
        if early < 0 or late < 0:
            raise ValueError("delta_dispatch early and late must be >= 0")
        if revisit not in REVISIT_FORMATS:
            raise ValueError(
                f"delta_dispatch revisit {revisit!r} not in {REVISIT_FORMATS}"
            )
        if rgap < 2:
            raise ValueError("delta_dispatch rgap must be >= 2")
        rules = early or late or revisit != "int4" or rgap != 2
        if rules and mode != "delta_int4":
            raise ValueError(
                "delta_dispatch early/late/revisit/rgap apply to delta_int4 only"
            )
        return cls(mode=mode, early=early, late=late, revisit=revisit, rgap=rgap)

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
        """Both sides keep the per-FFN-rank states (the delta modes)."""
        return self.mode in ("delta_fp32", "delta_int4")

    @property
    def sends_x(self) -> bool:
        """Rows are the MoE input x itself (no state, no FFN-side norm)."""
        return self.mode == "x_int4"

    def row_codes(
        self, layer_pos: int, num_layers: int, last: torch.Tensor
    ) -> torch.Tensor:
        """Format code of every row at the ``layer_pos``-th of ``num_layers``
        MoE layers; ``last`` is the MoE layer position of the row's previous
        visit to its FFN rank in this forward (``< 0``: first visit)."""
        if self.mode == "full_bf16":
            return torch.full_like(last, H_BF16)
        if self.mode == "x_int4":
            return torch.full_like(last, D_INT4)
        if self.mode == "delta_fp32":
            rest = torch.full_like(last, D_FP32)
        elif layer_pos < self.early or layer_pos >= num_layers - self.late:
            rest = torch.full_like(last, D_INT8)
        elif self.revisit == "int8":
            rest = torch.where(layer_pos - last >= self.rgap, D_INT8, D_INT4).to(
                last.dtype
            )
        else:
            rest = torch.full_like(last, D_INT4)
        return torch.where(last < 0, H_BF16, rest).to(last.dtype)

    def max_row_bytes(self, hidden: int) -> int:
        codes = {
            "full_bf16": (H_BF16,),
            "delta_fp32": (H_BF16, D_FP32),
            "delta_int4": (H_BF16, D_INT8, D_INT4),
            "x_int4": (D_INT4,),
        }[self.mode]
        return max(row_bytes(c, hidden) for c in codes)


# ------------------------------------------------------- torch reference
def _quantize_groups(
    d: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """d fp32 [R, G, QUANT_GROUP_SIZE] -> (q uint8, lo fp16, step fp16)."""
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


def _quantized_rows(code: int, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """d fp32 [R, hidden] -> (uint8 rows [R, row_bytes], deq fp32 [R, hidden])."""
    bits = CODE_BITS[code]
    rows, hidden = d.shape
    groups = hidden // QUANT_GROUP_SIZE
    q, lo, step = _quantize_groups(d.view(rows, groups, QUANT_GROUP_SIZE), bits)
    deq = _dequantize_groups(q, lo, step).view(rows, hidden)
    q = q.view(rows, hidden)
    if bits == 4:
        q = q[:, 0::2] | (q[:, 1::2] << 4)
    scales = torch.cat([lo, step], dim=-1).view(rows, groups * 2)
    return torch.cat([q, scales.view(torch.uint8)], dim=1), deq


def _dequantized_rows(code: int, rows: torch.Tensor, hidden: int) -> torch.Tensor:
    """uint8 rows [R, row_bytes] of an int format -> deq fp32 [R, hidden]."""
    bits = CODE_BITS[code]
    n = rows.shape[0]
    groups = hidden // QUANT_GROUP_SIZE
    data = hidden * bits // BITS_PER_BYTE
    packed, scales = rows[:, :data], rows[:, data:].contiguous().view(torch.float16)
    if bits == 4:
        q = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, hidden)
    else:
        q = packed
    scales = scales.view(n, groups, 2)
    lo, step = scales[..., :1], scales[..., 1:]
    deq = _dequantize_groups(q.view(n, groups, QUANT_GROUP_SIZE), lo, step)
    return deq.view(n, hidden)


def _byte_positions(offsets: torch.Tensor, size: int) -> torch.Tensor:
    return offsets.unsqueeze(1) + torch.arange(size, device=offsets.device)


def encode_rows_reference(
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
    """Torch reference of the fused Attention kernel for ``len(flat)`` rows:
    row ``i`` is token ``flat[i] % num_tokens`` of ``src`` against state row
    ``flat[i]`` in format ``codes[i]``, written to ``out`` (uint8, flat) at
    ``offsets[i]``; the states (and ``last``) are updated in place."""
    hidden = src.shape[1]
    tokens = flat % num_tokens
    for code in range(len(CODE_FORMATS)):
        sel = (codes == code).nonzero().view(-1)
        if not sel.numel():
            continue
        rows_flat, h = flat[sel], src[tokens[sel]]
        if code == H_BF16:
            data = h.to(torch.bfloat16).contiguous().view(torch.uint8)
            if cfg.stateful:
                assert state is not None
                state[rows_flat] = h.to(torch.bfloat16).float()
        else:
            base = (
                state[rows_flat]
                if cfg.stateful and state is not None
                else torch.zeros(len(sel), hidden, device=src.device)
            )
            d = h.float() - base
            if code == D_FP32:
                data, deq = d.contiguous().view(torch.uint8), d
            else:
                data, deq = _quantized_rows(code, d)
            if cfg.stateful and state is not None:
                state[rows_flat] = base + deq
        out[_byte_positions(offsets[sel], data.shape[1])] = data
    if cfg.stateful and last is not None:
        last[flat] = layer_pos
    meta.copy_(
        tokens.to(torch.int32)
        | (codes.to(torch.int32) << CODE_SHIFT)
        | (rank << RANK_SHIFT)
    )


def decode_rows_reference(
    cfg: DeltaDispatchConfig,
    data: torch.Tensor,
    meta: torch.Tensor,
    offsets: torch.Tensor,
    state_rows: torch.Tensor,
    state: torch.Tensor | None,
    h_out: torch.Tensor,
) -> None:
    """Torch reference of the fused FFN kernel: rebuild every row's value from
    the packed bytes into ``h_out`` (``h'``, or ``x`` when ``cfg.sends_x``) and
    update the states (rows ``state_rows``) in place."""
    hidden = h_out.shape[1]
    codes = (meta >> CODE_SHIFT) & CODE_MASK
    for code in range(len(CODE_FORMATS)):
        sel = (codes == code).nonzero().view(-1)
        if not sel.numel():
            continue
        size = row_bytes(code, hidden)
        rows = data[_byte_positions(offsets[sel], size)]
        if code == H_BF16:
            value = rows.contiguous().view(torch.bfloat16).float()
            if cfg.stateful:
                assert state is not None
                state[state_rows[sel]] = value
        else:
            if code == D_FP32:
                deq = rows.contiguous().view(torch.float32)
            else:
                deq = _dequantized_rows(code, rows, hidden)
            if cfg.stateful:
                assert state is not None
                value = state[state_rows[sel]] + deq
                state[state_rows[sel]] = value
            else:
                value = deq
        h_out[sel] = value.to(h_out.dtype)


def encode_rows(cfg: DeltaDispatchConfig, *args: Any) -> None:
    if args[0].is_cuda:
        # Lazy import: Triton is only needed (and installed) with CUDA.
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.encode_rows(cfg, *args)
        return
    encode_rows_reference(cfg, *args)


def decode_rows(cfg: DeltaDispatchConfig, *args: Any) -> None:
    if args[0].is_cuda:
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.decode_rows(cfg, *args)
        return
    decode_rows_reference(cfg, *args)


# ------------------------------------------------------------------ states
class _LayerOrder:
    """MoE layer index -> position in the forward (0 = first MoE layer)."""

    def __init__(self, moe_layers: Sequence[int]) -> None:
        order = sorted(int(i) for i in moe_layers)
        if not order:
            raise ValueError("delta dispatch needs the MoE layer indices")
        self.pos = {idx: p for p, idx in enumerate(order)}
        self.count = len(order)

    def position(self, layer_idx: int) -> int:
        try:
            return self.pos[int(layer_idx)]
        except KeyError:
            raise RuntimeError(f"layer {layer_idx} is not a known MoE layer") from None


@dataclass
class PackedDispatch:
    """Attention side: one stage's dispatch of one MoE layer, between
    ``prepare`` (formats, sizes) and ``encode`` (the bytes)."""

    stage: int
    layer_pos: int
    num_tokens: int
    # Per padded dispatch entry: the state row (FFN rank * num_tokens +
    # token), the format code and the byte offset in the packed block.
    flat: torch.Tensor
    codes: torch.Tensor
    offsets: torch.Tensor
    # Bytes per FFN rank, copied to host memory without waiting (readable
    # once the stream has passed ``prepare``).
    totals: torch.Tensor


class AttentionDeltaState:
    """Attention side (one rank): per stage, the copy of every FFN rank's
    states and the MoE layer position of each token's last visit to it."""

    def __init__(
        self,
        cfg: DeltaDispatchConfig,
        moe_layers: Sequence[int],
        routed_scale: float,
        num_ffn: int,
        max_tokens: int,
        hidden: int,
    ) -> None:
        if max_tokens > TOKEN_MASK + 1:
            raise ValueError(f"delta dispatch words hold tokens < {TOKEN_MASK + 1}")
        self.cfg, self.layers, self.scale = cfg, _LayerOrder(moe_layers), routed_scale
        self.num_ffn, self.max_tokens, self.hidden = num_ffn, max_tokens, hidden
        # stage -> fp32 [num_ffn * max_tokens, hidden], int32 [num_ffn * max_tokens]
        self._state: dict[int, torch.Tensor] = {}
        self._last: dict[int, torch.Tensor] = {}
        self._totals: dict[int, torch.Tensor] = {}
        self._table: torch.Tensor | None = None

    def _stage(
        self, stage: int, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if stage not in self._last:
            rows = self.num_ffn * self.max_tokens
            if self.cfg.stateful:
                self._state[stage] = torch.zeros(
                    rows, self.hidden, dtype=torch.float32, device=device
                )
            self._last[stage] = torch.full(
                (rows,), -1, dtype=torch.int32, device=device
            )
            self._totals[stage] = torch.zeros(
                self.num_ffn,
                dtype=torch.int64,
                pin_memory=device.type == "cuda",
            )
            self._table = row_bytes_table(self.hidden, device)
        return self._state.get(stage), self._last[stage], self._totals[stage]

    def prepare(
        self, layer_idx: int, stage: int, order: torch.Tensor, num_tokens: int
    ) -> PackedDispatch:
        """Formats, byte offsets and per-FFN byte totals of the dispatch whose
        destination-major entries are ``order`` (``P2pAll2AllEngine.begin_plan``:
        FFN rank * num_tokens + token, padded with larger values). Device work
        only; the totals reach host memory once the stream gets here."""
        if num_tokens > self.max_tokens:
            raise RuntimeError(
                f"{num_tokens} tokens exceed the delta states of {self.max_tokens}"
            )
        pos = self.layers.position(layer_idx)
        _, last, totals = self._stage(stage, order.device)
        if pos == 0:
            # A new forward: every token starts with a first visit.
            last.fill_(-1)
        valid = order < self.num_ffn * num_tokens
        flat = torch.where(valid, order, 0)
        codes = self.cfg.row_codes(pos, self.layers.count, last[flat])
        assert self._table is not None
        nbytes = self._table[codes.long()] * valid
        offsets = torch.cumsum(nbytes, 0) - nbytes
        per_ffn = torch.zeros(self.num_ffn, dtype=torch.int64, device=order.device)
        per_ffn.index_add_(0, flat // max(num_tokens, 1), nbytes)
        totals.copy_(per_ffn, non_blocking=True)
        return PackedDispatch(stage, pos, num_tokens, flat, codes, offsets, totals)

    def encode(
        self,
        packed: PackedDispatch,
        src: torch.Tensor,
        num_send: int,
        out: torch.Tensor,
        meta: torch.Tensor,
        rank: int,
    ) -> None:
        """Write the first ``num_send`` rows (``src``: ``h'``, or ``x`` when
        ``cfg.sends_x``) into ``out`` (uint8, flat) and their words into
        ``meta`` (int32; ``rank`` is this Attention rank), updating this
        side's states."""
        if not 0 <= rank < MAX_ATTENTION_RANKS:
            raise ValueError(
                f"delta dispatch words hold Attention ranks < {MAX_ATTENTION_RANKS}"
            )
        if not num_send:
            return
        state, last, _ = self._stage(packed.stage, src.device)
        encode_rows(
            self.cfg,
            src,
            state,
            last,
            packed.layer_pos,
            packed.flat[:num_send],
            packed.codes[:num_send],
            packed.offsets[:num_send],
            packed.num_tokens,
            rank,
            out,
            meta[:num_send],
        )

    def after_combine(
        self, packed: PackedDispatch, num_send: int, partials: torch.Tensor
    ) -> None:
        """Add each FFN rank's returned partial sums (``[num_send, hidden]``,
        dispatch order, before summing) to that rank's states."""
        if self.cfg.stateful and num_send:
            state = self._state[packed.stage]
            state.index_add_(
                0, packed.flat[:num_send], partials[:num_send].float(), alpha=self.scale
            )


class FFNDeltaState:
    """FFN side (one rank): per stage, the states of the tokens of every
    Attention rank (row Attention rank * max_tokens + token)."""

    def __init__(
        self,
        cfg: DeltaDispatchConfig,
        moe_layers: Sequence[int],
        routed_scale: float,
        hidden: int,
        num_attn: int,
        max_tokens: int,
    ) -> None:
        self.cfg, self.layers, self.scale = cfg, _LayerOrder(moe_layers), routed_scale
        self.hidden, self.num_attn, self.max_tokens = int(hidden), num_attn, max_tokens
        self._state: dict[int, torch.Tensor] = {}
        self._rows: dict[int, torch.Tensor] = {}
        # stage -> words of the received rows (handed over by the connector).
        self._meta: dict[int, torch.Tensor] = {}
        # stage -> h' in the model dtype, and the buffer handed to the experts.
        self._h: dict[int, torch.Tensor] = {}
        self._x: dict[int, torch.Tensor] = {}
        self._table: torch.Tensor | None = None

    def _buffer(
        self,
        store: dict[int, torch.Tensor],
        stage: int,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = store.get(stage)
        if buf is None or buf.shape[0] < rows or buf.dtype != dtype:
            # Zeros at first: the launch rows past the received ones stay finite.
            buf = torch.zeros((rows, self.hidden), dtype=dtype, device=device)
            store[stage] = buf
        return buf

    def receive(self, stage: int, meta: torch.Tensor) -> None:
        """The words (int32, one per received row) of the stage's next rows."""
        self._meta[stage] = meta

    def decode(
        self,
        layer_idx: int,
        stage: int,
        data: torch.Tensor,
        norm_into: Callable[[torch.Tensor, torch.Tensor], object],
        out_rows: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Rebuild the received rows (``data`` uint8 flat bytes, the words from
        ``receive``) and return the experts' input with ``out_rows`` >= rows
        (the experts kernel's row count; rows past the received ones hold
        finite values of earlier layers, masked by the launch's expert ids
        like the plain path's fixed-size buffers). ``norm_into(out, h)``
        writes ``Norm(h)`` into ``out``."""
        self.layers.position(layer_idx)
        meta = self._meta.pop(stage, None)
        if meta is None:
            raise RuntimeError(
                f"stage {stage}: no received rows before layer {layer_idx}"
            )
        n = int(meta.shape[0])
        device = data.device
        x = self._buffer(self._x, stage, out_rows, device, dtype)
        if not n:
            self._rows[stage] = meta.new_empty(0, dtype=torch.int64)
            return x[:out_rows]
        if self._table is None:
            self._table = row_bytes_table(self.hidden, device)
        codes = ((meta >> CODE_SHIFT) & CODE_MASK).long()
        nbytes = self._table[codes]
        offsets = torch.cumsum(nbytes, 0) - nbytes
        rows = (meta >> RANK_SHIFT).long() * self.max_tokens + (
            meta & TOKEN_MASK
        ).long()
        self._rows[stage] = rows
        state = None
        if self.cfg.stateful:
            state = self._state.get(stage)
            if state is None:
                state = torch.zeros(
                    self.num_attn * self.max_tokens,
                    self.hidden,
                    dtype=torch.float32,
                    device=device,
                )
                self._state[stage] = state
        if self.cfg.sends_x:
            decode_rows(self.cfg, data, meta, offsets, rows, state, x[:n])
            return x[:out_rows]
        h = self._buffer(self._h, stage, n, device, dtype)[:n]
        decode_rows(self.cfg, data, meta, offsets, rows, state, h)
        norm_into(x[:n], h)
        return x[:out_rows]

    def after_experts(self, stage: int, output: torch.Tensor) -> None:
        """Add the partial sums returned for this layer (its first rows)."""
        if self.cfg.stateful:
            rows = self._rows[stage]
            self._state[stage].index_add_(
                0, rows, output[: rows.shape[0]].float(), alpha=self.scale
            )


__all__ = [
    "CODE_FORMATS",
    "CODE_MASK",
    "CODE_SHIFT",
    "D_FP32",
    "D_INT4",
    "D_INT8",
    "EXTRA_CONFIG_KEY",
    "H_BF16",
    "MODES",
    "QUANT_GROUP_SIZE",
    "RANK_SHIFT",
    "AttentionDeltaState",
    "DeltaDispatchConfig",
    "FFNDeltaState",
    "PackedDispatch",
    "decode_rows_reference",
    "encode_rows_reference",
    "row_bytes",
]
