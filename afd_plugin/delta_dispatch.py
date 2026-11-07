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
      (first "int4" / "int8"):   send Q(h'),                   S_f = deq(Q(h'))
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
* ``1`` ``d_fp32``: ``d`` in fp32 (4 * hidden bytes); rebuilds ``h'`` exactly
  whenever ``S + (h' - S)`` rounds back to ``h'`` in fp32, which holds for the
  bf16 values of a model (a check mode, not a compression).
* ``2`` ``d_int8`` / ``3`` ``d_int4``: ``d`` quantized per group of
  ``QUANT_GROUP_SIZE`` values, asymmetric: per group the minimum ``lo`` and the
  step ``(max - lo) / (2^bits - 1)``, both rounded to fp16, and
  ``q = round((d - lo) / step)`` clamped to ``[0, 2^bits - 1]`` (the offline
  experiment's ``compress.py`` arithmetic). A row is the packed values (int4:
  two per byte, even index in the low nibble; int3: value ``i`` at bit ``3 i``
  of the row's bytes read little-endian, so every 8 values fill 3 bytes and a
  group 48) followed by the ``(lo, step)`` fp16 pairs of its groups.
* ``4`` ``h_int4``: ``h'`` itself quantized like ``d_int4`` (a first visit
  with ``first: "int4"``); the state becomes the dequantized value.
* ``5`` ``d_int3``: ``d`` quantized like ``d_int4`` in 3 bits.
* ``6`` ``h_int8``: ``h'`` itself quantized like ``d_int8`` (a first visit
  with ``first: "int8"``); the state becomes the dequantized value.

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
* ``delta_int8`` / ``delta_int4`` / ``delta_int3``: first visit ``h_bf16``
  (``h_int4`` / ``h_int8`` with ``first: "int4"`` / ``"int8"``, on every MoE
  layer); then ``d_int8`` on the ``early`` first and ``late`` last MoE layers
  and, with ``revisit`` naming a format wider than the mode's own (``"int8"``;
  for delta_int3 also ``"int4"``), that format on rows whose last visit to the
  FFN rank was ``rgap`` or more MoE layers before; else ``d_int8`` /
  ``d_int4`` / ``d_int3`` (the rules of the offline experiment's
  ``sim.FFNSim``).
* ``x_int4`` / ``x_int8``: no state: every row is ``x = Norm(h')`` in int4 /
  int8 (``d_int4`` / ``d_int8`` of ``x - 0``) and the FFN rank uses the
  dequantized ``x`` as is (plain quantization, the baselines; int8 stands in
  for the 8-bit dispatch of other systems).

Combine compression (``connector_extra_config = {"combine": {"mode": ...,
"ef": ...}}``, needs delta dispatch for the rows' tokens): each FFN rank
returns its partial sum of a row quantized like the int formats above
(``int8`` or ``int4``; every row of a layer the same size). With ``int4``,
``early`` / ``late`` put the first ``early`` and the last ``late`` MoE layers
in ``int8`` (the offline experiment's ``cearly`` / ``clate``: the layers whose
combine error costs the most). With ``ef`` (error feedback)
it adds to the partial the residual it cut from that (Attention rank, token)
at its previous visit in this forward and keeps the new residual:

    v = partial + e,   send Q(v),   e = v - deq(Q(v))

Both sides add ``deq(Q(v))``, what the Attention side received, to the FFN
rank's state, so the states stay identical. Every forward starts with ``e``
at zero.

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

MODES = (
    "off",
    "full_bf16",
    "delta_fp32",
    "delta_int4",
    "delta_int3",
    "x_int4",
    "x_int8",
    "delta_int8",
)
EXTRA_CONFIG_KEY = "delta_dispatch"
COMBINE_KEY = "combine"
COMBINE_MODES = ("off", "int8", "int4")
H_BF16, D_FP32, D_INT8, D_INT4, H_INT4, D_INT3, H_INT8 = 0, 1, 2, 3, 4, 5, 6
CODE_FORMATS = ("h_bf16", "d_fp32", "d_int8", "d_int4", "h_int4", "d_int3", "h_int8")
CODE_BITS = {D_INT8: 8, D_INT4: 4, H_INT4: 4, D_INT3: 3, H_INT8: 8}
# The first-visit formats: h' itself (the state becomes what was sent).
H_CODES = (H_BF16, H_INT4, H_INT8)
# The format of a quantized delta mode's later visits, and the formats
# ``first`` and ``revisit`` name.
DELTA_CODES = {"delta_int8": D_INT8, "delta_int4": D_INT4, "delta_int3": D_INT3}
# The format of every row of a plain-quantization (stateless) mode.
X_CODES = {"x_int4": D_INT4, "x_int8": D_INT8}
FIRST_FORMATS = {"bf16": H_BF16, "int4": H_INT4, "int8": H_INT8}
REVISIT_FORMATS = {"int3": D_INT3, "int4": D_INT4, "int8": D_INT8}
CODE_SHIFT = 23
CODE_MASK = 0x7
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
    """int64 [len(CODE_FORMATS)]: bytes of a row per format code."""
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
    # "": no revisit rule (the mode's own format).
    revisit: str = ""
    rgap: int = 2
    first: str = "bf16"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeltaDispatchConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("delta_dispatch must be a mapping")
        unknown = set(raw) - {"mode", "early", "late", "revisit", "rgap", "first"}
        if unknown:
            raise ValueError(f"unknown delta_dispatch keys: {sorted(unknown)}")
        mode = str(raw.get("mode", "off"))
        if mode not in MODES:
            raise ValueError(f"delta_dispatch mode {mode!r} not in {MODES}")
        early, late = int(raw.get("early", 0)), int(raw.get("late", 0))
        revisit, rgap = str(raw.get("revisit", "")), int(raw.get("rgap", 2))
        first = str(raw.get("first", "bf16"))
        if early < 0 or late < 0:
            raise ValueError("delta_dispatch early and late must be >= 0")
        if revisit and revisit not in REVISIT_FORMATS:
            raise ValueError(
                f"delta_dispatch revisit {revisit!r} not in {tuple(REVISIT_FORMATS)}"
            )
        # Naming the mode's own format is no rule (int4 for the modes without
        # a quantized difference, as before); a rule names a wider format.
        own = DELTA_CODES.get(mode, D_INT4)
        if revisit and REVISIT_FORMATS[revisit] == own:
            revisit = ""
        # The early / late rows are d_int8: already the rows of delta_int8.
        if own == D_INT8:
            early = late = 0
        if revisit and CODE_BITS[REVISIT_FORMATS[revisit]] <= CODE_BITS[own]:
            raise ValueError(
                f"delta_dispatch revisit {revisit!r} is not wider than the rows "
                f"of {mode}"
            )
        if rgap < 2:
            raise ValueError("delta_dispatch rgap must be >= 2")
        if first not in FIRST_FORMATS:
            raise ValueError(
                f"delta_dispatch first {first!r} not in {tuple(FIRST_FORMATS)}"
            )
        rules = early or late or revisit or rgap != 2 or first != "bf16"
        if rules and mode not in DELTA_CODES:
            raise ValueError(
                "delta_dispatch early/late/revisit/rgap/first apply to the "
                f"quantized delta modes {tuple(DELTA_CODES)} only"
            )
        return cls(
            mode=mode, early=early, late=late, revisit=revisit, rgap=rgap, first=first
        )

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
        return self.mode == "delta_fp32" or self.mode in DELTA_CODES

    @property
    def sends_x(self) -> bool:
        """Rows are the MoE input x itself (no state, no FFN-side norm)."""
        return self.mode in X_CODES

    def row_codes(
        self, layer_pos: int, num_layers: int, last: torch.Tensor
    ) -> torch.Tensor:
        """Format code of every row at the ``layer_pos``-th of ``num_layers``
        MoE layers; ``last`` is the MoE layer position of the row's previous
        visit to its FFN rank in this forward (``< 0``: first visit)."""
        if self.mode == "full_bf16":
            return torch.full_like(last, H_BF16)
        if self.mode in X_CODES:
            return torch.full_like(last, X_CODES[self.mode])
        if self.mode == "delta_fp32":
            rest = torch.full_like(last, D_FP32)
        elif layer_pos < self.early or layer_pos >= num_layers - self.late:
            rest = torch.full_like(last, D_INT8)
        elif self.revisit:
            rest = torch.where(
                layer_pos - last >= self.rgap,
                REVISIT_FORMATS[self.revisit],
                DELTA_CODES[self.mode],
            ).to(last.dtype)
        else:
            rest = torch.full_like(last, DELTA_CODES[self.mode])
        # First visits keep their format on the early / late layers too.
        return torch.where(last < 0, FIRST_FORMATS[self.first], rest).to(last.dtype)

    def _codes(self) -> tuple[int, ...]:
        """The formats the rows of this mode can have."""
        fixed = {
            "full_bf16": (H_BF16,),
            "delta_fp32": (H_BF16, D_FP32),
            **{mode: (code,) for mode, code in X_CODES.items()},
        }
        if self.mode in fixed:
            return fixed[self.mode]
        codes = {FIRST_FORMATS[self.first], DELTA_CODES[self.mode]}
        if self.early or self.late:
            codes.add(D_INT8)
        if self.revisit:
            codes.add(REVISIT_FORMATS[self.revisit])
        return tuple(sorted(codes))

    def max_row_bytes(self, hidden: int) -> int:
        return max(row_bytes(c, hidden) for c in self._codes())

    def min_row_bytes(self, hidden: int) -> int:
        return min(row_bytes(c, hidden) for c in self._codes())

    def fixed_row_bytes(self, hidden: int) -> int:
        """The row size when every row of the mode has the same size, else 0.
        Then a row's byte offset is its index times the size, the bytes per
        FFN rank its row count times the size, and the format depends only on
        whether the row is a first visit (``first_rest_codes``): a mode with an
        early / late / revisit rule mixes sizes."""
        sizes = {row_bytes(c, hidden) for c in self._codes()}
        return sizes.pop() if len(sizes) == 1 else 0

    def first_rest_codes(self) -> tuple[int, int]:
        """(format of a first visit, format of a later visit) of a mode whose
        rows all have the same size (``fixed_row_bytes``)."""
        if self.mode == "full_bf16":
            return H_BF16, H_BF16
        if self.mode in X_CODES:
            return X_CODES[self.mode], X_CODES[self.mode]
        rest = DELTA_CODES.get(self.mode)
        # early / late rows are d_int8: the same format only for delta_int8.
        if (
            rest is None
            or self.revisit
            or ((self.early or self.late) and rest != D_INT8)
        ):
            raise ValueError(f"{self} has rows of more than one size")
        return FIRST_FORMATS[self.first], rest


@dataclass(frozen=True)
class CombineConfig:
    """Combine compression: ``mode`` off / int8 / int4, ``ef`` error feedback;
    with int4, ``int8`` on the first ``early`` and last ``late`` MoE layers."""

    mode: str = "off"
    ef: bool = True
    early: int = 0
    late: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> CombineConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("combine must be a mapping")
        unknown = set(raw) - {"mode", "ef", "early", "late"}
        if unknown:
            raise ValueError(f"unknown combine keys: {sorted(unknown)}")
        mode = str(raw.get("mode", "off"))
        if mode not in COMBINE_MODES:
            raise ValueError(f"combine mode {mode!r} not in {COMBINE_MODES}")
        ef = raw.get("ef", True)
        if not isinstance(ef, bool):
            raise ValueError("combine ef must be true or false")
        early, late = int(raw.get("early", 0)), int(raw.get("late", 0))
        if early < 0 or late < 0:
            raise ValueError("combine early and late must be >= 0")
        if (early or late) and mode != "int4":
            raise ValueError("combine early/late (int8 layers) apply to mode int4 only")
        return cls(mode=mode, ef=ef, early=early, late=late)

    @classmethod
    def from_extra_config(cls, extra: Mapping[str, Any] | None) -> CombineConfig:
        return cls.from_mapping((extra or {}).get(COMBINE_KEY))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> CombineConfig:
        additional_config = getattr(vllm_config, "additional_config", None)
        return cls.from_extra_config(
            connector_extra_config_from_source(additional_config)
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def code(self) -> int:
        """The row format of the compressed partial sums (outside the int8
        layers of ``early`` / ``late``)."""
        if not self.enabled:
            raise ValueError("combine compression is off: no row format")
        return {"int8": D_INT8, "int4": D_INT4}[self.mode]

    def code_at(self, layer_pos: int, num_layers: int) -> int:
        """The row format at the ``layer_pos``-th of ``num_layers`` MoE layers."""
        if layer_pos < self.early or layer_pos >= num_layers - self.late:
            return D_INT8
        return self.code

    def row_bytes_at(self, layer_pos: int, num_layers: int, hidden: int) -> int:
        return row_bytes(self.code_at(layer_pos, num_layers), hidden)

    def max_row_bytes(self, hidden: int) -> int:
        codes = {self.code} | ({D_INT8} if self.early or self.late else set())
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
    elif bits == 3:
        # Every BITS_PER_BYTE values (value j at bit 3 j) fill 3 bytes, low first.
        shifts = bits * torch.arange(BITS_PER_BYTE, device=d.device)
        word = (q.view(rows, -1, BITS_PER_BYTE).long() << shifts).sum(-1, keepdim=True)
        shifts = BITS_PER_BYTE * torch.arange(bits, device=d.device)
        q = ((word >> shifts) & 0xFF).to(torch.uint8).view(rows, -1)
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
    elif bits == 3:
        shifts = BITS_PER_BYTE * torch.arange(bits, device=rows.device)
        word = (packed.reshape(n, -1, bits).long() << shifts).sum(-1, keepdim=True)
        shifts = bits * torch.arange(BITS_PER_BYTE, device=rows.device)
        q = ((word >> shifts) & ((1 << bits) - 1)).view(n, hidden)
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
    ``offsets[i]``; the states (and ``last``) are updated in place. With
    ``codes`` None (a mode whose rows all have one size) the format is the
    first-visit one where ``last[flat] < 0``, else the later-visit one."""
    hidden = src.shape[1]
    tokens = flat % num_tokens
    if codes is None:
        first, rest = cfg.first_rest_codes()
        if cfg.stateful and last is not None:
            codes = torch.where(last[flat] < 0, first, rest).to(torch.int32)
        else:
            codes = torch.full_like(flat, rest, dtype=torch.int32)
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
        elif code in (H_INT4, H_INT8):
            # h' itself: the state becomes what was sent.
            data, deq = _quantized_rows(code, h.float())
            if cfg.stateful:
                assert state is not None
                state[rows_flat] = deq
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
    offsets: torch.Tensor | None,
    state_rows: torch.Tensor | None,
    state: torch.Tensor | None,
    h_out: torch.Tensor,
    rows_out: torch.Tensor | None = None,
    max_tokens: int = 0,
    row_size: int = 0,
) -> None:
    """Torch reference of the fused FFN kernel: rebuild every row's value from
    the packed bytes into ``h_out`` (``h'``, or ``x`` when ``cfg.sends_x``) and
    update the states (rows ``state_rows``) in place. With ``state_rows``
    None the state row of a row is read from its word (Attention rank *
    ``max_tokens`` + token) and written to ``rows_out``; with ``offsets``
    None every row is ``row_size`` bytes (offset index * ``row_size``)."""
    hidden = h_out.shape[1]
    if state_rows is None:
        state_rows = (meta >> RANK_SHIFT).long() * max_tokens + (
            meta & TOKEN_MASK
        ).long()
        if rows_out is not None:
            rows_out.copy_(state_rows)
    if offsets is None:
        offsets = torch.arange(meta.shape[0], device=meta.device) * row_size
    codes = (meta >> CODE_SHIFT) & CODE_MASK
    for code in range(len(CODE_FORMATS)):
        sel = (codes == code).nonzero().view(-1)
        if not sel.numel():
            continue
        size = row_bytes(code, hidden)
        rows = data[_byte_positions(offsets[sel], size)]
        if code in H_CODES:
            # h' itself: the new state.
            if code == H_BF16:
                value = rows.contiguous().view(torch.bfloat16).float()
            else:
                value = _dequantized_rows(code, rows, hidden)
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


def encode_combine_reference(
    code: int,
    partial: torch.Tensor,
    rows: torch.Tensor,
    err: torch.Tensor | None,
    out: torch.Tensor,
    deq: torch.Tensor,
) -> None:
    """FFN side: quantize ``partial`` (``[n, hidden]``) plus, with error
    feedback, ``err`` at ``rows`` into ``out`` (uint8 flat, ``n`` rows of
    ``row_bytes(code)`` back to back); ``deq`` (fp32 ``[n, hidden]``) gets
    what the Attention side will rebuild and ``err`` the new residual."""
    n, hidden = partial.shape
    if not n:
        return
    v = partial.float()
    if err is not None:
        v = v + err[rows]
    packed, d = _quantized_rows(code, v)
    out[: packed.numel()].copy_(packed.view(-1))
    deq.copy_(d)
    if err is not None:
        err[rows] = v - d


def decode_combine_reference(code: int, data: torch.Tensor, out: torch.Tensor) -> None:
    """Attention side: ``out`` (fp32 ``[n, hidden]``) from the ``n`` packed rows
    of ``data`` (uint8 flat)."""
    n, hidden = out.shape
    if not n:
        return
    size = row_bytes(code, hidden)
    out.copy_(_dequantized_rows(code, data[: n * size].view(n, size), hidden))


def encode_combine(code: int, *args: Any) -> None:
    """``encode_combine_reference``, fused on CUDA."""
    if args[0].is_cuda:
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.encode_combine(code, *args)
    else:
        encode_combine_reference(code, *args)


def decode_combine(code: int, *args: Any) -> None:
    """``decode_combine_reference``, fused on CUDA."""
    if args[0].is_cuda:
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.decode_combine(code, *args)
    else:
        decode_combine_reference(code, *args)


def encode_rows(cfg: DeltaDispatchConfig, *args: Any) -> None:
    if args[0].is_cuda:
        # Lazy import: Triton is only needed (and installed) with CUDA.
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.encode_rows(cfg, *args)
        return
    encode_rows_reference(cfg, *args)


def decode_rows(cfg: DeltaDispatchConfig, *args: Any, **kwargs: Any) -> None:
    if args[0].is_cuda:
        from afd_plugin import delta_dispatch_triton

        delta_dispatch_triton.decode_rows(cfg, *args, **kwargs)
        return
    decode_rows_reference(cfg, *args, **kwargs)


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
    # A mode whose rows all have one size (``row_size``) leaves the codes to
    # the encode kernel (None) and its entries past the sent ones unmasked.
    flat: torch.Tensor
    codes: torch.Tensor | None
    offsets: torch.Tensor
    # Bytes per FFN rank, copied to host memory without waiting (readable
    # once the stream has passed ``prepare``); None with one row size (the
    # host knows them from the plan's row counts, ``ffn_bytes``).
    totals: torch.Tensor | None
    row_size: int = 0


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
        combine: CombineConfig | None = None,
    ) -> None:
        if max_tokens > TOKEN_MASK + 1:
            raise ValueError(f"delta dispatch words hold tokens < {TOKEN_MASK + 1}")
        self.cfg, self.layers, self.scale = cfg, _LayerOrder(moe_layers), routed_scale
        self.num_ffn, self.max_tokens, self.hidden = num_ffn, max_tokens, hidden
        self.combine = combine or CombineConfig()
        # stage -> fp32 [num_ffn * max_tokens, hidden]: rebuilt partial sums.
        self._back: dict[int, torch.Tensor] = {}
        # stage -> fp32 [num_ffn * max_tokens, hidden], int32 [num_ffn * max_tokens]
        self._state: dict[int, torch.Tensor] = {}
        self._last: dict[int, torch.Tensor] = {}
        self._totals: dict[int, torch.Tensor] = {}
        self._table: torch.Tensor | None = None
        # One row size for every row of the mode (0: sizes differ): the
        # offsets are index * size (``_offsets``, made once) and the codes are
        # chosen in the encode kernel.
        self._fixed = cfg.fixed_row_bytes(hidden)
        self._offsets: torch.Tensor | None = None

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
        if self._fixed:
            # One row size: no per-row work here. The sent entries come first
            # in ``order`` and only those are encoded and read back.
            if self._offsets is None or self._offsets.device != order.device:
                self._offsets = (
                    torch.arange(
                        self.num_ffn * self.max_tokens,
                        dtype=torch.int64,
                        device=order.device,
                    )
                    * self._fixed
                )
            return PackedDispatch(
                stage,
                pos,
                num_tokens,
                order,
                None,
                self._offsets[: order.shape[0]],
                None,
                self._fixed,
            )
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
            None if packed.codes is None else packed.codes[:num_send],
            packed.offsets[:num_send],
            packed.num_tokens,
            rank,
            out,
            meta[:num_send],
        )

    def ffn_bytes(
        self, packed: PackedDispatch, send_counts: Sequence[int]
    ) -> list[int]:
        """Bytes of this dispatch per FFN rank (``send_counts``: its rows per
        FFN rank from the plan)."""
        if packed.totals is None:
            return [int(c) * packed.row_size for c in send_counts]
        return [int(b) for b in packed.totals.tolist()]

    def combine_row_bytes(self, layer_pos: int) -> int:
        """Bytes of a compressed combine row at MoE layer position ``layer_pos``."""
        return self.combine.row_bytes_at(layer_pos, self.layers.count, self.hidden)

    def decode_combine(
        self, stage: int, data: torch.Tensor, num_rows: int, layer_pos: int
    ) -> torch.Tensor:
        """The returned partial sums (fp32 ``[num_rows, hidden]``, dispatch
        order) of MoE layer position ``layer_pos`` rebuilt from the compressed
        rows in ``data`` (uint8 flat)."""
        back = self._back.get(stage)
        if back is None:
            back = torch.empty(
                self.num_ffn * self.max_tokens,
                self.hidden,
                dtype=torch.float32,
                device=data.device,
            )
            self._back[stage] = back
        code = self.combine.code_at(layer_pos, self.layers.count)
        decode_combine(code, data, back[:num_rows])
        return back[:num_rows]

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
        combine: CombineConfig | None = None,
    ) -> None:
        self.cfg, self.layers, self.scale = cfg, _LayerOrder(moe_layers), routed_scale
        self.hidden, self.num_attn, self.max_tokens = int(hidden), num_attn, max_tokens
        self.combine = combine or CombineConfig()
        # Combine compression, per stage: the residuals of error feedback
        # (fp32 [num_attn * max_tokens, hidden], by state row), the compressed
        # rows of the last layer (uint8 flat) and their count, and the values
        # the Attention side rebuilds from them.
        self._err: dict[int, torch.Tensor] = {}
        self._cbytes: dict[int, torch.Tensor] = {}
        self._crows: dict[int, int] = {}
        self._deq: dict[int, torch.Tensor] = {}
        # stage -> MoE layer position of the rows being processed.
        self._pos: dict[int, int] = {}
        self._state: dict[int, torch.Tensor] = {}
        self._rows: dict[int, torch.Tensor] = {}
        # stage -> words of the received rows (handed over by the connector).
        self._meta: dict[int, torch.Tensor] = {}
        # stage -> h' in the model dtype, and the buffer handed to the experts.
        self._h: dict[int, torch.Tensor] = {}
        self._x: dict[int, torch.Tensor] = {}
        self._table: torch.Tensor | None = None
        # One row size for every row of the mode (0: sizes differ): the decode
        # kernel finds each row at index * size. It always reads each row's
        # state row from its word and writes it here (int32, per stage).
        self._fixed = cfg.fixed_row_bytes(self.hidden)
        self._row_ids: dict[int, torch.Tensor] = {}

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
        pos = self.layers.position(layer_idx)
        self._pos[stage] = pos
        if pos == 0 and stage in self._err:
            # A new forward: no residual is carried over from the last one.
            self._err[stage].zero_()
        # The previous layer's compressed rows are stale from here on.
        self._crows.pop(stage, None)
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
        offsets = None
        if not self._fixed:
            if self._table is None:
                self._table = row_bytes_table(self.hidden, device)
            nbytes = self._table[((meta >> CODE_SHIFT) & CODE_MASK).long()]
            offsets = torch.cumsum(nbytes, 0) - nbytes
        ids = self._row_ids.get(stage)
        if ids is None or ids.device != device:
            ids = torch.empty(
                self.num_attn * self.max_tokens, dtype=torch.int32, device=device
            )
            self._row_ids[stage] = ids
        rows = ids[:n]
        self._rows[stage] = rows
        kw = {"rows_out": rows, "max_tokens": self.max_tokens, "row_size": self._fixed}
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
            decode_rows(self.cfg, data, meta, offsets, None, state, x[:n], **kw)
            return x[:out_rows]
        h = self._buffer(self._h, stage, n, device, dtype)[:n]
        decode_rows(self.cfg, data, meta, offsets, None, state, h, **kw)
        norm_into(x[:n], h)
        return x[:out_rows]

    def after_experts(self, stage: int, output: torch.Tensor) -> None:
        """Take the partial sums returned for this layer (its first rows):
        compress them (combine compression) and add what the Attention side
        will get to the states."""
        rows = self._rows[stage]
        n = int(rows.shape[0])
        returned = output[:n]
        if self.combine.enabled:
            device = output.device
            capacity = self.num_attn * self.max_tokens
            data = self._cbytes.get(stage)
            if data is None:
                data = torch.empty(
                    capacity * self.combine.max_row_bytes(self.hidden),
                    dtype=torch.uint8,
                    device=device,
                )
                self._cbytes[stage] = data
            err = None
            if self.combine.ef:
                err = self._err.get(stage)
                if err is None:
                    err = torch.zeros(
                        capacity, self.hidden, dtype=torch.float32, device=device
                    )
                    self._err[stage] = err
            deq = self._buffer(self._deq, stage, capacity, device, torch.float32)[:n]
            code = self.combine.code_at(self._pos[stage], self.layers.count)
            encode_combine(code, returned, rows, err, data, deq)
            self._crows[stage] = n
            returned = deq
        if self.cfg.stateful and n:
            self._state[stage].index_add_(0, rows, returned.float(), alpha=self.scale)

    def combine_row_bytes(self, stage: int) -> int:
        """Bytes of a compressed combine row of the stage's last layer."""
        return self.combine.row_bytes_at(
            self._pos[stage], self.layers.count, self.hidden
        )

    def combine_bytes(self, stage: int, num_rows: int) -> torch.Tensor:
        """The compressed partial sums of the stage's last layer (uint8 flat,
        ``num_rows`` rows back to back) for the connector to send."""
        if self._crows.get(stage) != num_rows:
            raise RuntimeError(
                f"stage {stage}: {self._crows.get(stage)} compressed rows, the "
                f"connector sends {num_rows}"
            )
        return self._cbytes[stage][: num_rows * self.combine_row_bytes(stage)]


__all__ = [
    "COMBINE_KEY",
    "COMBINE_MODES",
    "CODE_FORMATS",
    "CODE_MASK",
    "CODE_SHIFT",
    "D_FP32",
    "D_INT3",
    "D_INT4",
    "D_INT8",
    "EXTRA_CONFIG_KEY",
    "H_BF16",
    "H_CODES",
    "H_INT4",
    "H_INT8",
    "MODES",
    "QUANT_GROUP_SIZE",
    "RANK_SHIFT",
    "AttentionDeltaState",
    "CombineConfig",
    "DeltaDispatchConfig",
    "FFNDeltaState",
    "PackedDispatch",
    "decode_combine_reference",
    "decode_rows_reference",
    "encode_combine_reference",
    "encode_rows_reference",
    "row_bytes",
]
