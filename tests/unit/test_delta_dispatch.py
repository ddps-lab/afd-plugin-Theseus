# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch arithmetic (afd_plugin/delta_dispatch.py), without vLLM."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from afd_plugin.delta_dispatch import (
    CODE_FORMATS,
    CODE_MASK,
    CODE_SHIFT,
    D_FP32,
    D_INT3,
    D_INT4,
    D_INT8,
    D_NVFP4,
    H_BF16,
    H_INT4,
    H_INT8,
    H_NVFP4,
    MAX_ATTENTION_RANKS,
    QUANT_GROUP_SIZE,
    RANK_SHIFT,
    TOKEN_MASK,
    AttentionDeltaState,
    CombineConfig,
    DeltaDispatchConfig,
    FFNDeltaState,
    decode_combine_reference,
    decode_rows_reference,
    encode_combine_reference,
    encode_rows_reference,
    row_bytes,
)

HIDDEN = 256
TOKENS = 6
MOE_LAYERS = (1, 2, 3, 4, 5, 6, 7)
ROUTED_SCALE = 1.5
EPS = 1e-6
NUM_EXPERTS, TOP_K = 8, 2
MODEL_HIDDEN = 2048
# The Triton kernels run on CUDA, or on the CPU in Triton's interpreter
# (TRITON_INTERPRET=1; slow, and the GPU's E4M3 conversion is not emulated).
KERNEL_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "")
)
needs_kernels = pytest.mark.skipif(
    not KERNEL_DEVICE, reason="needs CUDA or TRITON_INTERPRET=1 (Triton)"
)
# The kernel tests move tensors with .to(KERNEL_DEVICE, copy=True): on the CPU
# a plain .to() returns the same tensor, and the two sides' states would alias.
# The E4M3 conversion instructions (Ada and later).
HARDWARE_E4M3 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
    8,
    9,
)


def _assert_same_model_dtype(got: torch.Tensor, want: torch.Tensor) -> None:
    """Kernel output in the model dtype vs ``want`` (fp32) cast to it: equal on
    CUDA; the interpreter truncates fp32 -> bf16 instead of rounding, so there
    within one bf16 step."""
    if KERNEL_DEVICE == "cuda":
        assert torch.equal(got, want.to(got.dtype))
    else:
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=0)


NVFP4_KERNEL_VARIANTS = [
    ("two_pass", False),
    ("one_pass", False),
    pytest.param(
        "two_pass",
        True,
        marks=pytest.mark.skipif(not HARDWARE_E4M3, reason="needs E4M3 hardware"),
    ),
    pytest.param(
        "one_pass",
        True,
        marks=pytest.mark.skipif(not HARDWARE_E4M3, reason="needs E4M3 hardware"),
    ),
]


def _norm_into(out: torch.Tensor, h: torch.Tensor) -> None:
    """Stand-in for post_attention_layernorm (input and output in bf16)."""
    x = h.float()
    out.copy_((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(out.dtype))


def _norm(h: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(h)
    _norm_into(out, h)
    return out


def _expert(x: torch.Tensor, expert: int) -> torch.Tensor:
    """Stand-in for one expert (depends on the input and the expert)."""
    return torch.tanh(x.float() * (0.5 + 0.1 * expert))


def _offline_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """The offline experiment's quantizer (Theseus experiment/afd_delta/compress.py,
    ``_q_int``), copied: the vLLM path must reproduce it."""
    sh = x.shape
    g = x.reshape(*sh[:-1], sh[-1] // QUANT_GROUP_SIZE, QUANT_GROUP_SIZE)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    n = 2**bits - 1
    s = ((hi - lo).clamp_min(1e-12) / n).half().float()
    lo = lo.half().float()
    return (torch.round((g - lo) / s).clamp(0, n) * s + lo).reshape(sh)


_OFFLINE_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _offline_round_e2m1(y: torch.Tensor) -> torch.Tensor:
    """compress.py ``_round_e2m1``, copied."""
    grid = torch.tensor(_OFFLINE_E2M1, device=y.device)
    a = y.abs().clamp(max=6.0)
    lo_i = (torch.searchsorted(grid, a.contiguous(), right=True) - 1).clamp(0, 6)
    lo, hi = grid[lo_i], grid[lo_i + 1]
    up = (hi - a < a - lo) | ((hi - a == a - lo) & (lo_i % 2 == 1))
    return torch.where(up, hi, lo) * torch.sign(y)


def _offline_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """The offline experiment's NVFP4 (compress.py ``_q_nvfp4``), copied."""
    block = 16
    sh = x.shape
    gs = (x.abs().amax(-1, keepdim=True) / (448.0 * 6.0)).clamp_min(1e-30)
    g = x.reshape(*sh[:-1], sh[-1] // block, block)
    bs = g.abs().amax(-1, keepdim=True) / 6.0 / gs.unsqueeze(-1)
    bs = bs.to(torch.float8_e4m3fn).float()
    scale = bs * gs.unsqueeze(-1)
    q = torch.where(
        scale > 0,
        _offline_round_e2m1(g / scale.clamp_min(1e-30)),
        torch.zeros_like(g),
    )
    return (q * scale).reshape(sh)


def _order(owner: torch.Tensor, ids: torch.Tensor, num_ffn: int) -> torch.Tensor:
    """The dispatch order of P2pAll2AllEngine.begin_plan: (FFN rank * T +
    token) of every visited pair, FFN rank major, padded with F * T."""
    tokens = ids.shape[0]
    dest = (owner[ids].unsqueeze(0) == torch.arange(num_ffn).view(-1, 1, 1)).any(-1)
    flat = dest.view(-1).nonzero().view(-1)
    pad = torch.full((num_ffn * tokens - flat.numel(),), num_ffn * tokens)
    return torch.cat([flat, pad])


def _drive(
    cfg: DeltaDispatchConfig,
    num_ffn: int,
    num_attn: int,
    stages: int,
    combine: CombineConfig | None = None,
    forwards: int = 1,
):
    """Run Attention ranks and FFN ranks through the MoE layers the way the
    connector and the model do, with a loopback wire. Per (layer, stage,
    Attention rank) return the experts' input of every (FFN rank, token) row,
    Norm(h'), the rows' state indices, codes and words, the bytes per FFN
    rank, the returned partial sums (exact and as received) and, after each
    (layer, stage), the states of both sides."""
    gen = torch.Generator().manual_seed(0)
    owner = torch.arange(NUM_EXPERTS) % num_ffn
    combine = combine or CombineConfig()
    rb = 0
    attn = [
        AttentionDeltaState(
            cfg, MOE_LAYERS, ROUTED_SCALE, num_ffn, TOKENS, HIDDEN, combine
        )
        for _ in range(num_attn)
    ]
    ffn = [
        FFNDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, HIDDEN, num_attn, TOKENS, combine)
        for _ in range(num_ffn)
    ]
    h = {
        (a, s): torch.randn(TOKENS, HIDDEN, generator=gen).to(torch.bfloat16)
        for a in range(num_attn)
        for s in range(stages)
    }
    steps = []
    for layer in [layer for _ in range(forwards) for layer in MOE_LAYERS]:
        for s in range(stages):
            sent = {}
            for a in range(num_attn):
                h_prime = (
                    h[a, s].float() + torch.randn(TOKENS, HIDDEN, generator=gen) * 0.3
                ).to(torch.bfloat16)
                ids = torch.randint(0, NUM_EXPERTS, (TOKENS, TOP_K), generator=gen)
                weights = torch.rand(TOKENS, TOP_K, generator=gen)
                order = _order(owner, ids, num_ffn)
                packed = attn[a].prepare(layer, s, order, TOKENS)
                n = int((order < num_ffn * TOKENS).sum())
                counts = torch.bincount(order[:n] // TOKENS, minlength=num_ffn)
                totals = attn[a].ffn_bytes(packed, counts.tolist())
                out = torch.zeros(sum(totals) + 16, dtype=torch.uint8)
                meta = torch.zeros(n, dtype=torch.int32)
                src = _norm(h_prime) if cfg.sends_x else h_prime
                attn[a].encode(packed, src, n, out, meta, a)
                sent[a] = (h_prime, ids, weights, packed, n, out, meta, totals)
            # Every FFN rank receives its rows of all Attention ranks
            # (Attention rank major), rebuilds them and runs its experts.
            results: dict[int, list] = {a: [] for a in range(num_attn)}
            for f in range(num_ffn):
                datas, words, origin = [], [], []
                for a in range(num_attn):
                    _, ids, weights, packed, n, out, meta, totals = sent[a]
                    rows = (packed.flat[:n] // TOKENS == f).nonzero().view(-1)
                    if rows.numel():
                        start = int(packed.offsets[rows[0]])
                        datas.append(out[start : start + totals[f]])
                        words.append(meta[rows])
                        origin += [(a, int(i)) for i in rows]
                data = torch.cat(datas) if datas else torch.zeros(0, dtype=torch.uint8)
                word = torch.cat(words) if words else torch.zeros(0, dtype=torch.int32)
                ffn[f].receive(s, word)
                launch = len(origin) + 2  # fixed-size launch: two padding rows
                x = ffn[f].decode(layer, s, data, _norm_into, launch, torch.bfloat16)
                assert x.shape == (launch, HIDDEN)
                assert torch.isfinite(x[len(origin) :]).all()
                partial = torch.zeros(launch, HIDDEN)
                for r, (a, i) in enumerate(origin):
                    _, ids, weights, packed, _, _, _, _ = sent[a]
                    t = int(packed.flat[i]) % TOKENS
                    for k in range(TOP_K):
                        if int(owner[ids[t, k]]) == f:
                            partial[r] += weights[t, k] * _expert(x[r], int(ids[t, k]))
                partial = partial.to(torch.bfloat16)
                ffn[f].after_experts(s, partial)
                if combine.enabled:
                    # The compressed rows (this layer's row size), Attention
                    # rank major like the received ones: each Attention
                    # rank's share is a span.
                    rb = ffn[f].combine_row_bytes(s)
                    assert rb == attn[0].combine_row_bytes(MOE_LAYERS.index(layer))
                    cbytes = ffn[f].combine_bytes(s, len(origin)).clone()
                    assert cbytes.numel() == len(origin) * rb
                for r, (a, i) in enumerate(origin):
                    wire = cbytes[r * rb : (r + 1) * rb] if combine.enabled else None
                    results[a].append((i, partial[r], x[r].clone(), f, wire))
            for a in range(num_attn):
                h_prime, ids, weights, packed, n, _, meta, totals = sent[a]
                exact = torch.zeros(n, HIDDEN, dtype=torch.bfloat16)
                inputs = {}
                for i, part, x_row, f, _ in results[a]:
                    exact[i] = part
                    inputs[(f, int(packed.flat[i]) % TOKENS)] = x_row
                if combine.enabled:
                    # Gathered in the dispatch order (FFN rank major, as the
                    # connector reads the regions), rebuilt, into the states.
                    by_row = sorted(results[a], key=lambda item: item[0])
                    data = torch.cat([item[4] for item in by_row])
                    received = (
                        attn[a].decode_combine(s, data, n, packed.layer_pos).clone()
                    )
                    attn[a].after_combine(packed, n, received)
                    back = received.to(torch.bfloat16)
                else:
                    back = exact
                    attn[a].after_combine(packed, n, back)
                summed = torch.zeros(TOKENS, HIDDEN)
                summed.index_add_(0, packed.flat[:n] % TOKENS, back.float())
                shared = torch.randn(TOKENS, HIDDEN, generator=gen) * 0.2
                h[a, s] = (h_prime.float() + ROUTED_SCALE * summed + shared).to(
                    torch.bfloat16
                )
                steps.append(
                    {
                        "layer": layer,
                        "stage": s,
                        "attn": a,
                        "inputs": inputs,
                        "norm_h": _norm(h_prime),
                        "flat": packed.flat[:n].clone(),
                        # The codes as sent (the words); with one row size the
                        # encode kernel picks them (packed.codes is None).
                        "codes": (meta >> CODE_SHIFT) & CODE_MASK,
                        "meta": meta.clone(),
                        "bytes": torch.tensor(totals),
                        "exact": exact.float(),
                        "received": back.float(),
                    }
                )
            # Snapshots: the states right after this (layer, stage).
            steps[-1]["ffn_states"] = [
                None if f_state._state.get(s) is None else f_state._state[s].clone()
                for f_state in ffn
            ]
            steps[-1]["attn_states"] = [
                None if a_state._state.get(s) is None else a_state._state[s].clone()
                for a_state in attn
            ]
    return steps


def test_config_parsing() -> None:
    assert not DeltaDispatchConfig.from_extra_config(None).enabled
    assert not DeltaDispatchConfig.from_extra_config({}).enabled
    cfg = DeltaDispatchConfig.from_extra_config(
        {
            "delta_dispatch": {
                "mode": "delta_int4",
                "early": 4,
                "late": 1,
                "revisit": "int8",
            }
        }
    )
    assert (cfg.mode, cfg.early, cfg.late, cfg.revisit, cfg.rgap) == (
        "delta_int4",
        4,
        1,
        "int8",
        2,
    )
    for mode in ("x_int4", "x_int8", "x_nvfp4"):
        assert DeltaDispatchConfig(mode=mode).sends_x
        assert not DeltaDispatchConfig(mode=mode).stateful
    cfg = DeltaDispatchConfig.from_mapping({"mode": "delta_nvfp4", "first": "nvfp4"})
    assert (cfg.mode, cfg.first) == ("delta_nvfp4", "nvfp4")
    assert cfg.stateful and cfg.uses_nvfp4
    assert not DeltaDispatchConfig(mode="delta_int4").uses_nvfp4
    cfg = DeltaDispatchConfig.from_mapping({"mode": "delta_int8", "first": "int8"})
    assert (cfg.mode, cfg.first) == ("delta_int8", "int8")
    assert cfg.stateful and not cfg.sends_x
    for bad in (
        {"mode": "int3"},
        {"mode": "off", "extra": 1},
        {"mode": "delta_fp32", "early": 2},
        {"mode": "x_int4", "revisit": "int8"},
        {"mode": "x_int8", "revisit": "int8"},
        {"mode": "x_int8", "first": "int4"},
        {"mode": "delta_int4", "revisit": "fp8"},
        {"mode": "delta_int4", "rgap": 1},
    ):
        with pytest.raises(ValueError):
            DeltaDispatchConfig.from_mapping(bad)
    # Read from vLLM's additional_config like the connector extra config.
    vllm_config = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "ffn",
                "connector_extra_config": {"delta_dispatch": {"mode": "delta_fp32"}},
            }
        }
    )
    assert DeltaDispatchConfig.from_vllm_config(vllm_config).mode == "delta_fp32"
    assert not DeltaDispatchConfig.from_vllm_config(SimpleNamespace()).enabled
    assert not DeltaDispatchConfig.from_vllm_config(
        SimpleNamespace(additional_config={"afd": {"role": "ffn"}})
    ).enabled


def test_first_and_revisit_config() -> None:
    cfg = DeltaDispatchConfig.from_extra_config(
        {"delta_dispatch": {"mode": "delta_int3", "first": "int4"}}
    )
    assert (cfg.mode, cfg.first, cfg.revisit) == ("delta_int3", "int4", "")
    assert cfg.stateful and not cfg.sends_x
    assert DeltaDispatchConfig.from_mapping({"mode": "delta_int4"}).first == "bf16"
    # Naming the mode's own format is no revisit rule: equal configs, equal
    # repr (the settings digest); int4 stays no rule in the other modes.
    for mode, own in (
        ("delta_int8", "int8"),
        ("delta_int4", "int4"),
        ("delta_int3", "int3"),
        ("x_int4", "int4"),
        ("x_int8", "int4"),
        ("delta_fp32", "int4"),
    ):
        named = DeltaDispatchConfig.from_mapping({"mode": mode, "revisit": own})
        plain = DeltaDispatchConfig.from_mapping({"mode": mode})
        assert named == plain and repr(named) == repr(plain) and not named.revisit
    # Only formats wider than the mode's own are revisit rules.
    for mode, revisit in (
        ("delta_int4", "int8"),
        ("delta_int3", "int4"),
        ("delta_int3", "int8"),
    ):
        cfg = DeltaDispatchConfig.from_mapping(
            {"mode": mode, "revisit": revisit, "first": "int4"}
        )
        assert (cfg.revisit, cfg.first) == (revisit, "int4")
    wide_first = DeltaDispatchConfig.from_mapping(
        {"mode": "delta_int4", "first": "int8"}
    )
    assert wide_first.first == "int8"
    # delta_int8 rows are already int8: early / late are no rule (same setting).
    ruled = DeltaDispatchConfig.from_mapping(
        {"mode": "delta_int8", "early": 4, "late": 4}
    )
    assert ruled == DeltaDispatchConfig.from_mapping({"mode": "delta_int8"})
    for bad in (
        {"mode": "delta_int8", "revisit": "int4"},
        {"mode": "x_int8", "first": "int8"},
        {"mode": "delta_int4", "revisit": "int3"},
        {"mode": "x_int4", "revisit": "int3"},
        {"mode": "delta_int3", "revisit": "fp8"},
        {"mode": "delta_int3", "first": "fp8"},
        {"mode": "delta_int3", "first": "bf16", "extra": 1},
        {"mode": "delta_fp32", "first": "int4"},
        {"mode": "full_bf16", "first": "int4"},
        {"mode": "x_int4", "first": "int4"},
        {"mode": "off", "first": "int4"},
        # NVFP4 rows are 4-bit: an int4 revisit is not wider.
        {"mode": "delta_nvfp4", "revisit": "int4"},
        {"mode": "x_nvfp4", "first": "nvfp4"},
    ):
        with pytest.raises(ValueError):
            DeltaDispatchConfig.from_mapping(bad)


def test_row_formats_and_codes() -> None:
    assert row_bytes(H_BF16, MODEL_HIDDEN) == 4096
    assert row_bytes(D_FP32, MODEL_HIDDEN) == 8192
    assert row_bytes(D_INT8, MODEL_HIDDEN) == 2112
    assert row_bytes(D_INT4, MODEL_HIDDEN) == 1088
    assert row_bytes(H_INT4, MODEL_HIDDEN) == 1088
    assert row_bytes(H_INT8, MODEL_HIDDEN) == 2112
    assert row_bytes(D_INT3, MODEL_HIDDEN) == 832
    assert row_bytes(D_INT3, QUANT_GROUP_SIZE) == 52
    # NVFP4: 4-bit codes, an E4M3 byte per 16 values, the fp32 row scale.
    assert row_bytes(D_NVFP4, MODEL_HIDDEN) == 1156
    assert row_bytes(H_NVFP4, MODEL_HIDDEN) == 1156
    assert row_bytes(D_NVFP4, QUANT_GROUP_SIZE) == 76
    # (min, max) row bytes per config; the existing ones as before.
    bounds = {
        "delta_int4": (DeltaDispatchConfig(mode="delta_int4"), (1088, 4096)),
        "delta_int4 rules": (
            DeltaDispatchConfig(mode="delta_int4", early=2, late=1, revisit="int8"),
            (1088, 4096),
        ),
        "delta_fp32": (DeltaDispatchConfig(mode="delta_fp32"), (4096, 8192)),
        "full_bf16": (DeltaDispatchConfig(mode="full_bf16"), (4096, 4096)),
        "x_int4": (DeltaDispatchConfig(mode="x_int4"), (1088, 1088)),
        "x_int8": (DeltaDispatchConfig(mode="x_int8"), (2112, 2112)),
        "delta_int3": (DeltaDispatchConfig(mode="delta_int3"), (832, 4096)),
        "delta_int3 first": (
            DeltaDispatchConfig(mode="delta_int3", first="int4"),
            (832, 1088),
        ),
        "delta_int3 first revisit": (
            DeltaDispatchConfig(mode="delta_int3", first="int4", revisit="int4"),
            (832, 1088),
        ),
        "delta_int3 first late": (
            DeltaDispatchConfig(mode="delta_int3", first="int4", late=1),
            (832, 2112),
        ),
        "delta_int4 first": (
            DeltaDispatchConfig(mode="delta_int4", first="int4"),
            (1088, 1088),
        ),
        "delta_int4 first int8": (
            DeltaDispatchConfig(mode="delta_int4", first="int8"),
            (1088, 2112),
        ),
        "delta_int8": (DeltaDispatchConfig(mode="delta_int8"), (2112, 4096)),
        "delta_int8 first": (
            DeltaDispatchConfig(mode="delta_int8", first="int8"),
            (2112, 2112),
        ),
        "delta_nvfp4": (DeltaDispatchConfig(mode="delta_nvfp4"), (1156, 4096)),
        "delta_nvfp4 first": (
            DeltaDispatchConfig(mode="delta_nvfp4", first="nvfp4"),
            (1156, 1156),
        ),
        "delta_nvfp4 first late": (
            DeltaDispatchConfig(mode="delta_nvfp4", first="nvfp4", late=1),
            (1156, 2112),
        ),
        "x_nvfp4": (DeltaDispatchConfig(mode="x_nvfp4"), (1156, 1156)),
    }
    for name, (cfg, limits) in bounds.items():
        got = (cfg.min_row_bytes(MODEL_HIDDEN), cfg.max_row_bytes(MODEL_HIDDEN))
        assert got == limits, name
    # Previous visits of five rows before MoE layer position 5 of 9 (-1: none).
    last = torch.tensor([-1, 4, 3, 2, 0], dtype=torch.int32)
    cases = {
        "full_bf16": (DeltaDispatchConfig(mode="full_bf16"), [0, 0, 0, 0, 0]),
        "x_int4": (DeltaDispatchConfig(mode="x_int4"), [3, 3, 3, 3, 3]),
        "x_int8": (DeltaDispatchConfig(mode="x_int8"), [2, 2, 2, 2, 2]),
        "delta_fp32": (DeltaDispatchConfig(mode="delta_fp32"), [0, 1, 1, 1, 1]),
        "int4": (DeltaDispatchConfig(mode="delta_int4"), [0, 3, 3, 3, 3]),
        # gaps 1, 2, 3, 5
        "revisit": (
            DeltaDispatchConfig(mode="delta_int4", revisit="int8"),
            [0, 3, 2, 2, 2],
        ),
        "rgap3": (
            DeltaDispatchConfig(mode="delta_int4", revisit="int8", rgap=3),
            [0, 3, 3, 2, 2],
        ),
        "early": (DeltaDispatchConfig(mode="delta_int4", early=6), [0, 2, 2, 2, 2]),
        "late": (DeltaDispatchConfig(mode="delta_int4", late=4), [0, 2, 2, 2, 2]),
        "int4 first": (
            DeltaDispatchConfig(mode="delta_int4", first="int4"),
            [4, 3, 3, 3, 3],
        ),
        "int3": (DeltaDispatchConfig(mode="delta_int3"), [0, 5, 5, 5, 5]),
        "int3 first": (
            DeltaDispatchConfig(mode="delta_int3", first="int4"),
            [4, 5, 5, 5, 5],
        ),
        "int3 revisit int4": (
            DeltaDispatchConfig(mode="delta_int3", revisit="int4"),
            [0, 5, 3, 3, 3],
        ),
        "int3 first revisit int8 rgap3": (
            DeltaDispatchConfig(
                mode="delta_int3", first="int4", revisit="int8", rgap=3
            ),
            [4, 5, 5, 2, 2],
        ),
        # First visits keep their format on the early / late layers.
        "int3 first early": (
            DeltaDispatchConfig(mode="delta_int3", first="int4", early=6),
            [4, 2, 2, 2, 2],
        ),
        "int3 late": (DeltaDispatchConfig(mode="delta_int3", late=4), [0, 2, 2, 2, 2]),
        "nvfp4": (DeltaDispatchConfig(mode="delta_nvfp4"), [0, 8, 8, 8, 8]),
        "nvfp4 first revisit": (
            DeltaDispatchConfig(mode="delta_nvfp4", first="nvfp4", revisit="int8"),
            [7, 8, 2, 2, 2],
        ),
        "nvfp4 first late": (
            DeltaDispatchConfig(mode="delta_nvfp4", first="nvfp4", late=4),
            [7, 2, 2, 2, 2],
        ),
        "x_nvfp4": (DeltaDispatchConfig(mode="x_nvfp4"), [8, 8, 8, 8, 8]),
    }
    nvfp4_modes = {name for name, (cfg, _) in cases.items() if cfg.uses_nvfp4}
    assert nvfp4_modes == {
        "nvfp4",
        "nvfp4 first revisit",
        "nvfp4 first late",
        "x_nvfp4",
    }
    for name, (cfg, want) in cases.items():
        assert cfg.row_codes(5, 9, last).tolist() == want, name


@pytest.mark.parametrize("code,bits", [(D_INT4, 4), (D_INT8, 8), (D_INT3, 3)])
def test_quantization_matches_offline_experiment(code: int, bits: int) -> None:
    cfg = DeltaDispatchConfig(mode="delta_int3" if code == D_INT3 else "delta_int4")
    gen = torch.Generator().manual_seed(1)
    h = (torch.randn(TOKENS, HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(TOKENS, HIDDEN, generator=gen) * 3
    before = state.clone()
    size = row_bytes(code, HIDDEN)
    flat, offsets = torch.arange(TOKENS), torch.arange(TOKENS) * size
    out = torch.zeros(TOKENS * size, dtype=torch.uint8)
    meta = torch.zeros(TOKENS, dtype=torch.int32)
    last = torch.full((TOKENS,), -1, dtype=torch.int32)
    codes = torch.full((TOKENS,), code, dtype=torch.int32)
    encode_rows_reference(
        cfg, h, state, last, 3, flat, codes, offsets, TOKENS, 5, out, meta
    )
    # The Attention side's state moved by exactly the offline dequantized delta.
    assert torch.equal(state, before + _offline_quantize(h.float() - before, bits))
    assert torch.equal(meta & ((1 << CODE_SHIFT) - 1), flat.int())
    assert torch.equal((meta >> CODE_SHIFT) & CODE_MASK, codes)
    assert torch.equal(meta >> RANK_SHIFT, torch.full_like(meta, 5))
    assert torch.equal(last, torch.full_like(last, 3))
    # The FFN side rebuilds the same state from the bytes.
    ffn_state, h_out = before.clone(), torch.zeros(TOKENS, HIDDEN)
    decode_rows_reference(cfg, out, meta, offsets, flat, ffn_state, h_out)
    assert torch.equal(ffn_state, state)
    assert torch.equal(h_out, state)


@pytest.mark.parametrize(
    "mode,first,code,bits",
    [("delta_int3", "int4", H_INT4, 4), ("delta_int8", "int8", H_INT8, 8)],
)
def test_first_visit_in_int_replaces_the_state(
    mode: str, first: str, code: int, bits: int
) -> None:
    # h_int4 / h_int8: h' itself in the offline int quantizer; both sides
    # overwrite the (stale, nonzero) state with it instead of adding.
    cfg = DeltaDispatchConfig(mode=mode, first=first)
    gen = torch.Generator().manual_seed(3)
    h = (torch.randn(TOKENS, HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(TOKENS, HIDDEN, generator=gen) * 3
    size = row_bytes(code, HIDDEN)
    flat, offsets = torch.arange(TOKENS), torch.arange(TOKENS) * size
    out = torch.zeros(TOKENS * size, dtype=torch.uint8)
    meta = torch.zeros(TOKENS, dtype=torch.int32)
    last = torch.full((TOKENS,), -1, dtype=torch.int32)
    codes = torch.full((TOKENS,), code, dtype=torch.int32)
    encode_rows_reference(
        cfg, h, state, last, 0, flat, codes, offsets, TOKENS, 1, out, meta
    )
    want = _offline_quantize(h.float(), bits)
    assert torch.equal(state, want)
    assert torch.equal((meta >> CODE_SHIFT) & CODE_MASK, codes)
    ffn_state = torch.full((TOKENS, HIDDEN), 7.0)
    h_out = torch.zeros(TOKENS, HIDDEN)
    decode_rows_reference(cfg, out, meta, offsets, flat, ffn_state, h_out)
    assert torch.equal(ffn_state, want) and torch.equal(h_out, want)


@pytest.mark.parametrize("first_visit", [False, True])
def test_nvfp4_matches_offline_experiment(first_visit: bool) -> None:
    # d_nvfp4: the state moves by the offline NVFP4 of h' - S; h_nvfp4 (first
    # visit): the state becomes the offline NVFP4 of h'. Rows hold a few large
    # values; the FFN side rebuilds the same state from the bytes.
    cfg = DeltaDispatchConfig(mode="delta_nvfp4", first="nvfp4")
    gen = torch.Generator().manual_seed(7)
    h = torch.randn(TOKENS, HIDDEN, generator=gen) * 3
    h[torch.rand(TOKENS, HIDDEN, generator=gen) < 0.01] *= 20
    h = h.to(torch.bfloat16)
    state = torch.randn(TOKENS, HIDDEN, generator=gen) * 3
    before = state.clone()
    code = H_NVFP4 if first_visit else D_NVFP4
    size = row_bytes(code, HIDDEN)
    flat, offsets = torch.arange(TOKENS), torch.arange(TOKENS) * size
    out = torch.zeros(TOKENS * size, dtype=torch.uint8)
    meta = torch.zeros(TOKENS, dtype=torch.int32)
    last = torch.full((TOKENS,), -1, dtype=torch.int32)
    codes = torch.full((TOKENS,), code, dtype=torch.int32)
    encode_rows_reference(
        cfg, h, state, last, 2, flat, codes, offsets, TOKENS, 3, out, meta
    )
    if first_visit:
        want = _offline_nvfp4(h.float())
    else:
        want = before + _offline_nvfp4(h.float() - before)
    assert torch.equal(state, want)
    assert torch.equal((meta >> CODE_SHIFT) & CODE_MASK, codes)
    ffn_state, h_out = before.clone(), torch.zeros(TOKENS, HIDDEN)
    decode_rows_reference(cfg, out, meta, offsets, flat, ffn_state, h_out)
    assert torch.equal(ffn_state, want) and torch.equal(h_out, want)


def test_int3_bytes_are_a_little_endian_3_bit_stream() -> None:
    # Values that quantize to themselves (each group holds 0 and 7: lo 0,
    # step 1): the data bytes are the little-endian bytes of sum(q_i << 3 i),
    # followed by each group's fp16 (lo, step).
    cfg = DeltaDispatchConfig(mode="delta_int3")
    gen = torch.Generator().manual_seed(5)
    bits = 3
    levels = (1 << bits) - 1
    q = torch.randint(0, levels + 1, (HIDDEN,), generator=gen)
    q[::QUANT_GROUP_SIZE], q[1::QUANT_GROUP_SIZE] = 0, levels
    data_bytes = HIDDEN * bits // 8
    stream = sum(int(v) << (bits * i) for i, v in enumerate(q.tolist()))
    groups = HIDDEN // QUANT_GROUP_SIZE
    scales = torch.tensor([0.0, 1.0] * groups, dtype=torch.float16)
    want = list(stream.to_bytes(data_bytes, "little"))
    want += scales.view(torch.uint8).tolist()
    out = torch.zeros(row_bytes(D_INT3, HIDDEN), dtype=torch.uint8)
    state = torch.zeros(1, HIDDEN)
    encode_rows_reference(
        cfg,
        q.view(1, -1).to(torch.bfloat16),
        state,
        torch.full((1,), -1, dtype=torch.int32),
        1,
        torch.zeros(1, dtype=torch.int64),
        torch.full((1,), D_INT3, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int64),
        1,
        0,
        out,
        torch.zeros(1, dtype=torch.int32),
    )
    assert out.tolist() == want
    assert torch.equal(state.view(-1), q.float())


def test_row_words_hold_every_code_token_and_rank() -> None:
    # One row per format for the largest token and Attention rank: the word
    # stays a non-negative int32 and gives back token, code and rank, and the
    # bytes decode (no states: src is a broadcast view, no big buffers).
    cfg = DeltaDispatchConfig(mode="full_bf16")
    gen = torch.Generator().manual_seed(6)
    h = (torch.randn(1, HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    tokens = TOKEN_MASK + 1
    rank = MAX_ATTENTION_RANKS - 1
    codes = torch.arange(len(CODE_FORMATS), dtype=torch.int32)
    sizes = torch.tensor([row_bytes(int(c), HIDDEN) for c in codes])
    offsets = torch.cumsum(sizes, 0) - sizes
    flat = torch.full((len(codes),), TOKEN_MASK, dtype=torch.int64)
    out = torch.zeros(int(sizes.sum()), dtype=torch.uint8)
    meta = torch.zeros(len(codes), dtype=torch.int32)
    src = h.expand(tokens, HIDDEN)
    encode_rows_reference(
        cfg, src, None, None, 0, flat, codes, offsets, tokens, rank, out, meta
    )
    assert (meta >= 0).all()
    assert torch.equal(meta & TOKEN_MASK, flat.int())
    assert torch.equal((meta >> CODE_SHIFT) & CODE_MASK, codes)
    assert torch.equal(meta >> RANK_SHIFT, torch.full_like(meta, rank))
    h_out = torch.zeros(len(codes), HIDDEN)
    decode_rows_reference(cfg, out, meta, offsets, flat, None, h_out)
    want = {
        H_BF16: h.float(),
        D_FP32: h.float(),
        D_INT8: _offline_quantize(h.float(), 8),
        D_INT4: _offline_quantize(h.float(), 4),
        H_INT4: _offline_quantize(h.float(), 4),
        D_INT3: _offline_quantize(h.float(), 3),
        H_INT8: _offline_quantize(h.float(), 8),
        H_NVFP4: _offline_nvfp4(h.float()),
        D_NVFP4: _offline_nvfp4(h.float()),
    }
    for code in codes.tolist():
        assert torch.equal(h_out[code], want[code].view(-1)), CODE_FORMATS[code]
    # The Attention side's states only take tokens the word can hold.
    AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, 1, tokens, HIDDEN)
    with pytest.raises(ValueError):
        AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, 1, tokens + 1, HIDDEN)


def test_constant_group_has_no_nan() -> None:
    cfg = DeltaDispatchConfig(mode="delta_int4")
    h = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
    state = torch.zeros(TOKENS, HIDDEN)
    size = row_bytes(D_INT4, HIDDEN)
    flat, offsets = torch.arange(TOKENS), torch.arange(TOKENS) * size
    out = torch.zeros(TOKENS * size, dtype=torch.uint8)
    meta = torch.zeros(TOKENS, dtype=torch.int32)
    codes = torch.full((TOKENS,), D_INT4, dtype=torch.int32)
    last = torch.full((TOKENS,), -1, dtype=torch.int32)
    encode_rows_reference(
        cfg, h, state, last, 1, flat, codes, offsets, TOKENS, 0, out, meta
    )
    decoded, h_out = torch.zeros(TOKENS, HIDDEN), torch.zeros(TOKENS, HIDDEN)
    decode_rows_reference(cfg, out, meta, offsets, flat, decoded, h_out)
    assert torch.isfinite(state).all() and torch.equal(decoded, state)


@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
@pytest.mark.parametrize("num_ffn,num_attn,stages", [(1, 1, 1), (4, 2, 2)])
def test_uncompressed_modes_give_norm_of_h_prime(
    mode: str, num_ffn: int, num_attn: int, stages: int
) -> None:
    # Without compression every FFN rank hands its experts exactly Norm(h').
    steps = _drive(DeltaDispatchConfig(mode=mode), num_ffn, num_attn, stages)
    for step in steps:
        assert step["inputs"]
        for (_, t), x in step["inputs"].items():
            assert torch.equal(x, step["norm_h"][t])


@pytest.mark.parametrize(
    "mode,first,revisit,want_codes",
    [
        ("delta_int4", "bf16", "int4", {H_BF16, D_INT4, D_INT8}),
        ("delta_int4", "bf16", "int8", {H_BF16, D_INT4, D_INT8}),
        ("delta_int4", "int4", "int8", {H_INT4, D_INT4, D_INT8}),
        ("delta_int3", "bf16", "", {H_BF16, D_INT3, D_INT8}),
        ("delta_int3", "int4", "", {H_INT4, D_INT3, D_INT8}),
        ("delta_int3", "int4", "int4", {H_INT4, D_INT3, D_INT4, D_INT8}),
        ("delta_int8", "int8", "", {H_INT8, D_INT8}),
        ("delta_int8", "bf16", "", {H_BF16, D_INT8}),
        ("delta_nvfp4", "nvfp4", "", {H_NVFP4, D_NVFP4, D_INT8}),
        ("delta_nvfp4", "bf16", "int8", {H_BF16, D_NVFP4, D_INT8}),
    ],
)
def test_quantized_states_stay_identical_across_ffn_ranks(
    mode: str, first: str, revisit: str, want_codes: set[int]
) -> None:
    num_ffn, num_attn, stages = 4, 2, 2
    cfg = DeltaDispatchConfig(mode=mode, late=1, revisit=revisit, first=first)
    steps = _drive(cfg, num_ffn, num_attn, stages)
    codes = set()
    for step in steps:
        codes |= set(step["codes"].tolist())
        # The bytes per FFN rank are the sizes of its rows.
        sizes = torch.tensor([row_bytes(int(c), HIDDEN) for c in step["codes"]])
        per_ffn = torch.zeros(num_ffn, dtype=torch.int64)
        per_ffn.index_add_(0, step["flat"] // TOKENS, sizes)
        assert torch.equal(per_ffn, step["bytes"])
        # Errors stay bounded (no accumulation).
        for (_, t), x in step["inputs"].items():
            norm_h = step["norm_h"][t].float()
            assert (x.float() - norm_h).norm() / norm_h.norm() < 0.15
        if "ffn_states" not in step:
            continue
        # Closed loop: every FFN rank holds the Attention side's copy of its
        # states (FFN row Attention rank * T + token, Attention row FFN rank
        # * T + token).
        for f, ffn_state in enumerate(step["ffn_states"]):
            for a, attn_state in enumerate(step["attn_states"]):
                assert torch.equal(
                    ffn_state[a * TOKENS : (a + 1) * TOKENS],
                    attn_state[f * TOKENS : (f + 1) * TOKENS],
                )
    assert codes == want_codes


def test_first_visit_and_revisit_follow_the_offline_rules() -> None:
    # Four FFN ranks, revisit int8: a token's first row at an FFN rank in a
    # forward is bf16, rows after a gap of 2+ MoE layers int8, others int4.
    steps = _drive(DeltaDispatchConfig(mode="delta_int4", revisit="int8"), 4, 1, 1)
    last: dict[tuple[int, int], int] = {}
    for pos, step in enumerate(steps):
        for flat, code in zip(
            step["flat"].tolist(), step["codes"].tolist(), strict=True
        ):
            f, t = divmod(flat, TOKENS)
            prev = last.get((f, t))
            want = H_BF16 if prev is None else (D_INT8 if pos - prev >= 2 else D_INT4)
            assert code == want
            last[(f, t)] = pos
    assert any(len(set(s["codes"].tolist())) > 1 for s in steps)


@pytest.mark.parametrize(
    "mode,code,bits", [("x_int4", D_INT4, 4), ("x_int8", D_INT8, 8)]
)
def test_x_modes_are_the_plain_quantizer(mode: str, code: int, bits: int) -> None:
    # No state: every FFN rank gets x = Norm(h') quantized per row.
    for step in _drive(DeltaDispatchConfig(mode=mode), 4, 2, 1):
        assert set(step["codes"].tolist()) == {code}
        for (_, t), x in step["inputs"].items():
            want = _offline_quantize(step["norm_h"][t].float().view(1, -1), bits)
            assert torch.equal(x, want.view(-1).to(torch.bfloat16))


def test_x_nvfp4_is_the_plain_nvfp4() -> None:
    for step in _drive(DeltaDispatchConfig(mode="x_nvfp4"), 4, 2, 1):
        assert set(step["codes"].tolist()) == {D_NVFP4}
        for (_, t), x in step["inputs"].items():
            want = _offline_nvfp4(step["norm_h"][t].float().view(1, -1))
            assert torch.equal(x, want.view(-1).to(torch.bfloat16))


def test_missing_rows_are_an_error() -> None:
    cfg = DeltaDispatchConfig(mode="delta_fp32")
    ffn = FFNDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, HIDDEN, 1, TOKENS)
    no_rows = torch.zeros(0, dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        ffn.decode(MOE_LAYERS[0], 0, no_rows, _norm_into, 1, torch.bfloat16)
    attn = AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, 1, TOKENS, HIDDEN)
    with pytest.raises(RuntimeError):
        attn.prepare(99, 0, torch.arange(TOKENS), TOKENS)


@needs_kernels
@pytest.mark.parametrize("hidden", [MODEL_HIDDEN, QUANT_GROUP_SIZE])
@pytest.mark.parametrize(
    "mode,first",
    [
        ("delta_int4", "bf16"),
        ("delta_fp32", "bf16"),
        ("full_bf16", "bf16"),
        ("x_int4", "bf16"),
        ("x_int8", "bf16"),
        ("delta_int3", "int4"),
        ("delta_int4", "int4"),
        ("delta_int3", "bf16"),
        ("delta_int8", "int8"),
        ("delta_int8", "bf16"),
        ("delta_nvfp4", "nvfp4"),
        ("delta_nvfp4", "bf16"),
        ("x_nvfp4", "bf16"),
    ],
)
@pytest.mark.parametrize(("nvfp4_pass", "hardware"), NVFP4_KERNEL_VARIANTS)
def test_triton_kernels_match_reference(
    mode: str, first: str, hidden: int, nvfp4_pass: str, hardware: bool
) -> None:
    from afd_plugin import delta_dispatch_triton

    revisit = {"delta_int4": "int8", "delta_int3": "int4", "delta_nvfp4": "int8"}.get(
        mode, ""
    )
    cfg = DeltaDispatchConfig(mode=mode, revisit=revisit, first=first)
    if not cfg.uses_nvfp4 and (nvfp4_pass, hardware) != ("two_pass", False):
        pytest.skip("no NVFP4 rows: one kernel structure")
    gen = torch.Generator().manual_seed(2)
    tokens, rows, rank = 64, 40, MAX_ATTENTION_RANKS - 1
    flat = torch.randperm(3 * tokens, generator=gen)[:rows].sort().values
    # Per state row, the MoE layer of its last visit (-1: none).
    last = torch.randint(-1, 3, (3 * tokens,), generator=gen, dtype=torch.int32)
    codes = cfg.row_codes(3, 8, last[flat])
    sizes = torch.tensor([row_bytes(int(c), hidden) for c in codes])
    offsets = torch.cumsum(sizes, 0) - sizes
    h = (torch.randn(tokens, hidden, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(3 * tokens, hidden, generator=gen) * 3
    total = int(sizes.sum())
    ref_state = state.clone()
    ref_last = last.clone()
    ref_out = torch.zeros(total, dtype=torch.uint8)
    ref_meta = torch.zeros(rows, dtype=torch.int32)
    encode_rows_reference(
        cfg,
        h,
        ref_state,
        ref_last,
        3,
        flat,
        codes,
        offsets,
        tokens,
        rank,
        ref_out,
        ref_meta,
    )
    gpu_state = state.to(KERNEL_DEVICE, copy=True)
    gpu_last = last.to(KERNEL_DEVICE, copy=True)
    out = torch.zeros(total, dtype=torch.uint8, device=KERNEL_DEVICE)
    meta = torch.zeros(rows, dtype=torch.int32, device=KERNEL_DEVICE)
    delta_dispatch_triton.encode_rows(
        cfg,
        h.to(KERNEL_DEVICE, copy=True),
        gpu_state,
        gpu_last,
        3,
        flat.to(KERNEL_DEVICE, copy=True),
        codes.to(KERNEL_DEVICE, copy=True),
        offsets.to(KERNEL_DEVICE, copy=True),
        tokens,
        rank,
        out,
        meta,
        nvfp4_pass=nvfp4_pass,
        hardware_e4m3=hardware,
    )
    # Same bytes and words as the reference; the states differ from it only
    # by the fused multiply-add of the int dequantization (NVFP4: none).
    assert torch.equal(out.cpu(), ref_out) and torch.equal(meta.cpu(), ref_meta)
    torch.testing.assert_close(gpu_state.cpu(), ref_state, rtol=1e-6, atol=1e-6)
    if cfg.stateful:
        assert torch.equal(gpu_last.cpu(), ref_last)
    # The FFN kernel rebuilds the Attention kernel's states bit for bit.
    ffn_state = state.to(KERNEL_DEVICE, copy=True)
    h_out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=KERNEL_DEVICE)
    delta_dispatch_triton.decode_rows(
        cfg,
        out,
        meta,
        offsets.to(KERNEL_DEVICE, copy=True),
        flat.to(KERNEL_DEVICE, copy=True),
        ffn_state,
        h_out,
        hardware_e4m3=hardware,
    )
    if cfg.stateful:
        rows_idx = flat.to(KERNEL_DEVICE, copy=True)
        assert torch.equal(ffn_state[rows_idx], gpu_state[rows_idx])
        _assert_same_model_dtype(h_out, gpu_state[rows_idx])


@needs_kernels
@pytest.mark.parametrize(
    "mode,first",
    [
        ("delta_int4", "int4"),
        ("delta_int8", "int8"),
        ("x_int4", "bf16"),
        ("x_int8", "bf16"),
        ("full_bf16", "bf16"),
        (
            "delta_int3",
            "int4",
        ),  # sizes differ: offsets given, rows still from the words
        ("delta_nvfp4", "nvfp4"),
        ("x_nvfp4", "bf16"),
    ],
)
@pytest.mark.parametrize(("nvfp4_pass", "hardware"), NVFP4_KERNEL_VARIANTS)
def test_triton_one_row_size_paths_match_reference(
    mode: str, first: str, nvfp4_pass: str, hardware: bool
) -> None:
    # One row size: the encode kernel picks the codes (first visit or not)
    # and the host updates the visits after it; the decode kernel finds each
    # row at index * size and reads its state row from its word.
    from afd_plugin import delta_dispatch_triton

    cfg = DeltaDispatchConfig(mode=mode, first=first)
    if not cfg.uses_nvfp4 and (nvfp4_pass, hardware) != ("two_pass", False):
        pytest.skip("no NVFP4 rows: one kernel structure")
    size = cfg.fixed_row_bytes(MODEL_HIDDEN)
    gen = torch.Generator().manual_seed(5)
    tokens, rows, rank, max_tokens = 64, 40, 3, 96
    flat = torch.randperm(3 * tokens, generator=gen)[:rows].sort().values
    last = torch.randint(-1, 3, (3 * tokens,), generator=gen, dtype=torch.int32)
    if size:
        codes, offsets = None, torch.arange(rows) * size
    else:
        codes = cfg.row_codes(3, 8, last[flat])
        sizes = torch.tensor([row_bytes(int(c), MODEL_HIDDEN) for c in codes])
        offsets = torch.cumsum(sizes, 0) - sizes
    total = int(offsets[-1]) + cfg.max_row_bytes(MODEL_HIDDEN)
    h = (torch.randn(tokens, MODEL_HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(3 * tokens, MODEL_HIDDEN, generator=gen) * 3
    ref_state, ref_last = state.clone(), last.clone()
    ref_out = torch.zeros(total, dtype=torch.uint8)
    ref_meta = torch.zeros(rows, dtype=torch.int32)
    encode_rows_reference(
        cfg,
        h,
        ref_state,
        ref_last,
        3,
        flat,
        codes,
        offsets,
        tokens,
        rank,
        ref_out,
        ref_meta,
    )
    gpu_state, gpu_last = (
        state.to(KERNEL_DEVICE, copy=True),
        last.to(KERNEL_DEVICE, copy=True),
    )
    out = torch.zeros(total, dtype=torch.uint8, device=KERNEL_DEVICE)
    meta = torch.zeros(rows, dtype=torch.int32, device=KERNEL_DEVICE)
    delta_dispatch_triton.encode_rows(
        cfg,
        h.to(KERNEL_DEVICE, copy=True),
        gpu_state,
        gpu_last,
        3,
        flat.to(KERNEL_DEVICE, copy=True),
        None if codes is None else codes.to(KERNEL_DEVICE, copy=True),
        offsets.to(KERNEL_DEVICE, copy=True),
        tokens,
        rank,
        out,
        meta,
        nvfp4_pass=nvfp4_pass,
        hardware_e4m3=hardware,
    )
    assert torch.equal(out.cpu(), ref_out) and torch.equal(meta.cpu(), ref_meta)
    torch.testing.assert_close(gpu_state.cpu(), ref_state, rtol=1e-6, atol=1e-6)
    if cfg.stateful:
        assert torch.equal(gpu_last.cpu(), ref_last)
    # One FFN rank's rows (destination-major: the first k), its state rows
    # from the words: Attention rank * max_tokens + token.
    k = int((flat < tokens).sum())
    ffn_state = torch.randn(4 * max_tokens, MODEL_HIDDEN, generator=gen).to(
        KERNEL_DEVICE, copy=True
    )
    want = (rank * max_tokens + flat[:k] % tokens).to(KERNEL_DEVICE, copy=True)
    # The FFN holds what Attention held before this layer (delta rows add to
    # it; first visits overwrite it).
    ffn_state[want] = state[flat[:k]].to(KERNEL_DEVICE, copy=True)
    h_out = torch.empty(k, MODEL_HIDDEN, dtype=torch.bfloat16, device=KERNEL_DEVICE)
    rows_out = torch.empty(k, dtype=torch.int32, device=KERNEL_DEVICE)
    delta_dispatch_triton.decode_rows(
        cfg,
        out,
        meta[:k],
        None if size else offsets[:k].to(KERNEL_DEVICE, copy=True),
        None,
        ffn_state if cfg.stateful else None,
        h_out,
        rows_out=rows_out,
        max_tokens=max_tokens,
        row_size=size,
        hardware_e4m3=hardware,
    )
    assert torch.equal(rows_out.long(), want)
    if cfg.stateful:
        assert torch.equal(
            ffn_state[want], gpu_state[flat[:k].to(KERNEL_DEVICE, copy=True)]
        )
        _assert_same_model_dtype(h_out, ffn_state[want])


def test_each_forward_starts_with_first_visits() -> None:
    # The first MoE layer of a forward clears the visits: every row is bf16
    # again, whatever the previous forward left.
    cfg = DeltaDispatchConfig(mode="delta_int4")
    attn = AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, 2, TOKENS, HIDDEN)
    order = torch.arange(2 * TOKENS)
    h = torch.randn(TOKENS, HIDDEN).to(torch.bfloat16)
    for _forward in range(2):
        for pos, layer in enumerate(MOE_LAYERS):
            packed = attn.prepare(layer, 0, order, TOKENS)
            want = H_BF16 if pos == 0 else D_INT4
            assert set(packed.codes.tolist()) == {want}
            out = torch.zeros(int(packed.totals.sum()), dtype=torch.uint8)
            meta = torch.zeros(2 * TOKENS, dtype=torch.int32)
            attn.encode(packed, h, 2 * TOKENS, out, meta, 0)
    # One row size (first visits int4): the encode kernel picks the codes,
    # the words carry them.
    cfg = DeltaDispatchConfig(mode="delta_int4", first="int4")
    attn = AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, 2, TOKENS, HIDDEN)
    for _forward in range(2):
        for pos, layer in enumerate(MOE_LAYERS):
            packed = attn.prepare(layer, 0, order, TOKENS)
            assert packed.codes is None and packed.totals is None
            assert packed.row_size == row_bytes(D_INT4, HIDDEN)
            assert (
                attn.ffn_bytes(packed, [TOKENS, TOKENS])
                == [TOKENS * packed.row_size] * 2
            )
            out = torch.zeros(2 * TOKENS * packed.row_size, dtype=torch.uint8)
            meta = torch.zeros(2 * TOKENS, dtype=torch.int32)
            attn.encode(packed, h, 2 * TOKENS, out, meta, 0)
            want = H_INT4 if pos == 0 else D_INT4
            assert set(((meta >> CODE_SHIFT) & CODE_MASK).tolist()) == {want}


def test_combine_config_parsing() -> None:
    assert not CombineConfig.from_extra_config(None).enabled
    cfg = CombineConfig.from_extra_config({"combine": {"mode": "int4"}})
    assert cfg.enabled and cfg.ef and cfg.code == D_INT4
    assert cfg.max_row_bytes(MODEL_HIDDEN) == 1088
    assert {cfg.row_bytes_at(p, 27, MODEL_HIDDEN) for p in range(27)} == {1088}
    cfg = CombineConfig.from_mapping({"mode": "int8", "ef": False})
    assert cfg.code == D_INT8 and not cfg.ef
    assert cfg.max_row_bytes(MODEL_HIDDEN) == 2112
    cfg = CombineConfig.from_mapping({"mode": "int4", "early": 4, "late": 4})
    assert (cfg.early, cfg.late, cfg.code) == (4, 4, D_INT4)
    # 27 MoE layers: positions 0-3 and 23-26 int8, the rest int4.
    assert [cfg.code_at(p, 27) for p in (0, 3, 4, 22, 23, 26)] == [
        D_INT8,
        D_INT8,
        D_INT4,
        D_INT4,
        D_INT8,
        D_INT8,
    ]
    assert cfg.row_bytes_at(3, 27, MODEL_HIDDEN) == 2112
    assert cfg.row_bytes_at(4, 27, MODEL_HIDDEN) == 1088
    assert cfg.max_row_bytes(MODEL_HIDDEN) == 2112
    assert CombineConfig(mode="int4").max_row_bytes(MODEL_HIDDEN) == 1088
    cfg = CombineConfig.from_mapping({"mode": "nvfp4", "early": 1, "late": 2})
    assert cfg.code == D_NVFP4 and cfg.ef
    assert [cfg.code_at(p, 27) for p in (0, 1, 24, 25)] == [
        D_INT8,
        D_NVFP4,
        D_NVFP4,
        D_INT8,
    ]
    assert CombineConfig(mode="nvfp4").max_row_bytes(MODEL_HIDDEN) == 1156
    assert cfg.max_row_bytes(MODEL_HIDDEN) == 2112
    for bad in (
        {"mode": "fp8"},
        {"mode": "int4", "ef": 1},
        {"bits": 4},
        {"mode": "int8", "early": 2},
        {"mode": "off", "late": 1},
        {"mode": "int4", "early": -1},
    ):
        with pytest.raises(ValueError):
            CombineConfig.from_mapping(bad)


def _offline_combine(code: int, x: torch.Tensor) -> torch.Tensor:
    if code == D_NVFP4:
        return _offline_nvfp4(x)
    return _offline_quantize(x, 8 if code == D_INT8 else 4)


@pytest.mark.parametrize("code", [D_INT8, D_INT4, D_NVFP4])
def test_combine_rows_match_offline_quantizer(code: int) -> None:
    # Without error feedback the received value is the offline quantizer's
    # (compress.py) of the partial; with it, of partial + residual, and the
    # new residual is what was cut.
    gen = torch.Generator().manual_seed(4)
    n = 9
    partial = (torch.randn(n, HIDDEN, generator=gen) * 2).to(torch.bfloat16)
    rows = torch.randperm(2 * TOKENS, generator=gen)[:n]
    size = row_bytes(code, HIDDEN)
    data = torch.zeros(n * size + 5, dtype=torch.uint8)
    deq = torch.empty(n, HIDDEN)
    encode_combine_reference(code, partial, rows, None, data, deq)
    assert torch.equal(deq, _offline_combine(code, partial.float()))
    back = torch.empty(n, HIDDEN)
    decode_combine_reference(code, data, back)
    assert torch.equal(back, deq)
    err = torch.randn(2 * TOKENS, HIDDEN, generator=gen) * 0.1
    before = err.clone()
    encode_combine_reference(code, partial, rows, err, data, deq)
    v = partial.float() + before[rows]
    assert torch.equal(deq, _offline_combine(code, v))
    assert torch.equal(err[rows], v - deq)
    untouched = torch.ones(2 * TOKENS, dtype=torch.bool)
    untouched[rows] = False
    assert torch.equal(err[untouched], before[untouched])


@pytest.mark.parametrize("mode,step", [("int8", 0.2), ("int4", 0.2), ("nvfp4", 0.75)])
def test_combine_error_feedback_keeps_the_sum(mode: str, step: float) -> None:
    # Over the visits of a forward the received partials add up to the exact
    # ones minus the last residual (sigma-delta): the cut parts are not lost.
    combine = CombineConfig(mode=mode)
    ffn = FFNDeltaState(
        DeltaDispatchConfig(mode="delta_fp32"),
        MOE_LAYERS,
        1.0,
        HIDDEN,
        1,
        TOKENS,
        combine,
    )
    gen = torch.Generator().manual_seed(6)
    exact = torch.zeros(TOKENS, HIDDEN)
    received = torch.zeros(TOKENS, HIDDEN)
    rows = torch.arange(TOKENS)
    for pos, _ in enumerate(MOE_LAYERS):
        ffn._rows[0] = rows
        ffn._pos[0] = pos
        partial = (torch.randn(TOKENS, HIDDEN, generator=gen)).to(torch.bfloat16)
        ffn._state[0] = torch.zeros(TOKENS, HIDDEN)
        ffn.after_experts(0, partial)
        back = torch.empty(TOKENS, HIDDEN)
        decode_combine_reference(combine.code, ffn.combine_bytes(0, TOKENS), back)
        exact += partial.float()
        received += back
    torch.testing.assert_close(received, exact - ffn._err[0], rtol=0, atol=1e-5)
    # The residual stays at one quantization step (NVFP4: the half gap between
    # its largest values, 4 and 6, times the block scale), not the
    # accumulated error.
    assert float(ffn._err[0].abs().max()) < step


@pytest.mark.parametrize(
    "delta_mode",
    [
        "delta_int4",
        "delta_int3",
        "delta_int8",
        "delta_fp32",
        "x_int4",
        "x_int8",
        "delta_nvfp4",
        "x_nvfp4",
    ],
)
@pytest.mark.parametrize(
    "mode,ef",
    [
        ("int4", True),
        ("int8", True),
        ("int4", False),
        ("nvfp4", True),
        ("nvfp4", False),
    ],
)
def test_combine_keeps_both_states_identical(
    delta_mode: str, mode: str, ef: bool
) -> None:
    # Four FFN ranks, two Attention ranks, two stages, two forwards: both sides
    # add what the Attention side received, so the states stay identical; the
    # received partial sums are the offline quantizer's of partial (+ residual).
    revisit = {"delta_int4": "int8", "delta_int3": "int4"}.get(delta_mode, "")
    first = {"delta_int3": "int4", "delta_int8": "int8", "delta_nvfp4": "nvfp4"}.get(
        delta_mode, "bf16"
    )
    cfg = DeltaDispatchConfig(mode=delta_mode, revisit=revisit, first=first)
    steps = _drive(cfg, 4, 2, 2, CombineConfig(mode=mode, ef=ef), forwards=2)
    for step in steps:
        if "ffn_states" in step and cfg.stateful:
            for f, f_state in enumerate(step["ffn_states"]):
                for a, a_state in enumerate(step["attn_states"]):
                    if f_state is None or a_state is None:
                        continue
                    assert torch.equal(
                        f_state[a * TOKENS : (a + 1) * TOKENS],
                        a_state[f * TOKENS : (f + 1) * TOKENS],
                    )
        err = (step["received"] - step["exact"]).abs().max()
        assert 0 < float(err) < (1.0 if mode == "nvfp4" else 0.5)
    if not ef:
        code = CombineConfig(mode=mode).code
        for step in steps:
            assert torch.equal(
                step["received"],
                _offline_combine(code, step["exact"]).to(torch.bfloat16).float(),
            )


@pytest.mark.parametrize("ef", [False, True])
def test_combine_int8_layers_follow_early_and_late(ef: bool) -> None:
    # int4 with int8 on the first 2 and the last 1 MoE layers (the offline
    # cearly / clate): each layer's rows are the offline quantizer's at that
    # layer's width, both sides agree on the row size, the states stay equal.
    combine = CombineConfig(mode="int4", ef=ef, early=2, late=1)
    cfg = DeltaDispatchConfig(mode="delta_int4", first="int4")
    steps = _drive(cfg, 4, 2, 2, combine, forwards=2)
    count = len(MOE_LAYERS)
    wide = {MOE_LAYERS[p] for p in range(count) if p < 2 or p >= count - 1}
    for step in steps:
        bits = 8 if step["layer"] in wide else 4
        if not ef:
            assert torch.equal(
                step["received"],
                _offline_quantize(step["exact"], bits).to(torch.bfloat16).float(),
            )
        if "ffn_states" in step:
            for f, f_state in enumerate(step["ffn_states"]):
                for a, a_state in enumerate(step["attn_states"]):
                    assert torch.equal(
                        f_state[a * TOKENS : (a + 1) * TOKENS],
                        a_state[f * TOKENS : (f + 1) * TOKENS],
                    )
    # int8 layers cut less than int4 layers.
    err = {
        w: max(
            float((s["received"] - s["exact"]).abs().max())
            for s in steps
            if (s["layer"] in wide) == w
        )
        for w in (True, False)
    }
    assert err[True] < err[False]


def test_combine_residual_starts_at_zero_every_forward() -> None:
    combine = CombineConfig(mode="int4")
    ffn = FFNDeltaState(
        DeltaDispatchConfig(mode="delta_int4"),
        MOE_LAYERS,
        1.0,
        HIDDEN,
        1,
        TOKENS,
        combine,
    )
    ffn._err[0] = torch.ones(TOKENS, HIDDEN)
    ffn.receive(0, torch.zeros(0, dtype=torch.int32))
    ffn.decode(
        MOE_LAYERS[1],
        0,
        torch.zeros(0, dtype=torch.uint8),
        _norm_into,
        2,
        torch.bfloat16,
    )
    assert float(ffn._err[0].sum()) == TOKENS * HIDDEN
    ffn.receive(0, torch.zeros(0, dtype=torch.int32))
    ffn.decode(
        MOE_LAYERS[0],
        0,
        torch.zeros(0, dtype=torch.uint8),
        _norm_into,
        2,
        torch.bfloat16,
    )
    assert float(ffn._err[0].abs().sum()) == 0


@needs_kernels
@pytest.mark.parametrize("code", [D_INT8, D_INT4, D_NVFP4])
@pytest.mark.parametrize("ef", [False, True])
@pytest.mark.parametrize(("nvfp4_pass", "hardware"), NVFP4_KERNEL_VARIANTS)
def test_triton_combine_kernels_match_reference(
    code: int, ef: bool, nvfp4_pass: str, hardware: bool
) -> None:
    from afd_plugin import delta_dispatch_triton

    if code != D_NVFP4 and (nvfp4_pass, hardware) != ("two_pass", False):
        pytest.skip("not NVFP4: one kernel structure")
    gen = torch.Generator().manual_seed(8)
    n, slots = 40, 64
    partial = (torch.randn(n, MODEL_HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    rows = torch.randperm(slots, generator=gen)[:n]
    err = torch.randn(slots, MODEL_HIDDEN, generator=gen) * 0.1 if ef else None
    size = row_bytes(code, MODEL_HIDDEN)
    ref_data = torch.zeros(n * size, dtype=torch.uint8)
    ref_deq = torch.empty(n, MODEL_HIDDEN)
    ref_err = err.clone() if ef else None
    encode_combine_reference(code, partial, rows, ref_err, ref_data, ref_deq)
    data = torch.zeros(n * size, dtype=torch.uint8, device=KERNEL_DEVICE)
    deq = torch.empty(n, MODEL_HIDDEN, device=KERNEL_DEVICE)
    gpu_err = err.to(KERNEL_DEVICE, copy=True) if ef else None
    delta_dispatch_triton.encode_combine(
        code,
        partial.to(KERNEL_DEVICE, copy=True),
        rows.to(KERNEL_DEVICE, copy=True),
        gpu_err,
        data,
        deq,
        nvfp4_pass=nvfp4_pass,
        hardware_e4m3=hardware,
    )
    # Same bytes; deq and the residual differ from the reference only by the
    # fused multiply-add of the dequantization.
    assert torch.equal(data.cpu(), ref_data)
    torch.testing.assert_close(deq.cpu(), ref_deq, rtol=1e-6, atol=1e-6)
    if ef:
        torch.testing.assert_close(gpu_err.cpu(), ref_err, rtol=1e-6, atol=1e-6)
    # The Attention kernel rebuilds exactly what the FFN kernel added.
    back = torch.empty(n, MODEL_HIDDEN, device=KERNEL_DEVICE)
    delta_dispatch_triton.decode_combine(code, data, back, hardware_e4m3=hardware)
    assert torch.equal(back, deq)
