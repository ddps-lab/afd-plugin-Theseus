# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch arithmetic (afd_plugin/delta_dispatch.py), without vLLM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from afd_plugin.delta_dispatch import (
    CODE_MASK,
    CODE_SHIFT,
    D_FP32,
    D_INT4,
    D_INT8,
    H_BF16,
    QUANT_GROUP_SIZE,
    RANK_SHIFT,
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
    rb = combine.row_bytes(HIDDEN) if combine.enabled else 0
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
                out = torch.zeros(int(packed.totals.sum()) + 16, dtype=torch.uint8)
                meta = torch.zeros(n, dtype=torch.int32)
                src = _norm(h_prime) if cfg.sends_x else h_prime
                attn[a].encode(packed, src, n, out, meta, a)
                sent[a] = (h_prime, ids, weights, packed, n, out, meta)
            # Every FFN rank receives its rows of all Attention ranks
            # (Attention rank major), rebuilds them and runs its experts.
            results: dict[int, list] = {a: [] for a in range(num_attn)}
            for f in range(num_ffn):
                datas, words, origin = [], [], []
                for a in range(num_attn):
                    _, ids, weights, packed, n, out, meta = sent[a]
                    rows = (packed.flat[:n] // TOKENS == f).nonzero().view(-1)
                    if rows.numel():
                        start = int(packed.offsets[rows[0]])
                        datas.append(out[start : start + int(packed.totals[f])])
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
                    _, ids, weights, packed, _, _, _ = sent[a]
                    t = int(packed.flat[i]) % TOKENS
                    for k in range(TOP_K):
                        if int(owner[ids[t, k]]) == f:
                            partial[r] += weights[t, k] * _expert(x[r], int(ids[t, k]))
                partial = partial.to(torch.bfloat16)
                ffn[f].after_experts(s, partial)
                if combine.enabled:
                    # The compressed rows, Attention rank major like the
                    # received ones: each Attention rank's share is a span.
                    cbytes = ffn[f].combine_bytes(s, len(origin)).clone()
                    assert cbytes.numel() == len(origin) * rb
                for r, (a, i) in enumerate(origin):
                    wire = cbytes[r * rb : (r + 1) * rb] if combine.enabled else None
                    results[a].append((i, partial[r], x[r].clone(), f, wire))
            for a in range(num_attn):
                h_prime, ids, weights, packed, n, _, meta = sent[a]
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
                    received = attn[a].decode_combine(s, data, n).clone()
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
                        "codes": packed.codes[:n].clone(),
                        "meta": meta.clone(),
                        "bytes": packed.totals.clone(),
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
    assert DeltaDispatchConfig(mode="x_int4").sends_x
    assert not DeltaDispatchConfig(mode="x_int4").stateful
    for bad in (
        {"mode": "int3"},
        {"mode": "off", "extra": 1},
        {"mode": "delta_fp32", "early": 2},
        {"mode": "x_int4", "revisit": "int8"},
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


def test_row_formats_and_codes() -> None:
    assert row_bytes(H_BF16, MODEL_HIDDEN) == 4096
    assert row_bytes(D_FP32, MODEL_HIDDEN) == 8192
    assert row_bytes(D_INT8, MODEL_HIDDEN) == 2112
    assert row_bytes(D_INT4, MODEL_HIDDEN) == 1088
    assert DeltaDispatchConfig(mode="delta_int4").max_row_bytes(MODEL_HIDDEN) == 4096
    assert DeltaDispatchConfig(mode="x_int4").max_row_bytes(MODEL_HIDDEN) == 1088
    # Previous visits of five rows before MoE layer position 5 of 9 (-1: none).
    last = torch.tensor([-1, 4, 3, 2, 0], dtype=torch.int32)
    cases = {
        "full_bf16": (DeltaDispatchConfig(mode="full_bf16"), [0, 0, 0, 0, 0]),
        "x_int4": (DeltaDispatchConfig(mode="x_int4"), [3, 3, 3, 3, 3]),
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
    }
    for name, (cfg, want) in cases.items():
        assert cfg.row_codes(5, 9, last).tolist() == want, name


@pytest.mark.parametrize("code,bits", [(D_INT4, 4), (D_INT8, 8)])
def test_quantization_matches_offline_experiment(code: int, bits: int) -> None:
    cfg = DeltaDispatchConfig(mode="delta_int4")
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


@pytest.mark.parametrize("revisit", ["int4", "int8"])
def test_int4_states_stay_identical_across_ffn_ranks(revisit: str) -> None:
    num_ffn, num_attn, stages = 4, 2, 2
    cfg = DeltaDispatchConfig(mode="delta_int4", late=1, revisit=revisit)
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
    assert codes == ({H_BF16, D_INT4, D_INT8})


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


def test_x_int4_is_the_plain_quantizer() -> None:
    # No state: every FFN rank gets x = Norm(h') quantized to int4 per row.
    for step in _drive(DeltaDispatchConfig(mode="x_int4"), 4, 2, 1):
        assert set(step["codes"].tolist()) == {D_INT4}
        for (_, t), x in step["inputs"].items():
            want = _offline_quantize(step["norm_h"][t].float().view(1, -1), 4)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
@pytest.mark.parametrize("mode", ["delta_int4", "delta_fp32", "full_bf16", "x_int4"])
def test_triton_kernels_match_reference(mode: str) -> None:
    from afd_plugin import delta_dispatch_triton

    revisit = "int8" if mode == "delta_int4" else "int4"
    cfg = DeltaDispatchConfig(mode=mode, revisit=revisit)
    gen = torch.Generator().manual_seed(2)
    tokens, rows = 64, 40
    flat = torch.randperm(3 * tokens, generator=gen)[:rows].sort().values
    last = torch.randint(-1, 3, (rows,), generator=gen, dtype=torch.int32)
    codes = cfg.row_codes(3, 8, last)
    sizes = torch.tensor([row_bytes(int(c), MODEL_HIDDEN) for c in codes])
    offsets = torch.cumsum(sizes, 0) - sizes
    h = (torch.randn(tokens, MODEL_HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(3 * tokens, MODEL_HIDDEN, generator=gen) * 3
    total = int(sizes.sum())
    ref_state = state.clone()
    ref_out = torch.zeros(total, dtype=torch.uint8)
    ref_meta = torch.zeros(rows, dtype=torch.int32)
    encode_rows_reference(
        cfg,
        h,
        ref_state,
        last.clone(),
        3,
        flat,
        codes,
        offsets,
        tokens,
        1,
        ref_out,
        ref_meta,
    )
    gpu_state = state.cuda()
    out = torch.zeros(total, dtype=torch.uint8, device="cuda")
    meta = torch.zeros(rows, dtype=torch.int32, device="cuda")
    delta_dispatch_triton.encode_rows(
        cfg,
        h.cuda(),
        gpu_state,
        last.cuda(),
        3,
        flat.cuda(),
        codes.cuda(),
        offsets.cuda(),
        tokens,
        1,
        out,
        meta,
    )
    # Same bytes and words as the reference; the states differ from it only
    # by the fused multiply-add of the dequantization.
    assert torch.equal(out.cpu(), ref_out) and torch.equal(meta.cpu(), ref_meta)
    torch.testing.assert_close(gpu_state.cpu(), ref_state, rtol=1e-6, atol=1e-6)
    # The FFN kernel rebuilds the Attention kernel's states bit for bit.
    ffn_state = state.cuda()
    h_out = torch.empty(rows, MODEL_HIDDEN, dtype=torch.bfloat16, device="cuda")
    delta_dispatch_triton.decode_rows(
        cfg, out, meta, offsets.cuda(), flat.cuda(), ffn_state, h_out
    )
    if cfg.stateful:
        rows_idx = flat.cuda()
        assert torch.equal(ffn_state[rows_idx], gpu_state[rows_idx])
        assert torch.equal(h_out, gpu_state[rows_idx].to(torch.bfloat16))


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


def test_combine_config_parsing() -> None:
    assert not CombineConfig.from_extra_config(None).enabled
    cfg = CombineConfig.from_extra_config({"combine": {"mode": "int4"}})
    assert cfg.enabled and cfg.ef and cfg.code == D_INT4
    assert cfg.row_bytes(MODEL_HIDDEN) == 1088
    cfg = CombineConfig.from_mapping({"mode": "int8", "ef": False})
    assert cfg.code == D_INT8 and not cfg.ef
    assert cfg.row_bytes(MODEL_HIDDEN) == 2112
    for bad in ({"mode": "fp8"}, {"mode": "int4", "ef": 1}, {"bits": 4}):
        with pytest.raises(ValueError):
            CombineConfig.from_mapping(bad)


@pytest.mark.parametrize("code,bits", [(D_INT8, 8), (D_INT4, 4)])
def test_combine_rows_match_offline_quantizer(code: int, bits: int) -> None:
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
    assert torch.equal(deq, _offline_quantize(partial.float(), bits))
    back = torch.empty(n, HIDDEN)
    decode_combine_reference(code, data, back)
    assert torch.equal(back, deq)
    err = torch.randn(2 * TOKENS, HIDDEN, generator=gen) * 0.1
    before = err.clone()
    encode_combine_reference(code, partial, rows, err, data, deq)
    v = partial.float() + before[rows]
    assert torch.equal(deq, _offline_quantize(v, bits))
    assert torch.equal(err[rows], v - deq)
    untouched = torch.ones(2 * TOKENS, dtype=torch.bool)
    untouched[rows] = False
    assert torch.equal(err[untouched], before[untouched])


@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_combine_error_feedback_keeps_the_sum(mode: str) -> None:
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
    for _ in MOE_LAYERS:
        ffn._rows[0] = rows
        partial = (torch.randn(TOKENS, HIDDEN, generator=gen)).to(torch.bfloat16)
        ffn._state[0] = torch.zeros(TOKENS, HIDDEN)
        ffn.after_experts(0, partial)
        back = torch.empty(TOKENS, HIDDEN)
        decode_combine_reference(combine.code, ffn.combine_bytes(0, TOKENS), back)
        exact += partial.float()
        received += back
    torch.testing.assert_close(received, exact - ffn._err[0], rtol=0, atol=1e-5)
    # The residual stays at one quantization step, not the accumulated error.
    assert float(ffn._err[0].abs().max()) < 0.2


@pytest.mark.parametrize("delta_mode", ["delta_int4", "delta_fp32", "x_int4"])
@pytest.mark.parametrize("mode,ef", [("int4", True), ("int8", True), ("int4", False)])
def test_combine_keeps_both_states_identical(
    delta_mode: str, mode: str, ef: bool
) -> None:
    # Four FFN ranks, two Attention ranks, two stages, two forwards: both sides
    # add what the Attention side received, so the states stay identical; the
    # received partial sums are the offline quantizer's of partial (+ residual).
    revisit = "int8" if delta_mode == "delta_int4" else "int4"
    cfg = DeltaDispatchConfig(mode=delta_mode, revisit=revisit)
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
        assert 0 < float(err) < 0.5
    if not ef:
        bits = 8 if mode == "int8" else 4
        for step in steps:
            assert torch.equal(
                step["received"],
                _offline_quantize(step["exact"], bits).to(torch.bfloat16).float(),
            )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
@pytest.mark.parametrize("code", [D_INT8, D_INT4])
@pytest.mark.parametrize("ef", [False, True])
def test_triton_combine_kernels_match_reference(code: int, ef: bool) -> None:
    from afd_plugin import delta_dispatch_triton

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
    data = torch.zeros(n * size, dtype=torch.uint8, device="cuda")
    deq = torch.empty(n, MODEL_HIDDEN, device="cuda")
    gpu_err = err.cuda() if ef else None
    delta_dispatch_triton.encode_combine(
        code, partial.cuda(), rows.cuda(), gpu_err, data, deq
    )
    # Same bytes; deq and the residual differ from the reference only by the
    # fused multiply-add of the dequantization.
    assert torch.equal(data.cpu(), ref_data)
    torch.testing.assert_close(deq.cpu(), ref_deq, rtol=1e-6, atol=1e-6)
    if ef:
        torch.testing.assert_close(gpu_err.cpu(), ref_err, rtol=1e-6, atol=1e-6)
    # The Attention kernel rebuilds exactly what the FFN kernel added.
    back = torch.empty(n, MODEL_HIDDEN, device="cuda")
    delta_dispatch_triton.decode_combine(code, data, back)
    assert torch.equal(back, deq)
