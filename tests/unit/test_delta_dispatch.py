# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch arithmetic (afd_plugin/delta_dispatch.py), without vLLM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from afd_plugin.delta_dispatch import (
    QUANT_GROUP_SIZE,
    AttentionDeltaState,
    DeltaDispatchConfig,
    FFNDeltaState,
    decode_delta_reference,
    decode_plain,
    encode_delta_reference,
    encode_plain,
    row_bytes,
)

HIDDEN = 256
TOKENS = 5
MOE_LAYERS = (1, 2, 3, 4, 5, 6)
ROUTED_SCALE = 1.5
EPS = 1e-6
MODEL_HIDDEN = 2048
INT4_ROW_BYTES = 1088
INT8_ROW_BYTES = 2112


def _norm_into(out: torch.Tensor, h: torch.Tensor) -> None:
    """Stand-in for post_attention_layernorm (input and output in bf16)."""
    x = h.float()
    out.copy_((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(out.dtype))


def _ffn_output(x: torch.Tensor) -> torch.Tensor:
    """Stand-in for the experts' partial sum (bf16, depends on the input)."""
    return torch.tanh(x.float() * 0.7).to(torch.bfloat16)


def _run(
    cfg: DeltaDispatchConfig, stages: int = 1, out_rows: int = TOKENS
) -> list[dict]:
    """Drive both sides through the MoE layers like the model does; per layer
    and stage return the FFN input, Norm(h'), the two states and the row size."""
    attn = AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE)
    ffn = FFNDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE, HIDDEN)
    gen = torch.Generator().manual_seed(0)
    h = [
        torch.randn(TOKENS, HIDDEN, generator=gen).to(torch.bfloat16)
        for _ in range(stages)
    ]
    seen = []
    for layer in MOE_LAYERS:
        for stage in range(stages):
            a = torch.randn(TOKENS, HIDDEN, generator=gen) * 0.3
            h_prime = (h[stage].float() + a).to(torch.bfloat16)
            rows = attn.encode_dispatch(layer, stage, h_prime)
            assert rows.dtype == torch.uint8
            assert rows.shape[1] == ffn.expected_row_bytes(layer)
            x = ffn.decode_dispatch(
                layer, stage, rows, _norm_into, out_rows, torch.bfloat16
            )
            assert x.shape == (out_rows, HIDDEN)
            expected = torch.empty_like(h_prime)
            _norm_into(expected, h_prime)
            seen.append(
                {
                    "x": x[:TOKENS].clone(),
                    "norm_h": expected,
                    "attn_state": attn._state.get(stage, torch.empty(0)).clone(),
                    "ffn_state": ffn._state.get(stage, torch.empty(0)).clone(),
                    "row_bytes": rows.shape[1],
                }
            )
            out = _ffn_output(x[:TOKENS])
            filler = torch.full((out_rows - TOKENS, HIDDEN), 7.0, dtype=out.dtype)
            ffn.after_experts(layer, stage, torch.cat([out, filler]))
            attn.after_combine(layer, stage, out)
            shared = (torch.randn(TOKENS, HIDDEN, generator=gen) * 0.2).to(
                torch.bfloat16
            )
            h[stage] = (
                h_prime.float() + ROUTED_SCALE * out.float() + shared.float()
            ).to(torch.bfloat16)
    return seen


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


def test_config_parsing() -> None:
    assert not DeltaDispatchConfig.from_extra_config(None).enabled
    assert not DeltaDispatchConfig.from_extra_config({}).enabled
    cfg = DeltaDispatchConfig.from_extra_config(
        {"delta_dispatch": {"mode": "delta_int4", "early": 4, "late": 1}}
    )
    assert (cfg.mode, cfg.early, cfg.late) == ("delta_int4", 4, 1)
    with pytest.raises(ValueError):
        DeltaDispatchConfig.from_mapping({"mode": "int3"})
    with pytest.raises(ValueError):
        DeltaDispatchConfig.from_mapping({"mode": "off", "extra": 1})
    with pytest.raises(ValueError):
        DeltaDispatchConfig.from_mapping({"mode": "delta_fp32", "early": 2})
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


def test_row_formats() -> None:
    assert row_bytes("h_bf16", MODEL_HIDDEN) == 2 * MODEL_HIDDEN
    assert row_bytes("d_fp32", MODEL_HIDDEN) == 4 * MODEL_HIDDEN
    assert row_bytes("d_int4", MODEL_HIDDEN) == INT4_ROW_BYTES
    assert row_bytes("d_int8", MODEL_HIDDEN) == INT8_ROW_BYTES
    assert DeltaDispatchConfig(mode="delta_int4").max_row_bytes(MODEL_HIDDEN) == (
        2 * MODEL_HIDDEN
    )
    cfg = DeltaDispatchConfig(mode="delta_int4", early=3, late=1)
    formats = [cfg.row_format(p, 6) for p in range(6)]
    assert formats == ["h_bf16", "d_int8", "d_int8", "d_int4", "d_int4", "d_int8"]
    v = torch.randn(3, HIDDEN)
    assert torch.equal(decode_plain("d_fp32", encode_plain("d_fp32", v), HIDDEN), v)
    with pytest.raises(ValueError):
        decode_plain("d_fp32", encode_plain("h_bf16", v), HIDDEN)


@pytest.mark.parametrize("fmt,bits", [("d_int4", 4), ("d_int8", 8)])
def test_quantization_matches_offline_experiment(fmt: str, bits: int) -> None:
    gen = torch.Generator().manual_seed(1)
    h = (torch.randn(TOKENS, HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(TOKENS, HIDDEN, generator=gen) * 3
    before = state.clone()
    rows = encode_delta_reference(fmt, h, state)
    assert rows.shape == (TOKENS, row_bytes(fmt, HIDDEN))
    # The Attention side's state moved by exactly the offline dequantized delta.
    assert torch.equal(state, before + _offline_quantize(h.float() - before, bits))
    # The FFN side rebuilds the same state from the rows.
    ffn_state = before.clone()
    decode_delta_reference(fmt, rows, ffn_state)
    assert torch.equal(ffn_state, state)


def test_constant_group_has_no_nan() -> None:
    h = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
    state = torch.zeros(TOKENS, HIDDEN)
    rows = encode_delta_reference("d_int4", h, state)
    decoded = torch.zeros(TOKENS, HIDDEN)
    decode_delta_reference("d_int4", rows, decoded)
    assert torch.isfinite(state).all() and torch.equal(decoded, state)


@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
@pytest.mark.parametrize("stages", [1, 2])
def test_ffn_input_matches_norm_of_h_prime(mode: str, stages: int) -> None:
    # Without compression the FFN side hands the experts exactly Norm(h').
    for step in _run(DeltaDispatchConfig(mode=mode), stages):
        assert torch.equal(step["x"], step["norm_h"])


@pytest.mark.parametrize("stages", [1, 2])
def test_int4_states_stay_identical(stages: int) -> None:
    # Closed loop: both sides hold the same state at every layer, and the
    # first MoE layer (h' in full) and the int8 layers are where they belong.
    cfg = DeltaDispatchConfig(mode="delta_int4", early=2, late=1)
    steps = _run(cfg, stages)
    for step in steps:
        assert torch.equal(step["attn_state"], step["ffn_state"])
    sizes = [s["row_bytes"] for s in steps[::stages]]
    assert sizes == [
        row_bytes(f, HIDDEN)
        for f in ("h_bf16", "d_int8", "d_int4", "d_int4", "d_int4", "d_int8")
    ]
    # Errors stay bounded (no accumulation): last layer like the second.
    errs = [
        (s["x"].float() - s["norm_h"].float()).norm() / s["norm_h"].float().norm()
        for s in steps
    ]
    assert max(errs) < 0.1


def test_fixed_size_launch_rows() -> None:
    # More launch rows than received rows: the real rows still match.
    for step in _run(DeltaDispatchConfig(mode="delta_fp32"), out_rows=TOKENS + 3):
        assert torch.equal(step["x"], step["norm_h"])


def test_missing_state_is_an_error() -> None:
    cfg = DeltaDispatchConfig(mode="delta_fp32")
    attn = AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE)
    h = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        attn.encode_dispatch(MOE_LAYERS[1], 0, h)
    with pytest.raises(RuntimeError):
        attn.encode_dispatch(99, 0, h)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
@pytest.mark.parametrize("fmt", ["d_int4", "d_int8"])
def test_triton_kernels_match_reference(fmt: str) -> None:
    from afd_plugin import delta_dispatch_triton

    gen = torch.Generator().manual_seed(2)
    h = (torch.randn(64, MODEL_HIDDEN, generator=gen) * 3).to(torch.bfloat16)
    state = torch.randn(64, MODEL_HIDDEN, generator=gen) * 3
    ref_state = state.clone()
    ref_rows = encode_delta_reference(fmt, h, ref_state)
    gpu_state = state.cuda()
    rows = delta_dispatch_triton.encode_delta(fmt, h.cuda(), gpu_state)
    # Same quantized values and scales as the reference.
    assert torch.equal(rows.cpu(), ref_rows)
    # The state differs from the reference only by the fused multiply-add of
    # the dequantization (one rounding instead of two).
    torch.testing.assert_close(gpu_state.cpu(), ref_state, rtol=1e-6, atol=1e-6)
    # The FFN kernel rebuilds the Attention kernel's state bit for bit.
    ffn_state = state.cuda()
    h_out = torch.empty(64, MODEL_HIDDEN, dtype=torch.bfloat16, device="cuda")
    delta_dispatch_triton.decode_delta(fmt, rows, ffn_state, h_out)
    assert torch.equal(ffn_state, gpu_state)
    assert torch.equal(h_out, gpu_state.to(torch.bfloat16))
