# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch arithmetic (afd_plugin/delta_dispatch.py), without vLLM."""

from __future__ import annotations

import pytest
import torch

from afd_plugin.delta_dispatch import (
    AttentionDeltaState,
    DeltaDispatchConfig,
    FFNDeltaState,
    decode_rows,
    encode_rows,
    row_bytes,
)

HIDDEN = 64
TOKENS = 5
MOE_LAYERS = (1, 2, 3, 4, 5)
ROUTED_SCALE = 1.5
EPS = 1e-6


def _norm(h: torch.Tensor) -> torch.Tensor:
    """Stand-in for post_attention_layernorm (input and output in bf16)."""
    x = h.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(h.dtype)


def _ffn_output(x: torch.Tensor) -> torch.Tensor:
    """Stand-in for the experts' partial sum (bf16, depends on the input)."""
    return torch.tanh(x.float() * 0.7).to(torch.bfloat16)


def _run(
    mode: str, stages: int = 1, out_rows: int = TOKENS
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Drive both sides through the MoE layers like the model does and return,
    per layer and stage, (x the FFN side handed to the experts, Norm(h'))."""
    cfg = DeltaDispatchConfig(mode=mode)
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
            h_prime = (h[stage].float() + a).to(
                torch.bfloat16
            )  # residual after attention
            rows = attn.encode_dispatch(layer, stage, h_prime)
            assert rows.dtype == torch.uint8
            assert rows.shape[1] == ffn.expected_row_bytes(layer)
            x = ffn.decode_dispatch(layer, stage, rows, _norm, out_rows, torch.bfloat16)
            assert x.shape == (out_rows, HIDDEN)
            seen.append((x[:TOKENS].clone(), _norm(h_prime)))
            out = _ffn_output(x[:TOKENS])
            out_launch = torch.cat(
                [out, torch.full((out_rows - TOKENS, HIDDEN), 7.0, dtype=out.dtype)]
            )
            ffn.after_experts(layer, stage, out_launch)
            attn.after_combine(layer, stage, out)
            shared = (torch.randn(TOKENS, HIDDEN, generator=gen) * 0.2).to(
                torch.bfloat16
            )
            h[stage] = (
                h_prime.float() + ROUTED_SCALE * out.float() + shared.float()
            ).to(torch.bfloat16)
    return seen


def test_config_parsing() -> None:
    assert not DeltaDispatchConfig.from_extra_config(None).enabled
    assert not DeltaDispatchConfig.from_extra_config({}).enabled
    assert (
        DeltaDispatchConfig.from_extra_config(
            {"delta_dispatch": {"mode": "delta_fp32"}}
        ).mode
        == "delta_fp32"
    )
    with pytest.raises(ValueError):
        DeltaDispatchConfig.from_mapping({"mode": "int3"})
    with pytest.raises(ValueError):
        DeltaDispatchConfig.from_mapping({"mode": "off", "extra": 1})


def test_row_formats() -> None:
    assert row_bytes("h_bf16", HIDDEN) == 2 * HIDDEN
    assert row_bytes("d_fp32", HIDDEN) == 4 * HIDDEN
    assert DeltaDispatchConfig(mode="full_bf16").max_row_bytes(HIDDEN) == 2 * HIDDEN
    assert DeltaDispatchConfig(mode="delta_fp32").max_row_bytes(HIDDEN) == 4 * HIDDEN
    cfg = DeltaDispatchConfig(mode="delta_fp32")
    assert [cfg.row_format(p) for p in range(3)] == ["h_bf16", "d_fp32", "d_fp32"]
    v = torch.randn(3, HIDDEN)
    assert torch.equal(decode_rows("d_fp32", encode_rows("d_fp32", v), HIDDEN), v)
    with pytest.raises(ValueError):
        decode_rows("d_fp32", encode_rows("h_bf16", v), HIDDEN)


@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
@pytest.mark.parametrize("stages", [1, 2])
def test_ffn_input_matches_norm_of_h_prime(mode: str, stages: int) -> None:
    # The FFN side must hand the experts exactly Norm(h') at every layer.
    for x, expected in _run(mode, stages):
        assert torch.equal(x, expected)


def test_fixed_size_launch_rows() -> None:
    # More launch rows than received rows: the real rows still match.
    for x, expected in _run("delta_fp32", out_rows=TOKENS + 3):
        assert torch.equal(x, expected)


def test_missing_state_is_an_error() -> None:
    cfg = DeltaDispatchConfig(mode="delta_fp32")
    attn = AttentionDeltaState(cfg, MOE_LAYERS, ROUTED_SCALE)
    with pytest.raises(RuntimeError):
        attn.encode_dispatch(
            MOE_LAYERS[1], 0, torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
        )
    with pytest.raises(RuntimeError):
        attn.encode_dispatch(99, 0, torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16))
