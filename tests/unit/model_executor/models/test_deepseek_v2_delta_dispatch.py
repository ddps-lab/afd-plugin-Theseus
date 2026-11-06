# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch through the DeepSeek wrapper's glue code.

The Attention side's ``ResidualRecordingRMSNorm``, ``RoutedExpertsProxy`` and
``routed_moe_forward``, the model's ``_init_delta_dispatch`` and the FFN
side's ``compute_routed_experts_output`` run on CPU as in the server; the wire
is a loopback and the experts are a stand-in. The CPU norm rounds the residual
sum to the model dtype before normalizing, as vLLM's CUDA kernels do (the
torch reference normalizes the fp32 sum, which the FFN rank cannot rebuild);
that the FFN-side call reproduces the CUDA kernel bit for bit is checked on
CUDA.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
nn = torch.nn

from vllm.config import (  # noqa: E402
    CompilationConfig,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode  # noqa: E402
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.delta_dispatch import row_bytes  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from afd_plugin.model_executor.models import routed_moe  # noqa: E402

HIDDEN = 256
TOKENS = 5
LAYERS = 6  # layer 0 dense, layers 1-5 MoE
MOE_LAYERS = tuple(range(1, LAYERS))
TOP_K = 2
NUM_EXPERTS = 4
EPS = 1e-6
SCALE = 2.5
PAD_ROWS = 3  # fixed-size experts launch: rows past the received ones


def _rounded_sum_norm(self, x, residual=None):
    """vLLM's CUDA RMSNorm arithmetic: residual sum rounded to the model
    dtype, then ``x * rsqrt(mean(x^2) + eps) * w`` in fp32, rounded once."""
    if residual is not None:
        residual = (x.float() + residual.float()).to(x.dtype)
        x = residual
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    out = (xf * scale * self.weight.float()).to(x.dtype)
    return out if residual is None else (out, residual)


@pytest.fixture
def cpu_config(monkeypatch):
    monkeypatch.setattr(RMSNorm, "forward_native", _rounded_sum_norm)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


@pytest.fixture
def context(monkeypatch):
    """The forward context the proxy and the FFN model read the stage from."""
    ctx = SimpleNamespace(
        forward=SimpleNamespace(ubatch_idx=0),
        afd=SimpleNamespace(stage_idx=0, connector=None),
    )
    monkeypatch.setattr(routed_moe, "get_forward_context", lambda: ctx.forward)
    for module in (routed_moe, adapter):
        monkeypatch.setattr(
            module, "get_afd_metadata_from_forward_context", lambda: ctx.afd
        )
    monkeypatch.setattr(adapter, "get_tensor_model_parallel_world_size", lambda: 1)
    return ctx


def _vllm_config(delta: dict | None) -> SimpleNamespace:
    extra = {} if delta is None else {"delta_dispatch": delta}
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=True),
        additional_config={"afd": {"connector_extra_config": extra}},
    )


def _afd_config(role: str, num_ffn_ranks: int = 1) -> AFDConfig:
    return AFDConfig(
        role=role,
        connector="NixlRoutedAFDConnector",
        compute_gate_on_attention=True,
        num_ffn_ranks=num_ffn_ranks,
    )


def _routed_shell(cls: type, layer_idx: int) -> nn.Module:
    """A routed MoE module without its native construction (the delta wiring
    only looks at its type and, on Attention, its proxy)."""
    mlp = cls.__new__(cls)
    nn.Module.__init__(mlp)
    if cls is adapter.AFDDeepseekV2RoutedAttentionMoE:
        mlp.experts = routed_moe.RoutedExpertsProxy(layer_idx=layer_idx)
    return mlp


def _experts(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Stand-in for the experts' weighted partial sum (depends on every row)."""
    return (torch.tanh(x.float() * 0.7) * weights.sum(-1, keepdim=True)).to(x.dtype)


class _FFNLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        if layer_idx in MOE_LAYERS:
            self.mlp = _routed_shell(adapter.AFDDeepseekV2RoutedFFNMoE, layer_idx)
        else:
            self.mlp = nn.Identity()
        self.post_attention_layernorm = RMSNorm(HIDDEN, eps=EPS)
        self.inputs: list[torch.Tensor] = []

    def compute_routed_experts_output(
        self, x, topk_weights, topk_ids, a1q_scale=None
    ) -> torch.Tensor:
        assert x.shape[0] == topk_ids.shape[0] and a1q_scale is None
        self.inputs.append(x[:TOKENS].clone())
        return _experts(x, topk_weights)


class _AttnLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        if layer_idx in MOE_LAYERS:
            self.mlp = _routed_shell(adapter.AFDDeepseekV2RoutedAttentionMoE, layer_idx)
        else:
            self.mlp = nn.Identity()
        self.post_attention_layernorm = RMSNorm(HIDDEN, eps=EPS)


def _model(role: str, delta: dict | None, num_ffn_ranks: int = 1):
    """The parts of AFDDeepseekV2Model the delta wiring and the FFN compute use."""
    layer_cls = _AttnLayer if role == "attention" else _FFNLayer
    vllm_config = _vllm_config(delta)
    model = SimpleNamespace(
        start_layer=0,
        end_layer=LAYERS,
        layers=[layer_cls(idx) for idx in range(LAYERS)],
        config=SimpleNamespace(rms_norm_eps=EPS, routed_scaling_factor=SCALE),
        hidden_size=HIDDEN,
        ffn_delta=None,
        vllm_config=vllm_config,
    )
    model.config.hidden_size = HIDDEN
    cfg = adapter.DeltaDispatchConfig.from_vllm_config(vllm_config)
    if cfg.enabled:
        adapter.AFDDeepseekV2Model._init_delta_dispatch(
            model, cfg, vllm_config, _afd_config(role, num_ffn_ranks)
        )
    return model, cfg


class _Loopback:
    """Hands the dispatched rows to the FFN model and returns its partial sums
    (what the NIXL connector does between the two ranks)."""

    def __init__(self, cfg, ffn, ctx) -> None:
        self.delta_dispatch = cfg
        self.ffn, self.ctx = ffn, ctx
        self.sent: list[tuple[int, int, torch.dtype, int]] = []

    def quantize_for_dispatch(self, hidden_states):
        return hidden_states, None

    def send_attn_output(self, rows, context, *, topk_weights, topk_ids, a1q_scale):
        meta = context.metadata
        self.sent.append((meta.layer_idx, meta.stage_idx, rows.dtype, rows.shape[1]))
        self._pending = (rows.clone(), meta.layer_idx, topk_weights, topk_ids)

    def recv_ffn_output(self, ref_tensor, ubatch_idx):
        rows, layer_idx, weights, ids = self._pending
        assert self.ctx.afd.stage_idx == ubatch_idx
        num_recv = rows.shape[0]
        if rows.dtype != torch.uint8:
            # x rows land in the launch buffer; delta rows stay compact and
            # the FFN model rebuilds the launch rows.
            rows = torch.cat([rows, torch.zeros(PAD_ROWS, HIDDEN, dtype=rows.dtype)])
        launch_ids = torch.cat(
            [ids, torch.full((PAD_ROWS, TOP_K), -1, dtype=ids.dtype)]
        )
        launch_weights = torch.cat([weights, torch.zeros(PAD_ROWS, TOP_K)])
        out = adapter.AFDDeepseekV2Model.compute_routed_experts_output(
            self.ffn, rows, layer_idx, launch_weights, launch_ids
        )
        return out[:num_recv].clone()


class _Router:
    def __init__(self, gen) -> None:
        self.gen = gen

    def select_experts(self, *, hidden_states, router_logits, topk_indices_dtype):
        n = hidden_states.shape[0]
        ids = torch.randint(0, NUM_EXPERTS, (n, TOP_K), generator=self.gen)
        weights = torch.rand(n, TOP_K, generator=self.gen)
        return weights, ids.to(topk_indices_dtype)


def _run(delta: dict | None, ctx, stages: int) -> dict:
    """One forward over the MoE layers, ``stages`` micro-batches interleaved
    per layer like the pipeline; returns what the FFN experts saw, the final
    hidden states, the dispatched row sizes and both models."""
    attn, cfg = _model("attention", delta)
    ffn, _ = _model("ffn", delta)
    gen = torch.Generator().manual_seed(0)
    # Same loaded norm weights on both sides (the norm is a both-roles weight).
    for a_layer, f_layer in zip(attn.layers, ffn.layers, strict=True):
        weight = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
        with torch.no_grad():
            a_layer.post_attention_layernorm.weight.copy_(weight)
            f_layer.post_attention_layernorm.weight.copy_(weight)
    wire = _Loopback(cfg, ffn, ctx)
    ctx.afd.connector = wire
    router = _Router(gen)
    residual = [
        torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16() for _ in range(stages)
    ]
    hidden = [
        torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16() for _ in range(stages)
    ]
    for layer_idx in MOE_LAYERS:
        layer = attn.layers[layer_idx]
        for stage in range(stages):
            ctx.forward.ubatch_idx = stage
            attn_out = (hidden[stage].float() * 0.5).bfloat16()
            x, residual[stage] = layer.post_attention_layernorm(
                attn_out, residual[stage]
            )
            hidden[stage] = routed_moe.routed_moe_forward(
                x,
                gate=lambda h: (torch.zeros(h.shape[0], NUM_EXPERTS), None),
                router=router,
                proxy=layer.mlp.experts,
                shared_experts=lambda h: (h.float() * 0.1).bfloat16(),
                routed_scaling_factor=SCALE,
                tp_size=1,
            )
    return {
        "inputs": [x for layer in ffn.layers for x in layer.inputs],
        "hidden": hidden,
        "sent": wire.sent,
        "attn": attn,
        "ffn": ffn,
    }


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
def test_uncompressed_modes_reproduce_off(cpu_config, context, mode, stages):
    off = _run(None, context, stages)
    on = _run({"mode": mode}, context, stages)
    assert len(on["inputs"]) == len(MOE_LAYERS) * stages
    # The FFN experts get exactly the off path's x at every layer and stage,
    # so the whole forward is the same.
    for x_on, x_off in zip(on["inputs"], off["inputs"], strict=True):
        assert torch.equal(x_on, x_off)
    for h_on, h_off in zip(on["hidden"], off["hidden"], strict=True):
        assert torch.equal(h_on, h_off)
    for layer_idx, _, dtype, size in on["sent"]:
        first = layer_idx == MOE_LAYERS[0]
        fmt = "h_bf16" if mode == "full_bf16" or first else "d_fp32"
        assert dtype == torch.uint8 and size == row_bytes(fmt, HIDDEN)


def test_off_mode_keeps_native_modules(cpu_config, context):
    run = _run(None, context, 1)
    for layer in run["attn"].layers:
        assert type(layer.post_attention_layernorm) is RMSNorm
        if layer.mlp.__class__ is adapter.AFDDeepseekV2RoutedAttentionMoE:
            assert layer.mlp.experts.delta is None
    assert run["ffn"].ffn_delta is None
    assert {dtype for _, _, dtype, _ in run["sent"]} == {torch.bfloat16}


@pytest.mark.parametrize("stages", [1, 2])
def test_int4_rows_keep_both_states_identical(cpu_config, context, stages):
    on = _run({"mode": "delta_int4", "early": 2, "late": 1}, context, stages)
    off = _run(None, context, stages)
    attn_layers = on["attn"].layers
    proxies = [attn_layers[idx].mlp.experts for idx in MOE_LAYERS]
    # One state shared by the routed layers; their norms record h'.
    assert all(p.delta is proxies[0].delta for p in proxies)
    for idx in MOE_LAYERS:
        assert isinstance(
            attn_layers[idx].post_attention_layernorm,
            routed_moe.ResidualRecordingRMSNorm,
        )
    assert type(attn_layers[0].post_attention_layernorm) is RMSNorm
    # Closed loop: the FFN rank holds the Attention side's state, per stage.
    attn_state, ffn_state = proxies[0].delta._state, on["ffn"].ffn_delta._state
    assert sorted(attn_state) == sorted(ffn_state) == list(range(stages))
    for stage in range(stages):
        assert torch.equal(attn_state[stage], ffn_state[stage])
    formats = ["h_bf16", "d_int8", "d_int4", "d_int4", "d_int8"]
    sizes = [size for _, stage, _, size in on["sent"] if stage == 0]
    assert sizes == [row_bytes(f, HIDDEN) for f in formats]
    # Quantization error, not a broken path: close to the off inputs.
    for x_on, x_off in zip(on["inputs"], off["inputs"], strict=True):
        err = (x_on.float() - x_off.float()).norm() / x_off.float().norm()
        assert err < 0.1


def test_delta_dispatch_refuses_more_than_one_ffn_rank(cpu_config, context):
    with pytest.raises(ValueError, match="one FFN rank"):
        _model("ffn", {"mode": "delta_fp32"}, num_ffn_ranks=2)


def test_unwired_model_is_an_error(cpu_config, context):
    attn, _ = _model("attention", None)
    ffn, _ = _model("ffn", {"mode": "delta_fp32"})
    context.afd.connector = _Loopback(
        adapter.DeltaDispatchConfig(mode="delta_fp32"), ffn, context
    )
    layer = attn.layers[MOE_LAYERS[0]]
    x = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not wired"):
        layer.mlp.experts(
            x,
            torch.ones(TOKENS, TOP_K),
            torch.zeros(TOKENS, TOP_K, dtype=torch.int32),
        )


def test_nixl_connector_accepts_delta_dispatch_only():
    pytest.importorskip("nixl")
    from afd_plugin.connectors.gpu.nixl_routed import NixlRoutedAFDConnector

    NixlRoutedAFDConnector.parse_extra_config(
        {"delta_dispatch": {"mode": "delta_int4", "early": 4, "late": 1}}
    )
    NixlRoutedAFDConnector.parse_extra_config({})
    with pytest.raises(ValueError):
        NixlRoutedAFDConnector.parse_extra_config({"delta_dispatch": {"mode": "x"}})
    with pytest.raises(ValueError):
        NixlRoutedAFDConnector.parse_extra_config({"other": 1})


def test_norm_from_residual_writes_norm_and_keeps_h(cpu_config):
    norm = RMSNorm(HIDDEN, eps=EPS)
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(HIDDEN, generator=gen))
    h = torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16()
    before = h.clone()
    out = torch.full((TOKENS, HIDDEN), 7.0, dtype=torch.bfloat16)
    adapter.norm_from_residual(norm, out, h)
    assert torch.equal(out, norm(before))
    assert torch.equal(h, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 64, 255, 256, 300])
def test_cuda_norm_from_residual_matches_attention_norm(tokens):
    # The Attention side's fused add + norm and the FFN side's call on the
    # rebuilt h' give the same bits (same kernel, launch and reduction).
    # Eager like the server (--enforce-eager), with the IR op priority the
    # worker applies at start: the CUDA kernels, not the torch reference.
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    priority = config.kernel_config.ir_op_priority
    assert priority.fused_add_rms_norm[0] == "vllm_c"
    with set_current_vllm_config(config), priority.set_priority():
        norm = RMSNorm(2048, eps=EPS).to(device="cuda", dtype=torch.bfloat16)
        gen = torch.Generator(device="cuda").manual_seed(4)
        with torch.no_grad():
            norm.weight.copy_(
                1.0 + 0.1 * torch.randn(2048, generator=gen, device="cuda")
            )
        attn_out = torch.randn(tokens, 2048, generator=gen, device="cuda").bfloat16()
        residual = (
            4 * torch.randn(tokens, 2048, generator=gen, device="cuda")
        ).bfloat16()
        x, h_prime = norm(attn_out, residual)
        x, h_prime = x.clone(), h_prime.clone()
        out = torch.empty_like(x)
        adapter.norm_from_residual(norm, out, h_prime.clone())
        assert torch.equal(out, x)


class _HostAgent:
    """NIXL agent stand-in: a WRITE is a host memcpy, notifications queue up."""

    def __init__(self) -> None:
        self.posted: list[tuple[str, bytes]] = []
        self.inbox: dict[str, list[bytes]] = {}

    def get_xfer_descs(self, descs, mem_type):
        return list(descs)

    def initialize_xfer(self, op, src, dst, peer, notif):
        assert op == "WRITE"
        return src, dst, peer, notif

    def transfer(self, handle):
        src, dst, peer, notif = handle
        for (src_addr, size, _), (dst_addr, dst_size, _) in zip(src, dst, strict=True):
            assert size == dst_size
            ctypes.memmove(dst_addr, src_addr, size)
        self.posted.append((peer, notif))
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        pass

    def get_new_notifs(self):
        inbox, self.inbox = self.inbox, {}
        return inbox


def _host_connector(nixl_routed, *, role_rank: int, max_tokens: int):
    conn = nixl_routed.NixlRoutedAFDConnector.__new__(
        nixl_routed.NixlRoutedAFDConnector
    )
    conn.agent = _HostAgent()
    conn.role_rank, conn.max_tokens = role_rank, max_tokens
    conn._writes, conn._pending_combines, conn._notifs = {}, [], {}
    conn._peer_agents, conn._peer_regions = {}, {}
    return conn


@pytest.mark.parametrize("row_size", [4096, 2112, 1088])
def test_nixl_compact_rows_land_where_the_ffn_reads_them(row_size):
    pytest.importorskip("nixl")
    from afd_plugin.connectors.gpu import nixl_routed

    max_tokens, attn_size, slots, max_row = 8, 2, 2, 4096
    # FFN receive buffer: per (stage, Attention rank) a region of max_tokens
    # rows of the largest row size (registered and exchanged at init).
    region = torch.zeros(slots * attn_size * max_tokens, max_row, dtype=torch.uint8)
    gen = torch.Generator().manual_seed(5)
    sent = {}
    for stage in range(slots):
        for attn in range(attn_size):
            count = 3 + attn + stage
            conn = _host_connector(nixl_routed, role_rank=attn, max_tokens=max_tokens)
            conn._peer_agents[0] = "ffn-0"
            conn._peer_regions[0] = [(region.data_ptr(), max_row)]
            # Attention: the stage's send block, its rows packed at row_size.
            block = torch.zeros(max_tokens, max_row, dtype=torch.uint8)
            rows = nixl_routed._compact_rows(block, row_size)
            rows[:count] = torch.randint(
                0, 256, (count, row_size), generator=gen, dtype=torch.uint8
            )
            sent[stage, attn] = rows[:count].clone()
            conn._write(
                stage,
                0,
                [(rows.data_ptr(), rows.stride(0) * rows.element_size())],
                conn._region_row(stage, attn_size),
                count,
                f"d{stage}:{count}:{row_size}".encode(),
            )
    # FFN: each Attention rank's rows at the head of its region (the slicing
    # of NixlRoutedAFDConnector._complete_prefetch).
    for (stage, attn), rows in sent.items():
        part = nixl_routed._compact_rows(
            region[(stage * attn_size + attn) * max_tokens :][:max_tokens], row_size
        )[: rows.shape[0]]
        assert torch.equal(part, rows)


def test_nixl_dispatch_notification_carries_the_row_size():
    pytest.importorskip("nixl")
    from afd_plugin.connectors.gpu import nixl_routed

    conn = _host_connector(nixl_routed, role_rank=0, max_tokens=8)
    conn._peer_agents = {1: "attn-0", 2: "attn-1"}
    conn.agent.inbox = {"attn-0": [b"d7:3:1088"], "attn-1": [b"c7:1", b"d7:0:1088"]}
    assert conn._wait_counts(b"d7:", [1, 2]) == {1: (3, 1088), 2: (0, 1088)}
    conn.agent.inbox = {"attn-0": [b"d8:5"], "attn-1": [b"d8:2"]}
    assert conn._wait_counts(b"d8:", [1, 2]) == {1: (5, None), 2: (2, None)}
