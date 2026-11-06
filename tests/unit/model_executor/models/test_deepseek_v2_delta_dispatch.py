# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Delta dispatch through the DeepSeek wrapper's glue code.

The Attention side's ``ResidualRecordingRMSNorm``, ``RoutedExpertsProxy`` and
``routed_moe_forward``, the model's ``_init_delta_dispatch`` and the FFN
side's ``compute_routed_experts_output`` run on CPU as in the server, with one
or four FFN models; the wire is a loopback doing the connector's steps on the
real dispatch plan (``P2pAll2AllEngine``), and the experts are a stand-in.
The CPU norm rounds the residual sum to the model dtype before normalizing,
as vLLM's CUDA kernels do (the torch reference normalizes the fp32 sum, which
the FFN rank cannot rebuild); that the FFN-side call reproduces the CUDA
kernel bit for bit is checked on CUDA.
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
from vllm.distributed.device_communicators.p2p_nccl_all2all import (  # noqa: E402
    P2pAll2AllEngine,
)
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.delta_dispatch import (  # noqa: E402
    D_INT4,
    D_INT8,
    H_BF16,
    QUANT_GROUP_SIZE,
    row_bytes,
)
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from afd_plugin.model_executor.models import routed_moe  # noqa: E402

HIDDEN = 256
TOKENS = 5
LAYERS = 7  # layer 0 dense, layers 1-6 MoE
MOE_LAYERS = tuple(range(1, LAYERS))
TOP_K = 2
NUM_EXPERTS = 8
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
        scheduler_config=SimpleNamespace(max_num_batched_tokens=TOKENS),
        additional_config={"afd": {"connector_extra_config": extra}},
    )


def _afd_config(role: str, num_ffn: int, connector: str) -> AFDConfig:
    return AFDConfig(
        role=role,
        connector=connector,
        compute_gate_on_attention=True,
        num_attention_ranks=1,
        num_ffn_ranks=num_ffn,
    )


def _routed_shell(cls: type, layer_idx: int) -> nn.Module:
    """A routed MoE module without its native construction (the delta wiring
    only looks at its type and, on Attention, its proxy)."""
    mlp = cls.__new__(cls)
    nn.Module.__init__(mlp)
    if cls is adapter.AFDDeepseekV2RoutedAttentionMoE:
        mlp.experts = routed_moe.RoutedExpertsProxy(layer_idx=layer_idx)
    return mlp


def _expert(x: torch.Tensor, expert: int) -> torch.Tensor:
    return torch.tanh(x.float() * (0.5 + 0.1 * expert))


class _FFNLayer(nn.Module):
    """FFN rank ``ffn`` of ``num_ffn``: owns experts ``ffn * E/F`` onwards
    (the plan's contiguous placement); records its experts' inputs."""

    def __init__(self, layer_idx: int, ffn: int, num_ffn: int) -> None:
        super().__init__()
        if layer_idx in MOE_LAYERS:
            self.mlp = _routed_shell(adapter.AFDDeepseekV2RoutedFFNMoE, layer_idx)
        else:
            self.mlp = nn.Identity()
        self.post_attention_layernorm = RMSNorm(HIDDEN, eps=EPS)
        self.ffn, self.local = ffn, NUM_EXPERTS // num_ffn
        self.inputs: list[torch.Tensor] = []

    def compute_routed_experts_output(
        self, x, topk_weights, topk_ids, a1q_scale=None
    ) -> torch.Tensor:
        assert x.shape[0] == topk_ids.shape[0] and a1q_scale is None
        real = int((topk_ids >= 0).any(-1).sum())
        self.inputs.append(x[:real].clone())
        out = torch.zeros(x.shape[0], HIDDEN)
        for r in range(real):
            for k in range(TOP_K):
                e = int(topk_ids[r, k])
                if e // self.local == self.ffn:
                    out[r] += topk_weights[r, k] * _expert(x[r], e)
        return out.to(x.dtype)


class _AttnLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        if layer_idx in MOE_LAYERS:
            self.mlp = _routed_shell(adapter.AFDDeepseekV2RoutedAttentionMoE, layer_idx)
        else:
            self.mlp = nn.Identity()
        self.post_attention_layernorm = RMSNorm(HIDDEN, eps=EPS)


def _model(
    role: str,
    delta: dict | None,
    num_ffn: int = 1,
    ffn: int = 0,
    connector: str = "NixlRoutedAFDConnector",
):
    """The parts of AFDDeepseekV2Model the delta wiring and the FFN compute use."""
    vllm_config = _vllm_config(delta)
    layers = [
        _AttnLayer(idx) if role == "attention" else _FFNLayer(idx, ffn, num_ffn)
        for idx in range(LAYERS)
    ]
    model = SimpleNamespace(
        start_layer=0,
        end_layer=LAYERS,
        layers=layers,
        config=SimpleNamespace(
            rms_norm_eps=EPS, routed_scaling_factor=SCALE, hidden_size=HIDDEN
        ),
        hidden_size=HIDDEN,
        ffn_delta=None,
        vllm_config=vllm_config,
    )
    cfg = adapter.DeltaDispatchConfig.from_vllm_config(vllm_config)
    if cfg.enabled:
        adapter.AFDDeepseekV2Model._init_delta_dispatch(
            model, cfg, vllm_config, _afd_config(role, num_ffn, connector)
        )
    return model, cfg


class _Loopback:
    """The NIXL connector's steps between the ranks (``_send_attn_delta``,
    ``_complete_prefetch``, ``recv_attn_output``, ``recv_ffn_output``) on the
    real dispatch plan, handing each FFN model its rows."""

    def __init__(self, cfg, ffns, ctx) -> None:
        self.delta_dispatch, self.ffns, self.ctx = cfg, ffns, ctx
        num_ffn = len(ffns)
        self.engine = P2pAll2AllEngine(
            None, num_ffn, num_ffn + 1, NUM_EXPERTS // num_ffn, TOKENS, "cpu"
        )
        self.sent: list[dict] = []

    def quantize_for_dispatch(self, hidden_states):
        return hidden_states, None

    def send_attn_output(
        self,
        hidden_states,
        context,
        *,
        topk_weights,
        topk_ids,
        a1q_scale,
        delta=None,
        h_prime=None,
        layer_idx=None,
    ):
        cfg = self.delta_dispatch
        stage = context.metadata.stage_idx
        assert self.ctx.afd.stage_idx == stage
        pending = self.engine.begin_plan(topk_ids, slot=stage)
        packed = None
        if cfg.enabled:
            packed = delta.prepare(layer_idx, stage, pending.order, pending.num_tokens)
        plan = self.engine.finish_plan_local(pending)
        n = plan.num_send
        if cfg.enabled:
            totals = packed.totals.tolist()
            data = torch.zeros(sum(totals), dtype=torch.uint8)
            words = torch.zeros(n, dtype=torch.int32)
            src = hidden_states if cfg.sends_x else h_prime
            delta.encode(packed, src, n, data, words, 0)
            self.sent.append(
                {
                    "layer": layer_idx,
                    "stage": stage,
                    "codes": packed.codes[:n].tolist(),
                    "bytes": totals,
                    "x": hidden_states[plan.perm].clone(),
                }
            )
        ids, weights = topk_ids[plan.perm], topk_weights[plan.perm]
        rows = hidden_states[plan.perm]
        partials, row, byte = [], 0, 0
        for f, model in enumerate(self.ffns):
            count = plan.send_counts[f]
            launch_ids = torch.cat(
                [ids[row : row + count], torch.full((PAD_ROWS, TOP_K), -1)]
            ).to(topk_ids.dtype)
            launch_weights = torch.cat(
                [weights[row : row + count], torch.zeros(PAD_ROWS, TOP_K)]
            )
            if cfg.enabled:
                received = data[byte : byte + totals[f]]
                model.ffn_delta.receive(stage, words[row : row + count])
                byte += totals[f]
            else:
                received = torch.cat(
                    [rows[row : row + count], torch.zeros(PAD_ROWS, HIDDEN)]
                ).to(rows.dtype)
            out = adapter.AFDDeepseekV2Model.compute_routed_experts_output(
                model, received, context.metadata.layer_idx, launch_weights, launch_ids
            )
            partials.append(out[:count])
            row += count
        self._pending = (plan, torch.cat(partials), packed, delta)

    def recv_ffn_output(self, ref_tensor, ubatch_idx):
        plan, back, packed, delta = self._pending
        if self.delta_dispatch.enabled:
            delta.after_combine(packed, plan.num_send, back)
        out = torch.empty_like(ref_tensor)
        self.engine.combine_accumulate(plan, back, out)
        return out


class _Router:
    def __init__(self, gen) -> None:
        self.gen = gen

    def select_experts(self, *, hidden_states, router_logits, topk_indices_dtype):
        n = hidden_states.shape[0]
        scores = torch.rand(n, NUM_EXPERTS, generator=self.gen)
        ids = scores.topk(TOP_K, dim=-1).indices  # distinct experts per token
        weights = torch.rand(n, TOP_K, generator=self.gen)
        return weights, ids.to(topk_indices_dtype)


def _run(delta: dict | None, ctx, stages: int, num_ffn: int) -> dict:
    """One forward over the MoE layers, ``stages`` micro-batches interleaved
    per layer like the pipeline; returns what every FFN rank's experts saw,
    the final hidden states, the dispatched rows and the models."""
    attn, cfg = _model("attention", delta, num_ffn)
    ffns = [_model("ffn", delta, num_ffn, f)[0] for f in range(num_ffn)]
    gen = torch.Generator().manual_seed(0)
    # Same loaded norm weights on all ranks (the norm is a both-roles weight).
    for idx in range(LAYERS):
        weight = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
        with torch.no_grad():
            for model in (attn, *ffns):
                model.layers[idx].post_attention_layernorm.weight.copy_(weight)
    wire = _Loopback(cfg, ffns, ctx)
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
            ctx.afd.stage_idx = stage
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
        "inputs": [
            x for f in ffns for layer in f.layers for x in getattr(layer, "inputs", [])
        ],
        "hidden": hidden,
        "sent": wire.sent,
        "attn": attn,
        "ffns": ffns,
    }


def _offline_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    sh = x.shape
    g = x.reshape(*sh[:-1], sh[-1] // QUANT_GROUP_SIZE, QUANT_GROUP_SIZE)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    n = 2**bits - 1
    s = ((hi - lo).clamp_min(1e-12) / n).half().float()
    lo = lo.half().float()
    return (torch.round((g - lo) / s).clamp(0, n) * s + lo).reshape(sh)


@pytest.mark.parametrize("num_ffn", [1, 4])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("mode", ["full_bf16", "delta_fp32"])
def test_uncompressed_modes_reproduce_off(cpu_config, context, mode, stages, num_ffn):
    off = _run(None, context, stages, num_ffn)
    on = _run({"mode": mode}, context, stages, num_ffn)
    assert len(on["inputs"]) == len(off["inputs"]) >= len(MOE_LAYERS) * stages
    # Every FFN rank's experts get exactly the off path's x at every layer and
    # stage, so the whole forward is the same.
    for x_on, x_off in zip(on["inputs"], off["inputs"], strict=True):
        assert torch.equal(x_on, x_off)
    for h_on, h_off in zip(on["hidden"], off["hidden"], strict=True):
        assert torch.equal(h_on, h_off)


def test_off_mode_keeps_native_modules(cpu_config, context):
    run = _run(None, context, 1, 4)
    for layer in run["attn"].layers:
        assert type(layer.post_attention_layernorm) is RMSNorm
        if layer.mlp.__class__ is adapter.AFDDeepseekV2RoutedAttentionMoE:
            assert layer.mlp.experts.delta is None
    assert all(f.ffn_delta is None for f in run["ffns"])
    assert run["sent"] == []


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("num_ffn,revisit", [(1, "int4"), (4, "int4"), (4, "int8")])
def test_int4_rows_keep_both_states_identical(
    cpu_config, context, stages, num_ffn, revisit
):
    delta = {"mode": "delta_int4", "early": 2, "late": 1, "revisit": revisit}
    on = _run(delta, context, stages, num_ffn)
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
    # Closed loop: FFN rank f holds the Attention side's copy of its states.
    attn_state = proxies[0].delta._state
    for f, model in enumerate(on["ffns"]):
        for stage in range(stages):
            assert torch.equal(
                model.ffn_delta._state[stage][:TOKENS],
                attn_state[stage][f * TOKENS : (f + 1) * TOKENS],
            )
    # Formats: first MoE layer h' bf16; early (2 first) and late (last) MoE
    # layers d_int8; with several FFN ranks also first visits and revisits.
    for sent in on["sent"]:
        pos = MOE_LAYERS.index(sent["layer"])
        codes = set(sent["codes"])
        if pos == 0:
            assert codes == {H_BF16}
        elif pos < 2 or pos == len(MOE_LAYERS) - 1:
            assert codes <= {H_BF16, D_INT8}
        elif num_ffn == 1:
            assert codes == {D_INT4}
        sizes = [row_bytes(c, HIDDEN) for c in sent["codes"]]
        assert sum(sent["bytes"]) == sum(sizes)
    all_codes = {c for sent in on["sent"] for c in sent["codes"]}
    if num_ffn == 4:
        assert H_BF16 in all_codes and D_INT4 in all_codes


def test_x_int4_sends_the_plainly_quantized_x(cpu_config, context):
    on = _run({"mode": "x_int4"}, context, 1, 4)
    sent = iter(on["sent"])
    inputs = iter(on["inputs"])
    # The FFN models' inputs, per layer FFN rank by FFN rank, are the plan's x
    # rows quantized to int4 (no state, no FFN-side norm).
    by_layer: dict[int, list[torch.Tensor]] = {}
    for model in on["ffns"]:
        for idx in MOE_LAYERS:
            by_layer.setdefault(idx, []).extend(model.layers[idx].inputs)
    for item in sent:
        assert set(item["codes"]) == {D_INT4}
        want = _offline_quantize(item["x"].float(), 4).to(torch.bfloat16)
        got = torch.cat(by_layer[item["layer"]])
        assert torch.equal(got, want)
    assert next(inputs, None) is not None


def test_delta_dispatch_needs_the_routed_connector(cpu_config, context):
    with pytest.raises(ValueError, match="routed connector"):
        _model("ffn", {"mode": "delta_fp32"}, connector="P2pNcclAFDConnector")


def test_unwired_model_is_an_error(cpu_config, context):
    attn, _ = _model("attention", None)
    ffn, _ = _model("ffn", {"mode": "delta_fp32"})
    context.afd.connector = _Loopback(
        adapter.DeltaDispatchConfig(mode="delta_fp32"), [ffn], context
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
        {"delta_dispatch": {"mode": "delta_int4", "early": 4, "revisit": "int8"}}
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


def test_nixl_spans_land_where_the_ffn_reads_them():
    pytest.importorskip("nixl")
    from afd_plugin.connectors.gpu import nixl_routed

    max_tokens, attn_size, slots, max_row = 8, 2, 2, 4096
    # FFN receive buffers: per (stage, Attention rank) a region of max_tokens
    # rows of the largest row size, and one word per row.
    region = torch.zeros(slots * attn_size * max_tokens, max_row, dtype=torch.uint8)
    words = torch.zeros(slots * attn_size * max_tokens, 1, dtype=torch.int32)
    gen = torch.Generator().manual_seed(5)
    sent = {}
    for stage in range(slots):
        for attn in range(attn_size):
            count = 3 + attn + stage
            sizes = [(4096, 2112, 1088)[(i + attn) % 3] for i in range(count)]
            conn = _host_connector(nixl_routed, role_rank=attn, max_tokens=max_tokens)
            conn._peer_agents[0] = "ffn-0"
            conn._peer_regions[0] = [
                (region.data_ptr(), max_row),
                (words.data_ptr(), 4),
            ]
            # Attention: rows of mixed sizes packed back to back, and words.
            data = torch.randint(
                0, 256, (sum(sizes),), generator=gen, dtype=torch.uint8
            )
            meta = torch.randint(
                0, 1 << 20, (count, 1), generator=gen, dtype=torch.int32
            )
            sent[stage, attn] = (data, meta)
            conn._write_spans(
                stage,
                0,
                [(data.data_ptr(), data.numel()), (meta.data_ptr(), count * 4)],
                conn._region_row(stage, attn_size),
                f"d{stage}:{count}:{data.numel()}".encode(),
            )
    # FFN: each Attention rank's bytes at the head of its region (the slicing
    # of NixlRoutedAFDConnector._complete_prefetch).
    for (stage, attn), (data, meta) in sent.items():
        first = (stage * attn_size + attn) * max_tokens
        got = region[first:][:max_tokens].view(-1)[: data.numel()]
        assert torch.equal(got, data)
        assert torch.equal(words[first:][: meta.shape[0]], meta)


def test_nixl_dispatch_notification_carries_the_byte_count():
    pytest.importorskip("nixl")
    from afd_plugin.connectors.gpu import nixl_routed

    conn = _host_connector(nixl_routed, role_rank=0, max_tokens=8)
    conn._peer_agents = {1: "attn-0", 2: "attn-1"}
    # Delta dispatch: row count and byte count of the stage's rows.
    conn.agent.inbox = {"attn-0": [b"d7:3:5248"], "attn-1": [b"c7:1", b"d7:0:0"]}
    assert conn._wait_counts(b"d7:", [1, 2]) == {1: (3, 5248), 2: (0, 0)}
    conn.agent.inbox = {"attn-0": [b"d8:5"], "attn-1": [b"d8:2"]}
    assert conn._wait_counts(b"d8:", [1, 2]) == {1: (5, None), 2: (2, None)}
