# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One-hop routed AFD connector for CUDA deployments.

``P2pNcclRoutedAFDConnector`` sends every token of an MoE layer straight from
its Attention rank to the FFN ranks that own one of the token's selected
experts, and receives the per-rank partial sums back. It reuses the selective
all-to-all engine of vLLM's ``p2p_nccl`` all2all backend
(``P2pAll2AllEngine``). There is no fixed Attention-to-FFN mapping and no
FFN-to-FFN expert-parallel exchange: the FFN role runs vLLM's ``prerouted``
all2all backend, which performs no communication at all.

Topology:
    One NCCL world ordered ``[F0, ..., F(F-1), A0, ..., A(A-1)]`` (FFN ranks
    first). With ``E`` routed experts, FFN rank ``r`` owns experts
    ``[r * E / F, (r + 1) * E / F)`` (vLLM's linear expert placement) and
    Attention ranks own no experts. Any ``A`` and ``F`` are allowed.

Data plane (per MoE layer and micro-batch stage):
    1. Attention computes gate + top-k, quantizes the activations when the
       model's quantization allows it (fp8 with block scales), then runs the
       engine's plan (one count exchange, the only host sync) and dispatch on
       the *dispatch* communicator.
    2. FFN receives ``[hidden, topk_ids, topk_weights(, scale)]`` rows, runs
       the experts it owns through the ``prerouted`` backend and returns the
       weighted partial sums on the *combine* communicator.
    3. Attention accumulates the partials (fp32) into the layer output.
    DeepSeek's shared experts run on Attention while the round trip is in
    flight.

Ordering:
    Every rank issues the collectives of one communicator in the same order:
    ``exchange(s), dispatch(s)`` for stages ``s = 0, 1, ...`` on the dispatch
    communicator and ``combine(s)`` on the combine communicator. Attention
    issues all dispatches of a layer before its first combine (dual batch
    overlap), so the FFN side runs every stage on its own CUDA stream and
    chains the stages per communicator with events, which keeps a single
    collective of each communicator in flight.

Control plane:
    Attention role rank 0 sends the per-step control payload (stage ids,
    warmup/profile flags) to every FFN rank over a separate NCCL group,
    like ``P2pNcclAFDConnector``.

Requirements and limitations:
    - Eager mode on both roles: the count exchange is a host sync.
    - FFN role: ``--data-parallel-size F --enable-expert-parallel
      --all2all-backend prerouted``, TP=1, ``E % F == 0``.
    - Attention role: ``compute_gate_on_attention=true``. With TP>1 every
      TP rank dispatches a token slice and the result is all-gathered
      (unverified).
    - Attention-side quantization before dispatch is implemented for fp8
      checkpoints with dynamic block-scaled activations (``Fp8Config`` with
      ``weight_block_size``). Other quantizations send the model dtype and
      quantize on the FFN rank.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from transformers import PretrainedConfig
from vllm.distributed.device_communicators.p2p_nccl_all2all import (
    P2pAll2AllEngine,
    P2pAll2AllPlan,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import (
    dbo_switch_to_compute,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# DEBUG(p2p_nccl): remove after GPU validation. VLLM_P2P_NCCL_DEBUG=1 logs the
# first stages of every step (counts, bytes, host wait) on both roles.
_DEBUG = int(os.environ.get("VLLM_P2P_NCCL_DEBUG", "0") or 0)

_IDS_DTYPE = torch.int32
_WEIGHTS_DTYPE = torch.float32
_SCALE_DTYPE = torch.float32
_WIRE_DTYPES: tuple[torch.dtype, ...] = tuple(
    getattr(torch, name)
    for name in (
        "bfloat16",
        "float16",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if hasattr(torch, name)
)


def _should_log(n: int) -> bool:
    return n < 8 or n % 64 == 0


# ==============================
# Wire format
# ==============================


@dataclass(frozen=True)
class RoutedWireSpec:
    """Static per-deployment description of the rows on the wire.

    Both roles derive it from their own ``VllmConfig``; ``init_afd_connector``
    checks that every rank agrees with Attention role rank 0.
    """

    hidden_size: int
    top_k: int
    num_experts: int
    model_dtype: torch.dtype
    # dtype of the activation rows on the wire (model dtype or fp8).
    act_dtype: torch.dtype
    # Set when Attention quantizes before the dispatch.
    quant_dtype: torch.dtype | None
    block_shape: tuple[int, int] | None
    # Trailing shape of the per-token scale rows, None when no scale travels.
    scale_tail: tuple[int, ...] | None

    @property
    def quantize_on_attention(self) -> bool:
        return self.quant_dtype is not None

    def encode(self) -> list[int]:
        return [
            self.hidden_size,
            self.top_k,
            self.num_experts,
            _WIRE_DTYPES.index(self.act_dtype),
            self.scale_tail[0] if self.scale_tail else -1,
            self.block_shape[1] if self.block_shape else -1,
        ]


def _num_routed_experts(text_config: PretrainedConfig) -> int:
    # HF configs name the routed expert count differently per model family.
    for name in ("n_routed_experts", "num_experts", "num_local_experts"):
        value = getattr(text_config, name, None)
        if value:
            return int(value)
    raise RuntimeError("cannot determine the number of routed experts")


def _attention_quant_plan(
    quant_config: QuantizationConfig | None,
) -> tuple[torch.dtype, tuple[int, int]] | None:
    """Return ``(quant_dtype, block_shape)`` when Attention can quantize.

    Only fp8 with dynamic block-scaled activations is quantized on Attention:
    its per-token-group scales travel with the tokens. Static input scales
    live in the expert weights (not loaded on Attention) and dynamic
    per-tensor scales depend on the whole batch, so those checkpoints send the
    model dtype and the FFN rank quantizes, as vLLM's DeepEP backend does.
    """
    if not isinstance(quant_config, Fp8Config):
        return None
    if quant_config.activation_scheme != "dynamic":
        return None
    block_shape = quant_config.weight_block_size
    if not block_shape:
        return None
    return current_platform.fp8_dtype(), (int(block_shape[0]), int(block_shape[1]))


def resolve_wire_spec(vllm_config: VllmConfig) -> RoutedWireSpec:
    text_config = vllm_config.model_config.hf_text_config
    hidden_size = int(text_config.hidden_size)
    top_k = int(text_config.num_experts_per_tok)
    num_experts = _num_routed_experts(text_config)
    model_dtype = vllm_config.model_config.dtype
    plan = _attention_quant_plan(vllm_config.quant_config)
    if plan is None:
        return RoutedWireSpec(
            hidden_size=hidden_size,
            top_k=top_k,
            num_experts=num_experts,
            model_dtype=model_dtype,
            act_dtype=model_dtype,
            quant_dtype=None,
            block_shape=None,
            scale_tail=None,
        )
    quant_dtype, block_shape = plan
    block_k = block_shape[1]
    if hidden_size % block_k != 0:
        raise RuntimeError(
            f"hidden size {hidden_size} is not a multiple of the fp8 block "
            f"size {block_k}",
        )
    return RoutedWireSpec(
        hidden_size=hidden_size,
        top_k=top_k,
        num_experts=num_experts,
        model_dtype=model_dtype,
        act_dtype=quant_dtype,
        quant_dtype=quant_dtype,
        block_shape=block_shape,
        scale_tail=(hidden_size // block_k,),
    )


def _validate_routed_runtime(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    spec: RoutedWireSpec,
) -> None:
    parallel = vllm_config.parallel_config
    ffn_size = int(afd_config.num_ffn_ranks)
    if not bool(vllm_config.model_config.enforce_eager):
        raise RuntimeError(
            "P2pNcclRoutedAFDConnector requires --enforce-eager on both roles: "
            "the per-layer token count exchange is a host sync that cannot be "
            "captured in a CUDA graph",
        )
    if int(parallel.pipeline_parallel_size) != 1:
        raise RuntimeError("P2pNcclRoutedAFDConnector does not support PP")
    if int(parallel.prefill_context_parallel_size) != 1:
        raise RuntimeError("P2pNcclRoutedAFDConnector does not support PCP")
    if spec.num_experts % ffn_size != 0:
        raise RuntimeError(
            f"P2pNcclRoutedAFDConnector needs num_experts ({spec.num_experts}) "
            f"divisible by num_ffn_ranks ({ffn_size})",
        )
    if afd_config.role != "ffn":
        return
    problems: list[str] = []
    if not bool(parallel.enable_expert_parallel):
        problems.append("--enable-expert-parallel")
    if parallel.all2all_backend != "prerouted":
        problems.append("--all2all-backend prerouted")
    if int(parallel.tensor_parallel_size) != 1:
        problems.append("--tensor-parallel-size 1")
    if int(parallel.data_parallel_size) != ffn_size:
        problems.append(f"--data-parallel-size {ffn_size} (= num_ffn_ranks)")
    if bool(parallel.enable_eplb):
        problems.append("no EPLB")
    if parallel.expert_placement_strategy != "linear":
        problems.append("expert_placement_strategy=linear")
    if problems:
        raise RuntimeError(
            "P2pNcclRoutedAFDConnector FFN role requires: " + ", ".join(problems),
        )


# ==============================
# Transfer state
# ==============================


@dataclass(slots=True)
class RoutedTransferState(AFDTransferState):
    """FFN-side state between ``recv_attn_output`` and ``send_ffn_output``."""

    plan: P2pAll2AllPlan
    stream: torch.cuda.Stream


# ==============================
# Connector
# ==============================


class P2pNcclRoutedAFDConnector(AFDConnectorBase):
    """One-hop Attention -> owning FFN ranks -> Attention connector."""

    # Read by the FFN runner and the model wrappers to select the routed path.
    is_routed = True

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("routed connector_extra_config must be a mapping")
        if raw:
            raise ValueError(
                "P2pNcclRoutedAFDConnector does not support connector_extra_config",
            )
        return ConnectorExtraInfo()

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        self.attn_size = int(afd_config.num_attention_ranks)
        self.ffn_size = int(afd_config.num_ffn_ranks)
        self.world_size = self.attn_size + self.ffn_size
        self.is_ffn = afd_config.role == "ffn"
        # AFD world: FFN ranks first, then Attention ranks.
        self.world_rank = role_rank if self.is_ffn else self.ffn_size + role_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.spec = resolve_wire_spec(vllm_config)
        _validate_routed_runtime(vllm_config, afd_config, self.spec)
        self.num_local_experts = self.spec.num_experts // self.ffn_size
        self.max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)

        self.dispatch_pg: ProcessGroup | None = None
        self.combine_pg: ProcessGroup | None = None
        self.control_pg: ProcessGroup | None = None
        self.engine: P2pAll2AllEngine | None = None
        # Attention: plans of the stages whose dispatch is in flight.
        self._plans: dict[int, P2pAll2AllPlan] = {}
        self._empty_partial: torch.Tensor | None = None
        # FFN: one CUDA stream per stage plus per-communicator ordering chains.
        self._stage_streams: dict[int, torch.cuda.Stream] = {}
        self._dispatch_chain: torch.cuda.Event | None = None
        self._combine_chain: torch.cuda.Event | None = None
        self._empty_send: tuple[torch.Tensor, ...] | None = None
        self._empty_ids: torch.Tensor | None = None

        self.dp_metadata_list: dict[int, AFDDPMetadata] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self._num_steps = 0  # DEBUG(p2p_nccl)
        self._num_sends = 0  # DEBUG(p2p_nccl)
        self._num_recvs = 0  # DEBUG(p2p_nccl)
        self.control_plane = P2pNcclRoutedControlPlane(self)

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        # Like the two-hop connector, the NCCL groups are left to process
        # teardown; destroying them here can hang when peers already exited.
        self.engine = None
        self._plans.clear()
        self._stage_streams.clear()
        self._dispatch_chain = None
        self._combine_chain = None
        self._initialized = False

    def init_afd_connector(self) -> None:
        """Create the dispatch/combine/control NCCL groups and the engine.

        Collective: every Attention and FFN rank must call it. The control
        group only contains the FFN ranks and Attention role rank 0.
        """
        if self._initialized:
            return
        init_method = f"tcp://{self.afd_config.host}:{self.afd_config.port}"
        self.dispatch_pg = init_afd_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=self.world_size,
            rank=self.world_rank,
            group_name="afd_routed_dispatch",
            timeout=timedelta(minutes=2),
        )
        self.combine_pg = init_afd_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=self.world_size,
            rank=self.world_rank,
            group_name="afd_routed_combine",
            timeout=timedelta(minutes=2),
        )
        if self.is_ffn or self.role_rank == 0:
            # FFN ranks are 0..F-1 and Attention role rank 0 is world rank F,
            # so the world ranks double as the control-group ranks.
            self.control_pg = init_afd_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=self.ffn_size + 1,
                rank=self.world_rank,
                group_name="afd_routed_control",
                timeout=timedelta(minutes=30),
            )
        self._check_wire_spec()
        self.engine = P2pAll2AllEngine(
            group=self.dispatch_pg,
            rank=self.world_rank,
            world_size=self.world_size,
            num_local_experts=self.num_local_experts,
            max_tokens_per_rank=self.max_tokens,
            device=self.device,
            combine_group=self.combine_pg,
        )
        spec = self.spec
        if self.is_ffn:
            # FFN ranks have no tokens of their own: they take part in every
            # dispatch with zero rows of the right shapes and dtypes.
            self._empty_ids = torch.empty(
                (0, spec.top_k), dtype=_IDS_DTYPE, device=self.device
            )
            empty: list[torch.Tensor] = [
                torch.empty((0, spec.hidden_size), dtype=spec.act_dtype, device=self.device),
                self._empty_ids,
                torch.empty((0, spec.top_k), dtype=_WEIGHTS_DTYPE, device=self.device),
            ]
            if spec.scale_tail is not None:
                empty.append(
                    torch.empty(
                        (0, *spec.scale_tail), dtype=_SCALE_DTYPE, device=self.device
                    )
                )
            self._empty_send = tuple(empty)
        else:
            self._empty_partial = torch.empty(
                (0, spec.hidden_size), dtype=spec.model_dtype, device=self.device
            )
        logger.info(
            "routed AFD connector ready: role=%s role_rank=%d world_rank=%d/%d "
            "experts=%d (%d local per FFN rank) top_k=%d wire=%s scale_tail=%s",
            self.afd_config.role,
            self.role_rank,
            self.world_rank,
            self.world_size,
            spec.num_experts,
            self.num_local_experts,
            spec.top_k,
            spec.act_dtype,
            spec.scale_tail,
        )
        self._initialized = True

    def _check_wire_spec(self) -> None:
        """Compare this rank's wire spec with Attention role rank 0's."""
        assert self.dispatch_pg is not None
        mine = torch.tensor(self.spec.encode(), dtype=torch.int64, device=self.device)
        reference = mine.clone()
        dist.broadcast(reference, src=self.ffn_size, group=self.dispatch_pg)
        if not torch.equal(reference.cpu(), mine.cpu()):
            raise RuntimeError(
                "routed AFD wire spec mismatch: attention rank 0 has "
                f"{reference.tolist()} but this rank derived {mine.tolist()} "
                "(hidden, top_k, num_experts, dtype, scale_tail, block_k); "
                "check that both roles use the same model and quantization",
            )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------- attention data path
    def quantize_for_dispatch(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize activations on Attention when the wire spec says so."""
        spec = self.spec
        if not spec.quantize_on_attention:
            return hidden_states, None
        assert spec.block_shape is not None
        a1q, a1q_scale = moe_kernel_quantize_input(
            hidden_states,
            None,
            quant_dtype=spec.quant_dtype,
            per_act_token_quant=False,
            block_shape=list(spec.block_shape),
            is_scale_swizzled=False,
        )
        return a1q, a1q_scale

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Dispatch this stage's tokens to the FFN ranks owning their experts.

        Args:
            hidden_states: ``[tokens, hidden]`` activations, already quantized
                when ``quantize_for_dispatch`` did so.
            context: Transfer context; ``metadata.stage_idx`` selects the
                engine slot (one per micro-batch).
            **kwargs: ``topk_weights`` and ``topk_ids`` (``[tokens, top_k]``)
                and, for quantized activations, ``a1q_scale``.
        """
        if self.engine is None:
            raise RuntimeError("routed connector is not initialized")
        spec = self.spec
        topk_ids: torch.Tensor | None = kwargs.get("topk_ids")
        topk_weights: torch.Tensor | None = kwargs.get("topk_weights")
        a1q_scale: torch.Tensor | None = kwargs.get("a1q_scale")
        if topk_ids is None or topk_weights is None:
            raise ValueError("routed send_attn_output needs topk_ids and topk_weights")
        num_tokens = int(hidden_states.shape[0])
        if topk_ids.shape != (num_tokens, spec.top_k):
            raise ValueError(
                f"topk_ids shape {tuple(topk_ids.shape)} does not match "
                f"({num_tokens}, {spec.top_k})",
            )
        if hidden_states.dtype != spec.act_dtype:
            raise ValueError(
                f"hidden_states dtype {hidden_states.dtype} does not match the "
                f"wire dtype {spec.act_dtype}",
            )
        tensors: list[torch.Tensor] = [
            hidden_states,
            topk_ids.to(_IDS_DTYPE),
            topk_weights.to(_WEIGHTS_DTYPE),
        ]
        if spec.scale_tail is not None:
            if a1q_scale is None:
                raise ValueError("quantized activations need a1q_scale")
            if tuple(a1q_scale.shape) != (num_tokens, *spec.scale_tail):
                raise ValueError(
                    f"a1q_scale shape {tuple(a1q_scale.shape)} does not match "
                    f"({num_tokens}, {spec.scale_tail})",
                )
            tensors.append(a1q_scale.to(_SCALE_DTYPE))

        stage = int(context.metadata.stage_idx)
        if stage in self._plans:
            raise RuntimeError(f"stage {stage} already has a dispatch in flight")
        engine = self.engine
        pending = engine.begin_plan(tensors[1], slot=stage)
        # Dual batch overlap: give the CPU to the other micro-batch while the
        # counts travel, and run the collectives on the communication stream.
        # Without ubatching these calls are no-ops.
        dbo_yield_and_switch_from_compute_to_comm()
        engine.exchange_counts(pending)
        plan = engine.finish_plan(pending)
        engine.dispatch(plan, tuple(tensors))
        self._plans[stage] = plan
        # Back to the compute stream without waiting for the dispatch: the
        # shared experts can run while the rows are on the wire.
        dbo_switch_to_compute()
        # DEBUG(p2p_nccl): remove after GPU validation.
        if _DEBUG and _should_log(self._num_sends):
            logger.info(
                "routed[A%d] send#%d layer=%d stage=%d T=%d send=%s rows=%d "
                "host_wait=%.3fms",
                self.role_rank,
                self._num_sends,
                context.metadata.layer_idx,
                stage,
                num_tokens,
                plan.send_counts,
                plan.num_send,
                plan.host_wait_ms,
            )
        self._num_sends += 1

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Collect the partial sums of this stage into a new ``[tokens, hidden]``.

        ``ref_tensor`` only provides the shape, dtype and device of the result
        (the unquantized MoE input of the stage).
        """
        if self.engine is None:
            raise RuntimeError("routed connector is not initialized")
        plan = self._plans.pop(int(ubatch_idx), None)
        if plan is None:
            raise RuntimeError(f"no dispatch in flight for stage {ubatch_idx}")
        if ref_tensor.shape[0] != plan.num_tokens:
            raise ValueError(
                f"ref_tensor has {ref_tensor.shape[0]} rows, stage {ubatch_idx} "
                f"dispatched {plan.num_tokens}",
            )
        assert self._empty_partial is not None
        output = torch.empty_like(ref_tensor)
        dbo_yield_and_switch_from_compute_to_comm()
        back = self.engine.combine_send(plan, self._empty_partial)
        self.engine.combine_accumulate(plan, back, output)
        dbo_yield_and_switch_from_comm_to_compute()
        return output

    # ------------------------------------------------------- ffn data path
    def _stage_stream(self, stage: int) -> torch.cuda.Stream:
        stream = self._stage_streams.get(stage)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._stage_streams[stage] = stream
        return stream

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive the rows routed to this FFN rank for one stage.

        Runs the count exchange and the dispatch on the stage's own CUDA
        stream (chained after the previous stage's dispatch) and makes the
        current stream wait for the data.
        """
        if self.engine is None or self._empty_send is None:
            raise RuntimeError("routed connector is not initialized")
        stage = int(ubatch_idx)
        stream = self._stage_stream(stage)
        engine = self.engine
        with torch.cuda.stream(stream):
            if self._dispatch_chain is not None:
                stream.wait_event(self._dispatch_chain)
            pending = engine.begin_plan(self._empty_send[1], slot=stage)
            engine.exchange_counts(pending)
            plan = engine.finish_plan(pending)
            received = engine.dispatch(plan, self._empty_send)
            chain = torch.cuda.Event()
            chain.record(stream)
            self._dispatch_chain = chain
        torch.cuda.current_stream(self.device).wait_stream(stream)

        hidden_states, topk_ids, topk_weights = received[0], received[1], received[2]
        a1q_scale = received[3] if self.spec.scale_tail is not None else None
        # DEBUG(p2p_nccl): remove after GPU validation.
        if _DEBUG and _should_log(self._num_recvs):
            logger.info(
                "routed[F%d] recv#%d stage=%d rows=%d recv=%s host_wait=%.3fms",
                self.role_rank,
                self._num_recvs,
                stage,
                plan.num_recv,
                plan.recv_counts,
                plan.host_wait_ms,
            )
        self._num_recvs += 1
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=stage,
            # The metadata requires positive lengths; the real row count is
            # plan.num_recv (may be zero).
            seq_lens=[max(1, plan.num_recv)],
        )
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=AFDTransferContext(
                metadata=metadata,
                states=RoutedTransferState(plan=plan, stream=stream),
            ),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Return the partial sums of this stage on the stage's stream."""
        if self.engine is None:
            raise RuntimeError("routed connector is not initialized")
        state = context.states
        if not isinstance(state, RoutedTransferState):
            raise RuntimeError("send_ffn_output needs the state of recv_attn_output")
        plan, stream = state.plan, state.stream
        if ffn_output.shape[0] != plan.num_recv:
            raise ValueError(
                f"ffn_output has {ffn_output.shape[0]} rows, stage "
                f"{plan.slot} received {plan.num_recv}",
            )
        if ffn_output.dtype != self.spec.model_dtype:
            raise ValueError(
                f"ffn_output dtype {ffn_output.dtype} differs from the model "
                f"dtype {self.spec.model_dtype} the Attention side accumulates",
            )
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            stream.wait_event(compute_done)
            if self._combine_chain is not None:
                stream.wait_event(self._combine_chain)
            # The partial sums were allocated on the compute stream.
            ffn_output.record_stream(stream)
            self.engine.combine_send(plan, ffn_output)
            chain = torch.cuda.Event()
            chain.record(stream)
            self._combine_chain = chain


class P2pNcclRoutedControlPlane(AFDControlPlane):
    """Per-step control payload: Attention role rank 0 -> every FFN rank."""

    def __init__(self, connector: P2pNcclRoutedAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup
        # A new step: the FFN loop synchronized the device at the end of the
        # previous one, so the per-communicator chains start afresh.
        connector._dispatch_chain = None
        connector._combine_chain = None
        if connector._plans:
            logger.warning(
                "routed AFD connector: %d dispatches of the previous step were "
                "never collected",
                len(connector._plans),
            )
            connector._plans.clear()
        # DEBUG(p2p_nccl): remove after GPU validation.
        if _DEBUG and _should_log(connector._num_steps):
            logger.info(
                "routed[%s%d] step#%d stages=%s warmup=%s capture=%s profile=%s",
                "F" if connector.is_ffn else "A",
                connector.role_rank,
                connector._num_steps,
                sorted(int(idx) for idx in payload.dp_metadata_list),
                payload.is_warmup,
                payload.is_graph_capturing,
                payload.is_profile,
            )
        connector._num_steps += 1

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        if connector.control_pg is None or connector.is_ffn or connector.role_rank != 0:
            return
        send_control_payload(
            payload,
            dst=list(range(connector.ffn_size)),
            group=connector.control_pg,
            device=connector.device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.control_pg is None:
            raise RuntimeError("routed control process group is not initialized")
        return recv_control_payload(
            src=connector.ffn_size,
            group=connector.control_pg,
            device=connector.device,
        )


__all__ = [
    "P2pNcclRoutedAFDConnector",
    "P2pNcclRoutedControlPlane",
    "RoutedTransferState",
    "RoutedWireSpec",
    "resolve_wire_spec",
]
