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
    1. Attention computes gate + top-k, quantizes the activations only when
       the FFN experts kernel wants fp8 inputs that Attention can reproduce
       (see "Wire format" below), then runs the
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
    - Wire format: FFN role rank 0 derives it at init from the experts
      kernel it actually loaded and broadcasts it; Attention follows it and
      the other FFN ranks check theirs matches. Attention quantizes before
      the dispatch only for fp8 with dynamic per-token-group (block) or
      per-token scales; every other case (W8A16 such as Marlin, unquantized,
      static or per-tensor scales, int8, nvfp4/mxfp4, kernels that quantize
      internally) sends the model dtype and the FFN quantizes after receive.
"""

from __future__ import annotations

import os
import socket
import time
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
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_switch_to_comm_sync,
    dbo_switch_to_compute,
    dbo_yield,
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

# Under the ``vllm`` namespace: vLLM only configures handlers for ``vllm.*``.
logger = init_logger(f"vllm.{__name__}")

# DEBUG(p2p_nccl): off by default. VLLM_P2P_NCCL_DEBUG=1 logs the first
# stages of every step (counts, bytes, host wait) on both roles.
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
    # A prime period samples every micro-batch index (64 hit only stage 0 at 4).
    return n < 8 or n % 61 == 0


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
    # Per-token (one scale per row) instead of per-token-group (block) scales.
    per_token_quant: bool = False

    @property
    def quantize_on_attention(self) -> bool:
        return self.quant_dtype is not None

    @property
    def quant_mode(self) -> str:
        if self.quant_dtype is None:
            return "none"
        return "per_token" if self.per_token_quant else "block"

    def model_fields(self) -> tuple:
        return (self.hidden_size, self.top_k, self.num_experts, self.model_dtype)

    def encode(self) -> list[int]:
        mode = {"none": 0, "block": 1, "per_token": 2}[self.quant_mode]
        return [
            self.hidden_size,
            self.top_k,
            self.num_experts,
            _WIRE_DTYPES.index(self.model_dtype),
            _WIRE_DTYPES.index(self.act_dtype),
            mode,
            self.block_shape[0] if self.block_shape else -1,
            self.block_shape[1] if self.block_shape else -1,
            self.scale_tail[0] if self.scale_tail else -1,
        ]

    @classmethod
    def decode(cls, values: list[int]) -> RoutedWireSpec:
        hidden, top_k, num_experts, mdt, adt, mode, bn, bk, tail = (
            int(v) for v in values
        )
        model_dtype = _WIRE_DTYPES[mdt]
        if mode == 0:
            return _unquantized_spec(hidden, top_k, num_experts, model_dtype)
        act_dtype = _WIRE_DTYPES[adt]
        return cls(
            hidden_size=hidden,
            top_k=top_k,
            num_experts=num_experts,
            model_dtype=model_dtype,
            act_dtype=act_dtype,
            quant_dtype=act_dtype,
            block_shape=(bn, bk) if mode == 1 else None,
            scale_tail=(tail,),
            per_token_quant=mode == 2,
        )


def _unquantized_spec(
    hidden_size: int, top_k: int, num_experts: int, model_dtype: torch.dtype
) -> RoutedWireSpec:
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


def _num_routed_experts(text_config: PretrainedConfig) -> int:
    # HF configs name the routed expert count differently per model family.
    for name in ("n_routed_experts", "num_experts", "num_local_experts"):
        value = getattr(text_config, name, None)
        if value:
            return int(value)
    raise RuntimeError("cannot determine the number of routed experts")


def resolve_wire_spec(vllm_config: VllmConfig) -> RoutedWireSpec:
    """Model-level part of the wire spec, without quantization.

    Both roles derive it from their own config. The quantization part is
    decided by FFN role rank 0 at init (``ffn_wire_spec``) because only the
    FFN role holds the experts kernel and knows which inputs it wants.
    """
    text_config = vllm_config.model_config.hf_text_config
    return _unquantized_spec(
        int(text_config.hidden_size),
        int(text_config.num_experts_per_tok),
        _num_routed_experts(text_config),
        vllm_config.model_config.dtype,
    )


def _find_moe_kernels(model: Any) -> list[Any]:
    """The modular MoE kernels of a loaded model (one per MoE layer)."""
    root = model
    for _ in range(4):
        if isinstance(root, torch.nn.Module):
            break
        root = (
            getattr(root, "runnable", None)
            or getattr(root, "model", None)
            or getattr(root, "module", None)
        )
    if not isinstance(root, torch.nn.Module):
        raise RuntimeError(f"cannot find the FFN model module in {type(model)}")
    kernels: list[Any] = []
    for module in root.modules():
        quant_method = getattr(module, "quant_method", None)
        kernel = getattr(quant_method, "moe_kernel", None)
        if kernel is not None and all(kernel is not k for k in kernels):
            kernels.append(kernel)
    return kernels


def _kernel_wire_plan(
    kernel: Any,
) -> tuple[torch.dtype, tuple[int, int] | None, bool] | None:
    """``(fp8 dtype, block shape, per_token)`` if Attention may quantize.

    Mirrors the non-AFD ``p2p_nccl`` rule (quantize before the dispatch only
    when the scales travel per row), restricted to what Attention can
    reproduce without FFN state: fp8 with dynamic per-token-group or
    per-token scales. Static scales live in the FFN weights, per-tensor
    scales depend on the FFN's whole batch, and kernels that quantize
    internally want the model dtype.
    """
    fused_experts = kernel.fused_experts
    if fused_experts.expects_unquantized_inputs:
        return None
    qc = fused_experts.quant_config
    quant_dtype = qc.quant_dtype
    if quant_dtype is None or isinstance(quant_dtype, str):
        return None
    if quant_dtype != current_platform.fp8_dtype() or qc.a1_scale is not None:
        return None
    if qc.block_shape is not None and not qc.per_act_token_quant:
        return quant_dtype, (int(qc.block_shape[0]), int(qc.block_shape[1])), False
    if qc.per_act_token_quant and qc.block_shape is None:
        return quant_dtype, None, True
    return None


def ffn_wire_spec(base: RoutedWireSpec, model: Any) -> RoutedWireSpec:
    """Wire spec the FFN experts kernel of ``model`` wants (FFN role only)."""
    kernels = _find_moe_kernels(model)
    if not kernels:
        raise RuntimeError("routed AFD FFN role: the model has no MoE kernels")
    plans = {_kernel_wire_plan(k) for k in kernels}
    if len(plans) != 1:
        # Layers disagree: send the model dtype, each layer quantizes itself.
        logger.warning(
            "routed AFD: MoE layers want different input quantizations %s; "
            "sending %s and quantizing on the FFN rank",
            plans,
            base.model_dtype,
        )
        return base
    plan = plans.pop()
    if plan is None:
        return base
    quant_dtype, block_shape, per_token = plan
    if block_shape is not None:
        block_k = block_shape[1]
        if base.hidden_size % block_k != 0:
            raise RuntimeError(
                f"hidden size {base.hidden_size} is not a multiple of the fp8 "
                f"block size {block_k}",
            )
        scale_tail = (base.hidden_size // block_k,)
    else:
        scale_tail = (1,)
    return RoutedWireSpec(
        hidden_size=base.hidden_size,
        top_k=base.top_k,
        num_experts=base.num_experts,
        model_dtype=base.model_dtype,
        act_dtype=quant_dtype,
        quant_dtype=quant_dtype,
        block_shape=block_shape,
        scale_tail=scale_tail,
        per_token_quant=per_token,
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


@dataclass(frozen=True)
class RoutedPrefetch:
    """FFN-side receive of one stage, posted but not yet waited on.

    Returned by ``prefetch_attn_output`` and consumed by ``recv_attn_output``.
    ``received`` holds the per-tensor receive buffers (valid until the next
    dispatch on the same stage), ``ready`` is recorded on ``stream`` right
    after the receive.
    """

    stage: int
    plan: P2pAll2AllPlan
    received: list[torch.Tensor]
    stream: torch.cuda.Stream
    ready: torch.cuda.Event


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
        # Model-level fields only; init_afd_connector() fills in the
        # quantization that FFN role rank 0 derives from its experts kernel.
        self.spec = resolve_wire_spec(vllm_config)
        _validate_routed_runtime(vllm_config, afd_config, self.spec)
        self._ffn_model: Any | None = None
        self.num_local_experts = self.spec.num_experts // self.ffn_size
        self.max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)

        self.dispatch_pg: ProcessGroup | None = None
        self.combine_pg: ProcessGroup | None = None
        self.control_pg: ProcessGroup | None = None
        self.engine: P2pAll2AllEngine | None = None
        # Attention: plans of the stages whose dispatch is in flight.
        self._plans: dict[int, P2pAll2AllPlan] = {}
        # Attention: stage -> (plan, result buffer, result-received event).
        self._pending: dict[
            int, tuple[P2pAll2AllPlan, torch.Tensor, torch.cuda.Event]
        ] = {}
        self._result_stream: torch.cuda.Stream | None = None
        # Attention: (stage, plan) whose result receive waits for the next
        # dispatch (see send_attn_output), and the last MoE layer of a forward
        # (set by the Attention model runner after loading the model).
        self._deferred_result: tuple[int, P2pAll2AllPlan] | None = None
        self._last_moe_layer: int | None = None
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
    def bind_ffn_model(self, model: Any) -> None:
        """FFN role: hand over the loaded model before init_afd_connector()."""
        self._ffn_model = model

    def close(self) -> None:
        # Like the two-hop connector, the NCCL groups are left to process
        # teardown; destroying them here can hang when peers already exited.
        self.engine = None
        self._plans.clear()
        self._pending.clear()
        self._deferred_result = None
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
        if self.world_rank != 0:
            self._wait_for_rendezvous()
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
        self._exchange_wire_spec()
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
                torch.empty(
                    (0, spec.hidden_size), dtype=spec.act_dtype, device=self.device
                ),
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
            "experts=%d (%d local per FFN rank) top_k=%d wire=%s quant=%s "
            "scale_tail=%s",
            self.afd_config.role,
            self.role_rank,
            self.world_rank,
            self.world_size,
            spec.num_experts,
            self.num_local_experts,
            spec.top_k,
            spec.act_dtype,
            spec.quant_mode,
            spec.scale_tail,
        )
        self._initialized = True

    def _wait_for_rendezvous(self) -> None:
        """Wait until FFN role rank 0 opens the rendezvous port.

        The roles reach init_afd_connector at different times: Attention
        loads only the attention weights, FFN loads the experts and profiles
        first. The process groups use a short timeout so that a stalled peer
        is detected quickly during serving; only this initial wait is long.
        """
        host, port = self.afd_config.host, int(self.afd_config.port)
        wait_s = float(os.environ.get("AFD_RENDEZVOUS_WAIT_S", "1800"))
        deadline = time.monotonic() + wait_s
        while True:
            try:
                with socket.create_connection((host, port), timeout=5):
                    return
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"routed AFD: rendezvous {host}:{port} did not open within "
                        f"{wait_s:.0f}s (FFN role rank 0 not up?)",
                    ) from None
                time.sleep(2)

    def _exchange_wire_spec(self) -> None:
        """Adopt the wire spec of FFN role rank 0 (world rank 0).

        FFN ranks derive the spec from the experts kernel they loaded; the
        other ranks only check the model-level fields. FFN ranks must agree
        with rank 0 (same kernel everywhere).
        """
        assert self.dispatch_pg is not None
        base = self.spec
        if self.is_ffn:
            if self._ffn_model is None:
                raise RuntimeError(
                    "routed AFD FFN role: bind_ffn_model() must be called "
                    "before init_afd_connector()",
                )
            mine = ffn_wire_spec(base, self._ffn_model)
        else:
            mine = base
        reference = torch.tensor(mine.encode(), dtype=torch.int64, device=self.device)
        dist.broadcast(reference, src=0, group=self.dispatch_pg)
        spec = RoutedWireSpec.decode(reference.cpu().tolist())
        if spec.model_fields() != base.model_fields():
            raise RuntimeError(
                "routed AFD wire spec mismatch: FFN rank 0 has (hidden, top_k, "
                f"num_experts, dtype)={spec.model_fields()} but this rank has "
                f"{base.model_fields()}; check that both roles serve the same model",
            )
        if self.is_ffn and spec != mine:
            raise RuntimeError(
                f"routed AFD: FFN rank 0 wants wire {spec} but FFN rank "
                f"{self.role_rank} derived {mine}; all FFN ranks must load the "
                "same experts kernel (check --kernel-config and GPU types)",
            )
        self.spec = spec

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
        a1q, a1q_scale = moe_kernel_quantize_input(
            hidden_states,
            None,
            quant_dtype=spec.quant_dtype,
            per_act_token_quant=spec.per_token_quant,
            block_shape=list(spec.block_shape) if spec.block_shape else None,
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
        if stage in self._pending:
            raise RuntimeError(f"stage {stage} already has a dispatch in flight")
        engine = self.engine
        pending = engine.begin_plan(tensors[1], slot=stage)
        # Each micro-batch runs its count exchange and dispatch within its own
        # turn and gives the CPU away once, in recv_ffn_output (send, then
        # yield, as vLLM's DBO does with DeepEP low-latency). The next layer's
        # first dispatch therefore precedes the previous layer's last result,
        # which lets the FFN prefetch across layers (see below), and every
        # micro-batch thread yields the same number of times. The comm stream
        # waits for this micro-batch's compute before the count exchange.
        dbo_switch_to_comm_sync()
        engine.exchange_counts(pending)
        plan = engine.finish_plan(pending)
        engine.dispatch(plan, tuple(tensors))
        # Result receives, posted on their own stream and consumed only when
        # the stage needs them (recv_ffn_output). The FFN posts the receive
        # of the next stage before it launches the experts of the current
        # one (so those rows travel while the current stage is computed),
        # also across layers, and returns the current stage after that.
        # Every rank must launch the NCCL operations of the two communicators
        # in the same order, so the receive of a stage is posted right after
        # the next dispatch; only the last stage of the last MoE layer (and a
        # forward without micro-batches) posts its own right away:
        #   D0 D1 C0 D2 C1 | D0' C2 D1' C0' D2' C1' | ... C2''
        num_stages = max(1, len(self.dp_metadata_list)) if dbo_enabled() else 1
        deferred = self._deferred_result
        self._deferred_result = None
        if deferred is not None:
            expected = stage - 1 if stage > 0 else num_stages - 1
            if deferred[0] != expected:
                raise RuntimeError(
                    f"stage {stage} dispatched after stage {deferred[0]}; "
                    "stages must be dispatched in order",
                )
            self._post_result_recv(*deferred)
        if stage + 1 < num_stages:
            self._deferred_result = (stage, plan)
        elif num_stages > 1 and int(context.metadata.layer_idx) != self._last_moe_layer:
            if self._last_moe_layer is None:
                raise RuntimeError(
                    "routed connector needs set_moe_layer_indices() with micro-batches",
                )
            self._deferred_result = (stage, plan)
        else:
            self._post_result_recv(stage, plan)
        # Back to the compute stream without waiting for the dispatch: the
        # shared experts can run while the rows are on the wire.
        dbo_switch_to_compute()
        # DEBUG(p2p_nccl): only with VLLM_P2P_NCCL_DEBUG.
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

    def set_moe_layer_indices(self, layer_indices: Any) -> None:
        """Record the MoE layers of the model (the last one ends a forward)."""
        indices = [int(idx) for idx in layer_indices]
        self._last_moe_layer = max(indices) if indices else None

    def _post_result_recv(self, stage: int, plan: P2pAll2AllPlan) -> None:
        """Post the receive of ``stage``'s partial sums on the result stream."""
        assert self.engine is not None and self._empty_partial is not None
        result_stream = self._attn_result_stream()
        # Ordered after everything launched so far on the communication
        # stream, so the GPU runs the NCCL operations in launch order.
        result_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(result_stream):
            back = self.engine.combine_send(plan, self._empty_partial)
            received = torch.cuda.Event()
            received.record(result_stream)
        self._pending[stage] = (plan, back, received)

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
        # Let the other micro-batches compute and dispatch while this stage's
        # results travel back. The receive of this stage is posted by the next
        # dispatch: the next stage's in the same layer, or for the last stage
        # the first stage's of the next layer. Both run in the turns of the
        # other micro-batches before this one gets the CPU back.
        dbo_yield()
        pending = self._pending.pop(int(ubatch_idx), None)
        if pending is None:
            raise RuntimeError(f"no dispatch in flight for stage {ubatch_idx}")
        plan, back, received = pending
        if ref_tensor.shape[0] != plan.num_tokens:
            raise ValueError(
                f"ref_tensor has {ref_tensor.shape[0]} rows, stage {ubatch_idx} "
                f"dispatched {plan.num_tokens}",
            )
        output = torch.empty_like(ref_tensor)
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(received)
        # ``perm`` was allocated on the communication stream; the plan may be
        # released before the compute stream reads it below.
        back.record_stream(compute)
        plan.perm.record_stream(compute)
        self.engine.combine_accumulate(plan, back, output)
        return output

    def _attn_result_stream(self) -> torch.cuda.Stream:
        if self._result_stream is None:
            self._result_stream = torch.cuda.Stream(device=self.device)
        return self._result_stream

    # ------------------------------------------------------- ffn data path
    def _stage_stream(self, stage: int) -> torch.cuda.Stream:
        stream = self._stage_streams.get(stage)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._stage_streams[stage] = stream
        return stream

    def prefetch_attn_output(self, ubatch_idx: int = 0) -> RoutedPrefetch:
        """Run the count exchange of one stage and post its dispatch receive.

        Both run on the stage's own CUDA stream, chained after the previous
        stage's receive so every rank keeps the same NCCL order. The current
        stream does not wait: the FFN posts stage i+1 before it launches the
        experts of stage i, so the rows of i+1 travel while i is computed.
        ``recv_attn_output`` makes the compute stream wait for them.
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
            ready = torch.cuda.Event()
            ready.record(stream)
            self._dispatch_chain = ready
        # DEBUG(p2p_nccl): only with VLLM_P2P_NCCL_DEBUG.
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
        return RoutedPrefetch(
            stage=stage,
            plan=plan,
            received=received,
            stream=stream,
            ready=ready,
        )

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        prefetched: RoutedPrefetch | None = None,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive the rows routed to this FFN rank for one stage.

        Uses ``prefetched`` (from ``prefetch_attn_output``) when given,
        otherwise posts the receive now, then makes the current stream wait
        for the data.
        """
        stage = int(ubatch_idx)
        if prefetched is None:
            prefetched = self.prefetch_attn_output(stage)
        elif prefetched.stage != stage:
            raise RuntimeError(
                f"prefetched stage {prefetched.stage} consumed as stage {stage}",
            )
        plan, received, stream = prefetched.plan, prefetched.received, prefetched.stream
        torch.cuda.current_stream(self.device).wait_event(prefetched.ready)

        hidden_states, topk_ids, topk_weights = received[0], received[1], received[2]
        a1q_scale = received[3] if self.spec.scale_tail is not None else None
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
        if connector._deferred_result is not None:
            logger.warning(
                "routed AFD connector: result receive of stage %d of the "
                "previous step was never posted",
                connector._deferred_result[0],
            )
            connector._deferred_result = None
        if connector._plans:
            logger.warning(
                "routed AFD connector: %d dispatches of the previous step were "
                "never collected",
                len(connector._plans),
            )
            connector._plans.clear()
        # DEBUG(p2p_nccl): only with VLLM_P2P_NCCL_DEBUG.
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
    "ffn_wire_spec",
    "resolve_wire_spec",
]
