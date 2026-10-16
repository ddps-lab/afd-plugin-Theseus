# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One-hop routed AFD connector that moves the rows with NIXL.

``NixlRoutedAFDConnector`` keeps the routing of ``P2pNcclRoutedAFDConnector``
(Attention computes the gate and sends each token once to every FFN rank that
owns one of its experts, FFN runs its experts through the ``prerouted``
backend and returns the weighted partial sums) together with its NCCL groups
for the per-layer count exchange and the control plane. Only the rows travel
differently, as NIXL WRITEs followed by a notification:

    Attention a -> FFN f: rows into f's receive region of a.
    FFN f -> Attention a: partial sums into a's result region of f.

Every rank registers its receive buffers once at init, split into one region
of ``R = max_num_batched_tokens`` rows per peer, so nothing is registered per
layer. With two ranks per role:

    FFN f (written by Attention)            Attention a (written by FFN)
                  hidden  ids  weights                    results
    rows 0..R-1   <- Attention 0            rows 0..R-1   <- FFN 0
    rows R..2R-1  <- Attention 1            rows R..2R-1  <- FFN 1

Attention 1 sending 300 rows to FFN 0 writes rows R..R+299 of FFN 0's hidden,
ids and weights buffers in one WRITE with three descriptors. A region holds the
share of any batch because a token goes to an FFN rank at most once.

Both ends of a pair know its row count from the count exchange: a pair without
rows sends no WRITE and waits for no notification. A notification carries the
plan sequence number, which advances identically on every rank because every
rank takes part in every count exchange. That exchange is also what makes a
region safe to rewrite: a rank only reaches the next layer's exchange after it
has consumed this one.

Limitations: one micro-batch (no DBO), unquantized rows, Attention TP=1.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from nixl import nixl_agent, nixl_agent_config
from vllm.distributed.device_communicators.p2p_nccl_all2all import P2pAll2AllPlan
from vllm.logger import init_logger

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.gpu.routed import (
    _IDS_DTYPE,
    _WEIGHTS_DTYPE,
    P2pNcclRoutedAFDConnector,
    RoutedPrefetch,
    RoutedTransferState,
)
from afd_plugin.connectors.metadata import AFDTransferContext

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(f"vllm.{__name__}")

# Same as the NCCL groups of the routed connector: a stalled peer surfaces
# as an error instead of a hang.
TRANSFER_TIMEOUT_S = 120.0
NIXL_BACKEND = "UCX"


def _agent_name(world_rank: int) -> str:
    return f"afd-routed-{world_rank}"


class NixlRoutedAFDConnector(P2pNcclRoutedAFDConnector):
    """``P2pNcclRoutedAFDConnector`` with the rows carried by NIXL WRITEs."""

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        parallel = vllm_config.parallel_config
        if parallel.use_ubatching:
            raise RuntimeError(
                "NixlRoutedAFDConnector supports one micro-batch only: "
                "disable --enable-dbo and --ubatch-size",
            )
        if int(parallel.tensor_parallel_size) != 1:
            raise RuntimeError("NixlRoutedAFDConnector requires TP=1 on both roles")
        self.agent: nixl_agent | None = None
        # world rank -> NIXL agent name of the peers this rank writes to.
        self._peer_agents: dict[int, str] = {}
        # world rank -> (base address, row bytes) of every receive buffer of
        # the peer, and the peer's device id.
        self._peer_regions: dict[int, list[tuple[int, int]]] = {}
        self._peer_devices: dict[int, int] = {}
        # Attention: send rows (hidden, ids, weights) and the result regions;
        # FFN: receive regions (hidden, ids, weights) and the partial sums.
        self._rows: list[torch.Tensor] = []
        self._results: torch.Tensor | None = None
        self._writes: list[Any] = []
        self._notifs: dict[str, list[bytes]] = {}
        # Attention: stage -> plan of the dispatch whose results are pending.
        self._nixl_pending: dict[int, P2pAll2AllPlan] = {}

    # ------------------------------------------------------------ lifecycle
    def init_afd_connector(self) -> None:
        """Set up the NCCL groups and the engine, then the NIXL regions.

        Collective, like the parent: every Attention and FFN rank must call it.
        """
        if self._initialized:
            return
        super().init_afd_connector()
        spec = self.spec
        if spec.quantize_on_attention:
            raise RuntimeError(
                "NixlRoutedAFDConnector carries unquantized rows only; the FFN "
                f"experts kernel wants {spec.quant_mode} quantized inputs",
            )
        # One region of max_tokens rows per peer: FFN receives from every
        # Attention rank, Attention gets results back from every FFN rank.
        peers = self.attn_size if self.is_ffn else self.ffn_size
        rows = peers * self.max_tokens
        self._rows = [
            torch.empty(
                (rows, spec.hidden_size), dtype=spec.act_dtype, device=self.device
            ),
            torch.empty((rows, spec.top_k), dtype=_IDS_DTYPE, device=self.device),
            torch.empty((rows, spec.top_k), dtype=_WEIGHTS_DTYPE, device=self.device),
        ]
        self._results = torch.empty(
            (rows, spec.hidden_size), dtype=spec.model_dtype, device=self.device
        )
        self.agent = nixl_agent(
            _agent_name(self.world_rank),
            nixl_agent_config(backends=[NIXL_BACKEND]),
        )
        self.agent.register_memory([*self._rows, self._results])

        # Exchange agent metadata and region addresses over the dispatch
        # group. Peers write into FFN's receive rows and Attention's results.
        writable = self._rows if self.is_ffn else [self._results]
        mine = (
            self.agent.get_agent_metadata(),
            [(t.data_ptr(), t[0].numel() * t.element_size()) for t in writable],
            self.device.index,
            self.max_tokens,
        )
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.dispatch_pg)
        if {info[3] for info in everyone} != {self.max_tokens}:
            raise RuntimeError(
                "NixlRoutedAFDConnector needs the same max_num_batched_tokens on "
                f"every rank, got {[info[3] for info in everyone]}",
            )
        peer_ranks = (
            range(self.ffn_size, self.world_size)
            if self.is_ffn
            else range(self.ffn_size)
        )
        for peer in peer_ranks:
            metadata, regions, device, _ = everyone[peer]
            # add_remote_agent returns the name as bytes while notifications
            # are keyed by str, so peers are addressed by the known name.
            self.agent.add_remote_agent(metadata)
            self._peer_agents[peer] = _agent_name(peer)
            self._peer_regions[peer] = regions
            self._peer_devices[peer] = device
        logger.info(
            "NIXL routed AFD connector ready: role=%s world_rank=%d peers=%s "
            "region_rows=%d backend=%s",
            self.afd_config.role,
            self.world_rank,
            list(self._peer_agents),
            self.max_tokens,
            NIXL_BACKEND,
        )

    def close(self) -> None:
        if self.agent is not None:
            self._finish_writes()
        self._nixl_pending.clear()
        super().close()

    # ------------------------------------------------------------- transfer
    def _write(
        self,
        peer: int,
        local: list[torch.Tensor],
        region_row: int,
        num_rows: int,
        notif: bytes,
    ) -> None:
        """WRITE ``local`` into ``peer``'s receive buffers from ``region_row`` on.

        ``local`` holds one ``[num_rows, ...]`` tensor per receive buffer of the
        peer (see the module docstring); ``region_row`` is where this rank's
        region starts in them. The peer receives ``notif`` once all of it has
        landed.
        """
        assert self.agent is not None
        remote = [
            (
                addr + region_row * row_bytes,
                num_rows * row_bytes,
                self._peer_devices[peer],
            )
            for addr, row_bytes in self._peer_regions[peer]
        ]
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(local),
            self.agent.get_xfer_descs(remote, "VRAM"),
            self._peer_agents[peer],
            notif,
        )
        if self.agent.transfer(handle) == "ERR":
            raise RuntimeError(f"NIXL WRITE to world rank {peer} failed to post")
        self._writes.append(handle)

    def _finish_writes(self) -> None:
        """Wait until every posted WRITE is done, so its rows can be reused."""
        assert self.agent is not None
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        for handle in self._writes:
            while (state := self.agent.check_xfer_state(handle)) == "PROC":
                if time.monotonic() > deadline:
                    raise RuntimeError("NIXL WRITE did not complete in time")
            if state != "DONE":
                raise RuntimeError(f"NIXL WRITE ended in state {state}")
            self.agent.release_xfer_handle(handle)
        self._writes.clear()

    def _wait_notifs(self, expected: dict[int, bytes]) -> None:
        """Wait for ``expected[peer]`` from every listed peer (world rank)."""
        assert self.agent is not None
        missing = {self._peer_agents[peer]: notif for peer, notif in expected.items()}
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        while missing:
            for name, notifs in self.agent.get_new_notifs().items():
                self._notifs.setdefault(name, []).extend(notifs)
            for name in [n for n in missing if missing[n] in self._notifs.get(n, [])]:
                self._notifs[name].remove(missing.pop(name))
            if missing and time.monotonic() > deadline:
                raise RuntimeError(f"no NIXL notification from {sorted(missing)}")

    # ------------------------------------------------- attention data path
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Pack this stage's rows by FFN rank and WRITE each FFN its share."""
        if self.engine is None:
            raise RuntimeError("routed connector is not initialized")
        topk_ids: torch.Tensor | None = kwargs.get("topk_ids")
        topk_weights: torch.Tensor | None = kwargs.get("topk_weights")
        if topk_ids is None or topk_weights is None:
            raise ValueError("routed send_attn_output needs topk_ids and topk_weights")
        stage = int(context.metadata.stage_idx)
        if stage in self._nixl_pending:
            raise RuntimeError(f"stage {stage} already has a dispatch in flight")
        # A token goes to an FFN rank at most once, so a peer's region of
        # max_tokens rows holds the share of any batch up to max_tokens.
        if hidden_states.shape[0] > self.max_tokens:
            raise RuntimeError(
                f"{hidden_states.shape[0]} tokens exceed the NIXL regions of "
                f"{self.max_tokens} rows (max_num_batched_tokens)",
            )
        tensors = [
            hidden_states,
            topk_ids.to(_IDS_DTYPE),
            topk_weights.to(_WEIGHTS_DTYPE),
        ]
        plan = self.engine.plan(tensors[1], slot=stage)
        with torch.profiler.record_function("afd_nixl.dispatch_write"):
            # The send rows of the previous layer may still be on the wire.
            self._finish_writes()
            # Destination-major packing, as the NCCL engine's dispatch does.
            for src, packed in zip(tensors, self._rows, strict=True):
                torch.index_select(src, 0, plan.perm, out=packed[: plan.num_send])
            torch.cuda.current_stream(self.device).synchronize()
            offset = 0
            for ffn in range(self.ffn_size):
                count = plan.send_counts[ffn]
                if count:
                    rows = [t[offset : offset + count] for t in self._rows]
                    self._write(
                        ffn,
                        rows,
                        self.role_rank * self.max_tokens,
                        count,
                        f"d{plan.seq}".encode(),
                    )
                offset += count
        self._nixl_pending[stage] = plan

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Wait for the partial sums of this stage and accumulate them."""
        if self.engine is None or self._results is None:
            raise RuntimeError("routed connector is not initialized")
        plan = self._nixl_pending.pop(int(ubatch_idx), None)
        if plan is None:
            raise RuntimeError(f"no dispatch in flight for stage {ubatch_idx}")
        if ref_tensor.shape[0] != plan.num_tokens:
            raise ValueError(
                f"ref_tensor has {ref_tensor.shape[0]} rows, stage {ubatch_idx} "
                f"dispatched {plan.num_tokens}",
            )
        with torch.profiler.record_function("afd_nixl.combine_wait"):
            notif = f"c{plan.seq}".encode()
            self._wait_notifs(
                {ffn: notif for ffn in range(self.ffn_size) if plan.send_counts[ffn]}
            )
        # Gather the result regions back into the packed (``plan.perm``) order.
        back = torch.cat(
            [
                self._results[ffn * self.max_tokens : ffn * self.max_tokens + count]
                for ffn, count in enumerate(plan.send_counts[: self.ffn_size])
            ]
        )
        output = torch.empty_like(ref_tensor)
        self.engine.combine_accumulate(plan, back, output)
        return output

    # ------------------------------------------------------- ffn data path
    def prefetch_attn_output(self, ubatch_idx: int = 0) -> RoutedPrefetch:
        """Run the count exchange, then wait for and gather this stage's rows."""
        if self.engine is None or self._empty_send is None:
            raise RuntimeError("routed connector is not initialized")
        stage = int(ubatch_idx)
        plan = self.engine.plan(self._empty_send[1], slot=stage)
        attn_counts = plan.recv_counts[self.ffn_size :]
        with torch.profiler.record_function("afd_nixl.dispatch_wait"):
            notif = f"d{plan.seq}".encode()
            self._wait_notifs(
                {
                    self.ffn_size + attn: notif
                    for attn, count in enumerate(attn_counts)
                    if count
                }
            )
        # Gather the receive regions into one block grouped by source rank,
        # the order the NCCL engine's dispatch returns.
        received = [
            torch.cat(
                [
                    region[attn * self.max_tokens : attn * self.max_tokens + count]
                    for attn, count in enumerate(attn_counts)
                ]
            )
            for region in self._rows
        ]
        stream = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(stream)
        return RoutedPrefetch(
            stage=stage, plan=plan, received=received, stream=stream, ready=ready
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """WRITE each Attention rank its share of the partial sums."""
        if self._results is None:
            raise RuntimeError("routed connector is not initialized")
        state = context.states
        if not isinstance(state, RoutedTransferState):
            raise RuntimeError("send_ffn_output needs the state of recv_attn_output")
        plan = state.plan
        if ffn_output.shape[0] != plan.num_recv:
            raise ValueError(
                f"ffn_output has {ffn_output.shape[0]} rows, stage "
                f"{plan.slot} received {plan.num_recv}",
            )
        with torch.profiler.record_function("afd_nixl.combine_write"):
            # The partial sums of the previous layer may still be on the wire.
            self._finish_writes()
            self._results[: plan.num_recv].copy_(ffn_output)
            torch.cuda.current_stream(self.device).synchronize()
            offset = 0
            for attn, count in enumerate(plan.recv_counts[self.ffn_size :]):
                if count:
                    self._write(
                        self.ffn_size + attn,
                        [self._results[offset : offset + count]],
                        self.role_rank * self.max_tokens,
                        count,
                        f"c{plan.seq}".encode(),
                    )
                offset += count


__all__ = ["NixlRoutedAFDConnector"]
