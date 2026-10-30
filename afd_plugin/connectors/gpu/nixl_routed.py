# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One-hop routed AFD connector that moves the rows with NIXL.

``NixlRoutedAFDConnector`` keeps the routing of ``P2pNcclRoutedAFDConnector``
(Attention computes the gate and sends each token once to every FFN rank that
owns one of its experts, FFN runs its experts through the ``prerouted``
backend and returns the weighted partial sums) together with its NCCL groups
for the per-layer count exchange and the control plane. Only the rows travel
differently, as NIXL WRITEs followed by a notification:

    Attention a -> FFN f: rows into f's receive region of (stage, a).
    FFN f -> Attention a: partial sums into a's result region of (stage, f).

Every rank registers its receive buffers once at init, split into one region
of ``R = max_num_batched_tokens`` rows per (micro-batch stage, peer), so
nothing is registered per layer. With two ranks per role and two stages:

    FFN f (written by Attention)              Attention a (written by FFN)
                  hidden  ids  weights                      results
    rows 0..R-1    <- stage 0, Attention 0    rows 0..R-1    <- stage 0, FFN 0
    rows R..2R-1   <- stage 0, Attention 1    rows R..2R-1   <- stage 0, FFN 1
    rows 2R..3R-1  <- stage 1, Attention 0    rows 2R..3R-1  <- stage 1, FFN 0
    rows 3R..4R-1  <- stage 1, Attention 1    rows 3R..4R-1  <- stage 1, FFN 1

Attention 1 sending 300 rows of stage 0 to FFN 0 writes rows R..R+299 of FFN
0's hidden, ids and weights buffers in one WRITE with three descriptors. A
region holds the share of any stage because a token goes to an FFN rank at
most once and a stage never has more than R tokens.

Both ends of a pair know its row count from the count exchange: a pair without
rows sends no WRITE and waits for no notification. A notification carries the
plan sequence number, which advances identically on every rank because every
rank takes part in every count exchange, in stage order. That exchange is also
what makes a region safe to rewrite: a rank only reaches the next layer's
exchange of a stage after it has consumed that stage's rows.

Pipeline (the micro-batch stages of a layer overlap as four independent
resources: Attention GPU, Attention->FFN link, FFN GPU, FFN->Attention link,
and every step starts as soon as its input has arrived):

* Attention, per stage: count exchange, pack, WRITE, then the shared experts;
  ``recv_ffn_output`` hands the CPU to the other micro-batches once
  (``dbo_yield``) and polls for the partial sums when it gets it back. No
  receive has to be posted for a WRITE, so a stage's partial sums leave the
  FFN the moment its experts are done, whatever the other stages do (the NCCL
  parent has to post each result receive after the next dispatch).
* FFN, per stage: ``prefetch_attn_output`` only posts the count exchange and
  returns; ``recv_attn_output`` reads the counts, polls for the rows and
  gathers them on the stage's stream; ``send_ffn_output`` stages the partial
  sums and returns at once. The WRITE is posted from the polling loops as
  soon as the experts kernel has finished, so the FFN GPU goes straight from
  one stage's experts to the next while the previous partial sums are on the
  wire. ``flush_ffn_output`` (after the last MoE layer) posts what is left.

Every host wait is a polling loop that also drives the posted WRITEs to
completion, because NIXL sends the notification of a WRITE only when its
state is checked after it finished.

With the parent's fixed-size experts launch (unquantized Triton experts), the
FFN gathers the received rows into per-stage buffers of the regions' size
(Attention ranks x R rows) instead of tensors of the received size, so the
experts kernel always sees one shape; the regions themselves are not handed
over because their rows are not contiguous.

Limitations: unquantized rows, Attention TP=1.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from nixl import nixl_agent, nixl_agent_config
from vllm.distributed.device_communicators.p2p_nccl_all2all import (
    P2pAll2AllPendingPlan,
    P2pAll2AllPlan,
)
from vllm.logger import init_logger
from vllm.v1.worker.ubatching import (
    dbo_switch_to_comm_sync,
    dbo_switch_to_compute,
    dbo_yield,
)

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.gpu.routed import (
    _IDS_DTYPE,
    _WEIGHTS_DTYPE,
    P2pNcclRoutedAFDConnector,
    RoutedPrefetch,
    RoutedTransferState,
)
from afd_plugin.connectors.metadata import AFDA2FTransferPayload, AFDTransferContext

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(f"vllm.{__name__}")

# Same as the NCCL groups of the routed connector: a stalled peer surfaces
# as an error instead of a hang.
TRANSFER_TIMEOUT_S = 120.0
NIXL_BACKEND = "UCX"


def _agent_name(world_rank: int) -> str:
    return f"afd-routed-{world_rank}"


@dataclass(frozen=True)
class NixlPrefetch:
    """FFN-side count exchange of one stage, posted but not yet read.

    Returned by ``prefetch_attn_output``; ``recv_attn_output`` reads the
    counts, waits for the rows and turns it into a ``RoutedPrefetch``.
    ``exchanged`` is recorded on ``stream`` right after the exchange.
    """

    stage: int
    pending: P2pAll2AllPendingPlan
    stream: torch.cuda.Stream
    exchanged: torch.cuda.Event


@dataclass(frozen=True)
class PendingCombineWrite:
    """FFN-side partial sums of one stage, staged but not yet WRITTEN.

    ``ready`` is recorded on the compute stream after the staging copy (so
    after the experts kernel); the WRITEs in ``shares`` (peer world rank,
    rows, row of the peer's region) are posted once it has fired.
    """

    slot: int
    ready: torch.cuda.Event
    shares: list[tuple[int, torch.Tensor, int]]
    notif: bytes


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
        if int(parallel.tensor_parallel_size) != 1:
            raise RuntimeError("NixlRoutedAFDConnector requires TP=1 on both roles")
        # Micro-batch stages a forward can have; every stage owns its regions.
        self.num_slots = max(1, int(parallel.num_ubatches))
        self.agent: nixl_agent | None = None
        # world rank -> NIXL agent name of the peers this rank writes to.
        self._peer_agents: dict[int, str] = {}
        # world rank -> (base address, row bytes) of every receive buffer of
        # the peer, and the peer's device id.
        self._peer_regions: dict[int, list[tuple[int, int]]] = {}
        self._peer_devices: dict[int, int] = {}
        # Attention: send rows (hidden, ids, weights) and the result regions;
        # FFN: receive regions (hidden, ids, weights) and the partial sums.
        # All of them hold num_slots blocks, one per stage.
        self._rows: list[torch.Tensor] = []
        self._results: torch.Tensor | None = None
        # slot -> WRITE handles posted and not yet seen complete.
        self._writes: dict[int, list[Any]] = {}
        self._notifs: dict[str, list[bytes]] = {}
        # Attention: stage -> plan of the dispatch whose results are pending.
        self._nixl_pending: dict[int, P2pAll2AllPlan] = {}
        # FFN: partial sums staged and waiting for their experts kernel.
        self._pending_combines: list[PendingCombineWrite] = []
        # FFN: slot -> event after the last experts kernel that read the
        # slot's gather buffers (the next gather into them waits for it).
        self._slot_consumed: dict[int, torch.cuda.Event] = {}
        # FFN, fixed-size experts launch: per slot, the buffers the received
        # rows are gathered into (empty when the launch is off).
        self._fixed_received: list[list[torch.Tensor]] = []

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
        # One region of max_tokens rows per (stage, peer): FFN receives from
        # every Attention rank, Attention gets results back from every FFN
        # rank. The send buffers have the same size: Attention packs a
        # stage's tokens once per FFN rank they visit (up to ffn_size *
        # max_tokens rows per stage), FFN returns at most the rows of all
        # Attention ranks per stage.
        peers = self.attn_size if self.is_ffn else self.ffn_size
        rows = self.num_slots * peers * self.max_tokens
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
        if self.is_ffn and self._fixed_ok:
            # One set per stage: recv_attn_output masks the rows past the
            # received ones, and mask_rows_past tracks which rows still need
            # it per (slot, buffer address).
            self._fixed_received = [
                [
                    torch.empty(
                        (self._fixed_rows, *region.shape[1:]),
                        dtype=region.dtype,
                        device=self.device,
                    )
                    for region in self._rows
                ]
                for _ in range(self.num_slots)
            ]
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
            self.device.index or 0,
            (self.max_tokens, self.num_slots),
        )
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.dispatch_pg)
        if {info[3] for info in everyone} != {(self.max_tokens, self.num_slots)}:
            raise RuntimeError(
                "NixlRoutedAFDConnector needs the same max_num_batched_tokens and "
                "micro-batch count on every rank, got "
                f"{[info[3] for info in everyone]}",
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
            "region_rows=%d slots=%d backend=%s",
            self.afd_config.role,
            self.world_rank,
            list(self._peer_agents),
            self.max_tokens,
            self.num_slots,
            NIXL_BACKEND,
        )

    def close(self) -> None:
        if self.agent is not None:
            self.flush_ffn_output()
        self._nixl_pending.clear()
        self._slot_consumed.clear()
        super().close()

    def reset_step_state(self) -> None:
        super().reset_step_state()
        if self._nixl_pending:
            logger.warning(
                "NIXL routed AFD connector: %d dispatches of the previous step "
                "were never collected",
                len(self._nixl_pending),
            )
            self._nixl_pending.clear()

    # ------------------------------------------------------------- transfer
    def _region_row(self, stage: int, peers: int) -> int:
        """First row of this rank's region of ``stage`` in a peer's buffers.

        ``peers`` is how many ranks write into the peer (the receiver's view:
        Attention ranks for an FFN buffer, FFN ranks for an Attention one).
        """
        return (stage * peers + self.role_rank) * self.max_tokens

    def _write(
        self,
        slot: int,
        peer: int,
        local: list[torch.Tensor],
        region_row: int,
        num_rows: int,
        notif: bytes,
    ) -> None:
        """WRITE ``local`` into ``peer``'s receive buffers from ``region_row`` on.

        ``local`` holds one ``[num_rows, ...]`` tensor per receive buffer of the
        peer (see the module docstring); ``region_row`` is where this rank's
        region of the stage starts in them. The peer receives ``notif`` once
        all of it has landed and this side has seen the WRITE complete.
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
        # GPU memory in serving, host memory when the connector runs on CPU
        # tensors (tests).
        mem_type = "VRAM" if self.device.type == "cuda" else "DRAM"
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(local),
            self.agent.get_xfer_descs(remote, mem_type),
            self._peer_agents[peer],
            notif,
        )
        if self.agent.transfer(handle) == "ERR":
            raise RuntimeError(f"NIXL WRITE to world rank {peer} failed to post")
        self._writes.setdefault(slot, []).append(handle)

    def _service_writes(self) -> None:
        """Make progress without blocking: post the partial sums whose experts
        kernel has finished and release the WRITEs that have completed.

        NIXL sends the notification of a WRITE that was still in flight when
        it was posted only once its state is checked, so every host wait
        calls this in its loop and the peers see the rows promptly.
        """
        assert self.agent is not None
        # The staging copies are in stage order on one compute stream, so
        # the first unfinished one blocks the later ones as well.
        while self._pending_combines and self._pending_combines[0].ready.query():
            self._post_combine(self._pending_combines.pop(0))
        for slot, handles in list(self._writes.items()):
            remaining: list[Any] = []
            for handle in handles:
                state = self.agent.check_xfer_state(handle)
                if state == "PROC":
                    remaining.append(handle)
                elif state == "DONE":
                    self.agent.release_xfer_handle(handle)
                else:
                    raise RuntimeError(f"NIXL WRITE ended in state {state}")
            if remaining:
                self._writes[slot] = remaining
            else:
                del self._writes[slot]

    def _wait_until(self, done: Callable[[], bool], what: str) -> None:
        """Poll ``done`` (servicing the WRITEs meanwhile) until it holds."""
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        while True:
            self._service_writes()
            if done():
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"NIXL routed connector: {what} did not finish in time"
                )

    def _wait_event(self, event: torch.cuda.Event, what: str) -> None:
        self._wait_until(event.query, what)

    def _finish_writes(self, slot: int | None = None) -> None:
        """Wait until the WRITEs of ``slot`` (or of every slot) have completed."""

        def done() -> bool:
            if slot is None:
                return not self._writes
            return slot not in self._writes

        self._wait_until(done, "WRITE")

    def _wait_notifs(self, expected: dict[int, bytes]) -> None:
        """Wait for ``expected[peer]`` from every listed peer (world rank)."""
        assert self.agent is not None
        agent = self.agent
        missing = {self._peer_agents[peer]: notif for peer, notif in expected.items()}

        def done() -> bool:
            for name, notifs in agent.get_new_notifs().items():
                self._notifs.setdefault(name, []).extend(notifs)
            for name in [n for n in missing if missing[n] in self._notifs.get(n, [])]:
                self._notifs[name].remove(missing.pop(name))
            return not missing

        self._wait_until(done, f"notification from {sorted(missing)}")

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
        if stage >= self.num_slots:
            raise RuntimeError(
                f"stage {stage} but the connector has {self.num_slots} micro-batch "
                "slots (--enable-dbo / --ubatch-size)",
            )
        # A token goes to an FFN rank at most once, so a peer's region of
        # max_tokens rows holds the share of any stage up to max_tokens.
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
        engine = self.engine
        # Under micro-batching the count exchange and the packing go on the
        # communication stream (after this stage's attention), so the other
        # stages' attention kernels on the compute stream are not queued
        # behind the exchange's wait for the peers. Without it: no-ops.
        dbo_switch_to_comm_sync()
        stream = torch.cuda.current_stream(self.device)
        # The count exchange is the only dependency of the dispatch on this
        # stage's attention; the host waits for it here (polling, so the
        # WRITEs of the other stages keep completing) and never hands the
        # CPU away before the rows are on the wire.
        pending = engine.begin_plan(tensors[1], slot=stage)
        engine.exchange_counts(pending)
        exchanged = torch.cuda.Event()
        exchanged.record(stream)
        self._wait_event(exchanged, "count exchange")
        plan = engine.finish_plan(pending)
        # This stage's block of the send buffers: a token is packed once per
        # FFN rank it visits, so a stage packs up to ffn_size * max_tokens
        # rows. Its previous WRITEs were completed in recv_ffn_output before
        # the stage came round again.
        block_rows = self.ffn_size * self.max_tokens
        block = slice(stage * block_rows, (stage + 1) * block_rows)
        send_rows = [t[block] for t in self._rows]
        with torch.profiler.record_function("afd_nixl.dispatch_write"):
            # Destination-major packing, as the NCCL engine's dispatch does.
            for src, packed in zip(tensors, send_rows, strict=True):
                torch.index_select(src, 0, plan.perm, out=packed[: plan.num_send])
            packed_done = torch.cuda.Event()
            packed_done.record(stream)
            self._wait_event(packed_done, "packing")
            offset = 0
            for ffn in range(self.ffn_size):
                count = plan.send_counts[ffn]
                if count:
                    self._write(
                        stage,
                        ffn,
                        [t[offset : offset + count] for t in send_rows],
                        self._region_row(stage, self.attn_size),
                        count,
                        f"d{plan.seq}".encode(),
                    )
                offset += count
        # Back to the compute stream: the shared experts run while the rows
        # are on the wire.
        dbo_switch_to_compute()
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
        # Let the other micro-batches compute and dispatch while this stage's
        # rows travel and its experts run; the shared experts of this stage
        # are already enqueued. Without micro-batches this is a no-op.
        dbo_yield()
        stage = int(ubatch_idx)
        plan = self._nixl_pending.pop(stage, None)
        if plan is None:
            raise RuntimeError(f"no dispatch in flight for stage {stage}")
        if ref_tensor.shape[0] != plan.num_tokens:
            raise ValueError(
                f"ref_tensor has {ref_tensor.shape[0]} rows, stage {stage} "
                f"dispatched {plan.num_tokens}",
            )
        with torch.profiler.record_function("afd_nixl.dispatch_finish"):
            self._finish_writes(stage)
        with torch.profiler.record_function("afd_nixl.combine_wait"):
            notif = f"c{plan.seq}".encode()
            self._wait_notifs(
                {ffn: notif for ffn in range(self.ffn_size) if plan.send_counts[ffn]}
            )
        # Gather the result regions of the stage back into the packed
        # (``plan.perm``) order.
        rows_per_region = self.max_tokens
        back = torch.cat(
            [
                self._results[(stage * self.ffn_size + ffn) * rows_per_region :][:count]
                for ffn, count in enumerate(plan.send_counts[: self.ffn_size])
            ]
        )
        output = torch.empty_like(ref_tensor)
        if plan.perm.is_cuda:
            # ``perm`` was allocated on the communication stream; the plan
            # may be released before the compute stream reads it below.
            plan.perm.record_stream(torch.cuda.current_stream(self.device))
        self.engine.combine_accumulate(plan, back, output)
        return output

    # ------------------------------------------------------- ffn data path
    def prefetch_attn_output(  # type: ignore[override]
        self, ubatch_idx: int = 0
    ) -> NixlPrefetch:
        """Post the count exchange of this stage and return without waiting.

        On the stage's own stream, chained after the previous exchange so
        every rank issues them in the same order; ``recv_attn_output`` reads
        the counts and waits for the rows.
        """
        if self.engine is None or self._empty_send is None:
            raise RuntimeError("routed connector is not initialized")
        stage = int(ubatch_idx)
        if stage >= self.num_slots:
            raise RuntimeError(
                f"stage {stage} but the connector has {self.num_slots} micro-batch "
                "slots (--enable-dbo / --ubatch-size)",
            )
        stream = self._stage_stream(stage)
        engine = self.engine
        with torch.cuda.stream(stream):
            if self._dispatch_chain is not None:
                stream.wait_event(self._dispatch_chain)
            pending = engine.begin_plan(self._empty_send[1], slot=stage)
            engine.exchange_counts(pending)
            exchanged = torch.cuda.Event()
            exchanged.record(stream)
        self._dispatch_chain = exchanged
        self._service_writes()
        return NixlPrefetch(
            stage=stage, pending=pending, stream=stream, exchanged=exchanged
        )

    def _complete_prefetch(self, prefetched: NixlPrefetch) -> RoutedPrefetch:
        """Read the counts, wait for the rows and gather them on the stage stream."""
        assert self.engine is not None
        stage, stream = prefetched.stage, prefetched.stream
        self._wait_event(prefetched.exchanged, "count exchange")
        with torch.cuda.stream(stream):
            plan = self.engine.finish_plan(prefetched.pending)
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
        # Gather the stage's receive regions into one block grouped by source
        # rank, the order the NCCL engine's dispatch returns. With the
        # fixed-size launch the block is the head of the stage's fixed-size
        # buffer and the whole buffer goes on (recv_attn_output masks the
        # rest); the regions together are that size, so the block always
        # fits. The buffer is free once the experts of its previous use are
        # done (``_slot_consumed``).
        fixed = bool(self._fixed_received)
        rows_per_region = self.max_tokens
        received: list[torch.Tensor] = []
        with torch.cuda.stream(stream):
            consumed = self._slot_consumed.get(stage)
            if consumed is not None:
                stream.wait_event(consumed)
            for i, region in enumerate(self._rows):
                parts = [
                    region[(stage * self.attn_size + attn) * rows_per_region :][:count]
                    for attn, count in enumerate(attn_counts)
                ]
                if fixed:
                    gather = self._fixed_received[stage][i]
                    torch.cat(parts, out=gather[: plan.num_recv])
                    received.append(gather)
                else:
                    received.append(torch.cat(parts))
            ready = torch.cuda.Event()
            ready.record(stream)
        return RoutedPrefetch(
            stage=stage,
            plan=plan,
            received=received,
            stream=stream,
            ready=ready,
            fixed=fixed,
        )

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        prefetched: NixlPrefetch | RoutedPrefetch | None = None,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        stage = int(ubatch_idx)
        if prefetched is None:
            prefetched = self.prefetch_attn_output(stage)
        if isinstance(prefetched, NixlPrefetch):
            if prefetched.stage != stage:
                raise RuntimeError(
                    f"prefetched stage {prefetched.stage} consumed as stage {stage}",
                )
            prefetched = self._complete_prefetch(prefetched)
        return super().recv_attn_output(
            ubatch_idx=stage, prefetched=prefetched, **kwargs
        )

    def _post_combine(self, pending: PendingCombineWrite) -> None:
        for peer, rows, region_row in pending.shares:
            self._write(
                pending.slot, peer, [rows], region_row, rows.shape[0], pending.notif
            )

    def _flush_combines(self, slot: int | None = None) -> None:
        """Post the staged partial sums of ``slot`` (of every slot when None)
        and those staged before them, waiting for their experts kernels, then
        complete the slot's WRITEs."""
        targets = [
            pending
            for pending in self._pending_combines
            if slot is None or pending.slot == slot
        ]
        if targets:
            last = targets[-1]
            while True:
                head = self._pending_combines.pop(0)
                self._wait_event(head.ready, "experts kernel")
                self._post_combine(head)
                if head is last:
                    break
        self._finish_writes(slot)

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Stage this stage's partial sums; they are WRITTEN once computed."""
        if self._results is None:
            raise RuntimeError("routed connector is not initialized")
        state = context.states
        if not isinstance(state, RoutedTransferState):
            raise RuntimeError("send_ffn_output needs the state of recv_attn_output")
        plan = state.plan
        if ffn_output.shape[0] > plan.num_recv:
            # Fixed-size launch: only the first num_recv rows are real.
            ffn_output = ffn_output[: plan.num_recv]
        if ffn_output.shape[0] != plan.num_recv:
            raise ValueError(
                f"ffn_output has {ffn_output.shape[0]} rows, stage "
                f"{plan.slot} received {plan.num_recv}",
            )
        slot = plan.slot
        # The slot's previous partial sums must have left the send buffer
        # before the staging copy is enqueued (long done: Attention has
        # consumed them before it sent this stage's rows).
        self._flush_combines(slot)
        base = slot * self.attn_size * self.max_tokens
        staged = self._results[base : base + plan.num_recv]
        with torch.profiler.record_function("afd_nixl.combine_stage"):
            staged.copy_(ffn_output)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        # The experts kernel that read the slot's gather buffers precedes
        # this copy on the compute stream.
        self._slot_consumed[slot] = ready
        shares: list[tuple[int, torch.Tensor, int]] = []
        offset = 0
        for attn, count in enumerate(plan.recv_counts[self.ffn_size :]):
            if count:
                shares.append(
                    (
                        self.ffn_size + attn,
                        staged[offset : offset + count],
                        self._region_row(slot, self.ffn_size),
                    )
                )
            offset += count
        self._pending_combines.append(
            PendingCombineWrite(
                slot=slot, ready=ready, shares=shares, notif=f"c{plan.seq}".encode()
            )
        )
        self._service_writes()

    def flush_ffn_output(self) -> None:
        """Post every staged partial sum and complete every WRITE.

        Called after the last MoE layer of a forward (nothing polls after
        it) and at close.
        """
        if self.agent is None:
            return
        with torch.profiler.record_function("afd_nixl.combine_flush"):
            self._flush_combines()


__all__ = ["NixlPrefetch", "NixlRoutedAFDConnector", "PendingCombineWrite"]
