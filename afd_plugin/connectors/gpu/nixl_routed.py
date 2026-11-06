# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One-hop routed AFD connector that moves the rows with NIXL.

``NixlRoutedAFDConnector`` keeps the routing of ``P2pNcclRoutedAFDConnector``
(Attention computes the gate and sends each token once to every FFN rank that
owns one of its experts, FFN runs its experts through the ``prerouted``
backend and returns the weighted partial sums) together with its control
plane. The rows travel as NIXL WRITEs followed by a notification, and the
row counts travel in those notifications (no count exchange):

    Attention a -> FFN f: rows into f's receive region of (stage, a).
    FFN f -> Attention a: partial sums into a's result region of (stage, f).

Every rank registers its receive buffers once at init, split into one region
of ``R = max_num_batched_tokens`` rows per (micro-batch stage, peer), so
nothing is registered per layer. The registered buffers are pinned host
memory (host staging, below). With two ranks per role and two stages:

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

Row counts: every dispatch notification is ``d<seq>:<rows>``, and every
Attention rank sends one to every FFN rank for every stage (a bare
notification, without a WRITE, when it has no rows for that FFN), so an FFN
rank learns what each Attention rank sent when the rows arrive and no rank
waits for the others before sending. ``<seq>`` is the plan sequence number,
which advances identically on every rank (one plan per layer and stage). The
partial sums come back with ``c<seq>`` from the FFN ranks the rows went to;
Attention knows how many (the rows it sent).

A region is safe to rewrite without a barrier: an Attention rank sends a
stage's next rows to an FFN rank only after the partial sums of the rows it
sent that FFN before have come back, which the FFN posts after its experts
read the gathered rows; where it sent no rows, the FFN had nothing to read.
The result regions follow the same order the other way round.

Pipeline (the micro-batch stages of a layer overlap as four independent
resources: Attention GPU, Attention->FFN link, FFN GPU, FFN->Attention link,
and every step starts as soon as its input has arrived):

* Attention, per stage: plan (copy its own per-FFN counts to the host), pack
  (and copy into the pinned send buffers), WRITE, then the shared experts;
  ``recv_ffn_output`` hands the CPU to the other micro-batches once
  (``dbo_yield``) and polls for the
  partial sums when it gets it back. No receive has to be posted for a
  WRITE, so a stage's partial sums leave the FFN the moment its experts are
  done, whatever the other stages do (the NCCL parent has to post each
  result receive after the next dispatch).
* FFN, per stage: ``prefetch_attn_output`` only picks the stage's stream;
  ``recv_attn_output`` polls for every Attention rank's notification (it
  carries the row count) and gathers the rows onto the GPU on that stream;
  ``send_ffn_output`` copies the partial sums into the pinned send buffer and
  returns at once.
  The WRITE is posted from the polling loops as soon as that copy (so the
  experts kernel) has finished, so the FFN GPU goes straight from one
  stage's experts to the next while the previous partial sums are on the
  wire. ``flush_ffn_output`` (after the last MoE layer) posts what is left.

Every host wait is a polling loop that also drives the posted WRITEs to
completion, because NIXL sends the notification of a WRITE only when its
state is checked after it finished. The NIXL agent runs NIXL's progress
thread as well (NIXL's default).

Host staging: the NIXL buffers are pinned host memory, and the connector
itself moves the rows between them and the GPU, with asynchronous copies on
its streams that CUDA events order like the kernels around them, so UCX only
sends host memory over TCP and makes no CUDA calls. With the buffers in GPU
memory UCX copies between the GPU and its TCP buffers itself, with a
synchronous CUDA copy inside ``transfer()``; one of those copies was seen
never to return, leaving the FFN stuck in ``transfer()``.

With the parent's fixed-size experts launch (unquantized Triton experts), the
FFN gathers the received rows into per-stage buffers of the regions' size
(Attention ranks x R rows) instead of tensors of the received size, so the
experts kernel always sees one shape; the regions themselves are not handed
over because their rows are not contiguous.

Delta dispatch (``connector_extra_config = {"delta_dispatch": {...}}``, see
``afd_plugin/delta_dispatch.py``): every row has its own format. The hidden
buffers are uint8 rows of the largest format; a stage's rows are packed back
to back in the dispatch order at their own sizes, a fourth buffer carries one
int32 word per row (token, format, Attention rank), and the dispatch
notification carries the stage's byte count after the row count. The
Attention side works out the formats and per-FFN byte counts right after the
plan (before the host waits for it, so no extra synchronization), encodes the
rows into the pack buffer, and adds each FFN rank's returned partial sums to
that rank's states before summing them. The FFN side hands the words to the
FFN model, which rebuilds and normalizes the rows.

Limitations: unquantized rows, Attention TP=1.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from nixl import nixl_agent, nixl_agent_config
from vllm.distributed.device_communicators.p2p_nccl_all2all import (
    P2pAll2AllPlan,
)
from vllm.logger import init_logger
from vllm.v1.worker.ubatching import (
    dbo_switch_to_comm_sync,
    dbo_switch_to_compute,
    dbo_yield,
)

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import ConnectorExtraInfo
from afd_plugin.connectors.gpu.routed import (
    _IDS_DTYPE,
    _WEIGHTS_DTYPE,
    P2pNcclRoutedAFDConnector,
    RoutedPrefetch,
    RoutedTransferState,
)
from afd_plugin.connectors.metadata import AFDA2FTransferPayload, AFDTransferContext
from afd_plugin.delta_dispatch import (
    EXTRA_CONFIG_KEY,
    AttentionDeltaState,
    DeltaDispatchConfig,
    PackedDispatch,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(f"vllm.{__name__}")

# Same as the NCCL groups of the routed connector: a stalled peer surfaces
# as an error instead of a hang.
TRANSFER_TIMEOUT_S = 120.0
NIXL_BACKEND = "UCX"
# NIXL memory type of the buffers: pinned host memory, device id 0 (see "Host
# staging" in the module docstring).
NIXL_MEM_TYPE = "DRAM"
# AFD_NIXL_DEBUG=1: a host wait longer than _DEBUG_WAIT_S logs what it is
# still waiting for every _DEBUG_WAIT_S seconds, and SIGUSR1 dumps every
# thread's stack (no cost while nothing waits that long). =2 also logs every
# count exchange, WRITE and notification of the first _DEBUG_PLANS plans
# (adds host time on the hot path).
_DEBUG_LEVEL = int(os.environ.get("AFD_NIXL_DEBUG", "0") or 0)
_DEBUG = _DEBUG_LEVEL >= 1
_DEBUG_PLANS = 400
_DEBUG_WAIT_S = 5.0


def _agent_name(world_rank: int) -> str:
    return f"afd-routed-{world_rank}"


def _profiling() -> bool:
    """Whether a profiler is recording. A ``record_function`` range costs
    several microseconds of host time even when none is, so the ranges of
    this module are opened only while one records."""
    return torch.autograd.profiler._is_profiler_enabled


def _profile_range(name: str) -> contextlib.AbstractContextManager:
    """``record_function(name)`` while a profiler records, else nothing."""
    if _profiling():
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _mark(name: str) -> None:
    """A point in time in profiler traces: a range of zero length."""
    if _profiling():
        with torch.profiler.record_function(name):
            pass


@dataclass(frozen=True)
class NixlPrefetch:
    """FFN-side stage whose rows are not yet received.

    Returned by ``prefetch_attn_output``; ``recv_attn_output`` waits for the
    rows (and their counts) and turns it into a ``RoutedPrefetch``, gathering
    on ``stream``.
    """

    stage: int
    stream: torch.cuda.Stream


@dataclass(frozen=True)
class PendingCombineWrite:
    """FFN-side partial sums of one stage, staged but not yet WRITTEN.

    ``ready`` is recorded on the compute stream after the staging copy (so
    after the experts kernel); the WRITEs in ``shares`` (peer world rank,
    (address, row bytes) of the first staged row, row count, row of the
    peer's region) are posted once it has fired.
    """

    slot: int
    ready: torch.cuda.Event
    shares: list[tuple[int, tuple[int, int], int, int]]
    notif: bytes


class NixlRoutedAFDConnector(P2pNcclRoutedAFDConnector):
    """``P2pNcclRoutedAFDConnector`` with the rows carried by NIXL WRITEs."""

    # AFD_NIXL_DEBUG: seq of the plan being handled, count exchanges posted
    # (class defaults so connectors built without __init__ in tests have them).
    _dbg_plan = -1
    _dbg_exchanges = 0

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        # Delta dispatch is the one extra setting of the NIXL routed connector;
        # anything else is rejected by the parent as before.
        rest = dict(raw or {})
        DeltaDispatchConfig.from_mapping(rest.pop(EXTRA_CONFIG_KEY, None))
        return super().parse_extra_config(rest)

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
        # the peer.
        self._peer_regions: dict[int, list[tuple[int, int]]] = {}
        # Attention: send rows (hidden, ids, weights) and the result regions;
        # FFN: receive regions (hidden, ids, weights) and the partial sums.
        # All of them hold num_slots blocks, one per stage, and are the
        # buffers registered with NIXL.
        self._rows: list[torch.Tensor] = []
        self._results: torch.Tensor | None = None
        # Attention: the GPU buffers a stage's rows are packed into before
        # they are copied into the pinned send buffers.
        self._pack: list[torch.Tensor] = []
        # slot -> WRITE handles posted and not yet seen complete.
        self._writes: dict[int, list[Any]] = {}
        self._notifs: dict[str, list[bytes]] = {}
        # Attention: stage -> plan of the dispatch whose results are pending.
        self._nixl_pending: dict[int, P2pAll2AllPlan] = {}
        # Delta dispatch. Attention: stage -> (states, packed dispatch, rows
        # sent) until the results are back. FFN: stage -> received words.
        self._delta_pending: dict[
            int, tuple[AttentionDeltaState, PackedDispatch, int]
        ] = {}
        self._delta_meta: dict[int, torch.Tensor] = {}
        # FFN: partial sums staged and waiting for their experts kernel.
        self._pending_combines: list[PendingCombineWrite] = []
        # FFN: slot -> event after the last experts kernel that read the
        # slot's gather buffers (the next gather into them waits for it).
        self._slot_consumed: dict[int, torch.cuda.Event] = {}
        # FFN, fixed-size experts launch: per slot, the buffers the received
        # rows are gathered into (empty when the launch is off).
        self._fixed_received: list[list[torch.Tensor]] = []

    # ---------------------------------------------------------------- debug
    def _dbg(self, fmt: str, *args: Any) -> None:
        """Log while the first _DEBUG_PLANS plans are handled (AFD_NIXL_DEBUG=2)."""
        if _DEBUG_LEVEL >= 2 and self._dbg_plan < _DEBUG_PLANS:
            self._dbg_always(fmt, *args)

    def _dbg_always(self, fmt: str, *args: Any) -> None:
        if _DEBUG:
            logger.info(
                "nixl-debug[%s r%d plan%d] %s",
                self.afd_config.role,
                self.world_rank,
                self._dbg_plan,
                fmt % args,
            )

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
        # Pinned host memory (plain host memory when the connector runs on
        # CPU tensors in tests); see "Host staging" in the module docstring.
        pin = self.device.type == "cuda"

        def buffer(width: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty((rows, width), dtype=dtype, device="cpu", pin_memory=pin)

        delta = self.delta_dispatch
        if (
            delta.enabled
            and self.is_ffn
            and getattr(getattr(self._ffn_model, "model", None), "ffn_delta", None)
            is None
        ):
            raise RuntimeError("delta dispatch is not wired into the FFN model")
        self._rows = [
            buffer(delta.max_row_bytes(spec.hidden_size), torch.uint8)
            if delta.enabled
            else buffer(spec.hidden_size, spec.act_dtype),
            buffer(spec.top_k, _IDS_DTYPE),
            buffer(spec.top_k, _WEIGHTS_DTYPE),
            # Delta dispatch: one word per row (token, format, Attention rank).
            *([buffer(1, torch.int32)] if delta.enabled else []),
        ]
        self._results = buffer(spec.hidden_size, spec.model_dtype)
        self._pack = (
            [
                torch.empty(t.shape, dtype=t.dtype, device=self.device)
                for t in self._rows
            ]
            if not self.is_ffn
            else []
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
            nixl_agent_config(backends=[NIXL_BACKEND], enable_prog_thread=True),
        )
        self.agent.register_memory([*self._rows, self._results])

        # Exchange agent metadata and region addresses over the dispatch
        # group. Peers write into FFN's receive rows and Attention's results.
        writable = self._rows if self.is_ffn else [self._results]
        layout = (self.max_tokens, self.num_slots)
        mine = (
            self.agent.get_agent_metadata(),
            [(t.data_ptr(), t[0].numel() * t.element_size()) for t in writable],
            layout,
        )
        everyone: list[Any] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.dispatch_pg)
        if {info[2] for info in everyone} != {layout}:
            raise RuntimeError(
                "NixlRoutedAFDConnector needs the same max_num_batched_tokens "
                "and micro-batch count on every rank, got "
                f"{[info[2] for info in everyone]}",
            )
        peer_ranks = (
            range(self.ffn_size, self.world_size)
            if self.is_ffn
            else range(self.ffn_size)
        )
        for peer in peer_ranks:
            metadata, regions, _ = everyone[peer]
            # add_remote_agent returns the name as bytes while notifications
            # are keyed by str, so peers are addressed by the known name.
            self.agent.add_remote_agent(metadata)
            self._peer_agents[peer] = _agent_name(peer)
            self._peer_regions[peer] = regions
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
        if _DEBUG:
            # kill -USR1 <pid>: every thread's stack, the process keeps running.
            faulthandler.register(signal.SIGUSR1, all_threads=True)

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
        local: list[tuple[int, int]],
        region_row: int,
        num_rows: int,
        notif: bytes,
    ) -> None:
        """WRITE ``num_rows`` rows from every ``(address, row bytes)`` of
        ``local`` into ``peer``'s receive buffers from ``region_row`` on.

        ``local`` holds the first row to send of one registered buffer per
        receive buffer of the peer (see the module docstring); ``region_row``
        is where this rank's region of the stage starts in them. The peer
        receives ``notif`` once all of it has landed and this side has seen
        the WRITE complete. The descriptors are built from addresses, which
        costs less host time than slicing the tensors for NIXL to read back.
        """
        assert self.agent is not None
        src = [(addr, num_rows * row_bytes, 0) for addr, row_bytes in local]
        dst = [
            (addr + region_row * row_bytes, num_rows * row_bytes, 0)
            for addr, row_bytes in self._peer_regions[peer]
        ]
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(src, NIXL_MEM_TYPE),
            self.agent.get_xfer_descs(dst, NIXL_MEM_TYPE),
            self._peer_agents[peer],
            notif,
        )
        state = self.agent.transfer(handle)
        _mark(f"afd_nixl.write_post[{slot}:{notif.decode()}:{peer}]")
        if state == "ERR":
            raise RuntimeError(f"NIXL WRITE to world rank {peer} failed to post")
        self._writes.setdefault(slot, []).append(handle)
        self._dbg(
            "WRITE posted slot=%d peer=%d rows=%d region_row=%d notif=%s -> %s",
            slot,
            peer,
            num_rows,
            region_row,
            notif,
            state,
        )

    def _write_spans(
        self,
        slot: int,
        peer: int,
        spans: list[tuple[int, int]],
        region_row: int,
        notif: bytes,
    ) -> None:
        """WRITE every ``(address, bytes)`` of ``spans`` into the matching
        receive buffer of ``peer`` at the start of this rank's region
        (``region_row`` rows into the peer's buffer); like ``_write`` for
        delta dispatch rows, whose sizes differ per row."""
        assert self.agent is not None
        src = [(addr, nbytes, 0) for addr, nbytes in spans]
        dst = [
            (addr + region_row * row_bytes, nbytes, 0)
            for (addr, row_bytes), (_, nbytes) in zip(
                self._peer_regions[peer], spans, strict=True
            )
        ]
        handle = self.agent.initialize_xfer(
            "WRITE",
            self.agent.get_xfer_descs(src, NIXL_MEM_TYPE),
            self.agent.get_xfer_descs(dst, NIXL_MEM_TYPE),
            self._peer_agents[peer],
            notif,
        )
        state = self.agent.transfer(handle)
        _mark(f"afd_nixl.write_post[{slot}:{notif.decode()}:{peer}]")
        if state == "ERR":
            raise RuntimeError(f"NIXL WRITE to world rank {peer} failed to post")
        self._writes.setdefault(slot, []).append(handle)
        self._dbg(
            "WRITE posted slot=%d peer=%d bytes=%s region_row=%d notif=%s -> %s",
            slot,
            peer,
            [nbytes for _, nbytes in spans],
            region_row,
            notif,
            state,
        )

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
                    _mark(f"afd_nixl.write_done[{slot}]")
                else:
                    raise RuntimeError(f"NIXL WRITE ended in state {state}")
            if remaining:
                self._writes[slot] = remaining
            else:
                del self._writes[slot]

    def _wait_until(self, done: Callable[[], bool], what: str) -> None:
        """Poll ``done`` (servicing the WRITEs meanwhile) until it holds."""
        start = time.monotonic()
        deadline = start + TRANSFER_TIMEOUT_S
        report = start + _DEBUG_WAIT_S
        while True:
            self._service_writes()
            if done():
                if _DEBUG and time.monotonic() - start > _DEBUG_WAIT_S:
                    self._dbg_always(
                        "%s done after %.1f s", what, time.monotonic() - start
                    )
                return
            if _DEBUG and time.monotonic() > report:
                report += _DEBUG_WAIT_S
                self._dbg_always(
                    "still waiting %s for %.1f s: writes=%s pending_combines=%s "
                    "unconsumed_notifs=%s",
                    what,
                    time.monotonic() - start,
                    {
                        slot: [self.agent.check_xfer_state(h) for h in hs]
                        for slot, hs in self._writes.items()
                    }
                    if self.agent is not None
                    else {},
                    [
                        (p.slot, p.notif, p.ready.query())
                        for p in self._pending_combines
                    ],
                    {k: v for k, v in self._notifs.items() if v},
                )
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
                self._dbg("notifs from %s: %s", name, notifs)
                for notif in notifs:
                    _mark(f"afd_nixl.notif_in[{notif.decode()}:{name}]")
                self._notifs.setdefault(name, []).extend(notifs)
            for name in [n for n in missing if missing[n] in self._notifs.get(n, [])]:
                self._notifs[name].remove(missing.pop(name))
            return not missing

        self._wait_until(done, f"notification from {sorted(missing)}")

    def _wait_counts(
        self, prefix: bytes, peers: list[int]
    ) -> dict[int, tuple[int, int | None]]:
        """Wait for the notification ``<prefix><rows>[:<row bytes>]`` from every
        world rank of ``peers`` and return the row count and, for delta
        dispatch rows, the row size each one announced."""
        assert self.agent is not None
        agent = self.agent
        missing = {self._peer_agents[peer]: peer for peer in peers}
        counts: dict[int, tuple[int, int | None]] = {}

        def done() -> bool:
            for name, notifs in agent.get_new_notifs().items():
                self._dbg("notifs from %s: %s", name, notifs)
                for notif in notifs:
                    _mark(f"afd_nixl.notif_in[{notif.decode()}:{name}]")
                self._notifs.setdefault(name, []).extend(notifs)
            for name in list(missing):
                notifs = self._notifs.get(name, [])
                for notif in notifs:
                    if notif.startswith(prefix):
                        notifs.remove(notif)
                        count, _, size = notif[len(prefix) :].partition(b":")
                        counts[missing.pop(name)] = (
                            int(count),
                            int(size) if size else None,
                        )
                        break
            return not missing

        self._wait_until(done, f"notification {prefix!r} from {sorted(missing)}")
        return counts

    def _gather_rows(self, parts: list[torch.Tensor], out: torch.Tensor) -> None:
        """Copy ``parts`` (regions of the pinned NIXL buffers) one after
        another into ``out`` on the current stream, asynchronously like a
        kernel."""
        offset = 0
        for part in parts:
            out[offset : offset + part.shape[0]].copy_(part, non_blocking=True)
            offset += part.shape[0]

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
        if self.delta_dispatch.enabled:
            self._send_attn_delta(hidden_states, stage, topk_ids, topk_weights, kwargs)
            return
        tensors = [
            hidden_states,
            topk_ids.to(_IDS_DTYPE),
            topk_weights.to(_WEIGHTS_DTYPE),
        ]
        engine = self.engine
        # Under micro-batching the plan and the packing go on the
        # communication stream (after this stage's attention), so the other
        # stages' attention kernels on the compute stream are not queued
        # behind them. Without it: no-ops.
        dbo_switch_to_comm_sync()
        stream = torch.cuda.current_stream(self.device)
        # The plan is the only dependency of the dispatch on this stage's
        # attention; the host waits for its kernels here (polling, so the
        # WRITEs of the other stages keep completing), copies its own per-FFN
        # counts and never hands the CPU away before the rows are on the
        # wire. The FFN ranks learn the counts from the notifications.
        pending = self._begin_plan(tensors[1], stage, context.metadata.layer_idx)
        planned = torch.cuda.Event()
        planned.record(stream)
        with _profile_range(f"afd_nixl.plan_wait[{stage}]"):
            self._wait_event(planned, "plan")
        plan = engine.finish_plan_local(pending)
        self._dbg_plan = plan.seq
        self._dbg(
            "A send stage=%d seq=%d tokens=%d send=%s",
            stage,
            plan.seq,
            plan.num_tokens,
            plan.send_counts,
        )
        # This stage's block of the send buffers: a token is packed once per
        # FFN rank it visits, so a stage packs up to ffn_size * max_tokens
        # rows. Its previous WRITEs were completed in recv_ffn_output before
        # the stage came round again (and the copy out of its pack buffers
        # before those WRITEs were posted).
        block_rows = self.ffn_size * self.max_tokens
        block = slice(stage * block_rows, (stage + 1) * block_rows)
        send_rows = [t[block] for t in self._rows]
        pack_rows = [t[block] for t in self._pack]
        with _profile_range(f"afd_nixl.dispatch_write[{stage}]"):
            # Destination-major packing, as the NCCL engine's dispatch does,
            # into the GPU pack buffers, then copied into the pinned send
            # buffers on the same stream.
            n = plan.num_send
            for src, packed, sent in zip(tensors, pack_rows, send_rows, strict=True):
                torch.index_select(src, 0, plan.perm, out=packed[:n])
                sent[:n].copy_(packed[:n], non_blocking=True)
            packed_done = torch.cuda.Event()
            packed_done.record(stream)
            self._wait_event(packed_done, "packing")
            # (address, row bytes) of the first row of the stage's block.
            sources = [
                (t.data_ptr(), t.stride(0) * t.element_size()) for t in send_rows
            ]
            offset = 0
            for ffn in range(self.ffn_size):
                count = plan.send_counts[ffn]
                # Every FFN rank hears from every Attention rank for every
                # stage; the notification carries the row count.
                notif = f"d{plan.seq}:{count}".encode()
                if count:
                    self._write(
                        stage,
                        ffn,
                        [(addr + offset * rb, rb) for addr, rb in sources],
                        self._region_row(stage, self.attn_size),
                        count,
                        notif,
                    )
                else:
                    assert self.agent is not None
                    self.agent.send_notif(self._peer_agents[ffn], notif)
                    _mark(f"afd_nixl.write_post[{stage}:{notif.decode()}:{ffn}]")
                offset += count
        # Back to the compute stream: the shared experts run while the rows
        # are on the wire.
        dbo_switch_to_compute()
        self._nixl_pending[stage] = plan

    def _send_attn_delta(
        self,
        hidden_states: torch.Tensor,
        stage: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> None:
        """Delta dispatch variant of ``send_attn_output``: same plan, packing
        streams and WRITEs, with the hidden rows encoded per row format."""
        assert self.engine is not None
        delta = self.delta_dispatch
        state: AttentionDeltaState | None = kwargs.get("delta")
        layer_idx = kwargs.get("layer_idx")
        src = hidden_states if delta.sends_x else kwargs.get("h_prime")
        if state is None or layer_idx is None or src is None:
            raise RuntimeError(
                "delta dispatch needs the model's states, the layer and h' "
                "(RoutedExpertsProxy passes them)",
            )
        ids = topk_ids.to(_IDS_DTYPE)
        weights = topk_weights.to(_WEIGHTS_DTYPE)
        engine = self.engine
        dbo_switch_to_comm_sync()
        stream = torch.cuda.current_stream(self.device)
        pending = self._begin_plan(ids, stage, int(layer_idx))
        # Formats, byte offsets and per-FFN byte counts on the same stream,
        # before the event the host waits for anyway: the counts are in host
        # memory once it fires (no extra synchronization).
        packed = state.prepare(int(layer_idx), stage, pending.order, pending.num_tokens)
        planned = torch.cuda.Event()
        planned.record(stream)
        with _profile_range(f"afd_nixl.plan_wait[{stage}]"):
            self._wait_event(planned, "plan")
        plan = engine.finish_plan_local(pending)
        totals = packed.totals.tolist()
        self._dbg_plan = plan.seq
        self._dbg(
            "A send stage=%d seq=%d tokens=%d send=%s bytes=%s",
            stage,
            plan.seq,
            plan.num_tokens,
            plan.send_counts,
            totals,
        )
        block_rows = self.ffn_size * self.max_tokens
        block = slice(stage * block_rows, (stage + 1) * block_rows)
        send_rows = [t[block] for t in self._rows]
        pack_rows = [t[block] for t in self._pack]
        send_bytes, pack_bytes = send_rows[0].view(-1), pack_rows[0].view(-1)
        n, total = plan.num_send, sum(totals)
        with _profile_range(f"afd_nixl.dispatch_write[{stage}]"):
            # Rows (and their words) packed destination-major in one kernel
            # into the GPU pack buffers, the routing columns as in the plain
            # path, then everything copied into the pinned send buffers.
            state.encode(
                packed, src, n, pack_bytes, pack_rows[3].view(-1), self.role_rank
            )
            send_bytes[:total].copy_(pack_bytes[:total], non_blocking=True)
            for i, column in ((1, ids), (2, weights)):
                torch.index_select(column, 0, plan.perm, out=pack_rows[i][:n])
                send_rows[i][:n].copy_(pack_rows[i][:n], non_blocking=True)
            send_rows[3][:n].copy_(pack_rows[3][:n], non_blocking=True)
            packed_done = torch.cuda.Event()
            packed_done.record(stream)
            self._wait_event(packed_done, "packing")
            row_offset = byte_offset = 0
            for ffn in range(self.ffn_size):
                count, nbytes = plan.send_counts[ffn], totals[ffn]
                notif = f"d{plan.seq}:{count}:{nbytes}".encode()
                if count:
                    spans = [(send_bytes.data_ptr() + byte_offset, nbytes)] + [
                        (
                            t.data_ptr() + row_offset * t.stride(0) * t.element_size(),
                            count * t.stride(0) * t.element_size(),
                        )
                        for t in send_rows[1:]
                    ]
                    self._write_spans(
                        stage,
                        ffn,
                        spans,
                        self._region_row(stage, self.attn_size),
                        notif,
                    )
                else:
                    assert self.agent is not None
                    self.agent.send_notif(self._peer_agents[ffn], notif)
                    _mark(f"afd_nixl.write_post[{stage}:{notif.decode()}:{ffn}]")
                row_offset += count
                byte_offset += nbytes
        dbo_switch_to_compute()
        self._nixl_pending[stage] = plan
        self._delta_pending[stage] = (state, packed, n)

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
        with _profile_range(f"afd_nixl.dispatch_finish[{stage}]"):
            self._finish_writes(stage)
        self._dbg("A wait combine stage=%d seq=%d", stage, plan.seq)
        with _profile_range(f"afd_nixl.combine_wait[{stage}]"):
            notif = f"c{plan.seq}".encode()
            self._wait_notifs(
                {ffn: notif for ffn in range(self.ffn_size) if plan.send_counts[ffn]}
            )
        # Gather the result regions of the stage back into the packed
        # (``plan.perm``) order, on the GPU.
        rows_per_region = self.max_tokens
        counts = plan.send_counts[: self.ffn_size]
        back = torch.empty(
            (sum(counts), self._results.shape[1]),
            dtype=self._results.dtype,
            device=self.device,
        )
        self._gather_rows(
            [
                self._results[(stage * self.ffn_size + ffn) * rows_per_region :][:count]
                for ffn, count in enumerate(counts)
            ],
            back,
        )
        if self.delta_dispatch.enabled:
            # Delta dispatch: each FFN rank's partial sums into its states,
            # before they are summed.
            state, packed, sent = self._delta_pending.pop(stage)
            state.after_combine(packed, sent, back)
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
        """Pick the stage's stream and return without waiting; the counts
        come with the rows, which ``recv_attn_output`` waits for."""
        if self.engine is None:
            raise RuntimeError("routed connector is not initialized")
        stage = int(ubatch_idx)
        if stage >= self.num_slots:
            raise RuntimeError(
                f"stage {stage} but the connector has {self.num_slots} micro-batch "
                "slots (--enable-dbo / --ubatch-size)",
            )
        self._dbg_exchanges += 1
        self._dbg("F prefetch stage=%d #%d", stage, self._dbg_exchanges)
        self._service_writes()
        return NixlPrefetch(stage=stage, stream=self._stage_stream(stage))

    def _complete_prefetch(self, prefetched: NixlPrefetch) -> RoutedPrefetch:
        """Wait for every Attention rank's rows of the stage (their counts come
        with the notifications) and gather them on the stage stream."""
        assert self.engine is not None
        stage, stream = prefetched.stage, prefetched.stream
        # Plans are numbered identically on every rank (one per layer and
        # stage), so this stage's notifications carry this number.
        seq = self.engine.next_plan_seq()
        self._dbg_plan = seq
        with _profile_range(f"afd_nixl.dispatch_wait[{stage}]"):
            counts = self._wait_counts(
                f"d{seq}:".encode(),
                [self.ffn_size + attn for attn in range(self.attn_size)],
            )
        attn_counts = [
            counts[self.ffn_size + attn][0] for attn in range(self.attn_size)
        ]
        # Delta dispatch: the byte count of every Attention rank's rows.
        attn_bytes = [counts[self.ffn_size + attn][1] for attn in range(self.attn_size)]
        delta = self.delta_dispatch.enabled
        if any((nbytes is None) == delta for nbytes in attn_bytes):
            raise RuntimeError(
                f"dispatch byte counts {attn_bytes} with delta dispatch "
                f"{self.delta_dispatch.mode}",
            )
        plan = self.engine.recv_plan(stage, seq, [0] * self.ffn_size + attn_counts)
        self._dbg("F recv stage=%d seq=%d counts=%s", stage, plan.seq, attn_counts)
        # Gather the stage's receive regions into one block on the GPU grouped
        # by source rank, the order the NCCL engine's dispatch returns. With the
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
                if i == 0 and delta:
                    # Each Attention rank's rows are packed at the head of its
                    # region, at their own sizes.
                    total = sum(int(nbytes or 0) for nbytes in attn_bytes)
                    parts = [
                        region[(stage * self.attn_size + attn) * rows_per_region :][
                            :rows_per_region
                        ].view(-1)[: int(nbytes or 0)]
                        for attn, nbytes in enumerate(attn_bytes)
                    ]
                    gather = (
                        self._fixed_received[stage][i].view(-1)
                        if fixed
                        else torch.empty(total, dtype=region.dtype, device=self.device)
                    )
                    self._gather_rows(parts, gather[:total])
                    received.append(gather[:total])
                    continue
                parts = [
                    region[(stage * self.attn_size + attn) * rows_per_region :][:count]
                    for attn, count in enumerate(attn_counts)
                ]
                if fixed:
                    gather = self._fixed_received[stage][i]
                else:
                    gather = torch.empty(
                        (plan.num_recv, *region.shape[1:]),
                        dtype=region.dtype,
                        device=self.device,
                    )
                self._gather_rows(parts, gather[: plan.num_recv])
                received.append(gather)
                if i == 3 and delta:
                    self._delta_meta[stage] = gather[: plan.num_recv].view(-1)
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
        payload = super().recv_attn_output(
            ubatch_idx=stage, prefetched=prefetched, **kwargs
        )
        if self.delta_dispatch.enabled:
            # The words of the rows; the FFN model rebuilds the rows from them.
            assert self._ffn_model is not None
            self._ffn_model.model.ffn_delta.receive(stage, self._delta_meta.pop(stage))
        return payload

    def _post_combine(self, pending: PendingCombineWrite) -> None:
        self._dbg("F post combine slot=%d notif=%s", pending.slot, pending.notif)
        with _profile_range(f"afd_nixl.combine_post[{pending.slot}]"):
            for peer, rows, num_rows, region_row in pending.shares:
                self._write(
                    pending.slot, peer, [rows], region_row, num_rows, pending.notif
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
        with _profile_range(f"afd_nixl.combine_stage[{slot}]"):
            # Under host staging into pinned host memory: asynchronous,
            # ``ready`` fires once it has landed.
            staged.copy_(ffn_output, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        # The experts kernel that read the slot's gather buffers precedes
        # this copy on the compute stream.
        self._slot_consumed[slot] = ready
        shares: list[tuple[int, tuple[int, int], int, int]] = []
        addr, row_bytes = staged.data_ptr(), staged.stride(0) * staged.element_size()
        offset = 0
        for attn, count in enumerate(plan.recv_counts[self.ffn_size :]):
            if count:
                shares.append(
                    (
                        self.ffn_size + attn,
                        (addr + offset * row_bytes, row_bytes),
                        count,
                        self._region_row(slot, self.ffn_size),
                    )
                )
            offset += count
        self._pending_combines.append(
            PendingCombineWrite(
                slot=slot, ready=ready, shares=shares, notif=f"c{plan.seq}".encode()
            )
        )
        self._dbg(
            "F staged combine slot=%d seq=%d rows=%d", slot, plan.seq, plan.num_recv
        )
        self._service_writes()

    def flush_ffn_output(self) -> None:
        """Post every staged partial sum and complete every WRITE.

        Called after the last MoE layer of a forward (nothing polls after
        it) and at close.
        """
        if self.agent is None:
            return
        with _profile_range("afd_nixl.combine_flush"):
            self._flush_combines()


__all__ = ["NixlPrefetch", "NixlRoutedAFDConnector", "PendingCombineWrite"]
