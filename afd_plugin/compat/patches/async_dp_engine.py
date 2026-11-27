# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP engine scheduling.

This module patches:
1. ``vllm.v1.engine.core.EngineCoreProc.run_engine_core``
2. ``vllm.v1.engine.utils.launch_core_engines``
3. ``vllm.v1.engine.utils.DPCoordinator`` construction during launch
4. ``vllm.v1.engine.coordinator.DPCoordinatorProc.run_coordinator``
5. ``vllm.v1.engine.coordinator.DPCoordinatorProc.process_input_socket``
6. ``vllm.v1.engine.core_client.DPAsyncMPClient.add_request_async``

Why:
    vLLM 0.26.0's native MoE DP path uses ``DPEngineCoreProc`` and DP wave
    notifications. AFD async-DP Attention ranks are connector-driven and must
    step independently while keeping the original DP/EP topology for expert
    placement and weight loading.

How:
    AFD async configs are selected by plugin-owned
    ``additional_config["afd"]["async"]``. Attention-side MoE DP engine
    processes instantiate ``EngineCoreProc`` instead of ``DPEngineCoreProc``;
    coordinator stats remain enabled, but wave coordination and client
    ``FIRST_REQ`` wakeups are disabled.

Future plan:
    Remove the coordinator portion when the pinned vLLM includes upstream
    #49204's non-lockstep stats handling. Remove the remaining patch when vLLM
    exposes an external async-DP scheduling hook selectable by plugin config.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import vllm.v1.engine.coordinator as coordinator_module
import vllm.v1.engine.core as engine_core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils_module
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPAsyncMPClient

from afd_plugin.compat.vllm import (
    PINNED_DP_COORDINATOR_VLLM_VERSION,
    TARGET_VLLM_VERSION,
)
from afd_plugin.config import is_afd_async_dp, parse_optional_afd_config

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from vllm.config import ParallelConfig, VllmConfig
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.utils import (
        CoreEngineLaunch,
        EngineZmqAddresses,
        SignalCallback,
    )
    from vllm.v1.executor import Executor


# Patch reason: vLLM's MoE DP engine process uses DPEngineCoreProc, but AFD
# async Attention ranks are connector-driven and must not run DP wave logic.
# Patch functionality: keep upstream startup flow while selecting EngineCoreProc
# for AFD async Attention configs.
# Signature: matches upstream; no added parameters.
def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """Launch EngineCore busy loop in background process."""
    vllm_config: VllmConfig = kwargs["vllm_config"]
    if logging_config := getattr(vllm_config, "logging_config", None):
        engine_core_module.configure_logging(logging_config)

    # Ensure we can serialize transformer config after spawning
    engine_core_module.maybe_register_config_serialize_by_value()

    engine_core: EngineCoreProc | None = None
    signal_callback: SignalCallback | None = None
    clean_shutdown = False
    try:
        parallel_config: ParallelConfig = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        engine_core_module.set_process_title(process_title)
        engine_core_module.maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
        engine_core_module.decorate_logs()
        if parallel_config.numa_bind:
            engine_core_module.numa_utils.log_current_affinity_state(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
            )
            engine_core_module.logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        # declare engine core as None and register signal handler before
        # engine_core initialization in case engine core process is terminated
        # in the middle of initialization. For example, frontend process exits
        # unexpectedly, if signal handler is not registered, engine core process
        # will use default signal handler which will terminate the process
        # silently. If engine core model executor has been initialized, it will
        # not be terminated asap. Register signal handler early and only log a
        # message will make engine core process alive. engine core process and
        # its subprocesses(model executor processes) will be force killed in
        # CoreEngineProcManager.shutdown.
        engine_core = None
        # A termination signal can arrive while engine_core is still being
        # constructed. At that point the request cannot be recorded on
        # engine_core, so keep it in its own flag and apply it later.
        shutdown_requested = False

        def wakeup_engine():
            # Wakes up idle engine via input_queue when shutdown is requested
            # Not safe in a signal handler - we may interrupt the main thread
            # while it is holding the non-reentrant input_queue.mutex
            assert engine_core is not None
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            signal_name = engine_core_module.signal.Signals(signum).name
            engine_core_module.logger.info(
                "[shutdown] EngineCore: trigger received signal=%s",
                signal_name,
            )
            if signum == engine_core_module.signal.SIGTERM:
                engine_core_module.get_watch_dog().handle_failure(signal_name)
            # Record the request first so it survives the construction window.
            shutdown_requested = True
            if engine_core is not None:
                engine_core.shutdown_state = engine_core_module.EngineShutdownState.REQUESTED
            # signal_callback is armed only after engine_core exists. Don't
            # trigger it earlier: it is one-shot, and a wakeup cannot be
            # delivered while there is no input_queue to wake.
            # For more info: https://github.com/vllm-project/vllm/pull/52299#pullrequestreview-5370255271
            if signal_callback is not None:
                signal_callback.trigger()

        engine_core_module.signal.signal(engine_core_module.signal.SIGTERM, signal_handler)
        engine_core_module.signal.signal(engine_core_module.signal.SIGINT, signal_handler)

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            # ### PATCH START: AFD async-DP Attention engine selection
            # Async-DP Attention ranks are connector-driven, so use the regular
            # EngineCoreProc instead of DPEngineCoreProc while keeping the
            # original DP rank metadata.
            if _is_afd_async_attention_config(vllm_config):
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
            else:
                engine_core = engine_core_module.DPEngineCoreProc(*args, **kwargs)
            # ### PATCH END: AFD async-DP Attention engine selection
        else:
            # Non-MoE DP ranks are completely independent, so treat like DP=1.
            # Note that parallel_config.data_parallel_index will still reflect
            # the original DP rank.
            parallel_config.reconfigure_for_independent_dp_rank()
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

        assert engine_core is not None

        # Apply a signal received during construction before entering the
        # loop. run_busy_loop calls _handle_shutdown first, so the loop exits
        # right away instead of blocking on an idle input_queue.
        if shutdown_requested:
            engine_core.shutdown_state = engine_core_module.EngineShutdownState.REQUESTED

        # Arm the wakeup callback only now that input_queue exists, so a
        # later signal can still wake an idle busy loop.
        signal_callback = engine_core_module.SignalCallback(wakeup_engine)

        engine_core_module.start_watch_dog(f"engine_{dp_rank}", vllm_config.watchdog_config, engine_core_module.logger)
        engine_core.run_busy_loop()

    except SystemExit as e:
        engine_core_module.logger.info_once("[shutdown] EngineCore: exiting busy loop")
        clean_shutdown = (
            e.code in (None, 0)
            and engine_core is not None
            and engine_core.shutdown_state == engine_core_module.EngineShutdownState.SHUTTING_DOWN
            and not engine_core.has_work()
            and engine_core.vllm_config.shutdown_timeout == 0
        )
        raise
    except Exception as e:
        if engine_core is None:
            engine_core_module.logger.exception("EngineCore failed to start.")
        else:
            engine_core_module.logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        engine_core_module.signal.signal(engine_core_module.signal.SIGTERM, engine_core_module.signal.SIG_DFL)
        engine_core_module.signal.signal(engine_core_module.signal.SIGINT, engine_core_module.signal.SIG_DFL)
        if signal_callback is not None:
            signal_callback.stop()
        if engine_core is not None:
            engine_core.shutdown()
        if clean_shutdown:
            from vllm.platforms import current_platform

            if current_platform.is_rocm():
                # Cleanup above already unfreezes and collects the heap.
                # Freeze the surviving graph to skip another slow cyclic-GC
                # scan during finalization; process exit reclaims it.
                engine_core_module.gc.freeze()


# Patch reason: vLLM enables MoE DP wave coordination when launching DP cores,
# while AFD async-DP only needs coordinator stats.
# Patch functionality: preserve upstream engine launch behavior but disable wave
# coordination for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
@contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Iterator[CoreEngineLaunch]:
    """Launch engine and DP coordinator processes as needed."""
    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # Create a single tensor IPC queue for sharing multimodal tensors between
    # API servers and engine core. Returns a single queue since we only support
    # DP=1 for this data flow.
    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = engine_utils_module.get_mp_context().Queue()

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    coord_store = None

    create_coord_store = (
        parallel_config.enable_elastic_ep
        and parallel_config.data_parallel_backend != "ray"
        and dp_rank == 0
        and not parallel_config._coord_store_port
    )

    if create_coord_store:
        from vllm.distributed.utils import create_tcp_store

        coord_store = create_tcp_store(
            host,
            0,
            is_master=True,
            world_size=-1,
            wait_for_workers=False,
        )
        parallel_config._coord_store_port = coord_store.port
        engine_utils_module.logger.info(
            "Created Elastic EP coordination TCPStore on %s:%d",
            host,
            coord_store.port,
        )

    if run_coordinator:
        coordinator = engine_utils_module.DPCoordinator(
            parallel_config,
            # ### PATCH START: AFD async-DP coordinator wave mode
            # Keep DP coordinator stats, but disable wave coordination for
            # connector-driven async-DP.
            enable_wave_coordination=(
                vllm_config.model_config.is_moe and not is_afd_async_dp(vllm_config)
            ),
            # ### PATCH END: AFD async-DP coordinator wave mode
            logging_config=vllm_config.logging_config,
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        engine_utils_module.logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
        if coord_store is not None:
            coordinator._coord_store = coord_store
    else:
        coordinator = None

    # Hold a coordination TCPStore (alive for this frame) so engines pick DP
    # master ports at bind time; pre-allocated ports can be taken before use.
    if (
        dp_size > 1
        and not offline_mode
        and dp_rank == 0
        and local_engine_count > 0
        and not parallel_config.enable_elastic_ep
    ):
        from vllm.distributed.utils import create_tcp_store

        coord_store = create_tcp_store(
            host, 0, is_master=True, world_size=-1, wait_for_workers=False
        )
        parallel_config._coord_store_port = coord_store.port

    if parallel_config.data_parallel_backend == "ray":
        engine_utils_module.logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = engine_utils_module.CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_utils_module.CoreEngineLaunch(
            engine_actor_manager, coordinator, addresses, tensor_queue
        )
        return

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [engine_utils_module.CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # Rank > 0 handshakes with just the local cores it is managing.
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    handshake_address = engine_utils_module.get_engine_client_zmq_addr(
        handshake_local_only,
        host,
        parallel_config.data_parallel_rpc_port,
    )

    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = engine_utils_module.get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with engine_utils_module.zmq_socket_ctx(
        local_handshake_address, engine_utils_module.zmq.ROUTER, bind=True
    ) as handshake_socket:
        # Start local engines.
        if local_engine_count:
            local_engine_manager = engine_utils_module.CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        if (
            coord_store is not None
            and coordinator is None
            and (local_engine_manager is not None)
        ):
            local_engine_manager._coord_store = coord_store

        launch = engine_utils_module.CoreEngineLaunch(
            local_engine_manager, coordinator, addresses, tensor_queue
        )
        yield launch
        engine_utils_module.wait_for_engine_startup(
            handshake_socket,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            launch,
        )


# Patch reason: multiprocessing spawn imports the coordinator target in a fresh
# process, where parent-only class assignments are not retained.
# Patch functionality: use a plugin-owned process target that constructs the
# pinned coordinator and invokes AFD's patched socket loop after child import.
# Signature: matches upstream; no added parameters.
def run_coordinator(
    engine_count: int,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
    zmq_addr_pipe=None,
    min_stats_update_interval_ms: int = 100,
    enable_wave_coordination: bool = True,
):
    # ### PATCH START: AFD copied-coordinator module bindings
    DPCoordinatorProc = coordinator_module.DPCoordinatorProc
    logger = coordinator_module.logger
    # ### PATCH END: AFD copied-coordinator module bindings

    coordinator = DPCoordinatorProc(
        engine_count=engine_count,
        min_stats_update_interval_ms=min_stats_update_interval_ms,
        enable_wave_coordination=enable_wave_coordination,
    )
    try:
        # ### PATCH START: AFD spawn-safe coordinator loop
        process_input_socket(
            coordinator,
            front_publish_address,
            back_output_address,
            back_publish_address,
            zmq_addr_pipe,
        )
        # ### PATCH END: AFD spawn-safe coordinator loop
    except KeyboardInterrupt:
        logger.info("DP Coordinator process exiting")
    finally:
        if zmq_addr_pipe is not None:
            zmq_addr_pipe.close()


# Patch reason: vLLM 0.26.0 groups request-count updates by globally synchronized
# DP steps, but AFD async Attention engines advance independently.
# Patch functionality: preserve upstream coordinator transport and its 100 ms
# frontend publication cadence while skipping lockstep snapshot/warning behavior
# whenever wave coordination is disabled.
# Signature: matches upstream; no added parameters.
def process_input_socket(
    self,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
    zmq_addr_pipe=None,
):
    # ### PATCH START: AFD copied-coordinator module bindings
    time = coordinator_module.time
    msgspec = coordinator_module.msgspec
    zmq = coordinator_module.zmq
    logger = coordinator_module.logger
    make_zmq_socket = coordinator_module.make_zmq_socket
    MsgpackDecoder = coordinator_module.MsgpackDecoder
    EngineCoreOutputs = engine_core_module.EngineCoreOutputs
    EngineState = coordinator_module.EngineState
    # ### PATCH END: AFD copied-coordinator module bindings

    decoder = MsgpackDecoder(EngineCoreOutputs)

    # For tracking request wave progression.
    current_wave = 0
    engines_running = False

    # For tracking request counts for internal load-balancing.
    stats_changed = False
    last_stats_step = -1
    last_stats_wave = -1
    last_step_counts: list[list[int]] | None = None

    with (
        make_zmq_socket(
            path=front_publish_address,  # IPC
            ctx=self.ctx,
            socket_type=zmq.XPUB,
            bind=True,
        ) as publish_front,
        make_zmq_socket(
            path=back_output_address,  # IPC or TCP
            ctx=self.ctx,
            socket_type=zmq.PULL,
            bind=True,
        ) as output_back,
        make_zmq_socket(
            path=back_publish_address,  # IPC or TCP
            ctx=self.ctx,
            socket_type=zmq.XPUB,
            bind=True,
        ) as publish_back,
    ):
        if zmq_addr_pipe is not None:
            try:
                zmq_addr_pipe.send(
                    (
                        publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                    )
                )
            finally:
                zmq_addr_pipe.close()
        # Wait until all engines subscribe.
        for _ in self.engines:
            if publish_back.recv() != b"\x01":
                logger.error(
                    "DP Coordinator received unexpected message while "
                    "waiting for engines to subscribe"
                )
                return
        # Send ready message to engines.
        publish_back.send(b"READY")

        logger.info("All engine subscriptions received by DP coordinator")

        poller = zmq.Poller()
        poller.register(publish_front, zmq.POLLIN)
        poller.register(publish_back, zmq.POLLIN)
        poller.register(output_back, zmq.POLLIN)
        last_publish_time = 0
        while True:
            elapsed = int(time.time() * 1000) - last_publish_time
            # Send at stats_update_interval_ms interval if the stats have
            # changed, or otherwise every 5 seconds.
            wait_for = self.stats_update_interval_ms if stats_changed else 5000

            # ### PATCH START: AFD independent request-count cadence
            # Only lockstep DP engines have a shared step whose rank-local
            # updates need a short aggregation window.
            min_timeout = (
                50 if self.enable_wave_coordination and last_step_counts is None else 0
            )
            # ### PATCH END: AFD independent request-count cadence

            events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
            if not events:
                # Poller timeout - publish current stats to front-ends.
                if last_step_counts is not None:
                    engine_req_counts_list = last_step_counts
                    last_step_counts = None
                else:
                    engine_req_counts_list = self._get_engine_counts()
                    stats_changed = False

                to_publish = (engine_req_counts_list, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(to_publish))
                last_publish_time = int(time.time() * 1000)
                continue

            events = dict(events)
            wave_state_changed = False

            if publish_back in events:
                buffer = publish_back.recv()
                if buffer == b"\x01":
                    # NOTE(yongji): newly started engine subscribed
                    # We need to send READY message here instead of receiving
                    # SCALE_ELASTIC_EP notification from engine core client
                    # as SCALE_ELASTIC_EP is only sent when
                    # new engines finished initialization.
                    # Subscription message, on the other hand, is sent
                    # by each engine during initialization
                    publish_back.send(b"READY")
                elif buffer != b"\x00":
                    logger.error(
                        "DP Coordinator received unexpected message from engines"
                    )

            if publish_front in events:
                buffer = publish_front.recv()
                if buffer in (b"\x01", b"\x00"):
                    # Ignore subscription messages.
                    continue

                decoded = msgspec.msgpack.decode(buffer)
                if (
                    isinstance(decoded, (list, tuple))
                    and len(decoded) == 2
                    and decoded[0] == "SCALE_ELASTIC_EP"
                ):
                    # Handle scale up notification
                    new_engine_count = decoded[1]
                    current_count = len(self.engines)
                    if new_engine_count > current_count:
                        for _ in range(new_engine_count - current_count):
                            self.engines.append(EngineState())
                        # NOTE(yongji): handle the case
                        # where newly started engines have current_wave = 0
                        # if existing engines just finished a wave
                        # and engine_running isn't updated yet at
                        # CoordinatorProc requests routed to newly started
                        # engines may not wake up existing engines, as long
                        # as 0 < request.wave < existing engines'
                        # current_wave
                        # we note that 0 is the wave number for the new
                        # engine
                        logger.info(
                            "DPCoordinator scaled up from %s to %s engines",
                            current_count,
                            new_engine_count,
                        )
                    else:
                        self.engines = self.engines[:new_engine_count]
                        logger.info(
                            "DPCoordinator scaled down from %s to %s engines",
                            current_count,
                            new_engine_count,
                        )
                    continue  # Skip normal engine notification processing

                # Wave coordination: handle new-request messages from front-end.
                # Only process these when wave coordination is enabled
                if self.enable_wave_coordination:
                    # We received a message on the front-end XPUB socket,
                    # from an API server sending a new request while the
                    # engines are paused, so that we can wake the other
                    # engines.
                    engine_to_exclude, wave = decoded
                    if not engines_running:
                        if wave < current_wave:
                            # If the wave number is stale, ensure the message
                            # is handled by all the engines.
                            engine_to_exclude = None

                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(
                            publish_back, current_wave, engine_to_exclude
                        )

            if output_back in events:
                # We received a message from one of the engines.

                buffer = output_back.recv()
                outputs: EngineCoreOutputs = decoder.decode(buffer)

                assert not outputs.outputs
                assert outputs.utility_output is None

                eng_index = outputs.engine_index
                scheduler_stats = outputs.scheduler_stats
                if scheduler_stats:
                    # 1. Updated request load stats - update our local
                    # state with these.
                    stats = self.engines[eng_index].request_counts
                    # ### PATCH START: AFD independent request-count updates
                    # Step/wave ordering is meaningful only for lockstep DP.
                    # Async Attention ranks update their latest counts directly.
                    if self.enable_wave_coordination:
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order update "
                                "(wave=%d step=%d) from engine %d (expected > (wave=%d step=%d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                    # ### PATCH END: AFD independent request-count updates
                    stats[0] = scheduler_stats.num_waiting_reqs
                    stats[1] = scheduler_stats.num_running_reqs
                    stats_changed = True

                # Wave coordination: handle wave completion and start notifications
                # Only process these when wave coordination is enabled
                if self.enable_wave_coordination:
                    if (wave := outputs.wave_complete) is not None:
                        # 2. Notification from rank 0 engine that we've
                        # moved into the global paused state
                        # (engines_running==False).
                        if current_wave <= wave:
                            new_wave = wave + 1
                            logger.debug(
                                "Moving DP wave from %d to %d.",
                                current_wave,
                                new_wave,
                            )
                            current_wave = new_wave
                            engines_running = False
                            wave_state_changed = True
                    elif (wave := outputs.start_wave) is not None and (
                        wave > current_wave
                        or (wave == current_wave and not engines_running)
                    ):
                        # 3. The engine received request for a non-current wave
                        # so we must ensure that other engines progress to the
                        # next wave (race condition handling).
                        logger.debug(
                            "Starting wave %d after notification of "
                            "stale wave request from engine.",
                            wave,
                        )
                        current_wave = wave
                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

            if wave_state_changed:
                message = (None, current_wave, engines_running)
                publish_front.send(msgspec.msgpack.encode(message))


# Patch reason: vLLM sends FIRST_REQ wakeups to coordinate DP waves, which AFD
# async-DP engines intentionally do not use.
# Patch functionality: preserve request routing and stats updates while skipping
# FIRST_REQ for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
async def add_request_async(self, request: EngineCoreRequest) -> None:
    self._ensure_stats_update_task()

    request.current_wave = self.current_wave
    request.client_index = self.client_index

    # The engine that receives it wakes the other ranks if they are idle.
    chosen_engine = self.get_core_engine_for_request(request)
    request = core_client_module._with_packed_prompt(request)
    await self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)

    self._ensure_output_queue_task()


def _is_afd_async_attention_config(vllm_config: VllmConfig) -> bool:
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    return (
        afd_config is not None
        and is_afd_async_dp(vllm_config)
        and afd_config.role == "attention"
    )


def _should_patch_pinned_dp_coordinator() -> bool:
    """Return whether the copied coordinator exactly matches the installed vLLM."""

    try:
        import vllm

        version_text = str(vllm.__version__)
    except (AttributeError, ImportError):
        return False
    return version_text == PINNED_DP_COORDINATOR_VLLM_VERSION


def _is_target_vllm_compatible() -> bool:
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    version_text = str(version_value)
    if "dev" in version_text:
        return True
    return version_text.startswith(TARGET_VLLM_VERSION)


def apply_async_dp_engine_patch() -> bool:
    """Install the async-DP patches against the current vLLM bindings.

    vLLM-Ascend installs platform patches after general plugins and also wraps
    ``EngineCoreProc.run_engine_core``. Re-applying this installer after the
    final AFD Attention config is built ensures the process target captured by
    vLLM uses AFD's async-DP entry point. The caller scopes that late install to
    AFD async Attention, so non-AFD and FFN Ascend scheduling remain untouched.
    """

    if not _is_target_vllm_compatible():
        return False

    EngineCoreProc.run_engine_core = staticmethod(run_engine_core)
    engine_utils_module.launch_core_engines = launch_core_engines
    core_client_module.launch_core_engines = launch_core_engines
    # The copied coordinator is exact-release-only. Newer development versions
    # retain their native non-lockstep implementation and payload fields.
    if _should_patch_pinned_dp_coordinator():
        coordinator_module.DPCoordinatorProc.run_coordinator = staticmethod(
            run_coordinator
        )
        coordinator_module.DPCoordinatorProc.process_input_socket = process_input_socket
    DPAsyncMPClient.add_request_async = add_request_async
    engine_core_module.logger.debug("AFD async-DP engine patch applied")
    return True


apply_async_dp_engine_patch()


__all__ = ["apply_async_dp_engine_patch"]
