# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned CUDA profiler helpers for AFD GPU runners."""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

logger = logging.getLogger(__name__)

AFDGPUProfilerRole = Literal["attention", "ffn"]

_ENV_PREFIX: Final[dict[AFDGPUProfilerRole, str]] = {
    "attention": "AFD_GPU_ATTENTION_PROFILER",
    "ffn": "AFD_GPU_FFN_PROFILER",
}
_DEFAULT_DIR: Final[dict[AFDGPUProfilerRole, str]] = {
    "attention": "./profiler_logs/attn",
    "ffn": "./profiler_logs/ffn",
}
_DEFAULT_WAIT_STEPS: Final[int] = 2500
_DEFAULT_WARMUP_STEPS: Final[int] = 1
_DEFAULT_ACTIVE_STEPS: Final[int] = 10
_DEFAULT_REPEAT: Final[int] = 1
_DEFAULT_SKIP_FIRST_STEPS: Final[int] = 0
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"


@dataclass(frozen=True)
class AFDGPUProfilerConfig:
    enabled: bool
    wait: int
    warmup: int
    active: int
    repeat: int
    skip_first: int
    trace_dir: str
    # When set, the profiler (and its wait/warmup/active schedule) starts at
    # the first step on which this file exists instead of at runner init.
    trigger: str | None = None
    # Record ops of every thread, not only the one that starts the profiler.
    all_threads: bool = True


def afd_gpu_profiler_config(role: AFDGPUProfilerRole) -> AFDGPUProfilerConfig:
    """Read plugin-owned profiler settings for an AFD GPU runner role."""

    prefix = _ENV_PREFIX[role]
    return AFDGPUProfilerConfig(
        enabled=_env_bool(f"{prefix}_ENABLE", default=False),
        wait=_env_int(f"{prefix}_WAIT", default=_DEFAULT_WAIT_STEPS),
        warmup=_env_int(f"{prefix}_WARMUP", default=_DEFAULT_WARMUP_STEPS),
        active=_env_int(f"{prefix}_ACTIVE", default=_DEFAULT_ACTIVE_STEPS),
        repeat=_env_int(f"{prefix}_REPEAT", default=_DEFAULT_REPEAT),
        skip_first=_env_int(
            f"{prefix}_SKIP_FIRST",
            default=_DEFAULT_SKIP_FIRST_STEPS,
        ),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
        trigger=os.getenv(f"{prefix}_TRIGGER") or None,
        all_threads=_env_bool(f"{prefix}_ALL_THREADS", default=True),
    )


class TriggeredProfiler:
    """A torch profiler that starts each time a trigger file appears.

    The step-counted schedule of a profiler started at runner init lands
    wherever the request mix puts it; a run script can instead create the
    trigger file in every Attention and FFN pod, and all roles then record
    the same steps from the first step that sees the file.

    One trigger records one schedule (``wait``/``warmup``/``active`` times
    ``repeat``). When ``window_steps`` is given, the profiler stops after that
    many steps, removes the trigger file and waits for the file to be created
    again, so a run script can record several windows from one runner without
    restarting it. Each window uses a new profiler object from ``factory``.
    """

    def __init__(
        self,
        profiler: torch.profiler.profile,
        trigger: str,
        role: AFDGPUProfilerRole,
        *,
        factory: Callable[[], torch.profiler.profile] | None = None,
        window_steps: int | None = None,
    ) -> None:
        self.profiler = profiler
        self.trigger = trigger
        self.role = role
        self.started = False
        self.windows = 0
        self._factory = factory
        self._window_steps = window_steps
        self._steps_in_window = 0

    def step(self) -> None:
        if self.started:
            self.profiler.step()
            self._steps_in_window += 1
            if (
                self._window_steps is not None
                and self._steps_in_window >= self._window_steps
            ):
                self._finish_window()
        elif os.path.exists(self.trigger):
            if self.profiler is None or (self.windows > 0 and self._factory is not None):
                assert self._factory is not None
                self.profiler = self._factory()
            self.profiler.start()
            self.started = True
            self._steps_in_window = 0
            self.windows += 1
            logger.info(
                "AFD GPU %s profiler started: %s exists (window %d)",
                self.role,
                self.trigger,
                self.windows,
            )

    def _finish_window(self) -> None:
        # The schedule has already handed its trace to on_trace_ready on the
        # last step; stop() only releases the profiler.
        self.profiler.stop()
        self.started = False
        if self._factory is not None:
            # Free the finished profiler now, while no profiler session is
            # active. Left to the cyclic garbage collector it is freed at an
            # arbitrary later step, which can crash in libkineto once the next
            # session has been started.
            self.profiler = None  # type: ignore[assignment]
            gc.collect()
        try:
            os.remove(self.trigger)
        except FileNotFoundError:
            pass
        logger.info(
            "AFD GPU %s profiler window %d done after %d steps; "
            "waiting for %s again",
            self.role,
            self.windows,
            self._steps_in_window,
            self.trigger,
        )

    def stop(self) -> None:
        if self.started and self.profiler is not None:
            self.profiler.stop()
            self.started = False


def create_afd_gpu_profiler(
    role: AFDGPUProfilerRole,
) -> torch.profiler.profile | TriggeredProfiler | None:
    """Create a torch profiler when the plugin-owned env enables it.

    With ``<prefix>_TRIGGER`` set, the profiler is returned unstarted inside a
    ``TriggeredProfiler`` and starts at the first step its file exists; after
    the schedule's steps it removes the file and waits for the next trigger.
    """

    config = afd_gpu_profiler_config(role)
    if not config.enabled:
        return None

    import torch

    logger.info(
        "AFD GPU %s profiler enabled. Traces will be saved to: %s",
        role,
        config.trace_dir,
    )
    def make_profiler() -> torch.profiler.profile:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=config.wait,
                warmup=config.warmup,
                active=config.active,
                repeat=config.repeat,
                skip_first=config.skip_first,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.trace_dir),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            # vLLM DBO runs each micro-batch on its own Python thread, started
            # per forward; by default the profiler records ops of the starting
            # thread only, so the micro-batches' ranges and ops would be missing.
            # <prefix>_ALL_THREADS=0 records the starting thread only, as a
            # stock vLLM torch profiler does.
            experimental_config=(
                torch.profiler._ExperimentalConfig(profile_all_threads=True)
                if config.all_threads
                else None
            ),
        )

    profiler = make_profiler()
    if config.trigger is not None:
        # repeat == 0 means "cycle forever" in torch's schedule: no window end.
        window_steps = (
            config.skip_first
            + (config.wait + config.warmup + config.active) * config.repeat
            if config.repeat > 0
            else None
        )
        logger.info(
            "AFD GPU %s profiler waits for %s before its schedule starts "
            "(window steps: %s)",
            role,
            config.trigger,
            window_steps,
        )
        return TriggeredProfiler(
            profiler,
            config.trigger,
            role,
            factory=make_profiler,
            window_steps=window_steps,
        )
    profiler.start()
    return profiler


def step_afd_gpu_profiler(
    profiler: torch.profiler.profile | TriggeredProfiler | None,
) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_gpu_profiler(
    profiler: torch.profiler.profile | TriggeredProfiler | None,
) -> None:
    if profiler is not None:
        profiler.stop()


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


__all__ = [
    "AFDGPUProfilerConfig",
    "TriggeredProfiler",
    "afd_gpu_profiler_config",
    "create_afd_gpu_profiler",
    "step_afd_gpu_profiler",
    "stop_afd_gpu_profiler",
]
