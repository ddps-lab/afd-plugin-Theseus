from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.profiler import (
    TriggeredProfiler,
    afd_gpu_profiler_config,
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)

_ENV_NAMES = (
    "AFD_GPU_ATTENTION_PROFILER_ENABLE",
    "AFD_GPU_ATTENTION_PROFILER_WAIT",
    "AFD_GPU_ATTENTION_PROFILER_WARMUP",
    "AFD_GPU_ATTENTION_PROFILER_ACTIVE",
    "AFD_GPU_ATTENTION_PROFILER_REPEAT",
    "AFD_GPU_ATTENTION_PROFILER_SKIP_FIRST",
    "AFD_GPU_ATTENTION_PROFILER_DIR",
    "AFD_GPU_ATTENTION_PROFILER_TRIGGER",
    "AFD_GPU_FFN_PROFILER_ENABLE",
    "AFD_GPU_FFN_PROFILER_WAIT",
    "AFD_GPU_FFN_PROFILER_WARMUP",
    "AFD_GPU_FFN_PROFILER_ACTIVE",
    "AFD_GPU_FFN_PROFILER_REPEAT",
    "AFD_GPU_FFN_PROFILER_SKIP_FIRST",
    "AFD_GPU_FFN_PROFILER_DIR",
    "AFD_GPU_FFN_PROFILER_TRIGGER",
    "VLLM_TORCH_PROFILER_DIR",
)


@pytest.fixture(autouse=True)
def _clear_profiler_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_gpu_profiler_defaults_are_disabled():
    attention = afd_gpu_profiler_config("attention")
    ffn = afd_gpu_profiler_config("ffn")

    assert attention.enabled is False
    assert attention.wait == 2500
    assert attention.warmup == 1
    assert attention.active == 10
    assert attention.repeat == 1
    assert attention.skip_first == 0
    assert attention.trace_dir == "./profiler_logs/attn"
    assert ffn.enabled is False
    assert ffn.trace_dir == "./profiler_logs/ffn"
    assert attention.trigger is None


def test_gpu_profiler_dir_falls_back_to_vllm_torch_profiler_dir(monkeypatch):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/vllm-profile")

    assert afd_gpu_profiler_config("attention").trace_dir == "/tmp/vllm-profile"

    monkeypatch.setenv("AFD_GPU_ATTENTION_PROFILER_DIR", "/tmp/afd-attn")

    assert afd_gpu_profiler_config("attention").trace_dir == "/tmp/afd-attn"


def test_create_gpu_profiler_uses_configured_schedule(monkeypatch):
    profiler_module = _FakeTorchProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_ENABLE", "true")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_WAIT", "3")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_WARMUP", "4")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_ACTIVE", "5")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_REPEAT", "6")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_SKIP_FIRST", "7")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_DIR", "/tmp/afd-ffn")

    profiler = create_afd_gpu_profiler("ffn")

    assert profiler is profiler_module.created_profiler
    assert profiler.started is True
    assert profiler_module.schedule_kwargs == {
        "wait": 3,
        "warmup": 4,
        "active": 5,
        "repeat": 6,
        "skip_first": 7,
    }
    assert profiler_module.profile_kwargs["record_shapes"] is True
    assert profiler_module.profile_kwargs["profile_memory"] is False
    assert profiler_module.profile_kwargs["with_stack"] is False
    assert profiler_module.trace_dir == "/tmp/afd-ffn"


def test_triggered_gpu_profiler_starts_when_trigger_file_exists(monkeypatch, tmp_path):
    profiler_module = _FakeTorchProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(profiler=profiler_module),
    )
    trigger = tmp_path / "go"
    monkeypatch.setenv("AFD_GPU_ATTENTION_PROFILER_ENABLE", "1")
    monkeypatch.setenv("AFD_GPU_ATTENTION_PROFILER_TRIGGER", str(trigger))

    profiler = create_afd_gpu_profiler("attention")
    inner = profiler_module.created_profiler

    assert isinstance(profiler, TriggeredProfiler)
    assert inner.started is False
    step_afd_gpu_profiler(profiler)
    step_afd_gpu_profiler(profiler)
    assert inner.started is False
    assert inner.steps == 0

    trigger.touch()
    step_afd_gpu_profiler(profiler)  # starts; start() takes schedule step 0
    assert inner.started is True
    assert inner.steps == 0
    step_afd_gpu_profiler(profiler)
    step_afd_gpu_profiler(profiler)
    assert inner.steps == 2

    stop_afd_gpu_profiler(profiler)
    assert inner.stopped is True


def test_triggered_gpu_profiler_never_started_is_not_stopped(monkeypatch, tmp_path):
    profiler_module = _FakeTorchProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_ENABLE", "1")
    monkeypatch.setenv("AFD_GPU_FFN_PROFILER_TRIGGER", str(tmp_path / "never"))

    profiler = create_afd_gpu_profiler("ffn")
    step_afd_gpu_profiler(profiler)
    stop_afd_gpu_profiler(profiler)

    assert profiler_module.created_profiler.started is False
    assert profiler_module.created_profiler.stopped is False


def test_step_gpu_profiler_ignores_disabled_profiler():
    step_afd_gpu_profiler(None)

    profiler = _StepProfiler()
    step_afd_gpu_profiler(profiler)

    assert profiler.steps == 1


def test_stop_gpu_profiler_ignores_disabled_profiler():
    stop_afd_gpu_profiler(None)

    profiler = _StepProfiler()
    stop_afd_gpu_profiler(profiler)

    assert profiler.stopped is True


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self):
        self.steps += 1


class _FakeTorchProfiler:
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"

    def __init__(self):
        self.created_profiler = _StepProfiler()
        self.schedule_kwargs = None
        self.profile_kwargs = None
        self.trace_dir = None

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir):
        self.trace_dir = trace_dir
        return ("handler", trace_dir)

    def profile(self, **kwargs):
        self.profile_kwargs = kwargs
        return self.created_profiler
