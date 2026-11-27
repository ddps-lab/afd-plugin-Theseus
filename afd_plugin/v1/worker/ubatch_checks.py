# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Micro-batching checks shared by the attention and FFN workers.

Kept apart from the model runners so that the workers can import it on vLLM
releases that no longer ship the pre-V2 ``gpu_model_runner`` module.
"""

from __future__ import annotations

from vllm.config import VllmConfig


def fail_if_unsupported_ubatching(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if not bool(parallel_config.use_ubatching):
        return
    num_ubatches = int(parallel_config.num_ubatches)
    if num_ubatches < 2:
        raise RuntimeError(
            "AFD ubatching needs at least two ubatches; "
            f"got num_ubatches={num_ubatches}",
        )
    # ``--enable-dbo`` is exactly two ubatches; ``--ubatch-size N`` selects N.
    # More than two ubatches are only supported in eager mode: the CUDA graph
    # path captures exactly two ubatch streams.
    if num_ubatches != 2 and not bool(vllm_config.model_config.enforce_eager):
        raise RuntimeError(
            "AFD ubatching with more than two ubatches requires --enforce-eager; "
            f"got num_ubatches={num_ubatches}",
        )


fail_if_ubatching_enabled = fail_if_unsupported_ubatching


__all__ = ["fail_if_ubatching_enabled", "fail_if_unsupported_ubatching"]
