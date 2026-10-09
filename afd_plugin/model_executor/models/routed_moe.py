# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-side building blocks of the one-hop routed AFD path (CUDA).

With ``P2pNcclRoutedAFDConnector`` an MoE layer is split as follows:

* Attention: gate -> top-k (vLLM router) -> optional quantization ->
  dispatch to the FFN ranks owning the selected experts -> shared experts
  (DeepSeek) while the round trip is in flight -> combine.
* FFN: run the local experts on the received rows through vLLM's
  ``prerouted`` all2all backend (no communication) and return the weighted
  partial sums.

Model wrappers build their routed MoE modules from ``RoutedExpertsProxy``,
``routed_moe_forward`` and ``run_prerouted_experts`` so that DeepSeek and
Qwen3 share one implementation of the exchange.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.prepare_finalize.prerouted import (
    PreroutedPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models.utils import sequence_parallel_chunk

from afd_plugin.config import ROUTED_P2P_CONNECTOR, AFDConfig
from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)


def is_routed_afd(afd_config: AFDConfig) -> bool:
    return afd_config.connector == ROUTED_P2P_CONNECTOR


class RoutedExpertsProxy(nn.Module):
    """Parameter-free routed experts executed on the FFN ranks owning them."""

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_fn: Callable[[], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch, run ``shared_experts_fn`` meanwhile, and combine.

        Returns ``(routed_output, shared_output)``; ``routed_output`` is the
        sum of the per-rank partials, not yet scaled by
        ``routed_scaling_factor``.
        """
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RoutedExpertsProxy requires AFD forward metadata")
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        connector = afd_metadata.connector
        a1q, a1q_scale = connector.quantize_for_dispatch(hidden_states)
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=self.layer_idx,
            stage_idx=stage_idx,
            seq_len=max(1, int(hidden_states.shape[0])),
        )
        context = AFDTransferContext(metadata=metadata)
        connector.send_attn_output(
            a1q,
            context,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
        )
        shared_output = shared_experts_fn() if shared_experts_fn is not None else None
        routed_output = connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )
        return routed_output, shared_output

    def update_expert_map(self) -> None:
        """Satisfy the native EPLB model interface without local experts."""


def routed_moe_forward(
    hidden_states: torch.Tensor,
    *,
    gate: nn.Module,
    router: FusedMoERouter,
    proxy: RoutedExpertsProxy,
    shared_experts: nn.Module | None,
    routed_scaling_factor: float,
    tp_size: int,
) -> torch.Tensor:
    """Attention-side MoE forward for the routed path.

    With Attention TP > 1 the hidden states are replicated on the TP ranks,
    so every rank routes and dispatches its own token slice and the routed
    output is all-gathered afterwards (shared experts are TP-sharded MLPs
    that reduce their own output).
    """
    num_tokens, hidden_dim = hidden_states.shape
    local = sequence_parallel_chunk(hidden_states) if tp_size > 1 else hidden_states
    router_logits, _ = gate(local)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=local,
        router_logits=router_logits,
        topk_indices_dtype=torch.int32,
    )
    shared_fn = None
    if shared_experts is not None:
        shared_fn = lambda: shared_experts(hidden_states)  # noqa: E731
    routed_output, shared_output = proxy(local, topk_weights, topk_ids, shared_fn)
    if tp_size > 1:
        routed_output = tensor_model_parallel_all_gather(routed_output, 0)
        routed_output = routed_output[:num_tokens]
    # Same scaling as vLLM's MoERunner with apply_routed_scale_to_output: the
    # routed sum is scaled, except in fp16 where the shared output is divided
    # instead (the decoder layer compensates) to avoid overflow.
    if routed_scaling_factor != 1.0:
        if routed_output.dtype != torch.float16 or shared_output is None:
            routed_output = routed_output * routed_scaling_factor
        else:
            shared_output = shared_output * (1.0 / routed_scaling_factor)
    if shared_output is not None:
        routed_output = routed_output + shared_output
    return routed_output.view(num_tokens, hidden_dim)


def run_prerouted_experts(
    experts: MoERunner,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1q_scale: torch.Tensor | None,
) -> torch.Tensor:
    """FFN-side: run the local experts of ``experts`` on received rows.

    ``experts`` is the ``MoERunner`` returned by vLLM's ``FusedMoE`` factory,
    built with ``--all2all-backend prerouted``. Rows that arrived quantized
    (``hidden_states`` in the wire's fp8 dtype) are handed to the prerouted
    prepare/finalize as-is together with their scales.
    """
    moe_config = experts.moe_config
    routed_experts = experts.routed_experts
    num_tokens = int(hidden_states.shape[0])
    if num_tokens == 0:
        return torch.empty(
            (0, moe_config.hidden_dim),
            dtype=moe_config.in_dtype,
            device=hidden_states.device,
        )
    prepare_finalize = routed_experts.quant_method.moe_kernel.prepare_finalize
    if not isinstance(prepare_finalize, PreroutedPrepareAndFinalize):
        raise RuntimeError(
            "routed AFD FFN role must run with --enable-expert-parallel "
            "--all2all-backend prerouted (got "
            f"{type(prepare_finalize).__name__})",
        )
    if hidden_states.dtype == moe_config.in_dtype:
        kernel_input = hidden_states
    else:
        # Pre-quantized rows: the kernel input only fixes the output dtype and
        # the row count; the quantized data and scales go through the
        # prepare/finalize.
        prepare_finalize.set_prequantized(hidden_states, a1q_scale)
        kernel_input = torch.empty(
            (num_tokens, moe_config.hidden_dim),
            dtype=moe_config.in_dtype,
            device=hidden_states.device,
        )
    return routed_experts.forward_modular(
        kernel_input,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
    )


__all__ = [
    "RoutedExpertsProxy",
    "is_routed_afd",
    "routed_moe_forward",
    "run_prerouted_experts",
]
