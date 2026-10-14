# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen3 MoE AFD model wrapper for CUDA execution.

The wrapper keeps the native vLLM Qwen3 MoE lifecycle and injects a
role-aware decoder layer through the native ``decoder_layer_type`` hook.
Attention executes the native Attention, residual, and normalization path;
FFN executes the complete native dense MLP or MoE block.
"""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from transformers import Qwen3MoeConfig
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.models import qwen3_moe as native
from vllm.platforms import current_platform

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy
from afd_plugin.model_executor.models.routed_moe import (
    RoutedExpertsProxy,
    is_routed_afd,
    routed_moe_forward,
    run_prerouted_experts,
)

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_BOTH_ROLES = frozenset(("attention", "ffn"))


def _is_moe_layer(config: Qwen3MoeConfig, layer_idx: int) -> bool:
    """Return whether the native Qwen3 schedule selects MoE for one layer."""
    return (
        layer_idx not in config.mlp_only_layers
        and config.num_experts > 0
        and (layer_idx + 1) % config.decoder_sparse_step == 0
    )


def _weight_layer_path(name: str) -> tuple[int, str, tuple[str, ...]] | None:
    """Return ``(layer index, stage, remainder)`` for a decoder weight path."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2], tuple(parts[marker_idx + 3 :])
    return None


def _checkpoint_weight_roles(
    name: str,
    *,
    config: Qwen3MoeConfig | None = None,
    routed: bool = False,
) -> frozenset[str]:
    """Classify one native checkpoint path by its AFD execution owner."""
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES

    layer_idx, stage, remainder = layer_path
    if stage == "mlp":
        if not routed:
            return _FFN_ROLE
        # One-hop routed connector: dense MLPs and the gate run on Attention,
        # only the routed experts live on FFN.
        if config is None:
            raise ValueError("routed weight classification needs the model config")
        if not _is_moe_layer(config, layer_idx):
            return _ATTENTION_ROLE
        is_routed_experts = bool(remainder) and remainder[0] == "experts"
        return _FFN_ROLE if is_routed_experts else _ATTENTION_ROLE
    if stage in {"self_attn", "input_layernorm", "post_attention_layernorm"}:
        return _ATTENTION_ROLE
    return _BOTH_ROLES


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    config: Qwen3MoeConfig | None = None,
    routed: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain this role's paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name, config=config, routed=routed):
            yield name, loaded_weight


def _validate_supported_config(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Reject Qwen3 MoE modes not covered by the initial CUDA contract."""
    parallel_config = vllm_config.parallel_config
    if not current_platform.is_cuda():
        raise RuntimeError("AFD Qwen3 MoE currently supports CUDA only")
    if afd_config.compute_gate_on_attention and not is_routed_afd(afd_config):
        raise RuntimeError(
            "AFD Qwen3 MoE requires compute_gate_on_attention=false unless the "
            "one-hop routed connector (P2pNcclRoutedAFDConnector) is used",
        )
    if parallel_config.use_sequence_parallel_moe:
        raise RuntimeError(
            "AFD Qwen3 MoE does not support sequence-parallel MoE",
        )
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD Qwen3 MoE does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("AFD Qwen3 MoE does not support pipeline parallelism")
    if vllm_config.speculative_config is not None:
        raise RuntimeError("AFD Qwen3 MoE does not support speculative decoding")
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD Qwen3 MoE does not support LoRA")


class AFDQwen3MoeRoutedAttentionBlock(nn.Module):
    """Attention-side Qwen3 MoE block for the one-hop routed connector.

    Owns the gate (native attribute name ``gate``) and the top-k router; the
    routed experts run on the FFN ranks that own them.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        if getattr(config, "shared_expert_intermediate_size", 0) > 0:
            raise NotImplementedError(
                "routed AFD Qwen3 MoE does not support shared experts yet",
            )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = int(config.num_experts)
        self.gate = native.ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        # Same router as the native FusedMoE (plain top-k, softmax).
        self.router = create_fused_moe_router(
            top_k=int(config.num_experts_per_tok),
            global_num_experts=self.n_routed_experts,
            renormalize=config.norm_topk_prob,
        )
        self.experts = RoutedExpertsProxy(layer_idx=layer_idx)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "AFDQwen3MoeRoutedAttentionBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        output = routed_moe_forward(
            hidden_states,
            gate=self.gate,
            router=self.router,
            proxy=self.experts,
            shared_experts=None,
            routed_scaling_factor=1.0,
            tp_size=self.tp_size,
        )
        return output.squeeze(0) if is_input_1d else output


class AFDQwen3MoeRoutedFFNBlock(native.Qwen3MoeSparseMoeBlock):
    """FFN-side Qwen3 MoE block for the one-hop routed connector: experts only."""

    # Patch reason: native Qwen3MoeSparseMoeBlock constructs the gate and routes
    # inside FusedMoE.
    # Patch functionality: construct only the routed experts (``prerouted``
    # all2all backend) and feed them the routing received from Attention.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_moe.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = native.get_ep_group().device_group
        self.ep_rank = native.get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = int(config.num_experts)
        self.is_sequence_parallel = False
        self.enable_eplb = False
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_routed_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )
        # The gate (and any shared expert) is Attention-owned in this layout.
        self.gate = None
        self.shared_expert_gate = None
        self.shared_expert = None
        self.experts = native.FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "routed FFN MoE is driven by compute_routed_experts_output",
        )

    def compute_routed_experts_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return run_prerouted_experts(
            self.experts,
            hidden_states,
            topk_weights,
            topk_ids,
            a1q_scale,
        )


class AFDQwen3MoeDecoderLayer(native.Qwen3MoeDecoderLayer):
    """Qwen3 MoE decoder layer with separable Attention and FFN execution."""

    # Patch reason: native Qwen3 MoE constructs both Attention and FFN modules.
    # Patch functionality: construct only the large modules owned by the AFD role.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_moe.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: initialize the role-aware layer without native allocation.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role
        # One-hop routed connector: tokens go straight to the FFN ranks that
        # own their experts; the gate and dense MLPs stay on Attention.
        self.routed = is_routed_afd(afd_config)
        # ### PATCH END: initialize the role-aware layer without native allocation.

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config,
            "dual_chunk_attention_config",
            None,
        )
        # ### PATCH START: retain the native layer schedule for AFD role dispatch.
        layer_idx = native.extract_layer_index(prefix)
        self.layer_idx = layer_idx
        self.is_moe_layer = _is_moe_layer(config, layer_idx)
        # ### PATCH END: retain the native layer schedule for AFD role dispatch.

        # ### PATCH START: allocate only the active role's execution stage.
        if self.afd_role == "attention":
            self.self_attn = native.Qwen3MoeAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_parameters=config.rope_parameters,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=getattr(config, "head_dim", None),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                dual_chunk_attention_config=dual_chunk_attention_config,
            )
            if self.routed and self.is_moe_layer:
                self.mlp = AFDQwen3MoeRoutedAttentionBlock(
                    vllm_config=vllm_config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.mlp",
                )
            elif self.routed:
                # Dense layers run on Attention with the routed connector.
                self.mlp = native.Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = RemoteFFNProxy(layer_idx=layer_idx)
            self.input_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = native.PPMissingLayer()
            if self.is_moe_layer and self.routed:
                self.mlp = AFDQwen3MoeRoutedFFNBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.mlp",
                )
            elif self.is_moe_layer:
                self.mlp = native.Qwen3MoeSparseMoeBlock(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.mlp",
                )
            elif self.routed:
                # Dense layers run on Attention with the routed connector.
                self.mlp = native.PPMissingLayer()
            else:
                self.mlp = native.Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            self.input_layernorm = native.PPMissingLayer()
            self.post_attention_layernorm = native.PPMissingLayer()
        # ### PATCH END: allocate only the active role's execution stage.

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute this layer's complete native FFN block on the FFN role."""
        if self.afd_role != "ffn":
            raise RuntimeError("Qwen3 MoE FFN compute requires the AFD FFN role")
        return self.mlp(hidden_states)

    def compute_routed_experts_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the experts this FFN rank owns on the rows routed to it."""
        if self.afd_role != "ffn" or not self.routed or not self.is_moe_layer:
            raise RuntimeError(
                "compute_routed_experts_output requires a routed FFN MoE layer",
            )
        if not isinstance(self.mlp, AFDQwen3MoeRoutedFFNBlock):
            raise RuntimeError("FFN role does not own a routed Qwen3 MoE block")
        return self.mlp.compute_routed_experts_output(
            hidden_states,
            topk_weights,
            topk_ids,
            a1q_scale=a1q_scale,
        )


class AFDQwen3MoeModel(native.Qwen3MoeModel):
    """Native Qwen3 MoE model using role-aware decoder layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_supported_config(vllm_config, afd_config)
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=AFDQwen3MoeDecoderLayer,
        )
        self.afd_config = afd_config
        self.afd_role = afd_config.role

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states)

    def compute_routed_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_routed_experts_output(
            hidden_states,
            topk_weights,
            topk_ids,
            a1q_scale=a1q_scale,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx in range(self.config.num_hidden_layers)
            if _is_moe_layer(self.config, layer_idx)
        )


class AFDQwen3MoeForCausalLM(native.Qwen3MoeForCausalLM):
    """Qwen3 MoE causal LM wrapper for AFD CUDA execution."""

    # Patch reason: native Qwen3MoeForCausalLM has no model-class injection hook.
    # Patch functionality: construct AFDQwen3MoeModel and retain native lifecycle
    # and MoE metadata without requiring local experts on the Attention role.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_moe.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: avoid constructing the full native model first.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # ### PATCH END: avoid constructing the full native model first.
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen3MoeMLP exists
        # ### PATCH START: keep the inherited class mapping immutable.
        self.packed_modules_mapping = dict(self.packed_modules_mapping)
        # ### PATCH END: keep the inherited class mapping immutable.
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = [
                "gate_proj",
                "up_proj",
            ]
        # ### PATCH START: inject the role-aware model.
        self.model = AFDQwen3MoeModel(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        # ### PATCH END: inject the role-aware model.
        self.lm_head = native.ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=native.maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, native.PPMissingLayer):
                continue

            assert isinstance(layer, native.Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, native.Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        # ### PATCH START: Attention owns no local MoE but keeps native metadata.
        if example_layer is None and self.afd_role == "attention":
            moe_layer_indices = self.model.get_experts_layer_indices()
            if not moe_layer_indices:
                raise RuntimeError("No Qwen3MoE layer found in the model.layers.")
            self.num_moe_layers = 0
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = config.num_experts
            self.num_physical_experts = config.num_experts
            self.num_local_physical_experts = 0
            self.num_routed_experts = config.num_experts
            self.num_redundant_experts = 0
        elif example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = example_layer.n_logical_experts
            self.num_physical_experts = example_layer.n_physical_experts
            self.num_local_physical_experts = example_layer.n_local_physical_experts
            self.num_routed_experts = example_layer.n_routed_experts
            self.num_redundant_experts = example_layer.n_redundant_experts
        # ### PATCH END: Attention owns no local MoE but keeps native metadata.

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def compute_routed_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.compute_routed_experts_output(
            hidden_states,
            layer_idx,
            topk_weights,
            topk_ids,
            a1q_scale=a1q_scale,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    # Patch reason: the native loader would materialize both execution stages
    # even though each AFD role constructs only one stage's decoder modules.
    # Patch functionality: retain only role-owned checkpoint paths, then use the
    # native loader unchanged for mapping, packing, and loaded-parameter results.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_moe.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: filter the checkpoint stream by AFD execution role.
        role_weights = _iter_role_weights(
            weights,
            role=self.afd_role,
            config=self.config,
            routed=is_routed_afd(self.afd_config),
        )
        # ### PATCH END: filter the checkpoint stream by AFD execution role.
        return super().load_weights(role_weights)


__all__ = ["AFDQwen3MoeForCausalLM"]
