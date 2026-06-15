import logging
import math

import torch
from tensor_cast.layers.moe_layer import FusedMoETensorCast, MoELayer
from tensor_cast.performance_model.utils import bytes_of_tensor
from tensor_cast.quantize_utils import LinearQuantType
from tensor_cast.transformers.transformations import (
    maybe_enable_mtp,
    maybe_reuse_layers,
    patch_attention,
    patch_moe,
    quantize_model,
    shard_model,
    wrap_model,
)
from tensor_cast.utils import DTYPE_FP8

from ..custom_model_registry import (
    ModelProfile,
    register_custom_model,
    register_model_profile,
)
from ..model import TransformerModel
from ...layers.minimax_m3_attention import MiniMaxM3AttentionWrapper
from ...layers.utils import ModelWrapperBase

logger = logging.getLogger(__name__)

_EMPTY_VISUAL_LAYERS_ATTR = "_tensor_cast_empty_visual_layers"




class MiniMaxM3DecoderLayerWrapper(ModelWrapperBase):
    def __init__(self, layer):
        super().__init__(layer)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        eps = getattr(self._inner.input_layernorm, 'variance_epsilon', None) or self._inner.input_layernorm.eps
        input_ln_weight = self._inner.input_layernorm.weight.data
        post_attn_ln_weight = self._inner.post_attention_layernorm.weight.data
        use_gemma_norm = getattr(self._inner, "use_gemma_norm", False)
        if use_gemma_norm:
            input_ln_weight = 1.0 + input_ln_weight
            post_attn_ln_weight = 1.0 + post_attn_ln_weight

        residual = hidden_states
        hidden_states, residual = torch.ops.tensor_cast.add_rms_norm2(
            hidden_states, residual, input_ln_weight, eps,
        )

        attn_out, _ = self._inner.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = attn_out

        hidden_states, residual = torch.ops.tensor_cast.add_rms_norm2(
            hidden_states, residual, post_attn_ln_weight, eps,
        )

        hidden_states = self._inner.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


def _patch_minimax_m3_rmsnorm(model: TransformerModel) -> TransformerModel:
    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        candidates = [
            getattr(unwrapped, "language_model", None),
            getattr(getattr(unwrapped, "model", None), "language_model", None),
            getattr(getattr(unwrapped, "language_model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "layers"):
                unwrapped = candidate
                break
        else:
            model_attr = getattr(unwrapped, "model", None)
            if hasattr(model_attr, "layers"):
                unwrapped = model_attr
            else:
                logger.warning("Cannot find layers for M3 rmsnorm patch")
                return model

    for i, layer in enumerate(unwrapped.layers):
        inner = layer
        while hasattr(inner, "_inner"):
            inner = inner._inner
        if hasattr(inner, "input_layernorm") and hasattr(inner, "post_attention_layernorm"):
            wrapped = MiniMaxM3DecoderLayerWrapper(inner)
            parent = unwrapped
            unwrapped.layers[i] = wrapped

    return model

class MiniMaxM3ExpertMLP(torch.nn.Module):
    """M3 Expert MLP using gate_proj + up_proj + down_proj with standard swiglu.

    Forward uses silu(gate) * up (standard SwiGLU) instead of
    SwigluOAIAndMul(cat([gate, up])), so that after quantization the DFC
    (dispatch_ffn_combine) pass Case 2 can recognize:
      static_quant_linear(gate) -> silu -> mul(up) -> static_quant_linear(down)
    and fuse into a single dispatch_ffn_combine kernel.
    """

    def __init__(self, original_experts_module, expert_idx=None):
        super().__init__()
        if isinstance(original_experts_module, torch.nn.ModuleList) and expert_idx is None:
            expert = original_experts_module[0]
        elif expert_idx is not None:
            expert = original_experts_module[expert_idx]
        else:
            expert = original_experts_module
        self.hidden_size = expert.hidden_size
        self.intermediate_size = expert.intermediate_size

        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        with torch.no_grad():
            self.gate_proj.weight.copy_(expert.gate_proj.weight.data)
            self.up_proj.weight.copy_(expert.up_proj.weight.data)
            self.down_proj.weight.copy_(expert.down_proj.weight.data)

    def forward(self, hidden_states):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _MoeReturnCompat(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self._moe = moe

    def forward(self, hidden_states):
        result = self._moe(hidden_states)
        if isinstance(result, tuple):
            return result
        return result, None


class MiniMaxM3MoELayer(MoELayer):
    def __init__(self, moe_config, module, quant_type):
        super().__init__(moe_config, module)
        self.fused_moe = MiniMaxM3FusedMoETensorCast(
            self.moe_config,
            self.get_attr(module, "experts", None),
            self.get_attr(module, "shared_experts", None),
            self.get_attr(module, "shared_experts_gate", None),
            self.top_k,
        )
        self.fused_moe.routed_scaling_factor = module.routed_scaling_factor
        self.fused_moe.swiglu_alpha = module.experts.swiglu_alpha
        self.fused_moe.swiglu_limit = module.experts.swiglu_limit
        self.fused_moe.quant_type = quant_type
        self.fused_moe.refresh_expert_weight_cache()
        self.gate_weight = self.gate.weight.data if self.gate is not None else None
        self.num_local_experts = self.fused_moe.num_global_experts

    def route(
        self,
        hidden_states: torch.Tensor,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        if self.gate_weight is not None:
            scores = torch.ops.tensor_cast.moe_decode_score(hidden_states, self.gate_weight)
        else:
            gate_output = self.gate(hidden_states)
            if isinstance(gate_output, tuple) and len(gate_output) >= 2:
                if len(gate_output) == 3:
                    router_logits, topk_weights, topk_indices = gate_output
                else:
                    topk_indices, topk_weights = gate_output[0], gate_output[1]
                if topk_indices.shape[0] == hidden_states.shape[0]:
                    topk_indices = topk_indices.view(*hidden_states.shape[:-1], topk_indices.shape[-1])
                    topk_weights = topk_weights.view(*hidden_states.shape[:-1], topk_weights.shape[-1])
                return topk_indices, topk_weights
            scores = gate_output

        partial_weights, partial_indices = torch.ops.tensor_cast.moe_topk_index_partial(
            scores, self.top_k,
        )
        topk_weights, topk_indices = torch.ops.tensor_cast.moe_topk_index_merge(
            partial_weights, partial_indices,
        )
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        if topk_indices.shape[0] == hidden_states.shape[0]:
            topk_indices = topk_indices.view(*hidden_states.shape[:-1], topk_indices.shape[-1])
            topk_weights = topk_weights.view(*hidden_states.shape[:-1], topk_weights.shape[-1])

        return topk_indices, topk_weights


class MiniMaxM3FusedMoETensorCast(FusedMoETensorCast):
    routed_scaling_factor = 1.0
    swiglu_alpha = 1.0
    swiglu_limit = 1.0
    quant_type = None
    use_all_reduce_instead_of_slice_gather = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gate_up_weights = None
        self._gate_up_scales = None
        self._down_weights = None
        self._down_scales = None

    @staticmethod
    def _transpose_expert_weights(weight: torch.Tensor) -> list[torch.Tensor]:
        return [weight[i].transpose(0, 1).contiguous() for i in range(weight.shape[0])]

    @staticmethod
    def _bias_list(count: int) -> list[None]:
        return [None] * count

    @staticmethod
    def _scale_list(weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.ones((), device=weight.device, dtype=torch.float32) for weight in weights]

    @staticmethod
    def _split_by_inputs(tensor: torch.Tensor, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return list(torch.split(tensor, [x.shape[0] for x in inputs], dim=0))

    def refresh_expert_weight_cache(self):
        experts = self.experts.experts
        self._gate_up_weights = self._transpose_expert_weights(experts.gate_up_proj)
        self._down_weights = self._transpose_expert_weights(experts.down_proj)
        if self.quant_type == LinearQuantType.FP8:
            self._gate_up_weights = [weight.to(DTYPE_FP8) for weight in self._gate_up_weights]
            self._down_weights = [weight.to(DTYPE_FP8) for weight in self._down_weights]
            self._gate_up_scales = self._scale_list(self._gate_up_weights)
            self._down_scales = self._scale_list(self._down_weights)
        else:
            self._gate_up_scales = None
            self._down_scales = None

    def _grouped_matmul(
        self,
        x: list[torch.Tensor],
        weights: list[torch.Tensor],
        weight_scales: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        bias = self._bias_list(len(weights))
        if self.quant_type == LinearQuantType.FP8:
            return torch.ops.tensor_cast.grouped_matmul_fp8_bf16(
                x,
                weights,
                weight_scales,
                bias,
                out_dtype=x[0].dtype if x else torch.bfloat16,
            )
        return torch.ops.tensor_cast.grouped_matmul(x, weights, bias)



    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.ops.tensor_cast.m3_swiglu(gate, up, self.swiglu_alpha, self.swiglu_limit)

    def _apply_gate_quant(self, gate_up: torch.Tensor, group_size: int = 128):
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.ops.tensor_cast.m3_swiglu_quant(gate, up, self.swiglu_alpha, self.swiglu_limit, group_size)

    def _grouped_matmul_with_prequant(
        self,
        x: list[torch.Tensor],
        weights: list[torch.Tensor],
        weight_scales: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        bias = self._bias_list(len(weights))
        return torch.ops.tensor_cast.grouped_matmul_fp8_bf16(
            x,
            weights,
            weight_scales,
            bias,
            out_dtype=torch.bfloat16,
        )

    def _run_routed_experts(self, dispatched_hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        gate_up = self._grouped_matmul(dispatched_hidden_states, self._gate_up_weights, self._gate_up_scales)
        if self.quant_type == LinearQuantType.FP8:
            activated = self._apply_gate_quant(gate_up)
            activated_by_expert = self._split_by_inputs(activated, dispatched_hidden_states)
            down = self._grouped_matmul_with_prequant(activated_by_expert, self._down_weights, self._down_scales)
        else:
            activated = self._apply_gate(gate_up)
            activated_by_expert = self._split_by_inputs(activated, dispatched_hidden_states)
            down = self._grouped_matmul(activated_by_expert, self._down_weights, self._down_scales)
        return self._split_by_inputs(down, dispatched_hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        skip_shared_experts: bool = False,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        num_tokens = topk_indices.numel()
        split_sizes = self.get_split_sizes(num_tokens, self.top_k)

        expert_indices = topk_indices
        expert_weights = topk_weights

        reordered = torch.ops.tensor_cast.moe_post_reorder(
            hidden_states, topk_indices, self.num_global_experts,
        )
        seg_indptr = torch.ops.tensor_cast.moe_compute_seg_indptr(
            topk_indices, self.num_global_experts,
        )
        masked_m = torch.ops.tensor_cast.moe_compute_masked_m(seg_indptr)
        src2dst = torch.ops.tensor_cast.moe_compute_src2dst(
            topk_indices, self.num_global_experts,
        )
        gateup_input = torch.ops.tensor_cast.moe_fill_gateup_input(
            hidden_states, seg_indptr,
        )

        dispatched_hidden_states = self.dispatch_tokens(
            hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )

        experts_hidden_states = self._run_routed_experts(dispatched_hidden_states)
        combined_hidden_states = self.combine_tokens(
            experts_hidden_states,
            expert_indices,
            split_sizes[0],
            split_sizes[1],
            split_sizes[3],
        )
        final_hidden_states = (combined_hidden_states * expert_weights.unsqueeze(-1)).sum(dim=-2)
        final_hidden_states = final_hidden_states * self.routed_scaling_factor

        final_hidden_states = final_hidden_states.view(original_shape)

        if self.shared_experts and self.num_external_shared_experts == 0 and not skip_shared_experts:
            shared_output = self._run_shared_experts(hidden_states)
            if self.ep_group.world_size > 1:
                shared_output = self.ep_group.all_reduce(shared_output)
            final_hidden_states = final_hidden_states + shared_output

        if self.ep_group.world_size > 1:
            final_hidden_states = self.ep_group.all_reduce(final_hidden_states)

        return final_hidden_states.to(hidden_states.dtype)


def _patch_minimax_m3_hf_config(hf_config, model_id):
    if hasattr(hf_config, "text_config") and not hasattr(hf_config, "num_hidden_layers"):
        tc = hf_config.text_config
        for attr in ["num_hidden_layers", "hidden_size", "num_attention_heads",
                      "num_key_value_heads", "head_dim", "intermediate_size",
                      "vocab_size", "rms_norm_eps", "max_position_embeddings"]:
            if hasattr(tc, attr) and not hasattr(hf_config, attr):
                try:
                    setattr(hf_config, attr, getattr(tc, attr))
                except Exception:
                    pass


def _patch_m3_moe_return_compat(model):
    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "layers"):
            unwrapped = unwrapped.model
        else:
            return model
    for layer in unwrapped.layers:
        while hasattr(layer, "_inner"):
            layer = layer._inner
        block_sparse_moe = getattr(layer, "block_sparse_moe", None)
        if block_sparse_moe is not None and not isinstance(block_sparse_moe, _MoeReturnCompat):
            layer.block_sparse_moe = _MoeReturnCompat(block_sparse_moe)
    return model


def _ensure_empty_visual_layers_for_reuse(model: TransformerModel):
    # MiniMax-M3 text layers live under model.language_model.layers, but the current VL reuse path only
    # reaches language layers when visual layers are present. M3 simulations here do not compile visual
    # layers, so provide an empty visual-layer container to trigger language-layer reuse.
    unwrapped = model.unwrap()
    if not hasattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR):
        setattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR, torch.nn.ModuleList())


def _get_quantization_config_value(config, key, default=None):
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None and hasattr(config, "text_config"):
        quantization_config = getattr(config.text_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        return quantization_config.get(key, default)
    return getattr(quantization_config, key, default)


def _is_mxfp8_config(config) -> bool:
    quant_method = _get_quantization_config_value(config, "quant_method")
    return quant_method == "mxfp8"


def _get_weight_block_size(config):
    block_size = _get_quantization_config_value(config, "weight_block_size", (1, 32))
    return tuple(block_size)


def _mxfp8_tensor_weight_size(tensor: torch.Tensor, weight_block_size) -> int:
    # MiniMax-M3 MXFP8 expert weights are stored as one byte per weight element plus one byte per
    # scale block. The expert dimension is a batch dimension, so blocks cover the last two dims.
    rows_per_block, cols_per_block = weight_block_size
    leading_numel = math.prod(tensor.shape[:-2]) if tensor.ndim > 2 else 1
    out_dim, in_dim = tensor.shape[-2:]
    num_scale_blocks = (
        leading_numel
        * math.ceil(out_dim / rows_per_block)
        * math.ceil(in_dim / cols_per_block)
    )
    return tensor.numel() + num_scale_blocks


def estimate_minimax_m3_weight_size(model: TransformerModel) -> int:
    if not _is_mxfp8_config(model.hf_config):
        return model.get_weight_size_nested([model])

    weight_block_size = _get_weight_block_size(model.hf_config)
    total_size = 0
    for name, param in model.named_parameters():
        if (
            param.ndim == 3
            and (
                name.endswith(".gate_up_proj")
                or name.endswith(".down_proj")
            )
        ):
            total_size += _mxfp8_tensor_weight_size(param, weight_block_size)
        else:
            total_size += int(bytes_of_tensor(param))
    for _, buffer in model.named_buffers():
        total_size += int(bytes_of_tensor(buffer))
    return total_size


def patch_minimax_m3_attention(model: TransformerModel) -> TransformerModel:
    sparse_cfg = None
    text_config = model.text_config

    if hasattr(text_config, "sparse_attention_config"):
        sparse_cfg = text_config.sparse_attention_config
    layer_types = getattr(text_config, "layer_types", None)
    has_native_sparse_config = (
        layer_types is not None
        and any(layer_type == "minimax_m3_sparse" for layer_type in layer_types)
        and hasattr(text_config, "index_n_heads")
    )
    if sparse_cfg is None and has_native_sparse_config:
        sparse_cfg = {
            "use_sparse_attention": True,
            "sparse_attention_freq": [
                1 if layer_type == "minimax_m3_sparse" else 0
                for layer_type in layer_types
            ],
            "sparse_num_index_heads": text_config.index_n_heads,
            "sparse_index_dim": text_config.index_head_dim,
            "sparse_topk_blocks": text_config.index_topk_blocks,
            "sparse_block_size": text_config.index_block_size,
            "sparse_local_block": text_config.index_local_blocks,
        }
    if sparse_cfg is None or not sparse_cfg.get("use_sparse_attention", False):
        logger.info("No sparse_attention_config found, skipping M3 attention patch")
        return model

    sparse_attention_freq = sparse_cfg.get("sparse_attention_freq", [])
    num_indexer_heads = sparse_cfg.get("sparse_num_index_heads", 4)
    indexer_head_dim = sparse_cfg.get("sparse_index_dim", 128)
    indexer_rope_dim = getattr(text_config, "rotary_dim", 64)
    topk_blocks = sparse_cfg.get("sparse_topk_blocks", 16)
    block_size = sparse_cfg.get("sparse_block_size", 128)
    local_blocks = sparse_cfg.get("sparse_local_block", 1)

    hidden_size = text_config.hidden_size
    num_q_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", hidden_size // num_q_heads)

    tp_size = 1
    if model.parallel_group_manager is not None and model.parallel_group_manager.tp_group is not None:
        tp_size = model.parallel_group_manager.tp_group.world_size
    per_rank_q_heads = num_q_heads // tp_size
    per_rank_kv_heads = num_kv_heads // tp_size
    per_rank_indexer_heads = num_indexer_heads // tp_size if num_indexer_heads >= tp_size else num_indexer_heads

    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        candidates = [
            getattr(unwrapped, "language_model", None),
            getattr(getattr(unwrapped, "model", None), "language_model", None),
            getattr(getattr(unwrapped, "language_model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "layers"):
                unwrapped = candidate
                break
        else:
            model_attr = getattr(unwrapped, "model", None)
            if hasattr(model_attr, "layers"):
                unwrapped = model_attr
            else:
                logger.warning("Cannot find layers for M3 attention patch")
                return model

    for layer_idx, layer in enumerate(unwrapped.layers):
        self_attn = layer
        while hasattr(self_attn, "_inner"):
            self_attn = self_attn._inner
        if hasattr(self_attn, "self_attn"):
            self_attn = self_attn.self_attn

        is_sparse = (
            layer_idx < len(sparse_attention_freq) and sparse_attention_freq[layer_idx] == 1
        )

        wrapper = MiniMaxM3AttentionWrapper(
            original_module=self_attn,
            is_sparse_layer=is_sparse,
            hidden_size=hidden_size,
            num_q_heads=per_rank_q_heads,
            num_kv_heads=per_rank_kv_heads,
            head_dim=head_dim,
            num_indexer_heads=per_rank_indexer_heads,
            indexer_head_dim=indexer_head_dim,
            indexer_rope_dim=indexer_rope_dim,
            topk_blocks=topk_blocks,
            block_size=block_size,
            local_blocks=local_blocks,
        )

        parent = layer
        while hasattr(parent, "_inner") and hasattr(parent._inner, "self_attn"):
            parent = parent._inner
        if hasattr(parent, "self_attn"):
            parent.self_attn = wrapper
        else:
            logger.warning("Could not replace self_attn for layer %d", layer_idx)

    return model


@register_custom_model("minimax_m3_vl")
def _(model: TransformerModel):
    linear_quant_configs = model.model_config.quant_config.linear_configs
    quant_type = next(iter(linear_quant_configs.values())).quant_type if linear_quant_configs else None
    model = wrap_model(model)
    model = maybe_enable_mtp(model)
    _ensure_empty_visual_layers_for_reuse(model)
    model = maybe_reuse_layers(model)
    model = patch_minimax_m3_attention(model)
    model = _patch_minimax_m3_rmsnorm(model)
    model = patch_attention(model)
    model = patch_moe(
        model,
        lambda moe_config, module: MiniMaxM3MoELayer(
            moe_config,
            module,
            quant_type,
        ),
    )
    model = _patch_m3_moe_return_compat(model)
    model = quantize_model(model)
    model = shard_model(model)
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m3_vl",
        moe_module_name="MiniMaxM3VLSparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="num_local_experts",
        mtp_block_module_name="MiniMaxM3DecoderLayer",
        hf_config_patch_method=_patch_minimax_m3_hf_config,
        weight_size_estimator=estimate_minimax_m3_weight_size,
        language_layers_path_str="language_model.layers",
        language_module_path="language_model",
        visual_layers_module_path=_EMPTY_VISUAL_LAYERS_ATTR,
        visual_layers_path_str=_EMPTY_VISUAL_LAYERS_ATTR,
        moe_field_names_override={
            "shared_experts": "shared_experts",
        },
        custom_expert_module_type=None,
    )
)
