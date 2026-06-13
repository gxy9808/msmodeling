import logging

import torch
from tensor_cast.transformers.transformations import (
    maybe_enable_mtp,
    maybe_reuse_layers,
    patch_attention,
    patch_moe,
    quantize_model,
    shard_model,
    wrap_model,
)

from ..custom_model_registry import (
    ModelProfile,
    register_custom_model,
    register_model_profile,
)
from ..model import TransformerModel
from ...layers.minimax_m3_attention import MiniMaxM3AttentionWrapper

logger = logging.getLogger(__name__)


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


def patch_minimax_m3_attention(model: TransformerModel) -> TransformerModel:
    sparse_cfg = None
    text_config = model.text_config

    if hasattr(text_config, "sparse_attention_config"):
        sparse_cfg = text_config.sparse_attention_config
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
        language_model = unwrapped
        if hasattr(unwrapped, "language_model"):
            language_model = unwrapped.language_model
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            unwrapped = language_model.model
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
    model = wrap_model(model)
    model = maybe_enable_mtp(model)
    model = maybe_reuse_layers(model)
    model = patch_minimax_m3_attention(model)
    model = patch_attention(model)
    model = patch_moe(model)
    model = _patch_m3_moe_return_compat(model)
    model = quantize_model(model)
    model = shard_model(model)
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m3_vl",
        moe_module_name="MiniMaxM3SparseMoeBlock",
        moe_gate_returns_raw_logits=True,
        moe_num_experts_key="num_local_experts",
        mtp_block_module_name="MiniMaxM3DecoderLayer",
        hf_config_patch_method=_patch_minimax_m3_hf_config,
        language_layers_path_str="model.layers",
        language_module_path="model",
        moe_field_names_override={
            "shared_experts": "shared_experts",
        },
        custom_expert_module_type=MiniMaxM3ExpertMLP,
    )
)
