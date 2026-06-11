# coding=utf-8
# Copyright 2025 the MiniMax AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MiniMax-M3 model."""

from typing import Optional, Union
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from typing import Unpack
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.output_capturing import OutputRecorder

from .configuration_minimax_m3_vl import MiniMaxM3VLConfig


class MiniMaxM3RMSNorm(nn.Module):
    """Gemma-style RMSNorm: weight * rms(hidden) where weight is biased by 1."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (1.0 + self.weight) * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SwigluOAIAndMul(torch.nn.Module):
    """
    SwiGLU variant used by MiniMax-M3: swigluoai with alpha and limit.
    f(x, y) = (x * sigmoid(alpha * x)) * y, clamped to [-limit, limit]
    """

    def __init__(self, alpha=1.702, limit=7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        gate = gate * torch.sigmoid(self.alpha * gate)
        if self.limit is not None:
            gate = torch.clamp(gate, -self.limit, self.limit)
        return gate * up


class MiniMaxM3MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act == "silu":
            self.act_fn = ACT2FN["silu"]
        elif config.hidden_act == "swigluoai":
            self.act_fn = SwigluOAIAndMul(
                alpha=config.swiglu_alpha,
                limit=config.swiglu_limit,
            )
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(torch.cat([self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1)))


class MiniMaxM3Experts(nn.ModuleList):
    """ModuleList of routed experts."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        for _ in range(self.num_experts):
            self.append(MiniMaxM3MLP(config, intermediate_size=config.intermediate_size))

    def forward(self, hidden_states, top_k_idx, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_idx, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states


class MiniMaxM3SparseMoeBlock(nn.Module):
    """Sparse MoE with sigmoid routing, e_score_correction_bias, and shared experts."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.use_routing_bias = getattr(config, "use_routing_bias", False)

        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM3Experts(config)

        if self.n_shared_experts is not None:
            self.shared_experts = MiniMaxM3MLP(
                config,
                intermediate_size=config.shared_intermediate_size * self.n_shared_experts,
            )
        else:
            self.shared_experts = None

        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(config.num_local_experts, dtype=torch.float32)
            )
        else:
            self.e_score_correction_bias = None

    def _route_tokens(self, router_logits):
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights
        if self.e_score_correction_bias is not None:
            scores_for_choice = scores_for_choice + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states_flat)
        top_k_index, top_k_weights = self._route_tokens(router_logits)

        routed_out = self.experts(hidden_states_flat, top_k_index, top_k_weights)
        if self.routed_scaling_factor != 1.0:
            routed_out = routed_out * self.routed_scaling_factor

        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states_flat)
            final_hidden_states = routed_out + shared_out
        else:
            final_hidden_states = routed_out

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class MiniMaxM3Attention(nn.Module):
    """Multi-headed attention with per-head QK LayerNorm and partial rotary embeddings."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.rotary_dim = config.rotary_dim
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Per-head QK LayerNorm (M3 uses per_head, unlike M2's full-head norm)
        self.q_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Per-head QK LayerNorm: reshape to (..., num_heads, head_dim) then norm
        query_states = query_states.view(*input_shape, -1, self.head_dim)
        key_states = key_states.view(*input_shape, -1, self.head_dim)

        # Per-head norm
        q_flat = query_states.reshape(-1, self.head_dim).contiguous()
        k_flat = key_states.reshape(-1, self.head_dim).contiguous()
        q_normed = self.q_norm(q_flat).reshape(query_states.shape)
        k_normed = self.k_norm(k_flat).reshape(key_states.shape)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        query_states = q_normed.transpose(1, 2)
        key_states = k_normed.transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = _eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = MiniMaxM3Attention(config, layer_idx)

        # MoE layer frequency: layers with moe_layer_freq != 0 use MoE, else dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        self.is_moe_layer = (
            moe_layer_freq[layer_idx] != 0 if moe_layer_freq is not None else True
        )
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(config)
        else:
            self.mlp = MiniMaxM3MLP(config, intermediate_size=config.dense_intermediate_size)

        self.input_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            hidden_states, _ = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MiniMaxM3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.rope_type = "linear"  # required by @dynamic_rope_update
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.attention_scaling = 1.0

        # MiniMax M3 uses standard linear RoPE (no scaling).
        # Manually compute inv_freq to avoid depending on transformers-5.3's
        # rope_parameters / ROPE_INIT_FUNCTIONS API which requires fields
        # absent from the M3 config.
        rope_theta = getattr(config, "rope_theta", 5000000.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class MiniMaxM3PreTrainedModel(PreTrainedModel):
    config_class = MiniMaxM3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(MiniMaxM3SparseMoeBlock, index=1),
        "hidden_states": MiniMaxM3DecoderLayer,
        "attentions": MiniMaxM3Attention,
    }


def _patch_text_config(text_config):
    """Restore attributes lost when MiniMaxVLBaseConfig.__post_init__ re-coerces
    text_config via _coerce_sub_config -> PretrainedConfig(**dict).

    PretrainedConfig.__init__ only preserves attributes declared in its own
    __init__ signature; everything else (rope_scaling, attention_dropout, etc.)
    is silently dropped.  We read the original __dict__ to recover them.
    """
    # Collect all original keys that exist in the config's underlying data
    # but are not yet accessible via getattr.
    orig_dict = getattr(text_config, "__dict__", {})
    raw_dict = getattr(text_config, "to_dict", lambda: orig_dict)()
    for key in dir(text_config):
        if key.startswith("_"):
            continue
        try:
            getattr(text_config, key)
        except AttributeError:
            if key in raw_dict:
                try:
                    setattr(text_config, key, raw_dict[key])
                except AttributeError:
                    pass
    return text_config


@auto_docstring
class MiniMaxM3Model(MiniMaxM3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # MiniMaxM3VLConfig wraps text_config inside a vision-language config;
        # resolve the actual text backbone config.
        text_config = config.text_config if hasattr(config, "text_config") else config
        _patch_text_config(text_config)

        self.padding_idx = text_config.pad_token_id if hasattr(text_config, "pad_token_id") else None
        self.vocab_size = text_config.vocab_size

        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniMaxM3DecoderLayer(text_config, layer_idx) for layer_idx in range(text_config.num_hidden_layers)]
        )
        self.norm = MiniMaxM3RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = MiniMaxM3RotaryEmbedding(config=text_config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class MiniMaxM3SparseForCausalLM(MiniMaxM3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxM3Model(config)
        # Resolve text backbone config for lm_head
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_router_logits=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class MiniMaxM3SparseForSequenceClassification(GenericForSequenceClassification, MiniMaxM3PreTrainedModel):
    pass


class MiniMaxM3SparseForTokenClassification(GenericForTokenClassification, MiniMaxM3PreTrainedModel):
    pass


class MiniMaxM3SparseForQuestionAnswering(GenericForQuestionAnswering, MiniMaxM3PreTrainedModel):
    pass


__all__ = [
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForQuestionAnswering",
    "MiniMaxM3Model",
    "MiniMaxM3PreTrainedModel",
    "MiniMaxM3SparseForSequenceClassification",
    "MiniMaxM3SparseForTokenClassification",
]
