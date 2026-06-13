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

import math
from typing import Optional, Union, List
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


class MiniMaxM3StandardRMSNorm(nn.Module):
    """Standard RMSNorm: weight * rms_norm(hidden)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MiniMaxM3MultiHeadRMSNorm(nn.Module):
    """Per-head RMSNorm with optional layernorm-1p (Gemma-style) bias."""

    def __init__(self, num_heads, head_dim, eps=1e-6, apply_layernorm_1p=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim, dtype=torch.float32))
        self.variance_epsilon = eps
        self.apply_layernorm_1p = apply_layernorm_1p

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, self.num_heads, self.head_dim).to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.apply_layernorm_1p:
            hidden_states = hidden_states * (1.0 + self.weight[None, ...])
        else:
            hidden_states = hidden_states * self.weight[None, ...]
        hidden_states = hidden_states.view(-1, self.num_heads * self.head_dim)
        return hidden_states.to(orig_dtype)


def _get_norm_class(use_gemma_norm):
    if use_gemma_norm:
        return MiniMaxM3RMSNorm
    return MiniMaxM3StandardRMSNorm


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


def _get_sparse_attention_layer_ids(config):
    sparse_cfg = getattr(config, "sparse_attention_config", None)
    if sparse_cfg is None:
        return list(range(config.num_hidden_layers)), []
    use_sparse = sparse_cfg.get("use_sparse_attention", False) if isinstance(sparse_cfg, dict) else getattr(sparse_cfg, "use_sparse_attention", False)
    if not use_sparse:
        return list(range(config.num_hidden_layers)), []
    freq = sparse_cfg.get("sparse_attention_freq", []) if isinstance(sparse_cfg, dict) else getattr(sparse_cfg, "sparse_attention_freq", [])
    dense_ids = [i for i, f in enumerate(freq) if f == 0]
    sparse_ids = [i for i, f in enumerate(freq) if f != 0]
    if len(freq) < config.num_hidden_layers:
        dense_ids += list(range(len(freq), config.num_hidden_layers))
    return dense_ids, sparse_ids


def _get_sparse_disable_value_layer_ids(config):
    sparse_cfg = getattr(config, "sparse_attention_config", None)
    if sparse_cfg is None:
        return set()
    if isinstance(sparse_cfg, dict):
        disable_list = sparse_cfg.get("sparse_disable_index_value", [])
        if not disable_list:
            disable_list = sparse_cfg.get("sparse_disable_value", [])
    else:
        disable_list = getattr(sparse_cfg, "sparse_disable_index_value", [])
        if not disable_list:
            disable_list = getattr(sparse_cfg, "sparse_disable_value", [])
    return set(i for i, v in enumerate(disable_list) if v)


class MiniMaxM3Attention(nn.Module):
    """Multi-headed attention with QK normalization, partial rotary embeddings,
    and optional sparse attention index branch.

    Supports two modes selected by ``is_sparse_attention_layer``:

    * Dense (default): standard QKV attention.
    * Sparse: extra index branch (index_q/k/v_proj + index_o_proj) whose
      outputs are computed alongside the dense attention and summed into the
      dense output. In a production inference framework this would be
      dispatched to a sparse attention backend; here we compute the index
      branch as a parallel attention path with reduced dimensionality.
    """

    def __init__(self, config, layer_idx, is_sparse_attention_layer=False, disable_index_value=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        self.is_sparse_attention_layer = is_sparse_attention_layer
        self.disable_index_value = is_sparse_attention_layer and disable_index_value

        self.qk_norm_type = getattr(config, "qk_norm_type", "per_head")
        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)
        self.attention_output_gate = getattr(config, "attention_output_gate", False)

        if self.attention_output_gate:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim * 2, bias=False
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_qk_norm(config)

        if self.is_sparse_attention_layer:
            self._init_sparse_index_branch(config)

    def _init_qk_norm(self, config):
        norm_cls = _get_norm_class(self.use_gemma_norm)

        if self.qk_norm_type == "per_layer":
            self.q_norm = norm_cls(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = norm_cls(self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
        elif self.qk_norm_type == "per_head":
            self.q_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
        elif self.qk_norm_type == "multi_head":
            self.q_norm = MiniMaxM3MultiHeadRMSNorm(
                self.num_heads, self.head_dim,
                eps=config.rms_norm_eps,
                apply_layernorm_1p=self.use_gemma_norm,
            )
            self.k_norm = MiniMaxM3MultiHeadRMSNorm(
                self.num_key_value_heads, self.head_dim,
                eps=config.rms_norm_eps,
                apply_layernorm_1p=self.use_gemma_norm,
            )
        else:
            raise ValueError(f"Invalid qk_norm_type: {self.qk_norm_type}")

    def _init_sparse_index_branch(self, config):
        sparse_cfg = config.sparse_attention_config
        if isinstance(sparse_cfg, dict):
            self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
            self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        else:
            self.total_idx_heads = sparse_cfg.sparse_num_index_heads
            self.idx_head_dim = sparse_cfg.sparse_index_dim

        self.index_q_proj = nn.Linear(
            self.hidden_size, self.total_idx_heads * self.idx_head_dim, bias=False
        )
        self.index_k_proj = nn.Linear(
            self.hidden_size, self.idx_head_dim, bias=False
        )
        if self.disable_index_value:
            self.index_v_proj = None
            self.index_o_proj = None
        else:
            self.index_v_proj = nn.Linear(
                self.hidden_size, self.idx_head_dim, bias=False
            )
            self.index_o_proj = nn.Linear(
                self.total_idx_heads * self.idx_head_dim, self.hidden_size, bias=False
            )

        norm_cls = _get_norm_class(self.use_gemma_norm)
        self.index_q_norm = norm_cls(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = norm_cls(self.idx_head_dim, eps=config.rms_norm_eps)

    def _qk_norm(self, query_states, key_states):
        if self.qk_norm_type == "per_layer":
            orig_q_shape = query_states.shape
            orig_k_shape = key_states.shape
            q_flat = query_states.contiguous().reshape(-1, self.num_heads * self.head_dim)
            k_flat = key_states.contiguous().reshape(-1, self.num_key_value_heads * self.head_dim)
            q_normed = self.q_norm(q_flat).reshape(orig_q_shape)
            k_normed = self.k_norm(k_flat).reshape(orig_k_shape)
        elif self.qk_norm_type == "per_head":
            orig_q_shape = query_states.shape
            orig_k_shape = key_states.shape
            q_flat = query_states.contiguous().reshape(-1, self.head_dim)
            k_flat = key_states.contiguous().reshape(-1, self.head_dim)
            q_normed = self.q_norm(q_flat).reshape(orig_q_shape)
            k_normed = self.k_norm(k_flat).reshape(orig_k_shape)
        elif self.qk_norm_type == "multi_head":
            orig_q_shape = query_states.shape
            orig_k_shape = key_states.shape
            q_flat = query_states.contiguous().reshape(-1, self.num_heads * self.head_dim)
            k_flat = key_states.contiguous().reshape(-1, self.num_key_value_heads * self.head_dim)
            q_normed = self.q_norm(q_flat).reshape(orig_q_shape)
            k_normed = self.k_norm(k_flat).reshape(orig_k_shape)
        else:
            raise ValueError(f"Invalid qk_norm_type: {self.qk_norm_type}")
        return q_normed, k_normed

    def _index_qk_norm(self, idx_q, idx_k):
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape
        idx_q = self.index_q_norm(idx_q.contiguous().reshape(-1, self.idx_head_dim)).reshape(idx_q_shape)
        idx_k = self.index_k_norm(idx_k.contiguous().reshape(-1, self.idx_head_dim)).reshape(idx_k_shape)
        return idx_q, idx_k

    def _sparse_index_forward(self, hidden_states, cos, sin, input_shape):
        idx_q = self.index_q_proj(hidden_states)
        idx_k = self.index_k_proj(hidden_states)
        if not self.disable_index_value:
            idx_v = self.index_v_proj(hidden_states)
        else:
            idx_v = None

        idx_q = idx_q.view(*input_shape, -1, self.idx_head_dim)
        idx_k = idx_k.view(*input_shape, -1, self.idx_head_dim)

        idx_q, idx_k = self._index_qk_norm(idx_q, idx_k)

        idx_q = idx_q.transpose(1, 2)
        idx_k = idx_k.transpose(1, 2)
        if idx_v is not None:
            idx_v = idx_v.view(*input_shape, -1, self.idx_head_dim).transpose(1, 2)

        idx_q, idx_k = _apply_rotary_pos_emb(idx_q, idx_k, cos, sin)

        if idx_v is not None:
            idx_scores = torch.matmul(idx_q, idx_k.transpose(2, 3)) * (self.idx_head_dim ** -0.5)
            seq_len = idx_scores.shape[-1]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=idx_scores.device, dtype=idx_scores.dtype),
                diagonal=1,
            )
            idx_scores = idx_scores + causal_mask
            idx_weights = F.softmax(idx_scores, dim=-1, dtype=torch.float32).to(idx_q.dtype)
            idx_attn = torch.matmul(idx_weights, idx_v)
            idx_attn = idx_attn.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            idx_output = self.index_o_proj(idx_attn)
        else:
            idx_output = None

        return idx_output

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

        if self.attention_output_gate:
            q_gate = self.q_proj(hidden_states)
            q_gate = q_gate.view(*input_shape, -1, 2, self.head_dim)
            query_states = q_gate[..., 0, :].reshape(*input_shape, -1)
            gate = q_gate[..., 1, :].reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states)
            gate = None

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(*input_shape, -1, self.head_dim)
        key_states = key_states.view(*input_shape, -1, self.head_dim)

        query_states, key_states = self._qk_norm(query_states, key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
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

        if self.attention_output_gate and gate is not None:
            gate = torch.sigmoid(gate.float())
            attn_output = (attn_output * gate).to(attn_output.dtype)

        attn_output = self.o_proj(attn_output)

        if self.is_sparse_attention_layer:
            idx_output = self._sparse_index_forward(hidden_states, cos, sin, input_shape)
            if idx_output is not None:
                attn_output = attn_output + idx_output

        return attn_output, attn_weights


class MiniMaxM3DecoderLayer(nn.Module):
    """MiniMax Decoder Layer with MoE and optional sparse attention support.

    The attention block can be either dense or sparse depending on
    config.sparse_attention_config:

    * If sparse_attention_config is None (or absent), all layers run
      dense attention.
    * If present, the per-layer dense/sparse split is read from
      sparse_attention_config['sparse_attention_freq'].
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        sparse_attention_config = getattr(config, "sparse_attention_config", None)
        use_sparse = False
        if sparse_attention_config is not None:
            if isinstance(sparse_attention_config, dict):
                use_sparse = sparse_attention_config.get("use_sparse_attention", False)
            else:
                use_sparse = getattr(sparse_attention_config, "use_sparse_attention", False)
        if use_sparse:
            _, sparse_layer_ids = _get_sparse_attention_layer_ids(config)
            is_sparse_attention_layer = layer_idx in sparse_layer_ids
            disable_value_layer_ids = _get_sparse_disable_value_layer_ids(config)
            disable_index_value = layer_idx in disable_value_layer_ids
        else:
            is_sparse_attention_layer = False
            disable_index_value = False

        self.self_attn = MiniMaxM3Attention(
            config, layer_idx,
            is_sparse_attention_layer=is_sparse_attention_layer,
            disable_index_value=disable_index_value,
        )

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        self.is_moe_layer = (
            moe_layer_freq[layer_idx] != 0 if moe_layer_freq is not None else True
        )
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3SparseMoeBlock(config)
        else:
            self.mlp = MiniMaxM3MLP(config, intermediate_size=config.dense_intermediate_size)

        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)
        norm_cls = _get_norm_class(self.use_gemma_norm)
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

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


def _get_mtp_layer_indices(config):
    """Return set of layer indices that belong to MTP (multi-token prediction) modules."""
    num_mtp = getattr(config, "num_mtp_modules", 0)
    if num_mtp <= 0:
        return set()
    return set(range(config.num_hidden_layers, config.num_hidden_layers + num_mtp))


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
        use_gemma_norm = getattr(text_config, "use_gemma_norm", False)
        norm_cls = _get_norm_class(use_gemma_norm)
        self.norm = norm_cls(text_config.hidden_size, eps=text_config.rms_norm_eps)
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
