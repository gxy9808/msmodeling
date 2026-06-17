import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _get_eps(norm: torch.nn.Module) -> float:
    if hasattr(norm, "variance_epsilon"):
        return norm.variance_epsilon
    if hasattr(norm, "eps"):
        return norm.eps
    raise AttributeError(
        f"{type(norm).__name__} has neither 'variance_epsilon' nor 'eps' attribute"
    )


class GemmaRMSNormFusedWrapper(torch.nn.Module):
    """Wrapper that replaces GemmaRMSNorm with fused rms_norm and add_rms_norm2 ops."""

    def __init__(self, original_norm: torch.nn.Module):
        super().__init__()
        self._inner = original_norm
        self.weight = original_norm.weight
        self.eps = _get_eps(original_norm)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        effective_weight = 1.0 + self.weight
        if residual is not None:
            out, residual_out = torch.ops.tensor_cast.add_rms_norm2(
                x, residual, effective_weight, self.eps,
            )
            return out, residual_out
        return torch.ops.tensor_cast.rms_norm(x, effective_weight, self.eps)


class RMSNormFusedWrapper(torch.nn.Module):
    """Wrapper that replaces RMSNorm with fused rms_norm op."""

    def __init__(self, original_norm: torch.nn.Module, is_gemma: bool = False):
        super().__init__()
        self._inner = original_norm
        self.weight = original_norm.weight
        self.eps = _get_eps(original_norm)
        self.is_gemma = is_gemma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_gemma:
            effective_weight = 1.0 + self.weight
        else:
            effective_weight = self.weight
        return torch.ops.tensor_cast.rms_norm(x, effective_weight, self.eps)


def _fused_decoder_layer_forward(self, hidden_states, **kwargs):
    """Replacement forward that fuses residual+norm into add_rms_norm2."""
    residual = hidden_states
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

    attn_out = self.self_attn(hidden_states, **kwargs)
    if isinstance(attn_out, tuple):
        hidden_states = attn_out[0]
    else:
        hidden_states = attn_out

    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    hidden_states = self.mlp(hidden_states)

    hidden_states = hidden_states + residual
    return hidden_states


class MiniMaxM3AttentionWrapper(torch.nn.Module):
    """Wrapper for MiniMax-M3 attention that routes dense/sparse layers.

    Dense layers use the standard tensor_cast attention path.
    Sparse layers call minimax_indexer + minimax_sparse_attention.
    For sparse layers, Q/K/index Q/K norms are explicitly called as rms_norm ops
    so they appear as independent operators matching profiling's 234 calls.
    """

    def __init__(
        self,
        original_module: torch.nn.Module,
        is_sparse_layer: bool,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_indexer_heads: int,
        indexer_head_dim: int,
        indexer_rope_dim: int,
        topk_blocks: int,
        block_size: int,
        local_blocks: int,
        rotary_dim: int = 64,
    ):
        super().__init__()
        self._inner = original_module
        self.is_sparse_layer = is_sparse_layer
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_indexer_heads = num_indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_rope_dim = indexer_rope_dim
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.rotary_dim = rotary_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attention_meta = kwargs.get("attention_meta", None)

        if not self.is_sparse_layer or attention_meta is None:
            return self._inner(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        seq_lens = attention_meta.seq_lens
        query_lens = attention_meta.query_lens
        block_table = attention_meta.block_table_tensor

        inner = self._inner
        while hasattr(inner, "_inner"):
            inner = inner._inner

        # --- Q norm + K norm (all 60 layers have these) ---
        q_tensor = hidden_states[:, :self.num_q_heads * self.head_dim]
        k_tensor = hidden_states[:, :self.num_kv_heads * self.head_dim]
        q_normed = inner.q_norm(q_tensor)
        k_normed = inner.k_norm(k_tensor)

        # --- Q/K fused_rope (all layers, matching NPU InterleaveRope profiling) ---
        # sglang NPU: self.rotary_emb(positions, q, k) -> fused_rope_qk_mqa
        position_embeddings = kwargs.get("position_embeddings", None)
        fused_rope_residual = torch.zeros_like(hidden_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_3d = q_normed.view(-1, self.num_q_heads, self.head_dim)
            k_3d = k_normed.view(-1, self.num_kv_heads, self.head_dim)
            cos_2d = cos.reshape(-1, cos.shape[-1])
            sin_2d = sin.reshape(-1, sin.shape[-1])
            cos_sin = torch.cat([cos_2d, sin_2d], dim=-1)
            q_rope, k_rope = torch.ops.tensor_cast.fused_rope(
                q_3d, k_3d, cos_sin, self.rotary_dim, True
            )

        # --- Index Q norm + Index K norm (sparse layers only, via self.indexer) ---
        indexer = getattr(inner, "indexer", None)
        if indexer is not None:
            idx_q_tensor = hidden_states[:, :self.num_indexer_heads * self.indexer_head_dim]
            idx_k_tensor = hidden_states[:, :1 * self.indexer_head_dim]
            idx_q_normed = indexer.q_norm(idx_q_tensor)
            idx_k_normed = indexer.k_norm(idx_k_tensor)

            # --- Index Q/K fused_rope (sparse layers only) ---
            # sglang NPU: self.index_rotary_emb(positions, idx_q, idx_k) -> fused_rope_qk_mqa
            if position_embeddings is not None:
                cos, sin = position_embeddings
                idx_cos = cos[..., :self.indexer_head_dim].reshape(-1, self.indexer_head_dim)
                idx_sin = sin[..., :self.indexer_head_dim].reshape(-1, self.indexer_head_dim)
                idx_cos_sin = torch.cat([idx_cos, idx_sin], dim=-1)
                idx_q_3d = idx_q_normed.view(-1, self.num_indexer_heads, self.indexer_head_dim)
                idx_k_3d = idx_k_normed.view(-1, 1, self.indexer_head_dim)
                idx_q_rope, idx_k_rope = torch.ops.tensor_cast.fused_rope(
                    idx_q_3d, idx_k_3d, idx_cos_sin, self.rotary_dim, True
                )

        # --- Indexer + Sparse Attention ---
        topk_idx = torch.ops.tensor_cast.minimax_indexer(
            hidden_states,
            seq_lens,
            query_lens,
            block_table,
            hidden_size=self.hidden_size,
            num_indexer_heads=self.num_indexer_heads,
            indexer_head_dim=self.indexer_head_dim,
            indexer_rope_dim=self.indexer_rope_dim,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
        )

        query = hidden_states
        key_cache = kwargs.get("kv_cache", None)
        if key_cache is not None and isinstance(key_cache, (list, tuple)):
            key_cache_tensor = key_cache[0]
            value_cache_tensor = key_cache[1]
        else:
            key_cache_tensor = key_cache
            value_cache_tensor = key_cache

        out = torch.ops.tensor_cast.minimax_sparse_attention(
            query,
            key_cache_tensor,
            value_cache_tensor,
            topk_idx,
            seq_lens,
            query_lens,
            block_table,
            hidden_size=self.hidden_size,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
        )

        return out, None
