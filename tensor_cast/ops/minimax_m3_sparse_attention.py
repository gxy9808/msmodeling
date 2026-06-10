from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("minimax_indexer")
def _(
    hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    hidden_size: int,
    num_indexer_heads: int,
    indexer_head_dim: int,
    indexer_rope_dim: int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """
    MiniMax-M3 indexer fused op.

    Boundary:
      hidden -> index_q_proj/index_k_proj -> norm -> RoPE -> index K cache write
      -> index QK block score -> top-k block indices.

    Performance formula: see M3-msmodeling.md section 4.1.
    """
    total_tokens = hidden_states.shape[0]
    return torch.empty(
        (total_tokens, num_indexer_heads, topk_blocks),
        dtype=torch.int32,
        device="meta",
    )


@register_tensor_cast_op("minimax_sparse_attention")
def _(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    topk_blocks: int,
    block_size: int,
    local_blocks: int,
) -> torch.Tensor:
    """
    MiniMax-M3 sparse attention fused op.

    Boundary:
      read Q + selected K/V cache -> sparse QK/PV attention -> output O.

    Performance formula: see M3-msmodeling.md section 4.2.
    """
    return torch.empty_like(query).contiguous()
