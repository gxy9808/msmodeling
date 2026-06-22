from typing import Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("minimax_indexer")
def _(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """
    MiniMax-M3 indexer fused op.

    Boundary:
      index Q/K block score -> top-k block indices.

    Index q/k projections, norm, and RoPE are explicit ops in
    MiniMaxM3AttentionWrapper.forward.

    Performance formula: see M3-msmodeling.md section 4.1.
    """
    total_tokens = idx_q.shape[0]
    num_indexer_heads = idx_q.shape[1]
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
    hidden_size: int,
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
      sparse QK/PV attention body.

    Main q/k/v projection, QK norm, RoPE, cache write, and o_proj are explicit
    ops in MiniMaxM3AttentionWrapper.forward.

    Performance formula: see M3-msmodeling.md section 4.2.
    """
    return torch.empty_like(query).contiguous()
