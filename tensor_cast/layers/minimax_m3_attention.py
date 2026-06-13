import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MiniMaxM3AttentionWrapper(torch.nn.Module):
    """Wrapper for MiniMax-M3 attention that routes dense/sparse layers.

    Dense layers use the standard tensor_cast attention path.
    Sparse layers call minimax_indexer + minimax_sparse_attention.
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
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
        )

        return out, None
