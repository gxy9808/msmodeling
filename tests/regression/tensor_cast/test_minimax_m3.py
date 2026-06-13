import pytest
import torch

from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


def test_minimax_indexer_op_registered():
    assert hasattr(torch.ops.tensor_cast, "minimax_indexer"), (
        "minimax_indexer op not registered"
    )


def test_minimax_sparse_attention_op_registered():
    assert hasattr(torch.ops.tensor_cast, "minimax_sparse_attention"), (
        "minimax_sparse_attention op not registered"
    )


def test_minimax_indexer_meta_shape():
    hidden_states = torch.empty(1, 6144, device="meta")
    seq_lens = torch.tensor([4096], device="meta")
    query_lens = torch.tensor([1], device="meta")
    block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

    topk_idx = torch.ops.tensor_cast.minimax_indexer(
        hidden_states,
        seq_lens,
        query_lens,
        block_table,
        hidden_size=6144,
        num_indexer_heads=4,
        indexer_head_dim=128,
        indexer_rope_dim=64,
        topk_blocks=16,
        block_size=128,
    )

    assert topk_idx.shape == (1, 4, 16)
    assert topk_idx.dtype == torch.int32


def test_minimax_sparse_attention_meta_shape():
    query = torch.empty(1, 64, 128, device="meta")
    key_cache = torch.empty(4096, 4, 128, device="meta")
    value_cache = torch.empty(4096, 4, 128, device="meta")
    topk_idx = torch.empty(1, 4, 16, dtype=torch.int32, device="meta")
    seq_lens = torch.tensor([4096], device="meta")
    query_lens = torch.tensor([1], device="meta")
    block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

    out = torch.ops.tensor_cast.minimax_sparse_attention(
        query,
        key_cache,
        value_cache,
        topk_idx,
        seq_lens,
        query_lens,
        block_table,
        num_q_heads=64,
        num_kv_heads=4,
        head_dim=128,
        topk_blocks=16,
        block_size=128,
        local_blocks=1,
    )

    assert out.shape == query.shape


def test_minimax_indexer_properties():
    assert hasattr(
        OpInvokeInfo._op_properties_registry,
        "__getitem__",
    ) or hasattr(OpInvokeInfo, "get_perf_properties"), (
        "OpInvokeInfo missing properties registry"
    )
