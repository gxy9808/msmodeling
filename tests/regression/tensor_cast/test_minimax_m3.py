import pytest
import torch

from tensor_cast.device import TEST_DEVICE
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.builtin_model.minimax_m3 import MiniMaxM3DenseMLPWrapper, MiniMaxM3FusedMoETensorCast


class _FakeDenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.swiglu_alpha = 1.702
        self.swiglu_limit = 7.0
        self.gate_up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)


class _FakeEpGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.all_reduce_calls = 0

    def all_reduce(self, input_):
        self.all_reduce_calls += 1
        return input_


def test_minimax_indexer_op_registered():
    assert hasattr(torch.ops.tensor_cast, "minimax_indexer"), (
        "minimax_indexer op not registered"
    )


def test_minimax_sparse_attention_op_registered():
    assert hasattr(torch.ops.tensor_cast, "minimax_sparse_attention"), (
        "minimax_sparse_attention op not registered"
    )


def test_minimax_indexer_meta_shape():
    idx_q = torch.empty(1, 4, 128, device="meta")
    idx_k = torch.empty(1, 1, 128, device="meta")
    seq_lens = torch.tensor([4096], device="meta")
    query_lens = torch.tensor([1], device="meta")
    block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

    topk_idx = torch.ops.tensor_cast.minimax_indexer(
        idx_q,
        idx_k,
        seq_lens,
        query_lens,
        block_table,
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
        hidden_size=6144,
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
        OpInvokeInfo._op_properties_functors,
        "__getitem__",
    ) or hasattr(OpInvokeInfo, "get_perf_properties"), (
        "OpInvokeInfo missing properties registry"
    )


def test_minimax_m3_fused_moe_does_not_ep_all_reduce_like_deepseek():
    fused_moe = MiniMaxM3FusedMoETensorCast.__new__(MiniMaxM3FusedMoETensorCast)
    torch.nn.Module.__init__(fused_moe)
    fused_moe.top_k = 1
    fused_moe.routed_scaling_factor = 1.0
    fused_moe.shared_experts = torch.nn.Identity()
    fused_moe.num_external_shared_experts = 0
    fused_moe.ep_group = _FakeEpGroup(world_size=2)

    hidden_states = torch.ones(2, 4)
    topk_indices = torch.zeros(2, 1, dtype=torch.long)
    topk_weights = torch.ones(2, 1)

    fused_moe.get_split_sizes = lambda num_tokens, top_k: ([], [], [], [])
    fused_moe.dispatch_tokens = lambda hidden, indices, *_: [hidden]
    fused_moe._run_routed_experts = lambda dispatched: dispatched
    fused_moe.combine_tokens = lambda routed, indices, *_: routed[0].unsqueeze(-2)
    fused_moe._run_shared_experts = lambda hidden: torch.full_like(hidden, 2.0)

    output = fused_moe(hidden_states, topk_indices, topk_weights)

    assert fused_moe.ep_group.all_reduce_calls == 0
    assert torch.equal(output, torch.full_like(hidden_states, 3.0))


def test_minimax_m3_dense_mlp_wrapper_uses_m3_swiglu_quant():
    wrapper = MiniMaxM3DenseMLPWrapper(_FakeDenseMLP()).to("meta")
    hidden_states = torch.empty(2, 4, device="meta")

    perf_model = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
        output = wrapper(hidden_states)

    result = runtime.table_averages()
    assert output.shape == hidden_states.shape
    assert "tensor_cast.m3_swiglu_quant.default" in result
    assert "aten.sigmoid.default" not in result
