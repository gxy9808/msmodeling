import pytest
import torch
from torch._inductor.compile_fx import fake_tensor_prop

from tensor_cast import ops  # noqa: F401
from tensor_cast.compilation.freezing_passes.grouped_matmul_swiglu_pass import GroupedMatmulSwigluPass
from tensor_cast.compilation.freezing_passes.sink_split_pass import SinkSplitPass
from tensor_cast.compilation.passes.merge_linear_pass import MergeLinearPass
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import QuantConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme, quantize_linear_modules
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.builtin_model.minimax_m3 import (
    MiniMaxM3DenseMLPWrapper,
    MiniMaxM3FusedMoETensorCast,
    MiniMaxM3MoeExpertMLP,
)

from .test_common import get_linear_quant_config


class _FakeDenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.swiglu_alpha = 1.702
        self.swiglu_limit = 7.0
        self.gate_up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)


class _FakeExpertsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self.intermediate_dim = 4
        self.swiglu_alpha = 1.702
        self.swiglu_limit = 7.0
        self.gate_up_proj = torch.nn.Parameter(torch.randn(1, 8, 4))
        self.down_proj = torch.nn.Parameter(torch.randn(1, 4, 4))


class _FakeEpGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.rank_in_group = 0
        self.all_reduce_calls = 0

    def all_reduce(self, input_):
        self.all_reduce_calls += 1
        return input_


class _FakeExperts:
    def call_expert(self, expert_idx, x, *args, **kwargs):
        return x


def _wrap_expert_fp8(expert: MiniMaxM3MoeExpertMLP) -> MiniMaxM3MoeExpertMLP:
    quant_config = QuantConfig()
    quant_config.linear_configs["*"] = get_linear_quant_config(
        LinearQuantType.FP8,
        dynamic_quant_granularity=QuantGranularity.PER_SAMPLE,
        dynamic_quant_scheme=QuantScheme.SYMMETRIC,
    )
    quantize_linear_modules(
        expert,
        TensorCastQuantLinear,
        quant_config,
        default_config_name="default",
        strip_module_fn=None,
    )
    return expert


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
    fused_moe._global_tp_size = 1
    fused_moe.ep_group = _FakeEpGroup(world_size=2)
    fused_moe.experts = _FakeExperts()

    hidden_states = torch.ones(2, 4)
    topk_indices = torch.zeros(2, 1, dtype=torch.long)
    topk_weights = torch.ones(2, 1)

    fused_moe.get_split_sizes = lambda num_tokens, top_k: ([], [], [], [])
    fused_moe.dispatch_tokens = lambda hidden, indices, *_: [hidden]
    fused_moe.combine_tokens = lambda routed, indices, *_: routed[0].unsqueeze(-2)
    fused_moe._run_shared_experts = lambda hidden: torch.full_like(hidden, 2.0)

    output = fused_moe(hidden_states, topk_indices, topk_weights)

    assert fused_moe.ep_group.all_reduce_calls == 0
    assert torch.equal(output, torch.full_like(hidden_states, 3.0))


def test_minimax_m3_dense_mlp_wrapper_uses_m3_swiglu_quant():
    wrapper = MiniMaxM3DenseMLPWrapper(_FakeDenseMLP()).to("meta")
    quantize_linear_modules(
        wrapper,
        TensorCastQuantLinear,
        QuantConfig(linear_configs={"*": get_linear_quant_config(LinearQuantType.FP8)}),
        default_config_name="default",
        strip_module_fn=None,
    )
    hidden_states = torch.empty(2, 4, device="meta")

    perf_model = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
        output = wrapper(hidden_states)

    result = runtime.table_averages()
    assert output.shape == hidden_states.shape
    assert "tensor_cast.m3_swiglu_quant.default" in result
    assert "aten.sigmoid.default" not in result


def test_minimax_m3_moe_expert_uses_fp8_linear_for_gate_up_and_down():
    expert = _wrap_expert_fp8(MiniMaxM3MoeExpertMLP(_FakeExpertsModule(), 0)).to("meta")
    hidden_states = torch.empty(2, 4, device="meta")

    perf_model = AnalyticPerformanceModel(TEST_DEVICE)
    with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
        output = expert(hidden_states)

    result = runtime.table_averages()
    assert output.shape == hidden_states.shape
    fp8_line = next(line for line in result.splitlines() if "tensor_cast.fp8_linear.default" in line)
    quant_line = next(
        line for line in result.splitlines() if "tensor_cast.dynamic_quantize_symmetric.default" in line
    )
    assert fp8_line.split()[-1] == "3"
    assert quant_line.split()[-1] == "2"
    assert "tensor_cast.m3_swiglu_quant.default" in result


def _run_m3_moe_fp8_freezing_passes(gm: torch.fx.GraphModule, inputs):
    fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
    MergeLinearPass()(gm)
    fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
    for _ in range(3):
        SinkSplitPass()(gm)
        fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
    GroupedMatmulSwigluPass()(gm)
    for _ in range(3):
        SinkSplitPass()(gm)


def test_minimax_m3_moe_fp8_freezing_passes_fuse_gmm_swiglu_and_down():
    """M3 expert graph should match DeepSeek: GMM+m3_swiglu_quant, then down GMM."""
    dq = torch.ops.tensor_cast.dynamic_quantize_symmetric.default
    fp8 = torch.ops.tensor_cast.fp8_linear.default
    gmm = torch.ops.tensor_cast.grouped_matmul_fp8.default
    gmm_m3 = torch.ops.tensor_cast.grouped_matmul_fp8_m3_swiglu_quant.default
    m3sq = torch.ops.tensor_cast.m3_swiglu_quant.default
    split = torch.ops.aten.split_with_sizes.default

    def expert_fwd(x, wg, wu, wd, wsg, wsu, wsd):
        qx, ascale = dq(x, dims=[-1], scale_dtype=torch.float32, out_dtype=torch.int8)
        gate = fp8(qx, wg, ascale, wsg, None, torch.bfloat16)
        up = fp8(qx, wu, ascale, wsu, None, torch.bfloat16)
        mid, mscale = m3sq(gate, up, 1.702, 7.0, 128)
        return fp8(mid, wd, mscale, wsd, None, torch.bfloat16)

    class _TwoExpertMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in (0, 1):
                for name, shape in (
                    ("wg", (4, 4)),
                    ("wu", (4, 4)),
                    ("wd", (4, 4)),
                    ("wsg", (4,)),
                    ("wsu", (4,)),
                    ("wsd", (4,)),
                ):
                    param = torch.randn(shape) if len(shape) > 1 else torch.ones(shape)
                    setattr(self, f"{name}{i}", torch.nn.Parameter(param))

        def forward(self, x):
            cat = torch.cat([x[:3], x[3:]], dim=0)
            x0, x1 = split(cat, [3, 2], 0)
            o0 = expert_fwd(x0, self.wg0, self.wu0, self.wd0, self.wsg0, self.wsu0, self.wsd0)
            o1 = expert_fwd(x1, self.wg1, self.wu1, self.wd1, self.wsg1, self.wsu1, self.wsd1)
            return torch.cat([o0, o1], dim=0)

    inputs = [torch.empty(5, 4)]
    gm = torch.fx.symbolic_trace(_TwoExpertMoE())
    _run_m3_moe_fp8_freezing_passes(gm, inputs)

    def _count(target):
        return sum(1 for node in gm.graph.nodes if node.target == target)

    assert _count(fp8) == 0
    assert _count(m3sq) == 0
    assert _count(gmm_m3) == 1
    assert _count(gmm) == 1
