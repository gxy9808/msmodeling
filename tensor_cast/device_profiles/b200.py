import torch

from ..device import (
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)
from ..utils import DTYPE_FP4, DTYPE_FP8


class B200:
    STATIC_COST = StaticCost(
        mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=5 * 1e-6
    )

    # NVLink 5.0 switch-based full mesh, 8-GPU per node
    INTERCONNECT = CommGrid(
        grid=torch.arange(64 * 8).reshape(64, 8),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50 * 1e9, latency_s=10 * 1e-6, comm_efficiency=0.7
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=900 * 1e9,
                latency_s=1.0 * 1e-6,
                comm_efficiency=0.85,
                type=InterconnectType.FULL_MESH,
            ),
        },
    )

    B200_192G = DeviceProfile(
        name="B200",
        vendor="NVIDIA",
        mma_ops={
            # Blackwell Tensor Core peak FLOPS (dense, per GPU)
            # Source: NVIDIA DGX B200 spec — FP8 sparse=72 PFLOPS → dense=36 PFLOPS,
            # BF16 dense = FP8 dense / 2 = 2250 TOPS, FP32 dense = BF16 dense / 2 = 1125 TOPS
            # FP4 dense = FP8 dense * 2 = 9000 TOPS
            torch.float32: 1125 * 1e12,
            torch.bfloat16: 2250 * 1e12,
            torch.half: 2250 * 1e12,
            DTYPE_FP8: 4500 * 1e12,
            torch.int8: 4500 * 1e12,
            DTYPE_FP4: 9000 * 1e12,
        },
        gp_ops={
            # CUDA Core peak FLOPS (per GPU)
            # 128 SM * 128 FP32 cores/SM * 2 ops/clock * 2.1 GHz ≈ 68.7 TOPS
            # BF16/FP16 CUDA Core ≈ 2x FP32 ≈ 137 TOPS
            torch.float32: 68.7 * 1e12,
            torch.bfloat16: 137 * 1e12,
            torch.half: 137 * 1e12,
        },
        memory_size_bytes=192 * (1024**3),
        # HBM3e 8 TB/s per GPU (DGX B200 spec: 64 TB/s total / 8 GPUs)
        memory_bandwidth_bytes_ps=8.0 * (1024**4),
        # Decode场景MoE小batch GEMM利用率较低，需micro-benchmark回填
        compute_efficiency=0.55,
        # Decode场景随机访存多，KV cache非连续访问
        memory_efficiency=0.60,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )
