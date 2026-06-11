import torch

from ..device import (
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)
from ..utils import DTYPE_FP4


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
            torch.float32: 112.5 * 1e12,
            torch.bfloat16: 2250 * 1e12,
            torch.half: 2250 * 1e12,
            torch.int8: 4500 * 1e12,
            DTYPE_FP4: 4500 * 1e12,
        },
        gp_ops={
            torch.float32: 14 * 1e12,
            torch.bfloat16: 28 * 1e12,
            torch.half: 28 * 1e12,
        },
        memory_size_bytes=192 * (1024**3),
        memory_bandwidth_bytes_ps=8.0 * (1024**4),
        compute_efficiency=0.85,
        memory_efficiency=0.75,
        comm_grid=INTERCONNECT,
        static_cost=STATIC_COST,
    )
