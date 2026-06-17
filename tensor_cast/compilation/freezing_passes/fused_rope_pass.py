import logging
import operator

import torch
import torch.fx as fx

from ..pass_base import TensorCastGraphModulePass
from ..topo_sort import stable_topo_sort

logger = logging.getLogger(__name__)


class FusedRopePass(TensorCastGraphModulePass):
    """Replace apply_rope nodes with fused_rope nodes for NPU profiling alignment.

    On NPU, the InterleaveRope/fused_rope_qk_mqa kernel fuses cos/sin lookup
    and partial RoPE into a single operator. This pass replaces apply_rope
    with fused_rope so the simulation graph matches the NPU profiling
    operator count.

    apply_rope(q, k, cos, sin, is_neox) -> (q_out, k_out)
    fused_rope(q, k, cos, sin, rotary_dim, is_neox) -> (q_out, k_out)

    The shapes are compatible: both take BHSD input, output BSHD.
    fused_rope adds rotary_dim as an extra arg (inferred from cos shape).
    """

    def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target != torch.ops.tensor_cast.apply_rope.default:
                continue

            args = list(node.args)
            # apply_rope(q, k, cos, sin, is_neox)
            q_node = args[0]
            k_node = args[1]
            cos_node = args[2]
            sin_node = args[3]
            is_neox = args[4] if len(args) > 4 else True

            # Infer rotary_dim from cos tensor shape's last dimension
            # cos is typically (batch, seq_len, rotary_dim) or (num_tokens, rotary_dim)
            cos_meta = cos_node.meta.get("val", None) if hasattr(cos_node, "meta") else None
            if cos_meta is not None and hasattr(cos_meta, "shape"):
                rotary_dim = cos_meta.shape[-1]
            else:
                rotary_dim = -1

            with graph.inserting_before(node):
                fused_node = graph.create_node(
                    "call_function",
                    torch.ops.tensor_cast.fused_rope.default,
                    args=(q_node, k_node, cos_node, sin_node, rotary_dim, is_neox),
                )

            node.replace_all_uses_with(fused_node)
            modified = True

        if modified:
            stable_topo_sort(gm)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()

        return gm
