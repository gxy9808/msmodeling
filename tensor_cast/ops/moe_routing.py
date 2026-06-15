from typing import List, Optional

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("moe_decode_score")
def _(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (hidden_states.shape[0], gate_weight.shape[0]),
        dtype=torch.float32,
        device=hidden_states.device,
    )


@register_tensor_cast_op("moe_topk_index_partial")
def _(
    scores: torch.Tensor,
    top_k: int,
) -> List[torch.Tensor]:
    return (
        torch.empty((scores.shape[0], top_k), dtype=torch.float32, device=scores.device),
        torch.empty((scores.shape[0], top_k), dtype=torch.int32, device=scores.device),
    )


@register_tensor_cast_op("moe_topk_index_merge")
def _(
    partial_weights: torch.Tensor,
    partial_indices: torch.Tensor,
) -> List[torch.Tensor]:
    return (
        torch.empty_like(partial_weights),
        torch.empty_like(partial_indices),
    )


@register_tensor_cast_op("moe_post_reorder")
def _(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@register_tensor_cast_op("moe_fill_gateup_input")
def _(
    hidden_states: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@register_tensor_cast_op("moe_compute_seg_indptr")
def _(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    return torch.empty((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)


@register_tensor_cast_op("moe_compute_src2dst")
def _(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    return torch.empty((topk_ids.numel(),), dtype=torch.int32, device=topk_ids.device)


@register_tensor_cast_op("moe_compute_masked_m")
def _(
    seg_indptr: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(seg_indptr[:-1])
