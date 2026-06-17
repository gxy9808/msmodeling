from typing import Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("apply_rope")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed, k_embed = torch.empty_like(query), torch.empty_like(key)
    q_embed = q_embed.transpose(1, 2)
    k_embed = k_embed.transpose(1, 2)
    return q_embed.contiguous(), k_embed.contiguous()


@register_tensor_cast_op("fused_rope")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused rotary position embedding (RoPE) for Q and K tensors.

    Matches the NPU profiling operator InterleaveRope / fused_rope_qk_mqa
    which fuses cos/sin lookup + partial RoPE application into a single kernel.
    Signature matches sgl_kernel_npu.fused_rope_qk_mqa used in sglang NPU.

    Args:
        query: (num_tokens, num_heads, head_dim) 3D query tensor.
        key: (num_tokens, num_kv_heads, head_dim) 3D key tensor.
        cos_sin: (num_tokens, rotary_dim * 2) concatenated cos/sin for positions.
        rotary_dim: dimension to apply rotary embedding (partial RoPE).
        is_neox_style: True for GPT-NeoX style (rotate_half), False for interleaved.

    Returns:
        query_out: same shape as query.
        key_out: same shape as key.
    """
    return (
        torch.empty_like(query).contiguous(),
        torch.empty_like(key).contiguous(),
    )
