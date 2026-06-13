# MiniMax-M3 模型 msmodeling 适配实施记录

---

## 1. 适配目标

将 MiniMax-M3 模型接入 msmodeling（TensorCast）性能建模仿真框架，使其能够对 M3 的 Sparse Attention 机制进行精确的性能建模和仿真。

核心挑战：M3 的 Sparse Attention 包含 **Indexer**（index Q/K projection -> norm -> RoPE -> block score -> top-k）和 **Sparse Attention**（selected KV cache -> sparse QK/PV）两个独立建模边界，需在 msmodeling 的四层架构中逐层注册。

---

## 2. 思路方案

### 2.1 四层架构

msmodeling 的性能建模采用"一算子一建模"四层架构：

```
模型定义层  ->  注册 ModelProfile + @register_custom_model，识别模型结构
    |
抽象算子层  ->  封装 forward，将 attention 模块路由到虚拟算子
    |
虚拟算子层  ->  @register_tensor_cast_op 定义 meta-only op 签名（身份锚定）
    |
性能模型层  ->  @register_op_properties 注册 Roofline 拆解（MMA + GP + Bytes）
```

### 2.2 Op 拆分策略

将 M3 sparse layer 的 attention 拆成两个虚拟 op：

| 虚拟 Op | 建模边界 | 对应文档公式 |
|---------|---------|------------|
| `minimax_indexer` | hidden -> index Q/K proj -> norm -> RoPE -> index K cache write -> block score -> top-k indices | M3-msmodeling.md 4.1 |
| `minimax_sparse_attention` | Q + selected K/V cache -> sparse QK/PV attention -> output O | M3-msmodeling.md 4.2 |

Dense layer（前3层）复用已有 `tensor_cast.attention` 算子，无需新增。

### 2.3 参考先例

- **模型定义层**：参考 `minimax_m2.py`（同系列模型）和 `kimi_k25.py`（VL 模型 + 自定义 patch 链）
- **虚拟算子层**：参考 `ops/attention.py` 和 `ops/mla.py`
- **性能模型层**：参考 `_estimate_dsa_indexer_breakdown`（DeepSeek DSA Indexer 的拆解模式）

---

## 3. 代码修改详情

### 3.1 修改汇总表

| 层级 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 虚拟算子层 | `tensor_cast/ops/minimax_m3_sparse_attention.py` | **新建** | 注册 `minimax_indexer` 和 `minimax_sparse_attention` 两个 meta-only op |
| 虚拟算子导入 | `tensor_cast/ops/__init__.py` | **修改** | 添加 `minimax_m3_sparse_attention` 导入 |
| 抽象算子层 | `tensor_cast/layers/minimax_m3_attention.py` | **新建** | `MiniMaxM3AttentionWrapper`：dense 走标准 attention，sparse 调用 indexer + sparse_attention |
| 模型定义层 | `tensor_cast/transformers/builtin_model/minimax_m3.py` | **新建** | 注册 `minimax_m3_vl` 的 ModelProfile + `@register_custom_model`，含 `patch_minimax_m3_attention` |
| 性能模型层 | `tensor_cast/performance_model/__init__.py` | **修改** | 追加 `_estimate_minimax_indexer_breakdown` 和 `_estimate_minimax_sparse_attention_breakdown`，注册 `@register_op_properties` |
| 测试 | `tests/regression/tensor_cast/test_minimax_m3.py` | **新建** | op 注册 + meta shape 验证 |

### 3.2 虚拟算子层：`tensor_cast/ops/minimax_m3_sparse_attention.py`（新建）

注册两个 meta-only op，只做身份锚定和 shape 传播，不做真实计算。

```python
from typing import Optional
import torch
from ..utils import register_tensor_cast_op


@register_tensor_cast_op("minimax_indexer")
def _(
    hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    hidden_size: int,
    num_indexer_heads: int,
    indexer_head_dim: int,
    indexer_rope_dim: int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """
    MiniMax-M3 indexer fused op.

    Boundary:
      hidden -> index_q_proj/index_k_proj -> norm -> RoPE -> index K cache write
      -> index QK block score -> top-k block indices.

    Performance formula: see M3-msmodeling.md section 4.1.
    """
    total_tokens = hidden_states.shape[0]
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
      read Q + selected K/V cache -> sparse QK/PV attention -> output O.

    Performance formula: see M3-msmodeling.md section 4.2.
    """
    return torch.empty_like(query).contiguous()
```

**关键设计**：
- `minimax_indexer` 返回 `(T, N_indexer, K)` 的 int32 top-k index 张量
- `minimax_sparse_attention` 返回与 query 同形状的输出
- 所有静态参数通过 keyword-only args 传递（`*` 分隔），方便性能模型层从 `kwargs` 提取

### 3.3 虚拟算子导入：`tensor_cast/ops/__init__.py`（修改）

在导入列表中添加 `minimax_m3_sparse_attention`：

```python
from . import (  # noqa: F401
    attention,
    cat,
    communication,
    fused_moe,
    gmm,
    internal,
    layernorm,
    linear,
    mla,
    mtp,
    quantization,
    rotary_embedding,
    minimax_m3_sparse_attention,   # <-- 新增
    swiglu,
)
```

### 3.4 抽象算子层：`tensor_cast/layers/minimax_m3_attention.py`（新建）

封装 `MiniMaxM3AttentionWrapper`，根据 `is_sparse_layer` 标志路由：

- **Dense layer**：透传给原始 attention 模块（走已有 `tensor_cast.attention` 路径）
- **Sparse layer**：调用 `minimax_indexer` -> `minimax_sparse_attention`

```python
import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class MiniMaxM3AttentionWrapper(torch.nn.Module):
    """Wrapper for MiniMax-M3 attention that routes dense/sparse layers."""

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
        # ... 存储所有参数 ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attention_meta = kwargs.get("attention_meta", None)

        # Dense layer 或无 attention_meta 时走标准路径
        if not self.is_sparse_layer or attention_meta is None:
            return self._inner(
                hidden_states, attention_mask=attention_mask, **kwargs,
            )

        # Sparse layer: indexer + sparse_attention
        seq_lens = attention_meta.seq_lens
        query_lens = attention_meta.query_lens
        block_table = attention_meta.block_table_tensor

        topk_idx = torch.ops.tensor_cast.minimax_indexer(
            hidden_states, seq_lens, query_lens, block_table,
            hidden_size=self.hidden_size,
            num_indexer_heads=self.num_indexer_heads,
            indexer_head_dim=self.indexer_head_dim,
            indexer_rope_dim=self.indexer_rope_dim,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
        )

        # KV cache 支持 list/tuple 和单 tensor 格式
        key_cache = kwargs.get("kv_cache", None)
        if key_cache is not None and isinstance(key_cache, (list, tuple)):
            key_cache_tensor = key_cache[0]
            value_cache_tensor = key_cache[1]
        else:
            key_cache_tensor = key_cache
            value_cache_tensor = key_cache

        out = torch.ops.tensor_cast.minimax_sparse_attention(
            hidden_states, key_cache_tensor, value_cache_tensor, topk_idx,
            seq_lens, query_lens, block_table,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
        )

        return out, None
```

**关键设计**：
- 从 `attention_meta` 提取 `seq_lens`、`query_lens`、`block_table`，保证 per-request 的 Q_b、L_b 能传到性能模型层
- KV cache 支持 list/tuple 格式（paged attention）和单 tensor 格式
- 返回 `(output, None)` 与标准 attention 的 `(attn_output, attn_weights)` 格式对齐

### 3.5 模型定义层：`tensor_cast/transformers/builtin_model/minimax_m3.py`（新建）

完成三件事：

1. 注册 `minimax_m3_vl` 的 `ModelProfile`
2. 注册 `@register_custom_model("minimax_m3_vl")` 自定义 patch 函数
3. 实现 `patch_minimax_m3_attention` 替换 attention 模块

```python
import logging
import torch
from tensor_cast.transformers.transformations import (
    maybe_enable_mtp, maybe_reuse_layers, patch_attention,
    patch_moe, patch_rotary_emb, quantize_model, shard_model, wrap_model,
)
from ..custom_model_registry import (
    ModelProfile, register_custom_model, register_model_profile,
)
from ..model import TransformerModel
from ...layers.minimax_m3_attention import MiniMaxM3AttentionWrapper

logger = logging.getLogger(__name__)


def patch_minimax_m3_attention(model: TransformerModel) -> TransformerModel:
    """Replace MiniMax-M3 attention layers with TensorCast wrappers."""
    sparse_cfg = None
    text_config = model.text_config

    if hasattr(text_config, "sparse_attention_config"):
        sparse_cfg = text_config.sparse_attention_config
    if sparse_cfg is None or not sparse_cfg.get("use_sparse_attention", False):
        logger.info("No sparse_attention_config found, skipping M3 attention patch")
        return model

    # 从 sparse_attention_config 提取参数
    sparse_attention_freq = sparse_cfg.get("sparse_attention_freq", [])
    num_indexer_heads = sparse_cfg.get("sparse_num_index_heads", 4)
    indexer_head_dim = sparse_cfg.get("sparse_index_dim", 128)
    indexer_rope_dim = getattr(text_config, "rotary_dim", 64)
    topk_blocks = sparse_cfg.get("sparse_topk_blocks", 16)
    block_size = sparse_cfg.get("sparse_block_size", 128)
    local_blocks = sparse_cfg.get("sparse_local_block", 1)

    hidden_size = text_config.hidden_size
    num_q_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", hidden_size // num_q_heads)

    # TP 切分后 per-rank 的 head 数
    tp_size = 1
    if model.parallel_group_manager is not None and model.parallel_group_manager.tp_group is not None:
        tp_size = model.parallel_group_manager.tp_group.world_size
    per_rank_q_heads = num_q_heads // tp_size
    per_rank_kv_heads = num_kv_heads // tp_size
    per_rank_indexer_heads = num_indexer_heads // tp_size if num_indexer_heads >= tp_size else num_indexer_heads

    # 定位 layers 列表（VL 模型在 language_model.model.layers）
    unwrapped = model.unwrap()
    if not hasattr(unwrapped, "layers"):
        # ... VL 模型路径查找逻辑 ...

    for layer_idx, layer in enumerate(unwrapped.layers):
        # 穿透 wrapper 层找到原始 self_attn
        self_attn = layer
        while hasattr(self_attn, "_inner"):
            self_attn = self_attn._inner
        if hasattr(self_attn, "self_attn"):
            self_attn = self_attn.self_attn

        # 按 sparse_attention_freq 判断 dense/sparse
        is_sparse = (
            layer_idx < len(sparse_attention_freq) and sparse_attention_freq[layer_idx] == 1
        )

        wrapper = MiniMaxM3AttentionWrapper(
            original_module=self_attn,
            is_sparse_layer=is_sparse,
            hidden_size=hidden_size,
            num_q_heads=per_rank_q_heads,
            num_kv_heads=per_rank_kv_heads,
            head_dim=head_dim,
            num_indexer_heads=per_rank_indexer_heads,
            indexer_head_dim=indexer_head_dim,
            indexer_rope_dim=indexer_rope_dim,
            topk_blocks=topk_blocks,
            block_size=block_size,
            local_blocks=local_blocks,
        )

        # 替换 self_attn
        parent = layer
        while hasattr(parent, "_inner") and hasattr(parent._inner, "self_attn"):
            parent = parent._inner
        if hasattr(parent, "self_attn"):
            parent.self_attn = wrapper

    return model


@register_custom_model("minimax_m3_vl")
def _(model: TransformerModel):
    model = wrap_model(model)
    model = maybe_enable_mtp(model)
    model = maybe_reuse_layers(model)
    model = patch_rotary_emb(model)
    model = patch_attention(model)
    model = patch_minimax_m3_attention(model)   # <-- 新增
    model = patch_moe(model)
    model = quantize_model(model)
    model = shard_model(model)
    return model


register_model_profile(
    ModelProfile(
        model_type="minimax_m3_vl",
        moe_module_name="MiniMaxM3SparseMoeBlock",
        moe_gate_returns_raw_logits=False,
        moe_num_experts_key="num_local_experts",
        mtp_block_module_name="MiniMaxM3DecoderLayer",
        language_layers_path_str="language_model.model.layers",
        language_module_path="language_model",
        visual_module_path="vision_tower",
        visual_layers_module_path="vision_tower.encoder.blocks",
        visual_layers_path_str="vision_tower.encoder.blocks",
    )
)
```

**关键设计**：
- M3 是 VL 模型（`model_type="minimax_m3_vl"`），`text_config` 中存放语言模型参数
- `sparse_attention_config` 字典包含 `sparse_attention_freq` 列表，按 layer_idx 判断 dense/sparse
- TP 切分后 head 数需要除以 `tp_size`，indexer head 也需切分
- 通过 `while hasattr(_inner)` 穿透 wrapper 层找到原始 self_attn

### 3.6 性能模型层：`tensor_cast/performance_model/__init__.py`（修改）

在文件末尾 `_load_custom_op()` 之前，追加两个拆解函数和两个 `@register_op_properties` 注册。

#### 3.6.1 `minimax_indexer` 拆解

对应 M3-msmodeling.md 4.1 公式：

```
MMA = 4*T*H*N*D + 2*sum_b(Q_b*N*L_b*D)
GP  = 12*T*N*D + 6*T*N*D_r + sum_b(Q_b*N*L_b) + c_topk*sum_b(Q_b*N*B_n)
```

访存按各阶段读写相加（projection + norm/rope + cache write + score + topk）。

完整实现中，各阶段拆成命名变量再求和，方便 profiling 校正：

```python
def _estimate_minimax_indexer_breakdown(
    hidden_states, seq_lens, query_lens,
    hidden_size, num_indexer_heads, indexer_head_dim,
    indexer_rope_dim, topk_blocks, block_size,
):
    T = hidden_states.shape[0]
    H = hidden_size; N = num_indexer_heads; D = indexer_head_dim
    D_r = indexer_rope_dim; K = topk_blocks; B_s = block_size
    s = hidden_states.element_size()

    # --- MMA ---
    index_q_proj_mma = 2 * T * H * N * D          # 4.1.1
    index_k_proj_mma = 2 * T * H * N * D          # 4.1.1

    Q_b_list = query_lens.tolist()
    L_b_list = seq_lens.tolist()
    index_qk_mma = 0; sum_qb_nb_lb = 0; sum_qb_nb_bn = 0
    for Q_b, L_b in zip(Q_b_list, L_b_list):
        B_n = math.ceil(L_b / B_s) if B_s > 0 else 0
        index_qk_mma += 2 * Q_b * N * L_b * D      # 4.1.4
        sum_qb_nb_lb += Q_b * N * L_b
        sum_qb_nb_bn += Q_b * N * B_n

    # --- GP ---
    index_norm_gp = 12 * T * N * D                  # 4.1.2
    index_rope_gp = 6 * T * N * D_r                 # 4.1.2
    block_reduce_gp = sum_qb_nb_lb                  # 4.1.4
    c_topk = max(int(math.ceil(math.log2(max(K, 2)))), 1)
    topk_gp = c_topk * sum_qb_nb_bn                 # 4.1.5

    # --- Bytes ---
    bytes_projection  = 2*T*H*s + 2*H*N*D*s + 2*T*N*D*s      # 4.1.1
    bytes_norm_rope   = 4*T*N*D*s + 4*T*N*D*s + 2*N*D*s      # 4.1.2
    bytes_cache_write = 2*T*N*D*s                              # 4.1.3
    bytes_score = T*N*D*s                                      # 4.1.4
    for Q_b, L_b in zip(Q_b_list, L_b_list):
        B_n = math.ceil(L_b / B_s) if B_s > 0 else 0
        bytes_score += Q_b*N*L_b*D*s + 4*Q_b*N*B_n
    bytes_topk = 4*sum_qb_nb_bn + 4*T*N*K                     # 4.1.5

    return {
        "mma_total": index_q_proj_mma + index_k_proj_mma + index_qk_mma,
        "gp_total": index_norm_gp + index_rope_gp + block_reduce_gp + topk_gp,
        "bytes_total": bytes_projection + bytes_norm_rope + bytes_cache_write + bytes_score + bytes_topk,
    }
```

#### 3.6.2 `minimax_sparse_attention` 拆解

对应 M3-msmodeling.md 4.2 公式：

```
A_b = min(L_b, min(B_n, K+R)*B_s)     -- selected token 上界
MMA = 4*sum_b(Q_b*N_q*A_b*D)
GP  = 6*sum_b(Q_b*N_q*A_b)
```

```python
def _estimate_minimax_sparse_attention_breakdown(
    query, seq_lens, query_lens,
    num_q_heads, num_kv_heads, head_dim,
    topk_blocks, block_size, local_blocks,
):
    T = query.shape[0]; N_q = num_q_heads; N_kv = num_kv_heads
    D = head_dim; K = topk_blocks; B_s = block_size; R = local_blocks
    s = query.element_size()

    Q_b_list = query_lens.tolist()
    L_b_list = seq_lens.tolist()

    mma_total = 0; gp_total = 0
    qo_bytes = 2*s*T*N_q*D
    kv_bytes = 0
    topk_bytes = 4*T*N_kv*K

    for Q_b, L_b in zip(Q_b_list, L_b_list):
        B_n = math.ceil(L_b / B_s) if B_s > 0 else 0
        A_b = min(L_b, min(B_n, K + R) * B_s)
        mma_total += 4 * Q_b * N_q * A_b * D       # QK^T + PV
        gp_total += 6 * Q_b * N_q * A_b             # softmax
        kv_bytes += 2*s * Q_b * A_b * N_kv * D      # KV read

    return {
        "mma_total": mma_total,
        "gp_total": gp_total,
        "bytes_total": qo_bytes + kv_bytes + topk_bytes,
    }
```

#### 3.6.3 注册到 OpInvokeInfo

```python
@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.minimax_indexer.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    hidden_states = op_invoke_info.args[0]
    seq_lens = op_invoke_info.args[1]
    query_lens = op_invoke_info.args[2]
    # 从 kwargs 提取静态参数
    hidden_size = op_invoke_info.kwargs["hidden_size"]
    num_indexer_heads = op_invoke_info.kwargs["num_indexer_heads"]
    # ... 其余参数 ...

    breakdown = _estimate_minimax_indexer_breakdown(...)
    properties = op_invoke_info.get_memory_access_properties()
    _accumulate_compute_ops(properties, hidden_states.dtype,
                           mma_ops=breakdown["mma_total"],
                           gp_ops=breakdown["gp_total"])
    properties.memory_readwrite_bytes += breakdown["bytes_total"]
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.minimax_sparse_attention.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query = op_invoke_info.args[0]
    seq_lens = op_invoke_info.args[4]
    query_lens = op_invoke_info.args[5]
    # 从 kwargs 提取静态参数
    # ...

    breakdown = _estimate_minimax_sparse_attention_breakdown(...)
    # exclude KV cache 的自动访存统计，由拆解函数精确计算
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={1, 2})
    _accumulate_compute_ops(properties, query.dtype,
                           mma_ops=breakdown["mma_total"],
                           gp_ops=breakdown["gp_total"])
    properties.memory_readwrite_bytes += breakdown["bytes_total"]
    return properties
```

**关键设计**：
- `minimax_sparse_attention` 排除 KV cache 的自动访存统计（`exclude_input_ids={1, 2}`），因为 KV cache 的访存由拆解函数按 selected blocks 精确计算
- `minimax_indexer` 使用默认的自动访存 + 额外 `memory_readwrite_bytes`，因为 indexer 的 index K cache 访存需要精确建模

---

## 4. 测试验证

### 4.1 Ops 注册与 Meta Shape 验证

```python
import torch
from tensor_cast.ops import minimax_m3_sparse_attention

# 验证 op 注册
print(hasattr(torch.ops.tensor_cast, "minimax_indexer"))           # True
print(hasattr(torch.ops.tensor_cast, "minimax_sparse_attention"))  # True

# 验证 meta shape
hidden_states = torch.empty(1, 6144, device="meta")
seq_lens = torch.tensor([4096], device="meta")
query_lens = torch.tensor([1], device="meta")
block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

topk_idx = torch.ops.tensor_cast.minimax_indexer(
    hidden_states, seq_lens, query_lens, block_table,
    hidden_size=6144, num_indexer_heads=4,
    indexer_head_dim=128, indexer_rope_dim=64,
    topk_blocks=16, block_size=128,
)
print(topk_idx.shape)  # torch.Size([1, 4, 16])
print(topk_idx.dtype)  # torch.int32
```

**结果**：全部通过。

### 4.2 性能模型 Breakdown 验证

```python
from tensor_cast.performance_model import (
    _estimate_minimax_indexer_breakdown,
    _estimate_minimax_sparse_attention_breakdown,
)
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo

# 验证 op properties 注册
op_key = torch.ops.tensor_cast.minimax_indexer.default
print(op_key in OpInvokeInfo._op_properties_functors)  # True

# 验证 breakdown 计算
hidden_states = torch.empty(1, 6144, dtype=torch.bfloat16)
seq_lens = torch.tensor([4096], dtype=torch.long)
query_lens = torch.tensor([1], dtype=torch.long)

breakdown = _estimate_minimax_indexer_breakdown(
    hidden_states, seq_lens, query_lens,
    hidden_size=6144, num_indexer_heads=4,
    indexer_head_dim=128, indexer_rope_dim=64,
    topk_blocks=16, block_size=128,
)
print("MMA:", breakdown["mma_total"])    # 16777216
print("GP:", breakdown["gp_total"])      # 24576
print("Bytes:", breakdown["bytes_total"])  # 16818432
```

**结果**：MMA/GP/Bytes 均有合理非零输出，op properties 注册成功。

### 4.3 端到端 Runtime 仿真验证

```python
import torch
from tensor_cast.runtime import Runtime
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.device import TEST_DEVICE

perf_model = AnalyticPerformanceModel(TEST_DEVICE)

with Runtime(perf_model, TEST_DEVICE) as runtime, torch.no_grad():
    seq_lens = torch.tensor([4096], dtype=torch.long)
    query_lens = torch.tensor([1], dtype=torch.long)
    block_table = torch.empty(1, 32, dtype=torch.long)

    topk_idx = torch.ops.tensor_cast.minimax_indexer(
        torch.empty(1, 6144, dtype=torch.bfloat16),
        seq_lens, query_lens, block_table,
        hidden_size=6144, num_indexer_heads=4,
        indexer_head_dim=128, indexer_rope_dim=64,
        topk_blocks=16, block_size=128,
    )

    out = torch.ops.tensor_cast.minimax_sparse_attention(
        torch.empty(1, 64, 128, dtype=torch.bfloat16),
        torch.empty(4096, 4, 128, dtype=torch.bfloat16),
        torch.empty(4096, 4, 128, dtype=torch.bfloat16),
        topk_idx, seq_lens, query_lens, block_table,
        num_q_heads=64, num_kv_heads=4, head_dim=128,
        topk_blocks=16, block_size=128, local_blocks=1,
    )

print(runtime.table_averages())
```

**输出**：

```
--------------------------------------------  --------------  ------------  ----------
                    Name                      analytic total  analytic avg  # of Calls
--------------------------------------------  --------------  ------------  ----------
tensor_cast.minimax_indexer.default                 20.946us      20.946us           1
aten.empty.memory_format                            17.975us       3.595us           5
tensor_cast.minimax_sparse_attention.default         9.285us       9.285us           1
aten.lift_fresh.default                              0.000ns       0.000ns           2
aten.detach.default                                  0.000ns       0.000ns           5
--------------------------------------------  --------------  ------------  ----------
Total time for analytic: 48.205us
```

**结果**：`minimax_indexer` 和 `minimax_sparse_attention` 均成功出现在 Runtime 仿真结果中，耗时估算合理。

### 4.4 模型定义层验证

```python
from tensor_cast.layers.minimax_m3_attention import MiniMaxM3AttentionWrapper
from tensor_cast.transformers.custom_model_registry import get_model_profile

# 验证 Wrapper 可实例化
wrapper = MiniMaxM3AttentionWrapper(
    original_module=torch.nn.Linear(64, 64),
    is_sparse_layer=True, hidden_size=6144,
    num_q_heads=64, num_kv_heads=4, head_dim=128,
    num_indexer_heads=4, indexer_head_dim=128,
    indexer_rope_dim=64, topk_blocks=16,
    block_size=128, local_blocks=1,
)
print(wrapper.is_sparse_layer)  # True

# 验证 ModelProfile 注册
profile = get_model_profile("minimax_m3_vl")
print(profile.moe_module_name)          # MiniMaxM3SparseMoeBlock
print(profile.mtp_block_module_name)    # MiniMaxM3DecoderLayer
print(profile.language_layers_path_str)  # language_model.model.layers
```

**结果**：全部通过。

---

## 5. 使用方法

### 5.1 端到端仿真命令

```bash
python -m cli.inference.text_generate \
    --model-id minimax/MiniMax-M3 \
    --num-queries 1 \
    --query-length 128 \
    --decode \
    --tp-size 8 \
    --device ATLAS_800_A3_752T_128G_DIE
```

### 5.2 注册流程追溯

```
用户输入 model-id="minimax/MiniMax-M3"
    |
    v
ConfigResolver.resolve()
    |  model_type = "minimax_m3_vl"
    |  读取 text_config.sparse_attention_config
    |  ModelProfile 注册 -> MoE/Attention/MTP 静态信息
    |
    v
TransformerModel(model_id, config)
    |  import builtin_model.minimax_m3  (自动加载)
    |  -> register_model_profile(ModelProfile("minimax_m3_vl", ...))
    |  -> @register_custom_model("minimax_m3_vl")
    |
    v
custom_fn(model):
    |  wrap_model -> maybe_enable_mtp -> maybe_reuse_layers
    |  patch_rotary_emb -> patch_attention
    |  patch_minimax_m3_attention  <-- 核心: 替换 attention
    |     |  解析 sparse_attention_config
    |     |  识别 sparse_layer_ids (sparse_attention_freq[layer_idx]==1)
    |     |  替换 self_attn -> MiniMaxM3AttentionWrapper
    |  patch_moe -> quantize_model -> shard_model
    |
    v
model.forward(**inputs)
    |  每层 forward:
    |  MiniMaxM3AttentionWrapper.forward()
    |     |  dense layer: 走已有 tensor_cast.attention
    |     |  sparse layer:
    |     |     torch.ops.tensor_cast.minimax_indexer(...)
    |     |     torch.ops.tensor_cast.minimax_sparse_attention(...)
    |
    v
Runtime.__torch_dispatch__()
    |  -> 分别为两个 op 注册 OpInvokeInfo
    |  -> AnalyticPerformanceModel.process_op()
    |     |  minimax_indexer -> _estimate_minimax_indexer_breakdown
    |     |  minimax_sparse_attention -> _estimate_minimax_sparse_attention_breakdown
    |     |  Roofline: time = max(compute_time, memory_time) + static_cost
    |
    v
输出: table_averages(), total_execution_time_s(), export_chrome_trace()
```

---

## 6. 后续工作

1. **端到端对比验证**：在 B200/A5 上采集硬件实测 TTFT/TPOT，与 msmodeling 仿真结果做误差对比
2. **Index value/output 分支**：当 `sparse_disable_index_value=False` 的 sparse layer 出现时，需要额外建模 `index_v_proj` / `index_o_proj` 分支
3. **SwiGLU 变体**：M3 使用 `swigluoai`（带 alpha/limit 参数），可能需要定制 SwiGLU op 的 GP 估算
4. **Profiling 校正**：通过实际 kernel profiling 数据校正访存量的 lower/upper bound
