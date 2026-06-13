# MiniMax-M3 模型 msmodeling 适配实施记录

---

## 1. 适配目标

将 MiniMax-M3 模型接入 msmodeling（TensorCast）性能建模仿真框架，使其能够对 M3 的 Sparse Attention 机制进行精确的性能建模和仿真。

核心挑战：M3 的 Sparse Attention 包含 **Indexer**（index Q/K projection -> norm -> RoPE -> block score -> top-k）和 **Sparse Attention**（selected KV cache -> sparse QK/PV）两个独立建模边界，需在 msmodeling 的四层架构中逐层注册。

---

## 2. 思路方案

### 2.0 背景与问题演进

MiniMax-M3 的最初适配路径是将 `docs/modeling_minimax.py` 放入权重目录，通过 `trust_remote_code=True` 让 Transformers 加载本地 modeling。这个方式可以快速验证模型结构，但长期存在三个问题：

1. 本地 modeling 文件容易和上游 Transformers 实现漂移，后续维护成本高；
2. `AutoModel.from_config()` 在 `auto_map` / remote code 场景下需要额外处理 `trust_remote_code`，否则模型加载会在交互式确认后仍然失败；
3. 本地 modeling 放在 `docs/` 或权重目录中不属于 TensorCast 正式适配边界，不适合作为 PR 交付内容。

在 `huggingface/transformers#46600` 合入 MiniMax-M3 后，适配目标切换为：**完全使用 upstream Transformers 的 native MiniMax-M3 modeling，不再依赖仓内自写 modeling 文件**。因此本轮修改的重点从“补一个临时 modeling”调整为“让 TensorCast 能识别、patch、量化、分片并建模 upstream M3 结构”。

实际调试中还暴露出几个与 M3 结构相关的问题：

- **Sparse Attention 边界不同于普通 attention**：M3 sparse layer 不是完整 dense attention，而是先用 indexer 选择 block，再执行 sparse QK/PV，需要拆成独立虚拟 op 建模。
- **M3 expert 参数是 3D expert tensor**：upstream M3 的 routed expert 使用 `gate_up_proj` / `down_proj` 这类 `[E, *, *]` 权重，不是 TensorCast 原有 DFC 路线里每个 expert 一个 `gate_proj/up_proj/down_proj` module 的形态，因此无法自然融合成 TensorCast grouped matmul。
- **VL 模型层路径影响层复用**：M3 是 Vision-Language 模型，但本次文本仿真不编译 visual layers。原有 VL 层复用逻辑只有在能拿到 visual layers 时才继续处理 language layers，导致 M3 的 60 层完整展开，compile 时间很长。
- **完整 60 层 graph 暴露 multistream pass 递归栈风险**：即使层复用修复后 M3 常规路径不再强依赖该修复，`multistream_pass` 的 upward rank 本质是 DAG 上的反向动态规划，递归 DFS 对长图不够稳健。
- **MXFP8 权重显存估算偏大**：直接按 live tensor 或 safetensors 存储大小估算，会把 M3 3D expert / scale 的存储结构算偏，需要模型特定估算逻辑。

因此最终适配采用“native Transformers modeling + TensorCast profile/patch + M3 专属虚拟 op + M3 expert grouped matmul 转换 + 层复用修正 + 编译 pass 稳定性修正”的方案。

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

### 2.4 本轮最终修改范围

本轮整合后的代码改动可以分为七类：

1. **模型加载与配置解析**
   - 使用 upstream Transformers native MiniMax-M3 modeling；
   - `AutoModelConfigLoader` 在 `auto_map` 只有 `AutoConfig` 时仍视为 native supported，只有存在 `AutoModel*` remote mapping 时才切到 remote code；
   - `ConfigResolver` 从 HF config / text_config 解析 `dtype` / `torch_dtype`，避免初始化和执行 dtype 不一致。

2. **MiniMax-M3 ModelProfile 与 patch 链**
   - 注册 `model_type="minimax_m3_vl"`；
   - 设置语言层路径为 `language_model.layers`；
   - 构造空 `visual_layers` 容器，让 VL 模型也能触发 language layer reuse；
   - patch 顺序调整为 `wrap_model -> maybe_enable_mtp -> maybe_reuse_layers -> patch_minimax_m3_attention -> patch_attention -> patch_moe -> quantize_model -> shard_model`。

3. **Sparse Attention 虚拟 op**
   - 新增 `minimax_indexer`：覆盖 index Q/K projection、norm、RoPE、index K cache write、block score、top-k；
   - 新增 `minimax_sparse_attention`：覆盖 selected KV cache 读取、sparse QK/PV、输出 O；
   - 性能模型层分别为两个 op 注册 MMA、GP、Bytes 估算。

4. **MoE expert grouped matmul**
   - 保留 upstream M3 的 3D expert 权重形态；
   - 将 `[E, *, *]` expert 权重转成 TensorCast `grouped_matmul_fp8` 接受的 per-expert weight list；
   - 复用 TensorCast 现有 `dynamic_quantize_symmetric` 和 `grouped_matmul_fp8` op；
   - 保留 M3 的 `routed_scaling_factor`、`swiglu_alpha`、`swiglu_limit`。

5. **权重大小估算**
   - 为 `ModelProfile` 增加可选 `weight_size_estimator`；
   - M3 MXFP8 下对 3D expert weight 按 weight element + scale block 估算；
   - 避免模型加载后 live parameter dtype/shape 与真实 MXFP8 存储不一致导致显存偏大。

6. **编译稳定性**
   - `multistream_pass` upward rank 从递归 DFS 改为反向迭代 DP；
   - 增加 Gemma-style `add_rms_norm` / `add_rms_norm2` / quant pattern，避免 M3 RMSNorm residual fusion 路径需要通过关闭 fusion 规避。

7. **设备与测试**
   - 新增 B200 device profile；
   - 新增 MiniMax-M3 op 注册与 meta shape 回归测试；
   - 删除临时 `docs/modeling_minimax.py`，避免继续依赖自写 Transformers modeling。

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
| 模型加载 | `tensor_cast/transformers/utils.py` | **修改** | 修正 native Transformers / remote code 判断，支持 upstream M3 modeling |
| 配置解析 | `tensor_cast/core/config_resolver.py` | **修改** | 从 HF config / text_config 继承 dtype |
| 层复用 | `tensor_cast/transformers/builtin_model/minimax_m3.py` | **修改** | 为 M3 构造空 visual layers，使 VL language layers reuse 生效 |
| MoE grouped matmul | `tensor_cast/transformers/builtin_model/minimax_m3.py`、`tensor_cast/layers/moe_layer.py` | **修改** | 将 M3 3D expert 转换为 TensorCast grouped matmul 输入，并保留 M3 MoE 属性 |
| 编译稳定性 | `tensor_cast/compilation/passes/multistream_pass.py` | **修改** | upward rank 从递归 DFS 改为反向迭代 DP |
| RMSNorm fusion | `tensor_cast/compilation/patterns/rms_norm.py` | **修改** | 增加 Gemma-style add-rms-norm fusion pattern |
| 权重大小估算 | `tensor_cast/transformers/custom_model_registry.py`、`tensor_cast/transformers/model.py`、`minimax_m3.py` | **修改** | 增加可选模型级 weight size estimator，并为 M3 MXFP8 实现估算 |
| 设备 profile | `tensor_cast/device_profiles/b200.py` | **新建** | 增加 B200 profile |
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

### 3.7 其他关键修复

#### 3.7.1 切换到 upstream Transformers native modeling

早期验证阶段使用 `docs/modeling_minimax.py` 作为自定义 Transformers modeling，并复制到权重目录注册 AutoModel。这条路径的问题是：模型定义不再跟随 upstream Transformers 演进，而且 `AutoConfig` / `AutoModel` 在 remote code 判断上容易进入 `trust_remote_code` 分支。

最终实现删除了临时 modeling 文件，改为依赖 upstream Transformers 中的 `minimax_m3_vl` 实现。为此做了两处基础修改：

- `tensor_cast/transformers/utils.py`：加载 config 后检查 `auto_map`。如果 `auto_map` 只包含 `AutoConfig`，仍认为 Transformers native supported；只有出现 `AutoModel*` 映射时才视为需要 remote code。
- `tensor_cast/core/config_resolver.py`：从 HF config 或 text_config 读取 `dtype` / `torch_dtype`，写入 `ModelConfig.dtype`，避免 native M3 初始化时 dtype 和执行路径不一致。

这样 M3 的加载边界变为：

```
MiniMax-M3-MXFP8/config.json
    -> upstream transformers.models.minimax_m3_vl
    -> TensorCast ModelProfile("minimax_m3_vl")
    -> TensorCast custom patch
```

#### 3.7.2 语言层复用

MiniMax-M3 是 VL 模型，但本轮仿真主要面向语言模型路径，不编译 visual encoder。原有 `maybe_reuse_layers` 对 VL 模型的 language layers 复用依赖 `get_visual_layers(model)` 返回非空；M3 没有进入这条路径时，60 层 decoder 会完整展开，导致 compile 时间很长。

本轮修改在 `minimax_m3.py` 中新增 `_ensure_empty_visual_layers_for_reuse()`：

```python
_EMPTY_VISUAL_LAYERS_ATTR = "_tensor_cast_empty_visual_layers"

def _ensure_empty_visual_layers_for_reuse(model: TransformerModel):
    unwrapped = model.unwrap()
    if not hasattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR):
        setattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR, torch.nn.ModuleList())
```

同时在 `ModelProfile` 中设置：

```python
visual_layers_module_path=_EMPTY_VISUAL_LAYERS_ATTR
visual_layers_path_str=_EMPTY_VISUAL_LAYERS_ATTR
language_layers_path_str="language_model.layers"
language_module_path="language_model"
```

这样 `maybe_reuse_layers` 能继续处理 language layers。实际效果是 M3 language layers 被压缩为两个代表 region：dense/full layer 代表组和 sparse/MoE layer 代表组，其余层通过 `CopyLayerWrapper` 表示重复区域，避免 60 层真实 graph 全量进入 compile。

#### 3.7.3 M3 3D expert 转 TensorCast grouped matmul

upstream M3 routed expert 的权重不是每个 expert 一个子 module，而是集中存储为 3D tensor：

```
experts.gate_up_proj: [E, 2I, H]
experts.down_proj:    [E, H, I]
```

TensorCast 原有 DFC / grouped matmul 路线更适合 per-expert weight list。为了复用现有 `dynamic_quantize_symmetric` 和 `grouped_matmul_fp8`，新增 `MiniMaxM3FusedMoETensorCast`：

- 将 3D expert weight 按 expert 维拆成 list；
- 对每个 expert weight 做 transpose，使其匹配 grouped matmul 的输入格式；
- gate_up 和 down 分别调用一次 `tensor_cast.grouped_matmul_fp8`；
- gate_up 输出使用 M3 的 SwiGLU 变体：

```python
gate, up = gate_up.chunk(2, dim=-1)
gate = gate.clamp(max=self.swiglu_limit)
up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
activated = (up + 1.0) * glu
```

`ParallelMoELayer` 也做了通用修复：并行包装时不再强制重建成基础 `FusedMoETensorCast`，而是保留原 fused moe 的具体类型，并复制 `routed_scaling_factor`、`swiglu_alpha`、`swiglu_limit`、`quant_type` 等属性。

#### 3.7.4 MXFP8 权重大小估算

M3 MXFP8 expert 权重的真实存储不是简单的 BF16/FP8 dense tensor 大小。直接按模型加载后的 live tensor 或 safetensors 文件大小估算，会把 expert weight 和 scale 的关系算偏，出现模型权重大小显著大于预期的问题。

本轮为 `ModelProfile` 增加可选字段：

```python
weight_size_estimator: Optional[Callable] = None
```

`TransformerModel.weight_size` 优先调用模型 profile 提供的 estimator。M3 的 estimator 对 3D expert 权重按：

```
weight bytes = number_of_weight_elements
scale bytes  = number_of_scale_blocks
```

估算，其中 scale block 只覆盖最后两个维度，expert 维度作为 batch 维处理。非 3D expert 参数仍使用通用 `bytes_of_tensor()`。

#### 3.7.5 multistream upward rank 迭代 DP

在层复用修复前，M3 60 层完整展开会让 FX graph 的依赖链很长，`multistream_pass` 中 upward rank 的递归 DFS 可能触发 Python recursion limit。即使 M3 现在通过层复用绕开了主要压力，这个问题仍属于通用 compile pass 的脆弱点：

- 用户可以关闭层复用；
- 其他模型也可能生成很长的 FX graph；
- 某些模型结构不完全同构，无法稳定命中 reuse；
- upward rank 本质是 DAG 上的反向动态规划，不需要递归实现。

因此将 `_compute_upward_ranks()` 从递归 DFS 改为按拓扑序反向遍历：

```python
for node in reversed(nodes):
    self_cost = min(self._estimate_node_cost_s(node, stream_id) for stream_id in self._allowed_streams(node))
    max_succ_rank = 0.0
    for user in node.users.keys():
        if user in schedulable and user in self._ranks:
            max_succ_rank = max(max_succ_rank, self._ranks[user] + self.cross_stream_sync_overhead_s)
    self._ranks[node] = self_cost + max_succ_rank
```

当前实现依赖传入的 `nodes` 是 FX 拓扑序；FX graph 通常满足这一点。若未来该 pass 支持非拓扑输入，需要在进入该函数前显式 topo sort。

#### 3.7.6 Gemma-style Add RMSNorm fusion

M3 的 RMSNorm residual 路径更接近 Gemma 风格，即 norm weight 在计算时使用 `1.0 + weight`。早期规避方式是对 MiniMax-M3 定向关闭 add-rms-norm residual fusion，但这会降低编译优化覆盖面。

本轮在 `tensor_cast/compilation/patterns/rms_norm.py` 中增加 Gemma-style pattern：

- `GemmaAddRMSNormPattern`
- `GemmaAddRMSNorm2Pattern`
- `GemmaAddRMSNormQuantPattern`
- `GemmaAddRMSNormQuant2Pattern`

这些 pattern 在匹配时显式构造 `effective_weight = 1.0 + weight`，再替换为 TensorCast fused RMSNorm op，避免为了 M3 单独关闭 fusion。

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
  /Users/liujiaxu/Code/MiniMax-M3-MXFP8 \
  --num-queries 1 \
  --query-length 64 \
  --decode \
  --num-devices 8 \
  --tp-size 8 \
  --ep-size 8 \
  --quantize-linear-action FP8 \
  --log-level info \
  --compile
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
2. **EP + shared expert TP**：当前 TP8 路径已验证；EP8 + shared expert TP 仍需补齐 `moe_route_after_dp_transform=True` 和 `shared_experts.gate_up_proj` 的 colwise TP 规则
3. **MXFP8 scale 接入**：当前 M3 grouped matmul FP8 路径复用 TensorCast op，但 expert weight scale 仍是占位 scale，需要接入真实 MXFP8 scale
4. **Sparse op 通信建模**：`minimax_indexer` 和 `minimax_sparse_attention` 当前只建模计算和访存，若后续 kernel 实现包含 TP/EP 通信，需要拆出或在性能模型中补充通信项
5. **测试覆盖扩展**：现有回归测试只覆盖 op 注册和 meta shape，需要补充 TP1/TP8 compile、EP8、shared expert TP、op count 和 weight size 的回归测试
6. **Profiling 校正**：通过实际 kernel profiling 数据校正 indexer、sparse attention、grouped matmul 的访存量和 lower/upper bound
