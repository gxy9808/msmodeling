# MiniMax-M3 msmodeling 适配记录

## 1. 适配目标

MiniMax-M3 适配目标是在 TensorCast 中支持基于 upstream Transformers `minimax_m3_vl` modeling 的性能仿真，使模型能够完成加载、patch、量化、分片、编译和性能统计。

本次适配覆盖以下核心能力：

- 识别 MiniMax-M3 VL 模型结构，并注册对应 `ModelProfile`
- 对 MiniMax-M3 sparse attention 建立 TensorCast 虚拟算子和性能模型
- 将 MiniMax-M3 3D expert 权重转换到 TensorCast grouped matmul 路径
- 支持 MiniMax-M3 语言层复用，降低完整 60 层 compile 成本
- 修正 MXFP8 模型权重大小估算
- 增强 compile pass 对长图的稳定性
- 增加 B200 device profile 和基础回归测试

## 2. 总体设计

MiniMax-M3 适配沿用 TensorCast 的四层建模结构：

```text
模型定义层
  ModelProfile + @register_custom_model
  识别 MiniMax-M3 结构、patch 模型、接入量化与分片

抽象算子层
  MiniMaxM3AttentionWrapper
  将 dense attention 和 sparse attention 路径分流

虚拟算子层
  tensor_cast.minimax_indexer
  tensor_cast.minimax_sparse_attention
  定义 meta-only op 签名和 shape 传播

性能模型层
  @OpInvokeInfo.register_op_properties
  为两个 sparse op 建立 MMA / GP / Bytes 估算
```

MiniMax-M3 的 sparse layer 被拆成两个性能建模边界：

| 虚拟 op | 建模边界 |
| --- | --- |
| `tensor_cast.minimax_indexer` | hidden states -> index Q/K projection -> norm -> RoPE -> index K cache write -> block score -> top-k indices |
| `tensor_cast.minimax_sparse_attention` | Q + selected K/V cache -> sparse QK/PV attention -> output |

Dense/full attention layer 继续使用已有 TensorCast attention 路径。

## 3. 修改清单

| 文件 | 修改类型 | 内容 |
| --- | --- | --- |
| `tensor_cast/transformers/builtin_model/minimax_m3.py` | 新增 | MiniMax-M3 ModelProfile、custom patch、sparse attention patch、MoE 3D expert grouped matmul 转换、MXFP8 weight estimator |
| `tensor_cast/layers/minimax_m3_attention.py` | 新增 | `MiniMaxM3AttentionWrapper`，按 layer 类型分流 dense/sparse attention |
| `tensor_cast/ops/minimax_m3_sparse_attention.py` | 新增 | 注册 `minimax_indexer` 和 `minimax_sparse_attention` meta-only op |
| `tensor_cast/ops/__init__.py` | 修改 | 导入 MiniMax-M3 sparse attention op |
| `tensor_cast/performance_model/__init__.py` | 修改 | 增加 MiniMax-M3 indexer 和 sparse attention 性能模型 |
| `tensor_cast/layers/moe_layer.py` | 修改 | 并行包装 MoE 时保留具体 fused moe 类型和模型特定属性 |
| `tensor_cast/transformers/custom_model_registry.py` | 修改 | `ModelProfile` 增加可选 `weight_size_estimator` |
| `tensor_cast/transformers/model.py` | 修改 | `TransformerModel.weight_size` 支持调用模型级 estimator |
| `tensor_cast/transformers/utils.py` | 修改 | 调整 native Transformers / remote code 判断 |
| `tensor_cast/core/config_resolver.py` | 修改 | 从 HF config / text_config 继承 dtype |
| `tensor_cast/compilation/passes/multistream_pass.py` | 修改 | upward rank 从递归 DFS 改为反向迭代 DP |
| `tensor_cast/compilation/patterns/rms_norm.py` | 修改 | 增加 Gemma-style add RMSNorm fusion pattern |
| `tensor_cast/device_profiles/b200.py` | 新增 | B200 device profile |
| `tests/regression/tensor_cast/test_minimax_m3.py` | 新增 | MiniMax-M3 op 注册与 meta shape 回归测试 |

## 4. 模型定义层

MiniMax-M3 通过 `ModelProfile(model_type="minimax_m3_vl")` 注册模型信息。

关键配置：

```python
ModelProfile(
    model_type="minimax_m3_vl",
    moe_module_name="MiniMaxM3VLSparseMoeBlock",
    moe_gate_returns_raw_logits=False,
    moe_num_experts_key="num_local_experts",
    mtp_block_module_name="MiniMaxM3DecoderLayer",
    hf_config_patch_method=_patch_minimax_m3_hf_config,
    weight_size_estimator=estimate_minimax_m3_weight_size,
    language_layers_path_str="language_model.layers",
    language_module_path="language_model",
    visual_layers_module_path="_tensor_cast_empty_visual_layers",
    visual_layers_path_str="_tensor_cast_empty_visual_layers",
    custom_expert_module_type=None,
)
```

custom patch 链为：

```text
wrap_model
-> maybe_enable_mtp
-> _ensure_empty_visual_layers_for_reuse
-> maybe_reuse_layers
-> patch_minimax_m3_attention
-> patch_attention
-> patch_moe
-> _patch_m3_moe_return_compat
-> quantize_model
-> shard_model
```

其中 `_ensure_empty_visual_layers_for_reuse` 为文本仿真构造空 visual layers，使 VL 模型也能进入 language layers reuse 逻辑。

## 5. Sparse Attention 适配

### 5.1 抽象算子层

`MiniMaxM3AttentionWrapper` 包装原始 attention module：

- dense/full layer：透传给原 attention module，继续走已有 attention patch
- sparse layer：调用 `minimax_indexer` 和 `minimax_sparse_attention`

wrapper 从 `attention_meta` 中读取：

- `seq_lens`
- `query_lens`
- `block_table_tensor`

这些运行时长度会传入性能模型，用于计算每个 request 的 `Q_b`、`L_b` 和 selected block 上界。

### 5.2 虚拟算子层

`minimax_indexer` 返回 top-k block indices：

```python
torch.empty((total_tokens, num_indexer_heads, topk_blocks), dtype=torch.int32, device="meta")
```

`minimax_sparse_attention` 返回与 query 相同 shape 的输出：

```python
torch.empty_like(query).contiguous()
```

两个 op 都是 meta-only op，只承担图中身份锚定、shape 传播和性能模型入口的作用。

### 5.3 性能模型层

`minimax_indexer` 估算内容：

- Index Q projection
- Index K projection
- Index Q/K norm
- RoPE
- Index K cache write
- QK block score
- block max reduce
- top-k select

核心计算量：

```text
MMA = 2*T*H*N*D + 2*T*H*N*D + 2*sum_b(Q_b*N*L_b*D)
GP  = 12*T*N*D + 6*T*N*D_r + sum_b(Q_b*N*L_b) + c_topk*sum_b(Q_b*N*B_n)
```

`minimax_sparse_attention` 估算内容：

- selected K/V cache read
- sparse QK
- softmax
- sparse PV
- output write

其中：

```text
B_n = ceil(L_b / B_s)
A_b = min(L_b, min(B_n, K + R) * B_s)
```

核心计算量：

```text
MMA = 4*sum_b(Q_b*N_q*A_b*D)
GP  = 6*sum_b(Q_b*N_q*A_b)
```

当前 sparse op 只建模计算和访存，不单独拆出 TP/EP collective。

## 6. MoE 与 grouped matmul

MiniMax-M3 routed expert 权重是 3D tensor 形态：

```text
experts.gate_up_proj: [E, 2I, H]
experts.down_proj:    [E, H, I]
```

TensorCast grouped matmul 接口使用 per-expert weight list，因此新增 `MiniMaxM3FusedMoETensorCast`：

- 按 expert 维度拆分 3D weight
- 对每个 expert weight 转置并 contiguous
- FP8 路径对每个 expert input 调用 `dynamic_quantize_symmetric`
- gate_up 和 down 分别调用 `tensor_cast.grouped_matmul_fp8`
- 使用 MiniMax-M3 SwiGLU 变体完成 activation

MiniMax-M3 SwiGLU 计算：

```python
gate, up = gate_up.chunk(2, dim=-1)
gate = gate.clamp(max=self.swiglu_limit)
up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
activated = (up + 1.0) * glu
```

`ParallelMoELayer` 做了通用增强：并行包装时保留原 fused moe 的具体 class，并复制 `routed_scaling_factor`、`swiglu_alpha`、`swiglu_limit`、`quant_type` 等属性，避免 M3 特定行为在 parallel wrapper 中丢失。

## 7. 层复用

MiniMax-M3 文本模型共有 60 层，其中 dense/full layer 和 sparse/MoE layer 结构重复。适配后通过 `maybe_reuse_layers` 将语言层压缩成代表 region：

- dense/full layer 代表 region
- sparse/MoE layer 代表 region
- 其余同构层使用 `CopyLayerWrapper`

这使完整模型 compile 不再需要展开 60 个真实 decoder layer graph，显著降低编译时间和 graph pass 压力。

因为 M3 是 VL 模型，但本次文本仿真不编译 visual encoder，所以为模型构造空 visual layers：

```python
_EMPTY_VISUAL_LAYERS_ATTR = "_tensor_cast_empty_visual_layers"
setattr(unwrapped, _EMPTY_VISUAL_LAYERS_ATTR, torch.nn.ModuleList())
```

并在 `ModelProfile` 中将 visual layer path 指向该空容器，从而复用现有 VL language layers reuse 逻辑。

## 8. 权重大小估算

`ModelProfile` 增加：

```python
weight_size_estimator: Optional[Callable] = None
```

`TransformerModel.weight_size` 优先调用该 estimator。

MiniMax-M3 MXFP8 estimator 对 3D expert weight 按 MXFP8 存储结构估算：

```text
expert_weight_bytes = weight_numel
expert_scale_bytes  = scale_block_count
```

scale block 只覆盖 expert weight 的最后两个维度，expert 维度作为 batch 维处理。非 3D expert 参数仍使用通用 `bytes_of_tensor()` 估算。

## 9. 编译与 fusion

### 9.1 multistream upward rank

`multistream_pass` 的 upward rank 计算从递归 DFS 改为反向迭代 DP：

```python
for node in reversed(nodes):
    self_cost = min(self._estimate_node_cost_s(node, stream_id) for stream_id in self._allowed_streams(node))
    max_succ_rank = 0.0
    for user in node.users.keys():
        if user in schedulable and user in self._ranks:
            max_succ_rank = max(max_succ_rank, self._ranks[user] + self.cross_stream_sync_overhead_s)
    self._ranks[node] = self_cost + max_succ_rank
```

该修改避免长 FX graph 在 recursive rank 计算中触发 Python recursion limit。当前 M3 已有层复用，常规路径不再强依赖该修复，但该修改仍提升 compile pass 的通用健壮性。

### 9.2 Gemma-style Add RMSNorm

MiniMax-M3 RMSNorm residual fusion 需要支持 `1.0 + weight` 的 effective weight 形式。新增以下 pattern：

- `GemmaAddRMSNormPattern`
- `GemmaAddRMSNorm2Pattern`
- `GemmaAddRMSNormQuantPattern`
- `GemmaAddRMSNormQuant2Pattern`

这些 pattern 匹配时显式构造：

```python
effective_weight = 1.0 + weight
```

然后替换为 TensorCast fused RMSNorm op。

## 10. 验证

基础回归测试：

```bash
python -m pytest tests/regression/tensor_cast/test_minimax_m3.py -v
```

端到端仿真命令：

```bash
python3 -m cli.inference.text_generate /Users/liujiaxu/Code/MiniMax-M3-MXFP8 \
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

已验证的关键现象：

- `minimax_indexer` 和 `minimax_sparse_attention` 能出现在性能统计中
- routed expert 不再落到 `aten._grouped_mm.default`，而是走 `tensor_cast.grouped_matmul_fp8.default`
- 语言层复用后只保留代表 region，其余同构层通过 copy region 表示
- TP8 compile 能完成
- MXFP8 权重大小估算不再按 live tensor / safetensors 粗略放大

## 11. 当前限制与后续工作

1. **EP + shared expert TP 仍需完善**  
   当前 TP8 路径已验证；EP8 + shared expert TP 还需要补齐 `moe_route_after_dp_transform=True` 和 `shared_experts.gate_up_proj` 的 colwise TP 规则。

2. **MXFP8 scale 仍需接入真实 scale**  
   当前 M3 routed expert grouped matmul FP8 路径复用 TensorCast op，但 expert weight scale 仍为占位 scale。

3. **Sparse op 通信未单独建模**  
   `minimax_indexer` 和 `minimax_sparse_attention` 当前只建模计算和访存。如果后续 kernel 实现包含 TP/EP 通信，需要在性能模型中补充通信项。

4. **测试覆盖仍偏基础**  
   现有测试覆盖 op 注册和 meta shape，需要补充 TP1/TP8 compile、EP8、shared expert TP、op count 和 weight size 回归。

5. **性能公式需要 profiling 校正**  
   indexer、sparse attention、grouped matmul 的访存量和 lower/upper bound 需要结合真实 profiling 数据校正。
