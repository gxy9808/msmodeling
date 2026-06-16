# Minimax-M3 DECODE bs=16: Profiling vs msmodeling 仿真对比分析

## 测试配置

| 项目 | 值 |
|------|------|
| 模型 | Minimax-M3-MXFP8 |
| 硬件 | NVIDIA B200 (8卡) |
| 场景 | DECODE, bs=16 |
| 并行策略 | TP=8, EP=8 |
| 量化 | FP8 (MXFP8 权重, 动态激活) |

---

## 1. 总耗时对比

| 来源 | 单步总耗时 | TPS/Device |
|------|-----------|------------|
| GPU Profiling (实测) | 46.07 ms | ~347 token/s |
| msmodeling 仿真 (analytic) | 20.76 ms | 96.32 token/s |
| 差异 | 仿真偏低 54.9% | - |

仿真总耗时约为实测的 45.1%，说明 analytic model 对通信、kernel launch overhead 及 GPU 利用率的建模偏乐观。

---

## 2. 算子级别对比

### 2.1 核心计算算子

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比(仿真/实测) |
|------|-----------------|-----------|------|---------|---------|------|------------------|
| Expert GEMM (6144->384) | deep_gemm sm100_fp8 | 1.847ms | 57 | grouped_matmul_fp8 | 8.416ms | 114 | 4.56x(合并) |
| Expert GEMM (6144->6144->384) | deep_gemm sm100_fp8 | 2.471ms | 57 | (同上) | - | - | - |
| Shared Expert GEMM | mxfp8_block_scaled_matmul | 4.315ms | 240 | fp8_linear | 1.230ms | 132 | 0.28x |
| Attn QKV/O Proj | cutlass sgemm | 0.959ms | 57 | fp8_linear | 0.611ms | 57 | 0.64x |
| Attn Decode Score | decode_score_kernel | 5.738ms | 57 | minimax_sparse_attention | 0.289ms | 57 | **0.05x** |
| GQA Sparse Decode | gqa_share_sparse_decode | 0.718ms | 57 | (含在sparse_attention) | - | - | - |
| KV Cache Store | store_kvcache | 0.113ms | 60 | reshape_and_cache | 0.006ms | 3 | **0.05x** |

### 2.2 通信算子

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比 |
|------|-----------------|-----------|------|---------|---------|------|--------|
| AllReduce | all_reduce_one_shot_push | 15.070ms | 121 | all_reduce | 1.334ms | 120 | **0.09x** |
| AllToAll | (NCCL) | - | - | all_to_all | 1.015ms | 114 | ~1.0x |
| AllGather | ncclDevKernel_AllGather | 0.049ms | 1 | all_gather | 0.484ms | 58 | 9.9x |

### 2.3 归一化算子

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比 |
|------|-----------------|-----------|------|---------|---------|------|--------|
| FusedAddRMSNorm | FusedAddRMSNormKernel | 15.021ms | 120 | add+rsqrt+mul | ~1.16ms | ~554 | **0.08x** |
| RMSNorm | RMSNormKernel | 0.438ms | 234 | pow+mean+rsqrt+mul | ~1.07ms | ~127 | 2.44x |

### 2.4 MoE 路由与 TopK

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比 |
|------|-----------------|-----------|------|---------|---------|------|--------|
| TopK + Gating | topk_index_partial+topkGatingSigmoid | 2.093ms | 114 | topk+sigmoid+gather | 0.342ms | 171 | 0.16x |
| TopK Merge | topk_index_merge_kernel | 0.147ms | 57 | (未建模) | - | - | - |
| Post Reorder | post_reorder_triton_kernel | 1.588ms | 57 | init_routing_v2 | 0.115ms | 57 | **0.07x** |
| Merge Attn Out | merge_topk_attn_out_kernel | 0.135ms | 57 | unpermute_tokens | 0.116ms | 57 | 0.86x |

### 2.5 量化与反量化

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比 |
|------|-----------------|-----------|------|---------|---------|------|--------|
| MXFP8 Quant | mxfp8_block_scaled_matmul | 4.315ms | 240 | dynamic_quantize_symmetric | 2.105ms | 1956 | 0.49x |
| FP8 Downcast | downcast_to_mxfp | 0.375ms | 240 | (含在quantize) | - | - | - |
| SiLU+Mul+Quant | silu_and_mul_post_quant | 0.462ms | 57 | m3_swiglu_quant | 0.124ms | 57 | 0.27x |
| Fill Gateup Input | fill_gateup_input_triton | 0.330ms | 57 | (含在cat) | - | - | - |

### 2.6 拷贝与索引操作

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比 |
|------|-----------------|-----------|------|---------|---------|------|--------|
| Elementwise Copy | direct_copy_kernel_cuda | 1.708ms | 480 | _to_copy | 1.019ms | 498 | 0.60x |
| Cat Array | CatArrayBatchedCopy | 0.759ms | 480 | cat | 1.207ms | 1039 | 1.59x |
| Index Put | index_elementwise_kernel | 0.442ms | 57 | (含在gather等) | - | - | - |
| Memset/Fill | vectorized_elementwise_kernel(Fill) | 0.618ms | 480 | (未单独统计) | - | - | - |
| KV Index | create_flashinfer_kv_indices | 0.166ms | 1 | minimax_indexer | 0.401ms | 57 | 2.42x |

### 2.7 其他算子

| 功能 | Profiling Kernel | Prof. 耗时 | 次数 | 仿真算子 | 仿真耗时 | 次数 | 耗时比 |
|------|-----------------|-----------|------|---------|---------|------|--------|
| RoPE | fused_rope_kernel | 0.237ms | 117 | cos+sin+mul | 0.004ms | 2 | **0.02x** |
| Sort | radixSortKVInPlace | 0.291ms | 57 | (未建模) | - | - | - |
| Transpose+Pack | transpose_and_pack_fp32 | 0.198ms | 57 | (含在quantize) | - | - | - |
| Memcpy HtoD/DtoH | Memcpy HtoD/DtoH | 0.009ms | 2 | (未建模) | - | - | - |

---

## 3. 调用次数对比

| 算子类别 | Profiling 次数 | 仿真次数 | 说明 |
|---------|---------------|---------|------|
| Expert GEMM (FP8) | 57x2 shapes | 114 | 仿真将2种shape合并统计 |
| AllReduce | 121 | 120 | 基本一致(差1次,可能是MTP) |
| AllToAll | 114 | 114 | 完全一致 |
| AllGather | 1 (NCCL合并) | 58 | Profiling中NCCL合并了多次调用 |
| RMSNorm / FusedAddRMSNorm | 120+234=354 | 127+121=248 | 仿真拆分为基础算子,调用次数不同 |
| Dynamic Quantize | 240 (含在matmul) | 1956 | 仿真单独统计每次量化调用 |
| TopK | 57 | 57 | 一致 |
| Attention | 57(decode)+3(prefill) | 57(sparse_attn)+3(attn) | 基本一致 |
| SiLU+Mul+Quant | 57 | 57 | 一致 |
| KV Cache Store | 60 | 3 | 仿真只统计了attention层,未覆盖全部 |
| RoPE | 117 | 2 | 仿真将fused_rope拆为cos/sin,只计1次 |

---

## 4. 关键差异分析

### 4.1 仿真显著低估的算子 (仿真/实测 < 0.3)

| 算子 | 差异倍数 | 原因分析 |
|------|---------|---------|
| Attention Decode Score | 0.05x | 仿真用 analytic roofline 估算, 未考虑 sparse attention 实际访存模式和 block sparse 开销; 实测是高度优化的专用kernel |
| FusedAddRMSNorm | 0.08x | 仿真将 fused kernel 拆解为 add+rsqrt+mul 基础算子, 每个按 roofline 估算耗时极低; 实测 fused kernel 有显著的 kernel launch overhead 和内存带宽竞争 |
| AllReduce | 0.09x | 仿真 analytic model 严重低估通信耗时, 未考虑 NCCL 协议开销、同步等待和实际带宽利用率 |
| Post Reorder | 0.07x | 仿真 init_routing_v2 用简单元数据操作建模, 实测 post_reorder_triton_kernel 涉及复杂 token 重排和数据搬运 |
| RoPE | 0.02x | 仿真将 fused_rope 拆为 cos+sin+mul, 只计算了1次; 实测 fused_rope_kernel 被调用117次(每层+MTP) |
| SiLU+Mul+Quant | 0.27x | 仿真未考虑 fusion 后的额外内存开销和量化精度处理 |
| KV Cache Store | 0.05x | 仿真只建模了3次(attention层), 实际每层都需存储, 共60次 |

### 4.2 仿真显著高估的算子 (仿真/实测 > 2.0)

| 算子 | 差异倍数 | 原因分析 |
|------|---------|---------|
| AllGather | 9.9x | Profiling中NCCL将多次AllGather合并为1次统计; 仿真按58次单独统计 |
| Expert GEMM (grouped) | 4.56x(合并) | 仿真将2种shape合并为grouped_matmul统计, 总耗时偏高; 实测2种shape分别用deep_gemm, 有更好的batch效应 |
| KV Indexer | 2.42x | 仿真单次耗时(7.0us) vs Profiling(166us/1次), 调用次数差异大(57 vs 1) |
| RMSNorm (非fused) | 2.44x | 仿真拆分为基础算子后总耗时反而偏高, 说明拆分后的roofline估算对小张量不精确 |

### 4.3 仿真与实测较接近的算子 (0.5x ~ 2.0x)

| 算子 | 差异倍数 | 说明 |
|------|---------|------|
| Attn QKV/O Proj | 0.64x | 基本合理, 小幅低估 |
| AllToAll | ~1.0x | 匹配良好 |
| Elementwise Copy | 0.60x | 基本合理 |
| Merge Attn Out | 0.86x | 匹配良好 |
| Dynamic Quantize | 0.49x | 偏低, 但量级合理 |
| Cat | 1.59x | 仿真调用次数偏多 |

---

## 5. 耗时分布对比 (Top-5 瓶颈)

### Profiling 实测 Top-5

| 排名 | Kernel | 耗时 | 占比 |
|------|--------|------|------|
| 1 | AllReduce (push) | 15.07ms | 32.7% |
| 2 | FusedAddRMSNorm | 15.02ms | 32.6% |
| 3 | Decode Score (Sparse Attention) | 5.74ms | 12.5% |
| 4 | MXFP8 Block Scaled Matmul (Shared Expert) | 4.32ms | 9.4% |
| 5 | Expert GEMM (6144->6144->384) | 2.47ms | 5.4% |

### msmodeling 仿真 Top-5

| 排名 | 算子 | 耗时 | 占比 |
|------|------|------|------|
| 1 | grouped_matmul_fp8 (Expert GEMM) | 8.42ms | 37.6% |
| 2 | dynamic_quantize_symmetric | 2.11ms | 9.4% |
| 3 | all_reduce | 1.33ms | 6.0% |
| 4 | fp8_linear | 1.23ms | 5.5% |
| 5 | cat | 1.21ms | 5.4% |

**关键发现**: 瓶颈排序完全不同。实测瓶颈是 AllReduce(32.7%) 和 FusedAddRMSNorm(32.6%), 两者合计占 65.3%; 仿真中这两项仅占 11.1%。仿真认为瓶颈是 Expert GEMM(37.6%), 但实测中 Expert GEMM 仅占 ~9.4%。

---

## 6. 算子映射详细对比

### `mxfp8_block_scaled_matmul` (Profiling, 240次)

| 层类型 | 计算项 | 次数/层 | 层数 | 总计 |
|--------|-------|--------|------|------|
| Dense层(3) | qkv_proj + o_proj + gate_up_proj + down_proj | 4 | 3 | 12 |
| Sparse层(57) | qkv_proj + o_proj + index_q_proj + index_k_proj | 4 | 57 | 228 |
| **合计** | | | | **240** |

### `deep_gemm sm100_fp8` (Profiling, 114次)

| 层类型 | 计算项 | 次数/层 | 层数 | 总计 |
|--------|-------|--------|------|------|
| Sparse层(57) | expert gate_up_proj + expert down_proj | 2 | 57 | 114 |

### `fp8_linear` (仿真, 132次)

| 层类型 | 计算项 | 次数/层 | 层数 | 总计 |
|--------|-------|--------|------|------|
| Dense层(3) | q_proj + k_proj + v_proj + o_proj + gate_up_proj + down_proj | 6 | 3 | 18 |
| Sparse层(57) | shared_expert gate_up_proj + shared_expert down_proj | 2 | 57 | 114 |
| **合计** | | | | **132** |

> 注: Dense层 q_proj/k_proj/v_proj 未合并为 qkv_proj, 各自独立调用 fp8_linear (共3次), 而 Profiling 中合并为1次 mxfp8_block_scaled_matmul。

### `grouped_matmul_fp8_bf16` (仿真, 114次) — 替代原 `grouped_matmul_fp8`

| 层类型 | 计算项 | 次数/层 | 层数 | 总计 |
|--------|-------|--------|------|------|
| Sparse层(57) | expert gate_up_proj + expert down_proj | 2 | 57 | 114 |

> 注: 原 `grouped_matmul_fp8` 需要先用 `dynamic_quantize_symmetric` 将 bf16 输入量化为 int8+scale 再传入; 新算子 `grouped_matmul_fp8_bf16` 直接接受 bf16 输入, 内部完成量化与计算, 消除了 MoE expert 的预量化开销。

### `dynamic_quantize_symmetric` (仿真, 132次) — 从 1956 次降至 132 次

| 层类型 | 计算项 | 次数/层 | 层数 | 总计 |
|--------|-------|--------|------|------|
| Dense层(3) | gate_up_proj + down_proj 量化 | 2 | 3 | 6 |
| Sparse层(57) | shared_expert gate_up_proj + down_proj 量化 | 2 | 57 | 114 |
| Sparse层(57) | expert gate_up + down 预量化 | 2 | 57 | 114 | (已消除, 由 grouped_matmul_fp8_bf16 内部完成) |
| 其他 | m3_swiglu_quant + attn 量化 | - | - | 12 |

> 修改前 1956 次中包含 57×4=228 次 MoE expert 预量化 (gate_up 量化 + down 量化, 各2次), 现已消除。剩余 132 次为 non-expert 线性层的动态量化。

### `minimax_sparse_attention` (仿真, 57次)

| 计算项 | 修改前状态 | 修改后状态 | 对应 Profiling |
|--------|-----------|-----------|---------------|
| qkv_proj (Q/K/V投影) | 缺失 | 已覆盖 | mxfp8_block_scaled_matmul 57次 |
| QK norm + RoPE | 缺失 | 已覆盖 | 融合在 decode_score_kernel 中 |
| sparse attention 本身 | 有 | 有 | decode_score_kernel + gqa_share_sparse_decode |
| o_proj | 缺失 | 已覆盖 | mxfp8_block_scaled_matmul 57次 |

> 修改前: 288.969us (5.070us/层) — 仅建模 sparse attention 本身
> 修改后: 512.069us (8.984us/层) — 新增 qkv_proj + o_proj + QK norm + RoPE
> 增量: 223.1us 总计, 3.91us/层

### `minimax_indexer` (仿真, 57次)

| 计算项 | 状态 | 对应 Profiling |
|--------|------|---------------|
| index_q_proj + index_k_proj | 已覆盖(融合在算子内) | mxfp8_block_scaled_matmul 57次 |
| index norm + RoPE | 已覆盖(融合在算子内) | 融合在 indexer kernel 中 |
| block score + topk | 已覆盖(融合在算子内) | _topk_index_partial_kernel |

### Profiling 算子与仿真算子覆盖情况汇总

| Profiling 算子 | 次数 | 修改前仿真覆盖 | 修改后仿真覆盖 |
|--------------|------|--------------|--------------|
| mxfp8_block_scaled_matmul (Dense qkv+o+gate_up+down) | 12 | fp8_linear 18次 | fp8_linear 18次 |
| mxfp8_block_scaled_matmul (Sparse qkv_proj) | 57 | 缺失 | minimax_sparse_attention 内 |
| mxfp8_block_scaled_matmul (Sparse o_proj) | 57 | 缺失 | minimax_sparse_attention 内 |
| mxfp8_block_scaled_matmul (Sparse index_q_proj) | 57 | minimax_indexer 内 | minimax_indexer 内 |
| mxfp8_block_scaled_matmul (Sparse index_k_proj) | 57 | minimax_indexer 内 | minimax_indexer 内 |
| mxfp8_block_scaled_matmul (Sparse shared_expert) | 114 | fp8_linear 114次 | fp8_linear 114次 |
| deep_gemm (Sparse expert gate_up+down) | 114 | grouped_matmul_fp8 114次 | grouped_matmul_fp8_bf16 114次 |

---

## 7. 修改记录: minimax_sparse_attention 新增 qkv_proj 和 o_proj 建模

### 7.1 问题描述

修改前, `minimax_sparse_attention` 算子的 boundary 定义为 `read Q + selected K/V cache -> sparse QK/PV attention -> output O`, 假设输入已经是投影后的 Q, 输出不需要 o_proj 映射。但实际上:

- **输入** hidden_states 需要经过 qkv_proj 才能得到 Q/K/V, 这步计算被跳过
- **输出** sparse attention 的结果需要经过 o_proj 才能得到最终输出, 这步计算也被跳过

qkv_proj 和 o_proj 既没有作为独立算子出现在仿真图中, 也没有被 minimax_sparse_attention 的 performance model 内部建模, 计算开销完全丢失。

### 7.2 修改内容

**文件1: `tensor_cast/ops/minimax_m3_sparse_attention.py`**
- `minimax_sparse_attention` 算子新增 `hidden_size` kwargs 参数
- boundary 更新为: `hidden -> qkv_proj -> QK norm + RoPE -> sparse QK/PV attention -> o_proj -> output`

**文件2: `tensor_cast/layers/minimax_m3_attention.py`**
- 调用 `minimax_sparse_attention` 时传入 `hidden_size=self.hidden_size`

**文件3: `tensor_cast/performance_model/__init__.py`**
- `_estimate_minimax_sparse_attention_breakdown` 新增 `hidden_size` 参数, 并增加以下建模:

| 新增计算项 | MMA 公式 | 字节公式 |
|-----------|---------|---------|
| qkv_proj | `2 * T * H * (N_q*D + 2*N_kv*D)` | input + weight + output |
| QK norm | `12 * T * N_q * D + 12 * T * N_kv * D` (GP) | 4 * T * N_q * D * s + 4 * T * N_kv * D * s |
| RoPE | `6 * T * (N_q + N_kv) * D` (GP) | 2 * T * (N_q + N_kv) * D * s |
| o_proj | `2 * T * N_q * D * H` | input + weight + output |

- `register_op_properties` 读取 `hidden_size` 并传递给 breakdown 函数

### 7.3 修改前后对比

| 指标 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| minimax_sparse_attention 总耗时 | 288.969us | 512.069us | +223.1us (+77.2%) |
| minimax_sparse_attention 平均耗时/层 | 5.070us | 8.984us | +3.914us/层 |
| 仿真总耗时 | 22.375ms | 22.598ms | +0.223ms |

### 7.4 仍需改进的项

| 计算项 | 当前状态 | 说明 |
|--------|---------|------|
| Dense层 q/k/v 未合并 | 3次独立 fp8_linear | 应合并为1次 merged qkv_proj, 与 Profiling 的 mxfp8_block_scaled_matmul 对齐 |
| Sparse层 qkv_proj/o_proj | 已在 minimax_sparse_attention 内建模 | 但未作为独立 fp8_linear 算子出现, 无法单独观察耗时 |
| QK norm + RoPE | 已在 minimax_sparse_attention 内建模 | 与 SGLang 实测的 fused_rope_kernel 调用117次仍有差异 |

---

## 8. 修改记录: 引入 grouped_matmul_fp8_bf16 消除 MoE 预量化开销

### 8.1 问题描述

修改前, MoE expert 的 `grouped_matmul_fp8` 需要先用 `dynamic_quantize_symmetric` 将 bf16 输入量化为 int8+scale 再传入, 导致每个 MoE 层产生 4 次额外的量化调用 (gate_up 量化 2次 + down 量化 2次), 57 层共 228 次。这些预量化调用在 Profiling 中并不存在 (expert matmul 由 `deep_gemm` 直接完成量化与计算), 属于仿真框架的额外开销。

### 8.2 修改内容

**文件1: `tensor_cast/ops/gmm.py`**
- 新增 `grouped_matmul_fp8_bf16` 和 `grouped_matmul_mxfp4_bf16` 算子注册, 接受 bf16 输入, 无需外部预量化

**文件2: `tensor_cast/transformers/builtin_model/minimax_m3.py`**
- `_grouped_matmul` FP8 路径: 改用 `grouped_matmul_fp8_bf16`, 移除手动 `dynamic_quantize_symmetric` 调用
- `_grouped_matmul_with_prequant`: 同上, 移除预量化, 改用 `grouped_matmul_fp8_bf16`

**文件3: `tensor_cast/performance_model/__init__.py`**
- 为 `grouped_matmul_fp8_bf16` 和 `grouped_matmul_mxfp4_bf16` 注册 performance model, 内部复用 `_static_quant_linear_properties_helper` 进行建模

### 8.3 修改前后对比

| 指标 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| MoE expert matmul 算子 | `grouped_matmul_fp8` (114次) | `grouped_matmul_fp8_bf16` (114次) | 算子替换 |
| `dynamic_quantize_symmetric` 调用次数 | 1956 | 132 | -1824 (-93.3%) |
| `dynamic_quantize_symmetric` 总耗时 | 2.105ms | 0.265ms | -1.840ms |
| expert matmul 总耗时 | 8.416ms | 8.422ms | +0.006ms (基本不变) |
| 仿真总耗时 | 22.598ms | 20.764ms | -1.834ms (-8.1%) |
| TPS/Device | 88.5 token/s | 96.32 token/s | +8.8% |

> 注: expert matmul 耗时基本不变, 因为 `grouped_matmul_fp8_bf16` 的 performance model 内部仍包含量化的计算开销建模。仿真总耗时下降主要来自消除了独立的 `dynamic_quantize_symmetric` 调用的 kernel launch overhead 估算。

### 8.4 仍需改进的项

| 计算项 | 当前状态 | 说明 |
|--------|---------|------|
| expert 预量化已消除 | `grouped_matmul_fp8_bf16` 内部包含 | 与 Profiling 的 `deep_gemm` 行为更一致 |
| shared_expert 预量化仍独立 | 132次 `dynamic_quantize_symmetric` | shared_expert 的 fp8_linear 仍需外部量化, 但与 Profiling 行为一致 |

---

## 9. 仿真结果汇总 (截至最新修改)

### 最新仿真输出 (bs=16, DECODE)

| 算子 | 总耗时 | 平均耗时 | 调用次数 |
|------|--------|---------|---------|
| grouped_matmul_fp8_bf16 | 8.422ms | 73.873us | 114 |
| all_reduce | 1.334ms | 11.117us | 120 |
| fp8_linear | 1.230ms | 9.319us | 132 |
| cat | 1.207ms | 1.162us | 1039 |
| mul.Tensor | 1.159ms | 2.063us | 562 |
| add.Tensor | 1.121ms | 2.024us | 554 |
| _to_copy | 1.019ms | 2.046us | 498 |
| all_to_all | 1.015ms | 8.900us | 114 |
| minimax_sparse_attention | 512.069us | 8.984us | 57 |
| all_gather | 484.141us | 8.347us | 58 |
| minimax_indexer | 400.860us | 7.033us | 57 |
| mm.default | 350.539us | 6.044us | 58 |
| pow.Tensor_Scalar | 268.491us | 2.114us | 127 |
| dynamic_quantize_symmetric | 265.305us | 2.010us | 132 |
| mean.dim | 261.247us | 2.057us | 127 |
| rsqrt.default | 254.003us | 2.000us | 127 |
| clamp.default | 241.140us | 2.009us | 120 |
| sigmoid.default | 234.588us | 2.005us | 117 |
| sum.dim_IntList | 230.124us | 2.019us | 114 |
| m3_swiglu_quant | 124.192us | 2.179us | 57 |

| 总计 | **20.764ms** | TPS/Device | **96.32 token/s** |

### 仿真结果历史对比

| 版本 | 总耗时 | TPS/Device | 关键变化 |
|------|--------|-----------|---------|
| 初始版本 | 22.375ms | 89.38 token/s | - |
| +minimax_sparse_attention 建模 qkv_proj/o_proj | 22.598ms | 88.5 token/s | sparse attention 耗时 +223us |
| +grouped_matmul_fp8_bf16 消除预量化 | 20.764ms | 96.32 token/s | dynamic_quantize -1824次, 总耗时 -1.834ms |

---

## 10. 总结与改进建议

### 10.1 核心问题

1. **通信模型严重不足**: AllReduce 仿真 1.33ms vs 实测 15.07ms (0.09x), 最大误差来源。Analytic model 未建模 NCCL 协议开销、同步等待延迟和实际带宽利用率。
2. **Fused kernel 拆分导致低估**: FusedAddRMSNorm 被拆为基础算子, roofline 总和远低于实测 fused kernel 耗时, 未考虑 kernel launch overhead 和内存带宽竞争。
3. **Sparse Attention 建模不足**: minimax_sparse_attention 修改后已包含 qkv_proj 和 o_proj, 但核心的 sparse attention 计算部分(0.289ms) vs 实测 decode_score(5.738ms) 仍差距大, 未反映 block sparse attention 的实际访存模式。
4. **KV Cache 操作缺失**: reshape_and_cache 仅3次 vs 实测60次, 未完整建模每层的 KV 存储操作。
5. **RoPE 调用次数错误**: 仿真只计2次(cos+sin) vs 实测117次 fused_rope, 未逐层调用。

### 10.2 改进建议

| 优先级 | 改进项 | 预期收益 |
|--------|--------|---------|
| P0 | 引入 profiling-based 通信模型 (NCCL 实测带宽/延迟) | 消除 AllReduce 0.09x 误差 |
| P0 | 为 fused kernel (FusedAddRMSNorm, fused_rope) 建立专用 TC 算子 | 消除拆分导致的低估 |
| P1 | 为 minimax_sparse_attention 引入 profiling-based 性能模型 | 修正 sparse attention 核心计算的严重低估 |
| P1 | 补全 KV Cache Store 逐层调用建模 | 修正调用次数 3 vs 60 |
| P1 | Dense层 q/k/v 合并为 merged qkv_proj | 与 Profiling 的 mxfp8_block_scaled_matmul 对齐 |
| P2 | 为 MoE Post Reorder (token permutation) 建立专用算子 | 修正 0.07x 低估 |
| P2 | grouped_matmul_fp8 按 shape 分别建模 | 避免合并统计导致的高估 |
