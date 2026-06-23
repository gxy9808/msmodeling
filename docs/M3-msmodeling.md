# MiniMax-M3 模型在 msmodeling 中的适配指南

> 日期：2026-06-06（2026-06-07 更新：添加代码实现部分与测试验证结果）

---

## 1. 背景

MiniMax-M3 是 MiniMax 的第三代大语言模型，核心特点是 **Sparse Attention（稀疏注意力）机制** —— 在部分 attention layer 中使用 index-head 进行 top-k 稀疏注意力，与标准 dense attention 混合。

本指南分析如何在 msmodeling（TensorCast）中为 MiniMax-M3 注册模型适配，使其能够进行性能建模仿真。

---

## 2. MiniMax-M3 模型架构关键特征

> 注意：Minimax-M3是VL类模型，顶层
> `model_type="minimax_m3_vl"`；语言模型参数位于`text_config` 中。
> 本节描述的是`text_config` 对应的 MiniMax-M3 文本 backbone。

### 2.1 总体结构

```
MiniMaxM3SparseForCausalLM
  └── model (MiniMaxM3Model)
      ├── embed_tokens (VocabParallelEmbedding)
      ├── layers (MiniMaxM3DecoderLayer × num_hidden_layers)
      │   ├── input_layernorm (RMSNorm / GemmaRMSNorm)
      │   ├── self_attn (MiniMaxM3Attention)
      │   │   ├── qkv_proj (QKVParallelLinear)          # 标准 QKV 投影
      │   │   ├── o_proj (RowParallelLinear)             # 输出投影
      │   │   ├── q_norm / k_norm (RMSNorm per_head)    # 按 head_dim 归一化，共享权重
      │   │   ├── rotary_emb (partial RoPE)              # 部分维度 RoPE, rotary_dim=64
      │   │   ├── [sparse] index_q_proj / index_k_proj   # index 分支 Q/K 投影
      │   │   │   └── index_v_proj / index_o_proj        # 可选；M3-preview sparse layer 均关闭
      │   │   │   └── index_q/k_norm                     # index 分支 QK 归一化
      │   │   └── attn (RadixAttention)
      │   ├── mlp (MiniMaxM3MLP | MiniMaxM3MoE)
      │   │   ├── gate_up_proj (MergedColumnParallelLinear)
      │   │   ├── down_proj (RowParallelLinear)
      │   │   └── [MoE] experts (FusedMoE) + gate (ReplicatedLinear)
      │   │       └── [MoE] shared_experts (MiniMaxM3MLP)
      │   └── post_attention_layernorm (RMSNorm / GemmaRMSNorm)
      └── norm (RMSNorm / GemmaRMSNorm)
  └── lm_head (ParallelLMHead)
```

### 2.2 Sparse Attention 机制

MiniMax-M3 的 Sparse Attention 由`text_config.sparse_attention_config` 控制。 `Minimax-M3-preview` 中共有 60 层，`sparse_attention_freq` 前 3 层为 0（dense attention），layer 3-59 为 1（sparse attention）。对应配置如下：

```python
sparse_attention_config = {
    "use_sparse_attention": True,
    "sparse_attention_freq": [0, 0, 0, 1, 1, ...],  # 每层 0=dense, 1=sparse
    "sparse_num_index_heads": 4,                    # index 头数量
    "sparse_index_dim": 128,                        # index 头维度
    "sparse_block_size": 128,                       # 稀疏 block 大小
    "sparse_topk_blocks": 16,                       # top-k block 数量
    "sparse_local_block": 1,                        # 局部 block
    "sparse_disable_index_value": [0, 0, 0, 1, ...],# 每层 0/1 mask，1=跳过 index V/O 分支
    "sparse_score_type": "max",                     # score 聚合方式
}
```

`Minimax-M3-preview` 的所有 sparse layer（layer 3-59）都有`sparse_disable_index_value=1`，因此实际 sparse layer 只使用`index_q_proj` / `index_k_proj` 生成 top-k block 索引，不执行`index_v_proj` / `index_o_proj` 的输出加和路径。

**Sparse Layer 的 Attention 计算流程（`MiniMaxM3Attention.forward_prepare` + `forward_core`）：**

```
1. qkv_proj: hidden -> q, k, v                        # 标准 QKV 投影
2. q_norm/k_norm: q, k -> norm_q, norm_k               # per_head QK RMSNorm
3. rotary_emb: norm_q, norm_k -> q_rot, k_rot          # 部分维度 RoPE
   ── 以下仅 sparse layer 执行 ──
4. index_q/k_proj: hidden -> idx_q, idx_k              # index 分支 Q/K 投影
5. [可选] index_v_proj: hidden -> idx_v                # M3-preview sparse layer 均跳过
6. index_q/k_norm: idx_q, idx_k -> norm_idx_q, norm_idx_k
7. radix_attention(q, k, v, idx_q, idx_k, idx_v)       # HybridAttnBackend 路由
   ├── dense: flash_attention(q, k, v)                 # 标准 attention
   └── sparse: topk_index(idx_q, idx_k, cache)         # index 分支 top-k 选择
              + sparse_flash_attention(q, k, v, topk) # 稀疏 attention
8. o_proj: attn_output -> output                       # 主 attention 输出投影
9. [可选] index_o_proj: idx_o -> idx_output            # 仅 disable_index_value=False 时执行
10. [可选] output = output + idx_output                # M3-preview sparse layer 不执行
```

SGLang 的 sparse attention backend 在 sparse layer 中返回`idx_o, attn_output`。当`disable_index_value=True` 时，`idx_o` 分支被跳过， 仅返回`o_proj(attn_output)`；当该分支启用且 TP 下存在 index head replica 时， 实现会在`index_o_proj` 前将`idx_o` 除以`idx_replica_size`，避免 all-reduce 后重复累加。

### 2.3 其他关键特征

| 特征 | 说明 |
|------|------|
| **QK Normalization** | `qk_norm_type="per_head"`：将 Q/K reshape 为多个`head_dim` 向量后逐 head 归一化；权重形状为`(head_dim,)`，各 head 共享权重 |
| **Partial RoPE** | `rotary_dim=64`，仅部分 head_dim 施加 RoPE，其余不动 |
| **Gemma 风格** | `use_gemma_norm=True`，使用`x * (1 + weight)` 的 GemmaRMSNorm |
| **Attention Output Gate** | 代码支持 dense attention 场景下的可选 gating；M3-preview 中`attention_output_gate=false`，且 sparse layer 不支持该开关 |
| **MoE + Dense 混合** | `moe_layer_freq` 前 3 层为 0，layer 3-59 为 1：前 3 层 Dense MLP，后 57 层 MoE |
| **Sparse Attention 分布** | `sparse_attention_freq` 前 3 层为 0，layer 3-59 为 1；所有 sparse layer 均设置`sparse_disable_index_value=1` |
| **SwiGLU 变体** | `hidden_act="swigluoai"`，带`swiglu_alpha=1.702` 和`swiglu_limit=7.0` |
| **DP Attention** | 支持 Data Parallel Attention |
| **LayerCommunicator** | 控制 allreduce 融合和 reduce scatter |
| **MultiHeadRMSNorm** | 备选`qk_norm_type="multi_head"` 分支，每个 head 有独立 RMSNorm 权重；M3-preview 默认不是这个分支 |
| **MTP (Multi-Token Prediction)** | `num_mtp_modules=1`，支持 MTP；不支持 EAGLE3 |

---

## 3. msmodeling 四层架构与适配思路

TensorCast 的性能建模采用 **"一算子一建模"** 架构，模型适配通过四层协作完成：

```
模型定义层:  transformers/builtin_model/minimax_m3.py
   │  注册 ModelProfile + @register_custom_model
   ▼
抽象算子层:  layers/minimax_m3_attention.py (新建)
   │  封装 forward，调用虚拟算子
   ▼
虚拟算子层:  ops/minimax_m3_sparse_attention.py (新建)
   │  @register_tensor_cast_op → 定义 meta-only 算子签名
   ▼
性能模型层:  performance_model/__init__.py (追加注册)
   │  @register_op_properties → Roofline 拆解 (FLOPs + memory)
   └─ @register_op_estimator → 直接估算（可选）
```

**核心原则（参照 DSA-Indexer 的建模方法论）：**

1. **虚拟算子层**只做"身份锚定"，不参与任何计算或时间估算
2. **性能模型层**根据算子类型注册 FLOPs 和内存访问量的拆解函数
3. **拆解函数**严格依据虚拟算子 docstring 描述的算法步骤，把每一步划分到 MMA 或 GP 桶
4. **Roofline 模型**最终决定时间 = max(compute_time, memory_time)

---

## 4. Indexer 与 Sparse Attention 的 op 拆分和公式

MiniMax-M3 sparse layer 可以拆成两个主要建模边界：

1. `minimax_indexer`：从 index Q/K projection 开始，到 top-k block 输出结束
2. `minimax_sparse_attention`：主 attention 根据 top-k block 访问标准 K/V cache

术语约定：

| 术语 | 说明 |
|------|------|
| **MMA** | Matrix Multiply-Accumulate，对应 GEMM/BMM/attention QK 和 PV 等矩阵乘累加计算；本指南按 FMA = 2 FLOPs 估算，例如 $[M, K] \times [K, N]$ 的计算量为 $2MKN$。 |
| **GP** | General Purpose elementwise/reduction work，表示非矩阵乘的通用计算，如 RMSNorm、RoPE、mask、max reduce、top-k 选择、softmax 近似等。 |
| **FLOPs** | 浮点操作数。性能模型按 dtype 将 MMA/GP 分桶，再结合吞吐和带宽做 roofline 估算。 |
| **访存量** | 逻辑读写字节数，包括 activation、weight、cache、score/top-k index 的读写；真实 kernel 有 cache reuse 时可用 lower/upper bound 修正。 |
| **lower / upper** | 同一数据可能被多个 head 复用时的访存下界/上界。`lower` 假设复用充分，`upper` 假设几乎无复用。 |

变量约定：

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $Q_b$ | 第 $b$ 个 request 本轮 query token 数；decode 中通常为 1 |
| $L_b$ | 第 $b$ 个 request 当前总 KV 长度，包含 prefix 和本轮新增 token |
| $T$ | 本轮总 query token 数 |
| $H$ | hidden size；M3-preview 为 6144 |
| $B_n = \lceil L_b / B_s \rceil$ | 第 $b$ 个 request 的 KV block 数；在 $\sum_b$ 内按当前 request 的 $L_b$ 取值 |
| $N_q$ | 本 rank 上的 query heads |
| $N_{kv}$ | 本 rank 上的 KV heads |
| $N_{\mathrm{indexer}}$ / $D_{\mathrm{indexer}}$ | 本 rank 上的 index heads / index head dim；M3-preview 为总量 4 / 128 |
| $D$ | 标准 attention head dim；M3-preview 为 128。4.1 中 $D$ 是局部简写，含义见 4.1 开头说明 |
| $D_r$ | index RoPE 维度；M3-preview 可按`rotary_dim=64` |
| $K$ | `sparse_topk_blocks`；M3-preview 为 16 |
| $B_s$ | `sparse_block_size`；M3-preview 为 128 |
| $R$ | `sparse_local_block`；M3-preview 为 1 |
| $s$ | activation/cache dtype 字节数，bf16/fp16 时 $s=2$ |

### 4.1 Indexer

> **本节局部符号约定**：为简化公式，4.1 中的 $N$ 和 $D$ 分别表示
> $N_{\mathrm{indexer}}$ 和 $D_{\mathrm{indexer}}$，即 indexer head 数和
> indexer head dim；它们不表示 4.2 主 attention 中的 query/KV head 数或标准 head dim。

参照`_estimate_dsa_indexer_breakdown` 的拆解口径，Indexer 可以作为一个从 index Q/K projection 开始、到 top-k block 输出结束的 fused 虚拟 op： `minimax_indexer`。该 op 覆盖 index 分支的投影、norm、RoPE、index cache 写入、block score 和 top-k。

#### 4.1.1 Index Q/K Projection

MiniMax-M3 的 index 分支在 sparse layer 中额外执行：

```text
index_q_proj: hidden [T, H] -> idx_q [T, N, D]
index_k_proj: hidden [T, H] -> idx_k [T, N, D]
```

计算量：

$$
\begin{aligned}
\mathrm{index\_q\_proj\_mma} &= 2THND \\
\mathrm{index\_k\_proj\_mma} &= 2THND
\end{aligned}
$$

访存量：

$$
\begin{aligned}
\mathrm{read\_hidden\_bytes} &= 2THs \\
\mathrm{read\_wq\_bytes} &= HNDs \\
\mathrm{read\_wk\_bytes} &= HNDs \\
\mathrm{write\_idx\_qk\_bytes} &= 2TNDs
\end{aligned}
$$

#### 4.1.2 Index Q/K Norm 与 RoPE

`qk_norm_type="per_head"` 时，`idx_q` 和`idx_k` 都 reshape 到 $[-1, D]$ 后做 GemmaRMSNorm；随后对前 $D_r$ 维施加 RoPE。

计算量：

$$
\begin{aligned}
\mathrm{index\_norm\_gp} &\approx 12TND \\
\mathrm{index\_rope\_gp} &\approx 6TND_r
\end{aligned}
$$

访存量：

$$
\begin{aligned}
\mathrm{read\_write\_norm\_bytes} &\approx 4TNDs \\
\mathrm{read\_write\_rope\_bytes} &\approx 4TNDs \\
\mathrm{read\_norm\_weight\_bytes} &= 2NDs
\end{aligned}
$$

常量来源：

| 常量 | 来源 |
|------|------|
| $12$ | `idx_q` 和`idx_k` 两路都做 RMSNorm；每路按每元素约 $6$ 个 GP 操作估算，包括平方、reduce 均摊、rsqrt、scale、weight 和写回相关的 elementwise 开销，因此为 $2 \times 6$。 |
| $6$ | `idx_q` 和`idx_k` 两路都做 RoPE；每路对前 $D_r$ 维按每元素约 $3$ 个 GP 操作估算，包括 sin/cos 旋转中的乘加和交换/组合开销，因此为 $2 \times 3$。 |
| 第一个 $4$ | norm 访存包含 Q/K 两路，每路近似一次读输入、一次写输出，即 $2 \times 2TNDs$。 |
| 第二个 $4$ | RoPE 访存同样包含 Q/K 两路，每路近似一次读、一次写；即使只旋转前 $D_r$ 维，实际 kernel 往往按完整 head tensor 读写，因此保守写成 $4TNDs$。 |
| $2$ | norm weight 有 Q/K 两套参数，各自大小为 $ND$，因此为 $2NDs$。 |

GemmaRMSNorm 的`1 + weight` 可并入 norm 的 elementwise GP 桶，不需要单独建 MMA。

#### 4.1.3 Index Cache Write

写入当前 token 的 index K cache。

计算量：

$$
\mathrm{MMA}=0,\qquad \mathrm{GP}=0
$$

访存量：

$$
\begin{aligned}
\mathrm{read\_idx\_k\_bytes\_current} &= TNDs \\
\mathrm{write\_idx\_k\_cache\_bytes} &= TNDs
\end{aligned}
$$

#### 4.1.4 Index Block Score

对应 vLLM 中 `_index_block_score_kernel`（`vllm/models/minimax_m3/common/ops/index_topk.py`）的 score 部分：`idx_q` 与 index K cache 打分，然后按 `sparse_block_size` 聚合到 block score。index Q/K 都有 $N$ 个 head，打分时按 indexer head 对齐。

计算量按 request 求和：

$$
\begin{aligned}
\mathrm{index\_qk\_mma}
  &= 2\sum_b Q_bNL_bD \\
\mathrm{block\_reduce\_gp}
  &\approx \sum_b Q_bNL_b
  && \text{score\_type = max}
\end{aligned}
$$

M3-preview 使用`sparse_score_type="max"`，这里只按 max reduce 建模。

访存量建议按逻辑访问量估算：

$$
\begin{aligned}
\mathrm{read\_idx\_q\_bytes}
  &= TNDs \\
\mathrm{read\_idx\_k\_cache\_bytes}
  &= \sum_b \frac{Q_b}{B_q} N L_b D s \\
\mathrm{write\_score\_bytes}
  &= 4\sum_b Q_bNB_n
\end{aligned}
$$

其中 $B_q = 64$ 是kernel中Q的tile大小：key cache被加载一次后被 $B_q$ 个 query token 共享，因此 K cache 读量按 query tile 数 $\lceil Q_b / B_q \rceil$ 而非 query token 数计；近似写成 $Q_b / B_q$。

#### 4.1.5 Top-k Selection

从 block score 中选择 top-k block，并输出给 4.2 的 sparse attention：

```text
topk_idx: [N, T, K]  # 或按 kernel 排布为等价形状
```

计算量：

$$
\mathrm{topk\_gp}
\approx c_{\mathrm{topk}}\sum_b Q_bNB_n
$$

其中`c_topk` 是 top-k 选择的近似常数。若性能模型只需要稳定的一阶估算， 可取 $c_{\mathrm{topk}}=K$；若要更接近选择算法，可取 $c_{\mathrm{topk}}=\lceil \log_2(\max(K, 2)) \rceil$。

访存量：

$$
\begin{aligned}
\mathrm{read\_score\_bytes} &= 4\sum_b Q_bNB_n \\
\mathrm{write\_topk\_bytes} &= 4TNK
\end{aligned}
$$

#### 4.1.6 Indexer 汇总

`minimax_indexer` 按完整 fused 子图建模，从 index Q/K projection 一直计到 top-k block 输出。汇总时包含 projection、norm、RoPE、index K cache write、 block score 和 top-k selection。

$$
\begin{aligned}
\mathrm{MMA}_{\mathrm{indexer}}
  &= \mathrm{index\_q\_proj\_mma}
   + \mathrm{index\_k\_proj\_mma}
   + \mathrm{index\_qk\_mma} \\
\mathrm{GP}_{\mathrm{indexer}}
  &= \mathrm{index\_norm\_gp}
   + \mathrm{index\_rope\_gp}
   + \mathrm{block\_reduce\_gp}
   + \mathrm{topk\_gp}
\end{aligned}
$$

代入 4.1.1 到 4.1.5 的公式后：

$$
\begin{aligned}
\mathrm{MMA}_{\mathrm{indexer}}
  &= 4THND + 2\sum_b Q_bNL_bD \\
\mathrm{GP}_{\mathrm{indexer}}
  &\approx 12TND + 6TND_r
   + \sum_b Q_bNL_b
   + c_{\mathrm{topk}}\sum_b Q_bNB_n
\end{aligned}
$$

对应的逻辑访存量可按各阶段读写相加：

$$
\begin{aligned}
\mathrm{Bytes}_{\mathrm{indexer}}
  &\approx
    \underbrace{2THs + 2HNDs + 2TNDs}_{\text{index Q/K projection}} \\
  &\quad+
    \underbrace{4TNDs + 4TNDs + 2NDs}_{\text{index Q/K norm + RoPE}} \\
  &\quad+
    \underbrace{2TNDs}_{\text{index K cache write}} \\
  &\quad+
    \underbrace{TNDs + \sum_b \frac{Q_b}{B_q}NL_bDs + 4\sum_b Q_bNB_n}_{\text{block score}} \\
  &\quad+
    \underbrace{4\sum_b Q_bNB_n + 4TNK}_{\text{top-k selection}}
\end{aligned}
$$

其中 index K cache 读按 vLLM `_index_block_score_kernel` 的实测行为建模：每个 128-token K-block 被加载一次后被 $B_q = 64$ 个 query token 复用（`tl.dot(q, k)`），读量为 $\sum_b \frac{Q_b}{B_q}NL_bDs$。

### 4.2 Sparse Attention

Sparse attention 使用 Indexer 输出的`topk_idx`，从标准 K/V cache 中取 selected blocks 做主 attention。本节只需要建模`minimax_sparse_attention`。

计算量：

$$
\mathrm{MMA}_{\mathrm{sparse\_attn}}
  = 4N_qTDkB_k
$$

这里的系数 $4$ 来自 sparse attention 的两次矩阵乘：

$$
\underbrace{2N_qTDkB_k}_{QK^\top}
+
\underbrace{2N_qTDkB_k}_{PV}
= 4N_qTDkB_k
$$

每次矩阵乘按 FMA = 2 FLOPs 估算；$kB_k$ 表示每个 query token 访问的 selected token 上界。

访存量：

$$
\begin{aligned}
\mathrm{Bytes}_{Q+\mathrm{KV}}
  &=
    \underbrace{2sN_qTD}_{\text{Q read + O write}}
    +
    \underbrace{2sN_{kv}TkB_kD}_{\text{KV read}} \\
  &= 2sTD\left(N_q + N_{kv}kB_k\right)
\end{aligned}
$$

常量来源：

| 常量 | 来源 |
|------|------|
| 第一个 $2$ | Q 读一次，attention 输出 O 写一次，二者形状同为 $[T, N_q, D]$，因此为 $2sN_qTD$。 |
| 第二个 $2$ | KV 包含 K cache 和 V cache 两份数据，各读取一次；因此为 $2sN_{kv}TkB_kD$。 |
| $s$ | 每个元素的字节数，bf16/fp16 时 $s=2$。 |
| $kB_k$ | 每个 query token 实际访问的 selected token 上界：$k$ 个 block，每个 block $B_k$ 个 token。 |

---

## 5. 各层级详细修改方案

本章按第 3 章的四层架构顺序展开：**模型定义层 -> 抽象算子层 -> 虚拟算子层 -> 性能模型层**。适配目标是把 MiniMax-M3 sparse layer 显式拆成两个可建模边界：

1. `minimax_indexer`：从 index Q/K projection 到 top-k block 输出，公式见 4.1。
2. `minimax_sparse_attention`：根据 top-k block 访问标准 K/V cache 做 sparse attention，公式见 4.2。

### 5.1 模型定义层：注册 MiniMax-M3 ModelProfile

模型定义层负责识别 HF/SGLang 模型结构，并把模型里的 attention layer 替换或包装成 TensorCast 可追踪的抽象算子。建议新增：

```text
tensor_cast/transformers/builtin_model/minimax_m3.py
```

需要完成三件事：

1. 注册`MiniMaxM3` 的`ModelProfile` 和`@register_custom_model`。
2. 从`config.text_config.sparse_attention_config` 读取 sparse 参数。
3. 按 layer id 判断 dense/sparse layer，并把 attention 模块映射到抽象算子层。

关键配置字段：

| 字段 | 用途 |
|------|------|
| `hidden_size` | 第 4 章中的 $H$ |
| `num_attention_heads` | 标准 attention query head 总数 |
| `num_key_value_heads` | 标准 attention KV head 总数 |
| `head_dim` | 标准 attention head dim，即 4.2 的 $D$ |
| `rotary_dim` | index RoPE 维度 $D_r$ |
| `sparse_attention_freq` | 判断每层 dense / sparse |
| `sparse_num_index_heads` | indexer head 总数 $N_{\mathrm{indexer}}$ |
| `sparse_index_dim` | indexer head dim $D_{\mathrm{indexer}}$ |
| `sparse_topk_blocks` | top-k block 数 $K$ |
| `sparse_block_size` | block size $B_s$ |
| `sparse_local_block` | local block 数 $R$ |
| `sparse_score_type` | M3-preview 为`max` |

模型定义层不直接计算 FLOPs 或 bytes，只负责把这些静态配置传给后续 wrapper。对于 M3-preview，应固定以下判断：

```python
sparse_cfg = config.text_config.sparse_attention_config
is_sparse_layer = sparse_cfg["sparse_attention_freq"][layer_idx] == 1
```

注意：M3-preview 的 sparse layer 不建模 index value / index output 分支；第 5 章的适配路径只包含 index Q/K 和主 attention K/V。

### 5.2 抽象算子层：封装 MiniMax-M3 Attention Forward

抽象算子层负责把模型 forward 中的具体张量和 runtime metadata 组织成虚拟算子调用。建议新增：

```text
tensor_cast/layers/minimax_m3_attention.py
```

建议提供一个 wrapper，例如`MiniMaxM3AttentionWrapper`，职责如下：

| 职责 | 说明 |
|------|------|
| dense layer | 继续走已有 dense attention 建模路径 |
| sparse layer | 调用`minimax_indexer` 后，再调用`minimax_sparse_attention` |
| shape 提取 | 从 hidden states、QKV、seq lens、block table 中提取 $T, Q_b, L_b, B_n$ |
| config 传递 | 传递 $H, N_q, N_{kv}, D, K, B_s, R, N_{\mathrm{indexer}}, D_{\mathrm{indexer}}, D_r$ |

抽象算子层推荐的 sparse forward 逻辑：

```python
if is_sparse_layer:
    topk_idx = torch.ops.tensor_cast.minimax_indexer(
        hidden_states,
        seq_lens,
        query_lens,
        block_table,
        hidden_size=hidden_size,
        num_indexer_heads=num_indexer_heads,
        indexer_head_dim=indexer_head_dim,
        indexer_rope_dim=indexer_rope_dim,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )

    out = torch.ops.tensor_cast.minimax_sparse_attention(
        query,
        key_cache,
        value_cache,
        topk_idx,
        seq_lens,
        query_lens,
        block_table,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        topk_blocks=topk_blocks,
        block_size=block_size,
        local_blocks=local_blocks,
    )
else:
    out = torch.ops.tensor_cast.attention(...)
```

这里的`seq_lens` 和`query_lens` 很关键：性能模型应按 request 求和，不能用固定的平均序列长度替代。抽象算子层需要保证它们能传到性能模型层，至少能还原第 4 章公式中的 $Q_b$ 和 $L_b$。

### 5.3 虚拟算子层：定义 meta-only op 签名

虚拟算子层只做身份锚定和 shape 传播，不做真实计算，也不估算性能。建议新增：

```text
tensor_cast/ops/minimax_m3_sparse_attention.py
```

只需要注册两个 MiniMax-M3 专用 op。

#### 5.3.1 `minimax_indexer`

```python
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

    Performance formula: see section 4.1.
    """
    total_tokens = hidden_states.shape[0]
    return torch.empty(
        (total_tokens, num_indexer_heads, topk_blocks),
        dtype=torch.int32,
        device="meta",
    )
```

返回形状不要求和 kernel 完全一致，只要性能模型能从 op 参数中拿到 $T, N, K$ 即可。若 runtime 的 top-k index 实际按`[N, T, K]` 或 paged layout 排布，docstring 中说明等价即可。

#### 5.3.2 `minimax_sparse_attention`

```python
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

    Performance formula: see section 4.2.
    """
    return torch.empty_like(query).contiguous()
```

同时在`tensor_cast/ops/__init__.py` 中导入：

```python
from . import minimax_m3_sparse_attention  # noqa: F401
```

不建议在虚拟算子层注册`minimax_attention_hybrid` 或 index value 相关 op。M3-preview 的 sparse layer 只需要上面两个边界，dense layer 复用已有 attention / linear / norm / rotary / MoE 算子即可。

### 5.4 性能模型层：注册 op properties

性能模型层负责把两个虚拟 op 映射到 Roofline 需要的 MMA、GP 和 memory bytes。修改位置：

```text
tensor_cast/performance_model/__init__.py
```

建议新增两个拆解函数，并注册到对应 op：

```python
@register_op_properties(torch.ops.tensor_cast.minimax_indexer.default)
def _estimate_minimax_indexer_breakdown(...):
    ...

@register_op_properties(torch.ops.tensor_cast.minimax_sparse_attention.default)
def _estimate_minimax_sparse_attention_breakdown(...):
    ...
```

#### 5.4.1 `minimax_indexer` 拆解

`minimax_indexer` 使用 4.1.6 的完整 fused 口径，不再判断 projection / norm / RoPE 是否已被基础 TensorCast op 捕获。

输入量定义：

```python
T = sum(query_lens)
H = hidden_size
N = num_indexer_heads
D = indexer_head_dim
D_r = indexer_rope_dim
K = topk_blocks
B_s = block_size
B_n = ceil(seq_lens / B_s)
```

计算桶：

```text
MMA = 4 * T * H * N * D
    + 2 * sum_b(Q_b * N * L_b * D)

GP  = 12 * T * N * D
    + 6 * T * N * D_r
    + sum_b(Q_b * N * L_b)
    + c_topk * sum_b(Q_b * N * B_n)
```

访存桶按 4.1.6 的`Bytes_indexer` 汇总。实现时可以先把各阶段拆成命名变量，再求和，方便 profiling 校正：

```python
bytes_projection = 2*T*H*s + 2*H*N*D*s + 2*T*N*D*s
bytes_norm_rope = 4*T*N*D*s + 4*T*N*D*s + 2*N*D*s
bytes_cache_write = 2*T*N*D*s
bytes_score = T*N*D*s + sum_b(Q_b*N*L_b*D*s) + 4*sum_b(Q_b*N*B_n)
bytes_topk = 4*sum_b(Q_b*N*B_n) + 4*T*N*K
bytes_total = bytes_projection + bytes_norm_rope + bytes_cache_write + bytes_score + bytes_topk
```

#### 5.4.2 `minimax_sparse_attention` 拆解

`sparse attention` 使用 4.2 的口径。令每个 request 的 selected token 上界为：

```text
A_b = min(L_b, min(B_n, K + R) * B_s)
```

计算桶：

```text
MMA = 4 * sum_b(Q_b * N_q * A_b * D)
GP  = 6 * sum_b(Q_b * N_q * A_b)
```

访存桶：

```text
Q/O bytes = 2 * s * T * N_q * D
KV bytes  = 2 * s * sum_b(Q_b * A_b * N_kv * D)
topk bytes = 4 * T * N_kv * K
bytes_total = Q/O bytes + KV bytes + topk bytes
```

如果实现暂时使用 decode-only 快速估算，也可以令`A_b = min(L_b, (K + R) * B_s)`；但最终版本应保留 per-request 的 $Q_b, L_b$ 求和。

### 5.5 修改汇总表

| 层级 | 文件 | 主要改动 |
|------|------|----------|
| 模型定义层 | `tensor_cast/transformers/builtin_model/minimax_m3.py` | 注册 MiniMax-M3 profile，读取`text_config.sparse_attention_config`，按 layer id 标记 sparse layer |
| 抽象算子层 | `tensor_cast/layers/minimax_m3_attention.py` | 包装 attention forward，dense layer 复用已有 attention，sparse layer 调用`minimax_indexer` + `minimax_sparse_attention` |
| 虚拟算子层 | `tensor_cast/ops/minimax_m3_sparse_attention.py` | 注册两个 meta-only op：`minimax_indexer`、`minimax_sparse_attention` |
| 虚拟算子导入 | `tensor_cast/ops/__init__.py` | 导入`minimax_m3_sparse_attention` |
| 性能模型层 | `tensor_cast/performance_model/__init__.py` | 注册两个 op properties 拆解函数，实现第 4 章公式 |

## 6. 注册流程完整追溯

```
用户输入: python -m cli.inference.text_generate \
    --model-id minimax/MiniMax-M3 \
    --num-queries 1 --query-length 128 \
    --decode
    │
    ▼
UserInputConfig.from_args(args)
    │  model_id="minimax/MiniMax-M3"
    │
    ▼
build_model(user_input)
    │  tensor_cast/core/model_builder.py:50
    │
    ▼
ConfigResolver.resolve()
    │  1. 从 HuggingFace 加载 config.json
    │  2. model_type = "minimax_m3_vl"
    │  3. 读取 text_config.sparse_attention_config
    │  4. ModelProfile 注册 → attention / MoE / MTP 等静态信息
    │
    ▼
TransformerModel(model_id, config)
    │  tensor_cast/transformers/model.py
    │
    ├── import tensor_cast.transformers.builtin_model
    │   → import builtin_model.minimax_m3  (自动加载)
    │      │  → register_model_profile(ModelProfile("minimax_m3_vl", ...))
    │      │  → @register_custom_model("minimax_m3_vl") 注册自定义函数
    │      ▼
    ├── get_custom_model("minimax_m3_vl") → custom_fn
    │
    ├── custom_fn(self):  ← 自定义 patch 流程
    │   ├── wrap_model(self)                      # CausalLmWrapper
    │   ├── maybe_enable_mtp(self)                # MTP
    │   ├── maybe_reuse_layers(self)              # 层复用
    │   ├── patch_rotary_emb(self)                # RoPE 替换
    │   ├── patch_minimax_m3_attention(self)      # ← 新: 替换 attention
    │   │   ├── 解析 sparse_attention_config
    │   │   ├── 识别 sparse_layer_ids
    │   │   └── 替换 self_attn → MiniMaxM3AttentionWrapper
    │   ├── patch_moe(self)                       # MoE 替换
    │   ├── quantize_model(self)                  # 量化
    │   └── shard_model(self)                     # TP/EP 切分
    │
    ▼
generate_inputs() → 构造输入
    │
    ▼
Runtime.__enter__()
    │  → 激活 TorchDispatchMode
    │
    ▼
model.forward(**inputs)
    │
    ▼
在每层 forward 中:
  MiniMaxM3AttentionWrapper.forward()
    │
    ├── dense layer:
    │     调用已有 tensor_cast.attention / linear / norm / rotary / MoE 算子
    │
    └── sparse layer:
          1. 调用 torch.ops.tensor_cast.minimax_indexer(...)
          2. 调用 torch.ops.tensor_cast.minimax_sparse_attention(...)
    │
    ▼
Runtime.__torch_dispatch__(...)
    │  → 分别为 minimax_indexer / minimax_sparse_attention 注册 OpInvokeInfo
    │
    ▼
Runtime.__exit__()
    │
    ├── repeat_op_invoke_infos()
    └── replay_op_invoke_infos()
        │
        ▼
    AnalyticPerformanceModel.process_op()
        │
        ├── get_op_estimator(op, device)
        │   → 未注册专用 estimator → _estimate_default
        │
        ├── _estimate_default()
        │   │
        │   ├── minimax_indexer.get_perf_properties()
        │   │   │
        │   │   └── _estimate_minimax_indexer_breakdown()
        │   │       │
        │   │       └── 拆解为 4.1.6:
        │   │           ├── index_q_proj_mma + index_k_proj_mma
        │   │           ├── index_norm_gp + index_rope_gp
        │   │           ├── index_qk_mma + block_reduce_gp
        │   │           ├── topk_gp
        │   │           └── Bytes_indexer
        │   │
        │   ├── minimax_sparse_attention.get_perf_properties()
        │   │   │
        │   │   └── _estimate_minimax_sparse_attention_breakdown()
        │   │       │
        │   │       └── 拆解为 4.2:
        │   │           ├── sparse_qk_mma + sparse_pv_mma
        │   │           ├── softmax_gp + mask_gp
        │   │           └── Q/O + selected K/V + topk index bytes
        │   │
        │   └── Roofline: time = max(compute_time, memory_time) + static_cost
        │
        ▼
    RuntimeEvent(op_invoke_info, perf_results)
    self.event_list.append(event)
    │
    ▼
_build_model_timelines() → token 依赖 + stream 并行
    │
    ▼
total_execution_time_s()
table_averages()
get_breakdowns()
export_chrome_trace()
```

---

## 7. 测试验证

### 7.1 最小验证测试

```python
# tests/regression/tensor_cast/test_minimax_m3.py

import torch
from tensor_cast import Runtime, device_profiles
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel

def test_minimax_m3_sparse_attention_ops():
    """Verify sparse attention ops are registered and estimatable."""
    device = device_profiles.TEST_DEVICE
    perf_model = AnalyticPerformanceModel(device)

    with Runtime(perf_model, device) as runtime, torch.no_grad():
        seq_lens = torch.tensor([4096], device="meta")
        query_lens = torch.tensor([1], device="meta")
        block_table = torch.empty(1, 32, device="meta", dtype=torch.long)

        topk_idx = torch.ops.tensor_cast.minimax_indexer(
            torch.empty(1, 6144, device="meta"),
            seq_lens,
            query_lens,
            block_table,
            hidden_size=6144,
            num_indexer_heads=4,
            indexer_head_dim=128,
            indexer_rope_dim=64,
            topk_blocks=16,
            block_size=128,
        )

        out = torch.ops.tensor_cast.minimax_sparse_attention(
            torch.empty(1, 64, 128, device="meta"),
            torch.empty(4096, 4, 128, device="meta"),
            torch.empty(4096, 4, 128, device="meta"),
            topk_idx,
            seq_lens,
            query_lens,
            block_table,
            num_q_heads=64,
            num_kv_heads=4,
            head_dim=128,
            topk_blocks=16,
            block_size=128,
            local_blocks=1,
        )

    # Verify results
    table = runtime.table_averages()
    assert "minimax_indexer" in table
    assert "minimax_sparse_attention" in table

    total_time = runtime.total_execution_time_s()
    print(f"Total execution time: {total_time}")
    print(table)
```

### 7.2 端到端仿真测试

```python
# 终端命令
python -m cli.inference.text_generate \
    --model-id minimax/MiniMax-M3 \
    --num-queries 1 \
    --query-length 128 \
    --decode \
    --tp-size 8 \
    --device ATLAS_800_A3_752T_128G_DIE
```

端到端对比测试用例：

| 硬件 | 并发数 | 输入长度 | 输出长度 | msmodeling 仿真 TTFT | msmodeling 仿真 TPOT | 硬件实测 TTFT | 硬件实测 TPOT | 误差 |
|------|--------|----------|----------|----------------------|----------------------|----------------|----------------|------|
| B200 | 1 | 64 | 1k |  |  |  |  |  |
| B200 | 4 | 64 | 1k |  |  |  |  |  |
| B200 | 32 | 64 | 1k |  |  |  |  |  |
| B200 | 64 | 64 | 1k |  |  |  |  |  |
| A5 | 1 | 64 | 1k |  |  |  |  |  |
| A5 | 4 | 64 | 1k |  |  |  |  |  |
| A5 | 32 | 64 | 1k |  |  |  |  |  |
| A5 | 64 | 64 | 1k |  |  |  |  |  |

对比方法：输入长度固定为 64 tokens，输出长度固定为 1k tokens；分别在 B200 和 A5 上采集硬件实测结果，并与 msmodeling 仿真的 TTFT、TPOT 做误差对比。

---

## 8. 参考

### 8.1 msmodeling

- DSA-Indexer 建模方法论：`notes/msmodeling-DSA-Indexer-性能仿真建模分析.md`
- 现有类似模型适配：`tensor_cast/transformers/builtin_model/minimax_m2.py`
- 自定义 patch 链参考：`tensor_cast/transformers/builtin_model/kimi_k25.py`
- Op 定义示例：`tensor_cast/ops/attention.py`, `tensor_cast/ops/mla.py`
- Op Properties 注册示例：`tensor_cast/performance_model/__init__.py` (`dsa_indexer`, `attention`)
- 性能模型基类：`tensor_cast/performance_model/base.py`

### 8.2 mm-sglang-triton

- MiniMax-M3 模型定义：`python/sglang/srt/models/minimax_m3.py`
- Sparse Attention 后端：`python/sglang/srt/layers/attention/minimax_sparse_backend.py`
- Sparse Attention Triton 算子：`python/sglang/srt/layers/attention/minimax_sparse_ops/`

### 8.3 Minimax-M3-preview

- 模型配置：`config.json`
