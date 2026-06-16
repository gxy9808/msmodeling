# Minimax-M3 DECODE bs=16: Profiling vs msmodeling Simulation Comparison

## Test Configuration

| Item | Value |
|------|-------|
| Model | Minimax-M3-MXFP8 |
| Hardware | NVIDIA B200 (8x) |
| Scenario | DECODE, bs=16 |
| Parallelism | TP=8, EP=8 |
| Quantization | FP8 (MXFP8 weights, dynamic activation) |

---

## 1. Overall Latency Comparison

| Source | Step Latency | TPS/Device |
|--------|-------------|------------|
| GPU Profiling (real) | 46.07 ms | ~347 token/s |
| msmodeling Simulation (analytic) | 22.38 ms | 89.38 token/s |
| Delta | Sim 51.4% lower | - |

Simulation total latency is ~48.6% of real, indicating the analytic model is overly optimistic on communication, kernel launch overhead and GPU utilization.

---

## 2. Operator-Level Comparison

### 2.1 Core Compute Operators

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio(Sim/Real) |
|----------|-----------------|-----------|-------|-------------|---------|-------|-----------------|
| Expert GEMM (6144->384) | deep_gemm sm100_fp8 | 1.847ms | 57 | grouped_matmul_fp8 | 8.416ms | 114 | 4.56x(combined) |
| Expert GEMM (6144->6144->384) | deep_gemm sm100_fp8 | 2.471ms | 57 | (same as above) | - | - | - |
| Shared Expert GEMM | mxfp8_block_scaled_matmul | 4.315ms | 240 | fp8_linear | 1.230ms | 132 | 0.28x |
| Attn QKV/O Proj | cutlass sgemm | 0.959ms | 57 | fp8_linear | 0.611ms | 57 | 0.64x |
| Attn Decode Score | decode_score_kernel | 5.738ms | 57 | minimax_sparse_attention | 0.289ms | 57 | **0.05x** |
| GQA Sparse Decode | gqa_share_sparse_decode | 0.718ms | 57 | (in sparse_attention) | - | - | - |
| KV Cache Store | store_kvcache | 0.113ms | 60 | reshape_and_cache | 0.006ms | 3 | **0.05x** |

### 2.2 Communication Operators

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio |
|----------|-----------------|-----------|-------|-------------|---------|-------|-------|
| AllReduce | all_reduce_one_shot_push | 15.070ms | 121 | all_reduce | 1.334ms | 120 | **0.09x** |
| AllToAll | (NCCL) | - | - | all_to_all | 1.015ms | 114 | ~1.0x |
| AllGather | ncclDevKernel_AllGather | 0.049ms | 1 | all_gather | 0.484ms | 58 | 9.9x |

### 2.3 Normalization Operators

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio |
|----------|-----------------|-----------|-------|-------------|---------|-------|-------|
| FusedAddRMSNorm | FusedAddRMSNormKernel | 15.021ms | 120 | add+rsqrt+mul | ~1.16ms | ~554 | **0.08x** |
| RMSNorm | RMSNormKernel | 0.438ms | 234 | pow+mean+rsqrt+mul | ~1.07ms | ~127 | 2.44x |

### 2.4 MoE Routing and TopK

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio |
|----------|-----------------|-----------|-------|-------------|---------|-------|-------|
| TopK + Gating | topk_index_partial+topkGatingSigmoid | 2.093ms | 114 | topk+sigmoid+gather | 0.342ms | 171 | 0.16x |
| TopK Merge | topk_index_merge_kernel | 0.147ms | 57 | (not modeled) | - | - | - |
| Post Reorder | post_reorder_triton_kernel | 1.588ms | 57 | init_routing_v2 | 0.115ms | 57 | **0.07x** |
| Merge Attn Out | merge_topk_attn_out_kernel | 0.135ms | 57 | unpermute_tokens | 0.116ms | 57 | 0.86x |

### 2.5 Quantization Operators

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio |
|----------|-----------------|-----------|-------|-------------|---------|-------|-------|
| MXFP8 Quant | mxfp8_block_scaled_matmul | 4.315ms | 240 | dynamic_quantize_symmetric | 2.105ms | 1956 | 0.49x |
| FP8 Downcast | downcast_to_mxfp | 0.375ms | 240 | (in quantize) | - | - | - |
| SiLU+Mul+Quant | silu_and_mul_post_quant | 0.462ms | 57 | m3_swiglu_quant | 0.124ms | 57 | 0.27x |
| Fill Gateup Input | fill_gateup_input_triton | 0.330ms | 57 | (in cat) | - | - | - |

### 2.6 Copy and Index Operations

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio |
|----------|-----------------|-----------|-------|-------------|---------|-------|-------|
| Elementwise Copy | direct_copy_kernel_cuda | 1.708ms | 480 | _to_copy | 1.019ms | 498 | 0.60x |
| Cat Array | CatArrayBatchedCopy | 0.759ms | 480 | cat | 1.207ms | 1039 | 1.59x |
| Index Put | index_elementwise_kernel | 0.442ms | 57 | (in gather etc.) | - | - | - |
| Memset/Fill | vectorized_elementwise_kernel(Fill) | 0.618ms | 480 | (not tracked) | - | - | - |
| KV Index | create_flashinfer_kv_indices | 0.166ms | 1 | minimax_indexer | 0.401ms | 57 | 2.42x |

### 2.7 Other Operators

| Function | Profiling Kernel | Prof. Time | Calls | Sim Operator | Sim Time | Calls | Ratio |
|----------|-----------------|-----------|-------|-------------|---------|-------|-------|
| RoPE | fused_rope_kernel | 0.237ms | 117 | cos+sin+mul | 0.004ms | 2 | **0.02x** |
| Sort | radixSortKVInPlace | 0.291ms | 57 | (not modeled) | - | - | - |
| Transpose+Pack | transpose_and_pack_fp32 | 0.198ms | 57 | (in quantize) | - | - | - |
| Memcpy HtoD/DtoH | Memcpy HtoD/DtoH | 0.009ms | 2 | (not modeled) | - | - | - |

---

## 3. Call Count Comparison

| Operator Category | Profiling Calls | Sim Calls | Notes |
|------------------|----------------|-----------|-------|
| Expert GEMM (FP8) | 57x2 shapes | 114 | Sim combines 2 shapes |
| AllReduce | 121 | 120 | Nearly same (1 diff, possibly MTP) |
| AllToAll | 114 | 114 | Exact match |
| AllGather | 1 (NCCL merged) | 58 | Profiling merged multiple calls |
| RMSNorm/FusedAddRMSNorm | 120+234=354 | 127+121=248 | Sim splits into basic ops |
| Dynamic Quantize | 240 (in matmul) | 1956 | Sim counts each quantize call |
| TopK | 57 | 57 | Match |
| Attention | 57(decode)+3(prefill) | 57(sparse)+3(attn) | Nearly match |
| SiLU+Mul+Quant | 57 | 57 | Match |
| KV Cache Store | 60 | 3 | Sim only models attn layers |
| RoPE | 117 | 2 | Sim splits fused_rope into cos/sin, only 1 call |

---

## 4. Key Discrepancy Analysis

### 4.1 Significantly Underestimated by Sim (Sim/Real < 0.3)

| Operator | Ratio | Root Cause |
|----------|-------|-----------|
| Attention Decode Score | 0.05x | Sim uses analytic roofline, ignoring sparse attention actual memory access pattern and block sparse overhead; real kernel is highly optimized specialized kernel |
| FusedAddRMSNorm | 0.08x | Sim decomposes fused kernel into add+rsqrt+mul basic ops, each estimated very low by roofline; real fused kernel has significant kernel launch overhead and memory bandwidth contention |
| AllReduce | 0.09x | Sim analytic model severely underestimates communication, ignoring NCCL protocol overhead, sync wait and actual bandwidth utilization |
| Post Reorder | 0.07x | Sim init_routing_v2 uses simple metadata operation modeling; real post_reorder_triton_kernel involves complex token permutation and data movement |
| RoPE | 0.02x | Sim splits fused_rope into cos+sin+mul, only counts 1 call; real fused_rope_kernel called 117 times (per layer + MTP) |
| SiLU+Mul+Quant | 0.27x | Sim does not consider post-fusion extra memory overhead and quantization precision handling |
| KV Cache Store | 0.05x | Sim only models 3 calls (attn layers), real requires per-layer store, total 60 calls |

### 4.2 Significantly Overestimated by Sim (Sim/Real > 2.0)

| Operator | Ratio | Root Cause |
|----------|-------|-----------|
| AllGather | 9.9x | Profiling NCCL merges multiple AllGather into 1 call; Sim counts 58 individual calls |
| Expert GEMM (grouped) | 4.56x(combined) | Sim combines 2 shapes into grouped_matmul, total time higher; real uses separate deep_gemm with better batching |
| KV Indexer | 2.42x | Sim per-call (7.0us) vs Profiling (166us/1call), call count differs significantly (57 vs 1) |
| RMSNorm (non-fused) | 2.44x | Sim splits into basic ops, total time actually higher; roofline estimation imprecise for small tensors |

### 4.3 Reasonably Close (0.5x ~ 2.0x)

| Operator | Ratio | Notes |
|----------|-------|-------|
| Attn QKV/O Proj | 0.64x | Reasonable, slight underestimate |
| AllToAll | ~1.0x | Good match |
| Elementwise Copy | 0.60x | Reasonable |
| Merge Attn Out | 0.86x | Good match |
| Dynamic Quantize | 0.49x | Low but order of magnitude correct |
| Cat | 1.59x | Sim has more calls |

---

## 5. Latency Distribution Comparison (Top-5 Bottlenecks)

### Profiling Real Top-5

| Rank | Kernel | Time | Share |
|------|--------|------|-------|
| 1 | AllReduce (push) | 15.07ms | 32.7% |
| 2 | FusedAddRMSNorm | 15.02ms | 32.6% |
| 3 | Decode Score (Sparse Attention) | 5.74ms | 12.5% |
| 4 | MXFP8 Block Scaled Matmul (Shared Expert) | 4.32ms | 9.4% |
| 5 | Expert GEMM (6144->6144->384) | 2.47ms | 5.4% |

### msmodeling Simulation Top-5

| Rank | Operator | Time | Share |
|------|----------|------|-------|
| 1 | grouped_matmul_fp8 (Expert GEMM) | 8.42ms | 37.6% |
| 2 | dynamic_quantize_symmetric | 2.11ms | 9.4% |
| 3 | all_reduce | 1.33ms | 6.0% |
| 4 | fp8_linear | 1.23ms | 5.5% |
| 5 | cat | 1.21ms | 5.4% |

**Key Finding**: Bottleneck ranking is completely different. Real bottlenecks are AllReduce(32.7%) and FusedAddRMSNorm(32.6%), together 65.3%; Simulation shows these as only 11.1%. Simulation identifies Expert GEMM(37.6%) as bottleneck, but real shows it at only ~9.4%.

---

## 6. Summary and Improvement Suggestions

### 6.1 Core Issues

1. **Communication model severely insufficient**: AllReduce sim 1.33ms vs real 15.07ms (0.09x), largest error source. Analytic model does not model NCCL protocol overhead, sync wait latency and actual bandwidth utilization.
2. **Fused kernel decomposition causes underestimation**: FusedAddRMSNorm decomposed into basic ops, roofline total far below real fused kernel time, not considering kernel launch overhead and memory bandwidth contention.
3. **Sparse Attention modeling insufficient**: minimax_sparse_attention simple roofline (0.289ms) vs real decode_score (5.738ms), does not reflect block sparse attention actual memory access pattern.
4. **KV Cache operations missing**: reshape_and_cache only 3 calls vs real 60 calls, incomplete per-layer KV store modeling.
5. **RoPE call count error**: Sim counts only 2 calls (cos+sin) vs real 117 fused_rope calls, not per-layer.

### 6.2 Improvement Suggestions

| Priority | Improvement | Expected Benefit |
|----------|------------|-----------------|
| P0 | Introduce profiling-based communication model (NCCL real bandwidth/latency) | Eliminate AllReduce 0.09x error |
| P0 | Create dedicated TC operators for fused kernels (FusedAddRMSNorm, fused_rope) | Eliminate decomposition-induced underestimation |
| P1 | Introduce profiling-based perf model for minimax_sparse_attention | Fix 0.05x severe underestimation |
| P1 | Complete KV Cache Store per-layer call modeling | Fix call count 3 vs 60 |
| P2 | Create dedicated operator for MoE Post Reorder (token permutation) | Fix 0.07x underestimation |
| P2 | Model grouped_matmul_fp8 by shape separately | Avoid combined statistics overestimation |
