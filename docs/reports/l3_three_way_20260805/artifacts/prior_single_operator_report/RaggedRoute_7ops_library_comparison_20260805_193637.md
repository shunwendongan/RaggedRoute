# RaggedRoute 七算子 CUDA Candidate / library_baseline 实时性能对比

> 生成时间：2026-08-05T19:36:37+08:00；源码：`8d4c281eb28bb63e93f9b499343486938f8c3515`；所有数字来自本次实时运行。

## 实验合同与证据边界

- 硬件：NVIDIA GeForce RTX 3080，SM86，68 SM；驱动 591.86；CUDA 13.3.73；NSYS 2026.1.3；NCU 2026.2.1。Release build 含 `-lineinfo`，CUTLASS 通过仓库固定的 v4.6.1 Fetch 方式启用。
- 串行顺序：`dense_gemm, topk_gate, histogram, exclusive_scan, histogram_scan, permute, grouped_gemm, unpermute`；每个 suite 仅在前一 suite 的所有独立进程结束后启动。每个 suite 为 3 个独立 process、20 warmups、30 samples，保留仓库 case-specific kernel repeats/cache policy。
- correctness：CTest 8/8 通过；raw records 5571 条，`validation.ok=false` 为 0 条。Compute Sanitizer memcheck 在 correctness binary 上运行超过 6 分钟无输出，已终止并标记为 timeout，不能据此宣称 sanitizer PASS。
- 正式 speedup 只使用未 profile 的 release p50；NSYS/NCU duration 仅用于机制诊断。`not_collected`/`unsupported_or_unknown` 不替换为 0。

## Release 总览

| 算子/对照 | Candidate | `src/*/library_baseline` 对照 | Cases | Geomean speedup | 最差 speedup | 最大 p95 ratio |
|---|---|---|---:|---:|---:|---:|
| Dense GEMM | `cuda_register_tiled_v3_64x32_async` | `cublaslt` | 6 | 0.973x | 0.838x | 1.197x |
| Top-K Gate（candidate vs naive exact contract） | `cuda_local_pair_two_reduce_top2_v4` | `cuda_naive` | 668 | 1.046x | 0.684x | 1.580x |
| Histogram | `cuda_candidate` | `cub_device_histogram` | 12 | 1.425x | 1.109x | 0.950x |
| Exclusive Scan（naive vs CUB BlockScan） | `cuda_naive` | `cub_block_scan` | 10 | 0.939x | 0.832x | 1.231x |
| Histogram-Scan fused（Scan candidate） | `cuda_fused_histogram_scan` | `cuda_separate_current` | 5 | 1.507x | 1.010x | 1.001x |
| Token Permute | `cuda_candidate_from_ids` | `vllm_moe_permute` | 5 | 0.932x | 0.867x | 1.168x |
| Grouped GEMM | `cuda_grouped_sm86_fp32_v1` | `cutlass_grouped` | 10 | 0.977x | 0.501x | 2.192x |
| Unpermute | `cuda_warp_token_vec4` | `vllm_finalize_routing` | 72 | 1.077x | 0.892x | 1.168x |

说明：表中 speedup = baseline p50 / candidate p50，>1 表示 candidate 更快；最差值和 p95 ratio 用于识别 shape 回退，不能只看几何平均。

## Dense GEMM

- 严格配对 6 组；geomean 0.973x，最差 `dense_gemm.fp32.m1024_n1024_k1024.candidate_v3` 为 0.838x，最佳 `dense_gemm.fp32.m256_n256_k256.candidate_v3` 为 1.206x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `dense_gemm.fp32.m1024_n1024_k1024.candidate_v3` | L2_operator_steady/warm | 152.576 | 182.170 | 0.838x | 184.433 | 0.0073 |
| `dense_gemm.fp32.m1024_n1024_k1024.candidate_v3` | L1_kernel_body/warm | 155.136 | 183.091 | 0.847x | 186.706 | 0.0100 |
| `dense_gemm.fp32.m512_n512_k512.candidate_v3` | L1_kernel_body/warm | 26.112 | 28.365 | 0.921x | 28.570 | 0.0607 |
| `dense_gemm.fp32.m512_n512_k512.candidate_v3` | L2_operator_steady/warm | 27.136 | 26.931 | 1.008x | 28.524 | 0.0532 |
| `dense_gemm.fp32.m256_n256_k256.candidate_v3` | L1_kernel_body/warm | 12.851 | 12.032 | 1.068x | 16.701 | 0.2024 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\dense_gemm.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Top-K Gate（candidate vs naive exact contract）

- 严格配对 668 组；geomean 1.046x，最差 `topk_gate.release.main.e17_t4096` 为 0.684x，最佳 `topk_gate.release.cold_t2048_e64` 为 1.983x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `topk_gate.release.main.e17_t4096` | L2_operator_steady/warm | 7.849 | 11.479 | 0.684x | 12.279 | 0.1074 |
| `topk_gate.release.cub_pair.e17_t4096` | L1_kernel_body/warm | 8.335 | 11.372 | 0.733x | 12.816 | 0.1094 |
| `topk_gate.release.main.e63_t4096` | L2_operator_steady/warm | 9.933 | 13.133 | 0.756x | 15.448 | 0.0875 |
| `topk_gate.release.main.e33_t4096` | L1_kernel_body/warm | 9.738 | 12.872 | 0.757x | 13.699 | 0.0337 |
| `topk_gate.release.main.e48_t4096` | L1_kernel_body/warm | 10.199 | 12.989 | 0.785x | 13.871 | 0.0664 |

- Top‑K 的 v4 与 naive 具有同一项目 oracle，可报告 exact-contract 对照；仓库明确没有严格语义等价的外部 library baseline（tie/NaN/selected-softmax 合同不同），因此下列 library 数字只作诊断参考，不作为正式 library speedup：

  - v4 vs `vllm_row_packed_top2`：80 组，geomean 0.988x，最差 0.873x。
  - v4 vs `cub_block_radix_top2`：198 组，geomean 1.923x，最差 1.238x。
原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\topk_gate.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Histogram

- 严格配对 12 组；geomean 1.425x，最差 `histogram.r65536_e16.uniform.l2` 为 1.109x，最佳 `histogram.r128_e8.uniform.l2` 为 2.266x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `histogram.r65536_e16.uniform.l2` | L2_operator_steady/warm | 17.254 | 15.565 | 1.109x | 19.302 | 0.1064 |
| `histogram.r65536_e64.zipf14.l2` | L2_operator_steady/warm | 17.485 | 15.590 | 1.122x | 19.177 | 0.0923 |
| `histogram.r65536_e64.single_hot.l2` | L2_operator_steady/warm | 17.203 | 15.181 | 1.133x | 17.572 | 0.0833 |
| `histogram.r1048576_e64.single_hot.l2` | L2_operator_steady/warm | 20.275 | 17.715 | 1.145x | 20.741 | 0.1089 |
| `histogram.r1048576_e64.uniform.l2` | L2_operator_steady/warm | 20.634 | 17.715 | 1.165x | 23.255 | 0.1625 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\histogram.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Exclusive Scan（naive vs CUB BlockScan）

- 严格配对 10 组；geomean 0.939x，最差 `exclusive_scan.e64.l1` 为 0.832x，最佳 `exclusive_scan.e1.l2` 为 1.015x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `exclusive_scan.e64.l1` | L1_kernel_body/warm | 8.269 | 9.943 | 0.832x | 10.983 | 0.1073 |
| `exclusive_scan.e31.l2` | L2_operator_steady/warm | 8.177 | 8.832 | 0.926x | 9.360 | 0.0449 |
| `exclusive_scan.e64.l2` | L2_operator_steady/warm | 8.095 | 8.740 | 0.926x | 11.030 | 0.1114 |
| `exclusive_scan.e31.l1` | L1_kernel_body/warm | 8.212 | 8.806 | 0.933x | 9.322 | 0.0845 |
| `exclusive_scan.e32.l1` | L1_kernel_body/warm | 8.228 | 8.791 | 0.936x | 9.514 | 0.0589 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\exclusive_scan.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Histogram-Scan fused（Scan candidate）

- 严格配对 5 组；geomean 1.507x，最差 `histogram_scan.r4097_e64_single_hot` 为 1.010x，最佳 `histogram_scan.r1024_e64_zipf` 为 2.004x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `histogram_scan.r4097_e64_single_hot` | L2_operator_steady/warm | 25.247 | 24.991 | 1.010x | 26.877 | 0.0690 |
| `histogram_scan.r8192_e16_uniform` | L2_operator_steady/warm | 24.684 | 23.900 | 1.033x | 26.365 | 0.0563 |
| `histogram_scan.r4096_e64_uniform` | L2_operator_steady/warm | 19.548 | 10.409 | 1.878x | 10.844 | 0.0526 |
| `histogram_scan.r0_e1_uniform` | L2_operator_steady/warm | 16.328 | 8.238 | 1.982x | 8.915 | 0.0611 |
| `histogram_scan.r1024_e64_zipf` | L2_operator_steady/warm | 16.077 | 8.023 | 2.004x | 9.531 | 0.0930 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\histogram_scan.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Token Permute

- 严格配对 5 组；geomean 0.932x，最差 `permute.full_from_ids.tail` 为 0.867x，最佳 `permute.full_from_ids.large.uniform` 为 1.070x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `permute.full_from_ids.tail` | L2_operator_steady/warm | 26.112 | 30.106 | 0.867x | 37.796 | 0.1640 |
| `permute.full_from_ids.anchor.uniform` | L2_operator_steady/warm | 29.798 | 33.894 | 0.879x | 41.165 | 0.1395 |
| `permute.full_from_ids.anchor.zipf14` | L2_operator_steady/warm | 31.642 | 35.430 | 0.893x | 40.571 | 0.1416 |
| `permute.full_from_ids.large.single_hot` | L2_operator_steady/warm | 40.038 | 41.574 | 0.963x | 53.453 | 0.1735 |
| `permute.full_from_ids.large.uniform` | L2_operator_steady/warm | 40.858 | 38.195 | 1.070x | 53.156 | 0.1865 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\permute.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Grouped GEMM

- 严格配对 10 组；geomean 0.977x，最差 `grouped_gemm.uniform.t2048_e64_k128_n128` 为 0.501x，最佳 `grouped_gemm.single_expert.t512_e64_k128_n256` 为 1.500x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `grouped_gemm.uniform.t2048_e64_k128_n128` | L2_operator_steady/warm | 26.317 | 52.531 | 0.501x | 58.368 | 0.0609 |
| `grouped_gemm.zipf14.t2048_e64_k128_n128` | L2_operator_steady/warm | 44.237 | 62.362 | 0.709x | 67.400 | 0.0511 |
| `grouped_gemm.non_aligned.t512_e64_k127_n129` | L2_operator_steady/warm | 42.394 | 50.381 | 0.841x | 50.790 | 0.0074 |
| `grouped_gemm.uniform.t512_e64_k128_n128` | L2_operator_steady/warm | 24.371 | 27.034 | 0.902x | 28.375 | 0.0231 |
| `grouped_gemm.uniform.t512_e16_k128_n256` | L2_operator_steady/warm | 22.938 | 25.395 | 0.903x | 27.556 | 0.0352 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\grouped_gemm.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## Unpermute

- 严格配对 72 组；geomean 1.077x，最差 `unpermute.uniform.t64_n256.warm` 为 0.892x，最佳 `unpermute.zipf.t4096_n64.warm` 为 1.807x。

| Case | Level/cache | Baseline p50 (us) | Candidate p50 (us) | Speedup | Candidate p95 (us) | CV |
|---|---|---:|---:|---:|---:|---:|
| `unpermute.uniform.t64_n256.warm` | L2_operator_steady/warm | 7.895 | 8.847 | 0.892x | 10.026 | 0.0932 |
| `unpermute.zipf.t64_n1024.warm` | L2_operator_steady/warm | 8.038 | 8.868 | 0.906x | 10.087 | 0.0930 |
| `unpermute.zipf.t64_n64.warm` | L2_operator_steady/warm | 7.864 | 8.509 | 0.924x | 9.903 | 0.0875 |
| `unpermute.uniform.t512_n128.warm` | L1_kernel_body/warm | 8.253 | 8.909 | 0.926x | 10.119 | 0.0761 |
| `unpermute.uniform.t64_n64.warm` | L1_kernel_body/warm | 8.090 | 8.714 | 0.928x | 9.933 | 0.0905 |

原始/聚合/比较工件：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results\unpermute.jsonl`；聚合目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。

## NSYS / NCU 诊断（最大绝对 p50 差距 case）

选择规则：每个对照优先选择 L2 case 中 candidate/baseline p50 绝对差距最大的有效 case；Top‑K 额外保留 exact、vLLM diagnostic、CUB diagnostic。

### `dense_gemm`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_register_tiled_v3_64x32_async` | `unnamed>::dense_gemm_register_tiled_v3_64x32_async_kernel(const float *, const float *, float *, int, int)` | 3734385 | 34.623 | 69.090 | 75.037 | 14.500 | 1.510 | 85 | 15.360 |
| baseline | `cublaslt` | `ampere_sgemm_64x64_nn` | 3045376 | 24.105 | 60.673 | 52.338 | 18.699 | 0.940 | 126 | 9.472 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\dense_gemm.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\dense_gemm.candidate`。

### `topk_exact`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_local_pair_two_reduce_top2_v4` | `void unnamed>::topk_gate_local_pair_two_reduce_kernel<64, 4>(const float *, int *, float *, int)` | 55966 | 26.922 | 7.269 | 7.877 | 7.877 | 0.310 | 18 | 1.024 |
| baseline | `cuda_naive` | `unnamed>::topk_gate_naive_kernel(const float *, int *, float *, int, int)` | 235739 | 16.031 | 1.387 | 5.932 | 5.932 | 0.020 | 26 | 1.024 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\topk_exact.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\topk_exact.candidate`。

### `topk_vllm_diag`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_local_pair_two_reduce_top2_v4` | `void unnamed>::topk_gate_local_pair_two_reduce_kernel<8, 4>(const float *, int *, float *, int)` | 41727 | 6.287 | 0.022 | 0.289 | 0.115 | 0.000 | 18 | 1.024 |
| baseline | `vllm_row_packed_top2` | `void unnamed>::vllm_row_packed_top2_kernel<8>(const float *, int *, float *, int)` | 37567 | 6.875 | 0.022 | 0.261 | 0.101 | 0.000 | 21 | 1.024 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\topk_vllm_diag.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\topk_vllm_diag.candidate`。

### `topk_cub_diag`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_local_pair_two_reduce_top2_v4` | `unnamed>::topk_gate_two_expert_vector_kernel(const float *, int *, float *, int)` | 30207 | 16.229 | 0.191 | 0.809 | 0.586 | 0.040 | 18 | 1.024 |
| baseline | `cub_block_radix_top2` | `unnamed>::cub_block_radix_top2_kernel(const float *, int *, float *, int, int)` | 1419173 | 30.994 | 76.289 | 76.289 | 0.079 | 3.760 | 32 | 2.208 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\topk_cub_diag.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\topk_cub_diag.candidate`。

### `histogram`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_candidate` | `unnamed>::histogram_single_cta_shared_kernel(const int *, int *, int, int)` | 33568 | 14.966 | 0.008 | 0.236 | 0.094 | 0.000 | 28 | 1.280 |
| baseline | `cub_device_histogram` | `void DeviceHistogramSweepKernel<policy_hub<int, int, 1, 1, 1>::Policy1000, 256, 1, 1, const int *, int, Transforms<int, int, int>::ScaleTransform, Transforms<int, int, int>::PassThruTransform, int>(T5, array<int, T4>, array<int, T4>, array<T6 *, T4>, array<T6 *, T4>, array<T8, T4>, array<T7, T4>, T9, T9, T9, int, GridQueue<int>)` | 104094 | 24.401 | 0.108 | 2.265 | 1.069 | 0.000 | 54 | 2.060 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\histogram.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\histogram.candidate`。

### `exclusive_scan`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_naive` | `unnamed>::exclusive_scan_naive_kernel(const int *, int *, int)` | 47264 | 2.083 | 0.011 | 0.255 | 0.089 | 0.000 | 20 | 1.024 |
| baseline | `cub_block_scan` | `void unnamed>::block_scan_kernel<128>(const int *, int *, int)` | 31615 | 8.175 | 0.006 | 0.219 | 0.072 | 0.000 | 16 | 1.696 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\exclusive_scan.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\exclusive_scan.candidate`。

### `histogram_scan`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_fused_histogram_scan` | `unnamed>::histogram_exclusive_scan_fused_subwarp_kernel(const int *, int *, int *, int, int)` | 92608 | 16.220 | 0.054 | 0.339 | 0.339 | 0.000 | 22 | 1.280 |
| baseline | `cuda_separate_current` | `unnamed>::histogram_single_cta_shared_kernel(const int *, int *, int, int)` | 90399 | 16.380 | 0.032 | 0.347 | 0.342 | 0.000 | 28 | 1.280 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\histogram_scan.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\histogram_scan.candidate`。

### `permute`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_candidate_from_ids` | `unnamed>::exclusive_scan_naive_kernel(const int *, int *, int)` | 65470 | 2.083 | 0.024 | 0.253 | 0.087 | 0.000 | 20 | 1.024 |
| baseline | `vllm_moe_permute` | `void DeviceRadixSortSingleTileKernel<policy_selector_from_types<int, int, unsigned int>, 0, int, int, unsigned int, identity_decomposer_t>(const T3 *, T3 *, const T4 *, T4 *, T5, int, int, T6)` | 153983 | 16.589 | 0.952 | 0.952 | 0.379 | 0.010 | 128 | 34.880 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\permute.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\permute.candidate`。

### `grouped_gemm`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_grouped_sm86_fp32_v1` | `unnamed>::grouped_gemm_register16x32_async_v3_kernel(const float *, const float *, const int *, float *, int, int, int)` | 977324 | 29.790 | 46.464 | 48.585 | 24.605 | 1.000 | 106 | 7.952 |
| baseline | `cutlass_grouped` | `void Kernel<GemmGrouped<MmaPipelined<GemmShape<128, 128, 8>, PredicatedTileIterator<MatrixShape<128, 8>, float, RowMajor, 1, PitchLinearStripminedThreadMap<PitchLinearShape<8, 128>, 256, 1>, 1, 0, NoPermute>, RegularTileIterator<MatrixShape<128, 8>, float, ColumnMajor, 1, TransposePitchLinearThreadMapSimt<PitchLinearStripminedThreadMap<PitchLinearShape<8, 128>, 256, 1>>, 4>, PredicatedTileIterator<MatrixShape<8, 128>, float, RowMajor, 0, PitchLinearStripminedThreadMap<PitchLinearShape<128, 8>, 256, 1>, 1, 0, NoPermute>, RegularTileIterator<MatrixShape<8, 128>, float, RowMajor, 0, PitchLinearStripminedThreadMap<PitchLinearShape<128, 8>, 256, 1>, 4>, float, RowMajor, MmaPolicy<MmaSimt<GemmShape<32, 64, 8>, float, ColumnMajor, float, RowMajor, float, RowMajor, MmaSimtPolicy<MatrixShape<4, 8>, RowMajorInterleaved<2>, GemmShape<4, 4, 1>>, 1, 0, 0, bool>, MatrixShape<4, 0>, MatrixShape<0, 0>, 1>, NumericArrayConverter<float, float, 4, 2, Identity>, NumericArrayConverter<float, float, 4, 2, Identity>, bool>, Epilogue<GemmShape<128, 128, 8>, MmaSimt<GemmShape<32, 64, 8>, float, ColumnMajor, float, RowMajor, float, RowMajor, MmaSimtPolicy<MatrixShape<4, 8>, RowMajorInterleaved<2>, GemmShape<4, 4, 1>>, 1, 0, 0, bool>, 1, PredicatedTileIterator<OutputTileOptimalThreadMap<OutputTileShape<128, 1, 4, 4, 1>, OutputTileShape<1, 4, 2, 1, 8>, 256, 1, 32>, float, 0, NoPermute, 0>, FragmentIteratorSimt<GemmShape<32, 64, 8>, Mma<GemmShape<8, 8, 1>, float, ColumnMajor, float, RowMajor, float, RowMajor, OpMultiplyAdd, bool>, RowMajor, MmaSimtPolicy<MatrixShape<4, 8>, RowMajorInterleaved<2>, GemmShape<4, 4, 1>>>, TileIteratorSimt<GemmShape<32, 64, 8>, Mma<GemmShape<8, 8, 1>, float, ColumnMajor, float, RowMajor, float, RowMajor, OpMultiplyAdd, bool>, float, RowMajor, MmaSimtPolicy<MatrixShape<4, 8>, RowMajorInterleaved<2>, GemmShape<4, 4, 1>>>, SharedLoadIterator<OutputTileOptimalThreadMap<OutputTileShape<128, 1, 4, 4, 1>, OutputTileShape<1, 4, 2, 1, 8>, 256, 1, 32>::CompactedThreadMap, float, 4>, LinearCombination<float, 1, float, float, 0, 2, float>, MatrixShape<0, 17>, 1, 1>, GemmBatchedIdentityThreadblockSwizzle, 0, 0>>(Params)` | 480981 | 16.651 | 54.200 | 45.276 | 45.276 | 0.940 | 144 | 17.680 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\grouped_gemm.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\grouped_gemm.candidate`。

### `unpermute`

| Role | Variant | Kernel (NSYS hotspot) | GPU time (ns) | Occupancy % | SM % | Mem % | DRAM % | Waves/SM | Reg/thread | Shared/block KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | `cuda_warp_token_vec4` | `unnamed>::unpermute_warp_token_vec4_kernel(const float *, const int *, const float *, float *, int, int)` | 64256 | 65.573 | 13.948 | 54.290 | 54.290 | 1.250 | 34 | 1.024 |
| baseline | `vllm_finalize_routing` | `void unnamed>::finalize_routing<1>(const float *, float *, const float *, const int *, int, int, int)` | 214332 | 21.908 | 12.492 | 25.413 | 25.413 | 10.040 | 40 | 1.024 |

NSYS/NCU 工件目录：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys\unpermute.candidate` 与 `C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu\unpermute.candidate`。

## 机制结论与限制

- Dense GEMM candidate 的 NCU 显示较高寄存器数与约 34.6% achieved occupancy，且 grid tail 由 NSYS/NCU 可见；其 release 结果仍应以 shape 逐项看，不能由 profiler duration 推断总体胜负。
- Histogram candidate 在小/中等 routing shape 使用 shared single-CTA 路径；强库 CUB 在部分大 shape 通过多 kernel/初始化完成工作，release 差距应结合 p95 与 workspace 解读。
- Grouped GEMM candidate 与 CUTLASS 均为 strict FP32；candidate 的显式 SM86 调度需同时关注 registers、waves/SM 和 expert-tail imbalance。
- Permute 的 NSYS 热点可能是完整 from-ids 路径中的 histogram/scan 组件，而非最后 payload copy；这正是 L2 对照计入 mapping preparation 的结果。
- Scan section 同时报告原始 Exclusive Scan（naive vs CUB BlockScan）与远端新增融合 Histogram‑Scan candidate；融合 candidate 不是单独的 exclusive_scan primitive。
- 背景 GPU 查询中 `compute_applications.*` 字段在本机 nvidia-smi 不支持，报告保留该限制；逐 suite 的显存/温度/功耗快照仍写入 `serial_environment.json`。

## Artifact 索引

- 串行环境与顺序：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\serial_environment.json`；命令日志：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\serial_run.log`。
- Release raw/aggregate/comparison：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\results`、`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\aggregates`。
- NSYS：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\nsys`；NCU basic：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu`。
- NCU collection status：`C:\Users\Administrator\Documents\Playground\RaggedRoute-eval-8d4c281\out\evaluation\profiles\ncu_jobs_status.json`；失败数 0。
