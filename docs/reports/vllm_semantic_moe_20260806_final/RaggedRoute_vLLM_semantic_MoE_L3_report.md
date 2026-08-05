# RaggedRoute CUDA Candidate vs vLLM 生产语义 MoE L3 对照实验

> 生成时间：2026-08-05T16:55:19+00:00。发布性能来自未 profile 的 CUDA Event release；NSYS/NCU 只用于机制诊断，绝不作为速度比来源。

## 结论

在固定的 strict FP32 / IEEE、Top-2、E≤64 和相同四个真实 MoE 代理场景下，RaggedRoute CUDA candidate 的跨场景几何平均速度为 vLLM-semantic adapted chain 的 **1.755×**。这是 production-reference comparison：两端的数值合同与 chain 输入一致，但运行时/编译器栈、vLLM 的 alignment padding 与 fused layout 仍是生产实现的一部分，不能误写为 binary-identical vLLM 或自动推广到完整 vLLM serving。

## 实验合同与来源

| 项目 | 固定值 |
|---|---|
| RaggedRoute candidate | `cuda_all_candidates_chain`：Dense router GEMM → Top-K → fused Histogram/Scan → Permute → Grouped GEMM → Unpermute |
| vLLM semantic | router GEMM → `vllm_topk_softmax` → `moe_align_block_size` / sort metadata → vLLM Triton `fused_moe_kernel` → weighted reduction |
| vLLM upstream | `https://github.com/vllm-project/vllm` @ `66b3c0e61f1e477820212201adf1ed871df7ee98`，Apache-2.0；[source manifest](artifacts/SOURCE_MANIFEST.json) |
| GPU | RTX 3080, sm_86, 68 SM, 10 GiB |
| math | FP32 strict IEEE；vLLM container 设置 `TRITON_F32_DEFAULT=ieee` |
| release | 20 warmups, 30 samples, 3 kernel repeats/sample, 3 independent processes, fixed seed `20260805` |
| timed boundary | 输入已在 device；预分配 workspace；计入 routing、alignment/sort metadata、Grouped GEMM、weighted reduction；不计 allocation/JIT/autotune/CPU oracle |

vLLM adapter 将 RaggedRoute `[E,K,N]` expert 权重转为 vLLM `[E,N,K]`，且计入 vLLM device-side alignment padding 与 metadata 工作。vLLM 把 permutation/Grouped GEMM 组织为 fused kernel，故七阶段是语义映射而非逐 kernel 一一对应。所有阶段输出均在 each-process timed run 前通过独立 oracle：Top-K IDs/weights、alignment 元数据、Grouped GEMM、最终加权输出。

## 发布性能（唯一速度结论来源）

| 场景 | 线路 | p50 µs | p95 µs | CV | tokens/s | routes/s | Effective GB/s | TFLOP/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mixtral 原尺寸 decode (`T=8, E=8, K=4096, N=14336`) | CUDA candidate | 3634.859 | 3649.912 | 0.0034 | 2,201 | 4,402 | 453.25 | 0.517 |
| Mixtral 原尺寸 decode (`T=8, E=8, K=4096, N=14336`) | vLLM semantic | 7634.944 | 7754.172 | 0.0068 | 1,048 | 2,096 | 246.36 | 0.246 |
| ↳ CUDA candidate speedup vs vLLM | — | **2.100×** | — | — | — | — | — | — |
| 连续 decode 代理 (`T=128, E=8, K=1024, N=3584`) | CUDA candidate | 569.173 | 611.362 | 0.0341 | 224,888 | 449,775 | 228.98 | 3.307 |
| 连续 decode 代理 (`T=128, E=8, K=1024, N=3584`) | vLLM semantic | 843.776 | 914.313 | 0.0474 | 151,699 | 303,398 | 147.62 | 2.231 |
| ↳ CUDA candidate speedup vs vLLM | — | **1.482×** | — | — | — | — | — | — |
| Chunked prefill 代理 (`T=512, E=8, K=512, N=1792`) | CUDA candidate | 444.587 | 450.594 | 0.0188 | 1,151,631 | 2,303,263 | 123.95 | 4.242 |
| Chunked prefill 代理 (`T=512, E=8, K=512, N=1792`) | vLLM semantic | 805.376 | 823.057 | 0.0265 | 635,728 | 1,271,456 | 54.07 | 2.342 |
| ↳ CUDA candidate speedup vs vLLM | — | **1.812×** | — | — | — | — | — | — |
| E64 细粒度专家压力 (`T=1024, E=64, K=256, N=512`) | CUDA candidate | 216.405 | 229.222 | 0.0216 | 4,731,861 | 9,463,722 | 240.23 | 2.643 |
| E64 细粒度专家压力 (`T=1024, E=64, K=256, N=512`) | vLLM semantic | 364.203 | 458.069 | 0.0838 | 2,811,621 | 5,623,243 | 118.94 | 1.571 |
| ↳ CUDA candidate speedup vs vLLM | — | **1.683×** | — | — | — | — | — | — |

所有 24 条 record 均 `validation.ok=true`；每 case/线路都恰好含 process runs `{1,2,3}`。原始结果：[CUDA JSONL](artifacts/cuda_candidate_release.jsonl)、[vLLM JSONL](artifacts/release.jsonl)、[CUDA manifest](artifacts/cuda_candidate_release.jsonl.manifest.json)、[vLLM manifest](artifacts/release.jsonl.manifest.json)。

## NSYS 机制证据

### mixtral_decode

- CUDA hotspot: `raggedroute::ops::<unnamed>::grouped_gemm_register16x32_async_v3_kernel(const float *, const float *, const int *, float *, int, int, int)` — 94.9% GPU time.
- vLLM hotspot: `fused_moe_kernel` — 99.2% GPU time.
- [CUDA lane](analysis/mixtral_decode_cuda_timeline.svg) · [vLLM lane](analysis/mixtral_decode_vllm_timeline.svg)

### continuous_decode

- CUDA hotspot: `raggedroute::ops::<unnamed>::grouped_gemm_register16x32_async_v3_kernel(const float *, const float *, const int *, float *, int, int, int)` — 89.0% GPU time.
- vLLM hotspot: `fused_moe_kernel` — 95.2% GPU time.
- [CUDA lane](analysis/continuous_decode_cuda_timeline.svg) · [vLLM lane](analysis/continuous_decode_vllm_timeline.svg)

### chunked_prefill

- CUDA hotspot: `raggedroute::ops::<unnamed>::grouped_gemm_register16x32_async_v3_kernel(const float *, const float *, const int *, float *, int, int, int)` — 89.3% GPU time.
- vLLM hotspot: `fused_moe_kernel` — 95.0% GPU time.
- [CUDA lane](analysis/chunked_prefill_cuda_timeline.svg) · [vLLM lane](analysis/chunked_prefill_vllm_timeline.svg)


NSYS 显示两条线路均由 Grouped GEMM 主导；vLLM 的生产路径还显式可见 `topkGating`、`moe_align_block_size` / `count_and_sort_expert_tokens`、以及 weighted reduction。vLLM alignment padding 和 sorted-token metadata 未被移出 timing boundary。CUDA candidate 以 fused Histogram/Scan 和 token-owned permute 避免额外生产 metadata kernels，但其 Grouped GEMM 仍是主要优化目标。

## NCU basic：Grouped GEMM（诊断）

| 场景 | block | grid | active warps % | SM % | DRAM % | L1 % | L2 % | regs/thread | shared KiB | waves/SM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mixtral_decode | (128, 1, 1) | (272, 1, 1) | 32.243235 | 53.230563 | 59.448995 | 53.249852 | 21.96102 | 106.0 | 7.952 | 1.0 |
| continuous_decode | (128, 1, 1) | (272, 1, 1) | 31.398759 | 58.644431 | 67.010256 | 59.7469 | 56.494531 | 106.0 | 7.952 | 1.0 |
| chunked_prefill | (128, 1, 1) | (272, 1, 1) | 30.952937 | 67.035583 | 36.772372 | 70.856004 | 77.431979 | 106.0 | 7.952 | 1.0 |

CUDA candidate 的 NCU basic 已采集，原始 `.ncu-rep` 与 CSV 位于 `artifacts/profile/`。vLLM container 的 NCU 是 `not_collected`：容器未安装 Linux NCU，现有 NCU 2026.2.1 是 Windows CLI，无法 attach 到 Linux container。该限制没有用推测的 counters 填充，也不影响 release latency 的实测结论。没有升级 detailed，因为 basic 已足以表明 CUDA candidate 的 Grouped GEMM 受 SIMT FP32 吞吐/活跃 warp 与专家小 M tail 的共同制约；vLLM 侧只可依据 NSYS 作 kernel-time 诊断。

## 可复现性与边界

- `strict_semantic_comparison`：输入 seed、FP32/IEEE、Top-2、T/E/K/N、device-side metadata、steady-state 时间边界相同；vLLM adapter correctness gate 通过。
- `production_reference_comparison`：保留 vLLM 的 block alignment、排序/临时 buffer、Triton fused kernel 与 `[E,N,K]` layout；因此是 source-faithful-adapted，不是将 vLLM server/runtime 原样嵌入。
- 不纳入 cuBLAS 作为完整七阶段链基准：它不提供 Top-K、alignment/histogram、scan、permute 和 weighted unpermute。
- 未 profile 的 release 数据是唯一性能事实；NSYS/NCU 的 replay/trace 开销不可与表内 p50 混算。

## 工件校验

[hash manifest](analysis/SHA256SUMS.json) · [machine-readable summary](analysis/summary.json) · profiler evidence 在 [artifacts/profile](artifacts/profile)。
