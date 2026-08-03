# RTX 3080 Grouped GEMM SM86 strict-FP32 优化报告

## 结论

`cuda_grouped_sm86_fp32_v1` **未通过相对 CUTLASS v4.6.1 SIMT FP32 的晋级门禁**，
因此只保留为 benchmark-only candidate。public `grouped_gemm`、`kAuto` 和默认 L3 chain 均继续使用
naive；runtime 对 Grouped GEMM optimized implementation id 仍明确拒绝。

正式十 shape Release 结果：shape-balanced geomean `0.907x`，等权 ratio-of-sums `0.805x`，
仅 4/10 shape 的 median-of-process-medians 不慢于 CUTLASS。最大反例是
`T=2048/E=64/K=N=128/uniform`，candidate/CUTLASS 为 `0.467x`。

## 环境与协议

- Main release Git：`7fb8f43034a4`；context release Git：`a5df6eb9fcd4`；均为 clean Git。
- GPU：NVIDIA GeForce RTX 3080，`sm_86`，68 SM，10 GB，
  UUID `7c5e95c0e5a415d824a0c8c8b58d6f39`。
- CUDA compiler 13.3.73；driver/runtime 13.1；CUTLASS 4.6.1；NCU 2026.2.1；NSYS 2026.1.3。
- Release：`-O3 -lineinfo`、`86-real`、strict FP32；seed `20260729`、warm cache、
  3 processes、20 warmup、30 samples、5 repeats。
- L2 interval 排除 route/input generation、offset preparation、H2D 与 allocation；workspace 为 0。
- useful TFLOP/s 为 `2RKN/latency`，routes/s 为 `R/latency`。

## 正式 L2 配对结果

下表 p50 是三个独立进程的 process median 之中位数，speedup=`CUTLASS/candidate`；
p95/CV 的完整字段见机器可读 artifact。

| Case | CUTLASS p50 (us) | Candidate p50 (us) | speedup | p95 ratio | candidate CV |
|---|---:|---:|---:|---:|---:|
| many empty `T16/E64/K64/N64` | 14.131 | 12.186 | 1.160x | 0.854 | 0.318 |
| non-aligned `T512/E64/K127/N129` | 37.478 | 51.200 | 0.732x | 1.233 | 0.046 |
| single expert `T512/E64/K128/N256` | 20.685 | 13.722 | 1.507x | 0.746 | 0.345 |
| single-hot `T512/E64/K128/N128` | 20.890 | 14.848 | 1.407x | 0.823 | 0.288 |
| tail `T17/E8/K13/N11` | 10.240 | 9.830 | 1.042x | 0.621 | 0.499 |
| uniform `T2048/E64/K128/N128` | 24.576 | 52.634 | 0.467x | 1.959 | 0.097 |
| uniform `T512/E16/K128/N256` | 20.582 | 22.733 | 0.905x | 0.826 | 0.163 |
| uniform `T512/E64/K128/N128` | 22.118 | 26.419 | 0.837x | 1.183 | 0.079 |
| Zipf1.4 `T2048/E64/K128/N128` | 40.038 | 59.597 | 0.672x | 1.291 | 0.228 |
| Zipf1.4 `T512/E64/K128/N128` | 21.709 | 25.600 | 0.848x | 0.914 | 0.504 |

代表 throughput：

- `T512/E64/K=N=128/uniform`：candidate `1.360 TFLOP/s, 38.76M routes/s`；
  CUTLASS `1.517 TFLOP/s, 46.30M routes/s`。
- `T512/E64/K=N=128/Zipf1.4`：candidate `1.332 TFLOP/s, 40.00M routes/s`；
  CUTLASS `1.546 TFLOP/s, 47.17M routes/s`。
- `T2048/E64/K=N=128/uniform`：candidate `2.550 TFLOP/s, 77.82M routes/s`；
  CUTLASS `5.530 TFLOP/s, 166.67M routes/s`。

cuBLAS per-expert 的十 shape geomean/ratio-of-sums 为 `0.098x/0.044x`，naive 为
`0.542x/0.423x`，说明 candidate 明显改善了 naive，但没有超过主要性能分母 CUTLASS。

## Cold 与 L3 旁证

- 64 MiB cold-scrub Zipf：candidate/CUTLASS p50 `0.839x`，p95 ratio `1.078`，
  candidate CV `0.211`。
- six-op L3 `chain_from_logits` p50 相对 naive：uniform `1.568x`、Zipf `1.705x`。
  但 candidate CV 分别为 `0.794/0.373`，uniform p95 ratio 为 `1.176`；这些结果不稳定，
  且 chain candidate 只在 adapter 内直接调用，不是 runtime promotion。

## NSYS / NCU 诊断

旧 baseline chain 的 NSYS 中 naive Grouped GEMM 占 78.1% GPU kernel time。
V4 Zipf 的单算子 NSYS 捕获 20 warmup + 1 launch，steady median 21.088 us；不同 workload 的
profiler duration 不作为 release speedup。

| Variant / case | grid | waves/SM | duration | SM/memory | achieved occ. | regs/thread | shared/block |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive / Zipf | — | — | 56.256 us | 45.41% | 58.94% | 40 | 1,024 B |
| C3 cap=2 / Zipf | 136 | 0.50 | 46.848 us | 22.00% | 15.07% | 106 | 7,952 B |
| V4 / Zipf | 272 | 1.00 | 25.248 us | 31.41% / 34.14% | 28.41% | 106 | 7,952 B |
| V4 / uniform | 272 | 1.00 | 30.176 us | 22.65% / 30.84% | 26.82% | 106 | 7,952 B |

V4 证明解除 2 blocks/SM 上限是有效单变量优化，但 106 registers/thread 仍将理论 occupancy
限制在 33.33%。NCU basic 还报告 uniform SM active cycles 约 `+31.8%/-38.1%` 的不均衡。
这些信号已能解释 underfill 与残余瓶颈，故没有升级 detailed/full/source。

## 正确性与安全

- fresh Release build 成功；CTest 8/8。
- CPU oracle 覆盖空 expert、single-hot、非对齐 `K/N`、不同 expert weights 与 redzone。
- caller stream、invalid optimized IDs 与 unsupported runtime path 均有测试。
- Compute Sanitizer memcheck、initcheck、racecheck、synccheck 全部通过。

## 证据位置

可提交摘要与逐 shape CSV：
[artifact bundle](artifacts/20260802T190625Z-7fb8f43-grouped-gemm-sm86/)；
原始 JSONL、aggregate/comparison、`.ncu-rep`、`.nsys-rep` 和 sanitizer logs 保留在本地，
其 SHA256 记录于 bundle，但 Nsight 二进制不进入 Git。

## 后续建议

1. 先降低 register live range，并以 `T=2048` 两个反例作为硬门禁。
2. 避免每个 CTA 重建完整 prefix，比较一次 device precompute 的收益与 workspace/API 成本。
3. 用真实 trace/Dirichlet effective-expert sweep 训练 selector；当前静态 direct/V4 阈值不可发布。
4. 只有重新通过 CUTLASS 单算子门禁后，才考虑把 L3 adapter 证据转成 runtime 或 fusion 方案。
