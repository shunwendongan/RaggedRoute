# Grouped GEMM 实际性能记录

## 2026-08-30 / SM86 v5 direct-grid 与 v6 32x128 follow-up

- 正式证据固定在 clean SHA `c2205ed1ba1063fccce3cd417fd671798dbfb66f`：RTX 3080 / SM86、strict FP32、10 shape、5 个独立进程、20 warmup、30 samples/process、50 repeats、seed `20260829`；250/250 records 通过 CPU oracle。最大 process CV 为 `0.4968`，44/50 groups 超过 `0.10` WDDM 风险线，0 个超过 `0.50` evidence ceiling，未删除离群点。
- v5 `cuda_grouped_sm86_fp32_v5_balanced_direct` 保留 v2 `16x32x16` `cp.async` mainloop，对 balanced workload 以 direct `(column,row,expert)` grid 取代 CTA prefix/binary search/persistent serial traversal。
- v6 `cuda_grouped_sm86_fp32_v6_balanced_32x128` 将 balanced path 改为 `32x128x16`、256 threads、每线程 `4x4` outer product、aligned `float4` store 和 Ampere 两级 `cp.async`；只在 average rows、skew、K/N 对齐条件满足时进入，其余回退 v5，workspace 为 0。
- 对最快 CUTLASS/cuBLAS envelope，v6 ratio-of-sums `0.9946x`、geomean `1.0352x`、5/10 shape 获益，最大 p50/p95 回退 32.74%/86.90%，因此是 `reject`，不能写成整体超过库实现。v6 对 v5 仅 `1.0116x` ratio-of-sums，且 p95 gate 失败。
- 可用于简历的限定结果：uniform `T512/E64/K128/N128` 为 `18.4627 us` 对 CUTLASS `22.6304 us`，即 `1.2257x`；single-hot 为 `12.4006 us` 对最快 library `22.0262 us`，即 `1.7762x`；两者都是 5/5 process pairs 同向。必须同时披露 uniform T2048 `0.7534x`、non-aligned `0.8380x` 和 single-expert 对 cuBLAS `0.7835x` 反例。
- NCU detailed 验证了大 tile 确实减少 request amplification：v6 相对 v5 的 global load/store requests 减少 43.4%/75.5%，DRAM writes 减少 82.7%；但 v6 只有 0.941 waves/SM、31.60% achieved occupancy 和 31.28% issue active，SM active-cycle minimum 比均值低 54.17%。当前剩余瓶颈是 underfill/work imbalance 与 issue efficiency，不是 spill（local load/store 都为 0）。
- v7 `64x128` 只做了 dirty 三进程 smoke；uniform T2048 比 v6 慢约 6.6%，因 accumulator live range 和 tail-row 同步代价撤回，不作为正式性能声明。

决策：v6 作为 zero-workspace、benchmark-only 的 SM86 高性能研究 candidate 保留，`KernelFamily::kAuto` 不变。统一七算子 portfolio 仍以 `9732a0343c60f869fc4166a0cc3cabba2fd67bbb` 为事实基线；本小节是只针对 Grouped GEMM 的后续 clean 证据。完整数据见 [v5/v6 compact report](../reports/compact/20260830-c2205ed-grouped-v6/REPORT.md)。

## 2026-08-29 / 统一简历作品集复测与 detailed 归因

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`；十 shape、5-process clean Release，strict FP32。
- 该轮完整矩阵的最强自研是 v2 16x32，不是 v3/v4A。v2 对最快 CUTLASS/cuBLAS envelope ratio-of-sums `0.8697x`、geomean `0.9472x`、3/10 shape 获益；对 CUTLASS 单独为 `0.9259x`、4/10 获益。
- 局部 winner：single-hot `1.6411x`、many-empty `1.3365x`、Zipf T512 `1.0790x`；关键反例 uniform T2048 `0.5335x`、non-aligned `0.7210x`。不能写整体胜过 CUTLASS。
- NSYS 中 v2 占 uniform/Zipf L3 GPU kernel time 71.6%/86.0%。为诊断失败机制升级的 v3/CUTLASS detailed 显示：v3 occupancy/L2 hit 更高，但 issue active 23.87% 对 34.79%，MIO/barrier/long-scoreboard 716/452/404 对 62/60/138。
- 当时决策：所有 candidate 继续 benchmark-only；后续回到 v2 mainloop，只隔离 scheduling/load-balance，这直接导向了上方 v5/v6 实验。

统一证据：[compact report](../reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)；瓶颈分析：[interview bottleneck analysis](../interview/bottleneck-analysis.md)。

## 2026-07-31 / RTX 3080 strict-FP32 ragged baseline

- Git：`a9489abce704`；case `T=512,E=64,K=N=128,top_k=2,Zipf s=1.4`；逐 expert CPU GEMM 与空 expert 合同通过。

| Level / variant | p50 (us) | p95 (us) | CV |
|---|---:|---:|---:|
| L1 `cuda_naive` | 47.718 | 48.538 | 0.017 |
| L2 `cuda_naive` | 48.128 | 48.538 | 0.051 |
| L2 CUTLASS Grouped reference | 23.757 | 26.726 | 0.091 |
| L2 cuBLAS per-active-expert | 588.390 | 630.610 | 0.044 |

CUTLASS p50 比 naive 快 2.02×；逐 expert host loop 因大量 launch 极慢。NSYS 中 naive 占完整链 GPU kernel time 71.9%。NCU detailed：26.353 waves/SM、59.1% achieved occupancy（理论 100%）、SM/Memory 45.4%、DRAM 13.9%、L1/L2 hit 86.6%/56.9%、issue active 19.3%；long-scoreboard samples 2264，且无 local-memory spill。

结论：这是最高优先级。下一候选应改善 Zipf 下的 grouped tile 调度/尾部负载均衡，而不是先追 occupancy 数字；以 CUTLASS strict-FP32 为强 reference。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。

## 2026-08-03 / SM86 strict-FP32 candidate 诊断与未晋级决定

环境：RTX 3080 (`sm_86`, 68 SM, UUID `7c5e95c0e5a415d824a0c8c8b58d6f39`)，
CUDA compiler 13.3.73，driver/runtime 13.1，CUTLASS 4.6.1，NCU 2026.2.1，NSYS 2026.1.3。
代码基于 `927c585`；下表是正式 clean-Git 采集前的一进程诊断（20 warmup、30 samples、
5 repeats、warm cache、seed `20260729`），用于决定是否撤销 runtime 晋级，不作为最终 PR headline。

| Case | CUTLASS p50 (us) | Candidate p50 (us) | speedup | candidate p95 (us) | CV |
|---|---:|---:|---:|---:|---:|
| tail `T17/E8/K13/N11` | 10.854 | 4.915 | 2.208x | 5.233 | 0.139 |
| many-empty `T16/E64/K64/N64` | 15.565 | 10.957 | 1.421x | 21.821 | 0.347 |
| uniform `T512/E64/K128/N128` | 28.262 | 30.720 | 0.920x | 32.881 | 0.026 |
| Zipf1.4 `T512/E64/K128/N128` | 24.166 | 24.166 | 1.000x | 24.371 | 0.005 |
| single-hot `T512/E64/K128/N128` | 25.600 | 14.541 | 1.761x | 23.593 | 0.178 |
| uniform `T512/E16/K128/N256` | 23.142 | 25.702 | 0.900x | 26.010 | 0.008 |
| non-aligned `T512/E64/K127/N129` | 47.309 | 46.490 | 1.018x | 46.694 | 0.003 |
| uniform `T2048/E64/K128/N128` | 24.166 | 44.851 | 0.539x | 46.490 | 0.018 |
| Zipf1.4 `T2048/E64/K128/N128` | 40.960 | 60.621 | 0.676x | 61.143 | 0.006 |
| single expert `T512/E64/K128/N256` | 23.040 | 13.722 | 1.679x | 14.336 | 0.166 |

- Candidate/CUTLASS shape-balanced geomean：`1.110x`。
- Candidate/CUTLASS 等权 ratio-of-sums：`0.951x`，低于 `1.03x` 门槛。
- 只有 6/10 shapes 不慢于 CUTLASS，低于 80% 要求；最大 p50 回退远超 5%。
- tiny/single-hot case 的单进程 CV 超过 0.10，也不满足稳定性门槛。
- workspace 均为 0；所有记录的 validation 都通过 CPU reference。

### Profiler 诊断

Profiler duration 只用于原因分析，不参与上方 speedup：

| Variant / distribution | grid | waves/SM | NCU duration | SM/memory | achieved occupancy | registers | static shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive / Zipf | — | — | 56.256 us | 45.41% | 58.94% | 40 | 1,024 B |
| C3 cap=2 / Zipf | 136 | 0.50 | 46.848 us | 22.00% | 15.07% | 106 | 7,952 B |
| V4 full residency / Zipf | 272 | 1.00 | 25.248 us | 31.41% / 34.14% | 28.41% | 106 | 7,952 B |
| V4 full residency / uniform | 272 | 1.00 | 30.176 us | 22.65% / 30.84% | 26.82% | 106 | 7,952 B |

理论 occupancy 为 33.33%，由 106 registers/thread 限制。V4 修复 C3 的 underfill，
但 NCU 仍报告 Zipf 约 `+13.1%/-6.2%` 的 SM active-cycle 不均，uniform 约
`+31.8%/-38.1%`；basic 已足以区分机制，因此没有升级 full/source。
候选 NSYS 的 21 次（20 warmup + 1 capture）kernel median 为 21.088 us；旧 naive 完整链
NSYS 中 Grouped GEMM 占 78.1% GPU kernel time。两者 workload 边界不同，只用于各自诊断。

### 决定

候选保留为 benchmark-only 失败实验，`cuda_grouped_sm86_fp32_v1` 不进入 public runtime；
Grouped GEMM 的 `kAuto` 与 L3 chain 继续使用 naive。正式三进程 clean-Git evidence 会在首个
实现提交后生成，并以新小节和 artifact bundle 补充。

## 2026-08-03 / clean-Git 三进程最终记录

- Main suite：`7fb8f43034a4`，candidate/CUTLASS shape-geomean `0.907x`、
  ratio-of-sums `0.805x`，4/10 shapes 不慢于 CUTLASS。
- 最大回退：`T2048/E64/K=N=128/uniform` 为 `0.467x`；Zipf 同规模为 `0.672x`。
- cold-scrub Zipf 为 `0.839x`；L3 p50 虽相对 naive 为 uniform `1.568x`、Zipf `1.705x`，
  但 CV `0.794/0.373`，不能形成发布证据。
- fresh build、CTest 8/8 和四项 Compute Sanitizer 全通过，workspace 保持 0。

最终报告：[RTX 3080 Grouped GEMM SM86 strict-FP32](../reports/rtx3080-grouped-gemm-sm86-a5df6eb.md)；
[可提交 artifact 摘要](../reports/artifacts/20260802T190625Z-7fb8f43-grouped-gemm-sm86/)。

## 2026-08-27 / SM86 strict-FP32 v3 tile geometry experiment

- 固定代码 SHA：`657d29e54ce94097be00b0ac57aea4f5e1f4f143`；RTX 3080、CUDA 13.3.73、driver 591.86、NCU 2026.2.1、NSYS 2026.1.3。
- `cuda_grouped_sm86_fp32_v3` 只将 aligned large path 改为 `16x64x16`、256 threads、两级 `cp.async`；调度、strict-FP32 合同、workspace 与 v2 fallback 不变，`Auto` 未修改。
- 10-case、5-process、clean uninstrumented Release 对 CUTLASS 的 ratio-of-sums 为 `0.9141x`，shape geomean 为 `1.0189x`，仅 4/10 shape 获益。关键反例为 uniform `T2048/E64/K=N=128` 的 `0.4745x`；同规模 Zipf1.4 为 `0.8526x`。
- 全部 correctness 和 sanitizer 门禁通过且 workspace 为 0，但所有配对都有进程内 `CV > 0.10`，因此 evaluator 输出 `insufficient_evidence`，不是可晋级结论。
- NSYS Zipf L3 中 v2/v3 Grouped median 分别为 43.871/51.839 us，占 GPU kernel time 70.9%/74.5%；其他阶段基本不变。
- NCU detailed 显示 v3 global-load requests 从 93,328 降至 81,200，local load/store 均为 0；但 registers/thread 从 86 升至 96、static shared 从 7,952 升至 12,048 B、achieved occupancy 从 36.24% 降至 30.29%、`sm__issue_active.avg.pct_of_peak_sustained_elapsed` 从 33.24% 降至 25.38%。更宽 tile 的资源与发射代价没有被 load request 减少抵消。

决定：v3 作为显式失败研究路径保留，不进入 `Auto`，也不在同一版本叠加第二种调度机制。证据见 [统一 v3 报告](../reports/rtx3080-six-ops-v3-657d29e.md) 与 [compact bundle](../reports/compact/20260827-657d29e-six-ops-v3/REPORT.md)。

## 2026-08-28 / SM86 v4A descriptor scheduling experiment

- 256/512/1024-thread prepass 的 static/queue 矩阵与 v4B cache-order 均完成 correctness、sanitizer 和 clean five-process Release 筛选；没有改变 `16x32x16` v2 FP32 mainloop 合同或 `Auto`。
- 最优 queue-1024 相对 v2 的 ratio-of-sums 为 `0.9225x`、geomean `0.8448x`、仅 30% shape 获益，最大 p50/p95 回退为 55.6%/264.3%，因此拒绝 descriptor scheduling 方向。
- NCU detailed 的 descriptor prepass 是单 CTA（0.00245 waves/SM，0.12% SM throughput），说明扩大到 1024 threads 不会使用更多 SM；descriptor mainloop 的 84 registers/thread、35.79% achieved occupancy 和 barrier/long-scoreboard/mio-throttle stalls 才是下一轮应隔离的变量。

证据见 [v4 final report](../reports/rtx3080-sm86-v4-graph-promotion.md) 与 [raw compact bundle](../reports/compact/20260828-sm86-v4-raw-final/FINAL_REPORT.md)。
