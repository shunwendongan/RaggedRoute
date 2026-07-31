# Grouped GEMM 实际性能记录

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
