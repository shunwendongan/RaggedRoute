# Dense GEMM 实际性能记录

## 2026-07-31 / RTX 3080 strict-FP32 baseline

- Git：`a9489abce704`（clean Release，`sm_86 + -lineinfo`）；seed `20260729`；warm cache；3 processes × 30 samples。
- Case：row-major `M=N=K=256`，`alpha=1,beta=0`，CPU FP64-accumulation oracle 通过。

| Level / variant | p50 (us) | p95 (us) | CV | 说明 |
|---|---:|---:|---:|---|
| L1 `cuda_naive` | 27.034 | 27.136 | 0.004 | one thread per output |
| L2 `cuda_naive` | 27.034 | 27.136 | 0.003 | public wrapper |
| L2 cuBLASLt reference | 10.854 | 16.486 | 0.180 | strict pairing；reference p50 快 2.48× |

NCU basic：256 blocks、256 threads、0.627 waves/SM、40 registers/thread、48.5% achieved occupancy、SM/Memory 67.9%。Detailed：L2 hit 98.2%、DRAM 2.9%、long-scoreboard samples 852、无 local load/store。小网格 underfill 与缺少 tile 复用是下一步假设；不能把 profiler duration 当作上表 latency。

结论：保留 naive 作为教学/正确性基线；下一候选应先做 shared-memory tile/register blocking，再独立验证 `cp.async`。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。
