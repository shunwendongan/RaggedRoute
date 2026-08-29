# Dense GEMM 实际性能记录

## 2026-08-29 / 统一简历作品集复测

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`；RTX 3080 / SM86、strict FP32、clean Release、5 processes、20 warmup、30 samples/process、seed `20260828`。
- 最强树内路径仍是 `cuda_register_tiled_v3_64x32_async`。对每 shape 最快 cuBLASLt/cuBLAS envelope，256³/512³/1024³ 为 `1.0095x/0.9588x/0.8514x`；ratio-of-sums `0.8716x`，geomean `0.9376x`，1/3 shape 获益。
- 1024³ NCU basic：1.506 waves/SM、85 registers/thread、34.63% achieved occupancy、SM/memory `73.26%/74.50%`。结论仍是局部 256³ 持平、完整矩阵库实现胜出；`Auto` 不变，禁止写“整体超过 cuBLAS”。

统一证据：[compact report](../reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)；面试卡片：[operator performance](../interview/operator-performance.md#1-dense-gemm)。本轮作品集 CV ceiling 为 0.50，但历史实验当时的 0.10 policy decision 不回写。

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

## 2026-08-01–02 / SM86 explicit candidate campaign

- optimized id 1–4 分别覆盖 tiled scalar、2D mapping、vector staging 和 combined 路径；id 5/6 是 64x64 sync/`cp.async` v2，id 7 是 64x32 `cp.async` v3。
- 四个早期候选中 `cuda_tiled_vector` 的中心 p50 相对 naive 最好约 `1.231x`，但独立复跑 CV 为 `0.188`，不满足门禁。
- v3 在 512³/1024³ L2 分别比 v2 async 快约 `1.19x/1.25x`；1024³ L2 为 `182.630 us`，仍比 strict-FP32 cuBLAS `153.754 us` 慢约 `1.19x`。
- 256³ 的 candidate/cuBLAS 结果都有 `CV>0.10`，且当时 manifest 记录了竞争 GPU 进程；因此 `Auto` 继续选 `cuda_naive`，所有 optimized id 只是显式 research path。
- NCU 对 1024³ v3 显示 85 registers/thread、无 local spill，global-load sectors 比 v2 低 22.2%，但仍是 cuBLAS 的 1.74x；下一轮应隔离 staging mapping/CTA reuse，不把 TF32 当作 strict-FP32 等价优化。

证据：[四候选报告](../reports/rtx3080-dense-gemm-optimization-bfe4494.md)；[v2 报告](../reports/rtx3080-dense-gemm-candidate-v2-0d2cbfa.md)；[v3 报告](../reports/rtx3080-dense-gemm-v3-64x32-f1da1d2.md)。
