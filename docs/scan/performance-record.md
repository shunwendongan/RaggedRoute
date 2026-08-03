# Exclusive Scan 实际性能记录

## 2026-07-31 / RTX 3080 int32 metadata baseline

- Git：`a9489abce704`；case `E=64,R=4096`；`offsets[0]`、`offsets[E]` 与全部相邻差分精确通过。

| Level / variant | p50 (us) | p95 (us) | CV | Workspace |
|---|---:|---:|---:|---:|
| L1 `cuda_naive` | 8.192 | 9.069 | 0.055 | 0 B |
| L2 `cuda_naive` | 8.637 | 9.468 | 0.058 | 0 B |
| L2 CUB BlockScan reference | 9.175 | 9.707 | 0.068 | 0 B |
| L2 CUB DeviceScan | 27.008 | 29.069 | 0.069 | 1023 B |

严格配对中 naive 比 BlockScan 快 1.06×；DeviceScan 还需要 completion kernel。NCU basic：单 block/单 thread、0.0009 waves/SM、2.1% achieved occupancy，明确是 launch-bound tiny metadata work，而不是需要追求大数组 scan 吞吐。

结论：`E=64` 时保留 sequential naive；若未来 `E` 显著增大，再用独立 shape dispatch 比较 warp/block/device scan。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。

## 2026-08-03 / sm_86 单 warp候选研究

研究基于 clean `origin/main@927c585`，候选提交为 `bd68fd2`、`d26f8a6`、`32bf6c9`。所有候选通过 `E=1..64`、L1/L2、原地/分离、边界与随机输入，以及 memcheck/initcheck/racecheck/synccheck。

未修改 main 的 E=64 L2 library 基线：

| Variant | p50 (us) | p95 (us) | mean (us) | CV | calls/s | items/s | effective GB/s | Workspace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cuda_naive` | 8.586 | 9.239 | 8.577 | 0.051 | 116,465 | 7.45 M | 0.0598 | 0 B |
| CUB BlockScan | 8.768 | 10.822 | 9.138 | 0.112 | 114,051 | 7.30 M | 0.0497 | 0 B |
| CUB DeviceScan | 35.054 | 41.066 | 32.883 | 0.189 | 28,528 | 1.83 M | 0.0142 | 1023 B |

C2 5 进程稳定性复测覆盖 E=29..64、100 samples/process、1000 repeats/sample：

| 指标 | `cuda_warp_blocked_scalar` vs `cuda_naive` |
|---|---:|
| ratio-of-sums | `1.0503x` |
| shape-balanced geometric mean | `1.0496x` |
| p50 获益覆盖 | 33/36（91.7%） |
| 最坏 p50 退化 | 2.56%（E=32） |
| CV≤10% | 0/36 pairs |
| 最坏 p95 ratio | `7.042x`（E=31） |

结论：中心趋势超过 3% 目标，但稳定性和 p95 是硬失败，标记 `variance-limited`，不得晋升。C1 因两条 scan 链和不一致收益拒绝；C3 已由 cuobjdump 证明生成 64-bit load/store，但在 E=29..64 只有 6/36 shapes 比 C2 快至少 3%，相对 C2 ratio-of-sums 为 `0.9936x`，因此向量化也被拒绝。

`kAuto` 继续选择 `cuda_naive`，最终树不保留失败 candidate 源码。完整证据：[研究报告](../reports/rtx3080-scan-sm86-research-32bf6c9.md)、[候选 artifact bundle](../reports/artifacts/scan-sm86-research-32bf6c9/)、[clean-main library bundle](../reports/artifacts/scan-library-main-927c585/)。
