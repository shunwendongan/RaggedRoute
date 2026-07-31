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
