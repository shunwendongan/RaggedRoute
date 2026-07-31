# Expert Histogram 实际性能记录

## 2026-07-31 / RTX 3080 strict-FP32 baseline

- Git：`a9489abce704`；case `T=2048,E=64,top_k=2,Zipf s=1.4`；counts 与总数不变量精确通过。

| Level / variant | p50 (us) | p95 (us) | CV | Reset |
|---|---:|---:|---:|---|
| L1 `cuda_naive` | 9.175 | 10.540 | 0.100 | 排除 counts reset |
| L2 `cuda_naive` | 17.500 | 18.305 | 0.058 | 包含 counts reset |
| L2 CUB DeviceHistogram reference | 20.275 | 23.695 | 0.133 | 相同完整 L2 边界 |

严格配对中 naive p50 比 CUB reference 快 1.28×，且少用 767 B workspace；这不代表其他规模或分布仍然成立。NCU basic：16 blocks、0.039 waves/SM、15.7% achieved occupancy、SM 0.13%、Memory 0.82%，表明该固定小 metadata case 主要受 launch/underfill 限制，不能仅凭 atomic 名称断言带宽饱和。

结论：保留当前 naive；后续只有在更大 `R` 或更热点分布实测出现 atomic serialization 时，才测试 warp/block-private histogram。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。
