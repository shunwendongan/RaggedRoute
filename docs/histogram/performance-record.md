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

## 2026-08-03 / D0 strongest-baseline freeze

- 分支基点：`origin/main@927c585`；硬件 RTX 3080 / SM86 / 68 SM / 10 GiB；CUDA 13.3.73；driver 591.86。
- 协议：Release、5 processes、warmup 20、30 samples/process、seed `20260729`。
- D0 CTest 8/8；Compute Sanitizer memcheck/initcheck/racecheck/synccheck 全通过。

| L2 case | naive p50 (us) | CUB p50 (us) | strongest |
|---|---:|---:|---|
| `R=128,E=8,uniform` | 14.254 | 17.951 | naive |
| `R=4096,E=1,single-hot` | 15.933 | 17.551 | naive |
| `R=4096,E=64,Zipf-1.4` | 16.179 | 18.350 | naive |
| `R=65536,E=64,uniform` | 26.419 | 18.048 | CUB |
| `R=65536,E=64,single-hot` | 47.360 | 17.510 | CUB |
| `R=1M,E=64,uniform` | 323.533 | 25.344 | CUB |
| `R=1M,E=64,single-hot` | 579.738 | 18.330 | CUB |

数值为 5 个 process median 的中位数。WDDM 下存在双峰样本，原始 samples 不删除；manifest 记录了当时的 Douyin/QQ 等 GPU 背景进程。

## 2026-08-03 / H1–H3 与 dispatcher preflight

- H1：`R=65536,E=64,single-hot` 曾约 16 us，显著优于 naive；但 uniform/小 `R` 开销和跨进程稳定性不足，拒绝。
- H2：`R=128,E=8` 约 8.2 us，`R=4096,E=64,Zipf-1.4` 约 10.0 us；`R=65536` 明显退化，限定在 `R<=4096`。
- H3：`R=65536` 和 `R=1M` 的早期 smoke 显示潜力，但多工作区并发 benchmark 造成 GPU contention，正式数值必须以 clean、独占 GPU release 重跑。
- 5-process crossover preflight 在 `8192/16384` 方向不稳定，因此统一 `cuda_candidate` 在 `4097..32767` 回退 naive；不根据 Zipf/skew dispatch。
- H4 暂不实现：先收集最终 H3 的 NCU basic；只有 shared atomic counter 明确构成瓶颈才升级。

最终 promotion 结果、p50/p95/Gitems/s、NSYS/NCU 指标和 artifact bundle 将在 clean release 后补入本节。
