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
- H3：`R=65536` 和 `R=1M` 晋升；128 CTA cap 将 `R=1M,E=64` 的 global merge atomic 上界降为 8192。
- 5-process crossover preflight 在 `8192/16384` 方向不稳定，因此统一 `cuda_candidate` 在 `4097..32767` 回退 naive；不根据 Zipf/skew dispatch。
- H4 拒绝：NCU detailed 中 hot/uniform 均执行 32768 个 warp-level shared atomic 指令；shared pipe 仅约 3.7%/7.3%，没有 shared-atomic 饱和证据。

## 2026-08-03 / `cuda_candidate` promotion

- 测量 SHA：`88b670b6f0a5`，clean Release `sm_86`，5 processes，20 warmup，30 samples/process，seed `20260729`。
- Workspace 始终为 0；candidate p95 在全部 12 个 case 均低于 strongest baseline；所有 case 相对 naive 的 p50 均无回退。
- 预声明的中大型 winner `R=1M,E=64,uniform` 在 5/5 processes 中方向一致，p50 提升 23.9%。`R=65536,E=16` 有 1/5 process 出现 1.28% 噪声级反向，其余 case 为 5/5；聚合 p50/p95 仍分别改善 3.3%/14.8%。

| L2 case | strongest baseline | baseline p50 (us) | candidate p50 / p95 (us) | speedup | candidate Gitems/s |
|---|---|---:|---:|---:|---:|
| `R=128,E=8,uniform` | naive | 14.095 | 8.202 / 8.972 | 1.718× | 0.016 |
| `R=4096,E=1,single-hot` | naive | 15.094 | 10.220 / 11.255 | 1.477× | 0.401 |
| `R=4096,E=64,uniform` | naive | 16.159 | 10.097 / 11.337 | 1.600× | 0.406 |
| `R=4096,E=64,round-robin` | naive | 15.165 | 9.820 / 11.480 | 1.544× | 0.417 |
| `R=4096,E=64,Zipf-1.4` | naive | 16.046 | 9.943 / 11.356 | 1.614× | 0.412 |
| `R=65536,E=16,uniform` | CUB | 16.742 | 16.205 / 18.819 | 1.033× | 4.044 |
| `R=65536,E=64,uniform` | CUB | 16.179 | 15.718 / 17.459 | 1.029× | 4.169 |
| `R=65536,E=64,Zipf-1.4` | CUB | 16.742 | 15.923 / 17.861 | 1.051× | 4.116 |
| `R=65536,E=64,single-hot` | CUB | 16.282 | 15.718 / 18.022 | 1.036× | 4.169 |
| `R=1M,E=64,uniform` | CUB | 21.248 | 17.152 / 18.181 | 1.239× | 61.134 |
| `R=1M,E=64,Zipf-1.4` | CUB | 18.330 | 17.818 / 21.151 | 1.029× | 58.851 |
| `R=1M,E=64,single-hot` | CUB | 18.125 | 17.357 / 22.175 | 1.044× | 60.413 |

NSYS：small trace 中 21 次 single-CTA kernel、无逐次 L2 reset；large trace 中 21 次 block-private kernel 对应 21 次 L2 reset，无额外 merge kernel。NCU basic/detailed：small grid=1、`0.00245 waves/SM`，明确 launch/underfill；large grid=128、`0.3137 waves/SM`、21 registers/thread、0 local load/store、约 66% achieved occupancy、约 49% DRAM、long-scoreboard 为主要采样 stall。Profiler duration 仅作诊断，不参与上表 speedup。

结论：H2/H3 shape dispatcher 晋升为 SM86 Histogram `kAuto`；H1/H4 和不稳定交叉区间候选不进入 shipping path。
