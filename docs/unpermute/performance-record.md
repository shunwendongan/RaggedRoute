# Unpermute 实际性能记录

## 2026-07-31 / RTX 3080 strict-FP32 baseline

- Git：`a9489abce704`；case `T=1024,E=64,N=256,top_k=2,uniform`；weighted reduce 与 token-owned CPU reference 通过，未使用 global atomic。

| Level / variant | p50 (us) | p95 (us) | CV |
|---|---:|---:|---:|
| L1 `cuda_naive` | 9.626 | 13.420 | 0.167 |
| L2 `cuda_naive` | 9.498 | 12.931 | 0.140 |
| L2 vLLM adapted reference | 9.446 | 12.777 | 0.156 |
| L2 comparable naive | 9.754 | 12.931 | 0.148 |

严格配对的 p50 差距仅约 3%，且双方 CV 都超过 0.10，无法得出稳定领先结论。NCU basic：1024 blocks、2.510 waves/SM、80.6% achieved occupancy、SM 22.5%、Memory/DRAM 35.9%、28 registers/thread；当前证据更像中等 memory pressure，而不是 occupancy 不足。

结论：保留 naive；向量化 weighted reduce 必须在更安静环境和对齐/非对齐 shape 上重新验证，3% 差距不足以晋升。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。
