# Top-K Gate 实际性能记录

## 2026-07-31 / RTX 3080 strict-FP32 baseline

- Git：`a9489abce704`；case `T=2048,E=64,top_k=2`；selected-softmax、lower-id tie 与 NaN 合同全部通过。

| Level / variant | p50 (us) | p95 (us) | CV |
|---|---:|---:|---:|
| L1 `cuda_naive` | 14.029 | 15.624 | 0.072 |
| L2 `cuda_naive` | 13.875 | 15.137 | 0.034 |

没有同语义外部 reference，因此不报告伪 speedup。NSYS 中该 kernel 占完整链 GPU kernel time 的 10.4%。NCU：仅 8 blocks、0.020 waves/SM、15.8% achieved occupancy、SM 2.6%、Memory 9.3%；detailed 的 long-scoreboard/wait samples 为 45/11，无 local-memory spill。

结论：主要限制是一个 thread 串行扫描一行导致的 underfill。下一候选是 warp-per-token Top-2 + selected-softmax，必须保持 ids 精确、tie/NaN 行为不变。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。
