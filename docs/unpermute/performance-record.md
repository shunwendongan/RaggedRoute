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

## 2026-08-03 / SM86 candidate pre-promotion

- clean base：`927c585e031ba6a01e41c01d21db03cb0d1ca9d0`；开发分支 `codex/unpermute-warp-vectorized`。
- 原始基线复现：CTest 8/8，四类 Compute Sanitizer PASS。`T=1024,E=64,N=256,top_k=2` 的 5-process median p50 为 naive `10.138 us`、vLLM `10.547 us`，平均 CV `0.151/0.123`；历史约 3% 的 vLLM 优势没有复现，当前窗口仍应视为 inconclusive。
- dirty-screening 选择了 shape-dispatched hybrid：small N 为 4-warps/token，`N>=512` 为一 CTA/token；32-shape 筛选相对 vLLM p50 几何平均约 `1.088x`。该数字不用于晋升。
- large-N NSYS/NCU 证明原纯 warp 版本存在 grid underfill；具体指标和 rejected-candidate 账本见 [optimization plan](optimization-plan.md)。
- 正式 clean-release、L3、sanitizer、SASS 和最终 NSYS/NCU 结果将在 checkpoint commit 后追加；在完成前不宣称 candidate 已晋升。
