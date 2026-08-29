# Unpermute 实际性能记录

## 2026-08-29 / 统一简历作品集复测

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`；T={64,512,1024,4096}、N={64,128,256,1024}、uniform/Zipf，5-process clean Release。
- `cuda_warp_token_vec4` 对最快 vLLM/naive envelope ratio-of-sums `1.0154x`、geomean `1.0172x`、12/32 shape 获益、最大回退 8.03%；对 vLLM 单独为 `1.0369x`、15/32 获益。
- 局部优势集中在中大型 T、窄 N：Zipf T4096/N128 `1.2892x`、uniform T4096/N128 `1.2487x`、Zipf T4096/N64 `1.2405x`。
- Candidate NCU 为 0.314 waves/SM、26.90% occupancy、34 registers/thread、Memory/DRAM 45.72%。结论是局部 shape winner，仍不足以进入全局 Auto。

统一证据：[compact report](../reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)；面试卡片：[operator performance](../interview/operator-performance.md#7-unpermute)。

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

## 2026-08-03 / SM86 candidate formal rejection

- clean base：`927c585e031ba6a01e41c01d21db03cb0d1ca9d0`；开发分支 `codex/unpermute-warp-vectorized`。
- 原始基线复现：CTest 8/8，四类 Compute Sanitizer PASS。`T=1024,E=64,N=256,top_k=2` 的 5-process median p50 为 naive `10.138 us`、vLLM `10.547 us`，平均 CV `0.151/0.123`；历史约 3% 的 vLLM 优势没有复现，当前窗口仍应视为 inconclusive。
- dirty-screening 选择了 shape-dispatched hybrid：small N 为 4-warps/token，`N>=512` 为一 CTA/token；32-shape 筛选相对 vLLM p50 几何平均约 `1.088x`。该数字不用于晋升。
- large-N NSYS/NCU 证明原纯 warp 版本存在 grid underfill；具体指标和 rejected-candidate 账本见 [optimization plan](optimization-plan.md)。
- 两轮正式 5-process Release 的 warm p50 几何平均如下；profiler duration 未参与 speedup：

| Run | L1 vs vLLM | L2 vs vLLM | warm candidate groups with CV>0.10 |
|---|---:|---:|---:|
| 首轮 | 1.008x | 1.066x | 55/64 |
| 自动复测 | 1.088x | 1.056x | 63/64 |

- 首轮最差 L1/L2 p50 speedup 为 `0.264x/0.542x`，复测为 `0.446x/0.492x`；两轮
  p95 最大 ratio 分别达到 `7.142/6.064` 与 `3.625/5.375`。即使平均值部分超过
  `1.05x`，逐 shape p50≤5% 回退、p95≤3% 回退和稳定性门槛仍同时失败。
- L3 只替换 Unpermute 得到 `1.0016x`，p95 ratio `0.947`，baseline/candidate CV
  `0.435/0.305`；结论是没有可信回退，也没有可信收益。
- 最终 CTA 路径 NSYS（`T=64,N=1024`）median 为 `1.792 us`，仅用于机制诊断；NCU
  basic 表明 warp/CTA case 都受短 grid/underfill 主导。SASS 生成了
  `LDG.E.128/STG.E.128`，resource table 为 34 registers/thread、`LOCAL:0/STACK:0`。
- CTest 8/8、memcheck/initcheck/racecheck/synccheck、aligned warp/CTA、unaligned tail 和
  Top-4 fallback 均通过。candidate 保持 zero workspace、无 atomic，但**不接入公开
  optimized dispatch**，registry 标为 `in_tree_cuda_research`。

结论：candidate 不晋升。已合并的代码保留 benchmark-only 实现、完整 raw/aggregate/profile 证据和
拒绝记录，不进入 public optimized dispatch。完整报告见
[SM86 candidate evidence report](../reports/unpermute-sm86-candidate-eba8f02.md)。

## 2026-08-04 / SM86 v2 screening follow-up

- `cuda_warp_token_vec4` 仍是当前源码中最强的 retained candidate；本轮没有新增可保留的 V2
  kernel。
- token/feature-tiled 和 one-warp CTA 实验均未同时超过 V1 与 vLLM；最终 V2D 相对 V1 的
  L1/L2 几何平均为 `0.9834x/0.9801x`，因此不进入 dispatch。
- compact comparison、NSYS/NCU 指标、hash manifest 和 raw archive 清单见
  [v2 screening evidence](../reports/artifacts/20260804T034147Z-728b9f36a6ad-unpermute-v2-screen/)。
