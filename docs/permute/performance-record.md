# Token Permute 实际性能记录

## 2026-08-29 / 统一简历作品集复测

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`；5-process clean Release。
- Full-from-ids：`cuda_candidate_v2_from_ids` 对 adapted `vllm_moe_permute` ratio-of-sums `1.5671x`、geomean `1.5885x`、5/5 shape 获益、五进程方向 25/25，逐 shape `1.2896x–1.8501x`；v3 为 `1.5089x`。
- Pure-permute：v2/v3 对 retained token-owned 仅 `0.9841x/0.9892x`，分别 11/28 shape 获益。收益属于 fused preparation + copy 的完整边界，不能写成 pure copy kernel 普遍领先。
- v3 representative NCU 的 DRAM throughput 为 86.85%，确认 payload path bandwidth-bound；`Auto` 保持 naive。

统一证据：[compact report](../reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)；面试卡片：[operator performance](../interview/operator-performance.md#5-token-permute)。本轮 CV 0.50 口径不覆盖历史 0.10 promotion decision。

## 2026-07-31 / RTX 3080 strict-FP32 baseline

- Git：`a9489abce704`；case `T=512,E=64,K=256,top_k=2,Zipf s=1.4`；expert segment、`route_pos`、`sorted_route` 与逐行内容通过。

| Level / variant | p50 (us) | p95 (us) | CV | Mapping boundary |
|---|---:|---:|---:|---|
| L1 `cuda_naive` | 10.240 | 10.240 | 0.094 | cursor reset 排除，单 launch |
| L2 `cuda_naive` | 19.763 | 21.356 | 0.173 | 包含 cursor reset |
| L2 vLLM full-from-ids reference | 32.358 | 44.595 | 0.179 | histogram/sort/scan/expand |
| L2 `cuda_naive_from_ids` | 40.550 | 48.548 | 0.112 | histogram/scan/atomic permute |

严格 full-from-ids 配对中 vLLM p50 快 1.25×，但两方 CV 均超过 0.10，不能用于 promotion。NCU basic 对 kernel body 显示 1024 blocks、1.255 waves/SM、70.9% achieved occupancy、SM 10.3%、Memory/DRAM 19.6%，没有带宽饱和证据。

结论：保留 naive 基线；后续优先比较预计算位置与向量化 copy，但必须把 mapping preparation 在 L2/L3 中公平计入。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。

## 2026-08-03 / SM86 candidate campaign

- 分支基线：`927c585e031b`；目标 RTX 3080 / SM86 / strict FP32。
- 已实现并注册五个独立候选：atomic vectorized 128/64/256、token-owned Top-2、block-partial。
- API、caller stream、`4*E` workspace 与默认 naive dispatch 均保持不变；`cuda_candidate` 是经 high-repeat selection 选出的 explicit research/对照 alias。
- CTest 8/8 与 Compute Sanitizer memcheck/initcheck/racecheck/synccheck 全通过；两轮 candidate Release 和 library/chain Release 全部 `validation.ok=true`。
- 两轮 Release 均未满足稳定性与 promotion gate。Run 1 最好 ratio-of-sums 为 atomic-256 的 1.0373x，但只有 75% shapes 加速且 worst speedup 0.6300x；Run 2 没有候选同时满足 gate。所有候选 reject，默认 dispatch 保持 `cuda_naive`。
- 后续增加 L2-only 高重复 selection suite（50 warmups、50 samples、warm-case repeats=100）。token-owned Top-2 在两轮中均排名第一，ratio-of-sums 为 1.0456x/1.0463x，因此 `cuda_candidate` alias 已绑定 implementation ID 4；generic top-k 仍由该实现内部回退 atomic-128。
- vLLM full-from-ids、prepared mapping、L3、NSYS 与 NCU 只用于 selected candidate 的能力定位；`KernelFamily::kAuto` 是否晋升仍与 explicit `cuda_candidate` alias 分开处理。

最终 high-repeat library 对照中，`cuda_candidate_from_ids` 相对 `cuda_naive_from_ids` ratio-of-sums 为 1.0182x（4/5 case 加速）；相对 vLLM full-from-ids 为 0.7484x，即 aggregate latency 约高 33.6%。

完整数据、能力矩阵和 profiler 摘要：[中央报告](../reports/rtx3080-permute-sm86-5cb9bd4.md)；[initial bundle](../reports/artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/)；[selected token-owned bundle](../reports/artifacts/20260803T071455Z-af4947f-permute-selected-token-owned-v1/)。

## 2026-08-03 / SM86 v2 tile4 + fused preparation

- 代码证据 SHA：`bdc77c276878ea2d0198e69a1fbcb4f9bd4355b1`；RTX 3080 / CUDA 13.3 / strict FP32。`cuda_candidate` 现在显式指向 shape-dispatched v2；`KernelFamily::kAuto` 仍是 naive。
- v2 在 large/aligned Top-2 使用 128-thread、4-token CTA 的 direct cursor path；其余 shape 回退 implementation ID 4。full-from-ids 的 v2 先以单 CTA 融合 counts、exclusive scan 与 cursor reset，再执行该 selector；外部 workspace 不增长。
- CTest 8/8，以及 Compute Sanitizer memcheck/initcheck/racecheck/synccheck 均通过。full-from-ids 5 个同边界 case 相对旧 candidate-from-ids 的 ratio-of-sums 为 **1.7302x**，相对 pinned vLLM 为 **1.4110x**；vLLM 对照逐 case 为 1.5387x、1.4439x、1.0262x、1.6177x、1.5239x。
- 纯 permute 28-case v2 rerun ratio-of-sums 为 1.0755x（18/28 快），但全部配对的 CV 超过 0.10，且最差 shape 为 0.2487x。因此它不满足原严格 promotion policy；这次保留是用户明确要求的 explicit candidate 决定，不应被表述为稳定的默认升级。
- NSYS large-uniform 的诊断中，token-owned copy median 10.400 us，tile4-direct 9.568 us；v2 full path 由 10.432 us fused prepare + 9.632 us copy 构成。NCU detailed 显示 direct 的 0.627 waves/SM、34 registers/thread、41.7% achieved occupancy、48.6% DRAM throughput（ID4：2.510、34、63.6%、66.3%）。这些是 profiler 诊断值，非 Release latency。

完整报告与可审计 compact artifacts：[SM86 v2 report](../reports/rtx3080-permute-sm86-v2-bdc77c2.md)；[v2 artifact bundle](../reports/artifacts/20260803T181954Z-bdc77c2-permute-sm86-v2/)。

## 2026-08-27 / SM86 v3 two-token CTA experiment

- 固定代码 SHA：`657d29e54ce94097be00b0ac57aea4f5e1f4f143`。v3 新增 64-thread、2-token CTA，并按 shape 在 tile4/tile2/token-owned v1 间选择；workspace、atomic placement 语义和 `Auto` 均不变。
- 28-case、5-process、clean uninstrumented Release 相对 token-owned v1 的 ratio-of-sums 为 `0.9938x`，shape geomean `0.9980x`，仅 10/28 shape 获益。重点反例 `T4096/K256`、`T4096/K1024`、`T2048/K512`、`T8192/K256` 分别为 `0.9467x`、`0.9957x`、`0.9424x`、`0.9900x`。
- full-from-ids 相对 pinned vLLM 的趋势仍为 `1.5452x` ratio-of-sums、5/5 shape 获益，但两侧全部 `CV > 0.10`，不能写成正式 speedup。
- NCU basic 对 `T4096/K1024/single_hot` 显示 tile4/tile2 的 diagnostic duration 为 80.608/79.744 us，DRAM throughput 为 84.55%/85.11%，registers/thread 都是 34。tile2 将 waves/SM 从 1.255 增至 1.882，却把 achieved occupancy 从 77.79% 降到 58.20%；纯 CTA geometry 改动没有突破带宽上限。

决定：v3 不晋级，保留为显式 research path；正式 evaluator 因 WDDM 长尾输出 `insufficient_evidence`。证据见 [统一 v3 报告](../reports/rtx3080-six-ops-v3-657d29e.md) 与 [compact bundle](../reports/compact/20260827-657d29e-six-ops-v3/REPORT.md)。

## 2026-08-28 / Top-K 2/4/8 route-prep 与 gather 筛选

- shared-rank route-prep t1024 相对 v3 的 ratio-of-sums 为 `1.0439x`，但只有 40.3% shape 获益，最大 p50/p95 回退为 55.9%/111.9%；不晋级。
- Top-4 token-owned copy 相对 shared-rank t256 的 ratio-of-sums 为 `1.0170x`、72.9% shape 获益，仍未形成稳定的全面晋级证据。
- 不物化 `X_permuted` 的 gather fusion 相对 current postroute chain 为 `0.8191x`，并增加 4,195,072 B workspace；拒绝该方向。该结果不能被 CUDA Graph fixed replay 的 host speedup 抵消或改写。

详情见 [v4 final report](../reports/rtx3080-sm86-v4-graph-promotion.md)。
