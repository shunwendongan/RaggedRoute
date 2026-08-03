# Grouped GEMM SM86 strict-FP32 优化方案

## 1. 冻结合同

- 输入、权重、输出、accumulator 均为 FP32 row-major；不启用 TF32/FP16/BF16 Tensor Core。
- `E<=64`，支持空 expert、`M_e=1`、单/全 active expert、非对齐 `K/N` 和 `hidden=0`。
- `GroupedGemmArgs`、layout 与 workspace API 不变；workspace 始终为 0。
- CUTLASS v4.6.1 SIMT FP32 Grouped GEMM 是 promotion baseline；naive 和 per-expert cuBLAS 是辅助对照。
- public `kAuto` 保持 naive。候选只有通过全部门禁才允许进入显式 optimized runtime dispatch。

数学定义、流量模型、架构边界和文献见 [research-notes](research-notes.md)。

## 2. 已执行路线

1. 从 `origin/main@927c585` 建立 `codex/grouped-gemm-sm86-opt` 独立 worktree；原 Dense GEMM 工作树未触碰。
2. 复现 C0 历史 16x16 shared-memory tile。
3. C1 只加入 device prefix + persistent round-robin scheduler。
4. C2 只改成 16x32x16、128-thread register microtile。
5. C3 只加入对齐 `float4`/`cp.async` 双缓冲，并保留同步 tail fallback。
6. NCU 发现 C3 被人为限制在 2 blocks/SM；V4 只移除该上限，按 device attribute 与 occupancy API 缓存 grid limit。
7. Final 对 tiny upper-tile bound 选择 C0，否则选择 V4；随后执行十 shape CUTLASS 配对诊断。

## 3. Benchmark 与门禁

正式 suite 固定 seed `20260729`、warm cache、20 warmup、30 samples、5 kernel repeats、3 个独立进程。
十组 case 覆盖 tail、50%+ 空 expert、uniform/Zipf/single-hot、宽矩阵、非对齐 `K/N`、
`T=512/2048` 和单 active expert。性能结论只使用未 profile 的 Release A/B。

相对 CUTLASS 必须同时满足：

- correctness、CTest、memcheck/initcheck/racecheck/synccheck 全通过；
- `CV<=0.10`，至少 80% shapes 提速；
- 等权标准 workload ratio-of-sums `>=1.03x`；
- 任一 shape p50 回退不超过 5%，p95 ratio 不超过 1.03；
- workspace 为 0，GPU UUID、数值合同、seed、cache 与 measurement boundary 完全配对。

## 4. 决策记录

| 日期 | 版本/假设 | 代表 Zipf L2 p50 | 结论 |
|---|---|---:|---|
| 2026-08-03 | C0 16x16 sync | 56.730 us | 未复现历史优势；慢于 naive/CUTLASS |
| 2026-08-03 | C1 persistent scheduler | 57.037 us | scheduler 单独不足以抵消 mainloop 成本 |
| 2026-08-03 | C2 register 16x32 sync | 50.176 us | 有改善，仍慢于 naive |
| 2026-08-03 | C3 `cp.async`, cap=2 blocks/SM | 32.973 us | 胜 naive；NCU 显示 0.50 waves/SM、15.1% occupancy |
| 2026-08-03 | V4 full occupancy-derived residency | 22.528 us | 单变量有效；272 blocks、1.00 waves/SM |
| 2026-08-03 | Final direct/V4 selector | 23.347 us | 进入十 shape 诊断 |
| 2026-08-03 | CUTLASS reference | 20.890 us | 同进程代表 case 最快 |

十 shape 单进程诊断的 candidate/CUTLASS shape-geomean 为 `1.110x`，但 ratio-of-sums 仅
`0.951x`，并在 `T=2048` uniform/Zipf 上分别只有 `0.539x/0.676x`。因此最终候选
**未晋级**：保留 benchmark-only 代码与证据，public runtime 继续明确拒绝 Grouped GEMM optimized ID。

完整逐 shape 结果与 profiler 指标见 [performance-record](performance-record.md)。

## 5. 后续路线

- 第一优先级是降低 106 registers/thread 和重复 per-CTA prefix 成本，并专门复测 `T=2048` 反例。
- 增加真实 trace 与 distribution-aware sweep 后再设计 selector，禁止用当前静态阈值推断生产最优。
- 任何新候选仍从 correctness → unprofiled A/B → NSYS → filtered NCU basic 开始；basic 足够时不升级 full/source。
