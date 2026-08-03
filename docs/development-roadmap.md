# RaggedRoute 后续开发路线

> **状态：部分实现。** 里程碑 A1/A2 与 B4 的 profiler/text bundle 子集已实现，Histogram 已经过手工门禁晋升为第一个 shape-dispatched `Auto` candidate；A3 通用 promotion evaluator、真实 trace、working-set/plot 仍是计划。当前可执行范围及证据边界以 `implementation-status.md` 为准。

## 1. 依赖顺序

按以下顺序推进，前一里程碑验收后才能把后一里程碑用于正式性能结论：

1. 多 variant suite 与 registry（已实现）；
2. 可审计聚合与严格配对比较（已实现）；
3. 版本化 workload、真实 route trace 与 working-set sweep；
4. 三态 promotion evaluator；
5. profiler 指标、图表和 release bundle 固化。

`raggedroute.suite.v1`、`raggedroute.benchmark.v1` 与 `raggedroute.aggregate.v1` 在迁移期继续可读；v2 不覆盖已有 raw evidence。现有 v1 的计时与结果边界以 [benchmark-architecture.md](benchmark-architecture.md) 为准，本文只定义新增能力。

七个算子的优化假设、实验决策和实际性能记录统一从 [operator-optimization-index.md](operator-optimization-index.md) 进入；每个算子独立维护，避免把 Kernel Body、operator steady-state 和 chain 结果混在一起。

## 2. 里程碑 A：候选 variant 评估闭环

### A1. 多 variant suite 与 registry

- [x] 定义 `raggedroute.suite.v2`：一个 logical case 包含至少两个 `variants[]`，且恰好一个 variant 标记为 `promotion_baseline`。
- [x] 同一 logical case 的所有 variant 复用 case ID、seed、params、measurement level、cache mode、warmup、samples 与 kernel repeats；执行顺序确定性打乱。
- [x] runner 继续接受 suite v1，并将其视为单 variant case；v1 raw schema 与输出语义不变。
- [x] registry 将 variant 名传给同一个 typed adapter，并统一注入实现类别/版本/revision、依赖 revision、算法 ID 与 math mode；具体 library strategy 由对应 adapter 在接入时选择。
- [x] adapter 自有 `variant_config` 继续记录 tile、stage、vector width、scheduler 等参数；标准公平性字段由 registry 强制补全。

### A2. 聚合与公平配对

- [x] 定义 `raggedroute.aggregate.v2`，从 run manifest 完整保留 environment、protocol、seed、timing boundary、`excluded_steps`、workspace、case/variant/workload config 和全部配对键。
- [x] 新增 `raggedroute.comparison.v1` 与 `scripts/compare_results.py`。只有 GPU、build、case semantics、math mode、seed、measurement level、cache mode、repeats、samples 和排除项全部一致时才计算 speedup。
- [x] 缺失 baseline/candidate、重复配对键、字段缺失或任一公平性条件不一致时 fail closed，不生成部分 speedup。
- [x] shape-balanced 汇总使用 unweighted geometric mean；仅当所有 pair 都提供非负 `trace_weight` 时才计算 ratio-of-sums，否则保存 `null + reason`。

### A3. 质量门禁与三态 promotion

> **执行依赖：** A3 的 schema 与 policy 可以提前设计，但只有 B2 的 trace 证据和 B3 的 cache 语义到位后，evaluator 才能产生 `eligible` 或 `rejected`；在此之前一律为 `inconclusive`。

- [ ] 定义 `raggedroute.quality_gate.v1` 和 `raggedroute.promotion_decision.v1`，实现 evaluator，输出仅允许 `eligible`、`rejected`、`inconclusive`。
- [ ] evaluator 只生成证据和建议，不自动修改 runtime 默认 dispatch。
- [ ] 高 CV、缺少必需 trace、质量门禁未执行、配对缺失或协议不完整归为 `inconclusive`。
- [ ] 数据有效但未达到 speedup shape coverage、trace ratio-of-sums、最大单点退化或 workspace 门槛时归为 `rejected`。
- [ ] workspace 门槛固定为：baseline workspace 为 0 时，candidate 最多增加 **64 MiB**；baseline 非零时，增长不超过 **25%**。
- [ ] 只有 `eligible`、人工 review 通过并完成完整 release 复测后，才能另行提交默认 dispatch 变更。

## 3. 里程碑 B：完整工作负载与结果管线

### B1. Workload spec 与确定性生成

- [ ] 定义版本化 workload spec 和确定性 suite generator，显式记录 generator 版本与 seed。
- [ ] 覆盖 `T/E/K/N`、非对齐 shape、uniform、Zipf `s={0, 0.6, 1.0, 1.4}`、单热点、双热点、空 expert，以及 warm/cold/rotating working set。
- [ ] 每个 workload 保存分布参数与派生统计，保证 baseline/candidate 获得同一输入，而不是仅获得同名分布。

### B2. Route trace

- [ ] 定义 `raggedroute.route_trace.v1`：记录来源与 revision、frame、`T/E/top_k`、route weights、top-k ids 和内容 SHA256。
- [ ] 加载时校验数组长度、expert id 范围、每个 token 内 expert 去重、权重有效性及 SHA256；任何失败均拒绝运行。
- [ ] 只有 top-k ids 的 trace 可以真实驱动 Histogram 及后续 post-gate 算子。为 `chain_from_logits` 合成 logits 时必须标记 `synthetic_from_ids`，不得称为真实 router projection 或真实 gate latency。

### B3. Working set 与 cache 语义

- [ ] 实现 working-set ring，记录 requested bytes、实际 resolved bytes、副本数和设备 L2 容量。
- [ ] 明确 warm、cold-scrub 与 rotating 的操作顺序和计时排除项；cold-scrub 强制 `kernel_repeats=1`。
- [ ] cache policy 不同的记录不得聚合或互相计算 speedup。

### B4. Profiler、图表与发布制品

- [x] 为 NCU/NSYS 报告增加 text sidecar 与 metrics 抽取；不可用指标保存明确状态，不能用 0 代替缺失值。
- [x] profiler duration 只用于瓶颈诊断，永不进入正式 latency、speedup 或 promotion 计算。
- [ ] 增加 shape heatmap、working-set sweep、distribution/trace，以及 L1/L2/L3 breakdown 图表脚本；所有图表可追溯到 comparison/aggregate 输入。
- [x] 迁移到 `raggedroute.evidence_bundle.v2`：Git 固化 report、compact summary/comparison、normalized profiler metrics、manifest 与 `SHA256SUMS`；raw JSONL/full aggregate/run manifest/NCU/NSYS/SQLite 进入不可覆盖 Release ZIP，无法恢复的历史文件显式标记 `unavailable`。
- [ ] 补齐真实 trace 和可追溯图表；现有 v2 bundle 已记录 promotion/拒绝决策，但不因此升级缺失证据的复现等级。

## 4. 验收测试

- [x] suite v1/v2 兼容；多 variant 复用 case ID、seed、params、协议与计时边界。
- [x] 公平 join 拒绝矩阵覆盖 GPU、build、case config、math、seed、level、cache、repeats、排除项、candidate/baseline 缺失与重复。
- [x] comparison 验证单 shape speedup、unweighted geometric mean 和 trace ratio-of-sums；保留 baseline/candidate 原始 latency。
- [ ] promotion 覆盖三种状态、高 CV、缺 trace、门禁缺失、coverage、最大退化，以及 workspace 的 64 MiB/25% 两条边界。
- [ ] route trace 覆盖损坏 hash、长度错误、越界 id、token 内重复 expert 与非法权重；`synthetic_from_ids` 标签不可省略。
- [ ] working-set rotation 验证 requested/resolved bytes、副本轮转、L2 元数据和 cold-scrub repeats 约束。
- [x] profiler metric alias 与缺失指标状态、NSYS/NCU 文本 sidecar、freeze bundle 非覆盖和 `SHA256SUMS` 完整性校验已实现。
- [ ] shape/working-set/distribution 图表与 plot smoke 尚未实现。
- [x] 首次 clean Release naive/library/NSYS/NCU text evidence bundle 已在 RTX 3080 采集并提交。

## 5. 基线口径提醒

- Triton 仅用于学习与实现对照，**不是 RaggedRoute 的主性能 baseline**。
- CPU/PyTorch 是 correctness oracle；PyTorch 高层时间可另表报告，但**不是 CUDA kernel speedup 的主分母**。
- 主性能分母应是同机、同语义、同精度、同布局、同测量边界的 naive CUDA 或 cuBLAS/CUTLASS/CUB/production implementation。
- 没有同语义强基线时，不强行生成 speedup；改为报告 naive CUDA、optimized CUDA、绝对 latency/throughput 与实测 roof，并明确证据边界。
