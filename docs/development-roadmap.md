# RaggedRoute 后续开发路线

> **状态：规划中，未实现。** 本文描述候选 variant 评估闭环与完整工作负载/结果管线；出现的 schema、脚本、门禁和制品名称均是后续目标，不代表当前仓库已经提供这些能力。当前可执行范围仍以 `implementation-status.md` 为准。

## 1. 依赖顺序

按以下顺序推进，前一里程碑验收后才能把后一里程碑用于正式性能结论：

1. 多 variant suite 与 registry；
2. 可审计聚合与严格配对比较；
3. 版本化 workload、真实 route trace 与 working-set sweep；
4. 三态 promotion evaluator；
5. profiler 指标、图表和 release bundle 固化。

`raggedroute.suite.v1`、`raggedroute.benchmark.v1` 与 `raggedroute.aggregate.v1` 在迁移期继续可读；v2 不覆盖已有 raw evidence。现有 v1 的计时与结果边界以 [benchmark-architecture.md](benchmark-architecture.md) 为准，本文只定义新增能力。

## 2. 里程碑 A：候选 variant 评估闭环

### A1. 多 variant suite 与 registry

- [ ] 定义 `raggedroute.suite.v2`：一个 logical case 包含 `variants[]`，且恰好一个 variant 标记为 `promotion_baseline`。
- [ ] 同一 logical case 的所有 variant 复用 case ID、seed、params、measurement level、cache mode、warmup、samples 与 kernel repeats；执行顺序可以确定性打乱。
- [ ] runner 继续接受 suite v1，并将其视为单 variant case；v1 输出语义不得改变。
- [ ] registry 让同一个 typed adapter 复用输入、CPU oracle、reset、validation 与计时边界，只替换 implementation strategy，禁止 variant 私自增删被测步骤。
- [ ] `variant_config` 至少记录实现类别与版本、依赖 revision、算法 ID、math mode，以及 tile、stage、vector width、scheduler 等优化参数；关键公平性字段不得只写进 notes。

### A2. 聚合与公平配对

- [ ] 定义 `raggedroute.aggregate.v2`，完整保留 environment、protocol、seed、timing boundary、`excluded_steps`、workspace、case/variant/workload config 和全部配对键。
- [ ] 新增 `raggedroute.comparison.v1` 与 `scripts/compare_results.py`。只有 GPU、build、input/weight/accumulator/output dtype、math mode、layout、shape、seed、measurement level、cache mode、repeats 和排除项全部一致时才计算 speedup。
- [ ] 缺失 baseline/candidate、重复配对键、字段缺失或任一公平性条件不一致时 fail closed：不计算 speedup，并输出可审计原因。
- [ ] shape-balanced 汇总可用 unweighted geometric mean；真实 trace 总收益必须使用 `sum(weight * baseline_latency) / sum(weight * candidate_latency)`，不得用 weighted geometric mean 冒充部署收益。

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

- [ ] 为 NCU 报告增加 sidecar 与 metrics 抽取；不可用指标写 `null` 并附 `reason`，不能用 0 代替缺失值。
- [ ] profiler duration 只用于瓶颈诊断，永不进入正式 latency、speedup 或 promotion 计算。
- [ ] 增加 shape heatmap、working-set sweep、distribution/trace，以及 L1/L2/L3 breakdown 图表脚本；所有图表可追溯到 comparison/aggregate 输入。
- [ ] 实现 release bundle freeze，将 raw JSONL、manifest、aggregate、comparison、promotion decision、profiler metrics、图表与 `SHA256SUMS` 固化到 `docs/reports/artifacts/<run_id>/`，禁止覆盖既有 run。

## 4. 验收测试

- [ ] suite v1/v2 兼容；多 variant 确实复用 case ID、seed、输入、协议与计时边界。
- [ ] 公平 join 拒绝矩阵覆盖每个配对字段的缺失、不一致、重复与 baseline 缺失。
- [ ] comparison 验证单 shape speedup、unweighted geometric mean 和 trace ratio-of-sums；不得丢失原始 latency。
- [ ] promotion 覆盖三种状态、高 CV、缺 trace、门禁缺失、coverage、最大退化，以及 workspace 的 64 MiB/25% 两条边界。
- [ ] route trace 覆盖损坏 hash、长度错误、越界 id、token 内重复 expert 与非法权重；`synthetic_from_ids` 标签不可省略。
- [ ] working-set rotation 验证 requested/resolved bytes、副本轮转、L2 元数据和 cold-scrub repeats 约束。
- [ ] profiler metric alias、缺失指标 `null + reason`、plot smoke 与 freeze bundle 完整性校验通过。
- [ ] 完整 release 链从 suite 到 `SHA256SUMS` 可由 clean Release build 重现，且已有 artifact 不会被覆盖。

## 5. 基线口径提醒

- Triton 仅用于学习与实现对照，**不是 RaggedRoute 的主性能 baseline**。
- CPU/PyTorch 是 correctness oracle；PyTorch 高层时间可另表报告，但**不是 CUDA kernel speedup 的主分母**。
- 主性能分母应是同机、同语义、同精度、同布局、同测量边界的 naive CUDA 或 cuBLAS/CUTLASS/CUB/production implementation。
- 没有同语义强基线时，不强行生成 speedup；改为报告 naive CUDA、optimized CUDA、绝对 latency/throughput 与实测 roof，并明确证据边界。
