# RaggedRoute 后续开发路线

> **状态：部分实现；最新 Grouped GEMM 研究证据为 v5/v6 clean Release `c2205ed`，更新于 2026-08-30。** 七算子统一 portfolio 仍以 `9732a03` 为基线；A1/A2、三态 promotion evaluator、版本化 route-trace 输入、frame working set、shape heatmap 与 compact evidence 已实现；真实 captured/production trace、通用 working-set ring 和 cache sweep 仍是计划。当前可执行范围及证据边界以 `implementation-status.md` 为准。

> **环境边界：** 当前本地分支已在 Windows RTX 3080 / SM86 上完成 clean Release、Compute Sanitizer、NSYS 和 NCU 复测。v3 与非 Graph v4 仍受 WDDM CV 或实际 performance gate 限制；仅 fixed CUDA Graph replay 按可审计的授权 exception policy 晋级为显式实现。不得将 profiler duration 或 synthetic route fixture 写成发布性能。

## 0. 面向 AI Infra/CUDA 实习的当前优先级

1. **P0：关闭 F2 dispatch 与证据之间的缺口。** `HistogramExclusiveScan` 的 `Auto` 已指向 F2，但已有运行的 CV、fallback 与 L3 门禁失败。下一次 CUDA 窗口先重复 `configs/operators/scan/benchmark/fused_v2_promoted.json` 和对应 L3 suite；若仍失败，撤回默认选择而不是继续包装 speedup。
2. **P1：把 Grouped GEMM 作为唯一主 kernel 性能假设。** v5 direct-grid 已去掉 balanced workload 的 prefix/binary-search persistent traversal，v6 `32x128x16` 又将 global load/store requests 相对 v5 降低 43.4%/75.5%，在 uniform T512 对 CUTLASS 达 `1.2257x`。但完整 library envelope 仍只有 `0.9946x`，T2048 为 `0.7534x`；下一轮必须固定 v6 mainloop，只隔离 selector/tail-wave 映射，不再放大 tile 或叠加 descriptor queue。
3. **P1：补真实 workload，而不是扩充 synthetic shape。** route-trace schema、frame working set 和 evaluator 已实现；在匿名 captured/production trace 与通用 cache working set 到位前，real-trace policy 必须返回 `insufficient_evidence`。
4. **P2：整理真实 L3 证据。** PR #31/#32 合并到实验分支而非 `main`；未来需要以当前 `main` 重整、校验语义和来源后再决定是否合入，当前文档不得把它们写成已发布能力。
5. **P2：修复 evidence policy 漂移。** 清理 `l3_three_way_20260805` 中直接进入 Git 的 profiler 二进制，并增加扩展名/角色检查；超大 comparison JSON 应压缩为摘要，完整文件进入 immutable Release 资产。
6. **P3：低精度与新架构。** FP16/BF16 Tensor Core、Linux CUDA CI、H100/Blackwell 实卡验证属于后续增强；完成真实 kernel 与实卡验证前不进入简历成果。

## 1. 依赖顺序

按以下顺序推进，前一里程碑验收后才能把后一里程碑用于正式性能结论：

1. 多 variant suite 与 registry（已实现）；
2. 可审计聚合与严格配对比较（已实现）；
3. 版本化 workload 与 route-trace/frame working set（已实现；真实 trace 待输入）；
4. 三态 promotion evaluator（已实现）；
5. profiler 指标、shape heatmap 和 compact bundle（已实现；通用 working-set sweep 待补）。

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

> **执行边界：** evaluator 已实现；普通 synthetic matrix 可以产生三态结论，要求真实 trace 的 policy 只有在 `workload_source` 为 `captured` 或 `production` 时才有资格 `promote/reject`，否则为 `insufficient_evidence`。

- [x] 定义 `raggedroute.promotion_decision.v1`，实现 evaluator，输出仅允许 `promote`、`reject`、`insufficient_evidence`。
- [x] evaluator 只生成证据和建议，不自动修改 runtime 默认 dispatch。
- [x] 高 CV、缺少必需 trace、质量门禁未执行、配对缺失或协议不完整归为 `insufficient_evidence`。
- [x] 数据有效但未达到 speedup shape coverage、ratio-of-sums、最大单点退化或 workspace 门槛时归为 `reject`。
- [ ] workspace 门槛固定为：baseline workspace 为 0 时，candidate 最多增加 **64 MiB**；baseline 非零时，增长不超过 **25%**。
- [x] 只有 `promote`、人工 review 通过并完成完整 release 复测后，才能另行提交默认 dispatch 变更。

## 3. 里程碑 B：完整工作负载与结果管线

### B1. Workload spec 与确定性生成

- [ ] 定义版本化 workload spec 和确定性 suite generator，显式记录 generator 版本与 seed。
- [ ] 覆盖 `T/E/K/N`、非对齐 shape、uniform、Zipf `s={0, 0.6, 1.0, 1.4}`、单热点、双热点、空 expert，以及 warm/cold/rotating working set。
- [ ] 每个 workload 保存分布参数与派生统计，保证 baseline/candidate 获得同一输入，而不是仅获得同名分布。

### B2. Route trace

- [x] 定义 `raggedroute.route_trace.v1`：记录来源、trace ID、`T/E/top_k` 和多 frame top-k ids；normalized `.rrtrace` 及 run manifest 记录内容 SHA256。
- [x] 加载时校验数组长度、expert id 范围、每个 token 内 expert 去重、frame/trace ID 与 normalized 内容；任何失败均拒绝运行。
- [x] top-k ids trace 驱动 Token Permute、Grouped GEMM 与 `chain_from_logits`；`chain_from_tokens` 明确拒绝 trace，fixture 不冒充真实 router projection/gate latency。

### B3. Working set 与 cache 语义

- [ ] 实现 working-set ring，记录 requested bytes、实际 resolved bytes、副本数和设备 L2 容量。
- [ ] 明确 warm、cold-scrub 与 rotating 的操作顺序和计时排除项；cold-scrub 强制 `kernel_repeats=1`。
- [ ] cache policy 不同的记录不得聚合或互相计算 speedup。

### B4. Profiler、图表与发布制品

- [x] 为 NCU/NSYS 报告增加 text sidecar 与 metrics 抽取；不可用指标保存明确状态，不能用 0 代替缺失值。
- [x] profiler duration 只用于瓶颈诊断，永不进入正式 latency、speedup 或 promotion 计算。
- [x] 增加由 promotion decision 生成的 shape heatmap 与 compact summary；通用 working-set sweep、distribution 和 L1/L2/L3 breakdown 图仍待补。
- [x] 迁移到 `raggedroute.evidence_bundle.v2`：Git 固化 report、compact summary/comparison、normalized profiler metrics、manifest 与 `SHA256SUMS`；raw JSONL/full aggregate/run manifest/NCU/NSYS/SQLite 进入不可覆盖 Release ZIP，无法恢复的历史文件显式标记 `unavailable`。
- [ ] 增加仓库级 evidence policy gate：拒绝新的 `.ncu-rep`、`.nsys-rep`、SQLite 和未压缩 full aggregate/comparison 进入 Git；处理 `l3_three_way_20260805` 的既有例外。
- [ ] 补齐 captured/production trace；现有 synthetic fixture、可追溯 heatmap 与 v3 bundle 不因此升级真实 workload 的复现等级。

## 4. 验收测试

- [x] suite v1/v2 兼容；多 variant 复用 case ID、seed、params、协议与计时边界。
- [x] 公平 join 拒绝矩阵覆盖 GPU、build、case config、math、seed、level、cache、repeats、排除项、candidate/baseline 缺失与重复。
- [x] comparison 验证单 shape speedup、unweighted geometric mean 和 trace ratio-of-sums；保留 baseline/candidate 原始 latency。
- [x] promotion 覆盖三种状态、高 CV、缺真实 trace、provenance 缺失、correctness 优先级、无效 timing 与 JSON `null`。
- [x] route trace 覆盖 normalization、token 内重复 expert、frame working-set 展开；更完整的损坏输入矩阵仍可继续扩充。
- [ ] working-set rotation 验证 requested/resolved bytes、副本轮转、L2 元数据和 cold-scrub repeats 约束。
- [x] profiler metric alias 与缺失指标状态、NSYS/NCU 文本 sidecar、freeze bundle 非覆盖和 `SHA256SUMS` 完整性校验已实现。
- [x] shape heatmap 与 deterministic compact ZIP smoke 已实现；working-set/distribution 图仍待补。
- [x] 首次 clean Release naive/library/NSYS/NCU text evidence bundle 已在 RTX 3080 采集并提交。

## 5. 基线口径提醒

- Triton 仅用于学习与实现对照，**不是 RaggedRoute 的主性能 baseline**。
- CPU/PyTorch 是 correctness oracle；PyTorch 高层时间可另表报告，但**不是 CUDA kernel speedup 的主分母**。
- 主性能分母应是同机、同语义、同精度、同布局、同测量边界的 naive CUDA 或 cuBLAS/CUTLASS/CUB/production implementation。
- 没有同语义强基线时，不强行生成 speedup；改为报告 naive CUDA、optimized CUDA、绝对 latency/throughput 与实测 roof，并明确证据边界。
