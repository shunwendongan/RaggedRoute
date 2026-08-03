# RaggedRoute 仓库与性能证据审计（2026-08-03）

## 结论

审计基线是远端 `main@aea9fe96dda0`，包含已合并的 PR #1–#21。该 tree 在 RTX 3080 / SM86 上通过 Release 67/67 构建、CTest 8/8、Python 34/34、Triton 8/8 和全算子 smoke。当前问题主要在发布事实、文件角色一致性和证据可追溯性，而不是基本可执行性。

## 发现

| 严重度 | 发现 | 影响 | 整改 |
|---|---|---|---|
| P0 | README/实现状态声称所有 `Auto` 都是 naive，但 `select_kernel()` 已将 SM86 Histogram `Auto` 选到 optimized candidate | 用户看到的发布行为与源码不一致 | PR 1 同步文档与状态矩阵 |
| P0 | Unpermute bundle 的两个 throughput CSV 与 `SHA256SUMS` 不一致 | 当前不能通过完整性校验 | PR 3 由 raw JSONL 重生成并重新签名 |
| P1 | Dense-levels/Grouped GEMM checksum 只指向本机路径；Top-K raw benchmark 也未提交 | 仓库 clone 无法独立复核 | PR 3 将已找回且 hash 匹配的 raw 文件发布为 Release asset |
| P1 | 只有 Dense GEMM 完成 `cuda_naive/` 归位，其余六算子仍编译根目录 `baseline.cu` | PR #6 建立的 role 目录与实际 CMake 不一致 | PR 2 统一迁移并删除过期 `.gitkeep` |
| P1 | 62 个 active config 平铺，正式 suite、smoke、profile、policy 和历史实验混杂 | 难以找到当前发布入口 | PR 2 按 project/operator/cross-backend/policy 归档 |
| P1 | GitHub 无 workflow，21 个合并 PR 的 checks/reviews/comments 均为 0 | 回归只依赖本地人工执行 | PR 4 加 CPU CI、evidence gate 和手动 SM86 workflow |
| P2 | `docs/reports/artifacts` 有 395 个文件、约 59.7 MiB，raw 数据提交策略不一 | 仓库体积会持续增长 | PR 3 改为 Git 摘要 + immutable Release raw archive |
| P2 | 缺少项目级许可证 | 不适合公开发布 | PR 1 采用 Apache-2.0，保留第三方 notices |

## 性能与 Profile 优先级

1. **P0 Grouped GEMM**：naive 在代表 L3 trace 占 71.9% GPU kernel time；最终 candidate 对 CUTLASS ratio-of-sums 仅 `0.805x`，最差 shape `0.467x`。先解决调度、prefix/descriptor 重建和 106 registers/thread 压力。
2. **P1 Top-K Gate**：`T=2048,E=64` 单点有约 `1.46x` 潜力，但连续 E/T bucket 门禁未通过；需要以反例 shape 重测保守 dispatch。
3. **P1 Dense GEMM**：v3 已接近库实现，但 1024³ 仍比 cuBLAS 慢约 19%；优先分离 transaction amplification 与 barrier/MIO 成本。
4. **P1 Token Permute**：explicit alias 只小幅超过 naive，full-from-ids 边界仍比 vLLM 慢约 33.6%；先用 NSYS 分解 metadata preparation 与 row-copy。
5. **P2 Unpermute**：平均改善未通过稳定性、单 shape 和 p95 门禁，且 L3 占比低；在安静环境复测后再决定。
6. **P3 Scan/Histogram**：Scan 的独立算子是 launch-bound，应转向 metadata fusion；Histogram 已晋升，只需真实 trace/crossover 回归。

所有后续性能 PR 必须保持 strict-FP32 和原测量边界，先过 correctness，再做未插桩 Release A/B，然后使用 NSYS 选热点并对一次 post-warmup launch 采 NCU。Profiler duration 只解释机制，不用于性能晋级。
