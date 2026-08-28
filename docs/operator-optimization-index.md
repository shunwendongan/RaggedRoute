# 算子优化文档索引

本目录按 RaggedRoute 的七个语义算子分别保存优化方案、实验决策和实际性能记录；benchmark registry 另有一个 `histogram_exclusive_scan` 融合 adapter，因此当前是七阶段数据流、八个 adapter。

当前统一实测报告：[RTX 3080 七算子 naive benchmark 与 Nsight 分析（a9489ab）](reports/rtx3080-naive-profile-a9489ab.md)；最新本地 v3 研究轮次见 [六阶段 v3 统一复测（657d29e）](reports/rtx3080-six-ops-v3-657d29e.md)。

| 算子 | 当前 `Auto` | 显式/研究 candidate | 最新决策 |
|---|---|---|---|
| Dense GEMM | `cuda_naive` | optimized id 1–7，v3 为大 shape 最快自研路径 | 256³ 方差超限，不晋级 |
| Top-K Gate | `cuda_naive` | optimized id 1–4 | v4 exact-E/连续 bucket 均无晋级区间 |
| Histogram | `cuda_candidate` | small/sparse/block-private shape paths | **已晋级** SM86 `Auto` |
| Exclusive Scan | `cuda_naive` | standalone C2/S1/S2 已删除 | 受 WDDM tail/CV 限制 |
| Token Permute | `cuda_naive` | optimized id 1–5 与显式 v2/v3 shape dispatcher | v3 为 0.9938x；CV 超限，证据不足且不晋级 |
| Grouped GEMM | `cuda_naive` | 显式 SM86 v1/v2/v3 candidate | v3 对 CUTLASS 为 0.9141x；证据不足且不晋级 |
| Unpermute | `cuda_naive` | benchmark-only warp/CTA hybrid | 正式拒绝，不进入 public dispatch |

Histogram→Scan 融合：`cuda_fused_histogram_scan`（F2，SM86；`R<=4096` 单 CTA，
更大 R 回退）已绑定公共 fused API 且当前由该 primitive 的 `Auto` 选择；已有运行未通过
稳定性、fallback 与 L3 门禁，必须在独占 CUDA 环境复测或撤回默认选择。性能边界见
[F2 performance report](reports/rtx3080-histogram-scan-fused-sm86-v2.md)。

| 算子 | 定位 | 优化文档 | 性能记录 |
|---|---|---|---|
| Dense GEMM | 主算子：路由投影/通用矩阵乘 | [optimization-plan](dense_gemm/optimization-plan.md) | [performance-record](dense_gemm/performance-record.md) |
| Top-K Gate | 主算子：Top-2 选择与权重归一化 | [optimization-plan](topk_gate/optimization-plan.md) | [performance-record](topk_gate/performance-record.md) |
| Grouped GEMM | 主算子：ragged expert GEMM | [optimization-plan](grouped_gemm/optimization-plan.md) / [research-notes](grouped_gemm/research-notes.md) | [performance-record](grouped_gemm/performance-record.md) |
| Histogram | 配套算子：expert route 计数 | [optimization-plan](histogram/optimization-plan.md) | [performance-record](histogram/performance-record.md) |
| Exclusive Scan | 配套算子：计数到 expert offsets | [optimization-plan](scan/optimization-plan.md) | [performance-record](scan/performance-record.md) |
| Token Permute | 配套算子：按 expert 重排 token | [optimization-plan](permute/optimization-plan.md) | [performance-record](permute/performance-record.md) |
| Unpermute | 配套算子：还原并按权重合并 | [optimization-plan](unpermute/optimization-plan.md) | [performance-record](unpermute/performance-record.md) |

## 记录规则

- 每次实验先在对应的 `optimization-plan.md` 中登记假设、版本、输入 shape 和验收指标，再运行 benchmark/profile。
- 实际数字只写入对应的 `performance-record.md`；必须同时记录 baseline、candidate、测量层级、cache 模式、seed、GPU、软件版本和 Git revision。
- L1/L2/L3/L4 的含义遵循 [benchmark-architecture.md](benchmark-architecture.md)，不要把 Kernel Body 结果和完整 operator 或 chain 结果混在一起。
- Git 只保留报告、紧凑 CSV/comparison、归一化 profiler 指标、v2 manifest 与 `SHA256SUMS`；raw JSONL、完整 aggregate、run manifest、NCU/NSYS/SQLite 进入 manifest 指向的不可覆盖 `<run_id>-raw.zip` Release 资产。历史缺失文件必须标记 `storage: unavailable`，不得静默省略。
- 失败优化也要保留：写清楚假设、改动、反例 shape、指标变化和回退原因，不能只保留成功结果。
- 未实际运行的数字统一写 `[待实测]`，不能用论文、第三方仓库或理论峰值代替本项目实测。

## 推荐的单次迭代闭环

```text
提出假设 → 固定 workload/协议 → correctness → benchmark → ncu/nsys 诊断
→ 记录指标与失败样例 → 与 promotion baseline 严格配对比较 → 决定保留/回退
```

项目级 benchmark 公平性和证据边界见 [benchmark-architecture.md](benchmark-architecture.md)；当前实现/计划边界见 [implementation-status.md](implementation-status.md) 和 [development-roadmap.md](development-roadmap.md)。
