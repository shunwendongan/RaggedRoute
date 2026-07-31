# 算子优化文档索引

本目录按 RaggedRoute 的七个算子分别保存优化方案、实验决策和实际性能记录。七个算子与项目当前 API、benchmark adapter 和 `src/` 目录一一对应：

当前统一实测报告：[RTX 3080 七算子 naive benchmark 与 Nsight 分析（a9489ab）](reports/rtx3080-naive-profile-a9489ab.md)。

| 算子 | 定位 | 优化文档 | 性能记录 |
|---|---|---|---|
| Dense GEMM | 主算子：路由投影/通用矩阵乘 | [optimization-plan](dense_gemm/optimization-plan.md) | [performance-record](dense_gemm/performance-record.md) |
| Top-K Gate | 主算子：Top-2 选择与权重归一化 | [optimization-plan](topk_gate/optimization-plan.md) | [performance-record](topk_gate/performance-record.md) |
| Grouped GEMM | 主算子：ragged expert GEMM | [optimization-plan](grouped_gemm/optimization-plan.md) | [performance-record](grouped_gemm/performance-record.md) |
| Histogram | 配套算子：expert route 计数 | [optimization-plan](histogram/optimization-plan.md) | [performance-record](histogram/performance-record.md) |
| Exclusive Scan | 配套算子：计数到 expert offsets | [optimization-plan](scan/optimization-plan.md) | [performance-record](scan/performance-record.md) |
| Token Permute | 配套算子：按 expert 重排 token | [optimization-plan](permute/optimization-plan.md) | [performance-record](permute/performance-record.md) |
| Unpermute | 配套算子：还原并按权重合并 | [optimization-plan](unpermute/optimization-plan.md) | [performance-record](unpermute/performance-record.md) |

## 记录规则

- 每次实验先在对应的 `optimization-plan.md` 中登记假设、版本、输入 shape 和验收指标，再运行 benchmark/profile。
- 实际数字只写入对应的 `performance-record.md`；必须同时记录 baseline、candidate、测量层级、cache 模式、seed、GPU、软件版本和 Git revision。
- L1/L2/L3/L4 的含义遵循 [benchmark-architecture.md](benchmark-architecture.md)，不要把 Kernel Body 结果和完整 operator 或 chain 结果混在一起。
- Nsight Compute/Nsight Systems 原始输出、JSONL、CSV 和图表放在 `docs/reports/artifacts/<run_id>/` 或对应算子目录下的 `artifacts/<run_id>/`，正文只保存可审计摘要和链接。
- 失败优化也要保留：写清楚假设、改动、反例 shape、指标变化和回退原因，不能只保留成功结果。
- 未实际运行的数字统一写 `[待实测]`，不能用论文、第三方仓库或理论峰值代替本项目实测。

## 推荐的单次迭代闭环

```text
提出假设 → 固定 workload/协议 → correctness → benchmark → ncu/nsys 诊断
→ 记录指标与失败样例 → 与 promotion baseline 严格配对比较 → 决定保留/回退
```

项目级 benchmark 公平性和证据边界见 [benchmark-architecture.md](benchmark-architecture.md)；当前实现/计划边界见 [implementation-status.md](implementation-status.md) 和 [development-roadmap.md](development-roadmap.md)。
