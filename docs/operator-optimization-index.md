# 算子优化文档索引

本目录按 RaggedRoute 的七个语义算子分别保存优化方案、实验决策和实际性能记录；benchmark registry 另有一个 `histogram_exclusive_scan` 融合 adapter，因此当前是七阶段数据流、八个 adapter。

当前统一实测报告：[2026-08-29 / 9732a03 compact evidence](reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)；Grouped GEMM 的更新 follow-up 见 [2026-08-30 / c2205ed v5/v6 evidence](reports/compact/20260830-c2205ed-grouped-v6/REPORT.md)。后者不改写其他六算子矩阵。面试唯一入口见 [docs/interview](interview/README.md)。

| 算子 | 当前 `Auto` | 显式/研究 candidate | 最新决策 |
|---|---|---|---|
| Dense GEMM | `cuda_naive` | v3 64x32 `cp.async` | 对 cuBLAS envelope `0.8716x`；256³ 局部 `1.0095x`，不晋级 |
| Top-K Gate | `cuda_naive` | v4 two-reduction | 对 strict naive `1.0972x`，E64/T4096 `1.6934x`；最大回退 10.88%，不改 Auto |
| Histogram | shape-dispatched v1 | v2 E=1 fast path + existing dispatcher | 对 v1/CUB/naive envelope `1.1016x`，显式 research winner；本轮不改 Auto |
| Exclusive Scan | `cuda_naive` | 无 retained custom candidate | CUB Block/Warp 仅 `1.0249x/1.0290x`，tiny launch-bound |
| Token Permute | `cuda_naive` | v2 full-from-ids | 对 adapted vLLM `1.5671x`、5/5；pure path 仅 `0.9841x`，不改 Auto |
| Grouped GEMM | `cuda_naive` | v6 hybrid `32x128` balanced / v5 fallback | 对 CUTLASS/cuBLAS envelope `0.9946x`；uniform T512 对 CUTLASS `1.2257x`、single-hot 对最快 library `1.7762x`，但 T2048 `0.7534x`，不晋级 |
| Unpermute | `cuda_naive` | `cuda_warp_token_vec4` | envelope `1.0154x`、12/32；窄 N 局部 `1.2892x`，不晋级 |

作品集 CV ceiling 为 0.50：超过 0.10 仍披露为 WDDM 稳定性风险，但不再单独自动判 `insufficient_evidence`；完整矩阵仍需 ratio-of-sums、geomean、coverage、最大回退和跨进程方向共同成立。历史 0.10 policy decision 不回写。

六阶段/五阶段 chain：`cuda_postlogit_graph_fixed_v1` 与 `cuda_postroute_graph_fixed_v1` 只在固定 shape、setup 完成的 host replay 边界获得显式晋级；不是 kernel speedup，不进入七算子排名或 `Auto`。完整口径见 [v4 report](reports/rtx3080-sm86-v4-graph-promotion.md)。

Histogram→Scan 融合：`cuda_fused_histogram_scan`（F2，SM86；`R<=4096` 单 CTA，更大 R 回退）是独立跨算子 primitive。当前由该 primitive 的 `Auto` 选择，但历史 fallback/L3 门禁仍需独占环境复测；不得混入 standalone Histogram/Scan 排名。性能边界见 [F2 performance report](reports/rtx3080-histogram-scan-fused-sm86-v2.md)。

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
