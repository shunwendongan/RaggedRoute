# RTX 3080 / SM86 v4：CUDA Graph fixed replay 晋级记录

## 结论

`cuda_postlogit_graph_fixed_v1` 与 `cuda_postroute_graph_fixed_v1` 晋级为**显式、固定 shape 的 CUDA Graph replay** 实现。两者都不修改公开 API，且 `KernelFamily::kAuto` 保持不变。

这不是单 kernel 或 GPU-event speedup：下表是包含最终同步的 QPC host time-to-solution。capture、instantiate 与 upload 在 fixed replay 的 setup 阶段完成，因此不计入 steady replay；它们和回本次数单独报告。

| Scope | Baseline p50 | Graph p50 | p50 speedup | Graph p95 speedup | Setup / break-even |
|---|---:|---:|---:|---:|---:|
| postlogit, Top-2 | 84.750 us | 52.300 us | 1.6205x | 1.9864x | 262.7 us / 9 replays |
| postroute, Top-2 | 59.350 us | 47.800 us | 1.2416x | 1.4468x | 253.9 us / 22 replays |
| postroute, Top-4 | 86.400 us | 71.300 us | 1.2118x | 1.7855x | 242.9 us / 17 replays |
| postroute, Top-8 | 101.100 us | 81.000 us | 1.2481x | 1.3410x | 274.0 us / 14 replays |

## 晋级边界与例外

- 原始正式采集为 RTX 3080 / SM86，CUDA 13.3.73，driver 616.56，NCU 2026.2.1，NSYS 2026.1.3；Release binary SHA 为 `b3429c2abdfe`，所有记录 `build_git_dirty=false`。
- 每个配对有五个独立进程、20 warmups、30 samples，CPU oracle、repository checks、57 Python tests、8 CTest tests，以及 memcheck/initcheck/racecheck/synccheck 均通过。
- 新增 [fixed-replay policy](../../configs/policies/cuda_graph_wddm_fixed_replay_promotion.json) 显式选择 `host_time_to_solution_timing`。它保留 correctness、同 GPU/同 case/同 SHA/同 run、五进程、ratio-of-sums、coverage、p50/p95 回退与 64 MiB workspace 上限，只按用户授权豁免 Windows WDDM 的 CV 门禁。
- 该豁免只对 `fixed_capture_upload_replay`、`graph_setup_excluded=true`、`mixed_shape_cache_trace=false` 的两个 named variants 有效。param-update、exec-update、mixed-shape LRU cache miss 仍是 research，不可引用上表作为其 speedup。
- postroute fixed variant 是一个已定义的 Graph topology，内部包含 gather route-stage；它的结果只能表述为该完整显式 topology 的 host replay 加速，不能归因为“Graph 单独消除了 gather 的 GPU 成本”。

在这份有边界的 policy 下，postlogit 的 ratio-of-sums 是 `1.6205x`，postroute Top-2/4/8 的 ratio-of-sums 是 `1.2336x`，两个 decision 均为 `promote`。它们的原始严格 policy decision 仍为 `insufficient_evidence`，原因仅为 WDDM 多峰导致的 CV；原始 decision 未被改写。

## 未晋级的 v4 方向

| Candidate | Result | 决定 |
|---|---:|---|
| Grouped descriptor queue-1024 vs v2 | 0.9225x ratio-of-sums | 拒绝该方向 |
| shared-rank route-prep t1024 vs v3 | 1.0439x，但仅 40.3% shape 获益 | 不晋级 |
| Top-4 token-owned vs shared-rank t256 | 1.0170x，CV/覆盖不足 | 不晋级 |
| postroute gather fusion vs current | 0.8191x，且 workspace +4,195,072 B | 拒绝该方向 |

NSYS 诊断显示，代表性 Top-4 postroute 链 70.0%（uniform）/71.3%（Zipf-1.4）的 GPU time 仍在 Grouped GEMM v2。NCU detailed 还显示 descriptor mainloop 使用 84 registers/thread、35.79% achieved occupancy，且有 long-scoreboard、barrier 与 mio-throttle stalls；因此下一轮不应再把“更多 descriptor-prepass threads”当作主假设。上述 profiler 时间不参与晋级数值。

## 可审计制品

- 原始 v4 compact evidence（含被保留的严格 policy `insufficient_evidence` decisions、报告、SVG 与 SHA-256 manifest）：[raw final bundle](compact/20260828-sm86-v4-raw-final/FINAL_REPORT.md)。
- 用户授权 CV 例外后的两个 machine-readable decisions：[postlogit](compact/20260828-sm86-v4-graph-promotion/graph-postlogit-fixed-wddm-exception-decision.json) 与 [postroute](compact/20260828-sm86-v4-graph-promotion/graph-postroute-fixed-wddm-exception-decision.json)。
- 原始 JSONL、`.ncu-rep` 和 `.nsys-rep` 留在 ignored `out/research/six-ops-v4/`，不提交到 Git。
