# RTX 3080 / SM86 六阶段 CUDA v3 统一复测

## 结论

固定代码 `657d29e54ce94097be00b0ac57aea4f5e1f4f143` 上，Grouped GEMM v3、Token Permute v3、六阶段 L3 和 synthetic route-trace 的正式三态结论均为 `insufficient_evidence`。原因是 Windows WDDM 长尾使所有正式 case 的最大进程内 CV 超过 0.10；这不是 correctness 失败，也不能手工改写成 `reject`。

不过，不依赖 CV 门禁的 aggregate 方向同样不支持晋级：Grouped v3、Permute v3 和 L3 的 ratio-of-sums 分别为 `0.9141x`、`0.9938x` 和 `0.9743x`。因此 `KernelFamily::kAuto` 保持不变，两个 v3 仅作为显式 research path 保留。

| Scope | Baseline | Candidate | Shapes | Ratio-of-sums | Geomean | Win coverage | Decision |
|---|---|---|---:|---:|---:|---:|---|
| Grouped GEMM | CUTLASS Grouped | `cuda_grouped_sm86_fp32_v3` | 10 | 0.9141x | 1.0189x | 40.0% | `insufficient_evidence` |
| Token Permute | `cuda_token_owned_top2` | `cuda_candidate_v3` | 28 | 0.9938x | 0.9980x | 35.7% | `insufficient_evidence` |
| Permute full-from-ids | vLLM | `cuda_candidate_v3_from_ids` | 5 | 1.5452x | 1.5753x | 100.0% | `insufficient_evidence` |
| Six-stage L3 | integrated v2 | research v3 | 7 | 0.9743x | 0.9755x | 28.6% | `insufficient_evidence` |
| Route trace fixture | integrated v2 | research v3 | 3 | 0.9900x | 0.9899x | 33.3% | `insufficient_evidence` |

## Correctness 与环境

- GPU：NVIDIA GeForce RTX 3080，UUID `GPU-7c5e95c0-e5a4-15d8-24a0-c8c8b58d6f39`，compute capability 8.6，68 SM。
- CUDA compiler 13.3.73，driver 591.86，NCU 2026.2.1，NSYS 2026.1.3。
- Release binary 记录 `build_git_dirty=false`、`build_git_sha=657d29e54ce9`；所有正式 record 的 validation 均通过。
- Repository checks、57 个 Python tests、CTest 8/8，以及 memcheck/initcheck/racecheck/synccheck 全部通过。
- 正式窗口开始时 GPU 34°C、约 27.5 W；没有独立 compute workload，但 DWM、Lively Wallpaper、浏览器和 Codex 等 WDDM 桌面进程存在。不锁频、不改功耗。

## NSYS：端到端热点

代表 workload 为 `T=2048,E=64,K=N=128,top_k=2,Zipf s=1.4`，20 次 warmup 后采集一次六阶段链。NSYS 只用于诊断，以下时间不参与 promotion：

| Chain | Grouped kernel | Kernel median | GPU kernel time share |
|---|---|---:|---:|
| integrated v2 | `grouped_gemm_register16x32_sm86_v2_kernel` | 43.871 us | 70.9% |
| research v3 | `grouped_gemm_register16x64_sm86_v3_kernel` | 51.839 us | 74.5% |

Top-K、fused Histogram→Scan、Permute 和 Unpermute 的 median 基本不变。L3 回退因此主要归因于 Grouped v3；该 L3 shape 的 Permute selector 仍走 tile4，不应把回退归因于 tile2。

## NCU：机制解释

### Grouped GEMM v2 → v3

| Metric | v2 | v3 |
|---|---:|---:|
| `launch__registers_per_thread` | 86 | 96 |
| `launch__shared_mem_per_block` | 7,952 B | 12,048 B |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 36.24% | 30.29% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.24% | 25.38% |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 49.95% | 37.63% |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 93,328 | 81,200 |
| local load/store instructions | 0 / 0 | 0 / 0 |

v3 的 `16x64x16` tile 确实减少约 13.0% global-load requests，也没有 spill；但更大的 register live range 和 shared-memory footprint 降低了活跃 warp、issue activity 与 SM throughput。这个结果符合 Ampere 上“更宽 tile 与双缓冲并不自动更快”的反例。

### Permute tile4 → tile2

代表 workload 为 `T=4096,E=64,K=1024,top_k=2,single_hot`：

| Metric | tile4 v2 | tile2 v3 |
|---|---:|---:|
| `launch__block_size` | 128 | 64 |
| `launch__waves_per_multiprocessor` | 1.255 | 1.882 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 77.79% | 58.20% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 84.55% | 85.11% |
| `launch__registers_per_thread` | 34 | 34 |
| diagnostic `gpu__time_duration.sum` | 80.608 us | 79.744 us |

两版都已经接近相同 DRAM throughput，tile2 只增加 CTA/wave 数，没有减少 payload bytes 或建立新的数据复用，因此无法产生全矩阵 1.03x 收益。basic 已能回答这次 geometry 假设，没有升级 full/source。

`global_load_sectors`、L1/L2 hit rate、eligible warps 和 stall breakdown 在 fresh basic 中为 `not_collected`；未把它们写成 0。Grouped 的 detailed collection 提供了 global-load request 与 stall sample，但 profiler duration 仍只作诊断。

## Trace 与证据边界

版本化 `.rrtrace`、frame working set、SHA-256 provenance 和三态 evaluator 已实现。仓库内 `synthetic_fixture` 只验证 parser/runner/evaluator，policy 明确要求真实证据的 `workload_source` 为 `captured` 或 `production`，所以真实 trace 决策必须保持 `insufficient_evidence`。

可提交 compact evidence 位于 [compact bundle](compact/20260827-657d29e-six-ops-v3/REPORT.md)，包含五份 decision JSON、promotion CSV、shape heatmap、manifest 与 `SHA256SUMS`。它不是完整 `evidence_bundle.v2`，不会冒充 Release asset。原始 JSONL、`.ncu-rep`、`.nsys-rep` 留在 ignored `out/`；本地 `compact-evidence-final.zip` 与 Git 中的规范化内容逐文件一致，SHA-256 为 `b9713922220b487191ea2300e46197421b1b824eabf3801ffcb4043415e648fe`。
