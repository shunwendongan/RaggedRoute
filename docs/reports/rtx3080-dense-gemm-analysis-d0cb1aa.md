# RTX 3080 Dense GEMM：naive 与 cuBLAS/cuBLASLt 性能分析

日期：2026-08-01<br>
源码基线：远端 `main` `884d119d6ac393c9b80708547da6f89ae204a623` 的后续分析提交 `d0cb1aa40494e6c5ef6c3f924523c9e1d20f1832`<br>
分支：`codex/dense-gemm-analysis`

## 1. 结论先行

- `dense_gemm` 的 `library_baseline` 实际是 cuBLAS/cuBLASLt；CUB 只用于 Histogram/Scan，不能把 CUB 作为 dense GEMM 基线。
- 在相同的 Release、256×256×256、严格 FP32、row-major、warm-cache 和 L2 operator 边界下，`cuda_naive` 的中位数为 **27.546 μs**，cuBLASLt 为 **10.752 μs**，cuBLAS SGEMM 为 **10.803 μs**。因此库实现约快 **2.56×/2.55×**，naive 相对 cuBLASLt 的 speedup 为 **0.390×**。
- cuBLASLt 与 cuBLAS 在该形状发射同一个 `ampere_sgemm_32x32_sliced1x4_nn`，Release p50 几乎相同；差异主要来自同一个优化过的 SIMT GEMM kernel，而不是 API 名称本身。
- naive 的主瓶颈是标量 one-thread-per-output 路径产生的高访存指令量、tile 复用缺失和 L1TEX 依赖等待，不是 DRAM 带宽饱和，也不是寄存器 spill 或 Tensor Core 使用不足。
- 当前桌面/WDDM 环境使 cuBLAS/cuBLASLt 的 p95/CV 抖动较大，因此报告把 p50 作为中心估计；不把 profiler duration 或不稳定的 p95 比值当作 promotion 结论。

## 2. 测量合同与门禁

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080，compute capability 8.6，68 SM，10 GiB |
| Driver / CUDA | 591.86 / 13.3.73 |
| Nsight | Compute 2026.2.1，Systems 2026.1.3 |
| Build | Windows Release，`sm_86`，`-O3`，`-lineinfo`，MSVC 19.44 |
| Math | FP32 input/output，FP32 accumulation，`alpha=1`，`beta=0`，无 fast-math/TF32/Tensor Core 合同 |
| Shape/layout | `M=N=K=256`，A/B/C row-major |
| Benchmark | warmup 20，30 samples/process，10 kernel repeats/sample，3 independent processes，seed `20260729` |
| Boundary | L2 `operator_steady`；input generation、CPU reference、H2D、workspace allocation excluded |
| Correctness | CTest 8/8；三种 variant 的 256³ profile invocation 均 `validation_ok=true` |
| Dependency | cuBLAS/cuBLASLt enabled；CUTLASS/CCCL fetch disabled because this experiment only needs cuBLAS and local CUDA CCCL headers |

执行过的关键命令：

```powershell
ctest --preset test-rtx3080-sm86-release --output-on-failure
python scripts\run_benchmarks.py `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\benchmark_dense_gemm_library_release.json `
  --output profile\dense-gemm-d0cb1aa-20260801-1350-rerun\benchmark\paired.jsonl `
  --run-id dense-gemm-d0cb1aa-20260801-1350-rerun
```

正式 Release 原始制品位于 `profile/dense-gemm-d0cb1aa-20260801-1350-rerun/benchmark/`，该目录被 Git 忽略；其中的 manifest 记录了 GPU UUID、P-state、时钟和所有实际命令。

## 3. Release A/B 结果

以下数值来自无 profiler 的第二次完整运行。第一次运行也完成了 9 条记录和 pairing，但 cuBLAS/cuBLASLt 的 CV 更高，因此按预定规则保留为初始证据、使用第二次 run 作为最终中心估计。

| Variant | API/math path | p50 (μs) | p95 (μs) | CV | TFLOP/s @ p50 | Effective GB/s | Workspace |
|---|---|---:|---:|---:|---:|---:|---:|
| `cublaslt` | `cublasLtMatmul`, `CUBLAS_COMPUTE_32F_PEDANTIC` | 10.752 | 15.519 | 0.152 | 3.121 | 73.14 | 0 B |
| `cublas` | `cublasSgemm`, `CUBLAS_PEDANTIC_MATH` | 10.803 | 16.031 | 0.195 | 3.106 | 72.80 | 0 B |
| `cuda_naive` | scalar CUDA core, 256 threads/block | 27.546 | 27.863 | 0.0097 | 1.223 | 28.55 | 0 B |

严格 pairing comparison：

| Baseline | Compared variant | p50 ratio / speedup | p95 ratio |
|---|---|---:|---:|
| cuBLASLt | cuBLAS | 0.995× | 1.033× |
| cuBLASLt | `cuda_naive` | 0.390×（库约快 2.56×） | 1.795× |

所有 9 条记录的 `build_git_dirty=false`、build SHA、GPU UUID、shape、seed、math mode、cache mode、level、repeats 和 excluded steps 均匹配；每个 variant 有 3 个 process record 和 90 个 raw samples。由于 cuBLAS 两次运行仍有 CV>0.10，p95 仅作波动信息，不作为稳定性或晋升判据。

历史 naive 数据用于漂移检查，不与本次样本池化：

| Tested commit | Dense L2 p50 (μs) | 与本次 27.546 μs 的差异 |
|---|---:|---:|
| `e37c132` | 27.750 | -0.7% |
| `4114caa` | 27.853 | -1.1% |
| `a9489ab` | 26.931 | +2.3% |
| `d0cb1aa`（本次） | 27.546 | — |

这说明 naive 的中心延迟与之前 Release 记录一致；库 smoke 的 17×19×13 Debug 数字不用于 256³ 性能结论。

## 4. NSYS 系统证据

本次对三个 variant 分别执行 CUDA/NVTX trace，CPU sampling 和 context-switch sampling 均关闭。每条 trace 使用 20 次 warmup 后的相同 256³ 路径；NSYS 的时间只用于识别 kernel，不用于 Release A/B。

| Variant | Kernel | Instances | Total GPU time | Avg duration |
|---|---|---:|---:|---:|
| cuBLASLt | `ampere_sgemm_32x32_sliced1x4_nn` | 22 | 174,461 ns | 7,930 ns |
| cuBLAS | `ampere_sgemm_32x32_sliced1x4_nn` | 22 | 174,268 ns | 7,921 ns |
| naive | `raggedroute::ops::<unnamed>::dense_gemm_naive_kernel(...)` | 22 | 472,152 ns | 21,462 ns |

已有的 `profile/integration-baseline-20260731-1710/nsys-chain` 是 `chain_from_logits` 路径，未执行 dense GEMM，不能用来推断 dense 热点；仓库中更早的 `a9489ab` 7-op `chain_from_tokens` 报告包含过 dense，但 shape/commit/链路不同。本次单算子 NSYS 才是三种实现的直接 kernel 对照。

## 5. NCU 证据

### 5.1 Kernel 形状与资源

| Variant | Kernel launch | Grid/block | Waves/SM | Registers/thread | Static shared/block | Achieved occupancy |
|---|---|---:|---:|---:|---:|---:|
| cuBLASLt | `ampere_sgemm_32x32_sliced1x4_nn` | 64 / 128 | 0.31 | 86 | 32.77 KiB | 8.33% basic |
| cuBLAS | 同上 | 64 / 128 | 0.31 | 86 | 32.77 KiB | 8.33% basic |
| naive | `dense_gemm_naive_kernel` | 256 / 256 | 0.63 | 40 | 1.02 KiB driver allocation | 50.61% full |

库 kernel 虽然 grid 只有 64 blocks 且理论 occupancy 25%，仍明显快于 naive；因此“occupancy 越高越快”不成立。库 kernel 以更少的 global traffic 和更高的 tile reuse 完成相同 GEMM 工作。

### 5.2 naive full/source 关键指标

来自 `ncu-naive-full` 的真实 metric/value：

- `gpu__time_duration.sum`：27.072 μs（诊断上下文，不是 Release 分数）。
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` 与 `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`：66.55%；`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`：3.96%。这排除了“DRAM 已饱和”作为第一根因。
- `l1tex__throughput.avg.pct_of_peak_sustained_active`：83.72%；`l1tex__t_sector_hit_rate.pct`：66.64%；`lts__t_sector_hit_rate.pct`：99.40%。数据主要在 L2/L1 路径中反复访问，非大规模 DRAM streaming。
- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`：1,048,576；`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`：2,621,440；store 为 2,048 requests/8,192 sectors。
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio`：26.4/32 bytes，提示 load sector 利用率不满；store 为 32/32。
- `smsp__warps_eligible.avg.per_cycle_active`：0.712；`sm__issue_active.avg.pct_of_peak_sustained_elapsed`：28.28%。大量 active warps 并未处于可发射状态。
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`：7.208；PC sampling 的 `smsp__pcsamp_warps_issue_stalled_long_scoreboard`：681，wait 为 103，short scoreboard 为 3，MIO throttle 为 1。
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`：0；`derived__local_spilling_requests` 和 `derived__shared_spilling_requests` 均为 0。
- Roofline overview 仅达到本机 FP32 peak 约 4%，与 scalar CUDA core 路径一致；不能通过把它描述成 Tensor Core baseline 来解释或掩盖差距。

Source page 已收集 SASS/source-counter 数据，但该 Windows/Nsight 组合没有稳定地输出 `baseline.cu:<line>` 文件行映射；因此本报告只把源码中 `for (inner)` 的 A/B global load 表达式作为审查对象，不伪造 source-PC 行号。原始 source CSV 保存在 `profile/.../ncu-naive-full/analysis/ncu_full_source.csv`。

### 5.3 library detailed 关键指标

cuBLASLt 与 cuBLAS 的 detailed replay 均确认：

- 两者仍是同一 `ampere_sgemm_32x32_sliced1x4_nn`、64×128、86 registers/thread、0.31 waves/SM；Tensor pipe active 为 0，说明 pedantic FP32 选择的是 SIMT kernel。
- cuBLASLt：`sm__throughput` 29.31%、DRAM 5.81%、L2 hit 96.02%、无 local spill；cuBLAS：17.67%、5.00%、97.87%、无 local spill。
- detailed duration 为 replay/serialization 后的 14–15 μs，仅用于诊断；正式 p50 仍取 unprofiled Release 表。

## 6. 瓶颈判断

### P0：标量 per-output 计算缺少 tile reuse，并暴露 L1TEX latency

证据链：naive Release p50 比 cuBLASLt 高 2.56×；NCU 同时显示 global load 1,048,576 requests、L1 load sector 2,621,440、sector 利用率 26.4/32，且 long-scoreboard 7.208 cycles/issue-active、eligible warps 0.712。DRAM 仅 3.96%，local spill 为 0。最合理的解释是每个输出线程独立重复产生 A/B load，依赖等待和 LSU 指令量主导，而不是显存容量或带宽不够。

### P1：256 blocks 在 68-SM/资源模型下只有 0.63 waves/SM，存在尾波和 active-cycle imbalance

证据链：`launch__grid_size=256`、`launch__waves_per_multiprocessor=0.63`、achieved occupancy 50.61% 对比理论 100%；Workload Distribution 的最小 SM active cycle 约比平均值低 20%。这会放大短 GEMM 的尾部时间，但不是唯一根因，因为 cuBLAS kernel 只有 64 blocks、0.31 waves/SM 仍然更快。它应作为 tile/block geometry 的次级实验变量，而不是单独归因。

### P2：索引/边界算术和 scalar memory instruction mix 有可优化空间，但当前没有独立 source-line 计数证明其占主导

源码使用 linear index 后执行 row/column 除法与取模，再进入 ascending-K scalar loop；SASS source page 可见对应的 integer address-generation 指令。由于当前 source counter 没有可靠文件行映射，不能把它单独宣称为第一瓶颈，应在 tiled/二维映射候选中用 A/B benchmark 验证。

## 7. 排序后的优化实验（本次不实现）

1. **Strict-FP32 shared-memory tiled GEMM（最高优先级）**
   - 机制：二维 block tile，A/B tile staging，register accumulator；先保持 ascending-K FP32 累加和公共 API不变。
   - 预期信号：global load requests/sectors 显著下降、long-scoreboard 和 LSU pressure 下降；不能只看 occupancy 是否上升。
   - 风险：shared-memory footprint、bank conflict、register live range 和边界 tile；每一步都要验证无 spill。
   - 验证：CTest + dense oracle；同一 Release suite 与 cuBLASLt 配对，按 p50/p95/CV 和 shape regression 决定保留/拒绝。

2. **二维 launch/index mapping（第二优先级）**
   - 机制：用二维 grid/block 直接得到 row/column，去掉每个输出的 linear-index `/`、`%`；单独于 tile reuse 实验，避免机制混淆。
   - 预期信号：integer/ALU 指令量和 issue gap 下降；如果 global load/stall 不变且 end-to-end 无改善，明确拒绝。
   - 风险：非整除 M/N 的边界覆盖、grid stride 和结果顺序；必须保留现有 FP32 容差。

3. **对齐 vector load 或 Ampere `cp.async` staging（第三优先级）**
   - 只有在 tiled baseline 已正确且 NCU 仍显示 load sector 利用率/long scoreboard 为主瓶颈时才测试。
   - 先单 stage、再双 stage；确认 16-byte alignment、尾部处理、register/shared-memory occupancy 后才考虑 `cp.async`。不使用 Hopper TMA/WGMMA/TMEM，也不改变为 TF32/Tensor Core 数值合同。

## 8. 限制与制品

- cuBLAS/cuBLASLt 两次正式 run 的 CV 仍受 Windows 桌面 GPU 工作影响；因此库 p95 不能视为稳定 tail latency。没有锁频，也没有杀掉桌面进程。
- NCU/NSYS duration 受 replay、cache control 和串行化影响，不能替代 Release benchmark。
- 本次只有 256³ 正式形状；17×19×13 仅是 Debug smoke，结论不外推到非方阵、生产 trace 或其他 GPU。
- Source-PC 文件行映射未收集到，已标记为 unavailable；没有用猜测行号填充报告。
- 运行制品（包括 `.ncu-rep`、`.nsys-rep`、`.sqlite`、raw/aggregate JSON/CSV）保存在被忽略的 `profile/dense-gemm-d0cb1aa-20260801-1350-rerun/`；正式配对摘要为 `benchmark/paired.comparison.json`。
