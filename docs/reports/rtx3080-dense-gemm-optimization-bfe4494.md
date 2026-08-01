# RTX 3080 Dense GEMM：四轮 strict-FP32 优化实验

日期：2026-08-01
分支：`codex/dense-gemm-optimization`
最终实现/Release 基线：`bfe449418e8fe6dc05dd1c0eb27b8b0eb83bb2c0`

> 结论：四个候选均已作为显式、可调用的 research CUDA kernel 接入。`cuda_tiled_vector` 的中心 p50 最好（相对 naive 1.231x），但它的独立完整复跑 CV 为 0.188，超过 0.10 门槛；因此它是 **variance-limited research variant**，不是暂定默认。没有候选同时满足全部晋升条件，故 `KernelFamily::kCudaOptimized, id=0` 保持 fail-closed，`kAuto` 继续选 `cuda_naive`。

## 1. 固定合同与验证边界

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080，sm_86，68 SM，10 GiB |
| Driver / toolchain | 591.86 / CUDA-NVCC 13.3.73 / Nsight Compute 2026.2.1 / Nsight Systems 2026.1.3 |
| Build | Windows Release，`-O3`、`-lineinfo`、`sm_86`，不锁频 |
| 数值合同 | FP32 row-major，ascending-K FP32 accumulation，`alpha=1`、`beta=0`；不使用 TF32/Tensor Core |
| 正式形状与边界 | `M=N=K=256`，L2 operator steady，warm cache |
| Release 协议 | 3 独立进程 × 30 samples × 每 sample 10 launches，20 warmup，seed `20260729` |
| 排除项 | input generation、CPU oracle、H2D copy、workspace allocation |
| 正确性 | 每个 clean experiment commit 均执行 Release CTest 8/8、CPU oracle、Compute Sanitizer 的 memcheck/initcheck/racecheck/synccheck；所有 benchmark record 为 `validation.ok=true` |

每个 variant 的最终横向样本均为 3 process records、90 raw samples，strict pairing 已核验 GPU UUID、SHA、shape、seed、math/cache mode、level、repeats 和 excluded steps。所有 custom kernel 的外部 workspace 均为 0 B。正式数值只来自无 profiler 的 Release suite；下方的 NSYS/NCU duration 只用于诊断。

E4 的可复现关键命令（完整 benchmark 子进程命令在其 JSONL manifest）：

```powershell
ctest --preset test-rtx3080-sm86-release --output-on-failure
python scripts\run_sanitizers.py --build-dir out\build\rtx3080-sm86-release --output-dir profile\dense-gemm-bfe4494-exp4\sanitizer
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\benchmark_dense_gemm_optimization_exp4_release.json --output profile\dense-gemm-bfe4494-exp4\benchmark\release.jsonl --run-id dense-gemm-bfe4494-exp4-release
python scripts\profile_benchmarks.py compute --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\profile_dense_gemm_optimization_exp4.json --run-dir profile\dense-gemm-bfe4494-exp4\profile --set basic
python scripts\profile_benchmarks.py compute --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\profile_dense_gemm_optimization_exp4.json --run-dir profile\dense-gemm-bfe4494-exp4\profile --set detailed
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output=profile\dense-gemm-bfe4494-exp4\profile\reports\system.exp4 out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --operator dense_gemm --variant cuda_combined --level l2 --profile-once --warmup 20 --seed 20260729 --param M=256 --param N=256 --param K=256
python scripts\profile_benchmarks.py analyze --run-dir profile\dense-gemm-bfe4494-exp4\profile
```

## 2. 实现与 dispatch 状态

| ID | Benchmark variant | 单变量机制 | 当前状态 |
|---:|---|---|---|
| 1 | `cuda_tiled_scalar` | 16×16 shared A/B tile、linear CTA tile mapping、scalar staging、每线程 FP32 accumulator | 显式 research variant |
| 2 | `cuda_2d_mapping` | `(32,8)` block 与二维 grid，直接计算 row/column，无 shared memory | 显式 research variant |
| 3 | `cuda_tiled_vector` | ID 1 加 aligned `float4` staging；仅 A/B/C 16-byte 对齐且 M/N/K 是 16 倍数时使用，否则回退 ID 1 | variance-limited research variant |
| 4 | `cuda_combined` | `(16,16)` 二维 grid/block + scalar shared tile reuse | 显式 research variant |

四个 body 和 launcher 位于 `src/dense_gemm/cuda_candidate/optimized.cu`，ID 定义与有效性检查在 `src/dense_gemm/cuda_candidate/optimized_internal.h`。adapter/registry 暴露了四个 variant，L2 经 public `DenseGemmArgs.kernel={kCudaOptimized,id}` 调用。`select_kernel` 只允许 dense GEMM 的 ID 1–4；未知 ID、ID 0 与其他算子的 optimized family 都明确拒绝。非整除形状、`K=0`、null A/B（仅 `K=0`）、以及仅 4-byte 对齐的 vector fallback 都在 correctness tests 中覆盖。

## 3. 四轮实验与重排

不同 SHA 的 Release 样本不合并。下表给出每轮本地的 promotion-baseline 对照，用于解释重排；p50 是各进程中位数的中位数。

| 实验 | Commit | 对照 p50 (μs) | 候选 p50 (μs) | 当轮加速 | CV / 决策 |
|---|---|---:|---:|---:|---|
| E1 shared reuse | `cdd3554` | naive 27.443 | tiled scalar 26.112 | 1.051x | 0.031；保留进入累计 suite |
| E2 2D mapping | `44cec95` | naive 27.853 | 2D mapping 26.419 | 1.054x | 0.046；保留进入累计 suite |
| E3 vector staging | `755e257` | tiled scalar 26.112 | tiled vector 22.118 | 1.180x | 初测 0.103，完整复跑 0.188；不稳定 |
| E4 scalar combined | `bfe4494` | naive 27.238 | combined 26.010 | 1.047x | 0.005；未达到 1.05x 晋升线 |

E3 后按稳定 p50、p95/CV 和 NCU memory/stall 信号重排时，vector staging 有最高的中心收益；但它已经触发 `CV>0.10` 的重跑规则，复跑仍为 0.188。因此 E4 按预定机械门槛没有混入 vector path，而是只组合二维映射与 scalar shared reuse。这个选择不是否定 `float4` 的潜力，而是避免把不稳定测量提升为默认 dispatch。

## 4. 最终统一 Release 横向结果（唯一权威比较）

来源：`bfe4494` 的 `rtx3080_dense_gemm_optimization_exp4_v1`。`p50` 是 3 个 process median 的中位数；`p95/CV` 来自全部 90 samples。

| Variant | p50 (μs) | p95 (μs) | CV | TFLOP/s | relative to naive |
|---|---:|---:|---:|---:|---:|
| `cuda_naive` | 27.238 | 27.443 | 0.009 | 1.230 | 1.000x |
| `cuda_tiled_scalar` | 26.112 | 26.680 | 0.008 | 1.285 | 1.043x |
| `cuda_2d_mapping` | 26.419 | 26.803 | 0.008 | 1.270 | 1.031x |
| `cuda_tiled_vector` | **22.118** | 27.223 | 0.067 | **1.510** | **1.231x** |
| `cuda_combined` | 26.010 | **26.112** | **0.005** | 1.290 | 1.047x |
| `cublaslt` strict FP32 | 12.134 | 16.282 | 0.273 | 2.765 | 2.245x |
| `cublas` strict FP32 | 11.674 | 15.949 | 0.169 | 2.874 | 2.333x |

`cuda_combined` 的 p50 与 ID 1 相差仅 0.39%，但具备较低 p95/CV；它仍少于 promotion 所需的 1.05x。ID 3 的最终 suite 本身满足 p50/p95/workspace 线，但先前完整重跑没有通过稳定性线，所以不会选择性忽略反例。库的 CV 也高于 0.10，故它们用作 strict-FP32 性能参考而非 tail-latency 排名依据。

最终 naive 27.238 μs 相对既有 27.546 μs 基线偏差 -1.1%，未触发 5% 重跑阈值。由于 256³ 是唯一正式形状，上表不外推到其他矩阵、真实 MoE 流量或其他 GPU。

## 5. NSYS/NCU 取证

每个候选各有独立 CUDA/NVTX NSYS trace（CPU sampling/context-switch sampling 关闭），并以 trace 中的真实 demangled name 过滤 20 warmup 后的一个 NCU launch。E4 的 trace 捕获了 `dense_gemm_combined_scalar_kernel` 21 次、总 GPU 时间 489,207 ns、平均 23,296 ns；这些 profiler durations 不参与第 4 节的加速比。

| Variant | NSYS 平均 kernel duration (ns) | NCU detailed：grid/block, waves/SM | regs / shared | achieved occ. |
|---|---:|---|---|---:|
| tiled scalar | 23,354 | 256 / 256，0.627 | 38 / 3,072 B | 51.28% |
| 2D mapping | 23,617 | 256 / 256，0.627 | 40 / 1,024 B | 47.93% |
| tiled vector | 19,424 | 256 / 256，0.627 | 40 / 3,072 B | 51.73% |
| combined scalar | 23,296 | 256 / 256，0.627 | 37 / 3,072 B | 52.02% |

| Variant | global load requests | L2 hit | DRAM SOL | issue active | long-scoreboard | MIO throttle | barrier |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiled scalar | 65,536 | 99.10% | 3.62% | 31.71% | 279 | 289 | 120 |
| 2D mapping | 1,048,576 | 99.42% | 2.67% | 27.61% | 432 | 0 | 0 |
| tiled vector | 16,384 | 99.40% | 4.30% | 21.98% | 172 | 211 | 346 |
| combined scalar | 65,536 | 99.47% | 4.23% | 32.00% | 282 | 251 | 163 |

所有四个 detailed capture 都报告 local load/store 为 0、tensor pipe 为 0，因此不存在 spill 证据，也没有把 Tensor Core 作为等价优化。Detailed set 没有采集 global load/store sectors 或 eligible warps/scheduler；它们在 evidence 中保持 `not_collected`，不能解释为零。`float4` 的 16,384 requests 是 scalar tiled 的四分之一，支持其减少 load instruction/request count 的假设；但缺少 sectors，不能单凭它宣称减少同等比例的 transaction bytes。

### 瓶颈判断

1. **P0 — global load 指令数与 load-latency 仍是最有价值的优化方向。** 对比 2D mapping 的 1,048,576 load requests / 432 long-scoreboard 与 tiled scalar 的 65,536 / 279，同时 E1 的无 profiler p50 有 1.051x 改善。这是两种独立信号，支持 tile reuse；DRAM SOL 仅 2.67–4.30%、L2 hit 约 99%，不支持“显存带宽饱和”作为首因。
2. **P1 — vector staging 有最大的可重复中心值潜力，但测量稳定性是当前阻塞。** E3 的 Release p50 22.118 μs 和 NSYS 平均 19,424 ns 均优于 scalar tile，同时 request count 与 long-scoreboard 更低；但 E3 两次完整 suite 的 CV 为 0.103/0.188。应先解决或量化 WDDM/桌面干扰，不能以这个候选改变默认 dispatch。
3. **P2 — 16×16 tile 的同步/调度成本限制 scalar combine。** E4 虽将 reg 降到 37、p95/CV 最低，但 barrier 163、MIO 251、long-scoreboard 282，且 p50 只比 naive 快 1.047x。所有变体仅 256 CTAs，在 68 SM 上是 0.627 waves/SM；它是短 kernel 的尾波/underfill 因子，但 cuBLAS 更小的 grid 仍更快，故不能独立归因为首因。

## 6. 重排后的后续实验（本次未实现）

1. **优先：稳定性受控的 vector staging 复验。** 在隔离桌面 GPU 工作、记录温度/功耗/P-state 的条件下，至少两个新的 3×30 suite；保留标准是两次都 `CV≤0.10`、相对 naive p50 ≥1.05x、p95 不退化 >3%、无 spill。风险是把主机/显示器干扰误认为 kernel 效果。
2. **其次：仅在 ID 3 稳定后测试 Ampere `cp.async` 双 stage。** 代码位置为 A/B global-to-shared staging；预期是覆盖仍可见的 long-scoreboard/MIO。风险为 producer-consumer wait-group、16-byte 对齐/tail、额外 shared/register 降低 residency。保留标准为相对稳定 vector 的 p50 ≥1.05x、p95 不退化、detailed 中无 spill 且 long-scoreboard 或 MIO 有一致改善；否则拒绝。
3. **第三：8×16 或 16×8 tile 的 grid/CTA shape sweep。** 目标是将 256 CTAs 的 0.627 waves/SM 提升到更多波次，同时单独观察 reuse、barrier 与 register 代价。风险是更低的 tile reuse 或更差 coalescing；保留标准仍为 strict-FP32 correctness、无 spill 和同一 Release promotion policy。

## 7. 可审计制品与最终 dispatch 决策

每个 bundle 都含 raw JSONL、manifest、aggregate CSV/JSON、strict comparison、sanitizer log、环境、NSYS CSV/hotspot、NCU metrics CSV/JSON、完整 SHA256SUMS。`.ncu-rep`、`.nsys-rep` 和 `.sqlite` 保留在忽略的 `profile/` 中，bundle manifest 记录文件名、大小、SHA256 与 `committed=false`。

- E1: `docs/reports/artifacts/20260801T120145Z-cdd3554-dense-gemm-opt-exp1-v1/`
- E2: `docs/reports/artifacts/20260801T120145Z-44cec95-dense-gemm-opt-exp2-v1/`
- E3（含独立 variance rerun benchmark）: `docs/reports/artifacts/20260801T120145Z-755e257-dense-gemm-opt-exp3-v1/`
- E4 final: `docs/reports/artifacts/20260801T120145Z-bfe4494-dense-gemm-opt-exp4-v1/`

最终决定保持 fail-closed：研究 kernel 已在项目对应 CUDA 位置并可用 ID 1–4 显式复现；因没有稳定通过所有 promotion gate 的 winner，ID 0 不映射到任一实现、`kAuto` 不改变。这保留了最优 p50 的 vector kernel 和全部四轮数据，供下一次稳定性受控的优化继续使用，而不把不稳结果暴露给默认用户路径。
