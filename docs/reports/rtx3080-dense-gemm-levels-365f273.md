# RTX 3080 dense GEMM：cuda_candidate 与 cuBLAS 的 L1/L2 对比

## 结论先行

在严格 FP32、row-major、warm-cache 的同一测量边界下，候选 kernel 与 cuBLAS 的差距不是由 256³ 的 launch 计时误差单独造成的。补充的 512³ 和 1024³ 规格把单次计算拉长后，最快的 `cuda_tiled_vector` 仍比 cuBLAS 慢约 4.42–5.38 倍（L2）；`cuda_combined`/`cuda_tiled_scalar` 慢约 5.08–6.29 倍；`cuda_2d_mapping` 慢约 5.49–6.89 倍；naive 慢约 7.44–12.03 倍。

当前最有价值的候选仍是 `cuda_tiled_vector`，但没有达到暂定 kernel 晋升门槛，因此没有修改 `kAuto` 或默认 dispatch。

## 测量合同与证据边界

- GPU：NVIDIA GeForce RTX 3080，sm_86，68 SM，10 GiB；driver 591.86。
- CUDA/NVCC：13.3.73；NSYS 2026.1.3；NCU 2026.2.1；compute capability 8.6。
- FP32 row-major，`alpha=1`、`beta=0`、strict FP32，warm cache；3 个独立进程、20 次 warmup、30 个 sample、每个 sample 10 次 kernel repeat，seed `20260729`。
- 256³ 是原定正式规格；512³、1024³ 是用于降低短 kernel 计时误差的工业规格补充组，不与 256³ 合并成一个性能分数。
- Release p50/p95/CV 来自无 profiler 的 event timing；NCU/NSYS duration 只用于诊断，不用于加速比。
- 所有补充记录 `validation.ok=true`，每个 variant/level/shape 均为 3 records、90 raw samples，严格 pairing 通过。

## 无 profiler Release 结果

### 原定 256³ 主规格

下表的“cuBLAS 倍数”是 candidate p50 / cuBLAS p50，数值越大表示 candidate 越慢。cuBLAS 在该短规格的 CV 为 0.189–0.204，因此只把倍数解释为量级。

| Level | cuBLAS p50 / p95 / CV (µs) | cuda_tiled_vector | cuda_combined | cuda_tiled_scalar | cuda_2d_mapping |
|---|---:|---:|---:|---:|---:|
| L1 kernel body | 11.008 / 15.667 / 0.204 | 22.170 / 27.761 / 0.093 (2.01×) | 25.907 / 26.726 / 0.022 (2.35×) | 26.112 / 26.839 / 0.033 (2.37×) | 26.522 / 29.491 / 0.043 (2.41×) |
| L2 operator steady | 10.854 / 16.179 / 0.189 | 22.118 / 27.704 / 0.087 (2.04×) | 25.907 / 26.680 / 0.010 (2.39×) | 26.214 / 29.343 / 0.048 (2.42×) | 26.317 / 26.952 / 0.009 (2.42×) |

### 工业规格补充组：512³ 与 1024³

| Shape / Level | cuBLAS p50 (µs) | tiled_vector | combined | tiled_scalar | 2D mapping | naive |
|---|---:|---:|---:|---:|---:|---:|
| 512³ / L1 | 23.962 | 120.218 (5.02×) | 138.445 (5.78×) | 138.035 (5.76×) | 148.787 (6.21×) | 201.882 (8.43×) |
| 512³ / L2 | 27.136 | 120.064 (4.42×) | 137.830 (5.08×) | 138.650 (5.11×) | 148.890 (5.49×) | 201.933 (7.44×) |
| 1024³ / L1 | 154.931 | 835.021 (5.39×) | 972.954 (6.28×) | 975.718 (6.30×) | 1069.568 (6.90×) | 1862.912 (12.02×) |
| 1024³ / L2 | 155.034 | 834.662 (5.38×) | 974.234 (6.28×) | 975.872 (6.29×) | 1068.134 (6.89×) | 1864.602 (12.03×) |

1024³ 的 CV 为 cuBLAS 0.008–0.010、scalar/combined/2D 0.003–0.020、vector 0.026–0.038；因此该规格对“候选仍落后 cuBLAS”的结论比 256³ 更稳健。512³ 的 L2 cuBLAS CV=0.095，接近 0.10 方差门槛，故 1024³ 是更合适的确认规格。

### L1→L2 边界

在 1024³ 上，cuBLAS 的 L2 p50 比 L1 高 0.10 µs，vector 低 0.36 µs，combined 高 1.28 µs，scalar 高 0.15 µs，2D 低 1.43 µs；这些差异都小于 kernel 级差距，不能解释 5–12× 的差距。512³ 的 cuBLAS L1/L2 差值约 3.17 µs，但 L2 CV=0.095，属于方差敏感结果。256³ 所有 L1/L2 差值也都在约 0.2 µs 内，且 cuBLAS CV 很高。因此当前没有稳定证据表明 public operator/dispatch 是主要瓶颈。

## NCU detailed：1024³ L1/L2 真实 kernel

报告目录：`profile/dense-gemm-365f273-industrial-profile-v1/`。每个 case 只采集 warmup 后一个 launch，`--cache-control none --clock-control none`。

下表列出 L2 case；对应 L1 case 的 grid/block、寄存器、shared memory 和主要吞吐指标在测量噪声内一致，L1/L2 的差异以 Release 表格和上一节的边界分析为准。

| Kernel | Grid / block | Waves/SM | Reg/thread | Shared | Achieved occ. | L1 / L2 hit | Global load requests | Issue active | Long scoreboard / MIO / barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuBLAS `ampere_sgemm_64x64_nn` | 512 / 64 | 0.941 | 126 | 9.2 KiB | 24.1% | 7.4% / 92.7% | 1.082 M | 61.8% | 358 / 81 / 194 |
| `cuda_tiled_vector_kernel` | 4096 / 256 | 10.039 | 40 | 3 KiB | 96.5% | 5.1% / 98.2% | 1.049 M | 28.4% | 2352 / 20854 / 20691 |
| `cuda_combined_scalar_kernel` | 4096 / 256 | 10.039 | 37 | 3 KiB | 96.2% | 4.4% / 99.4% | 4.194 M | 43.4% | 6576 / 27277 / 7638 |
| `cuda_tiled_scalar_kernel` | 4096 / 256 | 10.039 | 38 | 3 KiB | 96.2% | 4.5% / 99.0% | 4.194 M | 43.6% | 6484 / 27164 / 7424 |
| `cuda_2d_mapping_kernel` | 4096 / 256 | 10.039 | 40 | 1 KiB | 95.9% | 87.5% / 99.5% | 67.109 M | 34.0% | 7226 / 131 / 0 |
| `dense_gemm_naive_kernel` | 4096 / 256 | 10.039 | 40 | 1 KiB | 96.6% | 22.7% / 100.3% | 67.109 M | 23.0% | 98276 / 5 / 0 |

关键解释：

1. tiled reuse 把 global load requests 从 naive/2D 的约 67.1 M 降到 scalar/combined 的 4.19 M，再降到 vector 的 1.05 M；这与 scalar/combined 相对 naive 的约 1.91×、vector 的约 2.23× kernel 改善方向一致。
2. vector 的 request 数与 cuBLAS 接近，但 issue active 只有 28.4%，并伴随大量 barrier（20691）和 memory/scoreboard stall；因此“减少请求”没有转化为 cuBLAS 级吞吐。
3. scalar/combined 的 shared-memory reuse 已有效，但 MIO throttle 约 27k、long scoreboard 约 6.5k，说明同步、共享内存访问和等待仍是主要代价；combined 与 scalar 的 NCU/Release 结果近似，二维映射并未带来额外收益。
4. 2D mapping 消除了线性 tile 的除法/取模，但仍重复读取 67.1 M requests；在 1024³ 只比 naive 快约 1.75×，所以 index mapping 不是第一优先级。
5. cuBLAS 使用 64×64 内部 kernel、126 registers/thread、约 24% achieved occupancy，却保持约 62% issue active 和 18.7% DRAM throughput；它通过更深的寄存器 blocking、库级调度和更高的计算/访存重叠获得优势。Tensor pipe 为 0%，这里没有把 TF32/Tensor Core 当作等价优化。

### NSYS 1024³ L2 hotspot

NSYS trace 位于 `profile/dense-gemm-365f273-industrial-profile-v1/nsys-*-l2/`，每条 trace 21 次 warm kernel：

| Variant | Kernel avg (µs) | Kernel total (µs) |
|---|---:|---:|
| cuBLAS | 144.873 | 3042.334 |
| cuda_tiled_vector | 950.780 | 19966.380 |
| cuda_combined | 1100.447 | 23109.386 |
| cuda_tiled_scalar | 1102.390 | 23150.198 |
| cuda_2d_mapping | 1213.047 | 25473.994 |
| cuda_naive | 2074.251 | 43559.263 |

这些 duration 只用于确认热点和 kernel 顺序；Release p50/p95 仍以上面的无 profiler 表格为准。

## 瓶颈优先级与后续实验建议

1. **P0：更高层次的 register blocking / warp tile。** 当前 16×16、单输出线程的 candidate 已有 reuse，但与 cuBLAS 的 64×64 内部 blocking 仍有数量级的 instruction/issue 差距。候选实验应保持 strict FP32，明确控制寄存器上限和 spill；若出现 local load/store 或 p95 回退则拒绝。
2. **P1：在保留 reuse 的前提下优化 staging。** vector staging 已证明请求数下降，但 barrier/long-scoreboard 仍高；可单独测试 Ampere `cp.async` 双缓冲或更少的同步点，要求 NCU 的 MIO/long-scoreboard 下降且 Release p50、p95、CV 同时改善。不要把 profiler duration 当作晋升依据。
3. **P2：tile 尺寸与线程到输出的映射。** 尝试 32×32/warp-level microtile 时必须检查 registers、occupancy、spill 和 shared-memory bank conflict；若寄存器压力使 occupancy 或 p95 变差，保留当前 vector。
4. **P3：单独去除线性 `/`、`%`。** 2D mapping 已隔离验证，收益远小于 tile reuse/vector staging；除非新的 NCU 显示 index instruction 明显占比，否则不优先。

## 可复现命令与限制

```powershell
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\benchmark_dense_gemm_industrial_supplement.json `
  --output profile\dense-gemm-365f273-industrial-v1\release.jsonl
python scripts\aggregate_results.py profile\dense-gemm-365f273-industrial-v1\release.jsonl `
  --json profile\dense-gemm-365f273-industrial-v1\aggregate.json `
  --csv profile\dense-gemm-365f273-industrial-v1\aggregate.csv `
  --manifest profile\dense-gemm-365f273-industrial-v1\release.jsonl.manifest.json
python scripts\profile_benchmarks.py analyze --run-dir profile\dense-gemm-365f273-industrial-profile-v1
```

`profile/` 按仓库策略被忽略；报告和补充 suite 配置被版本化，原始 `.ncu-rep/.nsys-rep/.sqlite` 保留在本机证据目录。当前 adapter 仍没有 candidate-vs-cuBLAS 的合法 L3 chain 对照，因而本报告不输出 L3 数字，也不把 256/512/1024 的结果外推到生产 MoE 流量或其他 GPU。
