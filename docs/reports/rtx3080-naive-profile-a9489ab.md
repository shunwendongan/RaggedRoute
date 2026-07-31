# RTX 3080 七算子 naive benchmark 与 Nsight 分析（a9489ab）

> 结论：七个 strict-FP32 naive CUDA 算子的 correctness、三进程 Release benchmark、完整链 NSYS 和逐算子 NCU 均已完成。结果建立了可审计的性能与资源占用基线，不修改默认 dispatch，也不构成 optimized variant 晋升结论。

## 1. 环境与证据边界

- tested commit：`a9489abce70478aa4811e3e867f6fa6e6a737bf8`（clean Git）；
- GPU：NVIDIA GeForce RTX 3080，SM 8.6，68 SM，10 GiB；driver 591.86；
- CUDA compiler 13.3.73；MSVC 19.44；NCU 2026.2.1；NSYS 2026.1.3；
- build：Release、`sm_86`、`-lineinfo`、strict FP32，未锁频，NCU 使用 `--clock-control none`；
- seed：`20260729`；Release 为 3 个独立进程、warmup 20、每进程 30 samples；
- correctness：CTest 8/8；Compute Sanitizer 的 memcheck/initcheck/racecheck/synccheck 全部 PASS；naive 78 条和 library 45 条 raw record 全部 `validation.ok=true`；
- 完整文本 bundle：[artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1](artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。

Profiler duration 受 replay、cache control 和序列化影响，只用于解释瓶颈；以下 p50/p95/CV 全部来自无 profiler 的 Release benchmark。

## 2. 七算子 naive Release 结果

主值为三个进程各自 p50 的 median；p95/CV 使用全部 90 个 batch-mean samples。

| Operator / case | Level | p50 (us) | p95 (us) | CV |
|---|---|---:|---:|---:|
| Dense GEMM `256×256×256` | L1 / L2 | 27.034 / 27.034 | 27.136 / 27.136 | 0.004 / 0.003 |
| Top-K Gate `T2048,E64` | L1 / L2 | 14.029 / 13.875 | 15.624 / 15.137 | 0.072 / 0.034 |
| Histogram Zipf-1.4 `T2048,E64,top2` | L1 / L2 | 9.175 / 17.500 | 10.540 / 18.305 | 0.100 / 0.058 |
| Exclusive Scan `E64,R4096` | L1 / L2 | 8.192 / 8.637 | 9.069 / 9.468 | 0.055 / 0.058 |
| Token Permute Zipf-1.4 `T512,K256` | L1 / L2 | 10.240 / 19.763 | 10.240 / 21.356 | 0.094 / 0.173 |
| Grouped GEMM Zipf-1.4 `T512,K128,N128` | L1 / L2 | 47.718 / 48.128 | 48.538 / 48.538 | 0.017 / 0.051 |
| Unpermute uniform `T1024,N256` | L1 / L2 | 9.626 / 9.498 | 13.420 / 12.931 | 0.167 / 0.140 |
| Full 7-op chain `T64,E16,K64,N64` | L3 | 48.128 | 108.544 | 0.384 |

Histogram 与 Permute 的 L2 包含必要的 counts/cursor reset，不能用 L1 代替 operator 成本。7/26 个 naive 聚合组 `CV>0.10`，最大为 0.473；WDDM/桌面环境的抖动仍然明显，尤其是短 kernel 与 L3 chain。

## 3. 合同匹配的强基线

同一 suite v2 case 中共享输入、seed、math mode、cache 和 L2 边界。表中的“reference advantage”是 `naive p50 / reference p50`；小于 1 表示 naive 更快。

| Operator | Reference | Reference p50 (us) | Comparable naive p50 (us) | Reference advantage |
|---|---|---:|---:|---:|
| Dense GEMM | cuBLASLt | 10.854 | 26.931 | 2.48× |
| Histogram | CUB DeviceHistogram | 20.275 | 15.811 | 0.78×（naive 快 1.28×） |
| Exclusive Scan | CUB BlockScan | 9.175 | 8.668 | 0.94×（naive 快 1.06×） |
| Token Permute full-from-ids | vLLM adapted | 32.358 | 40.550 | 1.25× |
| Grouped GEMM | CUTLASS Grouped | 23.757 | 47.923 | 2.02× |
| Unpermute | vLLM adapted | 9.446 | 9.754 | 1.03× |

Top-K 没有满足 lower-id tie、NaN 和 selected-softmax 合同的外部实现，因此没有制造虚假 speedup。部分 reference 组 CV 超过 0.10，例如 cuBLASLt、vLLM Permute/Unpermute；这些比值用于当前机器上的参考定位，不用于 promotion。

## 4. NSYS 完整 7 算子链

System case 为 `chain_from_tokens(T=512,E=64,K=N=128,uniform)`，执行 5 次 warmup 加 1 次验证调用。七个目标 kernel 都被观察到：

| Rank | Kernel | GPU kernel time share | Average duration (us) |
|---:|---|---:|---:|
| 1 | Grouped GEMM | 71.9% | 77.667 |
| 2 | Top-K Gate | 10.4% | 11.259 |
| 3 | Dense GEMM | 8.1% | 8.719 |
| 4 | Token Permute | 3.1% | 3.360 |
| 5 | Exclusive Scan | 3.0% | 3.189 |
| 6 | Unpermute | 2.2% | 2.395 |
| 7 | Histogram | 1.3% | 1.429 |

因此 detailed NCU 只追加到 Grouped GEMM、Top-K 和 Dense GEMM；热点按系统 GPU 时间选择，而不是按源码熟悉度选择。

同一 trace 的 CUDA API 与 GPU memory-operation 排名如下。它们覆盖整个短进程，包含一次性 stream/context 初始化、输入上传和输出验证，所以只用于定位系统开销，不是 L3 chain latency：

| Category | Rank | Operation | Time share | Total time | Calls / operations |
|---|---:|---|---:|---:|---:|
| CUDA API | 1 | `cudaStreamCreateWithFlags` | 93.5% | 131.542 ms | 1 |
| CUDA API | 2 | `cudaMemcpyAsync` | 2.8% | 3.901 ms | 8 |
| CUDA API | 3 | `cudaStreamSynchronize` | 2.3% | 3.174 ms | 9 |
| CUDA API | 4 | `cudaLaunchKernel` | 0.7% | 0.986 ms | 42 |
| GPU memory | 1 | Host-to-Device memcpy | 94.1% | 5.469 ms | 3 |
| GPU memory | 2 | Device-to-Host memcpy | 5.6% | 0.323 ms | 5 |
| GPU memory | 3 | Memset | 0.4% | 0.022 ms | 12 |

在稳态 7-op kernel 排名之外，API 表最明显的是一次性 stream 创建；memory 表则主要反映模型输入/权重上传。两者都不支持把 host 初始化或传输时间归因给任一 naive kernel。完整原始排名保存在 bundle 的 `nsys_cuda_api_sum.csv` 与 `nsys_cuda_gpu_mem_time_sum.csv`。

## 5. NCU 资源占用

每个 basic report 跳过 5 条匹配 warmup，只采 1 条 steady-state launch。

| Case | Grid / block | Waves/SM | Achieved occupancy | SM % | Memory % | DRAM % | Registers/thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 256 / 256 | 0.627 | 48.5% | 67.9 | 67.9 | 2.8 | 40 |
| Top-K | 8 / 256 | 0.020 | 16.0% | 2.7 | 9.5 | 6.4 | 26 |
| Histogram | 16 / 256 | 0.039 | 15.7% | 0.1 | 0.8 | 0.5 | 16 |
| Scan | 1 / 1 | 0.0009 | 2.1% | 0.02 | 0.34 | 0.09 | 20 |
| Permute | 1024 / 128 | 1.255 | 70.9% | 10.3 | 19.6 | 19.6 | 26 |
| Grouped GEMM | 10752 / 256 | 26.353 | 58.6% | 45.5 | 45.5 | 15.9 | 40 |
| Unpermute | 1024 / 256 | 2.510 | 80.6% | 22.5 | 35.9 | 35.9 | 28 |

Detailed evidence：

- **Grouped GEMM**：理论 occupancy 100%，实际 59.1%；SM/Memory 45.4%，DRAM 13.9%，L1/L2 hit 86.6%/56.9%，issue active 19.3%；PC-sampling long-scoreboard 2264，远高于 short-scoreboard 129、wait 170，且 local load/store 均为 0。结合 Zipf 输入与 NSYS 71.9% 占比，主问题是 ragged 调度/尾部不均衡叠加数据等待，不是寄存器 spill。
- **Dense GEMM**：只有 0.627 waves/SM，实际 occupancy 48.7%；L2 hit 98.2%，DRAM 2.9%，long-scoreboard samples 852，local load/store 为 0。当前 one-thread-per-output 路径既缺少 tile 复用，也无法用这个小网格充分填满 68 个 SM。
- **Top-K**：仅 8 blocks、0.020 waves/SM，实际 occupancy 15.8%，SM 2.6%、Memory 9.3%；long-scoreboard 45、wait 11。主要是一个 thread 串行扫描一行造成的严重 underfill/低并行度，而不是带宽峰值限制。

NCU 2026.2.1 的 `detailed` set 不含 SchedulerStats/WarpStateStats，因此 `eligible_warps_per_scheduler` 保留为 `not_collected`；stall 使用该 set 实际提供的 PC-sampling counters，单位是 sample count，不解释为百分比。

## 6. 后续实验（不在本 PR 实现）

1. **Grouped GEMM scheduler**：以 CUTLASS Grouped 2.02× reference advantage 为目标，先测试 Ampere-compatible grouped tile/work queue 或 shape bucket，保持 strict FP32 与相同 Zipf case；验证 tail/waves、long-scoreboard 和 L2 延迟，再由未插桩 benchmark 决定保留。
2. **Dense tiled CUDA Core path**：先建立正确的 shared-memory tile/register blocking，再单独评估 `cp.async` 双缓冲；每次只改变一个机制，并同时观察 register、occupancy、spill 与 cuBLASLt 差距。
3. **Warp-per-token Top-K**：用 warp shuffle 完成 Top-2 reduction 和 selected-softmax，严格保持 lower-id tie 与 NaN 合同；目标是把 8-block underfill 改为足够的 token/warp 并行度。

Histogram/Scan 在该小 metadata 合同下已经快于所选 CUB reference；不能仅因“库实现”标签而替换默认路径。所有下一步都必须重新通过 correctness、sanitizer 和同机未插桩 A/B benchmark。

## 7. 制品

- raw benchmark 与 manifest：[benchmark/](artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/benchmark/)
- normalized NSYS/NCU：[profile/](artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/profile/)
- bundle manifest：[manifest.json](artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/manifest.json)
- checksums：[SHA256SUMS](artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/SHA256SUMS)

本地 `.ncu-rep/.nsys-rep/.sqlite` 未提交；manifest 保存其文件名、字节数和 SHA256，以避免 Git 中加入与 Nsight 版本绑定的二进制制品。
