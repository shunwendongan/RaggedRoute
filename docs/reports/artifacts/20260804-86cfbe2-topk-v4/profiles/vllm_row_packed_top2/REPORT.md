# CUDA Kernel 性能分析报告

- Run: `vllm_row_packed_top2`
- GPU: `NVIDIA GeForce RTX 3080` / `sm86`
- CUDA/NCU/NSYS: 见 `vllm_row_packed_top2/manifest.json`
- 说明：Profiler duration 只用于诊断，不作为正式 benchmark 分数。

## Nsight Systems 热点

| Rank | Kernel | Time |
|---:|---|---:|
| 1 | `void raggedroute::benchmark::library_baseline::<unnamed>::vllm_row_packed_top2_kernel<(int)64>(const float *, int *, float *, int)` | 53663.0 |

## Nsight Compute 证据

### vllm_row_packed_top2_kernel

架构：`sm86`，SM 数：`68`

| Concept | Metric | Value | Status |
|---|---|---:|---|
| `duration_ns` | `gpu__time_duration.sum` | 3424.0 | collected |
| `sm_throughput_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 6.871268461296742 | collected |
| `memory_throughput_pct` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 21.255122950819672 | collected |
| `dram_throughput_pct` | `-` | N/A | not_collected |
| `occupancy_achieved_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | 26.666183788509535 | collected |
| `registers_per_thread` | `launch__registers_per_thread` | 23 | collected |
| `shared_mem_per_block` | `launch__shared_mem_per_block` | 1024 | collected |
| `stall_long_scoreboard` | `-` | N/A | not_collected |

诊断：
- `latency-or-underfill`：sm=6.9%, memory=21.3%
- `low-occupancy`：achieved occupancy=26.7%
- NCU rules（按估算收益排序）：
  - 73.3% `Achieved Occupancy`：The difference between calculated theoretical (100.0%) and measured achieved occupancy (26.7%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel. See the @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.
  - 19.3% `Workload Imbalance`：One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 46.12% above the average, while the minimum instance value is 19.13% below the average.
  - 未提供估算 `Bottleneck`：This kernel grid is too small to fill the available resources on this device, resulting in only 0.31 full waves across all SMs. Look at @section:LaunchStats:Launch Statistics@ for more details.

## 下一步

1. 优先处理证据最强、预计收益最大的单一瓶颈。
2. 修改后先跑 correctness，再用未受 profiler 干扰的 benchmark 做 A/B。
3. 只有 basic/detailed 证据不足时才升级 full/source，且保持 workload 不变。
