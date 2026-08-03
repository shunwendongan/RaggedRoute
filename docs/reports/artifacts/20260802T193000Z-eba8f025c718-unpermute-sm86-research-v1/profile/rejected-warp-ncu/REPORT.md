# CUDA Kernel 性能分析报告

- Run: `unpermute-i1-n1024-ncu`
- GPU: `NVIDIA GeForce RTX 3080` / `sm86`
- CUDA/NCU/NSYS: 见 `unpermute-i1-n1024-ncu/manifest.json`
- 说明：Profiler duration 只用于诊断，不作为正式 benchmark 分数。

## Nsight Compute 证据

### unpermute_warp_vec4_top2_kernel

架构：`sm86`，SM 数：`68`

| Concept | Metric | Value | Status |
|---|---|---:|---|
| `duration_ns` | `gpu__time_duration.sum` | 8864.0 | collected |
| `sm_throughput_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 0.5718122549762827 | collected |
| `memory_throughput_pct` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 8.207050879396986 | collected |
| `dram_throughput_pct` | `-` | N/A | not_collected |
| `occupancy_achieved_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | 8.226985839877795 | collected |
| `registers_per_thread` | `launch__registers_per_thread` | 34 | collected |
| `shared_mem_per_block` | `launch__shared_mem_per_block` | 1024 | collected |
| `stall_long_scoreboard` | `-` | N/A | not_collected |

诊断：
- `latency-or-underfill`：sm=0.6%, memory=8.2%
- `low-occupancy`：achieved occupancy=8.2%
- NCU rules（按估算收益排序）：
  - 91.8% `Achieved Occupancy`：The difference between calculated theoretical (100.0%) and measured achieved occupancy (8.2%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel. See the @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.
  - 76.5% `Launch Configuration`：The grid for this launch is configured to execute only 16 blocks, which is less than the 68 multiprocessors used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently with other workloads, consider reducing the block size to have at least one block per multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the @url:Hardware Model:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model@ description for more details on launch configurations.
  - 11.1% `Workload Imbalance`：One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 33.55% above the average, while the minimum instance value is 10.71% below the average.

## 下一步

1. 优先处理证据最强、预计收益最大的单一瓶颈。
2. 修改后先跑 correctness，再用未受 profiler 干扰的 benchmark 做 A/B。
3. 只有 basic/detailed 证据不足时才升级 full/source，且保持 workload 不变。
