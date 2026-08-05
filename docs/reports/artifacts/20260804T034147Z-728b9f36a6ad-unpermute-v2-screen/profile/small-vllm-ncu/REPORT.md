# CUDA Kernel 性能分析报告

- Run: `small-vllm_finalize_routing-t64-n256-ncu`
- GPU: `NVIDIA GeForce RTX 3080` / `sm86`
- CUDA/NCU/NSYS: 见 `small-vllm_finalize_routing-t64-n256-ncu/manifest.json`
- 说明：Profiler duration 只用于诊断，不作为正式 benchmark 分数。

## Nsight Compute 证据

### finalize_routing

架构：`sm86`，SM 数：`68`

| Concept | Metric | Value | Status |
|---|---|---:|---|
| `duration_ns` | `gpu__time_duration.sum` | 9344.0 | collected |
| `sm_throughput_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 0.3942958533219426 | collected |
| `memory_throughput_pct` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 1.9911668646080758 | collected |
| `dram_throughput_pct` | `-` | N/A | not_collected |
| `occupancy_achieved_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | 7.882980741047259 | collected |
| `registers_per_thread` | `launch__registers_per_thread` | 40 | collected |
| `shared_mem_per_block` | `launch__shared_mem_per_block` | 1024 | collected |
| `stall_long_scoreboard` | `-` | N/A | not_collected |

诊断：
- `latency-or-underfill`：sm=0.4%, memory=2.0%
- `low-occupancy`：achieved occupancy=7.9%
- NCU rules（按估算收益排序）：
  - 92.1% `Achieved Occupancy`：The difference between calculated theoretical (100.0%) and measured achieved occupancy (7.9%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel. See the @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.
  - 6.0% `Workload Imbalance`：One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 72.61% above the average, while the minimum instance value is 27.08% below the average.
  - 5.9% `Launch Configuration`：The grid for this launch is configured to execute only 64 blocks, which is less than the 68 multiprocessors used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently with other workloads, consider reducing the block size to have at least one block per multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the @url:Hardware Model:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model@ description for more details on launch configurations.

## 下一步

1. 优先处理证据最强、预计收益最大的单一瓶颈。
2. 修改后先跑 correctness，再用未受 profiler 干扰的 benchmark 做 A/B。
3. 只有 basic/detailed 证据不足时才升级 full/source，且保持 workload 不变。
