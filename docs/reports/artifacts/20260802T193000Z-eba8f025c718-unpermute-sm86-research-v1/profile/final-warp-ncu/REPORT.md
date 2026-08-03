# CUDA Kernel 性能分析报告

- Run: `unpermute-final-warp-ncu`
- GPU: `NVIDIA GeForce RTX 3080` / `sm86`
- CUDA/NCU/NSYS: 见 `unpermute-final-warp-ncu/manifest.json`
- 说明：Profiler duration 只用于诊断，不作为正式 benchmark 分数。

## Nsight Compute 证据

### unpermute_warp_token_vec4_kernel

架构：`sm86`，SM 数：`68`

| Concept | Metric | Value | Status |
|---|---|---:|---|
| `duration_ns` | `gpu__time_duration.sum` | 9184.0 | collected |
| `sm_throughput_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 2.606163891374929 | collected |
| `memory_throughput_pct` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 43.30492424242424 | collected |
| `dram_throughput_pct` | `-` | N/A | not_collected |
| `occupancy_achieved_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | 25.357238389564035 | collected |
| `registers_per_thread` | `launch__registers_per_thread` | 34 | collected |
| `shared_mem_per_block` | `launch__shared_mem_per_block` | 1024 | collected |
| `stall_long_scoreboard` | `-` | N/A | not_collected |

诊断：
- `latency-or-underfill`：sm=2.6%, memory=43.3%
- `low-occupancy`：achieved occupancy=25.4%
- NCU rules（按估算收益排序）：
  - 74.6% `Achieved Occupancy`：The difference between calculated theoretical (100.0%) and measured achieved occupancy (25.4%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel. See the @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.
  - 19.6% `Workload Imbalance`：One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 40.63% above the average, while the minimum instance value is 7.53% below the average.
  - 6.5% `Workload Imbalance`：One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 13.72% above the average, while the minimum instance value is 32.64% below the average.

## 下一步

1. 优先处理证据最强、预计收益最大的单一瓶颈。
2. 修改后先跑 correctness，再用未受 profiler 干扰的 benchmark 做 A/B。
3. 只有 basic/detailed 证据不足时才升级 full/source，且保持 workload 不变。
