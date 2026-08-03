# CUDA Kernel 性能分析报告

- Run: `unpermute-final-cta-ncu`
- GPU: `NVIDIA GeForce RTX 3080` / `sm86`
- CUDA/NCU/NSYS: 见 `unpermute-final-cta-ncu/manifest.json`
- 说明：Profiler duration 只用于诊断，不作为正式 benchmark 分数。

## Nsight Compute 证据

### unpermute_cta_token_vec4_kernel

架构：`sm86`，SM 数：`68`

| Concept | Metric | Value | Status |
|---|---|---:|---|
| `duration_ns` | `gpu__time_duration.sum` | 3328.0 | collected |
| `sm_throughput_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 1.6227263857364405 | collected |
| `memory_throughput_pct` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 21.722944630872483 | collected |
| `dram_throughput_pct` | `-` | N/A | not_collected |
| `occupancy_achieved_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | 14.936822941525287 | collected |
| `registers_per_thread` | `launch__registers_per_thread` | 34 | collected |
| `shared_mem_per_block` | `launch__shared_mem_per_block` | 1040 | collected |
| `stall_long_scoreboard` | `-` | N/A | not_collected |

诊断：
- `latency-or-underfill`：sm=1.6%, memory=21.7%
- `low-occupancy`：achieved occupancy=14.9%
- NCU rules（按估算收益排序）：
  - 85.1% `Achieved Occupancy`：The difference between calculated theoretical (100.0%) and measured achieved occupancy (14.9%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel. See the @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.
  - 8.1% `Workload Imbalance`：One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 22.99% above the average, while the minimum instance value is 13.50% below the average.
  - 6.8% `Workload Imbalance`：One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 24.51% above the average, while the minimum instance value is 100.00% below the average.

## 下一步

1. 优先处理证据最强、预计收益最大的单一瓶颈。
2. 修改后先跑 correctness，再用未受 profiler 干扰的 benchmark 做 A/B。
3. 只有 basic/detailed 证据不足时才升级 full/source，且保持 workload 不变。
