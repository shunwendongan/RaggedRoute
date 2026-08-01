# RaggedRoute Nsight 诊断：profile

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|
| 1 | `raggedroute::ops::<unnamed>::dense_gemm_2d_mapping_kernel(const float *, const float *, float *, int, int, int)` | 100.0 | 495960.0 |

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| dense.2d_mapping / basic | `dense_gemm_2d_mapping_kernel` | 0.6274509803921569 | 44.52111500307282 | 76.81349829436046 | 76.81349829436046 | 0.04284734917733089 | 40 |
| dense.2d_mapping / detailed | `dense_gemm_2d_mapping_kernel` | 0.6274509803921569 | 47.93049081645885 | 76.87261850272151 | 76.87261850272151 | 2.6748002283105023 | 40 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
