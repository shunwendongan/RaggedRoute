# RaggedRoute Nsight 诊断：profile

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|
| 1 | `raggedroute::ops::<unnamed>::dense_gemm_tiled_vector_kernel(const float *, const float *, float *, int, int, int)` | 100.0 | 407894.0 |

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| dense.tiled_vector / basic | `dense_gemm_tiled_vector_kernel` | 0.6274509803921569 | 48.42378912226268 | 69.60251810437806 | 69.60251810437806 | 4.181243272335845 | 40 |
| dense.tiled_vector / detailed | `dense_gemm_tiled_vector_kernel` | 0.6274509803921569 | 51.72530117278657 | 67.20536739758239 | 67.20536739758239 | 4.303431239848403 | 40 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
