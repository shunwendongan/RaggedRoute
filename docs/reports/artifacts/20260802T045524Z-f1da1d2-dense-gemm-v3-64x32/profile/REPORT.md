# RaggedRoute Nsight 诊断：dense-gemm-f1da1d2-v3-ncu-20260802T050000Z

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| dense.cublas.m1024 / basic | `ampere_sgemm_64x64_nn` | 0.9411764705882353 | 24.114178426719562 | 63.651243884345966 | 54.90827985899948 | 18.637717382977094 | 126 |
| dense.cublas.m1024 / detailed | `ampere_sgemm_64x64_nn` | 0.9411764705882353 | 24.07636506957494 | 63.614890266456946 | 54.87700657561712 | 18.383385734484296 | 126 |
| dense.cublas.m1024 / full | `ampere_sgemm_64x64_nn` | 0.9411764705882353 | 24.143103939573688 | 63.68561343023045 | 54.93793262167173 | 19.661548359837134 | 126 |
| dense.register_tiled_v2_async.m1024 / basic | `dense_gemm_register_tiled_v2_async_kernel` | 2.1512605042016806 | 49.86536089158008 | 79.13148953659133 | 79.13148953659133 | 11.576859956236325 | 70 |
| dense.register_tiled_v2_async.m1024 / detailed | `dense_gemm_register_tiled_v2_async_kernel` | 2.1512605042016806 | 49.89852280845344 | 79.2971096382721 | 79.2971096382721 | 12.132559226932669 | 70 |
| dense.register_tiled_v2_async.m1024 / full | `dense_gemm_register_tiled_v2_async_kernel` | 2.1512605042016806 | 49.873284055748194 | 79.14231291480166 | 79.14231291480166 | 12.001083113390768 | 70 |
| dense.register_tiled_v3_64x32_async.m1024 / basic | `dense_gemm_register_tiled_v3_64x32_async_kernel` | 1.5058823529411764 | 34.61785629947363 | 73.60769427225513 | 73.80531440963321 | 14.090453246034674 | 85 |
| dense.register_tiled_v3_64x32_async.m1024 / detailed | `dense_gemm_register_tiled_v3_64x32_async_kernel` | 1.5058823529411764 | 34.59830120709967 | 73.69499777882199 | 75.08720899503525 | 15.051738151806928 | 85 |
| dense.register_tiled_v3_64x32_async.m1024 / full | `dense_gemm_register_tiled_v3_64x32_async_kernel` | 1.5058823529411764 | 34.616628652936896 | 73.66658464034397 | 75.09104603837412 | 15.05357756734796 | 85 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
