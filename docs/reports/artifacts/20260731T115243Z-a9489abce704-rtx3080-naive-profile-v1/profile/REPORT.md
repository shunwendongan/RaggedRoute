# RaggedRoute Nsight 诊断：20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|
| 1 | `raggedroute::ops::<unnamed>::grouped_gemm_naive_kernel(const float *, const float *, const int *, float *, int, int)` | 71.9 | 466004.0 |
| 2 | `raggedroute::ops::<unnamed>::topk_gate_naive_kernel(const float *, int *, float *, int, int)` | 10.4 | 67552.0 |
| 3 | `raggedroute::ops::<unnamed>::dense_gemm_naive_kernel(const float *, const float *, float *, int, int, int)` | 8.1 | 52315.0 |
| 4 | `raggedroute::ops::<unnamed>::token_permute_naive_kernel(const float *, const int *, const int *, int *, float *, int *, int *, int, int, int)` | 3.1 | 20160.0 |
| 5 | `raggedroute::ops::<unnamed>::exclusive_scan_naive_kernel(const int *, int *, int)` | 3.0 | 19136.0 |
| 6 | `raggedroute::ops::<unnamed>::unpermute_naive_kernel(const float *, const int *, const float *, float *, int, int, int)` | 2.2 | 14368.0 |
| 7 | `raggedroute::ops::<unnamed>::histogram_naive_kernel(const int *, int *, int, int)` | 1.3 | 8576.0 |

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| dense.square / basic | `dense_gemm_naive_kernel` | 0.6274509803921569 | 48.503662733304076 | 67.91230711862745 | 67.91230711862745 | 2.795449918566775 | 40 |
| dense.square / detailed | `dense_gemm_naive_kernel` | 0.6274509803921569 | 48.73983571323899 | 47.310739648281334 | 50.37971509971509 | 2.8589750101584723 | 40 |
| grouped.skew / basic | `grouped_gemm_naive_kernel` | 26.35294117647059 | 58.56014587346503 | 45.54277235600397 | 45.54277235600397 | 15.902626632370398 | 40 |
| grouped.skew / detailed | `grouped_gemm_naive_kernel` | 26.35294117647059 | 59.088111366109395 | 45.44897696695248 | 45.44897696695248 | 13.851184600197433 | 40 |
| histogram.skew / basic | `histogram_naive_kernel` | 0.0392156862745098 | 15.737993133463519 | 0.1305687792945448 | 0.815093397975189 | 0.48415492957746487 | 16 |
| permute.skew / basic | `token_permute_naive_kernel` | 1.2549019607843137 | 70.9010865557283 | 10.336644190400028 | 19.56578947368421 | 19.56578947368421 | 26 |
| scan.e64 / basic | `exclusive_scan_naive_kernel` | 0.0009191176470588235 | 2.083333333333333 | 0.021120902544906287 | 0.33602707556292305 | 0.08529776674937967 | 20 |
| topk.short_rows / basic | `topk_gate_naive_kernel` | 0.0196078431372549 | 16.02246145127021 | 2.702830709966688 | 9.451726657012333 | 6.368328010948905 | 26 |
| topk.short_rows / detailed | `topk_gate_naive_kernel` | 0.0196078431372549 | 15.81869806032664 | 2.6479450725752365 | 9.259652443148344 | 6.054865542388332 | 26 |
| unpermute.uniform / basic | `unpermute_naive_kernel` | 2.5098039215686274 | 80.61617410372882 | 22.456017897918997 | 35.865053258145366 | 35.865053258145366 | 28 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
