# RaggedRoute Nsight 诊断：histogram-candidate-88b670b-ncu

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| histogram.hot.r1m_e1 / basic | `histogram_block_private_kernel` | 0.3137254901960784 | 30.775541429710678 | 7.25419144546353 | 51.72211021505376 | 51.72211021505376 | 21 |
| histogram.hot.r1m_e1 / detailed | `histogram_block_private_kernel` | 0.3137254901960784 | 65.97885665415409 | 14.444458829063258 | 48.58274647887325 | 48.58274647887325 | 21 |
| histogram.uniform.r1m_e64 / basic | `histogram_block_private_kernel` | 0.3137254901960784 | 30.888962622251043 | 6.924334002852915 | 37.73544520547945 | 37.73544520547945 | 21 |
| histogram.uniform.r1m_e64 / detailed | `histogram_block_private_kernel` | 0.3137254901960784 | 66.24542039118064 | 14.531578470264742 | 48.93298138869005 | 48.93298138869005 | 21 |
| histogram.zipf.r4096_e64 / basic | `histogram_single_cta_shared_kernel` | 0.0024509803921568627 | 16.183185603008326 | 0.064019893417532 | 0.75885173346447 | 0.1385809312638581 | 28 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
