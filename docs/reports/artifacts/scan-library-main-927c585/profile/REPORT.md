# RaggedRoute Nsight 诊断：scan-compute-32bf6c9

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| scan.blocked_scalar.e64 / detailed | `exclusive_scan_warp_blocked_scalar_kernel` | 0.0009191176470588235 | 2.2197732997481108 | 0.0034107672236639032 | 0.1951424425265409 | 0.0 | 16 |
| scan.blocked_vector.e64 / detailed | `exclusive_scan_warp_blocked_vector_kernel` | 0.0009191176470588235 | 2.2491112934098987 | 0.002750933254106455 | 0.1902009670595346 | 0.0 | 16 |
| scan.cub_block.e64 / basic | `block_scan_kernel` | 0.0012254901960784314 | 8.101159311892296 | 0.010660980810234541 | 0.0861295028616317 | 0.06326687116564417 | 16 |
| scan.naive.e64 / basic | `exclusive_scan_naive_kernel` | 0.0009191176470588235 | 2.0833333333333335 | 0.037809602475656436 | 1.3876090176073719 | 0.2271884272997032 | 20 |
| scan.striped.e32 / basic | `exclusive_scan_warp_striped_kernel` | 0.0009191176470588235 | 2.0833333333333335 | 0.003466034307962925 | 0.08838387485305459 | 0.04810652709359606 | 18 |
| scan.striped.e33 / basic | `exclusive_scan_warp_striped_kernel` | 0.0009191176470588235 | 2.083333333333333 | 0.004899012304685905 | 0.0832832091796604 | 0.05360546378653114 | 18 |
| scan.striped.e64 / detailed | `exclusive_scan_warp_striped_kernel` | 0.0009191176470588235 | 2.1791666666666667 | 0.004909809526667085 | 0.1967381386861314 | 0.0 | 18 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
