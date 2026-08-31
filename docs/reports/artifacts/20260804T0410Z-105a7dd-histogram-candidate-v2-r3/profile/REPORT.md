# RaggedRoute Nsight 诊断：histogram-v2-105a7dd-exclusive

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| crossover-cub_device_histogram / basic | `DeviceHistogramSweepKernel` | 0.029411764705882356 | 24.391014559563597 | 1.3554289124902603 | 0.9916614713216957 | 0.9916614713216957 | 54 |
| crossover-cuda_candidate / basic | `histogram_block_private_kernel` | 0.0392156862745098 | 16.15742114110551 | 0.7143018603494654 | 2.1856892010535556 | 0.22250791139240503 | 21 |
| crossover-cuda_candidate_v1 / basic | `histogram_block_private_kernel` | 0.0392156862745098 | 16.157680101751005 | 0.6839049675091035 | 2.181037504381353 | 0.22180599369085174 | 21 |
| crossover-cuda_naive / basic | `histogram_naive_kernel` | 0.3137254901960784 | 10.366959244404413 | 0.8553747021091783 | 2.298842151546241 | 0.03149649430324277 | 16 |
| large-cub_device_histogram / basic | `DeviceHistogramSweepKernel` | 0.8382352941176471 | 54.94699154328545 | 39.53736624930979 | 40.4083225667528 | 40.4083225667528 | 54 |
| large-cuda_candidate / basic | `histogram_block_private_kernel` | 0.3137254901960784 | 30.784097160379847 | 11.241699545676918 | 26.858346394984324 | 26.858346394984324 | 21 |
| large-cuda_candidate_v1 / basic | `histogram_block_private_kernel` | 0.3137254901960784 | 30.793399485388147 | 10.956710513254059 | 34.42125 | 34.42125 | 21 |
| large-cuda_naive / basic | `histogram_naive_kernel` | 10.03921568627451 | 79.80130825775719 | 0.8216987709394348 | 2.623858108804332 | 1.4929016675700666 | 16 |
| small-cub_device_histogram / basic | `DeviceHistogramSweepKernel` | 0.004901960784313725 | 24.3817669771515 | 0.17945257684166765 | 0.9850123609394313 | 0.9850123609394313 | 54 |
| small-cuda_candidate / basic | `histogram_single_bin_write_kernel` | 0.0009191176470588235 | 2.083333333333333 | 0.0006272423915497905 | 0.2812235692134198 | 0.16571969696969696 | 16 |
| small-cuda_candidate_v1 / basic | `histogram_single_cta_shared_kernel` | 0.0024509803921568627 | 16.082068233757756 | 0.09483548227999718 | 0.2975380428058889 | 0.1388888888888889 | 28 |
| small-cuda_naive / basic | `histogram_naive_kernel` | 0.0392156862745098 | 12.556454939162712 | 0.5925022327749581 | 1.8308206514009389 | 0.0914440203562341 | 16 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
