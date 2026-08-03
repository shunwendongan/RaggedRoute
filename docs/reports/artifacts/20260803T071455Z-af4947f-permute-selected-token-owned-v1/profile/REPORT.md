# RaggedRoute Nsight 诊断：ncu

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| permute.candidate.anchor.detailed / detailed | `token_permute_token_owned_top2_kernel` | 0.6274509803921569 | 41.75028973406774 | 2.150864238455912 | 10.904213659147869 | 10.904213659147869 | 34 |
| permute.candidate.wide_hot.detailed / detailed | `token_permute_token_owned_top2_kernel` | 2.5098039215686274 | 86.1453390501686 | 4.805751122295828 | 85.2361709770115 | 85.2361709770115 | 34 |
| permute.naive.anchor.detailed / detailed | `token_permute_naive_kernel` | 1.2549019607843137 | 74.77811484020592 | 15.462295464916668 | 11.527160101651843 | 11.527160101651843 | 26 |
| permute.naive.wide_hot.detailed / detailed | `token_permute_naive_kernel` | 5.019607843137255 | 95.64635348403809 | 14.20608354469104 | 81.62505155347813 | 81.62505155347813 | 26 |
| permute.token_owned.anchor / basic | `token_permute_token_owned_top2_kernel` | 0.6274509803921569 | 46.95637858509028 | 3.9313944568588868 | 10.57571964956195 | 10.57571964956195 | 34 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
