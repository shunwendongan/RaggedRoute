# RaggedRoute Nsight 诊断：ncu-detailed

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| permute.v2.direct.large_uniform / detailed | `token_permute_token_tile4_top2_kernel` | 0.6274509803921569 | 41.70847148688701 | 3.517263726334943 | 48.55271668822769 | 48.55271668822769 | 34 |
| permute.v2.old.large_uniform / detailed | `token_permute_token_owned_top2_kernel` | 2.5098039215686274 | 63.55022604665456 | 6.8180451147064325 | 66.25466417910447 | 66.25466417910447 | 34 |
| permute.v2.prepare.large_uniform / detailed | `token_permute_prepare_offsets_fused_kernel` | 0.0024509803921568627 | 15.435272862120117 | 0.4815716033598659 | 0.38148674474445043 | 0.3234375 | 40 |
| permute.v2.vllm.expand.large_uniform / detailed | `expand_rows` | 10.03921568627451 | 66.15161550719534 | 17.29648571294515 | 52.03419811320755 | 52.03419811320755 | 18 |
| permute.v2.warp.large_uniform / detailed | `token_permute_token_tile4_top2_kernel` | 0.6274509803921569 | 42.1651986555326 | 4.280762046028132 | 63.128682659932664 | 63.128682659932664 | 34 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
