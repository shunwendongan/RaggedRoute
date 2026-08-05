# RaggedRoute Nsight 诊断：ncu-basic

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| permute.v2.direct.large_uniform / basic | `token_permute_token_tile4_top2_kernel` | 0.6274509803921569 | 41.66647254800689 | 5.09929667228115 | 53.19056721194879 | 53.19056721194879 | 34 |
| permute.v2.old.large_uniform / basic | `token_permute_token_owned_top2_kernel` | 2.5098039215686274 | 66.9375460671646 | 5.030258275063556 | 58.03331413210445 | 58.03331413210445 | 34 |
| permute.v2.prepare.large_uniform / basic | `token_permute_prepare_offsets_fused_kernel` | 0.0024509803921568627 | 15.465806981164679 | 0.47793621059209246 | 0.3735548430972506 | 0.33360228401191655 | 40 |
| permute.v2.vllm.expand.large_uniform / basic | `expand_rows` | 10.03921568627451 | 66.30688090736362 | 13.745940241851587 | 50.357920294708634 | 50.357920294708634 | 18 |
| permute.v2.vllm.sort.large_uniform / basic | `DeviceRadixSortSingleTileKernel` | 0.007352941176470588 | 16.574591600039483 | 0.8478256764667007 | 1.0055478502080446 | 1.0055478502080446 | 128 |
| permute.v2.warp.large_uniform / basic | `token_permute_token_tile4_top2_kernel` | 0.6274509803921569 | 42.275475601155485 | 3.3010955710955714 | 48.559712346477056 | 48.559712346477056 | 34 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
