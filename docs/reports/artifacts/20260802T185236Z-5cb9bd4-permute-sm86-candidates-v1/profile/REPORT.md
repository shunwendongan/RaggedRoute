# RaggedRoute Nsight 诊断：ncu

> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。

## NSYS kernel 热点

| Rank | Kernel | Time % | Total time |
|---:|---|---:|---:|

## NCU headline metrics

| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |
|---|---|---:|---:|---:|---:|---:|---:|
| permute.atomic128.anchor / basic | `token_permute_atomic_vectorized_kernel` | 1.2549019607843137 | 54.71387515296961 | 8.138000009995203 | 22.944328528072834 | 22.944328528072834 | 38 |
| permute.atomic256.anchor / basic | `token_permute_atomic_vectorized_kernel` | 2.5098039215686274 | 68.85810607846568 | 12.348324863327786 | 20.513279262086517 | 20.513279262086517 | 38 |
| permute.atomic64.anchor / basic | `token_permute_atomic_vectorized_kernel` | 0.9411764705882353 | 40.517929013095014 | 4.762578173196128 | 18.336975524475523 | 18.336975524475523 | 38 |
| permute.block_partial.copy.anchor / basic | `token_permute_copy_from_positions_kernel` | 1.2549019607843137 | 42.06526566139028 | 7.3481053326316585 | 18.52720450281426 | 18.52720450281426 | 34 |
| permute.block_partial.placement.anchor / basic | `token_permute_block_partial_placement_kernel` | 0.00980392156862745 | 16.476383553848343 | 0.06195848384412674 | 0.2829824561403509 | 0.12397670025188914 | 16 |
| permute.candidate.anchor.detailed / detailed | `token_permute_atomic_vectorized_kernel` | 1.2549019607843137 | 64.06357942986615 | 8.17736107102975 | 18.525454775138957 | 17.822890025575447 | 38 |
| permute.candidate.wide_hot.detailed / detailed | `token_permute_atomic_vectorized_kernel` | 5.019607843137255 | 93.8056788984913 | 8.913880980628162 | 85.87526297335202 | 85.87526297335202 | 38 |
| permute.naive.anchor / basic | `token_permute_naive_kernel` | 1.2549019607843137 | 71.1032385079164 | 12.683809199257858 | 31.70955882352941 | 31.70955882352941 | 26 |
| permute.naive.anchor.detailed / detailed | `token_permute_naive_kernel` | 1.2549019607843137 | 74.7157168399748 | 13.538382527069631 | 18.259943181818183 | 18.259943181818183 | 26 |
| permute.naive.wide_hot.detailed / detailed | `token_permute_naive_kernel` | 5.019607843137255 | 91.2163990989945 | 12.943488107343832 | 81.32388032305433 | 81.32388032305433 | 26 |
| permute.token_owned.anchor / basic | `token_permute_token_owned_top2_kernel` | 0.6274509803921569 | 45.30830619868039 | 3.806277908761705 | 20.587008343265794 | 19.62274774774775 | 34 |

## 证据边界

- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。
- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。
- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。
