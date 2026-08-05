# RTX 3080 Top-K Gate Release evidence

> Latency and speedup below come from unprofiled Release benchmarks. Nsight duration is diagnostic only.

- Raw records: 16710; aggregate groups: 3342; independent process runs/group: 5.
- GPU UUIDs: 7c5e95c0e5a415d824a0c8c8b58d6f39.
- Auto decision: **cuda_naive**; promoted intervals: 0.

## Promotion result

V4 passed no shape-specific interval at both L1/L2 and the strong-library envelope; Auto remains `cuda_naive`.

| Level | Candidate | Bucket | Best T min | Ratio-of-sums | Max p50 regression | Max p95 regression | Max CV | Passed |
|---|---|---|---:|---:|---:|---:|---:|---|
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=2 | 8 | 0.961 | 0.092 | 3.069 | 1.209 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=3-8 | 2 | 0.925 | 0.241 | 4.486 | 1.147 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=9-16 | 16 | 0.954 | 0.276 | 1.436 | 1.227 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=17-32 | 32 | 1.082 | 0.376 | 2.242 | 1.177 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=33-64 | 128 | 1.177 | 0.336 | 1.195 | 1.064 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=2 | 2048 | 1.018 | -0.010 | -0.011 | 0.444 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=3-8 | 4096 | 0.963 | 0.124 | 1.260 | 0.481 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=9-16 | 8 | 0.976 | 0.278 | 2.416 | 1.227 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=17-32 | 64 | 1.088 | 0.326 | 4.514 | 1.177 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=33-64 | 64 | 1.168 | 0.329 | 5.350 | 1.076 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=2 | 2048 | 1.041 | -0.014 | -0.010 | 0.444 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=3-8 | 2 | 0.951 | 0.139 | 5.272 | 1.236 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=9-16 | 32 | 0.981 | 0.247 | 2.774 | 1.260 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=17-32 | 128 | 1.099 | 0.370 | 3.829 | 1.177 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=33-64 | 128 | 1.195 | 0.363 | 4.584 | 1.064 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | aligned E={8,16,32,64} | 4096 | 1.219 | 0.060 | 5.272 | 0.997 | no |
| L1_kernel_body | `cuda_local_pair_two_reduce_top2_v4` | E=2 | 4096 | 1.026 | -0.025 | 0.004 | 0.448 | no |
| L1_kernel_body | `cuda_local_pair_two_reduce_top2_v4` | E=3-8 | 4096 | 0.974 | 0.103 | 1.260 | 0.481 | no |
| L1_kernel_body | `cuda_local_pair_two_reduce_top2_v4` | E=9-16 | 1 | 0.990 | 0.238 | 5.767 | 1.227 | no |
| L1_kernel_body | `cuda_local_pair_two_reduce_top2_v4` | E=17-32 | 32 | 1.108 | 0.365 | 2.813 | 1.247 | no |
| L1_kernel_body | `cuda_local_pair_two_reduce_top2_v4` | E=33-64 | 128 | 1.199 | 0.319 | 3.954 | 1.169 | no |
| L1_kernel_body | `cuda_local_pair_two_reduce_top2_v4` | aligned E={8,16,32,64} | 256 | 1.253 | 0.094 | 2.915 | 1.267 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=2 | 1024 | 0.968 | 0.077 | 4.315 | 1.002 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=3-8 | 1 | 0.923 | 0.205 | 6.478 | 1.233 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=9-16 | 16 | 0.947 | 0.262 | 7.336 | 1.197 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=17-32 | 64 | 1.084 | 0.351 | 2.798 | 1.113 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=33-64 | 1 | 1.156 | 0.312 | 6.246 | 1.229 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=2 | 2 | 1.013 | 0.075 | 4.145 | 0.917 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=3-8 | 4096 | 0.980 | 0.107 | 0.037 | 1.130 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=9-16 | 8 | 0.980 | 0.287 | 7.530 | 1.218 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=17-32 | 32 | 1.081 | 0.411 | 2.093 | 1.154 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=33-64 | 4 | 1.163 | 0.319 | 3.055 | 1.229 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=2 | 4096 | 1.062 | -0.058 | -0.006 | 0.456 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=3-8 | 512 | 0.955 | 0.126 | 3.827 | 1.237 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=9-16 | 32 | 0.985 | 0.195 | 2.831 | 1.226 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=17-32 | 128 | 1.104 | 0.325 | 6.172 | 1.189 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=33-64 | 128 | 1.185 | 0.310 | 5.288 | 1.170 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | aligned E={8,16,32,64} | 512 | 1.216 | 0.099 | 2.638 | 1.204 | no |
| L2_operator_steady | `cuda_local_pair_two_reduce_top2_v4` | E=2 | 256 | 1.010 | 0.028 | 1.044 | 0.471 | no |
| L2_operator_steady | `cuda_local_pair_two_reduce_top2_v4` | E=3-8 | 4096 | 0.973 | 0.108 | 2.657 | 1.130 | no |
| L2_operator_steady | `cuda_local_pair_two_reduce_top2_v4` | E=9-16 | 16 | 0.999 | 0.193 | 7.457 | 1.132 | no |
| L2_operator_steady | `cuda_local_pair_two_reduce_top2_v4` | E=17-32 | 128 | 1.109 | 0.383 | 4.351 | 1.214 | no |
| L2_operator_steady | `cuda_local_pair_two_reduce_top2_v4` | E=33-64 | 128 | 1.192 | 0.306 | 5.249 | 1.143 | no |
| L2_operator_steady | `cuda_local_pair_two_reduce_top2_v4` | aligned E={8,16,32,64} | 256 | 1.252 | 0.100 | 5.160 | 1.104 | no |

## Strict external pairing

| Candidate | Denominator | Shapes | Ratio-of-sums | Max regression | Overall claim |
|---|---|---:|---:|---:|---|
| `cuda_subwarp_pair_top2_v2` | `cub_block_radix_top2` | 195 | 2.235 | -0.230 | yes |
| `cuda_vector_pair_top2_v3` | `cub_block_radix_top2` | 195 | 2.255 | -0.232 | yes |
| `cuda_local_pair_two_reduce_top2_v4` | `cub_block_radix_top2` | 195 | 2.272 | -0.241 | yes |
| `cuda_subwarp_pair_top2_v2` | `vllm_row_packed_top2` | 78 | 0.935 | 0.290 | mixed |
| `cuda_vector_pair_top2_v3` | `vllm_row_packed_top2` | 78 | 0.948 | 0.165 | mixed |
| `cuda_local_pair_two_reduce_top2_v4` | `vllm_row_packed_top2` | 78 | 0.974 | 0.210 | mixed |

## Representative L1 metrics

| T | E | Variant | p50 us | p90 us | p95 us | CV | Mrows/s | Effective GB/s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 64 | `cuda_local_pair_two_reduce_top2_v4` | 8.2125 | 8.9702 | 9.4797 | 0.216 | 3.897 | 1.060 |
| 32 | 64 | `cuda_naive` | 9.4515 | 17.4070 | 19.7105 | 0.330 | 3.386 | 0.921 |
| 32 | 64 | `cuda_subwarp_pair_top2_v2` | 8.5965 | 21.4641 | 21.5972 | 0.465 | 3.722 | 1.013 |
| 32 | 64 | `cuda_vector_pair_top2_v3` | 8.2688 | 21.4630 | 21.5880 | 0.493 | 3.870 | 1.053 |
| 32 | 64 | `cuda_warp_pair_top2_v1` | 8.2637 | 8.8689 | 8.9656 | 0.056 | 3.872 | 1.053 |
| 2048 | 8 | `cuda_local_pair_two_reduce_top2_v4` | 8.4531 | 82.3665 | 82.7663 | 1.267 | 242.277 | 11.629 |
| 2048 | 8 | `cuda_naive` | 8.5658 | 21.0637 | 21.1410 | 0.460 | 239.091 | 11.476 |
| 2048 | 8 | `cuda_subwarp_pair_top2_v2` | 9.0522 | 20.8180 | 20.8936 | 0.436 | 226.244 | 10.860 |
| 2048 | 8 | `cuda_vector_pair_top2_v3` | 8.9702 | 20.9347 | 21.1681 | 0.428 | 228.311 | 10.959 |
| 2048 | 8 | `cuda_warp_pair_top2_v1` | 10.1990 | 22.6120 | 22.7123 | 0.397 | 200.803 | 9.639 |
| 2048 | 33 | `cuda_local_pair_two_reduce_top2_v4` | 10.1888 | 22.5403 | 24.1731 | 0.427 | 201.005 | 29.749 |
| 2048 | 33 | `cuda_naive` | 10.4141 | 10.9179 | 11.0285 | 0.053 | 196.657 | 29.105 |
| 2048 | 33 | `cuda_subwarp_pair_top2_v2` | 9.8560 | 10.8554 | 11.3577 | 0.079 | 207.792 | 30.753 |
| 2048 | 33 | `cuda_vector_pair_top2_v3` | 9.8150 | 10.6301 | 10.8288 | 0.062 | 208.659 | 30.882 |
| 2048 | 33 | `cuda_warp_pair_top2_v1` | 10.0864 | 23.8961 | 24.2074 | 0.433 | 203.046 | 30.051 |
| 2048 | 64 | `cuda_local_pair_two_reduce_top2_v4` | 8.3968 | 21.7815 | 21.9305 | 0.494 | 243.902 | 66.341 |
| 2048 | 64 | `cuda_naive` | 14.5869 | 26.3506 | 30.3800 | 0.335 | 140.400 | 38.189 |
| 2048 | 64 | `cuda_subwarp_pair_top2_v2` | 10.0352 | 22.9734 | 23.2264 | 0.404 | 204.082 | 55.510 |
| 2048 | 64 | `cuda_vector_pair_top2_v3` | 9.8816 | 10.3639 | 10.6276 | 0.050 | 207.254 | 56.373 |
| 2048 | 64 | `cuda_warp_pair_top2_v1` | 9.6614 | 23.0840 | 23.2330 | 0.468 | 211.977 | 57.658 |
