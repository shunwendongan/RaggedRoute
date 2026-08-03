# RTX 3080 Top-K Gate Release evidence

> Latency and speedup below come from unprofiled Release benchmarks. Nsight duration is diagnostic only.

- Raw records: 13370; aggregate groups: 2674; independent process runs/group: 5.
- GPU UUIDs: 7c5e95c0e5a415d824a0c8c8b58d6f39.
- Auto decision: **cuda_naive**; promoted intervals: 0.

## Promotion result

No candidate passed a continuous E-bucket/T interval at both L1 and L2. Auto therefore remains `cuda_naive`.

| Level | Candidate | Bucket | Best T min | Ratio-of-sums | Max p50 regression | Max p95 regression | Max CV | Passed |
|---|---|---|---:|---:|---:|---:|---:|---|
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=2 | 64 | 0.971 | 0.098 | 0.047 | 0.087 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=3-8 | 2 | 0.929 | 0.208 | 7.460 | 1.225 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=9-16 | 1 | 0.941 | 0.259 | 5.550 | 1.089 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=17-32 | 32 | 1.064 | 0.360 | 0.435 | 0.407 | no |
| L1_kernel_body | `cuda_warp_pair_top2_v1` | E=33-64 | 128 | 1.193 | 0.310 | 0.911 | 0.640 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=2 | 64 | 1.019 | 0.028 | 0.006 | 0.087 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=3-8 | 4096 | 0.966 | 0.098 | 0.082 | 0.071 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=9-16 | 32 | 0.970 | 0.253 | 4.740 | 1.070 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=17-32 | 256 | 1.083 | 0.175 | 1.650 | 0.504 | no |
| L1_kernel_body | `cuda_subwarp_pair_top2_v2` | E=33-64 | 128 | 1.184 | 0.317 | 1.034 | 0.640 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=2 | 64 | 1.029 | -0.009 | -0.005 | 0.113 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=3-8 | 2 | 0.956 | 0.117 | 4.619 | 1.001 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=9-16 | 128 | 0.970 | 0.195 | 0.301 | 0.238 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=17-32 | 256 | 1.099 | 0.391 | 0.502 | 0.184 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | E=33-64 | 128 | 1.215 | 0.310 | 0.302 | 0.640 | no |
| L1_kernel_body | `cuda_vector_pair_top2_v3` | aligned E={8,16,32,64} | 256 | 1.213 | 0.101 | 0.152 | 0.640 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=2 | 128 | 0.972 | 0.075 | 0.147 | 0.076 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=3-8 | 1 | 0.928 | 0.215 | 4.601 | 0.923 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=9-16 | 16 | 0.945 | 0.252 | 7.438 | 1.208 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=17-32 | 32 | 1.075 | 0.423 | 5.875 | 1.173 | no |
| L2_operator_steady | `cuda_warp_pair_top2_v1` | E=33-64 | 128 | 1.171 | 0.425 | 0.253 | 0.676 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=2 | 2048 | 1.041 | -0.020 | 0.001 | 0.082 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=3-8 | 1 | 0.946 | 0.133 | 0.235 | 0.625 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=9-16 | 32 | 0.970 | 0.258 | 0.306 | 0.211 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=17-32 | 32 | 1.077 | 0.430 | 0.448 | 1.173 | no |
| L2_operator_steady | `cuda_subwarp_pair_top2_v2` | E=33-64 | 64 | 1.166 | 0.409 | 0.303 | 0.676 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=2 | 256 | 1.013 | 0.005 | 0.019 | 0.081 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=3-8 | 1 | 0.954 | 0.126 | 0.309 | 0.625 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=9-16 | 32 | 0.978 | 0.191 | 0.178 | 0.211 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=17-32 | 32 | 1.097 | 0.354 | 0.437 | 1.173 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | E=33-64 | 128 | 1.189 | 0.437 | 1.639 | 0.676 | no |
| L2_operator_steady | `cuda_vector_pair_top2_v3` | aligned E={8,16,32,64} | 1024 | 1.217 | 0.073 | 0.052 | 0.161 | no |

## Strict external pairing

| Candidate | Denominator | Shapes | Ratio-of-sums | Max regression | Overall claim |
|---|---|---:|---:|---:|---|
| `cuda_subwarp_pair_top2_v2` | `cub_block_radix_top2` | 195 | 2.221 | -0.251 | yes |
| `cuda_vector_pair_top2_v3` | `cub_block_radix_top2` | 195 | 2.234 | -0.263 | yes |
| `cuda_subwarp_pair_top2_v2` | `vllm_row_packed_top2` | 78 | 0.925 | 0.413 | mixed |
| `cuda_vector_pair_top2_v3` | `vllm_row_packed_top2` | 78 | 0.944 | 0.130 | mixed |

## Representative L1 metrics

| T | E | Variant | p50 us | p90 us | p95 us | CV | Mrows/s | Effective GB/s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 64 | `cuda_naive` | 9.4771 | 10.0270 | 10.4858 | 0.053 | 3.377 | 0.918 |
| 32 | 64 | `cuda_subwarp_pair_top2_v2` | 8.5043 | 9.4423 | 9.5539 | 0.074 | 3.763 | 1.023 |
| 32 | 64 | `cuda_vector_pair_top2_v3` | 8.6630 | 9.3092 | 9.4674 | 0.057 | 3.694 | 1.005 |
| 32 | 64 | `cuda_warp_pair_top2_v1` | 8.4019 | 9.1146 | 9.5391 | 0.068 | 3.809 | 1.036 |
| 2048 | 8 | `cuda_naive` | 8.5606 | 9.0030 | 9.2058 | 0.054 | 239.234 | 11.483 |
| 2048 | 8 | `cuda_subwarp_pair_top2_v2` | 8.9344 | 9.4740 | 9.6502 | 0.045 | 229.226 | 11.003 |
| 2048 | 8 | `cuda_vector_pair_top2_v3` | 8.7245 | 9.2170 | 9.3809 | 0.068 | 234.742 | 11.268 |
| 2048 | 8 | `cuda_warp_pair_top2_v1` | 10.0352 | 10.6025 | 10.7679 | 0.086 | 204.082 | 9.796 |
| 2048 | 33 | `cuda_naive` | 10.3168 | 10.8554 | 10.9875 | 0.044 | 198.511 | 29.380 |
| 2048 | 33 | `cuda_subwarp_pair_top2_v2` | 9.9430 | 10.7254 | 10.8805 | 0.052 | 205.973 | 30.484 |
| 2048 | 33 | `cuda_vector_pair_top2_v3` | 9.9072 | 10.6414 | 10.7827 | 0.058 | 206.718 | 30.594 |
| 2048 | 33 | `cuda_warp_pair_top2_v1` | 9.9686 | 10.7960 | 10.9701 | 0.063 | 205.444 | 30.406 |
| 2048 | 64 | `cuda_naive` | 14.0544 | 16.2202 | 18.0859 | 0.103 | 145.719 | 39.636 |
| 2048 | 64 | `cuda_subwarp_pair_top2_v2` | 9.6358 | 10.4980 | 10.6803 | 0.070 | 212.540 | 57.811 |
| 2048 | 64 | `cuda_vector_pair_top2_v3` | 9.9379 | 10.4090 | 10.5733 | 0.039 | 206.079 | 56.054 |
| 2048 | 64 | `cuda_warp_pair_top2_v1` | 9.6154 | 10.5902 | 10.9261 | 0.072 | 212.993 | 57.934 |
