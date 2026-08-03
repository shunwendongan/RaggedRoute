# Triton L1/L2/L3 reference evidence

- Git SHA: `d81f6b015d38` (clean Release build)
- GPU: NVIDIA GeForce RTX 3080 / SM86
- Promotion eligible: **false**
- Profiler durations are diagnostic only; the table uses unprofiled release CUDA Events.

## Cross-backend release

| Level | Target | Triton p50 (us) | CUDA p50 (us) | CUDA vs Triton | Triton p95 (us) | CUDA p95 (us) | Triton CV | CUDA CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L1_kernel_body | dense_gemm | 25.1904 | 26.1120 | 0.9647x | 30.1466 | 28.8614 | 0.1037 | 0.0774 |
| L2_operator_steady | dense_gemm | 24.4736 | 27.2384 | 0.8985x | 29.0560 | 27.8528 | 0.0944 | 0.0688 |
| L1_kernel_body | topk_gate | 12.4314 | 14.3360 | 0.8671x | 23.7926 | 15.3938 | 0.3153 | 0.0838 |
| L2_operator_steady | topk_gate | 12.1446 | 15.9949 | 0.7593x | 14.1691 | 18.1125 | 0.0745 | 0.0703 |
| L1_kernel_body | histogram | 22.5280 | 9.2160 | 2.4444x | 64.6144 | 41.6256 | 0.5386 | 0.7301 |
| L2_operator_steady | histogram | 20.2035 | 15.3190 | 1.3189x | 21.9013 | 16.8479 | 0.0634 | 0.0532 |
| L1_kernel_body | exclusive_scan | 10.2298 | 8.7194 | 1.1732x | 11.5768 | 11.3603 | 0.0582 | 0.1372 |
| L2_operator_steady | exclusive_scan | 10.1581 | 8.4326 | 1.2046x | 10.8959 | 11.2650 | 0.0419 | 0.1377 |
| L1_kernel_body | token_permute | 19.4560 | 9.2160 | 2.1111x | 61.6448 | 32.8704 | 0.5152 | 0.6439 |
| L2_operator_steady | token_permute | 28.4672 | 9.4208 | 3.0217x | 33.9046 | 27.4842 | 0.1204 | 0.4349 |
| L1_kernel_body | grouped_gemm | 73.5232 | 53.1456 | 1.3834x | 87.7670 | 55.9514 | 0.0789 | 0.1030 |
| L2_operator_steady | grouped_gemm | 79.1552 | 47.7184 | 1.6588x | 83.6915 | 49.5616 | 0.0447 | 0.0153 |
| L1_kernel_body | unpermute | 14.1568 | 10.0352 | 1.4107x | 22.9299 | 13.5603 | 0.2319 | 0.1806 |
| L2_operator_steady | unpermute | 13.6448 | 10.0096 | 1.3632x | 21.5066 | 11.3946 | 0.2337 | 0.1104 |
| L3_chain_steady | chain_from_tokens | 249.2416 | 122.1632 | 2.0402x | 260.7104 | 131.2666 | 0.0290 | 0.0329 |

## Profiler coverage

- NSYS: 7 L1 + 7 L2 + one complete seven-operator L3 chain.
- NCU basic: 7 L1 + 7 L2 + 7 emitted L3 operator kernels.
- Raw `.nsys-rep`/`.ncu-rep` files remain under the reproducible local `out/profile` run; their sizes and SHA256 values are recorded in `profile/manifest.json`.
- SQLite exports and raw Nsight binaries are intentionally not committed.

## L3 NSYS kernel-time breakdown

| Time (%) | Total time (ns) | Instances | Average (ns) | Kernel |
|---:|---:|---:|---:|---|
| 73.50 | 404222726 | 1114 | 362857.00 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<char *, (unsigned long long)1>>(int, T2, T3)` |
| 25.50 | 140439244 | 662 | 212143.90 | `_grouped_gemm_kernel` |
| 0.90 | 4927680 | 743 | 6632.10 | `_dense_gemm_kernel` |
| 0.00 | 44448 | 22 | 2020.40 | `_route_map_kernel` |
| 0.00 | 41183 | 22 | 1872.00 | `_unpermute_kernel` |
| 0.00 | 40640 | 22 | 1847.30 | `_permute_rows_kernel` |
| 0.00 | 40413 | 22 | 1837.00 | `_histogram_kernel` |
| 0.00 | 37631 | 22 | 1710.50 | `_top2_selected_softmax_kernel` |
| 0.00 | 27871 | 22 | 1266.90 | `_exclusive_scan_kernel` |

## NCU basic headline metrics

| Level | Operator | Kernel | Duration (ns) | SM (%) | Memory (%) | DRAM (%) | Achieved occupancy (%) | Registers/thread | Grid | Block |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | dense_gemm | `_dense_gemm_kernel` | 103808.00 | 10.50 | 16.10 | 16.10 | 16.62 | 66 | 16 | 256 |
| L2 | dense_gemm | `_dense_gemm_kernel` | 103584.00 | 10.51 | 16.13 | 16.13 | 16.62 | 66 | 16 | 256 |
| L1 | topk_gate | `_top2_selected_softmax_kernel` | 37504.00 | 11.62 | 43.95 | 43.95 | 25.35 | 19 | 2048 | 32 |
| L2 | topk_gate | `_top2_selected_softmax_kernel` | 37312.00 | 11.58 | 43.87 | 43.87 | 25.10 | 19 | 2048 | 32 |
| L1 | histogram | `_histogram_kernel` | 42272.00 | 0.29 | 2.64 | 2.64 | 7.81 | 44 | 1 | 128 |
| L2 | histogram | `_histogram_kernel` | 42240.00 | 0.28 | 2.64 | 2.64 | 7.81 | 44 | 1 | 128 |
| L1 | exclusive_scan | `_exclusive_scan_kernel` | 13984.00 | 0.01 | 0.77 | 0.77 | 2.08 | 16 | 1 | 32 |
| L2 | exclusive_scan | `_exclusive_scan_kernel` | 14912.00 | 0.01 | 0.74 | 0.74 | 2.08 | 16 | 1 | 32 |
| L1 | token_permute | `_permute_rows_kernel` | 44320.00 | 6.93 | 47.67 | 47.67 | 67.31 | 16 | 1024 | 128 |
| L2 | token_permute | `_route_map_kernel` | 21824.00 | 6.68 | 6.68 | 1.94 | 8.31 | 36 | 64 | 128 |
| L1 | grouped_gemm | `_grouped_gemm_kernel` | 561792.00 | 83.26 | 83.26 | 34.85 | 62.21 | 52 | 1408 | 256 |
| L2 | grouped_gemm | `_grouped_gemm_kernel` | 555296.00 | 82.97 | 82.97 | 35.41 | 62.23 | 52 | 1408 | 256 |
| L1 | unpermute | `_unpermute_kernel` | 115296.00 | 2.20 | 61.81 | 61.81 | 41.51 | 23 | 256 | 256 |
| L2 | unpermute | `_unpermute_kernel` | 111616.00 | 2.19 | 64.11 | 64.11 | 41.26 | 23 | 256 | 256 |
| L3 | dense_gemm | `_dense_gemm_kernel` | 62752.00 | 4.46 | 15.33 | 15.33 | 16.58 | 66 | 8 | 256 |
| L3 | topk_gate | `_top2_selected_softmax_kernel` | 22176.00 | 4.04 | 19.08 | 19.08 | 13.20 | 19 | 512 | 32 |
| L3 | histogram | `_histogram_kernel` | 27296.00 | 0.15 | 1.43 | 1.43 | 7.39 | 38 | 1 | 128 |
| L3 | exclusive_scan | `_exclusive_scan_kernel` | 14624.00 | 0.01 | 0.75 | 0.75 | 2.08 | 16 | 1 | 32 |
| L3 | token_permute | `_route_map_kernel` | 19872.00 | 3.52 | 3.52 | 1.38 | 8.30 | 27 | 64 | 128 |
| L3 | grouped_gemm | `_grouped_gemm_kernel` | 1577152.00 | 86.28 | 86.28 | 12.53 | 64.73 | 52 | 4096 | 256 |
| L3 | unpermute | `_unpermute_kernel` | 36992.00 | 2.94 | 45.19 | 45.19 | 17.46 | 23 | 128 | 256 |

## Validation

| Check | Status | Details |
|---|---|---|
| cmake_release_build | passed | Fresh configure and 67/67 build steps completed |
| ctest_release | passed | 8/8 tests |
| python_script_tests | passed | 34/34 tests |
| native_windows_triton | passed | 8/8 tests; torch 2.12.1+cu130; triton-windows 3.7.1.post27 |
| wsl_triton_compatibility | passed | 8/8 tests; torch 2.13.0+cu130; Triton 3.7.1 |
| compute_sanitizer_memcheck | passed | out/sanitizer/triton-three-levels-d81-final/memcheck.log |
| compute_sanitizer_initcheck | passed | out/sanitizer/triton-three-levels-d81-final/initcheck.log |
| compute_sanitizer_racecheck | passed | out/sanitizer/triton-three-levels-d81-final/racecheck.log |
| compute_sanitizer_synccheck | passed | out/sanitizer/triton-three-levels-d81-final/synccheck.log |
| cross_backend_release | passed | 14 L1/L2 comparisons plus one complete L3 comparison; 3 process runs |
| nsys | passed | 15 traces: 7 L1 + 7 L2 + one seven-operator L3 chain |
| ncu_basic | passed | 21 filtered actions: 7 L1 + 7 L2 + 7 L3 |

## Evidence boundary

- Cross-runtime ratios are reference comparisons only and cannot promote a CUDA candidate.
- NCU duration is replay-affected diagnostic context and is not a release score.
- Metrics absent from NCU basic remain `not_collected`; they are never encoded as zero.

See `profile/profile_summary.json`, `validation/summary.json`, and `manifest.json` for machine-readable evidence.
