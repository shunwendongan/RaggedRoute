# Triton L1/L2/L3 profiler summary

Profiler durations are diagnostic only; unprofiled release JSONL is the performance source.

- NSYS cases: 15 (7 L1 + 7 L2 + 1 L3 chain)
- NCU basic actions: 21 (7 L1 + 7 L2 + 7 L3)
- Promotion eligible: false

## NCU kernels

| Level | Operator | Emitted kernel |
|---|---|---|
| L1 | dense_gemm | `_dense_gemm_kernel` |
| L2 | dense_gemm | `_dense_gemm_kernel` |
| L1 | topk_gate | `_top2_selected_softmax_kernel` |
| L2 | topk_gate | `_top2_selected_softmax_kernel` |
| L1 | histogram | `_histogram_kernel` |
| L2 | histogram | `_histogram_kernel` |
| L1 | exclusive_scan | `_exclusive_scan_kernel` |
| L2 | exclusive_scan | `_exclusive_scan_kernel` |
| L1 | token_permute | `_permute_rows_kernel` |
| L2 | token_permute | `_route_map_kernel` |
| L1 | grouped_gemm | `_grouped_gemm_kernel` |
| L2 | grouped_gemm | `_grouped_gemm_kernel` |
| L1 | unpermute | `_unpermute_kernel` |
| L2 | unpermute | `_unpermute_kernel` |
| L3 | dense_gemm | `_dense_gemm_kernel` |
| L3 | topk_gate | `_top2_selected_softmax_kernel` |
| L3 | histogram | `_histogram_kernel` |
| L3 | exclusive_scan | `_exclusive_scan_kernel` |
| L3 | token_permute | `_route_map_kernel` |
| L3 | grouped_gemm | `_grouped_gemm_kernel` |
| L3 | unpermute | `_unpermute_kernel` |
