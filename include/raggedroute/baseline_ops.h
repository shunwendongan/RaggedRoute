#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace raggedroute::ops {

// These launchers are deliberately simple CUDA baselines. They provide a real,
// stable target for the benchmark adapters while optimized variants are added.
// All pointers are device pointers and all work is enqueued on caller_stream.

cudaError_t launch_dense_gemm_naive(const float* a, const float* b, float* c, int m, int n, int k,
                                    cudaStream_t caller_stream);

cudaError_t launch_topk_gate_naive(const float* logits, std::int32_t* expert_ids, float* weights,
                                   int tokens, int experts, cudaStream_t caller_stream);

cudaError_t launch_histogram_naive(const std::int32_t* expert_ids, std::int32_t* counts,
                                   int route_pairs, int experts, cudaStream_t caller_stream);

cudaError_t launch_exclusive_scan_naive(const std::int32_t* counts, std::int32_t* offsets,
                                        int experts, cudaStream_t caller_stream);

cudaError_t launch_token_permute_naive(const float* x, const std::int32_t* expert_ids,
                                       const std::int32_t* offsets, std::int32_t* cursors,
                                       float* x_permuted, std::int32_t* route_pos,
                                       std::int32_t* sorted_route, int tokens, int top_k,
                                       int hidden, cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_naive(const float* x_permuted, const float* expert_weights,
                                      const std::int32_t* offsets, float* y_permuted, int experts,
                                      int hidden, int output, int max_expert_tokens,
                                      cudaStream_t caller_stream);

cudaError_t launch_unpermute_naive(const float* y_permuted, const std::int32_t* route_pos,
                                   const float* route_weights, float* y, int tokens, int top_k,
                                   int output, cudaStream_t caller_stream);

cudaError_t launch_cache_scrub(std::uint32_t* buffer, std::size_t elements,
                               cudaStream_t caller_stream);

}  // namespace raggedroute::ops
