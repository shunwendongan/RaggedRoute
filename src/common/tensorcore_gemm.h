#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "raggedroute/types.h"

namespace raggedroute::ops {

// Internal benchmark/correctness interface. These launchers do not participate
// in the public operator dispatch. Inputs are row-major FP16/BF16; C is FP32.
struct TensorCoreConfig {
  int block_m = 64;
  int block_n = 32;
  int block_k = 32;
  bool asynchronous = true;
};

cudaError_t launch_dense_gemm_tensorcore(
    const void* a, const void* b, float* c, int m, int n, int k,
    ScalarType input_type, const TensorCoreConfig& config, cudaStream_t stream);

cudaError_t launch_grouped_gemm_tensorcore(
    const void* x, const void* weights, const std::int32_t* offsets, float* y,
    int experts, int hidden, int output, int max_expert_tokens,
    ScalarType input_type, const TensorCoreConfig& config, cudaStream_t stream);

cudaError_t launch_dense_gemm_lowp_simt(
    const void* a, const void* b, float* c, int m, int n, int k,
    ScalarType input_type, cudaStream_t stream);

cudaError_t launch_grouped_gemm_lowp_simt(
    const void* x, const void* weights, const std::int32_t* offsets, float* y,
    int experts, int hidden, int output, int max_expert_tokens,
    ScalarType input_type, cudaStream_t stream);

cudaError_t launch_gemm_quantize(const float* source, void* destination,
                               std::size_t elements, ScalarType input_type,
                               cudaStream_t stream);

// The compiled MMA path requires sm_80+. Architecture fallback is selected in
// device code, without a host device query or synchronization in the hot path.
// Benchmarks record the actual device capability to label that fallback.

}  // namespace raggedroute::ops
