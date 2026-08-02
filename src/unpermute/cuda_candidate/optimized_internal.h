#pragma once

#include <cuda_runtime_api.h>

namespace raggedroute::ops {

// Benchmark/research-only launcher. This candidate is intentionally absent from
// public select_kernel/unpermute dispatch because it did not pass promotion gates.
cudaError_t launch_unpermute_optimized(const float* y_permuted,
                                       const std::int32_t* route_pos,
                                       const float* route_weights, float* y, int tokens,
                                       int top_k, int output,
                                       cudaStream_t caller_stream);

}  // namespace raggedroute::ops
