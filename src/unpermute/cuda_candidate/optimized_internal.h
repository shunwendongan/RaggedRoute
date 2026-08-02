#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kUnpermuteWarpTokenVec4Implementation = 1;

inline bool is_unpermute_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kUnpermuteWarpTokenVec4Implementation;
}

cudaError_t launch_unpermute_optimized(const float* y_permuted,
                                       const std::int32_t* route_pos,
                                       const float* route_weights, float* y, int tokens,
                                       int top_k, int output,
                                       cudaStream_t caller_stream);

}  // namespace raggedroute::ops
