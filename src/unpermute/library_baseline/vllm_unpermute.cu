// Adapted from vLLM commit 837eae64580c885101ee95b073aafb27a485e7ce.
// Original files: moe_permute_unpermute_kernel.{h,inl}.
// Copyright The vLLM Team. Licensed under Apache-2.0.
// Modified for a standalone FP32 benchmark baseline and a scalar fallback for
// output widths that do not satisfy the upstream 16-byte vector contract.

#include <cstdint>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

template <bool Vectorized>
__global__ void finalize_routing(const float* permuted_rows, float* output,
                                 const float* route_weights, const std::int32_t* route_pos,
                                 int tokens, int top_k, int output_width) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;
  if constexpr (Vectorized) {
    const int vectors = output_width / 4;
    auto* target =
        reinterpret_cast<float4*>(output + static_cast<std::size_t>(token) * output_width);
    for (int vector = static_cast<int>(threadIdx.x); vector < vectors; vector += blockDim.x) {
      float4 sum = {0.0F, 0.0F, 0.0F, 0.0F};
      for (int rank = 0; rank < top_k; ++rank) {
        const int route = token * top_k + rank;
        const int source = route_pos[route];
        const auto* row = reinterpret_cast<const float4*>(
            permuted_rows + static_cast<std::size_t>(source) * output_width);
        const float4 value = row[vector];
        const float weight = route_weights[route];
        sum.x += weight * value.x;
        sum.y += weight * value.y;
        sum.z += weight * value.z;
        sum.w += weight * value.w;
      }
      target[vector] = sum;
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < output_width; column += blockDim.x) {
      float sum = 0.0F;
      for (int rank = 0; rank < top_k; ++rank) {
        const int route = token * top_k + rank;
        const int source = route_pos[route];
        sum += route_weights[route] *
               permuted_rows[static_cast<std::size_t>(source) * output_width + column];
      }
      output[static_cast<std::size_t>(token) * output_width + column] = sum;
    }
  }
}

}  // namespace

cudaError_t launch_vllm_finalize_routing(const float* permuted_rows, float* output,
                                         const float* route_weights, const std::int32_t* route_pos,
                                         int tokens, int top_k, int output_width,
                                         cudaStream_t stream) {
  if (tokens < 0 || top_k < 1 || output_width < 0 ||
      (tokens != 0 && (permuted_rows == nullptr || output == nullptr || route_weights == nullptr ||
                       route_pos == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0 || output_width == 0) return cudaSuccess;
  if (output_width % 4 == 0 && is_aligned_16(permuted_rows) && is_aligned_16(output)) {
    finalize_routing<true><<<tokens, 256, 0, stream>>>(permuted_rows, output, route_weights,
                                                       route_pos, tokens, top_k, output_width);
  } else {
    finalize_routing<false><<<tokens, 256, 0, stream>>>(permuted_rows, output, route_weights,
                                                        route_pos, tokens, top_k, output_width);
  }
  return cudaPeekAtLastError();
}

}  // namespace raggedroute::benchmark::library_baseline
