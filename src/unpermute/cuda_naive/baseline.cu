#include <cuda_runtime.h>

#include <limits>

#include "raggedroute/baseline_ops.h"

namespace raggedroute::ops {
namespace {

__global__ void unpermute_naive_kernel(const float* y_permuted, const std::int32_t* route_pos,
                                       const float* route_weights, float* y, int tokens, int top_k,
                                       int output) {
  const int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int elements = tokens * output;
  if (linear >= elements) return;
  const int token = linear / output;
  const int column = linear % output;
  float accumulator = 0.0F;
  for (int rank = 0; rank < top_k; ++rank) {
    const int route = token * top_k + rank;
    const int source_row = route_pos[route];
    accumulator +=
        route_weights[route] * y_permuted[static_cast<std::size_t>(source_row) * output + column];
  }
  y[linear] = accumulator;
}

}  // namespace

cudaError_t launch_unpermute_naive(const float* y_permuted, const std::int32_t* route_pos,
                                   const float* route_weights, float* y, int tokens, int top_k,
                                   int output, cudaStream_t caller_stream) {
  if (tokens < 0 || top_k <= 0 || output < 0) return cudaErrorInvalidValue;
  if (tokens == 0 || output == 0) return cudaSuccess;
  if (tokens > std::numeric_limits<int>::max() / top_k) return cudaErrorInvalidValue;
  if (tokens > std::numeric_limits<int>::max() / output) return cudaErrorInvalidValue;
  if (y_permuted == nullptr || route_pos == nullptr || route_weights == nullptr || y == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int elements = tokens * output;
  constexpr int kThreads = 256;
  const auto blocks =
      static_cast<unsigned int>((static_cast<std::size_t>(elements) + kThreads - 1) / kThreads);
  unpermute_naive_kernel<<<blocks, kThreads, 0, caller_stream>>>(
      y_permuted, route_pos, route_weights, y, tokens, top_k, output);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
