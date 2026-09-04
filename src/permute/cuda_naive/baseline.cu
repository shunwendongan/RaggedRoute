#include <cuda_runtime.h>

#include <limits>

#include "raggedroute/baseline_ops.h"

// token permute 的朴素 baseline。
// 按 expert_ids[T,top_k] 把 x[T,H] 拷贝到 x_permuted[T*top_k,H]，并同步记录
// route_pos / sorted_route。
namespace raggedroute::ops {
namespace {

__global__ void token_permute_naive_kernel(const float* x, const std::int32_t* expert_ids,
                                           const std::int32_t* offsets, std::int32_t* cursors,
                                           float* x_permuted, std::int32_t* route_pos,
                                           std::int32_t* sorted_route, int route_pairs, int top_k,
                                           int hidden) {
  const int route = static_cast<int>(blockIdx.x);
  if (route >= route_pairs) return;
  __shared__ int destination;
  if (threadIdx.x == 0) {
    const int expert = expert_ids[route];
    destination = offsets[expert] + atomicAdd(cursors + expert, 1);
    route_pos[route] = destination;
    if (sorted_route != nullptr) sorted_route[destination] = route;
  }
  __syncthreads();
  const int token = route / top_k;
  for (int column = static_cast<int>(threadIdx.x); column < hidden;
       column += static_cast<int>(blockDim.x)) {
    x_permuted[static_cast<std::size_t>(destination) * hidden + column] =
        x[static_cast<std::size_t>(token) * hidden + column];
  }
}

}  // namespace

cudaError_t launch_token_permute_naive(const float* x, const std::int32_t* expert_ids,
                                       const std::int32_t* offsets, std::int32_t* cursors,
                                       float* x_permuted, std::int32_t* route_pos,
                                       std::int32_t* sorted_route, int tokens, int top_k,
                                       int hidden, cudaStream_t caller_stream) {
  if (tokens < 0 || top_k <= 0 || hidden < 0) return cudaErrorInvalidValue;
  if (tokens == 0) return cudaSuccess;
  if (tokens > std::numeric_limits<int>::max() / top_k) return cudaErrorInvalidValue;
  const int route_pairs = tokens * top_k;
  if (expert_ids == nullptr || offsets == nullptr || cursors == nullptr || route_pos == nullptr ||
      (hidden != 0 && (x == nullptr || x_permuted == nullptr))) {
    return cudaErrorInvalidValue;
  }
  token_permute_naive_kernel<<<route_pairs, 128, 0, caller_stream>>>(
      x, expert_ids, offsets, cursors, x_permuted, route_pos, sorted_route, route_pairs, top_k,
      hidden);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
