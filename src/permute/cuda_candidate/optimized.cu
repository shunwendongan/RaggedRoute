#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr int kMaxExperts = 64;
constexpr int kBlockPartialThreads = 256;

bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

template <int Threads, bool Vectorized>
__global__ void token_permute_atomic_vectorized_kernel(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int route_pairs, int top_k, int hidden) {
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
  if constexpr (Vectorized) {
    const int vectors = hidden / 4;
    const auto* source =
        reinterpret_cast<const float4*>(x + static_cast<std::size_t>(token) * hidden);
    auto* target = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destination) * hidden);
    for (int vector = static_cast<int>(threadIdx.x); vector < vectors; vector += Threads) {
      target[vector] = source[vector];
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < hidden; column += Threads) {
      x_permuted[static_cast<std::size_t>(destination) * hidden + column] =
          x[static_cast<std::size_t>(token) * hidden + column];
    }
  }
}

template <int Threads, bool Vectorized>
__global__ void token_permute_token_owned_top2_kernel(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int hidden) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;

  __shared__ int destinations[2];
  if (threadIdx.x == 0) {
    const int route0 = token * 2;
    const int route1 = route0 + 1;
    const int expert0 = expert_ids[route0];
    const int expert1 = expert_ids[route1];
    destinations[0] = offsets[expert0] + atomicAdd(cursors + expert0, 1);
    destinations[1] = offsets[expert1] + atomicAdd(cursors + expert1, 1);
    route_pos[route0] = destinations[0];
    route_pos[route1] = destinations[1];
    if (sorted_route != nullptr) {
      sorted_route[destinations[0]] = route0;
      sorted_route[destinations[1]] = route1;
    }
  }
  __syncthreads();

  if constexpr (Vectorized) {
    const int vectors = hidden / 4;
    const auto* source =
        reinterpret_cast<const float4*>(x + static_cast<std::size_t>(token) * hidden);
    auto* target0 = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destinations[0]) * hidden);
    auto* target1 = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destinations[1]) * hidden);
    for (int vector = static_cast<int>(threadIdx.x); vector < vectors; vector += Threads) {
      const float4 value = source[vector];
      target0[vector] = value;
      target1[vector] = value;
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < hidden; column += Threads) {
      const float value = x[static_cast<std::size_t>(token) * hidden + column];
      x_permuted[static_cast<std::size_t>(destinations[0]) * hidden + column] = value;
      x_permuted[static_cast<std::size_t>(destinations[1]) * hidden + column] = value;
    }
  }
}

__global__ void token_permute_block_partial_placement_kernel(
    const std::int32_t* expert_ids, const std::int32_t* offsets, std::int32_t* cursors,
    std::int32_t* route_pos, std::int32_t* sorted_route, int route_pairs, int experts) {
  __shared__ int counts[kMaxExperts];
  __shared__ int bases[kMaxExperts];

  const int thread = static_cast<int>(threadIdx.x);
  if (thread < experts) {
    counts[thread] = 0;
    bases[thread] = 0;
  }
  __syncthreads();

  const int route = static_cast<int>(blockIdx.x) * blockDim.x + thread;
  int expert = 0;
  int local_rank = 0;
  if (route < route_pairs) {
    expert = expert_ids[route];
    local_rank = atomicAdd(counts + expert, 1);
  }
  __syncthreads();

  if (thread < experts) {
    const int count = counts[thread];
    if (count != 0) bases[thread] = atomicAdd(cursors + thread, count);
  }
  __syncthreads();

  if (route < route_pairs) {
    const int destination = offsets[expert] + bases[expert] + local_rank;
    route_pos[route] = destination;
    if (sorted_route != nullptr) sorted_route[destination] = route;
  }
}

template <int Threads, bool Vectorized>
__global__ void token_permute_copy_from_positions_kernel(
    const float* x, const std::int32_t* route_pos, float* x_permuted, int route_pairs,
    int top_k, int hidden) {
  const int route = static_cast<int>(blockIdx.x);
  if (route >= route_pairs) return;
  const int destination = route_pos[route];
  const int token = route / top_k;
  if constexpr (Vectorized) {
    const int vectors = hidden / 4;
    const auto* source =
        reinterpret_cast<const float4*>(x + static_cast<std::size_t>(token) * hidden);
    auto* target = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destination) * hidden);
    for (int vector = static_cast<int>(threadIdx.x); vector < vectors; vector += Threads) {
      target[vector] = source[vector];
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < hidden; column += Threads) {
      x_permuted[static_cast<std::size_t>(destination) * hidden + column] =
          x[static_cast<std::size_t>(token) * hidden + column];
    }
  }
}

cudaError_t validate_arguments(const float* x, const std::int32_t* expert_ids,
                               const std::int32_t* offsets, std::int32_t* cursors,
                               float* x_permuted, std::int32_t* route_pos, int tokens,
                               int experts, int top_k, int hidden,
                               std::uint32_t implementation_id) {
  if (tokens < 0 || experts <= 0 || experts > kMaxExperts || top_k <= 0 ||
      top_k > experts || hidden < 0 ||
      !is_token_permute_optimized_implementation(implementation_id)) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0) return cudaSuccess;
  if (tokens > std::numeric_limits<int>::max() / top_k) return cudaErrorInvalidValue;
  if (expert_ids == nullptr || offsets == nullptr || cursors == nullptr ||
      route_pos == nullptr ||
      (hidden != 0 && (x == nullptr || x_permuted == nullptr))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

template <int Threads>
cudaError_t launch_atomic_vectorized(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int route_pairs, int top_k, int hidden,
    cudaStream_t stream) {
  const bool vectorized =
      hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
  if (vectorized) {
    token_permute_atomic_vectorized_kernel<Threads, true>
        <<<route_pairs, Threads, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted,
                                              route_pos, sorted_route, route_pairs, top_k, hidden);
  } else {
    token_permute_atomic_vectorized_kernel<Threads, false>
        <<<route_pairs, Threads, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted,
                                              route_pos, sorted_route, route_pairs, top_k, hidden);
  }
  return cudaGetLastError();
}

cudaError_t launch_token_owned_top2(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int top_k, int hidden, cudaStream_t stream) {
  if (top_k != 2) {
    return launch_atomic_vectorized<128>(x, expert_ids, offsets, cursors, x_permuted,
                                         route_pos, sorted_route, tokens * top_k, top_k, hidden,
                                         stream);
  }
  const bool vectorized =
      hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
  if (vectorized) {
    token_permute_token_owned_top2_kernel<128, true>
        <<<tokens, 128, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                    sorted_route, tokens, hidden);
  } else {
    token_permute_token_owned_top2_kernel<128, false>
        <<<tokens, 128, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                    sorted_route, tokens, hidden);
  }
  return cudaGetLastError();
}

cudaError_t launch_block_partial(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int route_pairs, int experts, int top_k, int hidden,
    cudaStream_t stream) {
  const int blocks = (route_pairs + kBlockPartialThreads - 1) / kBlockPartialThreads;
  token_permute_block_partial_placement_kernel<<<blocks, kBlockPartialThreads, 0, stream>>>(
      expert_ids, offsets, cursors, route_pos, sorted_route, route_pairs, experts);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess || hidden == 0) return error;

  const bool vectorized =
      hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
  if (vectorized) {
    token_permute_copy_from_positions_kernel<128, true>
        <<<route_pairs, 128, 0, stream>>>(x, route_pos, x_permuted, route_pairs, top_k, hidden);
  } else {
    token_permute_copy_from_positions_kernel<128, false>
        <<<route_pairs, 128, 0, stream>>>(x, route_pos, x_permuted, route_pairs, top_k, hidden);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_token_permute_optimized(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int experts, int top_k, int hidden,
    std::uint32_t implementation_id, cudaStream_t caller_stream) {
  const cudaError_t validation =
      validate_arguments(x, expert_ids, offsets, cursors, x_permuted, route_pos, tokens,
                         experts, top_k, hidden, implementation_id);
  if (validation != cudaSuccess || tokens == 0) return validation;

  const int route_pairs = tokens * top_k;
  switch (implementation_id) {
    case kTokenPermuteAtomicVectorized128Implementation:
      return launch_atomic_vectorized<128>(x, expert_ids, offsets, cursors, x_permuted,
                                            route_pos, sorted_route, route_pairs, top_k, hidden,
                                            caller_stream);
    case kTokenPermuteAtomicVectorized64Implementation:
      return launch_atomic_vectorized<64>(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                           sorted_route, route_pairs, top_k, hidden,
                                           caller_stream);
    case kTokenPermuteAtomicVectorized256Implementation:
      return launch_atomic_vectorized<256>(x, expert_ids, offsets, cursors, x_permuted,
                                            route_pos, sorted_route, route_pairs, top_k, hidden,
                                            caller_stream);
    case kTokenPermuteTokenOwnedTop2Implementation:
      return launch_token_owned_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                     sorted_route, tokens, top_k, hidden, caller_stream);
    case kTokenPermuteBlockPartialImplementation:
      return launch_block_partial(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                  sorted_route, route_pairs, experts, top_k, hidden,
                                  caller_stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace raggedroute::ops
