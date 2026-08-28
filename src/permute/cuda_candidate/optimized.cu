#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr int kMaxExperts = 64;
constexpr int kBlockPartialThreads = 256;
constexpr int kTokenTileTokens = 4;
constexpr int kTokenTileThreads = 128;
constexpr int kTokenTile2Tokens = 2;
constexpr int kTokenTile2Threads = 64;

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

__device__ __forceinline__ unsigned int lanes_before(int lane) {
  return lane == 0 ? 0u : ((1u << lane) - 1u);
}

template <bool Aggregate, bool Vectorized>
__global__ void token_permute_token_tile4_top2_kernel(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int hidden) {
  __shared__ int destinations[kTokenTileTokens * 2];
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int first_token = static_cast<int>(blockIdx.x) * kTokenTileTokens;
  const int valid_tokens = min(kTokenTileTokens, tokens - first_token);

  if (warp == 0) {
    const int valid_routes = valid_tokens * 2;
    const bool valid = lane < valid_routes;
    const unsigned int active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const int route = first_token * 2 + lane;
      const int expert = expert_ids[route];
      int local_rank = 0;
      int base = 0;
      if constexpr (Aggregate) {
        const unsigned int peers = __match_any_sync(active, expert);
        const int leader = __ffs(static_cast<int>(peers)) - 1;
        local_rank = __popc(peers & lanes_before(lane));
        if (lane == leader) base = atomicAdd(cursors + expert, __popc(peers));
        base = __shfl_sync(peers, base, leader);
      } else {
        base = atomicAdd(cursors + expert, 1);
      }
      const int destination = offsets[expert] + base + local_rank;
      destinations[lane] = destination;
      route_pos[route] = destination;
      if (sorted_route != nullptr) sorted_route[destination] = route;
    }
  }
  __syncthreads();

  if (warp >= valid_tokens) return;
  const int token = first_token + warp;
  const int destination0 = destinations[warp * 2];
  const int destination1 = destinations[warp * 2 + 1];
  if constexpr (Vectorized) {
    const int vectors = hidden / 4;
    const auto* source =
        reinterpret_cast<const float4*>(x + static_cast<std::size_t>(token) * hidden);
    auto* target0 = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destination0) * hidden);
    auto* target1 = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destination1) * hidden);
    for (int vector = lane; vector < vectors; vector += warpSize) {
      const float4 value = source[vector];
      target0[vector] = value;
      target1[vector] = value;
    }
  } else {
    for (int column = lane; column < hidden; column += warpSize) {
      const float value = x[static_cast<std::size_t>(token) * hidden + column];
      x_permuted[static_cast<std::size_t>(destination0) * hidden + column] = value;
      x_permuted[static_cast<std::size_t>(destination1) * hidden + column] = value;
    }
  }
}

template <bool Vectorized>
__global__ void token_permute_token_tile2_top2_kernel(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int hidden) {
  __shared__ int destinations[kTokenTile2Tokens * 2];
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int first_token = static_cast<int>(blockIdx.x) * kTokenTile2Tokens;
  const int valid_tokens = min(kTokenTile2Tokens, tokens - first_token);

  if (warp == 0) {
    const int valid_routes = valid_tokens * 2;
    if (lane < valid_routes) {
      const int route = first_token * 2 + lane;
      const int expert = expert_ids[route];
      const int destination = offsets[expert] + atomicAdd(cursors + expert, 1);
      destinations[lane] = destination;
      route_pos[route] = destination;
      if (sorted_route != nullptr) sorted_route[destination] = route;
    }
  }
  __syncthreads();

  if (warp >= valid_tokens) return;
  const int token = first_token + warp;
  const int destination0 = destinations[warp * 2];
  const int destination1 = destinations[warp * 2 + 1];
  if constexpr (Vectorized) {
    const int vectors = hidden / 4;
    const auto* source =
        reinterpret_cast<const float4*>(x + static_cast<std::size_t>(token) * hidden);
    auto* target0 = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destination0) * hidden);
    auto* target1 = reinterpret_cast<float4*>(
        x_permuted + static_cast<std::size_t>(destination1) * hidden);
    for (int vector = lane; vector < vectors; vector += warpSize) {
      const float4 value = source[vector];
      target0[vector] = value;
      target1[vector] = value;
    }
  } else {
    for (int column = lane; column < hidden; column += warpSize) {
      const float value = x[static_cast<std::size_t>(token) * hidden + column];
      x_permuted[static_cast<std::size_t>(destination0) * hidden + column] = value;
      x_permuted[static_cast<std::size_t>(destination1) * hidden + column] = value;
    }
  }
}

__global__ void token_permute_prepare_offsets_fused_kernel(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    std::int32_t* cursors, int route_pairs, int experts) {
  __shared__ int shared_counts[kMaxExperts];
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread % warpSize;
  if (thread < experts) shared_counts[thread] = 0;
  __syncthreads();

  const int iterations = (route_pairs + static_cast<int>(blockDim.x) - 1) /
                         static_cast<int>(blockDim.x);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int route = iteration * static_cast<int>(blockDim.x) + thread;
    const bool valid = route < route_pairs;
    const unsigned int active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const int expert = expert_ids[route];
      const unsigned int peers = __match_any_sync(active, expert);
      const int leader = __ffs(static_cast<int>(peers)) - 1;
      if (lane == leader) atomicAdd(shared_counts + expert, __popc(peers));
    }
  }
  __syncthreads();

  if (thread == 0) {
    int running = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int count = shared_counts[expert];
      counts[expert] = count;
      offsets[expert] = running;
      cursors[expert] = 0;
      running += count;
    }
    offsets[experts] = running;
  }
}

template <int Threads>
__global__ void token_permute_routeprep_shared_rank_kernel(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    std::int32_t* route_pos, std::int32_t* sorted_route, int route_pairs, int experts) {
  __shared__ int shared_counts[kMaxExperts];
  __shared__ int shared_offsets[kMaxExperts + 1];
  __shared__ int shared_cursors[kMaxExperts];

  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  if (thread < experts) {
    shared_counts[thread] = 0;
    shared_cursors[thread] = 0;
  }
  __syncthreads();

  const int iterations = (route_pairs + Threads - 1) / Threads;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int route = iteration * Threads + thread;
    const bool valid = route < route_pairs;
    const unsigned int active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const int expert = expert_ids[route];
      const unsigned int peers = __match_any_sync(active, expert);
      const int leader = __ffs(static_cast<int>(peers)) - 1;
      if (lane == leader) atomicAdd(shared_counts + expert, __popc(peers));
    }
  }
  __syncthreads();

  if (thread == 0) {
    int running = 0;
    shared_offsets[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int count = shared_counts[expert];
      counts[expert] = count;
      offsets[expert] = running;
      running += count;
      shared_offsets[expert + 1] = running;
    }
    offsets[experts] = running;
  }
  __syncthreads();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int route = iteration * Threads + thread;
    const bool valid = route < route_pairs;
    const unsigned int active = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const int expert = expert_ids[route];
      const unsigned int peers = __match_any_sync(active, expert);
      const int leader = __ffs(static_cast<int>(peers)) - 1;
      const int local_rank = __popc(peers & lanes_before(lane));
      int base = 0;
      if (lane == leader) base = atomicAdd(shared_cursors + expert, __popc(peers));
      base = __shfl_sync(peers, base, leader);
      const int destination = shared_offsets[expert] + base + local_rank;
      route_pos[route] = destination;
      if (sorted_route != nullptr) sorted_route[destination] = route;
    }
  }
}

template <int TopK, bool Vectorized>
__global__ void token_permute_copy_token_owned_topk_kernel(
    const float* x, const std::int32_t* route_pos, float* x_permuted, int tokens,
    int hidden) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;
  constexpr int Threads = 128;
  const int route_begin = token * TopK;
  if constexpr (Vectorized) {
    const int vectors = hidden / 4;
    const auto* source =
        reinterpret_cast<const float4*>(x + static_cast<std::size_t>(token) * hidden);
    for (int vector = static_cast<int>(threadIdx.x); vector < vectors; vector += Threads) {
      const float4 value = source[vector];
#pragma unroll
      for (int rank = 0; rank < TopK; ++rank) {
        const int destination = route_pos[route_begin + rank];
        auto* target = reinterpret_cast<float4*>(
            x_permuted + static_cast<std::size_t>(destination) * hidden);
        target[vector] = value;
      }
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < hidden; column += Threads) {
      const float value = x[static_cast<std::size_t>(token) * hidden + column];
#pragma unroll
      for (int rank = 0; rank < TopK; ++rank) {
        const int destination = route_pos[route_begin + rank];
        x_permuted[static_cast<std::size_t>(destination) * hidden + column] = value;
      }
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

template <bool Aggregate>
cudaError_t launch_token_tile4_top2(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int top_k, int hidden, cudaStream_t stream) {
  if (top_k != 2) {
    return launch_token_owned_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                   sorted_route, tokens, top_k, hidden, stream);
  }
  const int blocks = (tokens + kTokenTileTokens - 1) / kTokenTileTokens;
  const bool vectorized =
      hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
  if (vectorized) {
    token_permute_token_tile4_top2_kernel<Aggregate, true>
        <<<blocks, kTokenTileThreads, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted,
                                                   route_pos, sorted_route, tokens, hidden);
  } else {
    token_permute_token_tile4_top2_kernel<Aggregate, false>
        <<<blocks, kTokenTileThreads, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted,
                                                   route_pos, sorted_route, tokens, hidden);
  }
  return cudaGetLastError();
}

cudaError_t launch_token_tile2_top2(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int top_k, int hidden, cudaStream_t stream) {
  if (top_k != 2) {
    return launch_token_owned_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                   sorted_route, tokens, top_k, hidden, stream);
  }
  const int blocks = (tokens + kTokenTile2Tokens - 1) / kTokenTile2Tokens;
  const bool vectorized =
      hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
  if (vectorized) {
    token_permute_token_tile2_top2_kernel<true>
        <<<blocks, kTokenTile2Threads, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted,
                                                    route_pos, sorted_route, tokens, hidden);
  } else {
    token_permute_token_tile2_top2_kernel<false>
        <<<blocks, kTokenTile2Threads, 0, stream>>>(x, expert_ids, offsets, cursors, x_permuted,
                                                    route_pos, sorted_route, tokens, hidden);
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

cudaError_t launch_token_permute_prepare_offsets_fused(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    std::int32_t* cursors, int tokens, int experts, int top_k,
    cudaStream_t caller_stream) {
  if (tokens < 0 || experts <= 0 || experts > kMaxExperts || top_k <= 0 ||
      top_k > experts || tokens > std::numeric_limits<int>::max() / top_k) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0) return cudaSuccess;
  if (expert_ids == nullptr || counts == nullptr || offsets == nullptr || cursors == nullptr) {
    return cudaErrorInvalidValue;
  }
  token_permute_prepare_offsets_fused_kernel<<<1, kBlockPartialThreads, 0, caller_stream>>>(
      expert_ids, counts, offsets, cursors, tokens * top_k, experts);
  return cudaGetLastError();
}

cudaError_t launch_token_permute_routeprep_shared_rank(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    std::int32_t* route_pos, std::int32_t* sorted_route, int tokens, int experts,
    int top_k, int threads, cudaStream_t caller_stream) {
  if (tokens < 0 || experts <= 0 || experts > kMaxExperts || top_k <= 0 ||
      top_k > experts || tokens > std::numeric_limits<int>::max() / top_k ||
      (threads != 256 && threads != 512 && threads != 1024)) {
    return cudaErrorInvalidValue;
  }
  if (counts == nullptr || offsets == nullptr ||
      (tokens > 0 && (expert_ids == nullptr || route_pos == nullptr))) {
    return cudaErrorInvalidValue;
  }
  const int route_pairs = tokens * top_k;
  if (threads == 256) {
    token_permute_routeprep_shared_rank_kernel<256><<<1, 256, 0, caller_stream>>>(
        expert_ids, counts, offsets, route_pos, sorted_route, route_pairs, experts);
  } else if (threads == 512) {
    token_permute_routeprep_shared_rank_kernel<512><<<1, 512, 0, caller_stream>>>(
        expert_ids, counts, offsets, route_pos, sorted_route, route_pairs, experts);
  } else {
    token_permute_routeprep_shared_rank_kernel<1024><<<1, 1024, 0, caller_stream>>>(
        expert_ids, counts, offsets, route_pos, sorted_route, route_pairs, experts);
  }
  return cudaGetLastError();
}

cudaError_t launch_token_permute_copy_token_owned_topk(
    const float* x, const std::int32_t* route_pos, float* x_permuted, int tokens,
    int top_k, int hidden, cudaStream_t caller_stream) {
  if (tokens < 0 || hidden < 0 || (top_k != 2 && top_k != 4 && top_k != 8)) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0 || hidden == 0) return cudaSuccess;
  if (x == nullptr || route_pos == nullptr || x_permuted == nullptr) return cudaErrorInvalidValue;
  const bool vectorized = hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
#define RR_LAUNCH_TOKEN_COPY(K)                                                               \
  do {                                                                                         \
    if (vectorized) {                                                                          \
      token_permute_copy_token_owned_topk_kernel<K, true><<<tokens, 128, 0, caller_stream>>>( \
          x, route_pos, x_permuted, tokens, hidden);                                           \
    } else {                                                                                   \
      token_permute_copy_token_owned_topk_kernel<K, false><<<tokens, 128, 0, caller_stream>>>(\
          x, route_pos, x_permuted, tokens, hidden);                                           \
    }                                                                                          \
  } while (false)
  if (top_k == 2) {
    RR_LAUNCH_TOKEN_COPY(2);
  } else if (top_k == 4) {
    RR_LAUNCH_TOKEN_COPY(4);
  } else {
    RR_LAUNCH_TOKEN_COPY(8);
  }
#undef RR_LAUNCH_TOKEN_COPY
  return cudaGetLastError();
}

cudaError_t launch_token_permute_copy_from_positions(
    const float* x, const std::int32_t* route_pos, float* x_permuted, int tokens,
    int top_k, int hidden, cudaStream_t caller_stream) {
  if (tokens < 0 || top_k <= 0 || hidden < 0 ||
      (tokens > 0 && tokens > std::numeric_limits<int>::max() / top_k)) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0 || hidden == 0) return cudaSuccess;
  if (x == nullptr || route_pos == nullptr || x_permuted == nullptr) return cudaErrorInvalidValue;
  const int route_pairs = tokens * top_k;
  const bool vectorized = hidden % 4 == 0 && is_aligned_16(x) && is_aligned_16(x_permuted);
  if (vectorized) {
    token_permute_copy_from_positions_kernel<128, true><<<route_pairs, 128, 0, caller_stream>>>(
        x, route_pos, x_permuted, route_pairs, top_k, hidden);
  } else {
    token_permute_copy_from_positions_kernel<128, false><<<route_pairs, 128, 0, caller_stream>>>(
        x, route_pos, x_permuted, route_pairs, top_k, hidden);
  }
  return cudaGetLastError();
}

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
    case kTokenPermuteTokenTile4DirectImplementation:
      return launch_token_tile4_top2<false>(x, expert_ids, offsets, cursors, x_permuted,
                                            route_pos, sorted_route, tokens, top_k, hidden,
                                            caller_stream);
    case kTokenPermuteTokenTile4WarpAggregatedImplementation:
      return launch_token_tile4_top2<true>(x, expert_ids, offsets, cursors, x_permuted,
                                           route_pos, sorted_route, tokens, top_k, hidden,
                                           caller_stream);
    case kTokenPermuteShapeDispatchedV2Implementation:
      if (top_k == 2 && tokens >= 1024 && hidden >= 128 && hidden % 4 == 0 &&
          is_aligned_16(x) && is_aligned_16(x_permuted)) {
        return launch_token_tile4_top2<false>(x, expert_ids, offsets, cursors, x_permuted,
                                              route_pos, sorted_route, tokens, top_k, hidden,
                                              caller_stream);
      }
      return launch_token_owned_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                     sorted_route, tokens, top_k, hidden, caller_stream);
    case kTokenPermuteTokenTile2DirectImplementation:
      return launch_token_tile2_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                     sorted_route, tokens, top_k, hidden, caller_stream);
    case kTokenPermuteShapeDispatchedV3Implementation: {
      const bool aligned_top2 =
          top_k == 2 && hidden > 0 && hidden % 4 == 0 && is_aligned_16(x) &&
          is_aligned_16(x_permuted);
      if (aligned_top2 && tokens >= 1024 && tokens <= 2048 && hidden <= 256) {
        return launch_token_tile4_top2<false>(x, expert_ids, offsets, cursors, x_permuted,
                                              route_pos, sorted_route, tokens, top_k, hidden,
                                              caller_stream);
      }
      if (aligned_top2 && (tokens >= 2048 || hidden >= 512)) {
        return launch_token_tile2_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                       sorted_route, tokens, top_k, hidden, caller_stream);
      }
      return launch_token_owned_top2(x, expert_ids, offsets, cursors, x_permuted, route_pos,
                                     sorted_route, tokens, top_k, hidden, caller_stream);
    }
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace raggedroute::ops
