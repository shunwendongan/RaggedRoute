#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;
constexpr unsigned int kItemsPerThread = 8;

__global__ void histogram_warp_aggregated_kernel(const std::int32_t* expert_ids,
                                                  std::int32_t* counts, int route_pairs,
                                                  int experts) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const unsigned int lane = threadIdx.x & (warpSize - 1);
  for (std::size_t route = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < static_cast<std::size_t>(route_pairs); route += stride) {
    const std::int32_t expert = expert_ids[route];
    const unsigned int active = __activemask();
    const bool in_range = expert >= 0 && expert < experts;
    const unsigned int valid = __ballot_sync(active, in_range);
    if (in_range) {
      const unsigned int peers = __match_any_sync(valid, expert);
      if (lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
        atomicAdd(counts + expert, static_cast<std::int32_t>(__popc(peers)));
      }
    }
  }
}

__global__ void histogram_single_cta_shared_kernel(const std::int32_t* expert_ids,
                                                    std::int32_t* counts, int route_pairs,
                                                    int experts) {
  __shared__ std::int32_t shared_counts[64];
  for (int expert = static_cast<int>(threadIdx.x); expert < experts;
       expert += static_cast<int>(blockDim.x)) {
    shared_counts[expert] = 0;
  }
  __syncthreads();

  for (std::size_t route = threadIdx.x; route < static_cast<std::size_t>(route_pairs);
       route += blockDim.x) {
    const std::int32_t expert = expert_ids[route];
    if (expert >= 0 && expert < experts) {
      atomicAdd(shared_counts + expert, std::int32_t{1});
    }
  }
  __syncthreads();

  for (int expert = static_cast<int>(threadIdx.x); expert < experts;
       expert += static_cast<int>(blockDim.x)) {
    counts[expert] = shared_counts[expert];
  }
}

__global__ void histogram_block_private_kernel(const std::int32_t* expert_ids,
                                                std::int32_t* counts, int route_pairs,
                                                int experts) {
  __shared__ std::int32_t shared_counts[64];
  for (int expert = static_cast<int>(threadIdx.x); expert < experts;
       expert += static_cast<int>(blockDim.x)) {
    shared_counts[expert] = 0;
  }
  __syncthreads();

  constexpr std::size_t kItemsPerBlock = kThreadsPerBlock * kItemsPerThread;
  const std::size_t tile_stride = static_cast<std::size_t>(gridDim.x) * kItemsPerBlock;
  for (std::size_t block_base = static_cast<std::size_t>(blockIdx.x) * kItemsPerBlock;
       block_base < static_cast<std::size_t>(route_pairs); block_base += tile_stride) {
#pragma unroll
    for (unsigned int item = 0; item < kItemsPerThread; ++item) {
      const std::size_t route = block_base + threadIdx.x + item * blockDim.x;
      if (route < static_cast<std::size_t>(route_pairs)) {
        const std::int32_t expert = expert_ids[route];
        if (expert >= 0 && expert < experts) {
          atomicAdd(shared_counts + expert, std::int32_t{1});
        }
      }
    }
  }
  __syncthreads();

  for (int expert = static_cast<int>(threadIdx.x); expert < experts;
       expert += static_cast<int>(blockDim.x)) {
    const std::int32_t partial = shared_counts[expert];
    if (partial != 0) atomicAdd(counts + expert, partial);
  }
}

unsigned int block_count_for(std::size_t elements) {
  const std::size_t blocks =
      elements / kThreadsPerBlock + (elements % kThreadsPerBlock != 0 ? 1 : 0);
  return static_cast<unsigned int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

cudaError_t validate_arguments(const std::int32_t* expert_ids, std::int32_t* counts,
                               int route_pairs, int experts) {
  if (route_pairs < 0 || experts <= 0 || experts > 64) return cudaErrorInvalidValue;
  if (counts == nullptr || (route_pairs != 0 && expert_ids == nullptr)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_histogram_optimized(const std::int32_t* expert_ids, std::int32_t* counts,
                                       int route_pairs, int experts,
                                       std::uint32_t implementation_id,
                                       cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(expert_ids, counts, route_pairs, experts);
  if (validation != cudaSuccess) return validation;
  if (implementation_id == kHistogramWarpAggregatedImplementation) {
    if (route_pairs == 0) return cudaSuccess;
    histogram_warp_aggregated_kernel<<<
        block_count_for(static_cast<std::size_t>(route_pairs)), kThreadsPerBlock, 0,
        caller_stream>>>(expert_ids, counts, route_pairs, experts);
    return cudaGetLastError();
  }
  if (implementation_id == kHistogramSingleCtaSharedImplementation) {
    histogram_single_cta_shared_kernel<<<1, kThreadsPerBlock, 0, caller_stream>>>(
        expert_ids, counts, route_pairs, experts);
    return cudaGetLastError();
  }
  if (implementation_id == kHistogramBlockPrivateImplementation) {
    if (route_pairs == 0) return cudaSuccess;
    constexpr std::size_t kItemsPerBlock = kThreadsPerBlock * kItemsPerThread;
    const std::size_t blocks = (static_cast<std::size_t>(route_pairs) + kItemsPerBlock - 1) /
                               kItemsPerBlock;
    const unsigned int launch_blocks =
        static_cast<unsigned int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
    histogram_block_private_kernel<<<launch_blocks, kThreadsPerBlock, 0,
                                     caller_stream>>>(expert_ids, counts, route_pairs, experts);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

}  // namespace raggedroute::ops
