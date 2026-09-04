#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "optimized_internal.h"
#include "raggedroute/baseline_ops.h"

// histogram 的优化 kernel 家族。
// 输入是 expert_ids[route_pairs]，输出是 counts[experts]，每个 expert 一个 bin，
// 同时对小 E 和单 expert 情况做了单独处理。
namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kItemsPerThread = 8;

__global__ void histogram_single_bin_write_kernel(std::int32_t* counts, int route_pairs) {
  if (threadIdx.x == 0) counts[0] = route_pairs;
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

__global__ void histogram_block_private_kernel(const std::int32_t* expert_ids, std::int32_t* counts,
                                               int route_pairs, int experts) {
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
  if (!is_histogram_benchmark_implementation(implementation_id)) {
    return cudaErrorInvalidValue;
  }
  if (implementation_id == kHistogramCandidateV2Implementation && experts == 1) {
    histogram_single_bin_write_kernel<<<1, 1, 0, caller_stream>>>(counts, route_pairs);
    return cudaGetLastError();
  }
  if (route_pairs <= kHistogramSingleCtaMaxRoutePairs) {
    histogram_single_cta_shared_kernel<<<1, kThreadsPerBlock, 0, caller_stream>>>(
        expert_ids, counts, route_pairs, experts);
    return cudaGetLastError();
  }
  if (route_pairs >= kHistogramBlockPrivateMinRoutePairs) {
    constexpr std::size_t kItemsPerBlock = kThreadsPerBlock * kItemsPerThread;
    const std::size_t blocks =
        (static_cast<std::size_t>(route_pairs) + kItemsPerBlock - 1) / kItemsPerBlock;
    const unsigned int launch_blocks = static_cast<unsigned int>(
        blocks < kHistogramBlockPrivateMaxBlocks ? blocks : kHistogramBlockPrivateMaxBlocks);
    histogram_block_private_kernel<<<launch_blocks, kThreadsPerBlock, 0, caller_stream>>>(
        expert_ids, counts, route_pairs, experts);
    return cudaGetLastError();
  }
  return launch_histogram_naive(expert_ids, counts, route_pairs, experts, caller_stream);
}

}  // namespace raggedroute::ops
