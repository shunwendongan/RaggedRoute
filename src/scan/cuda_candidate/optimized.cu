#include "optimized_internal.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace raggedroute::ops {
namespace {

constexpr int kSubwarpThreads = 16;
constexpr int kItemsPerSubwarpLane = 4;
constexpr unsigned kSubwarpMask = 0x0000ffffU;

__device__ __forceinline__ std::int32_t inclusive_sum_16(std::int32_t value, int lane,
                                                         unsigned mask) {
#pragma unroll
  for (int delta = 1; delta < kSubwarpThreads; delta <<= 1) {
    const std::int32_t predecessor = __shfl_up_sync(mask, value, delta, kSubwarpThreads);
    if (lane >= delta) value += predecessor;
  }
  return value;
}

__global__ void histogram_exclusive_scan_fused_subwarp_kernel(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    int route_pairs, int experts) {
  __shared__ std::int32_t shared_counts[64];
  for (int expert = static_cast<int>(threadIdx.x); expert < experts;
       expert += static_cast<int>(blockDim.x)) {
    shared_counts[expert] = 0;
  }
  __syncthreads();

  for (int route = static_cast<int>(threadIdx.x); route < route_pairs;
       route += static_cast<int>(blockDim.x)) {
    const std::int32_t expert = expert_ids[route];
    if (expert >= 0 && expert < experts) atomicAdd(shared_counts + expert, 1);
  }
  __syncthreads();

  if (threadIdx.x < kSubwarpThreads) {
    const int lane = static_cast<int>(threadIdx.x);
    const int base_index = lane * kItemsPerSubwarpLane;
    std::int32_t values[kItemsPerSubwarpLane] = {0, 0, 0, 0};
#pragma unroll
    for (int item = 0; item < kItemsPerSubwarpLane; ++item) {
      const int index = base_index + item;
      if (index < experts) values[item] = shared_counts[index];
    }
    std::int32_t lane_total = 0;
#pragma unroll
    for (int item = 0; item < kItemsPerSubwarpLane; ++item) lane_total += values[item];
    const std::int32_t inclusive = inclusive_sum_16(lane_total, lane, kSubwarpMask);
    const std::int32_t lane_base = inclusive - lane_total;
    const std::int32_t total = __shfl_sync(kSubwarpMask, inclusive, kSubwarpThreads - 1,
                                           kSubwarpThreads);
    std::int32_t prefix = lane_base;
#pragma unroll
    for (int item = 0; item < kItemsPerSubwarpLane; ++item) {
      const int index = base_index + item;
      if (index < experts) {
        counts[index] = values[item];
        offsets[index] = prefix;
      }
      prefix += values[item];
    }
    if (lane == 0) offsets[experts] = total;
  }
}

}  // namespace

cudaError_t launch_histogram_exclusive_scan_optimized(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    int route_pairs, int experts, std::uint32_t implementation_id,
    cudaStream_t caller_stream) {
  if (route_pairs < 0 || experts < 1 || experts > 64 || counts == nullptr ||
      offsets == nullptr || (route_pairs != 0 && expert_ids == nullptr) ||
      route_pairs > kHistogramExclusiveScanFusedMaxRoutePairs) {
    return cudaErrorInvalidValue;
  }
  if (implementation_id == kHistogramExclusiveScanFusedSubwarpImplementation) {
    histogram_exclusive_scan_fused_subwarp_kernel<<<1, 256, 0, caller_stream>>>(
        expert_ids, counts, offsets, route_pairs, experts);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace raggedroute::ops
