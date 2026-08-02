#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;

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
  if (validation != cudaSuccess || route_pairs == 0) return validation;
  if (implementation_id != kHistogramWarpAggregatedImplementation) {
    return cudaErrorInvalidValue;
  }
  histogram_warp_aggregated_kernel<<<
      block_count_for(static_cast<std::size_t>(route_pairs)), kThreadsPerBlock, 0,
      caller_stream>>>(expert_ids, counts, route_pairs, experts);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
