#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "raggedroute/baseline_ops.h"

// histogram 的朴素 baseline。
// 把 expert_ids[route_pairs] 计数到 counts[experts] 中，沿用和高性能版本一致的
// E<=64 路由契约。
namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBaselineBlocks = 65535;

__global__ void histogram_naive_kernel(const std::int32_t* expert_ids, std::int32_t* counts,
                                       int route_pairs, int experts) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t route = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < static_cast<std::size_t>(route_pairs); route += stride) {
    const std::int32_t expert = expert_ids[route];
    if (expert >= 0 && expert < experts) {
      atomicAdd(counts + expert, std::int32_t{1});
    }
  }
}

unsigned int block_count_for(std::size_t elements) {
  const std::size_t blocks =
      elements / kThreadsPerBlock + (elements % kThreadsPerBlock != 0 ? 1 : 0);
  return static_cast<unsigned int>(blocks < kMaxBaselineBlocks ? blocks : kMaxBaselineBlocks);
}

}  // namespace

cudaError_t launch_histogram_naive(const std::int32_t* expert_ids, std::int32_t* counts,
                                   int route_pairs, int experts, cudaStream_t caller_stream) {
  if (route_pairs < 0 || experts <= 0) {
    return cudaErrorInvalidValue;
  }
  if (route_pairs == 0) {
    return cudaSuccess;
  }
  if (expert_ids == nullptr || counts == nullptr) {
    return cudaErrorInvalidValue;
  }

  histogram_naive_kernel<<<block_count_for(static_cast<std::size_t>(route_pairs)), kThreadsPerBlock,
                           0, caller_stream>>>(expert_ids, counts, route_pairs, experts);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
