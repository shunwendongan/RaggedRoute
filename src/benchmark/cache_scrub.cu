#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "raggedroute/baseline_ops.h"

namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBaselineBlocks = 65535;

__global__ void cache_scrub_kernel(std::uint32_t* buffer, std::size_t elements) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;) {
    const std::uint32_t old_value = buffer[index];
    buffer[index] = old_value * 1664525U + 1013904223U + static_cast<std::uint32_t>(index);
    if (elements - index <= stride) {
      break;
    }
    index += stride;
  }
}

unsigned int block_count_for(std::size_t elements) {
  const std::size_t blocks =
      elements / kThreadsPerBlock + (elements % kThreadsPerBlock != 0 ? 1 : 0);
  return static_cast<unsigned int>(blocks < kMaxBaselineBlocks ? blocks : kMaxBaselineBlocks);
}

}  // namespace

cudaError_t launch_cache_scrub(std::uint32_t* buffer, std::size_t elements,
                               cudaStream_t caller_stream) {
  if (elements == 0) {
    return cudaSuccess;
  }
  if (buffer == nullptr) {
    return cudaErrorInvalidValue;
  }

  cache_scrub_kernel<<<block_count_for(elements), kThreadsPerBlock, 0, caller_stream>>>(buffer,
                                                                                        elements);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
