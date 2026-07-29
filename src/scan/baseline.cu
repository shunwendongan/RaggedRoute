#include <cuda_runtime.h>

#include <cstdint>

#include "raggedroute/baseline_ops.h"

namespace raggedroute::ops {
namespace {

__global__ void exclusive_scan_naive_kernel(const std::int32_t* counts, std::int32_t* offsets,
                                            int experts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  std::int32_t running_total = 0;
  for (int expert = 0; expert < experts; ++expert) {
    // Read before writing so counts and offsets may alias for an in-place scan.
    const std::int32_t count = counts[expert];
    offsets[expert] = running_total;
    running_total += count;
  }
  offsets[experts] = running_total;
}

}  // namespace

cudaError_t launch_exclusive_scan_naive(const std::int32_t* counts, std::int32_t* offsets,
                                        int experts, cudaStream_t caller_stream) {
  if (experts <= 0 || offsets == nullptr || counts == nullptr) {
    return cudaErrorInvalidValue;
  }

  exclusive_scan_naive_kernel<<<1, 1, 0, caller_stream>>>(counts, offsets, experts);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
