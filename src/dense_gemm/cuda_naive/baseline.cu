#include <cuda_runtime.h>

#include <cstddef>

#include "raggedroute/baseline_ops.h"

namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBaselineBlocks = 65535;

__global__ void dense_gemm_naive_kernel(const float* a, const float* b, float* c, int m, int n,
                                        int k) {
  const std::size_t output_elements = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t linear_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear_index < output_elements; linear_index += stride) {
    const int row = static_cast<int>(linear_index / static_cast<std::size_t>(n));
    const int column = static_cast<int>(linear_index % static_cast<std::size_t>(n));
    float accumulator = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
      accumulator += a[static_cast<std::size_t>(row) * k + inner] *
                     b[static_cast<std::size_t>(inner) * n + column];
    }
    c[linear_index] = accumulator;
  }
}

unsigned int block_count_for(std::size_t elements) {
  const std::size_t blocks =
      elements / kThreadsPerBlock + (elements % kThreadsPerBlock != 0 ? 1 : 0);
  return static_cast<unsigned int>(blocks < kMaxBaselineBlocks ? blocks : kMaxBaselineBlocks);
}

}  // namespace

cudaError_t launch_dense_gemm_naive(const float* a, const float* b, float* c, int m, int n, int k,
                                    cudaStream_t caller_stream) {
  if (m < 0 || n < 0 || k < 0) {
    return cudaErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return cudaSuccess;
  }
  if (c == nullptr || (k != 0 && (a == nullptr || b == nullptr))) {
    return cudaErrorInvalidValue;
  }

  const std::size_t output_elements = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  dense_gemm_naive_kernel<<<block_count_for(output_elements), kThreadsPerBlock, 0, caller_stream>>>(
      a, b, c, m, n, k);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
