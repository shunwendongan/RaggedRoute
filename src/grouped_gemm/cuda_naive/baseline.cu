#include <cuda_runtime.h>

#include "raggedroute/baseline_ops.h"

namespace raggedroute::ops {
namespace {

__global__ void grouped_gemm_naive_kernel(const float* x_permuted, const float* expert_weights,
                                          const std::int32_t* offsets, float* y_permuted,
                                          int hidden, int output) {
  const int expert = static_cast<int>(blockIdx.z);
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int begin = offsets[expert];
  const int end = offsets[expert + 1];
  if (column >= output) return;
  const int row_stride = static_cast<int>(gridDim.y * blockDim.y);
  for (int local_row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
       local_row < end - begin; local_row += row_stride) {
    const int row = begin + local_row;
    float accumulator = 0.0F;
    for (int inner = 0; inner < hidden; ++inner) {
      accumulator +=
          x_permuted[static_cast<std::size_t>(row) * hidden + inner] *
          expert_weights[(static_cast<std::size_t>(expert) * hidden + inner) * output + column];
    }
    y_permuted[static_cast<std::size_t>(row) * output + column] = accumulator;
  }
}

}  // namespace

cudaError_t launch_grouped_gemm_naive(const float* x_permuted, const float* expert_weights,
                                      const std::int32_t* offsets, float* y_permuted, int experts,
                                      int hidden, int output, int max_expert_tokens,
                                      cudaStream_t caller_stream) {
  if (experts <= 0 || hidden < 0 || output < 0 || max_expert_tokens < 0) {
    return cudaErrorInvalidValue;
  }
  if (max_expert_tokens == 0 || output == 0) return cudaSuccess;
  if (offsets == nullptr || y_permuted == nullptr ||
      (hidden != 0 && (x_permuted == nullptr || expert_weights == nullptr))) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(16, 16, 1);
  constexpr unsigned int kMaxGridY = 65535;
  const auto grid_x =
      static_cast<unsigned int>((static_cast<std::size_t>(output) + block.x - 1) / block.x);
  const auto required_grid_y = static_cast<unsigned int>(
      (static_cast<std::size_t>(max_expert_tokens) + block.y - 1) / block.y);
  const dim3 grid(grid_x, required_grid_y < kMaxGridY ? required_grid_y : kMaxGridY, experts);
  grouped_gemm_naive_kernel<<<grid, block, 0, caller_stream>>>(x_permuted, expert_weights, offsets,
                                                               y_permuted, hidden, output);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
