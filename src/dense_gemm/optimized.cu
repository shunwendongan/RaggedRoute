#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr int kTileExtent = 16;
constexpr int kThreadsPerTiledBlock = kTileExtent * kTileExtent;

__global__ void dense_gemm_tiled_scalar_kernel(const float* a, const float* b, float* c, int m,
                                                int n, int k) {
  __shared__ float a_tile[kTileExtent][kTileExtent];
  __shared__ float b_tile[kTileExtent][kTileExtent];

  const int tile_columns = (n + kTileExtent - 1) / kTileExtent;
  const int tile_index = static_cast<int>(blockIdx.x);
  const int tile_row = tile_index / tile_columns;
  const int tile_column = tile_index - tile_row * tile_columns;

  const int thread = static_cast<int>(threadIdx.x);
  const int local_row = thread / kTileExtent;
  const int local_column = thread - local_row * kTileExtent;
  const int row = tile_row * kTileExtent + local_row;
  const int column = tile_column * kTileExtent + local_column;

  float accumulator = 0.0F;
  for (int k_base = 0; k_base < k; k_base += kTileExtent) {
    const int a_column = k_base + local_column;
    const int b_row = k_base + local_row;
    a_tile[local_row][local_column] =
        row < m && a_column < k ? a[static_cast<std::size_t>(row) * k + a_column] : 0.0F;
    b_tile[local_row][local_column] =
        b_row < k && column < n ? b[static_cast<std::size_t>(b_row) * n + column] : 0.0F;
    __syncthreads();

    const int tile_k = k - k_base < kTileExtent ? k - k_base : kTileExtent;
    for (int inner = 0; inner < tile_k; ++inner) {
      accumulator += a_tile[local_row][inner] * b_tile[inner][local_column];
    }
    __syncthreads();
  }

  if (row < m && column < n) {
    c[static_cast<std::size_t>(row) * n + column] = accumulator;
  }
}

cudaError_t validate_launch_arguments(const float* a, const float* b, float* c, int m, int n,
                                      int k) {
  if (m < 0 || n < 0 || k < 0) return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || (k != 0 && (a == nullptr || b == nullptr))) return cudaErrorInvalidValue;
  return cudaSuccess;
}

cudaError_t launch_tiled_scalar(const float* a, const float* b, float* c, int m, int n, int k,
                                cudaStream_t caller_stream) {
  const std::size_t tile_rows = (static_cast<std::size_t>(m) + kTileExtent - 1) / kTileExtent;
  const std::size_t tile_columns = (static_cast<std::size_t>(n) + kTileExtent - 1) / kTileExtent;
  const std::size_t tiles = tile_rows * tile_columns;
  if (tiles > std::numeric_limits<unsigned int>::max()) return cudaErrorInvalidConfiguration;
  dense_gemm_tiled_scalar_kernel<<<static_cast<unsigned int>(tiles), kThreadsPerTiledBlock, 0,
                                   caller_stream>>>(a, b, c, m, n, k);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_dense_gemm_optimized(const float* a, const float* b, float* c, int m, int n,
                                        int k, std::uint32_t implementation_id,
                                        cudaStream_t caller_stream) {
  const cudaError_t validation = validate_launch_arguments(a, b, c, m, n, k);
  if (validation != cudaSuccess || m == 0 || n == 0) return validation;
  if (implementation_id != kDenseGemmTiledScalarImplementation) return cudaErrorInvalidValue;
  return launch_tiled_scalar(a, b, c, m, n, k, caller_stream);
}

}  // namespace raggedroute::ops
