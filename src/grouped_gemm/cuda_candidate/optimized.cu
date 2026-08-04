#include "grouped_optimized_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace raggedroute::ops {
namespace {

constexpr int kTile = 16;
constexpr int kThreads = kTile * kTile;
constexpr int kMaxExperts = 64;
constexpr int kMaxCachedDevices = 32;

cudaError_t validate_arguments(const float* x_permuted, const float* expert_weights,
                               const std::int32_t* offsets, float* y_permuted, int experts,
                               int hidden, int output, int max_expert_tokens) {
  if (experts <= 0 || experts > kMaxExperts || hidden < 0 || output < 0 ||
      max_expert_tokens < 0) {
    return cudaErrorInvalidValue;
  }
  if (max_expert_tokens == 0 || output == 0) return cudaSuccess;
  if (offsets == nullptr || y_permuted == nullptr ||
      (hidden != 0 && (x_permuted == nullptr || expert_weights == nullptr))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

__device__ __forceinline__ int find_problem(const int* tile_prefix, int experts, int tile_id) {
  int first = 0;
  int count = experts;
  while (count > 0) {
    const int step = count >> 1;
    const int candidate = first + step;
    if (tile_prefix[candidate + 1] <= tile_id) {
      first = candidate + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__device__ __forceinline__ void run_tiled16(
    const float* x_permuted, const float* expert_weights, float* y_permuted, int hidden,
    int output, int expert, int begin, int rows, int tile_row, int tile_column,
    float (&x_tile)[kTile][kTile], float (&weight_tile)[kTile][kTile]) {
  const int local_row = tile_row * kTile + static_cast<int>(threadIdx.y);
  const int row = begin + local_row;
  const int column = tile_column * kTile + static_cast<int>(threadIdx.x);
  const bool valid_row = local_row < rows;
  const bool valid_column = column < output;
  float accumulator = 0.0F;

  for (int tile_begin = 0; tile_begin < hidden; tile_begin += kTile) {
    const int x_inner = tile_begin + static_cast<int>(threadIdx.x);
    const int weight_inner = tile_begin + static_cast<int>(threadIdx.y);
    x_tile[threadIdx.y][threadIdx.x] =
        valid_row && x_inner < hidden
            ? x_permuted[static_cast<std::size_t>(row) * hidden + x_inner]
            : 0.0F;
    weight_tile[threadIdx.y][threadIdx.x] =
        valid_column && weight_inner < hidden
            ? expert_weights[(static_cast<std::size_t>(expert) * hidden + weight_inner) * output +
                             column]
            : 0.0F;
    __syncthreads();

    const int tile_k = min(kTile, hidden - tile_begin);
#pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      if (inner < tile_k) {
        accumulator =
            __fmaf_rn(x_tile[threadIdx.y][inner], weight_tile[inner][threadIdx.x], accumulator);
      }
    }
    __syncthreads();
  }

  if (valid_row && valid_column) {
    y_permuted[static_cast<std::size_t>(row) * output + column] = accumulator;
  }
}

__global__ void grouped_gemm_tiled16_sync_v0_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int hidden, int output) {
  const int expert = static_cast<int>(blockIdx.z);
  const int begin = offsets[expert];
  const int rows = offsets[expert + 1] - begin;
  const int tile_row = static_cast<int>(blockIdx.y);
  if (tile_row * kTile >= rows) return;

  __shared__ float x_tile[kTile][kTile];
  __shared__ float weight_tile[kTile][kTile];
  run_tiled16(x_permuted, expert_weights, y_permuted, hidden, output, expert, begin, rows,
              tile_row, static_cast<int>(blockIdx.x), x_tile, weight_tile);
}

__global__ void grouped_gemm_persistent16_v1_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output) {
  __shared__ int tile_prefix[kMaxExperts + 1];
  __shared__ float x_tile[kTile][kTile];
  __shared__ float weight_tile[kTile][kTile];

  const int tile_columns = (output + kTile - 1) / kTile;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    tile_prefix[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int rows = offsets[expert + 1] - offsets[expert];
      const int tile_rows = (rows + kTile - 1) / kTile;
      tile_prefix[expert + 1] = tile_prefix[expert] + tile_rows * tile_columns;
    }
  }
  __syncthreads();

  const int total_tiles = tile_prefix[experts];
  for (int tile_id = static_cast<int>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int>(gridDim.x)) {
    const int expert = find_problem(tile_prefix, experts, tile_id);
    const int local_tile = tile_id - tile_prefix[expert];
    const int tile_row = local_tile / tile_columns;
    const int tile_column = local_tile - tile_row * tile_columns;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    run_tiled16(x_permuted, expert_weights, y_permuted, hidden, output, expert, begin, rows,
                tile_row, tile_column, x_tile, weight_tile);
  }
}

template <typename Kernel>
cudaError_t persistent_grid_limit(Kernel kernel, int threads, std::atomic<int>* cache,
                                  int* grid_limit) {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  if (device < 0 || device >= kMaxCachedDevices) return cudaErrorInvalidDevice;
  int cached = cache[device].load(std::memory_order_acquire);
  if (cached > 0) {
    *grid_limit = cached;
    return cudaSuccess;
  }

  int multiprocessors = 0;
  error = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess) return error;
  int active_blocks_per_sm = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, threads, 0);
  if (error != cudaSuccess) return error;
  const int resident_blocks = std::max(1, std::min(2, active_blocks_per_sm));
  cached = multiprocessors * resident_blocks;
  cache[device].store(cached, std::memory_order_release);
  *grid_limit = cached;
  return cudaSuccess;
}

cudaError_t launch_persistent16(const float* x_permuted, const float* expert_weights,
                                const std::int32_t* offsets, float* y_permuted, int experts,
                                int hidden, int output, int max_expert_tokens,
                                cudaStream_t caller_stream) {
  static std::atomic<int> grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  cudaError_t error = persistent_grid_limit(grouped_gemm_persistent16_v1_kernel, kThreads,
                                            grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  const std::size_t upper_tiles =
      static_cast<std::size_t>(experts) *
      ((static_cast<std::size_t>(max_expert_tokens) + kTile - 1) / kTile) *
      ((static_cast<std::size_t>(output) + kTile - 1) / kTile);
  if (upper_tiles == 0 || upper_tiles > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return upper_tiles == 0 ? cudaSuccess : cudaErrorInvalidConfiguration;
  }
  const int blocks = std::min(grid_limit, static_cast<int>(upper_tiles));
  grouped_gemm_persistent16_v1_kernel<<<blocks, dim3(kTile, kTile), 0, caller_stream>>>(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_grouped_gemm_tiled16_sync_v0(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(x_permuted, expert_weights, offsets,
                                                    y_permuted, experts, hidden, output,
                                                    max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  const std::size_t grid_x = (static_cast<std::size_t>(output) + kTile - 1) / kTile;
  const std::size_t grid_y =
      (static_cast<std::size_t>(max_expert_tokens) + kTile - 1) / kTile;
  if (grid_x > std::numeric_limits<unsigned int>::max() ||
      grid_y > std::numeric_limits<unsigned int>::max()) {
    return cudaErrorInvalidConfiguration;
  }
  grouped_gemm_tiled16_sync_v0_kernel<<<
      dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
           static_cast<unsigned int>(experts)),
      dim3(kTile, kTile), 0, caller_stream>>>(x_permuted, expert_weights, offsets, y_permuted,
                                              hidden, output);
  return cudaGetLastError();
}

cudaError_t launch_grouped_gemm_persistent16_v1(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(x_permuted, expert_weights, offsets,
                                                    y_permuted, experts, hidden, output,
                                                    max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  return launch_persistent16(x_permuted, expert_weights, offsets, y_permuted, experts, hidden,
                             output, max_expert_tokens, caller_stream);
}

cudaError_t launch_grouped_gemm_optimized(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    std::uint32_t implementation_id, cudaStream_t caller_stream) {
  if (implementation_id == kGroupedGemmTiled16SyncV0Implementation) {
    return launch_grouped_gemm_tiled16_sync_v0(x_permuted, expert_weights, offsets, y_permuted,
                                                experts, hidden, output, max_expert_tokens,
                                                caller_stream);
  }
  if (implementation_id == kGroupedGemmPersistent16V1Implementation) {
    return launch_grouped_gemm_persistent16_v1(x_permuted, expert_weights, offsets, y_permuted,
                                                experts, hidden, output, max_expert_tokens,
                                                caller_stream);
  }
  if (implementation_id == kGroupedGemmRegister16x32SyncV2Implementation) {
    return launch_grouped_gemm_register16x32_sync_v2(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  if (implementation_id == kGroupedGemmRegister16x32AsyncV3Implementation) {
    return launch_grouped_gemm_register16x32_async_v3(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  if (implementation_id == kGroupedGemmRegister16x32AsyncFullV4Implementation) {
    return launch_grouped_gemm_register16x32_async_full_v4(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  if (implementation_id == kGroupedGemmSm86Fp32V1Implementation) {
    static std::atomic<int> grid_cache[kMaxCachedDevices]{};
    int grid_limit = 0;
    cudaError_t error = persistent_grid_limit(grouped_gemm_persistent16_v1_kernel, kThreads,
                                              grid_cache, &grid_limit);
    if (error != cudaSuccess) return error;
    const std::size_t direct_upper_tiles =
        static_cast<std::size_t>(experts) *
        ((static_cast<std::size_t>(max_expert_tokens) + kTile - 1) / kTile) *
        ((static_cast<std::size_t>(output) + kTile - 1) / kTile);
    if (direct_upper_tiles <= static_cast<std::size_t>(grid_limit)) {
      return launch_grouped_gemm_tiled16_sync_v0(x_permuted, expert_weights, offsets,
                                                  y_permuted, experts, hidden, output,
                                                  max_expert_tokens, caller_stream);
    }
    return launch_grouped_gemm_register16x32_async_full_v4(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  if (implementation_id == kGroupedGemmSm86Fp32V2Implementation) {
    return launch_grouped_gemm_sm86_fp32_v2(x_permuted, expert_weights, offsets, y_permuted,
                                             experts, hidden, output, max_expert_tokens,
                                             caller_stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace raggedroute::ops
