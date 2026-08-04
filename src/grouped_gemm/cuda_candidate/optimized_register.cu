#include "grouped_optimized_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace raggedroute::ops {
namespace {

constexpr int kBlockRows = 16;
constexpr int kBlockColumns = 32;
constexpr int kBlockDepth = 16;
constexpr int kSharedAStride = 20;
constexpr int kThreads = 128;
constexpr int kRowsPerThread = 2;
constexpr int kColumnsPerThread = 2;
constexpr int kMaxExperts = 64;
constexpr int kMaxCachedDevices = 32;

static_assert(kThreads * kRowsPerThread * kColumnsPerThread ==
              kBlockRows * kBlockColumns);

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

bool aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
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

__device__ __forceinline__ void build_tile_prefix_parallel(
    const std::int32_t* offsets, int experts, int tile_columns, int* tile_prefix,
    int* warp_totals) {
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  int tile_count = 0;
  if (thread < kMaxExperts && thread < experts) {
    const int rows = offsets[thread + 1] - offsets[thread];
    tile_count = ((rows + kBlockRows - 1) / kBlockRows) * tile_columns;
  }
  if (thread < kMaxExperts) {
#pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
      const int prior = __shfl_up_sync(0xffffffffU, tile_count, delta);
      if (lane >= delta) tile_count += prior;
    }
    if (lane == 31) warp_totals[warp] = tile_count;
  }
  __syncthreads();
  if (thread == 0) tile_prefix[0] = 0;
  if (thread < kMaxExperts && thread < experts) {
    tile_prefix[thread + 1] = tile_count + (warp == 0 ? 0 : warp_totals[0]);
  }
  __syncthreads();
}

__device__ __forceinline__ void stage_sync(
    const float* x_permuted, const float* expert_weights, int expert, int begin, int rows,
    int hidden, int output, int tile_row, int tile_column, int k_base, float* x_tile,
    float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  for (int index = thread; index < kBlockRows * kBlockDepth; index += kThreads) {
    const int local_row = index / kBlockDepth;
    const int inner = index - local_row * kBlockDepth;
    const int input_row = tile_row * kBlockRows + local_row;
    const int input_inner = k_base + inner;
    x_tile[local_row * kSharedAStride + inner] =
        input_row < rows && input_inner < hidden
            ? x_permuted[static_cast<std::size_t>(begin + input_row) * hidden + input_inner]
            : 0.0F;
  }
  for (int index = thread; index < kBlockDepth * kBlockColumns; index += kThreads) {
    const int inner = index / kBlockColumns;
    const int local_column = index - inner * kBlockColumns;
    const int weight_inner = k_base + inner;
    const int output_column = tile_column * kBlockColumns + local_column;
    weight_tile[inner * kBlockColumns + local_column] =
        weight_inner < hidden && output_column < output
            ? expert_weights[(static_cast<std::size_t>(expert) * hidden + weight_inner) * output +
                             output_column]
            : 0.0F;
  }
}

__device__ __forceinline__ void copy_float4_async(float* shared_destination,
                                                  const float* global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<float4*>(shared_destination) =
      *reinterpret_cast<const float4*>(global_source);
#endif
}

__device__ __forceinline__ void commit_async_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_async_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void stage_async(
    const float* x_permuted, const float* expert_weights, int expert, int begin, int hidden,
    int output, int tile_row, int tile_column, int k_base, float* x_tile,
    float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  if (thread < kBlockRows * kBlockDepth / 4) {
    const int element = thread * 4;
    const int local_row = element / kBlockDepth;
    const int inner = element - local_row * kBlockDepth;
    const float* source =
        x_permuted + static_cast<std::size_t>(begin + tile_row * kBlockRows + local_row) * hidden +
        k_base + inner;
    copy_float4_async(x_tile + local_row * kSharedAStride + inner, source);
  }
  const int element = thread * 4;
  const int inner = element / kBlockColumns;
  const int local_column = element - inner * kBlockColumns;
  const float* source =
      expert_weights + (static_cast<std::size_t>(expert) * hidden + k_base + inner) * output +
      tile_column * kBlockColumns + local_column;
  copy_float4_async(weight_tile + inner * kBlockColumns + local_column, source);
  commit_async_group();
}

__device__ __forceinline__ void accumulate(
    const float* x_tile, const float* weight_tile, int tile_k, int local_row, int local_column,
    float (&accumulator)[kRowsPerThread][kColumnsPerThread]) {
#pragma unroll
  for (int inner = 0; inner < kBlockDepth; ++inner) {
    if (inner < tile_k) {
      const float x0 = x_tile[local_row * kSharedAStride + inner];
      const float x1 = x_tile[(local_row + 4) * kSharedAStride + inner];
      const float w0 = weight_tile[inner * kBlockColumns + local_column];
      const float w1 = weight_tile[inner * kBlockColumns + local_column + 1];
      accumulator[0][0] = __fmaf_rn(x0, w0, accumulator[0][0]);
      accumulator[0][1] = __fmaf_rn(x0, w1, accumulator[0][1]);
      accumulator[1][0] = __fmaf_rn(x1, w0, accumulator[1][0]);
      accumulator[1][1] = __fmaf_rn(x1, w1, accumulator[1][1]);
    }
  }
}

__device__ __forceinline__ void store(
    float* y_permuted, int begin, int rows, int output, int tile_row, int tile_column,
    int local_row, int local_column,
    const float (&accumulator)[kRowsPerThread][kColumnsPerThread]) {
  const int output_column = tile_column * kBlockColumns + local_column;
#pragma unroll
  for (int row_index = 0; row_index < kRowsPerThread; ++row_index) {
    const int local_output_row = tile_row * kBlockRows + local_row + row_index * 4;
    if (local_output_row < rows && output_column < output) {
      float* destination = y_permuted +
                           static_cast<std::size_t>(begin + local_output_row) * output +
                           output_column;
      destination[0] = accumulator[row_index][0];
      if (output_column + 1 < output) destination[1] = accumulator[row_index][1];
    }
  }
}

__device__ __forceinline__ void thread_output_coordinates(int* local_row, int* local_column) {
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  *local_row = (warp >> 1) * 8 + (lane >> 3);
  *local_column = (warp & 1) * 16 + (lane & 7) * 2;
}

__global__ __launch_bounds__(kThreads, 4) void grouped_gemm_register16x32_sync_v2_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output) {
  __shared__ int tile_prefix[kMaxExperts + 1];
  __align__(16) __shared__ float x_tile[kBlockRows][kSharedAStride];
  __align__(16) __shared__ float weight_tile[kBlockDepth][kBlockColumns];

  const int tile_columns = (output + kBlockColumns - 1) / kBlockColumns;
  if (threadIdx.x == 0) {
    tile_prefix[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int rows = offsets[expert + 1] - offsets[expert];
      tile_prefix[expert + 1] =
          tile_prefix[expert] + ((rows + kBlockRows - 1) / kBlockRows) * tile_columns;
    }
  }
  __syncthreads();

  int local_row = 0;
  int local_column = 0;
  thread_output_coordinates(&local_row, &local_column);
  const int total_tiles = tile_prefix[experts];
  for (int tile_id = static_cast<int>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int>(gridDim.x)) {
    const int expert = find_problem(tile_prefix, experts, tile_id);
    const int local_tile = tile_id - tile_prefix[expert];
    const int tile_row = local_tile / tile_columns;
    const int tile_column = local_tile - tile_row * tile_columns;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    float accumulator[kRowsPerThread][kColumnsPerThread] = {};

    for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
      stage_sync(x_permuted, expert_weights, expert, begin, rows, hidden, output, tile_row,
                 tile_column, k_base, &x_tile[0][0], &weight_tile[0][0]);
      __syncthreads();
      const int remaining = hidden - k_base;
      const int tile_k = remaining < kBlockDepth ? remaining : kBlockDepth;
      accumulate(&x_tile[0][0], &weight_tile[0][0], tile_k, local_row, local_column,
                 accumulator);
      __syncthreads();
    }
    store(y_permuted, begin, rows, output, tile_row, tile_column, local_row, local_column,
          accumulator);
  }
}

__global__ __launch_bounds__(kThreads, 3) void grouped_gemm_register16x32_async_v3_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output) {
  __shared__ int tile_prefix[kMaxExperts + 1];
  __align__(16) __shared__ float x_tiles[2][kBlockRows][kSharedAStride];
  __align__(16) __shared__ float weight_tiles[2][kBlockDepth][kBlockColumns];

  const int tile_columns = output / kBlockColumns;
  if (threadIdx.x == 0) {
    tile_prefix[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int rows = offsets[expert + 1] - offsets[expert];
      tile_prefix[expert + 1] =
          tile_prefix[expert] + ((rows + kBlockRows - 1) / kBlockRows) * tile_columns;
    }
  }
  __syncthreads();

  int local_row = 0;
  int local_column = 0;
  thread_output_coordinates(&local_row, &local_column);
  const int total_tiles = tile_prefix[experts];
  for (int tile_id = static_cast<int>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int>(gridDim.x)) {
    const int expert = find_problem(tile_prefix, experts, tile_id);
    const int local_tile = tile_id - tile_prefix[expert];
    const int tile_row = local_tile / tile_columns;
    const int tile_column = local_tile - tile_row * tile_columns;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    const bool full_rows = (tile_row + 1) * kBlockRows <= rows;
    float accumulator[kRowsPerThread][kColumnsPerThread] = {};

    if (!full_rows) {
      for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
        stage_sync(x_permuted, expert_weights, expert, begin, rows, hidden, output, tile_row,
                   tile_column, k_base, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
        __syncthreads();
        accumulate(&x_tiles[0][0][0], &weight_tiles[0][0][0], kBlockDepth, local_row,
                   local_column, accumulator);
        __syncthreads();
      }
    } else if (hidden > 0) {
      stage_async(x_permuted, expert_weights, expert, begin, hidden, output, tile_row,
                  tile_column, 0, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
      wait_async_group();
      __syncthreads();
      int stage = 0;
      for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
        const int next_k = k_base + kBlockDepth;
        if (next_k < hidden) {
          stage_async(x_permuted, expert_weights, expert, begin, hidden, output, tile_row,
                      tile_column, next_k, &x_tiles[stage ^ 1][0][0],
                      &weight_tiles[stage ^ 1][0][0]);
        }
        accumulate(&x_tiles[stage][0][0], &weight_tiles[stage][0][0], kBlockDepth, local_row,
                   local_column, accumulator);
        if (next_k < hidden) {
          wait_async_group();
          __syncthreads();
          stage ^= 1;
        }
      }
      __syncthreads();
    }
    store(y_permuted, begin, rows, output, tile_row, tile_column, local_row, local_column,
          accumulator);
  }
}

__global__ __launch_bounds__(kThreads, 5) void grouped_gemm_register16x32_sm86_v2_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output) {
  __shared__ int tile_prefix[kMaxExperts + 1];
  __shared__ int prefix_warp_totals[2];
  __align__(16) __shared__ float x_tiles[2][kBlockRows][kSharedAStride];
  __align__(16) __shared__ float weight_tiles[2][kBlockDepth][kBlockColumns];

  const int tile_columns = output / kBlockColumns;
  build_tile_prefix_parallel(offsets, experts, tile_columns, tile_prefix, prefix_warp_totals);

  int local_row = 0;
  int local_column = 0;
  thread_output_coordinates(&local_row, &local_column);
  const int total_tiles = tile_prefix[experts];
  for (int tile_id = static_cast<int>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int>(gridDim.x)) {
    const int expert = find_problem(tile_prefix, experts, tile_id);
    const int local_tile = tile_id - tile_prefix[expert];
    const int tile_row = local_tile / tile_columns;
    const int tile_column = local_tile - tile_row * tile_columns;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    const bool full_rows = (tile_row + 1) * kBlockRows <= rows;
    float accumulator[kRowsPerThread][kColumnsPerThread] = {};

    if (!full_rows) {
      for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
        stage_sync(x_permuted, expert_weights, expert, begin, rows, hidden, output, tile_row,
                   tile_column, k_base, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
        __syncthreads();
        accumulate(&x_tiles[0][0][0], &weight_tiles[0][0][0], kBlockDepth, local_row,
                   local_column, accumulator);
        __syncthreads();
      }
    } else if (hidden > 0) {
      stage_async(x_permuted, expert_weights, expert, begin, hidden, output, tile_row,
                  tile_column, 0, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
      wait_async_group();
      __syncthreads();
      int stage = 0;
      for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
        const int next_k = k_base + kBlockDepth;
        if (next_k < hidden) {
          stage_async(x_permuted, expert_weights, expert, begin, hidden, output, tile_row,
                      tile_column, next_k, &x_tiles[stage ^ 1][0][0],
                      &weight_tiles[stage ^ 1][0][0]);
        }
        accumulate(&x_tiles[stage][0][0], &weight_tiles[stage][0][0], kBlockDepth, local_row,
                   local_column, accumulator);
        if (next_k < hidden) {
          wait_async_group();
          __syncthreads();
          stage ^= 1;
        }
      }
      __syncthreads();
    }
    store(y_permuted, begin, rows, output, tile_row, tile_column, local_row, local_column,
          accumulator);
  }
}

template <typename Kernel>
cudaError_t persistent_grid_limit(Kernel kernel, int resident_block_cap, std::atomic<int>* cache,
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
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, kThreads, 0);
  if (error != cudaSuccess) return error;
  const int resident_blocks =
      resident_block_cap > 0 ? std::min(resident_block_cap, active_blocks_per_sm)
                             : active_blocks_per_sm;
  cached = multiprocessors * std::max(1, resident_blocks);
  cache[device].store(cached, std::memory_order_release);
  *grid_limit = cached;
  return cudaSuccess;
}

cudaError_t persistent_block_count(int grid_limit, int experts, int output,
                                   int max_expert_tokens, int* blocks) {
  const std::size_t upper_tiles =
      static_cast<std::size_t>(experts) *
      ((static_cast<std::size_t>(max_expert_tokens) + kBlockRows - 1) / kBlockRows) *
      ((static_cast<std::size_t>(output) + kBlockColumns - 1) / kBlockColumns);
  if (upper_tiles == 0 || upper_tiles > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return upper_tiles == 0 ? cudaSuccess : cudaErrorInvalidConfiguration;
  }
  *blocks = std::min(grid_limit, static_cast<int>(upper_tiles));
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_grouped_gemm_sm86_fp32_v2(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(x_permuted, expert_weights, offsets,
                                                    y_permuted, experts, hidden, output,
                                                    max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (hidden == 0 || hidden % kBlockDepth != 0 || output % kBlockColumns != 0 ||
      !aligned_16(x_permuted) || !aligned_16(expert_weights) || !aligned_16(y_permuted)) {
    return launch_grouped_gemm_register16x32_sync_v2(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  static std::atomic<int> grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  cudaError_t error = persistent_grid_limit(grouped_gemm_register16x32_sm86_v2_kernel, 0,
                                             grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  int blocks = 0;
  error = persistent_block_count(grid_limit, experts, output, max_expert_tokens, &blocks);
  if (error != cudaSuccess || blocks == 0) return error;
  grouped_gemm_register16x32_sm86_v2_kernel<<<blocks, kThreads, 0, caller_stream>>>(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output);
  return cudaGetLastError();
}

cudaError_t launch_grouped_gemm_register16x32_sync_v2(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(x_permuted, expert_weights, offsets,
                                                    y_permuted, experts, hidden, output,
                                                    max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  static std::atomic<int> grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  cudaError_t error = persistent_grid_limit(grouped_gemm_register16x32_sync_v2_kernel, 2,
                                            grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  int blocks = 0;
  error = persistent_block_count(grid_limit, experts, output, max_expert_tokens, &blocks);
  if (error != cudaSuccess || blocks == 0) return error;
  grouped_gemm_register16x32_sync_v2_kernel<<<blocks, kThreads, 0, caller_stream>>>(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output);
  return cudaGetLastError();
}

cudaError_t launch_grouped_gemm_register16x32_async_v3(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(x_permuted, expert_weights, offsets,
                                                    y_permuted, experts, hidden, output,
                                                    max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (hidden == 0 || hidden % kBlockDepth != 0 || output % kBlockColumns != 0 ||
      !aligned_16(x_permuted) || !aligned_16(expert_weights) || !aligned_16(y_permuted)) {
    return launch_grouped_gemm_register16x32_sync_v2(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  static std::atomic<int> grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  cudaError_t error = persistent_grid_limit(grouped_gemm_register16x32_async_v3_kernel, 2,
                                            grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  int blocks = 0;
  error = persistent_block_count(grid_limit, experts, output, max_expert_tokens, &blocks);
  if (error != cudaSuccess || blocks == 0) return error;
  grouped_gemm_register16x32_async_v3_kernel<<<blocks, kThreads, 0, caller_stream>>>(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output);
  return cudaGetLastError();
}

cudaError_t launch_grouped_gemm_register16x32_async_full_v4(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_arguments(x_permuted, expert_weights, offsets,
                                                    y_permuted, experts, hidden, output,
                                                    max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (hidden == 0 || hidden % kBlockDepth != 0 || output % kBlockColumns != 0 ||
      !aligned_16(x_permuted) || !aligned_16(expert_weights) || !aligned_16(y_permuted)) {
    return launch_grouped_gemm_register16x32_sync_v2(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
        max_expert_tokens, caller_stream);
  }
  static std::atomic<int> grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  cudaError_t error = persistent_grid_limit(grouped_gemm_register16x32_async_v3_kernel, 0,
                                            grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  int blocks = 0;
  error = persistent_block_count(grid_limit, experts, output, max_expert_tokens, &blocks);
  if (error != cudaSuccess || blocks == 0) return error;
  grouped_gemm_register16x32_async_v3_kernel<<<blocks, kThreads, 0, caller_stream>>>(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
