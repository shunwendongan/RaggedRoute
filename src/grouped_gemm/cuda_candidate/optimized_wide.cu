#include "grouped_optimized_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace raggedroute::ops {
namespace {

// grouped GEMM 的宽 tile 版本。
// 仍然消费 x_permuted[R,H] 和 expert_weights[E,H,O]，只是当输出宽度 O
// 足够大时更偏向宽 tile 的分块和 staging 策略。

constexpr int kWideBlockRows = 16;
constexpr int kWideBlockColumns = 64;
constexpr int kWideBlockDepth = 16;
constexpr int kWideSharedAStride = 20;
constexpr int kWideThreads = 256;
constexpr int kWideOutputsPerThread = 4;
constexpr int kMaxExperts = 64;
constexpr int kMaxCachedDevices = 32;

cudaError_t validate_wide_arguments(const float* x_permuted, const float* expert_weights,
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

__device__ __forceinline__ int find_wide_problem(const int* tile_prefix, int experts,
                                                 int tile_id) {
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

__device__ __forceinline__ void build_wide_tile_prefix_parallel(
    const std::int32_t* offsets, int experts, int tile_columns, int* tile_prefix,
    int* warp_totals) {
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int warp = thread >> 5;
  int tile_count = 0;
  if (thread < kMaxExperts && thread < experts) {
    const int rows = offsets[thread + 1] - offsets[thread];
    tile_count = ((rows + kWideBlockRows - 1) / kWideBlockRows) * tile_columns;
  }
  if (thread < kMaxExperts) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, tile_count, offset);
      if (lane >= offset) tile_count += other;
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

__device__ __forceinline__ void copy_wide_float4_async(float* shared_destination,
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

__device__ __forceinline__ void commit_wide_async_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_wide_async_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void stage_wide_sync(
    const float* x_permuted, const float* expert_weights, int expert, int begin, int rows,
    int hidden, int output, int tile_row, int tile_column, int k_base, float* x_tile,
    float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  const int x_local_row = thread / kWideBlockDepth;
  const int x_inner = thread - x_local_row * kWideBlockDepth;
  const int input_row = tile_row * kWideBlockRows + x_local_row;
  x_tile[x_local_row * kWideSharedAStride + x_inner] =
      input_row < rows
          ? x_permuted[static_cast<std::size_t>(begin + input_row) * hidden + k_base + x_inner]
          : 0.0F;

  for (int index = thread; index < kWideBlockDepth * kWideBlockColumns;
       index += kWideThreads) {
    const int inner = index / kWideBlockColumns;
    const int local_column = index - inner * kWideBlockColumns;
    weight_tile[inner * kWideBlockColumns + local_column] =
        expert_weights[(static_cast<std::size_t>(expert) * hidden + k_base + inner) * output +
                       tile_column * kWideBlockColumns + local_column];
  }
}

__device__ __forceinline__ void stage_wide_async(
    const float* x_permuted, const float* expert_weights, int expert, int begin, int hidden,
    int output, int tile_row, int tile_column, int k_base, float* x_tile,
    float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  if (thread < kWideBlockRows * kWideBlockDepth / 4) {
    const int element = thread * 4;
    const int local_row = element / kWideBlockDepth;
    const int inner = element - local_row * kWideBlockDepth;
    const float* source =
        x_permuted +
        static_cast<std::size_t>(begin + tile_row * kWideBlockRows + local_row) * hidden +
        k_base + inner;
    copy_wide_float4_async(x_tile + local_row * kWideSharedAStride + inner, source);
  }

  const int element = thread * 4;
  const int inner = element / kWideBlockColumns;
  const int local_column = element - inner * kWideBlockColumns;
  const float* source =
      expert_weights + (static_cast<std::size_t>(expert) * hidden + k_base + inner) * output +
      tile_column * kWideBlockColumns + local_column;
  copy_wide_float4_async(weight_tile + inner * kWideBlockColumns + local_column, source);
  commit_wide_async_group();
}

__device__ __forceinline__ void accumulate_wide(const float* x_tile, const float* weight_tile,
                                                int local_row, int local_column,
                                                float (&accumulator)[kWideOutputsPerThread]) {
#pragma unroll
  for (int inner = 0; inner < kWideBlockDepth; ++inner) {
    const float x = x_tile[local_row * kWideSharedAStride + inner];
#pragma unroll
    for (int column = 0; column < kWideOutputsPerThread; ++column) {
      const float weight =
          weight_tile[inner * kWideBlockColumns + local_column + column];
      accumulator[column] = __fmaf_rn(x, weight, accumulator[column]);
    }
  }
}

__device__ __forceinline__ void store_wide(
    float* y_permuted, int begin, int rows, int output, int tile_row, int tile_column,
    int local_row, int local_column, const float (&accumulator)[kWideOutputsPerThread]) {
  const int output_row = tile_row * kWideBlockRows + local_row;
  if (output_row >= rows) return;
  float* destination = y_permuted + static_cast<std::size_t>(begin + output_row) * output +
                       tile_column * kWideBlockColumns + local_column;
#pragma unroll
  for (int column = 0; column < kWideOutputsPerThread; ++column) {
    destination[column] = accumulator[column];
  }
}

__global__ __launch_bounds__(kWideThreads, 2) void grouped_gemm_register16x64_sm86_v3_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output) {
  __shared__ int tile_prefix[kMaxExperts + 1];
  __shared__ int prefix_warp_totals[2];
  __align__(16) __shared__ float x_tiles[2][kWideBlockRows][kWideSharedAStride];
  __align__(16) __shared__ float weight_tiles[2][kWideBlockDepth][kWideBlockColumns];

  const int tile_columns = output / kWideBlockColumns;
  build_wide_tile_prefix_parallel(offsets, experts, tile_columns, tile_prefix,
                                  prefix_warp_totals);

  const int thread = static_cast<int>(threadIdx.x);
  const int local_row = thread / (kWideBlockColumns / kWideOutputsPerThread);
  const int local_column =
      (thread % (kWideBlockColumns / kWideOutputsPerThread)) * kWideOutputsPerThread;
  const int total_tiles = tile_prefix[experts];
  for (int tile_id = static_cast<int>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int>(gridDim.x)) {
    const int expert = find_wide_problem(tile_prefix, experts, tile_id);
    const int local_tile = tile_id - tile_prefix[expert];
    const int tile_row = local_tile / tile_columns;
    const int tile_column = local_tile - tile_row * tile_columns;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    const bool full_rows = (tile_row + 1) * kWideBlockRows <= rows;
    float accumulator[kWideOutputsPerThread] = {};

    if (!full_rows) {
      for (int k_base = 0; k_base < hidden; k_base += kWideBlockDepth) {
        stage_wide_sync(x_permuted, expert_weights, expert, begin, rows, hidden, output,
                        tile_row, tile_column, k_base, &x_tiles[0][0][0],
                        &weight_tiles[0][0][0]);
        __syncthreads();
        accumulate_wide(&x_tiles[0][0][0], &weight_tiles[0][0][0], local_row,
                        local_column, accumulator);
        __syncthreads();
      }
    } else if (hidden > 0) {
      stage_wide_async(x_permuted, expert_weights, expert, begin, hidden, output, tile_row,
                       tile_column, 0, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
      wait_wide_async_group();
      __syncthreads();
      int stage = 0;
      for (int k_base = 0; k_base < hidden; k_base += kWideBlockDepth) {
        const int next_k = k_base + kWideBlockDepth;
        if (next_k < hidden) {
          stage_wide_async(x_permuted, expert_weights, expert, begin, hidden, output, tile_row,
                           tile_column, next_k, &x_tiles[stage ^ 1][0][0],
                           &weight_tiles[stage ^ 1][0][0]);
        }
        accumulate_wide(&x_tiles[stage][0][0], &weight_tiles[stage][0][0], local_row,
                        local_column, accumulator);
        if (next_k < hidden) wait_wide_async_group();
        __syncthreads();
        stage ^= 1;
      }
    }
    store_wide(y_permuted, begin, rows, output, tile_row, tile_column, local_row,
               local_column, accumulator);
  }
}

template <typename Kernel>
cudaError_t wide_grid_limit(Kernel kernel, std::atomic<int>* cache, int* grid_limit) {
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
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel,
                                                         kWideThreads, 0);
  if (error != cudaSuccess) return error;
  cached = multiprocessors * std::max(1, active_blocks_per_sm);
  cache[device].store(cached, std::memory_order_release);
  *grid_limit = cached;
  return cudaSuccess;
}

cudaError_t wide_block_count(int grid_limit, int experts, int output, int max_expert_tokens,
                             int* blocks) {
  const std::size_t upper_tiles =
      static_cast<std::size_t>(experts) *
      ((static_cast<std::size_t>(max_expert_tokens) + kWideBlockRows - 1) /
       kWideBlockRows) *
      (static_cast<std::size_t>(output) / kWideBlockColumns);
  if (upper_tiles == 0 || upper_tiles > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return upper_tiles == 0 ? cudaSuccess : cudaErrorInvalidConfiguration;
  }
  *blocks = std::min(grid_limit, static_cast<int>(upper_tiles));
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_grouped_gemm_sm86_fp32_v3(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_wide_arguments(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
      max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;

  if (hidden == 0 || hidden % kWideBlockDepth != 0 ||
      output % kWideBlockColumns != 0 || max_expert_tokens < 32 ||
      !aligned_16(x_permuted) || !aligned_16(expert_weights) || !aligned_16(y_permuted)) {
    return launch_grouped_gemm_sm86_fp32_v2(x_permuted, expert_weights, offsets, y_permuted,
                                             experts, hidden, output, max_expert_tokens,
                                             caller_stream);
  }

  static std::atomic<int> grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  cudaError_t error =
      wide_grid_limit(grouped_gemm_register16x64_sm86_v3_kernel, grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  int blocks = 0;
  error = wide_block_count(grid_limit, experts, output, max_expert_tokens, &blocks);
  if (error != cudaSuccess || blocks == 0) return error;
  grouped_gemm_register16x64_sm86_v3_kernel<<<blocks, kWideThreads, 0, caller_stream>>>(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
