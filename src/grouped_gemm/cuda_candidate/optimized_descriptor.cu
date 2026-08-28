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
constexpr int kGemmThreads = 128;
constexpr int kRowsPerThread = 2;
constexpr int kColumnsPerThread = 2;
constexpr int kMaxExperts = 64;
constexpr int kMaxCachedDevices = 32;
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kWorkspaceLimit = 64ULL * 1024ULL * 1024ULL;

struct TileDesc {
  std::int32_t expert_id;
  std::int32_t local_tile_id;
};

struct WorkspaceView {
  TileDesc* descriptors = nullptr;
  std::int32_t* expert_prefix = nullptr;
  std::int32_t* total_tiles = nullptr;
  std::int32_t* queue_counter = nullptr;
  std::size_t upper_tiles = 0;
};

static_assert(sizeof(TileDesc) == 8);
static_assert(kGemmThreads * kRowsPerThread * kColumnsPerThread ==
              kBlockRows * kBlockColumns);

std::size_t align_up(std::size_t value) {
  return (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

bool aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

bool aligned_workspace(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % kWorkspaceAlignment == 0;
}

bool descriptor_upper_bound(int experts, int output, int max_expert_tokens,
                            std::size_t* upper_tiles) {
  if (experts <= 0 || experts > kMaxExperts || output <= 0 || max_expert_tokens <= 0) {
    return false;
  }
  const std::size_t tile_rows =
      (static_cast<std::size_t>(max_expert_tokens) + kBlockRows - 1) / kBlockRows;
  const std::size_t tile_columns =
      (static_cast<std::size_t>(output) + kBlockColumns - 1) / kBlockColumns;
  if (tile_rows != 0 && static_cast<std::size_t>(experts) >
                            std::numeric_limits<std::size_t>::max() / tile_rows) {
    return false;
  }
  const std::size_t expert_rows = static_cast<std::size_t>(experts) * tile_rows;
  if (tile_columns != 0 && expert_rows >
                               std::numeric_limits<std::size_t>::max() / tile_columns) {
    return false;
  }
  *upper_tiles = expert_rows * tile_columns;
  return *upper_tiles > 0 && *upper_tiles <=
                                  static_cast<std::size_t>(std::numeric_limits<int>::max());
}

std::size_t descriptor_workspace_bytes(std::size_t upper_tiles) {
  if (upper_tiles > std::numeric_limits<std::size_t>::max() / sizeof(TileDesc)) return 0;
  std::size_t bytes = align_up(upper_tiles * sizeof(TileDesc));
  bytes += align_up((kMaxExperts + 1) * sizeof(std::int32_t));
  bytes += align_up(2 * sizeof(std::int32_t));
  return bytes <= kWorkspaceLimit ? bytes : 0;
}

WorkspaceView make_workspace_view(void* workspace, std::size_t upper_tiles) {
  auto* bytes = static_cast<std::uint8_t*>(workspace);
  WorkspaceView view;
  view.upper_tiles = upper_tiles;
  view.descriptors = reinterpret_cast<TileDesc*>(bytes);
  bytes += align_up(upper_tiles * sizeof(TileDesc));
  view.expert_prefix = reinterpret_cast<std::int32_t*>(bytes);
  bytes += align_up((kMaxExperts + 1) * sizeof(std::int32_t));
  view.total_tiles = reinterpret_cast<std::int32_t*>(bytes);
  view.queue_counter = view.total_tiles + 1;
  return view;
}

template <int Threads, bool CacheOrder>
__global__ void grouped_gemm_descriptor_prepass_kernel(
    const std::int32_t* offsets, TileDesc* descriptors, std::int32_t* expert_prefix,
    std::int32_t* total_tiles, std::int32_t* queue_counter, int experts, int output) {
  __shared__ std::int32_t tile_counts[kMaxExperts];
  __shared__ std::int32_t shared_prefix[kMaxExperts + 1];

  const int thread = static_cast<int>(threadIdx.x);
  if (thread < kMaxExperts) {
    int count = 0;
    if (thread < experts) {
      const int rows = offsets[thread + 1] - offsets[thread];
      const int tile_rows = (rows + kBlockRows - 1) / kBlockRows;
      const int tile_columns = output / kBlockColumns;
      count = tile_rows * tile_columns;
    }
    tile_counts[thread] = count;
  }
  __syncthreads();

  if (thread == 0) {
    shared_prefix[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      shared_prefix[expert + 1] = shared_prefix[expert] + tile_counts[expert];
    }
    for (int expert = experts; expert < kMaxExperts; ++expert) {
      shared_prefix[expert + 1] = shared_prefix[experts];
    }
    *total_tiles = shared_prefix[experts];
    *queue_counter = 0;
  }
  __syncthreads();

  if (thread <= experts) expert_prefix[thread] = shared_prefix[thread];

  const int lane = thread & 31;
  const int warp = thread >> 5;
  constexpr int kWarps = Threads / 32;
  const int tile_columns = output / kBlockColumns;
  for (int expert = warp; expert < experts; expert += kWarps) {
    const int begin = shared_prefix[expert];
    const int count = tile_counts[expert];
    const int rows = offsets[expert + 1] - offsets[expert];
    const int tile_rows = (rows + kBlockRows - 1) / kBlockRows;
    for (int sequence = lane; sequence < count; sequence += 32) {
      int local_tile = sequence;
      if constexpr (CacheOrder) {
        const int tile_column = sequence / tile_rows;
        const int tile_row = sequence - tile_column * tile_rows;
        local_tile = tile_row * tile_columns + tile_column;
      }
      descriptors[begin + sequence] = {expert, local_tile};
    }
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

__device__ __forceinline__ void stage_sync(
    const float* x_permuted, const float* expert_weights, int expert, int begin, int rows,
    int hidden, int output, int tile_row, int tile_column, int k_base, float* x_tile,
    float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  for (int index = thread; index < kBlockRows * kBlockDepth; index += kGemmThreads) {
    const int local_row = index / kBlockDepth;
    const int inner = index - local_row * kBlockDepth;
    const int input_row = tile_row * kBlockRows + local_row;
    const int input_inner = k_base + inner;
    x_tile[local_row * kSharedAStride + inner] =
        input_row < rows && input_inner < hidden
            ? x_permuted[static_cast<std::size_t>(begin + input_row) * hidden + input_inner]
            : 0.0F;
  }
  for (int index = thread; index < kBlockDepth * kBlockColumns; index += kGemmThreads) {
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

__device__ __forceinline__ void stage_sync_gather(
    const float* x, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, int expert, int begin, int rows, int hidden, int output,
    int tile_row, int tile_column, int k_base, float* x_tile, float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  for (int index = thread; index < kBlockRows * kBlockDepth; index += kGemmThreads) {
    const int local_row = index / kBlockDepth;
    const int inner = index - local_row * kBlockDepth;
    const int input_row = tile_row * kBlockRows + local_row;
    const int input_inner = k_base + inner;
    float value = 0.0F;
    if (input_row < rows && input_inner < hidden) {
      const int packed_row = begin + input_row;
      const int token = sorted_route[packed_row] / top_k;
      value = x[static_cast<std::size_t>(token) * hidden + input_inner];
    }
    x_tile[local_row * kSharedAStride + inner] = value;
  }
  for (int index = thread; index < kBlockDepth * kBlockColumns; index += kGemmThreads) {
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

__device__ __forceinline__ void stage_async_gather(
    const float* x, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, int expert, int begin, int hidden, int output,
    int tile_row, int tile_column, int k_base, float* x_tile, float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  if (thread < kBlockRows * kBlockDepth / 4) {
    const int element = thread * 4;
    const int local_row = element / kBlockDepth;
    const int inner = element - local_row * kBlockDepth;
    const int packed_row = begin + tile_row * kBlockRows + local_row;
    const int token = sorted_route[packed_row] / top_k;
    const float* source = x + static_cast<std::size_t>(token) * hidden + k_base + inner;
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
    const float* x_tile, const float* weight_tile, int local_row, int local_column,
    float (&accumulator)[kRowsPerThread][kColumnsPerThread]) {
#pragma unroll
  for (int inner = 0; inner < kBlockDepth; ++inner) {
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

template <bool Queue, bool Gather>
__global__ __launch_bounds__(kGemmThreads, 5) void grouped_gemm_descriptor_kernel(
    const float* x_input, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, const std::int32_t* offsets,
    const TileDesc* descriptors, const std::int32_t* total_tiles,
    std::int32_t* queue_counter, float* y_permuted, int hidden, int output) {
  __align__(16) __shared__ float x_tiles[2][kBlockRows][kSharedAStride];
  __align__(16) __shared__ float weight_tiles[2][kBlockDepth][kBlockColumns];
  __shared__ TileDesc shared_desc;
  __shared__ int shared_tile_id;

  int local_row = 0;
  int local_column = 0;
  thread_output_coordinates(&local_row, &local_column);
  const int tile_columns = output / kBlockColumns;
  int static_tile = static_cast<int>(blockIdx.x);

  while (true) {
    if (threadIdx.x == 0) {
      const int tile_id = Queue ? atomicAdd(queue_counter, 1) : static_tile;
      shared_tile_id = tile_id;
      if (tile_id < *total_tiles) shared_desc = descriptors[tile_id];
    }
    __syncthreads();
    if (shared_tile_id >= *total_tiles) break;

    const int expert = shared_desc.expert_id;
    const int local_tile = shared_desc.local_tile_id;
    const int tile_row = local_tile / tile_columns;
    const int tile_column = local_tile - tile_row * tile_columns;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    const bool full_rows = (tile_row + 1) * kBlockRows <= rows;
    float accumulator[kRowsPerThread][kColumnsPerThread] = {};

    if (!full_rows) {
      for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
        if constexpr (Gather) {
          stage_sync_gather(x_input, sorted_route, top_k, expert_weights, expert, begin, rows,
                            hidden, output, tile_row, tile_column, k_base,
                            &x_tiles[0][0][0], &weight_tiles[0][0][0]);
        } else {
          stage_sync(x_input, expert_weights, expert, begin, rows, hidden, output, tile_row,
                     tile_column, k_base, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
        }
        __syncthreads();
        accumulate(&x_tiles[0][0][0], &weight_tiles[0][0][0], local_row, local_column,
                   accumulator);
        __syncthreads();
      }
    } else if (hidden > 0) {
      if constexpr (Gather) {
        stage_async_gather(x_input, sorted_route, top_k, expert_weights, expert, begin, hidden,
                           output, tile_row, tile_column, 0, &x_tiles[0][0][0],
                           &weight_tiles[0][0][0]);
      } else {
        stage_async(x_input, expert_weights, expert, begin, hidden, output, tile_row,
                    tile_column, 0, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
      }
      wait_async_group();
      __syncthreads();
      int stage = 0;
      for (int k_base = 0; k_base < hidden; k_base += kBlockDepth) {
        const int next_k = k_base + kBlockDepth;
        if (next_k < hidden) {
          if constexpr (Gather) {
            stage_async_gather(x_input, sorted_route, top_k, expert_weights, expert, begin,
                               hidden, output, tile_row, tile_column, next_k,
                               &x_tiles[stage ^ 1][0][0], &weight_tiles[stage ^ 1][0][0]);
          } else {
            stage_async(x_input, expert_weights, expert, begin, hidden, output, tile_row,
                        tile_column, next_k, &x_tiles[stage ^ 1][0][0],
                        &weight_tiles[stage ^ 1][0][0]);
          }
        }
        accumulate(&x_tiles[stage][0][0], &weight_tiles[stage][0][0], local_row,
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
    __syncthreads();
    if constexpr (!Queue) static_tile += static_cast<int>(gridDim.x);
  }
}

template <typename Kernel>
cudaError_t persistent_grid_limit(Kernel kernel, std::atomic<int>* cache, int* grid_limit) {
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
                                                         kGemmThreads, 0);
  if (error != cudaSuccess) return error;
  cached = multiprocessors * std::max(1, active_blocks_per_sm);
  cache[device].store(cached, std::memory_order_release);
  *grid_limit = cached;
  return cudaSuccess;
}

template <int Threads, bool Queue, bool CacheOrder>
cudaError_t launch_descriptor_variant(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    WorkspaceView workspace, cudaStream_t stream) {
  grouped_gemm_descriptor_prepass_kernel<Threads, CacheOrder><<<1, Threads, 0, stream>>>(
      offsets, workspace.descriptors, workspace.expert_prefix, workspace.total_tiles,
      workspace.queue_counter, experts, output);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  static std::atomic<int> static_grid_cache[kMaxCachedDevices]{};
  static std::atomic<int> queue_grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  if constexpr (Queue) {
    error = persistent_grid_limit(grouped_gemm_descriptor_kernel<true, false>, queue_grid_cache,
                                  &grid_limit);
  } else {
    error = persistent_grid_limit(grouped_gemm_descriptor_kernel<false, false>, static_grid_cache,
                                  &grid_limit);
  }
  if (error != cudaSuccess) return error;
  const int blocks = std::min(grid_limit, static_cast<int>(workspace.upper_tiles));
  if (blocks <= 0) return cudaSuccess;
  grouped_gemm_descriptor_kernel<Queue, false><<<blocks, kGemmThreads, 0, stream>>>(
      x_permuted, nullptr, 1, expert_weights, offsets, workspace.descriptors, workspace.total_tiles,
      workspace.queue_counter, y_permuted, hidden, output);
  return cudaGetLastError();
}

bool is_descriptor_implementation(std::uint32_t implementation_id) {
  return implementation_id >= kGroupedGemmSm86Fp32V4ADescStaticT256Implementation &&
         implementation_id <= kGroupedGemmSm86Fp32V4BCacheOrderImplementation;
}

__global__ void grouped_gemm_gather_naive_kernel(
    const float* x, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, const std::int32_t* offsets, float* y_permuted,
    int experts, int hidden, int output, int column_tiles) {
  const int block = static_cast<int>(blockIdx.x);
  const int packed_row = block / column_tiles;
  if (packed_row >= offsets[experts]) return;
  const int column = (block - packed_row * column_tiles) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
  if (column >= output) return;
  int low = 0;
  int high = experts;
  while (low + 1 < high) {
    const int middle = (low + high) >> 1;
    if (offsets[middle] <= packed_row) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const int expert = low;
  const int token = sorted_route[packed_row] / top_k;
  float accumulator = 0.0F;
  for (int inner = 0; inner < hidden; ++inner) {
    accumulator = __fmaf_rn(
        x[static_cast<std::size_t>(token) * hidden + inner],
        expert_weights[(static_cast<std::size_t>(expert) * hidden + inner) * output + column],
        accumulator);
  }
  y_permuted[static_cast<std::size_t>(packed_row) * output + column] = accumulator;
}

cudaError_t launch_gather_naive(
    const float* x, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, const std::int32_t* offsets, float* y_permuted,
    int experts, int hidden, int output, int max_expert_tokens, cudaStream_t stream) {
  constexpr int kThreads = 128;
  const int column_tiles = (output + kThreads - 1) / kThreads;
  if (column_tiles <= 0 || max_expert_tokens >
                               std::numeric_limits<int>::max() / column_tiles) {
    return cudaErrorInvalidConfiguration;
  }
  const int blocks = max_expert_tokens * column_tiles;
  grouped_gemm_gather_naive_kernel<<<blocks, kThreads, 0, stream>>>(
      x, sorted_route, top_k, expert_weights, offsets, y_permuted, experts, hidden,
      output, column_tiles);
  return cudaGetLastError();
}

}  // namespace

std::size_t grouped_gemm_descriptor_workspace_size(int experts, int output,
                                                   int max_expert_tokens) noexcept {
  std::size_t upper_tiles = 0;
  if (!descriptor_upper_bound(experts, output, max_expert_tokens, &upper_tiles)) return 0;
  return descriptor_workspace_bytes(upper_tiles);
}

cudaError_t launch_grouped_gemm_sm86_fp32_v4_descriptor(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    void* workspace, std::size_t workspace_bytes, std::uint32_t implementation_id,
    cudaStream_t caller_stream) {
  if (!is_descriptor_implementation(implementation_id) || experts <= 0 ||
      experts > kMaxExperts || hidden < 0 || output < 0 || max_expert_tokens < 0) {
    return cudaErrorInvalidValue;
  }
  if (max_expert_tokens == 0 || output == 0) return cudaSuccess;
  if (offsets == nullptr || y_permuted == nullptr ||
      (hidden != 0 && (x_permuted == nullptr || expert_weights == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (hidden == 0 || hidden % kBlockDepth != 0 || output % kBlockColumns != 0 ||
      !aligned_16(x_permuted) || !aligned_16(expert_weights) || !aligned_16(y_permuted)) {
    return launch_grouped_gemm_sm86_fp32_v2(x_permuted, expert_weights, offsets, y_permuted,
                                            experts, hidden, output, max_expert_tokens,
                                            caller_stream);
  }
  std::size_t upper_tiles = 0;
  if (!descriptor_upper_bound(experts, output, max_expert_tokens, &upper_tiles)) {
    return cudaErrorInvalidConfiguration;
  }
  const std::size_t required = descriptor_workspace_bytes(upper_tiles);
  if (required == 0 || workspace == nullptr || workspace_bytes < required ||
      !aligned_workspace(workspace)) {
    return launch_grouped_gemm_sm86_fp32_v2(x_permuted, expert_weights, offsets, y_permuted,
                                            experts, hidden, output, max_expert_tokens,
                                            caller_stream);
  }
  const WorkspaceView view = make_workspace_view(workspace, upper_tiles);
  switch (implementation_id) {
    case kGroupedGemmSm86Fp32V4ADescStaticT256Implementation:
      return launch_descriptor_variant<256, false, false>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    case kGroupedGemmSm86Fp32V4ADescStaticT512Implementation:
      return launch_descriptor_variant<512, false, false>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    case kGroupedGemmSm86Fp32V4ADescStaticT1024Implementation:
      return launch_descriptor_variant<1024, false, false>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    case kGroupedGemmSm86Fp32V4ADescQueueT256Implementation:
      return launch_descriptor_variant<256, true, false>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    case kGroupedGemmSm86Fp32V4ADescQueueT512Implementation:
      return launch_descriptor_variant<512, true, false>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    case kGroupedGemmSm86Fp32V4ADescQueueT1024Implementation:
      return launch_descriptor_variant<1024, true, false>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    case kGroupedGemmSm86Fp32V4BCacheOrderImplementation:
      return launch_descriptor_variant<256, false, true>(
          x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output,
          max_expert_tokens, view, caller_stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t launch_grouped_gemm_sm86_fp32_gather_v1(
    const float* x, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, const std::int32_t* offsets, float* y_permuted,
    int experts, int hidden, int output, int max_expert_tokens, void* workspace,
    std::size_t workspace_bytes, cudaStream_t caller_stream) {
  if (experts <= 0 || experts > kMaxExperts || top_k <= 0 || top_k > 64 || hidden < 0 ||
      output < 0 || max_expert_tokens < 0) {
    return cudaErrorInvalidValue;
  }
  if (max_expert_tokens == 0 || output == 0) return cudaSuccess;
  if (sorted_route == nullptr || offsets == nullptr || y_permuted == nullptr ||
      (hidden != 0 && (x == nullptr || expert_weights == nullptr))) {
    return cudaErrorInvalidValue;
  }
  std::size_t upper_tiles = 0;
  const bool optimized_contract =
      hidden > 0 && hidden % kBlockDepth == 0 && output % kBlockColumns == 0 &&
      aligned_16(x) && aligned_16(expert_weights) && aligned_16(y_permuted) &&
      descriptor_upper_bound(experts, output, max_expert_tokens, &upper_tiles);
  const std::size_t required =
      optimized_contract ? descriptor_workspace_bytes(upper_tiles) : 0;
  if (!optimized_contract || required == 0 || workspace == nullptr ||
      workspace_bytes < required || !aligned_workspace(workspace)) {
    return launch_gather_naive(x, sorted_route, top_k, expert_weights, offsets, y_permuted,
                               experts, hidden, output, max_expert_tokens, caller_stream);
  }

  const WorkspaceView view = make_workspace_view(workspace, upper_tiles);
  grouped_gemm_descriptor_prepass_kernel<256, false><<<1, 256, 0, caller_stream>>>(
      offsets, view.descriptors, view.expert_prefix, view.total_tiles, view.queue_counter,
      experts, output);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  static std::atomic<int> gather_grid_cache[kMaxCachedDevices]{};
  int grid_limit = 0;
  error = persistent_grid_limit(grouped_gemm_descriptor_kernel<false, true>,
                                gather_grid_cache, &grid_limit);
  if (error != cudaSuccess) return error;
  const int blocks = std::min(grid_limit, static_cast<int>(view.upper_tiles));
  if (blocks <= 0) return cudaSuccess;
  grouped_gemm_descriptor_kernel<false, true><<<blocks, kGemmThreads, 0, caller_stream>>>(
      x, sorted_route, top_k, expert_weights, offsets, view.descriptors, view.total_tiles,
      view.queue_counter, y_permuted, hidden, output);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
