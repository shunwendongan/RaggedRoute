#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr int kBlockRows = 32;
constexpr int kBlockColumns = 32;
constexpr int kBlockDepth = 16;
// Padding A's 16-float logical K row to 20 floats separates the four distinct
// row broadcasts used by a warp across shared-memory banks.
constexpr int kSharedAStride = 20;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr int kRowsPerThread = 4;
constexpr int kColumnsPerThread = 2;
constexpr int kPipelineStages = 2;
constexpr int kMinBlocksPerSm = 4;
constexpr unsigned int kMaxGridY = 65535U;

static_assert(kWarpsPerBlock == 4);
static_assert(kBlockRows * kBlockDepth / 4 == kThreadsPerBlock);
static_assert(kBlockDepth * kBlockColumns / 4 == kThreadsPerBlock);
static_assert(kThreadsPerBlock * kRowsPerThread * kColumnsPerThread == kBlockRows * kBlockColumns);
static_assert(kSharedAStride % 4 == 0);

__device__ __forceinline__ void copy_float4(float* destination, const float* source) {
  *reinterpret_cast<float4*>(destination) = *reinterpret_cast<const float4*>(source);
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
  copy_float4(shared_destination, global_source);
#endif
}

__device__ __forceinline__ void commit_async_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void wait_async_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups) : "memory");
#endif
}

__device__ __forceinline__ void stage_tile_sync(const float* a, const float* b, int n, int k,
                                                int block_row, int block_column, int k_base,
                                                float* a_tile, float* b_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  const int a_load_row = thread >> 2;
  const int a_load_column = (thread & 3) * 4;
  const int b_load_row = thread >> 3;
  const int b_load_column = (thread & 7) * 4;

  const float* a_source = a + static_cast<std::size_t>(block_row * kBlockRows + a_load_row) * k +
                          k_base + a_load_column;
  const float* b_source = b + static_cast<std::size_t>(k_base + b_load_row) * n +
                          block_column * kBlockColumns + b_load_column;
  copy_float4(a_tile + a_load_row * kSharedAStride + a_load_column, a_source);
  copy_float4(b_tile + b_load_row * kBlockColumns + b_load_column, b_source);
}

__device__ __forceinline__ void stage_tile_async(const float* a, const float* b, int n, int k,
                                                 int block_row, int block_column, int k_base,
                                                 float* a_tile, float* b_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  const int a_load_row = thread >> 2;
  const int a_load_column = (thread & 3) * 4;
  const int b_load_row = thread >> 3;
  const int b_load_column = (thread & 7) * 4;

  const float* a_source = a + static_cast<std::size_t>(block_row * kBlockRows + a_load_row) * k +
                          k_base + a_load_column;
  const float* b_source = b + static_cast<std::size_t>(k_base + b_load_row) * n +
                          block_column * kBlockColumns + b_load_column;
  copy_float4_async(a_tile + a_load_row * kSharedAStride + a_load_column, a_source);
  copy_float4_async(b_tile + b_load_row * kBlockColumns + b_load_column, b_source);
  // A and B form one per-thread cp.async group. The mainloop keeps at most the
  // current and next groups outstanding.
  commit_async_group();
}

__device__ __forceinline__ void accumulate_tile(
    const float* a_tile, const float* b_tile, int local_row, int local_column,
    float (&accumulator)[kRowsPerThread][kColumnsPerThread]) {
#pragma unroll
  for (int inner = 0; inner < kBlockDepth; ++inner) {
    float a_fragment[kRowsPerThread];
    float b_fragment[kColumnsPerThread];
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
      a_fragment[row] = a_tile[(local_row + row * 4) * kSharedAStride + inner];
    }
#pragma unroll
    for (int column = 0; column < kColumnsPerThread; ++column) {
      b_fragment[column] = b_tile[inner * kBlockColumns + local_column + column];
    }
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
      for (int column = 0; column < kColumnsPerThread; ++column) {
        accumulator[row][column] =
            __fmaf_rn(a_fragment[row], b_fragment[column], accumulator[row][column]);
      }
    }
  }
}

__device__ __forceinline__ void store_accumulator(
    float* c, int n, int block_row, int block_column, int local_row, int local_column,
    const float (&accumulator)[kRowsPerThread][kColumnsPerThread]) {
  const int output_column = block_column * kBlockColumns + local_column;
#pragma unroll
  for (int row = 0; row < kRowsPerThread; ++row) {
    const int output_row = block_row * kBlockRows + local_row + row * 4;
    float2 value = make_float2(accumulator[row][0], accumulator[row][1]);
    *reinterpret_cast<float2*>(c + static_cast<std::size_t>(output_row) * n + output_column) =
        value;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock, kMinBlocksPerSm) void
dense_gemm_register_tiled_v2_sync_kernel(const float* a, const float* b, float* c, int n, int k) {
  __align__(16) __shared__ float a_tile[kBlockRows][kSharedAStride];
  __align__(16) __shared__ float b_tile[kBlockDepth][kBlockColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  // Four warps cover a 2x2 arrangement of 16x16 warp tiles. Within a warp,
  // each lane owns four rows and two adjacent columns (eight accumulators).
  const int local_row = (warp >> 1) * 16 + (lane >> 3);
  const int local_column = (warp & 1) * 16 + (lane & 7) * 2;
  const int block_row = static_cast<int>(blockIdx.y);
  const int block_column = static_cast<int>(blockIdx.x);
  float accumulator[kRowsPerThread][kColumnsPerThread] = {};

  for (int k_base = 0; k_base < k; k_base += kBlockDepth) {
    stage_tile_sync(a, b, n, k, block_row, block_column, k_base, &a_tile[0][0], &b_tile[0][0]);
    __syncthreads();
    accumulate_tile(&a_tile[0][0], &b_tile[0][0], local_row, local_column, accumulator);
    if (k_base + kBlockDepth < k) __syncthreads();
  }

  store_accumulator(c, n, block_row, block_column, local_row, local_column, accumulator);
}

__global__ __launch_bounds__(kThreadsPerBlock, kMinBlocksPerSm) void
dense_gemm_register_tiled_v2_async_kernel(const float* a, const float* b, float* c, int n, int k) {
  __align__(16) __shared__ float a_tiles[kPipelineStages][kBlockRows][kSharedAStride];
  __align__(16) __shared__ float b_tiles[kPipelineStages][kBlockDepth][kBlockColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int local_row = (warp >> 1) * 16 + (lane >> 3);
  const int local_column = (warp & 1) * 16 + (lane & 7) * 2;
  const int block_row = static_cast<int>(blockIdx.y);
  const int block_column = static_cast<int>(blockIdx.x);
  const int tile_count = k / kBlockDepth;
  float accumulator[kRowsPerThread][kColumnsPerThread] = {};

  stage_tile_async(a, b, n, k, block_row, block_column, 0, &a_tiles[0][0][0], &b_tiles[0][0][0]);
  for (int tile = 0; tile < tile_count; ++tile) {
    const int current_stage = tile & 1;
    const int next_tile = tile + 1;
    if (next_tile < tile_count) {
      const int next_stage = next_tile & 1;
      stage_tile_async(a, b, n, k, block_row, block_column, next_tile * kBlockDepth,
                       &a_tiles[next_stage][0][0], &b_tiles[next_stage][0][0]);
      // Keep the newest group in flight and wait for the current stage.
      wait_async_group<1>();
    } else {
      wait_async_group<0>();
    }
    __syncthreads();
    accumulate_tile(&a_tiles[current_stage][0][0], &b_tiles[current_stage][0][0], local_row,
                    local_column, accumulator);
    // Before the next iteration overwrites this stage, every warp must have
    // completed its shared-memory reads. No trailing barrier is needed.
    if (next_tile < tile_count) __syncthreads();
  }

  store_accumulator(c, n, block_row, block_column, local_row, local_column, accumulator);
}

cudaError_t validate_v2_grid(int m, int n) {
  const std::size_t grid_x = static_cast<std::size_t>(n) / kBlockColumns;
  const std::size_t grid_y = static_cast<std::size_t>(m) / kBlockRows;
  if (grid_x == 0 || grid_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      grid_y == 0 || grid_y > kMaxGridY) {
    return cudaErrorInvalidConfiguration;
  }
  return cudaSuccess;
}

bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

cudaError_t validate_v2_arguments(const float* a, const float* b, const float* c, int m, int n,
                                  int k) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0 ||
      m % kBlockRows != 0 || n % kBlockColumns != 0 || k % kBlockDepth != 0 || !is_aligned_16(a) ||
      !is_aligned_16(b) || !is_aligned_16(c)) {
    return cudaErrorInvalidValue;
  }
  return validate_v2_grid(m, n);
}

}  // namespace

cudaError_t launch_dense_gemm_register_tiled_v2_sync(const float* a, const float* b, float* c,
                                                     int m, int n, int k,
                                                     cudaStream_t caller_stream) {
  const cudaError_t validation = validate_v2_arguments(a, b, c, m, n, k);
  if (validation != cudaSuccess) return validation;
  const dim3 grid(static_cast<unsigned int>(n / kBlockColumns),
                  static_cast<unsigned int>(m / kBlockRows));
  dense_gemm_register_tiled_v2_sync_kernel<<<grid, kThreadsPerBlock, 0, caller_stream>>>(a, b, c, n,
                                                                                         k);
  return cudaGetLastError();
}

cudaError_t launch_dense_gemm_register_tiled_v2_async(const float* a, const float* b, float* c,
                                                      int m, int n, int k,
                                                      cudaStream_t caller_stream) {
  const cudaError_t validation = validate_v2_arguments(a, b, c, m, n, k);
  if (validation != cudaSuccess) return validation;
  const dim3 grid(static_cast<unsigned int>(n / kBlockColumns),
                  static_cast<unsigned int>(m / kBlockRows));
  dense_gemm_register_tiled_v2_async_kernel<<<grid, kThreadsPerBlock, 0, caller_stream>>>(a, b, c,
                                                                                          n, k);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
