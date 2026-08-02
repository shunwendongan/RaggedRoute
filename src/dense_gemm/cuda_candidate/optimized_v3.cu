#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

// V3 enlarges the CTA from 32x32 to 64x32 without increasing the thread count.
// Four 32x16 warp tiles use an 8x2 register microtile. This preserves the v2
// vector/cp.async staging mechanism while increasing A reuse within a CTA.
constexpr int kBlockRows = 64;
constexpr int kBlockColumns = 32;
constexpr int kBlockDepth = 16;
constexpr int kSharedAStride = 20;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr int kRowsPerThread = 8;
constexpr int kColumnsPerThread = 2;
constexpr int kPipelineStages = 2;
constexpr int kMinBlocksPerSm = 4;
constexpr unsigned int kMaxGridY = 65535U;

static_assert(kWarpsPerBlock == 4);
static_assert(kBlockRows * kBlockDepth / 4 == 2 * kThreadsPerBlock);
static_assert(kBlockDepth * kBlockColumns / 4 == kThreadsPerBlock);
static_assert(kThreadsPerBlock * kRowsPerThread * kColumnsPerThread ==
              kBlockRows * kBlockColumns);
static_assert(kSharedAStride % 4 == 0);

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
  *reinterpret_cast<float4*>(shared_destination) = *reinterpret_cast<const float4*>(global_source);
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

__device__ __forceinline__ void stage_tile_async(const float* a, const float* b, int n, int k,
                                                 int block_row, int block_column, int k_base,
                                                 float* a_tile, float* b_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  const int a_load_row = thread >> 2;
  const int a_load_column = (thread & 3) * 4;
  const int b_load_row = thread >> 3;
  const int b_load_column = (thread & 7) * 4;

  const float* a_top_source =
      a + static_cast<std::size_t>(block_row * kBlockRows + a_load_row) * k + k_base +
      a_load_column;
  const float* a_bottom_source =
      a + static_cast<std::size_t>(block_row * kBlockRows + a_load_row + kBlockRows / 2) * k +
      k_base + a_load_column;
  const float* b_source = b + static_cast<std::size_t>(k_base + b_load_row) * n +
                          block_column * kBlockColumns + b_load_column;
  copy_float4_async(a_tile + a_load_row * kSharedAStride + a_load_column, a_top_source);
  copy_float4_async(a_tile + (a_load_row + kBlockRows / 2) * kSharedAStride + a_load_column,
                    a_bottom_source);
  copy_float4_async(b_tile + b_load_row * kBlockColumns + b_load_column, b_source);
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
    const float2 value = make_float2(accumulator[row][0], accumulator[row][1]);
    *reinterpret_cast<float2*>(c + static_cast<std::size_t>(output_row) * n + output_column) =
        value;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock, kMinBlocksPerSm) void
dense_gemm_register_tiled_v3_64x32_async_kernel(const float* a, const float* b, float* c, int n,
                                                 int k) {
  __align__(16) __shared__ float a_tiles[kPipelineStages][kBlockRows][kSharedAStride];
  __align__(16) __shared__ float b_tiles[kPipelineStages][kBlockDepth][kBlockColumns];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int local_row = (warp >> 1) * 32 + (lane >> 3);
  const int local_column = (warp & 1) * 16 + (lane & 7) * 2;
  const int block_row = static_cast<int>(blockIdx.y);
  const int block_column = static_cast<int>(blockIdx.x);
  const int tile_count = k / kBlockDepth;
  float accumulator[kRowsPerThread][kColumnsPerThread] = {};

  stage_tile_async(a, b, n, k, block_row, block_column, 0, &a_tiles[0][0][0],
                   &b_tiles[0][0][0]);
  for (int tile = 0; tile < tile_count; ++tile) {
    const int current_stage = tile & 1;
    const int next_tile = tile + 1;
    if (next_tile < tile_count) {
      const int next_stage = next_tile & 1;
      stage_tile_async(a, b, n, k, block_row, block_column, next_tile * kBlockDepth,
                       &a_tiles[next_stage][0][0], &b_tiles[next_stage][0][0]);
      wait_async_group<1>();
    } else {
      wait_async_group<0>();
    }
    __syncthreads();
    accumulate_tile(&a_tiles[current_stage][0][0], &b_tiles[current_stage][0][0], local_row,
                    local_column, accumulator);
    if (next_tile < tile_count) __syncthreads();
  }

  store_accumulator(c, n, block_row, block_column, local_row, local_column, accumulator);
}

cudaError_t validate_v3_grid(int m, int n) {
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

cudaError_t validate_v3_arguments(const float* a, const float* b, const float* c, int m, int n,
                                  int k) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0 ||
      m % kBlockRows != 0 || n % kBlockColumns != 0 || k % kBlockDepth != 0 || !is_aligned_16(a) ||
      !is_aligned_16(b) || !is_aligned_16(c)) {
    return cudaErrorInvalidValue;
  }
  return validate_v3_grid(m, n);
}

}  // namespace

cudaError_t launch_dense_gemm_register_tiled_v3_64x32_async(const float* a, const float* b,
                                                            float* c, int m, int n, int k,
                                                            cudaStream_t caller_stream) {
  const cudaError_t validation = validate_v3_arguments(a, b, c, m, n, k);
  if (validation != cudaSuccess) return validation;
  const dim3 grid(static_cast<unsigned int>(n / kBlockColumns),
                  static_cast<unsigned int>(m / kBlockRows));
  dense_gemm_register_tiled_v3_64x32_async_kernel<<<grid, kThreadsPerBlock, 0, caller_stream>>>(
      a, b, c, n, k);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
