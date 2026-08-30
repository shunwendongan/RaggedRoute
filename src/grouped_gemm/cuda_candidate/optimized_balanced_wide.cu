#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "grouped_optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr int kDepth = 16;
constexpr int kAStride = 20;
constexpr int kThreads = 256;
constexpr int kMaxExperts = 64;

template <int BlockRows, int BlockColumns>
struct WideConfig {
  static constexpr int kRowsPerThread = BlockRows / 8;
  static constexpr int kColumnsPerThread = BlockColumns / 32;
  static_assert(BlockRows == 16 || BlockRows == 32);
  static_assert(BlockColumns == 64 || BlockColumns == 128);
  static_assert(kColumnsPerThread == 2 || kColumnsPerThread == 4);
  static_assert(kThreads * kRowsPerThread * kColumnsPerThread == BlockRows * BlockColumns);
};

bool aligned_16_wide(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

cudaError_t validate_wide_arguments(const float* x_permuted, const float* expert_weights,
                                    const std::int32_t* offsets, float* y_permuted, int experts,
                                    int hidden, int output, int max_expert_tokens) {
  if (experts <= 0 || experts > kMaxExperts || hidden < 0 || output < 0 || max_expert_tokens < 0) {
    return cudaErrorInvalidValue;
  }
  if (max_expert_tokens == 0 || output == 0) return cudaSuccess;
  if (offsets == nullptr || y_permuted == nullptr ||
      (hidden != 0 && (x_permuted == nullptr || expert_weights == nullptr))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

__device__ __forceinline__ void copy_float4_async_wide(float* shared_destination,
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

__device__ __forceinline__ void commit_async_wide() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_async_wide() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

template <int BlockRows, int BlockColumns>
__device__ __forceinline__ void stage_sync_wide(const float* x_permuted,
                                                const float* expert_weights, int expert, int begin,
                                                int rows, int hidden, int output, int tile_row,
                                                int tile_column, int k_base, float* x_tile,
                                                float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  for (int index = thread; index < BlockRows * kDepth; index += kThreads) {
    const int local_row = index / kDepth;
    const int inner = index - local_row * kDepth;
    const int input_row = tile_row * BlockRows + local_row;
    x_tile[local_row * kAStride + inner] =
        input_row < rows
            ? x_permuted[static_cast<std::size_t>(begin + input_row) * hidden + k_base + inner]
            : 0.0F;
  }
  for (int index = thread; index < kDepth * BlockColumns; index += kThreads) {
    const int inner = index / BlockColumns;
    const int local_column = index - inner * BlockColumns;
    weight_tile[inner * BlockColumns + local_column] =
        expert_weights[(static_cast<std::size_t>(expert) * hidden + k_base + inner) * output +
                       tile_column * BlockColumns + local_column];
  }
}

template <int BlockRows, int BlockColumns>
__device__ __forceinline__ void stage_async_wide(const float* x_permuted,
                                                 const float* expert_weights, int expert, int begin,
                                                 int hidden, int output, int tile_row,
                                                 int tile_column, int k_base, float* x_tile,
                                                 float* weight_tile) {
  const int thread = static_cast<int>(threadIdx.x);
  constexpr int kAVectors = BlockRows * kDepth / 4;
  if (thread < kAVectors) {
    const int element = thread * 4;
    const int local_row = element / kDepth;
    const int inner = element - local_row * kDepth;
    const float* source =
        x_permuted + static_cast<std::size_t>(begin + tile_row * BlockRows + local_row) * hidden +
        k_base + inner;
    copy_float4_async_wide(x_tile + local_row * kAStride + inner, source);
  }
  constexpr int kWeightVectors = kDepth * BlockColumns / 4;
  for (int vector = thread; vector < kWeightVectors; vector += kThreads) {
    const int element = vector * 4;
    const int inner = element / BlockColumns;
    const int local_column = element - inner * BlockColumns;
    const float* source = expert_weights +
                          (static_cast<std::size_t>(expert) * hidden + k_base + inner) * output +
                          tile_column * BlockColumns + local_column;
    copy_float4_async_wide(weight_tile + inner * BlockColumns + local_column, source);
  }
  commit_async_wide();
}

template <int BlockRows, int BlockColumns>
__device__ __forceinline__ void accumulate_wide(
    const float* x_tile, const float* weight_tile, int local_row, int local_column,
    float (&accumulator)[WideConfig<BlockRows, BlockColumns>::kRowsPerThread]
                        [WideConfig<BlockRows, BlockColumns>::kColumnsPerThread]) {
#pragma unroll
  for (int inner = 0; inner < kDepth; ++inner) {
    float x_values[WideConfig<BlockRows, BlockColumns>::kRowsPerThread];
    float weight_values[WideConfig<BlockRows, BlockColumns>::kColumnsPerThread];
#pragma unroll
    for (int row = 0; row < WideConfig<BlockRows, BlockColumns>::kRowsPerThread; ++row) {
      x_values[row] = x_tile[(local_row + row) * kAStride + inner];
    }
#pragma unroll
    for (int column = 0; column < WideConfig<BlockRows, BlockColumns>::kColumnsPerThread;
         ++column) {
      weight_values[column] = weight_tile[inner * BlockColumns + local_column + column];
    }
#pragma unroll
    for (int row = 0; row < WideConfig<BlockRows, BlockColumns>::kRowsPerThread; ++row) {
#pragma unroll
      for (int column = 0; column < WideConfig<BlockRows, BlockColumns>::kColumnsPerThread;
           ++column) {
        accumulator[row][column] =
            __fmaf_rn(x_values[row], weight_values[column], accumulator[row][column]);
      }
    }
  }
}

template <int BlockRows, int BlockColumns>
__device__ __forceinline__ void store_wide(
    float* y_permuted, int begin, int rows, int output, int tile_row, int tile_column,
    int local_row, int local_column,
    const float (&accumulator)[WideConfig<BlockRows, BlockColumns>::kRowsPerThread]
                              [WideConfig<BlockRows, BlockColumns>::kColumnsPerThread]) {
#pragma unroll
  for (int row = 0; row < WideConfig<BlockRows, BlockColumns>::kRowsPerThread; ++row) {
    const int output_row = tile_row * BlockRows + local_row + row;
    if (output_row < rows) {
      float* destination = y_permuted + static_cast<std::size_t>(begin + output_row) * output +
                           tile_column * BlockColumns + local_column;
      if constexpr (WideConfig<BlockRows, BlockColumns>::kColumnsPerThread == 4) {
        const float4 value = {accumulator[row][0], accumulator[row][1], accumulator[row][2],
                              accumulator[row][3]};
        *reinterpret_cast<float4*>(destination) = value;
      } else {
        const float2 value = {accumulator[row][0], accumulator[row][1]};
        *reinterpret_cast<float2*>(destination) = value;
      }
    }
  }
}

// Single-variable v6 experiment: enlarge only the balanced direct-path tile from 16x32 to
// 32x128. The 4x4 thread micro-tile preserves coalesced float4 stores while reusing each staged
// A value across 128 columns and each staged weight across 32 rows. This directly tests the v5
// NCU diagnosis that fine-grained CTAs amplify global requests at larger balanced shapes.
template <int BlockRows, int BlockColumns>
__global__ __launch_bounds__(kThreads, 2) void grouped_gemm_sm86_balanced_wide_kernel(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int hidden, int output) {
  const int expert = static_cast<int>(blockIdx.z);
  const int begin = offsets[expert];
  const int rows = offsets[expert + 1] - begin;
  const int tile_row = static_cast<int>(blockIdx.y);
  if (tile_row * BlockRows >= rows) return;
  const int tile_column = static_cast<int>(blockIdx.x);

  __align__(16) __shared__ float x_tiles[2][BlockRows][kAStride];
  __align__(16) __shared__ float weight_tiles[2][kDepth][BlockColumns];

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int local_row = warp * WideConfig<BlockRows, BlockColumns>::kRowsPerThread;
  const int local_column = lane * WideConfig<BlockRows, BlockColumns>::kColumnsPerThread;
  const bool full_rows = (tile_row + 1) * BlockRows <= rows;
  float accumulator[WideConfig<BlockRows, BlockColumns>::kRowsPerThread]
                   [WideConfig<BlockRows, BlockColumns>::kColumnsPerThread] = {};

  if (!full_rows) {
    for (int k_base = 0; k_base < hidden; k_base += kDepth) {
      stage_sync_wide<BlockRows, BlockColumns>(
          x_permuted, expert_weights, expert, begin, rows, hidden, output, tile_row, tile_column,
          k_base, &x_tiles[0][0][0], &weight_tiles[0][0][0]);
      __syncthreads();
      accumulate_wide<BlockRows, BlockColumns>(&x_tiles[0][0][0], &weight_tiles[0][0][0],
                                               local_row, local_column, accumulator);
      __syncthreads();
    }
  } else if (hidden > 0) {
    stage_async_wide<BlockRows, BlockColumns>(x_permuted, expert_weights, expert, begin, hidden,
                                              output, tile_row, tile_column, 0, &x_tiles[0][0][0],
                                              &weight_tiles[0][0][0]);
    wait_async_wide();
    __syncthreads();
    int stage = 0;
    for (int k_base = 0; k_base < hidden; k_base += kDepth) {
      const int next_k = k_base + kDepth;
      if (next_k < hidden) {
        stage_async_wide<BlockRows, BlockColumns>(
            x_permuted, expert_weights, expert, begin, hidden, output, tile_row, tile_column,
            next_k, &x_tiles[stage ^ 1][0][0], &weight_tiles[stage ^ 1][0][0]);
      }
      accumulate_wide<BlockRows, BlockColumns>(&x_tiles[stage][0][0],
                                               &weight_tiles[stage][0][0], local_row, local_column,
                                               accumulator);
      if (next_k < hidden) {
        wait_async_wide();
        __syncthreads();
        stage ^= 1;
      }
    }
    __syncthreads();
  }
  store_wide<BlockRows, BlockColumns>(y_permuted, begin, rows, output, tile_row, tile_column,
                                      local_row, local_column, accumulator);
}

template <int BlockRows, int BlockColumns>
cudaError_t launch_balanced_wide_kernel(const float* x_permuted, const float* expert_weights,
                                        const std::int32_t* offsets, float* y_permuted, int experts,
                                        int hidden, int output, int max_expert_tokens,
                                        cudaStream_t caller_stream) {
  const std::size_t grid_x = static_cast<std::size_t>(output) / BlockColumns;
  const std::size_t grid_y =
      (static_cast<std::size_t>(max_expert_tokens) + BlockRows - 1) / BlockRows;
  if (grid_x == 0 || grid_y == 0 ||
      grid_x > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) ||
      grid_y > 65535U) {
    return cudaErrorInvalidConfiguration;
  }
  grouped_gemm_sm86_balanced_wide_kernel<BlockRows, BlockColumns>
      <<<dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
              static_cast<unsigned int>(experts)),
         kThreads, 0, caller_stream>>>(x_permuted, expert_weights, offsets, y_permuted, hidden,
                                       output);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_grouped_gemm_sm86_fp32_v6_balanced_32x128(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens, int route_pairs,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_wide_arguments(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (route_pairs <= 0 || max_expert_tokens > route_pairs) return cudaErrorInvalidValue;

  const std::int64_t average_ceiling =
      (static_cast<std::int64_t>(route_pairs) + experts - 1) / experts;
  const bool balanced = average_ceiling >= 32 &&
                        static_cast<std::int64_t>(max_expert_tokens) <= 2LL * average_ceiling;
  if (!balanced || hidden == 0 || hidden % kDepth != 0 || output % 128 != 0 ||
      !aligned_16_wide(x_permuted) || !aligned_16_wide(expert_weights) ||
      !aligned_16_wide(y_permuted)) {
    return launch_grouped_gemm_sm86_fp32_v5_balanced_direct(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens,
        route_pairs, caller_stream);
  }

  return launch_balanced_wide_kernel<32, 128>(x_permuted, expert_weights, offsets, y_permuted,
                                              experts, hidden, output, max_expert_tokens,
                                              caller_stream);
}

// Single-variable v8 experiment: keep the v6 128-column reuse, direct grid, strict-FP32
// mainloop, two-stage cp.async pipeline, and selector, but halve tile-M from 32 to 16.  On the
// RTX 3080 the v6 T2048 case launches too few useful row tiles to fill 68 SMs; this variant trades
// additional weight-tile loads for more CTAs and a shorter accumulator live range.  Non-selected
// shapes retain the complete v6 hybrid portfolio, including its v5/v2 fallbacks.
cudaError_t launch_grouped_gemm_sm86_fp32_v8_balanced_16x128(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens, int route_pairs,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_wide_arguments(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (route_pairs <= 0 || max_expert_tokens > route_pairs) return cudaErrorInvalidValue;

  const std::int64_t average_ceiling =
      (static_cast<std::int64_t>(route_pairs) + experts - 1) / experts;
  const bool balanced = average_ceiling >= 32 &&
                        static_cast<std::int64_t>(max_expert_tokens) <= 2LL * average_ceiling;
  if (!balanced || hidden == 0 || hidden % kDepth != 0 || output % 128 != 0 ||
      !aligned_16_wide(x_permuted) || !aligned_16_wide(expert_weights) ||
      !aligned_16_wide(y_permuted)) {
    return launch_grouped_gemm_sm86_fp32_v6_balanced_32x128(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens,
        route_pairs, caller_stream);
  }

  return launch_balanced_wide_kernel<16, 128>(x_permuted, expert_weights, offsets, y_permuted,
                                              experts, hidden, output, max_expert_tokens,
                                              caller_stream);
}

// Single-variable v9 experiment: keep tile-M=32 and the complete v6 pipeline/selector, but halve
// tile-N from 128 to 64.  This doubles the CTA count for N=128 while halving the per-CTA weight
// stage and accumulator footprint.  Unlike v8, the extra CTAs split the N dimension: they reload
// the smaller A tile instead of reloading a full Kx128 weight tile for every additional row tile.
cudaError_t launch_grouped_gemm_sm86_fp32_v9_balanced_32x64(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens, int route_pairs,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_wide_arguments(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (route_pairs <= 0 || max_expert_tokens > route_pairs) return cudaErrorInvalidValue;

  const std::int64_t average_ceiling =
      (static_cast<std::int64_t>(route_pairs) + experts - 1) / experts;
  const bool balanced = average_ceiling >= 32 &&
                        static_cast<std::int64_t>(max_expert_tokens) <= 2LL * average_ceiling;
  if (!balanced || hidden == 0 || hidden % kDepth != 0 || output % 64 != 0 ||
      !aligned_16_wide(x_permuted) || !aligned_16_wide(expert_weights) ||
      !aligned_16_wide(y_permuted)) {
    return launch_grouped_gemm_sm86_fp32_v6_balanced_32x128(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens,
        route_pairs, caller_stream);
  }

  return launch_balanced_wide_kernel<32, 64>(x_permuted, expert_weights, offsets, y_permuted,
                                             experts, hidden, output, max_expert_tokens,
                                             caller_stream);
}

// Single-variable v10 experiment: keep the v6 and v9 kernels unchanged and change only their
// selector.  The dirty v9 region screen showed that the 32x64 tile is useful in bounded SM86 CTA
// wave windows, while very small grids, K=256, and larger N=128 grids should retain v6.  The
// selector is deliberately expressed in CTA count so its hardware hypothesis is reviewable; it
// remains a benchmark-only RTX 3080 research portfolio rather than public kAuto policy.
cudaError_t launch_grouped_gemm_sm86_fp32_v10_wave_aware_portfolio(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens, int route_pairs,
    cudaStream_t caller_stream) {
  const cudaError_t validation = validate_wide_arguments(
      x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens);
  if (validation != cudaSuccess || max_expert_tokens == 0 || output == 0) return validation;
  if (route_pairs <= 0 || max_expert_tokens > route_pairs) return cudaErrorInvalidValue;

  const std::int64_t average_ceiling =
      (static_cast<std::int64_t>(route_pairs) + experts - 1) / experts;
  const bool balanced = average_ceiling >= 32 &&
                        static_cast<std::int64_t>(max_expert_tokens) <= 2LL * average_ceiling;
  const std::int64_t row_tiles = (static_cast<std::int64_t>(max_expert_tokens) + 31) / 32;
  const std::int64_t column_tiles = output > 0 ? output / 64 : 0;
  const std::int64_t ctas = row_tiles * column_tiles * experts;
  const bool n64_window = output == 64 && hidden > 0 && hidden <= 128 && hidden % kDepth == 0 &&
                          ctas >= 160 && ctas <= 320;
  const bool n128_window = output == 128 && hidden == 128 && ctas >= 160 && ctas <= 256;
  const bool use_v9 = balanced && (n64_window || n128_window) &&
                      aligned_16_wide(x_permuted) && aligned_16_wide(expert_weights) &&
                      aligned_16_wide(y_permuted);
  if (!use_v9) {
    return launch_grouped_gemm_sm86_fp32_v6_balanced_32x128(
        x_permuted, expert_weights, offsets, y_permuted, experts, hidden, output, max_expert_tokens,
        route_pairs, caller_stream);
  }

  return launch_balanced_wide_kernel<32, 64>(x_permuted, expert_weights, offsets, y_permuted,
                                             experts, hidden, output, max_expert_tokens,
                                             caller_stream);
}

}  // namespace raggedroute::ops
