#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

constexpr int kThreads = 32;
constexpr int kItemsPerThread = 2;
using BlockSort = cub::BlockRadixSort<std::uint64_t, kThreads, kItemsPerThread>;

__device__ __forceinline__ std::uint32_t float_to_ordered_bits(float value) {
  if (isnan(value)) value = -CUDART_INF_F;
  if (value == 0.0F) value = 0.0F;
  const std::uint32_t bits = __float_as_uint(value);
  return bits ^ ((bits & 0x80000000U) != 0 ? 0xFFFFFFFFU : 0x80000000U);
}

__device__ __forceinline__ float ordered_bits_to_float(std::uint32_t ordered) {
  const std::uint32_t bits = (ordered & 0x80000000U) != 0 ? (ordered ^ 0x80000000U) : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ std::uint64_t make_key(float value, int expert) {
  return (static_cast<std::uint64_t>(float_to_ordered_bits(value)) << 32U) |
         static_cast<std::uint32_t>(0xFFFFFFFFU - static_cast<std::uint32_t>(expert));
}

__device__ __forceinline__ int key_expert(std::uint64_t key) {
  return static_cast<int>(0xFFFFFFFFU - static_cast<std::uint32_t>(key));
}

__device__ __forceinline__ float key_value(std::uint64_t key) {
  return ordered_bits_to_float(static_cast<std::uint32_t>(key >> 32U));
}

__device__ __forceinline__ void write_weights(float first, float second, bool saw_non_nan,
                                              int& first_expert, int& second_expert,
                                              float& first_weight, float& second_weight) {
  first_weight = 0.5F;
  second_weight = 0.5F;
  if (!saw_non_nan) {
    first_expert = 0;
    second_expert = 1;
  } else if (first != second) {
    if (first == CUDART_INF_F || second == -CUDART_INF_F) {
      first_weight = 1.0F;
      second_weight = 0.0F;
    } else {
      const float relative = expf(second - first);
      first_weight = 1.0F / (1.0F + relative);
      second_weight = relative * first_weight;
    }
  }
}

__global__ void cub_block_radix_top2_kernel(const float* logits, std::int32_t* expert_ids,
                                            float* weights, int tokens, int experts) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;
  __shared__ typename BlockSort::TempStorage sort_storage;

  std::uint64_t keys[kItemsPerThread];
  bool local_non_nan = false;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int expert = threadIdx.x + item * kThreads;
    float value = -CUDART_INF_F;
    if (expert < experts) {
      const float raw = logits[static_cast<std::size_t>(token) * experts + expert];
      local_non_nan = local_non_nan || !isnan(raw);
      value = isnan(raw) ? -CUDART_INF_F : raw;
    }
    keys[item] = make_key(value, expert);
  }

  const bool saw_non_nan = __any_sync(0xFFFFFFFFU, static_cast<int>(local_non_nan)) != 0;
  BlockSort(sort_storage).SortDescending(keys);
  if (threadIdx.x == 0) {
    int first_expert = key_expert(keys[0]);
    int second_expert = key_expert(keys[1]);
    const float first = key_value(keys[0]);
    const float second = key_value(keys[1]);
    float first_weight = 0.5F;
    float second_weight = 0.5F;
    write_weights(first, second, saw_non_nan, first_expert, second_expert, first_weight,
                  second_weight);
    const std::size_t output = static_cast<std::size_t>(token) * 2;
    expert_ids[output] = first_expert;
    expert_ids[output + 1] = second_expert;
    weights[output] = first_weight;
    weights[output + 1] = second_weight;
  }
}

}  // namespace

cudaError_t launch_cub_block_radix_top2(const float* logits, std::int32_t* expert_ids,
                                        float* weights, int tokens, int experts,
                                        cudaStream_t stream) {
  if (tokens < 0 || experts < 2 || experts > 64) return cudaErrorInvalidValue;
  if (tokens == 0) return cudaSuccess;
  if (logits == nullptr || expert_ids == nullptr || weights == nullptr) {
    return cudaErrorInvalidValue;
  }
  cub_block_radix_top2_kernel<<<tokens, kThreads, 0, stream>>>(logits, expert_ids, weights, tokens,
                                                               experts);
  return cudaGetLastError();
}

}  // namespace raggedroute::benchmark::library_baseline
