/*
 * Adapted from vLLM csrc/libtorch_stable/moe/topk_softmax_kernels.cu at
 * 55c98e370aa058f567a9e682dc0652bdfba6b0bb.
 * Copyright 2025 The vLLM team.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RaggedRoute modification: retain the row-packed vector-load and iterative
 * argmax organization, but select raw FP32 logits and apply the repository's
 * deterministic Top-2 selected-softmax/NaN contract.
 */

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ bool candidate_precedes(float value, int expert, float incumbent,
                                                   int incumbent_expert) {
  return incumbent_expert < 0 || value > incumbent ||
         (value == incumbent && expert < incumbent_expert);
}

__device__ __forceinline__ void finish_selected_softmax(float first, float second, bool saw_non_nan,
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

template <int Experts>
struct RowPacking {
  static constexpr int kValuesPerThread = Experts == 2 ? 2 : 4;
  static constexpr int kThreadsPerRow = Experts / kValuesPerThread;
  static constexpr int kRowsPerWarp = kWarpSize / kThreadsPerRow;
  static constexpr int kRowsPerBlock = kRowsPerWarp * kWarpsPerBlock;
};

template <int Experts>
__global__ __launch_bounds__(kWarpsPerBlock* kWarpSize) void vllm_row_packed_top2_kernel(
    const float* logits, std::int32_t* expert_ids, float* weights, int tokens) {
  using Packing = RowPacking<Experts>;
  constexpr int kValuesPerThread = Packing::kValuesPerThread;
  constexpr int kThreadsPerRow = Packing::kThreadsPerRow;
  constexpr int kRowsPerWarp = Packing::kRowsPerWarp;
  const int lane = threadIdx.x % kWarpSize;
  const int lane_in_row = lane % kThreadsPerRow;
  const int row_in_warp = lane / kThreadsPerRow;
  const int global_warp =
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize;
  const int token = global_warp * kRowsPerWarp + row_in_warp;
  const bool valid_token = token < tokens;

  float values[kValuesPerThread];
  int indices[kValuesPerThread];
  bool local_non_nan = false;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    values[item] = -CUDART_INF_F;
    indices[item] = lane_in_row * kValuesPerThread + item;
  }
  if (valid_token) {
    const float* row = logits + static_cast<std::size_t>(token) * Experts;
    if constexpr (Experts == 2) {
      const float2 packed = *reinterpret_cast<const float2*>(row);
      values[0] = packed.x;
      values[1] = packed.y;
    } else {
      const float4 packed = reinterpret_cast<const float4*>(row)[lane_in_row];
      values[0] = packed.x;
      values[1] = packed.y;
      values[2] = packed.z;
      values[3] = packed.w;
    }
#pragma unroll
    for (int item = 0; item < kValuesPerThread; ++item) {
      const bool is_nan_value = isnan(values[item]);
      local_non_nan = local_non_nan || !is_nan_value;
      values[item] = is_nan_value ? -CUDART_INF_F : values[item];
    }
  }

  float selected_values[2] = {-CUDART_INF_F, -CUDART_INF_F};
  int selected_experts[2] = {-1, -1};
#pragma unroll
  for (int rank = 0; rank < 2; ++rank) {
    float local_value = values[0];
    int local_expert = indices[0];
#pragma unroll
    for (int item = 1; item < kValuesPerThread; ++item) {
      if (candidate_precedes(values[item], indices[item], local_value, local_expert)) {
        local_value = values[item];
        local_expert = indices[item];
      }
    }
#pragma unroll
    for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
      const float other_value = __shfl_xor_sync(0xFFFFFFFFU, local_value, mask, kThreadsPerRow);
      const int other_expert = __shfl_xor_sync(0xFFFFFFFFU, local_expert, mask, kThreadsPerRow);
      if (candidate_precedes(other_value, other_expert, local_value, local_expert)) {
        local_value = other_value;
        local_expert = other_expert;
      }
    }
    selected_values[rank] = local_value;
    selected_experts[rank] = local_expert;
#pragma unroll
    for (int item = 0; item < kValuesPerThread; ++item) {
      if (indices[item] == local_expert) values[item] = -CUDART_INF_F;
    }
  }

  const int group_base = (lane / kThreadsPerRow) * kThreadsPerRow;
  const unsigned mask =
      kThreadsPerRow == 32 ? 0xFFFFFFFFU : ((1U << kThreadsPerRow) - 1U) << group_base;
  const bool saw_non_nan = __any_sync(mask, static_cast<int>(local_non_nan)) != 0;
  if (valid_token && lane_in_row == 0) {
    float first_weight = 0.5F;
    float second_weight = 0.5F;
    finish_selected_softmax(selected_values[0], selected_values[1], saw_non_nan,
                            selected_experts[0], selected_experts[1], first_weight, second_weight);
    const std::size_t output = static_cast<std::size_t>(token) * 2;
    expert_ids[output] = selected_experts[0];
    expert_ids[output + 1] = selected_experts[1];
    weights[output] = first_weight;
    weights[output + 1] = second_weight;
  }
}

template <int Experts>
cudaError_t launch_specialized(const float* logits, std::int32_t* expert_ids, float* weights,
                               int tokens, cudaStream_t stream) {
  constexpr int kRowsPerBlock = RowPacking<Experts>::kRowsPerBlock;
  const int blocks = (tokens + kRowsPerBlock - 1) / kRowsPerBlock;
  vllm_row_packed_top2_kernel<Experts>
      <<<blocks, kWarpsPerBlock * kWarpSize, 0, stream>>>(logits, expert_ids, weights, tokens);
  return cudaGetLastError();
}

}  // namespace

bool supports_vllm_row_packed_top2(int experts, const float* logits) noexcept {
  const bool supported_experts = experts == 2 || experts == 4 || experts == 8 || experts == 16 ||
                                 experts == 32 || experts == 64;
  const std::size_t alignment = experts == 2 ? alignof(float2) : alignof(float4);
  return supported_experts && logits != nullptr &&
         reinterpret_cast<std::uintptr_t>(logits) % alignment == 0;
}

cudaError_t launch_vllm_row_packed_top2(const float* logits, std::int32_t* expert_ids,
                                        float* weights, int tokens, int experts,
                                        cudaStream_t stream) {
  if (tokens < 0 || experts < 2 || experts > 64) return cudaErrorInvalidValue;
  if (tokens == 0) return cudaSuccess;
  if (expert_ids == nullptr || weights == nullptr ||
      !supports_vllm_row_packed_top2(experts, logits)) {
    return cudaErrorInvalidValue;
  }
  switch (experts) {
    case 2:
      return launch_specialized<2>(logits, expert_ids, weights, tokens, stream);
    case 4:
      return launch_specialized<4>(logits, expert_ids, weights, tokens, stream);
    case 8:
      return launch_specialized<8>(logits, expert_ids, weights, tokens, stream);
    case 16:
      return launch_specialized<16>(logits, expert_ids, weights, tokens, stream);
    case 32:
      return launch_specialized<32>(logits, expert_ids, weights, tokens, stream);
    case 64:
      return launch_specialized<64>(logits, expert_ids, weights, tokens, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace raggedroute::benchmark::library_baseline
