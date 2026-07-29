#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

#include "raggedroute/baseline_ops.h"

namespace raggedroute::ops {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBaselineBlocks = 65535;

__device__ bool candidate_precedes(float value, int expert, float incumbent_value,
                                   int incumbent_expert) {
  return incumbent_expert < 0 || value > incumbent_value ||
         (value == incumbent_value && expert < incumbent_expert);
}

__global__ void topk_gate_naive_kernel(const float* logits, std::int32_t* expert_ids,
                                       float* weights, int tokens, int experts) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t token = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < static_cast<std::size_t>(tokens); token += stride) {
    float first_value = -CUDART_INF_F;
    float second_value = -CUDART_INF_F;
    int first_expert = -1;
    int second_expert = -1;
    bool saw_non_nan = false;

    const std::size_t row_start = token * static_cast<std::size_t>(experts);
    for (int expert = 0; expert < experts; ++expert) {
      const float input = logits[row_start + static_cast<std::size_t>(expert)];
      const bool input_is_nan = isnan(input);
      const float value = input_is_nan ? -CUDART_INF_F : input;
      saw_non_nan = saw_non_nan || !input_is_nan;

      if (candidate_precedes(value, expert, first_value, first_expert)) {
        second_value = first_value;
        second_expert = first_expert;
        first_value = value;
        first_expert = expert;
      } else if (candidate_precedes(value, expert, second_value, second_expert)) {
        second_value = value;
        second_expert = expert;
      }
    }

    float first_weight;
    float second_weight;
    if (!saw_non_nan || first_value == second_value) {
      // The explicit all-NaN fallback is ids 0/1 with equal weights. Equal
      // finite or infinite selected values have the same normalized result.
      if (!saw_non_nan) {
        first_expert = 0;
        second_expert = 1;
      }
      first_weight = 0.5F;
      second_weight = 0.5F;
    } else if (first_value == CUDART_INF_F || second_value == -CUDART_INF_F) {
      // Avoid inf-inf in the selected softmax.
      first_weight = 1.0F;
      second_weight = 0.0F;
    } else {
      const float relative_second = expf(second_value - first_value);
      first_weight = 1.0F / (1.0F + relative_second);
      second_weight = relative_second * first_weight;
    }

    const std::size_t output_start = token * 2;
    expert_ids[output_start] = static_cast<std::int32_t>(first_expert);
    expert_ids[output_start + 1] = static_cast<std::int32_t>(second_expert);
    weights[output_start] = first_weight;
    weights[output_start + 1] = second_weight;
  }
}

unsigned int block_count_for(std::size_t elements) {
  const std::size_t blocks =
      elements / kThreadsPerBlock + (elements % kThreadsPerBlock != 0 ? 1 : 0);
  return static_cast<unsigned int>(blocks < kMaxBaselineBlocks ? blocks : kMaxBaselineBlocks);
}

}  // namespace

cudaError_t launch_topk_gate_naive(const float* logits, std::int32_t* expert_ids, float* weights,
                                   int tokens, int experts, cudaStream_t caller_stream) {
  if (tokens < 0 || experts < 2) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0) {
    return cudaSuccess;
  }
  if (logits == nullptr || expert_ids == nullptr || weights == nullptr) {
    return cudaErrorInvalidValue;
  }

  topk_gate_naive_kernel<<<block_count_for(static_cast<std::size_t>(tokens)), kThreadsPerBlock, 0,
                           caller_stream>>>(logits, expert_ids, weights, tokens, experts);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
