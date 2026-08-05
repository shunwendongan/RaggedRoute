#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

#include "optimized_internal.h"

namespace raggedroute::ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kScalarWarpsPerBlock = 4;

struct Candidate {
  float value;
  int expert;
};

struct PackedCandidate {
  std::uint32_t ordered_value;
  std::uint32_t inverse_expert;
};

__device__ __forceinline__ std::uint32_t float_to_ordered_bits(float value) {
  if (value == 0.0F) value = 0.0F;
  const std::uint32_t bits = __float_as_uint(value);
  return bits ^ ((bits & 0x80000000U) != 0 ? 0xFFFFFFFFU : 0x80000000U);
}

__device__ __forceinline__ float ordered_bits_to_float(std::uint32_t ordered) {
  const std::uint32_t bits =
      (ordered & 0x80000000U) != 0 ? ordered ^ 0x80000000U : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ PackedCandidate pack_candidate(float value, int expert) {
  return {float_to_ordered_bits(value), 0xFFFFFFFFU - static_cast<std::uint32_t>(expert)};
}

__device__ __forceinline__ PackedCandidate invalid_packed_candidate() {
  return {float_to_ordered_bits(-CUDART_INF_F), 0U};
}

__device__ __forceinline__ bool packed_precedes(const PackedCandidate& left,
                                                const PackedCandidate& right) {
  return left.ordered_value > right.ordered_value ||
         (left.ordered_value == right.ordered_value &&
          left.inverse_expert > right.inverse_expert);
}

__device__ __forceinline__ bool packed_equal(const PackedCandidate& left,
                                             const PackedCandidate& right) {
  return left.ordered_value == right.ordered_value &&
         left.inverse_expert == right.inverse_expert;
}

__device__ __forceinline__ void insert_packed_candidate(PackedCandidate candidate,
                                                        PackedCandidate& first,
                                                        PackedCandidate& second) {
  if (packed_precedes(candidate, first)) {
    second = first;
    first = candidate;
  } else if (packed_precedes(candidate, second)) {
    second = candidate;
  }
}

template <int Width>
__device__ __forceinline__ PackedCandidate subgroup_reduce_max(PackedCandidate candidate) {
  static_assert(Width == 1 || Width == 2 || Width == 4 || Width == 8 || Width == 16,
                "unsupported packed subgroup width");
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    PackedCandidate other{
        __shfl_xor_sync(0xFFFFFFFFU, candidate.ordered_value, offset, Width),
        __shfl_xor_sync(0xFFFFFFFFU, candidate.inverse_expert, offset, Width)};
    if (packed_precedes(other, candidate)) candidate = other;
  }
  return candidate;
}

template <int Width>
__device__ __forceinline__ void subgroup_select_top2(PackedCandidate& first,
                                                     PackedCandidate& second) {
  const PackedCandidate global_first = subgroup_reduce_max<Width>(first);
  const PackedCandidate runner_up = packed_equal(first, global_first) ? second : first;
  const PackedCandidate global_second = subgroup_reduce_max<Width>(runner_up);
  first = global_first;
  second = global_second;
}

__device__ __forceinline__ Candidate unpack_candidate(const PackedCandidate& packed) {
  return {ordered_bits_to_float(packed.ordered_value),
          static_cast<int>(0xFFFFFFFFU - packed.inverse_expert)};
}

__device__ __forceinline__ Candidate invalid_candidate() { return {-CUDART_INF_F, -1}; }

__device__ __forceinline__ bool precedes(const Candidate& left, const Candidate& right) {
  return right.expert < 0 ||
         (left.expert >= 0 &&
          (left.value > right.value || (left.value == right.value && left.expert < right.expert)));
}

__device__ __forceinline__ void insert_candidate(Candidate candidate, Candidate& first,
                                                 Candidate& second) {
  if (precedes(candidate, first)) {
    second = first;
    first = candidate;
  } else if (precedes(candidate, second)) {
    second = candidate;
  }
}

__device__ __forceinline__ void merge_top2(Candidate& first, Candidate& second,
                                           const Candidate& other_first,
                                           const Candidate& other_second) {
  if (precedes(other_first, first)) {
    const Candidate old_first = first;
    first = other_first;
    second = precedes(other_second, old_first) ? other_second : old_first;
  } else {
    second = precedes(other_first, second) ? other_first : second;
  }
}

template <int Width>
__device__ __forceinline__ void subgroup_reduce_top2(Candidate& first, Candidate& second) {
  static_assert(Width == 2 || Width == 4 || Width == 8 || Width == 16 || Width == 32,
                "unsupported subgroup width");
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    Candidate other_first{__shfl_xor_sync(0xFFFFFFFFU, first.value, offset, Width),
                          __shfl_xor_sync(0xFFFFFFFFU, first.expert, offset, Width)};
    Candidate other_second{__shfl_xor_sync(0xFFFFFFFFU, second.value, offset, Width),
                           __shfl_xor_sync(0xFFFFFFFFU, second.expert, offset, Width)};
    merge_top2(first, second, other_first, other_second);
  }
}

__device__ __forceinline__ unsigned subgroup_mask(int lane, int width) {
  if (width == kWarpSize) return 0xFFFFFFFFU;
  const int group_base = (lane / width) * width;
  return ((1U << width) - 1U) << group_base;
}

__device__ __forceinline__ void write_result(std::int32_t* expert_ids, float* weights, int token,
                                             Candidate first, Candidate second, bool saw_non_nan) {
  float first_weight = 0.5F;
  float second_weight = 0.5F;
  if (!saw_non_nan) {
    first.expert = 0;
    second.expert = 1;
  } else if (first.value != second.value) {
    if (first.value == CUDART_INF_F || second.value == -CUDART_INF_F) {
      first_weight = 1.0F;
      second_weight = 0.0F;
    } else {
      const float relative_second = expf(second.value - first.value);
      first_weight = 1.0F / (1.0F + relative_second);
      second_weight = relative_second * first_weight;
    }
  }
  const std::size_t output = static_cast<std::size_t>(token) * 2;
  expert_ids[output] = static_cast<std::int32_t>(first.expert);
  expert_ids[output + 1] = static_cast<std::int32_t>(second.expert);
  weights[output] = first_weight;
  weights[output + 1] = second_weight;
}

template <int Width>
__global__ __launch_bounds__(kScalarWarpsPerBlock* kWarpSize) void topk_gate_scalar_pair_kernel(
    const float* logits, std::int32_t* expert_ids, float* weights, int tokens, int experts) {
  constexpr int kRowsPerWarp = kWarpSize / Width;
  const int lane = threadIdx.x % kWarpSize;
  const int lane_in_group = lane % Width;
  const int group_in_warp = lane / Width;
  const int global_warp = static_cast<int>(blockIdx.x) * kScalarWarpsPerBlock +
                          static_cast<int>(threadIdx.x) / kWarpSize;
  const int token = global_warp * kRowsPerWarp + group_in_warp;
  const bool valid_token = token < tokens;

  Candidate first = invalid_candidate();
  Candidate second = invalid_candidate();
  bool local_non_nan = false;
  if (valid_token) {
    const std::size_t row = static_cast<std::size_t>(token) * experts;
    for (int expert = lane_in_group; expert < experts; expert += Width) {
      const float raw = logits[row + static_cast<std::size_t>(expert)];
      const bool is_nan_value = isnan(raw);
      local_non_nan = local_non_nan || !is_nan_value;
      insert_candidate({is_nan_value ? -CUDART_INF_F : raw, expert}, first, second);
    }
  }

  subgroup_reduce_top2<Width>(first, second);
  const bool saw_non_nan =
      __any_sync(subgroup_mask(lane, Width), static_cast<int>(local_non_nan)) != 0;
  if (valid_token && lane_in_group == 0) {
    write_result(expert_ids, weights, token, first, second, saw_non_nan);
  }
}

__global__ void topk_gate_two_expert_kernel(const float* logits, std::int32_t* expert_ids,
                                            float* weights, int tokens) {
  const int token = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const std::size_t row = static_cast<std::size_t>(token) * 2;
  const float raw_first = logits[row];
  const float raw_second = logits[row + 1];
  const bool first_nan = isnan(raw_first);
  const bool second_nan = isnan(raw_second);
  Candidate first{first_nan ? -CUDART_INF_F : raw_first, 0};
  Candidate second{second_nan ? -CUDART_INF_F : raw_second, 1};
  if (precedes(second, first)) {
    const Candidate swap = first;
    first = second;
    second = swap;
  }
  write_result(expert_ids, weights, token, first, second, !first_nan || !second_nan);
}

template <int Experts, int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock* kWarpSize) void topk_gate_vector_pair_kernel(
    const float* logits, std::int32_t* expert_ids, float* weights, int tokens) {
  static_assert(Experts == 8 || Experts == 16 || Experts == 32 || Experts == 64,
                "unsupported vector specialization");
  constexpr int kItemsPerThread = 4;
  constexpr int kWidth = Experts / kItemsPerThread;
  constexpr int kRowsPerWarp = kWarpSize / kWidth;
  const int lane = threadIdx.x % kWarpSize;
  const int lane_in_group = lane % kWidth;
  const int group_in_warp = lane / kWidth;
  const int global_warp =
      static_cast<int>(blockIdx.x) * WarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize;
  const int token = global_warp * kRowsPerWarp + group_in_warp;
  const bool valid_token = token < tokens;

  Candidate first = invalid_candidate();
  Candidate second = invalid_candidate();
  bool local_non_nan = false;
  if (valid_token) {
    const float4 packed = reinterpret_cast<const float4*>(logits + static_cast<std::size_t>(token) *
                                                                       Experts)[lane_in_group];
    const float values[kItemsPerThread] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const float raw = values[item];
      const bool is_nan_value = isnan(raw);
      local_non_nan = local_non_nan || !is_nan_value;
      const int expert = lane_in_group * kItemsPerThread + item;
      insert_candidate({is_nan_value ? -CUDART_INF_F : raw, expert}, first, second);
    }
  }

  subgroup_reduce_top2<kWidth>(first, second);
  const bool saw_non_nan =
      __any_sync(subgroup_mask(lane, kWidth), static_cast<int>(local_non_nan)) != 0;
  if (valid_token && lane_in_group == 0) {
    write_result(expert_ids, weights, token, first, second, saw_non_nan);
  }
}

template <int Experts, int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock* kWarpSize) void
topk_gate_local_pair_two_reduce_kernel(const float* logits, std::int32_t* expert_ids,
                                       float* weights, int tokens) {
  static_assert(Experts == 4 || Experts == 8 || Experts == 16 || Experts == 32 || Experts == 64,
                "unsupported packed two-reduction specialization");
  constexpr int kItemsPerThread = 4;
  constexpr int kWidth = Experts / kItemsPerThread;
  constexpr int kRowsPerWarp = kWarpSize / kWidth;
  const int lane = threadIdx.x % kWarpSize;
  const int lane_in_group = lane % kWidth;
  const int group_in_warp = lane / kWidth;
  const int global_warp =
      static_cast<int>(blockIdx.x) * WarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize;
  const int token = global_warp * kRowsPerWarp + group_in_warp;
  const bool valid_token = token < tokens;

  PackedCandidate first = invalid_packed_candidate();
  PackedCandidate second = invalid_packed_candidate();
  bool local_non_nan = false;
  if (valid_token) {
    const float4 packed = reinterpret_cast<const float4*>(
        logits + static_cast<std::size_t>(token) * Experts)[lane_in_group];
    const float values[kItemsPerThread] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const float raw = values[item];
      const bool is_nan_value = isnan(raw);
      local_non_nan = local_non_nan || !is_nan_value;
      const int expert = lane_in_group * kItemsPerThread + item;
      insert_packed_candidate(
          pack_candidate(is_nan_value ? -CUDART_INF_F : raw, expert), first, second);
    }
  }

  subgroup_select_top2<kWidth>(first, second);
  const bool saw_non_nan =
      __any_sync(subgroup_mask(lane, kWidth), static_cast<int>(local_non_nan)) != 0;
  if (valid_token && lane_in_group == 0) {
    write_result(expert_ids, weights, token, unpack_candidate(first), unpack_candidate(second),
                 saw_non_nan);
  }
}

__global__ void topk_gate_two_expert_vector_kernel(const float* logits,
                                                   std::int32_t* expert_ids, float* weights,
                                                   int tokens) {
  const int token = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const float2 packed = reinterpret_cast<const float2*>(logits)[token];
  const bool first_nan = isnan(packed.x);
  const bool second_nan = isnan(packed.y);
  Candidate first{first_nan ? -CUDART_INF_F : packed.x, 0};
  Candidate second{second_nan ? -CUDART_INF_F : packed.y, 1};
  if (precedes(second, first)) {
    const Candidate swap = first;
    first = second;
    second = swap;
  }
  write_result(expert_ids, weights, token, first, second, !first_nan || !second_nan);
}

template <int Width>
cudaError_t launch_scalar_width(const float* logits, std::int32_t* expert_ids, float* weights,
                                int tokens, int experts, cudaStream_t stream) {
  constexpr int kRowsPerBlock = kScalarWarpsPerBlock * (kWarpSize / Width);
  const int blocks = (tokens + kRowsPerBlock - 1) / kRowsPerBlock;
  topk_gate_scalar_pair_kernel<Width><<<blocks, kScalarWarpsPerBlock * kWarpSize, 0, stream>>>(
      logits, expert_ids, weights, tokens, experts);
  return cudaGetLastError();
}

cudaError_t launch_v1(const float* logits, std::int32_t* expert_ids, float* weights, int tokens,
                      int experts, cudaStream_t stream) {
  return launch_scalar_width<32>(logits, expert_ids, weights, tokens, experts, stream);
}

cudaError_t launch_v2(const float* logits, std::int32_t* expert_ids, float* weights, int tokens,
                      int experts, cudaStream_t stream) {
  if (experts == 2) {
    constexpr int kThreads = 256;
    topk_gate_two_expert_kernel<<<(tokens + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        logits, expert_ids, weights, tokens);
    return cudaGetLastError();
  }
  if (experts <= 4)
    return launch_scalar_width<4>(logits, expert_ids, weights, tokens, experts, stream);
  if (experts <= 8)
    return launch_scalar_width<8>(logits, expert_ids, weights, tokens, experts, stream);
  if (experts <= 16)
    return launch_scalar_width<16>(logits, expert_ids, weights, tokens, experts, stream);
  return launch_scalar_width<32>(logits, expert_ids, weights, tokens, experts, stream);
}

template <int Experts, int WarpsPerBlock>
cudaError_t launch_vector_specialization(const float* logits, std::int32_t* expert_ids,
                                         float* weights, int tokens, cudaStream_t stream) {
  constexpr int kRowsPerBlock = WarpsPerBlock * (kWarpSize / (Experts / 4));
  const int blocks = (tokens + kRowsPerBlock - 1) / kRowsPerBlock;
  topk_gate_vector_pair_kernel<Experts, WarpsPerBlock>
      <<<blocks, WarpsPerBlock * kWarpSize, 0, stream>>>(logits, expert_ids, weights, tokens);
  return cudaGetLastError();
}

cudaError_t launch_v3(const float* logits, std::int32_t* expert_ids, float* weights, int tokens,
                      int experts, cudaStream_t stream) {
  if (reinterpret_cast<std::uintptr_t>(logits) % alignof(float4) != 0) {
    return launch_v2(logits, expert_ids, weights, tokens, experts, stream);
  }
  switch (experts) {
    case 8:
      return launch_vector_specialization<8, 1>(logits, expert_ids, weights, tokens, stream);
    case 16:
      return launch_vector_specialization<16, 2>(logits, expert_ids, weights, tokens, stream);
    case 32:
      return launch_vector_specialization<32, 4>(logits, expert_ids, weights, tokens, stream);
    case 64:
      return launch_vector_specialization<64, 8>(logits, expert_ids, weights, tokens, stream);
    default:
      return launch_v2(logits, expert_ids, weights, tokens, experts, stream);
  }
}

template <int Experts, int WarpsPerBlock>
cudaError_t launch_local_pair_two_reduce_specialization(const float* logits,
                                                        std::int32_t* expert_ids, float* weights,
                                                        int tokens, cudaStream_t stream) {
  static_assert(WarpsPerBlock == 2 || WarpsPerBlock == 4 || WarpsPerBlock == 8,
                "unsupported CTA warp count");
  constexpr int kRowsPerBlock = WarpsPerBlock * (kWarpSize / (Experts / 4));
  const int blocks = (tokens + kRowsPerBlock - 1) / kRowsPerBlock;
  topk_gate_local_pair_two_reduce_kernel<Experts, WarpsPerBlock>
      <<<blocks, WarpsPerBlock * kWarpSize, 0, stream>>>(logits, expert_ids, weights, tokens);
  return cudaGetLastError();
}

template <int WarpsPerBlock>
cudaError_t launch_v4_geometry(const float* logits, std::int32_t* expert_ids, float* weights,
                               int tokens, int experts, cudaStream_t stream) {
  if (experts == 2) {
    if (reinterpret_cast<std::uintptr_t>(logits) % alignof(float2) != 0) {
      return launch_v2(logits, expert_ids, weights, tokens, experts, stream);
    }
    constexpr int kThreads = 256;
    topk_gate_two_expert_vector_kernel<<<(tokens + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        logits, expert_ids, weights, tokens);
    return cudaGetLastError();
  }
  if (reinterpret_cast<std::uintptr_t>(logits) % alignof(float4) != 0) {
    return launch_v2(logits, expert_ids, weights, tokens, experts, stream);
  }
  switch (experts) {
    case 4:
      return launch_local_pair_two_reduce_specialization<4, WarpsPerBlock>(
          logits, expert_ids, weights, tokens, stream);
    case 8:
      return launch_local_pair_two_reduce_specialization<8, WarpsPerBlock>(
          logits, expert_ids, weights, tokens, stream);
    case 16:
      return launch_local_pair_two_reduce_specialization<16, WarpsPerBlock>(
          logits, expert_ids, weights, tokens, stream);
    case 32:
      return launch_local_pair_two_reduce_specialization<32, WarpsPerBlock>(
          logits, expert_ids, weights, tokens, stream);
    case 64:
      return launch_local_pair_two_reduce_specialization<64, WarpsPerBlock>(
          logits, expert_ids, weights, tokens, stream);
    default:
      return launch_v2(logits, expert_ids, weights, tokens, experts, stream);
  }
}

cudaError_t launch_v4(const float* logits, std::int32_t* expert_ids, float* weights, int tokens,
                      int experts, cudaStream_t stream) {
  return launch_v4_geometry<4>(logits, expert_ids, weights, tokens, experts, stream);
}

}  // namespace

bool is_topk_gate_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kTopKGateWarpPairV1Implementation ||
         implementation_id == kTopKGateSubwarpPairV2Implementation ||
         implementation_id == kTopKGateVectorPairV3Implementation ||
         implementation_id == kTopKGateLocalPairTwoReduceV4Implementation;
}

cudaError_t launch_topk_gate_optimized(const float* logits, std::int32_t* expert_ids,
                                       float* weights, int tokens, int experts,
                                       std::uint32_t implementation_id,
                                       cudaStream_t caller_stream) {
  if (tokens < 0 || experts < 2 || experts > 64 ||
      !is_topk_gate_optimized_implementation(implementation_id)) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0) return cudaSuccess;
  if (logits == nullptr || expert_ids == nullptr || weights == nullptr) {
    return cudaErrorInvalidValue;
  }
  switch (implementation_id) {
    case kTopKGateWarpPairV1Implementation:
      return launch_v1(logits, expert_ids, weights, tokens, experts, caller_stream);
    case kTopKGateSubwarpPairV2Implementation:
      return launch_v2(logits, expert_ids, weights, tokens, experts, caller_stream);
    case kTopKGateVectorPairV3Implementation:
      return launch_v3(logits, expert_ids, weights, tokens, experts, caller_stream);
    case kTopKGateLocalPairTwoReduceV4Implementation:
      return launch_v4(logits, expert_ids, weights, tokens, experts, caller_stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace raggedroute::ops
