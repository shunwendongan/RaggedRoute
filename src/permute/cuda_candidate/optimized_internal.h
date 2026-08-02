#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kTokenPermuteAtomicVectorized128Implementation = 1;
constexpr std::uint32_t kTokenPermuteAtomicVectorized64Implementation = 2;
constexpr std::uint32_t kTokenPermuteAtomicVectorized256Implementation = 3;
constexpr std::uint32_t kTokenPermuteTokenOwnedTop2Implementation = 4;
constexpr std::uint32_t kTokenPermuteBlockPartialImplementation = 5;

// This alias is intentionally evidence-controlled. It remains the simplest
// vectorized candidate until release A/B data justifies changing it.
constexpr std::uint32_t kTokenPermuteCandidateImplementation =
    kTokenPermuteAtomicVectorized128Implementation;

inline bool is_token_permute_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kTokenPermuteAtomicVectorized128Implementation ||
         implementation_id == kTokenPermuteAtomicVectorized64Implementation ||
         implementation_id == kTokenPermuteAtomicVectorized256Implementation ||
         implementation_id == kTokenPermuteTokenOwnedTop2Implementation ||
         implementation_id == kTokenPermuteBlockPartialImplementation;
}

cudaError_t launch_token_permute_optimized(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int experts, int top_k, int hidden,
    std::uint32_t implementation_id, cudaStream_t caller_stream);

}  // namespace raggedroute::ops
