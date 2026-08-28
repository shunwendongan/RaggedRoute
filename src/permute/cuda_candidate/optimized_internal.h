#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kTokenPermuteAtomicVectorized128Implementation = 1;
constexpr std::uint32_t kTokenPermuteAtomicVectorized64Implementation = 2;
constexpr std::uint32_t kTokenPermuteAtomicVectorized256Implementation = 3;
constexpr std::uint32_t kTokenPermuteTokenOwnedTop2Implementation = 4;
constexpr std::uint32_t kTokenPermuteBlockPartialImplementation = 5;
constexpr std::uint32_t kTokenPermuteTokenTile4DirectImplementation = 6;
constexpr std::uint32_t kTokenPermuteTokenTile4WarpAggregatedImplementation = 7;
constexpr std::uint32_t kTokenPermuteShapeDispatchedV2Implementation = 8;
constexpr std::uint32_t kTokenPermuteTokenTile2DirectImplementation = 9;
constexpr std::uint32_t kTokenPermuteShapeDispatchedV3Implementation = 10;

// SM86 v2 candidate. It uses the tile4 direct path only for large, aligned
// Top-2 rows and keeps the proven token-owned path as its conservative
// fallback. Generic top-k continues to use the existing fallback inside the
// token-owned implementation. KernelFamily::kAuto deliberately remains naive.
constexpr std::uint32_t kTokenPermuteCandidateImplementation =
    kTokenPermuteShapeDispatchedV2Implementation;

inline bool is_token_permute_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kTokenPermuteAtomicVectorized128Implementation ||
         implementation_id == kTokenPermuteAtomicVectorized64Implementation ||
         implementation_id == kTokenPermuteAtomicVectorized256Implementation ||
         implementation_id == kTokenPermuteTokenOwnedTop2Implementation ||
         implementation_id == kTokenPermuteBlockPartialImplementation ||
         implementation_id == kTokenPermuteTokenTile4DirectImplementation ||
         implementation_id == kTokenPermuteTokenTile4WarpAggregatedImplementation ||
         implementation_id == kTokenPermuteShapeDispatchedV2Implementation ||
         implementation_id == kTokenPermuteTokenTile2DirectImplementation ||
         implementation_id == kTokenPermuteShapeDispatchedV3Implementation;
}

cudaError_t launch_token_permute_prepare_offsets_fused(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    std::int32_t* cursors, int tokens, int experts, int top_k,
    cudaStream_t caller_stream);

cudaError_t launch_token_permute_routeprep_shared_rank(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    std::int32_t* route_pos, std::int32_t* sorted_route, int tokens, int experts,
    int top_k, int threads, cudaStream_t caller_stream);

cudaError_t launch_token_permute_copy_token_owned_topk(
    const float* x, const std::int32_t* route_pos, float* x_permuted, int tokens,
    int top_k, int hidden, cudaStream_t caller_stream);

cudaError_t launch_token_permute_copy_from_positions(
    const float* x, const std::int32_t* route_pos, float* x_permuted, int tokens,
    int top_k, int hidden, cudaStream_t caller_stream);

cudaError_t launch_token_permute_optimized(
    const float* x, const std::int32_t* expert_ids, const std::int32_t* offsets,
    std::int32_t* cursors, float* x_permuted, std::int32_t* route_pos,
    std::int32_t* sorted_route, int tokens, int experts, int top_k, int hidden,
    std::uint32_t implementation_id, cudaStream_t caller_stream);

}  // namespace raggedroute::ops
