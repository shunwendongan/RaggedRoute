#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kHistogramCandidateV1Implementation = 101;
constexpr std::uint32_t kHistogramCandidateV2Implementation = 102;
constexpr std::uint32_t kHistogramCandidateV3Implementation = 103;
constexpr std::uint32_t kHistogramCandidateV4Implementation = 104;
// Keep the public/default dispatch pinned to the retained main implementation.
// V2 and rejected CTA-cap experiments V3/V4 are benchmark-only controls.
constexpr std::uint32_t kHistogramCandidateImplementation = kHistogramCandidateV1Implementation;
constexpr int kHistogramSingleCtaMaxRoutePairs = 4096;
constexpr int kHistogramBlockPrivateMinRoutePairs = 32768;
constexpr std::uint32_t kHistogramBlockPrivateMaxBlocks = 128;

inline bool is_histogram_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kHistogramCandidateImplementation;
}

inline bool is_histogram_benchmark_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kHistogramCandidateV1Implementation ||
         implementation_id == kHistogramCandidateV2Implementation ||
         implementation_id == kHistogramCandidateV3Implementation ||
         implementation_id == kHistogramCandidateV4Implementation;
}

inline bool histogram_optimized_overwrites_output(std::uint32_t implementation_id,
                                                  int route_pairs, int experts) noexcept {
  return ((implementation_id == kHistogramCandidateV2Implementation ||
           implementation_id == kHistogramCandidateV3Implementation ||
           implementation_id == kHistogramCandidateV4Implementation) && experts == 1) ||
         route_pairs <= kHistogramSingleCtaMaxRoutePairs;
}

inline bool histogram_optimized_requires_external_reset(std::uint32_t implementation_id,
                                                        int route_pairs,
                                                        int experts) noexcept {
  return is_histogram_benchmark_implementation(implementation_id) &&
         !histogram_optimized_overwrites_output(implementation_id, route_pairs, experts);
}

cudaError_t launch_histogram_optimized(const std::int32_t* expert_ids, std::int32_t* counts,
                                       int route_pairs, int experts,
                                       std::uint32_t implementation_id, cudaStream_t caller_stream);

}  // namespace raggedroute::ops
