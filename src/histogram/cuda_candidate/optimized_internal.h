#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kHistogramCandidateV1Implementation = 101;
constexpr std::uint32_t kHistogramH5SingleBinImplementation = 102;
constexpr std::uint32_t kHistogramH6Cap256Implementation = 103;
constexpr std::uint32_t kHistogramH6Cap384Implementation = 104;
constexpr std::uint32_t kHistogramH6Cap512Implementation = 105;
// Auto remains on the promoted v1 while the v2 experiments are evaluated.
constexpr std::uint32_t kHistogramCandidateImplementation =
    kHistogramCandidateV1Implementation;
constexpr int kHistogramSingleCtaMaxRoutePairs = 4096;
constexpr int kHistogramBlockPrivateMinRoutePairs = 32768;
constexpr std::uint32_t kHistogramBlockPrivateMaxBlocks = 128;

inline bool is_histogram_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id >= kHistogramCandidateV1Implementation &&
         implementation_id <= kHistogramH6Cap512Implementation;
}

inline std::uint32_t histogram_block_private_max_blocks(
    std::uint32_t implementation_id) noexcept {
  switch (implementation_id) {
    case kHistogramH6Cap256Implementation:
      return 256;
    case kHistogramH6Cap384Implementation:
      return 384;
    case kHistogramH6Cap512Implementation:
      return 512;
    default:
      return kHistogramBlockPrivateMaxBlocks;
  }
}

inline bool histogram_optimized_overwrites_output(std::uint32_t implementation_id,
                                                  int route_pairs, int experts) noexcept {
  return (implementation_id == kHistogramH5SingleBinImplementation && experts == 1) ||
         route_pairs <= kHistogramSingleCtaMaxRoutePairs;
}

inline bool histogram_optimized_requires_external_reset(std::uint32_t implementation_id,
                                                        int route_pairs,
                                                        int experts) noexcept {
  return is_histogram_optimized_implementation(implementation_id) &&
         !histogram_optimized_overwrites_output(implementation_id, route_pairs, experts);
}

cudaError_t launch_histogram_optimized(const std::int32_t* expert_ids, std::int32_t* counts,
                                       int route_pairs, int experts,
                                       std::uint32_t implementation_id, cudaStream_t caller_stream);

}  // namespace raggedroute::ops
