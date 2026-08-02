#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kHistogramWarpAggregatedImplementation = 101;
constexpr std::uint32_t kHistogramSingleCtaSharedImplementation = 102;

inline bool is_histogram_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kHistogramWarpAggregatedImplementation ||
         implementation_id == kHistogramSingleCtaSharedImplementation;
}

inline bool histogram_optimized_requires_external_reset(
    std::uint32_t implementation_id) noexcept {
  return implementation_id == kHistogramWarpAggregatedImplementation;
}

cudaError_t launch_histogram_optimized(const std::int32_t* expert_ids, std::int32_t* counts,
                                       int route_pairs, int experts,
                                       std::uint32_t implementation_id,
                                       cudaStream_t caller_stream);

}  // namespace raggedroute::ops
