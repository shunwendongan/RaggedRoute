#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

// F2 is the promoted SM86 fused Histogram->Scan implementation. It keeps the
// single-CTA shared histogram and finalizes counts/offsets with a 16-lane,
// four-items-per-lane scan.
constexpr std::uint32_t kHistogramExclusiveScanFusedSubwarpImplementation = 2;
constexpr int kHistogramExclusiveScanFusedMaxRoutePairs = 4096;

inline bool is_histogram_exclusive_scan_optimized_implementation(
    std::uint32_t implementation_id) noexcept {
  return implementation_id == kHistogramExclusiveScanFusedSubwarpImplementation;
}

cudaError_t launch_histogram_exclusive_scan_optimized(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    int route_pairs, int experts, std::uint32_t implementation_id,
    cudaStream_t caller_stream);

}  // namespace raggedroute::ops
