#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

// Implementation 2 is the strongest historical candidate from the first SM86
// scan campaign. IDs 1 and 3 remain reserved by that campaign.
constexpr std::uint32_t kExclusiveScanWarpBlockedScalarLegacyImplementation = 2;
constexpr std::uint32_t kExclusiveScanSubwarp4ScalarImplementation = 4;
constexpr std::uint32_t kExclusiveScanSubwarp4VectorImplementation = 5;

constexpr std::uint32_t kHistogramExclusiveScanFusedScalarImplementation = 1;
constexpr std::uint32_t kHistogramExclusiveScanFusedSubwarpImplementation = 2;
constexpr int kHistogramExclusiveScanFusedMaxRoutePairs = 4096;

inline bool is_exclusive_scan_optimized_implementation(
    std::uint32_t implementation_id) noexcept {
  return implementation_id == kExclusiveScanWarpBlockedScalarLegacyImplementation ||
         implementation_id == kExclusiveScanSubwarp4ScalarImplementation ||
         implementation_id == kExclusiveScanSubwarp4VectorImplementation;
}

inline bool is_histogram_exclusive_scan_optimized_implementation(
    std::uint32_t implementation_id) noexcept {
  return implementation_id == kHistogramExclusiveScanFusedScalarImplementation ||
         implementation_id == kHistogramExclusiveScanFusedSubwarpImplementation;
}

cudaError_t launch_exclusive_scan_optimized(const std::int32_t* counts,
                                            std::int32_t* offsets, int experts,
                                            std::uint32_t implementation_id,
                                            cudaStream_t caller_stream);

cudaError_t launch_histogram_exclusive_scan_optimized(
    const std::int32_t* expert_ids, std::int32_t* counts, std::int32_t* offsets,
    int route_pairs, int experts, std::uint32_t implementation_id,
    cudaStream_t caller_stream);

}  // namespace raggedroute::ops
