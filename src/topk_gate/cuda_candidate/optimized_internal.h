#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kTopKGateWarpPairV1Implementation = 1;
constexpr std::uint32_t kTopKGateSubwarpPairV2Implementation = 2;
constexpr std::uint32_t kTopKGateVectorPairV3Implementation = 3;
constexpr std::uint32_t kTopKGateLocalPairTwoReduceV4Implementation = 4;

bool is_topk_gate_optimized_implementation(std::uint32_t implementation_id) noexcept;

cudaError_t launch_topk_gate_optimized(const float* logits, std::int32_t* expert_ids,
                                       float* weights, int tokens, int experts,
                                       std::uint32_t implementation_id, cudaStream_t caller_stream);

}  // namespace raggedroute::ops
