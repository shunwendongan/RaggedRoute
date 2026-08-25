#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kGroupedGemmTiled16SyncV0Implementation = 1;
constexpr std::uint32_t kGroupedGemmPersistent16V1Implementation = 2;
constexpr std::uint32_t kGroupedGemmRegister16x32SyncV2Implementation = 3;
constexpr std::uint32_t kGroupedGemmRegister16x32AsyncV3Implementation = 4;
constexpr std::uint32_t kGroupedGemmSm86Fp32V1Implementation = 5;
constexpr std::uint32_t kGroupedGemmRegister16x32AsyncFullV4Implementation = 6;
constexpr std::uint32_t kGroupedGemmSm86Fp32V2Implementation = 7;

cudaError_t launch_grouped_gemm_optimized(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    std::uint32_t implementation_id, cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_tiled16_sync_v0(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_persistent16_v1(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_register16x32_sync_v2(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_register16x32_async_v3(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_register16x32_async_full_v4(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_sm86_fp32_v2(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

}  // namespace raggedroute::ops
