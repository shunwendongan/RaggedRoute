#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace raggedroute::ops {

constexpr std::uint32_t kGroupedGemmTiled16SyncV0Implementation = 1;
constexpr std::uint32_t kGroupedGemmPersistent16V1Implementation = 2;
constexpr std::uint32_t kGroupedGemmRegister16x32SyncV2Implementation = 3;
constexpr std::uint32_t kGroupedGemmRegister16x32AsyncV3Implementation = 4;
constexpr std::uint32_t kGroupedGemmSm86Fp32V1Implementation = 5;
constexpr std::uint32_t kGroupedGemmRegister16x32AsyncFullV4Implementation = 6;
constexpr std::uint32_t kGroupedGemmSm86Fp32V2Implementation = 7;
constexpr std::uint32_t kGroupedGemmSm86Fp32V3Implementation = 8;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescStaticT256Implementation = 9;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescStaticT512Implementation = 10;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescStaticT1024Implementation = 11;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescQueueT256Implementation = 12;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescQueueT512Implementation = 13;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescQueueT1024Implementation = 14;
constexpr std::uint32_t kGroupedGemmSm86Fp32V4BCacheOrderImplementation = 15;

// The alias is benchmark-only. It is intentionally independent of public
// runtime dispatch and may be rebound after the release matrix is evaluated.
constexpr std::uint32_t kGroupedGemmSm86Fp32V4ADescImplementation =
    kGroupedGemmSm86Fp32V4ADescStaticT256Implementation;

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

cudaError_t launch_grouped_gemm_sm86_fp32_v3(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    cudaStream_t caller_stream);

std::size_t grouped_gemm_descriptor_workspace_size(int experts, int output,
                                                   int max_expert_tokens) noexcept;

cudaError_t launch_grouped_gemm_sm86_fp32_v4_descriptor(
    const float* x_permuted, const float* expert_weights, const std::int32_t* offsets,
    float* y_permuted, int experts, int hidden, int output, int max_expert_tokens,
    void* workspace, std::size_t workspace_bytes, std::uint32_t implementation_id,
    cudaStream_t caller_stream);

cudaError_t launch_grouped_gemm_sm86_fp32_gather_v1(
    const float* x, const std::int32_t* sorted_route, int top_k,
    const float* expert_weights, const std::int32_t* offsets, float* y_permuted,
    int experts, int hidden, int output, int max_expert_tokens, void* workspace,
    std::size_t workspace_bytes, cudaStream_t caller_stream);

}  // namespace raggedroute::ops
