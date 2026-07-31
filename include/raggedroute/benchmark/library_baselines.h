#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace raggedroute::benchmark::library_baseline {

struct DenseCublasLtPlan;
struct DenseCublasPlan;
struct GroupedCublasPlan;
struct GroupedCutlassPlan;

DenseCublasLtPlan* create_dense_cublaslt_plan(int m, int n, int k,
                                              std::size_t workspace_limit_bytes,
                                              std::size_t* workspace_bytes);
void destroy_dense_cublaslt_plan(DenseCublasLtPlan* plan) noexcept;
void launch_dense_cublaslt(DenseCublasLtPlan* plan, const float* a, const float* b, float* c,
                           void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

DenseCublasPlan* create_dense_cublas_plan();
void destroy_dense_cublas_plan(DenseCublasPlan* plan) noexcept;
void launch_dense_cublas(DenseCublasPlan* plan, const float* a, const float* b, float* c, int m,
                         int n, int k, cudaStream_t stream);

GroupedCublasPlan* create_grouped_cublas_plan();
void destroy_grouped_cublas_plan(GroupedCublasPlan* plan) noexcept;
void launch_grouped_cublas(GroupedCublasPlan* plan, const float* x, const float* weights,
                           float* output, const std::int32_t* host_offsets, int experts, int hidden,
                           int output_width, cudaStream_t stream);

GroupedCutlassPlan* create_grouped_cutlass_plan(const float* x, const float* weights, float* output,
                                                const std::int32_t* host_offsets, int experts,
                                                int hidden, int output_width);
void destroy_grouped_cutlass_plan(GroupedCutlassPlan* plan) noexcept;
std::size_t grouped_cutlass_workspace_bytes(const GroupedCutlassPlan* plan) noexcept;
void initialize_grouped_cutlass_plan(GroupedCutlassPlan* plan, void* workspace,
                                     std::size_t workspace_bytes, cudaStream_t stream);
void launch_grouped_cutlass(GroupedCutlassPlan* plan, void* workspace, std::size_t workspace_bytes,
                            cudaStream_t stream);

cudaError_t query_cub_histogram_workspace(std::size_t route_pairs, int experts,
                                          std::size_t* workspace_bytes);
cudaError_t launch_cub_histogram(const std::int32_t* ids, std::int32_t* counts,
                                 std::size_t route_pairs, int experts, void* workspace,
                                 std::size_t workspace_bytes, cudaStream_t stream);

cudaError_t query_cub_device_scan_workspace(int experts, std::size_t* workspace_bytes);
cudaError_t launch_cub_device_scan(const std::int32_t* counts, std::int32_t* offsets, int experts,
                                   void* workspace, std::size_t workspace_bytes,
                                   cudaStream_t stream);
cudaError_t launch_cub_block_scan(const std::int32_t* counts, std::int32_t* offsets, int experts,
                                  cudaStream_t stream);
cudaError_t launch_cub_warp_scan(const std::int32_t* counts, std::int32_t* offsets, int experts,
                                 cudaStream_t stream);

cudaError_t query_vllm_permute_workspace(int tokens, int experts, int top_k,
                                         std::size_t* workspace_bytes);
cudaError_t initialize_vllm_permute_workspace(void* workspace, std::size_t workspace_bytes,
                                              int tokens, int experts, int top_k,
                                              cudaStream_t stream);
cudaError_t prepare_vllm_permute_mapping(const std::int32_t* expert_ids, void* workspace,
                                         std::size_t workspace_bytes, int tokens, int experts,
                                         int top_k, cudaStream_t stream);
cudaError_t launch_vllm_expand_rows(const float* input, float* output, std::int32_t* route_pos,
                                    std::int32_t* optional_sorted_route, void* workspace,
                                    std::size_t workspace_bytes, int tokens, int experts, int top_k,
                                    int hidden, cudaStream_t stream);
cudaError_t launch_vllm_moe_permute(const float* input, const std::int32_t* expert_ids,
                                    float* output, std::int32_t* route_pos,
                                    std::int32_t* optional_sorted_route, void* workspace,
                                    std::size_t workspace_bytes, int tokens, int experts, int top_k,
                                    int hidden, cudaStream_t stream);

cudaError_t launch_vllm_finalize_routing(const float* permuted_rows, float* output,
                                         const float* route_weights, const std::int32_t* route_pos,
                                         int tokens, int top_k, int output_width,
                                         cudaStream_t stream);

}  // namespace raggedroute::benchmark::library_baseline
