// Adapted from vLLM commit 837eae64580c885101ee95b073aafb27a485e7ce.
// Original files: moe_permute_unpermute_kernel.{h,inl,cu}.
// Copyright The vLLM Team. Licensed under Apache-2.0.
// Modified for a standalone FP32 benchmark baseline, explicit workspace ownership,
// and a scalar fallback for rows that do not satisfy the upstream 16-byte contract.

#include <cstdint>
#include <cub/device/device_radix_sort.cuh>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

constexpr std::size_t kAlignment = 16;

bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % kAlignment == 0;
}

std::size_t align_up(std::size_t value) {
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

int expert_bits(int experts) {
  int bits = 0;
  unsigned int value = static_cast<unsigned int>(2 * experts - 1);
  while (value != 0) {
    ++bits;
    value >>= 1U;
  }
  return bits;
}

struct WorkspaceLayout {
  std::int32_t* source_rows = nullptr;
  std::int32_t* sorted_experts = nullptr;
  std::int32_t* sorted_routes = nullptr;
  std::int32_t* internal_sorted_route = nullptr;
  std::int64_t* expert_offsets = nullptr;
  void* sort_storage = nullptr;
  std::size_t sort_storage_bytes = 0;
  std::size_t required_bytes = 0;
};

cudaError_t make_layout(void* workspace, int tokens, int experts, int top_k,
                        WorkspaceLayout* layout) {
  if (layout == nullptr || tokens < 0 || experts < 1 || top_k < 1) {
    return cudaErrorInvalidValue;
  }
  const std::size_t routes = static_cast<std::size_t>(tokens) * top_k;
  std::size_t sort_bytes = 0;
  std::int32_t* null_data = nullptr;
  cudaError_t error =
      cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, null_data, null_data, null_data,
                                      null_data, routes, 0, expert_bits(experts));
  if (error != cudaSuccess) return error;
  sort_bytes = align_up(sort_bytes == 0 ? 1 : sort_bytes);

  std::size_t offset = 0;
  auto take = [&](std::size_t bytes) {
    const std::size_t current = offset;
    offset += align_up(bytes);
    return current;
  };
  const std::size_t source_offset = take(routes * sizeof(std::int32_t));
  const std::size_t expert_offset = take(routes * sizeof(std::int32_t));
  const std::size_t route_offset = take(routes * sizeof(std::int32_t));
  const std::size_t internal_offset = take(routes * sizeof(std::int32_t));
  const std::size_t offsets_offset =
      take((static_cast<std::size_t>(experts) + 1) * sizeof(std::int64_t));
  const std::size_t sort_offset = take(sort_bytes);
  layout->required_bytes = offset;
  layout->sort_storage_bytes = sort_bytes;
  if (workspace != nullptr) {
    auto* bytes = static_cast<std::uint8_t*>(workspace);
    layout->source_rows = reinterpret_cast<std::int32_t*>(bytes + source_offset);
    layout->sorted_experts = reinterpret_cast<std::int32_t*>(bytes + expert_offset);
    layout->sorted_routes = reinterpret_cast<std::int32_t*>(bytes + route_offset);
    layout->internal_sorted_route = reinterpret_cast<std::int32_t*>(bytes + internal_offset);
    layout->expert_offsets = reinterpret_cast<std::int64_t*>(bytes + offsets_offset);
    layout->sort_storage = bytes + sort_offset;
  }
  return cudaSuccess;
}

__global__ void initialize_source_rows(std::int32_t* source_rows, int routes) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < routes) source_rows[index] = index;
}

__device__ std::int64_t lower_bound_expert(const std::int32_t* sorted_experts, int routes,
                                           int expert) {
  int low = 0;
  int high = routes;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (sorted_experts[middle] < expert)
      low = middle + 1;
    else
      high = middle;
  }
  return low;
}

__global__ void compute_expert_offsets(const std::int32_t* sorted_experts, int routes, int experts,
                                       std::int64_t* offsets) {
  const int expert = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (expert <= experts) offsets[expert] = lower_bound_expert(sorted_experts, routes, expert);
}

template <bool Vectorized>
__global__ void expand_rows(const float* input, float* output, const std::int32_t* sorted_routes,
                            std::int32_t* route_pos, std::int32_t* sorted_route, int routes,
                            int top_k, int hidden) {
  const int destination = static_cast<int>(blockIdx.x);
  if (destination >= routes) return;
  const int source_route = sorted_routes[destination];
  if (threadIdx.x == 0) {
    route_pos[source_route] = destination;
    sorted_route[destination] = source_route;
  }
  const int source_token = source_route / top_k;
  if constexpr (Vectorized) {
    const auto* source = reinterpret_cast<const float4*>(input + source_token * hidden);
    auto* target = reinterpret_cast<float4*>(output + destination * hidden);
    const int vectors = hidden / 4;
    for (int index = static_cast<int>(threadIdx.x); index < vectors; index += blockDim.x) {
      target[index] = source[index];
    }
  } else {
    for (int column = static_cast<int>(threadIdx.x); column < hidden; column += blockDim.x) {
      output[static_cast<std::size_t>(destination) * hidden + column] =
          input[static_cast<std::size_t>(source_token) * hidden + column];
    }
  }
}

}  // namespace

cudaError_t query_vllm_permute_workspace(int tokens, int experts, int top_k,
                                         std::size_t* workspace_bytes) {
  if (workspace_bytes == nullptr) return cudaErrorInvalidValue;
  WorkspaceLayout layout;
  const cudaError_t error = make_layout(nullptr, tokens, experts, top_k, &layout);
  if (error == cudaSuccess) *workspace_bytes = layout.required_bytes;
  return error;
}

cudaError_t initialize_vllm_permute_workspace(void* workspace, std::size_t workspace_bytes,
                                              int tokens, int experts, int top_k,
                                              cudaStream_t stream) {
  WorkspaceLayout layout;
  cudaError_t error = make_layout(workspace, tokens, experts, top_k, &layout);
  if (error != cudaSuccess || workspace == nullptr || workspace_bytes < layout.required_bytes) {
    return cudaErrorInvalidValue;
  }
  const int routes = tokens * top_k;
  if (routes == 0) return cudaSuccess;
  initialize_source_rows<<<(routes + 255) / 256, 256, 0, stream>>>(layout.source_rows, routes);
  return cudaPeekAtLastError();
}

cudaError_t prepare_vllm_permute_mapping(const std::int32_t* expert_ids, void* workspace,
                                         std::size_t workspace_bytes, int tokens, int experts,
                                         int top_k, cudaStream_t stream) {
  WorkspaceLayout layout;
  cudaError_t error = make_layout(workspace, tokens, experts, top_k, &layout);
  if (error != cudaSuccess || workspace == nullptr || workspace_bytes < layout.required_bytes) {
    return cudaErrorInvalidValue;
  }
  const int routes = tokens * top_k;
  if (routes == 0) return cudaSuccess;
  error = cub::DeviceRadixSort::SortPairs(
      layout.sort_storage, layout.sort_storage_bytes, expert_ids, layout.sorted_experts,
      layout.source_rows, layout.sorted_routes, routes, 0, expert_bits(experts), stream);
  if (error != cudaSuccess) return error;
  const int entries = experts + 1;
  compute_expert_offsets<<<(entries + 255) / 256, 256, 0, stream>>>(layout.sorted_experts, routes,
                                                                    experts, layout.expert_offsets);
  return cudaPeekAtLastError();
}

cudaError_t launch_vllm_expand_rows(const float* input, float* output, std::int32_t* route_pos,
                                    std::int32_t* optional_sorted_route, void* workspace,
                                    std::size_t workspace_bytes, int tokens, int experts, int top_k,
                                    int hidden, cudaStream_t stream) {
  WorkspaceLayout layout;
  cudaError_t error = make_layout(workspace, tokens, experts, top_k, &layout);
  if (error != cudaSuccess || workspace == nullptr || workspace_bytes < layout.required_bytes ||
      input == nullptr || output == nullptr || route_pos == nullptr || hidden < 0) {
    return cudaErrorInvalidValue;
  }
  const int routes = tokens * top_k;
  if (routes == 0 || hidden == 0) return cudaSuccess;
  std::int32_t* sorted_route =
      optional_sorted_route == nullptr ? layout.internal_sorted_route : optional_sorted_route;
  if (hidden % 4 == 0 && is_aligned_16(input) && is_aligned_16(output)) {
    expand_rows<true><<<routes, 256, 0, stream>>>(input, output, layout.sorted_routes, route_pos,
                                                  sorted_route, routes, top_k, hidden);
  } else {
    expand_rows<false><<<routes, 256, 0, stream>>>(input, output, layout.sorted_routes, route_pos,
                                                   sorted_route, routes, top_k, hidden);
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_vllm_moe_permute(const float* input, const std::int32_t* expert_ids,
                                    float* output, std::int32_t* route_pos,
                                    std::int32_t* optional_sorted_route, void* workspace,
                                    std::size_t workspace_bytes, int tokens, int experts, int top_k,
                                    int hidden, cudaStream_t stream) {
  cudaError_t error = prepare_vllm_permute_mapping(expert_ids, workspace, workspace_bytes, tokens,
                                                   experts, top_k, stream);
  if (error != cudaSuccess) return error;
  return launch_vllm_expand_rows(input, output, route_pos, optional_sorted_route, workspace,
                                 workspace_bytes, tokens, experts, top_k, hidden, stream);
}

}  // namespace raggedroute::benchmark::library_baseline
