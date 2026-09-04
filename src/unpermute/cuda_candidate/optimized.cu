#include "optimized_internal.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "raggedroute/baseline_ops.h"

// token unpermute 的优化 kernel 家族。
// 处理 y_permuted[T*top_k,O]、route_pos[T*top_k] 和 route_weights[T*top_k]
// -> y[T,O]，并为常见的 top_k=2 场景准备了更快的路径。
namespace raggedroute::ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kLargeOutputThreshold = 512;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

__global__ void unpermute_warp_token_vec4_kernel(
    const float* y_permuted, const std::int32_t* route_pos,
    const float* route_weights, float* y, int tokens, int output) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int token = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  if (token >= tokens) return;

  int source0 = 0;
  int source1 = 0;
  float weight0 = 0.0F;
  float weight1 = 0.0F;
  if (lane == 0) {
    const int route = token * 2;
    source0 = route_pos[route];
    source1 = route_pos[route + 1];
    weight0 = route_weights[route];
    weight1 = route_weights[route + 1];
  }
  source0 = __shfl_sync(kFullWarpMask, source0, 0);
  source1 = __shfl_sync(kFullWarpMask, source1, 0);
  weight0 = __shfl_sync(kFullWarpMask, weight0, 0);
  weight1 = __shfl_sync(kFullWarpMask, weight1, 0);

  const int vectors = output / 4;
  const auto* source0_row = reinterpret_cast<const float4*>(
      y_permuted + static_cast<std::size_t>(source0) * output);
  const auto* source1_row = reinterpret_cast<const float4*>(
      y_permuted + static_cast<std::size_t>(source1) * output);
  auto* target_row =
      reinterpret_cast<float4*>(y + static_cast<std::size_t>(token) * output);
  for (int vector = lane; vector < vectors; vector += kWarpSize) {
    const float4 value0 = source0_row[vector];
    const float4 value1 = source1_row[vector];
    float4 result;
    result.x = weight0 * value0.x + weight1 * value1.x;
    result.y = weight0 * value0.y + weight1 * value1.y;
    result.z = weight0 * value0.z + weight1 * value1.z;
    result.w = weight0 * value0.w + weight1 * value1.w;
    target_row[vector] = result;
  }
}

__global__ void unpermute_cta_token_vec4_kernel(
    const float* y_permuted, const std::int32_t* route_pos,
    const float* route_weights, float* y, int tokens, int output) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;

  __shared__ int sources[2];
  __shared__ float weights[2];
  if (threadIdx.x == 0) {
    const int route = token * 2;
    sources[0] = route_pos[route];
    sources[1] = route_pos[route + 1];
    weights[0] = route_weights[route];
    weights[1] = route_weights[route + 1];
  }
  __syncthreads();

  const int vectors = output / 4;
  const auto* source0_row = reinterpret_cast<const float4*>(
      y_permuted + static_cast<std::size_t>(sources[0]) * output);
  const auto* source1_row = reinterpret_cast<const float4*>(
      y_permuted + static_cast<std::size_t>(sources[1]) * output);
  auto* target_row =
      reinterpret_cast<float4*>(y + static_cast<std::size_t>(token) * output);
  for (int vector = static_cast<int>(threadIdx.x); vector < vectors;
       vector += static_cast<int>(blockDim.x)) {
    const float4 value0 = source0_row[vector];
    const float4 value1 = source1_row[vector];
    float4 result;
    result.x = weights[0] * value0.x + weights[1] * value1.x;
    result.y = weights[0] * value0.y + weights[1] * value1.y;
    result.z = weights[0] * value0.z + weights[1] * value1.z;
    result.w = weights[0] * value0.w + weights[1] * value1.w;
    target_row[vector] = result;
  }
}

}  // namespace

cudaError_t launch_unpermute_optimized(const float* y_permuted,
                                       const std::int32_t* route_pos,
                                       const float* route_weights, float* y, int tokens,
                                       int top_k, int output,
                                       cudaStream_t caller_stream) {
  if (tokens < 0 || top_k <= 0 || top_k > 64 || output < 0) return cudaErrorInvalidValue;
  if (tokens == 0 || output == 0) return cudaSuccess;
  if (tokens > std::numeric_limits<int>::max() / top_k ||
      tokens > std::numeric_limits<int>::max() / output || y_permuted == nullptr ||
      route_pos == nullptr || route_weights == nullptr || y == nullptr) {
    return cudaErrorInvalidValue;
  }

  const bool vector_contract = top_k == 2 && output % 4 == 0 &&
                               is_aligned_16(y_permuted) && is_aligned_16(y);
  if (!vector_contract) {
    return launch_unpermute_naive(y_permuted, route_pos, route_weights, y, tokens, top_k,
                                  output, caller_stream);
  }

  if (output < kLargeOutputThreshold) {
    constexpr int kThreads = kWarpsPerBlock * kWarpSize;
    const unsigned int blocks =
        static_cast<unsigned int>((tokens + kWarpsPerBlock - 1) / kWarpsPerBlock);
    unpermute_warp_token_vec4_kernel<<<blocks, kThreads, 0, caller_stream>>>(
        y_permuted, route_pos, route_weights, y, tokens, output);
  } else {
    constexpr int kThreads = 256;
    unpermute_cta_token_vec4_kernel<<<static_cast<unsigned int>(tokens), kThreads, 0,
                                      caller_stream>>>(y_permuted, route_pos, route_weights, y,
                                                       tokens, output);
  }
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
