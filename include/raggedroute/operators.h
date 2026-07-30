#pragma once

#include <cstddef>
#include <cstdint>

#include "raggedroute/runtime.h"
#include "raggedroute/status.h"
#include "raggedroute/types.h"

namespace raggedroute {

// Unless a field says otherwise, non-null pointers are CUDA device pointers.
// P0 requires contiguous row-major storage, valid route ids in [0, experts),
// monotonic offsets with offsets[experts] equal to the route count, and
// non-overlapping input/output buffers. Device-side value validation would add
// a synchronization or an extra kernel and is therefore not part of the
// steady-state P0 contract.

struct DenseGemmArgs {
  const float* a = nullptr;
  const float* b = nullptr;
  float* c = nullptr;
  int m = 0;
  int n = 0;
  int k = 0;
  float alpha = 1.0F;
  float beta = 0.0F;
  ScalarType scalar_type = ScalarType::kFloat32;
  TensorLayout layout_a = TensorLayout::kRowMajorContiguous;
  TensorLayout layout_b = TensorLayout::kRowMajorContiguous;
  TensorLayout layout_c = TensorLayout::kRowMajorContiguous;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

struct TopKGateArgs {
  const float* logits = nullptr;
  std::int32_t* expert_ids = nullptr;
  float* weights = nullptr;
  int tokens = 0;
  int experts = 0;
  int top_k = 2;
  ScalarType scalar_type = ScalarType::kFloat32;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
  TopKNormalization normalization = TopKNormalization::kSelectedSoftmax;
  TopKTieBreak tie_break = TopKTieBreak::kLowerExpertId;
  TopKNaNPolicy nan_policy = TopKNaNPolicy::kNegativeInfinityAllNaNFallback01;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

struct HistogramArgs {
  const std::int32_t* expert_ids = nullptr;
  std::int32_t* counts = nullptr;
  int route_pairs = 0;
  int experts = 0;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

struct ExclusiveScanArgs {
  const std::int32_t* counts = nullptr;
  std::int32_t* offsets = nullptr;
  int experts = 0;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

struct TokenPermuteArgs {
  const float* x = nullptr;
  const std::int32_t* expert_ids = nullptr;
  const std::int32_t* offsets = nullptr;
  float* x_permuted = nullptr;
  std::int32_t* route_pos = nullptr;
  // Optional inverse mapping: sorted_route[route_pos[r]] == r.
  std::int32_t* sorted_route = nullptr;
  int tokens = 0;
  int experts = 0;
  int top_k = 0;
  int hidden = 0;
  ScalarType scalar_type = ScalarType::kFloat32;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

struct GroupedGemmArgs {
  const float* x_permuted = nullptr;
  const float* expert_weights = nullptr;
  const std::int32_t* offsets = nullptr;
  float* y_permuted = nullptr;
  int experts = 0;
  int hidden = 0;
  int output = 0;
  // Caller-provided launch bound. P0 never synchronizes to read offsets on the
  // host; R=tokens*top_k is a valid conservative value for a full chain.
  int max_expert_tokens = 0;
  ScalarType scalar_type = ScalarType::kFloat32;
  TensorLayout layout_x = TensorLayout::kRowMajorContiguous;
  TensorLayout layout_weights = TensorLayout::kRowMajorContiguous;
  TensorLayout layout_y = TensorLayout::kRowMajorContiguous;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

struct UnpermuteArgs {
  const float* y_permuted = nullptr;
  const std::int32_t* route_pos = nullptr;
  const float* route_weights = nullptr;
  float* y = nullptr;
  int tokens = 0;
  int top_k = 0;
  int output = 0;
  ScalarType scalar_type = ScalarType::kFloat32;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
  KernelVariant kernel_variant = KernelVariant::kAuto;
};

std::size_t get_dense_gemm_workspace_size(const DenseGemmArgs& args) noexcept;
std::size_t get_topk_gate_workspace_size(const TopKGateArgs& args) noexcept;
std::size_t get_histogram_workspace_size(const HistogramArgs& args) noexcept;
std::size_t get_exclusive_scan_workspace_size(const ExclusiveScanArgs& args) noexcept;
std::size_t get_token_permute_workspace_size(const TokenPermuteArgs& args) noexcept;
std::size_t get_grouped_gemm_workspace_size(const GroupedGemmArgs& args) noexcept;
std::size_t get_unpermute_workspace_size(const UnpermuteArgs& args) noexcept;

// Complete operator wrappers. They use the caller stream, never allocate on
// the hot path, never unconditionally synchronize, and include mandatory
// device-side reset work in the call boundary.
Status dense_gemm(const DenseGemmArgs& args, const RuntimeContext& context) noexcept;
Status topk_gate(const TopKGateArgs& args, const RuntimeContext& context) noexcept;
Status histogram(const HistogramArgs& args, const RuntimeContext& context) noexcept;
Status exclusive_scan(const ExclusiveScanArgs& args, const RuntimeContext& context) noexcept;
Status token_permute(const TokenPermuteArgs& args, const RuntimeContext& context) noexcept;
Status grouped_gemm(const GroupedGemmArgs& args, const RuntimeContext& context) noexcept;
Status unpermute(const UnpermuteArgs& args, const RuntimeContext& context) noexcept;

}  // namespace raggedroute
