#pragma once

#include <cstddef>
#include <cstdint>

#include "raggedroute/runtime.h"
#include "raggedroute/status.h"
#include "raggedroute/types.h"

namespace raggedroute {

// Unless a field says otherwise, non-null pointers are CUDA device pointers.
// v0.2 requires zero-stride contiguous row-major storage, valid route ids in
// [0, experts), monotonic offsets with offsets[experts] equal to the route
// count, and non-overlapping input/output buffers. Device-side value validation
// would add a synchronization or an extra kernel and is therefore not part of
// the steady-state v0.2 contract.

struct DenseGemmArgs {
  ConstTensorView a;
  ConstTensorView b;
  MutableTensorView c;
  int m = 0;
  int n = 0;
  int k = 0;
  float alpha = 1.0F;
  float beta = 0.0F;
  ScalarType accumulator_dtype = ScalarType::kFp32;
  KernelSelection kernel;
};

struct TopKGateArgs {
  ConstTensorView logits;
  std::int32_t* expert_ids = nullptr;
  MutableTensorView weights;
  int tokens = 0;
  int experts = 0;
  int top_k = 2;
  ScalarType accumulator_dtype = ScalarType::kFp32;
  TopKNormalization normalization = TopKNormalization::kSelectedSoftmax;
  TopKTieBreak tie_break = TopKTieBreak::kLowerExpertId;
  TopKNaNPolicy nan_policy = TopKNaNPolicy::kNegativeInfinityAllNaNFallback01;
  KernelSelection kernel;
};

struct HistogramArgs {
  const std::int32_t* expert_ids = nullptr;
  std::int32_t* counts = nullptr;
  int route_pairs = 0;
  int experts = 0;
  KernelSelection kernel;
};

struct ExclusiveScanArgs {
  const std::int32_t* counts = nullptr;
  std::int32_t* offsets = nullptr;
  int experts = 0;
  KernelSelection kernel;
};

struct TokenPermuteArgs {
  ConstTensorView x;
  const std::int32_t* expert_ids = nullptr;
  const std::int32_t* offsets = nullptr;
  MutableTensorView x_permuted;
  std::int32_t* route_pos = nullptr;
  // Optional inverse mapping: sorted_route[route_pos[r]] == r.
  std::int32_t* sorted_route = nullptr;
  int tokens = 0;
  int experts = 0;
  int top_k = 0;
  int hidden = 0;
  KernelSelection kernel;
};

struct GroupedGemmArgs {
  ConstTensorView x_permuted;
  ConstTensorView expert_weights;
  const std::int32_t* offsets = nullptr;
  MutableTensorView y_permuted;
  int experts = 0;
  int hidden = 0;
  int output = 0;
  // Caller-provided launch bound. v0.2 never synchronizes to read offsets on the
  // host; R=tokens*top_k is a valid conservative value for a full chain.
  int max_expert_tokens = 0;
  ScalarType accumulator_dtype = ScalarType::kFp32;
  KernelSelection kernel;
};

struct UnpermuteArgs {
  ConstTensorView y_permuted;
  const std::int32_t* route_pos = nullptr;
  ConstTensorView route_weights;
  MutableTensorView y;
  int tokens = 0;
  int top_k = 0;
  int output = 0;
  ScalarType accumulator_dtype = ScalarType::kFp32;
  KernelSelection kernel;
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
