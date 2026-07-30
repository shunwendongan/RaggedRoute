#pragma once

#include <optional>

#include "raggedroute/status.h"
#include "raggedroute/types.h"

namespace raggedroute {

// Floating operand roles used as the dispatch key. std::nullopt means that the
// role is not part of the operator contract (for example Histogram and Scan).
struct OperatorSignature {
  std::optional<TensorSpec> input;
  std::optional<TensorSpec> weight;
  std::optional<ScalarType> accumulator;
  std::optional<TensorSpec> output;
};

struct DispatchRequest {
  OperatorKind operator_kind = OperatorKind::kDenseGemm;
  DeviceArchitecture architecture = DeviceArchitecture::kAuto;
  KernelSelection requested_kernel;
  OperatorSignature signature;
};

struct DispatchDecision {
  OperatorKind operator_kind = OperatorKind::kDenseGemm;
  DeviceArchitecture architecture = DeviceArchitecture::kOther;
  KernelSelection kernel;
};

DeviceArchitecture classify_compute_capability(int major, int minor) noexcept;

// Queries the current CUDA device once. Callers that benchmark steady-state
// operators should cache the result in RuntimeContext::architecture.
Status query_current_device_architecture(DeviceArchitecture* architecture) noexcept;

// Resolves kAuto through the current CUDA device; explicit values require no
// CUDA runtime query.
Status resolve_device_architecture(DeviceArchitecture requested,
                                   DeviceArchitecture* resolved) noexcept;

// Pure, testable dispatch after architecture resolution. v0.2 intentionally
// accepts only the operator-appropriate all-FP32, contiguous row-major
// cuda_naive signatures on SM86.
Status select_kernel(const DispatchRequest& request, DispatchDecision* decision) noexcept;

}  // namespace raggedroute
