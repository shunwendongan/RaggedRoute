#pragma once

#include "raggedroute/status.h"
#include "raggedroute/types.h"

namespace raggedroute {

struct DispatchRequest {
  OperatorKind operator_kind = OperatorKind::kDenseGemm;
  DeviceArchitecture architecture = DeviceArchitecture::kAuto;
  KernelVariant requested_variant = KernelVariant::kAuto;
  ScalarType scalar_type = ScalarType::kFloat32;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
};

struct DispatchDecision {
  OperatorKind operator_kind = OperatorKind::kDenseGemm;
  DeviceArchitecture architecture = DeviceArchitecture::kOther;
  KernelVariant kernel_variant = KernelVariant::kCudaNaive;
};

DeviceArchitecture classify_compute_capability(int major, int minor) noexcept;

// Queries the current CUDA device once. Callers that benchmark steady-state
// operators should cache the result in RuntimeContext::architecture.
Status query_current_device_architecture(DeviceArchitecture* architecture) noexcept;

// Resolves kAuto through the current CUDA device; explicit values require no
// CUDA runtime query.
Status resolve_device_architecture(DeviceArchitecture requested,
                                   DeviceArchitecture* resolved) noexcept;

// Pure, testable dispatch after architecture resolution. P0 intentionally
// accepts only FP32 contiguous row-major cuda_naive on SM86.
Status select_kernel(const DispatchRequest& request, DispatchDecision* decision) noexcept;

}  // namespace raggedroute
