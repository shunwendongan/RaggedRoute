#pragma once

#include <cuda_runtime_api.h>

#include "raggedroute/dispatch.h"
#include "raggedroute/runtime.h"
#include "raggedroute/status.h"

namespace raggedroute::detail {

inline Status make_status(StatusCode code, const char* message,
                          cudaError_t cuda_error = cudaSuccess) noexcept {
  return {code, cuda_error, message};
}

inline Status cuda_status(cudaError_t error, const char* message) noexcept {
  return error == cudaSuccess ? success_status()
                              : make_status(StatusCode::kCudaError, message, error);
}

inline Status dispatch_operator(OperatorKind operator_kind, ScalarType scalar_type,
                                TensorLayout layout, KernelVariant requested_variant,
                                DeviceArchitecture requested_architecture,
                                DispatchDecision* decision) noexcept {
  DeviceArchitecture architecture = DeviceArchitecture::kOther;
  Status status = resolve_device_architecture(requested_architecture, &architecture);
  if (!status.ok()) return status;
  DispatchRequest request;
  request.operator_kind = operator_kind;
  request.architecture = architecture;
  request.requested_variant = requested_variant;
  request.scalar_type = scalar_type;
  request.layout = layout;
  return select_kernel(request, decision);
}

inline Status unsupported_layout(const char* message) noexcept {
  return make_status(StatusCode::kUnsupportedLayout, message);
}

inline Status invalid_argument(const char* message) noexcept {
  return make_status(StatusCode::kInvalidArgument, message);
}

}  // namespace raggedroute::detail
