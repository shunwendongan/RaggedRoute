#pragma once

// 运行时侧给各算子 wrapper 共用的辅助工具。
// 这里集中放 dispatch、参数校验和对齐检查，避免每个算子入口重复写同样的边界逻辑。

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

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

inline Status dispatch_operator(OperatorKind operator_kind, const OperatorSignature& signature,
                                KernelSelection requested_kernel,
                                DeviceArchitecture requested_architecture,
                                DispatchDecision* decision) noexcept {
  DeviceArchitecture architecture = DeviceArchitecture::kOther;
  Status status = resolve_device_architecture(requested_architecture, &architecture);
  if (!status.ok()) return status;
  DispatchRequest request;
  request.operator_kind = operator_kind;
  request.architecture = architecture;
  request.requested_kernel = requested_kernel;
  request.signature = signature;
  return select_kernel(request, decision);
}

inline Status require_naive_implementation(const DispatchDecision& decision) noexcept {
  if (decision.kernel.family != KernelFamily::kCudaNaive ||
      decision.kernel.implementation_id != 0) {
    return make_status(StatusCode::kUnsupportedKernelVariant,
                       "operator wrapper does not implement the selected kernel");
  }
  return success_status();
}

inline bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer == nullptr || reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

inline OperatorSignature make_compute_signature(const ConstTensorView& input,
                                                const ConstTensorView& weight,
                                                ScalarType accumulator,
                                                const MutableTensorView& output) {
  return {input.spec, weight.spec, accumulator, output.spec};
}

inline Status invalid_argument(const char* message) noexcept {
  return make_status(StatusCode::kInvalidArgument, message);
}

}  // namespace raggedroute::detail
