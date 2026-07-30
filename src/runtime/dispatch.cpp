#include <cuda_runtime_api.h>

#include "raggedroute/dispatch.h"
#include "operator_internal.h"

namespace raggedroute {

DeviceArchitecture classify_compute_capability(int major, int minor) noexcept {
  return major == 8 && minor == 6 ? DeviceArchitecture::kSm86 : DeviceArchitecture::kOther;
}

Status query_current_device_architecture(DeviceArchitecture* architecture) noexcept {
  if (architecture == nullptr) {
    return detail::invalid_argument("architecture output must not be null");
  }

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return detail::cuda_status(error, "cudaGetDevice failed while resolving architecture");
  }

  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return detail::cuda_status(error,
                               "cudaGetDeviceProperties failed while resolving architecture");
  }
  *architecture = classify_compute_capability(properties.major, properties.minor);
  return success_status();
}

Status resolve_device_architecture(DeviceArchitecture requested,
                                   DeviceArchitecture* resolved) noexcept {
  if (resolved == nullptr) {
    return detail::invalid_argument("resolved architecture output must not be null");
  }
  if (requested == DeviceArchitecture::kAuto) {
    return query_current_device_architecture(resolved);
  }
  *resolved = requested;
  return success_status();
}

Status select_kernel(const DispatchRequest& request, DispatchDecision* decision) noexcept {
  if (decision == nullptr) {
    return detail::invalid_argument("dispatch decision output must not be null");
  }
  switch (request.operator_kind) {
    case OperatorKind::kDenseGemm:
    case OperatorKind::kTopKGate:
    case OperatorKind::kHistogram:
    case OperatorKind::kExclusiveScan:
    case OperatorKind::kTokenPermute:
    case OperatorKind::kGroupedGemm:
    case OperatorKind::kUnpermute:
      break;
    default:
      return detail::invalid_argument("unknown operator kind");
  }
  if (request.scalar_type != ScalarType::kFloat32) {
    return detail::make_status(StatusCode::kUnsupportedDataType,
                               "P0 dispatch supports FP32 only");
  }
  if (request.layout != TensorLayout::kRowMajorContiguous) {
    return detail::make_status(StatusCode::kUnsupportedLayout,
                               "P0 dispatch requires contiguous row-major tensors");
  }
  if (request.architecture == DeviceArchitecture::kAuto) {
    return detail::invalid_argument("select_kernel requires a resolved architecture");
  }
  if (request.architecture != DeviceArchitecture::kSm86) {
    return detail::make_status(StatusCode::kUnsupportedArchitecture,
                               "P0 kernels are validated for SM86 only");
  }
  if (request.requested_variant != KernelVariant::kAuto &&
      request.requested_variant != KernelVariant::kCudaNaive) {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "requested kernel variant is not implemented");
  }

  decision->operator_kind = request.operator_kind;
  decision->architecture = DeviceArchitecture::kSm86;
  decision->kernel_variant = KernelVariant::kCudaNaive;
  return success_status();
}

}  // namespace raggedroute
