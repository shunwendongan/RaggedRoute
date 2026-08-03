#include "raggedroute/dispatch.h"

#include <cuda_runtime_api.h>

#include "../dense_gemm/cuda_candidate/optimized_internal.h"
#include "../topk_gate/cuda_candidate/optimized_internal.h"
#include "../histogram/cuda_candidate/optimized_internal.h"
#include "../permute/cuda_candidate/optimized_internal.h"
#include "operator_internal.h"

namespace raggedroute {
namespace {

bool has_zero_strides(const TensorSpec& spec) noexcept {
  return spec.strides[0] == 0 && spec.strides[1] == 0 && spec.strides[2] == 0;
}

bool has_p02_layout(const std::optional<TensorSpec>& spec) noexcept {
  return !spec || (spec->layout == TensorLayout::kRowMajorContiguous && has_zero_strides(*spec));
}

bool is_v2_storage_type(ScalarType type) noexcept {
  return type == ScalarType::kFp32 || type == ScalarType::kFp16 || type == ScalarType::kBf16;
}

bool has_roles(const OperatorSignature& signature, bool input, bool weight, bool accumulator,
               bool output) noexcept {
  return signature.input.has_value() == input && signature.weight.has_value() == weight &&
         signature.accumulator.has_value() == accumulator && signature.output.has_value() == output;
}

bool is_valid_dtype_signature(OperatorKind kind, const OperatorSignature& signature) noexcept {
  switch (kind) {
    case OperatorKind::kDenseGemm:
    case OperatorKind::kGroupedGemm: {
      if (!has_roles(signature, true, true, true, true)) return false;
      const ScalarType input = signature.input->dtype;
      return is_v2_storage_type(input) && signature.weight->dtype == input &&
             *signature.accumulator == ScalarType::kFp32 &&
             (signature.output->dtype == input || signature.output->dtype == ScalarType::kFp32);
    }
    case OperatorKind::kTopKGate:
      return has_roles(signature, true, false, true, true) &&
             signature.input->dtype == ScalarType::kFp32 &&
             *signature.accumulator == ScalarType::kFp32 &&
             signature.output->dtype == ScalarType::kFp32;
    case OperatorKind::kHistogram:
    case OperatorKind::kExclusiveScan:
      return has_roles(signature, false, false, false, false);
    case OperatorKind::kTokenPermute:
      return has_roles(signature, true, false, false, true) &&
             is_v2_storage_type(signature.input->dtype) &&
             signature.output->dtype == signature.input->dtype;
    case OperatorKind::kUnpermute: {
      if (!has_roles(signature, true, true, true, true)) return false;
      const ScalarType input = signature.input->dtype;
      return is_v2_storage_type(input) && signature.weight->dtype == ScalarType::kFp32 &&
             *signature.accumulator == ScalarType::kFp32 &&
             (signature.output->dtype == input || signature.output->dtype == ScalarType::kFp32);
    }
  }
  return false;
}

bool is_all_fp32(const OperatorSignature& signature) noexcept {
  return (!signature.input || signature.input->dtype == ScalarType::kFp32) &&
         (!signature.weight || signature.weight->dtype == ScalarType::kFp32) &&
         (!signature.accumulator || *signature.accumulator == ScalarType::kFp32) &&
         (!signature.output || signature.output->dtype == ScalarType::kFp32);
}

}  // namespace

DeviceArchitecture classify_compute_capability(int major, int minor) noexcept {
  if (major == 8 && minor == 6) return DeviceArchitecture::kSm86;
  if (major == 9 && minor == 0) return DeviceArchitecture::kSm90;
  return DeviceArchitecture::kOther;
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
  const OperatorSignature& signature = request.signature;
  bool roles_valid = false;
  switch (request.operator_kind) {
    case OperatorKind::kDenseGemm:
    case OperatorKind::kGroupedGemm:
    case OperatorKind::kUnpermute:
      roles_valid = has_roles(signature, true, true, true, true);
      break;
    case OperatorKind::kTopKGate:
      roles_valid = has_roles(signature, true, false, true, true);
      break;
    case OperatorKind::kTokenPermute:
      roles_valid = has_roles(signature, true, false, false, true);
      break;
    case OperatorKind::kHistogram:
    case OperatorKind::kExclusiveScan:
      roles_valid = has_roles(signature, false, false, false, false);
      break;
  }
  if (!roles_valid) {
    return detail::invalid_argument("dispatch signature roles do not match the operator");
  }
  if (!has_p02_layout(signature.input) || !has_p02_layout(signature.weight) ||
      !has_p02_layout(signature.output)) {
    return detail::make_status(StatusCode::kUnsupportedLayout,
                               "v0.2 dispatch requires zero-stride contiguous row-major tensors");
  }
  if (!is_valid_dtype_signature(request.operator_kind, signature) || !is_all_fp32(signature)) {
    return detail::make_status(StatusCode::kUnsupportedDataType,
                               "the requested dtype signature has no v0.2 kernel");
  }
  if (request.architecture == DeviceArchitecture::kAuto) {
    return detail::invalid_argument("select_kernel requires a resolved architecture");
  }
  if (request.architecture != DeviceArchitecture::kSm86) {
    return detail::make_status(StatusCode::kUnsupportedArchitecture,
                               "v0.2 kernels are validated for SM86 only");
  }
  const KernelSelection requested = request.requested_kernel;
  if (requested.family == KernelFamily::kAuto && requested.implementation_id != 0) {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "automatic kernel selection requires implementation_id=0");
  }
  if (requested.family == KernelFamily::kCudaNaive && requested.implementation_id != 0) {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "the naive family has no non-default implementation");
  }
  if (requested.family == KernelFamily::kCudaOptimized) {
    const bool implemented =
        (request.operator_kind == OperatorKind::kDenseGemm &&
         ops::is_dense_gemm_optimized_implementation(requested.implementation_id)) ||
        (request.operator_kind == OperatorKind::kTopKGate &&
         ops::is_topk_gate_optimized_implementation(requested.implementation_id)) ||
        (request.operator_kind == OperatorKind::kHistogram &&
         ops::is_histogram_optimized_implementation(requested.implementation_id)) ||
        (request.operator_kind == OperatorKind::kTokenPermute &&
         ops::is_token_permute_optimized_implementation(requested.implementation_id));
    if (!implemented) {
      return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                                 "requested optimized kernel is not implemented");
    }
    decision->operator_kind = request.operator_kind;
    decision->architecture = DeviceArchitecture::kSm86;
    decision->kernel = requested;
    return success_status();
  }
  if (requested.family != KernelFamily::kAuto && requested.family != KernelFamily::kCudaNaive) {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "requested kernel family is not implemented");
  }

  decision->operator_kind = request.operator_kind;
  decision->architecture = DeviceArchitecture::kSm86;
  decision->kernel =
      requested.family == KernelFamily::kAuto && request.operator_kind == OperatorKind::kHistogram
          ? KernelSelection{KernelFamily::kCudaOptimized, ops::kHistogramCandidateImplementation}
          : KernelSelection{KernelFamily::kCudaNaive, 0};
  return success_status();
}

}  // namespace raggedroute
