#pragma once

#include <cuda_runtime_api.h>

namespace raggedroute {

enum class StatusCode {
  kSuccess = 0,
  kInvalidArgument,
  kUnsupportedArchitecture,
  kUnsupportedDataType,
  kUnsupportedLayout,
  kUnsupportedKernelVariant,
  kInsufficientWorkspace,
  kCudaError,
};

// Status deliberately owns no memory. Operator hot paths can return it without
// allocating, throwing, or synchronizing the caller's CUDA stream.
struct Status {
  StatusCode code = StatusCode::kSuccess;
  cudaError_t cuda_error = cudaSuccess;
  const char* message = "success";

  constexpr bool ok() const noexcept { return code == StatusCode::kSuccess; }
  constexpr explicit operator bool() const noexcept { return ok(); }
};

constexpr Status success_status() noexcept { return {}; }

}  // namespace raggedroute
