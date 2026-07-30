#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "raggedroute/types.h"

namespace raggedroute {

// Per-call resources supplied by the caller. Workspace allocation and device
// discovery belong outside steady-state timing.
struct RuntimeContext {
  cudaStream_t stream = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  DeviceArchitecture architecture = DeviceArchitecture::kAuto;
};

}  // namespace raggedroute
