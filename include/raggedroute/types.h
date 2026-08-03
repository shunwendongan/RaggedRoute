#pragma once

#include <array>
#include <cstdint>

namespace raggedroute {

enum class OperatorKind {
  kDenseGemm = 0,
  kTopKGate,
  kHistogram,
  kExclusiveScan,
  kTokenPermute,
  kGroupedGemm,
  kUnpermute,
  kHistogramExclusiveScan,
};

// kOther is intentional: an architecture is not advertised as supported until
// a dedicated path has been implemented and validated on real hardware.
enum class DeviceArchitecture {
  kAuto = 0,
  kSm86,
  kSm90,
  kOther,
};

// Stable implementation families exposed by the public operator API. Library
// baselines such as cuBLAS/CUTLASS/CUB remain benchmark-only variants.
enum class KernelFamily {
  kAuto = 0,
  kCudaNaive,
  kCudaOptimized,
};

struct KernelSelection {
  KernelFamily family = KernelFamily::kAuto;
  // Zero selects the family default. Non-zero ids are operator-local research
  // controls and are not portable between operators.
  std::uint32_t implementation_id = 0;
};

// Storage vocabulary shared by the runtime and correctness framework. An enum
// value describes representation only; runtime support is a separate dispatch
// and capability decision.
enum class ScalarType {
  kFp32 = 0,
  kFp16,
  kBf16,
  kFp8E4M3,
  kFp8E5M2,
  kFp6E2M3,
  kFp6E3M2,
  kFp4E2M1,
};

enum class TensorLayout {
  kRowMajorContiguous = 0,
  kColumnMajorContiguous,
  kStrided,
};

// Strides are expressed in elements in logical dimension order. All-zero
// strides mean canonical contiguous strides inferred from the operator shape.
// v0.2 executes only all-zero, row-major-contiguous specs, but preserving the
// fields now prevents dtype work from forcing another public API rewrite.
struct TensorSpec {
  ScalarType dtype = ScalarType::kFp32;
  TensorLayout layout = TensorLayout::kRowMajorContiguous;
  std::array<std::int64_t, 3> strides = {0, 0, 0};
};

struct ConstTensorView {
  const void* data = nullptr;
  TensorSpec spec{};
};

struct MutableTensorView {
  void* data = nullptr;
  TensorSpec spec{};
};

enum class TopKNormalization {
  kSelectedSoftmax = 0,
};

enum class TopKTieBreak {
  kLowerExpertId = 0,
};

enum class TopKNaNPolicy {
  // NaN is ordered as -Inf. An all-NaN row returns ids {0, 1} and weights
  // {0.5, 0.5}; this is the fixed v0.2 contract.
  kNegativeInfinityAllNaNFallback01 = 0,
};

}  // namespace raggedroute
