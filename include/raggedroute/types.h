#pragma once

namespace raggedroute {

enum class OperatorKind {
  kDenseGemm = 0,
  kTopKGate,
  kHistogram,
  kExclusiveScan,
  kTokenPermute,
  kGroupedGemm,
  kUnpermute,
};

// kOther is intentional: an architecture is not advertised as supported until
// a dedicated path has been implemented and validated on real hardware.
enum class DeviceArchitecture {
  kAuto = 0,
  kSm86,
  kOther,
};

enum class KernelVariant {
  kAuto = 0,
  kCudaNaive,
  kExperimental,
};

enum class ScalarType {
  kFloat32 = 0,
  kFloat16,
  kBfloat16,
};

enum class TensorLayout {
  kRowMajorContiguous = 0,
  kColumnMajorContiguous,
  kStrided,
};

enum class TopKNormalization {
  kSelectedSoftmax = 0,
};

enum class TopKTieBreak {
  kLowerExpertId = 0,
};

enum class TopKNaNPolicy {
  // NaN is ordered as -Inf. An all-NaN row returns ids {0, 1} and weights
  // {0.5, 0.5}; this is the fixed P0 contract.
  kNegativeInfinityAllNaNFallback01 = 0,
};

}  // namespace raggedroute
