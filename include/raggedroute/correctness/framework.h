#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace raggedroute::correctness {

enum class ScalarType {
  kFp32,
  kFp16,
  kBf16,
  kFp8E4M3,
  kFp8E5M2,
  kFp6E2M3,
  kFp6E3M2,
  kFp4E2M1,
};

enum class MathMode {
  kStrictFp32,
  kTf32,
  kFp16AccFp32,
  kBf16AccFp32,
  kFp8AccFp32,
};

enum class CapabilityLevel {
  kRuntimeVerified,
  kCompileOnly,
  kReferenceOnly,
  kUnsupported,
};

enum class ScaleMode { kNone, kPerTensorFp32, kPerBlockFp32 };
enum class RoundingPolicy { kRoundToNearestEven, kSaturateFinite };
enum class Determinism { kRequired, kSemanticOnly };
enum class CheckStatus { kPass, kFail, kSkip };

struct QuantizationSpec {
  ScaleMode scale_mode = ScaleMode::kNone;
  std::size_t block_elements = 0;
};

struct TensorTypeSpec {
  ScalarType input = ScalarType::kFp32;
  ScalarType accumulator = ScalarType::kFp32;
  ScalarType output = ScalarType::kFp32;
  MathMode math_mode = MathMode::kStrictFp32;
  QuantizationSpec quantization;
};

struct DTypeTraits {
  ScalarType type = ScalarType::kFp32;
  int logical_bits = 32;
  int storage_unit_bits = 32;
  bool supports_nan = true;
  bool supports_infinity = true;
  bool saturates_finite = false;
  RoundingPolicy rounding = RoundingPolicy::kRoundToNearestEven;
  double max_finite = 0.0;
  double epsilon_at_one = 0.0;
  int minimum_native_compute_capability = 0;
  bool requires_arch_family_specific = false;
  std::vector<ScalarType> allowed_accumulators;
};

struct DeviceCapability {
  int major = 0;
  int minor = 0;
  bool architecture_family_specific = false;
  bool runtime_device_present = false;
};

struct CaseDescriptor {
  std::string case_id;
  std::string operator_name;
  std::string variant_name = "cuda_naive";
  std::map<std::string, std::int64_t> shape;
  TensorTypeSpec tensor_types;
  std::string data_pattern = "random";
  std::string route_distribution = "uniform";
  Determinism determinism = Determinism::kRequired;
  std::string device_arch = "unknown";
  std::uint64_t seed = 20260729ULL;
};

struct CheckFailure {
  std::string check;
  std::string message;
  std::optional<std::size_t> linear_index;
  std::optional<double> actual;
  std::optional<double> expected;
  std::optional<double> allowed_error;
};

struct NumericSummary {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double normalized_l2_error = 0.0;
  std::optional<std::size_t> worst_index;
};

struct CheckReport {
  CheckStatus status = CheckStatus::kPass;
  std::string case_id;
  std::string operator_name;
  std::string variant_name;
  cudaError_t launch_error = cudaSuccess;
  cudaError_t execution_error = cudaSuccess;
  bool canaries_ok = true;
  bool inputs_unchanged = true;
  NumericSummary numeric;
  std::vector<CheckFailure> failures;
  std::string skip_reason;

  bool ok() const;
  void fail(CheckFailure failure);
  void skip(std::string reason);
};

struct Top2Reference {
  std::vector<std::int32_t> ids;
  std::vector<double> weights;
};

std::string to_string(ScalarType value);
std::string to_string(MathMode value);
std::string to_string(CapabilityLevel value);
std::string to_string(CheckStatus value);
ScalarType parse_scalar_type(const std::string& value);
MathMode parse_math_mode(const std::string& value);

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char* name);
std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* name);
std::int32_t checked_int32(std::int64_t value, const char* name);

DTypeTraits dtype_traits(ScalarType type);
CapabilityLevel dtype_capability(ScalarType type, const DeviceCapability& device);
std::size_t logical_storage_bytes(ScalarType type, std::size_t elements);
std::size_t wrapper_storage_bytes(ScalarType type, std::size_t elements);
std::uint32_t encode_scalar(ScalarType type, double value);
double decode_scalar(ScalarType type, std::uint32_t storage);
float quantize_tf32(float value);
std::vector<double> make_deterministic_master_data(std::size_t elements, std::uint64_t seed,
                                                   double lower = -1.0, double upper = 1.0);
std::vector<double> quantize_master_data(const std::vector<double>& master, ScalarType type);

cudaError_t launch_dtype_roundtrip(ScalarType type, const float* input, float* output,
                                   std::size_t elements, cudaStream_t stream);

std::vector<double> dense_gemm_reference(const std::vector<double>& a, const std::vector<double>& b,
                                         const std::vector<double>& c, int m, int n, int k,
                                         double alpha, double beta);
Top2Reference top2_selected_softmax_reference(const std::vector<double>& logits, int tokens,
                                              int experts);
std::vector<std::int32_t> histogram_reference(const std::vector<std::int32_t>& ids, int experts);
std::vector<std::int32_t> exclusive_scan_reference(const std::vector<std::int32_t>& counts);
std::vector<double> grouped_gemm_reference(const std::vector<double>& x_permuted,
                                           const std::vector<double>& expert_weights,
                                           const std::vector<std::int32_t>& offsets, int experts,
                                           int hidden, int output);
std::vector<double> unpermute_reference(const std::vector<double>& y_permuted,
                                        const std::vector<std::int32_t>& route_pos,
                                        const std::vector<double>& route_weights, int tokens,
                                        int top_k, int output);

CheckReport compare_floating(const CaseDescriptor& descriptor, const std::vector<double>& actual,
                             const std::vector<double>& expected, double absolute_tolerance,
                             double relative_tolerance);
CheckReport compare_gemm(const CaseDescriptor& descriptor, const std::vector<double>& actual,
                         const std::vector<double>& expected, const std::vector<double>& a,
                         const std::vector<double>& b, int m, int n, int k, ScalarType accumulator);
CheckReport validate_histogram(const CaseDescriptor& descriptor,
                               const std::vector<std::int32_t>& ids,
                               const std::vector<std::int32_t>& counts, int experts);
CheckReport validate_scan(const CaseDescriptor& descriptor, const std::vector<std::int32_t>& counts,
                          const std::vector<std::int32_t>& offsets);
CheckReport validate_permute(const CaseDescriptor& descriptor, const std::vector<double>& x,
                             const std::vector<std::int32_t>& ids,
                             const std::vector<std::int32_t>& offsets,
                             const std::vector<double>& x_permuted,
                             const std::vector<std::int32_t>& route_pos,
                             const std::vector<std::int32_t>* sorted_route, int tokens, int top_k,
                             int hidden);

std::string case_to_json(const CaseDescriptor& descriptor);
CaseDescriptor case_from_json(const std::string& json);
void save_case_json(const std::string& path, const CaseDescriptor& descriptor);
CaseDescriptor load_case_json(const std::string& path);
std::string report_to_json(const CheckReport& report, const CaseDescriptor& descriptor);
void save_failure_artifact(const std::string& path, const CheckReport& report,
                           const CaseDescriptor& descriptor);

}  // namespace raggedroute::correctness
