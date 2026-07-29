#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {
namespace {

constexpr unsigned int kThreads = 256;

template <typename T>
__device__ float roundtrip_value(float value) {
  return static_cast<float>(T(value));
}

__global__ void dtype_roundtrip_kernel(ScalarType type, const float* input, float* output,
                                       std::size_t elements) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += stride) {
    switch (type) {
      case ScalarType::kFp32:
        output[index] = input[index];
        break;
      case ScalarType::kFp16:
        output[index] = roundtrip_value<__half>(input[index]);
        break;
      case ScalarType::kBf16:
        output[index] = roundtrip_value<__nv_bfloat16>(input[index]);
        break;
      case ScalarType::kFp8E4M3:
        output[index] = roundtrip_value<__nv_fp8_e4m3>(input[index]);
        break;
      case ScalarType::kFp8E5M2:
        output[index] = roundtrip_value<__nv_fp8_e5m2>(input[index]);
        break;
      case ScalarType::kFp6E2M3:
        output[index] = roundtrip_value<__nv_fp6_e2m3>(input[index]);
        break;
      case ScalarType::kFp6E3M2:
        output[index] = roundtrip_value<__nv_fp6_e3m2>(input[index]);
        break;
      case ScalarType::kFp4E2M1:
        output[index] = roundtrip_value<__nv_fp4_e2m1>(input[index]);
        break;
    }
  }
}

}  // namespace

bool CheckReport::ok() const { return status == CheckStatus::kPass && failures.empty(); }

void CheckReport::fail(CheckFailure failure) {
  status = CheckStatus::kFail;
  failures.push_back(std::move(failure));
}

void CheckReport::skip(std::string reason) {
  if (status != CheckStatus::kFail) status = CheckStatus::kSkip;
  skip_reason = std::move(reason);
}

std::string to_string(ScalarType value) {
  switch (value) {
    case ScalarType::kFp32:
      return "fp32";
    case ScalarType::kFp16:
      return "fp16";
    case ScalarType::kBf16:
      return "bf16";
    case ScalarType::kFp8E4M3:
      return "fp8_e4m3";
    case ScalarType::kFp8E5M2:
      return "fp8_e5m2";
    case ScalarType::kFp6E2M3:
      return "fp6_e2m3";
    case ScalarType::kFp6E3M2:
      return "fp6_e3m2";
    case ScalarType::kFp4E2M1:
      return "fp4_e2m1";
  }
  throw std::invalid_argument("unknown ScalarType");
}

std::string to_string(MathMode value) {
  switch (value) {
    case MathMode::kStrictFp32:
      return "strict_fp32";
    case MathMode::kTf32:
      return "tf32";
    case MathMode::kFp16AccFp32:
      return "fp16_acc_fp32";
    case MathMode::kBf16AccFp32:
      return "bf16_acc_fp32";
    case MathMode::kFp8AccFp32:
      return "fp8_acc_fp32";
  }
  throw std::invalid_argument("unknown MathMode");
}

std::string to_string(CapabilityLevel value) {
  switch (value) {
    case CapabilityLevel::kRuntimeVerified:
      return "runtime_verified";
    case CapabilityLevel::kCompileOnly:
      return "compile_only";
    case CapabilityLevel::kReferenceOnly:
      return "reference_only";
    case CapabilityLevel::kUnsupported:
      return "unsupported";
  }
  throw std::invalid_argument("unknown CapabilityLevel");
}

std::string to_string(CheckStatus value) {
  switch (value) {
    case CheckStatus::kPass:
      return "pass";
    case CheckStatus::kFail:
      return "fail";
    case CheckStatus::kSkip:
      return "skip";
  }
  throw std::invalid_argument("unknown CheckStatus");
}

ScalarType parse_scalar_type(const std::string& value) {
  if (value == "fp32") return ScalarType::kFp32;
  if (value == "fp16") return ScalarType::kFp16;
  if (value == "bf16") return ScalarType::kBf16;
  if (value == "fp8_e4m3") return ScalarType::kFp8E4M3;
  if (value == "fp8_e5m2") return ScalarType::kFp8E5M2;
  if (value == "fp6_e2m3") return ScalarType::kFp6E2M3;
  if (value == "fp6_e3m2") return ScalarType::kFp6E3M2;
  if (value == "fp4_e2m1") return ScalarType::kFp4E2M1;
  throw std::invalid_argument("unknown scalar type: " + value);
}

MathMode parse_math_mode(const std::string& value) {
  if (value == "strict_fp32") return MathMode::kStrictFp32;
  if (value == "tf32") return MathMode::kTf32;
  if (value == "fp16_acc_fp32") return MathMode::kFp16AccFp32;
  if (value == "bf16_acc_fp32") return MathMode::kBf16AccFp32;
  if (value == "fp8_acc_fp32") return MathMode::kFp8AccFp32;
  throw std::invalid_argument("unknown math mode: " + value);
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char* name) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(std::string(name) + " multiplication overflow");
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* name) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(std::string(name) + " addition overflow");
  }
  return lhs + rhs;
}

std::int32_t checked_int32(std::int64_t value, const char* name) {
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error(std::string(name) + " outside int32 range");
  }
  return static_cast<std::int32_t>(value);
}

DTypeTraits dtype_traits(ScalarType type) {
  DTypeTraits traits;
  traits.type = type;
  switch (type) {
    case ScalarType::kFp32:
      traits.logical_bits = traits.storage_unit_bits = 32;
      traits.max_finite = std::numeric_limits<float>::max();
      traits.epsilon_at_one = std::numeric_limits<float>::epsilon();
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kFp16:
      traits.logical_bits = traits.storage_unit_bits = 16;
      traits.max_finite = 65504.0;
      traits.epsilon_at_one = std::ldexp(1.0, -10);
      traits.minimum_native_compute_capability = 70;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kBf16:
      traits.logical_bits = traits.storage_unit_bits = 16;
      traits.max_finite = 3.3895313892515355e38;
      traits.epsilon_at_one = std::ldexp(1.0, -7);
      traits.minimum_native_compute_capability = 80;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kFp8E4M3:
      traits.logical_bits = traits.storage_unit_bits = 8;
      traits.supports_infinity = false;
      traits.saturates_finite = true;
      traits.rounding = RoundingPolicy::kSaturateFinite;
      traits.max_finite = 448.0;
      traits.epsilon_at_one = std::ldexp(1.0, -3);
      traits.minimum_native_compute_capability = 90;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kFp8E5M2:
      traits.logical_bits = traits.storage_unit_bits = 8;
      traits.saturates_finite = true;
      traits.rounding = RoundingPolicy::kSaturateFinite;
      traits.max_finite = 57344.0;
      traits.epsilon_at_one = std::ldexp(1.0, -2);
      traits.minimum_native_compute_capability = 90;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kFp6E2M3:
      traits.logical_bits = 6;
      traits.storage_unit_bits = 8;
      traits.supports_nan = traits.supports_infinity = false;
      traits.saturates_finite = true;
      traits.rounding = RoundingPolicy::kSaturateFinite;
      traits.max_finite = 7.5;
      traits.epsilon_at_one = std::ldexp(1.0, -3);
      traits.minimum_native_compute_capability = 100;
      traits.requires_arch_family_specific = true;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kFp6E3M2:
      traits.logical_bits = 6;
      traits.storage_unit_bits = 8;
      traits.supports_nan = traits.supports_infinity = false;
      traits.saturates_finite = true;
      traits.rounding = RoundingPolicy::kSaturateFinite;
      traits.max_finite = 28.0;
      traits.epsilon_at_one = std::ldexp(1.0, -2);
      traits.minimum_native_compute_capability = 100;
      traits.requires_arch_family_specific = true;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
    case ScalarType::kFp4E2M1:
      traits.logical_bits = 4;
      traits.storage_unit_bits = 8;
      traits.supports_nan = traits.supports_infinity = false;
      traits.saturates_finite = true;
      traits.rounding = RoundingPolicy::kSaturateFinite;
      traits.max_finite = 6.0;
      traits.epsilon_at_one = 0.5;
      traits.minimum_native_compute_capability = 100;
      traits.requires_arch_family_specific = true;
      traits.allowed_accumulators = {ScalarType::kFp32};
      break;
  }
  return traits;
}

CapabilityLevel dtype_capability(ScalarType type, const DeviceCapability& device) {
  const DTypeTraits traits = dtype_traits(type);
  const int capability = device.major * 10 + device.minor;
  const bool future = type == ScalarType::kFp8E4M3 || type == ScalarType::kFp8E5M2 ||
                      type == ScalarType::kFp6E2M3 || type == ScalarType::kFp6E3M2 ||
                      type == ScalarType::kFp4E2M1;
  if (capability < traits.minimum_native_compute_capability) {
    return future ? CapabilityLevel::kReferenceOnly : CapabilityLevel::kUnsupported;
  }
  if (future || (traits.requires_arch_family_specific && !device.architecture_family_specific)) {
    return CapabilityLevel::kCompileOnly;
  }
  return device.runtime_device_present ? CapabilityLevel::kRuntimeVerified
                                       : CapabilityLevel::kCompileOnly;
}

std::size_t logical_storage_bytes(ScalarType type, std::size_t elements) {
  const std::size_t bits = checked_mul(
      elements, static_cast<std::size_t>(dtype_traits(type).logical_bits), "logical dtype bits");
  return checked_add(bits, 7, "logical dtype rounding") / 8;
}

std::size_t wrapper_storage_bytes(ScalarType type, std::size_t elements) {
  return checked_mul(elements, static_cast<std::size_t>(dtype_traits(type).storage_unit_bits / 8),
                     "dtype wrapper bytes");
}

std::uint32_t encode_scalar(ScalarType type, double value) {
  const float input = static_cast<float>(value);
  switch (type) {
    case ScalarType::kFp32: {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &input, sizeof(bits));
      return bits;
    }
    case ScalarType::kFp16:
      return __half_raw(__float2half_rn(input)).x;
    case ScalarType::kBf16:
      return __nv_bfloat16_raw(__float2bfloat16_rn(input)).x;
    case ScalarType::kFp8E4M3:
      return __nv_fp8_e4m3(input).__x;
    case ScalarType::kFp8E5M2:
      return __nv_fp8_e5m2(input).__x;
    case ScalarType::kFp6E2M3:
      return __nv_fp6_e2m3(input).__x;
    case ScalarType::kFp6E3M2:
      return __nv_fp6_e3m2(input).__x;
    case ScalarType::kFp4E2M1:
      return __nv_fp4_e2m1(input).__x;
  }
  throw std::invalid_argument("unknown scalar type");
}

double decode_scalar(ScalarType type, std::uint32_t storage) {
  switch (type) {
    case ScalarType::kFp32: {
      float value = 0.0F;
      std::memcpy(&value, &storage, sizeof(value));
      return value;
    }
    case ScalarType::kFp16: {
      __half_raw raw{};
      raw.x = static_cast<unsigned short>(storage);
      return static_cast<float>(__half(raw));
    }
    case ScalarType::kBf16: {
      __nv_bfloat16_raw raw{};
      raw.x = static_cast<unsigned short>(storage);
      return static_cast<float>(__nv_bfloat16(raw));
    }
    case ScalarType::kFp8E4M3: {
      __nv_fp8_e4m3 value;
      value.__x = static_cast<__nv_fp8_storage_t>(storage);
      return static_cast<float>(value);
    }
    case ScalarType::kFp8E5M2: {
      __nv_fp8_e5m2 value;
      value.__x = static_cast<__nv_fp8_storage_t>(storage);
      return static_cast<float>(value);
    }
    case ScalarType::kFp6E2M3: {
      __nv_fp6_e2m3 value;
      value.__x = static_cast<__nv_fp6_storage_t>(storage);
      return static_cast<float>(value);
    }
    case ScalarType::kFp6E3M2: {
      __nv_fp6_e3m2 value;
      value.__x = static_cast<__nv_fp6_storage_t>(storage);
      return static_cast<float>(value);
    }
    case ScalarType::kFp4E2M1: {
      __nv_fp4_e2m1 value;
      value.__x = static_cast<__nv_fp4_storage_t>(storage);
      return static_cast<float>(value);
    }
  }
  throw std::invalid_argument("unknown scalar type");
}

float quantize_tf32(float value) {
  if (!std::isfinite(value) || value == 0.0F) return value;
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0U) return value;
  const std::uint32_t lsb = (bits >> 13U) & 1U;
  bits = (bits + 0x00000fffU + lsb) & 0xffffe000U;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<double> make_deterministic_master_data(std::size_t elements, std::uint64_t seed,
                                                   double lower, double upper) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
    throw std::invalid_argument("invalid deterministic master-data range");
  }
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> distribution(lower, upper);
  std::vector<double> values(elements);
  for (double& value : values) value = distribution(generator);
  return values;
}

std::vector<double> quantize_master_data(const std::vector<double>& master, ScalarType type) {
  std::vector<double> quantized(master.size());
  for (std::size_t index = 0; index < master.size(); ++index) {
    quantized[index] = decode_scalar(type, encode_scalar(type, master[index]));
  }
  return quantized;
}

cudaError_t launch_dtype_roundtrip(ScalarType type, const float* input, float* output,
                                   std::size_t elements, cudaStream_t stream) {
  if (elements == 0) return cudaSuccess;
  if (input == nullptr || output == nullptr) return cudaErrorInvalidValue;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  dtype_roundtrip_kernel<<<static_cast<unsigned int>(blocks > 65535 ? 65535 : blocks), kThreads, 0,
                           stream>>>(type, input, output, elements);
  return cudaGetLastError();
}

}  // namespace raggedroute::correctness
