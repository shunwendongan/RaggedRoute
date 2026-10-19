#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/tensorcore_gemm.h"
#include "raggedroute/benchmark/tensorcore_adapter.h"
#include "raggedroute/correctness/framework.h"
#include "raggedroute/correctness/guarded_buffer.h"

namespace rr = raggedroute;
namespace rc = raggedroute::correctness;
namespace rb = raggedroute::benchmark;

namespace {
std::size_t case_count = 0;

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

void require_report(const rc::CheckReport& report, const std::string& label) {
  require(report.ok(), label + (report.failures.empty() ? ": comparison failed"
                                                      : ": " + report.failures.front().message));
}

struct Inputs {
  std::vector<std::uint16_t> bits;
  std::vector<double> values;
};

Inputs make_input(std::size_t count, rr::ScalarType dtype, const std::string& pattern,
                  std::uint64_t seed) {
  auto master = rc::make_deterministic_master_data(count, seed);
  Inputs input;
  input.bits.resize(count);
  input.values.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    // Use exactly representable finite scales for the extreme-value cases.
    if (pattern == "zero") master[i] = 0.0;
    if (pattern == "cancellation") master[i] = (i % 2 == 0 ? 1.0 : -1.0) * (1.0 + (i % 7) / 8.0);
    if (pattern == "finite_extremes") master[i] = (i % 2 == 0 ? 64.0 : 1.0 / 1024.0) * (i % 3 == 0 ? -1.0 : 1.0);
    input.bits[i] = static_cast<std::uint16_t>(rc::encode_scalar(dtype, master[i]));
    input.values[i] = rc::decode_scalar(dtype, input.bits[i]);
  }
  return input;
}

rc::CaseDescriptor descriptor(rr::ScalarType dtype, const std::string& variant,
                              const std::string& pattern) {
  rc::CaseDescriptor result;
  result.case_id = "tensorcore_" + std::to_string(case_count);
  result.operator_name = "tensorcore_gemm";
  result.variant_name = variant;
  result.data_pattern = pattern;
  result.tensor_types.input = dtype;
  result.tensor_types.accumulator = result.tensor_types.output = rr::ScalarType::kFp32;
  result.tensor_types.math_mode = dtype == rr::ScalarType::kFp16 ? rc::MathMode::kFp16AccFp32
                                                               : rc::MathMode::kBf16AccFp32;
  return result;
}

void validate_observation(const rc::LaunchObservation& observation, const std::string& label) {
  require(observation.launch_error == cudaSuccess,
          label + " immediate error: " + cudaGetErrorString(observation.launch_error));
  require(observation.execution_error == cudaSuccess,
          label + " asynchronous error: " + cudaGetErrorString(observation.execution_error));
}

void dense_case(int m, int n, int k, rr::ScalarType dtype, const rr::ops::TensorCoreConfig& config,
                const std::string& pattern, bool simt, bool unaligned, cudaStream_t stream) {
  const std::size_t prefix = unaligned ? 1 : 0;
  const Inputs a = make_input(static_cast<std::size_t>(m) * k, dtype, pattern, 12345);
  const Inputs b = make_input(static_cast<std::size_t>(k) * n, dtype, pattern, 54321);
  auto a_bits = a.bits, b_bits = b.bits;
  a_bits.insert(a_bits.begin(), prefix, 0x1234);
  b_bits.insert(b_bits.begin(), prefix, 0x2345);
  rc::GuardedDeviceBuffer<std::uint16_t> da(a_bits.size(), stream), db(b_bits.size(), stream);
  rc::GuardedDeviceBuffer<float> dc(static_cast<std::size_t>(m) * n, stream);
  da.copy_from_host(a_bits, stream);
  db.copy_from_host(b_bits, stream);
  const auto before_a = rc::hash_vector(a_bits), before_b = rc::hash_vector(b_bits);
  const std::uint16_t* ap = da.data() == nullptr ? nullptr : da.data() + prefix;
  const std::uint16_t* bp = db.data() == nullptr ? nullptr : db.data() + prefix;
  const auto observation = rc::observe_launch([&] {
    return simt ? rr::ops::launch_dense_gemm_lowp_simt(ap, bp, dc.data(), m, n, k, dtype, stream)
        : rr::ops::launch_dense_gemm_tensorcore(ap, bp, dc.data(), m, n, k, dtype, config, stream);
  }, stream);
  const std::string label = "dense " + rc::to_string(dtype) + " " + std::to_string(m) + "x" +
      std::to_string(n) + "x" + std::to_string(k) + " " + pattern + (simt ? " SIMT" : " MMA");
  validate_observation(observation, label);
  const auto host = dc.copy_to_host(stream);
  const std::vector<double> actual(host.begin(), host.end());
  const auto expected = rc::dense_gemm_reference(a.values, b.values, {}, m, n, k, 1.0, 0.0);
  require_report(rc::compare_gemm(descriptor(dtype, simt ? "simt" : "mma", pattern), actual,
      expected, a.values, b.values, m, n, k, rr::ScalarType::kFp32), label);
  require(da.canaries_intact(stream) && db.canaries_intact(stream) && dc.canaries_intact(stream),
          label + " redzone corruption");
  require(da.payload_hash(stream) == before_a && db.payload_hash(stream) == before_b,
          label + " modified input");
  ++case_count;
}

void grouped_case(const std::vector<std::int32_t>& counts, int k, int n, rr::ScalarType dtype,
                  const rr::ops::TensorCoreConfig& config, const std::string& pattern,
                  bool simt, bool unaligned, cudaStream_t stream) {
  const int experts = static_cast<int>(counts.size());
  std::vector<std::int32_t> offsets(counts.size() + 1, 0);
  for (std::size_t e = 0; e < counts.size(); ++e) offsets[e + 1] = offsets[e] + counts[e];
  const int m = offsets.back();
  const int bound = *std::max_element(counts.begin(), counts.end()) + 33;
  const std::size_t prefix = unaligned ? 1 : 0;
  const Inputs a = make_input(static_cast<std::size_t>(m) * k, dtype, pattern, 12345);
  const Inputs b = make_input(static_cast<std::size_t>(experts) * k * n, dtype, pattern, 54321);
  auto a_bits = a.bits, b_bits = b.bits;
  a_bits.insert(a_bits.begin(), prefix, 0x1234);
  b_bits.insert(b_bits.begin(), prefix, 0x2345);
  // Device offsets determine whether any row exists. A positive conservative
  // launch envelope still needs nonnull payload addresses at the host boundary.
  if (m == 0 && k > 0 && a_bits.empty()) a_bits.push_back(0x1234);
  rc::GuardedDeviceBuffer<std::uint16_t> da(a_bits.size(), stream), db(b_bits.size(), stream);
  rc::GuardedDeviceBuffer<std::int32_t> device_offsets(offsets.size(), stream);
  rc::GuardedDeviceBuffer<float> dc(std::max<std::size_t>(1, static_cast<std::size_t>(m) * n), stream);
  da.copy_from_host(a_bits, stream);
  db.copy_from_host(b_bits, stream);
  device_offsets.copy_from_host(offsets, stream);
  const auto a_hash = rc::hash_vector(a_bits), b_hash = rc::hash_vector(b_bits);
  const auto offset_hash = rc::hash_vector(offsets);
  const auto empty_output_hash = (m == 0 || n == 0) ? dc.payload_hash(stream) : 0;
  const auto* ap = da.data() == nullptr ? nullptr : da.data() + prefix;
  const auto* bp = db.data() == nullptr ? nullptr : db.data() + prefix;
  const auto observation = rc::observe_launch([&] {
    return simt ? rr::ops::launch_grouped_gemm_lowp_simt(ap, bp, device_offsets.data(), dc.data(),
        experts, k, n, bound, dtype, stream)
      : rr::ops::launch_grouped_gemm_tensorcore(ap, bp, device_offsets.data(), dc.data(),
        experts, k, n, bound, dtype, config, stream);
  }, stream);
  const std::string label = "grouped " + rc::to_string(dtype) + " E" + std::to_string(experts) +
      " R" + std::to_string(m) + " K" + std::to_string(k) + " N" + std::to_string(n) +
      " " + pattern + (simt ? " SIMT" : " MMA");
  validate_observation(observation, label);
  auto host = dc.copy_to_host(stream);
  if (m == 0 || n == 0) {
    require(rc::hash_vector(host) == empty_output_hash, label + " empty output touched");
  }
  host.resize(static_cast<std::size_t>(m) * n);
  const std::vector<double> actual(host.begin(), host.end());
  const auto expected = rc::grouped_gemm_reference(a.values, b.values, offsets, experts, k, n);
  for (int e = 0; e < experts; ++e) {
    const std::size_t begin = static_cast<std::size_t>(offsets[e]);
    const std::size_t end = static_cast<std::size_t>(offsets[e + 1]);
    if (begin == end) continue;
    require_report(rc::compare_gemm(descriptor(dtype, simt ? "simt" : "mma", pattern),
        std::vector<double>(actual.begin() + begin * n, actual.begin() + end * n),
        std::vector<double>(expected.begin() + begin * n, expected.begin() + end * n),
        std::vector<double>(a.values.begin() + begin * k, a.values.begin() + end * k),
        std::vector<double>(b.values.begin() + static_cast<std::size_t>(e) * k * n,
                            b.values.begin() + static_cast<std::size_t>(e + 1) * k * n),
        static_cast<int>(end - begin), n, k, rr::ScalarType::kFp32), label);
  }
  require(da.canaries_intact(stream) && db.canaries_intact(stream) && dc.canaries_intact(stream) &&
          device_offsets.canaries_intact(stream), label + " redzone corruption");
  require(da.payload_hash(stream) == a_hash && db.payload_hash(stream) == b_hash &&
          device_offsets.payload_hash(stream) == offset_hash, label + " modified input/offsets");
  ++case_count;
}

void quantize_case(rr::ScalarType dtype, cudaStream_t stream) {
  std::vector<float> values = {0.0F, -0.0F, 1.0F, -1.0F, 1.0F + 1.0F/1024.0F,
      1.0F + 1.0F/2048.0F, 1.0F + 3.0F/2048.0F, -17.1234F, 0.0001F, 64.0F, 0.3333333F};
  rc::GuardedDeviceBuffer<float> source(values.size(), stream);
  rc::GuardedDeviceBuffer<std::uint16_t> target(values.size(), stream);
  source.copy_from_host(values, stream);
  validate_observation(rc::observe_launch([&] {
    return rr::ops::launch_gemm_quantize(source.data(), target.data(), values.size(), dtype, stream);
  }, stream), "GPU input conversion");
  const auto result = target.copy_to_host(stream);
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(result[i] == rc::encode_scalar(dtype, values[i]), "GPU quantize disagrees with CPU RNE");
  }
  require(source.payload_hash(stream) == rc::hash_vector(values) && source.canaries_intact(stream) &&
          target.canaries_intact(stream), "quantize input/redzone changed");
  validate_observation(rc::observe_launch([&] {
    return rr::ops::launch_gemm_quantize(nullptr, nullptr, 0, dtype, stream);
  }, stream), "empty conversion");
  ++case_count;
}

void adapter_cases(rr::ScalarType dtype, cudaStream_t stream) {
  const auto dtype_text = rc::to_string(dtype);
  for (const std::string op : {"dense_gemm_tc", "grouped_gemm_tc"}) {
    // Exercise the actual registered variants, including conversion and library boundaries.
    for (const auto& item : rb::tensorcore_gemm_variant_descriptors(op)) {
      rb::OptionMap options = {{"dtype", dtype_text}, {"N", "32"}, {"K", "32"}};
      if (op == "dense_gemm_tc") options["M"] = "32";
      else {
        options["T"] = "32";
        options["E"] = "4";
        options["top_k"] = "2";
        options["distribution"] = "single_hot";
        options["max_expert_tokens"] = "64";
      }
      auto adapter = rb::make_tensorcore_gemm_adapter(op, item.name);
      adapter->setup(options, 12345, stream);
      adapter->prepare_sample(rb::MeasurementLevel::kOperatorSteady, stream);
      adapter->enqueue(rb::MeasurementLevel::kOperatorSteady, stream);
      rc::cuda_check(cudaStreamSynchronize(stream), "adapter execution");
      const auto result = adapter->validate(stream);
      require(result.ok, op + "/" + item.name + ": " + result.message);
      const auto metadata = adapter->variant_config();
      for (const std::string field : {"implementation_category", "implementation_version",
          "implementation_revision", "dependency_revision", "algorithm_id", "math_mode",
          "actual_path", "quantized_reference_normalized_l2_error",
          "original_fp32_reference_normalized_l2_error"}) {
        require(metadata.count(field) != 0, "adapter metadata missing " + field);
      }
      if (item.name != "fp32_parent") {
        require(std::get<std::string>(metadata.at("math_mode")) == dtype_text + "_acc_fp32",
                "adapter precision contract mismatch");
      }
      ++case_count;
    }
  }
  for (const std::string distribution : {"uniform", "zipf"}) {
    for (const std::string variant : {"cuda_mma_sync", "cuda_mma_async"}) {
      auto adapter = rb::make_tensorcore_gemm_adapter("grouped_gemm_tc", variant);
      adapter->setup({{"dtype", dtype_text}, {"T", "127"}, {"E", "64"}, {"top_k", "2"},
          {"K", "33"}, {"N", "17"}, {"distribution", distribution}, {"zipf_s", "1.4"},
          {"max_expert_tokens", "254"}}, 12345, stream);
      adapter->enqueue(rb::MeasurementLevel::kOperatorSteady, stream);
      rc::cuda_check(cudaStreamSynchronize(stream), "skewed grouped adapter execution");
      const auto result = adapter->validate(stream);
      require(result.ok, distribution + " " + variant + ": " + result.message);
      ++case_count;
    }
  }
}

void run(bool quick, cudaStream_t stream) {
  const std::vector<std::array<int, 3>> dense_shapes = quick
      ? std::vector<std::array<int, 3>>{{64, 64, 64}, {17, 15, 33}}
      : std::vector<std::array<int, 3>>{{1, 1, 1}, {15, 16, 17}, {16, 17, 15}, {17, 15, 16},
          {31, 32, 33}, {32, 33, 31}, {33, 31, 32}, {63, 64, 65}, {64, 65, 63},
          {65, 63, 64}, {128, 128, 128}, {0, 17, 17}, {17, 0, 17}, {17, 17, 0}};
  const std::vector<std::vector<std::int32_t>> grouped_counts = quick
      ? std::vector<std::vector<std::int32_t>>{{32, 0, 32, 0}, {0, 1, 15, 16, 17, 0}}
      : std::vector<std::vector<std::int32_t>>{{0, 1, 15, 16, 17, 0}, {31, 32, 33},
          {63, 64, 65}, {65}, {0, 0, 0, 0}, {32, 0, 32, 0}};
  for (rr::ScalarType dtype : {rr::ScalarType::kFp16, rr::ScalarType::kBf16}) {
    quantize_case(dtype, stream);
    for (const int tn : {32, 64}) for (const int tk : {32, 64}) for (const bool async : {false, true}) {
      rr::ops::TensorCoreConfig config;
      config.block_m = 64;
      config.block_n = tn;
      config.block_k = tk;
      config.asynchronous = async;
      for (const auto& s : dense_shapes) dense_case(s[0], s[1], s[2], dtype, config, "random", false,
          s[0] % 2 != 0, stream);
      if (!quick) for (const std::string pattern : {"zero", "cancellation", "finite_extremes"}) {
        dense_case(33, 65, 63, dtype, config, pattern, false, false, stream);
      }
    }
    for (const int tm : {16, 32}) for (const int tk : {32, 64}) for (const bool async : {false, true}) {
      rr::ops::TensorCoreConfig config;
      config.block_m = tm;
      config.block_n = 64;
      config.block_k = tk;
      config.asynchronous = async;
      for (std::size_t i = 0; i < grouped_counts.size(); ++i) {
        grouped_case(grouped_counts[i], i % 2 == 0 ? 17 : 64, i % 2 == 0 ? 65 : 64, dtype,
            config, "random", false, i % 2 == 0, stream);
      }
      if (!quick) {
        std::vector<std::int32_t> many_empty(64, 0);
        many_empty[37] = 65;
        grouped_case(many_empty, 33, 17, dtype, config, "random", false, true, stream);
        grouped_case({1, 0, 17, 32}, 0, 17, dtype, config, "zero", false, false, stream);
        grouped_case({1, 0, 17, 32}, 17, 0, dtype, config, "zero", false, false, stream);
        for (const std::string pattern : {"zero", "cancellation", "finite_extremes"}) {
          grouped_case({15, 0, 16, 17}, 33, 17, dtype, config, pattern, false, false, stream);
        }
      }
    }
    rr::ops::TensorCoreConfig unused;
    dense_case(17, 33, 0, dtype, unused, "zero", false, false, stream);
    dense_case(17, 33, 65, dtype, unused, "random", true, true, stream);
    dense_case(17, 33, 0, dtype, unused, "zero", true, false, stream);
    grouped_case({15, 0, 17}, 33, 17, dtype, unused, "random", true, true, stream);
    rr::ops::TensorCoreConfig grouped_config;
    grouped_config.block_m = 32;
    grouped_config.block_n = 64;
    grouped_case({15, 0, 17}, 0, 17, dtype, grouped_config, "zero", false, false, stream);
    grouped_case({0, 0, 0, 0}, 33, 17, dtype, grouped_config, "zero", false, false, stream);
    validate_observation(rc::observe_launch([&] {
      return rr::ops::launch_grouped_gemm_tensorcore(nullptr, nullptr, nullptr, nullptr,
          0, 33, 17, 0, dtype, grouped_config, stream);
    }, stream), "zero experts no-op");
    adapter_cases(dtype, stream);
  }
}
}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick" || arg == "--sanitizer") quick = true;
    else { std::cerr << "Unknown argument: " << arg << '\n'; return 2; }
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
  }
  cudaStream_t stream = nullptr;
  try {
    rc::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create non-default stream");
    run(quick, stream);
    rc::cuda_check(cudaStreamDestroy(stream), "destroy stream");
    std::cout << "PASS Tensor Core GEMM " << case_count << " cases (FP16/BF16, non-default stream)\n";
    return 0;
  } catch (const std::exception& error) {
    if (stream != nullptr) cudaStreamDestroy(stream);
    std::cerr << "FAIL after " << case_count << " cases: " << error.what() << '\n';
    return 1;
  }
}
