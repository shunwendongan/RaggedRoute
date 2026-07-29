#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/correctness/framework.h"
#include "raggedroute/correctness/guarded_buffer.h"

namespace rc = raggedroute::correctness;
namespace ops = raggedroute::ops;

namespace {

rc::CaseDescriptor make_case(std::string id, std::string op, std::uint64_t seed = 20260729ULL) {
  rc::CaseDescriptor descriptor;
  descriptor.case_id = std::move(id);
  descriptor.operator_name = std::move(op);
  descriptor.seed = seed;
  return descriptor;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> as_double(const std::vector<float>& values) {
  std::vector<double> output(values.size());
  std::transform(values.begin(), values.end(), output.begin(),
                 [](float value) { return static_cast<double>(value); });
  return output;
}

std::vector<float> as_float(const std::vector<double>& values) {
  std::vector<float> output(values.size());
  std::transform(values.begin(), values.end(), output.begin(),
                 [](double value) { return static_cast<float>(value); });
  return output;
}

void require_report(const rc::CheckReport& report) {
  if (!report.ok()) {
    const std::string detail =
        report.failures.empty() ? report.skip_reason : report.failures.front().message;
    throw std::runtime_error(report.case_id + ": " + detail);
  }
}

void test_dtype_host_and_runtime(cudaStream_t stream) {
  int device = 0;
  rc::cuda_check(cudaGetDevice(&device), "get CUDA device");
  cudaDeviceProp properties{};
  rc::cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
  const rc::DeviceCapability capability{properties.major, properties.minor, false, true};
  const auto master = rc::make_deterministic_master_data(19, 20260729ULL, -3.0, 3.0);
  require(master == rc::make_deterministic_master_data(19, 20260729ULL, -3.0, 3.0),
          "FP64 master data must be deterministic for a seed");
  const std::vector<rc::ScalarType> types = {rc::ScalarType::kFp32,    rc::ScalarType::kFp16,
                                             rc::ScalarType::kBf16,    rc::ScalarType::kFp8E4M3,
                                             rc::ScalarType::kFp8E5M2, rc::ScalarType::kFp6E2M3,
                                             rc::ScalarType::kFp6E3M2, rc::ScalarType::kFp4E2M1};
  for (const rc::ScalarType type : types) {
    const rc::DTypeTraits traits = rc::dtype_traits(type);
    require(traits.logical_bits <= traits.storage_unit_bits,
            "logical bits must not exceed storage unit bits");
    require(traits.allowed_accumulators == std::vector<rc::ScalarType>({rc::ScalarType::kFp32}),
            "dtype accumulator policy must require FP32 accumulation");
    for (const double value : {0.0, -0.0, 0.5, 1.0, -1.0, traits.max_finite * 0.75}) {
      const double decoded = rc::decode_scalar(type, rc::encode_scalar(type, value));
      require(std::isfinite(decoded) && std::abs(decoded) <= traits.max_finite,
              "dtype host roundtrip violates finite range for " + rc::to_string(type));
    }
    if (traits.saturates_finite) {
      const double saturated =
          rc::decode_scalar(type, rc::encode_scalar(type, traits.max_finite * 4.0));
      require(std::isfinite(saturated) && std::abs(saturated) <= traits.max_finite,
              "saturating dtype overflow must remain finite for " + rc::to_string(type));
    }
    if (traits.supports_nan) {
      require(std::isnan(rc::decode_scalar(
                  type, rc::encode_scalar(type, std::numeric_limits<double>::quiet_NaN()))),
              "dtype NaN roundtrip policy is incorrect for " + rc::to_string(type));
    }
    if (traits.supports_infinity) {
      const double infinity_conversion =
          rc::decode_scalar(type, rc::encode_scalar(type, std::numeric_limits<double>::infinity()));
      require(traits.saturates_finite ? std::isfinite(infinity_conversion) &&
                                            std::abs(infinity_conversion) <= traits.max_finite
                                      : std::isinf(infinity_conversion),
              "dtype infinity conversion policy is incorrect for " + rc::to_string(type));
    }
    const auto quantized = rc::quantize_master_data(master, type);
    for (std::size_t index = 0; index < master.size(); ++index) {
      require(quantized[index] == rc::decode_scalar(type, rc::encode_scalar(type, master[index])),
              "master-data quantization must decode actual storage bits");
    }
    const rc::CapabilityLevel level = rc::dtype_capability(type, capability);
    if (type == rc::ScalarType::kFp32 || type == rc::ScalarType::kFp16 ||
        type == rc::ScalarType::kBf16) {
      require(level == rc::CapabilityLevel::kRuntimeVerified,
              "SM86 core dtype must be runtime verified");
    } else {
      require(level == rc::CapabilityLevel::kReferenceOnly,
              "future dtype must remain reference-only on SM86");
    }
  }
  require(rc::logical_storage_bytes(rc::ScalarType::kFp4E2M1, 2) == 1,
          "two FP4 values must use one logical byte");
  require(rc::logical_storage_bytes(rc::ScalarType::kFp6E2M3, 4) == 3,
          "four FP6 values must use three logical bytes");
  require(rc::wrapper_storage_bytes(rc::ScalarType::kFp6E3M2, 4) == 4,
          "FP6 wrapper storage must remain explicit");
  require(rc::quantize_tf32(1.0F) == 1.0F &&
              std::isinf(rc::quantize_tf32(std::numeric_limits<float>::infinity())),
          "TF32 quantizer must preserve exact and special values");

  const std::vector<float> input = {-3.25F, -1.0F, -0.0F, 0.5F, 1.0F, 3.25F};
  for (const rc::ScalarType type :
       {rc::ScalarType::kFp32, rc::ScalarType::kFp16, rc::ScalarType::kBf16}) {
    std::vector<float> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      expected[index] =
          static_cast<float>(rc::decode_scalar(type, rc::encode_scalar(type, input[index])));
    }
    rc::GuardedDeviceBuffer<float> device_input(input.size(), stream);
    rc::GuardedDeviceBuffer<float> device_output(input.size(), stream);
    device_input.copy_from_host(input, stream);
    const std::uint64_t input_hash = device_input.payload_hash(stream);
    const auto launch = rc::observe_launch(
        [&] {
          return rc::launch_dtype_roundtrip(type, device_input.data(), device_output.data(),
                                            input.size(), stream);
        },
        stream);
    require(launch.launch_error == cudaSuccess && launch.execution_error == cudaSuccess,
            "dtype runtime roundtrip launch failed for " + rc::to_string(type));
    require(device_output.copy_to_host(stream) == expected,
            "dtype runtime roundtrip differs from host encoding");
    require(device_input.payload_hash(stream) == input_hash,
            "dtype roundtrip changed the input payload");
    require(device_input.canaries_intact(stream) && device_output.canaries_intact(stream),
            "dtype roundtrip changed a redzone");
  }
}

void test_references_and_invariants() {
  rc::CaseDescriptor dense = make_case("framework.dense", "dense_gemm");
  dense.shape = {{"M", 2}, {"N", 2}, {"K", 2}};
  const auto gemm =
      rc::dense_gemm_reference({1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {}, 2, 2, 2, 1.0, 0.0);
  require(gemm == std::vector<double>({19.0, 22.0, 43.0, 50.0}),
          "dense GEMM reference is incorrect");
  require_report(rc::compare_gemm(dense, gemm, gemm, {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, 2,
                                  2, 2, rc::ScalarType::kFp32));

  const auto all_nan = rc::top2_selected_softmax_reference(
      {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN()},
      1, 3);
  require(all_nan.ids == std::vector<std::int32_t>({0, 1}) &&
              all_nan.weights == std::vector<double>({0.5, 0.5}),
          "all-NaN Top-K contract is incorrect");
  const auto infinities = rc::top2_selected_softmax_reference(
      {std::numeric_limits<double>::infinity(), 2.0, -1.0}, 1, 3);
  require(infinities.weights == std::vector<double>({1.0, 0.0}),
          "infinite Top-K contract is incorrect");

  rc::CaseDescriptor routing = make_case("framework.routing", "routing");
  const std::vector<std::int32_t> ids = {0, 1, 0, 1};
  const auto counts = rc::histogram_reference(ids, 2);
  const auto offsets = rc::exclusive_scan_reference(counts);
  require_report(rc::validate_histogram(routing, ids, counts, 2));
  require_report(rc::validate_scan(routing, counts, offsets));

  const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
  const std::vector<std::int32_t> route_pos = {0, 2, 1, 3};
  const std::vector<std::int32_t> sorted_route = {0, 2, 1, 3};
  const std::vector<double> xp = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
  require_report(
      rc::validate_permute(routing, x, ids, offsets, xp, route_pos, &sorted_route, 2, 2, 2));

  const std::vector<double> weights = {1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0};
  const auto yp = rc::grouped_gemm_reference(xp, weights, offsets, 2, 2, 2);
  const auto y = rc::unpermute_reference(yp, route_pos, {0.5, 0.5, 0.5, 0.5}, 2, 2, 2);
  require(y.size() == 4 &&
              std::all_of(y.begin(), y.end(), [](double value) { return std::isfinite(value); }),
          "Grouped GEMM + Unpermute reference chain failed");
}

void test_failure_detection_and_replay(cudaStream_t stream) {
  rc::CaseDescriptor descriptor = make_case("framework.failure", "dense_gemm", 17);
  descriptor.shape = {{"M", 2}, {"N", 2}, {"K", 2}};
  const rc::CheckReport mismatch = rc::compare_floating(descriptor, {1.0}, {2.0}, 0.0, 0.0);
  require(!mismatch.ok(), "numeric mismatch must be detected");
  bool overflow_detected = false;
  try {
    (void)rc::checked_mul(std::numeric_limits<std::size_t>::max(), 2, "selftest");
  } catch (const std::overflow_error&) {
    overflow_detected = true;
  }
  require(overflow_detected, "checked multiplication must reject overflow");

  rc::GuardedDeviceBuffer<float> buffer(4, stream);
  rc::cuda_check(cudaMemsetAsync(reinterpret_cast<unsigned char*>(buffer.data()) + buffer.bytes(),
                                 0, 1, stream),
                 "corrupt back canary");
  rc::cuda_check(cudaStreamSynchronize(stream), "sync canary corruption");
  require(!buffer.canaries_intact(stream), "redzone corruption must be detected");

  const auto replay_path =
      (std::filesystem::temp_directory_path() / "raggedroute_correctness_replay.json").string();
  rc::save_case_json(replay_path, descriptor);
  const auto loaded = rc::load_case_json(replay_path);
  require(loaded.case_id == descriptor.case_id && loaded.shape == descriptor.shape &&
              loaded.seed == descriptor.seed,
          "case JSON replay changed descriptor fields");
  rc::save_failure_artifact(replay_path, mismatch, descriptor);
  const auto loaded_failure_case = rc::load_case_json(replay_path);
  require(loaded_failure_case.case_id == descriptor.case_id,
          "failure artifact cannot be replayed as a case descriptor");
  std::filesystem::remove(replay_path);
}

void test_zero_and_stream_contract(cudaStream_t stream) {
  require(ops::launch_dense_gemm_naive(nullptr, nullptr, nullptr, 0, 5, 3, stream) == cudaSuccess,
          "empty dense GEMM must be a no-op");
  require(ops::launch_topk_gate_naive(nullptr, nullptr, nullptr, 0, 8, stream) == cudaSuccess,
          "empty Top-K must be a no-op");
  require(ops::launch_token_permute_naive(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                          nullptr, 0, 2, 7, stream) == cudaSuccess,
          "empty permute must be a no-op");
  require(ops::launch_grouped_gemm_naive(nullptr, nullptr, nullptr, nullptr, 4, 7, 5, 0, stream) ==
              cudaSuccess,
          "empty grouped GEMM must be a no-op");
  require(ops::launch_unpermute_naive(nullptr, nullptr, nullptr, nullptr, 0, 2, 7, stream) ==
              cudaSuccess,
          "empty unpermute must be a no-op");
  require(
      ops::launch_topk_gate_naive(nullptr, nullptr, nullptr, 1, 1, stream) == cudaErrorInvalidValue,
      "Top-K E=1 must be rejected");
  cudaGetLastError();

  rc::GuardedDeviceBuffer<float> zero_dense(4, stream);
  require(ops::launch_dense_gemm_naive(nullptr, nullptr, zero_dense.data(), 2, 2, 0, stream) ==
              cudaSuccess,
          "K=0 dense GEMM must launch without input matrices");
  rc::cuda_check(cudaStreamSynchronize(stream), "sync K=0 dense GEMM");
  require(zero_dense.copy_to_host(stream) == std::vector<float>(4, 0.0F),
          "K=0 dense GEMM must write mathematical zeros");

  rc::GuardedDeviceBuffer<std::int32_t> zero_offsets(2, stream);
  zero_offsets.copy_from_host({0, 2}, stream);
  rc::GuardedDeviceBuffer<float> zero_grouped(4, stream);
  require(ops::launch_grouped_gemm_naive(nullptr, nullptr, zero_offsets.data(), zero_grouped.data(),
                                         1, 0, 2, 2, stream) == cudaSuccess,
          "K=0 grouped GEMM must launch without input matrices");
  rc::cuda_check(cudaStreamSynchronize(stream), "sync K=0 grouped GEMM");
  require(zero_grouped.copy_to_host(stream) == std::vector<float>(4, 0.0F),
          "K=0 grouped GEMM must write mathematical zeros");

  const std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> b = {5.0F, 6.0F, 7.0F, 8.0F};
  rc::GuardedDeviceBuffer<float> device_a(a.size(), stream);
  rc::GuardedDeviceBuffer<float> device_b(b.size(), stream);
  rc::GuardedDeviceBuffer<float> device_c(4, stream);
  device_a.copy_from_host(a, stream);
  device_b.copy_from_host(b, stream);
  const auto launch = rc::observe_launch(
      [&] {
        return ops::launch_dense_gemm_naive(device_a.data(), device_b.data(), device_c.data(), 2, 2,
                                            2, stream);
      },
      stream);
  require(launch.launch_error == cudaSuccess && launch.execution_error == cudaSuccess,
          "caller-stream dense GEMM launch failed");
  require(device_c.copy_to_host(stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "caller stream did not preserve copy-before-kernel ordering");
  require(device_a.canaries_intact(stream) && device_b.canaries_intact(stream) &&
              device_c.canaries_intact(stream),
          "stream contract test changed a redzone");
}

void test_randomized_reference_suite() {
  std::mt19937_64 engine(20260729ULL);
  std::uniform_real_distribution<double> values(-2.0, 2.0);
  for (int iteration = 0; iteration < 32; ++iteration) {
    const int tokens = 1 + iteration % 11;
    const int experts = iteration % 2 == 0 ? 4 : 8;
    std::vector<double> logits(static_cast<std::size_t>(tokens) * experts);
    for (double& value : logits) value = values(engine);
    const auto first = rc::top2_selected_softmax_reference(logits, tokens, experts);
    const auto second = rc::top2_selected_softmax_reference(logits, tokens, experts);
    require(first.ids == second.ids && first.weights == second.weights,
            "Top-K reference must be deterministic");
    const auto counts = rc::histogram_reference(first.ids, experts);
    const auto offsets = rc::exclusive_scan_reference(counts);
    require(offsets.back() == static_cast<std::int32_t>(first.ids.size()),
            "randomized scan must cover every route");
    for (int token = 0; token < tokens; ++token) {
      const std::size_t index = static_cast<std::size_t>(token) * 2;
      require(first.ids[index] != first.ids[index + 1] && first.ids[index] >= 0 &&
                  first.ids[index + 1] < experts &&
                  std::abs(first.weights[index] + first.weights[index + 1] - 1.0) < 1.0e-12,
              "randomized Top-K invariants failed");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string suite = "all";
  std::string replay_path;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--suite" && index + 1 < argc)
      suite = argv[++index];
    else if (argument == "--replay" && index + 1 < argc)
      replay_path = argv[++index];
    else if (argument == "--help") {
      std::cout << "raggedroute_correctness_framework_tests [--suite "
                   "all|dtype|selftest|edge|randomized|stream] [--replay FILE]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cerr << "SKIP CUDA device unavailable\n";
    return 77;
  }
  try {
    cudaStream_t stream{};
    rc::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "create correctness framework stream");
    try {
      if (!replay_path.empty()) {
        const auto replay = rc::load_case_json(replay_path);
        require(!replay.case_id.empty() && !replay.operator_name.empty(),
                "replay descriptor is incomplete");
      }
      if (suite == "all" || suite == "dtype") test_dtype_host_and_runtime(stream);
      if (suite == "all" || suite == "selftest") test_failure_detection_and_replay(stream);
      if (suite == "all" || suite == "edge") test_references_and_invariants();
      if (suite == "all" || suite == "edge" || suite == "stream")
        test_zero_and_stream_contract(stream);
      if (suite == "all" || suite == "randomized") test_randomized_reference_suite();
    } catch (...) {
      cudaStreamDestroy(stream);
      throw;
    }
    rc::cuda_check(cudaStreamDestroy(stream), "destroy correctness framework stream");
    std::cout << "PASS correctness framework suite=" << suite << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL correctness framework suite=" << suite << ": " << error.what() << '\n';
    return 1;
  }
}
