#include <cuda_runtime_api.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/registry.h"
#include "raggedroute/benchmark/runner.h"
#include "raggedroute/correctness/framework.h"

namespace rr = raggedroute::benchmark;

namespace {

struct Case {
  std::string name;
  rr::OptionMap options;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void run_adapter_case(const Case& test_case, cudaStream_t stream, std::uint64_t seed) {
  auto adapter = rr::make_adapter(test_case.name, "cuda_naive");
  require(adapter->supports(rr::MeasurementLevel::kOperatorSteady),
          test_case.name + " does not support L2");
  adapter->setup(test_case.options, seed, stream);
  adapter->prepare_sample(rr::MeasurementLevel::kOperatorSteady, stream);
  adapter->enqueue(rr::MeasurementLevel::kOperatorSteady, stream);
  rr::cuda_check(cudaStreamSynchronize(stream), "correctness case sync");
  const auto result = adapter->validate(stream);
  require(result.ok, test_case.name + ": " + result.message);
  require(!adapter->case_config().empty(), test_case.name + " has no case config");
  require(!adapter->variant_config().empty(), test_case.name + " has no variant config");
  std::cout << "PASS " << test_case.name << " - " << result.message << '\n';
}

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status != cudaSuccess || devices == 0) {
    std::cerr << "SKIP CUDA device unavailable\n";
    return 77;
  }

  try {
    const auto operators = rr::available_operators();
    require(operators.size() == 7, "registry must expose exactly seven operators");
    require(rr::available_suites().size() == 2,
            "registry must expose the two non-overlapping chain suites");
    const auto summary = rr::summarize_samples({1.0, 2.0, 3.0, 4.0, 5.0});
    require(summary.p50_us == 3.0 && summary.min_us == 1.0,
            "statistics implementation is incorrect");
    const auto fp16 = raggedroute::correctness::dtype_traits(
        raggedroute::correctness::ScalarType::kFp16);
    require(fp16.logical_bits == 16 && fp16.minimum_native_compute_capability == 70,
            "correctness dtype traits are incorrect");
    require(raggedroute::correctness::parse_math_mode("tf32") ==
                raggedroute::correctness::MathMode::kTf32,
            "correctness math mode parser is incorrect");
    const auto gemm_reference = raggedroute::correctness::dense_gemm_reference(
        {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {}, 2, 2, 2, 1.0, 0.0);
    require(gemm_reference == std::vector<double>({19.0, 22.0, 43.0, 50.0}),
            "correctness GEMM reference is incorrect");
    raggedroute::correctness::CaseDescriptor descriptor;
    descriptor.case_id = "roundtrip";
    descriptor.operator_name = "dense_gemm";
    descriptor.shape = {{"M", 2}, {"N", 2}, {"K", 2}};
    const auto parsed_descriptor = raggedroute::correctness::case_from_json(
        raggedroute::correctness::case_to_json(descriptor));
    require(parsed_descriptor.case_id == descriptor.case_id &&
                parsed_descriptor.shape == descriptor.shape,
            "correctness JSON roundtrip is incorrect");

    cudaStream_t stream{};
    rr::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "create correctness stream");
    try {
      const std::vector<Case> cases = {
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}},
          {"topk_gate", {{"T", "5"}, {"E", "8"}, {"input_mode", "random"}}},
          {"topk_gate", {{"T", "3"}, {"E", "8"}, {"input_mode", "ties"}}},
          {"topk_gate", {{"T", "3"}, {"E", "8"}, {"input_mode", "all_nan"}}},
          {"topk_gate", {{"T", "3"}, {"E", "8"}, {"input_mode", "infinities"}}},
          {"histogram",
           {{"T", "11"}, {"E", "8"}, {"top_k", "2"}, {"distribution", "zipf"}, {"zipf_s", "1.4"}}},
          {"exclusive_scan", {{"E", "7"}, {"R", "23"}, {"distribution", "single_hot"}}},
          {"token_permute",
           {{"T", "5"},
            {"E", "4"},
            {"top_k", "2"},
            {"K", "7"},
            {"distribution", "zipf"},
            {"zipf_s", "1.0"},
            {"materialize_sorted_route", "true"}}},
          {"token_permute",
           {{"T", "5"},
            {"E", "4"},
            {"top_k", "2"},
            {"K", "7"},
            {"distribution", "single_hot"},
            {"materialize_sorted_route", "false"}}},
          {"grouped_gemm",
           {{"T", "5"},
            {"E", "4"},
            {"top_k", "2"},
            {"K", "7"},
            {"N", "5"},
            {"distribution", "zipf"},
            {"zipf_s", "1.4"}}},
          {"unpermute",
           {{"T", "5"},
            {"E", "4"},
            {"top_k", "2"},
            {"N", "7"},
            {"distribution", "zipf"},
            {"zipf_s", "1.4"}}},
      };
      std::uint64_t seed = 20260729;
      for (const auto& test_case : cases) {
        run_adapter_case(test_case, stream, seed++);
      }

      for (const auto& suite_name : rr::available_suites()) {
        auto chain = rr::make_suite_adapter(suite_name, "cuda_naive");
        rr::OptionMap options = {
            {"T", "5"}, {"E", "4"}, {"K", "7"}, {"N", "5"}, {"distribution", "uniform"}};
        chain->setup(options, seed++, stream);
        chain->prepare_sample(rr::MeasurementLevel::kChainSteady, stream);
        chain->enqueue(rr::MeasurementLevel::kChainSteady, stream);
        rr::cuda_check(cudaStreamSynchronize(stream), "chain correctness sync");
        const auto validation = chain->validate(stream);
        require(validation.ok, suite_name + ": " + validation.message);
        std::cout << "PASS " << suite_name << " - " << validation.message << '\n';
      }

      auto permute = rr::make_adapter("token_permute", "cuda_naive");
      permute->setup({{"T", "4"}, {"E", "4"}, {"top_k", "2"}, {"K", "4"}}, seed, stream);
      require(permute->repeat_policy(rr::MeasurementLevel::kKernelBody).max_repeats == 1,
              "stateful permute L1 must reject batched repeats");

      bool rejected = false;
      try {
        auto dense = rr::make_adapter("dense_gemm", "cuda_naive");
        dense->setup({{"unexpected", "1"}}, seed, stream);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "adapter must reject unknown operator-specific params");
    } catch (...) {
      cudaStreamDestroy(stream);
      throw;
    }
    rr::cuda_check(cudaStreamDestroy(stream), "destroy correctness stream");
    std::cout << "All correctness tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
