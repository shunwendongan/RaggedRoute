#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"
#include "raggedroute/benchmark/runner.h"
#include "raggedroute/correctness/framework.h"

namespace rr = raggedroute::benchmark;

namespace {

struct Case {
  std::string name;
  rr::OptionMap options;
  std::string variant = "cuda_naive";
  rr::MeasurementLevel level = rr::MeasurementLevel::kOperatorSteady;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void run_adapter_case(const Case& test_case, cudaStream_t stream, std::uint64_t seed) {
  auto adapter = rr::make_adapter(test_case.name, test_case.variant);
  require(adapter->supports(test_case.level),
          test_case.name + "/" + test_case.variant + " does not support the requested level");
  adapter->setup(test_case.options, seed, stream);
  adapter->prepare_sample(test_case.level, stream);
  adapter->enqueue(test_case.level, stream);
  rr::cuda_check(cudaStreamSynchronize(stream), "correctness case sync");
  const auto result = adapter->validate(stream);
  require(result.ok, test_case.name + "/" + test_case.variant + ": " + result.message);
  require(!adapter->case_config().empty(), test_case.name + " has no case config");
  const auto variant_config = adapter->variant_config();
  require(!variant_config.empty(), test_case.name + " has no variant config");
  for (const std::string field :
       {"implementation_category", "implementation_version", "implementation_revision",
        "dependency_revision", "algorithm_id", "math_mode"}) {
    require(variant_config.count(field) == 1,
            test_case.name + " variant config is missing " + field);
  }
  std::cout << "PASS " << test_case.name << "/" << test_case.variant << " - " << result.message
            << '\n';
}

void run_histogram_candidate_matrix(cudaStream_t stream, std::uint64_t* seed) {
  const std::vector<int> experts = {1, 8, 16, 31, 32, 33, 64};
  const std::vector<int> route_pairs = {1, 31, 32, 33, 255, 256, 257, 4096, 65536};
  const std::vector<std::pair<std::string, std::string>> distributions = {
      {"uniform", "0.0"}, {"round_robin", "0.0"}, {"zipf", "1.0"},
      {"zipf", "1.4"},    {"zipf", "2.0"},        {"single_hot", "0.0"},
  };
  for (std::size_t expert_index = 0; expert_index < experts.size(); ++expert_index) {
    for (std::size_t route_index = 0; route_index < route_pairs.size(); ++route_index) {
      const auto& distribution = distributions[(expert_index + route_index) % distributions.size()];
      run_adapter_case({"histogram",
                        {{"T", std::to_string(route_pairs[route_index])},
                         {"E", std::to_string(experts[expert_index])},
                         {"top_k", "1"},
                         {"distribution", distribution.first},
                         {"zipf_s", distribution.second}},
                        "cuda_candidate"},
                       stream, (*seed)++);
    }
  }
  for (const int route_count : {32, 256, 4096, 65536}) {
    run_adapter_case({"histogram",
                      {{"T", std::to_string(route_count / 2)},
                       {"E", "64"},
                       {"top_k", "2"},
                       {"distribution", "single_hot"}},
                      "cuda_candidate"},
                     stream, (*seed)++);
  }
}

bool has_variant(const std::string& operator_name, const std::string& variant) {
  const auto variants = rr::available_variants(operator_name);
  return std::find(variants.begin(), variants.end(), variant) != variants.end();
}

void run_library_alignment_fallbacks(cudaStream_t stream) {
#if RAGGEDROUTE_HAS_CCCL
  if (has_variant("token_permute", "vllm_moe_permute")) {
    constexpr int kTokens = 2;
    constexpr int kExperts = 2;
    constexpr int kTopK = 1;
    constexpr int kHidden = 8;
    rr::DeviceBuffer<float> input(kTokens * kHidden + 1);
    rr::DeviceBuffer<float> output(kTokens * kHidden + 1);
    rr::DeviceBuffer<std::int32_t> expert_ids(kTokens);
    rr::DeviceBuffer<std::int32_t> route_pos(kTokens);
    rr::DeviceBuffer<std::int32_t> sorted_route(kTokens);
    std::vector<float> host_input(1, -1.0F);
    for (int value = 0; value < kTokens * kHidden; ++value) {
      host_input.push_back(static_cast<float>(value));
    }
    input.copy_from_host(host_input, stream);
    rr::cuda_check(cudaMemsetAsync(output.data(), 0, output.bytes(), stream),
                   "initialize unaligned vLLM permute output storage");
    expert_ids.copy_from_host({1, 0}, stream);
    std::size_t workspace_bytes = 0;
    rr::cuda_check(rr::library_baseline::query_vllm_permute_workspace(kTokens, kExperts, kTopK,
                                                                      &workspace_bytes),
                   "query unaligned vLLM permute workspace");
    rr::DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
    rr::cuda_check(rr::library_baseline::initialize_vllm_permute_workspace(
                       workspace.data(), workspace_bytes, kTokens, kExperts, kTopK, stream),
                   "initialize unaligned vLLM permute workspace");
    require(reinterpret_cast<std::uintptr_t>(input.data() + 1) % alignof(float4) != 0,
            "permute fallback test input must be unaligned");
    rr::cuda_check(rr::library_baseline::launch_vllm_moe_permute(
                       input.data() + 1, expert_ids.data(), output.data() + 1, route_pos.data(),
                       sorted_route.data(), workspace.data(), workspace_bytes, kTokens, kExperts,
                       kTopK, kHidden, stream),
                   "launch unaligned vLLM permute");
    rr::cuda_check(cudaStreamSynchronize(stream), "sync unaligned vLLM permute");
    const auto output_host = output.copy_to_host(stream);
    const auto positions = route_pos.copy_to_host(stream);
    const auto sorted = sorted_route.copy_to_host(stream);
    require(positions == std::vector<std::int32_t>({1, 0}) &&
                sorted == std::vector<std::int32_t>({1, 0}),
            "unaligned vLLM permute mapping is incorrect");
    for (int column = 0; column < kHidden; ++column) {
      require(output_host[static_cast<std::size_t>(1 + column)] ==
                      static_cast<float>(kHidden + column) &&
                  output_host[static_cast<std::size_t>(1 + kHidden + column)] ==
                      static_cast<float>(column),
              "unaligned vLLM permute scalar fallback copied the wrong row");
    }
  }
#endif

#if RAGGEDROUTE_HAS_VLLM_UNPERMUTE
  if (has_variant("unpermute", "vllm_finalize_routing")) {
    constexpr int kTokens = 2;
    constexpr int kTopK = 2;
    constexpr int kWidth = 8;
    rr::DeviceBuffer<float> permuted(kTokens * kTopK * kWidth + 1);
    rr::DeviceBuffer<float> output(kTokens * kWidth + 1);
    rr::DeviceBuffer<float> weights(kTokens * kTopK);
    rr::DeviceBuffer<std::int32_t> route_pos(kTokens * kTopK);
    std::vector<float> host_permuted(1, -1.0F);
    for (int row = 0; row < kTokens * kTopK; ++row) {
      for (int column = 0; column < kWidth; ++column) {
        host_permuted.push_back(static_cast<float>(row + 1));
      }
    }
    permuted.copy_from_host(host_permuted, stream);
    rr::cuda_check(cudaMemsetAsync(output.data(), 0, output.bytes(), stream),
                   "initialize unaligned vLLM unpermute output storage");
    weights.copy_from_host({0.25F, 0.75F, 0.4F, 0.6F}, stream);
    route_pos.copy_from_host({2, 0, 3, 1}, stream);
    require(reinterpret_cast<std::uintptr_t>(permuted.data() + 1) % alignof(float4) != 0,
            "unpermute fallback test input must be unaligned");
    rr::cuda_check(rr::library_baseline::launch_vllm_finalize_routing(
                       permuted.data() + 1, output.data() + 1, weights.data(), route_pos.data(),
                       kTokens, kTopK, kWidth, stream),
                   "launch unaligned vLLM unpermute");
    rr::cuda_check(cudaStreamSynchronize(stream), "sync unaligned vLLM unpermute");
    const auto output_host = output.copy_to_host(stream);
    for (int column = 0; column < kWidth; ++column) {
      require(std::fabs(output_host[static_cast<std::size_t>(1 + column)] - 1.5F) <= 1.0e-6F &&
                  std::fabs(output_host[static_cast<std::size_t>(1 + kWidth + column)] - 2.8F) <=
                      1.0e-6F,
              "unaligned vLLM unpermute scalar fallback produced the wrong reduction");
    }
  }
#endif
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
    const auto fp16 = raggedroute::correctness::dtype_traits(raggedroute::ScalarType::kFp16);
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
    raggedroute::correctness::CheckReport failure_report;
    failure_report.case_id = descriptor.case_id;
    failure_report.operator_name = descriptor.operator_name;
    failure_report.fail({"numeric_class", "non-finite diagnostic", 0,
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(), 0.0});
    const auto failure_json = raggedroute::correctness::report_to_json(failure_report, descriptor);
    require(failure_json.find("\"actual\":null") != std::string::npos &&
                failure_json.find("\"expected\":null") != std::string::npos,
            "correctness failure artifacts must remain valid JSON for NaN/Inf");

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
          {"histogram", {{"T", "1"}, {"E", "1"}, {"top_k", "1"}, {"distribution", "single_hot"}}},
          {"histogram",
           {{"T", "33"}, {"E", "31"}, {"top_k", "1"}, {"distribution", "round_robin"}}},
          {"histogram", {{"T", "255"}, {"E", "32"}, {"top_k", "1"}, {"distribution", "uniform"}}},
          {"histogram",
           {{"T", "256"},
            {"E", "33"},
            {"top_k", "1"},
            {"distribution", "zipf"},
            {"zipf_s", "1.0"}}},
          {"histogram",
           {{"T", "257"},
            {"E", "64"},
            {"top_k", "1"},
            {"distribution", "zipf"},
            {"zipf_s", "2.0"}}},
          {"histogram",
           {{"T", "4096"}, {"E", "8"}, {"top_k", "1"}, {"distribution", "single_hot"}}},
          {"histogram", {{"T", "65536"}, {"E", "16"}, {"top_k", "1"}, {"distribution", "uniform"}}},
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

      const std::vector<Case> library_cases = {
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cublaslt"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cublas"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cuda_tiled_scalar"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cuda_2d_mapping"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cuda_tiled_vector"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cuda_combined"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cuda_register_tiled_v2_sync"},
          {"dense_gemm", {{"M", "5"}, {"N", "7"}, {"K", "3"}}, "cuda_register_tiled_v2_async"},
          {"dense_gemm",
           {{"M", "5"}, {"N", "7"}, {"K", "3"}},
           "cuda_register_tiled_v3_64x32_async"},
          {"histogram", {{"T", "11"}, {"E", "8"}, {"top_k", "2"}}, "cub_device_histogram"},
          {"histogram",
           {{"T", "257"}, {"E", "64"}, {"top_k", "1"}, {"distribution", "single_hot"}},
           "cuda_candidate"},
          {"histogram",
           {{"T", "8192"},
            {"E", "33"},
            {"top_k", "1"},
            {"distribution", "zipf"},
            {"zipf_s", "2.0"}},
           "cuda_candidate"},
          {"histogram",
           {{"T", "32768"},
            {"E", "33"},
            {"top_k", "1"},
            {"distribution", "zipf"},
            {"zipf_s", "2.0"}},
           "cuda_candidate"},
          {"exclusive_scan", {{"E", "7"}, {"R", "23"}}, "cub_device_scan"},
          {"exclusive_scan", {{"E", "33"}, {"R", "67"}}, "cub_block_scan"},
          {"exclusive_scan", {{"E", "31"}, {"R", "67"}}, "cub_warp_scan"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}},
           "cuda_naive_from_ids"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}},
           "cuda_atomic_vectorized_128"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "8"}},
           "cuda_atomic_vectorized_64"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "8"}},
           "cuda_atomic_vectorized_256"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}},
           "cuda_token_owned_top2"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "8"}},
           "cuda_block_partial"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "8"}},
           "cuda_candidate"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}},
           "cuda_candidate_from_ids"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}},
           "vllm_moe_permute"},
          {"token_permute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "8"}},
           "vllm_expand_rows",
           rr::MeasurementLevel::kKernelBody},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cublas_per_expert"},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cutlass_grouped", rr::MeasurementLevel::kKernelBody},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cuda_grouped_tiled16_sync_v0"},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cuda_grouped_persistent16_v1"},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cuda_grouped_register16x32_sync_v2"},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cuda_grouped_register16x32_async_v3"},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cuda_grouped_register16x32_async_full_v4"},
          {"grouped_gemm",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"K", "7"}, {"N", "5"}},
           "cuda_grouped_sm86_fp32_v1"},
          {"unpermute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"N", "7"}},
           "vllm_finalize_routing",
           rr::MeasurementLevel::kKernelBody},
          {"unpermute",
           {{"T", "5"}, {"E", "4"}, {"top_k", "2"}, {"N", "8"}},
           "vllm_finalize_routing",
           rr::MeasurementLevel::kOperatorSteady},
      };
      for (const auto& test_case : library_cases) {
        if (has_variant(test_case.name, test_case.variant)) {
          run_adapter_case(test_case, stream, seed++);
        } else {
          std::cout << "SKIP " << test_case.name << "/" << test_case.variant
                    << " - optional dependency unavailable\n";
        }
      }
      run_histogram_candidate_matrix(stream, &seed);
      const std::vector<rr::OptionMap> unpermute_candidate_shapes = {
          {{"T", "1"}, {"E", "64"}, {"top_k", "1"}, {"N", "1"},
           {"distribution", "round_robin"}},
          {{"T", "17"}, {"E", "64"}, {"top_k", "2"}, {"N", "3"},
           {"distribution", "uniform"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "4"}, {"N", "4"},
           {"distribution", "round_robin"}},
          {{"T", "512"}, {"E", "64"}, {"top_k", "64"}, {"N", "7"},
           {"distribution", "single_hot"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "2"}, {"N", "63"},
           {"distribution", "zipf"}, {"zipf_s", "1.4"}},
          {{"T", "4096"}, {"E", "64"}, {"top_k", "2"}, {"N", "64"},
           {"distribution", "uniform"}},
          {{"T", "17"}, {"E", "64"}, {"top_k", "2"}, {"N", "65"},
           {"distribution", "zipf"}, {"zipf_s", "1.4"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "2"}, {"N", "255"},
           {"distribution", "uniform"}},
          {{"T", "1024"}, {"E", "64"}, {"top_k", "2"}, {"N", "256"},
           {"distribution", "zipf"}, {"zipf_s", "1.4"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "2"}, {"N", "257"},
           {"distribution", "uniform"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "2"}, {"N", "1024"},
           {"distribution", "round_robin"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "2"}, {"N", "256"},
           {"distribution", "uniform"}, {"pointer_offset_elements", "1"}},
      };
      for (const auto& shape : unpermute_candidate_shapes) {
        run_adapter_case({"unpermute", shape, "cuda_warp_token_vec4"}, stream, seed++);
      }
      const std::vector<rr::OptionMap> dense_tiled_edge_shapes = {
          {{"M", "1"}, {"N", "1"}, {"K", "1"}},
          {{"M", "17"}, {"N", "19"}, {"K", "13"}},
          {{"M", "31"}, {"N", "33"}, {"K", "65"}},
          {{"M", "256"}, {"N", "256"}, {"K", "256"}},
      };
      for (const std::string& variant :
           {"cuda_tiled_scalar", "cuda_2d_mapping", "cuda_tiled_vector", "cuda_combined",
            "cuda_register_tiled_v2_sync", "cuda_register_tiled_v2_async",
            "cuda_register_tiled_v3_64x32_async"}) {
        if (!has_variant("dense_gemm", variant)) continue;
        for (const auto& shape : dense_tiled_edge_shapes) {
          run_adapter_case({"dense_gemm", shape, variant}, stream, seed++);
        }
      }
      const std::vector<rr::OptionMap> grouped_edge_shapes = {
          {{"T", "17"}, {"E", "8"}, {"top_k", "2"}, {"K", "13"}, {"N", "11"},
           {"distribution", "zipf"}, {"zipf_s", "1.0"}},
          {{"T", "16"}, {"E", "64"}, {"top_k", "2"}, {"K", "64"}, {"N", "64"},
           {"distribution", "round_robin"}},
          {{"T", "64"}, {"E", "64"}, {"top_k", "2"}, {"K", "128"}, {"N", "128"},
           {"distribution", "single_hot"}},
      };
      for (const std::string& variant :
           {"cuda_grouped_tiled16_sync_v0", "cuda_grouped_persistent16_v1",
            "cuda_grouped_register16x32_sync_v2", "cuda_grouped_register16x32_async_v3",
            "cuda_grouped_register16x32_async_full_v4",
            "cuda_grouped_sm86_fp32_v1"}) {
        for (const auto& shape : grouped_edge_shapes) {
          run_adapter_case({"grouped_gemm", shape, variant}, stream, seed++);
        }
      }
      run_library_alignment_fallbacks(stream);

      for (const auto& suite_name : rr::available_suites()) {
        for (const auto& variant : rr::available_suite_variants(suite_name)) {
          auto chain = rr::make_suite_adapter(suite_name, variant);
          rr::OptionMap options = {{"T", "5"},
                                   {"E", "4"},
                                   {"K", "7"},
                                   {"N", "5"},
                                   {"distribution", "uniform"}};
          chain->setup(options, seed++, stream);
          chain->prepare_sample(rr::MeasurementLevel::kChainSteady, stream);
          chain->enqueue(rr::MeasurementLevel::kChainSteady, stream);
          rr::cuda_check(cudaStreamSynchronize(stream), "chain correctness sync");
          const auto validation = chain->validate(stream);
          require(validation.ok, suite_name + "/" + variant + ": " + validation.message);
          std::cout << "PASS " << suite_name << "/" << variant << " - "
                    << validation.message << '\n';
        }
      }

      auto permute = rr::make_adapter("token_permute", "cuda_naive");
      permute->setup({{"T", "4"}, {"E", "8"}, {"top_k", "2"}, {"K", "4"}}, seed, stream);
      require(permute->repeat_policy(rr::MeasurementLevel::kKernelBody).max_repeats == 1,
              "stateful permute L1 must reject batched repeats");
      require(permute->workspace_bytes() == 8 * sizeof(std::int32_t),
              "permute must report its cursor workspace");
      const auto permute_l1 = permute->work_estimate(rr::MeasurementLevel::kKernelBody);
      const auto permute_l2 = permute->work_estimate(rr::MeasurementLevel::kOperatorSteady);
      require(permute_l2.logical_bytes - permute_l1.logical_bytes == 8 * sizeof(std::int32_t),
              "permute L2 work must include cursor reset bytes");
      const auto permute_l1_excluded = permute->excluded_steps(rr::MeasurementLevel::kKernelBody);
      const auto permute_l2_excluded =
          permute->excluded_steps(rr::MeasurementLevel::kOperatorSteady);
      require(std::find(permute_l1_excluded.begin(), permute_l1_excluded.end(), "cursor_reset") !=
                      permute_l1_excluded.end() &&
                  std::find(permute_l2_excluded.begin(), permute_l2_excluded.end(),
                            "cursor_reset") == permute_l2_excluded.end(),
              "permute reset exclusion must distinguish L1 from L2");

      require(std::get<std::string>(permute->case_config().at("placement_order")) ==
                  "unstable_atomic_cursor",
              "suite-v1 permute placement signature changed");
      if (has_variant("token_permute", "vllm_moe_permute")) {
        auto from_ids = rr::make_adapter("token_permute", "cuda_naive_from_ids");
        auto vllm_full = rr::make_adapter("token_permute", "vllm_moe_permute");
        const rr::OptionMap options = {{"T", "4"}, {"E", "8"}, {"top_k", "2"}, {"K", "7"}};
        from_ids->setup(options, seed, stream);
        vllm_full->setup(options, seed, stream);
        require(from_ids->case_config() == vllm_full->case_config(),
                "full permute baselines must have an identical logical boundary");
        const auto from_ids_work = from_ids->work_estimate(rr::MeasurementLevel::kOperatorSteady);
        const auto vllm_work = vllm_full->work_estimate(rr::MeasurementLevel::kOperatorSteady);
        require(from_ids_work.operator_metrics.count("cursor_reset_bytes") == 1 &&
                    from_ids_work.operator_metrics.count("counts_reset_bytes") == 1,
                "naive-from-ids work must include its resets");
        require(vllm_work.operator_metrics.count("cursor_reset_bytes") == 0 &&
                    vllm_work.operator_metrics.count("counts_reset_bytes") == 0,
                "vLLM full permute must not report naive cursor/count reset traffic");
      }

      auto histogram = rr::make_adapter("histogram", "cuda_naive");
      histogram->setup({{"T", "4"}, {"E", "8"}, {"top_k", "2"}}, seed, stream);
      const auto histogram_l1 = histogram->work_estimate(rr::MeasurementLevel::kKernelBody);
      const auto histogram_l2 = histogram->work_estimate(rr::MeasurementLevel::kOperatorSteady);
      require(histogram_l2.logical_bytes - histogram_l1.logical_bytes == 8 * sizeof(std::int32_t),
              "histogram L2 work must include counts reset bytes");
      const auto histogram_l1_excluded =
          histogram->excluded_steps(rr::MeasurementLevel::kKernelBody);
      const auto histogram_l2_excluded =
          histogram->excluded_steps(rr::MeasurementLevel::kOperatorSteady);
      require(std::find(histogram_l1_excluded.begin(), histogram_l1_excluded.end(),
                        "counts_reset") != histogram_l1_excluded.end() &&
                  std::find(histogram_l2_excluded.begin(), histogram_l2_excluded.end(),
                            "counts_reset") == histogram_l2_excluded.end(),
              "histogram reset exclusion must distinguish L1 from L2");
      if (has_variant("histogram", "cub_device_histogram")) {
        auto cub_histogram = rr::make_adapter("histogram", "cub_device_histogram");
        cub_histogram->setup({{"T", "4"}, {"E", "8"}, {"top_k", "2"}}, seed, stream);
        const auto cub_work = cub_histogram->work_estimate(rr::MeasurementLevel::kOperatorSteady);
        require(cub_work.operator_metrics.count("global_atomic_operations") == 0 &&
                    cub_work.operator_metrics.count("histogram_input_items") == 1,
                "CUB histogram metrics must not invent its internal atomic strategy");
      }

      auto chain = rr::make_suite_adapter("chain_from_logits", "cuda_naive");
      chain->setup({{"T", "4"}, {"E", "4"}, {"K", "4"}, {"N", "4"}}, seed, stream);
      require(chain->workspace_bytes() == 4 * sizeof(std::int32_t),
              "chain must report its cursor workspace");

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
