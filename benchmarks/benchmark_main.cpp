#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/registry.h"
#include "raggedroute/benchmark/runner.h"

namespace rr = raggedroute::benchmark;

namespace {

void usage(std::ostream& out) {
  out << "RaggedRoute CUDA benchmark\n\n"
      << "  raggedroute_benchmark --list\n"
      << "  raggedroute_benchmark --operator NAME [options] --param key=value ...\n\n"
      << "  raggedroute_benchmark --suite chain_from_tokens|chain_from_logits ...\n\n"
      << "Common options:\n"
      << "  --variant cuda_naive\n"
      << "  --level l1|l2\n"
      << "  --protocol smoke|release\n"
      << "  --cache-mode warm|cold_scrub\n"
      << "  --cache-scrub-bytes N\n"
      << "  --warmup N --kernel-repeats N --samples N\n"
      << "  --process-run N --seed N\n"
      << "  --run-id ID --case-id ID --output FILE.jsonl\n"
      << "  --expected-git-sha SHA   required by release orchestration\n"
      << "  --profile-once   warm up, then enqueue one validated invocation without timing\n"
      << "  --no-validate\n";
}

std::string require_value(int& index, int argc, char** argv, const std::string& option) {
  if (index + 1 >= argc) throw std::invalid_argument(option + " requires a value");
  return argv[++index];
}

int parse_positive(const std::string& value, const std::string& option, bool allow_zero = false) {
  std::size_t consumed = 0;
  const long long parsed = std::stoll(value, &consumed);
  if (consumed != value.size() || parsed < (allow_zero ? 0 : 1) || parsed > 2147483647LL) {
    throw std::invalid_argument("invalid " + option + ": " + value);
  }
  return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string operator_name;
    std::string suite_name;
    std::string variant_name = "cuda_naive";
    rr::RunOptions run;
    rr::OptionMap adapter_options;
    bool list = false;
    bool profile_once = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        usage(std::cout);
        return 0;
      } else if (arg == "--list") {
        list = true;
      } else if (arg == "--operator") {
        operator_name = require_value(i, argc, argv, arg);
      } else if (arg == "--suite") {
        suite_name = require_value(i, argc, argv, arg);
      } else if (arg == "--variant") {
        variant_name = require_value(i, argc, argv, arg);
      } else if (arg == "--level") {
        run.level = rr::parse_measurement_level(require_value(i, argc, argv, arg));
      } else if (arg == "--protocol") {
        run.protocol = rr::parse_protocol(require_value(i, argc, argv, arg));
      } else if (arg == "--cache-mode") {
        run.cache_mode = rr::parse_cache_mode(require_value(i, argc, argv, arg));
      } else if (arg == "--cache-scrub-bytes") {
        run.cache_scrub_bytes =
            static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, arg)));
      } else if (arg == "--warmup") {
        run.warmup = parse_positive(require_value(i, argc, argv, arg), arg, true);
      } else if (arg == "--kernel-repeats") {
        run.kernel_repeats = parse_positive(require_value(i, argc, argv, arg), arg);
      } else if (arg == "--samples") {
        run.samples = parse_positive(require_value(i, argc, argv, arg), arg);
      } else if (arg == "--process-run") {
        run.process_run = parse_positive(require_value(i, argc, argv, arg), arg);
      } else if (arg == "--seed") {
        run.seed = std::stoull(require_value(i, argc, argv, arg));
      } else if (arg == "--run-id") {
        run.run_id = require_value(i, argc, argv, arg);
      } else if (arg == "--case-id") {
        run.case_id = require_value(i, argc, argv, arg);
      } else if (arg == "--output") {
        run.output_path = require_value(i, argc, argv, arg);
      } else if (arg == "--expected-git-sha") {
        run.expected_git_sha = require_value(i, argc, argv, arg);
      } else if (arg == "--no-validate") {
        run.validate_after_measurement = false;
      } else if (arg == "--profile-once") {
        profile_once = true;
      } else if (arg == "--param") {
        const std::string assignment = require_value(i, argc, argv, arg);
        const auto separator = assignment.find('=');
        if (separator == std::string::npos || separator == 0) {
          throw std::invalid_argument("--param requires key=value");
        }
        adapter_options[assignment.substr(0, separator)] = assignment.substr(separator + 1);
      } else {
        throw std::invalid_argument("unknown option: " + arg);
      }
    }

    if (list) {
      for (const auto& name : rr::available_operators()) {
        std::cout << name << ":";
        for (const auto& variant : rr::available_variants(name)) {
          std::cout << ' ' << variant;
        }
        std::cout << '\n';
      }
      for (const auto& name : rr::available_suites()) {
        std::cout << "suite/" << name << ':';
        for (const auto& variant : rr::available_suite_variants(name)) {
          std::cout << ' ' << variant;
        }
        std::cout << '\n';
      }
      return 0;
    }
    if (operator_name.empty() == suite_name.empty()) {
      usage(std::cerr);
      throw std::invalid_argument("specify exactly one of --operator or --suite");
    }
    const std::string target_name = operator_name.empty() ? suite_name : operator_name;
    if (run.run_id.empty()) run.run_id = "manual";
    if (run.case_id.empty()) run.case_id = target_name + ".manual";

    rr::AdapterPtr adapter = operator_name.empty()
                                 ? rr::make_suite_adapter(suite_name, variant_name)
                                 : rr::make_adapter(operator_name, variant_name);
    cudaStream_t stream{};
    rr::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    try {
      adapter->setup(adapter_options, run.seed, stream);
      rr::cuda_check(cudaStreamSynchronize(stream), "setup synchronization");
      if (profile_once) {
        for (int iteration = 0; iteration < run.warmup; ++iteration) {
          adapter->prepare_sample(run.level, stream);
          adapter->enqueue(run.level, stream);
        }
        rr::cuda_check(cudaStreamSynchronize(stream), "profile warmup synchronization");
        adapter->prepare_sample(run.level, stream);
        rr::cuda_check(cudaStreamSynchronize(stream), "profile preparation synchronization");
        adapter->enqueue(run.level, stream);
        rr::cuda_check(cudaStreamSynchronize(stream), "profile invocation synchronization");
        const rr::ValidationResult validation = adapter->validate(stream);
        if (!validation.ok) {
          throw std::runtime_error("profile validation failed: " + validation.message);
        }
        std::cout << "{\"mode\":\"profile_once\",\"operator\":\"" << adapter->operator_name()
                  << "\",\"variant\":\"" << adapter->variant_name()
                  << "\",\"warmup\":" << run.warmup << ",\"validation_ok\":true}\n";
      } else {
        const rr::BenchmarkRecord record = rr::run_benchmark(*adapter, run, stream);
        const std::string json = rr::record_to_json(record);
        std::cout << json << '\n';
        if (!run.output_path.empty()) rr::append_jsonl(run.output_path, record);
      }
    } catch (...) {
      cudaStreamDestroy(stream);
      throw;
    }
    rr::cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
