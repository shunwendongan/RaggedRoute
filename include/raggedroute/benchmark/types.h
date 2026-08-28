#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace raggedroute::benchmark {

enum class MeasurementLevel {
  kKernelBody,
  kOperatorSteady,
  kChainSteady,
  kHostCall,
};

enum class Protocol { kSmoke, kRelease };
enum class CacheMode { kWarm, kColdScrub };

std::string to_string(MeasurementLevel value);
std::string to_string(Protocol value);
std::string to_string(CacheMode value);
MeasurementLevel parse_measurement_level(const std::string& value);
Protocol parse_protocol(const std::string& value);
CacheMode parse_cache_mode(const std::string& value);

using FieldValue = std::variant<std::int64_t, double, bool, std::string>;
using FieldMap = std::map<std::string, FieldValue>;
using OptionMap = std::map<std::string, std::string>;

struct RepeatPolicy {
  // 0 means there is no adapter-specific cap.
  int max_repeats = 0;
  std::string reason;
};

struct ValidationResult {
  bool ok = false;
  std::string message;
  std::optional<double> max_abs_error;
  std::optional<double> max_rel_error;
};

struct WorkEstimate {
  double logical_bytes = 0.0;
  double flops = 0.0;
  FieldMap operator_metrics;
};

struct MeasurementSummary {
  std::vector<double> raw_samples_us;
  double mean_us = 0.0;
  double stddev_us = 0.0;
  double cv = 0.0;
  double min_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p95_us = 0.0;
};

struct EnvironmentInfo {
  FieldMap fields;
};

struct RunOptions {
  MeasurementLevel level = MeasurementLevel::kKernelBody;
  Protocol protocol = Protocol::kSmoke;
  CacheMode cache_mode = CacheMode::kWarm;
  int warmup = 5;
  int kernel_repeats = 1;
  int samples = 10;
  int process_run = 1;
  std::uint64_t seed = 20260729ULL;
  std::size_t cache_scrub_bytes = 0;
  bool validate_after_measurement = true;
  std::string run_id;
  std::string case_id;
  std::string output_path;
  std::string expected_git_sha;
};

struct BenchmarkRecord {
  std::string schema_version = "raggedroute.benchmark.v1";
  std::string timestamp_utc;
  std::string run_id;
  std::string case_id;
  std::string operator_name;
  std::string variant_name;
  MeasurementLevel level = MeasurementLevel::kKernelBody;
  Protocol protocol = Protocol::kSmoke;
  CacheMode cache_mode = CacheMode::kWarm;
  int warmup = 0;
  int kernel_repeats = 0;
  int samples = 0;
  int process_run = 0;
  std::uint64_t seed = 0;
  std::vector<std::string> excluded_steps;
  std::size_t workspace_bytes = 0;
  FieldMap case_config;
  FieldMap variant_config;
  WorkEstimate work;
  EnvironmentInfo environment;
  MeasurementSummary timing;
  std::optional<MeasurementSummary> gpu_span_timing;
  std::optional<MeasurementSummary> host_time_to_solution_timing;
  std::optional<MeasurementSummary> cpu_submission_timing;
  ValidationResult validation;
};

}  // namespace raggedroute::benchmark
