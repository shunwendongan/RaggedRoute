#include "raggedroute/benchmark/runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"

#ifndef RAGGEDROUTE_GIT_SHA
#define RAGGEDROUTE_GIT_SHA "unknown"
#endif
#ifndef RAGGEDROUTE_GIT_DIRTY
#define RAGGEDROUTE_GIT_DIRTY "unknown"
#endif
#ifndef RAGGEDROUTE_BUILD_TYPE
#define RAGGEDROUTE_BUILD_TYPE "unknown"
#endif
#ifndef RAGGEDROUTE_CUDA_COMPILER_VERSION
#define RAGGEDROUTE_CUDA_COMPILER_VERSION "unknown"
#endif

namespace raggedroute::benchmark {
namespace {

double percentile(std::vector<double> sorted, double quantile) {
  if (sorted.empty()) throw std::invalid_argument("cannot summarize no samples");
  std::sort(sorted.begin(), sorted.end());
  const double position = quantile * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

std::string escape_json(const std::string& input) {
  std::ostringstream output;
  for (const unsigned char c : input) {
    switch (c) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (c < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                 << std::dec;
        } else {
          output << static_cast<char>(c);
        }
    }
  }
  return output.str();
}

void write_quoted(std::ostringstream& out, const std::string& value) {
  out << '"' << escape_json(value) << '"';
}

void write_number(std::ostringstream& out, double value) {
  if (std::isfinite(value)) {
    out << std::setprecision(12) << value;
  } else {
    out << "null";
  }
}

void write_field_value(std::ostringstream& out, const FieldValue& value) {
  std::visit(
      [&](const auto& item) {
        using Item = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<Item, std::string>) {
          write_quoted(out, item);
        } else if constexpr (std::is_same_v<Item, bool>) {
          out << (item ? "true" : "false");
        } else if constexpr (std::is_same_v<Item, double>) {
          write_number(out, item);
        } else {
          out << item;
        }
      },
      value);
}

void write_field_map(std::ostringstream& out, const FieldMap& fields) {
  out << '{';
  bool first = true;
  for (const auto& [name, value] : fields) {
    if (!first) out << ',';
    first = false;
    write_quoted(out, name);
    out << ':';
    write_field_value(out, value);
  }
  out << '}';
}

std::string now_utc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &timestamp);
#else
  gmtime_r(&timestamp, &utc);
#endif
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

class EventPair {
 public:
  EventPair() {
    cuda_check(cudaEventCreate(&start_), "cudaEventCreate(start)");
    try {
      cuda_check(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
    } catch (...) {
      cudaEventDestroy(start_);
      throw;
    }
  }
  ~EventPair() {
    cudaEventDestroy(stop_);
    cudaEventDestroy(start_);
  }
  cudaEvent_t start() const { return start_; }
  cudaEvent_t stop() const { return stop_; }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace

MeasurementSummary summarize_samples(const std::vector<double>& samples_us) {
  if (samples_us.empty()) throw std::invalid_argument("samples must not be empty");
  MeasurementSummary summary;
  summary.raw_samples_us = samples_us;
  summary.min_us = *std::min_element(samples_us.begin(), samples_us.end());
  summary.mean_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
                    static_cast<double>(samples_us.size());
  double squared_sum = 0.0;
  for (const double sample : samples_us) {
    const double delta = sample - summary.mean_us;
    squared_sum += delta * delta;
  }
  summary.stddev_us = std::sqrt(squared_sum / static_cast<double>(samples_us.size()));
  summary.cv = summary.mean_us == 0.0 ? 0.0 : summary.stddev_us / summary.mean_us;
  summary.p50_us = percentile(samples_us, 0.50);
  summary.p90_us = percentile(samples_us, 0.90);
  summary.p95_us = percentile(samples_us, 0.95);
  return summary;
}

EnvironmentInfo collect_environment() {
  EnvironmentInfo info;
  int device = 0;
  cuda_check(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  int runtime_version = 0;
  int driver_version = 0;
  int core_clock_khz = 0;
  int memory_clock_khz = 0;
  cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  cuda_check(cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device),
             "cudaDeviceGetAttribute(clock)");
  cuda_check(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device),
             "cudaDeviceGetAttribute(memory clock)");

  info.fields["build_type"] = std::string(RAGGEDROUTE_BUILD_TYPE);
  info.fields["build_git_sha"] = std::string(RAGGEDROUTE_GIT_SHA);
  info.fields["build_git_dirty"] = std::string(RAGGEDROUTE_GIT_DIRTY);
  info.fields["cuda_compiler"] = std::string(RAGGEDROUTE_CUDA_COMPILER_VERSION);
  info.fields["cuda_runtime"] = static_cast<std::int64_t>(runtime_version);
  info.fields["cuda_driver"] = static_cast<std::int64_t>(driver_version);
  info.fields["device_ordinal"] = static_cast<std::int64_t>(device);
  info.fields["gpu_name"] = std::string(properties.name);
  info.fields["compute_capability"] =
      std::to_string(properties.major) + "." + std::to_string(properties.minor);
  info.fields["global_memory_bytes"] = static_cast<std::int64_t>(properties.totalGlobalMem);
  info.fields["sm_count"] = static_cast<std::int64_t>(properties.multiProcessorCount);
  info.fields["warp_size"] = static_cast<std::int64_t>(properties.warpSize);
  info.fields["core_clock_khz"] = static_cast<std::int64_t>(core_clock_khz);
  info.fields["memory_clock_khz"] = static_cast<std::int64_t>(memory_clock_khz);
  info.fields["memory_bus_width_bits"] = static_cast<std::int64_t>(properties.memoryBusWidth);
  info.fields["pci_domain_id"] = static_cast<std::int64_t>(properties.pciDomainID);
  info.fields["pci_bus_id"] = static_cast<std::int64_t>(properties.pciBusID);
  info.fields["pci_device_id"] = static_cast<std::int64_t>(properties.pciDeviceID);
  std::ostringstream uuid;
  uuid << std::hex << std::setfill('0');
  for (const char byte : properties.uuid.bytes) {
    uuid << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(byte));
  }
  info.fields["gpu_uuid"] = uuid.str();
  return info;
}

BenchmarkRecord run_benchmark(BenchmarkAdapter& adapter, const RunOptions& options,
                              cudaStream_t stream) {
  if (!adapter.supports(options.level)) {
    throw std::invalid_argument(adapter.operator_name() + " does not support " +
                                to_string(options.level));
  }
  if (options.warmup < 0 || options.samples < 1 || options.kernel_repeats < 1 ||
      options.process_run < 1) {
    throw std::invalid_argument("warmup/samples/repeats/process_run are invalid");
  }
  if (options.protocol == Protocol::kRelease) {
    if (std::string(RAGGEDROUTE_BUILD_TYPE) != "Release") {
      throw std::invalid_argument("release protocol requires a Release build");
    }
    if (std::string(RAGGEDROUTE_GIT_DIRTY) != "false") {
      throw std::invalid_argument(
          "release protocol requires a binary configured from a clean worktree");
    }
    if (options.expected_git_sha.empty() ||
        options.expected_git_sha.rfind(std::string(RAGGEDROUTE_GIT_SHA), 0) != 0) {
      throw std::invalid_argument(
          "release protocol build Git SHA does not match the orchestrator SHA");
    }
    if (options.warmup < 10 || options.samples < 20) {
      throw std::invalid_argument("release protocol requires warmup>=10 and samples>=20");
    }
    if (!options.validate_after_measurement || options.output_path.empty()) {
      throw std::invalid_argument("release protocol requires validation and a JSONL output path");
    }
  }
  const RepeatPolicy policy = adapter.repeat_policy(options.level);
  if (policy.max_repeats > 0 && options.kernel_repeats > policy.max_repeats) {
    throw std::invalid_argument("kernel_repeats exceeds adapter policy: " + policy.reason);
  }

  DeviceBuffer<std::uint32_t> scrub;
  if (options.cache_mode == CacheMode::kColdScrub) {
    if (options.cache_scrub_bytes < sizeof(std::uint32_t)) {
      throw std::invalid_argument("cold_scrub requires --cache-scrub-bytes >= 4");
    }
    scrub.resize(options.cache_scrub_bytes / sizeof(std::uint32_t));
    cuda_check(cudaMemsetAsync(scrub.data(), 0, scrub.bytes(), stream),
               "initialize cache scrub buffer");
    cuda_check(cudaStreamSynchronize(stream), "cache scrub initialization synchronization");
  }

  auto scrub_cache = [&]() {
    if (scrub.size() != 0) {
      cuda_check(ops::launch_cache_scrub(scrub.data(), scrub.size(), stream), "launch_cache_scrub");
      cuda_check(cudaStreamSynchronize(stream), "cold cache scrub sync");
    }
  };

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    scrub_cache();
    adapter.prepare_sample(options.level, stream);
    for (int repeat = 0; repeat < options.kernel_repeats; ++repeat) {
      adapter.enqueue(options.level, stream);
    }
  }
  cuda_check(cudaStreamSynchronize(stream), "warmup synchronization");

  EventPair events;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.samples));
  for (int sample = 0; sample < options.samples; ++sample) {
    scrub_cache();
    adapter.prepare_sample(options.level, stream);
    cuda_check(cudaStreamSynchronize(stream), "sample preparation synchronization");
    cuda_check(cudaEventRecord(events.start(), stream), "cudaEventRecord(start)");
    for (int repeat = 0; repeat < options.kernel_repeats; ++repeat) {
      adapter.enqueue(options.level, stream);
    }
    cuda_check(cudaEventRecord(events.stop(), stream), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(stop)");
    float milliseconds = 0.0F;
    cuda_check(cudaEventElapsedTime(&milliseconds, events.start(), events.stop()),
               "cudaEventElapsedTime");
    samples.push_back(static_cast<double>(milliseconds) * 1000.0 /
                      static_cast<double>(options.kernel_repeats));
  }

  ValidationResult validation{true, "validation disabled", std::nullopt, std::nullopt};
  if (options.validate_after_measurement) {
    adapter.prepare_sample(options.level, stream);
    adapter.enqueue(options.level, stream);
    cuda_check(cudaStreamSynchronize(stream), "validation invocation sync");
    validation = adapter.validate(stream);
    if (!validation.ok) {
      throw std::runtime_error("post-benchmark validation failed: " + validation.message);
    }
  }

  BenchmarkRecord record;
  record.timestamp_utc = now_utc();
  record.run_id = options.run_id;
  record.case_id = options.case_id;
  record.operator_name = adapter.operator_name();
  record.variant_name = adapter.variant_name();
  record.level = options.level;
  record.protocol = options.protocol;
  record.cache_mode = options.cache_mode;
  record.warmup = options.warmup;
  record.kernel_repeats = options.kernel_repeats;
  record.samples = options.samples;
  record.process_run = options.process_run;
  record.seed = options.seed;
  record.excluded_steps = adapter.excluded_steps(options.level);
  record.workspace_bytes = adapter.workspace_bytes();
  record.case_config = adapter.case_config();
  record.variant_config = adapter.variant_config();
  record.work = adapter.work_estimate(options.level);
  record.environment = collect_environment();
  record.timing = summarize_samples(samples);
  record.validation = validation;
  return record;
}

std::string record_to_json(const BenchmarkRecord& record) {
  std::ostringstream out;
  out << '{';
  auto string_field = [&](const char* name, const std::string& value, bool comma = true) {
    write_quoted(out, name);
    out << ':';
    write_quoted(out, value);
    if (comma) out << ',';
  };
  auto int_field = [&](const char* name, std::int64_t value, bool comma = true) {
    write_quoted(out, name);
    out << ':' << value;
    if (comma) out << ',';
  };
  auto double_field = [&](const char* name, double value, bool comma = true) {
    write_quoted(out, name);
    out << ':';
    write_number(out, value);
    if (comma) out << ',';
  };

  string_field("schema_version", record.schema_version);
  string_field("timestamp_utc", record.timestamp_utc);
  string_field("run_id", record.run_id);
  string_field("case_id", record.case_id);
  string_field("operator", record.operator_name);
  string_field("variant", record.variant_name);
  string_field("measurement_level", to_string(record.level));
  string_field("protocol", to_string(record.protocol));
  string_field("cache_mode", to_string(record.cache_mode));
  int_field("warmup", record.warmup);
  int_field("kernel_repeats", record.kernel_repeats);
  int_field("samples", record.samples);
  int_field("process_run", record.process_run);
  int_field("seed", static_cast<std::int64_t>(record.seed));
  int_field("workspace_bytes", static_cast<std::int64_t>(record.workspace_bytes));

  write_quoted(out, "excluded_steps");
  out << ":[";
  for (std::size_t i = 0; i < record.excluded_steps.size(); ++i) {
    if (i != 0) out << ',';
    write_quoted(out, record.excluded_steps[i]);
  }
  out << "],";

  write_quoted(out, "case_config");
  out << ':';
  write_field_map(out, record.case_config);
  out << ',';
  write_quoted(out, "variant_config");
  out << ':';
  write_field_map(out, record.variant_config);
  out << ',';
  write_quoted(out, "environment");
  out << ':';
  write_field_map(out, record.environment.fields);
  out << ',';

  write_quoted(out, "work");
  out << ":{";
  double_field("logical_bytes", record.work.logical_bytes);
  double_field("flops", record.work.flops);
  double_field("effective_gbps_batch_p50",
               record.timing.p50_us == 0.0
                   ? 0.0
                   : record.work.logical_bytes / record.timing.p50_us / 1000.0);
  double_field("tflops_batch_p50", record.timing.p50_us == 0.0
                                       ? 0.0
                                       : record.work.flops / record.timing.p50_us / 1.0e6);
  write_quoted(out, "operator_metrics");
  out << ':';
  write_field_map(out, record.work.operator_metrics);
  out << "},";

  write_quoted(out, "timing");
  out << ":{";
  string_field("sample_semantics", record.kernel_repeats == 1
                                       ? "single_launch"
                                       : "event_batch_elapsed_divided_by_repeats");
  double_field("batch_mean_us_mean", record.timing.mean_us);
  double_field("batch_mean_us_stddev", record.timing.stddev_us);
  double_field("cv", record.timing.cv);
  double_field("batch_mean_us_min", record.timing.min_us);
  double_field("batch_mean_us_p50", record.timing.p50_us);
  double_field("batch_mean_us_p90", record.timing.p90_us);
  double_field("batch_mean_us_p95", record.timing.p95_us);
  write_quoted(out, "raw_batch_mean_samples_us");
  out << ":[";
  for (std::size_t i = 0; i < record.timing.raw_samples_us.size(); ++i) {
    if (i != 0) out << ',';
    write_number(out, record.timing.raw_samples_us[i]);
  }
  out << "]},";

  write_quoted(out, "validation");
  out << ":{";
  write_quoted(out, "ok");
  out << ':' << (record.validation.ok ? "true" : "false") << ',';
  string_field("message", record.validation.message);
  write_quoted(out, "max_abs_error");
  out << ':';
  if (record.validation.max_abs_error)
    write_number(out, *record.validation.max_abs_error);
  else
    out << "null";
  out << ',';
  write_quoted(out, "max_rel_error");
  out << ':';
  if (record.validation.max_rel_error)
    write_number(out, *record.validation.max_rel_error);
  else
    out << "null";
  out << "}}";
  return out.str();
}

void append_jsonl(const std::string& path, const BenchmarkRecord& record) {
  const std::filesystem::path output(path);
  if (output.has_parent_path()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::ofstream stream(path, std::ios::app);
  if (!stream) throw std::runtime_error("cannot open JSONL output: " + path);
  stream << record_to_json(record) << '\n';
  if (!stream) throw std::runtime_error("failed to write JSONL output: " + path);
}

}  // namespace raggedroute::benchmark
