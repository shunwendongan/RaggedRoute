#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raggedroute/benchmark/types.h"
#include "raggedroute/correctness/framework.h"
#include "raggedroute/dispatch.h"
#include "raggedroute/operators.h"

namespace raggedroute::benchmark {

inline void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

inline void operator_check(const ::raggedroute::Status& status, const char* operation) {
  if (!status.ok()) {
    std::string message(operation);
    message += ": ";
    message += status.message == nullptr ? "operator error" : status.message;
    if (status.cuda_error != cudaSuccess) {
      message += ": ";
      message += cudaGetErrorString(status.cuda_error);
    }
    throw std::runtime_error(message);
  }
}

inline ::raggedroute::DeviceArchitecture current_device_architecture() {
  ::raggedroute::DeviceArchitecture architecture = ::raggedroute::DeviceArchitecture::kOther;
  operator_check(::raggedroute::query_current_device_architecture(&architecture),
                 "query current CUDA architecture");
  return architecture;
}

inline ::raggedroute::RuntimeContext make_runtime_context(
    cudaStream_t stream, ::raggedroute::DeviceArchitecture architecture, void* workspace = nullptr,
    std::size_t workspace_bytes = 0) {
  ::raggedroute::RuntimeContext context;
  context.stream = stream;
  context.workspace = workspace;
  context.workspace_bytes = workspace_bytes;
  context.architecture = architecture;
  return context;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { resize(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }
  ~DeviceBuffer() { reset(); }

  void resize(std::size_t count) {
    reset();
    count_ = count;
    if (count_ != 0) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "cudaMalloc");
    }
  }

  void reset() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0;
  }

  void copy_from_host(const std::vector<T>& host, cudaStream_t stream) {
    if (host.size() != count_) {
      throw std::invalid_argument("host/device buffer size mismatch");
    }
    if (!host.empty()) {
      cuda_check(cudaMemcpyAsync(data_, host.data(), host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync H2D");
    }
  }

  std::vector<T> copy_to_host(cudaStream_t stream) const {
    std::vector<T> host(count_);
    if (count_ != 0) {
      cuda_check(
          cudaMemcpyAsync(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync D2H");
      cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    return host;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

inline std::string get_option(const OptionMap& options, const std::string& name,
                              const std::string& default_value) {
  const auto it = options.find(name);
  return it == options.end() ? default_value : it->second;
}

inline int get_int_option(const OptionMap& options, const std::string& name, int default_value,
                          int minimum = 1) {
  const auto text = get_option(options, name, std::to_string(default_value));
  std::size_t consumed = 0;
  const long long parsed = std::stoll(text, &consumed);
  if (consumed != text.size() || parsed < minimum || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("invalid --param " + name + "=" + text);
  }
  return static_cast<int>(parsed);
}

inline double get_double_option(const OptionMap& options, const std::string& name,
                                double default_value, double minimum = 0.0) {
  const auto text = get_option(options, name, std::to_string(default_value));
  std::size_t consumed = 0;
  const double parsed = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(parsed) || parsed < minimum) {
    throw std::invalid_argument("invalid --param " + name + "=" + text);
  }
  return parsed;
}

inline bool get_bool_option(const OptionMap& options, const std::string& name, bool default_value) {
  const auto text = get_option(options, name, default_value ? "true" : "false");
  if (text == "true" || text == "1") return true;
  if (text == "false" || text == "0") return false;
  throw std::invalid_argument("invalid boolean --param " + name + "=" + text);
}

inline int checked_int_product(int left, int right, const std::string& label) {
  if (left < 0 || right < 0 || (right != 0 && left > std::numeric_limits<int>::max() / right)) {
    throw std::invalid_argument(label + " exceeds int32 indexing range");
  }
  return left * right;
}

inline std::vector<int> parse_positive_int_list(const std::string& text,
                                                const std::string& label) {
  if (text.empty()) throw std::invalid_argument(label + " must not be empty");
  std::vector<int> values;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
    if (item.empty()) throw std::invalid_argument(label + " contains an empty item");
    std::size_t consumed = 0;
    const long long value = std::stoll(item, &consumed);
    if (consumed != item.size() || value < 1 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(label + " contains an invalid positive integer: " + item);
    }
    values.push_back(static_cast<int>(value));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return values;
}

inline std::vector<float> make_random_floats(std::size_t count, std::uint64_t seed,
                                             float low = -1.0F, float high = 1.0F) {
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<float> distribution(low, high);
  std::vector<float> values(count);
  for (auto& value : values) value = distribution(engine);
  return values;
}

inline std::vector<std::int32_t> make_route_ids(int tokens, int top_k, int experts,
                                                const std::string& distribution, double zipf_s,
                                                std::uint64_t seed) {
  if (tokens < 1 || top_k < 1 || experts < top_k) {
    throw std::invalid_argument("route shape requires tokens>=1 and experts>=top_k>=1");
  }
  if (distribution != "uniform" && distribution != "zipf" && distribution != "single_hot" &&
      distribution != "round_robin" && distribution != "duplicate_route") {
    throw std::invalid_argument("unsupported distribution: " + distribution);
  }

  std::mt19937_64 engine(seed);
  std::uniform_int_distribution<int> uniform(0, experts - 1);
  std::vector<double> probabilities(static_cast<std::size_t>(experts));
  for (int expert = 0; expert < experts; ++expert) {
    probabilities[static_cast<std::size_t>(expert)] =
        std::pow(static_cast<double>(expert + 1), -zipf_s);
  }
  std::discrete_distribution<int> zipf(probabilities.begin(), probabilities.end());
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * top_k);

  for (int token = 0; token < tokens; ++token) {
    for (int rank = 0; rank < top_k; ++rank) {
      int candidate = 0;
      if (distribution == "duplicate_route") {
        candidate = 0;
      } else if (distribution == "single_hot") {
        candidate = rank;
      } else if (distribution == "round_robin") {
        candidate = (token + rank) % experts;
      } else {
        do {
          candidate = distribution == "zipf" ? zipf(engine) : uniform(engine);
        } while (std::find(ids.begin() + static_cast<std::ptrdiff_t>(token * top_k),
                           ids.begin() + static_cast<std::ptrdiff_t>(token * top_k + rank),
                           candidate) !=
                 ids.begin() + static_cast<std::ptrdiff_t>(token * top_k + rank));
      }
      ids[static_cast<std::size_t>(token * top_k + rank)] = candidate;
    }
  }
  return ids;
}

struct RouteTraceFrame {
  std::string trace_id;
  std::string source_kind;
  std::string frame_id;
  int tokens = 0;
  int experts = 0;
  int top_k = 0;
  int frame_count = 0;
  std::vector<std::int32_t> expert_ids;
};

inline bool is_portable_route_trace_id(const std::string& value) {
  if (value.empty()) return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) != 0 || character == '.' || character == '_' || character == '-';
  });
}

inline RouteTraceFrame load_route_trace_frame(const std::string& path, int requested_frame) {
  if (path.empty() || requested_frame < 0) {
    throw std::invalid_argument("route_trace_path and route_trace_frame are invalid");
  }
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("cannot open route trace: " + path);
  std::string schema;
  RouteTraceFrame result;
  if (!std::getline(input, schema) || schema != "raggedroute.route_trace.v1" ||
      !std::getline(input, result.trace_id) || !std::getline(input, result.source_kind) ||
      !(input >> result.tokens >> result.experts >> result.top_k >> result.frame_count)) {
    throw std::invalid_argument("invalid route trace header: " + path);
  }
  if (!is_portable_route_trace_id(result.trace_id) ||
      (result.source_kind != "production" && result.source_kind != "captured" &&
       result.source_kind != "synthetic_fixture") ||
      result.tokens < 1 || result.experts < 1 || result.experts > 64 ||
      result.top_k < 1 || result.top_k > result.experts || result.frame_count < 1 ||
      requested_frame >= result.frame_count) {
    throw std::invalid_argument("invalid route trace metadata: " + path);
  }
  const std::size_t route_pairs =
      static_cast<std::size_t>(result.tokens) * static_cast<std::size_t>(result.top_k);
  std::set<std::string> frame_ids;
  for (int frame = 0; frame < result.frame_count; ++frame) {
    std::string frame_id;
    if (!(input >> frame_id) || !is_portable_route_trace_id(frame_id) ||
        !frame_ids.insert(frame_id).second) {
      throw std::invalid_argument("route trace frame id is invalid or duplicated: " + path);
    }
    std::vector<std::int32_t> ids(route_pairs);
    for (std::size_t route = 0; route < route_pairs; ++route) {
      if (!(input >> ids[route]) || ids[route] < 0 || ids[route] >= result.experts) {
        throw std::invalid_argument("route trace contains an invalid expert id: " + path);
      }
    }
    for (int token = 0; token < result.tokens; ++token) {
      const auto begin = ids.begin() + static_cast<std::ptrdiff_t>(token * result.top_k);
      const auto end = begin + result.top_k;
      std::vector<std::int32_t> selected(begin, end);
      std::sort(selected.begin(), selected.end());
      if (std::adjacent_find(selected.begin(), selected.end()) != selected.end()) {
        throw std::invalid_argument("route trace repeats an expert within one token: " + path);
      }
    }
    if (frame == requested_frame) {
      result.frame_id = std::move(frame_id);
      result.expert_ids = std::move(ids);
    }
  }
  std::string trailing;
  if (input >> trailing) throw std::invalid_argument("route trace has trailing data: " + path);
  return result;
}

inline std::vector<std::int32_t> counts_from_ids(const std::vector<std::int32_t>& ids,
                                                 int experts) {
  std::vector<std::int32_t> counts(static_cast<std::size_t>(experts), 0);
  for (const auto id : ids) {
    if (id < 0 || id >= experts) {
      throw std::invalid_argument("route id out of range");
    }
    ++counts[static_cast<std::size_t>(id)];
  }
  return counts;
}

inline std::vector<std::int32_t> offsets_from_counts(const std::vector<std::int32_t>& counts) {
  std::vector<std::int32_t> offsets(counts.size() + 1, 0);
  for (std::size_t i = 0; i < counts.size(); ++i) {
    offsets[i + 1] = offsets[i] + counts[i];
  }
  return offsets;
}

inline ValidationResult compare_floats(const std::vector<float>& actual,
                                       const std::vector<float>& expected, double atol,
                                       double rtol) {
  correctness::CaseDescriptor descriptor;
  descriptor.case_id = "benchmark_post_measurement_validation";
  descriptor.operator_name = "benchmark_adapter";
  descriptor.variant_name = "cuda_naive";
  std::vector<double> actual_double(actual.begin(), actual.end());
  std::vector<double> expected_double(expected.begin(), expected.end());
  const correctness::CheckReport report =
      correctness::compare_floating(descriptor, actual_double, expected_double, atol, rtol);
  if (report.ok()) {
    return {true, "matched reference", report.numeric.max_abs_error, report.numeric.max_rel_error};
  }
  const std::string message =
      report.failures.empty() ? "correctness comparison failed" : report.failures.front().message;
  return {false, message, report.numeric.max_abs_error, report.numeric.max_rel_error};
}

inline void top2_selected_softmax_reference(const std::vector<float>& logits, int tokens,
                                            int experts, std::vector<std::int32_t>& ids,
                                            std::vector<float>& weights) {
  if (tokens < 1 || experts < 2 || logits.size() != static_cast<std::size_t>(tokens) * experts) {
    throw std::invalid_argument("invalid Top-2 reference shape");
  }
  ids.assign(static_cast<std::size_t>(tokens) * 2, 0);
  weights.assign(static_cast<std::size_t>(tokens) * 2, 0.0F);
  auto precedes = [](float value, int expert, float incumbent, int incumbent_expert) {
    return incumbent_expert < 0 || value > incumbent ||
           (value == incumbent && expert < incumbent_expert);
  };
  for (int token = 0; token < tokens; ++token) {
    float first = -std::numeric_limits<float>::infinity();
    float second = -std::numeric_limits<float>::infinity();
    int first_id = -1;
    int second_id = -1;
    bool saw_non_nan = false;
    for (int expert = 0; expert < experts; ++expert) {
      const float raw = logits[static_cast<std::size_t>(token) * experts + expert];
      const bool is_nan = std::isnan(raw);
      const float value = is_nan ? -std::numeric_limits<float>::infinity() : raw;
      saw_non_nan = saw_non_nan || !is_nan;
      if (precedes(value, expert, first, first_id)) {
        second = first;
        second_id = first_id;
        first = value;
        first_id = expert;
      } else if (precedes(value, expert, second, second_id)) {
        second = value;
        second_id = expert;
      }
    }
    float first_weight = 0.5F;
    float second_weight = 0.5F;
    if (!saw_non_nan) {
      first_id = 0;
      second_id = 1;
    } else if (first != second) {
      if (first == std::numeric_limits<float>::infinity() ||
          second == -std::numeric_limits<float>::infinity()) {
        first_weight = 1.0F;
        second_weight = 0.0F;
      } else {
        const float relative = std::exp(second - first);
        first_weight = 1.0F / (1.0F + relative);
        second_weight = relative * first_weight;
      }
    }
    const auto base = static_cast<std::size_t>(token) * 2;
    ids[base] = first_id;
    ids[base + 1] = second_id;
    weights[base] = first_weight;
    weights[base + 1] = second_weight;
  }
}

}  // namespace raggedroute::benchmark
