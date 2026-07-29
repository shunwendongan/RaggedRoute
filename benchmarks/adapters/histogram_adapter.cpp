#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

class HistogramAdapter final : public BenchmarkAdapter {
 public:
  std::string operator_name() const override { return "histogram"; }
  std::string variant_name() const override { return "cuda_naive"; }
  std::string description() const override { return "One global atomicAdd per route pair"; }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel level) const override {
    if (level != MeasurementLevel::kKernelBody || max_count_ == 0) return {};
    const int cap = std::numeric_limits<std::int32_t>::max() / max_count_;
    return {cap, "L1 histogram accumulates counts across batched launches"};
  }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16);
    top_k_ = get_int_option(options, "top_k", 2);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    if (experts_ < top_k_ || experts_ > 64) {
      throw std::invalid_argument("current histogram adapter requires top_k<=E<=64");
    }
    (void)checked_int_product(tokens_, top_k_, "R=T*top_k");
    ids_host_ = make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed);
    expected_ = counts_from_ids(ids_host_, experts_);
    max_count_ = *std::max_element(expected_.begin(), expected_.end());
    ids_.resize(ids_host_.size());
    counts_.resize(expected_.size());
    ids_.copy_from_host(ids_host_, stream);
    reset_counts(stream);
  }
  void prepare_sample(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kKernelBody) reset_counts(stream);
  }
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kOperatorSteady) reset_counts(stream);
    cuda_check(ops::launch_histogram_naive(ids_.data(), counts_.data(),
                                           static_cast<int>(ids_host_.size()), experts_, stream),
               "launch_histogram_naive");
  }
  ValidationResult validate(cudaStream_t stream) override {
    const auto actual = counts_.copy_to_host(stream);
    if (actual != expected_) {
      return {false, "expert counts differ from CPU bincount", {}, {}};
    }
    return {true, "counts match CPU bincount", 0.0, 0.0};
  }
  FieldMap case_config() const override {
    return {{"T", static_cast<std::int64_t>(tokens_)},
            {"E", static_cast<std::int64_t>(experts_)},
            {"top_k", static_cast<std::int64_t>(top_k_)},
            {"R", static_cast<std::int64_t>(ids_host_.size())},
            {"count_dtype", std::string("int32")},
            {"distribution", distribution_},
            {"zipf_s", zipf_s_},
            {"max_count", static_cast<std::int64_t>(max_count_)}};
  }
  FieldMap variant_config() const override {
    return {{"atomic_scope", std::string("device")},
            {"privatization", false},
            {"threads_per_block", static_cast<std::int64_t>(256)}};
  }
  WorkEstimate work_estimate() const override {
    WorkEstimate work;
    work.logical_bytes =
        sizeof(std::int32_t) * static_cast<double>(ids_host_.size() + expected_.size());
    work.operator_metrics["global_atomic_operations"] = static_cast<std::int64_t>(ids_host_.size());
    return work;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel level) const override {
    std::vector<std::string> excluded = {"route_generation", "h2d_copy", "workspace_allocation"};
    if (level == MeasurementLevel::kKernelBody) excluded.push_back("counts_reset");
    return excluded;
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "top_k", "distribution", "zipf_s"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown histogram param: " + name);
    }
  }
  void reset_counts(cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(counts_.data(), 0, counts_.bytes(), stream),
               "reset histogram counts");
  }
  int tokens_ = 0, experts_ = 0, top_k_ = 0, max_count_ = 0;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<std::int32_t> ids_host_, expected_;
  DeviceBuffer<std::int32_t> ids_, counts_;
};

}  // namespace

AdapterPtr make_histogram_adapter() { return std::make_unique<HistogramAdapter>(); }

}  // namespace raggedroute::benchmark
