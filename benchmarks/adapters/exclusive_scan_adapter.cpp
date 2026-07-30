#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

class ExclusiveScanAdapter final : public BenchmarkAdapter {
 public:
  std::string operator_name() const override { return "exclusive_scan"; }
  std::string variant_name() const override { return "cuda_naive"; }
  std::string description() const override {
    return "Single-thread int32 exclusive scan for tiny expert counts";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    experts_ = get_int_option(options, "E", 16);
    if (experts_ > 64) throw std::invalid_argument("current scan adapter supports E<=64");
    routes_ = get_int_option(options, "R", 256);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    const auto ids = make_route_ids(routes_, 1, experts_, distribution_, zipf_s_, seed);
    counts_host_ = counts_from_ids(ids, experts_);
    expected_ = offsets_from_counts(counts_host_);
    counts_.resize(counts_host_.size());
    offsets_.resize(expected_.size());
    counts_.copy_from_host(counts_host_, stream);
  }
  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(ops::launch_exclusive_scan_naive(counts_.data(), offsets_.data(), experts_, stream),
                 "launch_exclusive_scan_naive");
      return;
    }
    ExclusiveScanArgs args;
    args.counts = counts_.data();
    args.offsets = offsets_.data();
    args.experts = experts_;
    operator_check(exclusive_scan(args, make_runtime_context(stream, architecture_)),
                   "exclusive_scan operator");
  }
  ValidationResult validate(cudaStream_t stream) override {
    const auto actual = offsets_.copy_to_host(stream);
    if (actual != expected_) {
      return {false, "exclusive offsets differ from CPU reference", {}, {}};
    }
    return {true, "offsets match CPU reference", 0.0, 0.0};
  }
  FieldMap case_config() const override {
    return {{"E", static_cast<std::int64_t>(experts_)},
            {"R", static_cast<std::int64_t>(routes_)},
            {"count_dtype", std::string("int32")},
            {"distribution", distribution_},
            {"zipf_s", zipf_s_}};
  }
  FieldMap variant_config() const override {
    return {{"threads", static_cast<std::int64_t>(1)}, {"algorithm", std::string("sequential")}};
  }
  WorkEstimate work_estimate() const override {
    WorkEstimate work;
    work.logical_bytes = sizeof(std::int32_t) * static_cast<double>(experts_ + experts_ + 1);
    work.operator_metrics["integer_additions"] = static_cast<std::int64_t>(experts_);
    return work;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"count_generation", "h2d_copy", "workspace_allocation"};
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"E", "R", "distribution", "zipf_s"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name))
        throw std::invalid_argument("unknown exclusive_scan param: " + name);
    }
  }
  int experts_ = 0, routes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<std::int32_t> counts_host_, expected_;
  DeviceBuffer<std::int32_t> counts_, offsets_;
};

}  // namespace

AdapterPtr make_exclusive_scan_adapter() { return std::make_unique<ExclusiveScanAdapter>(); }

}  // namespace raggedroute::benchmark
