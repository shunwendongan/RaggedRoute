#include <cmath>
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

class TopKGateAdapter final : public BenchmarkAdapter {
 public:
  std::string operator_name() const override { return "topk_gate"; }
  std::string variant_name() const override { return "cuda_naive"; }
  std::string description() const override {
    return "One CUDA thread per row with deterministic Top-2 selected-softmax";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16, 2);
    if (experts_ > 64) throw std::invalid_argument("current Top-K adapter supports 2<=E<=64");
    input_mode_ = get_option(options, "input_mode", "random");
    logits_host_ =
        make_random_floats(static_cast<std::size_t>(tokens_) * experts_, seed, -8.0F, 8.0F);
    if (input_mode_ == "ties") {
      std::fill(logits_host_.begin(), logits_host_.end(), 1.0F);
    } else if (input_mode_ == "all_nan") {
      std::fill(logits_host_.begin(), logits_host_.end(), std::numeric_limits<float>::quiet_NaN());
    } else if (input_mode_ == "infinities") {
      for (int token = 0; token < tokens_; ++token) {
        logits_host_[static_cast<std::size_t>(token) * experts_] =
            std::numeric_limits<float>::infinity();
        logits_host_[static_cast<std::size_t>(token) * experts_ + 1] =
            -std::numeric_limits<float>::infinity();
      }
    } else if (input_mode_ != "random") {
      throw std::invalid_argument("unsupported topk input_mode: " + input_mode_);
    }
    build_reference();
    logits_.resize(logits_host_.size());
    ids_.resize(ids_expected_.size());
    weights_.resize(weights_expected_.size());
    logits_.copy_from_host(logits_host_, stream);
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel, cudaStream_t stream) override {
    cuda_check(ops::launch_topk_gate_naive(logits_.data(), ids_.data(), weights_.data(), tokens_,
                                           experts_, stream),
               "launch_topk_gate_naive");
  }
  ValidationResult validate(cudaStream_t stream) override {
    const auto ids = ids_.copy_to_host(stream);
    const auto weights = weights_.copy_to_host(stream);
    if (ids != ids_expected_) {
      return {false, "Top-2 expert ids differ from deterministic reference", {}, {}};
    }
    return compare_floats(weights, weights_expected_, 1.0e-6, 1.0e-6);
  }
  FieldMap case_config() const override {
    return {{"T", static_cast<std::int64_t>(tokens_)},
            {"E", static_cast<std::int64_t>(experts_)},
            {"top_k", static_cast<std::int64_t>(2)},
            {"dtype", std::string("fp32")},
            {"normalization", std::string("selected_softmax")},
            {"tie_break", std::string("lower_expert_id")},
            {"nan_policy", std::string("negative_infinity_all_nan_0_1")},
            {"input_mode", input_mode_}};
  }
  FieldMap variant_config() const override {
    return {{"rows_per_thread", static_cast<std::int64_t>(1)},
            {"threads_per_block", static_cast<std::int64_t>(256)}};
  }
  WorkEstimate work_estimate() const override {
    WorkEstimate work;
    work.logical_bytes =
        sizeof(float) * static_cast<double>(tokens_) * experts_ +
        (sizeof(float) + sizeof(std::int32_t)) * static_cast<double>(tokens_) * 2.0;
    work.operator_metrics["rows"] = static_cast<std::int64_t>(tokens_);
    work.operator_metrics["comparisons_lower_bound"] =
        static_cast<std::int64_t>(tokens_) * experts_;
    return work;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"input_generation", "cpu_reference", "h2d_copy", "workspace_allocation"};
  }

 private:
  void build_reference() {
    top2_selected_softmax_reference(logits_host_, tokens_, experts_, ids_expected_,
                                    weights_expected_);
  }
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "input_mode"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown topk_gate param: " + name);
    }
  }
  int tokens_ = 0, experts_ = 0;
  std::string input_mode_;
  std::vector<float> logits_host_, weights_expected_;
  std::vector<std::int32_t> ids_expected_;
  DeviceBuffer<float> logits_, weights_;
  DeviceBuffer<std::int32_t> ids_;
};

}  // namespace

AdapterPtr make_topk_gate_adapter() { return std::make_unique<TopKGateAdapter>(); }

}  // namespace raggedroute::benchmark
