#include <algorithm>
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

class TokenPermuteAdapter final : public BenchmarkAdapter {
 public:
  std::string operator_name() const override { return "token_permute"; }
  std::string variant_name() const override { return "cuda_naive"; }
  std::string description() const override {
    return "Atomic-cursor placement with scalar FP32 row copy";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel level) const override {
    if (level == MeasurementLevel::kKernelBody) {
      return {1, "L1 token permute mutates cursor state; use one launch per event sample"};
    }
    return {};
  }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16);
    top_k_ = get_int_option(options, "top_k", 2);
    hidden_ = get_int_option(options, "K", 256);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    materialize_sorted_route_ = get_bool_option(options, "materialize_sorted_route", true);
    if (experts_ < top_k_ || experts_ > 64) {
      throw std::invalid_argument("current permute adapter requires top_k<=E<=64");
    }
    route_pairs_ = checked_int_product(tokens_, top_k_, "R=T*top_k");

    host_x_ = make_random_floats(static_cast<std::size_t>(tokens_) * hidden_, seed);
    host_ids_ = make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed + 1);
    host_counts_ = counts_from_ids(host_ids_, experts_);
    host_offsets_ = offsets_from_counts(host_counts_);

    x_.resize(host_x_.size());
    ids_.resize(host_ids_.size());
    offsets_.resize(host_offsets_.size());
    cursors_.resize(static_cast<std::size_t>(experts_));
    x_permuted_.resize(static_cast<std::size_t>(route_pairs_) * hidden_);
    route_pos_.resize(static_cast<std::size_t>(route_pairs_));
    if (materialize_sorted_route_) {
      sorted_route_.resize(static_cast<std::size_t>(route_pairs_));
    }
    x_.copy_from_host(host_x_, stream);
    ids_.copy_from_host(host_ids_, stream);
    offsets_.copy_from_host(host_offsets_, stream);
    cuda_check(cudaMemsetAsync(cursors_.data(), 0, cursors_.bytes(), stream),
               "initialize permute cursors");
  }

  void prepare_sample(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kKernelBody) reset_cursors(stream);
  }

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(ops::launch_token_permute_naive(
                     x_.data(), ids_.data(), offsets_.data(), cursors_.data(), x_permuted_.data(),
                     route_pos_.data(), materialize_sorted_route_ ? sorted_route_.data() : nullptr,
                     tokens_, top_k_, hidden_, stream),
                 "launch_token_permute_naive");
      return;
    }
    TokenPermuteArgs args;
    args.x.data = x_.data();
    args.expert_ids = ids_.data();
    args.offsets = offsets_.data();
    args.x_permuted.data = x_permuted_.data();
    args.route_pos = route_pos_.data();
    args.sorted_route = materialize_sorted_route_ ? sorted_route_.data() : nullptr;
    args.tokens = tokens_;
    args.experts = experts_;
    args.top_k = top_k_;
    args.hidden = hidden_;
    operator_check(token_permute(args, make_runtime_context(stream, architecture_, cursors_.data(),
                                                            cursors_.bytes())),
                   "token_permute operator");
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto positions = route_pos_.copy_to_host(stream);
    const auto output = x_permuted_.copy_to_host(stream);
    std::vector<std::int32_t> reverse;
    if (materialize_sorted_route_) reverse = sorted_route_.copy_to_host(stream);
    std::vector<bool> seen(static_cast<std::size_t>(route_pairs_), false);
    for (int route = 0; route < route_pairs_; ++route) {
      const int position = positions[static_cast<std::size_t>(route)];
      const int expert = host_ids_[static_cast<std::size_t>(route)];
      if (position < host_offsets_[static_cast<std::size_t>(expert)] ||
          position >= host_offsets_[static_cast<std::size_t>(expert + 1)]) {
        return {false, "route_pos escaped its expert segment", {}, {}};
      }
      if (seen[static_cast<std::size_t>(position)]) {
        return {false, "duplicate route_pos destination", {}, {}};
      }
      seen[static_cast<std::size_t>(position)] = true;
      if (materialize_sorted_route_ && reverse[static_cast<std::size_t>(position)] != route) {
        return {false, "sorted_route is not the inverse of route_pos", {}, {}};
      }
      const int token = route / top_k_;
      for (int column = 0; column < hidden_; ++column) {
        const float actual = output[static_cast<std::size_t>(position) * hidden_ + column];
        const float expected = host_x_[static_cast<std::size_t>(token) * hidden_ + column];
        if (actual != expected) {
          return {false, "permuted activation mismatch", {}, {}};
        }
      }
    }
    return {true, "mapping is bijective and rows match input", 0.0, 0.0};
  }

  FieldMap case_config() const override {
    return {{"T", static_cast<std::int64_t>(tokens_)},
            {"E", static_cast<std::int64_t>(experts_)},
            {"top_k", static_cast<std::int64_t>(top_k_)},
            {"K", static_cast<std::int64_t>(hidden_)},
            {"R", static_cast<std::int64_t>(route_pairs_)},
            {"dtype", std::string("fp32")},
            {"distribution", distribution_},
            {"zipf_s", zipf_s_},
            {"materialize_sorted_route", materialize_sorted_route_},
            {"placement_order", std::string("unstable_atomic_cursor")}};
  }
  FieldMap variant_config() const override {
    return {{"threads_per_route", static_cast<std::int64_t>(128)},
            {"copy", std::string("scalar")},
            {"cursor", std::string("global_atomic")}};
  }
  WorkEstimate work_estimate(MeasurementLevel level) const override {
    WorkEstimate work;
    work.logical_bytes = 2.0 * sizeof(float) * route_pairs_ * hidden_ +
                         2.0 * sizeof(std::int32_t) * route_pairs_ +
                         (materialize_sorted_route_ ? sizeof(std::int32_t) * route_pairs_ : 0.0);
    work.operator_metrics["copied_rows"] = static_cast<std::int64_t>(route_pairs_);
    if (level == MeasurementLevel::kOperatorSteady) {
      work.logical_bytes += static_cast<double>(cursors_.bytes());
      work.operator_metrics["cursor_reset_bytes"] = static_cast<std::int64_t>(cursors_.bytes());
    }
    return work;
  }
  std::size_t workspace_bytes() const override { return cursors_.bytes(); }
  std::vector<std::string> excluded_steps(MeasurementLevel level) const override {
    std::vector<std::string> excluded = {"input_generation", "h2d_copy", "workspace_allocation"};
    if (level == MeasurementLevel::kKernelBody) excluded.push_back("cursor_reset");
    return excluded;
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {
        "T", "E", "top_k", "K", "distribution", "zipf_s", "materialize_sorted_route"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown token_permute param: " + name);
    }
  }
  void reset_cursors(cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(cursors_.data(), 0, cursors_.bytes(), stream),
               "reset permute cursors");
  }

  int tokens_ = 0, experts_ = 0, top_k_ = 0, hidden_ = 0, route_pairs_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  bool materialize_sorted_route_ = true;
  std::string distribution_;
  std::vector<float> host_x_;
  std::vector<std::int32_t> host_ids_, host_counts_, host_offsets_;
  DeviceBuffer<float> x_, x_permuted_;
  DeviceBuffer<std::int32_t> ids_, offsets_, cursors_, route_pos_, sorted_route_;
};

}  // namespace

AdapterPtr make_token_permute_adapter() { return std::make_unique<TokenPermuteAdapter>(); }

}  // namespace raggedroute::benchmark
