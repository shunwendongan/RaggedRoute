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

class ChainAdapter final : public BenchmarkAdapter {
 public:
  explicit ChainAdapter(bool include_router_projection)
      : include_router_projection_(include_router_projection) {}

  std::string operator_name() const override {
    return include_router_projection_ ? "chain_from_tokens" : "chain_from_logits";
  }
  std::string variant_name() const override { return "cuda_naive"; }
  std::string description() const override {
    return include_router_projection_ ? "Seven-operator token-to-output baseline chain"
                                      : "Six-operator logits-to-output baseline chain";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kChainSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    tokens_ = get_int_option(options, "T", 64);
    experts_ = get_int_option(options, "E", 8, 2);
    if (experts_ > 64) throw std::invalid_argument("current chain adapters support E<=64");
    hidden_ = get_int_option(options, "K", 32);
    output_ = get_int_option(options, "N", 32);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    route_pairs_ = checked_int_product(tokens_, 2, "R=T*2");
    (void)checked_int_product(tokens_, output_, "T*N");

    x_host_ = make_random_floats(static_cast<std::size_t>(tokens_) * hidden_, seed);
    expert_weights_host_ =
        make_random_floats(static_cast<std::size_t>(experts_) * hidden_ * output_, seed + 1);
    logits_host_.assign(static_cast<std::size_t>(tokens_) * experts_, 0.0F);
    if (include_router_projection_) {
      if (distribution_ != "uniform") {
        throw std::invalid_argument(
            "chain_from_tokens currently labels random router output as uniform; "
            "use chain_from_logits for controlled skew");
      }
      router_weights_host_ =
          make_random_floats(static_cast<std::size_t>(hidden_) * experts_, seed + 2);
      for (int token = 0; token < tokens_; ++token) {
        for (int expert = 0; expert < experts_; ++expert) {
          float sum = 0.0F;
          for (int inner = 0; inner < hidden_; ++inner) {
            sum += x_host_[static_cast<std::size_t>(token) * hidden_ + inner] *
                   router_weights_host_[static_cast<std::size_t>(inner) * experts_ + expert];
          }
          logits_host_[static_cast<std::size_t>(token) * experts_ + expert] = sum;
        }
      }
    } else {
      const auto desired_ids =
          make_route_ids(tokens_, 2, experts_, distribution_, zipf_s_, seed + 2);
      std::fill(logits_host_.begin(), logits_host_.end(), -8.0F);
      for (int token = 0; token < tokens_; ++token) {
        logits_host_[static_cast<std::size_t>(token) * experts_ +
                     desired_ids[static_cast<std::size_t>(token) * 2]] = 2.0F;
        logits_host_[static_cast<std::size_t>(token) * experts_ +
                     desired_ids[static_cast<std::size_t>(token) * 2 + 1]] = 1.0F;
      }
    }
    top2_selected_softmax_reference(logits_host_, tokens_, experts_, ids_expected_,
                                    route_weights_expected_);
    counts_expected_ = counts_from_ids(ids_expected_, experts_);
    offsets_expected_ = offsets_from_counts(counts_expected_);
    active_experts_ =
        static_cast<int>(std::count_if(counts_expected_.begin(), counts_expected_.end(),
                                       [](std::int32_t count) { return count > 0; }));
    build_output_reference();
    allocate_and_copy(stream);
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level != MeasurementLevel::kChainSteady) {
      throw std::invalid_argument("chain adapter only supports L3");
    }
    if (include_router_projection_) {
      cuda_check(ops::launch_dense_gemm_naive(x_.data(), router_weights_.data(), logits_.data(),
                                              tokens_, experts_, hidden_, stream),
                 "chain dense_gemm");
    }
    cuda_check(ops::launch_topk_gate_naive(logits_.data(), ids_.data(), route_weights_.data(),
                                           tokens_, experts_, stream),
               "chain topk_gate");
    cuda_check(cudaMemsetAsync(counts_.data(), 0, counts_.bytes(), stream), "chain counts reset");
    cuda_check(
        ops::launch_histogram_naive(ids_.data(), counts_.data(), route_pairs_, experts_, stream),
        "chain histogram");
    cuda_check(ops::launch_exclusive_scan_naive(counts_.data(), offsets_.data(), experts_, stream),
               "chain exclusive_scan");
    cuda_check(cudaMemsetAsync(cursors_.data(), 0, cursors_.bytes(), stream), "chain cursor reset");
    cuda_check(ops::launch_token_permute_naive(
                   x_.data(), ids_.data(), offsets_.data(), cursors_.data(), x_permuted_.data(),
                   route_pos_.data(), nullptr, tokens_, 2, hidden_, stream),
               "chain token_permute");
    // Passing R is a truthful worst-case launch bound. No input-dependent host
    // max-M computation is hidden outside the L3 interval.
    cuda_check(ops::launch_grouped_gemm_naive(x_permuted_.data(), expert_weights_.data(),
                                              offsets_.data(), y_permuted_.data(), experts_,
                                              hidden_, output_, route_pairs_, stream),
               "chain grouped_gemm");
    cuda_check(
        ops::launch_unpermute_naive(y_permuted_.data(), route_pos_.data(), route_weights_.data(),
                                    y_.data(), tokens_, 2, output_, stream),
        "chain unpermute");
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto ids = ids_.copy_to_host(stream);
    if (ids != ids_expected_) {
      return {false, "chain Top-2 ids differ from reference", {}, {}};
    }
    const auto counts = counts_.copy_to_host(stream);
    if (counts != counts_expected_) {
      return {false, "chain histogram differs from reference", {}, {}};
    }
    const auto offsets = offsets_.copy_to_host(stream);
    if (offsets != offsets_expected_) {
      return {false, "chain offsets differ from reference", {}, {}};
    }
    const auto route_weights = route_weights_.copy_to_host(stream);
    const auto weight_result =
        compare_floats(route_weights, route_weights_expected_, 1.0e-6, 1.0e-6);
    if (!weight_result.ok) return weight_result;
    return compare_floats(y_.copy_to_host(stream), expected_output_, 1.0e-4, 4.0e-5 * hidden_);
  }

  FieldMap case_config() const override {
    return {
        {"T", static_cast<std::int64_t>(tokens_)},
        {"E", static_cast<std::int64_t>(experts_)},
        {"top_k", static_cast<std::int64_t>(2)},
        {"R", static_cast<std::int64_t>(route_pairs_)},
        {"K", static_cast<std::int64_t>(hidden_)},
        {"N", static_cast<std::int64_t>(output_)},
        {"dtype", std::string("fp32")},
        {"distribution",
         include_router_projection_ ? std::string("router_projection_random") : distribution_},
        {"zipf_s", zipf_s_},
        {"chain_entry", include_router_projection_ ? std::string("tokens") : std::string("logits")},
        {"included_operator_count", static_cast<std::int64_t>(include_router_projection_ ? 7 : 6)},
        {"active_experts", static_cast<std::int64_t>(active_experts_)}};
  }

  FieldMap variant_config() const override {
    return {{"components",
             include_router_projection_
                 ? std::string("dense_gemm,topk_gate,histogram,exclusive_scan,token_permute,"
                               "grouped_gemm,unpermute")
                 : std::string(
                       "topk_gate,histogram,exclusive_scan,token_permute,grouped_gemm,unpermute")},
            {"component_variant", std::string("cuda_naive")},
            {"grouped_max_m_policy", std::string("worst_case_R")},
            {"materialize_sorted_route", false}};
  }

  WorkEstimate work_estimate() const override {
    WorkEstimate work;
    if (include_router_projection_) {
      work.flops += 2.0 * tokens_ * hidden_ * experts_;
      work.logical_bytes += sizeof(float) * (static_cast<double>(tokens_) * hidden_ +
                                             static_cast<double>(hidden_) * experts_ +
                                             static_cast<double>(tokens_) * experts_);
    }
    work.flops += 2.0 * route_pairs_ * hidden_ * output_;
    work.flops += 3.0 * tokens_ * output_;
    work.logical_bytes += sizeof(float) * static_cast<double>(tokens_) * experts_;
    work.logical_bytes += 2.0 * sizeof(std::int32_t) * route_pairs_;
    work.logical_bytes += 3.0 * sizeof(std::int32_t) * experts_;
    work.logical_bytes += 2.0 * sizeof(float) * route_pairs_ * hidden_;
    work.logical_bytes +=
        sizeof(float) *
        (static_cast<double>(route_pairs_) * hidden_ + static_cast<double>(route_pairs_) * output_ +
         static_cast<double>(active_experts_) * hidden_ * output_);
    work.logical_bytes += sizeof(float) * (static_cast<double>(route_pairs_) * output_ +
                                           static_cast<double>(tokens_) * output_);
    work.operator_metrics["kernel_launches"] =
        static_cast<std::int64_t>(include_router_projection_ ? 9 : 8);
    return work;
  }

  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"input_generation", "cpu_reference", "h2d_copy", "workspace_allocation"};
  }

 private:
  void build_output_reference() {
    expected_output_.assign(static_cast<std::size_t>(tokens_) * output_, 0.0F);
    for (int token = 0; token < tokens_; ++token) {
      for (int column = 0; column < output_; ++column) {
        float combined = 0.0F;
        for (int rank = 0; rank < 2; ++rank) {
          const auto route = static_cast<std::size_t>(token) * 2 + rank;
          const int expert = ids_expected_[route];
          float projection = 0.0F;
          for (int inner = 0; inner < hidden_; ++inner) {
            projection +=
                x_host_[static_cast<std::size_t>(token) * hidden_ + inner] *
                expert_weights_host_[(static_cast<std::size_t>(expert) * hidden_ + inner) *
                                         output_ +
                                     column];
          }
          combined += route_weights_expected_[route] * projection;
        }
        expected_output_[static_cast<std::size_t>(token) * output_ + column] = combined;
      }
    }
  }

  void allocate_and_copy(cudaStream_t stream) {
    x_.resize(x_host_.size());
    expert_weights_.resize(expert_weights_host_.size());
    logits_.resize(logits_host_.size());
    if (include_router_projection_) router_weights_.resize(router_weights_host_.size());
    ids_.resize(static_cast<std::size_t>(route_pairs_));
    route_weights_.resize(static_cast<std::size_t>(route_pairs_));
    counts_.resize(static_cast<std::size_t>(experts_));
    offsets_.resize(static_cast<std::size_t>(experts_ + 1));
    cursors_.resize(static_cast<std::size_t>(experts_));
    x_permuted_.resize(static_cast<std::size_t>(route_pairs_) * hidden_);
    route_pos_.resize(static_cast<std::size_t>(route_pairs_));
    y_permuted_.resize(static_cast<std::size_t>(route_pairs_) * output_);
    y_.resize(static_cast<std::size_t>(tokens_) * output_);
    x_.copy_from_host(x_host_, stream);
    expert_weights_.copy_from_host(expert_weights_host_, stream);
    if (include_router_projection_) {
      router_weights_.copy_from_host(router_weights_host_, stream);
    } else {
      logits_.copy_from_host(logits_host_, stream);
    }
  }

  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "K", "N", "distribution", "zipf_s"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown chain param: " + name);
    }
  }

  bool include_router_projection_ = false;
  int tokens_ = 0, experts_ = 0, hidden_ = 0, output_ = 0;
  int route_pairs_ = 0, active_experts_ = 0;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<float> x_host_, router_weights_host_, logits_host_;
  std::vector<float> expert_weights_host_, route_weights_expected_;
  std::vector<float> expected_output_;
  std::vector<std::int32_t> ids_expected_, counts_expected_, offsets_expected_;
  DeviceBuffer<float> x_, router_weights_, logits_, route_weights_;
  DeviceBuffer<float> expert_weights_, x_permuted_, y_permuted_, y_;
  DeviceBuffer<std::int32_t> ids_, counts_, offsets_, cursors_, route_pos_;
};

}  // namespace

AdapterPtr make_chain_adapter(bool include_router_projection) {
  return std::make_unique<ChainAdapter>(include_router_projection);
}

}  // namespace raggedroute::benchmark
