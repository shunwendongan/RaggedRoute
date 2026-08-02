#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"
#include "../../src/unpermute/cuda_candidate/optimized_internal.h"

namespace raggedroute::benchmark {
namespace {

class UnpermuteAdapter final : public BenchmarkAdapter {
 public:
  explicit UnpermuteAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  std::string operator_name() const override { return "unpermute"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    return "Token-owned scalar gather and weighted reduce";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16);
    top_k_ = get_int_option(options, "top_k", 2);
    output_ = get_int_option(options, "N", 256);
    pointer_offset_elements_ = get_int_option(options, "pointer_offset_elements", 0, 0);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    if (experts_ < top_k_ || experts_ > 64) {
      throw std::invalid_argument("current unpermute adapter requires top_k<=E<=64");
    }
    route_pairs_ = checked_int_product(tokens_, top_k_, "R=T*top_k");
    (void)checked_int_product(tokens_, output_, "T*N");

    const auto ids = make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed + 1);
    const auto counts = counts_from_ids(ids, experts_);
    const auto offsets = offsets_from_counts(counts);
    std::vector<std::int32_t> cursor(offsets.begin(), offsets.end() - 1);
    route_pos_host_.resize(static_cast<std::size_t>(route_pairs_));
    for (int route = 0; route < route_pairs_; ++route) {
      const int expert = ids[static_cast<std::size_t>(route)];
      route_pos_host_[static_cast<std::size_t>(route)] = cursor[static_cast<std::size_t>(expert)]++;
    }
    y_permuted_host_ =
        make_random_floats(static_cast<std::size_t>(route_pairs_) * output_, seed + 2);
    route_weights_host_ =
        make_random_floats(static_cast<std::size_t>(route_pairs_), seed + 3, 0.05F, 1.0F);
    for (int token = 0; token < tokens_; ++token) {
      float sum = 0.0F;
      for (int rank = 0; rank < top_k_; ++rank) {
        sum += route_weights_host_[static_cast<std::size_t>(token * top_k_ + rank)];
      }
      for (int rank = 0; rank < top_k_; ++rank) {
        route_weights_host_[static_cast<std::size_t>(token * top_k_ + rank)] /= sum;
      }
    }
    expected_.assign(static_cast<std::size_t>(tokens_) * output_, 0.0F);
    for (int token = 0; token < tokens_; ++token) {
      for (int column = 0; column < output_; ++column) {
        float sum = 0.0F;
        for (int rank = 0; rank < top_k_; ++rank) {
          const int route = token * top_k_ + rank;
          const int source = route_pos_host_[static_cast<std::size_t>(route)];
          sum += route_weights_host_[static_cast<std::size_t>(route)] *
                 y_permuted_host_[static_cast<std::size_t>(source) * output_ + column];
        }
        expected_[static_cast<std::size_t>(token) * output_ + column] = sum;
      }
    }

    y_permuted_storage_host_.assign(static_cast<std::size_t>(pointer_offset_elements_), -17.0F);
    y_permuted_storage_host_.insert(y_permuted_storage_host_.end(), y_permuted_host_.begin(),
                                    y_permuted_host_.end());
    y_permuted_.resize(y_permuted_storage_host_.size());
    route_pos_.resize(route_pos_host_.size());
    route_weights_.resize(route_weights_host_.size());
    y_.resize(expected_.size() + static_cast<std::size_t>(pointer_offset_elements_));
    y_permuted_.copy_from_host(y_permuted_storage_host_, stream);
    route_pos_.copy_from_host(route_pos_host_, stream);
    route_weights_.copy_from_host(route_weights_host_, stream);
    cuda_check(cudaMemsetAsync(y_.data(), 0, y_.bytes(), stream),
               "initialize unpermute output storage");
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (is_candidate()) {
      cuda_check(ops::launch_unpermute_optimized(
                     y_permuted_data(), route_pos_.data(), route_weights_.data(), y_data(),
                     tokens_, top_k_, output_, stream),
                 "launch_unpermute_optimized");
      return;
    }
#if RAGGEDROUTE_HAS_VLLM_UNPERMUTE
    if (variant_name_ == "vllm_finalize_routing") {
      cuda_check(library_baseline::launch_vllm_finalize_routing(
                     y_permuted_data(), y_data(), route_weights_.data(), route_pos_.data(),
                     tokens_, top_k_, output_, stream),
                 "vLLM finalizeMoeRoutingKernel");
      return;
    }
#endif
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(
          ops::launch_unpermute_naive(y_permuted_data(), route_pos_.data(), route_weights_.data(),
                                      y_data(), tokens_, top_k_, output_, stream),
          "launch_unpermute_naive");
      return;
    }
    UnpermuteArgs args;
    args.y_permuted.data = y_permuted_data();
    args.route_pos = route_pos_.data();
    args.route_weights.data = route_weights_.data();
    args.y.data = y_data();
    args.tokens = tokens_;
    args.top_k = top_k_;
    args.output = output_;
    operator_check(unpermute(args, make_runtime_context(stream, architecture_)),
                   "unpermute operator");
  }
  ValidationResult validate(cudaStream_t stream) override {
    const auto storage = y_.copy_to_host(stream);
    const auto begin = storage.begin() + pointer_offset_elements_;
    return compare_floats(std::vector<float>(begin, storage.end()), expected_, 1.0e-6, 1.0e-5);
  }
  FieldMap case_config() const override {
    return {{"T", static_cast<std::int64_t>(tokens_)},
            {"E", static_cast<std::int64_t>(experts_)},
            {"top_k", static_cast<std::int64_t>(top_k_)},
            {"R", static_cast<std::int64_t>(route_pairs_)},
            {"N", static_cast<std::int64_t>(output_)},
            {"dtype", std::string("fp32")},
            {"accumulator_dtype", std::string("fp32")},
            {"pointer_offset_elements", static_cast<std::int64_t>(pointer_offset_elements_)},
            {"distribution", distribution_},
            {"zipf_s", zipf_s_}};
  }
  FieldMap variant_config() const override {
    if (is_candidate()) {
      return {{"ownership", std::string("warp_or_cta_per_token")},
              {"warps_per_cta", static_cast<std::int64_t>(output_ >= 512 ? 8 : 4)},
              {"items_per_lane", static_cast<std::int64_t>(1)},
              {"metadata_policy", std::string("lane0_shuffle_broadcast")},
              {"large_output_metadata_policy", std::string("thread0_shared_broadcast")},
              {"large_output_threshold", static_cast<std::int64_t>(512)},
              {"vector_width_bytes", static_cast<std::int64_t>(16)},
              {"alignment_policy", std::string("float4_fast_naive_fallback")},
              {"top_k_policy", std::string("top2_fast_generic_fallback")}};
    }
    if (variant_name_ == "vllm_finalize_routing") {
      return {{"upstream_symbol", std::string("finalizeMoeRoutingKernelLauncher")},
              {"ownership", std::string("token_owned")},
              {"vector_width_bytes", static_cast<std::int64_t>(output_ % 4 == 0 ? 16 : 4)},
              {"alignment_policy", std::string("float4_fast_scalar_fallback")}};
    }
    return {{"ownership", std::string("token_owned")},
            {"vector_width_bytes", static_cast<std::int64_t>(4)}};
  }
  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    work.flops = static_cast<double>((2 * top_k_ - 1) * tokens_ * output_);
    work.logical_bytes =
        sizeof(float) * static_cast<double>(top_k_) * tokens_ * output_ +
        sizeof(float) * static_cast<double>(tokens_) * output_ +
        (sizeof(float) + sizeof(std::int32_t)) * static_cast<double>(top_k_) * tokens_;
    work.operator_metrics["gather_rows"] = static_cast<std::int64_t>(route_pairs_);
    return work;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"route_generation", "mapping_generation", "input_generation", "h2d_copy",
            "workspace_allocation"};
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "top_k", "N", "distribution", "zipf_s",
                                           "pointer_offset_elements"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown unpermute param: " + name);
    }
  }
  bool is_candidate() const { return variant_name_ == "cuda_warp_token_vec4"; }
  const float* y_permuted_data() const {
    return y_permuted_.data() + pointer_offset_elements_;
  }
  float* y_data() { return y_.data() + pointer_offset_elements_; }

  int tokens_ = 0, experts_ = 0, top_k_ = 0, output_ = 0, route_pairs_ = 0;
  int pointer_offset_elements_ = 0;
  std::string variant_name_;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<std::int32_t> route_pos_host_;
  std::vector<float> y_permuted_host_, y_permuted_storage_host_, route_weights_host_, expected_;
  DeviceBuffer<std::int32_t> route_pos_;
  DeviceBuffer<float> y_permuted_, route_weights_, y_;
};

}  // namespace

AdapterPtr make_unpermute_adapter(const std::string& variant_name) {
  return std::make_unique<UnpermuteAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
