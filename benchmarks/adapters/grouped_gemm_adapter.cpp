#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

class GroupedGemmAdapter final : public BenchmarkAdapter {
 public:
  explicit GroupedGemmAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  ~GroupedGemmAdapter() override {
#if RAGGEDROUTE_HAS_CUBLAS
    library_baseline::destroy_grouped_cublas_plan(cublas_plan_);
#endif
#if RAGGEDROUTE_HAS_CUTLASS
    library_baseline::destroy_grouped_cutlass_plan(cutlass_plan_);
#endif
  }
  std::string operator_name() const override { return "grouped_gemm"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    return "Single-launch FP32 grouped GEMM with one grid-z slice per expert";
  }
  bool supports(MeasurementLevel level) const override {
    if (variant_name_ == "cublas_per_expert") {
      return level == MeasurementLevel::kOperatorSteady;
    }
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16);
    top_k_ = get_int_option(options, "top_k", 2);
    hidden_ = get_int_option(options, "K", 128);
    output_ = get_int_option(options, "N", 128);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    if (experts_ < top_k_ || experts_ > 64) {
      throw std::invalid_argument("current grouped GEMM adapter requires top_k<=E<=64");
    }
    route_pairs_ = checked_int_product(tokens_, top_k_, "R=T*top_k");

    const auto ids = make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed + 1);
    counts_ = counts_from_ids(ids, experts_);
    offsets_host_ = offsets_from_counts(counts_);
    max_expert_tokens_ = *std::max_element(counts_.begin(), counts_.end());
    active_experts_ = static_cast<int>(
        std::count_if(counts_.begin(), counts_.end(), [](int count) { return count > 0; }));
    x_host_ = make_random_floats(static_cast<std::size_t>(route_pairs_) * hidden_, seed);
    weights_host_ =
        make_random_floats(static_cast<std::size_t>(experts_) * hidden_ * output_, seed + 2);
    expected_.assign(static_cast<std::size_t>(route_pairs_) * output_, 0.0F);
    for (int expert = 0; expert < experts_; ++expert) {
      for (int row = offsets_host_[static_cast<std::size_t>(expert)];
           row < offsets_host_[static_cast<std::size_t>(expert + 1)]; ++row) {
        for (int column = 0; column < output_; ++column) {
          float sum = 0.0F;
          for (int inner = 0; inner < hidden_; ++inner) {
            sum += x_host_[static_cast<std::size_t>(row) * hidden_ + inner] *
                   weights_host_[(static_cast<std::size_t>(expert) * hidden_ + inner) * output_ +
                                 column];
          }
          expected_[static_cast<std::size_t>(row) * output_ + column] = sum;
        }
      }
    }

    x_.resize(x_host_.size());
    weights_.resize(weights_host_.size());
    offsets_.resize(offsets_host_.size());
    output_buffer_.resize(expected_.size());
    x_.copy_from_host(x_host_, stream);
    weights_.copy_from_host(weights_host_, stream);
    offsets_.copy_from_host(offsets_host_, stream);
#if RAGGEDROUTE_HAS_CUBLAS
    if (variant_name_ == "cublas_per_expert") {
      cublas_plan_ = library_baseline::create_grouped_cublas_plan();
    }
#endif
#if RAGGEDROUTE_HAS_CUTLASS
    if (variant_name_ == "cutlass_grouped") {
      cutlass_plan_ = library_baseline::create_grouped_cutlass_plan(
          x_.data(), weights_.data(), output_buffer_.data(), offsets_host_.data(), experts_,
          hidden_, output_);
      library_workspace_bytes_ = library_baseline::grouped_cutlass_workspace_bytes(cutlass_plan_);
      library_workspace_.resize(library_workspace_bytes_);
      library_baseline::initialize_grouped_cutlass_plan(cutlass_plan_, library_workspace_.data(),
                                                        library_workspace_bytes_, stream);
    }
#endif
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
#if RAGGEDROUTE_HAS_CUBLAS
    if (variant_name_ == "cublas_per_expert") {
      library_baseline::launch_grouped_cublas(cublas_plan_, x_.data(), weights_.data(),
                                              output_buffer_.data(), offsets_host_.data(), experts_,
                                              hidden_, output_, stream);
      return;
    }
#endif
#if RAGGEDROUTE_HAS_CUTLASS
    if (variant_name_ == "cutlass_grouped") {
      library_baseline::launch_grouped_cutlass(cutlass_plan_, library_workspace_.data(),
                                               library_workspace_bytes_, stream);
      return;
    }
#endif
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(ops::launch_grouped_gemm_naive(x_.data(), weights_.data(), offsets_.data(),
                                                output_buffer_.data(), experts_, hidden_, output_,
                                                max_expert_tokens_, stream),
                 "launch_grouped_gemm_naive");
      return;
    }
    GroupedGemmArgs args;
    args.x_permuted.data = x_.data();
    args.expert_weights.data = weights_.data();
    args.offsets = offsets_.data();
    args.y_permuted.data = output_buffer_.data();
    args.experts = experts_;
    args.hidden = hidden_;
    args.output = output_;
    args.max_expert_tokens = max_expert_tokens_;
    operator_check(grouped_gemm(args, make_runtime_context(stream, architecture_)),
                   "grouped_gemm operator");
  }
  ValidationResult validate(cudaStream_t stream) override {
    return compare_floats(output_buffer_.copy_to_host(stream), expected_, 1.0e-4, 2.0e-5 * hidden_);
  }
  FieldMap case_config() const override {
    return {{"T", static_cast<std::int64_t>(tokens_)},
            {"E", static_cast<std::int64_t>(experts_)},
            {"top_k", static_cast<std::int64_t>(top_k_)},
            {"R", static_cast<std::int64_t>(route_pairs_)},
            {"K", static_cast<std::int64_t>(hidden_)},
            {"N", static_cast<std::int64_t>(output_)},
            {"dtype", std::string("fp32")},
            {"accumulator_dtype", std::string("fp32")},
            {"distribution", distribution_},
            {"zipf_s", zipf_s_},
            {"active_experts", static_cast<std::int64_t>(active_experts_)},
            {"max_expert_tokens", static_cast<std::int64_t>(max_expert_tokens_)}};
  }
  FieldMap variant_config() const override {
    if (variant_name_ == "cublas_per_expert") {
      return {{"api", std::string("cublasSgemm")},
              {"launches", static_cast<std::int64_t>(active_experts_)},
              {"scheduler", std::string("host_loop_active_experts")},
              {"math_mode", std::string("strict_fp32")}};
    }
    if (variant_name_ == "cutlass_grouped") {
      return {{"api", std::string("cutlass::gemm::device::GemmGrouped")},
              {"scheduler", std::string("device_only")},
              {"operator_class", std::string("simt_fp32")},
              {"threadblock_shape", std::string("128x128x8")}};
    }
    return {{"tile_m", static_cast<std::int64_t>(16)},
            {"tile_n", static_cast<std::int64_t>(16)},
            {"scheduler", std::string("grid_z_per_expert")}};
  }
  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    work.flops = 2.0 * route_pairs_ * hidden_ * output_;
    work.logical_bytes = sizeof(float) * (static_cast<double>(route_pairs_) * hidden_ +
                                          static_cast<double>(route_pairs_) * output_ +
                                          static_cast<double>(active_experts_) * hidden_ * output_);
    work.operator_metrics["total_gemm_problems"] = static_cast<std::int64_t>(experts_);
    work.operator_metrics["active_gemm_problems"] = static_cast<std::int64_t>(active_experts_);
    return work;
  }
  std::size_t workspace_bytes() const override { return library_workspace_bytes_; }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"route_generation", "offset_preparation", "input_generation", "h2d_copy",
            "workspace_allocation"};
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "top_k", "K", "N", "distribution", "zipf_s"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown grouped_gemm param: " + name);
    }
  }
  int tokens_ = 0, experts_ = 0, top_k_ = 0, hidden_ = 0, output_ = 0;
  std::string variant_name_;
#if RAGGEDROUTE_HAS_CUBLAS
  library_baseline::GroupedCublasPlan* cublas_plan_ = nullptr;
#endif
#if RAGGEDROUTE_HAS_CUTLASS
  library_baseline::GroupedCutlassPlan* cutlass_plan_ = nullptr;
#endif
  std::size_t library_workspace_bytes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  int route_pairs_ = 0, max_expert_tokens_ = 0, active_experts_ = 0;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<std::int32_t> counts_, offsets_host_;
  std::vector<float> x_host_, weights_host_, expected_;
  DeviceBuffer<std::int32_t> offsets_;
  DeviceBuffer<float> x_, weights_, output_buffer_;
  DeviceBuffer<std::uint8_t> library_workspace_;
};

}  // namespace

AdapterPtr make_grouped_gemm_adapter(const std::string& variant_name) {
  return std::make_unique<GroupedGemmAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
