#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"
#include "topk_gate/cuda_candidate/optimized_internal.h"

namespace raggedroute::benchmark {
namespace {

bool is_optimized_variant(const std::string& name) {
  return name == "cuda_warp_pair_top2_v1" || name == "cuda_subwarp_pair_top2_v2" ||
         name == "cuda_vector_pair_top2_v3";
}

bool is_library_variant(const std::string& name) {
  return name == "cub_block_radix_top2" || name == "vllm_row_packed_top2";
}

std::uint32_t implementation_id(const std::string& name) {
  if (name == "cuda_warp_pair_top2_v1") return ops::kTopKGateWarpPairV1Implementation;
  if (name == "cuda_subwarp_pair_top2_v2") return ops::kTopKGateSubwarpPairV2Implementation;
  if (name == "cuda_vector_pair_top2_v3") return ops::kTopKGateVectorPairV3Implementation;
  throw std::invalid_argument("unsupported optimized topk_gate variant: " + name);
}

class TopKGateAdapter final : public BenchmarkAdapter {
 public:
  explicit TopKGateAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  std::string operator_name() const override { return "topk_gate"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    if (variant_name_ == "cuda_warp_pair_top2_v1") return "Warp-register Top-2 pair merge";
    if (variant_name_ == "cuda_subwarp_pair_top2_v2")
      return "Shape-specialized subwarp Top-2 pair merge";
    if (variant_name_ == "cuda_vector_pair_top2_v3")
      return "Aligned float4 subwarp Top-2 pair merge with scalar fallback";
    if (variant_name_ == "cub_block_radix_top2")
      return "CUB BlockRadixSort composite-key strict Top-2";
    if (variant_name_ == "vllm_row_packed_top2") return "Adapted vLLM row-packed vector Top-2";
    return "One CUDA thread per row with deterministic Top-2 selected-softmax";
  }
  bool supports(MeasurementLevel level) const override {
    if (is_library_variant(variant_name_)) return level == MeasurementLevel::kKernelBody;
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16, 2);
    if (experts_ > 64) throw std::invalid_argument("current Top-K adapter supports 2<=E<=64");
    input_alignment_ = get_int_option(options, "input_alignment", 16);
    if (input_alignment_ != 4 && input_alignment_ != 16) {
      throw std::invalid_argument("topk_gate input_alignment must be 4 or 16");
    }
    input_mode_ = get_option(options, "input_mode", "random");
    logits_host_ =
        make_random_floats(static_cast<std::size_t>(tokens_) * experts_, seed, -8.0F, 8.0F);
    populate_special_input();
    build_reference();

    const std::size_t offset = input_alignment_ == 4 ? 1 : 0;
    logits_storage_.resize(logits_host_.size() + offset);
    logits_data_ = logits_storage_.data() + offset;
    ids_.resize(ids_expected_.size());
    weights_.resize(weights_expected_.size());
    cuda_check(cudaMemcpyAsync(logits_data_, logits_host_.data(),
                               logits_host_.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync topk logits H2D");

    if (variant_name_ == "vllm_row_packed_top2" &&
        !library_baseline::supports_vllm_row_packed_top2(experts_, logits_data_)) {
      throw std::invalid_argument(
          "vllm_row_packed_top2 supports only E={2,4,8,16,32,64} with required alignment");
    }
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level == MeasurementLevel::kKernelBody) {
      if (variant_name_ == "cub_block_radix_top2") {
#if RAGGEDROUTE_HAS_CCCL
        cuda_check(library_baseline::launch_cub_block_radix_top2(
                       logits_data_, ids_.data(), weights_.data(), tokens_, experts_, stream),
                   "launch_cub_block_radix_top2");
        return;
#else
        throw std::runtime_error("CUB baseline was not compiled");
#endif
      }
      if (variant_name_ == "vllm_row_packed_top2") {
        cuda_check(library_baseline::launch_vllm_row_packed_top2(
                       logits_data_, ids_.data(), weights_.data(), tokens_, experts_, stream),
                   "launch_vllm_row_packed_top2");
        return;
      }
      if (is_optimized_variant(variant_name_)) {
        cuda_check(
            ops::launch_topk_gate_optimized(logits_data_, ids_.data(), weights_.data(), tokens_,
                                            experts_, implementation_id(variant_name_), stream),
            "launch_topk_gate_optimized");
        return;
      }
      cuda_check(ops::launch_topk_gate_naive(logits_data_, ids_.data(), weights_.data(), tokens_,
                                             experts_, stream),
                 "launch_topk_gate_naive");
      return;
    }

    TopKGateArgs args;
    args.logits.data = logits_data_;
    args.expert_ids = ids_.data();
    args.weights.data = weights_.data();
    args.tokens = tokens_;
    args.experts = experts_;
    if (is_optimized_variant(variant_name_)) {
      args.kernel = {KernelFamily::kCudaOptimized, implementation_id(variant_name_)};
    }
    operator_check(topk_gate(args, make_runtime_context(stream, architecture_)),
                   "topk_gate operator");
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto ids = ids_.copy_to_host(stream);
    const auto weights = weights_.copy_to_host(stream);
    if (ids != ids_expected_) {
      return {false, "Top-2 expert ids differ from deterministic reference", {}, {}};
    }
    for (int token = 0; token < tokens_; ++token) {
      const std::size_t output = static_cast<std::size_t>(token) * 2;
      if (ids[output] < 0 || ids[output] >= experts_ || ids[output + 1] < 0 ||
          ids[output + 1] >= experts_ || ids[output] == ids[output + 1]) {
        return {false, "Top-2 ids are not unique and in range", {}, {}};
      }
      const float first = weights[output];
      const float second = weights[output + 1];
      if (!std::isfinite(first) || !std::isfinite(second) || first < 0.0F || second < 0.0F ||
          std::abs((first + second) - 1.0F) > 1.0e-6F) {
        return {false, "Top-2 weights violate finite/nonnegative/unit-sum invariants", {}, {}};
      }
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
            {"input_mode", input_mode_},
            {"input_alignment_bytes", static_cast<std::int64_t>(input_alignment_)}};
  }

  FieldMap variant_config() const override {
    std::string load_path = "scalar";
    if (variant_name_ == "cuda_vector_pair_top2_v3") {
      const bool vector_shape = experts_ == 8 || experts_ == 16 || experts_ == 32 || experts_ == 64;
      load_path = input_alignment_ == 16 && vector_shape ? "float4" : "v2_fallback";
    } else if (variant_name_ == "vllm_row_packed_top2") {
      load_path = experts_ == 2 ? "float2" : "float4";
    }
    return {{"load_path", load_path},
            {"workspace_bytes", static_cast<std::int64_t>(0)},
            {"launch_count", static_cast<std::int64_t>(1)}};
  }

  WorkEstimate work_estimate(MeasurementLevel) const override {
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
  void populate_special_input() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (input_mode_ == "random") return;
    if (input_mode_ == "ties") {
      std::fill(logits_host_.begin(), logits_host_.end(), 1.0F);
      return;
    }
    if (input_mode_ == "all_nan") {
      std::fill(logits_host_.begin(), logits_host_.end(), nan);
      return;
    }
    for (int token = 0; token < tokens_; ++token) {
      float* row = logits_host_.data() + static_cast<std::size_t>(token) * experts_;
      if (input_mode_ == "duplicate_max") {
        std::fill(row, row + experts_, -3.0F);
        row[0] = 7.0F;
        row[experts_ - 1] = 7.0F;
      } else if (input_mode_ == "signed_zero") {
        for (int expert = 0; expert < experts_; ++expert) row[expert] = expert % 2 ? 0.0F : -0.0F;
      } else if (input_mode_ == "mixed_nan_neg_inf") {
        std::fill(row, row + experts_, -inf);
        row[0] = nan;
        if (experts_ > 2) row[2] = nan;
      } else if (input_mode_ == "single_infinity" || input_mode_ == "infinities") {
        row[experts_ - 1] = inf;
      } else if (input_mode_ == "multiple_infinity") {
        row[0] = inf;
        row[experts_ - 1] = inf;
      } else if (input_mode_ == "extreme") {
        std::fill(row, row + experts_, -1.0e30F);
        row[0] = 1.0e30F;
        row[1] = 0.0F;
      } else {
        throw std::invalid_argument("unsupported topk input_mode: " + input_mode_);
      }
    }
  }

  void build_reference() {
    top2_selected_softmax_reference(logits_host_, tokens_, experts_, ids_expected_,
                                    weights_expected_);
  }
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "input_mode", "input_alignment"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown topk_gate param: " + name);
    }
  }

  int tokens_ = 0;
  int experts_ = 0;
  int input_alignment_ = 16;
  std::string variant_name_;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  std::string input_mode_;
  std::vector<float> logits_host_, weights_expected_;
  std::vector<std::int32_t> ids_expected_;
  DeviceBuffer<float> logits_storage_, weights_;
  float* logits_data_ = nullptr;
  DeviceBuffer<std::int32_t> ids_;
};

}  // namespace

AdapterPtr make_topk_gate_adapter(const std::string& variant_name) {
  return std::make_unique<TopKGateAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
