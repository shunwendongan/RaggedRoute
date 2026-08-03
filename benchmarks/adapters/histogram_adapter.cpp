#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "histogram/cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

bool is_optimized_histogram_variant(const std::string& variant_name) {
  return variant_name == "cuda_candidate" || variant_name == "cuda_candidate_v1" ||
         variant_name == "cuda_candidate_v2";
}

std::uint32_t optimized_histogram_implementation(const std::string& variant_name) {
  if (variant_name == "cuda_candidate") return ops::kHistogramCandidateImplementation;
  if (variant_name == "cuda_candidate_v1") return ops::kHistogramCandidateV1Implementation;
  if (variant_name == "cuda_candidate_v2") return ops::kHistogramCandidateV2Implementation;
  throw std::invalid_argument("unsupported optimized histogram variant: " + variant_name);
}

class HistogramAdapter final : public BenchmarkAdapter {
 public:
  explicit HistogramAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  std::string operator_name() const override { return "histogram"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    if (variant_name_ == "cub_device_histogram") {
      return "CUB device-wide even histogram over discrete int32 expert ids";
    }
    if (is_optimized_histogram_variant(variant_name_)) {
      return "Shape-dispatched SM86 shared-memory histogram candidate";
    }
    return "One global atomicAdd per route pair";
  }
  bool supports(MeasurementLevel level) const override {
    if (variant_name_ == "cub_device_histogram") {
      return level == MeasurementLevel::kOperatorSteady;
    }
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel level) const override {
    if (variant_name_ == "cub_device_histogram") return {};
    if (overwrites_output()) return {};
    if (level != MeasurementLevel::kKernelBody || max_count_ == 0) return {};
    const int cap = std::numeric_limits<std::int32_t>::max() / max_count_;
    return {cap, "L1 histogram accumulates counts across batched launches"};
  }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
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
    active_experts_ = static_cast<int>(std::count_if(
        expected_.begin(), expected_.end(), [](std::int32_t count) { return count != 0; }));
    ids_.resize(ids_host_.size());
    counts_.resize(expected_.size());
    ids_.copy_from_host(ids_host_, stream);
    reset_counts(stream);
#if RAGGEDROUTE_HAS_CCCL
    if (variant_name_ == "cub_device_histogram") {
      cuda_check(library_baseline::query_cub_histogram_workspace(ids_host_.size(), experts_,
                                                                 &library_workspace_bytes_),
                 "query CUB histogram workspace");
      library_workspace_.resize(library_workspace_bytes_);
    }
#endif
  }
  void prepare_sample(MeasurementLevel level, cudaStream_t stream) override {
    if (variant_name_ == "cub_device_histogram") return;
    if (overwrites_output()) return;
    if (level == MeasurementLevel::kKernelBody) reset_counts(stream);
  }
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
#if RAGGEDROUTE_HAS_CCCL
    if (variant_name_ == "cub_device_histogram") {
      cuda_check(library_baseline::launch_cub_histogram(
                     ids_.data(), counts_.data(), ids_host_.size(), experts_,
                     library_workspace_.data(), library_workspace_bytes_, stream),
                 "CUB DeviceHistogram::HistogramEven");
      return;
    }
#endif
    if (is_optimized_histogram_variant(variant_name_)) {
      const std::uint32_t implementation = optimized_histogram_implementation(variant_name_);
      if (level == MeasurementLevel::kKernelBody) {
        cuda_check(ops::launch_histogram_optimized(ids_.data(), counts_.data(),
                                                   static_cast<int>(ids_host_.size()), experts_,
                                                   implementation, stream),
                   "launch_histogram_optimized");
        return;
      }
      HistogramArgs args;
      args.expert_ids = ids_.data();
      args.counts = counts_.data();
      args.route_pairs = static_cast<int>(ids_host_.size());
      args.experts = experts_;
      args.kernel = {KernelFamily::kCudaOptimized, implementation};
      operator_check(histogram(args, make_runtime_context(stream, architecture_)),
                     "optimized histogram operator");
      return;
    }
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(ops::launch_histogram_naive(ids_.data(), counts_.data(),
                                             static_cast<int>(ids_host_.size()), experts_, stream),
                 "launch_histogram_naive");
      return;
    }
    HistogramArgs args;
    args.expert_ids = ids_.data();
    args.counts = counts_.data();
    args.route_pairs = static_cast<int>(ids_host_.size());
    args.experts = experts_;
    args.kernel = {KernelFamily::kCudaNaive, 0};
    operator_check(histogram(args, make_runtime_context(stream, architecture_)),
                   "histogram operator");
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
            {"active_experts", static_cast<std::int64_t>(active_experts_)},
            {"max_count", static_cast<std::int64_t>(max_count_)}};
  }
  FieldMap variant_config() const override {
    if (variant_name_ == "cub_device_histogram") {
      return {{"api", std::string("cub::DeviceHistogram::HistogramEven")},
              {"lower_level", static_cast<std::int64_t>(0)},
              {"upper_level", static_cast<std::int64_t>(experts_)},
              {"output_reset", std::string("inside_library_call")}};
    }
    if (is_optimized_histogram_variant(variant_name_)) {
      const std::uint32_t implementation = optimized_histogram_implementation(variant_name_);
      const std::string path =
          implementation == ops::kHistogramCandidateV2Implementation && experts_ == 1
              ? "single_bin_direct_write"
              : ids_host_.size() <= static_cast<std::size_t>(ops::kHistogramSingleCtaMaxRoutePairs)
              ? "single_cta_shared_overwrite"
              : (ids_host_.size() >=
                         static_cast<std::size_t>(ops::kHistogramBlockPrivateMinRoutePairs)
                     ? "block_private_shared_global_merge"
                     : "cuda_naive_fallback");
      return {{"algorithm_path", path},
              {"dispatch_key", std::string("route_pairs")},
              {"single_cta_max_route_pairs",
               static_cast<std::int64_t>(ops::kHistogramSingleCtaMaxRoutePairs)},
              {"block_private_min_route_pairs",
               static_cast<std::int64_t>(ops::kHistogramBlockPrivateMinRoutePairs)},
              {"block_private_max_ctas",
               static_cast<std::int64_t>(ops::kHistogramBlockPrivateMaxBlocks)},
              {"implementation_id", static_cast<std::int64_t>(implementation)},
              {"workspace_bytes", static_cast<std::int64_t>(0)},
              {"threads_per_block", static_cast<std::int64_t>(256)},
              {"shared_bytes_per_cta", static_cast<std::int64_t>(64 * sizeof(std::int32_t))}};
    }
    return {{"atomic_scope", std::string("device")},
            {"privatization", false},
            {"output_mode", std::string("accumulate_l1_reset_then_accumulate_l2")},
            {"threads_per_block", static_cast<std::int64_t>(256)}};
  }
  std::size_t workspace_bytes() const override { return library_workspace_bytes_; }
  WorkEstimate work_estimate(MeasurementLevel level) const override {
    WorkEstimate work;
    work.logical_bytes =
        sizeof(std::int32_t) * static_cast<double>(ids_host_.size() + expected_.size());
    work.operator_metrics["common_useful_bytes"] =
        static_cast<std::int64_t>(sizeof(std::int32_t) * (ids_host_.size() + expected_.size()));
    work.operator_metrics["histogram_input_items"] = static_cast<std::int64_t>(ids_host_.size());
    work.operator_metrics["histogram_bins"] = static_cast<std::int64_t>(experts_);
    work.operator_metrics["active_experts"] = static_cast<std::int64_t>(active_experts_);
    if (variant_name_ == "cuda_naive" ||
        (is_optimized_histogram_variant(variant_name_) && !uses_single_cta() &&
         !uses_block_private() && !uses_single_bin_direct())) {
      work.operator_metrics["global_atomic_operations"] =
          static_cast<std::int64_t>(ids_host_.size());
      work.operator_metrics["kernel_launches"] = static_cast<std::int64_t>(1);
    }
    if (is_optimized_histogram_variant(variant_name_) && uses_single_cta() &&
        !uses_single_bin_direct()) {
      work.operator_metrics["shared_atomic_operations_upper_bound"] =
          static_cast<std::int64_t>(ids_host_.size());
      work.operator_metrics["kernel_launches"] = static_cast<std::int64_t>(1);
    }
    if (is_optimized_histogram_variant(variant_name_) && uses_block_private()) {
      constexpr std::int64_t kItemsPerBlock = 256 * 8;
      const std::int64_t uncapped_blocks =
          (static_cast<std::int64_t>(ids_host_.size()) + kItemsPerBlock - 1) / kItemsPerBlock;
      const std::int64_t blocks = std::min(
          uncapped_blocks,
          static_cast<std::int64_t>(ops::kHistogramBlockPrivateMaxBlocks));
      work.operator_metrics["shared_atomic_operations_upper_bound"] =
          static_cast<std::int64_t>(ids_host_.size());
      work.operator_metrics["global_atomic_operations_upper_bound"] =
          blocks * static_cast<std::int64_t>(experts_);
      work.operator_metrics["histogram_ctas"] = blocks;
      work.operator_metrics["kernel_launches"] = static_cast<std::int64_t>(1);
    }
    if (level == MeasurementLevel::kOperatorSteady && !overwrites_output()) {
      work.logical_bytes += static_cast<double>(counts_.bytes());
      work.operator_metrics["counts_reset_bytes"] = static_cast<std::int64_t>(counts_.bytes());
      if (variant_name_ != "cub_device_histogram") {
        work.operator_metrics["memset_operations"] = static_cast<std::int64_t>(1);
      }
    }
    return work;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel level) const override {
    std::vector<std::string> excluded = {"route_generation", "h2d_copy", "workspace_allocation"};
    if (level == MeasurementLevel::kKernelBody && !overwrites_output()) {
      excluded.push_back("counts_reset");
    }
    return excluded;
  }

 private:
  bool uses_single_cta() const {
    return is_optimized_histogram_variant(variant_name_) &&
           ids_host_.size() <= static_cast<std::size_t>(ops::kHistogramSingleCtaMaxRoutePairs);
  }
  bool uses_block_private() const {
    return is_optimized_histogram_variant(variant_name_) && !uses_single_bin_direct() &&
           ids_host_.size() >= static_cast<std::size_t>(ops::kHistogramBlockPrivateMinRoutePairs);
  }
  bool uses_single_bin_direct() const {
    return is_optimized_histogram_variant(variant_name_) && experts_ == 1 &&
           optimized_histogram_implementation(variant_name_) ==
               ops::kHistogramCandidateV2Implementation;
  }
  bool overwrites_output() const {
    if (!is_optimized_histogram_variant(variant_name_)) return false;
    return ops::histogram_optimized_overwrites_output(
        optimized_histogram_implementation(variant_name_), static_cast<int>(ids_host_.size()),
        experts_);
  }

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
  int tokens_ = 0, experts_ = 0, top_k_ = 0, max_count_ = 0, active_experts_ = 0;
  std::string variant_name_;
  std::size_t library_workspace_bytes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<std::int32_t> ids_host_, expected_;
  DeviceBuffer<std::int32_t> ids_, counts_;
  DeviceBuffer<std::uint8_t> library_workspace_;
};

}  // namespace

AdapterPtr make_histogram_adapter(const std::string& variant_name) {
  return std::make_unique<HistogramAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
