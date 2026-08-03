#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "scan/cuda_candidate/optimized_internal.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

bool is_fused_candidate(const std::string& variant) {
  return variant == "cuda_fused_scalar" || variant == "cuda_fused_subwarp";
}

std::uint32_t fused_implementation(const std::string& variant) {
  if (variant == "cuda_fused_scalar") {
    return ops::kHistogramExclusiveScanFusedScalarImplementation;
  }
  if (variant == "cuda_fused_subwarp") {
    return ops::kHistogramExclusiveScanFusedSubwarpImplementation;
  }
  throw std::invalid_argument("unsupported fused histogram-scan variant: " + variant);
}

class HistogramExclusiveScanAdapter final : public BenchmarkAdapter {
 public:
  explicit HistogramExclusiveScanAdapter(std::string variant)
      : variant_(std::move(variant)) {}

  std::string operator_name() const override { return "histogram_exclusive_scan"; }
  std::string variant_name() const override { return variant_; }
  std::string description() const override {
    if (variant_ == "cuda_separate_current") {
      return "Current promoted Histogram followed by current Exclusive Scan";
    }
    if (variant_ == "cuda_fused_scalar") {
      return "Single-CTA shared histogram with scalar scan finalization";
    }
    if (variant_ == "cuda_fused_subwarp") {
      return "Single-CTA shared histogram with 16-lane scan finalization";
    }
    return "CUB DeviceHistogram followed by a CUB scan primitive";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    route_pairs_ = get_int_option(options, "R", 1024, 0);
    experts_ = get_int_option(options, "E", 64);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    if (experts_ > 64) throw std::invalid_argument("histogram-scan requires E<=64");

    if (route_pairs_ != 0) {
      ids_host_ = make_route_ids(route_pairs_, 1, experts_, distribution_, zipf_s_, seed);
    }
    expected_counts_ = counts_from_ids(ids_host_, experts_);
    expected_offsets_ = offsets_from_counts(expected_counts_);
    active_experts_ = static_cast<int>(std::count_if(
        expected_counts_.begin(), expected_counts_.end(),
        [](std::int32_t count) { return count != 0; }));

    ids_.resize(ids_host_.size());
    counts_.resize(expected_counts_.size());
    offsets_.resize(expected_offsets_.size());
    if (!ids_host_.empty()) ids_.copy_from_host(ids_host_, stream);

#if RAGGEDROUTE_HAS_CCCL
    if (is_cub_variant()) {
      cuda_check(library_baseline::query_cub_histogram_workspace(
                     ids_host_.size(), experts_, &histogram_workspace_bytes_),
                 "query CUB histogram workspace");
      histogram_workspace_.resize(histogram_workspace_bytes_);
      if (variant_ == "cub_histogram_device_scan") {
        cuda_check(library_baseline::query_cub_device_scan_workspace(
                       experts_, &scan_workspace_bytes_),
                   "query CUB scan workspace");
        scan_workspace_.resize(scan_workspace_bytes_);
      }
      if (variant_ == "cub_histogram_warp_scan" && experts_ > 32) {
        throw std::invalid_argument("cub_histogram_warp_scan requires E<=32");
      }
    }
#endif
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level != MeasurementLevel::kOperatorSteady) {
      throw std::invalid_argument("histogram_exclusive_scan supports L2 only");
    }
    if (variant_ == "cuda_separate_current") {
      HistogramArgs histogram_args;
      histogram_args.expert_ids = ids_.data();
      histogram_args.counts = counts_.data();
      histogram_args.route_pairs = route_pairs_;
      histogram_args.experts = experts_;
      operator_check(histogram(histogram_args, make_runtime_context(stream, architecture_)),
                     "current histogram operator");
      ExclusiveScanArgs scan_args;
      scan_args.counts = counts_.data();
      scan_args.offsets = offsets_.data();
      scan_args.experts = experts_;
      operator_check(exclusive_scan(scan_args, make_runtime_context(stream, architecture_)),
                     "current exclusive_scan operator");
      return;
    }
    if (is_fused_candidate(variant_)) {
      HistogramExclusiveScanArgs args;
      args.expert_ids = ids_.data();
      args.counts = counts_.data();
      args.offsets = offsets_.data();
      args.route_pairs = route_pairs_;
      args.experts = experts_;
      args.kernel = {KernelFamily::kCudaOptimized, fused_implementation(variant_)};
      operator_check(histogram_exclusive_scan(args,
                                              make_runtime_context(stream, architecture_)),
                     "fused histogram_exclusive_scan operator");
      return;
    }
#if RAGGEDROUTE_HAS_CCCL
    if (is_cub_variant()) {
      cuda_check(library_baseline::launch_cub_histogram(
                     ids_.data(), counts_.data(), ids_host_.size(), experts_,
                     histogram_workspace_.data(), histogram_workspace_bytes_, stream),
                 "CUB DeviceHistogram::HistogramEven");
      if (variant_ == "cub_histogram_warp_scan") {
        cuda_check(library_baseline::launch_cub_warp_scan(
                       counts_.data(), offsets_.data(), experts_, stream),
                   "CUB WarpScan::ExclusiveSum");
      } else if (variant_ == "cub_histogram_block_scan") {
        cuda_check(library_baseline::launch_cub_block_scan(
                       counts_.data(), offsets_.data(), experts_, stream),
                   "CUB BlockScan::ExclusiveSum");
      } else if (variant_ == "cub_histogram_device_scan") {
        cuda_check(library_baseline::launch_cub_device_scan(
                       counts_.data(), offsets_.data(), experts_, scan_workspace_.data(),
                       scan_workspace_bytes_, stream),
                   "CUB DeviceScan::ExclusiveSum");
      }
      return;
    }
#endif
    throw std::invalid_argument("unknown histogram_exclusive_scan variant: " + variant_);
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto counts = counts_.copy_to_host(stream);
    if (counts != expected_counts_) {
      return {false, "fused expert counts differ from CPU reference", {}, {}};
    }
    const auto offsets = offsets_.copy_to_host(stream);
    if (offsets != expected_offsets_) {
      return {false, "fused offsets differ from CPU reference", {}, {}};
    }
    return {true, "counts and offsets match CPU reference", 0.0, 0.0};
  }

  FieldMap case_config() const override {
    return {{"R", static_cast<std::int64_t>(route_pairs_)},
            {"E", static_cast<std::int64_t>(experts_)},
            {"count_dtype", std::string("int32")},
            {"distribution", distribution_},
            {"zipf_s", zipf_s_},
            {"active_experts", static_cast<std::int64_t>(active_experts_)}};
  }

  FieldMap variant_config() const override {
    if (variant_ == "cuda_separate_current") {
      return {{"algorithm", std::string("promoted_histogram_then_scan")},
              {"kernel_launches", static_cast<std::int64_t>(2)}};
    }
    if (is_fused_candidate(variant_)) {
      return {{"algorithm", variant_ == "cuda_fused_scalar"
                                ? std::string("single_cta_shared_scalar_finalize")
                                : std::string("single_cta_shared_subwarp_finalize")},
              {"fused_max_route_pairs",
               static_cast<std::int64_t>(ops::kHistogramExclusiveScanFusedMaxRoutePairs)},
              {"kernel_launches", static_cast<std::int64_t>(route_pairs_ <=
                                                                     ops::kHistogramExclusiveScanFusedMaxRoutePairs
                                                                 ? 1
                                                                 : 2)},
              {"workspace_bytes", static_cast<std::int64_t>(0)}};
    }
    return {{"histogram_api", std::string("cub::DeviceHistogram::HistogramEven")},
            {"scan_api", variant_ == "cub_histogram_warp_scan"
                             ? std::string("cub::WarpScan::ExclusiveSum")
                             : variant_ == "cub_histogram_block_scan"
                                   ? std::string("cub::BlockScan::ExclusiveSum")
                                   : std::string("cub::DeviceScan::ExclusiveSum")}};
  }

  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    work.logical_bytes = sizeof(std::int32_t) *
                         static_cast<double>(ids_host_.size() + expected_counts_.size() +
                                             expected_offsets_.size());
    work.operator_metrics["histogram_input_items"] =
        static_cast<std::int64_t>(ids_host_.size());
    work.operator_metrics["histogram_bins"] = static_cast<std::int64_t>(experts_);
    work.operator_metrics["integer_additions"] = static_cast<std::int64_t>(experts_);
    work.operator_metrics["kernel_launches"] =
        static_cast<std::int64_t>(is_fused_candidate(variant_) &&
                                          route_pairs_ <=
                                              ops::kHistogramExclusiveScanFusedMaxRoutePairs
                                      ? 1
                                      : variant_ == "cub_histogram_device_scan" ? 3 : 2);
    return work;
  }

  std::size_t workspace_bytes() const override {
    return histogram_workspace_bytes_ + scan_workspace_bytes_;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"route_generation", "h2d_copy", "workspace_allocation"};
  }

 private:
  bool is_cub_variant() const { return variant_.rfind("cub_histogram_", 0) == 0; }
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"R", "E", "distribution", "zipf_s"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) {
        throw std::invalid_argument("unknown histogram_exclusive_scan param: " + name);
      }
    }
  }

  std::string variant_;
  int route_pairs_ = 0;
  int experts_ = 0;
  int active_experts_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::size_t histogram_workspace_bytes_ = 0;
  std::size_t scan_workspace_bytes_ = 0;
  std::vector<std::int32_t> ids_host_, expected_counts_, expected_offsets_;
  DeviceBuffer<std::int32_t> ids_, counts_, offsets_;
  DeviceBuffer<std::uint8_t> histogram_workspace_, scan_workspace_;
};

}  // namespace

AdapterPtr make_histogram_exclusive_scan_adapter(const std::string& variant_name) {
  return std::make_unique<HistogramExclusiveScanAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
