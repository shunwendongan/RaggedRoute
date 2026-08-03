#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "scan/cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

bool is_optimized_scan_variant(const std::string& variant_name) {
  return variant_name == "cuda_warp_blocked_scalar_legacy" ||
         variant_name == "cuda_subwarp4_scalar" ||
         variant_name == "cuda_subwarp4_vector";
}

std::uint32_t optimized_scan_implementation(const std::string& variant_name) {
  if (variant_name == "cuda_warp_blocked_scalar_legacy") {
    return ops::kExclusiveScanWarpBlockedScalarLegacyImplementation;
  }
  if (variant_name == "cuda_subwarp4_scalar") {
    return ops::kExclusiveScanSubwarp4ScalarImplementation;
  }
  if (variant_name == "cuda_subwarp4_vector") {
    return ops::kExclusiveScanSubwarp4VectorImplementation;
  }
  throw std::invalid_argument("unsupported optimized scan variant: " + variant_name);
}

class ExclusiveScanAdapter final : public BenchmarkAdapter {
 public:
  explicit ExclusiveScanAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  std::string operator_name() const override { return "exclusive_scan"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    if (variant_name_ == "cuda_warp_blocked_scalar_legacy") {
      return "Historical 32-lane two-items-per-lane scan research baseline";
    }
    if (variant_name_ == "cuda_subwarp4_scalar") {
      return "SM86 16-lane four-items-per-lane scalar scan candidate";
    }
    if (variant_name_ == "cuda_subwarp4_vector") {
      return "SM86 16-lane four-items-per-lane int4 scan candidate";
    }
    return "Single-thread int32 exclusive scan for tiny expert counts";
  }
  bool supports(MeasurementLevel level) const override {
    if (variant_name_ == "cub_device_scan") {
      return level == MeasurementLevel::kOperatorSteady;
    }
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
#if RAGGEDROUTE_HAS_CCCL
    if (variant_name_ == "cub_device_scan") {
      cuda_check(
          library_baseline::query_cub_device_scan_workspace(experts_, &library_workspace_bytes_),
          "query CUB DeviceScan workspace");
      library_workspace_.resize(library_workspace_bytes_);
    }
    if (variant_name_ == "cub_warp_scan" && experts_ > 32) {
      throw std::invalid_argument("cub_warp_scan requires E<=32");
    }
#endif
  }
  void prepare_sample(MeasurementLevel, cudaStream_t) override {}
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
#if RAGGEDROUTE_HAS_CCCL
    if (variant_name_ == "cub_device_scan") {
      cuda_check(library_baseline::launch_cub_device_scan(counts_.data(), offsets_.data(), experts_,
                                                          library_workspace_.data(),
                                                          library_workspace_bytes_, stream),
                 "CUB DeviceScan::ExclusiveSum");
      return;
    }
    if (variant_name_ == "cub_block_scan") {
      cuda_check(library_baseline::launch_cub_block_scan(counts_.data(), offsets_.data(), experts_,
                                                         stream),
                 "CUB BlockScan::ExclusiveSum");
      return;
    }
    if (variant_name_ == "cub_warp_scan") {
      cuda_check(
          library_baseline::launch_cub_warp_scan(counts_.data(), offsets_.data(), experts_, stream),
          "CUB WarpScan::ExclusiveSum");
      return;
    }
#endif
    if (is_optimized_scan_variant(variant_name_)) {
      const std::uint32_t implementation = optimized_scan_implementation(variant_name_);
      if (level == MeasurementLevel::kKernelBody) {
        cuda_check(ops::launch_exclusive_scan_optimized(counts_.data(), offsets_.data(), experts_,
                                                        implementation, stream),
                   "launch_exclusive_scan_optimized");
        return;
      }
      ExclusiveScanArgs args;
      args.counts = counts_.data();
      args.offsets = offsets_.data();
      args.experts = experts_;
      args.kernel = {KernelFamily::kCudaOptimized, implementation};
      operator_check(exclusive_scan(args, make_runtime_context(stream, architecture_)),
                     "optimized exclusive_scan operator");
      return;
    }
    if (level == MeasurementLevel::kKernelBody) {
      cuda_check(
          ops::launch_exclusive_scan_naive(counts_.data(), offsets_.data(), experts_, stream),
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
    if (variant_name_ == "cub_device_scan") {
      return {{"api", std::string("cub::DeviceScan::ExclusiveSum")}, {"completion_kernel", true}};
    }
    if (variant_name_ == "cub_block_scan") {
      return {{"api", std::string("cub::BlockScan::ExclusiveSum")},
              {"threads", static_cast<std::int64_t>(128)}};
    }
    if (variant_name_ == "cub_warp_scan") {
      return {{"api", std::string("cub::WarpScan::ExclusiveSum")},
              {"threads", static_cast<std::int64_t>(32)}};
    }
    if (variant_name_ == "cuda_warp_blocked_scalar_legacy") {
      return {{"threads", static_cast<std::int64_t>(32)},
              {"items_per_thread", static_cast<std::int64_t>(2)},
              {"algorithm", std::string("historical_warp_blocked_scalar")},
              {"promotion_eligible", false}};
    }
    if (variant_name_ == "cuda_subwarp4_scalar" ||
        variant_name_ == "cuda_subwarp4_vector") {
      const bool vector_path =
          variant_name_ == "cuda_subwarp4_vector" && experts_ % 4 == 0;
      return {{"threads", static_cast<std::int64_t>(16)},
              {"items_per_thread", static_cast<std::int64_t>(4)},
              {"algorithm", std::string("subwarp4_blocked_scan")},
              {"memory_path", vector_path ? std::string("int4_when_aligned")
                                           : std::string("scalar")}};
    }
    return {{"threads", static_cast<std::int64_t>(1)}, {"algorithm", std::string("sequential")}};
  }
  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    work.logical_bytes = sizeof(std::int32_t) * static_cast<double>(experts_ + experts_ + 1);
    work.operator_metrics["integer_additions"] = static_cast<std::int64_t>(experts_);
    return work;
  }
  std::size_t workspace_bytes() const override { return library_workspace_bytes_; }
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
  std::string variant_name_;
  std::size_t library_workspace_bytes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<std::int32_t> counts_host_, expected_;
  DeviceBuffer<std::int32_t> counts_, offsets_;
  DeviceBuffer<std::uint8_t> library_workspace_;
};

}  // namespace

AdapterPtr make_exclusive_scan_adapter(const std::string& variant_name) {
  return std::make_unique<ExclusiveScanAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
