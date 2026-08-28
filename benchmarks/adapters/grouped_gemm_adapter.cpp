#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "grouped_gemm/cuda_candidate/grouped_optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {
namespace {

bool is_optimized_grouped_variant(const std::string& variant) {
  return variant == "cuda_grouped_tiled16_sync_v0" ||
         variant == "cuda_grouped_persistent16_v1" ||
         variant == "cuda_grouped_register16x32_sync_v2" ||
         variant == "cuda_grouped_register16x32_async_v3" ||
         variant == "cuda_grouped_register16x32_async_full_v4" ||
         variant == "cuda_grouped_sm86_fp32_v2" ||
         variant == "cuda_grouped_sm86_fp32_v3" ||
         variant == "cuda_grouped_sm86_fp32_v1";
}

bool is_descriptor_grouped_variant(const std::string& variant) {
  return variant == "cuda_grouped_sm86_fp32_v4a_desc_static_t256" ||
         variant == "cuda_grouped_sm86_fp32_v4a_desc_static_t512" ||
         variant == "cuda_grouped_sm86_fp32_v4a_desc_static_t1024" ||
         variant == "cuda_grouped_sm86_fp32_v4a_desc_queue_t256" ||
         variant == "cuda_grouped_sm86_fp32_v4a_desc_queue_t512" ||
         variant == "cuda_grouped_sm86_fp32_v4a_desc_queue_t1024" ||
         variant == "cuda_grouped_sm86_fp32_v4a_desc" ||
         variant == "cuda_grouped_sm86_fp32_v4b_cache_order";
}

std::uint32_t descriptor_grouped_implementation(const std::string& variant) {
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc_static_t256") {
    return ops::kGroupedGemmSm86Fp32V4ADescStaticT256Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc_static_t512") {
    return ops::kGroupedGemmSm86Fp32V4ADescStaticT512Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc_static_t1024") {
    return ops::kGroupedGemmSm86Fp32V4ADescStaticT1024Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc_queue_t256") {
    return ops::kGroupedGemmSm86Fp32V4ADescQueueT256Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc_queue_t512") {
    return ops::kGroupedGemmSm86Fp32V4ADescQueueT512Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc_queue_t1024") {
    return ops::kGroupedGemmSm86Fp32V4ADescQueueT1024Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4a_desc") {
    return ops::kGroupedGemmSm86Fp32V4ADescImplementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v4b_cache_order") {
    return ops::kGroupedGemmSm86Fp32V4BCacheOrderImplementation;
  }
  throw std::invalid_argument("unsupported descriptor grouped_gemm variant: " + variant);
}

int descriptor_prepass_threads(const std::string& variant) {
  if (variant.find("t1024") != std::string::npos) return 1024;
  if (variant.find("t512") != std::string::npos) return 512;
  return 256;
}

std::uint32_t optimized_grouped_implementation(const std::string& variant) {
  if (variant == "cuda_grouped_tiled16_sync_v0") {
    return ops::kGroupedGemmTiled16SyncV0Implementation;
  }
  if (variant == "cuda_grouped_persistent16_v1") {
    return ops::kGroupedGemmPersistent16V1Implementation;
  }
  if (variant == "cuda_grouped_register16x32_sync_v2") {
    return ops::kGroupedGemmRegister16x32SyncV2Implementation;
  }
  if (variant == "cuda_grouped_register16x32_async_v3") {
    return ops::kGroupedGemmRegister16x32AsyncV3Implementation;
  }
  if (variant == "cuda_grouped_register16x32_async_full_v4") {
    return ops::kGroupedGemmRegister16x32AsyncFullV4Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v1") {
    return ops::kGroupedGemmSm86Fp32V1Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v2") {
    return ops::kGroupedGemmSm86Fp32V2Implementation;
  }
  if (variant == "cuda_grouped_sm86_fp32_v3") {
    return ops::kGroupedGemmSm86Fp32V3Implementation;
  }
  throw std::invalid_argument("unsupported optimized grouped_gemm variant: " + variant);
}

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
    if (variant_name_ == "cuda_grouped_tiled16_sync_v0") {
      return "16x16 synchronous shared-memory strict-FP32 grouped GEMM";
    }
    if (variant_name_ == "cuda_grouped_persistent16_v1") {
      return "Persistent-CTA 16x16 strict-FP32 grouped GEMM";
    }
    if (variant_name_ == "cuda_grouped_register16x32_sync_v2") {
      return "Persistent 16x32 register-tiled synchronous strict-FP32 grouped GEMM";
    }
    if (variant_name_ == "cuda_grouped_register16x32_async_v3") {
      return "Persistent 16x32 register-tiled cp.async strict-FP32 grouped GEMM";
    }
    if (variant_name_ == "cuda_grouped_register16x32_async_full_v4") {
      return "Full-residency persistent 16x32 cp.async strict-FP32 grouped GEMM";
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v1") {
      return "Explicit SM86 strict-FP32 grouped GEMM with direct/persistent selection";
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v2") {
      return "SM86 v2 warp-prefix launch-bounds grouped GEMM (research candidate)";
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v3") {
      return "SM86 v3 16x64 cp.async large-aligned grouped GEMM (research candidate)";
    }
    if (is_descriptor_grouped_variant(variant_name_)) {
      return "SM86 v4 descriptor-prepass 16x32 cp.async grouped GEMM (research candidate)";
    }
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
    route_trace_path_ = get_option(options, "route_trace_path", "");
    route_trace_frame_ = get_int_option(options, "route_trace_frame", 0, 0);
    if (experts_ < top_k_ || experts_ > 64) {
      throw std::invalid_argument("current grouped GEMM adapter requires top_k<=E<=64");
    }
    route_pairs_ = checked_int_product(tokens_, top_k_, "R=T*top_k");

    std::vector<std::int32_t> ids;
    if (route_trace_path_.empty()) {
      ids = make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed + 1);
    } else {
      const RouteTraceFrame trace = load_route_trace_frame(route_trace_path_, route_trace_frame_);
      if (trace.tokens != tokens_ || trace.experts != experts_ || trace.top_k != top_k_) {
        throw std::invalid_argument("route trace shape does not match grouped_gemm params");
      }
      ids = trace.expert_ids;
      route_trace_id_ = trace.trace_id;
      route_trace_source_kind_ = trace.source_kind;
      route_trace_frame_id_ = trace.frame_id;
      route_trace_frame_count_ = trace.frame_count;
      distribution_ = "route_trace";
      zipf_s_ = 0.0;
    }
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
    if (is_descriptor_grouped_variant(variant_name_)) {
      candidate_workspace_bytes_ = ops::grouped_gemm_descriptor_workspace_size(
          experts_, output_, max_expert_tokens_);
      if (candidate_workspace_bytes_ == 0 ||
          candidate_workspace_bytes_ > 64ULL * 1024ULL * 1024ULL) {
        throw std::runtime_error("descriptor grouped GEMM workspace exceeds research limit");
      }
      candidate_workspace_.resize(candidate_workspace_bytes_);
    }
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
    if (is_descriptor_grouped_variant(variant_name_)) {
      cuda_check(ops::launch_grouped_gemm_sm86_fp32_v4_descriptor(
                     x_.data(), weights_.data(), offsets_.data(), output_buffer_.data(), experts_,
                     hidden_, output_, max_expert_tokens_, candidate_workspace_.data(),
                     candidate_workspace_bytes_, descriptor_grouped_implementation(variant_name_),
                     stream),
                 "launch_grouped_gemm_sm86_fp32_v4_descriptor benchmark-only candidate");
      return;
    }
    if (is_optimized_grouped_variant(variant_name_)) {
      const std::uint32_t implementation = optimized_grouped_implementation(variant_name_);
      cuda_check(ops::launch_grouped_gemm_optimized(
                     x_.data(), weights_.data(), offsets_.data(), output_buffer_.data(), experts_,
                     hidden_, output_, max_expert_tokens_, implementation, stream),
                 "launch_grouped_gemm_optimized benchmark-only candidate");
      return;
    }
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
    FieldMap config = {{"T", static_cast<std::int64_t>(tokens_)},
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
    if (!route_trace_path_.empty()) {
      config["workload_source"] = route_trace_source_kind_;
      config["route_trace_id"] = route_trace_id_;
      config["route_trace_frame_id"] = route_trace_frame_id_;
      config["route_trace_frame"] = static_cast<std::int64_t>(route_trace_frame_);
      config["route_trace_frame_count"] = static_cast<std::int64_t>(route_trace_frame_count_);
    }
    return config;
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
    if (variant_name_ == "cuda_grouped_tiled16_sync_v0") {
      return {{"tile_m", static_cast<std::int64_t>(16)},
              {"tile_n", static_cast<std::int64_t>(16)},
              {"tile_k", static_cast<std::int64_t>(16)},
              {"threads_per_block", static_cast<std::int64_t>(256)},
              {"scheduler", std::string("grid_z_per_expert")},
              {"staging", std::string("synchronous_shared_memory")}};
    }
    if (variant_name_ == "cuda_grouped_persistent16_v1") {
      return {{"tile_m", static_cast<std::int64_t>(16)},
              {"tile_n", static_cast<std::int64_t>(16)},
              {"tile_k", static_cast<std::int64_t>(16)},
              {"threads_per_block", static_cast<std::int64_t>(256)},
              {"scheduler", std::string("device_prefix_persistent_round_robin")},
              {"staging", std::string("synchronous_shared_memory")}};
    }
    if (variant_name_ == "cuda_grouped_register16x32_sync_v2" ||
        variant_name_ == "cuda_grouped_register16x32_async_v3" ||
        variant_name_ == "cuda_grouped_register16x32_async_full_v4") {
      const bool asynchronous = variant_name_ != "cuda_grouped_register16x32_sync_v2";
      const bool full_residency =
          variant_name_ == "cuda_grouped_register16x32_async_full_v4";
      return {{"tile_m", static_cast<std::int64_t>(16)},
              {"tile_n", static_cast<std::int64_t>(32)},
              {"tile_k", static_cast<std::int64_t>(16)},
              {"threads_per_block", static_cast<std::int64_t>(128)},
              {"outputs_per_thread", static_cast<std::int64_t>(4)},
              {"scheduler", std::string("device_prefix_persistent_round_robin")},
              {"resident_blocks", std::string(full_residency ? "occupancy_api_full"
                                                             : "capped_at_2_per_sm")},
              {"staging", std::string(asynchronous ? "sm86_cp_async_double_buffered"
                                                    : "synchronous_shared_memory")},
              {"fallback", std::string(asynchronous ? "register16x32_sync_for_tail"
                                                     : "none")}};
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v1") {
      return {{"scheduler", std::string("explicit_direct_or_persistent")},
              {"direct_path", std::string("tiled16_sync_v0")},
              {"persistent_path", std::string("register16x32_async_full_v4")},
              {"selection", std::string("device_attributes_occupancy_and_upper_tiles")},
              {"math_path", std::string("cuda_core_strict_fp32")},
              {"runtime_status", std::string("benchmark_only_not_promoted")}};
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v2") {
      return {{"algorithm_id", std::string("warp_prefix_launch_bounds_5")},
              {"tile_m", static_cast<std::int64_t>(16)},
              {"tile_n", static_cast<std::int64_t>(32)},
              {"tile_k", static_cast<std::int64_t>(16)},
              {"threads_per_block", static_cast<std::int64_t>(128)},
              {"outputs_per_thread", static_cast<std::int64_t>(4)},
              {"scheduler", std::string("warp_prefix_persistent_round_robin")},
              {"resident_blocks", std::string("launch_bounds_min_5_per_sm")},
              {"staging", std::string("sm86_cp_async_double_buffered")},
              {"fallback", std::string("register16x32_sync_for_tail")},
              {"math_mode", std::string("strict_fp32")},
              {"runtime_status", std::string("explicit_research_candidate")}};
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v3") {
      return {{"algorithm_id", std::string("large_aligned_register16x64_v3")},
              {"tile_m", static_cast<std::int64_t>(16)},
              {"tile_n", static_cast<std::int64_t>(64)},
              {"tile_k", static_cast<std::int64_t>(16)},
              {"threads_per_block", static_cast<std::int64_t>(256)},
              {"outputs_per_thread", static_cast<std::int64_t>(4)},
              {"scheduler", std::string("warp_prefix_persistent_round_robin")},
              {"staging", std::string("sm86_cp_async_double_buffered")},
              {"selection",
               std::string("aligned_k16_n64_max_expert_tokens_ge_32_else_v2")},
              {"fallback", std::string("cuda_grouped_sm86_fp32_v2")},
              {"math_mode", std::string("strict_fp32")},
               {"runtime_status", std::string("explicit_research_candidate")}};
    }
    if (is_descriptor_grouped_variant(variant_name_)) {
      const bool queue = variant_name_.find("_queue_") != std::string::npos;
      const bool cache_order = variant_name_ == "cuda_grouped_sm86_fp32_v4b_cache_order";
      return {{"algorithm_id", std::string(cache_order ? "descriptor_cache_order_v4b"
                                                        : "descriptor_prepass_v4a")},
              {"tile_m", static_cast<std::int64_t>(16)},
              {"tile_n", static_cast<std::int64_t>(32)},
              {"tile_k", static_cast<std::int64_t>(16)},
              {"gemm_threads_per_block", static_cast<std::int64_t>(128)},
              {"prepass_threads_per_block",
               static_cast<std::int64_t>(descriptor_prepass_threads(variant_name_))},
              {"scheduler", std::string(queue ? "descriptor_global_atomic_queue"
                                               : "descriptor_static_round_robin")},
              {"descriptor_order", std::string(cache_order ? "column_major_within_expert"
                                                             : "row_major_within_expert")},
              {"staging", std::string("sm86_cp_async_double_buffered")},
              {"fallback", std::string("cuda_grouped_sm86_fp32_v2")},
              {"math_mode", std::string("strict_fp32")},
              {"runtime_status", std::string("explicit_research_candidate")}};
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
  std::size_t workspace_bytes() const override {
    return library_workspace_bytes_ + candidate_workspace_bytes_;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    return {"route_generation", "offset_preparation", "input_generation", "h2d_copy",
            "workspace_allocation"};
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T", "E", "top_k", "K", "N", "distribution", "zipf_s",
                                           "route_trace_path", "route_trace_frame"};
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
  std::size_t candidate_workspace_bytes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  int route_pairs_ = 0, max_expert_tokens_ = 0, active_experts_ = 0;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::string route_trace_path_, route_trace_id_, route_trace_source_kind_, route_trace_frame_id_;
  int route_trace_frame_ = 0, route_trace_frame_count_ = 0;
  std::vector<std::int32_t> counts_, offsets_host_;
  std::vector<float> x_host_, weights_host_, expected_;
  DeviceBuffer<std::int32_t> offsets_;
  DeviceBuffer<float> x_, weights_, output_buffer_;
  DeviceBuffer<std::uint8_t> library_workspace_;
  DeviceBuffer<std::uint8_t> candidate_workspace_;
};

}  // namespace

AdapterPtr make_grouped_gemm_adapter(const std::string& variant_name) {
  return std::make_unique<GroupedGemmAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
