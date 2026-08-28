#include <algorithm>
#include <chrono>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "grouped_gemm/cuda_candidate/grouped_optimized_internal.h"
#include "permute/cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/registry.h"
#include "scan/cuda_candidate/optimized_internal.h"
#include "unpermute/cuda_candidate/optimized_internal.h"

namespace raggedroute::benchmark {
namespace {

bool is_shared_rank_postroute(const std::string& variant) {
  return variant == "cuda_postroute_shared_rank_t256_v1" ||
         variant == "cuda_postroute_shared_rank_t512_v1" ||
         variant == "cuda_postroute_shared_rank_t1024_v1" ||
         variant == "cuda_postroute_token_owned_v4" ||
         variant == "cuda_postroute_grouped_v4a_desc" ||
         variant == "cuda_postroute_gather_grouped_v1" ||
         variant == "cuda_postroute_graph_fixed_v1" ||
         variant == "cuda_postroute_graph_param_update_v2" ||
         variant == "cuda_postroute_graph_exec_update_v3" ||
         variant == "cuda_postroute_graph_cache_v4";
}

int routeprep_threads(const std::string& variant) {
  if (variant == "cuda_postroute_shared_rank_t512_v1") return 512;
  if (variant == "cuda_postroute_shared_rank_t1024_v1") return 1024;
  return 256;
}

bool uses_token_owned_copy(const std::string& variant) {
  return variant == "cuda_postroute_token_owned_v4" || variant == "cuda_postroute_grouped_v4a_desc";
}

bool uses_descriptor_grouped(const std::string& variant) {
  return variant == "cuda_postroute_grouped_v4a_desc";
}

bool uses_gather_grouped(const std::string& variant) {
  return variant == "cuda_postroute_gather_grouped_v1" ||
         variant == "cuda_postroute_graph_fixed_v1" ||
         variant == "cuda_postroute_graph_param_update_v2" ||
         variant == "cuda_postroute_graph_exec_update_v3" ||
         variant == "cuda_postroute_graph_cache_v4";
}

bool is_graph_variant(const std::string& variant) {
  return variant == "cuda_postroute_graph_fixed_v1" ||
         variant == "cuda_postroute_graph_param_update_v2" ||
         variant == "cuda_postroute_graph_exec_update_v3" ||
         variant == "cuda_postroute_graph_cache_v4";
}

bool uses_graph_pointer_rotation(const std::string& variant) {
  return variant == "cuda_postroute_graph_param_update_v2" ||
         variant == "cuda_postroute_graph_exec_update_v3";
}

struct PostRouteCacheRequest {
  int tokens = 0;
  int top_k = 0;
  int route_pairs = 0;
  int active_experts = 0;
  std::vector<float> route_weights;
  std::vector<float> expected;
  std::vector<std::int32_t> ids;
  std::vector<std::int32_t> counts;
  std::vector<std::int32_t> offsets;
};

struct PostRouteGraphCacheKey {
  std::string variant;
  int tokens = 0;
  int experts = 0;
  int top_k = 0;
  int hidden = 0;
  int output = 0;
  std::string dtype;
  std::string topology_class;

  bool operator==(const PostRouteGraphCacheKey& other) const {
    return variant == other.variant && tokens == other.tokens && experts == other.experts &&
           top_k == other.top_k && hidden == other.hidden && output == other.output &&
           dtype == other.dtype && topology_class == other.topology_class;
  }
};

struct PostRouteGraphCacheEntry {
  PostRouteGraphCacheKey key;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  std::uint64_t last_use = 0;
  std::size_t kernel_nodes = 0;
};

class PostRouteChainAdapter final : public BenchmarkAdapter {
 public:
  explicit PostRouteChainAdapter(std::string variant) : variant_name_(std::move(variant)) {}
  ~PostRouteChainAdapter() override {
    destroy_graph_cache();
    if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
    if (graph_rotating_ != nullptr) cudaGraphDestroy(graph_rotating_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    if (graph_update_stream_ != nullptr) cudaStreamDestroy(graph_update_stream_);
  }

  std::string operator_name() const override { return "chain_from_route_ids"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    if (uses_gather_grouped(variant_name_)) {
      return "Five-stage post-routing chain with shared-rank mapping and Permute-to-Grouped gather "
             "fusion";
    }
    if (uses_descriptor_grouped(variant_name_)) {
      return "Five-stage post-routing chain with token-owned copy and Grouped descriptor prepass";
    }
    if (uses_token_owned_copy(variant_name_)) {
      return "Five-stage post-routing chain with Top-K token-owned payload copy";
    }
    if (is_shared_rank_postroute(variant_name_)) {
      return "Five-stage post-routing chain with single-CTA shared-rank route preparation";
    }
    if (variant_name_ == "cuda_postroute_current_v1") {
      return "Five-stage post-routing chain with retained fused scan, Permute v2 and Grouped v2";
    }
    return "Five-stage post-routing CUDA-naive chain from prepared route IDs and weights";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kChainSteady || level == MeasurementLevel::kHostCall;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 64);
    experts_ = get_int_option(options, "E", 16, 2);
    top_k_ = get_int_option(options, "top_k", 2, 1);
    hidden_ = get_int_option(options, "K", 64);
    output_ = get_int_option(options, "N", 64);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.4);
    graph_cache_tokens_text_ = get_option(options, "graph_cache_tokens", std::to_string(tokens_));
    graph_cache_topks_text_ = get_option(options, "graph_cache_topks", std::to_string(top_k_));
    std::vector<int> request_tokens = {tokens_};
    std::vector<int> request_topks = {top_k_};
    if (variant_name_ == "cuda_postroute_graph_cache_v4") {
      request_tokens = parse_positive_int_list(graph_cache_tokens_text_, "graph_cache_tokens");
      request_topks = parse_positive_int_list(graph_cache_topks_text_, "graph_cache_topks");
    }
    max_tokens_capacity_ = 0;
    max_route_pairs_capacity_ = 0;
    for (const int request_tokens_value : request_tokens) {
      (void)checked_int_product(request_tokens_value, output_, "cache request T*N");
      max_tokens_capacity_ = std::max(max_tokens_capacity_, request_tokens_value);
      for (const int request_top_k : request_topks) {
        validate_route_shape(request_tokens_value, request_top_k);
        max_route_pairs_capacity_ = std::max(
            max_route_pairs_capacity_,
            checked_int_product(request_tokens_value, request_top_k, "cache request R=T*top_k"));
      }
    }

    host_x_ = make_random_floats(static_cast<std::size_t>(max_tokens_capacity_) * hidden_, seed);
    host_weights_ =
        make_random_floats(static_cast<std::size_t>(experts_) * hidden_ * output_, seed + 1);
    std::set<std::pair<int, int>> unique_shapes;
    std::uint64_t request_seed = seed + 2;
    for (const int request_tokens_value : request_tokens) {
      for (const int request_top_k : request_topks) {
        if (!unique_shapes.emplace(request_tokens_value, request_top_k).second) continue;
        cache_requests_.push_back(
            make_cache_request(request_tokens_value, request_top_k, request_seed));
        request_seed += 17;
      }
    }
    if (cache_requests_.empty()) throw std::invalid_argument("graph cache request set is empty");
    apply_cache_request(0);

    x_.resize(host_x_.size());
    expert_weights_.resize(host_weights_.size());
    ids_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    route_weights_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    counts_.resize(host_counts_.size());
    offsets_.resize(host_offsets_.size());
    cursors_.resize(static_cast<std::size_t>(experts_));
    x_permuted_.resize(static_cast<std::size_t>(max_route_pairs_capacity_) * hidden_);
    route_pos_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    sorted_route_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    y_permuted_.resize(static_cast<std::size_t>(max_route_pairs_capacity_) * output_);
    y_.resize(static_cast<std::size_t>(max_tokens_capacity_) * output_);
    x_.copy_from_host(host_x_, stream);
    expert_weights_.copy_from_host(host_weights_, stream);
    copy_active_request_to_device(stream);

    if (uses_graph_pointer_rotation(variant_name_)) {
      x_rotating_.resize(host_x_.size());
      y_rotating_.resize(static_cast<std::size_t>(max_tokens_capacity_) * output_);
      x_rotating_.copy_from_host(host_x_, stream);
    }

    if (uses_descriptor_grouped(variant_name_) || uses_gather_grouped(variant_name_)) {
      descriptor_workspace_bytes_ =
          ops::grouped_gemm_descriptor_workspace_size(experts_, output_, max_route_pairs_capacity_);
      if (descriptor_workspace_bytes_ > kWorkspaceLimit) {
        throw std::invalid_argument("post-route descriptor workspace exceeds 64 MiB");
      }
      descriptor_workspace_.resize(descriptor_workspace_bytes_);
    }
    if (is_graph_variant(variant_name_)) initialize_graph(stream);
  }

  void prepare_sample(MeasurementLevel, cudaStream_t stream) override {
    if (variant_name_ != "cuda_postroute_graph_cache_v4" || cache_requests_.size() <= 1) return;
    cache_request_cursor_ = (cache_request_cursor_ + 1) % cache_requests_.size();
    apply_cache_request(cache_request_cursor_);
    copy_active_request_to_device(stream);
  }

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level != MeasurementLevel::kChainSteady && level != MeasurementLevel::kHostCall) {
      throw std::invalid_argument("chain_from_route_ids supports L3 and graph L4");
    }
    if (is_graph_variant(variant_name_)) {
      launch_graph(stream);
      return;
    }
    enqueue_direct(stream);
  }

  void enqueue_direct(cudaStream_t stream) {
    const RuntimeContext context = make_runtime_context(stream, architecture_);
    if (is_shared_rank_postroute(variant_name_)) {
      cuda_check(
          ops::launch_token_permute_routeprep_shared_rank(
              ids_.data(), counts_.data(), offsets_.data(), route_pos_.data(), sorted_route_.data(),
              tokens_, experts_, top_k_, routeprep_threads(variant_name_), stream),
          "post-route shared-rank preparation");
      if (!uses_gather_grouped(variant_name_)) {
        if (uses_token_owned_copy(variant_name_)) {
          cuda_check(ops::launch_token_permute_copy_token_owned_topk(
                         active_x_data(), route_pos_.data(), x_permuted_.data(), tokens_, top_k_,
                         hidden_, stream),
                     "post-route token-owned payload copy");
        } else {
          cuda_check(ops::launch_token_permute_copy_from_positions(
                         active_x_data(), route_pos_.data(), x_permuted_.data(), tokens_, top_k_,
                         hidden_, stream),
                     "post-route route-owned payload copy");
        }
      }
    } else {
      if (variant_name_ == "cuda_postroute_current_v1") {
        HistogramExclusiveScanArgs fused;
        fused.expert_ids = ids_.data();
        fused.counts = counts_.data();
        fused.offsets = offsets_.data();
        fused.route_pairs = route_pairs_;
        fused.experts = experts_;
        fused.kernel = {KernelFamily::kCudaOptimized,
                        ops::kHistogramExclusiveScanFusedSubwarpImplementation};
        operator_check(histogram_exclusive_scan(fused, context), "post-route fused histogram scan");
      } else {
        HistogramArgs histogram_args;
        histogram_args.expert_ids = ids_.data();
        histogram_args.counts = counts_.data();
        histogram_args.route_pairs = route_pairs_;
        histogram_args.experts = experts_;
        operator_check(histogram(histogram_args, context), "post-route histogram");
        ExclusiveScanArgs scan_args;
        scan_args.counts = counts_.data();
        scan_args.offsets = offsets_.data();
        scan_args.experts = experts_;
        operator_check(exclusive_scan(scan_args, context), "post-route scan");
      }
      TokenPermuteArgs permute_args;
      permute_args.x.data = active_x_data();
      permute_args.expert_ids = ids_.data();
      permute_args.offsets = offsets_.data();
      permute_args.x_permuted.data = x_permuted_.data();
      permute_args.route_pos = route_pos_.data();
      permute_args.sorted_route = sorted_route_.data();
      permute_args.tokens = tokens_;
      permute_args.experts = experts_;
      permute_args.top_k = top_k_;
      permute_args.hidden = hidden_;
      if (variant_name_ == "cuda_postroute_current_v1") {
        permute_args.kernel = {KernelFamily::kCudaOptimized,
                               ops::kTokenPermuteShapeDispatchedV2Implementation};
      }
      operator_check(
          token_permute(permute_args, make_runtime_context(stream, architecture_, cursors_.data(),
                                                           cursors_.bytes())),
          "post-route token permute");
    }

    if (uses_gather_grouped(variant_name_)) {
      cuda_check(ops::launch_grouped_gemm_sm86_fp32_gather_v1(
                     active_x_data(), sorted_route_.data(), top_k_, expert_weights_.data(),
                     offsets_.data(), y_permuted_.data(), experts_, hidden_, output_, route_pairs_,
                     descriptor_workspace_.data(), descriptor_workspace_bytes_, stream),
                 "post-route gathered grouped GEMM");
    } else if (uses_descriptor_grouped(variant_name_)) {
      cuda_check(
          ops::launch_grouped_gemm_sm86_fp32_v4_descriptor(
              x_permuted_.data(), expert_weights_.data(), offsets_.data(), y_permuted_.data(),
              experts_, hidden_, output_, route_pairs_, descriptor_workspace_.data(),
              descriptor_workspace_bytes_, ops::kGroupedGemmSm86Fp32V4ADescImplementation, stream),
          "post-route descriptor grouped GEMM");
    } else if (variant_name_ != "cuda_naive") {
      cuda_check(ops::launch_grouped_gemm_optimized(
                     x_permuted_.data(), expert_weights_.data(), offsets_.data(),
                     y_permuted_.data(), experts_, hidden_, output_, route_pairs_,
                     ops::kGroupedGemmSm86Fp32V2Implementation, stream),
                 "post-route Grouped GEMM v2");
    } else {
      GroupedGemmArgs grouped;
      grouped.x_permuted.data = x_permuted_.data();
      grouped.expert_weights.data = expert_weights_.data();
      grouped.offsets = offsets_.data();
      grouped.y_permuted.data = y_permuted_.data();
      grouped.experts = experts_;
      grouped.hidden = hidden_;
      grouped.output = output_;
      grouped.max_expert_tokens = route_pairs_;
      operator_check(grouped_gemm(grouped, context), "post-route grouped GEMM");
    }

    if (variant_name_ == "cuda_naive") {
      UnpermuteArgs unpermute_args;
      unpermute_args.y_permuted.data = y_permuted_.data();
      unpermute_args.route_pos = route_pos_.data();
      unpermute_args.route_weights.data = route_weights_.data();
      unpermute_args.y.data = active_y_data();
      unpermute_args.tokens = tokens_;
      unpermute_args.top_k = top_k_;
      unpermute_args.output = output_;
      operator_check(unpermute(unpermute_args, context), "post-route unpermute");
    } else {
      cuda_check(ops::launch_unpermute_optimized(y_permuted_.data(), route_pos_.data(),
                                                 route_weights_.data(), active_y_data(), tokens_,
                                                 top_k_, output_, stream),
                 "post-route retained unpermute");
    }
  }

  ValidationResult validate(cudaStream_t stream) override {
    if (counts_.copy_to_host(stream) != host_counts_) {
      return {false, "post-route counts differ from CPU oracle", {}, {}};
    }
    if (offsets_.copy_to_host(stream) != host_offsets_) {
      return {false, "post-route offsets differ from CPU oracle", {}, {}};
    }
    const auto route_pos = route_pos_.copy_to_host(stream);
    const auto sorted_route = sorted_route_.copy_to_host(stream);
    std::vector<bool> seen(static_cast<std::size_t>(route_pairs_), false);
    for (int route = 0; route < route_pairs_; ++route) {
      const int packed = route_pos[static_cast<std::size_t>(route)];
      const int expert = host_ids_[static_cast<std::size_t>(route)];
      if (packed < host_offsets_[static_cast<std::size_t>(expert)] ||
          packed >= host_offsets_[static_cast<std::size_t>(expert + 1)] ||
          seen[static_cast<std::size_t>(packed)] ||
          sorted_route[static_cast<std::size_t>(packed)] != route) {
        return {false, "post-route mapping is not an expert-segment bijection", {}, {}};
      }
      seen[static_cast<std::size_t>(packed)] = true;
    }
    const auto actual = copy_active_output_to_host(stream);
    auto result = compare_floats(actual, expected_, 1.0e-4, 4.0e-5 * hidden_ * top_k_);
    if (!result.ok) result.message = "post-route final output: " + result.message;
    return result;
  }

  FieldMap case_config() const override {
    FieldMap config = {{"T", static_cast<std::int64_t>(tokens_)},
                       {"E", static_cast<std::int64_t>(experts_)},
                       {"top_k", static_cast<std::int64_t>(top_k_)},
                       {"R", static_cast<std::int64_t>(route_pairs_)},
                       {"K", static_cast<std::int64_t>(hidden_)},
                       {"N", static_cast<std::int64_t>(output_)},
                       {"dtype", std::string("fp32")},
                       {"distribution", distribution_},
                       {"zipf_s", zipf_s_},
                       {"chain_entry", std::string("prepared_route_ids_and_weights")},
                       {"chain_scope", std::string("five_stage_post_routing_l3")},
                       {"included_operator_count", static_cast<std::int64_t>(5)},
                       {"active_experts", static_cast<std::int64_t>(active_experts_)}};
    return config;
  }

  FieldMap variant_config() const override {
    FieldMap config = {
        {"components", std::string("histogram,scan,permute,grouped_gemm,unpermute")},
        {"mapping", is_shared_rank_postroute(variant_name_) ? std::string("single_cta_shared_rank")
                    : variant_name_ == "cuda_postroute_current_v1"
                        ? std::string("fused_histogram_scan_then_global_atomic")
                        : std::string("separate_histogram_scan_global_atomic")},
        {"routeprep_threads", static_cast<std::int64_t>(is_shared_rank_postroute(variant_name_)
                                                            ? routeprep_threads(variant_name_)
                                                            : 0)},
        {"payload", uses_gather_grouped(variant_name_)
                        ? std::string("not_materialized_grouped_gather")
                    : uses_token_owned_copy(variant_name_) ? std::string("token_owned_topk")
                                                           : std::string("route_owned")},
        {"grouped", uses_gather_grouped(variant_name_)
                        ? std::string("cuda_postroute_gather_grouped_v1")
                    : uses_descriptor_grouped(variant_name_)
                        ? std::string("cuda_grouped_sm86_fp32_v4a_desc_selected_alias")
                    : variant_name_ == "cuda_naive" ? std::string("cuda_naive")
                                                    : std::string("cuda_grouped_sm86_fp32_v2")},
        {"unpermute", variant_name_ == "cuda_naive"
                          ? std::string("cuda_naive")
                          : std::string("cuda_warp_token_vec4_top2_else_generic_fallback")},
        {"runtime_status", variant_name_ == "cuda_postroute_graph_fixed_v1"
                               ? std::string("promoted_explicit_fixed_shape_only")
                               : std::string("benchmark_only_not_promoted")}};
    if (is_graph_variant(variant_name_)) {
      config["graph_mode"] = variant_name_ == "cuda_postroute_graph_fixed_v1"
                                 ? std::string("fixed_capture_upload_replay")
                             : variant_name_ == "cuda_postroute_graph_param_update_v2"
                                 ? std::string("kernel_node_set_params_double_buffer_rotation")
                             : variant_name_ == "cuda_postroute_graph_exec_update_v3"
                                 ? std::string("recapture_cudaGraphExecUpdate")
                                 : std::string("lru16_shape_key_capture_on_miss");
      config["graph_measurement_boundary"] = std::string("steady_replay_or_dynamic_update");
      config["graph_kernel_nodes"] = static_cast<std::int64_t>(
          variant_name_ == "cuda_postroute_graph_cache_v4" ? graph_cached_kernel_nodes_
                                                           : graph_kernel_nodes_.size());
      config["graph_cache_capacity"] = static_cast<std::int64_t>(16);
      config["graph_cache_entries"] = static_cast<std::int64_t>(graph_cache_.size());
      config["graph_cache_hits"] = static_cast<std::int64_t>(graph_cache_hits_);
      config["graph_cache_misses"] = static_cast<std::int64_t>(graph_cache_misses_);
      std::size_t graph_memory_delta = graph_memory_delta_bytes_;
      if (variant_name_ == "cuda_postroute_graph_cache_v4") {
        std::size_t free_after = 0, total_after = 0;
        cuda_check(cudaMemGetInfo(&free_after, &total_after),
                   "cudaMemGetInfo while reporting graph cache");
        graph_memory_delta = graph_memory_baseline_free_bytes_ > free_after
                                 ? graph_memory_baseline_free_bytes_ - free_after
                                 : 0;
      }
      config["driver_graph_memory_delta_bytes"] = static_cast<std::int64_t>(graph_memory_delta);
      config["graph_capture_instantiate_upload_us"] =
          variant_name_ == "cuda_postroute_graph_cache_v4" ? graph_cache_miss_setup_us_total_
                                                           : graph_setup_us_;
      config["graph_cache_miss_setup_us_total"] = graph_cache_miss_setup_us_total_;
      config["graph_cache_key_fields"] = std::string("variant,T,E,top_k,K,N,dtype,topology_class");
      config["graph_setup_excluded"] = variant_name_ != "cuda_postroute_graph_cache_v4";
      config["graph_pointer_rotation"] = uses_graph_pointer_rotation(variant_name_);
      config["graph_cache_request_keys"] = static_cast<std::int64_t>(cache_requests_.size());
      config["graph_cache_tokens"] = graph_cache_tokens_text_;
      config["graph_cache_topks"] = graph_cache_topks_text_;
      config["mixed_shape_cache_trace"] =
          variant_name_ == "cuda_postroute_graph_cache_v4" && cache_requests_.size() > 1;
      config["graph_working_set_count"] = static_cast<std::int64_t>(
          variant_name_ == "cuda_postroute_graph_cache_v4" ? cache_requests_.size()
          : uses_graph_pointer_rotation(variant_name_)     ? 2
                                                           : 1);
    }
    return config;
  }

  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    double average_tokens = tokens_;
    double average_route_pairs = route_pairs_;
    double average_active_experts = active_experts_;
    if (variant_name_ == "cuda_postroute_graph_cache_v4" && !cache_requests_.empty()) {
      average_tokens = 0.0;
      average_route_pairs = 0.0;
      average_active_experts = 0.0;
      for (const auto& request : cache_requests_) {
        average_tokens += request.tokens;
        average_route_pairs += request.route_pairs;
        average_active_experts += request.active_experts;
      }
      const double request_count = static_cast<double>(cache_requests_.size());
      average_tokens /= request_count;
      average_route_pairs /= request_count;
      average_active_experts /= request_count;
      work.operator_metrics["graph_cache_request_keys"] =
          static_cast<std::int64_t>(cache_requests_.size());
    }
    work.flops =
        2.0 * average_route_pairs * hidden_ * output_ + 2.0 * average_route_pairs * output_;
    work.logical_bytes =
        2.0 * sizeof(std::int32_t) * average_route_pairs + 3.0 * sizeof(std::int32_t) * experts_ +
        sizeof(float) * (average_route_pairs * hidden_ + average_route_pairs * output_ +
                         average_active_experts * hidden_ * output_ + average_tokens * output_);
    if (uses_gather_grouped(variant_name_)) {
      work.operator_metrics["elided_x_permuted_bytes"] =
          static_cast<std::int64_t>(2.0 * sizeof(float) * average_route_pairs * hidden_);
    }
    const std::int64_t physical_kernel_launches =
        variant_name_ == "cuda_postroute_graph_cache_v4" && graph_cached_kernel_nodes_ != 0
            ? static_cast<std::int64_t>(graph_cached_kernel_nodes_)
        : is_graph_variant(variant_name_) && !graph_kernel_nodes_.empty()
            ? static_cast<std::int64_t>(graph_kernel_nodes_.size())
            : static_cast<std::int64_t>(variant_name_ == "cuda_naive"        ? 5
                                        : uses_gather_grouped(variant_name_) ? 4
                                                                             : 4);
    work.operator_metrics["kernel_launches"] = physical_kernel_launches;
    work.operator_metrics["host_graph_launches"] =
        static_cast<std::int64_t>(is_graph_variant(variant_name_) ? 1 : 0);
    return work;
  }

  std::size_t workspace_bytes() const override {
    return cursors_.bytes() + descriptor_workspace_bytes_;
  }
  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    std::vector<std::string> steps = {"route_generation", "weight_generation", "cpu_reference",
                                      "h2d_copy", "workspace_allocation"};
    if (variant_name_ == "cuda_postroute_graph_fixed_v1" ||
        variant_name_ == "cuda_postroute_graph_param_update_v2") {
      steps.insert(steps.end(), {"graph_capture", "graph_instantiate", "graph_upload"});
    }
    return steps;
  }

 private:
  void capture_graph(cudaStream_t stream, cudaGraph_t* graph) {
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin post-route graph capture");
    enqueue_direct(stream);
    cuda_check(cudaStreamEndCapture(stream, graph), "end post-route graph capture");
  }

  static void collect_kernel_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>* nodes_out,
                                   std::vector<cudaKernelNodeParams>* params_out) {
    std::size_t node_count = 0;
    cuda_check(cudaGraphGetNodes(graph, nullptr, &node_count), "query graph node count");
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (node_count != 0) {
      cuda_check(cudaGraphGetNodes(graph, nodes.data(), &node_count), "get graph nodes");
    }
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type{};
      cuda_check(cudaGraphNodeGetType(node, &type), "get graph node type");
      if (type != cudaGraphNodeTypeKernel) continue;
      cudaKernelNodeParams params{};
      cuda_check(cudaGraphKernelNodeGetParams(node, &params), "get graph kernel params");
      if (nodes_out != nullptr) nodes_out->push_back(node);
      params_out->push_back(params);
    }
  }

  void initialize_graph(cudaStream_t stream) {
    // Prime occupancy caches and prove the direct body before capture. Setup work is
    // explicitly excluded from steady-state graph measurements.
    enqueue_direct(stream);
    cuda_check(cudaStreamSynchronize(stream), "post-route graph priming sync");
    std::size_t free_before = 0, total_before = 0;
    cuda_check(cudaMemGetInfo(&free_before, &total_before), "cudaMemGetInfo before graph");
    graph_memory_baseline_free_bytes_ = free_before;
    if (variant_name_ == "cuda_postroute_graph_cache_v4") {
      graph_cache_hits_ = 0;
      graph_cache_misses_ = 0;
      return;
    }
    const auto setup_begin = std::chrono::steady_clock::now();
    capture_graph(stream, &graph_);
    cuda_check(cudaGraphInstantiate(&graph_exec_, graph_, 0), "instantiate post-route graph");
    cuda_check(cudaGraphUpload(graph_exec_, stream), "upload post-route graph");
    cuda_check(cudaStreamSynchronize(stream), "post-route graph upload sync");
    graph_setup_us_ =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - setup_begin)
            .count();
    std::size_t free_after = 0, total_after = 0;
    cuda_check(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after graph");
    graph_memory_delta_bytes_ = free_before > free_after ? free_before - free_after : 0;
    collect_kernel_nodes(graph_, &graph_kernel_nodes_, &graph_kernel_params_primary_);
    if (variant_name_ == "cuda_postroute_graph_param_update_v2") {
      graph_use_rotating_buffers_ = true;
      capture_graph(stream, &graph_rotating_);
      collect_kernel_nodes(graph_rotating_, nullptr, &graph_kernel_params_rotating_);
      graph_use_rotating_buffers_ = false;
      if (graph_kernel_params_primary_.size() != graph_kernel_params_rotating_.size()) {
        throw std::runtime_error("post-route pointer-update graph topology changed");
      }
      for (std::size_t index = 0; index < graph_kernel_params_primary_.size(); ++index) {
        if (graph_kernel_params_primary_[index].func != graph_kernel_params_rotating_[index].func) {
          throw std::runtime_error("post-route pointer-update graph kernel order changed");
        }
      }
    }
    cuda_check(cudaStreamCreateWithFlags(&graph_update_stream_, cudaStreamNonBlocking),
               "create graph update stream");
    graph_cache_misses_ = 1;
  }

  void launch_graph(cudaStream_t stream) {
    if (variant_name_ == "cuda_postroute_graph_cache_v4") {
      cudaGraphExec_t exec = get_or_create_cached_graph(stream);
      cuda_check(cudaGraphLaunch(exec, stream), "launch cached post-route graph");
      return;
    }
    if (variant_name_ == "cuda_postroute_graph_param_update_v2") {
      graph_use_rotating_buffers_ = !graph_use_rotating_buffers_;
      const auto& params = graph_use_rotating_buffers_ ? graph_kernel_params_rotating_
                                                       : graph_kernel_params_primary_;
      for (std::size_t index = 0; index < graph_kernel_nodes_.size(); ++index) {
        cuda_check(cudaGraphExecKernelNodeSetParams(graph_exec_, graph_kernel_nodes_[index],
                                                    &params[index]),
                   "set post-route graph kernel params");
      }
    } else if (variant_name_ == "cuda_postroute_graph_exec_update_v3") {
      graph_use_rotating_buffers_ = !graph_use_rotating_buffers_;
      cudaGraph_t updated = nullptr;
      capture_graph(graph_update_stream_, &updated);
      cudaGraphExecUpdateResultInfo result_info{};
      const cudaError_t update_error = cudaGraphExecUpdate(graph_exec_, updated, &result_info);
      cudaGraphDestroy(updated);
      cuda_check(update_error, "cudaGraphExecUpdate post-route graph");
      if (result_info.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error("post-route cudaGraphExecUpdate rejected compatible topology");
      }
    }
    cuda_check(cudaGraphLaunch(graph_exec_, stream), "launch post-route graph");
  }

  const float* active_x_data() const {
    return graph_use_rotating_buffers_ ? x_rotating_.data() : x_.data();
  }
  float* active_y_data() { return graph_use_rotating_buffers_ ? y_rotating_.data() : y_.data(); }

  std::vector<float> copy_active_output_to_host(cudaStream_t stream) {
    std::vector<float> actual(static_cast<std::size_t>(tokens_) * output_);
    cuda_check(cudaMemcpyAsync(actual.data(), active_y_data(), actual.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "copy post-route active output");
    cuda_check(cudaStreamSynchronize(stream), "sync post-route active output copy");
    return actual;
  }

  void validate_route_shape(int tokens, int top_k) const {
    if (experts_ > 64 || top_k > experts_) {
      throw std::invalid_argument("post-route chain requires top_k<=E<=64");
    }
    if (top_k != 2 && top_k != 4 && top_k != 8) {
      throw std::invalid_argument("chain_from_route_ids research suite supports top_k=2,4,8");
    }
    (void)checked_int_product(tokens, top_k, "R=T*top_k");
  }

  std::vector<float> build_reference_for(const PostRouteCacheRequest& request) const {
    std::vector<float> expected(static_cast<std::size_t>(request.tokens) * output_, 0.0F);
    for (int token = 0; token < request.tokens; ++token) {
      for (int column = 0; column < output_; ++column) {
        float combined = 0.0F;
        for (int rank = 0; rank < request.top_k; ++rank) {
          const std::size_t route = static_cast<std::size_t>(token) * request.top_k + rank;
          const int expert = request.ids[route];
          float projection = 0.0F;
          for (int inner = 0; inner < hidden_; ++inner) {
            projection +=
                host_x_[static_cast<std::size_t>(token) * hidden_ + inner] *
                host_weights_[(static_cast<std::size_t>(expert) * hidden_ + inner) * output_ +
                              column];
          }
          combined += request.route_weights[route] * projection;
        }
        expected[static_cast<std::size_t>(token) * output_ + column] = combined;
      }
    }
    return expected;
  }

  PostRouteCacheRequest make_cache_request(int tokens, int top_k, std::uint64_t seed) const {
    PostRouteCacheRequest request;
    request.tokens = tokens;
    request.top_k = top_k;
    request.route_pairs = checked_int_product(tokens, top_k, "cache request R=T*top_k");
    request.ids = make_route_ids(tokens, top_k, experts_, distribution_, zipf_s_, seed);
    request.route_weights =
        make_random_floats(static_cast<std::size_t>(request.route_pairs), seed + 1, 0.01F, 1.0F);
    for (int token = 0; token < tokens; ++token) {
      float sum = 0.0F;
      for (int rank = 0; rank < top_k; ++rank) {
        sum += request.route_weights[static_cast<std::size_t>(token) * top_k + rank];
      }
      for (int rank = 0; rank < top_k; ++rank) {
        request.route_weights[static_cast<std::size_t>(token) * top_k + rank] /= sum;
      }
    }
    request.counts = counts_from_ids(request.ids, experts_);
    request.offsets = offsets_from_counts(request.counts);
    request.active_experts = static_cast<int>(std::count_if(
        request.counts.begin(), request.counts.end(), [](int count) { return count > 0; }));
    request.expected = build_reference_for(request);
    return request;
  }

  void apply_cache_request(std::size_t index) {
    const PostRouteCacheRequest& request = cache_requests_.at(index);
    tokens_ = request.tokens;
    top_k_ = request.top_k;
    route_pairs_ = request.route_pairs;
    active_experts_ = request.active_experts;
    host_ids_ = request.ids;
    host_route_weights_ = request.route_weights;
    host_counts_ = request.counts;
    host_offsets_ = request.offsets;
    expected_ = request.expected;
  }

  void copy_active_request_to_device(cudaStream_t stream) {
    cuda_check(
        cudaMemcpyAsync(ids_.data(), host_ids_.data(), host_ids_.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream),
        "copy post-route cache ids");
    cuda_check(
        cudaMemcpyAsync(route_weights_.data(), host_route_weights_.data(),
                        host_route_weights_.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
        "copy post-route cache route weights");
  }

  PostRouteGraphCacheKey current_cache_key() const {
    return {variant_name_, tokens_, experts_, top_k_,
            hidden_,       output_, "fp32",   "shared_rank_gather_unpermute"};
  }

  void destroy_graph_cache() noexcept {
    for (auto& entry : graph_cache_) {
      if (entry.exec != nullptr) cudaGraphExecDestroy(entry.exec);
      if (entry.graph != nullptr) cudaGraphDestroy(entry.graph);
      entry.exec = nullptr;
      entry.graph = nullptr;
    }
    graph_cache_.clear();
  }

  cudaGraphExec_t get_or_create_cached_graph(cudaStream_t stream) {
    const PostRouteGraphCacheKey key = current_cache_key();
    ++graph_cache_clock_;
    for (auto& entry : graph_cache_) {
      if (entry.key == key) {
        entry.last_use = graph_cache_clock_;
        ++graph_cache_hits_;
        graph_cached_kernel_nodes_ = entry.kernel_nodes;
        return entry.exec;
      }
    }

    ++graph_cache_misses_;
    const auto setup_begin = std::chrono::steady_clock::now();
    PostRouteGraphCacheEntry entry;
    entry.key = key;
    entry.last_use = graph_cache_clock_;
    capture_graph(stream, &entry.graph);
    cuda_check(cudaGraphInstantiate(&entry.exec, entry.graph, 0),
               "instantiate cached post-route graph");
    cuda_check(cudaGraphUpload(entry.exec, stream), "upload cached post-route graph");
    std::vector<cudaKernelNodeParams> params;
    collect_kernel_nodes(entry.graph, nullptr, &params);
    entry.kernel_nodes = params.size();
    graph_cached_kernel_nodes_ = entry.kernel_nodes;
    cudaGraphExec_t result_exec = entry.exec;
    graph_cache_miss_setup_us_total_ +=
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - setup_begin)
            .count();

    if (graph_cache_.size() == kGraphCacheCapacity) {
      auto lru = std::min_element(
          graph_cache_.begin(), graph_cache_.end(),
          [](const auto& left, const auto& right) { return left.last_use < right.last_use; });
      if (lru->exec != nullptr) cudaGraphExecDestroy(lru->exec);
      if (lru->graph != nullptr) cudaGraphDestroy(lru->graph);
      *lru = std::move(entry);
    } else {
      graph_cache_.push_back(std::move(entry));
    }
    return result_exec;
  }

  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T",
                                           "E",
                                           "top_k",
                                           "K",
                                           "N",
                                           "distribution",
                                           "zipf_s",
                                           "graph_cache_tokens",
                                           "graph_cache_topks"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) {
        throw std::invalid_argument("unknown chain_from_route_ids param: " + name);
      }
    }
  }

  static constexpr std::size_t kWorkspaceLimit = 64ULL * 1024ULL * 1024ULL;
  static constexpr std::size_t kGraphCacheCapacity = 16;
  std::string variant_name_;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  int tokens_ = 0, experts_ = 0, top_k_ = 0, hidden_ = 0, output_ = 0;
  int route_pairs_ = 0, active_experts_ = 0;
  int max_tokens_capacity_ = 0, max_route_pairs_capacity_ = 0;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<float> host_x_, host_weights_, host_route_weights_, expected_;
  std::vector<std::int32_t> host_ids_, host_counts_, host_offsets_;
  std::vector<PostRouteCacheRequest> cache_requests_;
  std::size_t cache_request_cursor_ = 0;
  std::string graph_cache_tokens_text_, graph_cache_topks_text_;
  DeviceBuffer<float> x_, x_rotating_, expert_weights_, route_weights_, x_permuted_, y_permuted_;
  DeviceBuffer<float> y_, y_rotating_;
  DeviceBuffer<std::int32_t> ids_, counts_, offsets_, cursors_, route_pos_, sorted_route_;
  std::size_t descriptor_workspace_bytes_ = 0;
  DeviceBuffer<std::uint8_t> descriptor_workspace_;
  cudaGraph_t graph_ = nullptr;
  cudaGraph_t graph_rotating_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaStream_t graph_update_stream_ = nullptr;
  std::vector<cudaGraphNode_t> graph_kernel_nodes_;
  std::vector<cudaKernelNodeParams> graph_kernel_params_primary_, graph_kernel_params_rotating_;
  std::size_t graph_memory_delta_bytes_ = 0;
  double graph_setup_us_ = 0.0;
  std::uint64_t graph_cache_hits_ = 0, graph_cache_misses_ = 0;
  bool graph_use_rotating_buffers_ = false;
  std::vector<PostRouteGraphCacheEntry> graph_cache_;
  std::uint64_t graph_cache_clock_ = 0;
  std::size_t graph_cached_kernel_nodes_ = 0;
  std::size_t graph_memory_baseline_free_bytes_ = 0;
  double graph_cache_miss_setup_us_total_ = 0.0;
};

}  // namespace

AdapterPtr make_postroute_chain_adapter(const std::string& variant_name) {
  return std::make_unique<PostRouteChainAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
