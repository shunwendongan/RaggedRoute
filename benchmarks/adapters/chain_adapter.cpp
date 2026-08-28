#include <algorithm>
#include <chrono>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "dense_gemm/cuda_candidate/optimized_internal.h"
#include "grouped_gemm/cuda_candidate/grouped_optimized_internal.h"
#include "permute/cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/benchmark/adapter_utils.h"
#include "raggedroute/benchmark/library_baselines.h"
#include "raggedroute/benchmark/registry.h"
#include "scan/cuda_candidate/optimized_internal.h"
#include "topk_gate/cuda_candidate/optimized_internal.h"
#include "unpermute/cuda_candidate/optimized_internal.h"

namespace raggedroute::benchmark {
namespace {

bool is_postlogit_graph_variant(const std::string& variant) {
  return variant == "cuda_postlogit_graph_fixed_v1" ||
         variant == "cuda_postlogit_graph_param_update_v2" ||
         variant == "cuda_postlogit_graph_exec_update_v3" ||
         variant == "cuda_postlogit_graph_cache_v4";
}

bool uses_postlogit_graph_pointer_rotation(const std::string& variant) {
  return variant == "cuda_postlogit_graph_param_update_v2" ||
         variant == "cuda_postlogit_graph_exec_update_v3";
}

struct PostLogitCacheRequest {
  int tokens = 0;
  int route_pairs = 0;
  int active_experts = 0;
  std::vector<float> logits;
  std::vector<float> route_weights;
  std::vector<float> expected_output;
  std::vector<std::int32_t> ids;
  std::vector<std::int32_t> counts;
  std::vector<std::int32_t> offsets;
};

struct PostLogitGraphCacheKey {
  std::string variant;
  int tokens = 0;
  int experts = 0;
  int top_k = 2;
  int hidden = 0;
  int output = 0;
  std::string dtype;
  std::string topology_class;

  bool operator==(const PostLogitGraphCacheKey& other) const {
    return variant == other.variant && tokens == other.tokens && experts == other.experts &&
           top_k == other.top_k && hidden == other.hidden && output == other.output &&
           dtype == other.dtype && topology_class == other.topology_class;
  }
};

struct PostLogitGraphCacheEntry {
  PostLogitGraphCacheKey key;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  std::uint64_t last_use = 0;
  std::size_t kernel_nodes = 0;
};

class ChainAdapter final : public BenchmarkAdapter {
 public:
  ChainAdapter(bool include_router_projection, const std::string& variant_name)
      : include_router_projection_(include_router_projection), variant_name_(variant_name) {}
  ~ChainAdapter() override {
    destroy_graph_cache();
    if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
    if (graph_rotating_ != nullptr) cudaGraphDestroy(graph_rotating_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    if (graph_update_stream_ != nullptr) cudaStreamDestroy(graph_update_stream_);
#if RAGGEDROUTE_HAS_CUBLAS
    library_baseline::destroy_dense_cublaslt_plan(dense_cublaslt_plan_);
#endif
#if RAGGEDROUTE_HAS_CUTLASS
    library_baseline::destroy_grouped_cutlass_plan(grouped_cutlass_plan_);
#endif
  }

  std::string operator_name() const override {
    return include_router_projection_ ? "chain_from_tokens" : "chain_from_logits";
  }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    const std::string prefix = include_router_projection_ ? "Seven-operator token-to-output chain"
                                                          : "Six-operator logits-to-output chain";
    if (variant_name_ == "cuda_permute_candidate") {
      return prefix + " with the selected optimized Permute";
    }
    if (variant_name_ == "cuda_grouped_sm86_fp32_v1") {
      return prefix + " with benchmark-only SM86 grouped GEMM candidate";
    }
    if (variant_name_ == "cuda_unpermute_candidate") {
      return prefix + " with optimized Unpermute";
    }
    if (variant_name_ == "cuda_fused_histogram_scan") {
      return prefix + " with subwarp-finalized fused Histogram-Scan";
    }
    if (variant_name_ == "cuda_all_candidates_chain") {
      return prefix + " with every selected CUDA candidate";
    }
    if (variant_name_ == "cuda_postlogit_retained_main") {
      return prefix + " with the retained main post-logit candidates and Grouped GEMM v1";
    }
    if (variant_name_ == "cuda_postlogit_integrated_latest") {
      return prefix + " with the retained main post-logit candidates and Grouped GEMM v2";
    }
    if (variant_name_ == "cuda_postlogit_research_v3") {
      return prefix + " with Permute v3 and Grouped GEMM v3 research candidates";
    }
    if (variant_name_ == "library_all_baselines_chain") {
      return prefix + " with repository library baselines (diagnostic contract)";
    }
    return prefix + " baseline";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kChainSteady || level == MeasurementLevel::kHostCall;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 64);
    experts_ = get_int_option(options, "E", 8, 2);
    const int top_k = get_int_option(options, "top_k", 2, 1);
    if (top_k != 2) {
      throw std::invalid_argument("current chain adapters require top_k=2");
    }
    if (experts_ > 64) throw std::invalid_argument("current chain adapters support E<=64");
    hidden_ = get_int_option(options, "K", 32);
    output_ = get_int_option(options, "N", 32);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    route_trace_path_ = get_option(options, "route_trace_path", "");
    route_trace_frame_ = get_int_option(options, "route_trace_frame", 0, 0);
    if (include_router_projection_ && !route_trace_path_.empty()) {
      throw std::invalid_argument("route traces are supported only by chain_from_logits");
    }
    graph_cache_tokens_text_ = get_option(options, "graph_cache_tokens", std::to_string(tokens_));
    std::vector<int> request_tokens = {tokens_};
    if (variant_name_ == "cuda_postlogit_graph_cache_v4") {
      if (include_router_projection_ || !route_trace_path_.empty()) {
        throw std::invalid_argument(
            "post-logit graph cache shape cycle requires synthetic logits input");
      }
      request_tokens = parse_positive_int_list(graph_cache_tokens_text_, "graph_cache_tokens");
      tokens_ = request_tokens.front();
    }
    max_tokens_capacity_ = 0;
    for (const int request_token_count : request_tokens) {
      (void)checked_int_product(request_token_count, output_, "cache request T*N");
      max_tokens_capacity_ = std::max(max_tokens_capacity_, request_token_count);
    }
    max_route_pairs_capacity_ = checked_int_product(max_tokens_capacity_, 2, "max cache R=T*2");
    route_pairs_ = checked_int_product(tokens_, 2, "R=T*2");

    x_host_ = make_random_floats(static_cast<std::size_t>(max_tokens_capacity_) * hidden_, seed);
    expert_weights_host_ =
        make_random_floats(static_cast<std::size_t>(experts_) * hidden_ * output_, seed + 1);
    if (variant_name_ == "cuda_postlogit_graph_cache_v4") {
      std::set<int> unique_tokens;
      std::uint64_t request_seed = seed + 2;
      for (const int request_token_count : request_tokens) {
        if (!unique_tokens.insert(request_token_count).second) continue;
        cache_requests_.push_back(make_cache_request(request_token_count, request_seed));
        request_seed += 17;
      }
      if (cache_requests_.empty()) throw std::invalid_argument("post-logit graph cache is empty");
      apply_cache_request(0);
      allocate_and_copy(stream);
      initialize_graph(stream);
      return;
    }
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
      std::vector<std::int32_t> desired_ids;
      if (route_trace_path_.empty()) {
        desired_ids = make_route_ids(tokens_, 2, experts_, distribution_, zipf_s_, seed + 2);
      } else {
        const RouteTraceFrame trace = load_route_trace_frame(route_trace_path_, route_trace_frame_);
        if (trace.tokens != tokens_ || trace.experts != experts_ || trace.top_k != 2) {
          throw std::invalid_argument("route trace shape does not match chain params");
        }
        desired_ids = trace.expert_ids;
        route_trace_id_ = trace.trace_id;
        route_trace_source_kind_ = trace.source_kind;
        route_trace_frame_id_ = trace.frame_id;
        route_trace_frame_count_ = trace.frame_count;
        distribution_ = "route_trace";
        zipf_s_ = 0.0;
      }
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
    if (variant_name_ == "library_all_baselines_chain") setup_library_chain(stream);
    if (is_postlogit_graph_variant(variant_name_)) initialize_graph(stream);
  }

  void prepare_sample(MeasurementLevel, cudaStream_t stream) override {
    if (variant_name_ != "cuda_postlogit_graph_cache_v4" || cache_requests_.size() <= 1) return;
    cache_request_cursor_ = (cache_request_cursor_ + 1) % cache_requests_.size();
    apply_cache_request(cache_request_cursor_);
    copy_active_logits_to_device(stream);
  }

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level != MeasurementLevel::kChainSteady && level != MeasurementLevel::kHostCall) {
      throw std::invalid_argument("chain adapter supports L3 and graph L4");
    }
    if (is_postlogit_graph_variant(variant_name_)) {
      launch_graph(stream);
      return;
    }
    enqueue_direct(stream);
  }

  void enqueue_direct(cudaStream_t stream) {
    const RuntimeContext context = make_runtime_context(stream, architecture_);
    const bool all_candidates = variant_name_ == "cuda_all_candidates_chain";
    const bool all_libraries = variant_name_ == "library_all_baselines_chain";
    const bool retained_postlogit = variant_name_ == "cuda_postlogit_retained_main";
    const bool integrated_postlogit = variant_name_ == "cuda_postlogit_integrated_latest";
    const bool research_v3 = variant_name_ == "cuda_postlogit_research_v3";
    const bool postlogit_graph = is_postlogit_graph_variant(variant_name_);
    const bool unified_postlogit =
        retained_postlogit || integrated_postlogit || research_v3 || postlogit_graph;
    if (include_router_projection_ && all_libraries) {
#if RAGGEDROUTE_HAS_CUBLAS
      library_baseline::launch_dense_cublaslt(
          dense_cublaslt_plan_, x_.data(), router_weights_.data(), logits_.data(),
          dense_library_workspace_.data(), dense_library_workspace_bytes_, stream);
#else
      throw std::runtime_error("library chain requires cuBLASLt support");
#endif
    } else if (include_router_projection_) {
      DenseGemmArgs args;
      args.a.data = x_.data();
      args.b.data = router_weights_.data();
      args.c.data = logits_.data();
      args.m = tokens_;
      args.n = experts_;
      args.k = hidden_;
      if (all_candidates) {
        args.kernel = {KernelFamily::kCudaOptimized,
                       ops::kDenseGemmRegisterTiledV3_64x32AsyncImplementation};
      }
      operator_check(dense_gemm(args, context), "chain dense_gemm operator");
    }
    if (all_libraries) {
#if RAGGEDROUTE_HAS_CCCL
      cuda_check(
          library_baseline::launch_cub_block_radix_top2(
              active_logits_data(), ids_.data(), route_weights_.data(), tokens_, experts_, stream),
          "chain CUB BlockRadixSort Top-2");
#else
      throw std::runtime_error("library chain requires CCCL support");
#endif
    } else {
      TopKGateArgs topk_args;
      topk_args.logits.data = active_logits_data();
      topk_args.expert_ids = ids_.data();
      topk_args.weights.data = route_weights_.data();
      topk_args.tokens = tokens_;
      topk_args.experts = experts_;
      if (all_candidates || unified_postlogit) {
        topk_args.kernel = {KernelFamily::kCudaOptimized,
                            ops::kTopKGateLocalPairTwoReduceV4Implementation};
      }
      operator_check(topk_gate(topk_args, context), "chain topk_gate operator");
    }

    if (all_libraries) {
#if RAGGEDROUTE_HAS_CCCL
      cuda_check(
          library_baseline::launch_cub_histogram(
              ids_.data(), counts_.data(), static_cast<std::size_t>(route_pairs_), experts_,
              histogram_library_workspace_.data(), histogram_library_workspace_bytes_, stream),
          "chain CUB DeviceHistogram");
      cuda_check(library_baseline::launch_cub_block_scan(counts_.data(), offsets_.data(), experts_,
                                                         stream),
                 "chain CUB BlockScan");
#else
      throw std::runtime_error("library chain requires CCCL support");
#endif
    } else if (variant_name_ == "cuda_fused_histogram_scan" || all_candidates ||
               unified_postlogit) {
      HistogramExclusiveScanArgs fused_args;
      fused_args.expert_ids = ids_.data();
      fused_args.counts = counts_.data();
      fused_args.offsets = offsets_.data();
      fused_args.route_pairs = route_pairs_;
      fused_args.experts = experts_;
      fused_args.kernel = {KernelFamily::kCudaOptimized,
                           ops::kHistogramExclusiveScanFusedSubwarpImplementation};
      operator_check(histogram_exclusive_scan(fused_args, context),
                     "chain fused histogram_exclusive_scan operator");
    } else {
      HistogramArgs histogram_args;
      histogram_args.expert_ids = ids_.data();
      histogram_args.counts = counts_.data();
      histogram_args.route_pairs = route_pairs_;
      histogram_args.experts = experts_;
      operator_check(histogram(histogram_args, context), "chain histogram operator");

      ExclusiveScanArgs scan_args;
      scan_args.counts = counts_.data();
      scan_args.offsets = offsets_.data();
      scan_args.experts = experts_;
      operator_check(exclusive_scan(scan_args, context), "chain exclusive_scan operator");
    }

    TokenPermuteArgs permute_args;
    permute_args.x.data = x_.data();
    permute_args.expert_ids = ids_.data();
    permute_args.offsets = offsets_.data();
    permute_args.x_permuted.data = x_permuted_.data();
    permute_args.route_pos = route_pos_.data();
    permute_args.tokens = tokens_;
    permute_args.experts = experts_;
    permute_args.top_k = 2;
    permute_args.hidden = hidden_;
    if (variant_name_ == "cuda_permute_candidate" || all_candidates || unified_postlogit) {
      permute_args.kernel = {KernelFamily::kCudaOptimized,
                             research_v3 ? ops::kTokenPermuteShapeDispatchedV3Implementation
                                         : ops::kTokenPermuteCandidateImplementation};
    }
    if (all_libraries) {
#if RAGGEDROUTE_HAS_CCCL
      cuda_check(library_baseline::launch_vllm_moe_permute(
                     x_.data(), ids_.data(), x_permuted_.data(), route_pos_.data(), nullptr,
                     permute_library_workspace_.data(), permute_library_workspace_bytes_, tokens_,
                     experts_, 2, hidden_, stream),
                 "chain vLLM moe_permute");
#else
      throw std::runtime_error("library chain requires CCCL support");
#endif
    } else {
      operator_check(
          token_permute(permute_args, make_runtime_context(stream, architecture_, cursors_.data(),
                                                           cursors_.bytes())),
          "chain token_permute operator");
    }

    // Passing R is a truthful worst-case launch bound. No input-dependent host
    // max-M computation is hidden outside the L3 interval.
    GroupedGemmArgs grouped_args;
    grouped_args.x_permuted.data = x_permuted_.data();
    grouped_args.expert_weights.data = expert_weights_.data();
    grouped_args.offsets = offsets_.data();
    grouped_args.y_permuted.data = y_permuted_.data();
    grouped_args.experts = experts_;
    grouped_args.hidden = hidden_;
    grouped_args.output = output_;
    grouped_args.max_expert_tokens = route_pairs_;
    if (all_libraries) {
#if RAGGEDROUTE_HAS_CUTLASS
      library_baseline::launch_grouped_cutlass(grouped_cutlass_plan_,
                                               grouped_library_workspace_.data(),
                                               grouped_library_workspace_bytes_, stream);
#else
      throw std::runtime_error("library chain requires CUTLASS support");
#endif
    } else if (variant_name_ == "cuda_grouped_sm86_fp32_v1" || all_candidates ||
               unified_postlogit) {
      const std::uint32_t grouped_implementation =
          research_v3                               ? ops::kGroupedGemmSm86Fp32V3Implementation
          : integrated_postlogit || postlogit_graph ? ops::kGroupedGemmSm86Fp32V2Implementation
                                                    : ops::kGroupedGemmSm86Fp32V1Implementation;
      cuda_check(ops::launch_grouped_gemm_optimized(
                     static_cast<const float*>(grouped_args.x_permuted.data),
                     static_cast<const float*>(grouped_args.expert_weights.data),
                     grouped_args.offsets, static_cast<float*>(grouped_args.y_permuted.data),
                     grouped_args.experts, grouped_args.hidden, grouped_args.output,
                     grouped_args.max_expert_tokens, grouped_implementation, stream),
                 "chain benchmark-only grouped_gemm candidate");
    } else {
      operator_check(grouped_gemm(grouped_args, context), "chain grouped_gemm operator");
    }

    UnpermuteArgs unpermute_args;
    unpermute_args.y_permuted.data = y_permuted_.data();
    unpermute_args.route_pos = route_pos_.data();
    unpermute_args.route_weights.data = route_weights_.data();
    unpermute_args.y.data = active_y_data();
    unpermute_args.tokens = tokens_;
    unpermute_args.top_k = 2;
    unpermute_args.output = output_;
    if (all_libraries) {
#if RAGGEDROUTE_HAS_VLLM_UNPERMUTE
      cuda_check(library_baseline::launch_vllm_finalize_routing(
                     y_permuted_.data(), active_y_data(), route_weights_.data(), route_pos_.data(),
                     tokens_, 2, output_, stream),
                 "chain vLLM finalize routing");
      return;
#else
      throw std::runtime_error("library chain requires vLLM unpermute support");
#endif
    }
    if (variant_name_ == "cuda_unpermute_candidate" || all_candidates || unified_postlogit) {
      cuda_check(
          ops::launch_unpermute_optimized(
              static_cast<const float*>(unpermute_args.y_permuted.data), unpermute_args.route_pos,
              static_cast<const float*>(unpermute_args.route_weights.data),
              static_cast<float*>(unpermute_args.y.data), unpermute_args.tokens,
              unpermute_args.top_k, unpermute_args.output, stream),
          "chain research unpermute candidate");
      return;
    }
    operator_check(unpermute(unpermute_args, context), "chain unpermute operator");
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto ids = copy_ids_prefix_to_host(stream);
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
    const auto route_weights = copy_route_weights_prefix_to_host(stream);
    const double weight_tolerance =
        variant_name_ == "library_all_baselines_chain" ? 2.0e-5 : 1.0e-6;
    auto weight_result =
        compare_floats(route_weights, route_weights_expected_, weight_tolerance, weight_tolerance);
    if (!weight_result.ok) {
      weight_result.message = "chain route weights: " + weight_result.message;
      return weight_result;
    }
    const auto actual_output = copy_active_output_to_host(stream);
    auto output_result = compare_floats(actual_output, expected_output_, 1.0e-4, 4.0e-5 * hidden_);
    if (!output_result.ok) output_result.message = "chain final output: " + output_result.message;
    return output_result;
  }

  FieldMap case_config() const override {
    FieldMap config = {
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
    const bool retained_postlogit = variant_name_ == "cuda_postlogit_retained_main";
    const bool integrated_postlogit = variant_name_ == "cuda_postlogit_integrated_latest";
    const bool research_v3 = variant_name_ == "cuda_postlogit_research_v3";
    const bool postlogit_graph = is_postlogit_graph_variant(variant_name_);
    const bool unified_postlogit =
        retained_postlogit || integrated_postlogit || research_v3 || postlogit_graph;
    FieldMap config = {
        {"components",
         include_router_projection_
             ? std::string("dense_gemm,topk_gate,histogram,exclusive_scan,token_permute,"
                           "grouped_gemm,unpermute")
             : std::string(
                   "topk_gate,histogram,exclusive_scan,token_permute,grouped_gemm,unpermute")},
        {"component_variant",
         research_v3 ? std::string("postlogit_research_permute_v3_grouped_v3")
         : integrated_postlogit || postlogit_graph
             ? std::string("postlogit_integrated_latest_grouped_v2")
         : retained_postlogit ? std::string("postlogit_retained_main_grouped_v1")
         : variant_name_ == "cuda_all_candidates_chain"
             ? std::string("all_selected_cuda_candidates")
         : variant_name_ == "library_all_baselines_chain"
             ? std::string("all_repository_library_baselines_diagnostic")
         : variant_name_ == "cuda_permute_candidate"
             ? std::string("cuda_naive_except_token_permute_candidate")
         : variant_name_ == "cuda_grouped_sm86_fp32_v1"
             ? std::string("cuda_naive_except_benchmark_only_grouped_sm86_fp32_v1")
         : variant_name_ == "cuda_unpermute_candidate" ? std::string("mixed")
         : variant_name_ == "cuda_fused_histogram_scan"
             ? std::string("cuda_naive_except_fused_histogram_scan")
             : std::string("cuda_naive")},
        {"permute_variant",
         research_v3         ? std::string("cuda_candidate_v3_from_offsets")
         : unified_postlogit ? std::string("cuda_candidate_v2_from_offsets")
         : variant_name_ == "cuda_all_candidates_chain"
             ? std::string("cuda_candidate_from_ids_equivalent")
         : variant_name_ == "library_all_baselines_chain" ? std::string("vllm_moe_permute")
         : variant_name_ == "cuda_permute_candidate"      ? std::string("cuda_token_owned_top2")
                                                          : std::string("cuda_naive")},
        {"runtime_status", variant_name_ == "library_all_baselines_chain"
                               ? std::string("diagnostic_not_strictly_comparable")
                           : variant_name_ == "cuda_grouped_sm86_fp32_v1" || unified_postlogit ||
                                   variant_name_ == "cuda_all_candidates_chain"
                               ? std::string("benchmark_only_not_promoted")
                               : std::string("public_runtime")},
        {"unpermute_variant",
         unified_postlogit                                ? std::string("cuda_warp_token_vec4")
         : variant_name_ == "cuda_all_candidates_chain"   ? std::string("cuda_warp_token_vec4")
         : variant_name_ == "library_all_baselines_chain" ? std::string("vllm_finalize_routing")
         : variant_name_ == "cuda_unpermute_candidate"    ? std::string("cuda_warp_token_vec4")
                                                          : std::string("cuda_naive")},
        {"grouped_variant",
         research_v3                                    ? std::string("cuda_grouped_sm86_fp32_v3")
         : integrated_postlogit || postlogit_graph      ? std::string("cuda_grouped_sm86_fp32_v2")
         : retained_postlogit                           ? std::string("cuda_grouped_sm86_fp32_v1")
         : variant_name_ == "cuda_all_candidates_chain" ? std::string("cuda_grouped_sm86_fp32_v1")
         : variant_name_ == "library_all_baselines_chain"
             ? std::string("cutlass_grouped_fixed_problem_metadata")
             : std::string("cuda_naive")},
        {"grouped_max_m_policy", variant_name_ == "library_all_baselines_chain"
                                     ? std::string("fixed_host_offsets_from_deterministic_oracle")
                                     : std::string("worst_case_R")},
        {"histogram_scan_variant",
         unified_postlogit ? std::string("cuda_fused_histogram_scan_v2")
         : variant_name_ == "cuda_all_candidates_chain" ? std::string("cuda_fused_histogram_scan")
         : variant_name_ == "library_all_baselines_chain"
             ? std::string("cub_device_histogram_plus_cub_block_scan")
         : variant_name_ == "cuda_fused_histogram_scan" ? std::string("cuda_fused_histogram_scan")
                                                        : std::string("separate")},
        {"dense_variant", variant_name_ == "cuda_all_candidates_chain"
                              ? std::string("cuda_register_tiled_v3_64x32_async")
                          : variant_name_ == "library_all_baselines_chain"
                              ? std::string("cublaslt")
                              : std::string("cuda_naive")},
        {"topk_variant", unified_postlogit ? std::string("cuda_local_pair_two_reduce_top2_v4")
                         : variant_name_ == "cuda_all_candidates_chain"
                             ? std::string("cuda_local_pair_two_reduce_top2_v4")
                         : variant_name_ == "library_all_baselines_chain"
                             ? std::string("cub_block_radix_top2_benchmark_only")
                             : std::string("cuda_naive")},
        {"strict_comparison_eligible", variant_name_ != "library_all_baselines_chain"},
        {"materialize_sorted_route", false}};
    if (postlogit_graph) {
      config["graph_mode"] = variant_name_ == "cuda_postlogit_graph_fixed_v1"
                                 ? std::string("fixed_capture_upload_replay")
                             : variant_name_ == "cuda_postlogit_graph_param_update_v2"
                                 ? std::string("kernel_node_set_params_double_buffer_rotation")
                             : variant_name_ == "cuda_postlogit_graph_exec_update_v3"
                                 ? std::string("recapture_cudaGraphExecUpdate")
                                 : std::string("lru16_shape_key_capture_on_miss");
      config["graph_kernel_nodes"] = static_cast<std::int64_t>(
          variant_name_ == "cuda_postlogit_graph_cache_v4" ? graph_cached_kernel_nodes_
                                                           : graph_kernel_nodes_.size());
      config["graph_cache_capacity"] = static_cast<std::int64_t>(16);
      config["graph_cache_entries"] = static_cast<std::int64_t>(graph_cache_.size());
      config["graph_cache_hits"] = static_cast<std::int64_t>(graph_cache_hits_);
      config["graph_cache_misses"] = static_cast<std::int64_t>(graph_cache_misses_);
      std::size_t graph_memory_delta = graph_memory_delta_bytes_;
      if (variant_name_ == "cuda_postlogit_graph_cache_v4") {
        std::size_t free_after = 0, total_after = 0;
        cuda_check(cudaMemGetInfo(&free_after, &total_after),
                   "cudaMemGetInfo while reporting post-logit graph cache");
        graph_memory_delta = graph_memory_baseline_free_bytes_ > free_after
                                 ? graph_memory_baseline_free_bytes_ - free_after
                                 : 0;
      }
      config["driver_graph_memory_delta_bytes"] = static_cast<std::int64_t>(graph_memory_delta);
      config["graph_capture_instantiate_upload_us"] =
          variant_name_ == "cuda_postlogit_graph_cache_v4" ? graph_cache_miss_setup_us_total_
                                                           : graph_setup_us_;
      config["graph_cache_miss_setup_us_total"] = graph_cache_miss_setup_us_total_;
      config["graph_cache_key_fields"] = std::string("variant,T,E,top_k,K,N,dtype,topology_class");
      config["graph_setup_excluded"] = variant_name_ != "cuda_postlogit_graph_cache_v4";
      config["graph_pointer_rotation"] = uses_postlogit_graph_pointer_rotation(variant_name_);
      config["graph_cache_request_keys"] = static_cast<std::int64_t>(cache_requests_.size());
      config["graph_cache_tokens"] = graph_cache_tokens_text_;
      config["mixed_shape_cache_trace"] =
          variant_name_ == "cuda_postlogit_graph_cache_v4" && cache_requests_.size() > 1;
      config["graph_working_set_count"] = static_cast<std::int64_t>(
          variant_name_ == "cuda_postlogit_graph_cache_v4"       ? cache_requests_.size()
          : uses_postlogit_graph_pointer_rotation(variant_name_) ? 2
                                                                 : 1);
    }
    return config;
  }

  WorkEstimate work_estimate(MeasurementLevel) const override {
    WorkEstimate work;
    double average_tokens = tokens_;
    double average_route_pairs = route_pairs_;
    double average_active_experts = active_experts_;
    if (variant_name_ == "cuda_postlogit_graph_cache_v4" && !cache_requests_.empty()) {
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
    if (include_router_projection_) {
      work.flops += 2.0 * average_tokens * hidden_ * experts_;
      work.logical_bytes +=
          sizeof(float) * (average_tokens * hidden_ + static_cast<double>(hidden_) * experts_ +
                           average_tokens * experts_);
    }
    work.flops += 2.0 * average_route_pairs * hidden_ * output_;
    work.flops += 3.0 * average_tokens * output_;
    work.logical_bytes += sizeof(float) * average_tokens * experts_;
    work.logical_bytes += 2.0 * sizeof(std::int32_t) * average_route_pairs;
    work.logical_bytes += 3.0 * sizeof(std::int32_t) * experts_;
    work.logical_bytes += 2.0 * sizeof(std::int32_t) * experts_;
    work.logical_bytes += 2.0 * sizeof(float) * average_route_pairs * hidden_;
    work.logical_bytes +=
        sizeof(float) * (average_route_pairs * hidden_ + average_route_pairs * output_ +
                         average_active_experts * hidden_ * output_);
    work.logical_bytes +=
        sizeof(float) * (average_route_pairs * output_ + average_tokens * output_);
    const bool fused_histogram_scan = variant_name_ == "cuda_fused_histogram_scan" ||
                                      variant_name_ == "cuda_postlogit_retained_main" ||
                                      variant_name_ == "cuda_postlogit_integrated_latest" ||
                                      variant_name_ == "cuda_postlogit_research_v3" ||
                                      is_postlogit_graph_variant(variant_name_) ||
                                      variant_name_ == "cuda_all_candidates_chain";
    work.operator_metrics["kernel_launches"] =
        variant_name_ == "cuda_postlogit_graph_cache_v4" && graph_cached_kernel_nodes_ != 0
            ? static_cast<std::int64_t>(graph_cached_kernel_nodes_)
        : is_postlogit_graph_variant(variant_name_) && !graph_kernel_nodes_.empty()
            ? static_cast<std::int64_t>(graph_kernel_nodes_.size())
            : static_cast<std::int64_t>((include_router_projection_ ? 9 : 8) -
                                        (fused_histogram_scan ? 1 : 0));
    work.operator_metrics["host_graph_launches"] =
        static_cast<std::int64_t>(is_postlogit_graph_variant(variant_name_) ? 1 : 0);
    work.operator_metrics["counts_reset_bytes"] =
        static_cast<std::int64_t>(sizeof(std::int32_t) * experts_);
    work.operator_metrics["cursor_reset_bytes"] = static_cast<std::int64_t>(cursors_.bytes());
    return work;
  }

  std::size_t workspace_bytes() const override {
    return cursors_.bytes() + dense_library_workspace_bytes_ + histogram_library_workspace_bytes_ +
           permute_library_workspace_bytes_ + grouped_library_workspace_bytes_;
  }

  std::vector<std::string> excluded_steps(MeasurementLevel) const override {
    std::vector<std::string> steps = {"input_generation", "cpu_reference", "h2d_copy",
                                      "workspace_allocation"};
    if (variant_name_ == "cuda_postlogit_graph_fixed_v1" ||
        variant_name_ == "cuda_postlogit_graph_param_update_v2") {
      steps.insert(steps.end(), {"graph_capture", "graph_instantiate", "graph_upload"});
    }
    return steps;
  }

 private:
  void capture_graph(cudaStream_t stream, cudaGraph_t* graph) {
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "begin post-logit graph capture");
    enqueue_direct(stream);
    cuda_check(cudaStreamEndCapture(stream, graph), "end post-logit graph capture");
  }

  static void collect_kernel_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t>* nodes_out,
                                   std::vector<cudaKernelNodeParams>* params_out) {
    std::size_t node_count = 0;
    cuda_check(cudaGraphGetNodes(graph, nullptr, &node_count), "query post-logit graph nodes");
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (node_count != 0) {
      cuda_check(cudaGraphGetNodes(graph, nodes.data(), &node_count), "get post-logit graph nodes");
    }
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type{};
      cuda_check(cudaGraphNodeGetType(node, &type), "get post-logit graph node type");
      if (type != cudaGraphNodeTypeKernel) continue;
      cudaKernelNodeParams params{};
      cuda_check(cudaGraphKernelNodeGetParams(node, &params), "get post-logit kernel node params");
      if (nodes_out != nullptr) nodes_out->push_back(node);
      params_out->push_back(params);
    }
  }

  void initialize_graph(cudaStream_t stream) {
    enqueue_direct(stream);
    cuda_check(cudaStreamSynchronize(stream), "post-logit graph priming sync");
    std::size_t free_before = 0, total_before = 0;
    cuda_check(cudaMemGetInfo(&free_before, &total_before), "cudaMemGetInfo before graph");
    graph_memory_baseline_free_bytes_ = free_before;
    if (variant_name_ == "cuda_postlogit_graph_cache_v4") {
      graph_cache_hits_ = 0;
      graph_cache_misses_ = 0;
      return;
    }
    const auto setup_begin = std::chrono::steady_clock::now();
    capture_graph(stream, &graph_);
    cuda_check(cudaGraphInstantiate(&graph_exec_, graph_, 0), "instantiate post-logit graph");
    cuda_check(cudaGraphUpload(graph_exec_, stream), "upload post-logit graph");
    cuda_check(cudaStreamSynchronize(stream), "post-logit graph upload sync");
    graph_setup_us_ =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - setup_begin)
            .count();
    std::size_t free_after = 0, total_after = 0;
    cuda_check(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after graph");
    graph_memory_delta_bytes_ = free_before > free_after ? free_before - free_after : 0;
    collect_kernel_nodes(graph_, &graph_kernel_nodes_, &graph_kernel_params_primary_);
    if (variant_name_ == "cuda_postlogit_graph_param_update_v2") {
      graph_use_rotating_buffers_ = true;
      capture_graph(stream, &graph_rotating_);
      collect_kernel_nodes(graph_rotating_, nullptr, &graph_kernel_params_rotating_);
      graph_use_rotating_buffers_ = false;
      if (graph_kernel_params_primary_.size() != graph_kernel_params_rotating_.size()) {
        throw std::runtime_error("post-logit pointer-update graph topology changed");
      }
      for (std::size_t index = 0; index < graph_kernel_params_primary_.size(); ++index) {
        if (graph_kernel_params_primary_[index].func != graph_kernel_params_rotating_[index].func) {
          throw std::runtime_error("post-logit pointer-update graph kernel order changed");
        }
      }
    }
    cuda_check(cudaStreamCreateWithFlags(&graph_update_stream_, cudaStreamNonBlocking),
               "create post-logit graph update stream");
    graph_cache_misses_ = 1;
  }

  void launch_graph(cudaStream_t stream) {
    if (variant_name_ == "cuda_postlogit_graph_cache_v4") {
      cudaGraphExec_t exec = get_or_create_cached_graph(stream);
      cuda_check(cudaGraphLaunch(exec, stream), "launch cached post-logit graph");
      return;
    }
    if (variant_name_ == "cuda_postlogit_graph_param_update_v2") {
      graph_use_rotating_buffers_ = !graph_use_rotating_buffers_;
      const auto& params = graph_use_rotating_buffers_ ? graph_kernel_params_rotating_
                                                       : graph_kernel_params_primary_;
      for (std::size_t index = 0; index < graph_kernel_nodes_.size(); ++index) {
        cuda_check(cudaGraphExecKernelNodeSetParams(graph_exec_, graph_kernel_nodes_[index],
                                                    &params[index]),
                   "set post-logit graph kernel params");
      }
    } else if (variant_name_ == "cuda_postlogit_graph_exec_update_v3") {
      graph_use_rotating_buffers_ = !graph_use_rotating_buffers_;
      cudaGraph_t updated = nullptr;
      capture_graph(graph_update_stream_, &updated);
      cudaGraphExecUpdateResultInfo result_info{};
      const cudaError_t update_error = cudaGraphExecUpdate(graph_exec_, updated, &result_info);
      cudaGraphDestroy(updated);
      cuda_check(update_error, "cudaGraphExecUpdate post-logit graph");
      if (result_info.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error("post-logit cudaGraphExecUpdate rejected compatible topology");
      }
    }
    cuda_check(cudaGraphLaunch(graph_exec_, stream), "launch post-logit graph");
  }

  const float* active_logits_data() const {
    return graph_use_rotating_buffers_ ? logits_rotating_.data() : logits_.data();
  }
  float* active_y_data() { return graph_use_rotating_buffers_ ? y_rotating_.data() : y_.data(); }

  template <typename T>
  std::vector<T> copy_prefix_to_host(const DeviceBuffer<T>& buffer, std::size_t count,
                                     cudaStream_t stream, const char* operation) const {
    std::vector<T> result(count);
    cuda_check(cudaMemcpyAsync(result.data(), buffer.data(), count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream),
               operation);
    cuda_check(cudaStreamSynchronize(stream), "sync chain prefix copy");
    return result;
  }

  std::vector<std::int32_t> copy_ids_prefix_to_host(cudaStream_t stream) const {
    return copy_prefix_to_host(ids_, static_cast<std::size_t>(route_pairs_), stream,
                               "copy chain ids prefix");
  }
  std::vector<float> copy_route_weights_prefix_to_host(cudaStream_t stream) const {
    return copy_prefix_to_host(route_weights_, static_cast<std::size_t>(route_pairs_), stream,
                               "copy chain route weights prefix");
  }
  std::vector<float> copy_active_output_to_host(cudaStream_t stream) const {
    const DeviceBuffer<float>& source = graph_use_rotating_buffers_ ? y_rotating_ : y_;
    return copy_prefix_to_host(source, static_cast<std::size_t>(tokens_) * output_, stream,
                               "copy chain output prefix");
  }

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

  std::vector<float> build_output_reference_for(const PostLogitCacheRequest& request) const {
    std::vector<float> expected(static_cast<std::size_t>(request.tokens) * output_, 0.0F);
    for (int token = 0; token < request.tokens; ++token) {
      for (int column = 0; column < output_; ++column) {
        float combined = 0.0F;
        for (int rank = 0; rank < 2; ++rank) {
          const auto route = static_cast<std::size_t>(token) * 2 + rank;
          const int expert = request.ids[route];
          float projection = 0.0F;
          for (int inner = 0; inner < hidden_; ++inner) {
            projection +=
                x_host_[static_cast<std::size_t>(token) * hidden_ + inner] *
                expert_weights_host_[(static_cast<std::size_t>(expert) * hidden_ + inner) *
                                         output_ +
                                     column];
          }
          combined += request.route_weights[route] * projection;
        }
        expected[static_cast<std::size_t>(token) * output_ + column] = combined;
      }
    }
    return expected;
  }

  PostLogitCacheRequest make_cache_request(int tokens, std::uint64_t seed) const {
    PostLogitCacheRequest request;
    request.tokens = tokens;
    request.route_pairs = checked_int_product(tokens, 2, "post-logit cache R=T*2");
    const auto desired_ids = make_route_ids(tokens, 2, experts_, distribution_, zipf_s_, seed);
    request.logits.assign(static_cast<std::size_t>(tokens) * experts_, -8.0F);
    for (int token = 0; token < tokens; ++token) {
      request.logits[static_cast<std::size_t>(token) * experts_ +
                     desired_ids[static_cast<std::size_t>(token) * 2]] = 2.0F;
      request.logits[static_cast<std::size_t>(token) * experts_ +
                     desired_ids[static_cast<std::size_t>(token) * 2 + 1]] = 1.0F;
    }
    top2_selected_softmax_reference(request.logits, tokens, experts_, request.ids,
                                    request.route_weights);
    request.counts = counts_from_ids(request.ids, experts_);
    request.offsets = offsets_from_counts(request.counts);
    request.active_experts =
        static_cast<int>(std::count_if(request.counts.begin(), request.counts.end(),
                                       [](std::int32_t count) { return count > 0; }));
    request.expected_output = build_output_reference_for(request);
    return request;
  }

  void apply_cache_request(std::size_t index) {
    const PostLogitCacheRequest& request = cache_requests_.at(index);
    tokens_ = request.tokens;
    route_pairs_ = request.route_pairs;
    active_experts_ = request.active_experts;
    logits_host_ = request.logits;
    ids_expected_ = request.ids;
    route_weights_expected_ = request.route_weights;
    counts_expected_ = request.counts;
    offsets_expected_ = request.offsets;
    expected_output_ = request.expected_output;
  }

  void copy_active_logits_to_device(cudaStream_t stream) {
    cuda_check(cudaMemcpyAsync(logits_.data(), logits_host_.data(),
                               logits_host_.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
               "copy active post-logit cache input");
  }

  PostLogitGraphCacheKey current_cache_key() const {
    return {variant_name_, tokens_, experts_, 2,
            hidden_,       output_, "fp32",   "topk_fusedscan_permute_grouped_unpermute"};
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
    const PostLogitGraphCacheKey key = current_cache_key();
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
    PostLogitGraphCacheEntry entry;
    entry.key = key;
    entry.last_use = graph_cache_clock_;
    capture_graph(stream, &entry.graph);
    cuda_check(cudaGraphInstantiate(&entry.exec, entry.graph, 0),
               "instantiate cached post-logit graph");
    cuda_check(cudaGraphUpload(entry.exec, stream), "upload cached post-logit graph");
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

  void allocate_and_copy(cudaStream_t stream) {
    x_.resize(x_host_.size());
    expert_weights_.resize(expert_weights_host_.size());
    logits_.resize(static_cast<std::size_t>(max_tokens_capacity_) * experts_);
    if (include_router_projection_) router_weights_.resize(router_weights_host_.size());
    ids_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    route_weights_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    counts_.resize(static_cast<std::size_t>(experts_));
    offsets_.resize(static_cast<std::size_t>(experts_ + 1));
    cursors_.resize(static_cast<std::size_t>(experts_));
    x_permuted_.resize(static_cast<std::size_t>(max_route_pairs_capacity_) * hidden_);
    route_pos_.resize(static_cast<std::size_t>(max_route_pairs_capacity_));
    y_permuted_.resize(static_cast<std::size_t>(max_route_pairs_capacity_) * output_);
    y_.resize(static_cast<std::size_t>(max_tokens_capacity_) * output_);
    x_.copy_from_host(x_host_, stream);
    expert_weights_.copy_from_host(expert_weights_host_, stream);
    if (include_router_projection_) {
      router_weights_.copy_from_host(router_weights_host_, stream);
    } else {
      copy_active_logits_to_device(stream);
    }
    if (uses_postlogit_graph_pointer_rotation(variant_name_)) {
      if (include_router_projection_) {
        throw std::invalid_argument("post-logit graph pointer rotation requires logits input");
      }
      logits_rotating_.resize(logits_host_.size());
      y_rotating_.resize(static_cast<std::size_t>(max_tokens_capacity_) * output_);
      logits_rotating_.copy_from_host(logits_host_, stream);
    }
  }

  void setup_library_chain(cudaStream_t stream) {
    if (!include_router_projection_) {
      throw std::invalid_argument(
          "library_all_baselines_chain is only defined for chain_from_tokens");
    }
#if RAGGEDROUTE_HAS_CUBLAS && RAGGEDROUTE_HAS_CCCL && RAGGEDROUTE_HAS_CUTLASS && \
    RAGGEDROUTE_HAS_VLLM_UNPERMUTE
    dense_cublaslt_plan_ = library_baseline::create_dense_cublaslt_plan(
        tokens_, experts_, hidden_, 64ULL * 1024ULL * 1024ULL, &dense_library_workspace_bytes_);
    dense_library_workspace_.resize(dense_library_workspace_bytes_);

    cuda_check(
        library_baseline::query_cub_histogram_workspace(
            static_cast<std::size_t>(route_pairs_), experts_, &histogram_library_workspace_bytes_),
        "query chain CUB histogram workspace");
    histogram_library_workspace_.resize(histogram_library_workspace_bytes_);

    cuda_check(library_baseline::query_vllm_permute_workspace(tokens_, experts_, 2,
                                                              &permute_library_workspace_bytes_),
               "query chain vLLM permute workspace");
    permute_library_workspace_.resize(permute_library_workspace_bytes_);
    cuda_check(library_baseline::initialize_vllm_permute_workspace(
                   permute_library_workspace_.data(), permute_library_workspace_bytes_, tokens_,
                   experts_, 2, stream),
               "initialize chain vLLM permute workspace");

    // The repository's CUTLASS baseline requires fixed host problem metadata.
    // The deterministic chain input produces the same offsets on every sample;
    // the report marks this library path diagnostic rather than strictly L3-comparable.
    grouped_cutlass_plan_ = library_baseline::create_grouped_cutlass_plan(
        x_permuted_.data(), expert_weights_.data(), y_permuted_.data(), offsets_expected_.data(),
        experts_, hidden_, output_);
    grouped_library_workspace_bytes_ =
        library_baseline::grouped_cutlass_workspace_bytes(grouped_cutlass_plan_);
    grouped_library_workspace_.resize(grouped_library_workspace_bytes_);
    library_baseline::initialize_grouped_cutlass_plan(grouped_cutlass_plan_,
                                                      grouped_library_workspace_.data(),
                                                      grouped_library_workspace_bytes_, stream);
#else
    (void)stream;
    throw std::runtime_error(
        "library_all_baselines_chain requires cuBLASLt, CCCL, CUTLASS, and vLLM baselines");
#endif
  }

  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {"T",
                                           "E",
                                           "top_k",
                                           "K",
                                           "N",
                                           "distribution",
                                           "zipf_s",
                                           "route_trace_path",
                                           "route_trace_frame",
                                           "graph_cache_tokens"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown chain param: " + name);
    }
  }

  bool include_router_projection_ = false;
  std::string variant_name_;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  int tokens_ = 0, experts_ = 0, hidden_ = 0, output_ = 0;
  int route_pairs_ = 0, active_experts_ = 0;
  int max_tokens_capacity_ = 0, max_route_pairs_capacity_ = 0;
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::string route_trace_path_, route_trace_id_, route_trace_source_kind_, route_trace_frame_id_;
  int route_trace_frame_ = 0, route_trace_frame_count_ = 0;
  std::vector<float> x_host_, router_weights_host_, logits_host_;
  std::vector<float> expert_weights_host_, route_weights_expected_;
  std::vector<float> expected_output_;
  std::vector<std::int32_t> ids_expected_, counts_expected_, offsets_expected_;
  std::vector<PostLogitCacheRequest> cache_requests_;
  std::size_t cache_request_cursor_ = 0;
  std::string graph_cache_tokens_text_;
  DeviceBuffer<float> x_, router_weights_, logits_, logits_rotating_, route_weights_;
  DeviceBuffer<float> expert_weights_, x_permuted_, y_permuted_, y_, y_rotating_;
  DeviceBuffer<std::int32_t> ids_, counts_, offsets_, cursors_, route_pos_;
  std::size_t dense_library_workspace_bytes_ = 0;
  std::size_t histogram_library_workspace_bytes_ = 0;
  std::size_t permute_library_workspace_bytes_ = 0;
  std::size_t grouped_library_workspace_bytes_ = 0;
  DeviceBuffer<std::uint8_t> dense_library_workspace_, histogram_library_workspace_;
  DeviceBuffer<std::uint8_t> permute_library_workspace_, grouped_library_workspace_;
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
  static constexpr std::size_t kGraphCacheCapacity = 16;
  std::vector<PostLogitGraphCacheEntry> graph_cache_;
  std::uint64_t graph_cache_clock_ = 0;
  std::size_t graph_cached_kernel_nodes_ = 0;
  std::size_t graph_memory_baseline_free_bytes_ = 0;
  double graph_cache_miss_setup_us_total_ = 0.0;
#if RAGGEDROUTE_HAS_CUBLAS
  library_baseline::DenseCublasLtPlan* dense_cublaslt_plan_ = nullptr;
#endif
#if RAGGEDROUTE_HAS_CUTLASS
  library_baseline::GroupedCutlassPlan* grouped_cutlass_plan_ = nullptr;
#endif
};

}  // namespace

AdapterPtr make_chain_adapter(bool include_router_projection, const std::string& variant_name) {
  return std::make_unique<ChainAdapter>(include_router_projection, variant_name);
}

}  // namespace raggedroute::benchmark
