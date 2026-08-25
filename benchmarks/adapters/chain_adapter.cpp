#include <algorithm>
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

class ChainAdapter final : public BenchmarkAdapter {
 public:
  ChainAdapter(bool include_router_projection, const std::string& variant_name)
      : include_router_projection_(include_router_projection), variant_name_(variant_name) {}
  ~ChainAdapter() override {
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
    if (variant_name_ == "library_all_baselines_chain") {
      return prefix + " with repository library baselines (diagnostic contract)";
    }
    return prefix + " baseline";
  }
  bool supports(MeasurementLevel level) const override {
    return level == MeasurementLevel::kChainSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel) const override { return {}; }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
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
    if (variant_name_ == "library_all_baselines_chain") setup_library_chain(stream);
  }

  void prepare_sample(MeasurementLevel, cudaStream_t) override {}

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    if (level != MeasurementLevel::kChainSteady) {
      throw std::invalid_argument("chain adapter only supports L3");
    }
    const RuntimeContext context = make_runtime_context(stream, architecture_);
    const bool all_candidates = variant_name_ == "cuda_all_candidates_chain";
    const bool all_libraries = variant_name_ == "library_all_baselines_chain";
    const bool retained_postlogit = variant_name_ == "cuda_postlogit_retained_main";
    const bool integrated_postlogit = variant_name_ == "cuda_postlogit_integrated_latest";
    const bool unified_postlogit = retained_postlogit || integrated_postlogit;
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
      cuda_check(library_baseline::launch_cub_block_radix_top2(
                     logits_.data(), ids_.data(), route_weights_.data(), tokens_, experts_, stream),
                 "chain CUB BlockRadixSort Top-2");
#else
      throw std::runtime_error("library chain requires CCCL support");
#endif
    } else {
      TopKGateArgs topk_args;
      topk_args.logits.data = logits_.data();
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
                             ops::kTokenPermuteCandidateImplementation};
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
          integrated_postlogit ? ops::kGroupedGemmSm86Fp32V2Implementation
                               : ops::kGroupedGemmSm86Fp32V1Implementation;
      cuda_check(
          ops::launch_grouped_gemm_optimized(
              static_cast<const float*>(grouped_args.x_permuted.data),
              static_cast<const float*>(grouped_args.expert_weights.data), grouped_args.offsets,
              static_cast<float*>(grouped_args.y_permuted.data), grouped_args.experts,
              grouped_args.hidden, grouped_args.output, grouped_args.max_expert_tokens,
              grouped_implementation, stream),
          "chain benchmark-only grouped_gemm candidate");
    } else {
      operator_check(grouped_gemm(grouped_args, context), "chain grouped_gemm operator");
    }

    UnpermuteArgs unpermute_args;
    unpermute_args.y_permuted.data = y_permuted_.data();
    unpermute_args.route_pos = route_pos_.data();
    unpermute_args.route_weights.data = route_weights_.data();
    unpermute_args.y.data = y_.data();
    unpermute_args.tokens = tokens_;
    unpermute_args.top_k = 2;
    unpermute_args.output = output_;
    if (all_libraries) {
#if RAGGEDROUTE_HAS_VLLM_UNPERMUTE
      cuda_check(library_baseline::launch_vllm_finalize_routing(
                     y_permuted_.data(), y_.data(), route_weights_.data(), route_pos_.data(),
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
    const double weight_tolerance =
        variant_name_ == "library_all_baselines_chain" ? 2.0e-5 : 1.0e-6;
    auto weight_result =
        compare_floats(route_weights, route_weights_expected_, weight_tolerance, weight_tolerance);
    if (!weight_result.ok) {
      weight_result.message = "chain route weights: " + weight_result.message;
      return weight_result;
    }
    auto output_result =
        compare_floats(y_.copy_to_host(stream), expected_output_, 1.0e-4, 4.0e-5 * hidden_);
    if (!output_result.ok) output_result.message = "chain final output: " + output_result.message;
    return output_result;
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
    const bool retained_postlogit = variant_name_ == "cuda_postlogit_retained_main";
    const bool integrated_postlogit = variant_name_ == "cuda_postlogit_integrated_latest";
    const bool unified_postlogit = retained_postlogit || integrated_postlogit;
    return {
        {"components",
         include_router_projection_
             ? std::string("dense_gemm,topk_gate,histogram,exclusive_scan,token_permute,"
                           "grouped_gemm,unpermute")
             : std::string(
                   "topk_gate,histogram,exclusive_scan,token_permute,grouped_gemm,unpermute")},
        {"component_variant",
         integrated_postlogit ? std::string("postlogit_integrated_latest_grouped_v2")
         : retained_postlogit ? std::string("postlogit_retained_main_grouped_v1")
         : variant_name_ == "cuda_all_candidates_chain" ? std::string("all_selected_cuda_candidates")
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
         unified_postlogit ? std::string("cuda_candidate_v2_from_offsets")
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
         unified_postlogit ? std::string("cuda_warp_token_vec4")
         : variant_name_ == "cuda_all_candidates_chain"     ? std::string("cuda_warp_token_vec4")
         : variant_name_ == "library_all_baselines_chain" ? std::string("vllm_finalize_routing")
         : variant_name_ == "cuda_unpermute_candidate"    ? std::string("cuda_warp_token_vec4")
                                                          : std::string("cuda_naive")},
        {"grouped_variant", integrated_postlogit
                                ? std::string("cuda_grouped_sm86_fp32_v2")
                            : retained_postlogit
                                ? std::string("cuda_grouped_sm86_fp32_v1")
                            : variant_name_ == "cuda_all_candidates_chain"
                                ? std::string("cuda_grouped_sm86_fp32_v1")
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
        {"topk_variant", unified_postlogit
                             ? std::string("cuda_local_pair_two_reduce_top2_v4")
                         : variant_name_ == "cuda_all_candidates_chain"
                             ? std::string("cuda_local_pair_two_reduce_top2_v4")
                         : variant_name_ == "library_all_baselines_chain"
                             ? std::string("cub_block_radix_top2_benchmark_only")
                             : std::string("cuda_naive")},
        {"strict_comparison_eligible", variant_name_ != "library_all_baselines_chain"},
        {"materialize_sorted_route", false}};
  }

  WorkEstimate work_estimate(MeasurementLevel) const override {
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
    work.logical_bytes += 2.0 * sizeof(std::int32_t) * experts_;
    work.logical_bytes += 2.0 * sizeof(float) * route_pairs_ * hidden_;
    work.logical_bytes +=
        sizeof(float) *
        (static_cast<double>(route_pairs_) * hidden_ + static_cast<double>(route_pairs_) * output_ +
         static_cast<double>(active_experts_) * hidden_ * output_);
    work.logical_bytes += sizeof(float) * (static_cast<double>(route_pairs_) * output_ +
                                           static_cast<double>(tokens_) * output_);
    const bool fused_histogram_scan = variant_name_ == "cuda_fused_histogram_scan" ||
                                      variant_name_ == "cuda_postlogit_retained_main" ||
                                      variant_name_ == "cuda_postlogit_integrated_latest" ||
                                      variant_name_ == "cuda_all_candidates_chain";
    work.operator_metrics["kernel_launches"] = static_cast<std::int64_t>(
        (include_router_projection_ ? 9 : 8) - (fused_histogram_scan ? 1 : 0));
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
    const std::set<std::string> allowed = {"T", "E", "K", "N", "distribution", "zipf_s"};
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
  double zipf_s_ = 0.0;
  std::string distribution_;
  std::vector<float> x_host_, router_weights_host_, logits_host_;
  std::vector<float> expert_weights_host_, route_weights_expected_;
  std::vector<float> expected_output_;
  std::vector<std::int32_t> ids_expected_, counts_expected_, offsets_expected_;
  DeviceBuffer<float> x_, router_weights_, logits_, route_weights_;
  DeviceBuffer<float> expert_weights_, x_permuted_, y_permuted_, y_;
  DeviceBuffer<std::int32_t> ids_, counts_, offsets_, cursors_, route_pos_;
  std::size_t dense_library_workspace_bytes_ = 0;
  std::size_t histogram_library_workspace_bytes_ = 0;
  std::size_t permute_library_workspace_bytes_ = 0;
  std::size_t grouped_library_workspace_bytes_ = 0;
  DeviceBuffer<std::uint8_t> dense_library_workspace_, histogram_library_workspace_;
  DeviceBuffer<std::uint8_t> permute_library_workspace_, grouped_library_workspace_;
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
