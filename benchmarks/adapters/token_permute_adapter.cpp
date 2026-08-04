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
#include "permute/cuda_candidate/optimized_internal.h"

namespace raggedroute::benchmark {
namespace {

bool is_optimized_variant(const std::string& name) {
  return name == "cuda_atomic_vectorized_128" || name == "cuda_atomic_vectorized_64" ||
         name == "cuda_atomic_vectorized_256" || name == "cuda_token_owned_top2" ||
         name == "cuda_block_partial" || name == "cuda_token_tile4_direct" ||
         name == "cuda_token_tile4_warp_aggregated" || name == "cuda_candidate_v2" ||
         name == "cuda_candidate" ||
         name == "cuda_candidate_from_ids" ||
         name == "cuda_fused_prepare_token_owned_from_ids" ||
         name == "cuda_fused_prepare_warp_aggregated_from_ids" ||
         name == "cuda_candidate_v2_from_ids";
}

bool is_from_ids_variant(const std::string& name) {
  return name == "cuda_naive_from_ids" || name == "cuda_candidate_from_ids" ||
         name == "cuda_fused_prepare_token_owned_from_ids" ||
         name == "cuda_fused_prepare_warp_aggregated_from_ids" ||
         name == "cuda_candidate_v2_from_ids";
}

bool is_fused_prepare_variant(const std::string& name) {
  return name == "cuda_fused_prepare_token_owned_from_ids" ||
         name == "cuda_fused_prepare_warp_aggregated_from_ids" ||
         name == "cuda_candidate_v2_from_ids";
}

std::uint32_t optimized_implementation(const std::string& name) {
  if (name == "cuda_atomic_vectorized_128")
    return ops::kTokenPermuteAtomicVectorized128Implementation;
  if (name == "cuda_atomic_vectorized_64")
    return ops::kTokenPermuteAtomicVectorized64Implementation;
  if (name == "cuda_atomic_vectorized_256")
    return ops::kTokenPermuteAtomicVectorized256Implementation;
  if (name == "cuda_token_owned_top2")
    return ops::kTokenPermuteTokenOwnedTop2Implementation;
  if (name == "cuda_block_partial") return ops::kTokenPermuteBlockPartialImplementation;
  if (name == "cuda_token_tile4_direct")
    return ops::kTokenPermuteTokenTile4DirectImplementation;
  if (name == "cuda_token_tile4_warp_aggregated" ||
      name == "cuda_fused_prepare_warp_aggregated_from_ids") {
    return ops::kTokenPermuteTokenTile4WarpAggregatedImplementation;
  }
  if (name == "cuda_fused_prepare_token_owned_from_ids")
    return ops::kTokenPermuteTokenOwnedTop2Implementation;
  if (name == "cuda_candidate_v2") return ops::kTokenPermuteShapeDispatchedV2Implementation;
  if (name == "cuda_candidate_v2_from_ids")
    return ops::kTokenPermuteShapeDispatchedV2Implementation;
  if (name == "cuda_candidate" || name == "cuda_candidate_from_ids")
    return ops::kTokenPermuteCandidateImplementation;
  return 0;
}

class TokenPermuteAdapter final : public BenchmarkAdapter {
 public:
  explicit TokenPermuteAdapter(const std::string& variant_name) : variant_name_(variant_name) {}
  std::string operator_name() const override { return "token_permute"; }
  std::string variant_name() const override { return variant_name_; }
  std::string description() const override {
    if (variant_name_ == "cuda_naive_from_ids") {
      return "Naive histogram, scan, and atomic-cursor permute from Top-K ids";
    }
    if (variant_name_ == "cuda_candidate_from_ids") {
      return "Histogram, scan, and selected CUDA candidate from Top-K ids";
    }
    if (variant_name_ == "cuda_atomic_vectorized_128") {
      return "128-thread atomic placement with aligned float4 row copy";
    }
    if (variant_name_ == "cuda_atomic_vectorized_64") {
      return "64-thread atomic placement with aligned float4 row copy";
    }
    if (variant_name_ == "cuda_atomic_vectorized_256") {
      return "256-thread atomic placement with aligned float4 row copy";
    }
    if (variant_name_ == "cuda_token_owned_top2") {
      return "Top-2 token-owned placement with one input load and two destination stores";
    }
    if (variant_name_ == "cuda_block_partial") {
      return "Block-private placement ranks followed by vectorized row copy";
    }
    if (variant_name_ == "cuda_token_tile4_direct") {
      return "Four Top-2 tokens per CTA with direct global cursor atomics";
    }
    if (variant_name_ == "cuda_token_tile4_warp_aggregated") {
      return "Four Top-2 tokens per CTA with expert-keyed warp-aggregated cursor atomics";
    }
    if (variant_name_ == "cuda_fused_prepare_token_owned_from_ids") {
      return "Fused single-CTA counts/scan/reset followed by token-owned Top-2 permute";
    }
    if (variant_name_ == "cuda_fused_prepare_warp_aggregated_from_ids") {
      return "Fused single-CTA counts/scan/reset followed by warp-aggregated tile4 permute";
    }
    if (variant_name_ == "cuda_candidate_v2") {
      return "Shape-dispatched SM86 v2 candidate: tile4 direct for large aligned Top-2, ID4 fallback";
    }
    if (variant_name_ == "cuda_candidate_v2_from_ids") {
      return "Fused counts/scan/reset followed by the shape-dispatched SM86 v2 candidate";
    }
    if (variant_name_ == "cuda_candidate") {
      return "SM86 v2 shape-dispatched candidate: tile4 direct for large aligned Top-2, token-owned fallback";
    }
    if (variant_name_ == "vllm_moe_permute") {
      return "Adapted vLLM radix-sort mapping and vectorized row expansion";
    }
    if (variant_name_ == "vllm_expand_rows") {
      return "Adapted vLLM row expansion with mapping prepared outside timing";
    }
    return "Atomic-cursor placement with scalar FP32 row copy";
  }
  bool supports(MeasurementLevel level) const override {
    if (is_from_ids_variant(variant_name_) || variant_name_ == "vllm_moe_permute") {
      return level == MeasurementLevel::kOperatorSteady;
    }
    if (variant_name_ == "vllm_expand_rows") {
      return level == MeasurementLevel::kKernelBody;
    }
    return level == MeasurementLevel::kKernelBody || level == MeasurementLevel::kOperatorSteady;
  }
  RepeatPolicy repeat_policy(MeasurementLevel level) const override {
    if ((variant_name_ == "cuda_naive" || is_optimized_variant(variant_name_)) &&
        !is_from_ids_variant(variant_name_) && level == MeasurementLevel::kKernelBody) {
      return {1, "L1 token permute mutates cursor state; use one launch per event sample"};
    }
    return {};
  }

  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    reject_unknown(options);
    architecture_ = current_device_architecture();
    tokens_ = get_int_option(options, "T", 128);
    experts_ = get_int_option(options, "E", 16);
    top_k_ = get_int_option(options, "top_k", 2);
    hidden_ = get_int_option(options, "K", 256);
    distribution_ = get_option(options, "distribution", "uniform");
    zipf_s_ = get_double_option(options, "zipf_s", 1.0);
    materialize_sorted_route_ = get_bool_option(options, "materialize_sorted_route", true);
    if (experts_ < top_k_ || experts_ > 64) {
      throw std::invalid_argument("current permute adapter requires top_k<=E<=64");
    }
    route_pairs_ = checked_int_product(tokens_, top_k_, "R=T*top_k");

    host_x_ = make_random_floats(static_cast<std::size_t>(tokens_) * hidden_, seed);
    host_ids_ = make_route_ids(tokens_, top_k_, experts_, distribution_, zipf_s_, seed + 1);
    host_counts_ = counts_from_ids(host_ids_, experts_);
    host_offsets_ = offsets_from_counts(host_counts_);

    x_.resize(host_x_.size());
    ids_.resize(host_ids_.size());
    offsets_.resize(host_offsets_.size());
    counts_.resize(host_counts_.size());
    cursors_.resize(static_cast<std::size_t>(experts_));
    x_permuted_.resize(static_cast<std::size_t>(route_pairs_) * hidden_);
    route_pos_.resize(static_cast<std::size_t>(route_pairs_));
    if (materialize_sorted_route_) {
      sorted_route_.resize(static_cast<std::size_t>(route_pairs_));
    }
    x_.copy_from_host(host_x_, stream);
    ids_.copy_from_host(host_ids_, stream);
    offsets_.copy_from_host(host_offsets_, stream);
    cuda_check(cudaMemsetAsync(cursors_.data(), 0, cursors_.bytes(), stream),
               "initialize permute cursors");
#if RAGGEDROUTE_HAS_CCCL
    if (variant_name_ == "vllm_moe_permute" || variant_name_ == "vllm_expand_rows") {
      cuda_check(library_baseline::query_vllm_permute_workspace(tokens_, experts_, top_k_,
                                                                &library_workspace_bytes_),
                 "query vLLM permute workspace");
      library_workspace_.resize(library_workspace_bytes_);
      cuda_check(library_baseline::initialize_vllm_permute_workspace(
                     library_workspace_.data(), library_workspace_bytes_, tokens_, experts_, top_k_,
                     stream),
                 "initialize vLLM permute workspace");
      if (variant_name_ == "vllm_expand_rows") {
        cuda_check(library_baseline::prepare_vllm_permute_mapping(
                       ids_.data(), library_workspace_.data(), library_workspace_bytes_, tokens_,
                       experts_, top_k_, stream),
                   "prepare vLLM copy-only mapping");
      }
    }
#endif
  }

  void prepare_sample(MeasurementLevel level, cudaStream_t stream) override {
    if ((variant_name_ == "cuda_naive" || is_optimized_variant(variant_name_)) &&
        !is_from_ids_variant(variant_name_) && level == MeasurementLevel::kKernelBody) {
      reset_cursors(stream);
    }
  }

  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
#if RAGGEDROUTE_HAS_CCCL
    if (variant_name_ == "vllm_expand_rows") {
      cuda_check(
          library_baseline::launch_vllm_expand_rows(
              x_.data(), x_permuted_.data(), route_pos_.data(),
              materialize_sorted_route_ ? sorted_route_.data() : nullptr, library_workspace_.data(),
              library_workspace_bytes_, tokens_, experts_, top_k_, hidden_, stream),
          "vLLM expandInputRowsKernel");
      return;
    }
    if (variant_name_ == "vllm_moe_permute") {
      cuda_check(
          library_baseline::launch_vllm_moe_permute(
              x_.data(), ids_.data(), x_permuted_.data(), route_pos_.data(),
              materialize_sorted_route_ ? sorted_route_.data() : nullptr, library_workspace_.data(),
              library_workspace_bytes_, tokens_, experts_, top_k_, hidden_, stream),
          "vLLM moe_permute_with_scratch baseline");
      return;
    }
#endif
    if (is_from_ids_variant(variant_name_)) {
      if (is_fused_prepare_variant(variant_name_)) {
        cuda_check(ops::launch_token_permute_prepare_offsets_fused(
                       ids_.data(), counts_.data(), offsets_.data(), cursors_.data(), tokens_,
                       experts_, top_k_, stream),
                   "fused counts/scan/reset in permute-from-ids candidate");
        cuda_check(ops::launch_token_permute_optimized(
                       x_.data(), ids_.data(), offsets_.data(), cursors_.data(),
                       x_permuted_.data(), route_pos_.data(),
                       materialize_sorted_route_ ? sorted_route_.data() : nullptr, tokens_,
                       experts_, top_k_, hidden_, optimized_implementation(variant_name_), stream),
                   "optimized permute after fused preparation");
        return;
      }
      HistogramArgs histogram_args;
      histogram_args.expert_ids = ids_.data();
      histogram_args.counts = counts_.data();
      histogram_args.route_pairs = route_pairs_;
      histogram_args.experts = experts_;
      operator_check(histogram(histogram_args, make_runtime_context(stream, architecture_)),
                     "histogram in permute-from-ids baseline");
      ExclusiveScanArgs scan_args;
      scan_args.counts = counts_.data();
      scan_args.offsets = offsets_.data();
      scan_args.experts = experts_;
      operator_check(exclusive_scan(scan_args, make_runtime_context(stream, architecture_)),
                     "scan in permute-from-ids baseline");
      enqueue_operator(stream);
      return;
    }
    if (level == MeasurementLevel::kKernelBody) {
      if (is_optimized_variant(variant_name_)) {
        cuda_check(
            ops::launch_token_permute_optimized(
                x_.data(), ids_.data(), offsets_.data(), cursors_.data(), x_permuted_.data(),
                route_pos_.data(), materialize_sorted_route_ ? sorted_route_.data() : nullptr,
                tokens_, experts_, top_k_, hidden_, optimized_implementation(variant_name_),
                stream),
            "launch_token_permute_optimized");
        return;
      }
      cuda_check(ops::launch_token_permute_naive(
                     x_.data(), ids_.data(), offsets_.data(), cursors_.data(), x_permuted_.data(),
                     route_pos_.data(), materialize_sorted_route_ ? sorted_route_.data() : nullptr,
                     tokens_, top_k_, hidden_, stream),
                 "launch_token_permute_naive");
      return;
    }
    enqueue_operator(stream);
  }

  void enqueue_operator(cudaStream_t stream) {
    TokenPermuteArgs args;
    args.x.data = x_.data();
    args.expert_ids = ids_.data();
    args.offsets = offsets_.data();
    args.x_permuted.data = x_permuted_.data();
    args.route_pos = route_pos_.data();
    args.sorted_route = materialize_sorted_route_ ? sorted_route_.data() : nullptr;
    args.tokens = tokens_;
    args.experts = experts_;
    args.top_k = top_k_;
    args.hidden = hidden_;
    if (is_optimized_variant(variant_name_)) {
      args.kernel = {KernelFamily::kCudaOptimized, optimized_implementation(variant_name_)};
    }
    operator_check(token_permute(args, make_runtime_context(stream, architecture_, cursors_.data(),
                                                            cursors_.bytes())),
                   "token_permute operator");
  }

  ValidationResult validate(cudaStream_t stream) override {
    const auto positions = route_pos_.copy_to_host(stream);
    const auto output = x_permuted_.copy_to_host(stream);
    std::vector<std::int32_t> reverse;
    if (materialize_sorted_route_) reverse = sorted_route_.copy_to_host(stream);
    std::vector<bool> seen(static_cast<std::size_t>(route_pairs_), false);
    for (int route = 0; route < route_pairs_; ++route) {
      const int position = positions[static_cast<std::size_t>(route)];
      const int expert = host_ids_[static_cast<std::size_t>(route)];
      if (position < host_offsets_[static_cast<std::size_t>(expert)] ||
          position >= host_offsets_[static_cast<std::size_t>(expert + 1)]) {
        return {false, "route_pos escaped its expert segment", {}, {}};
      }
      if (seen[static_cast<std::size_t>(position)]) {
        return {false, "duplicate route_pos destination", {}, {}};
      }
      seen[static_cast<std::size_t>(position)] = true;
      if (materialize_sorted_route_ && reverse[static_cast<std::size_t>(position)] != route) {
        return {false, "sorted_route is not the inverse of route_pos", {}, {}};
      }
      const int token = route / top_k_;
      for (int column = 0; column < hidden_; ++column) {
        const float actual = output[static_cast<std::size_t>(position) * hidden_ + column];
        const float expected = host_x_[static_cast<std::size_t>(token) * hidden_ + column];
        if (actual != expected) {
          return {false, "permuted activation mismatch", {}, {}};
        }
      }
    }
    return {true, "mapping is bijective and rows match input", 0.0, 0.0};
  }

  FieldMap case_config() const override {
    FieldMap config = {{"T", static_cast<std::int64_t>(tokens_)},
                       {"E", static_cast<std::int64_t>(experts_)},
                       {"top_k", static_cast<std::int64_t>(top_k_)},
                       {"K", static_cast<std::int64_t>(hidden_)},
                       {"R", static_cast<std::int64_t>(route_pairs_)},
                       {"dtype", std::string("fp32")},
                       {"distribution", distribution_},
                       {"zipf_s", zipf_s_},
                       {"materialize_sorted_route", materialize_sorted_route_}};
    if (is_from_ids_variant(variant_name_) || variant_name_ == "vllm_moe_permute") {
      config["placement_order"] = std::string("unspecified_within_expert");
      config["input_boundary"] = std::string("topk_ids_to_permuted_rows");
    } else if (variant_name_ == "vllm_expand_rows") {
      config["placement_order"] = std::string("unspecified_within_expert");
      config["input_boundary"] = std::string("prepared_mapping_to_permuted_rows");
    } else {
      // Keep the suite-v1 evidence signature byte-for-byte compatible.
      config["placement_order"] = std::string("unstable_atomic_cursor");
    }
    return config;
  }
  FieldMap variant_config() const override {
    if (variant_name_ == "cuda_naive_from_ids") {
      return {{"components", std::string("histogram,exclusive_scan,token_permute")},
              {"placement", std::string("global_atomic_cursor")},
              {"copy", std::string("scalar")}};
    }
    if (variant_name_ == "cuda_candidate_from_ids") {
      return {{"components", std::string("histogram,exclusive_scan,cuda_candidate")},
              {"placement", std::string("global_atomic_cursor")},
              {"copy", std::string("float4_fast_scalar_fallback")},
              {"implementation_id",
               static_cast<std::int64_t>(ops::kTokenPermuteCandidateImplementation)}};
    }
    if (is_fused_prepare_variant(variant_name_)) {
      return {{"components", std::string("fused_counts_scan_reset,cuda_candidate")},
              {"placement", variant_name_ == "cuda_fused_prepare_warp_aggregated_from_ids"
                                    ? std::string("warp_aggregated_tile4")
                                : variant_name_ == "cuda_candidate_v2_from_ids"
                                    ? std::string("shape_dispatch_tile4_direct_or_token_owned")
                                    : std::string("token_owned_top2_atomic")},
              {"copy", std::string("float4_fast_scalar_fallback")},
              {"kernel_launches", static_cast<std::int64_t>(2)},
              {"implementation_id",
               static_cast<std::int64_t>(optimized_implementation(variant_name_))}};
    }
    if (variant_name_ == "vllm_moe_permute") {
      return {{"upstream_symbol", std::string("moe_permute_with_scratch")},
              {"mapping", std::string("cub_radix_sort_and_expert_scan")},
              {"vector_width_bytes", static_cast<std::int64_t>(hidden_ % 4 == 0 ? 16 : 4)},
              {"alignment_policy", std::string("float4_fast_scalar_fallback")}};
    }
    if (variant_name_ == "vllm_expand_rows") {
      return {{"upstream_symbol", std::string("expandInputRowsKernelLauncher")},
              {"mapping", std::string("prepared_outside_timing")},
              {"vector_width_bytes", static_cast<std::int64_t>(hidden_ % 4 == 0 ? 16 : 4)},
              {"alignment_policy", std::string("float4_fast_scalar_fallback")}};
    }
    if (is_optimized_variant(variant_name_)) {
      const std::uint32_t implementation = optimized_implementation(variant_name_);
      std::string placement = "global_atomic_cursor";
      std::int64_t threads = 128;
      std::int64_t launches = 1;
      if (implementation == ops::kTokenPermuteAtomicVectorized64Implementation) threads = 64;
      if (implementation == ops::kTokenPermuteAtomicVectorized256Implementation) threads = 256;
      if (implementation == ops::kTokenPermuteTokenOwnedTop2Implementation) {
        placement = "token_owned_top2_atomic";
      }
      if (implementation == ops::kTokenPermuteBlockPartialImplementation) {
        placement = "block_private_rank_global_reserve";
        threads = 256;
        launches = hidden_ == 0 ? 1 : 2;
      }
      if (implementation == ops::kTokenPermuteTokenTile4DirectImplementation) {
        placement = "token_tile4_direct_atomic";
      }
      if (implementation == ops::kTokenPermuteTokenTile4WarpAggregatedImplementation) {
        placement = "token_tile4_warp_aggregated_atomic";
      }
      if (implementation == ops::kTokenPermuteShapeDispatchedV2Implementation) {
        placement = "shape_dispatch_tile4_direct_or_token_owned";
      }
      return {{"implementation_id", static_cast<std::int64_t>(implementation)},
              {"threads", threads},
              {"kernel_launches", launches},
              {"placement", placement},
              {"copy", std::string("float4_fast_scalar_fallback")},
              {"vector_width_bytes", static_cast<std::int64_t>(hidden_ % 4 == 0 ? 16 : 4)}};
    }
    return {{"threads_per_route", static_cast<std::int64_t>(128)},
            {"copy", std::string("scalar")},
            {"cursor", std::string("global_atomic")}};
  }
  WorkEstimate work_estimate(MeasurementLevel level) const override {
    WorkEstimate work;
    work.logical_bytes = 2.0 * sizeof(float) * route_pairs_ * hidden_ +
                         2.0 * sizeof(std::int32_t) * route_pairs_ +
                         (materialize_sorted_route_ ? sizeof(std::int32_t) * route_pairs_ : 0.0);
    work.operator_metrics["copied_rows"] = static_cast<std::int64_t>(route_pairs_);
    if ((variant_name_ == "cuda_naive" ||
         (is_optimized_variant(variant_name_) && !is_from_ids_variant(variant_name_))) &&
        level == MeasurementLevel::kOperatorSteady) {
      work.logical_bytes += static_cast<double>(cursors_.bytes());
      work.operator_metrics["cursor_reset_bytes"] = static_cast<std::int64_t>(cursors_.bytes());
    }
    if (is_from_ids_variant(variant_name_)) {
      work.logical_bytes +=
          static_cast<double>(ids_.bytes() + 3 * counts_.bytes() + offsets_.bytes() +
                              cursors_.bytes());
      work.operator_metrics["counts_reset_bytes"] = static_cast<std::int64_t>(counts_.bytes());
      work.operator_metrics["cursor_reset_bytes"] = static_cast<std::int64_t>(cursors_.bytes());
      work.operator_metrics["dynamic_mapping_included"] = true;
    } else if (variant_name_ == "vllm_moe_permute") {
      // CUB's physical radix-sort traffic is intentionally not invented here.
      work.operator_metrics["dynamic_mapping_included"] = true;
    }
    return work;
  }
  std::size_t workspace_bytes() const override {
    if (variant_name_ == "vllm_moe_permute" || variant_name_ == "vllm_expand_rows") {
      return library_workspace_bytes_;
    }
    if (is_from_ids_variant(variant_name_)) {
      return cursors_.bytes() + counts_.bytes() + offsets_.bytes();
    }
    return cursors_.bytes();
  }
  std::vector<std::string> excluded_steps(MeasurementLevel level) const override {
    std::vector<std::string> excluded = {"input_generation", "h2d_copy", "workspace_allocation"};
    if (variant_name_ == "vllm_expand_rows") {
      excluded.push_back("mapping_generation");
    } else if ((variant_name_ == "cuda_naive" || is_optimized_variant(variant_name_)) &&
               !is_from_ids_variant(variant_name_) &&
               level == MeasurementLevel::kKernelBody) {
      excluded.push_back("cursor_reset");
    }
    return excluded;
  }

 private:
  static void reject_unknown(const OptionMap& options) {
    const std::set<std::string> allowed = {
        "T", "E", "top_k", "K", "distribution", "zipf_s", "materialize_sorted_route"};
    for (const auto& [name, unused] : options) {
      (void)unused;
      if (!allowed.count(name)) throw std::invalid_argument("unknown token_permute param: " + name);
    }
  }
  void reset_cursors(cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(cursors_.data(), 0, cursors_.bytes(), stream),
               "reset permute cursors");
  }

  int tokens_ = 0, experts_ = 0, top_k_ = 0, hidden_ = 0, route_pairs_ = 0;
  std::string variant_name_;
  std::size_t library_workspace_bytes_ = 0;
  DeviceArchitecture architecture_ = DeviceArchitecture::kOther;
  double zipf_s_ = 0.0;
  bool materialize_sorted_route_ = true;
  std::string distribution_;
  std::vector<float> host_x_;
  std::vector<std::int32_t> host_ids_, host_counts_, host_offsets_;
  DeviceBuffer<float> x_, x_permuted_;
  DeviceBuffer<std::int32_t> ids_, counts_, offsets_, cursors_, route_pos_, sorted_route_;
  DeviceBuffer<std::uint8_t> library_workspace_;
};

}  // namespace

AdapterPtr make_token_permute_adapter(const std::string& variant_name) {
  return std::make_unique<TokenPermuteAdapter>(variant_name);
}

}  // namespace raggedroute::benchmark
