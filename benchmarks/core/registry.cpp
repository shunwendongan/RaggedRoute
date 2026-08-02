#include "raggedroute/benchmark/registry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#if RAGGEDROUTE_HAS_CCCL
#include <cub/version.cuh>
#endif
#if RAGGEDROUTE_HAS_CUTLASS
#include <cutlass/version.h>
#endif

namespace raggedroute::benchmark {
namespace {

class RegisteredAdapter final : public BenchmarkAdapter {
 public:
  RegisteredAdapter(AdapterPtr inner, VariantDescriptor descriptor)
      : inner_(std::move(inner)), descriptor_(std::move(descriptor)) {}

  std::string operator_name() const override { return inner_->operator_name(); }
  std::string variant_name() const override { return descriptor_.name; }
  std::string description() const override { return inner_->description(); }
  bool supports(MeasurementLevel level) const override { return inner_->supports(level); }
  RepeatPolicy repeat_policy(MeasurementLevel level) const override {
    return inner_->repeat_policy(level);
  }
  void setup(const OptionMap& options, std::uint64_t seed, cudaStream_t stream) override {
    inner_->setup(options, seed, stream);
  }
  void prepare_sample(MeasurementLevel level, cudaStream_t stream) override {
    inner_->prepare_sample(level, stream);
  }
  void enqueue(MeasurementLevel level, cudaStream_t stream) override {
    inner_->enqueue(level, stream);
  }
  ValidationResult validate(cudaStream_t stream) override { return inner_->validate(stream); }
  FieldMap case_config() const override { return inner_->case_config(); }
  FieldMap variant_config() const override {
    FieldMap config = inner_->variant_config();
    config.insert_or_assign("implementation_category", descriptor_.implementation_category);
    config.insert_or_assign("implementation_version", descriptor_.implementation_version);
    config.insert_or_assign("implementation_revision", std::string(RAGGEDROUTE_GIT_SHA));
    config.insert_or_assign("dependency_revision", descriptor_.dependency_revision);
    config.insert_or_assign("algorithm_id", descriptor_.algorithm_id);
    config.insert_or_assign("math_mode", descriptor_.math_mode);
    return config;
  }
  WorkEstimate work_estimate(MeasurementLevel level) const override {
    return inner_->work_estimate(level);
  }
  std::size_t workspace_bytes() const override { return inner_->workspace_bytes(); }
  std::vector<std::string> excluded_steps(MeasurementLevel level) const override {
    return inner_->excluded_steps(level);
  }

 private:
  AdapterPtr inner_;
  VariantDescriptor descriptor_;
};

VariantDescriptor naive_descriptor(const std::string& algorithm_id) {
  return {"cuda_naive",     "in_tree_cuda", "raggedroute.cuda_naive.v1",
          "not_applicable", algorithm_id,   "strict_fp32"};
}

VariantDescriptor optimized_dense_descriptor(std::string name, std::string algorithm_id) {
  return {std::move(name), "in_tree_cuda", "raggedroute.cuda_optimized.v1", "not_applicable",
          std::move(algorithm_id), "strict_fp32"};
}

VariantDescriptor optimized_dense_v2_descriptor(std::string name, std::string algorithm_id) {
  return {std::move(name), "in_tree_cuda", "raggedroute.cuda_optimized.v2", "not_applicable",
          std::move(algorithm_id), "strict_fp32"};
}

VariantDescriptor descriptor(std::string name, std::string category, std::string version,
                             std::string dependency, std::string algorithm_id) {
  return {std::move(name),       std::move(category),     std::move(version),
          std::move(dependency), std::move(algorithm_id), "strict_fp32"};
}

std::string cuda_library_revision() {
  return std::string("CUDA Toolkit ") + RAGGEDROUTE_CUDATOOLKIT_VERSION;
}

std::string cccl_revision() {
#if RAGGEDROUTE_HAS_CCCL
  return "CUB_VERSION=" + std::to_string(CUB_VERSION) +
         "; configured=" + RAGGEDROUTE_CCCL_CONFIG_REVISION;
#else
  return "unavailable";
#endif
}

std::string cutlass_revision() {
#if RAGGEDROUTE_HAS_CUTLASS
  return "CUTLASS_VERSION=" + std::to_string(CUTLASS_VERSION) +
         "; configured=" + RAGGEDROUTE_CUTLASS_CONFIG_REVISION;
#else
  return "unavailable";
#endif
}

const VariantDescriptor& find_descriptor(const std::vector<VariantDescriptor>& descriptors,
                                         const std::string& target_name,
                                         const std::string& variant_name) {
  const auto found = std::find_if(
      descriptors.begin(), descriptors.end(),
      [&](const VariantDescriptor& descriptor) { return descriptor.name == variant_name; });
  if (found == descriptors.end()) {
    throw std::invalid_argument("unsupported variant '" + variant_name + "' for '" + target_name +
                                "'");
  }
  return *found;
}

std::vector<std::string> descriptor_names(const std::vector<VariantDescriptor>& descriptors) {
  std::vector<std::string> names;
  names.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) names.push_back(descriptor.name);
  return names;
}

}  // namespace

std::vector<std::string> available_operators() {
  return {"dense_gemm",    "topk_gate",    "histogram", "exclusive_scan",
          "token_permute", "grouped_gemm", "unpermute"};
}

std::vector<VariantDescriptor> available_variant_descriptors(const std::string& operator_name) {
  if (operator_name == "dense_gemm") {
    std::vector<VariantDescriptor> variants = {naive_descriptor("thread_per_output")};
    variants.push_back(optimized_dense_descriptor("cuda_tiled_scalar",
                                                  "shared_tile16_linear_cta_scalar"));
    variants.push_back(optimized_dense_descriptor("cuda_2d_mapping",
                                                  "direct_2d_row_column_scalar"));
    variants.push_back(optimized_dense_descriptor("cuda_tiled_vector",
                                                  "shared_tile16_linear_cta_float4"));
    variants.push_back(optimized_dense_descriptor("cuda_combined",
                                                  "shared_tile16_direct_2d_scalar"));
    variants.push_back(optimized_dense_v2_descriptor("cuda_register_tiled_v2_sync",
                                                     "shared_tile32_register_4x2_sync"));
    variants.push_back(optimized_dense_v2_descriptor("cuda_register_tiled_v2_async",
                                                     "shared_tile32_register_4x2_cp_async_2stage"));
    variants.push_back(optimized_dense_v2_descriptor("cuda_register_tiled_v3_64x32_async",
                                                     "shared_tile64x32_register_8x2_cp_async_2stage"));
#if RAGGEDROUTE_HAS_CUBLAS
    variants.push_back(descriptor("cublaslt", "nvidia_cuda_library", "cublasLtMatmul.v1",
                                  cuda_library_revision(), "cublaslt_heuristic_0"));
    variants.push_back(descriptor("cublas", "nvidia_cuda_library", "cublasSgemm.v1",
                                  cuda_library_revision(), "cublas_sgemm_pedantic"));
#endif
    return variants;
  }
  if (operator_name == "topk_gate") return {naive_descriptor("serial_row_top2")};
  if (operator_name == "histogram") {
    std::vector<VariantDescriptor> variants = {naive_descriptor("global_atomic")};
#if RAGGEDROUTE_HAS_CCCL
    variants.push_back(descriptor("cub_device_histogram", "nvidia_cccl",
                                  "cub::DeviceHistogram::HistogramEven", cccl_revision(),
                                  "histogram_even_discrete_int32"));
#endif
    return variants;
  }
  if (operator_name == "exclusive_scan") {
    std::vector<VariantDescriptor> variants = {naive_descriptor("single_thread_exclusive")};
#if RAGGEDROUTE_HAS_CCCL
    variants.push_back(descriptor("cub_device_scan", "nvidia_cccl", "cub::DeviceScan::ExclusiveSum",
                                  cccl_revision(), "device_scan_plus_terminal_offset"));
    variants.push_back(descriptor("cub_block_scan", "nvidia_cccl", "cub::BlockScan::ExclusiveSum",
                                  cccl_revision(), "block_scan_128_threads"));
    variants.push_back(descriptor("cub_warp_scan", "nvidia_cccl", "cub::WarpScan::ExclusiveSum",
                                  cccl_revision(), "warp_scan_32_threads"));
#endif
    return variants;
  }
  if (operator_name == "token_permute") {
    std::vector<VariantDescriptor> variants = {naive_descriptor("atomic_cursor_scalar_copy")};
    variants.push_back(descriptor("cuda_naive_from_ids", "in_tree_cuda",
                                  "raggedroute.cuda_naive_from_ids.v1", "not_applicable",
                                  "histogram_scan_atomic_permute"));
#if RAGGEDROUTE_HAS_CCCL
    const std::string vllm_dependency =
        "vllm@837eae64580c885101ee95b073aafb27a485e7ce; " + cccl_revision();
    variants.push_back(descriptor("vllm_moe_permute", "adapted_production_cuda",
                                  "vllm.moe_permute_with_scratch.fp32.v1", vllm_dependency,
                                  "radix_sort_scan_expand_rows"));
    variants.push_back(descriptor("vllm_expand_rows", "adapted_production_cuda",
                                  "vllm.expandInputRowsKernelLauncher.fp32.v1", vllm_dependency,
                                  "prepared_mapping_expand_rows"));
#endif
    return variants;
  }
  if (operator_name == "grouped_gemm") {
    std::vector<VariantDescriptor> variants = {naive_descriptor("grid_z_per_expert")};
#if RAGGEDROUTE_HAS_CUBLAS
    variants.push_back(descriptor("cublas_per_expert", "nvidia_cuda_library",
                                  "cublasSgemm.per_active_expert.v1", cuda_library_revision(),
                                  "host_loop_active_experts"));
#endif
#if RAGGEDROUTE_HAS_CUTLASS
    variants.push_back(descriptor("cutlass_grouped", "nvidia_cutlass",
                                  "cutlass::gemm::device::GemmGrouped.fp32.v1", cutlass_revision(),
                                  "device_scheduled_simt_fp32"));
#endif
    return variants;
  }
  if (operator_name == "unpermute") {
    std::vector<VariantDescriptor> variants = {
        naive_descriptor("token_owned_scalar_gather_reduce")};
    variants.push_back(descriptor("cuda_warp_token_vec4", "in_tree_cuda",
                                  "raggedroute.unpermute.cuda_candidate.v1", "not_applicable",
                                  "shape_dispatched_warp_or_cta_top2_float4"));
    variants.push_back(descriptor("vllm_finalize_routing", "adapted_production_cuda",
                                  "vllm.finalizeMoeRoutingKernelLauncher.fp32.v1",
                                  "vllm@837eae64580c885101ee95b073aafb27a485e7ce",
                                  "token_owned_vector_finalize"));
    return variants;
  }
  return {};
}

std::vector<std::string> available_variants(const std::string& operator_name) {
  return descriptor_names(available_variant_descriptors(operator_name));
}

AdapterPtr make_adapter(const std::string& operator_name, const std::string& variant_name) {
  const auto descriptors = available_variant_descriptors(operator_name);
  if (descriptors.empty()) throw std::invalid_argument("unknown operator: " + operator_name);
  const VariantDescriptor descriptor = find_descriptor(descriptors, operator_name, variant_name);
  AdapterPtr adapter;
  if (operator_name == "dense_gemm") adapter = make_dense_gemm_adapter(variant_name);
  if (operator_name == "topk_gate") adapter = make_topk_gate_adapter(variant_name);
  if (operator_name == "histogram") adapter = make_histogram_adapter(variant_name);
  if (operator_name == "exclusive_scan") adapter = make_exclusive_scan_adapter(variant_name);
  if (operator_name == "token_permute") adapter = make_token_permute_adapter(variant_name);
  if (operator_name == "grouped_gemm") adapter = make_grouped_gemm_adapter(variant_name);
  if (operator_name == "unpermute") adapter = make_unpermute_adapter(variant_name);
  if (!adapter) throw std::logic_error("operator registry has no adapter factory");
  return std::make_unique<RegisteredAdapter>(std::move(adapter), descriptor);
}

std::vector<std::string> available_suites() { return {"chain_from_tokens", "chain_from_logits"}; }

std::vector<VariantDescriptor> available_suite_variant_descriptors(const std::string& suite_name) {
  if (suite_name == "chain_from_tokens" || suite_name == "chain_from_logits") {
    return {naive_descriptor("sequential_cuda_naive_chain"),
            descriptor("cuda_unpermute_candidate", "in_tree_cuda",
                       "raggedroute.chain.unpermute_candidate.v1", "not_applicable",
                       "naive_chain_with_warp_token_vec4_unpermute")};
  }
  return {};
}

std::vector<std::string> available_suite_variants(const std::string& suite_name) {
  return descriptor_names(available_suite_variant_descriptors(suite_name));
}

AdapterPtr make_suite_adapter(const std::string& suite_name, const std::string& variant_name) {
  const auto descriptors = available_suite_variant_descriptors(suite_name);
  if (descriptors.empty()) throw std::invalid_argument("unknown suite: " + suite_name);
  const VariantDescriptor descriptor = find_descriptor(descriptors, suite_name, variant_name);
  AdapterPtr adapter;
  if (suite_name == "chain_from_tokens") adapter = make_chain_adapter(true, variant_name);
  if (suite_name == "chain_from_logits") adapter = make_chain_adapter(false, variant_name);
  if (!adapter) throw std::logic_error("suite registry has no adapter factory");
  return std::make_unique<RegisteredAdapter>(std::move(adapter), descriptor);
}

}  // namespace raggedroute::benchmark
