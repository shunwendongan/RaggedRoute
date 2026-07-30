#include "raggedroute/benchmark/registry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

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
  if (operator_name == "dense_gemm") return {naive_descriptor("thread_per_output")};
  if (operator_name == "topk_gate") return {naive_descriptor("serial_row_top2")};
  if (operator_name == "histogram") return {naive_descriptor("global_atomic")};
  if (operator_name == "exclusive_scan") return {naive_descriptor("single_thread_exclusive")};
  if (operator_name == "token_permute") return {naive_descriptor("atomic_cursor_scalar_copy")};
  if (operator_name == "grouped_gemm") return {naive_descriptor("grid_z_per_expert")};
  if (operator_name == "unpermute") {
    return {naive_descriptor("token_owned_scalar_gather_reduce")};
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
    return {naive_descriptor("sequential_cuda_naive_chain")};
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
