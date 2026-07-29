#include "raggedroute/benchmark/registry.h"

#include <stdexcept>

namespace raggedroute::benchmark {

std::vector<std::string> available_operators() {
  return {"dense_gemm",    "topk_gate",    "histogram", "exclusive_scan",
          "token_permute", "grouped_gemm", "unpermute"};
}

std::vector<std::string> available_variants(const std::string& operator_name) {
  for (const auto& candidate : available_operators()) {
    if (operator_name == candidate) return {"cuda_naive"};
  }
  return {};
}

AdapterPtr make_adapter(const std::string& operator_name, const std::string& variant_name) {
  if (variant_name != "cuda_naive") {
    throw std::invalid_argument("unsupported variant '" + variant_name + "' for operator '" +
                                operator_name + "'");
  }
  if (operator_name == "dense_gemm") return make_dense_gemm_adapter();
  if (operator_name == "topk_gate") return make_topk_gate_adapter();
  if (operator_name == "histogram") return make_histogram_adapter();
  if (operator_name == "exclusive_scan") return make_exclusive_scan_adapter();
  if (operator_name == "token_permute") return make_token_permute_adapter();
  if (operator_name == "grouped_gemm") return make_grouped_gemm_adapter();
  if (operator_name == "unpermute") return make_unpermute_adapter();
  throw std::invalid_argument("unknown operator: " + operator_name);
}

std::vector<std::string> available_suites() { return {"chain_from_tokens", "chain_from_logits"}; }

AdapterPtr make_suite_adapter(const std::string& suite_name, const std::string& variant_name) {
  if (variant_name != "cuda_naive") {
    throw std::invalid_argument("unsupported suite variant: " + variant_name);
  }
  if (suite_name == "chain_from_tokens") return make_chain_adapter(true);
  if (suite_name == "chain_from_logits") return make_chain_adapter(false);
  throw std::invalid_argument("unknown suite: " + suite_name);
}

}  // namespace raggedroute::benchmark
