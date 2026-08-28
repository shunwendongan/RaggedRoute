#pragma once

#include <string>
#include <vector>

#include "raggedroute/benchmark/adapter.h"

namespace raggedroute::benchmark {

struct VariantDescriptor {
  std::string name;
  std::string implementation_category;
  std::string implementation_version;
  std::string dependency_revision;
  std::string algorithm_id;
  std::string math_mode;
};

std::vector<std::string> available_operators();
std::vector<VariantDescriptor> available_variant_descriptors(const std::string& operator_name);
std::vector<std::string> available_variants(const std::string& operator_name);
AdapterPtr make_adapter(const std::string& operator_name, const std::string& variant_name);
std::vector<std::string> available_suites();
std::vector<VariantDescriptor> available_suite_variant_descriptors(const std::string& suite_name);
std::vector<std::string> available_suite_variants(const std::string& suite_name);
AdapterPtr make_suite_adapter(const std::string& suite_name, const std::string& variant_name);

AdapterPtr make_dense_gemm_adapter(const std::string& variant_name);
AdapterPtr make_topk_gate_adapter(const std::string& variant_name);
AdapterPtr make_histogram_adapter(const std::string& variant_name);
AdapterPtr make_exclusive_scan_adapter(const std::string& variant_name);
AdapterPtr make_histogram_exclusive_scan_adapter(const std::string& variant_name);
AdapterPtr make_token_permute_adapter(const std::string& variant_name);
AdapterPtr make_grouped_gemm_adapter(const std::string& variant_name);
AdapterPtr make_unpermute_adapter(const std::string& variant_name);
AdapterPtr make_chain_adapter(bool include_router_projection, const std::string& variant_name);
AdapterPtr make_postroute_chain_adapter(const std::string& variant_name);

}  // namespace raggedroute::benchmark
