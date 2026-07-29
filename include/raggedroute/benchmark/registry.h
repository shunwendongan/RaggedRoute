#pragma once

#include <string>
#include <vector>

#include "raggedroute/benchmark/adapter.h"

namespace raggedroute::benchmark {

std::vector<std::string> available_operators();
std::vector<std::string> available_variants(const std::string& operator_name);
AdapterPtr make_adapter(const std::string& operator_name, const std::string& variant_name);
std::vector<std::string> available_suites();
AdapterPtr make_suite_adapter(const std::string& suite_name, const std::string& variant_name);

AdapterPtr make_dense_gemm_adapter();
AdapterPtr make_topk_gate_adapter();
AdapterPtr make_histogram_adapter();
AdapterPtr make_exclusive_scan_adapter();
AdapterPtr make_token_permute_adapter();
AdapterPtr make_grouped_gemm_adapter();
AdapterPtr make_unpermute_adapter();
AdapterPtr make_chain_adapter(bool include_router_projection);

}  // namespace raggedroute::benchmark
