#pragma once

#include <string>
#include <vector>

#include "raggedroute/benchmark/registry.h"

namespace raggedroute::benchmark {

// Separate benchmark operators keep the strict-FP32 public adapters unchanged.
AdapterPtr make_tensorcore_gemm_adapter(const std::string& operator_name,
                                      const std::string& variant_name);
std::vector<VariantDescriptor> tensorcore_gemm_variant_descriptors(
    const std::string& operator_name);

}  // namespace raggedroute::benchmark
