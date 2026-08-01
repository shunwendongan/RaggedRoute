#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace raggedroute::ops {

// These ids are DenseGemm-local research controls.  The public API exposes the
// family through KernelSelection; benchmark names remain the stable user-facing
// experiment identifiers.
constexpr std::uint32_t kDenseGemmTiledScalarImplementation = 1;
constexpr std::uint32_t kDenseGemm2dMappingImplementation = 2;
constexpr std::uint32_t kDenseGemmTiledVectorImplementation = 3;
constexpr std::uint32_t kDenseGemmCombinedImplementation = 4;

inline bool is_dense_gemm_optimized_implementation(std::uint32_t implementation_id) noexcept {
  return implementation_id == kDenseGemmTiledScalarImplementation ||
         implementation_id == kDenseGemm2dMappingImplementation ||
         implementation_id == kDenseGemmTiledVectorImplementation ||
         implementation_id == kDenseGemmCombinedImplementation;
}

cudaError_t launch_dense_gemm_optimized(const float* a, const float* b, float* c, int m, int n,
                                        int k, std::uint32_t implementation_id,
                                        cudaStream_t caller_stream);

}  // namespace raggedroute::ops
