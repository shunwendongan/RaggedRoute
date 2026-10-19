#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "raggedroute/types.h"

namespace raggedroute::benchmark::tensorcore_baseline {

// Benchmark-only low-precision library contracts: row-major FP16/BF16 inputs,
// FP32 accumulation/output, alpha=1 and beta=0. Plans own all scratch allocation.
struct GemmPlan;
GemmPlan* create_gemm_plan(ScalarType type, bool use_lt, int m, int n, int k,
                           std::size_t workspace_limit_bytes = 8U * 1024U * 1024U);
void destroy_gemm_plan(GemmPlan* plan) noexcept;
std::size_t gemm_workspace_bytes(const GemmPlan* plan) noexcept;
// A cuBLAS plan accepts different shapes (used for each nonempty expert).
// cuBLASLt plans require the shape selected at creation.
void launch_gemm(GemmPlan* plan, const void* a, const void* b, float* output,
                 int m, int n, int k, cudaStream_t stream);

struct GroupedCutlassPlan;
// A nullptr result means the TensorOp implementation does not support the
// storage alignment/shape. It never silently substitutes a SIMT kernel.
// Host offsets are inspected only to validate the fixture. The timed launch
// reconstructs every problem/pointer from device_offsets on the given stream.
GroupedCutlassPlan* create_grouped_cutlass_plan(
    ScalarType type, const void* x, const void* weights, float* output,
    const std::int32_t* device_offsets, const std::int32_t* host_offsets,
    int experts, int hidden, int output_width);
void destroy_grouped_cutlass_plan(GroupedCutlassPlan* plan) noexcept;
// Includes all per-expert device metadata and any CUTLASS scratch.
std::size_t grouped_cutlass_workspace_bytes(const GroupedCutlassPlan* plan) noexcept;
// The metadata preparation kernel is deliberately included in every call.
void launch_grouped_cutlass(GroupedCutlassPlan* plan, cudaStream_t stream);

}  // namespace raggedroute::benchmark::tensorcore_baseline
