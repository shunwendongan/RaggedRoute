#include "../runtime/operator_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

namespace raggedroute {

std::size_t get_dense_gemm_workspace_size(const DenseGemmArgs& args) noexcept {
  (void)args;
  return 0;
}

Status dense_gemm(const DenseGemmArgs& args, const RuntimeContext& context) noexcept {
  if (args.m < 0 || args.n < 0 || args.k < 0) {
    return detail::invalid_argument("dense_gemm dimensions must be non-negative");
  }
  if (args.alpha != 1.0F || args.beta != 0.0F) {
    return detail::invalid_argument("P0 dense_gemm supports alpha=1 and beta=0 only");
  }
  DispatchDecision decision;
  Status status = detail::dispatch_operator(
      OperatorKind::kDenseGemm,
      detail::make_compute_signature(args.a, args.b, args.accumulator_dtype, args.c), args.kernel,
      context.architecture, &decision);
  if (!status.ok()) return status;
  status = detail::require_naive_implementation(decision);
  if (!status.ok()) return status;
  if (args.m == 0 || args.n == 0) return success_status();
  if (args.c.data == nullptr ||
      (args.k != 0 && (args.a.data == nullptr || args.b.data == nullptr))) {
    return detail::invalid_argument("dense_gemm received a null device pointer");
  }
  if (!detail::is_aligned(args.a.data, alignof(float)) ||
      !detail::is_aligned(args.b.data, alignof(float)) ||
      !detail::is_aligned(args.c.data, alignof(float))) {
    return detail::invalid_argument("dense_gemm FP32 buffers must be 4-byte aligned");
  }

  return detail::cuda_status(ops::launch_dense_gemm_naive(static_cast<const float*>(args.a.data),
                                                          static_cast<const float*>(args.b.data),
                                                          static_cast<float*>(args.c.data), args.m,
                                                          args.n, args.k, context.stream),
                             "dense_gemm kernel launch failed");
}

}  // namespace raggedroute
