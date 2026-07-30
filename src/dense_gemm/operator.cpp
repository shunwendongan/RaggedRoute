#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

#include "../runtime/operator_internal.h"

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
  if (args.layout_a != TensorLayout::kRowMajorContiguous ||
      args.layout_b != TensorLayout::kRowMajorContiguous ||
      args.layout_c != TensorLayout::kRowMajorContiguous) {
    return detail::unsupported_layout("dense_gemm requires contiguous row-major A, B, and C");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kDenseGemm, args.scalar_type,
                                            args.layout_a, args.kernel_variant,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  (void)decision;
  if (args.m == 0 || args.n == 0) return success_status();
  if (args.c == nullptr || (args.k != 0 && (args.a == nullptr || args.b == nullptr))) {
    return detail::invalid_argument("dense_gemm received a null device pointer");
  }

  return detail::cuda_status(
      ops::launch_dense_gemm_naive(args.a, args.b, args.c, args.m, args.n, args.k,
                                   context.stream),
      "dense_gemm kernel launch failed");
}

}  // namespace raggedroute
