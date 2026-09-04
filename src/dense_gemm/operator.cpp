#include "../runtime/operator_internal.h"
#include "cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

// dense GEMM 的公开入口。
// 先校验 [M,K] x [K,N] -> [M,N] 的 FP32 契约，再做架构与 kernel 派发，
// 最后启动 naive 或 optimized 分支。
namespace raggedroute {

// dense GEMM 不需要额外的 workspace。
std::size_t get_dense_gemm_workspace_size(const DenseGemmArgs& args) noexcept {
  (void)args;
  return 0;
}

// 校验 dense GEMM 的输入形状、dtype 和派发条件，并启动对应 kernel。
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

  const float* a = static_cast<const float*>(args.a.data);
  const float* b = static_cast<const float*>(args.b.data);
  float* c = static_cast<float*>(args.c.data);
  if (decision.kernel.family == KernelFamily::kCudaNaive) {
    return detail::cuda_status(ops::launch_dense_gemm_naive(a, b, c, args.m, args.n, args.k,
                                                            context.stream),
                               "dense_gemm naive kernel launch failed");
  }
  if (decision.kernel.family == KernelFamily::kCudaOptimized) {
    return detail::cuda_status(
        ops::launch_dense_gemm_optimized(a, b, c, args.m, args.n, args.k,
                                         decision.kernel.implementation_id, context.stream),
        "dense_gemm optimized kernel launch failed");
  }
  return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                             "dense_gemm dispatch selected an unknown kernel family");
}

}  // namespace raggedroute
