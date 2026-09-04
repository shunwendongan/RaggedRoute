#include "../runtime/operator_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

// grouped GEMM 的公开入口。
// 这里负责校验 routed-row 的布局和尺寸、根据架构和 kernel 做派发，
// 最后启动对应实现去计算 x_permuted[R,H] x weights[E,H,O] -> y_permuted[R,O]。
namespace raggedroute {

// grouped GEMM 不需要额外的 workspace。
std::size_t get_grouped_gemm_workspace_size(const GroupedGemmArgs& args) noexcept {
  (void)args;
  return 0;
}

// 校验 grouped GEMM 的 routed-row 布局和 expert 参数，并派发到对应 kernel。
Status grouped_gemm(const GroupedGemmArgs& args, const RuntimeContext& context) noexcept {
  if (args.experts <= 0 || args.experts > 64 || args.hidden < 0 || args.output < 0 ||
      args.max_expert_tokens < 0) {
    return detail::invalid_argument("grouped_gemm requires 1<=E<=64 and non-negative dimensions");
  }
  DispatchDecision decision;
  Status status = detail::dispatch_operator(
      OperatorKind::kGroupedGemm,
      detail::make_compute_signature(args.x_permuted, args.expert_weights, args.accumulator_dtype,
                                     args.y_permuted),
      args.kernel, context.architecture, &decision);
  if (!status.ok()) return status;
  status = detail::require_naive_implementation(decision);
  if (!status.ok()) return status;
  if (args.max_expert_tokens == 0 || args.output == 0) return success_status();
  if (args.offsets == nullptr || args.y_permuted.data == nullptr ||
      (args.hidden != 0 &&
       (args.x_permuted.data == nullptr || args.expert_weights.data == nullptr))) {
    return detail::invalid_argument("grouped_gemm received a null device pointer");
  }
  if (!detail::is_aligned(args.x_permuted.data, alignof(float)) ||
      !detail::is_aligned(args.expert_weights.data, alignof(float)) ||
      !detail::is_aligned(args.y_permuted.data, alignof(float))) {
    return detail::invalid_argument("grouped_gemm FP32 buffers must be 4-byte aligned");
  }

  return detail::cuda_status(ops::launch_grouped_gemm_naive(
                                 static_cast<const float*>(args.x_permuted.data),
                                 static_cast<const float*>(args.expert_weights.data), args.offsets,
                                 static_cast<float*>(args.y_permuted.data), args.experts,
                                 args.hidden, args.output, args.max_expert_tokens, context.stream),
                             "grouped_gemm kernel launch failed");
}

}  // namespace raggedroute
