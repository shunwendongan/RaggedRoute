#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

#include "../runtime/operator_internal.h"

namespace raggedroute {

std::size_t get_grouped_gemm_workspace_size(const GroupedGemmArgs& args) noexcept {
  (void)args;
  return 0;
}

Status grouped_gemm(const GroupedGemmArgs& args, const RuntimeContext& context) noexcept {
  if (args.experts <= 0 || args.experts > 64 || args.hidden < 0 || args.output < 0 ||
      args.max_expert_tokens < 0) {
    return detail::invalid_argument(
        "grouped_gemm requires 1<=E<=64 and non-negative dimensions");
  }
  if (args.layout_x != TensorLayout::kRowMajorContiguous ||
      args.layout_weights != TensorLayout::kRowMajorContiguous ||
      args.layout_y != TensorLayout::kRowMajorContiguous) {
    return detail::unsupported_layout(
        "grouped_gemm requires contiguous row-major activations, weights, and output");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kGroupedGemm, args.scalar_type,
                                            args.layout_x, args.kernel_variant,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  (void)decision;
  if (args.max_expert_tokens == 0 || args.output == 0) return success_status();
  if (args.offsets == nullptr || args.y_permuted == nullptr ||
      (args.hidden != 0 && (args.x_permuted == nullptr || args.expert_weights == nullptr))) {
    return detail::invalid_argument("grouped_gemm received a null device pointer");
  }

  return detail::cuda_status(
      ops::launch_grouped_gemm_naive(args.x_permuted, args.expert_weights, args.offsets,
                                     args.y_permuted, args.experts, args.hidden, args.output,
                                     args.max_expert_tokens, context.stream),
      "grouped_gemm kernel launch failed");
}

}  // namespace raggedroute
