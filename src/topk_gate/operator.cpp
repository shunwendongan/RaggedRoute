#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

#include "../runtime/operator_internal.h"

namespace raggedroute {

std::size_t get_topk_gate_workspace_size(const TopKGateArgs& args) noexcept {
  (void)args;
  return 0;
}

Status topk_gate(const TopKGateArgs& args, const RuntimeContext& context) noexcept {
  if (args.tokens < 0 || args.experts < 2 || args.experts > 64 || args.top_k != 2) {
    return detail::invalid_argument("topk_gate requires tokens>=0, 2<=E<=64, and top_k=2");
  }
  if (args.normalization != TopKNormalization::kSelectedSoftmax ||
      args.tie_break != TopKTieBreak::kLowerExpertId ||
      args.nan_policy != TopKNaNPolicy::kNegativeInfinityAllNaNFallback01) {
    return detail::invalid_argument("topk_gate received unsupported semantic policy");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kTopKGate, args.scalar_type,
                                            args.layout, args.kernel_variant,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  (void)decision;
  if (args.tokens == 0) return success_status();
  if (args.logits == nullptr || args.expert_ids == nullptr || args.weights == nullptr) {
    return detail::invalid_argument("topk_gate received a null device pointer");
  }

  return detail::cuda_status(
      ops::launch_topk_gate_naive(args.logits, args.expert_ids, args.weights, args.tokens,
                                  args.experts, context.stream),
      "topk_gate kernel launch failed");
}

}  // namespace raggedroute
