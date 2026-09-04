#include "../runtime/operator_internal.h"
#include "cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

// Top-K gate 的公开入口。
// 先校验 logits[T,E]、强制 top_k=2，再派发到确定性的 Top-2 selected-softmax
// kernel family。
namespace raggedroute {

// Top-K gate 不需要额外的 workspace。
std::size_t get_topk_gate_workspace_size(const TopKGateArgs& args) noexcept {
  (void)args;
  return 0;
}

// 选择每个 token 的前 2 个 expert，并写出 ids 与 selected-softmax 权重。
Status topk_gate(const TopKGateArgs& args, const RuntimeContext& context) noexcept {
  if (args.tokens < 0 || args.experts < 2 || args.experts > 64 || args.top_k != 2) {
    return detail::invalid_argument("topk_gate requires tokens>=0, 2<=E<=64, and top_k=2");
  }
  if (args.normalization != TopKNormalization::kSelectedSoftmax ||
      args.tie_break != TopKTieBreak::kLowerExpertId ||
      args.nan_policy != TopKNaNPolicy::kNegativeInfinityAllNaNFallback01) {
    return detail::invalid_argument("topk_gate received unsupported semantic policy");
  }

  OperatorSignature signature;
  signature.input = args.logits.spec;
  signature.accumulator = args.accumulator_dtype;
  signature.output = args.weights.spec;
  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kTopKGate, signature, args.kernel,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  if (args.tokens == 0) return success_status();
  if (args.logits.data == nullptr || args.expert_ids == nullptr || args.weights.data == nullptr) {
    return detail::invalid_argument("topk_gate received a null device pointer");
  }
  if (!detail::is_aligned(args.logits.data, alignof(float)) ||
      !detail::is_aligned(args.weights.data, alignof(float))) {
    return detail::invalid_argument("topk_gate FP32 buffers must be 4-byte aligned");
  }

  cudaError_t error = cudaErrorInvalidValue;
  if (decision.kernel.family == KernelFamily::kCudaNaive) {
    error = ops::launch_topk_gate_naive(static_cast<const float*>(args.logits.data),
                                        args.expert_ids, static_cast<float*>(args.weights.data),
                                        args.tokens, args.experts, context.stream);
  } else if (decision.kernel.family == KernelFamily::kCudaOptimized) {
    error = ops::launch_topk_gate_optimized(static_cast<const float*>(args.logits.data),
                                            args.expert_ids, static_cast<float*>(args.weights.data),
                                            args.tokens, args.experts,
                                            decision.kernel.implementation_id, context.stream);
  } else {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "topk_gate dispatch selected an unsupported kernel");
  }
  return detail::cuda_status(error, "topk_gate kernel launch failed");
}

}  // namespace raggedroute
