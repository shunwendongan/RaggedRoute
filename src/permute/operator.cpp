#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "../runtime/operator_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

namespace raggedroute {

std::size_t get_token_permute_workspace_size(const TokenPermuteArgs& args) noexcept {
  return args.experts > 0 ? static_cast<std::size_t>(args.experts) * sizeof(std::int32_t) : 0;
}

Status token_permute(const TokenPermuteArgs& args, const RuntimeContext& context) noexcept {
  if (args.tokens < 0 || args.experts <= 0 || args.experts > 64 || args.top_k <= 0 ||
      args.top_k > args.experts || args.hidden < 0) {
    return detail::invalid_argument("token_permute requires T>=0, 1<=top_k<=E<=64, and hidden>=0");
  }
  if (args.tokens > 0 && args.tokens > (std::numeric_limits<int>::max() / args.top_k)) {
    return detail::invalid_argument("token_permute route count exceeds int32 indexing");
  }

  OperatorSignature signature;
  signature.input = args.x.spec;
  signature.output = args.x_permuted.spec;
  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kTokenPermute, signature, args.kernel,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  status = detail::require_naive_implementation(decision);
  if (!status.ok()) return status;
  if (args.tokens == 0) return success_status();
  if (args.expert_ids == nullptr || args.offsets == nullptr || args.route_pos == nullptr ||
      (args.hidden != 0 && (args.x.data == nullptr || args.x_permuted.data == nullptr))) {
    return detail::invalid_argument("token_permute received a null device pointer");
  }
  if (!detail::is_aligned(args.x.data, alignof(float)) ||
      !detail::is_aligned(args.x_permuted.data, alignof(float))) {
    return detail::invalid_argument("token_permute FP32 buffers must be 4-byte aligned");
  }

  const std::size_t workspace_bytes = get_token_permute_workspace_size(args);
  if (context.workspace == nullptr || context.workspace_bytes < workspace_bytes) {
    return detail::make_status(StatusCode::kInsufficientWorkspace,
                               "token_permute requires int32 cursor workspace for every expert");
  }
  if (reinterpret_cast<std::uintptr_t>(context.workspace) % alignof(std::int32_t) != 0) {
    return detail::invalid_argument("token_permute workspace is not int32 aligned");
  }

  // L2/L3 include the cursor reset; L1 callers use the raw launcher and make
  // the reset an explicit precondition.
  status =
      detail::cuda_status(cudaMemsetAsync(context.workspace, 0, workspace_bytes, context.stream),
                          "token_permute cursor reset failed");
  if (!status.ok()) return status;
  return detail::cuda_status(
      ops::launch_token_permute_naive(
          static_cast<const float*>(args.x.data), args.expert_ids, args.offsets,
          static_cast<std::int32_t*>(context.workspace), static_cast<float*>(args.x_permuted.data),
          args.route_pos, args.sorted_route, args.tokens, args.top_k, args.hidden, context.stream),
      "token_permute kernel launch failed");
}

}  // namespace raggedroute
