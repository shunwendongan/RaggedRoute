#include <limits>

#include "../runtime/operator_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

namespace raggedroute {

std::size_t get_unpermute_workspace_size(const UnpermuteArgs& args) noexcept {
  (void)args;
  return 0;
}

Status unpermute(const UnpermuteArgs& args, const RuntimeContext& context) noexcept {
  if (args.tokens < 0 || args.top_k <= 0 || args.top_k > 64 || args.output < 0) {
    return detail::invalid_argument("unpermute requires tokens>=0, 1<=top_k<=64, and output>=0");
  }
  if (args.tokens > 0 && args.tokens > std::numeric_limits<int>::max() / args.top_k) {
    return detail::invalid_argument("unpermute route count exceeds int32 indexing");
  }
  if (args.tokens > 0 && args.output > 0 &&
      args.tokens > std::numeric_limits<int>::max() / args.output) {
    return detail::invalid_argument("unpermute output count exceeds int32 indexing");
  }

  DispatchDecision decision;
  Status status =
      detail::dispatch_operator(OperatorKind::kUnpermute,
                                detail::make_compute_signature(args.y_permuted, args.route_weights,
                                                               args.accumulator_dtype, args.y),
                                args.kernel, context.architecture, &decision);
  if (!status.ok()) return status;
  status = detail::require_naive_implementation(decision);
  if (!status.ok()) return status;
  if (args.tokens == 0 || args.output == 0) return success_status();
  if (args.y_permuted.data == nullptr || args.route_pos == nullptr ||
      args.route_weights.data == nullptr || args.y.data == nullptr) {
    return detail::invalid_argument("unpermute received a null device pointer");
  }
  if (!detail::is_aligned(args.y_permuted.data, alignof(float)) ||
      !detail::is_aligned(args.route_weights.data, alignof(float)) ||
      !detail::is_aligned(args.y.data, alignof(float))) {
    return detail::invalid_argument("unpermute FP32 buffers must be 4-byte aligned");
  }

  return detail::cuda_status(
      ops::launch_unpermute_naive(static_cast<const float*>(args.y_permuted.data), args.route_pos,
                                  static_cast<const float*>(args.route_weights.data),
                                  static_cast<float*>(args.y.data), args.tokens, args.top_k,
                                  args.output, context.stream),
      "unpermute kernel launch failed");
}

}  // namespace raggedroute
