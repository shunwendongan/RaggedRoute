#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

#include <limits>

#include "../runtime/operator_internal.h"

namespace raggedroute {

std::size_t get_unpermute_workspace_size(const UnpermuteArgs& args) noexcept {
  (void)args;
  return 0;
}

Status unpermute(const UnpermuteArgs& args, const RuntimeContext& context) noexcept {
  if (args.tokens < 0 || args.top_k <= 0 || args.top_k > 64 || args.output < 0) {
    return detail::invalid_argument(
        "unpermute requires tokens>=0, 1<=top_k<=64, and output>=0");
  }
  if (args.tokens > 0 && args.tokens > std::numeric_limits<int>::max() / args.top_k) {
    return detail::invalid_argument("unpermute route count exceeds int32 indexing");
  }
  if (args.tokens > 0 && args.output > 0 &&
      args.tokens > std::numeric_limits<int>::max() / args.output) {
    return detail::invalid_argument("unpermute output count exceeds int32 indexing");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kUnpermute, args.scalar_type,
                                            args.layout, args.kernel_variant,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  (void)decision;
  if (args.tokens == 0 || args.output == 0) return success_status();
  if (args.y_permuted == nullptr || args.route_pos == nullptr || args.route_weights == nullptr ||
      args.y == nullptr) {
    return detail::invalid_argument("unpermute received a null device pointer");
  }

  return detail::cuda_status(
      ops::launch_unpermute_naive(args.y_permuted, args.route_pos, args.route_weights, args.y,
                                  args.tokens, args.top_k, args.output, context.stream),
      "unpermute kernel launch failed");
}

}  // namespace raggedroute
