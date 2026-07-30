#include <cuda_runtime_api.h>

#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

#include "../runtime/operator_internal.h"

namespace raggedroute {

std::size_t get_histogram_workspace_size(const HistogramArgs& args) noexcept {
  (void)args;
  return 0;
}

Status histogram(const HistogramArgs& args, const RuntimeContext& context) noexcept {
  if (args.route_pairs < 0 || args.experts <= 0 || args.experts > 64) {
    return detail::invalid_argument("histogram requires route_pairs>=0 and 1<=E<=64");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kHistogram, ScalarType::kFloat32,
                                            args.layout, args.kernel_variant,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  (void)decision;
  if (args.counts == nullptr || (args.route_pairs != 0 && args.expert_ids == nullptr)) {
    return detail::invalid_argument("histogram received a null device pointer");
  }

  // L2/L3 include this reset. The low-level launcher intentionally assumes
  // zeroed counts so that L1 can isolate the atomic kernel body.
  status = detail::cuda_status(
      cudaMemsetAsync(args.counts, 0, static_cast<std::size_t>(args.experts) * sizeof(*args.counts),
                      context.stream),
      "histogram counts reset failed");
  if (!status.ok()) return status;
  return detail::cuda_status(
      ops::launch_histogram_naive(args.expert_ids, args.counts, args.route_pairs, args.experts,
                                  context.stream),
      "histogram kernel launch failed");
}

}  // namespace raggedroute
