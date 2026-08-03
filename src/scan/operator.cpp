#include "../runtime/operator_internal.h"
#include "cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

namespace raggedroute {

std::size_t get_exclusive_scan_workspace_size(const ExclusiveScanArgs& args) noexcept {
  (void)args;
  return 0;
}

Status exclusive_scan(const ExclusiveScanArgs& args, const RuntimeContext& context) noexcept {
  if (args.experts <= 0 || args.experts > 64) {
    return detail::invalid_argument("exclusive_scan requires 1<=E<=64");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kExclusiveScan, {}, args.kernel,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  if (args.counts == nullptr || args.offsets == nullptr) {
    return detail::invalid_argument("exclusive_scan received a null device pointer");
  }
  if (!detail::is_aligned(args.counts, alignof(std::int32_t)) ||
      !detail::is_aligned(args.offsets, alignof(std::int32_t))) {
    return detail::invalid_argument("exclusive_scan buffers must be 4-byte aligned");
  }
  if (decision.kernel.family == KernelFamily::kCudaNaive) {
    return detail::cuda_status(
        ops::launch_exclusive_scan_naive(args.counts, args.offsets, args.experts, context.stream),
        "exclusive_scan naive kernel launch failed");
  }
  if (decision.kernel.family == KernelFamily::kCudaOptimized) {
    return detail::cuda_status(
        ops::launch_exclusive_scan_optimized(args.counts, args.offsets, args.experts,
                                             decision.kernel.implementation_id, context.stream),
        "exclusive_scan optimized kernel launch failed");
  }
  return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                             "exclusive_scan dispatch selected an unknown kernel");
}

}  // namespace raggedroute
