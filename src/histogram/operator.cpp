#include <cuda_runtime_api.h>

#include "../runtime/operator_internal.h"
#include "cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

// histogram 的公开入口。
// 这里负责校验 route-id 缓冲区、按需清零 counts，再把 expert_ids[route_pairs]
// -> counts[experts] 的计算派发到选中的 kernel。
namespace raggedroute {

// histogram 不需要额外的 workspace。
std::size_t get_histogram_workspace_size(const HistogramArgs& args) noexcept {
  (void)args;
  return 0;
}

// 先清零 counts，再启动 histogram kernel，把 route id 统计到各个 expert。
Status histogram(const HistogramArgs& args, const RuntimeContext& context) noexcept {
  if (args.route_pairs < 0 || args.experts <= 0 || args.experts > 64) {
    return detail::invalid_argument("histogram requires route_pairs>=0 and 1<=E<=64");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kHistogram, {}, args.kernel,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  if (args.counts == nullptr || (args.route_pairs != 0 && args.expert_ids == nullptr)) {
    return detail::invalid_argument("histogram received a null device pointer");
  }

  // L2/L3 include this reset. The low-level launcher intentionally assumes
  // zeroed counts so that L1 can isolate the atomic kernel body.
  const bool optimized = decision.kernel.family == KernelFamily::kCudaOptimized;
  if (!optimized && (decision.kernel.family != KernelFamily::kCudaNaive ||
                     decision.kernel.implementation_id != 0)) {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "histogram dispatch selected an unknown kernel");
  }
  if (!optimized || ops::histogram_optimized_requires_external_reset(
                        decision.kernel.implementation_id, args.route_pairs, args.experts)) {
    status = detail::cuda_status(
        cudaMemsetAsync(args.counts, 0,
                        static_cast<std::size_t>(args.experts) * sizeof(*args.counts),
                        context.stream),
        "histogram counts reset failed");
    if (!status.ok()) return status;
  }
  const cudaError_t launch =
      optimized ? ops::launch_histogram_optimized(args.expert_ids, args.counts, args.route_pairs,
                                                  args.experts, decision.kernel.implementation_id,
                                                  context.stream)
                : ops::launch_histogram_naive(args.expert_ids, args.counts, args.route_pairs,
                                              args.experts, context.stream);
  return detail::cuda_status(launch, "histogram kernel launch failed");
}

}  // namespace raggedroute
