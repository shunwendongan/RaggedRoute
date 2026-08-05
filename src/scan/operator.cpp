#include "../runtime/operator_internal.h"
#include "cuda_candidate/optimized_internal.h"
#include "raggedroute/baseline_ops.h"
#include "raggedroute/operators.h"

#include <cstddef>
#include <cstdint>

namespace raggedroute {
namespace {

bool ranges_overlap(const void* left, std::size_t left_bytes, const void* right,
                    std::size_t right_bytes) noexcept {
  if (left_bytes == 0 || right_bytes == 0) return false;
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
  return left_begin < right_begin + right_bytes && right_begin < left_begin + left_bytes;
}

Status launch_histogram_scan_separate(const HistogramExclusiveScanArgs& args,
                                      const RuntimeContext& context) noexcept {
  HistogramArgs histogram_args;
  histogram_args.expert_ids = args.expert_ids;
  histogram_args.counts = args.counts;
  histogram_args.route_pairs = args.route_pairs;
  histogram_args.experts = args.experts;
  histogram_args.kernel = {};
  Status status = histogram(histogram_args, context);
  if (!status.ok()) return status;

  ExclusiveScanArgs scan_args;
  scan_args.counts = args.counts;
  scan_args.offsets = args.offsets;
  scan_args.experts = args.experts;
  scan_args.kernel = {};
  return exclusive_scan(scan_args, context);
}

}  // namespace

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
  return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                             "exclusive_scan dispatch selected an unknown kernel");
}

std::size_t get_histogram_exclusive_scan_workspace_size(
    const HistogramExclusiveScanArgs& args) noexcept {
  (void)args;
  return 0;
}

Status histogram_exclusive_scan(const HistogramExclusiveScanArgs& args,
                                const RuntimeContext& context) noexcept {
  if (args.route_pairs < 0 || args.experts <= 0 || args.experts > 64) {
    return detail::invalid_argument(
        "histogram_exclusive_scan requires route_pairs>=0 and 1<=E<=64");
  }
  if (args.counts == nullptr || args.offsets == nullptr ||
      (args.route_pairs != 0 && args.expert_ids == nullptr)) {
    return detail::invalid_argument("histogram_exclusive_scan received a null device pointer");
  }
  if (!detail::is_aligned(args.expert_ids, alignof(std::int32_t)) ||
      !detail::is_aligned(args.counts, alignof(std::int32_t)) ||
      !detail::is_aligned(args.offsets, alignof(std::int32_t))) {
    return detail::invalid_argument("histogram_exclusive_scan buffers must be 4-byte aligned");
  }

  const std::size_t ids_bytes =
      static_cast<std::size_t>(args.route_pairs) * sizeof(std::int32_t);
  const std::size_t counts_bytes = static_cast<std::size_t>(args.experts) * sizeof(std::int32_t);
  const std::size_t offsets_bytes =
      static_cast<std::size_t>(args.experts + 1) * sizeof(std::int32_t);
  if (ranges_overlap(args.expert_ids, ids_bytes, args.counts, counts_bytes) ||
      ranges_overlap(args.expert_ids, ids_bytes, args.offsets, offsets_bytes) ||
      ranges_overlap(args.counts, counts_bytes, args.offsets, offsets_bytes)) {
    return detail::invalid_argument(
        "histogram_exclusive_scan input and outputs must be pairwise non-overlapping");
  }

  DispatchDecision decision;
  Status status = detail::dispatch_operator(OperatorKind::kHistogramExclusiveScan, {}, args.kernel,
                                            context.architecture, &decision);
  if (!status.ok()) return status;
  if (decision.kernel.family == KernelFamily::kCudaNaive) {
    return launch_histogram_scan_separate(args, context);
  }
  if (decision.kernel.family != KernelFamily::kCudaOptimized ||
      decision.kernel.implementation_id != ops::kHistogramExclusiveScanFusedSubwarpImplementation) {
    return detail::make_status(StatusCode::kUnsupportedKernelVariant,
                               "histogram_exclusive_scan selected an unknown kernel");
  }
  if (args.route_pairs <= ops::kHistogramExclusiveScanFusedMaxRoutePairs) {
    return detail::cuda_status(
        ops::launch_histogram_exclusive_scan_optimized(
            args.expert_ids, args.counts, args.offsets, args.route_pairs, args.experts,
            decision.kernel.implementation_id, context.stream),
        "histogram_exclusive_scan fused kernel launch failed");
  }
  // F2 is a single-CTA implementation. Larger inputs use the production
  // Histogram+Scan path and keep the public workspace contract at zero.
  return launch_histogram_scan_separate(args, context);
}

}  // namespace raggedroute
