#pragma once

#include <cstdint>

namespace raggedroute::ops {

// This header is generated from the controlled L2 promotion report. It starts
// fail-closed: no shape is promoted until the release gate produces a route.
struct TopKGateAutoResolution {
  std::uint32_t implementation_id = 0;
  const char* reason = "no_promoted_shape";
};

inline TopKGateAutoResolution resolve_topk_gate_auto_policy(int tokens, int experts,
                                                            const void* logits) noexcept {
  (void)tokens;
  (void)experts;
  (void)logits;
  return {};
}

inline constexpr const char* kTopKGateAutoPolicyVersion = "controlled-l2-pending-v1";

}  // namespace raggedroute::ops
