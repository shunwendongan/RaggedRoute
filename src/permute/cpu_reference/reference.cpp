#include <algorithm>
#include <cmath>

#include "raggedroute/correctness/framework.h"

// token permute 的 CPU 参考实现。
// 输入是 x[T,H]、ids[T,top_k] 和 offsets[E+1]，输出为 x_permuted[T*top_k,H]
// 以及反向路径要用的 route position 信息。
namespace raggedroute::correctness {
namespace {

CheckReport report_for(const CaseDescriptor& descriptor) {
  CheckReport report;
  report.case_id = descriptor.case_id;
  report.operator_name = descriptor.operator_name;
  report.variant_name = descriptor.variant_name;
  return report;
}

}  // namespace

CheckReport validate_permute(const CaseDescriptor& descriptor, const std::vector<double>& x,
                             const std::vector<std::int32_t>& ids,
                             const std::vector<std::int32_t>& offsets,
                             const std::vector<double>& xp,
                             const std::vector<std::int32_t>& route_pos,
                             const std::vector<std::int32_t>* sorted_route, int tokens, int top_k,
                             int hidden) {
  CheckReport report = report_for(descriptor);
  if (tokens < 0 || top_k < 1 || hidden < 0) {
    report.fail({"shape", "negative or invalid permute dimension"});
    return report;
  }
  const std::size_t routes = checked_mul(static_cast<std::size_t>(tokens), top_k, "routes");
  if (ids.size() != routes || route_pos.size() != routes ||
      x.size() != checked_mul(static_cast<std::size_t>(tokens), hidden, "X") ||
      xp.size() != checked_mul(routes, hidden, "X permuted") || offsets.size() < 2 ||
      (sorted_route != nullptr && sorted_route->size() != routes)) {
    report.fail({"shape", "permute buffer size mismatch"});
    return report;
  }

  std::vector<bool> seen(routes, false);
  for (std::size_t route = 0; route < routes; ++route) {
    const int expert = ids[route];
    if (expert < 0 || static_cast<std::size_t>(expert + 1) >= offsets.size()) {
      report.fail({"id_range", "expert id out of range", route});
      continue;
    }
    const int position = route_pos[route];
    if (position < 0 || static_cast<std::size_t>(position) >= routes) {
      report.fail({"position_range", "route_pos out of range", route});
      continue;
    }
    if (position < offsets[expert] || position >= offsets[expert + 1]) {
      report.fail({"expert_segment", "route outside expert segment", route});
    }
    if (seen[static_cast<std::size_t>(position)]) {
      report.fail({"bijection", "duplicate permute destination", route});
    }
    seen[static_cast<std::size_t>(position)] = true;
    if (sorted_route != nullptr && (*sorted_route)[position] != static_cast<int>(route)) {
      report.fail({"inverse_mapping", "sorted_route is not inverse", route});
    }
    const std::size_t token = route / static_cast<std::size_t>(top_k);
    for (int column = 0; column < hidden; ++column) {
      const double actual = xp[static_cast<std::size_t>(position) * hidden + column];
      const double expected = x[token * hidden + column];
      if (actual != expected && !(std::isnan(actual) && std::isnan(expected))) {
        report.fail({"row_copy", "permuted row differs from source", route, actual, expected, 0.0});
      }
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
    report.fail({"bijection", "one or more destinations were not written"});
  }
  if (offsets.front() != 0 || offsets.back() != static_cast<int>(routes)) {
    report.fail({"offsets", "permute offsets have incorrect endpoints"});
  }
  return report;
}

}  // namespace raggedroute::correctness
