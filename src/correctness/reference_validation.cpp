#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {
namespace {

CheckReport report_for(const CaseDescriptor& descriptor) {
  CheckReport report;
  report.case_id = descriptor.case_id;
  report.operator_name = descriptor.operator_name;
  report.variant_name = descriptor.variant_name;
  return report;
}

bool same_class(double actual, double expected) {
  if (std::isnan(expected)) return std::isnan(actual);
  if (std::isinf(expected)) {
    return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
  }
  return std::isfinite(actual);
}

double unit_roundoff(ScalarType type) { return dtype_traits(type).epsilon_at_one * 0.5; }

}  // namespace

std::vector<double> dense_gemm_reference(const std::vector<double>& a, const std::vector<double>& b,
                                         const std::vector<double>& c, int m, int n, int k,
                                         double alpha, double beta) {
  if (m < 0 || n < 0 || k < 0) throw std::invalid_argument("negative GEMM dimension");
  const std::size_t a_size = checked_mul(static_cast<std::size_t>(m), k, "A size");
  const std::size_t b_size = checked_mul(static_cast<std::size_t>(k), n, "B size");
  const std::size_t output_size = checked_mul(static_cast<std::size_t>(m), n, "C size");
  if (a.size() != a_size || b.size() != b_size || (!c.empty() && c.size() != output_size)) {
    throw std::invalid_argument("GEMM reference buffer size mismatch");
  }
  if (beta != 0.0 && c.empty()) throw std::invalid_argument("beta requires initial C");
  std::vector<double> output(output_size, 0.0);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        sum += a[static_cast<std::size_t>(row) * k + inner] *
               b[static_cast<std::size_t>(inner) * n + column];
      }
      const std::size_t index = static_cast<std::size_t>(row) * n + column;
      output[index] = alpha * sum + (c.empty() ? 0.0 : beta * c[index]);
    }
  }
  return output;
}

Top2Reference top2_selected_softmax_reference(const std::vector<double>& logits, int tokens,
                                              int experts) {
  if (tokens < 0 || experts < 2 ||
      logits.size() != checked_mul(static_cast<std::size_t>(tokens), experts, "logits")) {
    throw std::invalid_argument("invalid Top-2 shape");
  }
  Top2Reference result;
  result.ids.resize(static_cast<std::size_t>(tokens) * 2);
  result.weights.resize(static_cast<std::size_t>(tokens) * 2);
  const auto precedes = [](double value, int id, double incumbent, int incumbent_id) {
    return incumbent_id < 0 || value > incumbent || (value == incumbent && id < incumbent_id);
  };
  for (int token = 0; token < tokens; ++token) {
    double first = -std::numeric_limits<double>::infinity();
    double second = first;
    int first_id = -1;
    int second_id = -1;
    bool saw_non_nan = false;
    for (int expert = 0; expert < experts; ++expert) {
      const double raw = logits[static_cast<std::size_t>(token) * experts + expert];
      const bool is_nan = std::isnan(raw);
      const double value = is_nan ? -std::numeric_limits<double>::infinity() : raw;
      saw_non_nan = saw_non_nan || !is_nan;
      if (precedes(value, expert, first, first_id)) {
        second = first;
        second_id = first_id;
        first = value;
        first_id = expert;
      } else if (precedes(value, expert, second, second_id)) {
        second = value;
        second_id = expert;
      }
    }
    double first_weight = 0.5;
    double second_weight = 0.5;
    if (!saw_non_nan) {
      first_id = 0;
      second_id = 1;
    } else if (first != second) {
      if (first == std::numeric_limits<double>::infinity() ||
          second == -std::numeric_limits<double>::infinity()) {
        first_weight = 1.0;
        second_weight = 0.0;
      } else {
        const double relative = std::exp(second - first);
        first_weight = 1.0 / (1.0 + relative);
        second_weight = relative * first_weight;
      }
    }
    const std::size_t base = static_cast<std::size_t>(token) * 2;
    result.ids[base] = first_id;
    result.ids[base + 1] = second_id;
    result.weights[base] = first_weight;
    result.weights[base + 1] = second_weight;
  }
  return result;
}

std::vector<std::int32_t> histogram_reference(const std::vector<std::int32_t>& ids, int experts) {
  if (experts < 1) throw std::invalid_argument("experts must be positive");
  std::vector<std::int32_t> counts(static_cast<std::size_t>(experts), 0);
  for (std::int32_t id : ids) {
    if (id < 0 || id >= experts) throw std::invalid_argument("route id out of range");
    if (counts[static_cast<std::size_t>(id)] == std::numeric_limits<std::int32_t>::max()) {
      throw std::overflow_error("histogram count overflow");
    }
    ++counts[static_cast<std::size_t>(id)];
  }
  return counts;
}

std::vector<std::int32_t> exclusive_scan_reference(const std::vector<std::int32_t>& counts) {
  std::vector<std::int32_t> offsets(counts.size() + 1, 0);
  std::int64_t total = 0;
  for (std::size_t expert = 0; expert < counts.size(); ++expert) {
    if (counts[expert] < 0) throw std::invalid_argument("negative expert count");
    offsets[expert] = checked_int32(total, "offset");
    total += counts[expert];
    checked_int32(total, "scan total");
  }
  offsets.back() = checked_int32(total, "scan total");
  return offsets;
}

std::vector<double> grouped_gemm_reference(const std::vector<double>& x,
                                           const std::vector<double>& weights,
                                           const std::vector<std::int32_t>& offsets, int experts,
                                           int hidden, int output) {
  if (experts < 1 || hidden < 0 || output < 0 ||
      offsets.size() != static_cast<std::size_t>(experts) + 1) {
    throw std::invalid_argument("invalid grouped GEMM shape");
  }
  for (int expert = 0; expert < experts; ++expert) {
    if (offsets[expert] < 0 || offsets[expert] > offsets[expert + 1]) {
      throw std::invalid_argument("non-monotonic grouped offsets");
    }
  }
  const std::size_t routes = static_cast<std::size_t>(offsets.back());
  if (x.size() != checked_mul(routes, hidden, "grouped X") ||
      weights.size() !=
          checked_mul(checked_mul(experts, hidden, "grouped weights"), output, "grouped weights")) {
    throw std::invalid_argument("grouped GEMM buffer size mismatch");
  }
  std::vector<double> y(checked_mul(routes, output, "grouped output"), 0.0);
  for (int expert = 0; expert < experts; ++expert) {
    for (int row = offsets[expert]; row < offsets[expert + 1]; ++row) {
      for (int column = 0; column < output; ++column) {
        for (int inner = 0; inner < hidden; ++inner) {
          y[static_cast<std::size_t>(row) * output + column] +=
              x[static_cast<std::size_t>(row) * hidden + inner] *
              weights[(static_cast<std::size_t>(expert) * hidden + inner) * output + column];
        }
      }
    }
  }
  return y;
}

std::vector<double> unpermute_reference(const std::vector<double>& y_permuted,
                                        const std::vector<std::int32_t>& route_pos,
                                        const std::vector<double>& route_weights, int tokens,
                                        int top_k, int output) {
  if (tokens < 0 || top_k < 1 || output < 0) throw std::invalid_argument("invalid unpermute");
  const std::size_t routes = checked_mul(static_cast<std::size_t>(tokens), top_k, "routes");
  if (route_pos.size() != routes || route_weights.size() != routes ||
      y_permuted.size() != checked_mul(routes, output, "Yp")) {
    throw std::invalid_argument("unpermute buffer size mismatch");
  }
  std::vector<double> y(checked_mul(static_cast<std::size_t>(tokens), output, "Y"), 0.0);
  for (int token = 0; token < tokens; ++token) {
    for (int column = 0; column < output; ++column) {
      for (int rank = 0; rank < top_k; ++rank) {
        const std::size_t route = static_cast<std::size_t>(token) * top_k + rank;
        const int source = route_pos[route];
        if (source < 0 || static_cast<std::size_t>(source) >= routes) {
          throw std::invalid_argument("route_pos out of range");
        }
        y[static_cast<std::size_t>(token) * output + column] +=
            route_weights[route] * y_permuted[static_cast<std::size_t>(source) * output + column];
      }
    }
  }
  return y;
}

CheckReport compare_floating(const CaseDescriptor& descriptor, const std::vector<double>& actual,
                             const std::vector<double>& expected, double atol, double rtol) {
  CheckReport report = report_for(descriptor);
  if (actual.size() != expected.size()) {
    report.fail({"shape", "floating output size mismatch"});
    return report;
  }
  long double error_squared = 0.0L;
  long double reference_squared = 0.0L;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!same_class(actual[index], expected[index])) {
      report.fail({"numeric_class", "NaN/Inf/finite class mismatch", index, actual[index],
                   expected[index], 0.0});
      continue;
    }
    if (!std::isfinite(expected[index])) continue;
    const double absolute = std::abs(actual[index] - expected[index]);
    const double denominator = std::max(std::abs(expected[index]), 1.0e-30);
    const double relative = absolute / denominator;
    const double allowed = atol + rtol * denominator;
    if (absolute > report.numeric.max_abs_error) {
      report.numeric.max_abs_error = absolute;
      report.numeric.worst_index = index;
    }
    report.numeric.max_rel_error = std::max(report.numeric.max_rel_error, relative);
    error_squared += static_cast<long double>(absolute) * absolute;
    reference_squared += static_cast<long double>(expected[index]) * expected[index];
    if (absolute > allowed) {
      report.fail({"numeric", "floating value exceeds tolerance", index, actual[index],
                   expected[index], allowed});
    }
  }
  report.numeric.normalized_l2_error =
      reference_squared == 0.0L ? static_cast<double>(std::sqrt(error_squared))
                                : static_cast<double>(std::sqrt(error_squared / reference_squared));
  return report;
}

CheckReport compare_gemm(const CaseDescriptor& descriptor, const std::vector<double>& actual,
                         const std::vector<double>& expected, const std::vector<double>& a,
                         const std::vector<double>& b, int m, int n, int k,
                         ScalarType accumulator) {
  CheckReport report = report_for(descriptor);
  if (actual.size() != expected.size() || expected.size() != static_cast<std::size_t>(m) * n) {
    report.fail({"shape", "GEMM output size mismatch"});
    return report;
  }
  const double u = unit_roundoff(accumulator);
  const double ku = static_cast<double>(k) * u;
  const double gamma = ku < 1.0 ? ku / (1.0 - ku) : std::numeric_limits<double>::infinity();
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      const std::size_t index = static_cast<std::size_t>(row) * n + column;
      if (!same_class(actual[index], expected[index])) {
        report.fail({"numeric_class", "GEMM value class mismatch", index, actual[index],
                     expected[index], 0.0});
        continue;
      }
      if (!std::isfinite(expected[index])) continue;
      double sum_abs = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        sum_abs += std::abs(a[static_cast<std::size_t>(row) * k + inner] *
                            b[static_cast<std::size_t>(inner) * n + column]);
      }
      const double output_rounding =
          descriptor.tensor_types.output == ScalarType::kFp32
              ? 0.0
              : unit_roundoff(descriptor.tensor_types.output) * std::abs(expected[index]);
      const double allowed = 4.0 * gamma * sum_abs + output_rounding + 1.0e-7;
      const double absolute = std::abs(actual[index] - expected[index]);
      if (absolute > report.numeric.max_abs_error) {
        report.numeric.max_abs_error = absolute;
        report.numeric.worst_index = index;
      }
      report.numeric.max_rel_error = std::max(
          report.numeric.max_rel_error, absolute / std::max(std::abs(expected[index]), 1.0e-30));
      if (absolute > allowed) {
        report.fail({"gemm_forward_error", "GEMM exceeds accumulation-aware bound", index,
                     actual[index], expected[index], allowed});
      }
    }
  }
  return report;
}

CheckReport validate_histogram(const CaseDescriptor& descriptor,
                               const std::vector<std::int32_t>& ids,
                               const std::vector<std::int32_t>& counts, int experts) {
  CheckReport report = report_for(descriptor);
  const auto expected = histogram_reference(ids, experts);
  if (counts.size() != expected.size()) {
    report.fail({"shape", "histogram count size mismatch"});
    return report;
  }
  std::int64_t total = 0;
  for (std::size_t expert = 0; expert < counts.size(); ++expert) {
    if (counts[expert] != expected[expert]) {
      report.fail({"exact_counts", "histogram count differs", expert,
                   static_cast<double>(counts[expert]), static_cast<double>(expected[expert]),
                   0.0});
    }
    if (counts[expert] < 0) report.fail({"non_negative", "negative histogram count", expert});
    total += counts[expert];
  }
  if (total != static_cast<std::int64_t>(ids.size())) {
    report.fail({"count_sum", "sum(counts) differs from route count"});
  }
  return report;
}

CheckReport validate_scan(const CaseDescriptor& descriptor, const std::vector<std::int32_t>& counts,
                          const std::vector<std::int32_t>& offsets) {
  CheckReport report = report_for(descriptor);
  const auto expected = exclusive_scan_reference(counts);
  if (offsets.size() != expected.size()) {
    report.fail({"shape", "scan offset size mismatch"});
    return report;
  }
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    if (offsets[index] != expected[index]) {
      report.fail({"exact_offsets", "exclusive offset differs", index,
                   static_cast<double>(offsets[index]), static_cast<double>(expected[index]), 0.0});
    }
    if (index != 0 && offsets[index] < offsets[index - 1]) {
      report.fail({"monotonic", "exclusive offsets decrease", index});
    }
  }
  return report;
}

CheckReport validate_permute(const CaseDescriptor& descriptor, const std::vector<double>& x,
                             const std::vector<std::int32_t>& ids,
                             const std::vector<std::int32_t>& offsets,
                             const std::vector<double>& xp,
                             const std::vector<std::int32_t>& route_pos,
                             const std::vector<std::int32_t>* sorted_route, int tokens, int top_k,
                             int hidden) {
  CheckReport report = report_for(descriptor);
  const std::size_t routes = static_cast<std::size_t>(tokens) * top_k;
  if (ids.size() != routes || route_pos.size() != routes ||
      x.size() != static_cast<std::size_t>(tokens) * hidden || xp.size() != routes * hidden ||
      offsets.size() < 2 || (sorted_route != nullptr && sorted_route->size() != routes)) {
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
