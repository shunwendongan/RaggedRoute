#include <algorithm>
#include <cmath>
#include <limits>

#include "raggedroute/correctness/framework.h"

// 正确性结果比较与报告生成辅助函数。
// 这里把原始输出转换成结构化的 pass/fail 报告，检查 shape、数值类别和误差界。
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

}  // namespace raggedroute::correctness
