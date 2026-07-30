#include <stdexcept>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {

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

}  // namespace raggedroute::correctness
