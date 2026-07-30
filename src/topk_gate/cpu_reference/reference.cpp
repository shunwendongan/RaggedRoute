#include <cmath>
#include <limits>
#include <stdexcept>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {

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

}  // namespace raggedroute::correctness
