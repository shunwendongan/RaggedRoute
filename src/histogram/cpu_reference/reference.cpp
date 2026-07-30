#include <limits>
#include <stdexcept>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {

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

}  // namespace raggedroute::correctness
