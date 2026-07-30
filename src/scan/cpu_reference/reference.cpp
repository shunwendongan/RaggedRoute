#include <stdexcept>

#include "raggedroute/correctness/framework.h"

namespace raggedroute::correctness {

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

}  // namespace raggedroute::correctness
