#include <stdexcept>

#include "raggedroute/correctness/framework.h"

// token unpermute 的 CPU 参考实现。
// 形状是 y_permuted[T*top_k,O]、route_pos[T*top_k] 和 route_weights[T*top_k]，
// 最终通过加权求和还原出 y[T,O]。
namespace raggedroute::correctness {

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

}  // namespace raggedroute::correctness
