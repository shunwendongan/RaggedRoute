#include <stdexcept>

#include "raggedroute/correctness/framework.h"

// dense GEMM 的 CPU 参考实现。
// 张量形状固定为 A[M,K]、B[K,N]、C[M,N]；保留 alpha / beta 参数，
// 方便 correctness runner 直接对照严格 FP32、行主序的运行时契约。
namespace raggedroute::correctness {

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

}  // namespace raggedroute::correctness
