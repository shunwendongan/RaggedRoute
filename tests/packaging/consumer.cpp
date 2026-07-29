#include "raggedroute/baseline_ops.h"
#include "raggedroute/correctness/framework.h"

int main() {
  auto* dense_gemm = &raggedroute::ops::launch_dense_gemm_naive;
  const auto traits =
      raggedroute::correctness::dtype_traits(raggedroute::correctness::ScalarType::kFp16);
  return dense_gemm == nullptr || traits.logical_bits != 16;
}
