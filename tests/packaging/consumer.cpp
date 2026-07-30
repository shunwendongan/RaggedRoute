#include "raggedroute/dispatch.h"
#include "raggedroute/operators.h"
#include "raggedroute/correctness/framework.h"

int main() {
  auto* dense_gemm = &raggedroute::dense_gemm;
  raggedroute::DispatchRequest request;
  request.architecture = raggedroute::DeviceArchitecture::kSm86;
  request.scalar_type = raggedroute::ScalarType::kFloat32;
  request.layout = raggedroute::TensorLayout::kRowMajorContiguous;
  raggedroute::DispatchDecision decision;
  const auto dispatch_status = raggedroute::select_kernel(request, &decision);
  const auto traits =
      raggedroute::correctness::dtype_traits(raggedroute::correctness::ScalarType::kFp16);
  return dense_gemm == nullptr || !dispatch_status.ok() ||
         decision.kernel_variant != raggedroute::KernelVariant::kCudaNaive ||
         traits.logical_bits != 16;
}
