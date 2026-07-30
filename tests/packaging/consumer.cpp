#include "raggedroute/correctness/framework.h"
#include "raggedroute/dispatch.h"
#include "raggedroute/operators.h"

int main() {
  auto* dense_gemm = &raggedroute::dense_gemm;
  raggedroute::DispatchRequest request;
  request.architecture = raggedroute::DeviceArchitecture::kSm86;
  request.signature.input = raggedroute::TensorSpec{};
  request.signature.weight = raggedroute::TensorSpec{};
  request.signature.accumulator = raggedroute::ScalarType::kFp32;
  request.signature.output = raggedroute::TensorSpec{};
  raggedroute::DispatchDecision decision;
  const auto dispatch_status = raggedroute::select_kernel(request, &decision);
  const auto traits = raggedroute::correctness::dtype_traits(raggedroute::ScalarType::kFp16);
  return dense_gemm == nullptr || !dispatch_status.ok() ||
         decision.kernel.family != raggedroute::KernelFamily::kCudaNaive ||
         traits.logical_bits != 16;
}
