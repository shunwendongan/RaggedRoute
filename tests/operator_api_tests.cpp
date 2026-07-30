#include <cuda_runtime_api.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/correctness/guarded_buffer.h"
#include "raggedroute/dispatch.h"
#include "raggedroute/operators.h"

namespace rc = raggedroute::correctness;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_status(const raggedroute::Status& status, const char* operation) {
  if (!status.ok()) {
    const char* message = status.message == nullptr ? "operator error" : status.message;
    throw std::runtime_error(std::string(operation) + ": " + message + ": " +
                             cudaGetErrorString(status.cuda_error));
  }
}

void test_pure_dispatch() {
  using namespace raggedroute;
  require(classify_compute_capability(8, 6) == DeviceArchitecture::kSm86,
          "8.6 must classify as SM86");
  require(classify_compute_capability(8, 0) == DeviceArchitecture::kOther,
          "SM80 must not silently claim SM86 support");
  require(classify_compute_capability(9, 0) == DeviceArchitecture::kOther,
          "Hopper must not silently claim P0 support");

  DispatchRequest request;
  request.operator_kind = OperatorKind::kDenseGemm;
  request.architecture = DeviceArchitecture::kSm86;
  request.requested_variant = KernelVariant::kAuto;
  request.scalar_type = ScalarType::kFloat32;
  request.layout = TensorLayout::kRowMajorContiguous;
  DispatchDecision decision;
  require_status(select_kernel(request, &decision), "SM86 auto dispatch");
  require(decision.kernel_variant == KernelVariant::kCudaNaive,
          "SM86 auto dispatch must select cuda_naive in P0");

  request.architecture = DeviceArchitecture::kOther;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedArchitecture,
          "unsupported hardware must be rejected explicitly");
  request.architecture = DeviceArchitecture::kSm86;
  request.scalar_type = ScalarType::kFloat16;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedDataType,
          "FP16 must not be claimed before its P1 implementation");
  request.scalar_type = ScalarType::kFloat32;
  request.layout = TensorLayout::kStrided;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedLayout,
          "strided tensors must be rejected by the P0 contract");
  request.layout = TensorLayout::kRowMajorContiguous;
  request.requested_variant = KernelVariant::kExperimental;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "unimplemented variants must not be selected");
  require(select_kernel(request, nullptr).code == StatusCode::kInvalidArgument,
          "null dispatch output must be rejected");
}

void test_argument_and_workspace_contracts(const raggedroute::RuntimeContext& context) {
  using namespace raggedroute;
  DenseGemmArgs invalid_dense;
  invalid_dense.m = -1;
  require(dense_gemm(invalid_dense, context).code == StatusCode::kInvalidArgument,
          "negative dense GEMM dimensions must be rejected");

  TokenPermuteArgs permute;
  permute.tokens = 1;
  permute.experts = 2;
  permute.top_k = 1;
  permute.hidden = 0;
  std::int32_t placeholder = 0;
  permute.expert_ids = &placeholder;
  permute.offsets = &placeholder;
  permute.route_pos = &placeholder;
  require(get_token_permute_workspace_size(permute) == 2 * sizeof(std::int32_t),
          "permute workspace query is wrong");
  require(token_permute(permute, context).code == StatusCode::kInsufficientWorkspace,
          "permute must reject missing caller workspace before launch");
}

void test_dense_gemm(const raggedroute::RuntimeContext& context) {
  rc::GuardedDeviceBuffer<float> a(4, context.stream);
  rc::GuardedDeviceBuffer<float> b(4, context.stream);
  rc::GuardedDeviceBuffer<float> c(4, context.stream);
  a.copy_from_host({1.0F, 2.0F, 3.0F, 4.0F}, context.stream);
  b.copy_from_host({5.0F, 6.0F, 7.0F, 8.0F}, context.stream);

  raggedroute::DenseGemmArgs args;
  args.a = a.data();
  args.b = b.data();
  args.c = c.data();
  args.m = 2;
  args.n = 2;
  args.k = 2;
  require_status(raggedroute::dense_gemm(args, context), "public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "public dense_gemm result is wrong");
  require(a.canaries_intact(context.stream) && b.canaries_intact(context.stream) &&
              c.canaries_intact(context.stream),
          "dense_gemm changed a redzone");
}

void test_histogram_reset(const raggedroute::RuntimeContext& context) {
  rc::GuardedDeviceBuffer<std::int32_t> ids(4, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> counts(2, context.stream);
  ids.copy_from_host({0, 1, 1, 1}, context.stream);
  counts.copy_from_host({99, 99}, context.stream);

  raggedroute::HistogramArgs args;
  args.expert_ids = ids.data();
  args.counts = counts.data();
  args.route_pairs = 4;
  args.experts = 2;
  require_status(raggedroute::histogram(args, context), "public histogram");
  require(counts.copy_to_host(context.stream) == std::vector<std::int32_t>({1, 3}),
          "histogram wrapper did not reset counts before atomic accumulation");
  require(ids.canaries_intact(context.stream) && counts.canaries_intact(context.stream),
          "histogram changed a redzone");
}

void test_permute_workspace_reset(const raggedroute::RuntimeContext& context) {
  rc::GuardedDeviceBuffer<float> x(4, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> ids(4, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> offsets(3, context.stream);
  rc::GuardedDeviceBuffer<float> x_permuted(8, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> route_pos(4, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> sorted_route(4, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> cursors(2, context.stream);
  x.copy_from_host({1.0F, 2.0F, 3.0F, 4.0F}, context.stream);
  ids.copy_from_host({0, 1, 0, 1}, context.stream);
  offsets.copy_from_host({0, 2, 4}, context.stream);
  cursors.copy_from_host({99, 99}, context.stream);

  raggedroute::TokenPermuteArgs args;
  args.x = x.data();
  args.expert_ids = ids.data();
  args.offsets = offsets.data();
  args.x_permuted = x_permuted.data();
  args.route_pos = route_pos.data();
  args.sorted_route = sorted_route.data();
  args.tokens = 2;
  args.experts = 2;
  args.top_k = 2;
  args.hidden = 2;
  auto permute_context = context;
  permute_context.workspace = cursors.data();
  permute_context.workspace_bytes = cursors.bytes();
  require_status(raggedroute::token_permute(args, permute_context), "public token_permute");

  const auto positions = route_pos.copy_to_host(context.stream);
  const auto output = x_permuted.copy_to_host(context.stream);
  const auto inverse = sorted_route.copy_to_host(context.stream);
  const auto final_cursors = cursors.copy_to_host(context.stream);
  require(final_cursors == std::vector<std::int32_t>({2, 2}),
          "permute wrapper did not reset cursor workspace");
  std::vector<bool> seen(4, false);
  const std::vector<std::int32_t> host_ids = {0, 1, 0, 1};
  const std::vector<std::int32_t> host_offsets = {0, 2, 4};
  const std::vector<float> host_x = {1.0F, 2.0F, 3.0F, 4.0F};
  for (int route = 0; route < 4; ++route) {
    const int position = positions[static_cast<std::size_t>(route)];
    const int expert = host_ids[static_cast<std::size_t>(route)];
    require(position >= host_offsets[static_cast<std::size_t>(expert)] &&
                position < host_offsets[static_cast<std::size_t>(expert + 1)],
            "route_pos escaped its expert segment");
    require(!seen[static_cast<std::size_t>(position)], "route_pos must be bijective");
    seen[static_cast<std::size_t>(position)] = true;
    require(inverse[static_cast<std::size_t>(position)] == route,
            "sorted_route is not the inverse of route_pos");
    const int token = route / 2;
    for (int column = 0; column < 2; ++column) {
      require(output[static_cast<std::size_t>(position) * 2 + column] ==
                  host_x[static_cast<std::size_t>(token) * 2 + column],
              "permuted row does not equal its source token row");
    }
  }
  require(x.canaries_intact(context.stream) && ids.canaries_intact(context.stream) &&
              offsets.canaries_intact(context.stream) && x_permuted.canaries_intact(context.stream) &&
              route_pos.canaries_intact(context.stream) && sorted_route.canaries_intact(context.stream) &&
              cursors.canaries_intact(context.stream),
          "permute changed a redzone");
}

}  // namespace

int main() {
  try {
    test_pure_dispatch();
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::cerr << "SKIP CUDA device unavailable\n";
      return 77;
    }
    raggedroute::DeviceArchitecture architecture = raggedroute::DeviceArchitecture::kOther;
    require_status(raggedroute::query_current_device_architecture(&architecture),
                   "query current architecture");
    if (architecture != raggedroute::DeviceArchitecture::kSm86) {
      std::cerr << "SKIP P0 operator runtime is validated for SM86 only\n";
      return 77;
    }

    cudaStream_t stream{};
    rc::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "create operator API stream");
    try {
      raggedroute::RuntimeContext context;
      context.stream = stream;
      context.architecture = architecture;
      test_argument_and_workspace_contracts(context);
      test_dense_gemm(context);
      test_histogram_reset(context);
      test_permute_workspace_reset(context);
    } catch (...) {
      cudaStreamDestroy(stream);
      throw;
    }
    rc::cuda_check(cudaStreamDestroy(stream), "destroy operator API stream");
    std::cout << "PASS operator API tests\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL operator API tests: " << error.what() << '\n';
    return 1;
  }
}
