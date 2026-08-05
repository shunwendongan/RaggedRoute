#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../src/runtime/operator_internal.h"
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

raggedroute::OperatorSignature fp32_signature(raggedroute::OperatorKind kind) {
  using namespace raggedroute;
  OperatorSignature signature;
  switch (kind) {
    case OperatorKind::kDenseGemm:
    case OperatorKind::kGroupedGemm:
    case OperatorKind::kUnpermute:
      signature.input = TensorSpec{};
      signature.weight = TensorSpec{};
      signature.accumulator = ScalarType::kFp32;
      signature.output = TensorSpec{};
      break;
    case OperatorKind::kTopKGate:
      signature.input = TensorSpec{};
      signature.accumulator = ScalarType::kFp32;
      signature.output = TensorSpec{};
      break;
    case OperatorKind::kTokenPermute:
      signature.input = TensorSpec{};
      signature.output = TensorSpec{};
      break;
    case OperatorKind::kHistogram:
    case OperatorKind::kExclusiveScan:
    case OperatorKind::kHistogramExclusiveScan:
      break;
  }
  return signature;
}

void test_pure_dispatch() {
  using namespace raggedroute;
  require(classify_compute_capability(8, 6) == DeviceArchitecture::kSm86,
          "8.6 must classify as SM86");
  require(classify_compute_capability(8, 0) == DeviceArchitecture::kOther,
          "SM80 must not silently claim SM86 support");
  require(classify_compute_capability(9, 0) == DeviceArchitecture::kSm90,
          "Hopper must classify as SM90 without implying runtime support");

  const std::vector<OperatorKind> operators = {
      OperatorKind::kDenseGemm,     OperatorKind::kTopKGate,     OperatorKind::kHistogram,
      OperatorKind::kExclusiveScan, OperatorKind::kTokenPermute, OperatorKind::kGroupedGemm,
      OperatorKind::kUnpermute, OperatorKind::kHistogramExclusiveScan};
  for (const OperatorKind kind : operators) {
    DispatchRequest request;
    request.operator_kind = kind;
    request.architecture = DeviceArchitecture::kSm86;
    request.signature = fp32_signature(kind);
    DispatchDecision decision;
    require_status(select_kernel(request, &decision), "SM86 FP32 auto dispatch");
    if (kind == OperatorKind::kHistogram) {
      require(decision.kernel.family == KernelFamily::kCudaOptimized &&
                  decision.kernel.implementation_id == 101,
              "SM86 Histogram auto dispatch must select the promoted candidate");
    } else if (kind == OperatorKind::kHistogramExclusiveScan) {
      require(decision.kernel.family == KernelFamily::kCudaOptimized &&
                  decision.kernel.implementation_id == 2,
              "SM86 fused Histogram-Scan auto dispatch must select F2");
    } else {
      require(decision.kernel.family == KernelFamily::kCudaNaive &&
                  decision.kernel.implementation_id == 0,
              "non-Histogram SM86 auto dispatch must select cuda_naive implementation zero");
    }
  }

  DispatchRequest request;
  request.operator_kind = OperatorKind::kDenseGemm;
  request.architecture = DeviceArchitecture::kSm86;
  request.signature = fp32_signature(request.operator_kind);
  DispatchDecision decision;
  request.architecture = DeviceArchitecture::kOther;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedArchitecture,
          "unsupported hardware must be rejected explicitly");
  request.architecture = DeviceArchitecture::kSm90;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedArchitecture,
          "SM90 classification must not imply unvalidated runtime support");
  request.architecture = DeviceArchitecture::kSm86;
  request.signature.input->dtype = ScalarType::kFp16;
  request.signature.weight->dtype = ScalarType::kFp16;
  request.signature.output->dtype = ScalarType::kFp16;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedDataType,
          "a valid FP16 signature must remain unsupported before its kernel exists");
  request.signature.input->dtype = ScalarType::kBf16;
  request.signature.weight->dtype = ScalarType::kBf16;
  request.signature.output->dtype = ScalarType::kBf16;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedDataType,
          "a valid BF16 signature must remain unsupported before its kernel exists");
  request.signature = fp32_signature(request.operator_kind);
  request.signature.input->layout = TensorLayout::kStrided;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedLayout,
          "strided tensors must be rejected by the v0.2 contract");
  request.signature.input->layout = TensorLayout::kRowMajorContiguous;
  request.signature.input->strides[0] = 1;
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedLayout,
          "explicit strides must be rejected until strided kernels exist");
  request.signature = fp32_signature(request.operator_kind);
  request.signature.weight.reset();
  require(select_kernel(request, &decision).code == StatusCode::kInvalidArgument,
          "missing dtype roles must be rejected as malformed dispatch requests");
  request.signature = fp32_signature(request.operator_kind);
  request.requested_kernel = {KernelFamily::kCudaOptimized, 0};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "the provisional optimized default must remain unavailable before promotion");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 1};
  require_status(select_kernel(request, &decision), "explicit tiled dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 1,
          "dense GEMM tiled experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 2};
  require_status(select_kernel(request, &decision), "explicit 2D dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 2,
          "dense GEMM 2D experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 3};
  require_status(select_kernel(request, &decision), "explicit vector tiled dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 3,
          "dense GEMM vector experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 4};
  require_status(select_kernel(request, &decision), "explicit combined dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 4,
          "combined dense GEMM experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 5};
  require_status(select_kernel(request, &decision), "explicit v2 sync dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 5,
          "v2 sync dense GEMM experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 6};
  require_status(select_kernel(request, &decision), "explicit v2 async dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 6,
          "v2 async dense GEMM experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 7};
  require_status(select_kernel(request, &decision), "explicit v3 async dense GEMM dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 7,
          "v3 async dense GEMM experiment must preserve its explicit implementation id");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 8};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "unimplemented optimized dense GEMM ids must be rejected");
  request.operator_kind = OperatorKind::kTopKGate;
  request.signature = fp32_signature(request.operator_kind);
  request.requested_kernel = {KernelFamily::kCudaOptimized, 0};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "Top-K optimized id zero must remain unavailable before measured promotion");
  for (const std::uint32_t implementation : {1U, 2U, 3U, 4U}) {
    request.requested_kernel = {KernelFamily::kCudaOptimized, implementation};
    require_status(select_kernel(request, &decision), "explicit Top-K candidate dispatch");
    require(decision.kernel.family == KernelFamily::kCudaOptimized &&
                decision.kernel.implementation_id == implementation,
            "Top-K candidate dispatch must preserve its explicit implementation id");
  }
  request.requested_kernel = {KernelFamily::kCudaOptimized, 5};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "unknown Top-K optimized ids must be rejected");
  request.operator_kind = OperatorKind::kGroupedGemm;
  request.signature = fp32_signature(request.operator_kind);
  for (const std::uint32_t implementation : {1U, 2U, 3U, 4U, 5U, 6U}) {
    request.requested_kernel = {KernelFamily::kCudaOptimized, implementation};
    require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
            "benchmark-only grouped GEMM candidates must not be runtime-dispatchable");
  }
  request.requested_kernel = {KernelFamily::kCudaOptimized, 101};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "Histogram implementation ids must be rejected for dense GEMM");

  request.operator_kind = OperatorKind::kHistogram;
  request.signature = fp32_signature(request.operator_kind);
  request.requested_kernel = {KernelFamily::kCudaOptimized, 101};
  require_status(select_kernel(request, &decision), "explicit histogram candidate dispatch");
  require(decision.kernel.family == KernelFamily::kCudaOptimized &&
              decision.kernel.implementation_id == 101,
          "Histogram dispatch must preserve its operator-local implementation id");
  request.requested_kernel = {KernelFamily::kCudaNaive, 0};
  require_status(select_kernel(request, &decision), "explicit naive histogram dispatch");
  require(
      decision.kernel.family == KernelFamily::kCudaNaive && decision.kernel.implementation_id == 0,
      "explicit Histogram cuda_naive must remain available after promotion");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 1};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "dense GEMM implementation ids must be rejected for Histogram");

  request.operator_kind = OperatorKind::kTokenPermute;
  request.signature = fp32_signature(request.operator_kind);
  for (std::uint32_t implementation = 1; implementation <= 8; ++implementation) {
    request.requested_kernel = {KernelFamily::kCudaOptimized, implementation};
    require_status(select_kernel(request, &decision), "explicit optimized token permute dispatch");
    require(decision.kernel.family == KernelFamily::kCudaOptimized &&
                decision.kernel.implementation_id == implementation,
            "token permute must preserve its explicit implementation id");
  }
  request.requested_kernel = {KernelFamily::kCudaOptimized, 9};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "unimplemented optimized token permute ids must be rejected");

  request.operator_kind = OperatorKind::kUnpermute;
  request.signature = fp32_signature(request.operator_kind);
  request.requested_kernel = {KernelFamily::kCudaOptimized, 1};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "research-only unpermute candidate must not enter public dispatch");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 0};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "unpermute optimized default must remain unavailable");
  request.requested_kernel = {KernelFamily::kCudaOptimized, 2};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "unknown optimized unpermute ids must be rejected");
  request.requested_kernel = {KernelFamily::kCudaNaive, 1};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "the naive family must reject unknown implementation ids");
  request.requested_kernel = {KernelFamily::kAuto, 1};
  require(select_kernel(request, &decision).code == StatusCode::kUnsupportedKernelVariant,
          "automatic dispatch must reject non-zero implementation ids");
  request.requested_kernel = {};
  require(select_kernel(request, nullptr).code == StatusCode::kInvalidArgument,
          "null dispatch output must be rejected");

  DispatchDecision wrapper_decision;
  wrapper_decision.kernel = {KernelFamily::kCudaOptimized, 0};
  require(detail::require_naive_implementation(wrapper_decision).code ==
              StatusCode::kUnsupportedKernelVariant,
          "a wrapper must fail closed when dispatch selects an unimplemented family");
  wrapper_decision.kernel = {KernelFamily::kCudaNaive, 7};
  require(detail::require_naive_implementation(wrapper_decision).code ==
              StatusCode::kUnsupportedKernelVariant,
          "a wrapper must fail closed when dispatch selects an unimplemented id");
  wrapper_decision.kernel = {KernelFamily::kCudaNaive, 0};
  require(detail::require_naive_implementation(wrapper_decision).ok(),
          "a wrapper must accept the implemented cuda_naive default");
}

void test_argument_and_workspace_contracts(const raggedroute::RuntimeContext& context) {
  using namespace raggedroute;
  DenseGemmArgs invalid_dense;
  invalid_dense.m = -1;
  require(dense_gemm(invalid_dense, context).code == StatusCode::kInvalidArgument,
          "negative dense GEMM dimensions must be rejected");

  DenseGemmArgs future_dense;
  future_dense.a.spec.dtype = ScalarType::kFp16;
  future_dense.b.spec.dtype = ScalarType::kFp16;
  future_dense.c.spec.dtype = ScalarType::kFp16;
  require(dense_gemm(future_dense, context).code == StatusCode::kUnsupportedDataType,
          "the public wrapper must not reinterpret an FP16 signature as FP32");
  future_dense = {};
  future_dense.a.spec.strides[0] = 1;
  require(dense_gemm(future_dense, context).code == StatusCode::kUnsupportedLayout,
          "the public wrapper must reject explicit strides before a strided kernel exists");
  future_dense = {};
  future_dense.m = 1;
  future_dense.n = 1;
  alignas(float) unsigned char misaligned_storage[sizeof(float) + 1]{};
  future_dense.c.data = misaligned_storage + 1;
  require(dense_gemm(future_dense, context).code == StatusCode::kInvalidArgument,
          "the public wrapper must reject misaligned FP32 payloads before launch");

  TokenPermuteArgs mismatched_permute;
  mismatched_permute.experts = 2;
  mismatched_permute.top_k = 1;
  mismatched_permute.x.spec.dtype = ScalarType::kFp16;
  mismatched_permute.x_permuted.spec.dtype = ScalarType::kBf16;
  require(token_permute(mismatched_permute, context).code == StatusCode::kUnsupportedDataType,
          "permute input and output storage dtypes must match");

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

void test_exclusive_scan(const raggedroute::RuntimeContext& context) {
  using raggedroute::ExclusiveScanArgs;
  using raggedroute::KernelFamily;
  using raggedroute::KernelSelection;
  using raggedroute::StatusCode;

  for (const int experts : {1, 31, 32, 33, 63, 64}) {
    std::vector<std::int32_t> counts(static_cast<std::size_t>(experts));
    std::vector<std::int32_t> expected(static_cast<std::size_t>(experts + 1), 0);
    for (int index = 0; index < experts; ++index) {
      counts[static_cast<std::size_t>(index)] = index % 5;
      expected[static_cast<std::size_t>(index + 1)] =
          expected[static_cast<std::size_t>(index)] + counts[static_cast<std::size_t>(index)];
    }
    rc::GuardedDeviceBuffer<std::int32_t> device_counts(counts.size(), context.stream);
    rc::GuardedDeviceBuffer<std::int32_t> device_offsets(expected.size(), context.stream);
    device_counts.copy_from_host(counts, context.stream);
    ExclusiveScanArgs args;
    args.counts = device_counts.data();
    args.offsets = device_offsets.data();
    args.experts = experts;
    require(raggedroute::get_exclusive_scan_workspace_size(args) == 0,
            "exclusive_scan must remain workspace-free");
    require_status(raggedroute::exclusive_scan(args, context), "default exclusive_scan");
    require(device_offsets.copy_to_host(context.stream) == expected,
            "default exclusive_scan produced incorrect offsets");
    require(device_counts.canaries_intact(context.stream) &&
                device_offsets.canaries_intact(context.stream),
            "default exclusive_scan changed a redzone");

    std::vector<std::int32_t> in_place_input = counts;
    in_place_input.push_back(-1);
    rc::GuardedDeviceBuffer<std::int32_t> in_place(in_place_input.size(), context.stream);
    in_place.copy_from_host(in_place_input, context.stream);
    args.counts = in_place.data();
    args.offsets = in_place.data();
    for (const KernelSelection kernel : {KernelSelection{KernelFamily::kCudaNaive, 0}}) {
      in_place.copy_from_host(in_place_input, context.stream);
      args.kernel = kernel;
      require_status(raggedroute::exclusive_scan(args, context), "in-place exclusive_scan");
      require(in_place.copy_to_host(context.stream) == expected,
              "in-place exclusive_scan produced incorrect offsets");
    }
    require(in_place.canaries_intact(context.stream), "in-place exclusive_scan changed a redzone");
  }

  constexpr int kFourByteAlignedExperts = 33;
  std::vector<std::int32_t> four_byte_counts(kFourByteAlignedExperts + 2, 0);
  std::vector<std::int32_t> four_byte_offsets(kFourByteAlignedExperts + 3, -1);
  std::vector<std::int32_t> four_byte_expected(kFourByteAlignedExperts + 1, 0);
  for (int index = 0; index < kFourByteAlignedExperts; ++index) {
    four_byte_counts[static_cast<std::size_t>(index + 1)] = index % 7;
    four_byte_expected[static_cast<std::size_t>(index + 1)] =
        four_byte_expected[static_cast<std::size_t>(index)] + index % 7;
  }
  rc::GuardedDeviceBuffer<std::int32_t> four_byte_counts_device(four_byte_counts.size(),
                                                                context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> four_byte_offsets_device(four_byte_offsets.size(),
                                                                 context.stream);
  four_byte_counts_device.copy_from_host(four_byte_counts, context.stream);
  four_byte_offsets_device.copy_from_host(four_byte_offsets, context.stream);
  ExclusiveScanArgs four_byte_args;
  four_byte_args.counts = four_byte_counts_device.data() + 1;
  four_byte_args.offsets = four_byte_offsets_device.data() + 1;
  four_byte_args.experts = kFourByteAlignedExperts;
  require(reinterpret_cast<std::uintptr_t>(four_byte_args.counts) % 8 == 4 &&
              reinterpret_cast<std::uintptr_t>(four_byte_args.offsets) % 8 == 4,
          "test setup must exercise pointers that are 4-byte but not 8-byte aligned");
  require_status(raggedroute::exclusive_scan(four_byte_args, context),
                 "4-byte-aligned exclusive_scan");
  const std::vector<std::int32_t> four_byte_actual =
      four_byte_offsets_device.copy_to_host(context.stream);
  require(std::vector<std::int32_t>(four_byte_actual.begin() + 1,
                                    four_byte_actual.begin() + kFourByteAlignedExperts + 2) ==
              four_byte_expected,
          "4-byte-aligned exclusive_scan produced incorrect offsets");
  require(four_byte_actual.front() == -1 && four_byte_actual.back() == -1,
          "4-byte-aligned exclusive_scan wrote outside its output span");

  std::vector<std::int32_t> max_safe_counts(64, 0);
  max_safe_counts.front() = std::numeric_limits<std::int32_t>::max();
  std::vector<std::int32_t> max_safe_expected(65, std::numeric_limits<std::int32_t>::max());
  max_safe_expected.front() = 0;
  rc::GuardedDeviceBuffer<std::int32_t> max_safe_device(max_safe_expected.size(), context.stream);
  max_safe_counts.push_back(-1);
  max_safe_device.copy_from_host(max_safe_counts, context.stream);
  ExclusiveScanArgs max_safe_args;
  max_safe_args.counts = max_safe_device.data();
  max_safe_args.offsets = max_safe_device.data();
  max_safe_args.experts = 64;
  require_status(raggedroute::exclusive_scan(max_safe_args, context),
                 "maximum-safe in-place exclusive_scan");
  require(max_safe_device.copy_to_host(context.stream) == max_safe_expected,
          "maximum-safe in-place exclusive_scan produced incorrect offsets");
  require(max_safe_device.canaries_intact(context.stream),
          "maximum-safe in-place exclusive_scan changed a redzone");

  rc::GuardedDeviceBuffer<std::int32_t> aligned_storage(66, context.stream);

  ExclusiveScanArgs invalid;
  invalid.experts = 1;
  require(raggedroute::exclusive_scan(invalid, context).code == StatusCode::kInvalidArgument,
          "exclusive_scan must reject null pointers");
  invalid.counts = aligned_storage.data();
  invalid.offsets = aligned_storage.data();
  invalid.experts = 65;
  require(raggedroute::exclusive_scan(invalid, context).code == StatusCode::kInvalidArgument,
          "exclusive_scan must reject E>64");
  invalid.experts = 0;
  require(raggedroute::exclusive_scan(invalid, context).code == StatusCode::kInvalidArgument,
          "exclusive_scan must reject E=0");
  invalid.experts = 1;
  invalid.kernel = {KernelFamily::kCudaOptimized, 1};
  require(
      raggedroute::exclusive_scan(invalid, context).code == StatusCode::kUnsupportedKernelVariant,
      "exclusive_scan must reject unpromoted optimized implementation ids");

  alignas(std::int32_t) unsigned char misaligned_storage[2 * sizeof(std::int32_t) + 1]{};
  invalid.kernel = {};
  invalid.counts = reinterpret_cast<const std::int32_t*>(misaligned_storage + 1);
  invalid.offsets = reinterpret_cast<std::int32_t*>(misaligned_storage + 1);
  require(raggedroute::exclusive_scan(invalid, context).code == StatusCode::kInvalidArgument,
          "exclusive_scan must reject pointers that are not 4-byte aligned");
}

void test_histogram_exclusive_scan(const raggedroute::RuntimeContext& context) {
  using raggedroute::HistogramExclusiveScanArgs;
  using raggedroute::KernelFamily;
  using raggedroute::KernelSelection;
  using raggedroute::StatusCode;

  for (const auto [route_pairs, experts] :
       {std::pair{0, 1}, std::pair{1, 8}, std::pair{257, 33}, std::pair{4096, 64},
        std::pair{4097, 64}}) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(route_pairs));
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(experts), 0);
    for (int route = 0; route < route_pairs; ++route) {
      ids[static_cast<std::size_t>(route)] = route % experts;
      ++expected_counts[static_cast<std::size_t>(route % experts)];
    }
    std::vector<std::int32_t> expected_offsets(static_cast<std::size_t>(experts + 1), 0);
    for (int expert = 0; expert < experts; ++expert) {
      expected_offsets[static_cast<std::size_t>(expert + 1)] =
          expected_offsets[static_cast<std::size_t>(expert)] +
          expected_counts[static_cast<std::size_t>(expert)];
    }

    rc::GuardedDeviceBuffer<std::int32_t> device_ids(ids.size(), context.stream);
    rc::GuardedDeviceBuffer<std::int32_t> device_counts(expected_counts.size(), context.stream);
    rc::GuardedDeviceBuffer<std::int32_t> device_offsets(expected_offsets.size(), context.stream);
    if (!ids.empty()) device_ids.copy_from_host(ids, context.stream);

    HistogramExclusiveScanArgs args;
    args.expert_ids = ids.empty() ? nullptr : device_ids.data();
    args.counts = device_counts.data();
    args.offsets = device_offsets.data();
    args.route_pairs = route_pairs;
    args.experts = experts;
    require(raggedroute::get_histogram_exclusive_scan_workspace_size(args) == 0,
            "fused metadata API must remain workspace-free");

    std::vector<KernelSelection> kernels = {{KernelFamily::kCudaNaive, 0},
                                            {KernelFamily::kCudaOptimized, 2}};
    for (const KernelSelection kernel : kernels) {
      args.kernel = kernel;
      require_status(raggedroute::histogram_exclusive_scan(args, context),
                     "histogram_exclusive_scan");
      require(device_counts.copy_to_host(context.stream) == expected_counts,
              "histogram_exclusive_scan produced incorrect counts");
      require(device_offsets.copy_to_host(context.stream) == expected_offsets,
              "histogram_exclusive_scan produced incorrect offsets");
    }
    require(device_counts.canaries_intact(context.stream) &&
                device_offsets.canaries_intact(context.stream),
            "histogram_exclusive_scan changed a redzone");
  }

  rc::GuardedDeviceBuffer<std::int32_t> storage(66, context.stream);
  HistogramExclusiveScanArgs invalid;
  invalid.expert_ids = storage.data();
  invalid.counts = storage.data();
  invalid.offsets = storage.data();
  invalid.route_pairs = 1;
  invalid.experts = 1;
  require(raggedroute::histogram_exclusive_scan(invalid, context).code ==
              StatusCode::kInvalidArgument,
          "histogram_exclusive_scan must reject overlapping outputs");
  invalid.counts = nullptr;
  require(raggedroute::histogram_exclusive_scan(invalid, context).code ==
              StatusCode::kInvalidArgument,
          "histogram_exclusive_scan must reject null outputs");
  invalid.counts = storage.data() + 2;
  invalid.offsets = storage.data() + 4;
  invalid.kernel = {KernelFamily::kCudaOptimized, 99};
  require(raggedroute::histogram_exclusive_scan(invalid, context).code ==
              StatusCode::kUnsupportedKernelVariant,
          "histogram_exclusive_scan must reject unknown implementation ids");
}

void test_dense_gemm(const raggedroute::RuntimeContext& context) {
  rc::GuardedDeviceBuffer<float> a(4, context.stream);
  rc::GuardedDeviceBuffer<float> b(4, context.stream);
  rc::GuardedDeviceBuffer<float> c(4, context.stream);
  a.copy_from_host({1.0F, 2.0F, 3.0F, 4.0F}, context.stream);
  b.copy_from_host({5.0F, 6.0F, 7.0F, 8.0F}, context.stream);

  raggedroute::DenseGemmArgs args;
  args.a.data = a.data();
  args.b.data = b.data();
  args.c.data = c.data();
  args.m = 2;
  args.n = 2;
  args.k = 2;
  require_status(raggedroute::dense_gemm(args, context), "public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 1};
  require_status(raggedroute::dense_gemm(args, context), "explicit tiled public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit tiled public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 2};
  require_status(raggedroute::dense_gemm(args, context), "explicit 2D public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit 2D public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 3};
  require_status(raggedroute::dense_gemm(args, context), "explicit vector tiled public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit vector tiled public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 4};
  require_status(raggedroute::dense_gemm(args, context), "explicit combined public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit combined public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 5};
  require_status(raggedroute::dense_gemm(args, context), "explicit v2 sync public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit v2 sync public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 6};
  require_status(raggedroute::dense_gemm(args, context), "explicit v2 async public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit v2 async public dense_gemm result is wrong");
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 7};
  require_status(raggedroute::dense_gemm(args, context), "explicit v3 async public dense_gemm");
  require(c.copy_to_host(context.stream) == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}),
          "explicit v3 async public dense_gemm result is wrong");
  args.a.data = nullptr;
  args.b.data = nullptr;
  args.k = 0;
  for (const std::uint32_t implementation : {1U, 5U, 6U, 7U}) {
    args.kernel = {raggedroute::KernelFamily::kCudaOptimized, implementation};
    require_status(raggedroute::dense_gemm(args, context), "explicit optimized K=0 dense_gemm");
    require(c.copy_to_host(context.stream) == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F}),
            "explicit optimized K=0 dense_gemm must write zero outputs");
  }
  require(a.canaries_intact(context.stream) && b.canaries_intact(context.stream) &&
              c.canaries_intact(context.stream),
          "dense_gemm changed a redzone");
}

void test_dense_gemm_vector_alignment_fallback(const raggedroute::RuntimeContext& context) {
  constexpr int kM = 17;
  constexpr int kN = 19;
  constexpr int kK = 13;
  rc::GuardedDeviceBuffer<float> a(static_cast<std::size_t>(kM) * kK + 1, context.stream);
  rc::GuardedDeviceBuffer<float> b(static_cast<std::size_t>(kK) * kN + 1, context.stream);
  rc::GuardedDeviceBuffer<float> c(static_cast<std::size_t>(kM) * kN + 1, context.stream);
  std::vector<float> a_host(a.size(), -1.0F);
  std::vector<float> b_host(b.size(), -1.0F);
  for (int index = 0; index < kM * kK; ++index) {
    a_host[static_cast<std::size_t>(index + 1)] = static_cast<float>((index % 11) - 5) * 0.25F;
  }
  for (int index = 0; index < kK * kN; ++index) {
    b_host[static_cast<std::size_t>(index + 1)] = static_cast<float>((index % 13) - 6) * 0.125F;
  }
  a.copy_from_host(a_host, context.stream);
  b.copy_from_host(b_host, context.stream);
  require(reinterpret_cast<std::uintptr_t>(a.data() + 1) % alignof(float) == 0 &&
              reinterpret_cast<std::uintptr_t>(a.data() + 1) % alignof(float4) != 0,
          "vector fallback input must be only 4-byte aligned");

  raggedroute::DenseGemmArgs args;
  args.a.data = a.data() + 1;
  args.b.data = b.data() + 1;
  args.c.data = c.data() + 1;
  args.m = kM;
  args.n = kN;
  args.k = kK;
  for (const std::uint32_t implementation : {3U, 5U, 6U, 7U}) {
    args.kernel = {raggedroute::KernelFamily::kCudaOptimized, implementation};
    require_status(raggedroute::dense_gemm(args, context),
                   "unaligned vectorized dense_gemm fallback");
    const auto output = c.copy_to_host(context.stream);
    for (int row = 0; row < kM; ++row) {
      for (int column = 0; column < kN; ++column) {
        float expected = 0.0F;
        for (int inner = 0; inner < kK; ++inner) {
          expected += a_host[static_cast<std::size_t>(1 + row * kK + inner)] *
                      b_host[static_cast<std::size_t>(1 + inner * kN + column)];
        }
        require(std::fabs(output[static_cast<std::size_t>(1 + row * kN + column)] - expected) <=
                    2.0e-5F * kK,
                "unaligned vectorized dense_gemm fallback produced the wrong output");
      }
    }
  }
  require(a.canaries_intact(context.stream) && b.canaries_intact(context.stream) &&
              c.canaries_intact(context.stream),
          "unaligned vector dense_gemm fallback changed a redzone");
}

void test_topk_gate(const raggedroute::RuntimeContext& context) {
  constexpr int kTokens = 3;
  constexpr int kExperts = 4;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  rc::GuardedDeviceBuffer<float> logits(kTokens * kExperts, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> ids(kTokens * 2, context.stream);
  rc::GuardedDeviceBuffer<float> weights(kTokens * 2, context.stream);
  logits.copy_from_host({1.0F, 3.0F, 3.0F, 2.0F, nan, nan, nan, nan,
                         std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(), 0.0F},
                        context.stream);

  raggedroute::TopKGateArgs args;
  args.logits.data = logits.data();
  args.expert_ids = ids.data();
  args.weights.data = weights.data();
  args.tokens = kTokens;
  args.experts = kExperts;
  for (const std::uint32_t implementation : {0U, 1U, 2U, 3U, 4U}) {
    args.kernel = implementation == 0
                      ? raggedroute::KernelSelection{}
                      : raggedroute::KernelSelection{raggedroute::KernelFamily::kCudaOptimized,
                                                     implementation};
    require_status(raggedroute::topk_gate(args, context), "public topk_gate candidate");
    require(ids.copy_to_host(context.stream) == std::vector<std::int32_t>({1, 2, 0, 1, 0, 1}),
            "public topk_gate ids violate deterministic semantics");
    require(weights.copy_to_host(context.stream) == std::vector<float>({0.5F, 0.5F, 0.5F,
                                                                         0.5F, 0.5F, 0.5F}),
            "public topk_gate weights violate selected-softmax semantics");
  }
  require(logits.canaries_intact(context.stream) && ids.canaries_intact(context.stream) &&
              weights.canaries_intact(context.stream),
          "topk_gate changed a redzone");
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

  counts.copy_from_host({77, 88}, context.stream);
  args.expert_ids = nullptr;
  args.route_pairs = 0;
  require_status(raggedroute::histogram(args, context), "zero-route public histogram");
  require(counts.copy_to_host(context.stream) == std::vector<std::int32_t>({0, 0}),
          "zero-route histogram did not clear all counts");
  require(ids.canaries_intact(context.stream) && counts.canaries_intact(context.stream),
          "histogram changed a redzone");

  ids.copy_from_host({0, 1, 1, 1}, context.stream);
  counts.copy_from_host({55, 66}, context.stream);
  args.expert_ids = ids.data();
  args.route_pairs = 4;
  args.kernel = {raggedroute::KernelFamily::kCudaOptimized, 101};
  require_status(raggedroute::histogram(args, context), "small histogram candidate");
  require(counts.copy_to_host(context.stream) == std::vector<std::int32_t>({1, 3}),
          "small histogram candidate did not overwrite old counts correctly");
  require(ids.copy_to_host(context.stream) == std::vector<std::int32_t>({0, 1, 1, 1}),
          "small histogram candidate changed its input");
  require(ids.canaries_intact(context.stream) && counts.canaries_intact(context.stream),
          "small histogram candidate changed a redzone");

  args.expert_ids = nullptr;
  args.route_pairs = 0;
  counts.copy_from_host({22, 11}, context.stream);
  require_status(raggedroute::histogram(args, context), "zero-route histogram candidate");
  require(counts.copy_to_host(context.stream) == std::vector<std::int32_t>({0, 0}),
          "zero-route histogram candidate did not clear counts");
  require(ids.canaries_intact(context.stream) && counts.canaries_intact(context.stream),
          "zero-route histogram candidate changed a redzone");

  for (const int route_pairs : {8192, 32768}) {
    rc::GuardedDeviceBuffer<std::int32_t> candidate_ids(static_cast<std::size_t>(route_pairs),
                                                        context.stream);
    rc::GuardedDeviceBuffer<std::int32_t> candidate_counts(2, context.stream);
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(route_pairs));
    for (int route = 0; route < route_pairs; ++route) host_ids[route] = route & 1;
    candidate_ids.copy_from_host(host_ids, context.stream);
    candidate_counts.copy_from_host({9, 9}, context.stream);
    args.expert_ids = candidate_ids.data();
    args.counts = candidate_counts.data();
    args.route_pairs = route_pairs;
    require_status(raggedroute::histogram(args, context), route_pairs == 8192
                                                              ? "candidate naive fallback"
                                                              : "candidate block-private path");
    require(candidate_counts.copy_to_host(context.stream) ==
                std::vector<std::int32_t>({route_pairs / 2, route_pairs / 2}),
            "histogram candidate dispatch path produced incorrect counts");
    require(candidate_ids.copy_to_host(context.stream) == host_ids,
            "histogram candidate dispatch path changed its input");
    require(candidate_ids.canaries_intact(context.stream) &&
                candidate_counts.canaries_intact(context.stream),
            "histogram candidate dispatch path changed a redzone");
  }
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
  args.x.data = x.data();
  args.expert_ids = ids.data();
  args.offsets = offsets.data();
  args.x_permuted.data = x_permuted.data();
  args.route_pos = route_pos.data();
  args.sorted_route = sorted_route.data();
  args.tokens = 2;
  args.experts = 2;
  args.top_k = 2;
  args.hidden = 2;
  auto permute_context = context;
  permute_context.workspace = cursors.data();
  permute_context.workspace_bytes = cursors.bytes();
  const std::vector<std::int32_t> host_ids = {0, 1, 0, 1};
  const std::vector<std::int32_t> host_offsets = {0, 2, 4};
  const std::vector<float> host_x = {1.0F, 2.0F, 3.0F, 4.0F};

  for (std::uint32_t implementation = 0; implementation <= 8; ++implementation) {
    cursors.copy_from_host({99, 99}, context.stream);
    args.kernel = implementation == 0
                      ? raggedroute::KernelSelection{}
                      : raggedroute::KernelSelection{
                            raggedroute::KernelFamily::kCudaOptimized, implementation};
    require_status(raggedroute::token_permute(args, permute_context), "public token_permute");

    const auto positions = route_pos.copy_to_host(context.stream);
    const auto output = x_permuted.copy_to_host(context.stream);
    const auto inverse = sorted_route.copy_to_host(context.stream);
    const auto final_cursors = cursors.copy_to_host(context.stream);
    require(final_cursors == std::vector<std::int32_t>({2, 2}),
            "permute wrapper did not reset cursor workspace");
    std::vector<bool> seen(4, false);
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
  }
  require(
      x.canaries_intact(context.stream) && ids.canaries_intact(context.stream) &&
          offsets.canaries_intact(context.stream) && x_permuted.canaries_intact(context.stream) &&
          route_pos.canaries_intact(context.stream) &&
          sorted_route.canaries_intact(context.stream) && cursors.canaries_intact(context.stream),
      "permute changed a redzone");
}

void test_permute_optimized_unaligned_duplicate_top2(
    const raggedroute::RuntimeContext& context) {
  constexpr int kTokens = 2;
  constexpr int kTopK = 2;
  constexpr int kHidden = 4;
  rc::GuardedDeviceBuffer<float> x(kTokens * kHidden + 1, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> ids(kTokens * kTopK, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> offsets(3, context.stream);
  rc::GuardedDeviceBuffer<float> output(kTokens * kTopK * kHidden + 1, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> route_pos(kTokens * kTopK, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> sorted_route(kTokens * kTopK, context.stream);
  rc::GuardedDeviceBuffer<std::int32_t> cursors(2, context.stream);
  x.copy_from_host({-1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F},
                   context.stream);
  ids.copy_from_host({0, 0, 1, 1}, context.stream);
  offsets.copy_from_host({0, 2, 4}, context.stream);
  require(reinterpret_cast<std::uintptr_t>(x.data() + 1) % alignof(float4) != 0 &&
              reinterpret_cast<std::uintptr_t>(output.data() + 1) % alignof(float4) != 0,
          "permute scalar fallback buffers must be intentionally unaligned");

  raggedroute::TokenPermuteArgs args;
  args.x.data = x.data() + 1;
  args.expert_ids = ids.data();
  args.offsets = offsets.data();
  args.x_permuted.data = output.data() + 1;
  args.route_pos = route_pos.data();
  args.sorted_route = sorted_route.data();
  args.tokens = kTokens;
  args.experts = 2;
  args.top_k = kTopK;
  args.hidden = kHidden;
  auto permute_context = context;
  permute_context.workspace = cursors.data();
  permute_context.workspace_bytes = cursors.bytes();

  for (std::uint32_t implementation = 1; implementation <= 8; ++implementation) {
    args.kernel = {raggedroute::KernelFamily::kCudaOptimized, implementation};
    require_status(raggedroute::token_permute(args, permute_context),
                   "unaligned duplicate-id optimized token_permute");
    const auto positions = route_pos.copy_to_host(context.stream);
    const auto inverse = sorted_route.copy_to_host(context.stream);
    const auto values = output.copy_to_host(context.stream);
    std::vector<bool> seen(4, false);
    for (int route = 0; route < 4; ++route) {
      const int position = positions[static_cast<std::size_t>(route)];
      require(position >= 0 && position < 4 && !seen[static_cast<std::size_t>(position)],
              "duplicate expert ids must still produce distinct destinations");
      seen[static_cast<std::size_t>(position)] = true;
      require(inverse[static_cast<std::size_t>(position)] == route,
              "duplicate-id inverse mapping is incorrect");
      const int token = route / kTopK;
      for (int column = 0; column < kHidden; ++column) {
        require(values[static_cast<std::size_t>(1 + position * kHidden + column)] ==
                    static_cast<float>(1 + token * kHidden + column),
                "unaligned optimized permute copied the wrong row");
      }
    }
  }
  require(x.canaries_intact(context.stream) && ids.canaries_intact(context.stream) &&
              offsets.canaries_intact(context.stream) &&
              output.canaries_intact(context.stream) &&
              route_pos.canaries_intact(context.stream) &&
              sorted_route.canaries_intact(context.stream) &&
              cursors.canaries_intact(context.stream),
          "optimized permute fallback changed a redzone");
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
      test_dense_gemm_vector_alignment_fallback(context);
      test_topk_gate(context);
      test_exclusive_scan(context);
      test_histogram_exclusive_scan(context);
      test_histogram_reset(context);
      test_permute_workspace_reset(context);
      test_permute_optimized_unaligned_duplicate_top2(context);
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
