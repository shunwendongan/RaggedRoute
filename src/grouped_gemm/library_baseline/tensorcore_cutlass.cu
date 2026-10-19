/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Uses the NVIDIA CUTLASS grouped GEMM interface. Device-only scheduling and
 * timed device metadata preparation preserve RaggedRoute's device-offset input.
 **************************************************************************************************/

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "raggedroute/benchmark/tensorcore_baselines.h"

namespace raggedroute::benchmark::tensorcore_baseline {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
  }
}

template <typename T>
void allocate(T** destination, int count) {
  check_cuda(cudaMalloc(reinterpret_cast<void**>(destination),
                         static_cast<std::size_t>(count) * sizeof(T)),
             "cudaMalloc(CUTLASS TensorOp metadata)");
}

// Four warps, conventional Ampere m16n8k16 Tensor Core instruction. Both inputs
// remain row major: CUTLASS supplies the required shared-memory permutation.
template <typename Element>
using TensorOpKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

template <typename Element>
__global__ void tensorcore_cutlass_prepare_metadata(
    const Element* input, const Element* weights, float* output,
    const std::int32_t* offsets, int experts, int hidden, int width,
    cutlass::gemm::GemmCoord* problems, Element** a, Element** b,
    float** c, float** d, std::int64_t* lda, std::int64_t* ldb,
    std::int64_t* ldc, std::int64_t* ldd) {
  const int expert = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (expert >= experts) return;
  const auto start = offsets[expert];
  const auto rows = offsets[expert + 1] - start;
  problems[expert] = cutlass::gemm::GemmCoord(rows, width, hidden);
  a[expert] = const_cast<Element*>(input) + static_cast<std::size_t>(start) * hidden;
  b[expert] = const_cast<Element*>(weights) +
              static_cast<std::size_t>(expert) * hidden * width;
  c[expert] = output + static_cast<std::size_t>(start) * width;
  d[expert] = c[expert];
  lda[expert] = hidden;
  ldb[expert] = width;
  ldc[expert] = width;
  ldd[expert] = width;
}

}  // namespace

struct GroupedCutlassPlan {
  virtual ~GroupedCutlassPlan() = default;
  virtual void launch(cudaStream_t stream) = 0;
  virtual std::size_t workspace_bytes() const noexcept = 0;
};

namespace {

template <typename Element>
struct TypedGroupedCutlassPlan final : GroupedCutlassPlan {
  using Gemm = cutlass::gemm::device::GemmGrouped<TensorOpKernel<Element>>;
  Gemm operation;
  const Element* input = nullptr;
  const Element* weights = nullptr;
  float* output = nullptr;
  const std::int32_t* offsets = nullptr;
  int experts = 0;
  int hidden = 0;
  int width = 0;
  int threadblocks = 0;
  cutlass::gemm::GemmCoord* problems = nullptr;
  Element** a = nullptr;
  Element** b = nullptr;
  float** c = nullptr;
  float** d = nullptr;
  std::int64_t* lda = nullptr;
  std::int64_t* ldb = nullptr;
  std::int64_t* ldc = nullptr;
  std::int64_t* ldd = nullptr;
  void* scratch = nullptr;
  std::size_t scratch_bytes = 0;

  ~TypedGroupedCutlassPlan() override {
    cudaFree(problems);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(lda);
    cudaFree(ldb);
    cudaFree(ldc);
    cudaFree(ldd);
    cudaFree(scratch);
  }

  typename Gemm::Arguments arguments() const {
    return typename Gemm::Arguments(
        problems, experts, threadblocks,
        typename Gemm::EpilogueOutputOp::Params(1.0F, 0.0F),
        a, b, c, d, lda, ldb, ldc, ldd);
  }

  void initialize(const void* x, const void* w, float* y,
                  const std::int32_t* device_offsets, int e, int k, int n) {
    input = static_cast<const Element*>(x);
    weights = static_cast<const Element*>(w);
    output = y;
    offsets = device_offsets;
    experts = e;
    hidden = k;
    width = n;
    // No host expert-size scheduling: this is occupancy only and is independent
    // of the values in offsets. All expert metadata is rebuilt in launch().
    threadblocks = Gemm::sufficient();
    if (threadblocks <= 0) {
      throw std::runtime_error("not_supported: CUTLASS TensorOp has insufficient resources");
    }
    allocate(&problems, experts);
    allocate(&a, experts);
    allocate(&b, experts);
    allocate(&c, experts);
    allocate(&d, experts);
    allocate(&lda, experts);
    allocate(&ldb, experts);
    allocate(&ldc, experts);
    allocate(&ldd, experts);
    const auto args = arguments();
    check_cutlass(Gemm::can_implement(args), "CUTLASS TensorOp can_implement");
    scratch_bytes = Gemm::get_workspace_size(args);
    if (scratch_bytes != 0) {
      check_cuda(cudaMalloc(&scratch, scratch_bytes), "cudaMalloc(CUTLASS TensorOp scratch)");
    }
    check_cutlass(operation.initialize(args, scratch), "CUTLASS TensorOp initialize");
  }

  void launch(cudaStream_t stream) override {
    tensorcore_cutlass_prepare_metadata<<<(experts + 127) / 128, 128, 0, stream>>>(
        input, weights, output, offsets, experts, hidden, width,
        problems, a, b, c, d, lda, ldb, ldc, ldd);
    check_cuda(cudaGetLastError(), "CUTLASS TensorOp prepare metadata launch");
    check_cutlass(operation.run(stream), "CUTLASS TensorOp grouped run");
  }

  std::size_t workspace_bytes() const noexcept override {
    return scratch_bytes + static_cast<std::size_t>(experts) *
        (sizeof(cutlass::gemm::GemmCoord) + 2 * sizeof(Element*) +
         2 * sizeof(float*) + 4 * sizeof(std::int64_t));
  }
};

template <typename Element>
GroupedCutlassPlan* create_typed_plan(const void* x, const void* weights,
                                     float* output, const std::int32_t* offsets,
                                     int experts, int hidden, int width) {
  auto* plan = new TypedGroupedCutlassPlan<Element>;
  try {
    plan->initialize(x, weights, output, offsets, experts, hidden, width);
    return plan;
  } catch (...) {
    delete plan;
    throw;
  }
}

}  // namespace

GroupedCutlassPlan* create_grouped_cutlass_plan(
    ScalarType type, const void* x, const void* weights, float* output,
    const std::int32_t* device_offsets, const std::int32_t* host_offsets,
    int experts, int hidden, int output_width) {
  if (type != ScalarType::kFp16 && type != ScalarType::kBf16) {
    throw std::invalid_argument("CUTLASS TensorOp requires FP16 or BF16 input");
  }
  if (experts < 1 || hidden < 0 || output_width < 0 ||
      device_offsets == nullptr || host_offsets == nullptr || host_offsets[0] != 0) {
    throw std::invalid_argument("invalid CUTLASS TensorOp grouped dimensions/offsets");
  }
  for (int expert = 0; expert < experts; ++expert) {
    if (host_offsets[expert + 1] < host_offsets[expert]) {
      throw std::invalid_argument("CUTLASS TensorOp offsets must be nondecreasing");
    }
  }
  // CUTLASS 2.x Grouped::can_implement is a permissive stub. Explicitly check
  // 128-bit operand and epilogue access alignment instead of trusting the stub.
  // K/N tails without this alignment are genuinely unsupported by this variant.
  const auto aligned = [](const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
  };
  if (hidden == 0 || output_width == 0 || hidden % 8 != 0 || output_width % 8 != 0 ||
      !aligned(x) || !aligned(weights) || !aligned(output)) {
    return nullptr;
  }
  if (weights == nullptr ||
      (host_offsets[experts] != 0 && (x == nullptr || output == nullptr))) {
    throw std::invalid_argument("null CUTLASS TensorOp grouped operand");
  }
  if (type == ScalarType::kFp16) {
    return create_typed_plan<cutlass::half_t>(x, weights, output, device_offsets,
                                            experts, hidden, output_width);
  }
  return create_typed_plan<cutlass::bfloat16_t>(x, weights, output, device_offsets,
                                              experts, hidden, output_width);
}

void destroy_grouped_cutlass_plan(GroupedCutlassPlan* plan) noexcept { delete plan; }

std::size_t grouped_cutlass_workspace_bytes(const GroupedCutlassPlan* plan) noexcept {
  return plan == nullptr ? 0 : plan->workspace_bytes();
}

void launch_grouped_cutlass(GroupedCutlassPlan* plan, cudaStream_t stream) {
  if (plan == nullptr) throw std::invalid_argument("null CUTLASS TensorOp grouped plan");
  plan->launch(stream);
}

}  // namespace raggedroute::benchmark::tensorcore_baseline
