/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * 改写自 NVIDIA CUTLASS example 24（v4.6.1）。
 * RaggedRoute 的形状契约是 x_permuted[R,H]、expert_weights[E,H,O]、
 * offsets[E+1] 和 y_permuted[R,O]，这里每个 expert slice 对应一次 GEMM。
 * 这个独立 plan 只保留 RaggedRoute 需要的 device-scheduled FP32 row-major 路径。
 **************************************************************************************************/

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

using Layout = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, Layout, cutlass::ComplexTransform::kNone, 1, float, Layout,
    cutlass::ComplexTransform::kNone, 1, float, Layout, float, cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>, Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + " failed with CUTLASS status " +
                             cutlassGetStatusString(status));
  }
}

template <typename T>
T* copy_vector_to_device(const std::vector<T>& source) {
  if (source.empty()) return nullptr;
  T* destination = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&destination), source.size() * sizeof(T)),
             "cudaMalloc(CUTLASS metadata)");
  check_cuda(
      cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
      "cudaMemcpy(CUTLASS metadata)");
  return destination;
}

}  // namespace

struct GroupedCutlassPlan {
  Gemm operation;
  std::vector<cutlass::gemm::GemmCoord> host_problems;
  cutlass::gemm::GemmCoord* problems = nullptr;
  float** a = nullptr;
  float** b = nullptr;
  float** c = nullptr;
  float** d = nullptr;
  std::int64_t* lda = nullptr;
  std::int64_t* ldb = nullptr;
  std::int64_t* ldc = nullptr;
  std::int64_t* ldd = nullptr;
  int problem_count = 0;
  int threadblock_count = 0;
  std::size_t workspace_bytes = 0;
  bool initialized = false;
};

void destroy_grouped_cutlass_plan(GroupedCutlassPlan* plan) noexcept {
  if (plan == nullptr) return;
  cudaFree(plan->problems);
  cudaFree(plan->a);
  cudaFree(plan->b);
  cudaFree(plan->c);
  cudaFree(plan->d);
  cudaFree(plan->lda);
  cudaFree(plan->ldb);
  cudaFree(plan->ldc);
  cudaFree(plan->ldd);
  delete plan;
}

GroupedCutlassPlan* create_grouped_cutlass_plan(const float* x, const float* weights, float* output,
                                                const std::int32_t* host_offsets, int experts,
                                                int hidden, int output_width) {
  if (x == nullptr || weights == nullptr || output == nullptr || host_offsets == nullptr ||
      experts < 1 || hidden < 1 || output_width < 1) {
    throw std::invalid_argument("invalid CUTLASS grouped GEMM plan");
  }
  auto* plan = new GroupedCutlassPlan;
  try {
    std::vector<float*> host_a;
    std::vector<float*> host_b;
    std::vector<float*> host_c;
    std::vector<std::int64_t> host_lda;
    std::vector<std::int64_t> host_ldb;
    std::vector<std::int64_t> host_ldc;
    for (int expert = 0; expert < experts; ++expert) {
      const int rows = host_offsets[expert + 1] - host_offsets[expert];
      if (rows == 0) continue;
      plan->host_problems.emplace_back(rows, output_width, hidden);
      host_a.push_back(const_cast<float*>(x) +
                       static_cast<std::size_t>(host_offsets[expert]) * hidden);
      host_b.push_back(const_cast<float*>(weights) +
                       static_cast<std::size_t>(expert) * hidden * output_width);
      host_c.push_back(output + static_cast<std::size_t>(host_offsets[expert]) * output_width);
      host_lda.push_back(hidden);
      host_ldb.push_back(output_width);
      host_ldc.push_back(output_width);
    }
    plan->problem_count = static_cast<int>(plan->host_problems.size());
    if (plan->problem_count == 0) return plan;
    plan->threadblock_count = Gemm::sufficient(plan->host_problems.data(), plan->problem_count);
    if (plan->threadblock_count <= 0) {
      throw std::runtime_error("CUTLASS grouped GEMM has insufficient device resources");
    }
    plan->problems = copy_vector_to_device(plan->host_problems);
    plan->a = copy_vector_to_device(host_a);
    plan->b = copy_vector_to_device(host_b);
    plan->c = copy_vector_to_device(host_c);
    plan->d = copy_vector_to_device(host_c);
    plan->lda = copy_vector_to_device(host_lda);
    plan->ldb = copy_vector_to_device(host_ldb);
    plan->ldc = copy_vector_to_device(host_ldc);
    plan->ldd = copy_vector_to_device(host_ldc);

    typename Gemm::Arguments arguments(plan->problems, plan->problem_count, plan->threadblock_count,
                                       typename Gemm::EpilogueOutputOp::Params(1.0F, 0.0F), plan->a,
                                       plan->b, plan->c, plan->d, plan->lda, plan->ldb, plan->ldc,
                                       plan->ldd, plan->host_problems.data());
    plan->workspace_bytes = plan->operation.get_workspace_size(arguments);
    return plan;
  } catch (...) {
    destroy_grouped_cutlass_plan(plan);
    throw;
  }
}

std::size_t grouped_cutlass_workspace_bytes(const GroupedCutlassPlan* plan) noexcept {
  return plan == nullptr ? 0 : plan->workspace_bytes;
}

void initialize_grouped_cutlass_plan(GroupedCutlassPlan* plan, void* workspace,
                                     std::size_t workspace_bytes, cudaStream_t stream) {
  if (plan == nullptr || workspace_bytes < plan->workspace_bytes ||
      (plan->workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument("invalid CUTLASS grouped GEMM initialization workspace");
  }
  if (plan->problem_count == 0) {
    plan->initialized = true;
    return;
  }
  typename Gemm::Arguments arguments(plan->problems, plan->problem_count, plan->threadblock_count,
                                     typename Gemm::EpilogueOutputOp::Params(1.0F, 0.0F), plan->a,
                                     plan->b, plan->c, plan->d, plan->lda, plan->ldb, plan->ldc,
                                     plan->ldd, plan->host_problems.data());
  check_cutlass(plan->operation.initialize(arguments, workspace, stream),
                "CUTLASS grouped initialize");
  plan->initialized = true;
}

void launch_grouped_cutlass(GroupedCutlassPlan* plan, void*, std::size_t, cudaStream_t stream) {
  if (plan == nullptr || !plan->initialized) {
    throw std::invalid_argument("CUTLASS grouped GEMM plan is not initialized");
  }
  if (plan->problem_count == 0) return;
  check_cutlass(plan->operation.run(stream), "CUTLASS grouped run");
}

}  // namespace raggedroute::benchmark::library_baseline
