#include <cublasLt.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

}  // namespace

struct DenseCublasLtPlan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t workspace_bytes = 0;
};

struct DenseCublasPlan {
  cublasHandle_t handle = nullptr;
};

void destroy_dense_cublaslt_plan(DenseCublasLtPlan* plan) noexcept {
  if (plan == nullptr) return;
  if (plan->preference != nullptr) cublasLtMatmulPreferenceDestroy(plan->preference);
  if (plan->c_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->c_layout);
  if (plan->b_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->b_layout);
  if (plan->a_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->a_layout);
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->handle != nullptr) cublasLtDestroy(plan->handle);
  delete plan;
}

DenseCublasLtPlan* create_dense_cublaslt_plan(int m, int n, int k,
                                              std::size_t workspace_limit_bytes,
                                              std::size_t* workspace_bytes) {
  if (m < 0 || n < 0 || k < 0 || workspace_bytes == nullptr) {
    throw std::invalid_argument("invalid cuBLASLt GEMM plan shape");
  }
  auto* plan = new DenseCublasLtPlan;
  try {
    check(cublasLtCreate(&plan->handle), "cublasLtCreate");
    check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
    check(cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_32F, m, k, k),
          "cublasLtMatrixLayoutCreate(A)");
    check(cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_32F, k, n, n),
          "cublasLtMatrixLayoutCreate(B)");
    check(cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_32F, m, n, n),
          "cublasLtMatrixLayoutCreate(C)");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check(cublasLtMatrixLayoutSetAttribute(plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "cublasLtMatrixLayoutSetAttribute(A order)");
    check(cublasLtMatrixLayoutSetAttribute(plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "cublasLtMatrixLayoutSetAttribute(B order)");
    check(cublasLtMatrixLayoutSetAttribute(plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "cublasLtMatrixLayoutSetAttribute(C order)");
    check(cublasLtMatmulPreferenceCreate(&plan->preference), "cublasLtMatmulPreferenceCreate");
    check(cublasLtMatmulPreferenceSetAttribute(
              plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit_bytes,
              sizeof(workspace_limit_bytes)),
          "cublasLtMatmulPreferenceSetAttribute");
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    check(cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation, plan->a_layout,
                                         plan->b_layout, plan->c_layout, plan->c_layout,
                                         plan->preference, 1, &heuristic, &returned),
          "cublasLtMatmulAlgoGetHeuristic");
    if (returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLASLt found no strict-FP32 row-major algorithm");
    }
    plan->algorithm = heuristic.algo;
    plan->workspace_bytes = heuristic.workspaceSize;
    *workspace_bytes = plan->workspace_bytes;
    return plan;
  } catch (...) {
    destroy_dense_cublaslt_plan(plan);
    throw;
  }
}

void launch_dense_cublaslt(DenseCublasLtPlan* plan, const float* a, const float* b, float* c,
                           void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  if (plan == nullptr || a == nullptr || b == nullptr || c == nullptr ||
      workspace_bytes < plan->workspace_bytes ||
      (plan->workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument("invalid cuBLASLt GEMM launch");
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check(cublasLtMatmul(plan->handle, plan->operation, &alpha, a, plan->a_layout, b, plan->b_layout,
                       &beta, c, plan->c_layout, c, plan->c_layout, &plan->algorithm, workspace,
                       workspace_bytes, stream),
        "cublasLtMatmul");
}

DenseCublasPlan* create_dense_cublas_plan() {
  auto* plan = new DenseCublasPlan;
  try {
    check(cublasCreate(&plan->handle), "cublasCreate");
    check(cublasSetMathMode(plan->handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    return plan;
  } catch (...) {
    destroy_dense_cublas_plan(plan);
    throw;
  }
}

void destroy_dense_cublas_plan(DenseCublasPlan* plan) noexcept {
  if (plan == nullptr) return;
  if (plan->handle != nullptr) cublasDestroy(plan->handle);
  delete plan;
}

void launch_dense_cublas(DenseCublasPlan* plan, const float* a, const float* b, float* c, int m,
                         int n, int k, cudaStream_t stream) {
  if (plan == nullptr || a == nullptr || b == nullptr || c == nullptr) {
    throw std::invalid_argument("invalid cuBLAS GEMM launch");
  }
  check(cublasSetStream(plan->handle, stream), "cublasSetStream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check(
      cublasSgemm(plan->handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n),
      "cublasSgemm");
}

}  // namespace raggedroute::benchmark::library_baseline
