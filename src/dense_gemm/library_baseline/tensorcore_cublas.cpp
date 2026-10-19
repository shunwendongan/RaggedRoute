#include <cublasLt.h>
#include <cublas_v2.h>

#include <initializer_list>
#include <stdexcept>
#include <string>

#include "raggedroute/benchmark/tensorcore_baselines.h"

namespace raggedroute::benchmark::tensorcore_baseline {
namespace {

void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

cudaDataType_t input_type(ScalarType type) {
  if (type == ScalarType::kFp16) return CUDA_R_16F;
  if (type == ScalarType::kBf16) return CUDA_R_16BF;
  throw std::invalid_argument("Tensor Core cuBLAS requires FP16 or BF16 input");
}

}  // namespace

struct GemmPlan {
  cudaDataType_t type = CUDA_R_16F;
  bool use_lt = false;
  int m = 0;
  int n = 0;
  int k = 0;
  cublasHandle_t blas = nullptr;
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
};

void destroy_gemm_plan(GemmPlan* plan) noexcept {
  if (plan == nullptr) return;
  cudaFree(plan->workspace);
  if (plan->preference != nullptr) cublasLtMatmulPreferenceDestroy(plan->preference);
  if (plan->c_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->c_layout);
  if (plan->b_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->b_layout);
  if (plan->a_layout != nullptr) cublasLtMatrixLayoutDestroy(plan->a_layout);
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->lt != nullptr) cublasLtDestroy(plan->lt);
  if (plan->blas != nullptr) cublasDestroy(plan->blas);
  delete plan;
}

GemmPlan* create_gemm_plan(ScalarType type, bool use_lt, int m, int n, int k,
                           std::size_t workspace_limit_bytes) {
  if (m < 0 || n < 0 || k < 0) {
    throw std::invalid_argument("negative Tensor Core cuBLAS GEMM dimension");
  }
  const auto storage = input_type(type);
  auto* plan = new GemmPlan;
  plan->type = storage;
  plan->use_lt = use_lt;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  try {
    if (!use_lt) {
      check(cublasCreate(&plan->blas), "cublasCreate(Tensor Core)");
      const auto math = static_cast<cublasMath_t>(
          CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
      check(cublasSetMathMode(plan->blas, math), "cublasSetMathMode(FP32 reduction)");
      // Explicit workspace avoids allocation in the timed launches. Setting a
      // stream resets the library workspace, so launch sets it again afterwards.
      plan->workspace_bytes = workspace_limit_bytes;
    } else if (m != 0 && n != 0 && k != 0) {
      check(cublasLtCreate(&plan->lt), "cublasLtCreate(Tensor Core)");
      check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "cublasLtMatmulDescCreate(FP32 accumulate)");
      check(cublasLtMatrixLayoutCreate(&plan->a_layout, storage, m, k, k),
            "cublasLtMatrixLayoutCreate(A)");
      check(cublasLtMatrixLayoutCreate(&plan->b_layout, storage, k, n, n),
            "cublasLtMatrixLayoutCreate(B)");
      check(cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_32F, m, n, n),
            "cublasLtMatrixLayoutCreate(C)");
      const auto order = CUBLASLT_ORDER_ROW;
      for (const auto layout : {plan->a_layout, plan->b_layout, plan->c_layout}) {
        check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                               &order, sizeof(order)),
              "cublasLtMatrixLayoutSetAttribute(row major)");
      }
      check(cublasLtMatmulPreferenceCreate(&plan->preference),
            "cublasLtMatmulPreferenceCreate");
      check(cublasLtMatmulPreferenceSetAttribute(
                plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_limit_bytes, sizeof(workspace_limit_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");
      cublasLtMatmulHeuristicResult_t heuristic{};
      int returned = 0;
      const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
          plan->lt, plan->operation, plan->a_layout, plan->b_layout,
          plan->c_layout, plan->c_layout, plan->preference, 1, &heuristic, &returned);
      if (heuristic_status == CUBLAS_STATUS_NOT_SUPPORTED) {
        throw std::runtime_error("not_supported: cuBLASLt low-precision algorithm query");
      }
      check(heuristic_status, "cublasLtMatmulAlgoGetHeuristic(low precision)");
      if (returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("not_supported: cuBLASLt found no low-precision row-major algorithm");
      }
      plan->algorithm = heuristic.algo;
      plan->workspace_bytes = heuristic.workspaceSize;
    }
    if (plan->workspace_bytes != 0) {
      check_cuda(cudaMalloc(&plan->workspace, plan->workspace_bytes),
                 "cudaMalloc(Tensor Core cuBLAS workspace)");
    }
    return plan;
  } catch (...) {
    destroy_gemm_plan(plan);
    throw;
  }
}

std::size_t gemm_workspace_bytes(const GemmPlan* plan) noexcept {
  return plan == nullptr ? 0 : plan->workspace_bytes;
}

void launch_gemm(GemmPlan* plan, const void* a, const void* b, float* output,
                 int m, int n, int k, cudaStream_t stream) {
  if (plan == nullptr || m < 0 || n < 0 || k < 0 ||
      (plan->use_lt && (m != plan->m || n != plan->n || k != plan->k))) {
    throw std::invalid_argument("invalid Tensor Core cuBLAS launch shape/plan");
  }
  if (m == 0 || n == 0) return;
  if (output == nullptr || (k != 0 && (a == nullptr || b == nullptr))) {
    throw std::invalid_argument("null Tensor Core cuBLAS launch operand");
  }
  if (k == 0) {
    check_cuda(cudaMemsetAsync(output, 0, static_cast<std::size_t>(m) * n * sizeof(float), stream),
               "cudaMemsetAsync(K=0 output)");
    return;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  if (plan->use_lt) {
    check(cublasLtMatmul(plan->lt, plan->operation, &alpha, a, plan->a_layout,
                         b, plan->b_layout, &beta, output, plan->c_layout,
                         output, plan->c_layout, &plan->algorithm, plan->workspace,
                         plan->workspace_bytes, stream),
          "cublasLtMatmul(low precision)");
  } else {
    check(cublasSetStream(plan->blas, stream), "cublasSetStream");
    check(cublasSetWorkspace(plan->blas, plan->workspace, plan->workspace_bytes),
          "cublasSetWorkspace");
    // Row-major A*B is column-major B^T*A^T with dimensions exchanged.
    check(cublasGemmEx(plan->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                       &alpha, b, plan->type, n, a, plan->type, k,
                       &beta, output, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "cublasGemmEx(FP16/BF16 input, FP32 output)");
  }
}

}  // namespace raggedroute::benchmark::tensorcore_baseline
