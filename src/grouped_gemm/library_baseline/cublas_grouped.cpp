#include <cublas_v2.h>

#include <stdexcept>
#include <string>

#include "raggedroute/benchmark/library_baselines.h"

// grouped GEMM 的按 expert 切分 cuBLAS baseline。
// 每个 expert 取 offsets[e] 到 offsets[e+1] 的行段，把 x[rows,H] 乘上 weights[H,O]，
// 再写入 output[rows,O] 的对应切片。
namespace raggedroute::benchmark::library_baseline {
namespace {

void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

}  // namespace

struct GroupedCublasPlan {
  cublasHandle_t handle = nullptr;
};

GroupedCublasPlan* create_grouped_cublas_plan() {
  auto* plan = new GroupedCublasPlan;
  try {
    check(cublasCreate(&plan->handle), "cublasCreate(grouped)");
    check(cublasSetMathMode(plan->handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode(grouped)");
    return plan;
  } catch (...) {
    destroy_grouped_cublas_plan(plan);
    throw;
  }
}

void destroy_grouped_cublas_plan(GroupedCublasPlan* plan) noexcept {
  if (plan == nullptr) return;
  if (plan->handle != nullptr) cublasDestroy(plan->handle);
  delete plan;
}

void launch_grouped_cublas(GroupedCublasPlan* plan, const float* x, const float* weights,
                           float* output, const std::int32_t* host_offsets, int experts, int hidden,
                           int output_width, cudaStream_t stream) {
  if (plan == nullptr || x == nullptr || weights == nullptr || output == nullptr ||
      host_offsets == nullptr || experts < 1 || hidden < 0 || output_width < 0) {
    throw std::invalid_argument("invalid per-expert cuBLAS launch");
  }
  check(cublasSetStream(plan->handle, stream), "cublasSetStream(grouped)");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  for (int expert = 0; expert < experts; ++expert) {
    const int rows = host_offsets[expert + 1] - host_offsets[expert];
    if (rows == 0 || hidden == 0 || output_width == 0) continue;
    const float* expert_x = x + static_cast<std::size_t>(host_offsets[expert]) * hidden;
    const float* expert_weight = weights + static_cast<std::size_t>(expert) * hidden * output_width;
    float* expert_output = output + static_cast<std::size_t>(host_offsets[expert]) * output_width;
    check(cublasSgemm(plan->handle, CUBLAS_OP_N, CUBLAS_OP_N, output_width, rows, hidden, &alpha,
                      expert_weight, output_width, expert_x, hidden, &beta, expert_output,
                      output_width),
          "cublasSgemm(per-expert)");
  }
}

}  // namespace raggedroute::benchmark::library_baseline
