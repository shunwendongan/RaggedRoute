#include "../../common/tensorcore_gemm.cuh"

namespace raggedroute::ops {
namespace {

using namespace tensorcore_detail;

template <int BM, int BN, int BK, bool Asynchronous, bool BFloat16>
__global__ __launch_bounds__(kThreads) void grouped_gemm_mma_kernel(
    const std::uint16_t* x, const std::uint16_t* weights,
    const std::int32_t* offsets, float* y, int n, int k,
    unsigned int column_tiles) {
  const int expert = static_cast<int>(blockIdx.z);
  const int begin = offsets[expert];
  const int end = offsets[expert + 1];
  const int row_base = static_cast<int>(blockIdx.x / column_tiles) * BM;
  // Device offsets are the actual row count. A conservative launch envelope
  // does not cause padded rows or empty experts to touch any payload.
  if (begin < 0 || end <= begin || row_base >= end - begin) return;
  constexpr int Stages = Asynchronous ? 2 : 1;
  __align__(16) __shared__ std::uint16_t a_shared[Stages * BM * (BK + 8)];
  __align__(16) __shared__ std::uint16_t b_shared[Stages * BK * (BN + 8)];
  const int column_base = static_cast<int>(blockIdx.x % column_tiles) * BN;
  // Do not form arithmetic on null inputs for the supported K=0 zero-fill case.
  const auto* a = k == 0 ? x : x + static_cast<std::size_t>(begin) * k;
  const auto* b = k == 0 ? weights : weights + static_cast<std::size_t>(expert) * k * n;
  compute<BM, BN, BK, Asynchronous, BFloat16>(a, b,
      y + static_cast<std::size_t>(begin) * n, end - begin, n, k,
      row_base, column_base, a_shared, b_shared);
}

template <bool BFloat16>
__global__ void grouped_gemm_lowp_simt_kernel(
    const std::uint16_t* x, const std::uint16_t* weights,
    const std::int32_t* offsets, float* y, int n, int k,
    unsigned int column_tiles) {
  const int expert = static_cast<int>(blockIdx.z);
  const int begin = offsets[expert];
  const int end = offsets[expert + 1];
  const int row_base = static_cast<int>(blockIdx.x / column_tiles) * 16;
  if (begin < 0 || end <= begin || row_base >= end - begin) return;
  const int column_base = static_cast<int>(blockIdx.x % column_tiles) * 32;
  const auto* a = k == 0 ? x : x + static_cast<std::size_t>(begin) * k;
  const auto* b = k == 0 ? weights : weights + static_cast<std::size_t>(expert) * k * n;
  simt_tile<16, 32, 1, BFloat16>(a, b, y + static_cast<std::size_t>(begin) * n,
      end - begin, n, k, row_base, column_base);
}

cudaError_t grouped_grid(int experts, int max_rows, int n, int bm, int bn,
                         dim3& grid, unsigned int& columns) {
  columns = ceil_div(n, bn);
  const auto blocks = static_cast<std::uint64_t>(ceil_div(max_rows, bm)) * columns;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  grid = dim3(static_cast<unsigned int>(blocks), 1U, static_cast<unsigned int>(experts));
  return cudaSuccess;
}

template <int BM, int BK, bool Asynchronous>
void launch_grouped(const void* x, const void* weights,
                    const std::int32_t* offsets, float* y, int n, int k,
                    ScalarType type, dim3 grid, unsigned int columns,
                    cudaStream_t stream) {
  if (type == ScalarType::kBf16)
    grouped_gemm_mma_kernel<BM, 64, BK, Asynchronous, true><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x), static_cast<const std::uint16_t*>(weights),
        offsets, y, n, k, columns);
  else
    grouped_gemm_mma_kernel<BM, 64, BK, Asynchronous, false><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x), static_cast<const std::uint16_t*>(weights),
        offsets, y, n, k, columns);
}

template <int BM, int BK>
void launch_grouped_staging(const void* x, const void* weights,
                            const std::int32_t* offsets, float* y, int n, int k,
                            ScalarType type, bool asynchronous, dim3 grid,
                            unsigned int columns, cudaStream_t stream) {
  if (asynchronous)
    launch_grouped<BM, BK, true>(x, weights, offsets, y, n, k, type, grid, columns, stream);
  else
    launch_grouped<BM, BK, false>(x, weights, offsets, y, n, k, type, grid, columns, stream);
}

}  // namespace

cudaError_t launch_grouped_gemm_tensorcore(
    const void* x, const void* weights, const std::int32_t* offsets, float* y,
    int experts, int hidden, int output, int max_expert_tokens,
    ScalarType input_type, const TensorCoreConfig& config, cudaStream_t stream) {
  const auto status = validate_grouped(x, weights, offsets, y, experts,
      hidden, output, max_expert_tokens, input_type);
  if (status != cudaSuccess) return status;
  if ((config.block_m != 16 && config.block_m != 32) || config.block_n != 64 ||
      (config.block_k != 32 && config.block_k != 64)) return cudaErrorInvalidValue;
  if (experts == 0 || output == 0 || max_expert_tokens == 0) return cudaSuccess;
  dim3 grid;
  unsigned int columns = 0;
  const auto grid_status = grouped_grid(experts, max_expert_tokens, output,
      config.block_m, config.block_n, grid, columns);
  if (grid_status != cudaSuccess) return grid_status;
  if (config.block_m == 16) {
    if (config.block_k == 32)
      launch_grouped_staging<16, 32>(x, weights, offsets, y, output, hidden,
          input_type, config.asynchronous, grid, columns, stream);
    else
      launch_grouped_staging<16, 64>(x, weights, offsets, y, output, hidden,
          input_type, config.asynchronous, grid, columns, stream);
  } else {
    if (config.block_k == 32)
      launch_grouped_staging<32, 32>(x, weights, offsets, y, output, hidden,
          input_type, config.asynchronous, grid, columns, stream);
    else
      launch_grouped_staging<32, 64>(x, weights, offsets, y, output, hidden,
          input_type, config.asynchronous, grid, columns, stream);
  }
  return cudaGetLastError();
}

cudaError_t launch_grouped_gemm_lowp_simt(
    const void* x, const void* weights, const std::int32_t* offsets, float* y,
    int experts, int hidden, int output, int max_expert_tokens,
    ScalarType input_type, cudaStream_t stream) {
  const auto status = validate_grouped(x, weights, offsets, y, experts,
      hidden, output, max_expert_tokens, input_type);
  if (status != cudaSuccess || experts == 0 || output == 0 || max_expert_tokens == 0)
    return status;
  dim3 grid;
  unsigned int columns = 0;
  const auto grid_status = grouped_grid(experts, max_expert_tokens, output,
      16, 32, grid, columns);
  if (grid_status != cudaSuccess) return grid_status;
  if (input_type == ScalarType::kBf16)
    grouped_gemm_lowp_simt_kernel<true><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x), static_cast<const std::uint16_t*>(weights),
        offsets, y, output, hidden, columns);
  else
    grouped_gemm_lowp_simt_kernel<false><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x), static_cast<const std::uint16_t*>(weights),
        offsets, y, output, hidden, columns);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
