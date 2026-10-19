#include "../../common/tensorcore_gemm.cuh"

namespace raggedroute::ops {
namespace {

using namespace tensorcore_detail;

template <int BM, int BN, int BK, bool Asynchronous, bool BFloat16>
__global__ __launch_bounds__(kThreads) void dense_gemm_mma_kernel(
    const std::uint16_t* a, const std::uint16_t* b, float* c,
    int m, int n, int k, unsigned int column_tiles) {
  constexpr int Stages = Asynchronous ? 2 : 1;
  __align__(16) __shared__ std::uint16_t a_shared[Stages * BM * (BK + 8)];
  __align__(16) __shared__ std::uint16_t b_shared[Stages * BK * (BN + 8)];
  const int row_base = static_cast<int>(blockIdx.x / column_tiles) * BM;
  const int column_base = static_cast<int>(blockIdx.x % column_tiles) * BN;
  compute<BM, BN, BK, Asynchronous, BFloat16>(a, b, c, m, n, k,
      row_base, column_base, a_shared, b_shared);
}

template <bool BFloat16>
__global__ void dense_gemm_lowp_simt_kernel(const std::uint16_t* a,
                                            const std::uint16_t* b, float* c,
                                            int m, int n, int k,
                                            unsigned int column_tiles) {
  const int row_base = static_cast<int>(blockIdx.x / column_tiles) * 16;
  const int column_base = static_cast<int>(blockIdx.x % column_tiles) * 32;
  simt_tile<16, 32, 1, BFloat16>(a, b, c, m, n, k, row_base, column_base);
}

template <bool BFloat16>
__global__ void gemm_quantize_kernel(const float* source, void* destination,
                                      std::size_t elements) {
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if constexpr (BFloat16)
      static_cast<__nv_bfloat16*>(destination)[index] = __float2bfloat16_rn(source[index]);
    else
      static_cast<__half*>(destination)[index] = __float2half_rn(source[index]);
  }
}

cudaError_t dense_grid(int m, int n, int bm, int bn,
                       unsigned int& blocks, unsigned int& column_tiles) {
  column_tiles = ceil_div(n, bn);
  const auto tiles = static_cast<std::uint64_t>(ceil_div(m, bm)) * column_tiles;
  if (tiles > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    return cudaErrorInvalidConfiguration;
  blocks = static_cast<unsigned int>(tiles);
  return cudaSuccess;
}

template <int BN, int BK, bool Asynchronous>
void launch_dense(const void* a, const void* b, float* c, int m, int n, int k,
                  ScalarType type, unsigned int blocks,
                  unsigned int column_tiles, cudaStream_t stream) {
  if (type == ScalarType::kBf16)
    dense_gemm_mma_kernel<64, BN, BK, Asynchronous, true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b),
        c, m, n, k, column_tiles);
  else
    dense_gemm_mma_kernel<64, BN, BK, Asynchronous, false><<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b),
        c, m, n, k, column_tiles);
}

template <int BN, int BK>
void launch_dense_staging(const void* a, const void* b, float* c, int m, int n, int k,
                          ScalarType type, bool asynchronous, unsigned int blocks,
                          unsigned int column_tiles, cudaStream_t stream) {
  if (asynchronous)
    launch_dense<BN, BK, true>(a, b, c, m, n, k, type, blocks, column_tiles, stream);
  else
    launch_dense<BN, BK, false>(a, b, c, m, n, k, type, blocks, column_tiles, stream);
}

}  // namespace

cudaError_t launch_dense_gemm_tensorcore(
    const void* a, const void* b, float* c, int m, int n, int k,
    ScalarType input_type, const TensorCoreConfig& config, cudaStream_t stream) {
  const auto status = validate_dense(a, b, c, m, n, k, input_type);
  if (status != cudaSuccess) return status;
  if (config.block_m != 64 || (config.block_n != 32 && config.block_n != 64) ||
      (config.block_k != 32 && config.block_k != 64)) return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  unsigned int blocks = 0, columns = 0;
  const auto grid_status = dense_grid(m, n, config.block_m, config.block_n, blocks, columns);
  if (grid_status != cudaSuccess) return grid_status;
  if (config.block_n == 32) {
    if (config.block_k == 32)
      launch_dense_staging<32, 32>(a, b, c, m, n, k, input_type,
                                  config.asynchronous, blocks, columns, stream);
    else
      launch_dense_staging<32, 64>(a, b, c, m, n, k, input_type,
                                  config.asynchronous, blocks, columns, stream);
  } else {
    if (config.block_k == 32)
      launch_dense_staging<64, 32>(a, b, c, m, n, k, input_type,
                                  config.asynchronous, blocks, columns, stream);
    else
      launch_dense_staging<64, 64>(a, b, c, m, n, k, input_type,
                                  config.asynchronous, blocks, columns, stream);
  }
  return cudaGetLastError();
}

cudaError_t launch_dense_gemm_lowp_simt(
    const void* a, const void* b, float* c, int m, int n, int k,
    ScalarType input_type, cudaStream_t stream) {
  const auto status = validate_dense(a, b, c, m, n, k, input_type);
  if (status != cudaSuccess || m == 0 || n == 0) return status;
  unsigned int blocks = 0, columns = 0;
  const auto grid_status = dense_grid(m, n, 16, 32, blocks, columns);
  if (grid_status != cudaSuccess) return grid_status;
  if (input_type == ScalarType::kBf16)
    dense_gemm_lowp_simt_kernel<true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b),
        c, m, n, k, columns);
  else
    dense_gemm_lowp_simt_kernel<false><<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(a), static_cast<const std::uint16_t*>(b),
        c, m, n, k, columns);
  return cudaGetLastError();
}

cudaError_t launch_gemm_quantize(const float* source, void* destination,
                               std::size_t elements, ScalarType input_type,
                               cudaStream_t stream) {
  if (!supported_type(input_type)) return cudaErrorInvalidValue;
  if (elements == 0) return cudaSuccess;
  if (source == nullptr || destination == nullptr || !aligned(source, alignof(float)) ||
      !aligned(destination, 2)) return cudaErrorInvalidValue;
  const std::size_t needed = (elements - 1) / 256 + 1;
  const auto blocks = static_cast<unsigned int>(needed < 65535 ? needed : 65535);
  if (input_type == ScalarType::kBf16)
    gemm_quantize_kernel<true><<<blocks, 256, 0, stream>>>(source, destination, elements);
  else
    gemm_quantize_kernel<false><<<blocks, 256, 0, stream>>>(source, destination, elements);
  return cudaGetLastError();
}

}  // namespace raggedroute::ops
