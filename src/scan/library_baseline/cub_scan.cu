#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/warp/warp_scan.cuh>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {
namespace {

__global__ void finish_device_scan(const std::int32_t* counts, std::int32_t* offsets, int experts) {
  if (threadIdx.x == 0) offsets[experts] = offsets[experts - 1] + counts[experts - 1];
}

template <int BlockThreads>
__global__ void block_scan_kernel(const std::int32_t* counts, std::int32_t* offsets, int experts) {
  using Scan = cub::BlockScan<std::int32_t, BlockThreads>;
  __shared__ typename Scan::TempStorage storage;
  const int thread = static_cast<int>(threadIdx.x);
  const std::int32_t value = thread < experts ? counts[thread] : 0;
  std::int32_t prefix = 0;
  std::int32_t aggregate = 0;
  Scan(storage).ExclusiveSum(value, prefix, aggregate);
  if (thread < experts) offsets[thread] = prefix;
  if (thread == 0) offsets[experts] = aggregate;
}

__global__ void warp_scan_kernel(const std::int32_t* counts, std::int32_t* offsets, int experts) {
  using Scan = cub::WarpScan<std::int32_t>;
  __shared__ typename Scan::TempStorage storage;
  const int lane = static_cast<int>(threadIdx.x);
  const std::int32_t value = lane < experts ? counts[lane] : 0;
  std::int32_t prefix = 0;
  std::int32_t aggregate = 0;
  Scan(storage).ExclusiveSum(value, prefix, aggregate);
  if (lane < experts) offsets[lane] = prefix;
  if (lane == 0) offsets[experts] = aggregate;
}

}  // namespace

cudaError_t query_cub_device_scan_workspace(int experts, std::size_t* workspace_bytes) {
  if (workspace_bytes == nullptr || experts < 1) return cudaErrorInvalidValue;
  std::int32_t* null_data = nullptr;
  return cub::DeviceScan::ExclusiveSum(nullptr, *workspace_bytes, null_data, null_data, experts);
}

cudaError_t launch_cub_device_scan(const std::int32_t* counts, std::int32_t* offsets, int experts,
                                   void* workspace, std::size_t workspace_bytes,
                                   cudaStream_t stream) {
  if (counts == nullptr || offsets == nullptr || experts < 1) return cudaErrorInvalidValue;
  cudaError_t error =
      cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes, counts, offsets, experts, stream);
  if (error != cudaSuccess) return error;
  finish_device_scan<<<1, 1, 0, stream>>>(counts, offsets, experts);
  return cudaPeekAtLastError();
}

cudaError_t launch_cub_block_scan(const std::int32_t* counts, std::int32_t* offsets, int experts,
                                  cudaStream_t stream) {
  if (counts == nullptr || offsets == nullptr || experts < 1 || experts > 128) {
    return cudaErrorInvalidValue;
  }
  block_scan_kernel<128><<<1, 128, 0, stream>>>(counts, offsets, experts);
  return cudaPeekAtLastError();
}

cudaError_t launch_cub_warp_scan(const std::int32_t* counts, std::int32_t* offsets, int experts,
                                 cudaStream_t stream) {
  if (counts == nullptr || offsets == nullptr || experts < 1 || experts > 32) {
    return cudaErrorInvalidValue;
  }
  warp_scan_kernel<<<1, 32, 0, stream>>>(counts, offsets, experts);
  return cudaPeekAtLastError();
}

}  // namespace raggedroute::benchmark::library_baseline
