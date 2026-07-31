#include <cub/device/device_histogram.cuh>

#include "raggedroute/benchmark/library_baselines.h"

namespace raggedroute::benchmark::library_baseline {

cudaError_t query_cub_histogram_workspace(std::size_t route_pairs, int experts,
                                          std::size_t* workspace_bytes) {
  if (workspace_bytes == nullptr || experts < 1) return cudaErrorInvalidValue;
  std::int32_t* null_ids = nullptr;
  std::int32_t* null_counts = nullptr;
  return cub::DeviceHistogram::HistogramEven(nullptr, *workspace_bytes, null_ids, null_counts,
                                             experts + 1, 0, experts, route_pairs);
}

cudaError_t launch_cub_histogram(const std::int32_t* ids, std::int32_t* counts,
                                 std::size_t route_pairs, int experts, void* workspace,
                                 std::size_t workspace_bytes, cudaStream_t stream) {
  if ((route_pairs != 0 && ids == nullptr) || counts == nullptr || experts < 1) {
    return cudaErrorInvalidValue;
  }
  return cub::DeviceHistogram::HistogramEven(workspace, workspace_bytes, ids, counts, experts + 1,
                                             0, experts, route_pairs, stream);
}

}  // namespace raggedroute::benchmark::library_baseline
