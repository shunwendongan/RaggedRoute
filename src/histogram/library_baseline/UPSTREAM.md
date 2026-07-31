# Histogram library baseline provenance

- API: `cub::DeviceHistogram::HistogramEven` from NVIDIA CCCL/CUB.
- Provider: bundled CUDA Toolkit headers or CMake `AUTO|SYSTEM|FETCH`; Fetch pins CCCL `v3.4.0`.
- Integration: RaggedRoute-owned wrapper over the public CUB API; no CUB source snapshot or binary
  is committed.
- Semantic restriction: int32 expert ids in `[0, E)` map exactly to `E` discrete bins. The CUB
  internal algorithm is treated as opaque and is not reported as a particular atomic strategy.
- Upstream API:
  <https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceHistogram.html>
- License: CCCL repository license, retained by the selected provider:
  <https://github.com/NVIDIA/cccl/blob/v3.4.0/LICENSE>.
