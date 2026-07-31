# Exclusive Scan library baseline provenance

- APIs: `cub::DeviceScan::ExclusiveSum`, `cub::BlockScan::ExclusiveSum`, and
  `cub::WarpScan::ExclusiveSum` from NVIDIA CCCL/CUB.
- Provider: bundled CUDA Toolkit headers or CMake `AUTO|SYSTEM|FETCH`; Fetch pins CCCL `v3.4.0`.
- Integration: RaggedRoute-owned wrappers over public CUB APIs; no CUB source snapshot or binary is
  committed. A one-thread completion kernel materializes `offsets[E]` for DeviceScan so all
  variants implement the same `E+1` output contract.
- Shape restrictions: BlockScan supports `E<=128`; the current operator contract limits `E<=64`;
  WarpScan is registered only for `E<=32`.
- Upstream: <https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html>.
- License: <https://github.com/NVIDIA/cccl/blob/v3.4.0/LICENSE>.
