# Token Permute production baseline provenance

- Upstream project: vLLM, fixed commit `837eae64580c885101ee95b073aafb27a485e7ce`.
- Relevant upstream files:
  - `csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu`
  - `csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h`
  - `csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl`
  - `csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu`
- Stable source root:
  <https://github.com/vllm-project/vllm/tree/837eae64580c885101ee95b073aafb27a485e7ce/csrc/libtorch_stable/moe>
- Adapted symbols/contracts: `moe_permute_with_scratch` and
  `expandInputRowsKernelLauncher`.
- Local changes: removed PyTorch and CUTLASS runtime dependencies, specialized to standalone FP32,
  made caller stream and workspace explicit, and added a scalar fallback when a row does not meet
  the 16-byte vector contract. CUB radix sorting remains part of the full from-ids boundary.
- License: vLLM Apache-2.0; retained at `third_party/licenses/vLLM-Apache-2.0.txt`.
