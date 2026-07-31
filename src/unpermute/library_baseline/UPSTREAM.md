# Unpermute production baseline provenance

- Upstream project: vLLM, fixed commit `837eae64580c885101ee95b073aafb27a485e7ce`.
- Relevant upstream files:
  - `csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h`
  - `csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl`
- Stable source root:
  <https://github.com/vllm-project/vllm/tree/837eae64580c885101ee95b073aafb27a485e7ce/csrc/libtorch_stable/moe/permute_unpermute_kernels>
- Adapted symbol/contract: `finalizeMoeRoutingKernelLauncher`.
- Local changes: removed PyTorch/CUTLASS dependencies, specialized to standalone FP32, preserved
  caller-stream execution, and added a scalar fallback when output rows are not 16-byte aligned.
- License: vLLM Apache-2.0; retained at `third_party/licenses/vLLM-Apache-2.0.txt`.
