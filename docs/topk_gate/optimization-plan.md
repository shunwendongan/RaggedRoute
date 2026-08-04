# Top-K Gate SM86 optimization design

## Contract and evidence boundary

The target is FP32 `TopKGateArgs` on RTX 3080 / SM86 for `2<=E<=64`,
`top_k=2`, caller stream, and zero workspace. Selection uses raw logits;
lower expert id wins ties; NaN ranks as negative infinity; all-NaN rows return
ids 0/1 and weights 0.5/0.5; selected-softmax normalizes only the two winners.
Ids must match exactly and weights use `atol=rtol=1e-6`.

Profiler duration is diagnostic only. Promotion and speedup claims come from
unprofiled Release A/B runs with identical case config, GPU UUID, seed, cache
mode, repeats, and excluded steps.

## Baselines

- `cuda_naive`: one thread scans one row serially.
- `cub_block_radix_top2`: strict composite-key
  `BlockRadixSort<uint64_t,32,2>`; sort, selected-softmax, and output are all
  inside the measured launch.
- `vllm_row_packed_top2`: Apache-2.0 vLLM structure pinned at
  `55c98e370aa058f567a9e682dc0652bdfba6b0bb`; only supported/aligned
  power-of-two E shapes are paired.

## Single-variable candidate chain

1. `cuda_warp_pair_top2_v1`: 32 lanes per token. Each lane forms a local
   ordered Top-2 and shuffle rounds merge two pairs, avoiding two full warp
   argmax reductions.
2. `cuda_subwarp_pair_top2_v2`: direct E=2 path; next-power-of-two subgroup
   for E<=32; 32 lanes for E>32. Sentinel lanes remain active in collectives.
3. `cuda_vector_pair_top2_v3`: aligned float4 specialization for
   E={8,16,32,64}, with rows-per-warp and CTA warps chosen to keep roughly
   16 rows/CTA. Unsupported or 4-byte-only aligned inputs fall back to v2.

Shared-memory staging, cp.async, tensor cores, persistent queues, TMA, and
WGMMA are excluded: the row has no useful data reuse, k is two, and the
baseline evidence points to grid underfill/lane utilization rather than a
reusable transfer pipeline. Hopper/Blackwell mechanisms are not applicable to
SM86.

## Literature-to-design map

- [vLLM Top-K gating](https://github.com/vllm-project/vllm/blob/55c98e370aa058f567a9e682dc0652bdfba6b0bb/csrc/libtorch_stable/moe/topk_softmax_kernels.cu):
  row packing, vector loads, four-warps/CTA.
- [TensorRT-LLM moeTopKFuncs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh):
  sortable float/index keys and lane-local candidates.
- [WarpSelect](https://arxiv.org/abs/1702.08734): register-resident,
  single-pass warp selection, specialized here to exact k=2 pair merge.
- [Online normalizer calculation](https://arxiv.org/abs/1805.02867):
  normalization without materializing a full softmax; here only selected
  logits are normalized.
- [RTop-K](https://arxiv.org/abs/2409.00822),
  [AIR Top-K](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap294.html),
  and [RadiK](https://arxiv.org/abs/2501.14336): algorithm/distribution
  taxonomy, not direct implementations for short exact k=2 rows.
- [Qrita](https://arxiv.org/abs/2602.01518): duplicate and deterministic
  ordering analysis only; vocabulary-scale pivot search is intentionally not
  transplanted.

## Verification and promotion

CTest covers every E from 2 through 64, three T/seed combinations,
misalignment, edge-value modes, explicit implementation ids, non-default
stream, and guarded redzones. Compute Sanitizer runs memcheck, racecheck,
initcheck, and synccheck.

Promotion buckets are E=2, 3-8, 9-16, 17-32, and 33-64, with aligned
power-of-two v3 evaluated separately. A candidate is enabled only from a
measured T threshold where every larger tested shape passes: ratio-of-sums
p50 speedup >=1.05 vs naive, per-shape p50 regression <=3%, p95 regression
<=5%, all-sample CV<=0.10, no spill, no extra launch, and zero workspace.
Otherwise Auto remains naive and the rejection evidence is retained.

## PR packaging status (2026-08-04)

This PR intentionally retains v4 as an explicit implementation id for reproducibility and follow-up experiments, while keeping `Auto` on `cuda_naive`. The completed Release campaign found no interval that simultaneously passed the in-tree L1/L2 and strong-library gates. v4 must therefore not be selected by automatic dispatch in this revision; the attached reports are evidence of the candidate and its rejection, not a production speedup claim.

## 2026-08-04 v4 campaign: local pair plus two reductions

The new candidate is `cuda_local_pair_two_reduce_top2_v4`. It keeps four
warps per CTA while changing only the selection mechanism: every lane builds a
register-local Top-2, one subgroup reduction selects the global winner, and a
second reduction selects the best of the winning lane's local runner-up and
the other lanes' local winners. FP32 values are converted to sortable bits and
paired with the complemented expert id, preserving lower-id ties without a
64-bit SM100 `redux` dependency. Aligned power-of-two rows use float2/float4;
all other inputs fall back to v2. Public API, one-launch behavior, and zero
workspace are unchanged.

The experiment compares 2, 4, and 8 warps per CTA as a separate launch-geometry
variable. Only four warps are retained because the multi-process geometry sweep
did not establish a stable alternative winner under the CV gate. The final
promotion evaluator is shape-specific for E={2,4,8,16,32,64}: every selected
T interval must have no p50 regression, at least 1.05x ratio-of-sums versus the
fastest of naive/v1/v2/v3 at both L1 and L2, and at least 1.03x versus the
fastest contract-equivalent CUB/vLLM baseline at L1. Per-shape p95 regression
is limited to 5%, CV to 0.10, and five independent processes are mandatory.

Production references were rechecked at fixed upstream revisions:

- [vLLM row-packed gating](https://github.com/vllm-project/vllm/blob/f0de1a604cad003379e5bb4dfc3cc5d2a1f25fa8/csrc/libtorch_stable/moe/topk_softmax_kernels.cu)
  and [packed Top-K reducer](https://github.com/vllm-project/vllm/blob/f0de1a604cad003379e5bb4dfc3cc5d2a1f25fa8/csrc/libtorch_stable/moe/moeTopKFuncs.cuh):
  row packing, local candidate ownership, deterministic packed keys, and
  architecture-gated fast reductions.
- [TensorRT-LLM Top-K reducer](https://github.com/NVIDIA/TensorRT-LLM/blob/624521576f44517c7cabb0f09ed9444adafb7093/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh)
  and [FlashInfer Top-K reducer](https://github.com/flashinfer-ai/flashinfer/blob/76c583655cf789051fca5870fef2adb8e544b576/csrc/fused_moe/moeTopKFuncs.cuh):
  packed value/index comparison and local-candidate sorting; SM100-only redux
  paths are deliberately not transferred to SM86.
- [RTop-K](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca1b93fc0f3560ba84eb0bc8de6d8f91-Abstract-Conference.html),
  [RadiK](https://arxiv.org/abs/2501.14336), and
  [Qrita](https://arxiv.org/abs/2602.01518) inform distribution and
  determinism tests. Their pivot/radix regimes target much longer arrays or
  larger k and remain rejected for exact E<=64, k=2 routing.
