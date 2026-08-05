# Top-K Gate SM86 v4 rejection record

- Decision: **rejected; no Auto dispatch change, branch, push, release upload, or PR**.
- Hardware: GeForce RTX 3080 10 GiB, sm_86; GPU UUID hash `7c5e95c0e5a415d824a0c8c8b58d6f39`.
- Base commit: `728b9f3`; local detached measurement commit: `86cfbe2a805d8880b73896dece2c77031768245d`.
- Candidate: `cuda_local_pair_two_reduce_top2_v4`, four warps/CTA.
- Contract: strict FP32, `2 <= E <= 64`, `top_k=2`, deterministic ties/NaN, selected softmax, one launch, zero workspace.

## Correctness gate

- Release CTest: 8/8 passed.
- Compute Sanitizer: memcheck, initcheck, racecheck, and synccheck passed.
- The exhaustive matrix covered every `E=2..64`, tail rows, 4/16-byte alignment, ties, NaN/Inf/extremes, non-default stream, redzones, and explicit implementation id 4.

## Release evidence

- 16,710/16,710 declared strict keys are present with no duplicate keys; every post-measurement validation passed.
- Each aggregate group has five independent process runs, 20 warmups, 30 samples, and 100 repeats.
- The original orchestration shell was interrupted after 14,277 records. A key-set-based deterministic resume ran only the 2,433 missing commands. The manifest records `orchestration_status=interrupted_and_resumed`; the final expected and observed key sets match exactly.
- Promotion was evaluated per exact `E in {2,4,8,16,32,64}` and every tested `T_min`, against the fastest in-tree implementation at L1/L2 and the fastest available CUB/vLLM envelope at L1.
- Promoted intervals: **0**.

The closest large-E region still fails the reviewed gates. For `E=64,T>=2048`, v4 reaches about 1.079x ratio-of-sums at L2, but only about 1.025x at L1 and about 1.003x against the strong-library envelope; it also has single-shape regressions and CV above 0.10. At `E=32,T=4096`, L2 and the library ratio cross their speed thresholds, but L1 is about 0.964x and the p95/CV stability gates fail. Smaller E regions do not reach the required speedup envelope.

The unprofiled global strict vLLM pairing also remains mixed: v4 is 0.9745x by ratio-of-sums over 78 paired shapes, despite beating the CUB baseline by 2.2725x over 195 paired shapes. The fastest strong-library envelope therefore blocks promotion.

## Profiler diagnosis

At diagnostic `T=2048,E=64`, NSYS measured v4 at 2.304 us, vLLM at 2.528 us, v3 at 3.072 us, v2 at 4.288 us, and CUB at 31.359 us. These profiler durations were not used for promotion.

NCU basic reported v4 at 18 registers/thread, 27.19% achieved occupancy, 0.31 waves, 10.13% SM throughput, and 8.21% memory throughput, with no local-memory spill. Together with the very short kernel duration, this diagnoses launch geometry/underfill rather than sustained compute or bandwidth saturation. No detailed/source escalation was needed.

## Design and literature map

The rejected implementation used float2/float4 row packing, register-local Top-2, two subgroup reductions, and deterministic sortable FP32/index keys. It intentionally excluded shared-memory staging, cp.async, Tensor Cores, persistent queues, TMA, and WGMMA: these short rows have no useful staging reuse, and the measured bottleneck is underfill rather than a reusable transfer or matrix-math pipeline.

- [vLLM row-packed gating](https://github.com/vllm-project/vllm/blob/f0de1a604cad003379e5bb4dfc3cc5d2a1f25fa8/csrc/libtorch_stable/moe/topk_softmax_kernels.cu) and [packed reducer](https://github.com/vllm-project/vllm/blob/f0de1a604cad003379e5bb4dfc3cc5d2a1f25fa8/csrc/libtorch_stable/moe/moeTopKFuncs.cuh): row packing, local candidates, packed deterministic keys, and architecture-gated reductions.
- [TensorRT-LLM reducer](https://github.com/NVIDIA/TensorRT-LLM/blob/624521576f44517c7cabb0f09ed9444adafb7093/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh) and [FlashInfer reducer](https://github.com/flashinfer-ai/flashinfer/blob/76c583655cf789051fca5870fef2adb8e544b576/csrc/fused_moe/moeTopKFuncs.cuh): value/index packing and lane-local sorting. Their SM100-only redux paths were not transferred to SM86.
- [RTop-K](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca1b93fc0f3560ba84eb0bc8de6d8f91-Abstract-Conference.html), [RadiK](https://arxiv.org/abs/2501.14336), and [Qrita](https://arxiv.org/abs/2602.01518): distribution/determinism taxonomy and adversarial tests. Their pivot/radix regimes target longer arrays or larger k and were not transplanted into exact `E<=64,k=2` routing.

## Evidence map

- Raw Release JSONL: `topk-v4-release.jsonl` (43,372,989 bytes, SHA256 `f4a646d6993036c2179cfd1bd49d297e559021b703388c1444d036fa252fa5f5`)
- Run manifest: `topk-v4-release.jsonl.manifest.json` (17,674,961 bytes, SHA256 `22c716e7bdf952f12f6aa852b679260dcf768077af66a8ebfb29531d6d112ff1`)
- Aggregate v2 JSON: `topk-v4-release.aggregate-v2.json` (22,036,302 bytes, SHA256 `a90ca9bcfab4d821a4c3e202e1c3d7e2946bfdfb8291fd07c01a329a1d16838c`)
- Aggregate CSV: `topk-v4-release.aggregate-v2.csv` (877,787 bytes, SHA256 `dff3829c4ab5fc829221be3fd8c425b5c41cca77e61070a1141293dfb49d0cf2`)
- Strict comparison: `topk-v4-release.comparison.json` (2,528,631 bytes, SHA256 `57048b33db0793a896b8793d741805b7fa823e96e03efa0f1db6bc41a6255bda`)
- Exact-shape decision: `promotion-v4/promotion.json` (142,739 bytes, SHA256 `f54370c0ad4f5a1f42d7711f15e4e20d889478f37bf188342705e73483c930fd`)
- Human-readable aggregate report: `promotion-v4/REPORT.md` (8,116 bytes, SHA256 `b34026b3abb8b0295251f5b190bc1bbcae3672476f2872587530282fe3b1de2d`)
- Sanitizer logs: `sanitizer-v4/`
- NSYS/NCU reports and parsed analysis: `profile-v4-system/`
- Rejected 2/8-warps geometry trials: `v4-geometry*/`

Because no interval passed, L3 `chain_from_logits`, final dispatch-boundary profiling, evidence-v2 repository packaging, release upload, and CI/PR work were intentionally not run.
