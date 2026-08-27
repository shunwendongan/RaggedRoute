# Kernel research campaigns

`kernel-research.json` is the active correctness/performance campaign consumed
by the Codex kernel workflow. It currently points at the six-stage v3 research
evaluation: Grouped GEMM v3 and Token Permute v3 are compared with their
strongest retained baselines, then composed against the integrated-v2 L3 chain.

`campaigns/<operator>.json` retains the measurement contract for every
operator. The active file must be byte-for-byte equivalent to the selected
campaign after JSON normalization. Campaigns preserve strict FP32 semantics,
use unprofiled release timing for decisions, and use NSYS/NCU only to explain
the measured result. The route-trace fixture is explicitly synthetic and may
validate the pipeline, but the real-trace policy must return
`insufficient_evidence` until a captured or production trace is supplied.
