# Kernel research campaigns

`kernel-research.json` is the active correctness/performance campaign consumed
by the Codex kernel workflow. It currently points at Grouped GEMM, the highest
GPU-time hotspot in the measured seven-operator chain.

`campaigns/<operator>.json` retains the measurement contract for every
operator. The active file must be byte-for-byte equivalent to the selected
campaign after JSON normalization. Campaigns preserve strict FP32 semantics,
use unprofiled release timing for decisions, and use NSYS/NCU only to explain
the measured result.
