# External Top-K baseline unavailable for the current contract

RaggedRoute Top-K is a batched, per-token Top-2 gate with deterministic tie handling, explicit
NaN behavior, and selected-entry softmax weights. The CUB DeviceTopK interfaces available in the
current CCCL/CUDA environment describe a different global/batched selection and output-ordering
contract and do not provide a drop-in implementation of all these semantics.

Accordingly, the registry exposes only `cuda_naive` for now. A future external variant may be
added only after it matches every case-config field and passes the same CPU oracle. Until then,
the valid performance chain is `cuda_naive -> optimized CUDA -> measured roof`; no synthetic
library speedup is emitted.
