# Rejected drop-in baselines

CUB DeviceTopK is not used as a primary denominator. Its device-wide/batched
selection and output-ordering contract is not a drop-in match for one
deterministic Top-2 per token with selected-softmax and the repository's NaN
fallback. A benchmark-only BlockRadixSort baseline is supplied instead because
its complete semantic boundary can be implemented and timed in one launch.

RTop-K, AIR Top-K, RadiK, and Qrita target long arrays, larger k, approximate
or pivot/early-stop regimes. They inform algorithm classification and
distribution tests, but transplanting them into E<=64, k=2 would add control
overhead and/or weaken exact determinism. They are therefore literature
references, not performance baselines.
