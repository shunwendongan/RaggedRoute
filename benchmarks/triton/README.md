# Triton Top-K auxiliary baseline

This directory contains a WSL2-only, benchmark-level Triton comparison for
the strict FP32 RaggedRoute Top-K Gate contract. It is intentionally not
registered as a C++ runtime variant and is not eligible for Auto promotion.

Pinned userspace dependencies are `torch==2.13.0`, `triton==3.7.1`, and
`numpy==2.5.1`. The profiler packages are NVIDIA Nsight Compute 2026.2.1 and
Nsight Systems 2026.1.3 from NVIDIA's official WSL CUDA apt repository. No
Linux NVIDIA driver is installed; WSL uses the host driver.

Setup and validation:

```bash
bash scripts/setup_wsl_triton.sh "$PWD"
python=~/.cache/raggedroute/topk-triton-venv/bin/python
"$python" benchmarks/triton/topk_gate.py correctness
```

Release results are absolute WSL2 measurements. Cross-OS ratios against the
Windows C++ harness are diagnostic context only and must not be described as
strictly paired speedups.
