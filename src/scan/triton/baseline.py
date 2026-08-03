# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for RaggedRoute from FlagGems cumsum kernels.
# Modifications: small-E int32 exclusive scan and terminal offset output.

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.jit
def _exclusive_scan_kernel(
    counts_ptr,
    offsets_ptr,
    EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_E)
    valid = lanes < EXPERTS
    values = tl.load(counts_ptr + lanes, mask=valid, other=0).to(tl.int32)
    inclusive = tl.cumsum(values, axis=0)
    tl.store(offsets_ptr, 0)
    tl.store(offsets_ptr + lanes + 1, inclusive, mask=valid)


def launch_exclusive_scan(
    counts: torch.Tensor,
    offsets: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Write ``offsets[0]=0`` and the inclusive prefix into ``offsets[1:]``."""
    if counts.dtype != torch.int32 or offsets.dtype != torch.int32:
        raise TypeError("exclusive_scan requires int32 tensors")
    if not (counts.is_cuda and offsets.is_cuda):
        raise ValueError("exclusive_scan tensors must be CUDA tensors")
    if not (counts.is_contiguous() and offsets.is_contiguous()):
        raise ValueError("exclusive_scan tensors must be contiguous")
    experts = counts.numel()
    if experts > 64 or offsets.numel() != experts + 1:
        raise ValueError("exclusive_scan requires E <= 64 and E+1 offsets")
    block_e = triton.next_power_of_2(max(experts, 1))
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        _exclusive_scan_kernel[(1,)](counts, offsets, experts, BLOCK_E=block_e, num_warps=1)
