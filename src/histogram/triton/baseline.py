# Copyright (c) 2019-2026 OpenAI
# SPDX-License-Identifier: MIT
#
# RaggedRoute 对 Triton `tl.histogram` primitive 的改写。

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(
    ids_ptr,
    counts_ptr,
    ROUTES: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BINS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = offsets < ROUTES
    values = tl.load(ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    valid = mask & (values >= 0) & (values < EXPERTS)
    local = tl.histogram(values, BINS, mask=valid)
    bins = tl.arange(0, BINS)
    tl.atomic_add(counts_ptr + bins, local, mask=bins < EXPERTS)


def launch_histogram(
    expert_ids: torch.Tensor,
    counts: torch.Tensor,
    *,
    reset: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Count int32 route ids. ``reset=True`` matches the L2 operator boundary."""
    if expert_ids.dtype != torch.int32 or counts.dtype != torch.int32:
        raise TypeError("histogram requires int32 ids and counts")
    if not (expert_ids.is_cuda and counts.is_cuda):
        raise ValueError("histogram tensors must be CUDA tensors")
    if not (expert_ids.is_contiguous() and counts.is_contiguous()):
        raise ValueError("histogram tensors must be contiguous")
    experts = counts.numel()
    if experts > 64:
        raise ValueError("histogram supports E <= 64")
    routes = expert_ids.numel()
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        if reset:
            counts.zero_()
        if routes == 0 or experts == 0:
            return
        block_r = min(triton.next_power_of_2(routes), 4096)
        bins = triton.next_power_of_2(experts)
        _histogram_kernel[(triton.cdiv(routes, block_r),)](
            expert_ids, counts, routes, experts, BLOCK_R=block_r, BINS=bins, num_warps=4
        )
