# Copyright (c) 2019-2026 OpenAI
# SPDX-License-Identifier: MIT
#
# Adapted for RaggedRoute from Triton's portable grouped-GEMM tutorial.
# Modifications: direct contiguous expert storage, device offsets, ragged/empty
# expert masking, strict IEEE FP32, preallocated output, and no TMA path.

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["EXPERTS", "HIDDEN", "OUTPUT", "MAX_M"],
)
@triton.jit
def _grouped_gemm_kernel(
    x_ptr,
    weights_ptr,
    offsets_ptr,
    out_ptr,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUTPUT: tl.constexpr,
    MAX_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.program_id(1)
    num_n_tiles = tl.cdiv(OUTPUT, BLOCK_N)
    tile_m = tile // num_n_tiles
    tile_n = tile % num_n_tiles
    start = tl.load(offsets_ptr + expert).to(tl.int64)
    end = tl.load(offsets_ptr + expert + 1).to(tl.int64)
    rows = end - start
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (start + offs_m[:, None]) * HIDDEN + offs_k[None, :]
    w_ptrs = weights_ptr + expert * HIDDEN * OUTPUT + offs_k[:, None] * OUTPUT + offs_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, HIDDEN, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < rows) & (offs_k[None, :] + k_start < HIDDEN), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] + k_start < HIDDEN) & (offs_n[None, :] < OUTPUT), other=0.0)
        accumulator = tl.dot(x, w, acc=accumulator, input_precision="ieee", out_dtype=tl.float32)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * OUTPUT
    out_ptrs = out_ptr + (start + offs_m[:, None]) * OUTPUT + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=(offs_m[:, None] < rows) & (offs_n[None, :] < OUTPUT))


def launch_grouped_gemm(
    x_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    offsets: torch.Tensor,
    y_permuted: torch.Tensor,
    max_expert_tokens: int,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Enqueue strict-FP32 ragged expert GEMMs using a caller-provided M bound."""
    if any(t.dtype != torch.float32 for t in (x_permuted, expert_weights, y_permuted)) or offsets.dtype != torch.int32:
        raise TypeError("grouped_gemm requires FP32 payload and int32 offsets")
    if not all(t.is_cuda and t.is_contiguous() for t in (x_permuted, expert_weights, offsets, y_permuted)):
        raise ValueError("grouped_gemm tensors must be contiguous CUDA tensors")
    if x_permuted.ndim != 2 or expert_weights.ndim != 3 or y_permuted.ndim != 2:
        raise ValueError("grouped_gemm expects x[R,K], weights[E,K,N], y[R,N]")
    routes, hidden = x_permuted.shape
    experts, weight_hidden, output = expert_weights.shape
    if weight_hidden != hidden or offsets.numel() != experts + 1 or y_permuted.shape != (routes, output):
        raise ValueError("grouped_gemm shape mismatch")
    if experts > 64 or max_expert_tokens < 0 or max_expert_tokens > routes:
        raise ValueError("grouped_gemm requires E <= 64 and a valid max_expert_tokens")
    if routes == 0 or experts == 0 or output == 0:
        return
    if hidden == 0:
        context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with context:
            y_permuted.zero_()
        return
    if max_expert_tokens == 0:
        return
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        grid = lambda meta: (
            triton.cdiv(max_expert_tokens, meta["BLOCK_M"]) * triton.cdiv(output, meta["BLOCK_N"]),
            experts,
        )
        _grouped_gemm_kernel[grid](
            x_permuted,
            expert_weights,
            offsets,
            y_permuted,
            experts,
            hidden,
            output,
            max_expert_tokens,
        )
