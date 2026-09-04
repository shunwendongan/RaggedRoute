# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# 改写自 TransformerEngine 的 _unpermute_kernel 和 vLLM 的 moe_fused_mul_sum_kernel。
# 修改点：增加 route_pos 间接索引、固定 Top-K 的 FP32 累加，并支持预分配的
# caller-stream 输出缓冲区。

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.jit
def _unpermute_kernel(
    y_permuted_ptr,
    route_pos_ptr,
    route_weights_ptr,
    out_ptr,
    TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    OUTPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    tokens = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = tokens < TOKENS
    col_mask = cols < OUTPUT
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for rank in tl.static_range(TOP_K):
        route = tokens * TOP_K + rank
        position = tl.load(route_pos_ptr + route, mask=token_mask, other=0).to(tl.int64)
        weight = tl.load(route_weights_ptr + route, mask=token_mask, other=0.0).to(tl.float32)
        values = tl.load(
            y_permuted_ptr + position[:, None] * OUTPUT + cols[None, :],
            mask=token_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += values * weight[:, None]
    tl.store(
        out_ptr + tokens[:, None] * OUTPUT + cols[None, :],
        accumulator,
        mask=token_mask[:, None] & col_mask[None, :],
    )


def launch_unpermute(
    y_permuted: torch.Tensor,
    route_pos: torch.Tensor,
    route_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """收集 routed rows，并执行严格 FP32 的 Top-K 加权归约。"""
    if any(t.dtype != torch.float32 for t in (y_permuted, route_weights, out)) or route_pos.dtype != torch.int32:
        raise TypeError("unpermute requires FP32 payload/weights and int32 route_pos")
    if not all(t.is_cuda and t.is_contiguous() for t in (y_permuted, route_pos, route_weights, out)):
        raise ValueError("unpermute tensors must be contiguous CUDA tensors")
    if y_permuted.ndim != 2 or out.ndim != 2 or top_k <= 0:
        raise ValueError("unpermute expects y_permuted[R,N], out[T,N], and top_k > 0")
    tokens, output = out.shape
    routes = tokens * top_k
    if y_permuted.shape != (routes, output) or route_pos.numel() != routes or route_weights.numel() != routes:
        raise ValueError("unpermute shape mismatch")
    if tokens == 0 or output == 0:
        return
    block_m = 4 if tokens >= 32 else 1
    block_n = 256
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        _unpermute_kernel[(triton.cdiv(output, block_n), triton.cdiv(tokens, block_m))](
            y_permuted,
            route_pos,
            route_weights,
            out,
            tokens,
            top_k,
            output,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8 if block_m * block_n >= 1024 else 4,
            num_stages=2,
        )
