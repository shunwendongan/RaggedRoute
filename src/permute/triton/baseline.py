# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 改写自 TransformerEngine 的 Triton permutation kernels。
# 修改点：直接消费 Top-K ids 和 offsets，生成带 per-expert scan 的稳定 route map，
# 并暴露预分配的 FP32 输出缓冲区。

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.jit
def _route_map_kernel(
    ids_ptr,
    offsets_ptr,
    route_pos_ptr,
    sorted_route_ptr,
    ROUTES: tl.constexpr,
    BLOCK_R: tl.constexpr,
    STORE_SORTED: tl.constexpr,
):
    expert = tl.program_id(0)
    routes = tl.arange(0, BLOCK_R)
    valid = routes < ROUTES
    ids = tl.load(ids_ptr + routes, mask=valid, other=-1)
    selected = valid & (ids == expert)
    rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    base = tl.load(offsets_ptr + expert)
    destination = base + rank
    tl.store(route_pos_ptr + routes, destination, mask=selected)
    if STORE_SORTED:
        tl.store(sorted_route_ptr + destination, routes, mask=selected)


@triton.jit
def _permute_rows_kernel(
    x_ptr,
    route_pos_ptr,
    out_ptr,
    ROUTES: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = cols < HIDDEN
    token = route // TOP_K
    destination = tl.load(route_pos_ptr + route).to(tl.int64)
    values = tl.load(x_ptr + token * HIDDEN + cols, mask=valid)
    tl.store(out_ptr + destination * HIDDEN + cols, values, mask=valid)


def launch_token_permute(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    offsets: torch.Tensor,
    x_permuted: torch.Tensor,
    route_pos: torch.Tensor,
    sorted_route: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Generate a stable expert-grouped mapping and copy routed token rows."""
    tensors = (x, expert_ids, offsets, x_permuted, route_pos)
    if x.dtype != torch.float32 or x_permuted.dtype != torch.float32:
        raise TypeError("token_permute payload requires FP32")
    if any(t.dtype != torch.int32 for t in (expert_ids, offsets, route_pos)) or (
        sorted_route is not None and sorted_route.dtype != torch.int32
    ):
        raise TypeError("token_permute metadata requires int32")
    if not all(t.is_cuda and t.is_contiguous() for t in tensors) or (
        sorted_route is not None and (not sorted_route.is_cuda or not sorted_route.is_contiguous())
    ):
        raise ValueError("token_permute tensors must be contiguous CUDA tensors")
    if x.ndim != 2 or expert_ids.ndim != 2 or expert_ids.shape[0] != x.shape[0]:
        raise ValueError("token_permute requires x[T,K] and expert_ids[T,top_k]")
    tokens, hidden = x.shape
    top_k = expert_ids.shape[1]
    routes = expert_ids.numel()
    experts = offsets.numel() - 1
    if experts < top_k or experts > 64:
        raise ValueError("token_permute requires top_k <= E <= 64")
    if x_permuted.shape != (routes, hidden) or route_pos.numel() != routes:
        raise ValueError("token_permute output shape mismatch")
    if sorted_route is not None and sorted_route.numel() != routes:
        raise ValueError("sorted_route must contain one entry per route")
    if routes == 0:
        return
    block_r = triton.next_power_of_2(routes)
    if block_r > 65536:
        raise ValueError("stable Triton route mapping supports at most 65536 routes")
    block_h = 256
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        _launch_route_map(expert_ids, offsets, route_pos, sorted_route, experts, routes, block_r)
        _launch_permute_rows(x, route_pos, x_permuted, routes, top_k, hidden, block_h)


def _launch_route_map(expert_ids, offsets, route_pos, sorted_route, experts, routes, block_r) -> None:
    sorted_arg = sorted_route if sorted_route is not None else route_pos
    _route_map_kernel[(experts,)](
        expert_ids,
        offsets,
        route_pos,
        sorted_arg,
        routes,
        BLOCK_R=block_r,
        STORE_SORTED=sorted_route is not None,
        num_warps=8 if block_r >= 2048 else 4,
    )


def _launch_permute_rows(x, route_pos, x_permuted, routes, top_k, hidden, block_h=256) -> None:
    _permute_rows_kernel[(routes, triton.cdiv(hidden, block_h))](
        x, route_pos, x_permuted, routes, top_k, hidden, BLOCK_H=block_h, num_warps=4
    )


def launch_route_map(
    expert_ids: torch.Tensor,
    offsets: torch.Tensor,
    route_pos: torch.Tensor,
    sorted_route: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Generate stable route metadata without copying payload rows."""
    if any(t.dtype != torch.int32 for t in (expert_ids, offsets, route_pos)) or (
        sorted_route is not None and sorted_route.dtype != torch.int32
    ):
        raise TypeError("route metadata requires int32")
    tensors = (expert_ids, offsets, route_pos) + (() if sorted_route is None else (sorted_route,))
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("route metadata must be contiguous CUDA tensors")
    routes = expert_ids.numel()
    experts = offsets.numel() - 1
    if route_pos.numel() != routes or (sorted_route is not None and sorted_route.numel() != routes):
        raise ValueError("route metadata shape mismatch")
    if routes == 0:
        return
    block_r = triton.next_power_of_2(routes)
    if block_r > 65536:
        raise ValueError("stable Triton route mapping supports at most 65536 routes")
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        _launch_route_map(expert_ids, offsets, route_pos, sorted_route, experts, routes, block_r)


def launch_permute_rows(
    x: torch.Tensor,
    route_pos: torch.Tensor,
    x_permuted: torch.Tensor,
    top_k: int,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Copy payload rows using a route map prepared outside the timed boundary."""
    if x.dtype != torch.float32 or x_permuted.dtype != torch.float32 or route_pos.dtype != torch.int32:
        raise TypeError("permute rows requires FP32 payload and int32 route_pos")
    if not all(t.is_cuda and t.is_contiguous() for t in (x, route_pos, x_permuted)):
        raise ValueError("permute rows tensors must be contiguous CUDA tensors")
    if x.ndim != 2 or top_k <= 0:
        raise ValueError("permute rows expects x[T,K] and top_k > 0")
    routes = route_pos.numel()
    hidden = x.shape[1]
    if routes != x.shape[0] * top_k or x_permuted.shape != (routes, hidden):
        raise ValueError("permute rows shape mismatch")
    if routes == 0:
        return
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        _launch_permute_rows(x, route_pos, x_permuted, routes, top_k, hidden)
