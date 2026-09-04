# Copyright (c) 2019-2026 OpenAI
# SPDX-License-Identifier: MIT
#
# 改写自 Triton 的矩阵乘法教程。
# 修改点：严格 IEEE FP32、SM86 安全配置、预分配输出、支持 caller stream，
# 并补齐所有边界 masking。

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _dense_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_start < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k_start < K) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee", out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def launch_dense_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Enqueue strict-FP32 ``out = a @ b`` into a preallocated output."""
    if a.dtype != torch.float32 or b.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("dense_gemm Triton baseline requires FP32 tensors")
    if not (a.is_cuda and b.is_cuda and out.is_cuda):
        raise ValueError("dense_gemm tensors must be CUDA tensors")
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("dense_gemm tensors must be contiguous row-major")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2 or a.shape[1] != b.shape[0] or out.shape != (a.shape[0], b.shape[1]):
        raise ValueError("dense_gemm shape mismatch")
    m, k = a.shape
    _, n = b.shape
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        if m == 0 or n == 0:
            return
        if k == 0:
            out.zero_()
            return
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
        _dense_gemm_kernel[grid](
            a,
            b,
            out,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
        )
