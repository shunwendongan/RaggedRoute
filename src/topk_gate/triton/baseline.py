# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for RaggedRoute from FlagGems topk_gating_softmax_kernel.
# Modifications: fixed Top-2 selected-softmax and RaggedRoute's exact tie,
# NaN, all-NaN, and infinity semantics.

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


@triton.jit
def _top2_selected_softmax_kernel(
    logits_ptr,
    ids_ptr,
    weights_ptr,
    TOKENS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    valid = cols < EXPERTS
    raw = tl.load(logits_ptr + token * EXPERTS + cols, mask=valid, other=-float("inf")).to(tl.float32)
    non_nan = raw == raw
    values = tl.where(non_nan & valid, raw, -float("inf"))
    first_value = tl.max(values, axis=0)
    first_id = tl.min(tl.where(valid & (values == first_value), cols, BLOCK_E), axis=0)
    remaining = tl.where(valid & (cols != first_id), values, -float("inf"))
    second_value = tl.max(remaining, axis=0)
    second_id = tl.min(tl.where(valid & (cols != first_id) & (remaining == second_value), cols, BLOCK_E), axis=0)
    saw_non_nan = tl.sum((non_nan & valid).to(tl.int32), axis=0) > 0
    first_id = tl.where(saw_non_nan, first_id, 0)
    second_id = tl.where(saw_non_nan, second_id, 1)

    equal = first_value == second_value
    terminal = (first_value == float("inf")) | (second_value == -float("inf"))
    relative_second = tl.exp(second_value - first_value)
    first_weight = 1.0 / (1.0 + relative_second)
    second_weight = relative_second * first_weight
    first_weight = tl.where(equal, 0.5, tl.where(terminal, 1.0, first_weight))
    second_weight = tl.where(equal, 0.5, tl.where(terminal, 0.0, second_weight))

    out = token * 2
    tl.store(ids_ptr + out, first_id.to(tl.int32))
    tl.store(ids_ptr + out + 1, second_id.to(tl.int32))
    tl.store(weights_ptr + out, first_weight)
    tl.store(weights_ptr + out + 1, second_weight)


def launch_topk_gate(
    logits: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Enqueue deterministic Top-2 selected-softmax into preallocated outputs."""
    if logits.dtype != torch.float32 or weights.dtype != torch.float32 or expert_ids.dtype != torch.int32:
        raise TypeError("topk_gate requires FP32 logits/weights and int32 ids")
    if not (logits.is_cuda and expert_ids.is_cuda and weights.is_cuda):
        raise ValueError("topk_gate tensors must be CUDA tensors")
    if logits.ndim != 2 or logits.shape[1] < 2 or logits.shape[1] > 64:
        raise ValueError("topk_gate requires logits shaped [T, E] with 2 <= E <= 64")
    if not (logits.is_contiguous() and expert_ids.is_contiguous() and weights.is_contiguous()):
        raise ValueError("topk_gate tensors must be contiguous")
    tokens, experts = logits.shape
    if expert_ids.numel() != tokens * 2 or weights.numel() != tokens * 2:
        raise ValueError("topk_gate outputs must contain T*2 elements")
    if tokens == 0:
        return
    block_e = triton.next_power_of_2(experts)
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with context:
        _top2_selected_softmax_kernel[(tokens,)](
            logits, expert_ids, weights, tokens, experts, BLOCK_E=block_e, num_warps=1
        )
