#!/usr/bin/env python3
"""Strict FP32 Top-K Gate Triton comparison for WSL2/SM86.

This is benchmark-only evidence. It is not a RaggedRoute dispatch variant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any, Callable

import torch
import triton
import triton.language as tl


@triton.jit
def topk_gate_triton_kernel(
    logits,
    expert_ids,
    weights,
    experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < experts
    raw = tl.load(logits + token * experts + offsets, mask=in_bounds, other=float("nan"))
    non_nan = in_bounds & (raw == raw)
    values = tl.where(non_nan, raw, -float("inf"))
    first_id = tl.argmax(values, axis=0, tie_break_left=True)
    first = tl.max(values, axis=0)
    remaining = tl.where(offsets == first_id, -float("inf"), values)
    second_id = tl.argmax(remaining, axis=0, tie_break_left=True)
    second = tl.max(remaining, axis=0)
    # -inf cannot be made smaller to exclude first_id from argmax. If the
    # second value is -inf, every remaining expert ties there, so the contract
    # requires the lowest valid id different from first_id.
    second_id = tl.where(second == -float("inf"), tl.where(first_id == 0, 1, 0), second_id)
    saw_non_nan = tl.sum(non_nan.to(tl.int32), axis=0) != 0

    equal = first == second
    dominant = (first == float("inf")) | (second == -float("inf"))
    relative = tl.exp(second - first)
    first_weight = tl.where(equal, 0.5, tl.where(dominant, 1.0, 1.0 / (1.0 + relative)))
    second_weight = tl.where(equal, 0.5, tl.where(dominant, 0.0, relative * first_weight))
    first_id = tl.where(saw_non_nan, first_id, 0)
    second_id = tl.where(saw_non_nan, second_id, 1)
    first_weight = tl.where(saw_non_nan, first_weight, 0.5)
    second_weight = tl.where(saw_non_nan, second_weight, 0.5)

    output = token * 2
    tl.store(expert_ids + output, first_id)
    tl.store(expert_ids + output + 1, second_id)
    tl.store(weights + output, first_weight)
    tl.store(weights + output + 1, second_weight)


def launch(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> None:
    tokens, experts = logits.shape
    block = triton.next_power_of_2(experts)
    topk_gate_triton_kernel[(tokens,)](
        logits, ids, weights, experts=experts, BLOCK_SIZE=block, num_warps=1
    )


def make_input(tokens: int, experts: int, seed: int, mode: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.empty((tokens, experts), dtype=torch.float32).uniform_(
        -8.0, 8.0, generator=generator
    )
    if mode == "random":
        pass
    elif mode in {"ties", "all_equal"}:
        values.fill_(1.0)
    elif mode == "duplicate_max":
        values.fill_(-3.0)
        values[:, 0] = 7.0
        values[:, -1] = 7.0
    elif mode == "signed_zero":
        values[:, 0::2] = -0.0
        values[:, 1::2] = 0.0
    elif mode == "all_nan":
        values.fill_(float("nan"))
    elif mode == "mixed_nan_neg_inf":
        values.fill_(-float("inf"))
        values[:, 0] = float("nan")
        if experts > 2:
            values[:, 2] = float("nan")
    elif mode in {"single_infinity", "infinities"}:
        values[:, -1] = float("inf")
    elif mode == "multiple_infinity":
        values[:, 0] = float("inf")
        values[:, -1] = float("inf")
    elif mode == "extreme":
        values.fill_(-1.0e30)
        values[:, 0] = 1.0e30
        values[:, 1] = 0.0
    else:
        raise ValueError(f"unsupported input mode: {mode}")
    return values


def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, experts = logits.shape
    ids = torch.empty((tokens, 2), dtype=torch.int32)
    weights = torch.empty((tokens, 2), dtype=torch.float32)
    for token in range(tokens):
        row = logits[token].tolist()
        saw_non_nan = any(not math.isnan(value) for value in row)
        ranked = sorted(
            range(experts), key=lambda expert: (-(row[expert] if not math.isnan(row[expert]) else -math.inf), expert)
        )
        first_id, second_id = (ranked[0], ranked[1]) if saw_non_nan else (0, 1)
        first = row[first_id] if not math.isnan(row[first_id]) else -math.inf
        second = row[second_id] if not math.isnan(row[second_id]) else -math.inf
        if not saw_non_nan or first == second:
            first_weight, second_weight = 0.5, 0.5
        elif first == math.inf or second == -math.inf:
            first_weight, second_weight = 1.0, 0.0
        else:
            relative = math.exp(second - first)
            first_weight = 1.0 / (1.0 + relative)
            second_weight = relative * first_weight
        ids[token] = torch.tensor((first_id, second_id), dtype=torch.int32)
        weights[token] = torch.tensor((first_weight, second_weight), dtype=torch.float32)
    return ids, weights


def validate_case(tokens: int, experts: int, seed: int, mode: str) -> dict[str, float]:
    host = make_input(tokens, experts, seed, mode)
    expected_ids, expected_weights = reference(host)
    logits = host.cuda()
    ids = torch.empty((tokens, 2), dtype=torch.int32, device="cuda")
    weights = torch.empty((tokens, 2), dtype=torch.float32, device="cuda")
    launch(logits, ids, weights)
    torch.cuda.synchronize()
    actual_ids = ids.cpu()
    actual_weights = weights.cpu()
    if not torch.equal(actual_ids, expected_ids):
        raise AssertionError(f"ids mismatch for T={tokens}, E={experts}, mode={mode}")
    if not torch.allclose(actual_weights, expected_weights, atol=1.0e-6, rtol=1.0e-6):
        error = torch.max(torch.abs(actual_weights - expected_weights)).item()
        raise AssertionError(f"weights mismatch ({error}) for T={tokens}, E={experts}, mode={mode}")
    if not torch.all(torch.isfinite(actual_weights)) or not torch.all(actual_weights >= 0):
        raise AssertionError("weights violate finite/nonnegative invariants")
    sums = actual_weights.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("weights do not sum to one")
    return {
        "max_abs_error": torch.max(torch.abs(actual_weights - expected_weights)).item(),
        "max_rel_error": torch.max(
            torch.abs(actual_weights - expected_weights) / torch.clamp(torch.abs(expected_weights), min=1.0e-30)
        ).item(),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "platform": "wsl2",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_uuid": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip(),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
    }


def benchmark_case(tokens: int, experts: int, seed: int, mode: str, warmup: int,
                   samples: int, repeats: int) -> tuple[list[float], dict[str, float]]:
    host = make_input(tokens, experts, seed, mode)
    logits = host.cuda()
    ids = torch.empty((tokens, 2), dtype=torch.int32, device="cuda")
    weights = torch.empty((tokens, 2), dtype=torch.float32, device="cuda")
    launch(logits, ids, weights)
    torch.cuda.synchronize()
    for _ in range(warmup):
        for _ in range(repeats):
            launch(logits, ids, weights)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    timings: list[float] = []
    for _ in range(samples):
        start.record()
        for _ in range(repeats):
            launch(logits, ids, weights)
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop) * 1000.0 / repeats)
    validation = validate_case(tokens, experts, seed, mode)
    return timings, validation


def command_correctness(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")
    seeds = (20260803, 20260804, 20260805)
    for experts in range(2, 65):
        for index, tokens in enumerate((1, 3, 33)):
            validate_case(tokens, experts, seeds[index], "random")
    for mode in (
        "duplicate_max", "ties", "signed_zero", "all_nan", "mixed_nan_neg_inf",
        "single_infinity", "multiple_infinity", "extreme",
    ):
        validate_case(33, 63, 20260803, mode)
    print("PASS Triton Top-K correctness: E=2..64, three T/seeds, semantic edge modes")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "raggedroute.triton_topk_suite.v1":
        raise ValueError("unsupported Triton suite")
    if not 1 <= args.process_run <= int(config["process_runs"]):
        raise ValueError("process-run outside declared range")
    if args.output.exists():
        raise FileExistsError(args.output)
    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("release Triton benchmark requires a clean worktree")
    env = environment()
    records = []
    for experts in config["experts"]:
        for tokens in config["tokens"]:
            timings, validation = benchmark_case(
                tokens, experts, int(config["seed"]), config["input_mode"],
                int(config["warmup"]), int(config["samples"]), int(config["kernel_repeats"]),
            )
            mean = statistics.fmean(timings)
            stddev = statistics.pstdev(timings)
            logical_bytes = 4 * tokens * experts + 16 * tokens
            p50 = percentile(timings, 0.50)
            records.append({
                "schema_version": "raggedroute.triton_benchmark.v1",
                "run_id": args.run_id,
                "suite_id": config["suite_id"],
                "process_run": args.process_run,
                "operator": "topk_gate",
                "variant": "triton_row_top2",
                "measurement_level": config["measurement_level"],
                "cache_mode": config["cache_mode"],
                "seed": config["seed"],
                "warmup": config["warmup"],
                "samples": config["samples"],
                "kernel_repeats": config["kernel_repeats"],
                "case_config": {"T": tokens, "E": experts, "dtype": "fp32", "top_k": 2,
                                "input_mode": config["input_mode"]},
                "variant_config": {"implementation_category": "wsl2_triton_auxiliary",
                                   "implementation_version": "raggedroute.triton.topk.v1",
                                   "dependency_revision": f"triton@{triton.__version__}",
                                   "algorithm_id": "row_next_power_of_two_two_argmax",
                                   "math_mode": "strict_fp32", "workspace_bytes": 0,
                                   "launch_count": 1, "num_warps": 1},
                "environment": {**env, "build_git_sha": sha[:12]},
                "timing": {"sample_semantics": "event_batch_elapsed_divided_by_repeats",
                           "raw_batch_mean_samples_us": timings, "p50_us": p50,
                           "p90_us": percentile(timings, 0.90), "p95_us": percentile(timings, 0.95),
                           "mean_us": mean, "stddev_us": stddev, "cv": stddev / mean},
                "throughput": {"rows_per_second": tokens * 1.0e6 / p50,
                               "effective_logical_gbps": logical_bytes / p50 / 1000.0},
                "validation": {"ok": True, **validation},
                "workspace_bytes": 0,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                           encoding="utf-8")
    manifest = {
        "schema_version": "raggedroute.triton_run_manifest.v1",
        "run_id": args.run_id,
        "suite": config,
        "process_run": args.process_run,
        "repo_commit": sha,
        "environment": env,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def command_profile_once(args: argparse.Namespace) -> int:
    validate_case(args.tokens, args.experts, args.seed, args.input_mode)
    host = make_input(args.tokens, args.experts, args.seed, args.input_mode)
    logits = host.cuda()
    ids = torch.empty((args.tokens, 2), dtype=torch.int32, device="cuda")
    weights = torch.empty((args.tokens, 2), dtype=torch.float32, device="cuda")
    for _ in range(args.warmup):
        launch(logits, ids, weights)
    torch.cuda.synchronize()
    launch(logits, ids, weights)
    torch.cuda.synchronize()
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    correctness = commands.add_parser("correctness")
    correctness.set_defaults(handler=command_correctness)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--config", required=True, type=pathlib.Path)
    benchmark.add_argument("--output", required=True, type=pathlib.Path)
    benchmark.add_argument("--run-id", required=True)
    benchmark.add_argument("--process-run", required=True, type=int)
    benchmark.set_defaults(handler=command_benchmark)
    profile = commands.add_parser("profile-once")
    profile.add_argument("--tokens", type=int, required=True)
    profile.add_argument("--experts", type=int, required=True)
    profile.add_argument("--seed", type=int, default=20260803)
    profile.add_argument("--input-mode", default="random")
    profile.add_argument("--warmup", type=int, default=20)
    profile.set_defaults(handler=command_profile_once)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
