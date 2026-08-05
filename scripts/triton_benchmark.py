#!/usr/bin/env python3
"""Correctness-gated Triton benchmark worker for RaggedRoute's seven operators."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

import torch
import triton


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dense_gemm.triton import launch_dense_gemm
from grouped_gemm.triton import launch_grouped_gemm
from histogram.triton import launch_histogram
from permute.triton import launch_permute_rows, launch_route_map, launch_token_permute
from scan.triton import launch_exclusive_scan
from topk_gate.triton import launch_topk_gate
from unpermute.triton import launch_unpermute


TRITON_VARIANT = "triton_reference"
LEVEL_NAMES = {
    "l1": "L1_kernel_body",
    "l2": "L2_operator_steady",
    "l3": "L3_chain_steady",
}
ALGORITHMS = {
    "dense_gemm": "triton_grouped_pid_ieee_fp32",
    "topk_gate": "triton_row_top2_selected_softmax",
    "histogram": "triton_masked_local_histogram_atomic_merge",
    "exclusive_scan": "triton_single_program_cumsum",
    "token_permute": "triton_per_expert_scan_stable_permute",
    "grouped_gemm": "triton_grid_z_ragged_ieee_fp32",
    "unpermute": "triton_token_owned_weighted_gather",
    "chain_from_tokens": "triton_seven_operator_chain_from_tokens",
}


def git_output(*args: str) -> str:
    if args == ("rev-parse", "--short=12", "HEAD") and os.environ.get(
        "RAGGEDROUTE_BUILD_GIT_SHA"
    ):
        return os.environ["RAGGEDROUTE_BUILD_GIT_SHA"][:12]
    if args == ("status", "--porcelain") and os.environ.get(
        "RAGGEDROUTE_BUILD_GIT_DIRTY"
    ) is not None:
        return "container-mounted-worktree" if os.environ["RAGGEDROUTE_BUILD_GIT_DIRTY"] == "true" else ""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples)
    return {
        "sample_semantics": "event_batch_elapsed_divided_by_repeats",
        "batch_mean_us_mean": mean,
        "batch_mean_us_stddev": stddev,
        "cv": 0.0 if mean == 0 else stddev / mean,
        "batch_mean_us_min": min(samples),
        "batch_mean_us_p50": percentile(samples, 0.50),
        "batch_mean_us_p90": percentile(samples, 0.90),
        "batch_mean_us_p95": percentile(samples, 0.95),
        "raw_batch_mean_samples_us": samples,
    }


def cuda_driver_version() -> int | str:
    try:
        library = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if library.cuDriverGetVersion(ctypes.byref(value)) != 0:
            return "unavailable"
        return value.value
    except (AttributeError, OSError):
        return "unavailable"


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid).replace("-", "").lower()
    return {
        "build_git_dirty": "true" if git_output("status", "--porcelain") else "false",
        "build_git_sha": git_output("rev-parse", "--short=12", "HEAD"),
        "build_type": "Release",
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_compiler": f"Triton {triton.__version__}",
        "cuda_driver": cuda_driver_version(),
        "cuda_runtime": torch._C._cuda_getCompiledVersion(),
        "device_ordinal": 0,
        "global_memory_bytes": properties.total_memory,
        "gpu_name": properties.name,
        "gpu_uuid": uuid,
        "pci_bus_id": properties.pci_bus_id,
        "pci_device_id": properties.pci_device_id,
        "pci_domain_id": properties.pci_domain_id,
        "sm_count": properties.multi_processor_count,
        "warp_size": properties.warp_size,
    }


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_params(entries: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid --param {entry!r}")
        name, value = entry.split("=", 1)
        if not name or name in result:
            raise ValueError(f"duplicate or empty parameter {name!r}")
        result[name] = parse_value(value)
    return result


def option(params: dict[str, Any], name: str, default: Any) -> Any:
    return params.get(name, default)


def make_floats(shape: tuple[int, ...], seed: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator)


def make_route_ids(tokens: int, top_k: int, experts: int, distribution: str, zipf_s: float, seed: int) -> torch.Tensor:
    if experts < top_k:
        raise ValueError("route generation requires E >= top_k")
    if tokens == 0:
        return torch.empty((0, top_k), dtype=torch.int32)
    if distribution == "uniform":
        probabilities = torch.ones(experts, dtype=torch.float64)
    elif distribution == "zipf":
        probabilities = torch.arange(1, experts + 1, dtype=torch.float64).pow(-zipf_s)
    else:
        raise ValueError(f"unsupported distribution {distribution!r}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = [torch.multinomial(probabilities, top_k, replacement=False, generator=generator) for _ in range(tokens)]
    return torch.stack(rows).to(torch.int32)


def offsets_from_ids(ids: torch.Tensor, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(ids.flatten().to(torch.int64), minlength=experts).to(torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)))
    return counts, offsets


def top2_reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids: list[list[int]] = []
    weights: list[list[float]] = []
    for row in logits.tolist():
        ordered = sorted(
            ((-math.inf if math.isnan(value) else value, index) for index, value in enumerate(row)),
            key=lambda item: (-item[0], item[1]),
        )
        saw_non_nan = any(not math.isnan(value) for value in row)
        first, second = ordered[:2]
        ids.append([first[1], second[1]] if saw_non_nan else [0, 1])
        if not saw_non_nan or first[0] == second[0]:
            weights.append([0.5, 0.5])
        elif first[0] == math.inf or second[0] == -math.inf:
            weights.append([1.0, 0.0])
        else:
            relative = math.exp(second[0] - first[0])
            first_weight = 1.0 / (1.0 + relative)
            weights.append([first_weight, relative * first_weight])
    return torch.tensor(ids, dtype=torch.int32), torch.tensor(weights, dtype=torch.float32)


@dataclass
class PreparedCase:
    operator: str
    launch: Callable[[str], None]
    prepare_sample: Callable[[str], None]
    validate: Callable[[], dict[str, Any]]
    case_config: dict[str, Any]
    variant_details: dict[str, Any]
    work: dict[str, Any]
    workspace_bytes: int = 0

    def excluded_steps(self, level: str) -> list[str]:
        excluded = ["input_generation", "cpu_reference", "h2d_copy", "workspace_allocation", "jit_compilation", "autotune"]
        if self.operator == "histogram" and level == "l1":
            excluded.append("counts_reset")
        if self.operator == "token_permute" and level == "l1":
            excluded.append("mapping_generation")
        return excluded


def prepare_dense(params: dict[str, Any], seed: int) -> PreparedCase:
    m, n, k = int(option(params, "M", 256)), int(option(params, "N", 256)), int(option(params, "K", 256))
    a_host, b_host = make_floats((m, k), seed), make_floats((k, n), seed + 1)
    expected = (a_host.double() @ b_host.double()).float()
    a, b, out = a_host.cuda(), b_host.cuda(), torch.empty((m, n), device="cuda", dtype=torch.float32)
    return PreparedCase(
        "dense_gemm",
        lambda level: launch_dense_gemm(a, b, out),
        lambda level: None,
        lambda: validation_close(out.cpu(), expected, 2e-5, 2e-5),
        {"K": k, "M": m, "N": n, "accumulator_dtype": "fp32", "alpha": 1.0, "beta": 0.0, "dtype": "fp32", "layout_a": "row_major", "layout_b": "row_major", "layout_c": "row_major"},
        {"tile_policy": "sm86_autotune", "input_precision": "ieee"},
        {"logical_bytes": float(4 * (m * k + k * n + m * n)), "flops": float(2 * m * n * k), "operator_metrics": {"output_elements": m * n}},
    )


def prepare_topk(params: dict[str, Any], seed: int) -> PreparedCase:
    tokens, experts = int(option(params, "T", 2048)), int(option(params, "E", 64))
    mode = str(option(params, "input_mode", "random"))
    logits_host = make_floats((tokens, experts), seed, -8.0, 8.0)
    if mode == "ties":
        logits_host.fill_(1.0)
    elif mode == "all_nan":
        logits_host.fill_(math.nan)
    elif mode == "infinities" and tokens:
        logits_host[:, 0] = math.inf
        logits_host[:, 1] = -math.inf
    elif mode != "random":
        raise ValueError(f"unsupported input_mode {mode!r}")
    expected_ids, expected_weights = top2_reference(logits_host)
    logits = logits_host.cuda()
    ids = torch.empty((tokens, 2), device="cuda", dtype=torch.int32)
    weights = torch.empty((tokens, 2), device="cuda", dtype=torch.float32)
    def validate() -> dict[str, Any]:
        actual_ids, actual_weights = ids.cpu(), weights.cpu()
        if not torch.equal(actual_ids, expected_ids):
            return {"ok": False, "message": "Top-2 ids differ from deterministic reference", "max_abs_error": None, "max_rel_error": None}
        return validation_close(actual_weights, expected_weights, 1e-6, 1e-6)
    return PreparedCase(
        "topk_gate", lambda level: launch_topk_gate(logits, ids, weights), lambda level: None, validate,
        {"T": tokens, "E": experts, "top_k": 2, "dtype": "fp32", "normalization": "selected_softmax", "tie_break": "lower_expert_id", "nan_policy": "negative_infinity_all_nan_0_1", "input_mode": mode},
        {"rows_per_program": 1, "expert_block": triton.next_power_of_2(experts)},
        {"logical_bytes": float(4 * tokens * experts + 8 * tokens * 2), "flops": 0.0, "operator_metrics": {"rows": tokens, "comparisons_lower_bound": tokens * experts}},
    )


def prepare_histogram(params: dict[str, Any], seed: int) -> PreparedCase:
    tokens, experts, top_k = int(option(params, "T", 2048)), int(option(params, "E", 64)), int(option(params, "top_k", 2))
    distribution, zipf_s = str(option(params, "distribution", "uniform")), float(option(params, "zipf_s", 0.0))
    ids_host = make_route_ids(tokens, top_k, experts, distribution, zipf_s, seed)
    expected, _ = offsets_from_ids(ids_host, experts)
    ids, counts = ids_host.cuda(), torch.empty((experts,), device="cuda", dtype=torch.int32)
    def launch(level: str) -> None:
        launch_histogram(ids, counts, reset=level == "l2")
    def prepare_sample(level: str) -> None:
        if level == "l1": counts.zero_()
    return PreparedCase(
        "histogram", launch, prepare_sample,
        lambda: validation_exact(counts.cpu(), expected, "counts match CPU bincount"),
        {"T": tokens, "E": experts, "top_k": top_k, "R": ids.numel(), "count_dtype": "int32", "distribution": distribution, "zipf_s": zipf_s, "max_count": int(expected.max().item()) if experts else 0},
        {"primitive": "tl.histogram", "output_reset": "inside_l2_call"},
        {"logical_bytes": float(4 * (ids.numel() + experts)), "flops": 0.0, "operator_metrics": {"histogram_input_items": ids.numel(), "histogram_bins": experts}},
    )


def prepare_scan(params: dict[str, Any], seed: int) -> PreparedCase:
    experts, routes = int(option(params, "E", 64)), int(option(params, "R", 4096))
    distribution, zipf_s = str(option(params, "distribution", "uniform")), float(option(params, "zipf_s", 0.0))
    ids = make_route_ids(routes, 1, experts, distribution, zipf_s, seed)
    counts_host, expected = offsets_from_ids(ids, experts)
    counts, offsets = counts_host.cuda(), torch.empty((experts + 1,), device="cuda", dtype=torch.int32)
    return PreparedCase(
        "exclusive_scan", lambda level: launch_exclusive_scan(counts, offsets), lambda level: None,
        lambda: validation_exact(offsets.cpu(), expected, "offsets match CPU reference"),
        {"E": experts, "R": routes, "count_dtype": "int32", "distribution": distribution, "zipf_s": zipf_s},
        {"primitive": "tl.cumsum", "programs": 1},
        {"logical_bytes": float(4 * (experts + experts + 1)), "flops": 0.0, "operator_metrics": {"integer_additions": experts}},
    )


def prepare_permute(params: dict[str, Any], seed: int) -> PreparedCase:
    tokens, experts, top_k, hidden = (int(option(params, "T", 512)), int(option(params, "E", 64)), int(option(params, "top_k", 2)), int(option(params, "K", 256)))
    distribution, zipf_s = str(option(params, "distribution", "uniform")), float(option(params, "zipf_s", 0.0))
    materialize = bool(option(params, "materialize_sorted_route", True))
    x_host = make_floats((tokens, hidden), seed)
    ids_host = make_route_ids(tokens, top_k, experts, distribution, zipf_s, seed + 1)
    _, offsets_host = offsets_from_ids(ids_host, experts)
    routes = tokens * top_k
    x, ids, offsets = x_host.cuda(), ids_host.cuda(), offsets_host.cuda()
    out = torch.empty((routes, hidden), device="cuda", dtype=torch.float32)
    route_pos = torch.empty((routes,), device="cuda", dtype=torch.int32)
    sorted_route = torch.empty_like(route_pos) if materialize else None
    launch_route_map(ids, offsets, route_pos, sorted_route)
    def launch(level: str) -> None:
        if level == "l1": launch_permute_rows(x, route_pos, out, top_k)
        else: launch_token_permute(x, ids, offsets, out, route_pos, sorted_route)
    def validate() -> dict[str, Any]:
        positions = route_pos.cpu()
        if sorted(positions.tolist()) != list(range(routes)):
            return {"ok": False, "message": "route_pos is not bijective", "max_abs_error": None, "max_rel_error": None}
        if sorted_route is not None and not torch.equal(sorted_route[route_pos].cpu(), torch.arange(routes, dtype=torch.int32)):
            return {"ok": False, "message": "sorted_route is not inverse", "max_abs_error": None, "max_rel_error": None}
        return validation_close(out[route_pos].cpu(), x_host.repeat_interleave(top_k, dim=0), 0.0, 0.0)
    return PreparedCase(
        "token_permute", launch, lambda level: None, validate,
        {"T": tokens, "E": experts, "top_k": top_k, "K": hidden, "R": routes, "dtype": "fp32", "distribution": distribution, "zipf_s": zipf_s, "materialize_sorted_route": materialize, "placement_order": "stable_within_expert", "input_boundary": "topk_ids_to_permuted_rows"},
        {"mapping": "per_expert_tl_cumsum", "copy_block": 256, "l1_boundary": "prepared_mapping_copy_only"},
        {"logical_bytes": float(8 * routes * hidden + 8 * routes + (4 * routes if materialize else 0)), "flops": 0.0, "operator_metrics": {"copied_rows": routes, "dynamic_mapping_included_l2": True}},
        workspace_bytes=0,
    )


def prepare_grouped(params: dict[str, Any], seed: int) -> PreparedCase:
    tokens, experts, top_k = int(option(params, "T", 512)), int(option(params, "E", 64)), int(option(params, "top_k", 2))
    hidden, output = int(option(params, "K", 128)), int(option(params, "N", 128))
    distribution, zipf_s = str(option(params, "distribution", "uniform")), float(option(params, "zipf_s", 0.0))
    ids = make_route_ids(tokens, top_k, experts, distribution, zipf_s, seed + 1)
    counts, offsets_host = offsets_from_ids(ids, experts)
    routes = tokens * top_k
    max_m = int(counts.max().item()) if experts else 0
    active = int((counts > 0).sum().item())
    x_host, weights_host = make_floats((routes, hidden), seed), make_floats((experts, hidden, output), seed + 2)
    expected = torch.empty((routes, output), dtype=torch.float32)
    for expert in range(experts):
        begin, end = int(offsets_host[expert]), int(offsets_host[expert + 1])
        if end > begin:
            expected[begin:end] = (x_host[begin:end].double() @ weights_host[expert].double()).float()
    x, weights, offsets, out = x_host.cuda(), weights_host.cuda(), offsets_host.cuda(), torch.empty_like(expected, device="cuda")
    return PreparedCase(
        "grouped_gemm", lambda level: launch_grouped_gemm(x, weights, offsets, out, max_m), lambda level: None,
        lambda: validation_close(out.cpu(), expected, 3e-5, 3e-5),
        {"T": tokens, "E": experts, "top_k": top_k, "R": routes, "K": hidden, "N": output, "dtype": "fp32", "accumulator_dtype": "fp32", "distribution": distribution, "zipf_s": zipf_s, "active_experts": active, "max_expert_tokens": max_m},
        {"scheduler": "grid_z_per_expert", "input_precision": "ieee", "tile_policy": "sm86_autotune"},
        {"logical_bytes": float(4 * (routes * hidden + experts * hidden * output + routes * output)), "flops": float(2 * routes * hidden * output), "operator_metrics": {"active_experts": active, "max_expert_tokens": max_m}},
    )


def prepare_unpermute(params: dict[str, Any], seed: int) -> PreparedCase:
    tokens, experts, top_k, output = (int(option(params, "T", 1024)), int(option(params, "E", 64)), int(option(params, "top_k", 2)), int(option(params, "N", 256)))
    distribution, zipf_s = str(option(params, "distribution", "uniform")), float(option(params, "zipf_s", 0.0))
    ids = make_route_ids(tokens, top_k, experts, distribution, zipf_s, seed + 1)
    flat = ids.flatten().to(torch.int64)
    order = torch.argsort(flat, stable=True)
    route_pos_host = torch.empty(tokens * top_k, dtype=torch.int32)
    route_pos_host[order] = torch.arange(tokens * top_k, dtype=torch.int32)
    y_host = make_floats((tokens * top_k, output), seed)
    raw_weights = make_floats((tokens, top_k), seed + 2, 0.1, 1.0)
    weights_host = raw_weights / raw_weights.sum(dim=1, keepdim=True)
    expected = torch.empty((tokens, output), dtype=torch.float32)
    for token in range(tokens):
        value = torch.zeros(output, dtype=torch.float64)
        for rank in range(top_k):
            route = token * top_k + rank
            value += y_host[int(route_pos_host[route])].double() * float(weights_host[token, rank])
        expected[token] = value.float()
    y, route_pos, weights, out = y_host.cuda(), route_pos_host.cuda(), weights_host.cuda(), torch.empty_like(expected, device="cuda")
    return PreparedCase(
        "unpermute", lambda level: launch_unpermute(y, route_pos, weights, out, top_k), lambda level: None,
        lambda: validation_close(out.cpu(), expected, 1e-6, 1e-6),
        {"T": tokens, "E": experts, "top_k": top_k, "R": tokens * top_k, "N": output, "dtype": "fp32", "accumulator_dtype": "fp32", "pointer_offset_elements": 0, "distribution": distribution, "zipf_s": zipf_s},
        {"ownership": "token_owned", "block_n": 256, "accumulation": "fp32"},
        {"logical_bytes": float(4 * (tokens * top_k * output + tokens * output + 2 * tokens * top_k)), "flops": float((2 * top_k - 1) * tokens * output), "operator_metrics": {"tokens": tokens, "top_k": top_k}},
    )


def prepare_chain_from_tokens(params: dict[str, Any], seed: int) -> PreparedCase:
    tokens = int(option(params, "T", 512))
    experts = int(option(params, "E", 64))
    top_k = int(option(params, "top_k", 2))
    hidden = int(option(params, "K", 128))
    output = int(option(params, "N", 128))
    distribution = str(option(params, "distribution", "uniform"))
    zipf_s = float(option(params, "zipf_s", 1.0))
    if top_k != 2:
        raise ValueError("chain_from_tokens requires top_k=2")
    if distribution != "uniform":
        raise ValueError("chain_from_tokens labels random router projection as uniform")
    routes = tokens * top_k

    x_host = make_floats((tokens, hidden), seed)
    expert_weights_host = make_floats((experts, hidden, output), seed + 1)
    router_weights_host = make_floats((hidden, experts), seed + 2)
    logits_expected = (x_host.double() @ router_weights_host.double()).float()
    ids_expected, route_weights_expected = top2_reference(logits_expected)
    counts_expected, offsets_expected = offsets_from_ids(ids_expected, experts)
    flat_ids = ids_expected.flatten().to(torch.int64)
    sorted_route_expected = torch.argsort(flat_ids, stable=True).to(torch.int32)
    route_pos_expected = torch.empty(routes, dtype=torch.int32)
    route_pos_expected[sorted_route_expected.to(torch.int64)] = torch.arange(
        routes, dtype=torch.int32
    )
    repeated_x = x_host.repeat_interleave(top_k, dim=0)
    x_permuted_expected = repeated_x[sorted_route_expected.to(torch.int64)]
    y_permuted_expected = torch.empty((routes, output), dtype=torch.float32)
    for expert in range(experts):
        begin = int(offsets_expected[expert])
        end = int(offsets_expected[expert + 1])
        if end > begin:
            y_permuted_expected[begin:end] = (
                x_permuted_expected[begin:end].double()
                @ expert_weights_host[expert].double()
            ).float()
    output_expected = torch.empty((tokens, output), dtype=torch.float32)
    for token in range(tokens):
        value = torch.zeros(output, dtype=torch.float64)
        for rank in range(top_k):
            route = token * top_k + rank
            value += (
                y_permuted_expected[int(route_pos_expected[route])].double()
                * float(route_weights_expected[token, rank])
            )
        output_expected[token] = value.float()

    x = x_host.cuda()
    router_weights = router_weights_host.cuda()
    expert_weights = expert_weights_host.cuda()
    logits = torch.empty((tokens, experts), device="cuda", dtype=torch.float32)
    ids = torch.empty((tokens, top_k), device="cuda", dtype=torch.int32)
    route_weights = torch.empty((tokens, top_k), device="cuda", dtype=torch.float32)
    counts = torch.empty((experts,), device="cuda", dtype=torch.int32)
    offsets = torch.empty((experts + 1,), device="cuda", dtype=torch.int32)
    x_permuted = torch.empty((routes, hidden), device="cuda", dtype=torch.float32)
    route_pos = torch.empty((routes,), device="cuda", dtype=torch.int32)
    y_permuted = torch.empty((routes, output), device="cuda", dtype=torch.float32)
    result = torch.empty((tokens, output), device="cuda", dtype=torch.float32)
    fp32_epsilon = torch.finfo(torch.float32).eps
    route_weight_tolerance = max(2.0e-5, 0.5 * fp32_epsilon * hidden)
    grouped_atol = max(3.0e-5, 4.0 * fp32_epsilon * hidden)
    output_atol = max(4.0e-5, 8.0 * fp32_epsilon * hidden)

    def launch(level: str) -> None:
        if level != "l3":
            raise ValueError("chain_from_tokens only supports l3")
        launch_dense_gemm(x, router_weights, logits)
        launch_topk_gate(logits, ids, route_weights)
        launch_histogram(ids, counts, reset=True)
        launch_exclusive_scan(counts, offsets)
        launch_token_permute(x, ids, offsets, x_permuted, route_pos, None)
        # Use R as a truthful worst-case launch bound.  No input-dependent
        # host max-M preparation is hidden outside the L3 interval.
        launch_grouped_gemm(
            x_permuted, expert_weights, offsets, y_permuted, routes
        )
        launch_unpermute(y_permuted, route_pos, route_weights, result, top_k)

    def validate() -> dict[str, Any]:
        checks = (
            (logits.cpu(), logits_expected, 2e-5, 2e-5, "router logits"),
            # The Top-2 kernel itself is tested at 1e-6.  In the chain its
            # logits come from strict-FP32 GEMM, while the independent oracle
            # forms logits in FP64 then rounds once, so permit propagated GEMM
            # rounding without weakening the exact ID/tie checks below.
            (
                route_weights.cpu(), route_weights_expected,
                route_weight_tolerance, route_weight_tolerance, "route weights",
            ),
            (x_permuted.cpu(), x_permuted_expected, 0.0, 0.0, "permuted rows"),
            (y_permuted.cpu(), y_permuted_expected, 1e-4, grouped_atol, "grouped GEMM"),
            (result.cpu(), output_expected, 1e-4, output_atol, "chain output"),
        )
        if not torch.equal(ids.cpu(), ids_expected):
            return {"ok": False, "message": "chain Top-2 ids differ", "max_abs_error": None, "max_rel_error": None}
        if not torch.equal(counts.cpu(), counts_expected):
            return {"ok": False, "message": "chain histogram differs", "max_abs_error": None, "max_rel_error": None}
        if not torch.equal(offsets.cpu(), offsets_expected):
            return {"ok": False, "message": "chain offsets differ", "max_abs_error": None, "max_rel_error": None}
        if not torch.equal(route_pos.cpu(), route_pos_expected):
            return {"ok": False, "message": "chain route_pos differs", "max_abs_error": None, "max_rel_error": None}
        max_abs = 0.0
        max_rel = 0.0
        for actual, expected, rtol, atol, name in checks:
            outcome = validation_close(actual, expected, rtol, atol)
            if not outcome["ok"]:
                outcome["message"] = f"{name} differs from independent reference"
                return outcome
            max_abs = max(max_abs, float(outcome["max_abs_error"]))
            max_rel = max(max_rel, float(outcome["max_rel_error"]))
        return {
            "ok": True,
            "message": "all seven stages matched independent references",
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
        }

    active_experts = int((counts_expected > 0).sum().item())
    logical_bytes = float(
        4 * (tokens * hidden + hidden * experts + tokens * experts)
        + 4 * tokens * experts
        + 8 * routes
        + 4 * (3 * experts + 1)
        + 8 * routes * hidden
        + 4 * (routes * hidden + routes * output + active_experts * hidden * output)
        + 4 * (routes * output + tokens * output)
    )
    flops = float(
        2 * tokens * hidden * experts
        + 2 * routes * hidden * output
        + (2 * top_k - 1) * tokens * output
    )
    return PreparedCase(
        "chain_from_tokens",
        launch,
        lambda level: None,
        validate,
        {
            "T": tokens,
            "E": experts,
            "top_k": top_k,
            "R": routes,
            "K": hidden,
            "N": output,
            "dtype": "fp32",
            "distribution": "router_projection_random",
            "zipf_s": zipf_s,
            "chain_entry": "tokens",
            "included_operator_count": 7,
            "active_experts": active_experts,
        },
        {
            "components": "dense_gemm,topk_gate,histogram,exclusive_scan,token_permute,grouped_gemm,unpermute",
            "component_variant": "triton_reference",
            "grouped_max_m_policy": "worst_case_R",
            "materialize_sorted_route": False,
            "input_precision": "ieee",
            "integer_math_mode": "exact_int32",
        },
        {
            "logical_bytes": logical_bytes,
            "flops": flops,
            "operator_metrics": {
                "kernel_launches": 9,
                "counts_reset_bytes": 4 * experts,
                "mapping_generation_included": True,
            },
        },
    )


PREPARERS = {
    "dense_gemm": prepare_dense,
    "topk_gate": prepare_topk,
    "histogram": prepare_histogram,
    "exclusive_scan": prepare_scan,
    "token_permute": prepare_permute,
    "grouped_gemm": prepare_grouped,
    "unpermute": prepare_unpermute,
    "chain_from_tokens": prepare_chain_from_tokens,
}


def validation_exact(actual: torch.Tensor, expected: torch.Tensor, message: str) -> dict[str, Any]:
    return {"ok": bool(torch.equal(actual, expected)), "message": message if torch.equal(actual, expected) else "exact output mismatch", "max_abs_error": 0.0 if torch.equal(actual, expected) else None, "max_rel_error": 0.0 if torch.equal(actual, expected) else None}


def validation_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    difference = (actual.double() - expected.double()).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    denominator = expected.double().abs().clamp_min(1e-30)
    max_rel = float((difference / denominator).max().item()) if difference.numel() else 0.0
    ok = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
    return {"ok": ok, "message": "matched independent reference" if ok else "floating output mismatch", "max_abs_error": max_abs, "max_rel_error": max_rel}


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 6):
        raise RuntimeError(f"validated Triton baseline target is sm86, found sm{properties.major}{properties.minor}")
    params = parse_params(args.param)
    if args.operator == "chain_from_tokens" and args.level != "l3":
        raise ValueError("chain_from_tokens only supports l3")
    if args.operator != "chain_from_tokens" and args.level == "l3":
        raise ValueError("l3 requires --suite chain_from_tokens")
    prepared = PREPARERS[args.operator](params, args.seed)
    prepared.prepare_sample(args.level)
    prepared.launch(args.level)
    torch.cuda.synchronize()
    validation = prepared.validate()
    if not validation["ok"]:
        raise RuntimeError(validation["message"])

    for _ in range(args.warmup):
        prepared.prepare_sample(args.level)
        prepared.launch(args.level)
    torch.cuda.synchronize()
    if args.profile_once:
        if args.profiler_api_capture:
            torch.cuda.cudart().cudaProfilerStart()
        prepared.prepare_sample(args.level)
        prepared.launch(args.level)
        torch.cuda.synchronize()
        if args.profiler_api_capture:
            torch.cuda.cudart().cudaProfilerStop()
        return None

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(args.samples):
        prepared.prepare_sample(args.level)
        start.record()
        for _ in range(args.kernel_repeats):
            prepared.launch(args.level)
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop) * 1000.0 / args.kernel_repeats))

    revision = git_output("rev-parse", "--short=12", "HEAD")
    variant_config = {
        "implementation_category": "in_tree_triton_reference",
        "implementation_version": "raggedroute.triton_reference.v1",
        "implementation_revision": revision,
        "dependency_revision": f"torch={torch.__version__};triton={triton.__version__}",
        "algorithm_id": ALGORITHMS[args.operator],
        "math_mode": "exact_int32" if args.operator == "exclusive_scan" else "strict_fp32",
        "compiler_stack": f"Triton {triton.__version__}; PyTorch CUDA {torch.version.cuda}",
        **prepared.variant_details,
    }
    record = {
        "schema_version": "raggedroute.benchmark.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": args.run_id,
        "case_id": args.case_id,
        "operator": args.operator,
        "variant": TRITON_VARIANT,
        "measurement_level": LEVEL_NAMES[args.level],
        "protocol": args.protocol,
        "cache_mode": args.cache_mode,
        "warmup": args.warmup,
        "kernel_repeats": args.kernel_repeats,
        "samples": args.samples,
        "process_run": args.process_run,
        "seed": args.seed,
        "workspace_bytes": prepared.workspace_bytes,
        "excluded_steps": prepared.excluded_steps(args.level),
        "case_config": prepared.case_config,
        "variant_config": variant_config,
        "environment": environment(),
        "work": {**prepared.work, "effective_gbps_batch_p50": None, "tflops_batch_p50": None},
        "timing": summarize(samples),
        "validation": validation,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--operator",
        choices=sorted(name for name in PREPARERS if name != "chain_from_tokens"),
    )
    target.add_argument("--suite", choices=["chain_from_tokens"])
    parser.add_argument("--variant", default=TRITON_VARIANT, choices=[TRITON_VARIANT])
    parser.add_argument("--level", default="l2", choices=sorted(LEVEL_NAMES))
    parser.add_argument("--protocol", default="smoke", choices=["smoke", "release"])
    parser.add_argument("--cache-mode", default="warm", choices=["warm"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--kernel-repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--process-run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--run-id", default="triton-standalone")
    parser.add_argument("--case-id", default="triton-standalone")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument("--profiler-api-capture", action="store_true")
    args = parser.parse_args()
    if args.suite is not None:
        args.operator = args.suite
    if min(args.warmup, args.kernel_repeats, args.samples, args.process_run) < 1:
        raise ValueError("warmup/repeats/samples/process-run must be positive")
    record = run(args)
    if record is not None:
        if args.output is None:
            print(json.dumps(record, ensure_ascii=False))
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a", encoding="utf-8", newline="\n") as stream:
                json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
            print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
