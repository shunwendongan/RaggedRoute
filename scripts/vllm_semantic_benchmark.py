#!/usr/bin/env python3
"""Run a vLLM component-semantic MoE adapter under a pinned container.

The worker keeps vLLM's production semantics for top-k routing, alignment and
fused MoE GEMM, while adapting tensor layouts and the timing contract to
RaggedRoute's seven-stage ``chain_from_tokens`` benchmark.  Upstream source is
archived separately under ``third_party/vllm_moe`` with its Apache-2.0 headers.
"""

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
from typing import Any

import torch
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import vllm_topk_softmax


ROOT = pathlib.Path(__file__).resolve().parents[1]
VARIANT = "vllm_component_semantic_adapter"
VLLM_SOURCE_REF = os.environ.get("VLLM_SOURCE_REF", "v0.26.0")
VLLM_SOURCE_TAG_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
VLLM_IMAGE_DIGEST = "sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
LEVEL_NAME = "L3_chain_steady"


def git_output(*args: str) -> str:
    if args == ("rev-parse", "--short=12", "HEAD") and os.environ.get("RAGGEDROUTE_BUILD_GIT_SHA"):
        return os.environ["RAGGEDROUTE_BUILD_GIT_SHA"][:12]
    if args == ("status", "--porcelain") and os.environ.get("RAGGEDROUTE_BUILD_GIT_DIRTY") is not None:
        return "container-mounted-worktree" if os.environ["RAGGEDROUTE_BUILD_GIT_DIRTY"] == "true" else ""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
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
        library = ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if library.cuDriverGetVersion(ctypes.byref(value)) != 0:
            return "unavailable"
        return value.value
    except (AttributeError, OSError):
        return "unavailable"


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "build_git_sha": git_output("rev-parse", "--short=12", "HEAD"),
        "build_git_dirty": "true" if git_output("status", "--porcelain") else "false",
        "build_type": "Release",
        "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid).replace("-", "").lower(),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "global_memory_bytes": properties.total_memory,
        "cuda_driver": cuda_driver_version(),
        "cuda_runtime": torch._C._cuda_getCompiledVersion(),
        "cuda_compiler": f"Triton {__import__('triton').__version__}",
        "torch": torch.__version__,
        "triton": __import__('triton').__version__,
        "vllm": __import__('vllm').__version__,
        "vllm_source_ref": VLLM_SOURCE_REF,
        "vllm_source_tag_commit": VLLM_SOURCE_TAG_COMMIT,
        "vllm_image_digest": VLLM_IMAGE_DIGEST,
    }


def make_floats(shape: tuple[int, ...], seed: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator)


def top2_reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids, weights = [], []
    for row in logits.tolist():
        ranked = sorted(enumerate(row), key=lambda item: (-item[1], item[0]))[:2]
        first, second = ranked
        relative = math.exp(second[1] - first[1])
        first_weight = 1.0 / (1.0 + relative)
        ids.append([first[0], second[0]])
        weights.append([first_weight, relative * first_weight])
    return torch.tensor(ids, dtype=torch.int32), torch.tensor(weights, dtype=torch.float32)


def validation_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    difference = (actual.double() - expected.double()).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    denominator = expected.double().abs().clamp_min(1e-30)
    max_rel = float((difference / denominator).max().item()) if difference.numel() else 0.0
    ok = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
    return {"ok": ok, "message": "matched independent reference" if ok else "floating output mismatch", "max_abs_error": max_abs, "max_rel_error": max_rel}


@dataclass
class Prepared:
    params: dict[str, Any]
    work: dict[str, Any]
    x: torch.Tensor
    router_weights: torch.Tensor
    expert_weights: torch.Tensor
    logits: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    token_expert_indices: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    fused_output: torch.Tensor
    result: torch.Tensor
    logits_expected: torch.Tensor
    ids_expected: torch.Tensor
    route_weights_expected: torch.Tensor
    output_expected: torch.Tensor
    sorted_ids_expected: torch.Tensor
    aligned_expert_ids_expected: torch.Tensor
    padded_routes_expected: int
    block_size: int = 32

    def launch(self) -> None:
        torch.mm(self.x, self.router_weights, out=self.logits)
        vllm_topk_softmax(self.topk_weights, self.topk_ids, self.token_expert_indices, self.logits, True)
        ops.moe_align_block_size(
            self.topk_ids, self.params["E"], self.block_size,
            self.sorted_token_ids, self.expert_ids, self.num_tokens_post_padded, None,
        )
        config = {
            "BLOCK_SIZE_M": self.block_size,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        }
        invoke_fused_moe_triton_kernel(
            self.x, self.expert_weights, self.fused_output,
            None, None, self.topk_weights, self.sorted_token_ids,
            self.expert_ids, self.num_tokens_post_padded, True, 2,
            config, tl.float32, False, False, False, False, False,
        )
        torch.sum(self.fused_output, dim=1, out=self.result)

    def validate(self) -> dict[str, Any]:
        if not torch.equal(self.topk_ids.cpu(), self.ids_expected):
            return {"ok": False, "message": "vLLM Top-K ids differ", "max_abs_error": None, "max_rel_error": None}
        route_check = validation_close(self.topk_weights.cpu(), self.route_weights_expected, 2e-5, 2e-5)
        if not route_check["ok"]:
            route_check["message"] = "vLLM route weights differ"
            return route_check
        padded_routes = int(self.num_tokens_post_padded.cpu().item())
        if padded_routes != self.padded_routes_expected:
            return {"ok": False, "message": "vLLM alignment padded-route count differs", "max_abs_error": None, "max_rel_error": None}
        blocks = padded_routes // self.block_size
        sorted_positions = self.sorted_token_ids[:padded_routes].cpu().tolist()
        if sorted(sorted_positions) != list(range(self.params["R"])) + [self.params["R"]] * (padded_routes - self.params["R"]):
            return {"ok": False, "message": "vLLM alignment does not contain every route exactly once with sentinel padding", "max_abs_error": None, "max_rel_error": None}
        routed_experts = self.ids_expected.reshape(-1).tolist()
        for block, expert in enumerate(self.expert_ids[:blocks].cpu().tolist()):
            if expert < 0 or expert >= self.params["E"]:
                return {"ok": False, "message": "vLLM alignment emitted an invalid block expert id", "max_abs_error": None, "max_rel_error": None}
            positions = sorted_positions[block * self.block_size:(block + 1) * self.block_size]
            if any(position != self.params["R"] and routed_experts[position] != expert for position in positions):
                return {"ok": False, "message": "vLLM sorted route expert does not match its alignment block", "max_abs_error": None, "max_rel_error": None}
        output_check = validation_close(self.result.cpu(), self.output_expected, 2e-4, 4e-4)
        if not output_check["ok"]:
            output_check["message"] = "vLLM final output differs"
            return output_check
        return {
            "ok": True,
            "message": "vLLM Top-K, route positions, alignment structure, fused grouped GEMM and weighted reduction matched independent reference",
            "max_abs_error": max(route_check["max_abs_error"], output_check["max_abs_error"]),
            "max_rel_error": max(route_check["max_rel_error"], output_check["max_rel_error"]),
        }


def prepare(params: dict[str, Any], seed: int) -> Prepared:
    tokens, experts, hidden, output = params["T"], params["E"], params["K"], params["N"]
    topk = 2
    x_host = make_floats((tokens, hidden), seed)
    router_host = make_floats((hidden, experts), seed + 1)
    # vLLM's fused kernel consumes [E, N, K], unlike RaggedRoute's [E, K, N].
    expert_host = make_floats((experts, hidden, output), seed + 2)
    expert_vllm_host = expert_host.permute(0, 2, 1).contiguous()
    logits_expected = (x_host.double() @ router_host.double()).float()
    ids_expected, route_weights_expected = top2_reference(logits_expected)
    sorted_ids, aligned_experts = [], []
    flattened_ids = ids_expected.reshape(-1).tolist()
    for expert in range(experts):
        assigned = [route for route, value in enumerate(flattened_ids) if value == expert]
        padded = ((len(assigned) + 31) // 32) * 32
        sorted_ids.extend(assigned)
        sorted_ids.extend([tokens * topk] * (padded - len(assigned)))
        aligned_experts.extend([expert] * (padded // 32))
    sorted_ids_expected = torch.tensor(sorted_ids, dtype=torch.int32)
    aligned_expert_ids_expected = torch.tensor(aligned_experts, dtype=torch.int32)
    output_expected = torch.zeros((tokens, output), dtype=torch.float32)
    for token in range(tokens):
        for rank in range(topk):
            expert = int(ids_expected[token, rank])
            output_expected[token] += (
                (x_host[token].double() @ expert_host[expert].double()).float()
                * route_weights_expected[token, rank]
            )
    x, router, weights = x_host.cuda(), router_host.cuda(), expert_vllm_host.cuda()
    logits = torch.empty((tokens, experts), device="cuda", dtype=torch.float32)
    topk_weights = torch.empty((tokens, topk), device="cuda", dtype=torch.float32)
    topk_ids = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    routes = tokens * topk
    max_padded = routes + experts * (32 - 1)
    sorted_token_ids = torch.empty((max_padded,), device="cuda", dtype=torch.int32)
    expert_ids = torch.empty(((max_padded + 31) // 32,), device="cuda", dtype=torch.int32)
    num_tokens_post_padded = torch.empty((1,), device="cuda", dtype=torch.int32)
    fused_output = torch.empty((tokens, topk, output), device="cuda", dtype=torch.float32)
    result = torch.empty((tokens, output), device="cuda", dtype=torch.float32)
    logical_bytes = float(4 * (tokens * hidden + hidden * experts + tokens * experts + routes * hidden + experts * hidden * output + tokens * topk * output + tokens * output))
    flops = float(2 * tokens * hidden * experts + 2 * routes * hidden * output + (2 * topk - 1) * tokens * output)
    return Prepared(
        params={**params, "R": routes, "top_k": topk, "dtype": "fp32", "distribution": "router_projection_random", "chain_entry": "tokens", "vllm_alignment_block": 32},
        work={"logical_bytes": logical_bytes, "flops": flops, "operator_metrics": {"kernel_launches": 6, "vllm_alignment_included": True, "vllm_padding_upper_bound": max_padded}},
        x=x, router_weights=router, expert_weights=weights, logits=logits, topk_weights=topk_weights,
        topk_ids=topk_ids, token_expert_indices=token_expert_indices, sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids, num_tokens_post_padded=num_tokens_post_padded, fused_output=fused_output,
        result=result, logits_expected=logits_expected, ids_expected=ids_expected,
        route_weights_expected=route_weights_expected, output_expected=output_expected,
        sorted_ids_expected=sorted_ids_expected, aligned_expert_ids_expected=aligned_expert_ids_expected,
        padded_routes_expected=len(sorted_ids),
    )


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (8, 6):
        raise RuntimeError(f"vLLM semantic benchmark targets sm86, found sm{props.major}{props.minor}")
    params = {name: int(value) for name, value in (item.split("=", 1) for item in args.param)}
    prepared = prepare(params, args.seed)
    prepared.launch()
    torch.cuda.synchronize()
    validation = prepared.validate()
    if not validation["ok"]:
        raise RuntimeError(validation["message"])
    for _ in range(args.warmup):
        prepared.launch()
    torch.cuda.synchronize()
    if args.profile_once:
        torch.cuda.cudart().cudaProfilerStart()
        prepared.launch()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return None
    samples = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.samples):
        start.record()
        for _ in range(args.kernel_repeats):
            prepared.launch()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop) * 1000.0 / args.kernel_repeats))
    record = {
        "schema_version": "raggedroute.benchmark.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": args.run_id, "case_id": args.case_id, "operator": "chain_from_tokens",
        "variant": VARIANT, "measurement_level": LEVEL_NAME, "protocol": args.protocol,
        "cache_mode": args.cache_mode, "warmup": args.warmup, "kernel_repeats": args.kernel_repeats,
        "samples": args.samples, "process_run": args.process_run, "seed": args.seed,
        "excluded_steps": ["input_generation", "cpu_reference", "h2d_copy", "workspace_allocation", "jit_compilation", "autotune"],
        "case_config": prepared.params,
        "variant_config": {
            "implementation_category": "vllm_component_semantic_adapter",
            "implementation_version": "vllm.component_moe.sm86.fp32.v2",
            "upstream_repository": "https://github.com/vllm-project/vllm",
            "source_ref": VLLM_SOURCE_REF,
            "source_tag_commit": VLLM_SOURCE_TAG_COMMIT,
            "image_digest": VLLM_IMAGE_DIGEST,
            "executed_vllm_components": ["vllm_topk_softmax", "moe_align_block_size", "invoke_fused_moe_triton_kernel"],
            "non_vllm_components": ["torch.mm router projection", "torch.sum weighted reduction"],
            "math_mode": "strict_fp32", "layout_adaptation": "RaggedRoute [E,K,N] weights transposed to vLLM [E,N,K]",
            "vllm_alignment": "device-side moe_align_block_size with block_size=32; padding included in L3",
            "grouped_kernel": "vllm fused_moe_kernel Triton component path",
            "production_full_ffn": False,
            "sm86_config_policy": "fixed compatibility tile 32x32x32; not vLLM production config lookup",
        },
        "environment": environment(), "work": {**prepared.work, "effective_gbps_batch_p50": None, "tflops_batch_p50": None},
        "timing": summarize(samples), "validation": validation,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="smoke", choices=["smoke", "release"])
    parser.add_argument("--cache-mode", default="warm", choices=["warm"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--kernel-repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--process-run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--run-id", default="vllm-semantic-standalone")
    parser.add_argument("--case-id", default="vllm-semantic-standalone")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--profile-once", action="store_true")
    args = parser.parse_args()
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
