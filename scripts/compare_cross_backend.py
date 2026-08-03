#!/usr/bin/env python3
"""Compare cross-toolchain CUDA and Triton baselines without promotion semantics."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] * (1 - (position - low)) + ordered[high] * (position - low)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("missing benchmark records")
    process_runs = {record["process_run"] for record in records}
    if len(process_runs) != len(records):
        raise ValueError("duplicate process records")
    raw = [sample for record in records for sample in record["timing"]["raw_batch_mean_samples_us"]]
    medians = [record["timing"]["batch_mean_us_p50"] for record in records]
    mean = statistics.fmean(raw)
    return {"median_of_process_medians_us": statistics.median(medians), "all_samples_p95_us": percentile(raw, 0.95), "all_samples_cv": 0.0 if mean == 0 else statistics.pstdev(raw) / mean, "process_runs": len(process_runs), "record": records[0]}


def verify_pair(reference: dict[str, Any], candidate: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    ref, cand = reference["record"], candidate["record"]
    for field in ("operator", "measurement_level", "protocol", "cache_mode", "warmup", "kernel_repeats", "samples", "seed"):
        if ref[field] != cand[field]:
            raise ValueError(f"{case['id']}: mismatched {field}")
    expected_level = {"l1": "L1_kernel_body", "l2": "L2_operator_steady"}
    if ref["measurement_level"] not in {expected_level[level] for level in case["levels"]}:
        raise ValueError(f"{case['id']}: undeclared measurement level")
    for name, value in case.get("params", {}).items():
        if ref["case_config"].get(name) != value or cand["case_config"].get(name) != value:
            raise ValueError(f"{case['id']}: parameter {name} does not match manifest")
    for field in ("gpu_uuid", "gpu_name", "compute_capability", "cuda_driver", "build_git_sha", "build_type"):
        if ref["environment"].get(field) != cand["environment"].get(field):
            raise ValueError(f"{case['id']}: mismatched environment.{field}")
    if ref["variant_config"].get("math_mode") != cand["variant_config"].get("math_mode"):
        raise ValueError(f"{case['id']}: mismatched strict math mode")
    # C++ adapters name their host-side setup more specifically (for example
    # ``route_generation`` or ``count_generation``), while Triton consistently
    # reports it as input generation.  The comparable timing boundary is that
    # transfers and workspace allocation are outside both timed regions; the
    # full exclusion lists remain in the emitted records for audit.
    required_exclusions = {"h2d_copy", "workspace_allocation"}
    if not required_exclusions <= set(ref["excluded_steps"]) or not required_exclusions <= set(cand["excluded_steps"]):
        raise ValueError(f"{case['id']}: missing required excluded work")
    return {"reference_toolchain": {"cuda_compiler": ref["environment"].get("cuda_compiler"), "cuda_runtime": ref["environment"].get("cuda_runtime"), "compiler_stack": ref["variant_config"].get("compiler_stack")}, "candidate_toolchain": {"cuda_compiler": cand["environment"].get("cuda_compiler"), "cuda_runtime": cand["environment"].get("cuda_runtime"), "compiler_stack": cand["variant_config"].get("compiler_stack")}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "raggedroute.cross_backend_manifest.v1":
        raise ValueError("manifest must be a cross-backend manifest")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or any(record.get("schema_version") != "raggedroute.benchmark.v1" or not record.get("validation", {}).get("ok") for record in records):
        raise ValueError("input must contain successful benchmark.v1 records")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["case_id"], record["variant"], record["measurement_level"])].append(record)
    comparisons = []
    for case in manifest["suite"]["cases"]:
        reference_variant = next(variant for variant in case["variants"] if variant.get("reference_baseline") is True)
        for level in case["levels"]:
            level_name = {"l1": "L1_kernel_body", "l2": "L2_operator_steady"}[level]
            reference_records = groups[(case["id"], reference_variant["name"], level_name)]
            reference = summarize(reference_records)
            for variant in case["variants"]:
                if variant is reference_variant:
                    continue
                candidate_records = groups[(case["id"], variant["name"], level_name)]
                if not candidate_records:
                    raise ValueError(f"{case['id']}: missing records for {variant['name']} at {level_name}")
                candidate = summarize(candidate_records)
                toolchains = verify_pair(reference, candidate, case)
                comparisons.append({"case_id": case["id"], "operator": case["operator"], "measurement_level": level_name, "reference_variant": reference_variant["name"], "candidate_variant": variant["name"], "reference_latency_us": reference["median_of_process_medians_us"], "candidate_latency_us": candidate["median_of_process_medians_us"], "candidate_vs_reference_speedup": reference["median_of_process_medians_us"] / candidate["median_of_process_medians_us"], "reference_p95_us": reference["all_samples_p95_us"], "candidate_p95_us": candidate["all_samples_p95_us"], "reference_cv": reference["all_samples_cv"], "candidate_cv": candidate["all_samples_cv"], "reference_process_runs": reference["process_runs"], "candidate_process_runs": candidate["process_runs"], "toolchain_differences": toolchains, "promotion_eligible": False, "promotion_ineligibility_reason": "cross-backend CUDA runtime/compiler stacks are intentionally compared as reference evidence only"})
    result = {"schema_version": "raggedroute.cross_backend_comparison.v1", "source": str(args.input.resolve()), "manifest": str(args.manifest.resolve()), "suite_id": manifest["suite"].get("suite_id"), "promotion_eligible": False, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"error: {error}")
