#!/usr/bin/env python3
"""Fail-closed comparison of paired RaggedRoute aggregate.v2 results."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any


ENVIRONMENT_FIELDS = (
    "gpu_uuid",
    "gpu_name",
    "compute_capability",
    "build_git_sha",
    "build_type",
    "cuda_compiler",
    "cuda_runtime",
    "cuda_driver",
)
PAIR_FIELDS = (
    "protocol",
    "warmup",
    "kernel_repeats",
    "samples",
    "seed",
    "excluded_steps",
)


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_field(mapping: dict[str, Any], field: str, context: str) -> Any:
    if field not in mapping:
        raise ValueError(f"{context} is missing {field}")
    return mapping[field]


def assert_pairable(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    context = f"{baseline['case_id']}: {baseline['variant']} vs {candidate['variant']}"
    for field in ("measurement_level", "cache_mode", *PAIR_FIELDS):
        if require_field(candidate, field, context) != require_field(
            baseline, field, context
        ):
            raise ValueError(f"{context} mismatches {field}")

    baseline_environment = require_mapping(
        require_field(baseline, "environment", context), "baseline environment"
    )
    candidate_environment = require_mapping(
        require_field(candidate, "environment", context), "candidate environment"
    )
    for field in ENVIRONMENT_FIELDS:
        if require_field(candidate_environment, field, context) != require_field(
            baseline_environment, field, context
        ):
            raise ValueError(f"{context} mismatches environment.{field}")

    baseline_case = require_mapping(
        require_field(baseline, "case_config", context), "baseline case_config"
    )
    candidate_case = require_mapping(
        require_field(candidate, "case_config", context), "candidate case_config"
    )
    if not baseline_case or baseline_case != candidate_case:
        raise ValueError(f"{context} mismatches case_config")

    baseline_variant = require_mapping(
        require_field(baseline, "variant_config", context), "baseline variant_config"
    )
    candidate_variant = require_mapping(
        require_field(candidate, "variant_config", context), "candidate variant_config"
    )
    baseline_math = require_field(baseline_variant, "math_mode", context)
    candidate_math = require_field(candidate_variant, "math_mode", context)
    if baseline_math != candidate_math:
        raise ValueError(f"{context} mismatches variant_config.math_mode")

    if baseline.get("suite_case") != candidate.get("suite_case"):
        raise ValueError(f"{context} mismatches suite_case")
    if baseline.get("workload_config") != candidate.get("workload_config"):
        raise ValueError(f"{context} mismatches workload_config")


def compare_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    paired: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        key = (
            require_field(group, "run_id", "aggregate group"),
            require_field(group, "case_id", "aggregate group"),
            require_field(group, "operator", "aggregate group"),
            require_field(group, "measurement_level", "aggregate group"),
            require_field(group, "cache_mode", "aggregate group"),
        )
        paired[key].append(group)

    comparisons: list[dict[str, Any]] = []
    for key, members in sorted(paired.items()):
        names = [member.get("variant") for member in members]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"{key[1]} has missing or duplicate variants")
        baselines = [member for member in members if member.get("promotion_baseline") is True]
        candidates = [member for member in members if member.get("promotion_baseline") is False]
        if len(baselines) != 1:
            raise ValueError(f"{key[1]} requires exactly one promotion baseline")
        if not candidates:
            raise ValueError(f"{key[1]} has no candidate variant")
        baseline = baselines[0]
        baseline_latency = float(
            require_field(baseline, "median_of_process_medians_us", key[1])
        )
        if not math.isfinite(baseline_latency) or baseline_latency <= 0:
            raise ValueError(f"{key[1]} has invalid baseline latency")

        for candidate in candidates:
            assert_pairable(baseline, candidate)
            candidate_latency = float(
                require_field(candidate, "median_of_process_medians_us", key[1])
            )
            if not math.isfinite(candidate_latency) or candidate_latency <= 0:
                raise ValueError(f"{key[1]} has invalid candidate latency")
            workload = require_mapping(
                candidate.get("workload_config", {}), "workload_config"
            )
            comparisons.append(
                {
                    "run_id": key[0],
                    "case_id": key[1],
                    "operator": key[2],
                    "measurement_level": key[3],
                    "cache_mode": key[4],
                    "baseline_variant": baseline["variant"],
                    "candidate_variant": candidate["variant"],
                    "baseline_latency_us": baseline_latency,
                    "candidate_latency_us": candidate_latency,
                    "speedup": baseline_latency / candidate_latency,
                    "baseline_workspace_bytes": baseline["workspace_bytes"],
                    "candidate_workspace_bytes": candidate["workspace_bytes"],
                    "workspace_growth_bytes": candidate["workspace_bytes"]
                    - baseline["workspace_bytes"],
                    "trace_weight": workload.get("trace_weight"),
                    "pairing_verified": True,
                }
            )

    if not comparisons:
        raise ValueError("aggregate contains no baseline/candidate comparisons")
    by_candidate: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for comparison in comparisons:
        key = (
            comparison["operator"],
            comparison["candidate_variant"],
            comparison["measurement_level"],
            comparison["cache_mode"],
        )
        by_candidate[key].append(comparison)
    candidate_summaries = []
    for candidate_key, members in sorted(by_candidate.items()):
        operator, candidate_variant, measurement_level, cache_mode = candidate_key
        geometric_mean = math.exp(
            sum(math.log(item["speedup"]) for item in members) / len(members)
        )
        weights = [item["trace_weight"] for item in members]
        if all(
            isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and weight >= 0
            for weight in weights
        ):
            weighted_baseline = sum(
                float(item["trace_weight"]) * item["baseline_latency_us"]
                for item in members
            )
            weighted_candidate = sum(
                float(item["trace_weight"]) * item["candidate_latency_us"]
                for item in members
            )
            if weighted_candidate <= 0 or weighted_baseline <= 0:
                raise ValueError("trace weights produce a non-positive weighted latency")
            trace_ratio = weighted_baseline / weighted_candidate
            trace_reason = None
        else:
            trace_ratio = None
            trace_reason = (
                "one or more paired workload_config objects omit a non-negative "
                "trace_weight"
            )
        candidate_summaries.append(
            {
                "operator": operator,
                "candidate_variant": candidate_variant,
                "measurement_level": measurement_level,
                "cache_mode": cache_mode,
                "paired_cases": len(members),
                "shape_balanced_geometric_mean_speedup": geometric_mean,
                "trace_ratio_of_sums_speedup": trace_ratio,
                "trace_ratio_unavailable_reason": trace_reason,
            }
        )

    return {
        "comparisons": comparisons,
        "summary": {
            "paired_candidates": len(comparisons),
            "candidate_variants": candidate_summaries,
        },
    }


def compare_document(aggregate: dict[str, Any]) -> dict[str, Any]:
    if aggregate.get("schema_version") != "raggedroute.aggregate.v2":
        raise ValueError("comparison requires raggedroute.aggregate.v2")
    groups = aggregate.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("aggregate.v2 contains no groups")
    result = compare_groups(groups)
    return {
        "schema_version": "raggedroute.comparison.v1",
        "source_schema": aggregate["schema_version"],
        "suite_id": aggregate.get("suite_id"),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    output = compare_document(aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
