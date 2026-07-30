#!/usr/bin/env python3
"""Aggregate raw JSONL without hiding per-process or per-sample evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any


SUITE_V2 = "raggedroute.suite.v2"
MANIFEST_V2 = "raggedroute.run_manifest.v2"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate records only after proving their measurement conditions match."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["run_id"],
            record["case_id"],
            record["operator"],
            record["variant"],
            record["measurement_level"],
            record["cache_mode"],
        )
        groups[key].append(record)

    invariant_fields = (
        "protocol",
        "warmup",
        "kernel_repeats",
        "samples",
        "seed",
        "excluded_steps",
        "workspace_bytes",
    )
    work_fields = ("logical_bytes", "flops", "operator_metrics")
    environment_fields = (
        "gpu_uuid",
        "build_git_sha",
        "build_type",
        "compute_capability",
        "cuda_compiler",
        "cuda_runtime",
        "cuda_driver",
    )
    summary: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        process_runs = {member["process_run"] for member in members}
        canonical = members[0]
        for member in members[1:]:
            if member["case_config"] != canonical["case_config"]:
                raise ValueError(f"{key[1]} mixes incompatible case_config values")
            if member["variant_config"] != canonical["variant_config"]:
                raise ValueError(f"{key[1]} mixes incompatible variant_config values")
            for field in invariant_fields:
                if member.get(field) != canonical.get(field):
                    raise ValueError(f"{key[1]} mixes incompatible {field} values")
            for field in work_fields:
                if member["work"].get(field) != canonical["work"].get(field):
                    raise ValueError(f"{key[1]} mixes incompatible work.{field} values")
            for field in environment_fields:
                if member["environment"].get(field) != canonical["environment"].get(field):
                    raise ValueError(f"{key[1]} mixes environments at {field}")
        if len(members) != len(process_runs):
            raise ValueError(f"{key[1]} contains duplicate records for a process run")
        if canonical.get("protocol") == "release" and len(process_runs) < 3:
            raise ValueError(f"{key[1]} release result has fewer than 3 process runs")
        process_medians = [
            member["timing"]["batch_mean_us_p50"] for member in members
        ]
        raw = [
            sample
            for member in members
            for sample in member["timing"]["raw_batch_mean_samples_us"]
        ]
        mean = statistics.fmean(raw)
        stddev = statistics.pstdev(raw)
        summary.append(
            {
                "run_id": key[0],
                "case_id": key[1],
                "operator": key[2],
                "variant": key[3],
                "measurement_level": key[4],
                "cache_mode": key[5],
                "process_runs": len(process_runs),
                "records": len(members),
                "raw_samples": len(raw),
                "median_of_process_medians_us": statistics.median(process_medians),
                "all_samples_p50_us": percentile(raw, 0.50),
                "all_samples_p95_us": percentile(raw, 0.95),
                "all_samples_mean_us": mean,
                "all_samples_stddev_us": stddev,
                "all_samples_cv": 0.0 if mean == 0 else stddev / mean,
                "case_config": canonical["case_config"],
                "variant_config": canonical["variant_config"],
                "work": canonical["work"],
            }
        )
    return summary


def level_name(level: str) -> str:
    names = {
        "l1": "L1_kernel_body",
        "l2": "L2_operator_steady",
        "l3": "L3_chain_steady",
        "l4": "L4_host_call",
    }
    if level not in names:
        raise ValueError(f"unsupported suite level {level!r}")
    return names[level]


def aggregate_records_v2(
    records: list[dict[str, Any]], suite: dict[str, Any]
) -> list[dict[str, Any]]:
    """Preserve every field required for strict baseline/candidate pairing."""
    if suite.get("schema_version") != SUITE_V2:
        raise ValueError("aggregate.v2 requires a raggedroute.suite.v2 manifest")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite v2 manifest has no cases")

    summaries = aggregate_records(records)
    records_by_group: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["run_id"],
            record["case_id"],
            record["operator"],
            record["variant"],
            record["measurement_level"],
            record["cache_mode"],
        )
        records_by_group[key].append(record)

    case_by_id = {case.get("id"): case for case in cases}
    if None in case_by_id or len(case_by_id) != len(cases):
        raise ValueError("suite v2 case ids must be present and unique")

    declared: set[tuple[str, str, str, str]] = set()
    common = suite.get("common", {})
    for case in cases:
        target = case.get("operator") or case.get("suite")
        if not target:
            raise ValueError(f"case {case['id']} has no target")
        variants = case.get("variants", [])
        if len([v for v in variants if v.get("promotion_baseline") is True]) != 1:
            raise ValueError(f"case {case['id']} has an invalid promotion baseline")
        for variant in variants:
            for level in case.get("levels", ["l1"]):
                declared.add((case["id"], target, variant["name"], level_name(level)))

    observed = {
        (summary["case_id"], summary["operator"], summary["variant"], summary["measurement_level"])
        for summary in summaries
    }
    missing = declared - observed
    unexpected = observed - declared
    if missing:
        raise ValueError(f"aggregate.v2 is missing declared case/variant groups: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"aggregate.v2 contains undeclared groups: {sorted(unexpected)}")

    required_variant_fields = {
        "implementation_category",
        "implementation_version",
        "implementation_revision",
        "dependency_revision",
        "algorithm_id",
        "math_mode",
    }
    enriched: list[dict[str, Any]] = []
    for summary in summaries:
        case = case_by_id[summary["case_id"]]
        variant = next(
            candidate
            for candidate in case["variants"]
            if candidate["name"] == summary["variant"]
        )
        key = (
            summary["run_id"],
            summary["case_id"],
            summary["operator"],
            summary["variant"],
            summary["measurement_level"],
            summary["cache_mode"],
        )
        members = records_by_group[key]
        canonical = members[0]
        expected_process_runs = int(suite.get("process_runs", 1))
        if summary["process_runs"] != expected_process_runs:
            raise ValueError(
                f"{summary['case_id']}/{summary['variant']} has {summary['process_runs']} "
                f"process runs; expected {expected_process_runs}"
            )
        absent = required_variant_fields - canonical["variant_config"].keys()
        if absent:
            raise ValueError(
                f"{summary['case_id']}/{summary['variant']} variant_config is missing {sorted(absent)}"
            )
        entry = dict(summary)
        entry.update(
            {
                "promotion_baseline": variant.get("promotion_baseline") is True,
                "protocol": canonical["protocol"],
                "warmup": canonical["warmup"],
                "kernel_repeats": canonical["kernel_repeats"],
                "samples": canonical["samples"],
                "seed": canonical["seed"],
                "excluded_steps": canonical["excluded_steps"],
                "workspace_bytes": canonical["workspace_bytes"],
                "environment": canonical["environment"],
                "environment_snapshots": [member["environment"] for member in members],
                "workload_config": case.get("workload", {}),
                "suite_case": {
                    "params": case.get("params", {}),
                    "cache_mode": case.get(
                        "cache_mode", common.get("cache_mode", "warm")
                    ),
                    "kernel_repeats_by_level": case.get(
                        "kernel_repeats_by_level", {}
                    ),
                },
            }
        )
        enriched.append(entry)
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--json", required=True, type=pathlib.Path)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        help="run manifest v2; when present, emit raggedroute.aggregate.v2",
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "raggedroute.benchmark.v1":
                raise ValueError(f"line {line_number}: unsupported schema")
            if not record.get("validation", {}).get("ok"):
                raise ValueError(f"line {line_number}: validation did not pass")
            records.append(record)
    if not records:
        raise ValueError("input contains no records")

    if args.manifest:
        with args.manifest.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != MANIFEST_V2:
            raise ValueError("--manifest must be raggedroute.run_manifest.v2")
        run_ids = {record["run_id"] for record in records}
        if run_ids != {manifest.get("run_id")}:
            raise ValueError("manifest run_id does not match the raw records")
        summary = aggregate_records_v2(records, manifest.get("suite", {}))
        output = {
            "schema_version": "raggedroute.aggregate.v2",
            "source": str(args.input.resolve()),
            "manifest": str(args.manifest.resolve()),
            "suite_id": manifest["suite"].get("suite_id"),
            "protocol": manifest["suite"].get("protocol", "smoke"),
            "groups": summary,
        }
    else:
        summary = aggregate_records(records)
        output = {
            "schema_version": "raggedroute.aggregate.v1",
            "source": str(args.input.resolve()),
            "groups": summary,
        }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    columns = [
        "run_id", "case_id", "operator", "variant", "measurement_level",
        "cache_mode", "process_runs", "records", "raw_samples",
        "median_of_process_medians_us", "all_samples_p50_us",
        "all_samples_p95_us", "all_samples_mean_us", "all_samples_stddev_us",
        "all_samples_cv", "promotion_baseline",
    ]
    with args.csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {args.json}")
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
