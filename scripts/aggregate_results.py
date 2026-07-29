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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--json", required=True, type=pathlib.Path)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
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

    summary: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        process_runs = {member["process_run"] for member in members}
        canonical = members[0]
        for member in members[1:]:
            if member["case_config"] != canonical["case_config"]:
                raise ValueError(f"{key[1]} mixes incompatible case_config values")
            if member["variant_config"] != canonical["variant_config"]:
                raise ValueError(f"{key[1]} mixes incompatible variant_config values")
            for environment_key in ("gpu_uuid", "build_git_sha", "build_type"):
                if member["environment"].get(environment_key) != canonical[
                    "environment"
                ].get(environment_key):
                    raise ValueError(
                        f"{key[1]} mixes environments at {environment_key}"
                    )
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
        row = {
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
            "case_config": members[0]["case_config"],
            "variant_config": members[0]["variant_config"],
            "work": members[0]["work"],
        }
        summary.append(row)

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
        "all_samples_cv",
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
