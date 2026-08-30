#!/usr/bin/env python3
"""Summarize a matched multi-process candidate matrix and library envelope.

The script keeps every recorded sample.  CV above 0.10 is disclosed as a
Windows/WDDM stability risk; only CV above the configured evidence ceiling is
reported as an evidence-ceiling violation.  Promotion decisions remain the
responsibility of ``evaluate_promotion.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"record {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError("input contains no records")
    return records


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sequence")
    return float(statistics.median(values))


def group_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("case_id")), str(record.get("variant")))].append(record)

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, members in grouped.items():
        case_id, variant = key
        process_runs = [int(member.get("process_run", 0)) for member in members]
        if len(process_runs) != len(set(process_runs)):
            raise ValueError(f"{case_id}/{variant} has duplicate process_run values")
        if any(member.get("protocol") != "release" for member in members):
            raise ValueError(f"{case_id}/{variant} contains a non-release record")
        if any(not member.get("validation", {}).get("ok", False) for member in members):
            raise ValueError(f"{case_id}/{variant} contains a failed validation")
        configs = {json.dumps(member.get("case_config", {}), sort_keys=True) for member in members}
        if len(configs) != 1:
            raise ValueError(f"{case_id}/{variant} contains mismatched case_config values")
        result[key] = {
            "case_id": case_id,
            "variant": variant,
            "case_config": members[0].get("case_config", {}),
            "process_runs": sorted(process_runs),
            "process_p50_us": {
                str(int(member["process_run"])): float(member["timing"]["batch_mean_us_p50"])
                for member in members
            },
            "p50_us": median(
                [float(member["timing"]["batch_mean_us_p50"]) for member in members]
            ),
            "p95_us": median(
                [float(member["timing"]["batch_mean_us_p95"]) for member in members]
            ),
            "max_process_cv": max(float(member["timing"]["cv"]) for member in members),
            "workspace_bytes": max(int(member.get("workspace_bytes", 0)) for member in members),
            "build_git_sha": members[0].get("environment", {}).get("build_git_sha"),
            "build_git_dirty": members[0].get("environment", {}).get("build_git_dirty"),
            "gpu_uuid": members[0].get("environment", {}).get("gpu_uuid"),
        }
    return result


def comparison_row(
    case_id: str,
    baseline_name: str,
    baseline: dict[str, Any],
    candidate_name: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if baseline["process_runs"] != candidate["process_runs"]:
        raise ValueError(f"{case_id}: process_run sets do not match")
    process_speedups = []
    for process_run in baseline["process_runs"]:
        key = str(process_run)
        process_speedups.append(
            baseline["process_p50_us"][key] / candidate["process_p50_us"][key]
        )
    return {
        "case_id": case_id,
        "case_config": candidate["case_config"],
        "baseline": baseline_name,
        "candidate": candidate_name,
        "baseline_p50_us": baseline["p50_us"],
        "candidate_p50_us": candidate["p50_us"],
        "p50_speedup": baseline["p50_us"] / candidate["p50_us"],
        "baseline_p95_us": baseline["p95_us"],
        "candidate_p95_us": candidate["p95_us"],
        "p95_speedup": baseline["p95_us"] / candidate["p95_us"],
        "baseline_max_process_cv": baseline["max_process_cv"],
        "candidate_max_process_cv": candidate["max_process_cv"],
        "candidate_faster_processes": sum(speedup > 1.0 for speedup in process_speedups),
        "paired_processes": len(process_speedups),
        "process_speedups": process_speedups,
        "workspace_growth_bytes": candidate["workspace_bytes"] - baseline["workspace_bytes"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("comparison has no rows")
    speedups = [float(row["p50_speedup"]) for row in rows]
    process_pairs = sum(int(row["paired_processes"]) for row in rows)
    process_wins = sum(int(row["candidate_faster_processes"]) for row in rows)
    return {
        "paired_shapes": len(rows),
        "ratio_of_sums_p50_speedup": sum(float(row["baseline_p50_us"]) for row in rows)
        / sum(float(row["candidate_p50_us"]) for row in rows),
        "shape_geomean_p50_speedup": math.exp(statistics.fmean(math.log(x) for x in speedups)),
        "shape_win_fraction": sum(x > 1.0 for x in speedups) / len(speedups),
        "maximum_p50_regression_fraction": max(
            float(row["candidate_p50_us"]) / float(row["baseline_p50_us"]) - 1.0
            for row in rows
        ),
        "maximum_p95_regression_fraction": max(
            float(row["candidate_p95_us"]) / float(row["baseline_p95_us"]) - 1.0
            for row in rows
        ),
        "maximum_candidate_process_cv": max(float(row["candidate_max_process_cv"]) for row in rows),
        "maximum_baseline_process_cv": max(float(row["baseline_max_process_cv"]) for row in rows),
        "candidate_faster_process_pairs": process_wins,
        "paired_process_pairs": process_pairs,
        "candidate_faster_process_fraction": process_wins / process_pairs,
        "all_processes_candidate_faster_shapes": sum(
            int(row["candidate_faster_processes"]) == int(row["paired_processes"])
            for row in rows
        ),
        "majority_candidate_faster_shapes": sum(
            int(row["candidate_faster_processes"]) * 2 > int(row["paired_processes"])
            for row in rows
        ),
        "maximum_workspace_growth_bytes": max(int(row["workspace_growth_bytes"]) for row in rows),
    }


def library_envelope_rows(
    groups: dict[tuple[str, str], dict[str, Any]],
    cases: list[str],
    library_names: list[str],
    candidate_name: str,
) -> list[dict[str, Any]]:
    """Select the lowest median process-p50 library for every declared case."""
    rows = []
    for case_id in cases:
        available = [(name, groups.get((case_id, name))) for name in library_names]
        if any(group is None for _, group in available):
            raise ValueError(f"{case_id}: one or more library envelope variants are missing")
        envelope_name, envelope = min(
            ((name, group) for name, group in available if group is not None),
            key=lambda item: float(item[1]["p50_us"]),
        )
        candidate = groups.get((case_id, candidate_name))
        if candidate is None:
            raise ValueError(f"{case_id}: candidate {candidate_name} is missing")
        row = comparison_row(case_id, envelope_name, envelope, candidate_name, candidate)
        row["baseline"] = "fastest_library_envelope"
        row["envelope_winner"] = envelope_name
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--library", action="append", default=[])
    parser.add_argument("--cv-risk-threshold", type=float, default=0.10)
    parser.add_argument("--cv-evidence-ceiling", type=float, default=0.50)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-csv", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if not args.baseline or not args.library:
        raise ValueError("at least one --baseline and one --library are required")

    records = read_records(args.input)
    groups = group_records(records)
    cases = sorted({case_id for case_id, variant in groups if variant == args.candidate})
    if not cases:
        raise ValueError("candidate has no cases")

    comparisons: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    for baseline_name in args.baseline:
        rows = []
        for case_id in cases:
            baseline = groups.get((case_id, baseline_name))
            candidate = groups.get((case_id, args.candidate))
            if baseline is None or candidate is None:
                raise ValueError(f"{case_id}: missing {baseline_name} or {args.candidate}")
            rows.append(comparison_row(case_id, baseline_name, baseline, args.candidate, candidate))
        comparisons[baseline_name] = {"summary": summarize(rows), "rows": rows}
        flat_rows.extend(rows)

    envelope_rows = library_envelope_rows(groups, cases, args.library, args.candidate)
    comparisons["fastest_library_envelope"] = {
        "members": args.library,
        "selection": "lowest median-of-process-p50 per predeclared shape",
        "summary": summarize(envelope_rows),
        "rows": envelope_rows,
    }
    flat_rows.extend(envelope_rows)

    all_groups = list(groups.values())
    output = {
        "schema_version": "raggedroute.candidate_matrix_summary.v1",
        "source": str(args.input),
        "candidate": args.candidate,
        "record_count": len(records),
        "case_count": len(cases),
        "variant_count": len({record.get("variant") for record in records}),
        "cv_policy": {
            "risk_disclosure_threshold": args.cv_risk_threshold,
            "evidence_ceiling": args.cv_evidence_ceiling,
            "behavior": "risk disclosure above 0.10; no automatic downgrade until the ceiling",
        },
        "evidence_quality": {
            "all_validation_ok": all(record.get("validation", {}).get("ok") for record in records),
            "all_release": all(record.get("protocol") == "release" for record in records),
            "all_clean_build": all(
                str(record.get("environment", {}).get("build_git_dirty")) == "false"
                for record in records
            ),
            "maximum_process_cv": max(float(record["timing"]["cv"]) for record in records),
            "groups_above_cv_risk_threshold": sum(
                float(group["max_process_cv"]) > args.cv_risk_threshold for group in all_groups
            ),
            "groups_above_cv_evidence_ceiling": sum(
                float(group["max_process_cv"]) > args.cv_evidence_ceiling for group in all_groups
            ),
        },
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "baseline",
        "envelope_winner",
        "candidate",
        "baseline_p50_us",
        "candidate_p50_us",
        "p50_speedup",
        "baseline_p95_us",
        "candidate_p95_us",
        "p95_speedup",
        "baseline_max_process_cv",
        "candidate_max_process_cv",
        "candidate_faster_processes",
        "paired_processes",
        "workspace_growth_bytes",
    ]
    with args.output_csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
