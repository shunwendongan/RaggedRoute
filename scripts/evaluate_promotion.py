#!/usr/bin/env python3
"""Evidence-driven three-state CUDA candidate promotion evaluator."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"record {line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError("input contains no benchmark records")
    return records


def timing_spec(policy: dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve the timing boundary selected by a policy.

    CUDA-event timing remains the default for operator promotion.  CUDA Graph
    submission experiments are different: their release claim is host
    time-to-solution, so a policy can explicitly select that separately
    recorded boundary without reinterpreting profiler output as benchmark data.
    """
    section = str(policy.get("timing_source", "timing"))
    if section == "timing":
        return (
            section,
            str(policy.get("timing_p50_field", "batch_mean_us_p50")),
            str(policy.get("timing_p95_field", "batch_mean_us_p95")),
            str(policy.get("timing_cv_field", "cv")),
        )
    return (
        section,
        str(policy.get("timing_p50_field", "p50_us")),
        str(policy.get("timing_p95_field", "p95_us")),
        str(policy.get("timing_cv_field", "cv")),
    )


def median_field(records: list[dict[str, Any]], section: str, field: str) -> float:
    return statistics.median(float(record[section][field]) for record in records)


def finite_positive_timing(record: dict[str, Any], section: str, field: str) -> bool:
    try:
        value = float(record.get(section, {}).get(field, math.nan))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def evaluate(
    records: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("variant") in {baseline, candidate}:
            groups[(str(record.get("case_id")), str(record.get("variant")))].append(record)
    cases = sorted({case for case, _ in groups})
    evidence_failures: list[str] = []
    correctness_failures: list[str] = []
    rows = []
    minimum_processes = int(policy.get("minimum_independent_process_runs", 5))
    minimum_paired_shapes = int(policy.get("minimum_paired_shapes", 1))
    # Portfolio policy: CV above 0.10 is disclosed as a Windows/WDDM risk, but
    # it no longer auto-downgrades otherwise complete evidence.  The default
    # fail-closed ceiling is 0.50; versioned historical policies can still
    # request their original stricter threshold explicitly.
    maximum_cv = float(policy.get("maximum_all_samples_cv", 0.50))
    timing_section, p50_field, p95_field, cv_field = timing_spec(policy)
    ignore_timing_cv = bool(policy.get("ignore_timing_cv", False))
    require_real = bool(policy.get("require_real_route_trace", False))
    required_level = policy.get("required_measurement_level")
    for case in cases:
        baseline_records = groups.get((case, baseline), [])
        candidate_records = groups.get((case, candidate), [])
        if not baseline_records or not candidate_records:
            evidence_failures.append(f"{case}: missing paired variant")
            continue
        for name, current in ((baseline, baseline_records), (candidate, candidate_records)):
            process_runs = {int(record.get("process_run", 0)) for record in current}
            if len(process_runs) < minimum_processes:
                evidence_failures.append(
                    f"{case}/{name}: {len(process_runs)} processes < {minimum_processes}"
                )
            if len(process_runs) != len(current):
                evidence_failures.append(f"{case}/{name}: duplicate process_run records")
            if any(record.get("protocol") != "release" for record in current):
                evidence_failures.append(f"{case}/{name}: non-release record")
            if any(str(record.get("environment", {}).get("build_git_dirty")) != "false"
                   for record in current):
                evidence_failures.append(f"{case}/{name}: dirty or unknown build")
            if any(record.get("environment", {}).get("build_type") != "Release"
                   for record in current):
                evidence_failures.append(f"{case}/{name}: non-Release or unknown build type")
            if required_level and any(record.get("measurement_level") != required_level
                                      for record in current):
                evidence_failures.append(
                    f"{case}/{name}: measurement_level does not match {required_level}"
                )
            if any(not record.get("validation", {}).get("ok", False) for record in current):
                correctness_failures.append(f"{case}/{name}: correctness failed")
            if not ignore_timing_cv:
                try:
                    cv = max(float(record.get(timing_section, {}).get(cv_field, math.inf))
                             for record in current)
                except (TypeError, ValueError):
                    cv = math.inf
                if not math.isfinite(cv) or cv > maximum_cv:
                    evidence_failures.append(
                        f"{case}/{name}: max {timing_section}.{cv_field} {cv:.4f} > "
                        f"{maximum_cv:.4f}"
                    )
            for field in (p50_field, p95_field):
                if not all(finite_positive_timing(record, timing_section, field)
                           for record in current):
                    evidence_failures.append(
                        f"{case}/{name}: invalid {timing_section}.{field}"
                    )
            if name == candidate:
                for field, expected in policy.get("required_candidate_variant_config", {}).items():
                    if any(record.get("variant_config", {}).get(field) != expected
                           for record in current):
                        evidence_failures.append(
                            f"{case}/{name}: candidate variant_config.{field} does not match "
                            f"the promotion policy"
                        )
        baseline_processes = {int(record.get("process_run", 0)) for record in baseline_records}
        candidate_processes = {int(record.get("process_run", 0)) for record in candidate_records}
        if policy.get("require_same_process_runs", True) and baseline_processes != candidate_processes:
            evidence_failures.append(f"{case}: process_run set mismatch")
        baseline_gpu = {record.get("environment", {}).get("gpu_uuid") for record in baseline_records}
        candidate_gpu = {record.get("environment", {}).get("gpu_uuid") for record in candidate_records}
        if policy.get("require_same_gpu_uuid", True) and (
            len(baseline_gpu) != 1 or baseline_gpu != candidate_gpu
        ):
            evidence_failures.append(f"{case}: GPU UUID mismatch")
        if policy.get("require_same_case_config", True):
            configs = {
                json.dumps(record.get("case_config", {}), sort_keys=True)
                for record in baseline_records + candidate_records
            }
            if len(configs) != 1:
                evidence_failures.append(f"{case}: case_config mismatch")
        build_shas = {
            record.get("environment", {}).get("build_git_sha")
            for record in baseline_records + candidate_records
        }
        if policy.get("require_same_build_git_sha", True) and (
            len(build_shas) != 1 or None in build_shas or "" in build_shas
        ):
            evidence_failures.append(f"{case}: build Git SHA mismatch or missing")
        run_ids = {record.get("run_id") for record in baseline_records + candidate_records}
        if policy.get("require_same_run_id", True) and (
            len(run_ids) != 1 or None in run_ids or "" in run_ids
        ):
            evidence_failures.append(f"{case}: run_id mismatch or missing")
        workload_sources = {
            record.get("case_config", {}).get("workload_source")
            for record in baseline_records + candidate_records
        }
        if require_real and not workload_sources <= {"production", "captured"}:
            evidence_failures.append(f"{case}: real route trace is required")
        valid_timings = all(
            finite_positive_timing(record, timing_section, field)
            for record in baseline_records + candidate_records
            for field in (p50_field, p95_field)
        )
        if not valid_timings:
            continue
        baseline_p50 = median_field(baseline_records, timing_section, p50_field)
        candidate_p50 = median_field(candidate_records, timing_section, p50_field)
        baseline_p95 = median_field(baseline_records, timing_section, p95_field)
        candidate_p95 = median_field(candidate_records, timing_section, p95_field)
        baseline_workspace = max(int(record.get("workspace_bytes", 0)) for record in baseline_records)
        candidate_workspace = max(int(record.get("workspace_bytes", 0)) for record in candidate_records)
        rows.append({
            "case_id": case,
            "baseline_p50_us": baseline_p50,
            "candidate_p50_us": candidate_p50,
            "baseline_p95_us": baseline_p95,
            "candidate_p95_us": candidate_p95,
            "p50_speedup": baseline_p50 / candidate_p50,
            "p95_speedup": baseline_p95 / candidate_p95,
            "p50_regression_fraction": candidate_p50 / baseline_p50 - 1.0,
            "p95_regression_fraction": candidate_p95 / baseline_p95 - 1.0,
            "baseline_workspace_bytes": baseline_workspace,
            "candidate_workspace_bytes": candidate_workspace,
        })
    if not cases:
        evidence_failures.append("no paired cases for requested variants")
    if len(rows) < minimum_paired_shapes:
        evidence_failures.append(
            f"paired shapes {len(rows)} < required {minimum_paired_shapes}"
        )
    if rows:
        ratio_of_sums = sum(row["baseline_p50_us"] for row in rows) / sum(
            row["candidate_p50_us"] for row in rows
        )
        geomean = math.exp(statistics.fmean(math.log(row["p50_speedup"]) for row in rows))
        win_fraction = sum(row["p50_speedup"] > 1.0 for row in rows) / len(rows)
        max_p50_regression = max(row["p50_regression_fraction"] for row in rows)
        max_p95_regression = max(row["p95_regression_fraction"] for row in rows)
        max_workspace_growth = max(
            row["candidate_workspace_bytes"] - row["baseline_workspace_bytes"] for row in rows
        )
    else:
        ratio_of_sums = geomean = win_fraction = math.nan
        max_p50_regression = max_p95_regression = math.inf
        max_workspace_growth = math.inf
    performance_failures = []
    checks = (
        (ratio_of_sums >= float(policy.get("minimum_ratio_of_sums_p50_speedup", 1.03)),
         f"ratio-of-sums {ratio_of_sums:.4f} below threshold"),
        (win_fraction >= float(policy.get("minimum_speedup_shape_coverage", 0.80)),
         f"shape win fraction {win_fraction:.4f} below threshold"),
        (geomean >= float(policy.get("minimum_geometric_mean_p50_speedup", 0.0)),
         f"geometric mean {geomean:.4f} below threshold"),
        (max_p50_regression <= float(policy.get("maximum_single_shape_p50_regression_fraction", 0.05)),
         f"max p50 regression {max_p50_regression:.4f} above threshold"),
        (max_p95_regression <= float(policy.get("maximum_single_shape_p95_regression_fraction", 0.03)),
         f"max p95 regression {max_p95_regression:.4f} above threshold"),
        (max_workspace_growth <= int(policy.get("maximum_workspace_growth_bytes", 0)),
         f"workspace growth {max_workspace_growth} bytes above threshold"),
        (not rows or max(row["candidate_workspace_bytes"] for row in rows) <=
         int(policy.get("maximum_candidate_workspace_bytes", 2**63 - 1)),
         "candidate workspace exceeds absolute threshold"),
    )
    performance_failures.extend(message for passed, message in checks if not passed)
    if correctness_failures:
        decision = "reject"
        reasons = correctness_failures
    elif evidence_failures:
        decision = "insufficient_evidence"
        reasons = sorted(set(evidence_failures))
    elif performance_failures:
        decision = "reject"
        reasons = performance_failures
    else:
        decision = "promote"
        reasons = ["all correctness, stability, performance, and workspace gates passed"]
    return {
        "schema_version": "raggedroute.promotion_decision.v1",
        "baseline": baseline,
        "candidate": candidate,
        "measurement": {
            "timing_source": timing_section,
            "p50_field": p50_field,
            "p95_field": p95_field,
            "cv_field": cv_field,
            "timing_cv_enforced": not ignore_timing_cv,
            "policy_exception": policy.get("stability_exception"),
        },
        "decision": decision,
        "reasons": reasons,
        "summary": {
            "paired_shapes": len(rows),
            "ratio_of_sums_p50_speedup": json_number(ratio_of_sums),
            "shape_balanced_geometric_mean_speedup": json_number(geomean),
            "shape_win_fraction": json_number(win_fraction),
            "maximum_p50_regression_fraction": json_number(max_p50_regression),
            "maximum_p95_regression_fraction": json_number(max_p95_regression),
            "maximum_workspace_growth_bytes": (
                max_workspace_growth if math.isfinite(max_workspace_growth) else None
            ),
        },
        "evidence_failures": sorted(set(evidence_failures)),
        "performance_failures": performance_failures,
        "rows": rows,
    }


def markdown(document: dict[str, Any]) -> str:
    summary = document["summary"]
    def metric(name: str, suffix: str = "") -> str:
        value = summary[name]
        return "not_collected" if value is None else f"{value:.4f}{suffix}"
    lines = [
        "# CUDA candidate promotion decision",
        "",
        f"- Decision: `{document['decision']}`",
        f"- Baseline: `{document['baseline']}`",
        f"- Candidate: `{document['candidate']}`",
        f"- Ratio-of-sums p50 speedup: {metric('ratio_of_sums_p50_speedup', 'x')}",
        f"- Geometric mean speedup: {metric('shape_balanced_geometric_mean_speedup', 'x')}",
        f"- Shape win fraction: " + (
            "not_collected" if summary["shape_win_fraction"] is None
            else f"{summary['shape_win_fraction']:.1%}"
        ),
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in document["reasons"])
    lines.extend([
        "",
        "## Paired shapes",
        "",
        "| Case | Baseline p50 (us) | Candidate p50 (us) | Speedup | p95 speedup |",
        "|---|---:|---:|---:|---:|",
    ])
    lines.extend(
        f"| `{row['case_id']}` | {row['baseline_p50_us']:.3f} | "
        f"{row['candidate_p50_us']:.3f} | {row['p50_speedup']:.4f}x | "
        f"{row['p95_speedup']:.4f}x |"
        for row in document["rows"]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument(
        "--case-prefix", action="append", default=[],
        help="Keep only case_id values beginning with one of these prefixes."
    )
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    input_records = read_records(args.input)
    if args.case_prefix:
        input_records = [
            record for record in input_records
            if any(str(record.get("case_id", "")).startswith(prefix)
                   for prefix in args.case_prefix)
        ]
        if not input_records:
            raise ValueError("no benchmark records matched --case-prefix")
    result = evaluate(input_records, args.baseline, args.candidate, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result), encoding="utf-8")
    print(result["decision"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
