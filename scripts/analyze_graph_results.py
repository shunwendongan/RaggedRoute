#!/usr/bin/env python3
"""Summarize CUDA Graph host/GPU/submission timing and amortization boundaries."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def records(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def median_metric(rows: list[dict[str, Any]], section: str, field: str) -> float:
    return statistics.median(float(row[section][field]) for row in rows)


def median_variant_metric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get("variant_config", {}).get(field) for row in rows]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.median(numeric) if numeric else None


def any_variant_bool(rows: list[dict[str, Any]], field: str) -> bool:
    return any(bool(row.get("variant_config", {}).get(field, False)) for row in rows)


def analyze(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["variant"]))].append(row)
    cases = sorted({case for case, _ in grouped})
    output_rows = []
    for case in cases:
        baseline_rows = grouped.get((case, baseline), [])
        if not baseline_rows:
            continue
        baseline_host = median_metric(
            baseline_rows, "host_time_to_solution_timing", "p50_us"
        )
        for (current_case, variant), current in sorted(grouped.items()):
            if current_case != case or variant == baseline or "graph" not in variant:
                continue
            host = median_metric(current, "host_time_to_solution_timing", "p50_us")
            gpu = median_metric(current, "gpu_span_timing", "p50_us")
            submission = median_metric(current, "cpu_submission_timing", "p50_us")
            setup = median_variant_metric(current, "graph_capture_instantiate_upload_us") or 0.0
            cache_misses = median_variant_metric(current, "graph_cache_misses")
            cache_hits = median_variant_metric(current, "graph_cache_hits")
            cache_entries = median_variant_metric(current, "graph_cache_entries")
            mixed_shape_trace = any_variant_bool(current, "mixed_shape_cache_trace")
            is_cache = "_graph_cache_" in variant
            setup_per_miss = (
                setup / cache_misses
                if is_cache and cache_misses is not None and cache_misses > 0.0
                else setup
            )
            amortized = {str(replays): host + setup / replays for replays in (1, 10, 100, 1000)}
            break_even = (
                math.ceil(setup / (baseline_host - host))
                if not mixed_shape_trace and baseline_host > host and setup > 0.0
                else None
            )
            output_rows.append(
                {
                    "case_id": case,
                    "variant": variant,
                    "baseline_host_p50_us": baseline_host,
                    "host_p50_us": host,
                    "gpu_span_p50_us": gpu,
                    "cpu_submission_p50_us": submission,
                    "steady_host_speedup": None if mixed_shape_trace else baseline_host / host,
                    "graph_setup_us": setup,
                    "graph_setup_us_per_miss": setup_per_miss,
                    "amortized_host_us": amortized,
                    "break_even_replays": break_even,
                    "graph_cache_entries": cache_entries,
                    "graph_cache_hits": cache_hits,
                    "graph_cache_misses": cache_misses,
                    "mixed_shape_cache_trace": mixed_shape_trace,
                    "speedup_claim_eligible": not mixed_shape_trace,
                }
            )
    return {"schema_version": "raggedroute.graph_analysis.v2", "baseline": baseline,
            "rows": output_rows}


def markdown(document: dict[str, Any]) -> str:
    lines = [
        "# CUDA Graph timing analysis",
        "",
        f"Baseline: `{document['baseline']}`. Host time-to-solution is the primary metric.",
        "",
        "| Case | Graph variant | Host p50 (us) | GPU p50 (us) | Submit p50 (us) | Host speedup | Cache H/M/E | Break-even replays |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in document["rows"]:
        break_even = "not_applicable" if row["mixed_shape_cache_trace"] else (
            "not_reached" if row["break_even_replays"] is None else str(row["break_even_replays"])
        )
        speedup = "not_a_speed_claim" if row["steady_host_speedup"] is None else f"{row['steady_host_speedup']:.4f}x"
        cache = "/".join(
            "not_collected" if row[name] is None else str(int(row[name]))
            for name in ("graph_cache_hits", "graph_cache_misses", "graph_cache_entries")
        )
        lines.append(
            f"| `{row['case_id']}` | `{row['variant']}` | {row['host_p50_us']:.3f} | "
            f"{row['gpu_span_p50_us']:.3f} | {row['cpu_submission_p50_us']:.3f} | "
            f"{speedup} | {cache} | {break_even} |"
        )
    lines.extend([
        "",
        "Mixed-shape cache traces describe LRU behavior and miss cost; they are not per-shape "
        "operator speedup claims. Amortized values use the recorded aggregate graph setup time "
        "and are therefore diagnostic, not promotion evidence.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument("--case-prefix", action="append", default=[])
    args = parser.parse_args()
    input_rows = records(args.input)
    if args.case_prefix:
        input_rows = [
            row for row in input_rows
            if any(str(row.get("case_id", "")).startswith(prefix)
                   for prefix in args.case_prefix)
        ]
        if not input_rows:
            raise ValueError("no graph records matched --case-prefix")
    document = analyze(input_rows, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(document), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
