#!/usr/bin/env python3
"""Validate and aggregate five WSL2 Triton Top-K release shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def aggregate(raw_paths: list[pathlib.Path], manifest_paths: list[pathlib.Path],
              controlled_paths: list[pathlib.Path]) -> dict[str, Any]:
    if not (len(raw_paths) == len(manifest_paths) == len(controlled_paths) == 5):
        raise ValueError("exactly five raw, manifest, and controlled shard paths are required")
    canonical: dict[str, Any] | None = None
    runs: set[int] = set()
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for raw_path, manifest_path, controlled_path in zip(raw_paths, manifest_paths, controlled_paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "raggedroute.triton_run_manifest.v1":
            raise ValueError(f"{manifest_path}: invalid manifest schema")
        if not controlled.get("locked") or not controlled.get("restored"):
            raise ValueError(f"{controlled_path}: clock lock/restore gate failed")
        current = {
            "run_id": manifest["run_id"], "suite": manifest["suite"],
            "repo_commit": manifest["repo_commit"], "environment": manifest["environment"],
        }
        if canonical is None:
            canonical = current
        elif current != canonical:
            raise ValueError(f"{manifest_path}: shard provenance mismatch")
        process_run = int(manifest["process_run"])
        if process_run in runs:
            raise ValueError(f"duplicate process run {process_run}")
        runs.add(process_run)
        records = load_lines(raw_path)
        expected_records = len(manifest["suite"]["tokens"]) * len(manifest["suite"]["experts"])
        if len(records) != expected_records:
            raise ValueError(f"{raw_path}: expected {expected_records} records, got {len(records)}")
        for record in records:
            if record.get("process_run") != process_run or not record.get("validation", {}).get("ok"):
                raise ValueError(f"{raw_path}: invalid process-run or correctness record")
            case = record["case_config"]
            grouped[(int(case["T"]), int(case["E"]))].append(record)
    assert canonical is not None
    if runs != {1, 2, 3, 4, 5}:
        raise ValueError(f"incomplete process runs: {sorted(runs)}")

    rows = []
    for (tokens, experts), records in sorted(grouped.items()):
        if len(records) != 5:
            raise ValueError(f"T={tokens},E={experts}: expected five records")
        samples = [sample for record in records for sample in record["timing"]["raw_batch_mean_samples_us"]]
        mean = statistics.fmean(samples)
        stddev = statistics.pstdev(samples)
        p50 = percentile(samples, 0.50)
        logical_bytes = 4 * tokens * experts + 16 * tokens
        rows.append({
            "T": tokens, "E": experts, "process_runs": 5, "raw_samples": len(samples),
            "p50_us": p50, "p90_us": percentile(samples, 0.90),
            "p95_us": percentile(samples, 0.95), "mean_us": mean,
            "stddev_us": stddev, "cv": stddev / mean,
            "rows_per_second": tokens * 1.0e6 / p50,
            "effective_logical_gbps": logical_bytes / p50 / 1000.0,
            "workspace_bytes": 0, "launch_count": 1,
        })
    return {
        "schema_version": "raggedroute.triton_aggregate.v1",
        "evidence_boundary": "wsl2_auxiliary_non_promotion",
        "run_id": canonical["run_id"], "repo_commit": canonical["repo_commit"],
        "environment": canonical["environment"], "rows": rows,
    }


def write_outputs(result: dict[str, Any], output_json: pathlib.Path,
                  output_csv: pathlib.Path, report: pathlib.Path) -> None:
    if output_json.exists() or output_csv.exists() or report.exists():
        raise FileExistsError("refusing to overwrite Triton analysis evidence")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["rows"][0].keys())
        writer.writeheader()
        writer.writerows(result["rows"])
    representative = [row for row in result["rows"] if (row["T"], row["E"]) in {
        (2048, 8), (2048, 33), (2048, 64), (32, 64)
    }]
    lines = [
        "# WSL2 Triton Top-K auxiliary release evidence", "",
        "> These absolute results are auxiliary and are not eligible for Windows Auto promotion.", "",
        "| T | E | p50 us | p90 us | p95 us | CV | Mrows/s | Effective GB/s |", 
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in representative:
        lines.append(
            f"| {row['T']} | {row['E']} | {row['p50_us']:.4f} | {row['p90_us']:.4f} | "
            f"{row['p95_us']:.4f} | {row['cv']:.3f} | {row['rows_per_second']/1e6:.3f} | "
            f"{row['effective_logical_gbps']:.3f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs=5, type=pathlib.Path, required=True)
    parser.add_argument("--manifests", nargs=5, type=pathlib.Path, required=True)
    parser.add_argument("--controlled", nargs=5, type=pathlib.Path, required=True)
    parser.add_argument("--json", required=True, type=pathlib.Path)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    parser.add_argument("--report", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = aggregate(args.raw, args.manifests, args.controlled)
    write_outputs(result, args.json, args.csv, args.report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}")
        raise SystemExit(2)
