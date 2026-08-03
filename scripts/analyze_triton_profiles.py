#!/usr/bin/env python3
"""Normalize NCU and NSYS evidence from the WSL2 Triton Top-K profiler."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pathlib
import sys

import profile_benchmarks as profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    reports = args.run_dir / "reports"
    analysis = args.run_dir / "analysis"
    ncu = profile.find_tool("ncu")
    report_python = profile.find_ncu_report_path()
    if not ncu or report_python is None:
        raise RuntimeError("Windows NCU or ncu_report.py is unavailable")
    sys.path.insert(0, str(report_python))
    ncu_report = importlib.import_module("ncu_report")
    supported = profile.query_supported_metrics(ncu)
    actions = []
    for report in sorted(reports.glob("*.ncu-rep")):
        context = ncu_report.load_report(str(report.resolve()))
        case_id = report.name[: -len(".basic.ncu-rep")]
        current_actions = list(profile.iter_actions(context))
        if not current_actions:
            raise ValueError(f"no NCU action in {report}")
        for action in current_actions:
            actions.append({
                "case_id": case_id,
                "profile_set": "basic",
                "kernel": profile.action_name(action),
                "metrics": profile.metric_snapshot(action, supported),
            })
    if not actions:
        raise ValueError("no Triton NCU reports found")
    profile.write_json(analysis / "ncu_metrics.json", actions)
    profile.write_ncu_csv(analysis / "ncu_metrics.csv", actions)

    hotspots = {}
    for path in sorted(analysis.glob("*.cuda_gpu_kern_sum.csv")):
        case_id = path.name[: -len(".cuda_gpu_kern_sum.csv")]
        hotspots[case_id] = profile.rank_nsys_kernels(path)
    profile.write_json(analysis / "nsys_hotspots.json", hotspots)
    lines = [
        "# WSL2 Triton Top-K Nsight diagnostics", "",
        "> Profiler duration is diagnostic only; release latency comes from unprofiled CUDA events.", "",
        "| Case | Kernel | Waves/SM | Occupancy % | SM % | Memory % | DRAM % | Registers/thread |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    wanted = ("waves_per_sm", "occupancy_achieved_pct", "sm_throughput_pct",
              "memory_throughput_pct", "dram_throughput_pct", "registers_per_thread")
    for action in actions:
        values = [action["metrics"][name].get("value") for name in wanted]
        lines.append(
            f"| {action['case_id']} | `{action['kernel']}` | "
            + " | ".join(str(value) if value is not None else "n/a" for value in values)
            + " |"
        )
    (analysis / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
