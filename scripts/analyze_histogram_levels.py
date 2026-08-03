#!/usr/bin/env python3
"""Pair Histogram L1/L2 aggregates and quantify reset/wrapper overhead."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any


def load_groups(path: pathlib.Path, level: str) -> dict[tuple[str, str], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = "L1_kernel_body" if level == "l1" else "L2_operator_steady"
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for group in document.get("groups", []):
        if group.get("operator") != "histogram" or group.get("measurement_level") != expected:
            continue
        key = (str(group["case_id"]), str(group["variant"]))
        if key in groups:
            raise ValueError(f"duplicate {level} group: {key}")
        groups[key] = group
    return groups


def analyze(l1_path: pathlib.Path, l2_path: pathlib.Path) -> dict[str, Any]:
    l1 = load_groups(l1_path, "l1")
    l2 = load_groups(l2_path, "l2")
    rows: list[dict[str, Any]] = []
    for key in sorted(set(l1) & set(l2)):
        first, second = l1[key], l2[key]
        for field in ("case_config", "cache_mode", "seed", "process_runs", "samples"):
            if first.get(field) != second.get(field):
                raise ValueError(f"{key} mismatches {field}")
        l1_p50 = float(first["median_of_process_medians_us"])
        l2_p50 = float(second["median_of_process_medians_us"])
        l1_p95 = float(first["all_samples_p95_us"])
        l2_p95 = float(second["all_samples_p95_us"])
        route_pairs = int(first["case_config"]["R"])
        l1_gitems = route_pairs / l1_p50 / 1000.0
        l2_gitems = route_pairs / l2_p50 / 1000.0
        rows.append(
            {
                "case_id": key[0],
                "variant": key[1],
                "R": route_pairs,
                "E": int(first["case_config"]["E"]),
                "distribution": first["case_config"]["distribution"],
                "l1_p50_us": l1_p50,
                "l2_p50_us": l2_p50,
                "l2_minus_l1_p50_us": l2_p50 - l1_p50,
                "l2_over_l1_p50_ratio": l2_p50 / l1_p50,
                "l1_p95_us": l1_p95,
                "l2_p95_us": l2_p95,
                "l2_minus_l1_p95_us": l2_p95 - l1_p95,
                "l2_over_l1_p95_ratio": l2_p95 / l1_p95,
                "reset_wrapper_share": (l2_p50 - l1_p50) / l2_p50,
                "l1_gitems_s": l1_gitems,
                "l2_gitems_s": l2_gitems,
                "gitems_s_loss": l1_gitems - l2_gitems,
                "throughput_loss_fraction": (l1_gitems - l2_gitems) / l1_gitems,
            }
        )
    return {
        "schema_version": "raggedroute.histogram_level_gap.v1",
        "l1_source": str(l1_path.resolve()),
        "l2_source": str(l2_path.resolve()),
        "unpaired_l1": [list(key) for key in sorted(set(l1) - set(l2))],
        "unpaired_l2": [list(key) for key in sorted(set(l2) - set(l1))],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1", required=True, type=pathlib.Path)
    parser.add_argument("--l2", required=True, type=pathlib.Path)
    parser.add_argument("--json", required=True, type=pathlib.Path)
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = analyze(args.l1, args.l2)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(result["rows"][0]) if result["rows"] else []
    with args.csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    print(f"wrote {args.json}")
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
