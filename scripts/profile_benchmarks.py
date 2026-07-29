#!/usr/bin/env python3
"""Capture Nsight Compute reports for diagnostic cases only.

The target uses --profile-once and does not emit benchmark timing records.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import Any


KERNEL_PATTERNS = {
    "dense_gemm": "dense_gemm_naive_kernel",
    "topk_gate": "topk_gate_naive_kernel",
    "histogram": "histogram_naive_kernel",
    "exclusive_scan": "exclusive_scan_naive_kernel",
    "token_permute": "token_permute_naive_kernel",
    "grouped_gemm": "grouped_gemm_naive_kernel",
    "unpermute": "unpermute_naive_kernel",
}


def load_config(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != "raggedroute.profile_suite.v1":
        raise ValueError("unsupported profile suite schema")
    if not config.get("cases"):
        raise ValueError("profile suite is empty")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    binary = args.binary.resolve()
    output_dir = args.output_dir.resolve()
    if not args.dry_run and not binary.is_file():
        raise FileNotFoundError(binary)
    output_dir.mkdir(parents=True, exist_ok=True)
    ncu_executable = shutil.which(args.ncu) or args.ncu

    selected = [
        case
        for case in config["cases"]
        if not args.case_ids or case["id"] in set(args.case_ids)
    ]
    if not selected:
        raise ValueError("no profile case matched --case")
    for case in selected:
        operator = case["operator"]
        report = output_dir / case["id"]
        if report.with_suffix(".ncu-rep").exists():
            raise FileExistsError(report.with_suffix(".ncu-rep"))
        pattern = case.get("kernel_pattern", KERNEL_PATTERNS[operator])
        command = [
            ncu_executable,
            "--target-processes", "all",
            "--kernel-name-base", "demangled",
            "--kernel-name", f"regex:.*{pattern}.*",
            "--set", case.get("set", "full"),
            "--cache-control", case.get("cache_control", "all"),
            "--clock-control", case.get("clock_control", "base"),
            "--export", str(report),
            str(binary),
            "--operator", operator,
            "--variant", case.get("variant", "cuda_naive"),
            "--level", case.get("level", "l1"),
            "--profile-once",
            "--seed", str(case.get("seed", 20260729)),
        ]
        for name, value in sorted(case.get("params", {}).items()):
            if isinstance(value, bool):
                value = "true" if value else "false"
            command.extend(["--param", f"{name}={value}"])
        print("+", shlex.join(command), flush=True)
        if not args.dry_run:
            if pathlib.Path(ncu_executable).suffix.lower() in {".bat", ".cmd"}:
                subprocess.run(
                    [
                        os.environ.get("COMSPEC", "cmd.exe"),
                        "/d",
                        "/c",
                        "call",
                        ncu_executable,
                        *command[1:],
                    ],
                    check=True,
                )
            else:
                subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
