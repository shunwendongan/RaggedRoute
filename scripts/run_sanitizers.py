#!/usr/bin/env python3
"""Run the small correctness edge suite under each Compute Sanitizer tool."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--executable",
        type=Path,
        help="Override the correctness framework executable path.",
    )
    return parser.parse_args()


def find_executable(build_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    names = (
        "raggedroute_correctness_framework_tests.exe",
        "raggedroute_correctness_framework_tests",
    )
    for name in names:
        candidate = build_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "correctness framework executable not found; build the selected configuration first"
    )


def main() -> int:
    args = parse_args()
    sanitizer = shutil.which("compute-sanitizer") or shutil.which("compute-sanitizer.bat")
    if sanitizer is None:
        print("Compute Sanitizer was not found on PATH", file=sys.stderr)
        return 2
    try:
        executable = find_executable(args.build_dir, args.executable)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    tools = ("memcheck", "initcheck", "racecheck", "synccheck")
    log_paths = {tool: args.output_dir / f"{tool}.log" for tool in tools}
    existing = [path for path in log_paths.values() if path.exists()]
    if existing:
        print(
            "refusing to overwrite existing sanitizer logs: "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for tool in tools:
        command = [
            sanitizer,
            "--tool",
            tool,
            "--error-exitcode",
            "99",
            str(executable),
            "--suite",
            "edge",
        ]
        execution = command
        if Path(sanitizer).suffix.lower() in {".bat", ".cmd"}:
            execution = [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/d",
                "/c",
                "call",
                sanitizer,
                *command[1:],
            ]
        completed = subprocess.run(execution, text=True, capture_output=True, check=False)
        log_path = log_paths[tool]
        log_path.write_text(
            "Command: " + shlex.join(command) + "\n\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            failures.append(f"{tool} (exit {completed.returncode}; {log_path})")
        else:
            print(f"PASS {tool}: {log_path}")
    if failures:
        print("Compute Sanitizer failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
