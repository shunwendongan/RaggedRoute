#!/usr/bin/env python3
"""Run C++ and Triton RaggedRoute variants under one reference-only suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import shlex
import subprocess
import sys
from typing import Any


SCHEMA = "raggedroute.cross_backend_suite.v1"
MANIFEST = "raggedroute.cross_backend_manifest.v1"
LEVEL_NAMES = {
    "l1": "L1_kernel_body",
    "l2": "L2_operator_steady",
    "l3": "L3_chain_steady",
}
OPERATORS = {
    "dense_gemm", "topk_gate", "histogram", "exclusive_scan",
    "token_permute", "grouped_gemm", "unpermute",
}
SUITES = {"chain_from_tokens"}


def command_output(command: list[str], cwd: pathlib.Path) -> str:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def load_suite(path: pathlib.Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != SCHEMA:
        raise ValueError(f"expected schema_version={SCHEMA!r}")
    if suite.get("protocol") not in {"smoke", "release"}:
        raise ValueError("cross-backend suite requires smoke or release protocol")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cross-backend suite requires cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case ids must be present and unique")
        seen.add(case_id)
        operator, target_suite = case.get("operator"), case.get("suite")
        if (operator is None) == (target_suite is None):
            raise ValueError(f"{case_id}: specify exactly one operator or suite")
        if operator is not None and operator not in OPERATORS:
            raise ValueError(f"{case_id}: unknown operator")
        if target_suite is not None and target_suite not in SUITES:
            raise ValueError(f"{case_id}: unknown suite")
        if not case.get("levels") or any(level not in LEVEL_NAMES for level in case["levels"]):
            raise ValueError(f"{case_id}: invalid levels")
        if target_suite is not None and case["levels"] != ["l3"]:
            raise ValueError(f"{case_id}: chain_from_tokens requires exactly level l3")
        if operator is not None and "l3" in case["levels"]:
            raise ValueError(f"{case_id}: individual operators do not support l3")
        variants = case.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError(f"{case_id}: requires at least two variants")
        names = [variant.get("name") for variant in variants]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"{case_id}: variant names must be unique")
        if {variant.get("backend") for variant in variants} - {"cpp", "triton"}:
            raise ValueError(f"{case_id}: backend must be cpp or triton")
        references = [variant for variant in variants if variant.get("reference_baseline") is True]
        if len(references) != 1 or references[0].get("backend") != "triton":
            raise ValueError(f"{case_id}: requires exactly one Triton reference_baseline")
    return suite


def git_state(repo: pathlib.Path) -> tuple[str, bool, str]:
    sha = command_output(["git", "rev-parse", "HEAD"], repo)
    dirty = command_output(["git", "status", "--porcelain"], repo)
    branch = command_output(["git", "branch", "--show-current"], repo)
    return sha, sha.startswith("unavailable:") or dirty.startswith("unavailable:") or bool(dirty), branch


def validate_release(suite: dict[str, Any], dirty: bool) -> None:
    common = suite.get("common", {})
    if int(suite.get("process_runs", 0)) < 3:
        raise ValueError("release requires at least three process runs")
    if int(common.get("warmup", 0)) < 10 or int(common.get("samples", 0)) < 20:
        raise ValueError("release requires warmup >= 10 and samples >= 20")
    if dirty:
        raise ValueError("release requires a clean git worktree")


def merged_common(suite: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    values = dict(suite.get("common", {}))
    values.update(case.get("common", {}))
    return values


def command_for_cpp(binary: pathlib.Path, suite: dict[str, Any], case: dict[str, Any], variant: dict[str, Any], level: str, process_run: int, run_id: str, output: pathlib.Path, expected_sha: str) -> list[str]:
    common = merged_common(suite, case)
    repeats = case.get("kernel_repeats_by_level", {}).get(level, common.get("kernel_repeats", 1))
    target_flag = "--suite" if "suite" in case else "--operator"
    target_name = case.get("suite", case.get("operator"))
    command = [str(binary), target_flag, target_name, "--variant", variant["name"], "--level", level, "--protocol", suite["protocol"], "--cache-mode", case.get("cache_mode", common.get("cache_mode", "warm")), "--warmup", str(common.get("warmup", 5)), "--kernel-repeats", str(repeats), "--samples", str(common.get("samples", 10)), "--process-run", str(process_run), "--seed", str(common.get("seed", 20260729)), "--run-id", run_id, "--case-id", case["id"], "--output", str(output)]
    if expected_sha:
        command.extend(["--expected-git-sha", expected_sha])
    for name, value in sorted(case.get("params", {}).items()):
        if isinstance(value, bool):
            value = str(value).lower()
        command.extend(["--param", f"{name}={value}"])
    return command


def command_for_triton(python: pathlib.Path, repo: pathlib.Path, suite: dict[str, Any], case: dict[str, Any], level: str, process_run: int, run_id: str, output: pathlib.Path) -> list[str]:
    common = merged_common(suite, case)
    repeats = case.get("kernel_repeats_by_level", {}).get(level, common.get("kernel_repeats", 1))
    target_flag = "--suite" if "suite" in case else "--operator"
    target_name = case.get("suite", case.get("operator"))
    command = [str(python), str(repo / "scripts" / "triton_benchmark.py"), target_flag, target_name, "--variant", "triton_reference", "--level", level, "--protocol", suite["protocol"], "--cache-mode", case.get("cache_mode", common.get("cache_mode", "warm")), "--warmup", str(common.get("warmup", 5)), "--kernel-repeats", str(repeats), "--samples", str(common.get("samples", 10)), "--process-run", str(process_run), "--seed", str(common.get("seed", 20260729)), "--run-id", run_id, "--case-id", case["id"], "--output", str(output)]
    for name, value in sorted(case.get("params", {}).items()):
        if isinstance(value, bool):
            value = str(value).lower()
        command.extend(["--param", f"{name}={value}"])
    return command


def output_path_for_cpp(binary: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    """Return a path the native C++ benchmark can open.

    The supported local setup runs the Python/Triton worker in WSL while the
    existing CUDA benchmark is a native Windows executable.  Windows does not
    understand WSL's ``/mnt/c/...`` spelling, so translate only that worker's
    output argument.  Keeping the Triton worker on its POSIX path lets both
    append to the same JSONL file.
    """
    if os.name == "nt" or binary.suffix.lower() != ".exe":
        return output
    translated = subprocess.check_output(["wslpath", "-w", str(output)], text=True).strip()
    if not translated:
        raise RuntimeError(f"could not translate C++ output path {output}")
    return pathlib.Path(translated)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--python", dest="python_executable", type=pathlib.Path, default=pathlib.Path(sys.executable))
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    suite = load_suite(args.config.resolve())
    # Preserve a venv's ``bin/python`` symlink. Resolving it selects the base
    # interpreter and silently drops the venv's Torch/Triton site-packages.
    binary, python_executable, output = args.binary.resolve(), args.python_executable, args.output.resolve()
    if not args.dry_run and not binary.is_file():
        raise FileNotFoundError(binary)
    if not python_executable.is_file():
        raise FileNotFoundError(python_executable)
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"refusing to append to existing output: {output}")
    sha, dirty, branch = git_state(repo)
    if suite["protocol"] == "release":
        validate_release(suite, dirty)
    run_id = args.run_id or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{sha[:12]}-{suite.get('suite_id', 'cross-backend')}"
    output.parent.mkdir(parents=True, exist_ok=True)
    cpp_output = output_path_for_cpp(binary, output)
    gpu_query = ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"]
    manifest = {"schema_version": MANIFEST, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "run_id": run_id, "suite": suite, "suite_path": str(args.config.resolve()), "binary": str(binary), "python": str(python_executable), "output": str(output), "repo_commit": sha, "repo_dirty": dirty, "repo_branch": branch, "host": os.environ.get("COMPUTERNAME") or os.uname().nodename, "gpu_snapshot_start": command_output(gpu_query, repo), "commands": []}
    work = [(case, variant, level) for case in suite["cases"] for variant in case["variants"] for level in case["levels"]]
    for process_run in range(1, int(suite.get("process_runs", 1)) + 1):
        ordered = list(work)
        random.Random(int(suite.get("common", {}).get("seed", 20260729)) + process_run).shuffle(ordered)
        for case, variant, level in ordered:
            command = command_for_cpp(binary, suite, case, variant, level, process_run, run_id, cpp_output, sha if suite["protocol"] == "release" else "") if variant["backend"] == "cpp" else command_for_triton(python_executable, repo, suite, case, level, process_run, run_id, output)
            manifest["commands"].append(command)
            print("+", shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=repo, check=True)
    manifest["gpu_snapshot_end"] = command_output(gpu_query, repo)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
