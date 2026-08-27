#!/usr/bin/env python3
"""Run a versioned RaggedRoute benchmark suite as isolated processes.

This is intentionally separate from CTest/correctness.  Release suites enforce
clean-source, Release-build, validation, sample-count, and multi-process gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import pathlib
import platform
import random
import re
import shlex
import subprocess
import sys
from typing import Any


SCHEMA_V1 = "raggedroute.suite.v1"
SCHEMA_V2 = "raggedroute.suite.v2"
SUPPORTED_SCHEMAS = {SCHEMA_V1, SCHEMA_V2}
PORTABLE_TRACE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def load_route_trace_header(path: pathlib.Path) -> dict[str, Any]:
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 7 or tokens[0] != "raggedroute.route_trace.v1":
        raise ValueError(f"invalid normalized route trace: {path}")
    trace_id, source_kind = tokens[1], tokens[2]
    token_count, experts, top_k, frame_count = map(int, tokens[3:7])
    route_pairs = token_count * top_k
    expected = 7 + frame_count * (1 + route_pairs)
    if (
        not PORTABLE_TRACE_ID.fullmatch(trace_id)
        or source_kind not in {"production", "captured", "synthetic_fixture"}
        or token_count < 1
        or not 1 <= top_k <= experts <= 64
        or frame_count < 1
        or len(tokens) != expected
    ):
        raise ValueError(f"invalid normalized route trace metadata: {path}")
    frame_ids: list[str] = []
    cursor = 7
    for _ in range(frame_count):
        frame_id = tokens[cursor]
        if not PORTABLE_TRACE_ID.fullmatch(frame_id) or frame_id in frame_ids:
            raise ValueError(f"invalid or duplicate route trace frame id: {path}")
        frame_ids.append(frame_id)
        try:
            ids = [int(value) for value in tokens[cursor + 1 : cursor + 1 + route_pairs]]
        except ValueError as error:
            raise ValueError(f"non-integer expert id in route trace: {path}") from error
        if any(value < 0 or value >= experts for value in ids):
            raise ValueError(f"out-of-range expert id in route trace: {path}")
        for token in range(token_count):
            selected = ids[token * top_k : (token + 1) * top_k]
            if len(set(selected)) != top_k:
                raise ValueError(f"repeated per-token expert id in route trace: {path}")
        cursor += 1 + route_pairs
    return {
        "trace_id": trace_id,
        "source_kind": source_kind,
        "T": token_count,
        "E": experts,
        "top_k": top_k,
        "frame_count": frame_count,
        "frame_ids": frame_ids,
    }


def command_output(command: list[str], cwd: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def file_sha256(path: pathlib.Path) -> str:
    if not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        suite = json.load(stream)
    schema = suite.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported schema_version={schema!r}")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("suite must contain a non-empty cases list")
    expanded_cases: list[dict[str, Any]] = []
    for source_case in suite["cases"]:
        matrix = source_case.get("matrix")
        if matrix is None:
            expanded_cases.append(source_case)
            continue
        if not isinstance(matrix, dict) or not matrix:
            raise ValueError(f"case {source_case.get('id')} has an invalid matrix")
        names = sorted(matrix)
        values = [matrix[name] for name in names]
        if any(not isinstance(items, list) or not items for items in values):
            raise ValueError(f"case {source_case.get('id')} matrix axes must be non-empty lists")
        for combination in itertools.product(*values):
            current = dict(source_case)
            current.pop("matrix")
            current["params"] = dict(source_case.get("params", {}))
            suffix = []
            for name, value in zip(names, combination):
                current["params"][name] = value
                suffix.append(f"{name.lower()}{value}")
            current["id"] = source_case["id"] + "." + "_".join(suffix)
            expanded_cases.append(current)
    trace_expanded_cases: list[dict[str, Any]] = []
    repo = pathlib.Path(__file__).resolve().parents[1]
    for source_case in expanded_cases:
        trace_spec = source_case.get("route_trace")
        if trace_spec is None:
            trace_expanded_cases.append(source_case)
            continue
        if not isinstance(trace_spec, dict) or not isinstance(trace_spec.get("path"), str):
            raise ValueError(f"case {source_case.get('id')} has an invalid route_trace")
        relative_path = pathlib.Path(trace_spec["path"])
        trace_path = relative_path if relative_path.is_absolute() else repo / relative_path
        header = load_route_trace_header(trace_path.resolve())
        requested = trace_spec.get("frames", "all")
        frames = list(range(header["frame_count"])) if requested == "all" else requested
        if not isinstance(frames, list) or not frames or any(
            not isinstance(frame, int) or frame < 0 or frame >= header["frame_count"]
            for frame in frames
        ):
            raise ValueError(f"case {source_case.get('id')} has invalid trace frames")
        for frame in frames:
            current = dict(source_case)
            current.pop("route_trace")
            current["params"] = dict(source_case.get("params", {}))
            for name in ("T", "E", "top_k"):
                configured = current["params"].get(name)
                if configured is not None and configured != header[name]:
                    raise ValueError(
                        f"case {source_case.get('id')} {name} conflicts with route trace"
                    )
                current["params"][name] = header[name]
            current["params"]["route_trace_path"] = str(relative_path).replace("\\", "/")
            current["params"]["route_trace_frame"] = frame
            frame_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in header["frame_ids"][frame]
            )
            current["id"] = source_case["id"] + f".trace_{frame_id}"
            trace_expanded_cases.append(current)
    suite["cases"] = trace_expanded_cases
    ids = [case.get("id") for case in suite["cases"]]
    if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("every case requires a unique non-empty id")
    for case in suite["cases"]:
        if bool(case.get("operator")) == bool(case.get("suite")):
            raise ValueError(
                f"case {case['id']} requires exactly one operator or suite"
            )
        if schema == SCHEMA_V1:
            if "variants" in case:
                raise ValueError(f"suite v1 case {case['id']} cannot define variants")
            continue
        if "variant" in case:
            raise ValueError(f"suite v2 case {case['id']} must use variants[]")
        variants = case.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError(
                f"suite v2 case {case['id']} requires at least two variants"
            )
        names = [variant.get("name") for variant in variants if isinstance(variant, dict)]
        if len(names) != len(variants) or any(not name for name in names):
            raise ValueError(f"case {case['id']} has an invalid variant entry")
        if len(names) != len(set(names)):
            raise ValueError(f"case {case['id']} has duplicate variant names")
        baselines = [
            variant for variant in variants if variant.get("promotion_baseline") is True
        ]
        if len(baselines) != 1:
            raise ValueError(
                f"case {case['id']} requires exactly one promotion_baseline"
            )
    return suite


def case_variants(
    suite: dict[str, Any], case: dict[str, Any]
) -> list[dict[str, Any]]:
    if suite["schema_version"] == SCHEMA_V2:
        return list(case["variants"])
    return [
        {
            "name": case.get("variant", "cuda_naive"),
            "promotion_baseline": True,
        }
    ]


def git_state(repo: pathlib.Path) -> tuple[str, bool, str]:
    sha = command_output(["git", "rev-parse", "HEAD"], repo)
    status = command_output(["git", "status", "--porcelain"], repo)
    branch = command_output(["git", "branch", "--show-current"], repo)
    unavailable = sha.startswith("unavailable:") or status.startswith("unavailable:")
    # Release evaluation must fail closed when Git provenance cannot be read.
    return sha, unavailable or bool(status), branch


def validate_release(suite: dict[str, Any], dirty: bool) -> None:
    common = suite.get("common", {})
    if suite.get("process_runs", 0) < 3:
        raise ValueError("release suites require process_runs >= 3")
    if common.get("warmup", 0) < 10 or common.get("samples", 0) < 20:
        raise ValueError("release suites require warmup >= 10 and samples >= 20")
    if dirty:
        raise ValueError("release suites require a clean git worktree")
    for case in suite["cases"]:
        if not case.get("levels"):
            raise ValueError(f"release case {case['id']} has no measurement levels")


def case_command(
    binary: pathlib.Path,
    suite: dict[str, Any],
    case: dict[str, Any],
    level: str,
    process_run: int,
    run_id: str,
    output: pathlib.Path,
    expected_git_sha: str = "",
    variant: dict[str, Any] | None = None,
) -> list[str]:
    common = dict(suite.get("common", {}))
    common.update(case.get("common", {}))
    repeats = case.get("kernel_repeats_by_level", {}).get(
        level, common.get("kernel_repeats", 1)
    )
    target_flag = "--operator" if case.get("operator") else "--suite"
    target_name = case.get("operator") or case["suite"]
    variant_name = (
        variant["name"] if variant is not None else case.get("variant", "cuda_naive")
    )
    command = [
        str(binary),
        target_flag,
        target_name,
        "--variant",
        variant_name,
        "--level",
        level,
        "--protocol",
        suite.get("protocol", "smoke"),
        "--cache-mode",
        case.get("cache_mode", common.get("cache_mode", "warm")),
        "--warmup",
        str(common.get("warmup", 5)),
        "--kernel-repeats",
        str(repeats),
        "--samples",
        str(common.get("samples", 10)),
        "--process-run",
        str(process_run),
        "--seed",
        # Independent processes intentionally replay identical inputs. Changing
        # the seed would mix workload variance into process-noise statistics.
        str(int(common.get("seed", 20260729))),
        "--run-id",
        run_id,
        "--case-id",
        case["id"],
        "--output",
        str(output),
    ]
    scrub_bytes = case.get(
        "cache_scrub_bytes", common.get("cache_scrub_bytes", 0)
    )
    if scrub_bytes:
        command.extend(["--cache-scrub-bytes", str(scrub_bytes)])
    if expected_git_sha:
        command.extend(["--expected-git-sha", expected_git_sha])
    for name, value in sorted(case.get("params", {}).items()):
        if isinstance(value, bool):
            value = "true" if value else "false"
        command.extend(["--param", f"{name}={value}"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    suite = load_suite(args.config.resolve())
    binary = args.binary.resolve()
    output = args.output.resolve()
    if not args.dry_run and not binary.is_file():
        raise FileNotFoundError(binary)
    if output.exists() and not args.dry_run:
        raise FileExistsError(
            f"refusing to overwrite or append an existing run: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    sha, dirty, branch = git_state(repo)
    if suite.get("protocol", "smoke") == "release":
        validate_release(suite, dirty)
    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + sha[:12]
        + "-"
        + suite.get("suite_id", "suite")
    )

    gpu_query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    gpu_process_query = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    manifest = {
        "schema_version": (
            "raggedroute.run_manifest.v2"
            if suite["schema_version"] == SCHEMA_V2
            else "raggedroute.run_manifest.v1"
        ),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "suite": suite,
        "suite_path": str(args.config.resolve()),
        "binary": str(binary),
        "output": str(output),
        "repo_commit": sha,
        "repo_dirty": dirty,
        "repo_branch": branch,
        "host": os.environ.get("COMPUTERNAME") or platform.node(),
        "gpu_snapshot_fields": [
            "name", "uuid", "pci_bus_id", "driver_version", "memory_mib",
            "pstate", "sm_clock_mhz", "memory_clock_mhz", "power_w",
            "power_limit_w", "temperature_c",
        ],
        "gpu_snapshot_start": command_output(gpu_query, repo),
        "gpu_processes_start": command_output(gpu_process_query, repo),
        "nvcc": command_output(["nvcc", "--version"], repo),
        "compile_commands_sha256": file_sha256(
            binary.parent / "compile_commands.json"
        ),
        "route_traces": [
            {
                "path": path,
                "sha256": file_sha256(
                    (repo / path).resolve() if not pathlib.Path(path).is_absolute()
                    else pathlib.Path(path)
                ),
            }
            for path in sorted(
                {
                    str(case.get("params", {}).get("route_trace_path"))
                    for case in suite["cases"]
                    if case.get("params", {}).get("route_trace_path")
                }
            )
        ],
        "commands": [],
    }

    process_runs = int(suite.get("process_runs", 1))
    seed = int(suite.get("common", {}).get("seed", 20260729))
    for process_run in range(1, process_runs + 1):
        work = [
            (case, variant, level)
            for case in suite["cases"]
            for variant in case_variants(suite, case)
            for level in case.get("levels", ["l1"])
        ]
        random.Random(seed + process_run).shuffle(work)
        for case, variant, level in work:
            command = case_command(
                binary, suite, case, level, process_run, run_id, output,
                sha if suite.get("protocol", "smoke") == "release" else "",
                variant,
            )
            manifest["commands"].append(command)
            print("+", shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=repo, check=True)

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest["gpu_snapshot_end"] = command_output(gpu_query, repo)
    manifest["gpu_processes_end"] = command_output(gpu_process_query, repo)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"wrote {output}")
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # CLI boundary
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
