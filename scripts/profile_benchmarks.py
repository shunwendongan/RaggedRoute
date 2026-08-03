#!/usr/bin/env python3
"""Collect and analyze reproducible RaggedRoute Nsight evidence.

Profiler durations are diagnostic evidence only. Release latency always comes
from the unprofiled benchmark runner.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from resolve_cuda_toolkit import runtime_environment


SCHEMA = "raggedroute.profile_run.v1"
PROFILE_SCHEMAS = {"raggedroute.profile_suite.v1", "raggedroute.profile_suite.v2"}
KERNEL_PATTERNS = {
    "dense_gemm": "dense_gemm_naive_kernel",
    "topk_gate": "topk_gate_naive_kernel",
    "histogram": "histogram_naive_kernel",
    "exclusive_scan": "exclusive_scan_naive_kernel",
    "token_permute": "token_permute_naive_kernel",
    "grouped_gemm": "grouped_gemm_naive_kernel",
    "unpermute": "unpermute_naive_kernel",
}
NSYS_STATS_REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_time_sum",
)

METRIC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "device_name": ("device__attribute_display_name",),
    "cc_major": ("device__attribute_compute_capability_major",),
    "cc_minor": ("device__attribute_compute_capability_minor",),
    "sm_count": ("device__attribute_multiprocessor_count",),
    "duration_ns": ("gpu__time_duration.sum", "gpu__time_duration.avg"),
    "grid_size": ("launch__grid_size",),
    "block_size": ("launch__block_size",),
    "waves_per_sm": ("launch__waves_per_multiprocessor",),
    "registers_per_thread": ("launch__registers_per_thread",),
    "shared_mem_per_block": ("launch__shared_mem_per_block",),
    "occupancy_theoretical_pct": ("sm__maximum_warps_per_active_cycle_pct",),
    "occupancy_achieved_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "sm_throughput_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "memory_throughput_pct": (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "dram_throughput_pct": (
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "dram_read_bytes": ("dram__bytes_read.sum",),
    "dram_write_bytes": ("dram__bytes_write.sum",),
    "l1_throughput_pct": ("l1tex__throughput.avg.pct_of_peak_sustained_active",),
    "l1_hit_rate_pct": (
        "l1tex__t_sector_hit_rate.pct",
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    ),
    "l2_hit_rate_pct": ("lts__t_sector_hit_rate.pct",),
    "global_load_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",),
    "global_load_requests": ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",),
    "global_store_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",),
    "global_store_requests": ("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",),
    "local_load_instructions": (
        "smsp__sass_inst_executed_op_local_ld.sum",
        "smsp__inst_executed_op_local_ld.sum",
    ),
    "local_store_instructions": (
        "smsp__sass_inst_executed_op_local_st.sum",
        "smsp__inst_executed_op_local_st.sum",
    ),
    "tensor_active_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ),
    "eligible_warps_per_scheduler": ("smsp__warps_eligible.avg.per_cycle_active",),
    "issue_active_pct": (
        "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
    ),
    "stall_long_scoreboard": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    ),
    "stall_short_scoreboard": (
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    ),
    "stall_wait": (
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__warps_issue_stalled_wait_per_warp_active.pct",
        "smsp__pcsamp_warps_issue_stalled_wait",
    ),
    "stall_barrier": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__warps_issue_stalled_barrier_per_warp_active.pct",
        "smsp__pcsamp_warps_issue_stalled_barrier",
    ),
    "stall_math_pipe": (
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    ),
    "stall_memory_throttle": (
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
        "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    ),
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_config(path: pathlib.Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    schema = config.get("schema_version")
    if schema not in PROFILE_SCHEMAS:
        raise ValueError("unsupported profile suite schema")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("profile suite is empty")
    ids = [case.get("id") for case in cases]
    operators = [case.get("operator") for case in cases]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("profile case ids must be non-empty and unique")
    if any(operator not in KERNEL_PATTERNS for operator in operators):
        raise ValueError("profile suite contains an unknown operator")
    if schema == "raggedroute.profile_suite.v2":
        if set(operators) != set(KERNEL_PATTERNS) or len(cases) != len(KERNEL_PATTERNS):
            raise ValueError("profile suite v2 requires exactly one case for every operator")
        system_case = config.get("system_case")
        if not isinstance(system_case, dict) or not system_case.get("suite"):
            raise ValueError("profile suite v2 requires a system_case")
    return config


def find_tool(name: str) -> str | None:
    return shutil.which(name) or shutil.which(f"{name}.bat") or shutil.which(f"{name}.exe")


def run_command(
    command: list[str], *, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    actual = command
    if os.name == "nt" and pathlib.Path(command[0]).suffix.casefold() in {".bat", ".cmd"}:
        actual = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "call", *command]
    print("+", shlex.join(command), flush=True)
    return subprocess.run(
        actual,
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def version(tool: str | None, arguments: list[str]) -> str | None:
    if not tool:
        return None
    result = run_command([tool, *arguments], capture=True, check=False)
    return (result.stdout + result.stderr).strip() or None


def find_ncu_report_path() -> pathlib.Path | None:
    override = os.environ.get("NCU_REPORT_PYTHON")
    candidates: list[pathlib.Path] = []
    if override:
        candidates.append(pathlib.Path(override))
    ncu = find_tool("ncu")
    if ncu:
        root = pathlib.Path(ncu).resolve().parent
        candidates.extend((root / "extras" / "python", root.parent / "extras" / "python"))
    for variable in ("ProgramFiles", "ProgramW6432"):
        base = os.environ.get(variable)
        if base:
            candidates.extend(
                (pathlib.Path(base) / "NVIDIA Corporation").glob(
                    "Nsight Compute */extras/python"
                )
            )
    found = [path.resolve() for path in candidates if (path / "ncu_report.py").is_file()]
    return sorted(found, key=lambda path: tuple(map(int, re.findall(r"\d+", str(path)))), reverse=True)[0] if found else None


def doctor_data() -> dict[str, Any]:
    resolved_cuda = None
    if os.name == "nt":
        updated, resolved_cuda = runtime_environment(pathlib.Path(__file__).resolve().parents[1])
        os.environ.update(updated)
    tools = {name: find_tool(name) for name in ("nvcc", "ncu", "nsys", "nvidia-smi", "python")}
    gpu: dict[str, Any] | None = None
    if tools["nvidia-smi"]:
        result = run_command(
            [
                tools["nvidia-smi"],
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
            if len(fields) == 5:
                gpu = dict(
                    zip(
                        ("name", "uuid", "driver", "memory_mib", "compute_capability"),
                        fields,
                    )
                )
    return {
        "schema_version": SCHEMA,
        "created_utc": now_utc(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "gpu": gpu,
        "tools": tools,
        "cuda_toolkit": resolved_cuda.as_dict() if resolved_cuda else None,
        "versions": {
            "nvcc": version(tools["nvcc"], ["--version"]),
            "ncu": version(tools["ncu"], ["--version"]),
            "nsys": version(tools["nsys"], ["--version"]),
        },
        "ncu_report_python": str(find_ncu_report_path()) if find_ncu_report_path() else None,
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_paths(run_dir: pathlib.Path, create: bool) -> dict[str, pathlib.Path]:
    root = run_dir.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(root)
    reports = root / "reports"
    analysis = root / "analysis"
    if create:
        reports.mkdir(exist_ok=True)
        analysis.mkdir(exist_ok=True)
    return {
        "root": root,
        "reports": reports,
        "analysis": analysis,
        "manifest": root / "manifest.json",
    }


def append_manifest(paths: dict[str, pathlib.Path], event: dict[str, Any]) -> None:
    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": SCHEMA,
            "run_id": paths["root"].name,
            "created_utc": now_utc(),
            "events": [],
        }
    event["created_utc"] = now_utc()
    manifest["events"].append(event)
    write_json(paths["manifest"], manifest)


def value_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def target_command(binary: pathlib.Path, case: dict[str, Any], *, system: bool = False) -> list[str]:
    command = [str(binary)]
    if system:
        command.extend(("--suite", str(case["suite"])))
    else:
        command.extend(("--operator", str(case["operator"])))
    command.extend(
        (
            "--variant",
            str(case.get("variant", "cuda_naive")),
            "--level",
            str(case.get("level", "l3" if system else "l1")),
            "--profile-once",
            "--warmup",
            str(case.get("warmup", 5)),
            "--seed",
            str(case.get("seed", 20260729)),
        )
    )
    for name, value in sorted(case.get("params", {}).items()):
        command.extend(("--param", f"{name}={value_string(value)}"))
    return command


def compute_commands(
    binary: pathlib.Path,
    config: dict[str, Any],
    reports: pathlib.Path,
    ncu: str,
    case_ids: Iterable[str] | None = None,
    profile_set: str | None = None,
) -> list[tuple[str, list[str], pathlib.Path]]:
    selected_ids = set(case_ids or ())
    selected = [case for case in config["cases"] if not selected_ids or case["id"] in selected_ids]
    if not selected:
        raise ValueError("no profile case matched --case")
    commands = []
    for case in selected:
        current_set = profile_set or case.get("set", "basic")
        report_base = reports / f"{case['id']}.{current_set}"
        report = pathlib.Path(f"{report_base}.ncu-rep")
        warmup = int(case.get("warmup", 5))
        pattern = case.get("kernel_pattern", KERNEL_PATTERNS[case["operator"]])
        command = [
            ncu,
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:.*{pattern}.*",
            "--set",
            current_set,
            "--cache-control",
            str(case.get("cache_control", "all")),
            "--clock-control",
            "none",
            "--launch-skip",
            str(warmup),
            "--launch-count",
            "1",
            "--export",
            str(report_base),
            *target_command(binary, case),
        ]
        commands.append((case["id"], command, report))
    return commands


def extract_csv(text: str) -> str:
    lines = text.replace("\r\n", "\n").splitlines()
    for index, line in enumerate(lines):
        if "," in line and any(word in line.casefold() for word in ("time", "name", "instances")):
            return "\n".join(lines[index:]).strip() + "\n"
    return text.strip() + "\n"


def number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def rank_nsys_kernels(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ranked = []
    for row in rows:
        normalized = {str(key).strip(): value for key, value in row.items() if key is not None}
        name_key = next((key for key in normalized if key.casefold() == "name"), None)
        total_key = next(
            (key for key in normalized if "total" in key.casefold() and "time" in key.casefold()),
            None,
        )
        pct_key = next((key for key in normalized if "time" in key.casefold() and "%" in key), None)
        if name_key:
            ranked.append(
                {
                    "name": normalized[name_key],
                    "total_time": number(normalized.get(total_key)) if total_key else None,
                    "time_pct": number(normalized.get(pct_key)) if pct_key else None,
                    "raw": normalized,
                }
            )
    ranked.sort(key=lambda item: item["total_time"] or item["time_pct"] or 0, reverse=True)
    return ranked


def export_nsys_stats(
    nsys: str, report: pathlib.Path, analysis: pathlib.Path
) -> dict[str, dict[str, Any]]:
    """Export every normalized NSYS summary from one immutable trace."""
    stats: dict[str, dict[str, Any]] = {}
    for report_name in NSYS_STATS_REPORTS:
        command = [
            nsys,
            "stats",
            "--force-export=true",
            "--report",
            report_name,
            "--format",
            "csv",
            str(report),
        ]
        result = run_command(command, capture=True, check=False)
        output = analysis / f"nsys_{report_name}.csv"
        if result.returncode == 0:
            output.write_text(extract_csv(result.stdout), encoding="utf-8")
            stats[report_name] = {
                "status": "collected",
                "path": str(output),
                "command": command,
            }
        else:
            if output.exists():
                output.unlink()
            stats[report_name] = {
                "status": "failed",
                "error": (result.stderr or result.stdout).strip(),
                "command": command,
            }
    return stats


def safe_metric(action: Any, name: str) -> tuple[Any, str | None]:
    try:
        metric = action[name]
        value = metric.value()
    except Exception:
        return None, None
    try:
        unit = metric.unit()
    except Exception:
        unit = None
    return value, unit


def query_supported_metrics(ncu: str) -> set[str]:
    result = run_command(
        [ncu, "--query-metrics", "--query-metrics-mode", "all", "--devices", "0"],
        capture=True,
        check=False,
    )
    return {line.split(maxsplit=1)[0] for line in result.stdout.splitlines() if line.strip()} if result.returncode == 0 else set()


def metric_snapshot(action: Any, supported: set[str]) -> dict[str, dict[str, Any]]:
    collected = set(action.metric_names())
    snapshot: dict[str, dict[str, Any]] = {}
    for concept, candidates in METRIC_CONCEPTS.items():
        selected = next((name for name in candidates if name in collected), None)
        if selected:
            value, unit = safe_metric(action, selected)
            snapshot[concept] = {"status": "collected", "metric": selected, "value": value, "unit": unit}
        elif any(name in supported for name in candidates):
            snapshot[concept] = {"status": "not_collected", "metric": None, "value": None, "unit": None}
        else:
            snapshot[concept] = {"status": "unsupported_or_unknown", "metric": None, "value": None, "unit": None}
    return snapshot


def iter_actions(context: Any):
    for range_index in range(context.num_ranges()):
        current = context.range_by_idx(range_index)
        if current is None:
            continue
        for action_index in range(current.num_actions()):
            action = current.action_by_idx(action_index)
            if action is not None:
                yield action


def action_name(action: Any) -> str:
    try:
        return str(action.name())
    except Exception:
        return "unknown"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_ncu_csv(path: pathlib.Path, actions: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("case_id", "profile_set", "kernel", "concept", "status", "metric", "value", "unit"),
        )
        writer.writeheader()
        for action in actions:
            for concept, record in action["metrics"].items():
                writer.writerow(
                    {
                        "case_id": action["case_id"],
                        "profile_set": action["profile_set"],
                        "kernel": action["kernel"],
                        "concept": concept,
                        **record,
                    }
                )


def markdown_report(run_id: str, hotspots: list[dict[str, Any]], actions: list[dict[str, Any]]) -> str:
    lines = [
        f"# RaggedRoute Nsight 诊断：{run_id}",
        "",
        "> NCU/NSYS duration 只用于诊断，不是 Release benchmark latency。",
        "",
        "## NSYS kernel 热点",
        "",
        "| Rank | Kernel | Time % | Total time |",
        "|---:|---|---:|---:|",
    ]
    for index, item in enumerate(hotspots[:15], 1):
        lines.append(f"| {index} | `{item['name']}` | {item.get('time_pct')} | {item.get('total_time')} |")
    lines.extend(
        [
            "",
            "## NCU headline metrics",
            "",
            "| Case / set | Kernel | Waves/SM | Achieved occupancy % | SM % | Memory % | DRAM % | Registers/thread |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    wanted = (
        "waves_per_sm",
        "occupancy_achieved_pct",
        "sm_throughput_pct",
        "memory_throughput_pct",
        "dram_throughput_pct",
        "registers_per_thread",
    )
    for action in actions:
        values = [action["metrics"][name].get("value") for name in wanted]
        lines.append(
            f"| {action['case_id']} / {action['profile_set']} | `{action['kernel']}` | "
            + " | ".join(str(value) if value is not None else "n/a" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- `not_collected` 表示当前 collection set 未采集；`unsupported_or_unknown` 表示当前 GPU/NCU 未确认支持。",
            "- Release p50/p95/CV 必须来自独立的无 profiler benchmark bundle。",
            "- 该系统 trace 使用仓库定义的固定输入，不宣称是真实生产 route trace。",
            "",
        ]
    )
    return "\n".join(lines)


def command_doctor(args: argparse.Namespace) -> int:
    data = doctor_data()
    if args.output:
        write_json(args.output.resolve(), data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    required = (data["tools"].get("ncu"), data["tools"].get("nsys"), data["ncu_report_python"])
    return 0 if all(required) else 2


def command_compute(args: argparse.Namespace) -> int:
    config = load_config(args.config.resolve())
    binary = args.binary.resolve()
    if not args.dry_run and not binary.is_file():
        raise FileNotFoundError(binary)
    ncu = find_tool("ncu")
    if not ncu:
        raise RuntimeError("ncu was not found")
    reports = args.run_dir.resolve() / "reports" if args.dry_run else run_paths(args.run_dir, True)["reports"]
    commands = compute_commands(binary, config, reports, ncu, args.case_ids, args.profile_set)
    paths = None if args.dry_run else run_paths(args.run_dir, True)
    for case_id, command, report in commands:
        if report.exists():
            raise FileExistsError(report)
        if args.dry_run:
            print("+", shlex.join(command), flush=True)
        else:
            run_command(command)
            if not report.is_file():
                raise RuntimeError(f"NCU did not create {report}")
            append_manifest(
                paths,
                {"kind": "compute", "case_id": case_id, "command": command, "report": str(report)},
            )
    return 0


def command_system(args: argparse.Namespace) -> int:
    config = load_config(args.config.resolve())
    if config.get("schema_version") != "raggedroute.profile_suite.v2":
        raise ValueError("system collection requires profile suite v2")
    binary = args.binary.resolve()
    if not args.dry_run and not binary.is_file():
        raise FileNotFoundError(binary)
    nsys = find_tool("nsys")
    if not nsys:
        raise RuntimeError("nsys was not found")
    reports = args.run_dir.resolve() / "reports" if args.dry_run else run_paths(args.run_dir, True)["reports"]
    report_base = reports / "system.full7"
    report = pathlib.Path(f"{report_base}.nsys-rep")
    if report.exists():
        raise FileExistsError(report)
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=false",
        f"--output={report_base}",
        *target_command(binary, config["system_case"], system=True),
    ]
    if args.dry_run:
        print("+", shlex.join(command), flush=True)
        return 0
    paths = run_paths(args.run_dir, True)
    run_command(command)
    if not report.is_file():
        raise RuntimeError(f"NSYS did not create {report}")
    stats = export_nsys_stats(nsys, report, paths["analysis"])
    hotspots = rank_nsys_kernels(paths["analysis"] / "nsys_cuda_gpu_kern_sum.csv")
    write_json(paths["analysis"] / "nsys_hotspots.json", hotspots)
    append_manifest(paths, {"kind": "system", "command": command, "report": str(report), "stats": stats})
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    paths = run_paths(args.run_dir, False)
    ncu = find_tool("ncu")
    report_path = find_ncu_report_path()
    if not ncu or report_path is None:
        raise RuntimeError("ncu or ncu_report.py was not found")
    sys.path.insert(0, str(report_path))
    ncu_report = importlib.import_module("ncu_report")
    supported = query_supported_metrics(ncu)
    actions: list[dict[str, Any]] = []
    raw_reports = []
    for report in sorted(paths["reports"].glob("*.ncu-rep")):
        stem = report.name[: -len(".ncu-rep")]
        case_id, profile_set = stem.rsplit(".", 1)
        context = ncu_report.load_report(str(report))
        report_actions = list(iter_actions(context))
        if not report_actions:
            raise ValueError(f"no NCU actions found in {report}")
        for action in report_actions:
            actions.append(
                {
                    "case_id": case_id,
                    "profile_set": profile_set,
                    "kernel": action_name(action),
                    "metrics": metric_snapshot(action, supported),
                }
            )
        raw_reports.append(
            {"path": str(report), "bytes": report.stat().st_size, "sha256": sha256_file(report)}
        )
    if not actions:
        raise ValueError("run contains no NCU reports")
    nsys_reports = sorted(paths["reports"].glob("*.nsys-rep"))
    if len(nsys_reports) > 1:
        raise ValueError("profile run must contain at most one NSYS system report")
    nsys_stats: dict[str, dict[str, Any]] = {}
    if nsys_reports:
        nsys = find_tool("nsys")
        if not nsys:
            raise RuntimeError("nsys was not found")
        nsys_stats = export_nsys_stats(nsys, nsys_reports[0], paths["analysis"])
    for report in nsys_reports:
        raw_reports.append(
            {"path": str(report), "bytes": report.stat().st_size, "sha256": sha256_file(report)}
        )
    write_json(paths["analysis"] / "ncu_metrics.json", actions)
    write_ncu_csv(paths["analysis"] / "ncu_metrics.csv", actions)
    hotspots = rank_nsys_kernels(paths["analysis"] / "nsys_cuda_gpu_kern_sum.csv")
    if hotspots:
        write_json(paths["analysis"] / "nsys_hotspots.json", hotspots)
    (paths["analysis"] / "REPORT.md").write_text(
        markdown_report(paths["root"].name, hotspots, actions), encoding="utf-8"
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8")) if paths["manifest"].is_file() else {"schema_version": SCHEMA, "run_id": paths["root"].name, "events": []}
    manifest["raw_reports"] = raw_reports
    manifest["analysis"] = {
        "ncu_actions": len(actions),
        "nsys_stats": nsys_stats,
        "created_utc": now_utc(),
    }
    write_json(paths["manifest"], manifest)
    print(paths["analysis"] / "REPORT.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--output", type=pathlib.Path)
    doctor.set_defaults(handler=command_doctor)

    for name, handler in (("system", command_system), ("compute", command_compute)):
        current = subparsers.add_parser(name)
        current.add_argument("--binary", required=True, type=pathlib.Path)
        current.add_argument("--config", required=True, type=pathlib.Path)
        current.add_argument("--run-dir", required=True, type=pathlib.Path)
        current.add_argument("--dry-run", action="store_true")
        if name == "compute":
            current.add_argument("--case", action="append", dest="case_ids")
            current.add_argument("--set", dest="profile_set", choices=("basic", "detailed", "full"))
        current.set_defaults(handler=handler)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True, type=pathlib.Path)
    analyze.set_defaults(handler=command_analyze)
    return parser


def main() -> int:
    if os.name == "nt":
        updated, _ = runtime_environment(pathlib.Path(__file__).resolve().parents[1])
        os.environ.update(updated)
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
