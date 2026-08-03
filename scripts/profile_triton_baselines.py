#!/usr/bin/env python3
"""Collect L1/L2/L3 Triton NSYS and NCU evidence on native Windows.

Profiler durations are diagnostic only. Release performance comes from the
unprofiled cross-backend benchmark runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import profile_benchmarks as profile_common


SCHEMA = "raggedroute.triton_profile_suite.v1"
MANIFEST_SCHEMA = "raggedroute.triton_profile_manifest.v1"
OPERATORS = {
    "dense_gemm", "topk_gate", "histogram", "exclusive_scan",
    "token_permute", "grouped_gemm", "unpermute",
}


def load_config(path: pathlib.Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    operators = config.get("operators")
    if not isinstance(operators, list) or len(operators) != len(OPERATORS):
        raise ValueError("profile config requires all seven individual operators")
    names = [item.get("operator") for item in operators]
    if set(names) != OPERATORS or len(names) != len(set(names)):
        raise ValueError("profile config operator coverage is incomplete or duplicated")
    for item in operators:
        if set(item.get("kernels", {})) != {"l1", "l2"}:
            raise ValueError(f"{item.get('operator')}: requires l1/l2 kernel regex")
    l3 = config.get("l3", {})
    if l3.get("suite") != "chain_from_tokens":
        raise ValueError("L3 profile must use chain_from_tokens")
    if {item.get("operator") for item in l3.get("kernels", [])} != OPERATORS:
        raise ValueError("L3 profile requires one emitted kernel per operator")
    return config


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def param_args(params: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in sorted(params.items()):
        if isinstance(value, bool):
            value = str(value).lower()
        result.extend(("--param", f"{name}={value}"))
    return result


def target_command(
    python: pathlib.Path,
    worker: pathlib.Path,
    target_flag: str,
    target: str,
    level: str,
    warmup: int,
    seed: int,
    params: dict[str, Any],
) -> list[str]:
    return [
        str(python), str(worker), target_flag, target, "--level", level,
        "--protocol", "smoke", "--warmup", str(warmup),
        "--kernel-repeats", "1", "--samples", "1", "--profile-once",
        "--seed", str(seed), *param_args(params),
    ]


def system_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in config["operators"]:
        for level in ("l1", "l2"):
            result.append({
                "id": f"{level}.{item['operator']}", "level": level,
                "target_flag": "--operator", "target": item["operator"],
                "params": item.get("params", {}),
                "expected_kernels": [item["kernels"][level]],
            })
    l3 = config["l3"]
    result.append({
        "id": "l3.chain_from_tokens", "level": "l3", "target_flag": "--suite",
        "target": "chain_from_tokens", "params": l3.get("params", {}),
        "expected_kernels": [item["regex"] for item in l3["kernels"]],
    })
    return result


def compute_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in config["operators"]:
        for level in ("l1", "l2"):
            result.append({
                "id": f"{level}.{item['operator']}", "level": level,
                "operator": item["operator"], "target_flag": "--operator",
                "target": item["operator"], "params": item.get("params", {}),
                "kernel": item["kernels"][level],
            })
    l3 = config["l3"]
    for item in l3["kernels"]:
        result.append({
            "id": f"l3.{item['operator']}", "level": "l3",
            "operator": item["operator"], "target_flag": "--suite",
            "target": "chain_from_tokens", "params": l3.get("params", {}),
            "kernel": item["regex"],
        })
    return result


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return profile_common.run_command(command, capture=capture)


def manifest_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "manifest.json"


def load_manifest(run_dir: pathlib.Path, config: pathlib.Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config.resolve()),
        "environment": profile_common.doctor_data(),
        "events": [],
    }


def save_event(run_dir: pathlib.Path, config: pathlib.Path, event: dict[str, Any]) -> None:
    manifest = load_manifest(run_dir, config)
    manifest["events"].append(event)
    write_json(manifest_path(run_dir), manifest)


def parse_nsys_json(output: str) -> list[dict[str, Any]]:
    start = output.find("[")
    if start < 0:
        raise ValueError("NSYS stats did not emit JSON")
    value = json.loads(output[start:])
    if not isinstance(value, list):
        raise ValueError("NSYS kernel summary must be a list")
    return value


def command_system(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    nsys = profile_common.find_tool("nsys")
    if not nsys:
        raise RuntimeError("nsys was not found")
    worker = pathlib.Path(__file__).resolve().with_name("triton_benchmark.py")
    run_dir = args.run_dir.resolve()
    reports = run_dir / "reports" / "nsys"
    analysis = run_dir / "analysis" / "nsys"
    reports.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    for case in system_cases(config):
        base = reports / case["id"]
        report = pathlib.Path(f"{base}.nsys-rep")
        if report.exists():
            raise FileExistsError(report)
        target = target_command(
            args.python.resolve(), worker, case["target_flag"], case["target"],
            case["level"], int(config.get("warmup", 20)),
            int(config.get("seed", 20260729)), case["params"],
        )
        command = [
            nsys, "profile", "--trace=cuda,nvtx", "--sample=none",
            "--cpuctxsw=none", "--force-overwrite=false", f"--output={base}",
            *target,
        ]
        if args.dry_run:
            print("+", shlex.join(command))
            continue
        run(command)
        stats_command = [
            nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "json",
            "--output", "-", str(report),
        ]
        completed = run(stats_command, capture=True)
        kernels = parse_nsys_json(completed.stdout + completed.stderr)
        names = [str(item.get("Name", "")) for item in kernels]
        missing = [regex for regex in case["expected_kernels"] if not any(regex in name for name in names)]
        if missing:
            raise ValueError(f"{case['id']}: NSYS missing emitted kernels {missing}")
        summary_path = analysis / f"{case['id']}.json"
        write_json(summary_path, kernels)
        save_event(run_dir, config_path, {
            "kind": "system", "case_id": case["id"], "level": case["level"],
            "command": command, "stats_command": stats_command,
            "report": str(report), "report_bytes": report.stat().st_size,
            "report_sha256": sha256(report), "summary": str(summary_path),
        })
    return 0


def command_compute(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    ncu = profile_common.find_tool("ncu")
    if not ncu:
        raise RuntimeError("ncu was not found")
    worker = pathlib.Path(__file__).resolve().with_name("triton_benchmark.py")
    run_dir = args.run_dir.resolve()
    reports = run_dir / "reports" / "ncu"
    reports.mkdir(parents=True, exist_ok=True)
    warmup = int(config.get("warmup", 20))
    for case in compute_cases(config):
        base = reports / f"{case['id']}.basic"
        report = pathlib.Path(f"{base}.ncu-rep")
        if report.exists():
            raise FileExistsError(report)
        target = target_command(
            args.python.resolve(), worker, case["target_flag"], case["target"],
            case["level"], warmup, int(config.get("seed", 20260729)), case["params"],
        )
        command = [
            ncu, "--set", "basic", "--clock-control", "none",
            "--target-processes", "all", "--kernel-name", f"regex:{case['kernel']}",
            "--launch-skip", str(warmup + 1), "--launch-count", "1",
            "--export", str(base), *target,
        ]
        if args.dry_run:
            print("+", shlex.join(command))
            continue
        run(command)
        if not report.is_file():
            raise RuntimeError(f"NCU did not create {report}")
        save_event(run_dir, config_path, {
            "kind": "compute", "case_id": case["id"], "level": case["level"],
            "operator": case["operator"], "kernel_regex": case["kernel"],
            "command": command, "report": str(report),
            "report_bytes": report.stat().st_size, "report_sha256": sha256(report),
        })
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = json.loads(manifest_path(run_dir).read_text(encoding="utf-8"))
    report_path = profile_common.find_ncu_report_path()
    ncu = profile_common.find_tool("ncu")
    if report_path is None or not ncu:
        raise RuntimeError("ncu or ncu_report.py was not found")
    sys.path.insert(0, str(report_path))
    ncu_report = importlib.import_module("ncu_report")
    supported = profile_common.query_supported_metrics(ncu)
    ncu_actions = []
    for event in manifest["events"]:
        if event["kind"] != "compute":
            continue
        context = ncu_report.load_report(event["report"])
        actions = list(profile_common.iter_actions(context))
        if len(actions) != 1:
            raise ValueError(f"{event['case_id']}: expected one NCU action")
        action = actions[0]
        kernel = profile_common.action_name(action)
        if event["kernel_regex"] not in kernel:
            raise ValueError(f"{event['case_id']}: NCU kernel mismatch")
        ncu_actions.append({
            "case_id": event["case_id"], "level": event["level"],
            "operator": event["operator"], "kernel": kernel,
            "metrics": profile_common.metric_snapshot(action, supported),
        })
    nsys = []
    for path in sorted((run_dir / "analysis" / "nsys").glob("*.json")):
        kernels = json.loads(path.read_text(encoding="utf-8"))
        nsys.append({"case_id": path.stem, "kernels": kernels})
    expected_nsys = 15
    expected_ncu = 21
    if len(nsys) != expected_nsys or len(ncu_actions) != expected_ncu:
        raise ValueError(
            f"incomplete profile evidence: nsys={len(nsys)}/{expected_nsys}, "
            f"ncu={len(ncu_actions)}/{expected_ncu}"
        )
    summary = {
        "schema_version": "raggedroute.triton_profile_summary.v1",
        "promotion_eligible": False,
        "nsys_cases": nsys,
        "ncu_actions": ncu_actions,
        "raw_reports": [
            {key: event[key] for key in ("kind", "case_id", "report", "report_bytes", "report_sha256")}
            for event in manifest["events"]
        ],
    }
    analysis = run_dir / "analysis"
    write_json(analysis / "profile_summary.json", summary)
    lines = [
        "# Triton L1/L2/L3 profiler summary", "",
        "Profiler durations are diagnostic only; unprofiled release JSONL is the performance source.", "",
        f"- NSYS cases: {len(nsys)} (7 L1 + 7 L2 + 1 L3 chain)",
        f"- NCU basic actions: {len(ncu_actions)} (7 L1 + 7 L2 + 7 L3)",
        "- Promotion eligible: false", "", "## NCU kernels", "",
        "| Level | Operator | Emitted kernel |", "|---|---|---|",
    ]
    for item in ncu_actions:
        lines.append(f"| {item['level'].upper()} | {item['operator']} | `{item['kernel']}` |")
    lines.append("")
    (analysis / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    manifest["analysis"] = {
        "profile_summary": str(analysis / "profile_summary.json"),
        "report": str(analysis / "REPORT.md"),
        "nsys_cases": len(nsys), "ncu_actions": len(ncu_actions),
    }
    write_json(manifest_path(run_dir), manifest)
    print(analysis / "REPORT.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("system", command_system), ("compute", command_compute),
        ("analyze", command_analyze),
    ):
        current = subparsers.add_parser(name)
        current.add_argument("--config", required=True, type=pathlib.Path)
        current.add_argument("--run-dir", required=True, type=pathlib.Path)
        if name != "analyze":
            current.add_argument("--python", required=True, type=pathlib.Path)
            current.add_argument("--dry-run", action="store_true")
        current.set_defaults(handler=handler)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
