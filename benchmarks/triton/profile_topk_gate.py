#!/usr/bin/env python3
"""Collect NSYS-first and NCU-basic Triton Top-K evidence in WSL2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import shlex
import shutil
import subprocess
from typing import Any


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command), flush=True)
    return subprocess.run(command, check=True, capture_output=capture, text=True, encoding="utf-8")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_csv(text: str) -> str:
    lines = text.replace("\r\n", "\n").splitlines()
    for index, line in enumerate(lines):
        if "," in line and any(word in line.casefold() for word in ("time", "name", "instances")):
            return "\n".join(lines[index:]).strip() + "\n"
    return text.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--python", default=str(pathlib.Path.home() / ".cache/raggedroute/topk-triton-venv/bin/python"))
    parser.add_argument("--ncu", default="/opt/nvidia/nsight-compute/2026.2.1/ncu")
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "raggedroute.triton_profile_suite.v1":
        raise ValueError("invalid Triton profile suite")
    if not pathlib.Path(args.python).is_file() or not pathlib.Path(args.ncu).is_file() or not shutil.which(args.nsys):
        raise RuntimeError("Triton Python, NCU, or NSYS is unavailable")
    reports = args.run_dir / "reports"
    analysis = args.run_dir / "analysis"
    reports.mkdir(parents=True)
    analysis.mkdir()
    events: list[dict[str, Any]] = []
    target = pathlib.Path(__file__).with_name("topk_gate.py")

    selected = [
        case for case in config["cases"]
        if not args.case_ids or case["id"] in set(args.case_ids)
    ]
    if not selected:
        raise ValueError("no Triton profile case matched --case")
    for case in selected:
        base_command = [
            args.python, str(target), "profile-once", "--tokens", str(case["T"]),
            "--experts", str(case["E"]), "--warmup", str(config["warmup"]),
            "--seed", str(config["seed"]), "--input-mode", config["input_mode"],
        ]
        nsys_base = reports / f"{case['id']}.nsys"
        nsys_command = [
            args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--force-overwrite=false", f"--output={nsys_base}", *base_command,
        ]
        run(nsys_command)
        nsys_report = pathlib.Path(f"{nsys_base}.nsys-rep")
        if not nsys_report.is_file():
            raise RuntimeError(f"NSYS report missing: {nsys_report}")
        events.append({"kind": "nsys", "case_id": case["id"], "command": nsys_command,
                       "report": str(nsys_report), "sha256": sha256(nsys_report)})
        for report_name in ("cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum"):
            result = run([args.nsys, "stats", "--force-export=true", "--report", report_name,
                          "--format", "csv", str(nsys_report)], capture=True)
            (analysis / f"{case['id']}.{report_name}.csv").write_text(
                extract_csv(result.stdout), encoding="utf-8"
            )

        ncu_base = reports / f"{case['id']}.basic"
        ncu_command = [
            args.ncu, "--target-processes", "all", "--kernel-name-base", "demangled",
            "--kernel-name", f"regex:.*{config['kernel_pattern']}.*", "--set", "basic",
            "--cache-control", "none", "--clock-control", "none", "--launch-skip", "21",
            "--launch-count", "1", "--export", str(ncu_base), *base_command,
        ]
        run(ncu_command)
        ncu_report = pathlib.Path(f"{ncu_base}.ncu-rep")
        if not ncu_report.is_file():
            raise RuntimeError(f"NCU report missing: {ncu_report}")
        events.append({"kind": "ncu", "case_id": case["id"], "set": "basic",
                       "command": ncu_command, "report": str(ncu_report), "sha256": sha256(ncu_report)})
        raw = run([args.ncu, "--import", str(ncu_report), "--csv", "--page", "raw"], capture=True)
        (analysis / f"{case['id']}.basic.raw.csv").write_text(raw.stdout, encoding="utf-8")

    environment = {
        "python": run([args.python, "--version"], capture=True).stdout.strip(),
        "ncu": run([args.ncu, "--version"], capture=True).stdout.strip(),
        "nsys": run([args.nsys, "--version"], capture=True).stdout.strip(),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
                            "--format=csv,noheader"], capture=True).stdout.strip(),
    }
    manifest = {
        "schema_version": "raggedroute.triton_profile_run.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config, "environment": environment, "events": events,
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}")
        raise SystemExit(2)
