#!/usr/bin/env python3
"""Run a Top-K release suite with a temporary, recorded graphics-clock lock."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any


def run(command: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, capture_output=capture, text=True, encoding="utf-8")


def snapshot(nvidia_smi: str) -> dict[str, Any]:
    command = [
        nvidia_smi,
        "--query-gpu=name,uuid,pstate,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory",
        "--format=csv,noheader,nounits",
    ]
    result = run(command, capture=True)
    return {"command": command, "csv": result.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--graphics-clock-mhz", type=int, default=1710)
    parser.add_argument("--run-id")
    parser.add_argument("--process-run", type=int)
    args = parser.parse_args()

    nvidia_smi = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi was not found")
    repo = pathlib.Path(__file__).resolve().parents[1]
    wrapper_manifest = args.output.with_suffix(args.output.suffix + ".controlled.json")
    if wrapper_manifest.exists():
        raise FileExistsError(wrapper_manifest)

    manifest: dict[str, Any] = {
        "schema_version": "raggedroute.controlled_topk_release.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "graphics_clock_mhz": args.graphics_clock_mhz,
        "pre_lock": snapshot(nvidia_smi),
        "lock_command": [nvidia_smi, "-lgc", f"{args.graphics_clock_mhz},{args.graphics_clock_mhz}"],
        "restore_command": [nvidia_smi, "-rgc"],
        "release_command": [
            sys.executable,
            "scripts/run_benchmarks.py",
            "--binary",
            str(args.binary),
            "--config",
            str(args.config),
            "--output",
            str(args.output),
        ],
        "locked": False,
        "restored": False,
    }
    if args.run_id:
        manifest["release_command"].extend(["--run-id", args.run_id])
    if args.process_run is not None:
        manifest["release_command"].extend(["--process-run", str(args.process_run)])
    failure: BaseException | None = None
    try:
        run(manifest["lock_command"])
        manifest["locked"] = True
        manifest["post_lock"] = snapshot(nvidia_smi)
        run(manifest["release_command"])
    except BaseException as error:
        failure = error
        manifest["failure"] = repr(error)
    finally:
        if manifest["locked"]:
            restore = run(manifest["restore_command"], check=False, capture=True)
            manifest["restore_returncode"] = restore.returncode
            manifest["restore_stdout"] = restore.stdout.strip()
            manifest["restore_stderr"] = restore.stderr.strip()
            manifest["restored"] = restore.returncode == 0
        try:
            manifest["post_restore"] = snapshot(nvidia_smi)
        except BaseException as error:
            manifest["post_restore_error"] = repr(error)
        wrapper_manifest.parent.mkdir(parents=True, exist_ok=True)
        wrapper_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure
    if not manifest["restored"]:
        raise RuntimeError("graphics-clock reset failed; inspect controlled manifest")
    print(f"wrote {wrapper_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
