#!/usr/bin/env python3
"""Run one WSL2 Triton release shard under a recorded temporary GPU clock lock."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import shutil
import subprocess
from typing import Any


def run(command: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, capture_output=capture, text=True, encoding="utf-8")


def snapshot(tool: str) -> str:
    return run([tool, "--query-gpu=name,uuid,pstate,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits"], capture=True).stdout.strip()


def wsl_path(path: pathlib.Path, distro: str) -> str:
    return run(["wsl", "-d", distro, "--", "wslpath", "-a", str(path.resolve())], capture=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--process-run", required=True, type=int)
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--venv", default="$HOME/.cache/raggedroute/topk-triton-venv")
    parser.add_argument("--graphics-clock-mhz", default=1710, type=int)
    args = parser.parse_args()
    nvidia_smi = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi not found")
    controlled = args.output.with_suffix(args.output.suffix + ".controlled.json")
    if controlled.exists():
        raise FileExistsError(controlled)
    repo = pathlib.Path(__file__).resolve().parents[1]
    venv_python = f"{args.venv}/bin/python"
    if venv_python.startswith("$HOME/"):
        quoted_python = '"$HOME/' + venv_python[len("$HOME/"):] + '"'
    else:
        quoted_python = shlex.quote(venv_python)
    command = [
        "benchmarks/triton/topk_gate.py", "benchmark",
        "--config", wsl_path(args.config, args.distro), "--output", wsl_path(args.output, args.distro),
        "--run-id", args.run_id, "--process-run", str(args.process_run),
    ]
    shell = (
        f"cd {shlex.quote(wsl_path(repo, args.distro))} && {quoted_python} "
        + " ".join(shlex.quote(x) for x in command)
    )
    manifest: dict[str, Any] = {
        "schema_version": "raggedroute.controlled_triton_wsl.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pre_lock": snapshot(nvidia_smi),
        "lock_command": [nvidia_smi, "-lgc", f"{args.graphics_clock_mhz},{args.graphics_clock_mhz}"],
        "restore_command": [nvidia_smi, "-rgc"],
        "wsl_command": ["wsl", "-d", args.distro, "--", "bash", "-lc", shell],
        "locked": False,
        "restored": False,
    }
    failure: BaseException | None = None
    try:
        run(manifest["lock_command"])
        manifest["locked"] = True
        manifest["post_lock"] = snapshot(nvidia_smi)
        run(manifest["wsl_command"])
    except BaseException as error:
        failure = error
        manifest["failure"] = repr(error)
    finally:
        if manifest["locked"]:
            restore = run(manifest["restore_command"], capture=True, check=False)
            manifest["restore_returncode"] = restore.returncode
            manifest["restored"] = restore.returncode == 0
        manifest["post_restore"] = snapshot(nvidia_smi)
        controlled.parent.mkdir(parents=True, exist_ok=True)
        controlled.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failure:
        raise failure
    if not manifest["restored"]:
        raise RuntimeError("GPU clock restore failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
