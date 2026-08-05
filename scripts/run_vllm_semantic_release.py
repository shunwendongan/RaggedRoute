#!/usr/bin/env python3
"""Run the pinned vLLM-semantic chain in independent Docker processes.

The worker is intentionally invoked once per case/process. No CUDA context is
shared between records, and the manifest retains the exact command and GPU
snapshot for every invocation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import subprocess
from typing import Any


CASES = [
    ("moe.mixtral_decode_exact.t8_e8_k4096_n14336", {"T": 8, "E": 8, "K": 4096, "N": 14336}),
    ("moe.continuous_decode_proxy.t128_e8_k1024_n3584", {"T": 128, "E": 8, "K": 1024, "N": 3584}),
    ("moe.chunked_prefill_proxy.t512_e8_k512_n1792", {"T": 512, "E": 8, "K": 512, "N": 1792}),
    ("moe.finegrained_prefill_proxy.t1024_e64_k256_n512", {"T": 1024, "E": 64, "K": 256, "N": 512}),
]


def output(command: list[str], cwd: pathlib.Path) -> str:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--docker", type=pathlib.Path, default=pathlib.Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe"))
    parser.add_argument("--image", default="vllm/vllm-openai:latest")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--git-dirty", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--kernel-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--start-process", type=int, default=1, choices=(1, 2, 3))
    args = parser.parse_args()
    repo = args.repo.resolve()
    docker = args.docker.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and args.start_process == 1:
        raise FileExistsError(f"refusing to append to existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.parent / "vllm-triton-cache"
    cache.mkdir(parents=True, exist_ok=True)
    rel_output = output_path.relative_to(repo).as_posix()
    container_output = f"/workspace/{rel_output}"
    container_cache = f"/workspace/{cache.relative_to(repo).as_posix()}"
    gpu_query = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"]
    manifest: dict[str, Any] = {
        "schema_version": "raggedroute.vllm_semantic_release_manifest.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "repo": str(repo),
        "git_sha": args.git_sha,
        "git_dirty": args.git_dirty,
        "upstream_commit": "66b3c0e61f1e477820212201adf1ed871df7ee98",
        "image": args.image,
        "output": str(output_path),
        "contract": {"warmup": args.warmup, "samples": args.samples, "kernel_repeats": args.kernel_repeats, "process_runs": 3, "seed": args.seed, "cache_mode": "warm"},
        "gpu_snapshot_start": output(gpu_query, repo),
        "commands": [],
        "results": [],
    }
    dirty_value = "true" if args.git_dirty else "false"
    existing_records = []
    if output_path.exists():
        existing_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest["results"].extend({"process_run": row.get("process_run"), "case_id": row.get("case_id"), "variant": row.get("variant")} for row in existing_records)
    for process_run in range(1, 4):
        for case_id, params in CASES:
            command = [
                str(docker), "run", "--rm", "--gpus", "all", "--ipc=host",
                "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                "-e", "TRITON_F32_DEFAULT=ieee",
                "-e", f"RAGGEDROUTE_BUILD_GIT_SHA={args.git_sha[:12]}",
                "-e", f"RAGGEDROUTE_BUILD_GIT_DIRTY={dirty_value}",
                "-e", f"VLLM_UPSTREAM_COMMIT=66b3c0e61f1e477820212201adf1ed871df7ee98",
                "-e", f"TRITON_CACHE_DIR={container_cache}",
                "-v", f"{repo}:/workspace", "-w", "/workspace",
                "--entrypoint", "/usr/bin/python3", args.image,
                "scripts/vllm_semantic_benchmark.py",
                "--protocol", "release", "--cache-mode", "warm",
                "--warmup", str(args.warmup), "--samples", str(args.samples),
                "--kernel-repeats", str(args.kernel_repeats), "--process-run", str(process_run),
                "--seed", str(args.seed), "--run-id", args.run_id, "--case-id", case_id,
                "--output", container_output,
            ]
            for name, value in sorted(params.items()):
                command.extend(["--param", f"{name}={value}"])
            manifest["commands"].append(command)
            if process_run < args.start_process:
                continue
            print("+", shlex.join(command), flush=True)
            subprocess.run(command, cwd=repo, check=True)
            manifest["results"].append({"process_run": process_run, "case_id": case_id, "variant": "vllm_semantic_chain"})
    manifest["gpu_snapshot_end"] = output(gpu_query, repo)
    manifest_path = pathlib.Path(str(output_path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
