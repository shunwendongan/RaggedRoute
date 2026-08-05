#!/usr/bin/env python3
"""Run the in-tree CUDA candidate chain in isolated native processes."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--kernel-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--expected-sha", default="af1b71be162902330623783c571f957a66eda776")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    binary = args.binary.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "raggedroute.cuda_chain_release_manifest.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "binary": str(binary),
        "output": str(output),
        "contract": {"warmup": args.warmup, "samples": args.samples, "kernel_repeats": args.kernel_repeats, "process_runs": 3, "seed": args.seed, "cache_mode": "warm"},
        "commands": [],
    }
    for process_run in range(1, 4):
        for case_id, params in CASES:
            command = [
                str(binary), "--suite", "chain_from_tokens", "--variant", "cuda_all_candidates_chain",
                "--level", "l3", "--protocol", "release", "--cache-mode", "warm",
                "--warmup", str(args.warmup), "--kernel-repeats", str(args.kernel_repeats),
                "--samples", str(args.samples), "--process-run", str(process_run), "--seed", str(args.seed),
                "--run-id", args.run_id, "--case-id", case_id, "--output", str(output),
                "--expected-git-sha", args.expected_sha,
            ]
            for name, value in sorted(params.items()):
                command.extend(["--param", f"{name}={value}"])
            manifest["commands"].append(command)
            print("+", shlex.join(command), flush=True)
            subprocess.run(command, cwd=repo, check=True)
    manifest_path = pathlib.Path(str(output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
