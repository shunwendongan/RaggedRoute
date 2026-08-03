#!/usr/bin/env python3
"""Freeze selected textual Top-K evidence while inventorying raw artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_text(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def copy_compact_json(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    target.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def inventory(paths: list[pathlib.Path], committed: bool = False) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve()),
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "committed": committed,
        }
        for path in sorted(paths)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-dir", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark-report-dir", required=True, type=pathlib.Path)
    parser.add_argument("--ncu-dir", required=True, type=pathlib.Path)
    parser.add_argument("--nsys-dir", required=True, type=pathlib.Path)
    parser.add_argument("--sanitizer-dir", required=True, type=pathlib.Path)
    parser.add_argument("--doctor", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    args = parser.parse_args()

    destination = args.output_root / args.run_id
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    copied: list[pathlib.Path] = []

    benchmark_names = (
        "topk-gate-release.aggregate.json",
        "topk-gate-release.aggregate.csv",
        "topk-gate-release.comparison.json",
        "topk-gate-preflight.aggregate.csv",
        "topk-gate-preflight.comparison.json",
    )
    for name in benchmark_names:
        source = args.benchmark_dir / name
        target = destination / "benchmark" / name
        copied.append(
            copy_compact_json(source, target)
            if source.suffix == ".json"
            else copy_text(source, target)
        )
    for source in sorted(args.benchmark_report_dir.glob("*")):
        if source.is_file() and source.suffix.casefold() in {".json", ".csv", ".md"}:
            copied.append(copy_text(source, destination / "benchmark" / source.name))

    for source in sorted((args.ncu_dir / "analysis").glob("*")):
        if source.is_file() and source.suffix.casefold() in {".json", ".csv", ".md"}:
            copied.append(copy_text(source, destination / "profile" / "ncu" / source.name))
    if (args.ncu_dir / "manifest.json").is_file():
        copied.append(
            copy_text(
                args.ncu_dir / "manifest.json", destination / "profile" / "ncu" / "manifest.json"
            )
        )
    for source in sorted((args.nsys_dir / "analysis").glob("*.csv")):
        copied.append(copy_text(source, destination / "profile" / "nsys" / source.name))
    copied.append(copy_text(args.doctor, destination / "environment.json"))
    for source in sorted(args.sanitizer_dir.glob("*.log")):
        copied.append(copy_text(source, destination / "sanitizer" / source.name))

    raw_benchmark = inventory(
        [
            args.benchmark_dir / "topk-gate-release.jsonl",
            args.benchmark_dir / "topk-gate-release.jsonl.manifest.json",
            args.benchmark_dir / "topk-gate-preflight.jsonl",
            args.benchmark_dir / "topk-gate-preflight.jsonl.manifest.json",
        ]
    )
    raw_profiler = inventory(
        list((args.ncu_dir / "reports").glob("*.ncu-rep"))
        + list((args.nsys_dir / "reports").glob("*.nsys-rep"))
    )
    manifest = {
        "schema_version": "raggedroute.topk_evidence_bundle.v1",
        "run_id": args.run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": {
            "profiler_duration_is_release_latency": False,
            "raw_benchmark_jsonl_committed": False,
            "raw_profiler_binaries_committed": False,
        },
        "raw_benchmark_inventory": raw_benchmark,
        "raw_profiler_inventory": raw_profiler,
        "command_contract": {
            "correctness": "ctest --preset test-rtx3080-sm86-release --output-on-failure",
            "sanitizers": "python scripts/run_sanitizers.py --build-dir out/build/rtx3080-sm86-release --output-dir out/sanitizer/rtx3080-topk-gate",
            "release": "python scripts/run_benchmarks.py --binary out/build/rtx3080-sm86-release/raggedroute_benchmark.exe --config configs/operators/topk_gate/benchmark/release.json --output out/benchmark/topk-gate-release.jsonl",
            "nsys": "nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none ... --profile-once",
            "ncu": "python scripts/profile_benchmarks.py compute --config configs/operators/topk_gate/profile/optimization.json --run-dir out/profile/topk-gate-ncu",
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    copied.append(manifest_path)

    checksum_lines = []
    for path in sorted(copied):
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(destination).as_posix()}")
    (destination / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
