#!/usr/bin/env python3
"""Freeze textual benchmark/profile evidence into a non-overwriting bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
from typing import Any


TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log"}
RAW_PROFILE_SUFFIXES = {".ncu-rep", ".nsys-rep", ".sqlite"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_text_tree(source: pathlib.Path, destination: pathlib.Path) -> list[pathlib.Path]:
    copied: list[pathlib.Path] = []
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def raw_profile_inventory(profile_dir: pathlib.Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted((profile_dir / "reports").glob("*")):
        if path.is_file() and path.suffix.casefold() in RAW_PROFILE_SUFFIXES:
            inventory.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "committed": False,
                }
            )
    return inventory


def freeze_bundle(
    run_id: str,
    profile_dir: pathlib.Path,
    benchmark_dir: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path:
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(destination)
    if not (profile_dir / "analysis" / "ncu_metrics.json").is_file():
        raise ValueError("profile analysis is incomplete")
    if not list(benchmark_dir.glob("*.jsonl")):
        raise ValueError("benchmark directory contains no raw JSONL")

    destination.mkdir(parents=True)
    copied = copy_text_tree(benchmark_dir, destination / "benchmark")
    copied.extend(copy_text_tree(profile_dir / "analysis", destination / "profile"))
    for name in ("manifest.json", "environment.json"):
        source = profile_dir / name
        if source.is_file():
            target = destination / "profile" / name
            shutil.copy2(source, target)
            copied.append(target)

    bundle_manifest = {
        "schema_version": "raggedroute.evidence_bundle.v1",
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_source": str(benchmark_dir.resolve()),
        "profile_source": str(profile_dir.resolve()),
        "raw_profiler_reports": raw_profile_inventory(profile_dir),
        "policy": {
            "profiler_duration_is_release_latency": False,
            "raw_profiler_binaries_committed": False,
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    copied.append(manifest_path)

    checksum_lines = []
    for path in sorted(set(copied)):
        relative = path.relative_to(destination).as_posix()
        checksum_lines.append(f"{sha256_file(path)}  {relative}")
    (destination / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-dir", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    destination = freeze_bundle(
        args.run_id,
        args.profile_dir.resolve(),
        args.benchmark_dir.resolve(),
        args.output_root.resolve(),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
