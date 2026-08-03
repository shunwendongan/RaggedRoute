#!/usr/bin/env python3
"""Fail-closed merge of independently-run benchmark process shards."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def records(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--manifest-output", required=True, type=pathlib.Path)
    parser.add_argument("--raw", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--manifests", type=pathlib.Path, nargs="+", required=True)
    args = parser.parse_args()
    if len(args.raw) != len(args.manifests):
        raise ValueError("--raw and --manifests must have the same number of paths")
    if args.output.exists() or args.manifest_output.exists():
        raise FileExistsError("refusing to overwrite merged benchmark evidence")

    canonical: dict[str, Any] | None = None
    expected_runs: set[int] = set()
    merged: list[dict[str, Any]] = []
    for raw_path, manifest_path in zip(args.raw, args.manifests):
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != "raggedroute.run_manifest.v2":
            raise ValueError(f"{manifest_path}: expected raggedroute.run_manifest.v2")
        current = {
            "run_id": manifest.get("run_id"),
            "suite": manifest.get("suite"),
            "repo_commit": manifest.get("repo_commit"),
            "repo_branch": manifest.get("repo_branch"),
            "binary": manifest.get("binary"),
        }
        if canonical is None:
            canonical = current
        elif current != canonical:
            raise ValueError(f"{manifest_path}: provenance or suite differs from the first shard")
        shard_rows = records(raw_path)
        if not shard_rows:
            raise ValueError(f"{raw_path}: contains no records")
        shard_runs = {int(row["process_run"]) for row in shard_rows}
        declared = set(manifest.get("executed_process_runs", []))
        if len(shard_runs) != 1 or shard_runs != declared:
            raise ValueError(f"{raw_path}: records do not match manifest executed_process_runs")
        if expected_runs & shard_runs:
            raise ValueError(f"{raw_path}: duplicate process run {sorted(shard_runs)}")
        expected_runs |= shard_runs
        if any(row.get("run_id") != canonical["run_id"] for row in shard_rows):
            raise ValueError(f"{raw_path}: record run_id differs from the manifest")
        merged.extend(shard_rows)

    assert canonical is not None
    required = set(range(1, int(canonical["suite"].get("process_runs", 1)) + 1))
    if expected_runs != required:
        raise ValueError(f"missing process runs: expected {sorted(required)}, got {sorted(expected_runs)}")
    merged.sort(key=lambda row: (row["process_run"], row["case_id"], row["variant"], row["measurement_level"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
                           encoding="utf-8", newline="\n")
    merged_manifest = read_json(args.manifests[0])
    merged_manifest["merged_from"] = [str(path.resolve()) for path in args.raw]
    merged_manifest["merged_manifest_inputs"] = [str(path.resolve()) for path in args.manifests]
    merged_manifest["executed_process_runs"] = sorted(expected_runs)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.manifest_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}")
        raise SystemExit(2)
