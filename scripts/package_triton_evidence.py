#!/usr/bin/env python3
"""Freeze consistent L1/L2/L3 Triton benchmark and profiler evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
from typing import Any


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path: pathlib.Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or any(
        item.get("schema_version") != "raggedroute.benchmark.v1"
        or not item.get("validation", {}).get("ok")
        for item in values
    ):
        raise ValueError(f"{path}: invalid or failed benchmark records")
    return values


def validate_benchmark(
    raw: pathlib.Path,
    manifest_path: pathlib.Path,
    comparison_path: pathlib.Path,
    expected_levels: set[str],
    expected_cases: int,
) -> dict[str, Any]:
    values = records(raw)
    manifest = read_json(manifest_path)
    comparison = read_json(comparison_path)
    if manifest.get("schema_version") != "raggedroute.cross_backend_manifest.v1":
        raise ValueError(f"{manifest_path}: wrong manifest schema")
    if comparison.get("schema_version") != "raggedroute.cross_backend_comparison.v1":
        raise ValueError(f"{comparison_path}: wrong comparison schema")
    if comparison.get("promotion_eligible") is not False:
        raise ValueError("cross-backend comparison must remain promotion-ineligible")
    levels = {item["measurement_level"] for item in values}
    if levels != expected_levels:
        raise ValueError(f"unexpected measurement levels: {levels}")
    if len(comparison.get("comparisons", [])) != expected_cases:
        raise ValueError("comparison count does not match expected logical cases")
    keys: dict[tuple[str, str, str], set[int]] = {}
    for item in values:
        key = (item["case_id"], item["variant"], item["measurement_level"])
        keys.setdefault(key, set()).add(int(item["process_run"]))
    if any(runs != {1, 2, 3} for runs in keys.values()):
        raise ValueError("release evidence requires process runs {1,2,3} for every group")
    git_shas = {item["environment"]["build_git_sha"] for item in values}
    if len(git_shas) != 1 or any(item["environment"].get("build_git_dirty") != "false" for item in values):
        raise ValueError("release evidence must come from one clean Git build")
    return {
        "records": len(values), "groups": len(keys), "git_sha": next(iter(git_shas)),
        "comparisons": comparison["comparisons"],
    }


def copy_file(source: pathlib.Path, destination: pathlib.Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--l1-l2-raw", required=True, type=pathlib.Path)
    parser.add_argument("--l1-l2-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--l1-l2-comparison", required=True, type=pathlib.Path)
    parser.add_argument("--l3-raw", required=True, type=pathlib.Path)
    parser.add_argument("--l3-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--l3-comparison", required=True, type=pathlib.Path)
    parser.add_argument("--profile-dir", required=True, type=pathlib.Path)
    parser.add_argument("--validation", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    args = parser.parse_args()

    l1_l2 = validate_benchmark(
        args.l1_l2_raw, args.l1_l2_manifest, args.l1_l2_comparison,
        {"L1_kernel_body", "L2_operator_steady"}, 14,
    )
    l3 = validate_benchmark(
        args.l3_raw, args.l3_manifest, args.l3_comparison,
        {"L3_chain_steady"}, 1,
    )
    if l1_l2["git_sha"] != l3["git_sha"]:
        raise ValueError("L1/L2 and L3 releases use different Git SHAs")
    profile_summary_path = args.profile_dir / "analysis" / "profile_summary.json"
    profile_report_path = args.profile_dir / "analysis" / "REPORT.md"
    profile_manifest_path = args.profile_dir / "manifest.json"
    profile = read_json(profile_summary_path)
    if profile.get("schema_version") != "raggedroute.triton_profile_summary.v1":
        raise ValueError("profile summary schema mismatch")
    if profile.get("promotion_eligible") is not False:
        raise ValueError("profile evidence must remain promotion-ineligible")
    if len(profile.get("nsys_cases", [])) != 15 or len(profile.get("ncu_actions", [])) != 21:
        raise ValueError("profile evidence must contain 15 NSYS cases and 21 NCU actions")
    validation = read_json(args.validation)
    if validation.get("schema_version") != "raggedroute.validation_summary.v1":
        raise ValueError("validation summary schema mismatch")
    if validation.get("git_sha") != l1_l2["git_sha"]:
        raise ValueError("validation and release Git SHAs differ")
    if any(item.get("status") != "passed" for item in validation.get("checks", [])):
        raise ValueError("validation summary contains a failed check")

    destination = args.output_root.resolve() / args.run_id
    if destination.exists():
        raise FileExistsError(destination)
    files = {
        "benchmark/l1_l2_release.jsonl": args.l1_l2_raw,
        "benchmark/l1_l2_manifest.json": args.l1_l2_manifest,
        "benchmark/l1_l2_comparison.json": args.l1_l2_comparison,
        "benchmark/l3_release.jsonl": args.l3_raw,
        "benchmark/l3_manifest.json": args.l3_manifest,
        "benchmark/l3_comparison.json": args.l3_comparison,
        "profile/profile_summary.json": profile_summary_path,
        "profile/REPORT.md": profile_report_path,
        "profile/manifest.json": profile_manifest_path,
        "validation/summary.json": args.validation,
    }
    repo = pathlib.Path(__file__).resolve().parents[1]
    config_sources = {
        "benchmark_rtx3080_cross_backend_release.json":
            repo / "configs" / "cross_backend" / "benchmark" / "l1_l2_release.json",
        "benchmark_rtx3080_cross_backend_l3_release.json":
            repo / "configs" / "cross_backend" / "benchmark" / "l3_release.json",
        "profile_triton_three_levels.json":
            repo / "configs" / "cross_backend" / "profile" / "triton_three_levels.json",
    }
    for name, source in config_sources.items():
        files[f"configs/{name}"] = source
    for relative, source in files.items():
        copy_file(source.resolve(), destination / relative)

    comparisons = [*l1_l2["comparisons"], *l3["comparisons"]]
    lines = [
        "# Triton L1/L2/L3 reference evidence", "",
        f"- Git SHA: `{l1_l2['git_sha']}` (clean Release build)",
        "- GPU: NVIDIA GeForce RTX 3080 / SM86",
        "- Promotion eligible: **false**",
        "- Profiler durations are diagnostic only; the table uses unprofiled release CUDA Events.",
        "", "## Cross-backend release", "",
        "| Level | Target | Triton p50 (us) | CUDA p50 (us) | CUDA vs Triton | Triton p95 (us) | CUDA p95 (us) | Triton CV | CUDA CV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['measurement_level']} | {item['operator']} | "
            f"{item['reference_latency_us']:.4f} | {item['candidate_latency_us']:.4f} | "
            f"{item['candidate_vs_reference_speedup']:.4f}x | "
            f"{item['reference_p95_us']:.4f} | {item['candidate_p95_us']:.4f} | "
            f"{item['reference_cv']:.4f} | {item['candidate_cv']:.4f} |"
        )
    lines.extend([
        "", "## Profiler coverage", "",
        "- NSYS: 7 L1 + 7 L2 + one complete seven-operator L3 chain.",
        "- NCU basic: 7 L1 + 7 L2 + 7 emitted L3 operator kernels.",
        "- Raw `.nsys-rep`/`.ncu-rep` files remain under the reproducible local `out/profile` run; their sizes and SHA256 values are recorded in `profile/manifest.json`.",
        "- SQLite exports and raw Nsight binaries are intentionally not committed.",
        "", "## L3 NSYS kernel-time breakdown", "",
        "| Time (%) | Total time (ns) | Instances | Average (ns) | Kernel |",
        "|---:|---:|---:|---:|---|",
    ])
    l3_nsys = next(item for item in profile["nsys_cases"] if item["case_id"] == "l3.chain_from_tokens")
    for item in l3_nsys["kernels"]:
        lines.append(
            f"| {float(item.get('Time (%)', 0.0)):.2f} | {int(item.get('Total Time (ns)', 0))} | "
            f"{int(item.get('Instances', 0))} | {float(item.get('Avg (ns)', 0.0)):.2f} | "
            f"`{item.get('Name', 'unknown')}` |"
        )
    lines.extend([
        "", "## NCU basic headline metrics", "",
        "| Level | Operator | Kernel | Duration (ns) | SM (%) | Memory (%) | DRAM (%) | Achieved occupancy (%) | Registers/thread | Grid | Block |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    def metric(action: dict[str, Any], name: str) -> str:
        value = action["metrics"].get(name, {})
        if value.get("status") != "collected":
            return value.get("status", "not_collected")
        raw = value.get("value")
        return f"{raw:.2f}" if isinstance(raw, float) else str(raw)

    for action in profile["ncu_actions"]:
        lines.append(
            f"| {action['level'].upper()} | {action['operator']} | `{action['kernel']}` | "
            f"{metric(action, 'duration_ns')} | {metric(action, 'sm_throughput_pct')} | "
            f"{metric(action, 'memory_throughput_pct')} | {metric(action, 'dram_throughput_pct')} | "
            f"{metric(action, 'occupancy_achieved_pct')} | {metric(action, 'registers_per_thread')} | "
            f"{metric(action, 'grid_size')} | {metric(action, 'block_size')} |"
        )
    lines.extend([
        "", "## Validation", "",
        "| Check | Status | Details |", "|---|---|---|",
    ])
    for item in validation["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['details']} |")
    lines.extend([
        "", "## Evidence boundary", "",
        "- Cross-runtime ratios are reference comparisons only and cannot promote a CUDA candidate.",
        "- NCU duration is replay-affected diagnostic context and is not a release score.",
        "- Metrics absent from NCU basic remain `not_collected`; they are never encoded as zero.",
        "", "See `profile/profile_summary.json`, `validation/summary.json`, and `manifest.json` for machine-readable evidence.", "",
    ])
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    artifact_files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        artifact_files.append({
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    evidence_manifest = {
        "schema_version": "raggedroute.triton_evidence_bundle.v1",
        "run_id": args.run_id,
        "git_sha": l1_l2["git_sha"],
        "promotion_eligible": False,
        "benchmark": {
            "l1_l2_records": l1_l2["records"], "l1_l2_groups": l1_l2["groups"],
            "l3_records": l3["records"], "l3_groups": l3["groups"],
        },
        "profile": {"nsys_cases": 15, "ncu_actions": 21},
        "files": artifact_files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(evidence_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"error: {error}")
