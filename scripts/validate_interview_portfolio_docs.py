#!/usr/bin/env python3
"""Validate the canonical interview-portfolio evidence and documentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib


EVIDENCE_SHA = "9732a0343c60f869fc4166a0cc3cabba2fd67bbb"
BUNDLE = pathlib.Path("docs/reports/compact/20260829-9732a03-interview-portfolio")
PRIMARY_EXPECTED = {
    "dense_v3_vs_library_envelope": (0.871553, 1.009524),
    "topk_v4_vs_exact_naive": (1.097225, 1.693444),
    "histogram_v2_vs_reference_envelope": (1.101552, 4.302632),
    "scan_cub_block_vs_naive": (1.024946, 1.076496),
    "permute_full_v2_vs_vllm": (1.567106, 1.850121),
    "grouped_v2_vs_external_envelope": (0.869669, 1.641089),
    "unpermute_vec4_vs_reference_envelope": (1.015376, 1.289228),
}
PERFORMANCE_RECORDS = (
    "docs/dense_gemm/performance-record.md",
    "docs/topk_gate/performance-record.md",
    "docs/histogram/performance-record.md",
    "docs/scan/performance-record.md",
    "docs/permute/performance-record.md",
    "docs/grouped_gemm/performance-record.md",
    "docs/unpermute/performance-record.md",
)
CANONICAL_DOCS = (
    "README.md",
    "README.zh-CN.md",
    "docs/interview/README.md",
    "docs/interview/operator-performance.md",
    "docs/interview/bottleneck-analysis.md",
    "docs/interview/question-bank.md",
    "docs/implementation-status.md",
    "docs/operator-optimization-index.md",
    *PERFORMANCE_RECORDS,
)


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def close(actual: object, expected: float, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def validate(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    bundle = root / BUNDLE
    required_bundle_files = {
        "summary.json",
        "operator_summary.csv",
        "release_pairs.json",
        "release_pairs.csv",
        "ncu_metrics.json",
        "ncu_metrics.csv",
        "nsys_hotspots.json",
        "nsys_hotspots.csv",
        "environment-and-commands.json",
        "REPORT.md",
        "SHA256SUMS",
    }
    if not bundle.is_dir():
        return [f"missing portfolio bundle: {BUNDLE.as_posix()}"]
    found = {path.name for path in bundle.iterdir() if path.is_file()}
    if found != required_bundle_files:
        errors.append(
            f"portfolio bundle file set mismatch: missing={sorted(required_bundle_files - found)}, "
            f"extra={sorted(found - required_bundle_files)}"
        )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jsonl", ".ncu-rep", ".nsys-rep", ".sqlite", ".zip"}
    ]
    if forbidden:
        errors.append(f"raw or binary artifacts are tracked in compact bundle: {forbidden}")

    sums_path = bundle / "SHA256SUMS"
    if sums_path.is_file():
        checksums: dict[str, str] = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("  ", 1)
            if len(parts) != 2:
                errors.append(f"malformed checksum line: {line!r}")
                continue
            expected_hash, name = parts
            checksums[name] = expected_hash
        expected_names = required_bundle_files - {"SHA256SUMS"}
        if set(checksums) != expected_names:
            errors.append("SHA256SUMS does not cover the exact compact file set")
        for name, expected_hash in checksums.items():
            path = bundle / name
            if path.is_file() and digest(path) != expected_hash:
                errors.append(f"checksum mismatch: {(BUNDLE / name).as_posix()}")

    summary_path = bundle / "summary.json"
    if not summary_path.is_file():
        return errors + ["missing summary.json"]
    summary = load_json(summary_path)
    if summary.get("evidence_commit") != EVIDENCE_SHA:
        errors.append("summary evidence_commit does not match the frozen SHA")
    for field, expected in (("release_records", 3725), ("aggregate_groups", 745), ("independent_processes", 5)):
        if summary.get(field) != expected:
            errors.append(f"summary {field} does not equal {expected}")
    if not close(summary.get("cv_threshold"), 0.50):
        errors.append("summary CV threshold is not 0.50")
    primary = {item.get("study_id"): item for item in summary.get("primary_operator_studies", [])}
    if set(primary) != set(PRIMARY_EXPECTED):
        errors.append(f"primary study set mismatch: {sorted(primary)}")
    for study_id, (ratio, best) in PRIMARY_EXPECTED.items():
        item = primary.get(study_id, {})
        if not close(item.get("ratio_of_sums_p50_speedup"), ratio):
            errors.append(f"{study_id}: ratio-of-sums drift")
        if not close(item.get("best_shape_speedup"), best):
            errors.append(f"{study_id}: best-shape speedup drift")
        if item.get("auto_dispatch_changed") is not False:
            errors.append(f"{study_id}: Auto must remain unchanged")

    csv_path = bundle / "operator_summary.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_ids = {row["study_id"] for row in csv.DictReader(handle)}
        json_ids = {item["study_id"] for item in summary.get("all_studies", [])}
        if csv_ids != json_ids:
            errors.append("operator_summary.csv study IDs do not match summary.json")

    manifest_path = bundle / "environment-and-commands.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("evidence_commit") != EVIDENCE_SHA:
            errors.append("environment manifest SHA drift")
        environment = manifest.get("environment", {})
        expected_environment = {
            "gpu_name": "NVIDIA GeForce RTX 3080",
            "compute_capability": "8.6",
            "sm_count": 68,
            "build_type": "Release",
            "build_git_dirty": "false",
        }
        for field, expected in expected_environment.items():
            if environment.get(field) != expected:
                errors.append(f"environment {field} does not equal {expected!r}")

    policy = load_json(root / "configs/policies/portfolio_aggressive.json")
    if policy.get("schema_version") != "raggedroute.cuda_v4_portfolio_aggressive_policy.v2":
        errors.append("portfolio policy schema was not advanced to v2")
    if not close(policy.get("maximum_all_samples_cv"), 0.50):
        errors.append("portfolio policy maximum_all_samples_cv is not 0.50")

    profile = load_json(root / "configs/project/profile/interview_portfolio_baselines.json")
    fused_baseline = next(
        (case for case in profile.get("cases", []) if case.get("id") == "histogram_scan.baseline.r4096_e64"),
        None,
    )
    if fused_baseline is None:
        errors.append("missing histogram+scan baseline profile case")
    elif fused_baseline.get("kernel_pattern") != "histogram_single_cta_shared_kernel":
        errors.append("Windows-safe histogram+scan baseline kernel pattern drift")

    texts: dict[str, str] = {}
    for relative in CANONICAL_DOCS:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing canonical document: {relative}")
            continue
        texts[relative] = path.read_text(encoding="utf-8")
    for relative in ("README.md", "README.zh-CN.md", "docs/interview/README.md", "docs/interview/operator-performance.md", "docs/interview/bottleneck-analysis.md", "docs/implementation-status.md"):
        if relative in texts and EVIDENCE_SHA not in texts[relative]:
            errors.append(f"canonical document does not cite full evidence SHA: {relative}")
    for relative in PERFORMANCE_RECORDS:
        if relative in texts and "2026-08-29" not in texts[relative]:
            errors.append(f"performance record lacks latest date: {relative}")
        if relative in texts and EVIDENCE_SHA not in texts[relative]:
            errors.append(f"performance record lacks evidence SHA: {relative}")

    required_claims = {
        "README.md": ("0.8716x", "1.0972x", "1.1016x", "1.5671x", "0.8697x", "1.0154x", "CV<=0.50"),
        "README.zh-CN.md": ("0.8716x", "1.0972x", "1.1016x", "1.5671x", "0.8697x", "1.0154x", "CV<=0.50"),
        "docs/interview/operator-performance.md": (
            "cuda_register_tiled_v3_64x32_async",
            "cuda_local_pair_two_reduce_top2_v4",
            "cuda_candidate_v2",
            "cuda_candidate_v2_from_ids",
            "cuda_grouped_sm86_fp32_v2",
            "cuda_warp_token_vec4",
        ),
    }
    for relative, claims in required_claims.items():
        text = texts.get(relative, "")
        for claim in claims:
            if claim not in text:
                errors.append(f"missing canonical claim {claim!r} in {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"interview portfolio validation failed: {len(errors)} error(s)")
        return 1
    print("interview portfolio validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
