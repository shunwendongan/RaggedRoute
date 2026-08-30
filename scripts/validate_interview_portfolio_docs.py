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
GOVERNANCE_BASE_SHA = "b74616e445938e077743c997835a8eb7ec177b7d"
GROUPED_EVIDENCE_SHA = "c2205ed1ba1063fccce3cd417fd671798dbfb66f"
GROUPED_BUNDLE = pathlib.Path("docs/reports/compact/20260830-c2205ed-grouped-v6")
GROUPED_REQUIRED_FILES = {
    "commands.md",
    "decision-v6-vs-cublas-per-expert.json",
    "decision-v6-vs-cuda-grouped-sm86-fp32-v2.json",
    "decision-v6-vs-cuda-grouped-sm86-fp32-v5-balanced-direct.json",
    "decision-v6-vs-cutlass-grouped.json",
    "environment.json",
    "profiler-metrics.json",
    "REPORT.md",
    "v5-matrix-rows.csv",
    "v5-matrix-summary.json",
    "v6-matrix-rows.csv",
    "v6-matrix-summary.json",
    "SHA256SUMS",
}
LATEST_GROUPED_EVIDENCE_SHA = "dea7c066a83a5df700aa60c03fd51446c6b4c5e5"
LATEST_GROUPED_BUNDLE = pathlib.Path(
    "docs/reports/compact/20260830-dea7c06-grouped-v9-v10"
)
LATEST_GROUPED_REQUIRED_FILES = {
    "commands.md",
    "environment.json",
    "ncu-basic.csv",
    "ncu-basic.json",
    "ncu-detailed.csv",
    "ncu-detailed.json",
    "nsys-cutlass-kernels.csv",
    "nsys-v6-fallback-kernels.csv",
    "nsys-v9-kernels.csv",
    "raw-artifact-inventory.json",
    "REPORT.md",
    "sanitizer-summary.csv",
    "SHA256SUMS",
    "v10-selector-rows.csv",
    "v10-selector-summary.json",
    "v9-matrix-rows.csv",
    "v9-matrix-summary.json",
}
PRIMARY_EXPECTED = {
    "dense_v3_vs_library_envelope": (0.871553, 1.009524),
    "topk_v4_vs_exact_naive": (1.097225, 1.693444),
    "histogram_v2_vs_reference_envelope": (1.101552, 4.302632),
    "scan_cub_block_vs_naive": (1.024946, 1.076496),
    "permute_full_v2_vs_vllm": (1.567106, 1.850121),
    "grouped_v2_vs_external_envelope": (0.869669, 1.641089),
    "unpermute_vec4_vs_reference_envelope": (1.015376, 1.289228),
}
EXPECTED_NCU_CASES = {
    ("candidate", "dense.candidate.m1024", "basic"),
    ("candidate", "topk.candidate.t2048_e64", "basic"),
    ("candidate", "histogram.candidate.r1m_e64", "basic"),
    ("candidate", "histogram_scan.candidate.r4096_e64", "basic"),
    ("candidate", "scan.candidate.e64", "basic"),
    ("candidate", "permute.candidate.t4096_k1024", "basic"),
    ("candidate", "grouped.candidate.t2048_zipf14", "basic"),
    ("candidate", "unpermute.candidate.t1024_n256", "basic"),
    ("baseline", "dense.baseline.m1024", "basic"),
    ("baseline", "topk.baseline.t2048_e64", "basic"),
    ("baseline", "histogram.baseline.r1m_e64", "basic"),
    ("baseline", "histogram_scan.baseline.r4096_e64", "basic"),
    ("baseline", "scan.baseline.e64", "basic"),
    ("baseline", "permute.baseline.t2048_k256", "basic"),
    ("baseline", "grouped.baseline.t2048_zipf14", "basic"),
    ("baseline", "unpermute.baseline.t1024_n256", "basic"),
    ("candidate", "grouped.candidate.t2048_zipf14", "detailed"),
    ("baseline", "grouped.baseline.t2048_zipf14", "detailed"),
}
EXPECTED_NSYS_KERNELS = {
    "grouped_gemm",
    "token_permute",
    "topk_gate",
    "histogram_exclusive_scan",
    "unpermute",
}
ALLOWED_METRIC_STATUSES = {"collected", "not_collected", "unsupported_or_unknown"}
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
    "docs/portfolio-completion-audit.md",
    "docs/implementation-status.md",
    "docs/operator-optimization-index.md",
    "docs/development-roadmap.md",
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

    ncu_path = bundle / "ncu_metrics.json"
    if ncu_path.is_file():
        ncu_records = load_json(ncu_path)
        observed_ncu_cases = {
            (record.get("role"), record.get("case_id"), record.get("profile_set"))
            for record in ncu_records
        }
        if observed_ncu_cases != EXPECTED_NCU_CASES:
            errors.append(
                "portfolio NCU coverage mismatch: "
                f"missing={sorted(EXPECTED_NCU_CASES - observed_ncu_cases)}, "
                f"extra={sorted(observed_ncu_cases - EXPECTED_NCU_CASES)}"
            )
        if len(ncu_records) != len(EXPECTED_NCU_CASES):
            errors.append("portfolio NCU records contain duplicate case/set entries")
        for record in ncu_records:
            metrics = record.get("metrics", {})
            device = metrics.get("device_name", {})
            cc_major = metrics.get("cc_major", {})
            cc_minor = metrics.get("cc_minor", {})
            if device.get("status") != "collected" or device.get("value") != "NVIDIA GeForce RTX 3080":
                errors.append(f"NCU device drift: {record.get('case_id')}")
            if cc_major.get("value") != 8 or cc_minor.get("value") != 6:
                errors.append(f"NCU compute capability drift: {record.get('case_id')}")
            for concept, metric in metrics.items():
                status = metric.get("status")
                if status not in ALLOWED_METRIC_STATUSES:
                    errors.append(
                        f"invalid NCU metric status {status!r}: {record.get('case_id')}:{concept}"
                    )
                elif status == "collected" and (
                    metric.get("metric") is None or metric.get("value") is None
                ):
                    errors.append(
                        f"collected NCU concept lacks metric/value: {record.get('case_id')}:{concept}"
                    )
                elif status != "collected" and metric.get("value") is not None:
                    errors.append(
                        f"unavailable NCU concept has numeric value: {record.get('case_id')}:{concept}"
                    )

    nsys_path = bundle / "nsys_hotspots.json"
    if nsys_path.is_file():
        nsys_records = load_json(nsys_path)
        distributions = {record.get("distribution") for record in nsys_records}
        if distributions != {"uniform", "zipf14"}:
            errors.append(f"portfolio NSYS distribution coverage drift: {sorted(distributions)}")
        for distribution in ("uniform", "zipf14"):
            records = [
                record for record in nsys_records if record.get("distribution") == distribution
            ]
            if len(records) != 5:
                errors.append(f"{distribution} NSYS hotspot count is not 5")
                continue
            kernels = {record.get("kernel", "") for record in records}
            for expected_kernel in EXPECTED_NSYS_KERNELS:
                if not any(expected_kernel in kernel for kernel in kernels):
                    errors.append(
                        f"{distribution} NSYS is missing {expected_kernel} hotspot"
                    )
            if not close(sum(float(record.get("time_pct", 0.0)) for record in records), 100.0, 0.05):
                errors.append(f"{distribution} NSYS hotspot percentages do not sum to 100")

    grouped_bundle = root / GROUPED_BUNDLE
    if not grouped_bundle.is_dir():
        errors.append(f"missing Grouped v6 bundle: {GROUPED_BUNDLE.as_posix()}")
    else:
        grouped_found = {path.name for path in grouped_bundle.iterdir() if path.is_file()}
        if grouped_found != GROUPED_REQUIRED_FILES:
            errors.append(
                "Grouped v6 bundle file set mismatch: "
                f"missing={sorted(GROUPED_REQUIRED_FILES - grouped_found)}, "
                f"extra={sorted(grouped_found - GROUPED_REQUIRED_FILES)}"
            )
        grouped_forbidden = [
            path.relative_to(root).as_posix()
            for path in grouped_bundle.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".jsonl", ".ncu-rep", ".nsys-rep", ".sqlite", ".zip"}
        ]
        if grouped_forbidden:
            errors.append(f"raw or binary artifacts are tracked in Grouped bundle: {grouped_forbidden}")
        grouped_sums = grouped_bundle / "SHA256SUMS"
        if grouped_sums.is_file():
            checksums: dict[str, str] = {}
            for line in grouped_sums.read_text(encoding="utf-8").splitlines():
                parts = line.split("  ", 1)
                if len(parts) != 2:
                    errors.append(f"malformed Grouped checksum line: {line!r}")
                    continue
                checksums[parts[1]] = parts[0]
            expected_names = GROUPED_REQUIRED_FILES - {"SHA256SUMS"}
            if set(checksums) != expected_names:
                errors.append("Grouped SHA256SUMS does not cover the exact compact file set")
            for name, expected_hash in checksums.items():
                path = grouped_bundle / name
                if path.is_file() and digest(path) != expected_hash:
                    errors.append(f"checksum mismatch: {(GROUPED_BUNDLE / name).as_posix()}")

        grouped_summary_path = grouped_bundle / "v6-matrix-summary.json"
        if grouped_summary_path.is_file():
            grouped_summary = load_json(grouped_summary_path)
            expected_summary = {
                "candidate": "cuda_grouped_sm86_fp32_v6_balanced_32x128",
                "record_count": 250,
                "case_count": 10,
                "variant_count": 5,
            }
            for field, expected in expected_summary.items():
                if grouped_summary.get(field) != expected:
                    errors.append(f"Grouped summary {field} does not equal {expected!r}")
            quality = grouped_summary.get("evidence_quality", {})
            if quality.get("all_validation_ok") is not True or quality.get("all_clean_build") is not True:
                errors.append("Grouped v6 evidence is not validated clean Release evidence")
            if not close(quality.get("maximum_process_cv"), 0.496793366473):
                errors.append("Grouped v6 maximum CV drift")
            if quality.get("groups_above_cv_evidence_ceiling") != 0:
                errors.append("Grouped v6 contains a CV evidence-ceiling violation")
            envelope = grouped_summary.get("comparisons", {}).get(
                "fastest_library_envelope", {}
            ).get("summary", {})
            if not close(envelope.get("ratio_of_sums_p50_speedup"), 0.9945583293423579):
                errors.append("Grouped v6 library-envelope ratio drift")
            if not close(envelope.get("shape_geomean_p50_speedup"), 1.0351901128615002):
                errors.append("Grouped v6 library-envelope geomean drift")

        grouped_rows_path = grouped_bundle / "v6-matrix-rows.csv"
        if grouped_rows_path.is_file():
            with grouped_rows_path.open("r", encoding="utf-8", newline="") as handle:
                grouped_rows = list(csv.DictReader(handle))
            envelope_rows = {
                row["case_id"]: row
                for row in grouped_rows
                if row.get("baseline") == "fastest_library_envelope"
            }
            expected_shapes = {
                "grouped.v6.uniform.t512_e64_k128_n128": 1.225734862846226,
                "grouped.v6.single_hot.t512_e64_k128_n128": 1.7762180325280665,
                "grouped.v6.uniform.t2048_e64_k128_n128": 0.7533617836250541,
            }
            for case_id, expected in expected_shapes.items():
                if not close(envelope_rows.get(case_id, {}).get("p50_speedup"), expected):
                    errors.append(f"Grouped v6 headline shape drift: {case_id}")

        grouped_environment_path = grouped_bundle / "environment.json"
        if grouped_environment_path.is_file():
            grouped_environment = load_json(grouped_environment_path)
            if grouped_environment.get("source_commit") != GROUPED_EVIDENCE_SHA:
                errors.append("Grouped environment source commit drift")
            if grouped_environment.get("source_clean") is not True:
                errors.append("Grouped environment is not clean")

    latest_grouped_bundle = root / LATEST_GROUPED_BUNDLE
    if not latest_grouped_bundle.is_dir():
        errors.append(
            f"missing Grouped V9/V10 bundle: {LATEST_GROUPED_BUNDLE.as_posix()}"
        )
    else:
        latest_grouped_found = {
            path.name for path in latest_grouped_bundle.iterdir() if path.is_file()
        }
        if latest_grouped_found != LATEST_GROUPED_REQUIRED_FILES:
            errors.append(
                "Grouped V9/V10 bundle file set mismatch: "
                f"missing={sorted(LATEST_GROUPED_REQUIRED_FILES - latest_grouped_found)}, "
                f"extra={sorted(latest_grouped_found - LATEST_GROUPED_REQUIRED_FILES)}"
            )
        latest_grouped_forbidden = [
            path.relative_to(root).as_posix()
            for path in latest_grouped_bundle.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in {".jsonl", ".ncu-rep", ".nsys-rep", ".sqlite", ".zip"}
        ]
        if latest_grouped_forbidden:
            errors.append(
                "raw or binary artifacts are tracked in Grouped V9/V10 bundle: "
                f"{latest_grouped_forbidden}"
            )

        latest_grouped_sums = latest_grouped_bundle / "SHA256SUMS"
        if latest_grouped_sums.is_file():
            checksums: dict[str, str] = {}
            for line in latest_grouped_sums.read_text(encoding="utf-8").splitlines():
                parts = line.split("  ", 1)
                if len(parts) != 2:
                    errors.append(
                        f"malformed Grouped V9/V10 checksum line: {line!r}"
                    )
                    continue
                checksums[parts[1]] = parts[0]
            expected_names = LATEST_GROUPED_REQUIRED_FILES - {"SHA256SUMS"}
            if set(checksums) != expected_names:
                errors.append(
                    "Grouped V9/V10 SHA256SUMS does not cover the exact compact file set"
                )
            for name, expected_hash in checksums.items():
                path = latest_grouped_bundle / name
                if path.is_file() and digest(path) != expected_hash:
                    errors.append(
                        f"checksum mismatch: {(LATEST_GROUPED_BUNDLE / name).as_posix()}"
                    )

        v9_summary_path = latest_grouped_bundle / "v9-matrix-summary.json"
        if v9_summary_path.is_file():
            v9_summary = load_json(v9_summary_path)
            expected_summary = {
                "candidate": "cuda_grouped_sm86_fp32_v9_balanced_32x64",
                "record_count": 375,
                "case_count": 15,
                "variant_count": 5,
            }
            for field, expected in expected_summary.items():
                if v9_summary.get(field) != expected:
                    errors.append(f"Grouped V9 summary {field} does not equal {expected!r}")
            cv_policy = v9_summary.get("cv_policy", {})
            if not close(cv_policy.get("risk_disclosure_threshold"), 0.10):
                errors.append("Grouped V9 CV risk threshold drift")
            if not close(cv_policy.get("evidence_ceiling"), 0.50):
                errors.append("Grouped V9 CV evidence ceiling drift")
            quality = v9_summary.get("evidence_quality", {})
            if (
                quality.get("all_validation_ok") is not True
                or quality.get("all_release") is not True
                or quality.get("all_clean_build") is not True
            ):
                errors.append("Grouped V9 evidence is not validated clean Release evidence")
            if not close(quality.get("maximum_process_cv"), 0.504134866414):
                errors.append("Grouped V9 maximum CV drift")
            if quality.get("groups_above_cv_risk_threshold") != 62:
                errors.append("Grouped V9 CV risk-disclosure group count drift")
            if quality.get("groups_above_cv_evidence_ceiling") != 1:
                errors.append("Grouped V9 CV evidence-ceiling group count drift")

            comparisons = v9_summary.get("comparisons", {})
            envelope = comparisons.get("fastest_library_envelope", {}).get(
                "summary", {}
            )
            expected_envelope = {
                "ratio_of_sums_p50_speedup": 1.0916490327497934,
                "shape_geomean_p50_speedup": 1.1396610481789666,
                "shape_win_fraction": 0.7333333333333333,
                "maximum_p50_regression_fraction": 0.33197388853198473,
                "maximum_p95_regression_fraction": 0.8318990937167612,
            }
            for field, expected in expected_envelope.items():
                if not close(envelope.get(field), expected):
                    errors.append(f"Grouped V9 library-envelope {field} drift")
            if envelope.get("candidate_faster_process_pairs") != 53:
                errors.append("Grouped V9 library-envelope process-win count drift")
            if envelope.get("paired_process_pairs") != 75:
                errors.append("Grouped V9 library-envelope process-pair count drift")

            v6 = comparisons.get(
                "cuda_grouped_sm86_fp32_v6_balanced_32x128", {}
            ).get("summary", {})
            if not close(v6.get("ratio_of_sums_p50_speedup"), 1.0485972412756566):
                errors.append("Grouped V9 versus V6 ratio drift")
            if not close(v6.get("shape_geomean_p50_speedup"), 1.0385930912340482):
                errors.append("Grouped V9 versus V6 geomean drift")

        v9_rows_path = latest_grouped_bundle / "v9-matrix-rows.csv"
        if v9_rows_path.is_file():
            with v9_rows_path.open("r", encoding="utf-8", newline="") as handle:
                v9_rows = list(csv.DictReader(handle))
            envelope_rows = {
                row["case_id"]: row
                for row in v9_rows
                if row.get("baseline") == "fastest_library_envelope"
            }
            expected_shapes = {
                "grouped.v10.uniform.t512_e32_k128_n64": 1.8818704380595692,
                "grouped.v10.zipf14.t2048_e64_k128_n64": 1.4366306130042634,
                "grouped.v10.uniform.t4096_e64_k128_n64": 1.3660714849970312,
                "grouped.v10.uniform.t2048_e64_k256_n64": 0.9072058247008087,
                "grouped.v10.uniform.t2048_e64_k128_n128": 0.773565950490079,
                "grouped.v10.nonaligned.t512_e64_k127_n129": 0.7507654681595411,
            }
            for case_id, expected in expected_shapes.items():
                if not close(envelope_rows.get(case_id, {}).get("p50_speedup"), expected):
                    errors.append(f"Grouped V9 headline/counterexample drift: {case_id}")

        v10_summary_path = latest_grouped_bundle / "v10-selector-summary.json"
        if v10_summary_path.is_file():
            v10_summary = load_json(v10_summary_path)
            expected_summary = {
                "candidate": "cuda_grouped_sm86_fp32_v10_wave_aware_portfolio",
                "record_count": 375,
                "case_count": 15,
                "variant_count": 5,
            }
            for field, expected in expected_summary.items():
                if v10_summary.get(field) != expected:
                    errors.append(f"Grouped V10 summary {field} does not equal {expected!r}")
            v10_vs_v9 = v10_summary.get("comparisons", {}).get(
                "cuda_grouped_sm86_fp32_v9_balanced_32x64", {}
            ).get("summary", {})
            if not close(
                v10_vs_v9.get("ratio_of_sums_p50_speedup"), 0.9774727556181431
            ):
                errors.append("Grouped V10 versus V9 ratio drift")
            if not close(
                v10_vs_v9.get("shape_geomean_p50_speedup"), 0.9894806758590254
            ):
                errors.append("Grouped V10 versus V9 geomean drift")
            if v10_vs_v9.get("candidate_faster_process_pairs") != 36:
                errors.append("Grouped V10 versus V9 process-win count drift")

        latest_grouped_environment_path = latest_grouped_bundle / "environment.json"
        if latest_grouped_environment_path.is_file():
            latest_grouped_environment = load_json(latest_grouped_environment_path)
            if (
                latest_grouped_environment.get("release_git_sha")
                != LATEST_GROUPED_EVIDENCE_SHA
            ):
                errors.append("Grouped V9/V10 environment source commit drift")
            if latest_grouped_environment.get("release_git_clean") is not True:
                errors.append("Grouped V9/V10 environment is not clean")
            gpu = latest_grouped_environment.get("gpu", {})
            expected_gpu = {
                "name": "NVIDIA GeForce RTX 3080",
                "compute_capability": "8.6",
                "sm_count": 68,
            }
            for field, expected in expected_gpu.items():
                if gpu.get(field) != expected:
                    errors.append(
                        f"Grouped V9/V10 environment GPU {field} does not equal {expected!r}"
                    )

        expected_v9_basic = {
            "grouped.v10.cutlass.t4096_e64_k128_n64",
            "grouped.v10.v6_portfolio_fallback.t4096_e64_k128_n64",
            "grouped.v10.v9_32x64.t4096_e64_k128_n64",
        }
        expected_v9_detailed = {
            "grouped.v10.v6_portfolio_fallback.t4096_e64_k128_n64",
            "grouped.v10.v9_32x64.t4096_e64_k128_n64",
        }
        for name, profile_set, expected_cases in (
            ("ncu-basic.json", "basic", expected_v9_basic),
            ("ncu-detailed.json", "detailed", expected_v9_detailed),
        ):
            path = latest_grouped_bundle / name
            if not path.is_file():
                continue
            records = load_json(path)
            observed_cases = {record.get("case_id") for record in records}
            if observed_cases != expected_cases or len(records) != len(expected_cases):
                errors.append(f"Grouped V9/V10 {profile_set} NCU coverage drift")
            for record in records:
                if record.get("profile_set") != profile_set:
                    errors.append(
                        f"Grouped V9/V10 NCU set drift: {record.get('case_id')}"
                    )
                for concept, metric in record.get("metrics", {}).items():
                    if metric.get("status") not in ALLOWED_METRIC_STATUSES:
                        errors.append(
                            "invalid Grouped V9/V10 NCU metric status: "
                            f"{record.get('case_id')}:{concept}"
                        )

        sanitizer_path = latest_grouped_bundle / "sanitizer-summary.csv"
        if sanitizer_path.is_file():
            with sanitizer_path.open("r", encoding="utf-8", newline="") as handle:
                sanitizer_rows = list(csv.DictReader(handle))
            if len(sanitizer_rows) != 24:
                errors.append("Grouped V9/V10 targeted sanitizer coverage is not 24 cases")
            if any(row.get("status") != "pass" for row in sanitizer_rows):
                errors.append("Grouped V9/V10 targeted sanitizer contains a failure")

    policy = load_json(root / "configs/policies/portfolio_aggressive.json")
    if policy.get("schema_version") != "raggedroute.cuda_v4_portfolio_aggressive_policy.v2":
        errors.append("portfolio policy schema was not advanced to v2")
    if not close(policy.get("maximum_all_samples_cv"), 0.50):
        errors.append("portfolio policy maximum_all_samples_cv is not 0.50")
    default_policy = load_json(root / "configs/policies/default_promotion.json")
    if default_policy.get("schema_version") != "raggedroute.promotion_policy.v2":
        errors.append("default promotion policy schema was not advanced to v2")
    if not close(default_policy.get("maximum_all_samples_cv"), 0.50):
        errors.append("default promotion policy maximum_all_samples_cv is not 0.50")

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
    for relative in (
        "README.md",
        "README.zh-CN.md",
        "docs/interview/README.md",
        "docs/interview/operator-performance.md",
        "docs/interview/bottleneck-analysis.md",
        "docs/implementation-status.md",
        "docs/operator-optimization-index.md",
        "docs/development-roadmap.md",
    ):
        if relative in texts and EVIDENCE_SHA not in texts[relative]:
            errors.append(f"canonical document does not cite full evidence SHA: {relative}")
        if relative in texts and GROUPED_EVIDENCE_SHA not in texts[relative]:
            errors.append(f"canonical document does not cite Grouped evidence SHA: {relative}")
        if relative in texts and LATEST_GROUPED_EVIDENCE_SHA not in texts[relative]:
            errors.append(
                f"canonical document does not cite latest Grouped evidence SHA: {relative}"
            )
    for relative in PERFORMANCE_RECORDS:
        if relative in texts and "2026-08-29" not in texts[relative]:
            errors.append(f"performance record lacks latest date: {relative}")
        if relative in texts and EVIDENCE_SHA not in texts[relative]:
            errors.append(f"performance record lacks evidence SHA: {relative}")
    for relative in (
        "docs/portfolio-completion-audit.md",
        "docs/implementation-status.md",
    ):
        if relative in texts and GOVERNANCE_BASE_SHA not in texts[relative]:
            errors.append(f"completion document lacks PR #38 merge SHA: {relative}")
    cleanup_path = root / "docs/cleanup-review.md"
    if cleanup_path.is_file():
        cleanup_text = cleanup_path.read_text(encoding="utf-8")
        if GOVERNANCE_BASE_SHA not in cleanup_text or "PR #38" not in cleanup_text:
            errors.append("cleanup review does not record the merged governance baseline")

    required_claims = {
        "README.md": (
            "0.8716x",
            "1.0972x",
            "1.1016x",
            "1.5671x",
            "1.0916x",
            "1.8819x",
            "1.3661x",
            "0.9072x",
            "0.7736x",
            "0.7508x",
            "1.0154x",
            "CV<=0.50",
        ),
        "README.zh-CN.md": (
            "0.8716x",
            "1.0972x",
            "1.1016x",
            "1.5671x",
            "1.0916x",
            "1.8819x",
            "1.3661x",
            "0.9072x",
            "0.7736x",
            "0.7508x",
            "1.0154x",
            "CV<=0.50",
        ),
        "docs/interview/operator-performance.md": (
            "cuda_register_tiled_v3_64x32_async",
            "cuda_local_pair_two_reduce_top2_v4",
            "cuda_candidate_v2",
            "cuda_candidate_v2_from_ids",
            "cuda_grouped_sm86_fp32_v9_balanced_32x64",
            "cuda_grouped_sm86_fp32_v10_wave_aware_portfolio",
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
