#!/usr/bin/env python3
"""Build a compact, auditable interview-portfolio evidence bundle.

The script consumes frozen unprofiled Release evidence plus normalized NSYS/NCU
JSON.  It never uses profiler duration for a speedup claim.  Per-shape library
envelopes choose the fastest comparable baseline under the same case, process,
level, cache mode, seed, and timing boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Study:
    study_id: str
    operator: str
    candidate: str
    baselines: tuple[str, ...]
    level: str
    prefixes: tuple[str, ...]
    excludes: tuple[str, ...] = ()
    cache_mode: str | None = "warm"
    expected_shapes: int | None = None
    candidate_scope: str = "in_tree_candidate"
    primary: bool = False
    contract_note: str = "same operator contract"

    def accepts_case(self, case_id: str) -> bool:
        return any(case_id.startswith(prefix) for prefix in self.prefixes) and not any(
            excluded in case_id for excluded in self.excludes
        )


STUDIES = (
    Study(
        "dense_v3_vs_library_envelope",
        "dense_gemm",
        "cuda_register_tiled_v3_64x32_async",
        ("cublaslt", "cublas"),
        "L2_operator_steady",
        ("dense_gemm.fp32.",),
        expected_shapes=3,
        primary=True,
        contract_note="strict FP32; fastest cuBLASLt/cuBLAS per shape",
    ),
    Study(
        "topk_v4_vs_exact_naive",
        "topk_gate",
        "cuda_local_pair_two_reduce_top2_v4",
        ("cuda_naive",),
        "L2_operator_steady",
        ("topk_gate.v4_preflight.in_tree.",),
        expected_shapes=30,
        primary=True,
        contract_note="full tie/NaN/selected-softmax contract",
    ),
    Study(
        "topk_v4_vs_external_envelope",
        "topk_gate",
        "cuda_local_pair_two_reduce_top2_v4",
        ("vllm_row_packed_top2", "cub_block_radix_top2"),
        "L1_kernel_body",
        ("topk_gate.v4_preflight.library.",),
        expected_shapes=30,
        contract_note="limited random-input subdomain only; not full contract equivalence",
    ),
    Study(
        "histogram_v2_vs_reference_envelope",
        "histogram",
        "cuda_candidate_v2",
        ("cuda_candidate_v1", "cub_device_histogram", "cuda_naive"),
        "L2_operator_steady",
        ("histogram.v2.",),
        expected_shapes=15,
        primary=True,
        contract_note="fastest shipping-v1/CUB/naive reference per shape",
    ),
    Study(
        "histogram_v2_vs_shipping_v1",
        "histogram",
        "cuda_candidate_v2",
        ("cuda_candidate_v1",),
        "L2_operator_steady",
        ("histogram.v2.",),
        expected_shapes=15,
        contract_note="explicit research v2 versus current shipping candidate v1",
    ),
    Study(
        "scan_cub_block_vs_naive",
        "exclusive_scan",
        "cub_block_scan",
        ("cuda_naive",),
        "L2_operator_steady",
        ("exclusive_scan.",),
        expected_shapes=5,
        candidate_scope="library_reference",
        primary=True,
        contract_note="tiny E<=64 metadata scan",
    ),
    Study(
        "scan_cub_warp_vs_naive",
        "exclusive_scan",
        "cub_warp_scan",
        ("cuda_naive",),
        "L2_operator_steady",
        ("exclusive_scan.e1.", "exclusive_scan.e31.", "exclusive_scan.e32."),
        expected_shapes=3,
        candidate_scope="library_reference",
        contract_note="warp-supported E<=32 subset",
    ),
    Study(
        "scan_cub_device_vs_naive",
        "exclusive_scan",
        "cub_device_scan",
        ("cuda_naive",),
        "L2_operator_steady",
        ("exclusive_scan.",),
        expected_shapes=5,
        candidate_scope="library_reference",
        contract_note="device-wide primitive on tiny E<=64 metadata",
    ),
    Study(
        "permute_pure_v2_vs_token_owned",
        "token_permute",
        "cuda_candidate_v2",
        ("cuda_token_owned_top2",),
        "L2_operator_steady",
        ("permute.",),
        excludes=("permute.v3.full.",),
        cache_mode=None,
        expected_shapes=28,
        contract_note="pure-permute boundary; prepared routing metadata",
    ),
    Study(
        "permute_pure_v3_vs_token_owned",
        "token_permute",
        "cuda_candidate_v3",
        ("cuda_token_owned_top2",),
        "L2_operator_steady",
        ("permute.",),
        excludes=("permute.v3.full.",),
        cache_mode=None,
        expected_shapes=28,
        contract_note="pure-permute boundary; prepared routing metadata",
    ),
    Study(
        "permute_full_v2_vs_vllm",
        "token_permute",
        "cuda_candidate_v2_from_ids",
        ("vllm_moe_permute",),
        "L2_operator_steady",
        ("permute.v3.full.",),
        expected_shapes=5,
        primary=True,
        contract_note="full-from-ids preparation plus payload movement",
    ),
    Study(
        "permute_full_v3_vs_vllm",
        "token_permute",
        "cuda_candidate_v3_from_ids",
        ("vllm_moe_permute",),
        "L2_operator_steady",
        ("permute.v3.full.",),
        expected_shapes=5,
        contract_note="full-from-ids preparation plus payload movement",
    ),
    Study(
        "grouped_v1_vs_external_envelope",
        "grouped_gemm",
        "cuda_grouped_sm86_fp32_v1",
        ("cutlass_grouped", "cublas_per_expert"),
        "L2_operator_steady",
        ("grouped_gemm.v3.",),
        expected_shapes=10,
        contract_note="strict FP32; fastest CUTLASS/cuBLAS-per-expert per shape",
    ),
    Study(
        "grouped_v2_vs_external_envelope",
        "grouped_gemm",
        "cuda_grouped_sm86_fp32_v2",
        ("cutlass_grouped", "cublas_per_expert"),
        "L2_operator_steady",
        ("grouped_gemm.v3.",),
        expected_shapes=10,
        primary=True,
        contract_note="strict FP32; fastest CUTLASS/cuBLAS-per-expert per shape",
    ),
    Study(
        "grouped_v2_vs_cutlass",
        "grouped_gemm",
        "cuda_grouped_sm86_fp32_v2",
        ("cutlass_grouped",),
        "L2_operator_steady",
        ("grouped_gemm.v3.",),
        expected_shapes=10,
        contract_note="strict FP32 CUTLASS Grouped promotion baseline",
    ),
    Study(
        "grouped_v3_vs_external_envelope",
        "grouped_gemm",
        "cuda_grouped_sm86_fp32_v3",
        ("cutlass_grouped", "cublas_per_expert"),
        "L2_operator_steady",
        ("grouped_gemm.v3.",),
        expected_shapes=10,
        contract_note="strict FP32; fastest CUTLASS/cuBLAS-per-expert per shape",
    ),
    Study(
        "grouped_v4a_queue1024_vs_external_envelope",
        "grouped_gemm",
        "cuda_grouped_sm86_fp32_v4a_desc_queue_t1024",
        ("cutlass_grouped", "cublas_per_expert"),
        "L2_operator_steady",
        ("grouped_gemm.v3.",),
        expected_shapes=10,
        contract_note="strict FP32 descriptor queue-1024 experiment",
    ),
    Study(
        "unpermute_vec4_vs_vllm",
        "unpermute",
        "cuda_warp_token_vec4",
        ("vllm_finalize_routing",),
        "L2_operator_steady",
        ("unpermute.",),
        expected_shapes=32,
        contract_note="adapted vLLM reference under matched routing metadata",
    ),
    Study(
        "unpermute_vec4_vs_reference_envelope",
        "unpermute",
        "cuda_warp_token_vec4",
        ("vllm_finalize_routing", "cuda_naive"),
        "L2_operator_steady",
        ("unpermute.",),
        expected_shapes=32,
        primary=True,
        contract_note="fastest vLLM/naive reference per shape",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--profile-doctor", type=Path, required=True)
    parser.add_argument("--ncu-candidate", type=Path, required=True)
    parser.add_argument("--ncu-baseline", type=Path, required=True)
    parser.add_argument("--ncu-candidate-manifest", type=Path, required=True)
    parser.add_argument("--ncu-baseline-manifest", type=Path, required=True)
    parser.add_argument("--nsys-uniform", type=Path, required=True)
    parser.add_argument("--nsys-zipf", type=Path, required=True)
    parser.add_argument("--nsys-uniform-manifest", type=Path, required=True)
    parser.add_argument("--nsys-zipf-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_value(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def median_latency(group: dict[str, Any]) -> float:
    return float(group["median_of_process_medians_us"])


def process_latency(record: dict[str, Any]) -> float:
    timing = record.get("gpu_span_timing") or record.get("timing") or {}
    for key in ("p50_us", "batch_mean_us_p50"):
        if timing.get(key) is not None:
            return float(timing[key])
    raise KeyError(f"missing process p50 for {record.get('case_id')}/{record.get('variant')}")


def compact_params(group: dict[str, Any]) -> dict[str, Any]:
    return dict(sorted((group.get("case_config") or {}).items()))


def classify(summary: dict[str, Any], study: Study, cv_threshold: float) -> str:
    if study.candidate_scope == "library_reference":
        return "library_reference_wins" if summary["ratio_of_sums_p50_speedup"] > 1.0 else "naive_reference_wins"
    passes_matrix = (
        summary["ratio_of_sums_p50_speedup"] >= 1.01
        and summary["shape_geomean_p50_speedup"] >= 1.00
        and summary["benefit_coverage_fraction"] >= 0.60
        and summary["maximum_p50_regression_fraction"] <= 0.10
        and summary["maximum_cv"] <= cv_threshold
    )
    if passes_matrix:
        return "matrix_winner_research_only"
    if summary["best_shape_speedup"] > 1.0:
        return "local_winner_only"
    return "library_reference_wins"


def analyze_study(
    study: Study,
    groups: list[dict[str, Any]],
    process_index: dict[tuple[Any, ...], dict[str, Any]],
    cv_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        group
        for group in groups
        if group.get("operator") == study.operator
        and group.get("measurement_level") == study.level
        and (study.cache_mode is None or group.get("cache_mode") == study.cache_mode)
        and study.accepts_case(str(group.get("case_id", "")))
    ]
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in selected:
        by_case[str(group["case_id"])][str(group["variant"])] = group

    pairs: list[dict[str, Any]] = []
    process_wins = 0
    process_pairs = 0
    shapes_with_majority_process_gain = 0
    for case_id in sorted(by_case):
        variants = by_case[case_id]
        candidate = variants.get(study.candidate)
        baselines = [variants[name] for name in study.baselines if name in variants]
        if candidate is None or not baselines:
            continue
        baseline = min(baselines, key=median_latency)
        baseline_p50 = median_latency(baseline)
        candidate_p50 = median_latency(candidate)
        speedup = baseline_p50 / candidate_p50
        baseline_p95 = float(baseline["all_samples_p95_us"])
        candidate_p95 = float(candidate["all_samples_p95_us"])
        case_cache_mode = str(candidate.get("cache_mode"))

        case_process_wins = 0
        case_process_pairs = 0
        process_ids = sorted(
            {
                key[5]
                for key in process_index
                if key[:5]
                == (case_id, study.operator, study.candidate, study.level, case_cache_mode)
            }
        )
        for process_run in process_ids:
            candidate_record = process_index.get(
                (case_id, study.operator, study.candidate, study.level, case_cache_mode, process_run)
            )
            baseline_records = [
                process_index.get(
                    (case_id, study.operator, baseline_name, study.level, case_cache_mode, process_run)
                )
                for baseline_name in study.baselines
            ]
            baseline_records = [record for record in baseline_records if record is not None]
            if candidate_record is None or not baseline_records:
                continue
            baseline_record = min(baseline_records, key=process_latency)
            won = process_latency(candidate_record) < process_latency(baseline_record)
            case_process_wins += int(won)
            case_process_pairs += 1
        process_wins += case_process_wins
        process_pairs += case_process_pairs
        if case_process_pairs and case_process_wins > case_process_pairs / 2:
            shapes_with_majority_process_gain += 1

        pairs.append(
            {
                "study_id": study.study_id,
                "operator": study.operator,
                "case_id": case_id,
                "params": compact_params(candidate),
                "measurement_level": study.level,
                "cache_mode": case_cache_mode,
                "candidate_variant": study.candidate,
                "baseline_variant": baseline["variant"],
                "candidate_p50_us": round_value(candidate_p50),
                "baseline_p50_us": round_value(baseline_p50),
                "p50_speedup": round_value(speedup),
                "candidate_p95_us": round_value(candidate_p95),
                "baseline_p95_us": round_value(baseline_p95),
                "p95_speedup": round_value(baseline_p95 / candidate_p95),
                "candidate_cv": round_value(float(candidate["all_samples_cv"])),
                "baseline_cv": round_value(float(baseline["all_samples_cv"])),
                "candidate_workspace_bytes": int(candidate.get("workspace_bytes") or 0),
                "baseline_workspace_bytes": int(baseline.get("workspace_bytes") or 0),
                "process_direction_wins": case_process_wins,
                "process_direction_pairs": case_process_pairs,
            }
        )

    if study.expected_shapes is not None and len(pairs) != study.expected_shapes:
        raise ValueError(
            f"{study.study_id}: expected {study.expected_shapes} paired shapes, found {len(pairs)}"
        )
    if not pairs:
        raise ValueError(f"{study.study_id}: no paired shapes")

    ratios = [float(pair["p50_speedup"]) for pair in pairs]
    ratio_of_sums = sum(float(pair["baseline_p50_us"]) for pair in pairs) / sum(
        float(pair["candidate_p50_us"]) for pair in pairs
    )
    geomean = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    best = max(pairs, key=lambda pair: float(pair["p50_speedup"]))
    worst = min(pairs, key=lambda pair: float(pair["p50_speedup"]))
    max_p50_regression = max(
        0.0,
        max(float(pair["candidate_p50_us"]) / float(pair["baseline_p50_us"]) - 1.0 for pair in pairs),
    )
    max_p95_regression = max(
        0.0,
        max(float(pair["candidate_p95_us"]) / float(pair["baseline_p95_us"]) - 1.0 for pair in pairs),
    )
    summary: dict[str, Any] = {
        "study_id": study.study_id,
        "operator": study.operator,
        "candidate_variant": study.candidate,
        "baseline_variants": list(study.baselines),
        "candidate_scope": study.candidate_scope,
        "measurement_level": study.level,
        "cache_mode": study.cache_mode or "declared_per_case",
        "contract_note": study.contract_note,
        "paired_shapes": len(pairs),
        "ratio_of_sums_p50_speedup": round_value(ratio_of_sums),
        "shape_geomean_p50_speedup": round_value(geomean),
        "benefit_shapes": sum(ratio > 1.0 for ratio in ratios),
        "benefit_coverage_fraction": round_value(sum(ratio > 1.0 for ratio in ratios) / len(ratios)),
        "maximum_p50_regression_fraction": round_value(max_p50_regression),
        "maximum_p95_regression_fraction": round_value(max_p95_regression),
        "maximum_cv": round_value(
            max(max(float(pair["candidate_cv"]), float(pair["baseline_cv"])) for pair in pairs)
        ),
        "cv_threshold": cv_threshold,
        "cv_policy": "disclose_risk; CV above 0.10 does not auto-downgrade; CV above 0.50 is insufficient evidence",
        "process_direction_wins": process_wins,
        "process_direction_pairs": process_pairs,
        "process_direction_consistency_fraction": round_value(process_wins / process_pairs if process_pairs else None),
        "shapes_with_majority_process_gain": shapes_with_majority_process_gain,
        "best_shape_case_id": best["case_id"],
        "best_shape_params": best["params"],
        "best_shape_speedup": best["p50_speedup"],
        "best_shape_baseline_variant": best["baseline_variant"],
        "worst_shape_case_id": worst["case_id"],
        "worst_shape_params": worst["params"],
        "worst_shape_speedup": worst["p50_speedup"],
        "maximum_candidate_workspace_bytes": max(int(pair["candidate_workspace_bytes"]) for pair in pairs),
        "auto_dispatch_changed": False,
    }
    summary["classification"] = classify(summary, study, cv_threshold)
    return summary, pairs


NCU_FIELDS = (
    "device_name",
    "cc_major",
    "cc_minor",
    "sm_count",
    "duration_ns",
    "grid_size",
    "block_size",
    "waves_per_sm",
    "registers_per_thread",
    "shared_mem_per_block",
    "occupancy_theoretical_pct",
    "occupancy_achieved_pct",
    "sm_throughput_pct",
    "memory_throughput_pct",
    "dram_throughput_pct",
    "l1_throughput_pct",
    "l1_hit_rate_pct",
    "l2_hit_rate_pct",
    "local_load_instructions",
    "local_store_instructions",
    "eligible_warps_per_scheduler",
    "issue_active_pct",
    "stall_long_scoreboard",
    "stall_short_scoreboard",
    "stall_wait",
    "stall_barrier",
    "stall_math_pipe",
    "stall_memory_throttle",
)


def compact_ncu(paths: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for role, path in paths:
        for report in load_json(path):
            metrics = report.get("metrics", {})
            compact.append(
                {
                    "role": role,
                    "case_id": report.get("case_id"),
                    "profile_set": report.get("profile_set"),
                    "kernel": report.get("kernel"),
                    "metrics": {field: metrics.get(field, {"status": "not_collected", "metric": None, "value": None, "unit": None}) for field in NCU_FIELDS},
                }
            )
    return compact


def compact_nsys(paths: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for distribution, path in paths:
        for hotspot in load_json(path):
            raw = hotspot.get("raw", {})
            compact.append(
                {
                    "distribution": distribution,
                    "kernel": hotspot.get("name"),
                    "time_pct": float(hotspot.get("time_pct", 0.0)),
                    "total_time_ns": float(hotspot.get("total_time", 0.0)),
                    "instances": int(str(raw.get("Instances", "0")).replace(",", "")),
                    "average_ns": float(str(raw.get("Avg (ns)", "0")).replace(",", "")),
                    "median_ns": float(str(raw.get("Med (ns)", "0")).replace(",", "")),
                    "minimum_ns": float(str(raw.get("Min (ns)", "0")).replace(",", "")),
                    "maximum_ns": float(str(raw.get("Max (ns)", "0")).replace(",", "")),
                }
            )
    return compact


def sanitize_paths(value: Any, repo_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_paths(item, repo_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_paths(item, repo_root) for item in value]
    if isinstance(value, str):
        root = str(repo_root.resolve())
        if value.lower().startswith(root.lower()):
            suffix = value[len(root) :].lstrip("\\/")
            return "." if not suffix else str(Path(".") / Path(suffix))
    return value


def compact_profile_events(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    manifest = load_json(path)
    return [
        {
            "kind": event.get("kind"),
            "case_id": event.get("case_id"),
            "command": sanitize_paths(event.get("command", []), repo_root),
        }
        for event in manifest.get("events", [])
    ]


def write_csv_files(output_dir: Path, studies: list[dict[str, Any]], pairs: list[dict[str, Any]], ncu: list[dict[str, Any]], nsys: list[dict[str, Any]]) -> None:
    with (output_dir / "operator_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "study_id", "operator", "candidate_variant", "classification", "measurement_level",
            "paired_shapes", "ratio_of_sums_p50_speedup", "shape_geomean_p50_speedup",
            "benefit_shapes", "benefit_coverage_fraction", "maximum_p50_regression_fraction",
            "maximum_p95_regression_fraction", "maximum_cv", "process_direction_consistency_fraction",
            "best_shape_case_id", "best_shape_speedup", "worst_shape_case_id", "worst_shape_speedup",
            "contract_note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(studies)

    with (output_dir / "release_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "study_id", "operator", "case_id", "measurement_level", "cache_mode", "candidate_variant",
            "baseline_variant", "candidate_p50_us", "baseline_p50_us", "p50_speedup", "candidate_p95_us",
            "baseline_p95_us", "p95_speedup", "candidate_cv", "baseline_cv", "candidate_workspace_bytes",
            "baseline_workspace_bytes", "process_direction_wins", "process_direction_pairs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)

    with (output_dir / "ncu_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["role", "case_id", "profile_set", "kernel"]
        for metric in NCU_FIELDS:
            fields.extend((metric, f"{metric}_status"))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for report in ncu:
            row = {key: report[key] for key in ("role", "case_id", "profile_set", "kernel")}
            for metric in NCU_FIELDS:
                entry = report["metrics"][metric]
                row[metric] = entry.get("value")
                row[f"{metric}_status"] = entry.get("status")
            writer.writerow(row)

    with (output_dir / "nsys_hotspots.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["distribution", "kernel", "time_pct", "total_time_ns", "instances", "average_ns", "median_ns", "minimum_ns", "maximum_ns"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(nsys)


def report_markdown(primary: list[dict[str, Any]], source_sha: str, cv_threshold: float) -> str:
    lines = [
        "# RTX 3080 / SM86 interview portfolio evidence",
        "",
        f"Release evidence commit: `{source_sha}`. All speedups below use clean, unprofiled, five-process Release data. NCU/NSYS durations are diagnostic only.",
        "",
        "| Operator study | Candidate | Strongest comparable baseline | Ratio-of-sums | Shape geomean | Coverage | Best shape | Max CV | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in primary:
        baseline = "/".join(item["baseline_variants"])
        lines.append(
            f"| `{item['study_id']}` | `{item['candidate_variant']}` | `{baseline}` | "
            f"{item['ratio_of_sums_p50_speedup']:.4f}x | {item['shape_geomean_p50_speedup']:.4f}x | "
            f"{item['benefit_shapes']}/{item['paired_shapes']} | {item['best_shape_speedup']:.4f}x | "
            f"{item['maximum_cv']:.4f} | `{item['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence policy",
            "",
            f"This portfolio policy uses `CV <= {cv_threshold:.2f}` as the evidence ceiling. Values above 0.10 remain visible stability risks but do not automatically become `insufficient_evidence`. Matrix claims still require five complete processes, matched semantics and timing boundaries, ratio-of-sums, geomean, coverage, maximum regression, and cross-process direction checks.",
            "",
            "A `matrix_winner_research_only` label does not change `KernelFamily::kAuto`; `local_winner_only` must be described with its winning shape and full-matrix counterexample. External Top-K results apply only to the declared random-input subdomain.",
            "",
            "## Files",
            "",
            "- `summary.json` and `operator_summary.csv`: matrix-level decisions.",
            "- `release_pairs.json` and `release_pairs.csv`: compact per-shape Release evidence.",
            "- `ncu_metrics.json/csv`: selected normalized profiler metrics with explicit missing statuses.",
            "- `nsys_hotspots.json/csv`: uniform and Zipf L3 hotspot summaries.",
            "- `environment-and-commands.json`: provenance and executable command boundary.",
            "- `SHA256SUMS`: integrity hashes for every compact file in this bundle.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    aggregate = load_json(args.aggregate)
    groups = aggregate.get("groups", [])
    raw = load_jsonl(args.raw)
    if len(groups) != 745:
        raise ValueError(f"expected 745 aggregate groups, found {len(groups)}")
    if len(raw) != 3725:
        raise ValueError(f"expected 3725 raw records, found {len(raw)}")
    if not all(record.get("validation", {}).get("ok") for record in raw):
        raise ValueError("raw Release evidence contains failed validation")

    process_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in raw:
        key = (
            record.get("case_id"), record.get("operator"), record.get("variant"),
            record.get("measurement_level"), record.get("cache_mode"), record.get("process_run"),
        )
        if key in process_index:
            raise ValueError(f"duplicate raw process record: {key}")
        process_index[key] = record

    study_summaries: list[dict[str, Any]] = []
    release_pairs: list[dict[str, Any]] = []
    for study in STUDIES:
        summary, pairs = analyze_study(study, groups, process_index, args.cv_threshold)
        study_summaries.append(summary)
        release_pairs.extend(pairs)

    release_manifest = load_json(args.release_manifest)
    source_sha = str(release_manifest.get("repo_commit") or groups[0]["environment"]["build_git_sha"])
    dirty_values = {str(group["environment"].get("build_git_dirty")).lower() for group in groups}
    if dirty_values != {"false"}:
        raise ValueError(f"Release evidence is not uniformly clean: {dirty_values}")
    process_counts = {int(group["process_runs"]) for group in groups}
    if process_counts != {5}:
        raise ValueError(f"Release evidence does not uniformly contain five processes: {process_counts}")

    ncu = compact_ncu((("candidate", args.ncu_candidate), ("baseline", args.ncu_baseline)))
    nsys = compact_nsys((("uniform", args.nsys_uniform), ("zipf14", args.nsys_zipf)))
    primary_ids = {study.study_id for study in STUDIES if study.primary}
    primary = [summary for summary in study_summaries if summary["study_id"] in primary_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "schema_version": "raggedroute.interview_portfolio_summary.v1",
        "evidence_commit": source_sha,
        "suite_id": aggregate.get("suite_id"),
        "protocol": aggregate.get("protocol"),
        "release_records": len(raw),
        "aggregate_groups": len(groups),
        "independent_processes": 5,
        "warmup": 20,
        "samples": 30,
        "seed": 20260828,
        "cv_threshold": args.cv_threshold,
        "primary_operator_studies": primary,
        "all_studies": study_summaries,
    }
    write_json(args.output_dir / "summary.json", summary_payload)
    write_json(args.output_dir / "release_pairs.json", release_pairs)
    write_json(args.output_dir / "ncu_metrics.json", ncu)
    write_json(args.output_dir / "nsys_hotspots.json", nsys)
    write_csv_files(args.output_dir, study_summaries, release_pairs, ncu, nsys)

    repo_root = Path.cwd()
    source_paths = [
        args.aggregate, args.raw, args.release_manifest, args.profile_doctor, args.ncu_candidate, args.ncu_baseline,
        args.ncu_candidate_manifest, args.ncu_baseline_manifest, args.nsys_uniform, args.nsys_zipf,
        args.nsys_uniform_manifest, args.nsys_zipf_manifest,
    ]
    environment = groups[0]["environment"]
    profile_doctor = load_json(args.profile_doctor)
    commands_manifest = {
        "schema_version": "raggedroute.interview_portfolio_manifest.v1",
        "evidence_commit": source_sha,
        "repo_dirty_during_release": False,
        "environment": environment,
        "profiler_environment": sanitize_paths(profile_doctor, repo_root),
        "release_protocol": {
            "suite_id": aggregate.get("suite_id"),
            "processes": 5,
            "warmup": 20,
            "samples": 30,
            "seed": 20260828,
            "timing_boundary": "CUDA Event; unprofiled Release; per-case declared kernel repeats",
            "command": [
                "python", "scripts/run_benchmarks.py", "--binary", "out/build/rtx3080-sm86-release/raggedroute_benchmark.exe",
                "--config", "configs/project/benchmark/interview_portfolio_release.json", "--output", "out/interview-portfolio/release-9732a03-20260829-rerun.jsonl",
            ],
            "aggregate_command": [
                "python", "scripts/aggregate_results.py", "out/interview-portfolio/release-9732a03-20260829-rerun.jsonl",
                "--json", "out/interview-portfolio/release-9732a03-20260829.aggregate.json",
                "--csv", "out/interview-portfolio/release-9732a03-20260829.aggregate.csv",
                "--manifest", "out/interview-portfolio/release-9732a03-20260829-rerun.jsonl.manifest.json",
            ],
        },
        "profile_events": {
            "ncu_candidate": compact_profile_events(args.ncu_candidate_manifest, repo_root),
            "ncu_baseline": compact_profile_events(args.ncu_baseline_manifest, repo_root),
            "nsys_uniform": compact_profile_events(args.nsys_uniform_manifest, repo_root),
            "nsys_zipf14": compact_profile_events(args.nsys_zipf_manifest, repo_root),
        },
        "summary_command": sanitize_paths(
            [
                "python", "scripts/summarize_interview_portfolio.py",
                "--aggregate", str(args.aggregate), "--raw", str(args.raw),
                "--release-manifest", str(args.release_manifest),
                "--profile-doctor", str(args.profile_doctor),
                "--ncu-candidate", str(args.ncu_candidate), "--ncu-baseline", str(args.ncu_baseline),
                "--ncu-candidate-manifest", str(args.ncu_candidate_manifest),
                "--ncu-baseline-manifest", str(args.ncu_baseline_manifest),
                "--nsys-uniform", str(args.nsys_uniform), "--nsys-zipf", str(args.nsys_zipf),
                "--nsys-uniform-manifest", str(args.nsys_uniform_manifest),
                "--nsys-zipf-manifest", str(args.nsys_zipf_manifest),
                "--output-dir", str(args.output_dir), "--cv-threshold", f"{args.cv_threshold:.2f}",
            ],
            repo_root,
        ),
        "source_artifacts": [
            {"path": sanitize_paths(str(path), repo_root), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        ],
        "retention": {
            "tracked": "compact JSON/CSV/report/checksums only",
            "ignored_or_release_asset": "raw JSONL, full aggregate, run manifest, .ncu-rep, .nsys-rep, SQLite",
        },
    }
    write_json(args.output_dir / "environment-and-commands.json", commands_manifest)
    (args.output_dir / "REPORT.md").write_text(
        report_markdown(primary, source_sha, args.cv_threshold),
        encoding="utf-8",
        newline="\n",
    )

    checksum_files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(checksum_files) + 1} files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
