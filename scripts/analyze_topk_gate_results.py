#!/usr/bin/env python3
"""Apply the reviewed Top-K promotion gates and emit normalized evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Callable


CANDIDATES = (
    "cuda_warp_pair_top2_v1",
    "cuda_subwarp_pair_top2_v2",
    "cuda_vector_pair_top2_v3",
    "cuda_local_pair_two_reduce_top2_v4",
)
OLD_IN_TREE = ("cuda_naive", *CANDIDATES[:-1])
V4 = "cuda_local_pair_two_reduce_top2_v4"
POWER_EXPERTS = (2, 4, 8, 16, 32, 64)
LEVELS = ("L1_kernel_body", "L2_operator_steady")
THRESHOLDS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
BUCKETS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("E=2", lambda value: value == 2),
    ("E=3-8", lambda value: 3 <= value <= 8),
    ("E=9-16", lambda value: 9 <= value <= 16),
    ("E=17-32", lambda value: 17 <= value <= 32),
    ("E=33-64", lambda value: 33 <= value <= 64),
)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def interval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratio = sum(row["baseline_latency_us"] for row in rows) / sum(
        row["candidate_latency_us"] for row in rows
    )
    max_p50_regression = max(
        row["candidate_latency_us"] / row["baseline_latency_us"] - 1.0 for row in rows
    )
    max_p95_regression = max(row["p95_ratio"] - 1.0 for row in rows)
    max_cv = max(
        max(row["baseline_all_samples_cv"], row["candidate_all_samples_cv"])
        for row in rows
    )
    minimum_process_runs = min(
        min(row["baseline_process_runs"], row["candidate_process_runs"]) for row in rows
    )
    gates = {
        "ratio_of_sums_p50_ge_1_05": ratio >= 1.05,
        "max_single_shape_p50_regression_le_0_03": max_p50_regression <= 0.03,
        "max_single_shape_p95_regression_le_0_05": max_p95_regression <= 0.05,
        "max_all_samples_cv_le_0_10": max_cv <= 0.10,
        "minimum_process_runs_ge_5": minimum_process_runs >= 5,
        "zero_workspace_growth": all(row["workspace_growth_bytes"] == 0 for row in rows),
        "pairing_verified": all(row["pairing_verified"] for row in rows),
    }
    return {
        "shapes": len(rows),
        "ratio_of_sums_p50_speedup": ratio,
        "max_single_shape_p50_regression": max_p50_regression,
        "max_single_shape_p95_regression": max_p95_regression,
        "max_all_samples_cv": max_cv,
        "minimum_process_runs": minimum_process_runs,
        "gates": gates,
        "passed": all(gates.values()),
    }


def arbitrary_pair_summary(
    groups: list[dict[str, Any]], prefix: str, candidate: str, denominator: str
) -> dict[str, Any]:
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        if group["case_id"].startswith(prefix) and group["measurement_level"] == "L1_kernel_body":
            by_case[group["case_id"]][group["variant"]] = group
    for variants in by_case.values():
        if candidate in variants and denominator in variants:
            paired.append((variants[candidate], variants[denominator]))
    ratios = [right["all_samples_p50_us"] / left["all_samples_p50_us"] for left, right in paired]
    ratio_of_sums = sum(right["all_samples_p50_us"] for left, right in paired) / sum(
        left["all_samples_p50_us"] for left, right in paired
    )
    maximum_regression = max(1.0 / ratio - 1.0 for ratio in ratios)
    return {
        "candidate": candidate,
        "denominator": denominator,
        "paired_shapes": len(paired),
        "ratio_of_sums_speedup": ratio_of_sums,
        "maximum_single_shape_regression": maximum_regression,
        "overall_claim_allowed": ratio_of_sums >= 1.03 and maximum_regression <= 0.05,
    }


def shape_groups(
    groups: list[dict[str, Any]], prefix: str
) -> dict[tuple[str, int, int], dict[str, dict[str, Any]]]:
    result: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        if not group["case_id"].startswith(prefix):
            continue
        key = (
            group["measurement_level"],
            int(group["case_config"]["E"]),
            int(group["case_config"]["T"]),
        )
        result[key][group["variant"]] = group
    return result


def paired_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_p50_us": candidate["all_samples_p50_us"],
        "baseline_p50_us": baseline["all_samples_p50_us"],
        "candidate_p95_us": candidate["all_samples_p95_us"],
        "baseline_p95_us": baseline["all_samples_p95_us"],
        "candidate_cv": candidate["all_samples_cv"],
        "baseline_cv": baseline["all_samples_cv"],
        "process_runs": min(candidate["process_runs"], baseline["process_runs"]),
        "workspace_ok": candidate["variant_config"].get("workspace_bytes") == 0,
        "launch_ok": candidate["variant_config"].get("launch_count") == 1,
    }


def evaluate_v4_shape_dispatch(groups: list[dict[str, Any]]) -> dict[str, Any]:
    main = shape_groups(groups, "topk_gate.release.main.")
    cub = shape_groups(groups, "topk_gate.release.cub_pair.")
    vllm = shape_groups(groups, "topk_gate.release.vllm_pair.")
    attempts: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []

    for experts in POWER_EXPERTS:
        selected_promotion = None
        for threshold in THRESHOLDS:
            old_pairs: dict[str, list[dict[str, Any]]] = {level: [] for level in LEVELS}
            external_pairs: list[dict[str, Any]] = []
            selected_t = [value for value in THRESHOLDS if value >= threshold]
            complete = True
            for tokens in selected_t:
                for level in LEVELS:
                    variants = main.get((level, experts, tokens), {})
                    if V4 not in variants or not all(name in variants for name in OLD_IN_TREE):
                        complete = False
                        continue
                    best_old = min(
                        (variants[name] for name in OLD_IN_TREE),
                        key=lambda item: item["all_samples_p50_us"],
                    )
                    old_pairs[level].append(paired_gate(variants[V4], best_old))

                library_options: list[tuple[dict[str, Any], dict[str, Any], str]] = []
                cub_variants = cub.get(("L1_kernel_body", experts, tokens), {})
                if V4 in cub_variants and "cub_block_radix_top2" in cub_variants:
                    library_options.append(
                        (cub_variants[V4], cub_variants["cub_block_radix_top2"], "cub_block_radix_top2")
                    )
                vllm_variants = vllm.get(("L1_kernel_body", experts, tokens), {})
                if V4 in vllm_variants and "vllm_row_packed_top2" in vllm_variants:
                    library_options.append(
                        (vllm_variants[V4], vllm_variants["vllm_row_packed_top2"], "vllm_row_packed_top2")
                    )
                if not library_options:
                    complete = False
                else:
                    candidate, baseline, name = min(
                        library_options, key=lambda item: item[1]["all_samples_p50_us"]
                    )
                    external_pairs.append({**paired_gate(candidate, baseline), "baseline": name})

            def summarize(pairs: list[dict[str, Any]], minimum_speedup: float) -> dict[str, Any]:
                if not pairs:
                    return {"passed": False, "pairs": 0}
                ratio = sum(pair["baseline_p50_us"] for pair in pairs) / sum(
                    pair["candidate_p50_us"] for pair in pairs
                )
                no_p50_regression = all(
                    pair["candidate_p50_us"] <= pair["baseline_p50_us"] for pair in pairs
                )
                max_p95_regression = max(
                    pair["candidate_p95_us"] / pair["baseline_p95_us"] - 1.0 for pair in pairs
                )
                max_cv = max(
                    max(pair["candidate_cv"], pair["baseline_cv"]) for pair in pairs
                )
                min_runs = min(pair["process_runs"] for pair in pairs)
                passed = (
                    ratio >= minimum_speedup
                    and no_p50_regression
                    and max_p95_regression <= 0.05
                    and max_cv <= 0.10
                    and min_runs >= 5
                    and all(pair["workspace_ok"] and pair["launch_ok"] for pair in pairs)
                )
                return {
                    "passed": passed,
                    "pairs": len(pairs),
                    "ratio_of_sums_p50_speedup": ratio,
                    "no_single_shape_p50_regression": no_p50_regression,
                    "max_single_shape_p95_regression": max_p95_regression,
                    "max_all_samples_cv": max_cv,
                    "minimum_process_runs": min_runs,
                }

            l1 = summarize(old_pairs["L1_kernel_body"], 1.05)
            l2 = summarize(old_pairs["L2_operator_steady"], 1.05)
            library = summarize(external_pairs, 1.03)
            passed = complete and l1["passed"] and l2["passed"] and library["passed"]
            attempt = {
                "E": experts,
                "T_min": threshold,
                "complete": complete,
                "in_tree_l1": l1,
                "in_tree_l2": l2,
                "library_envelope_l1": library,
                "passed": passed,
            }
            attempts.append(attempt)
            if selected_promotion is None and passed:
                selected_promotion = attempt
        if selected_promotion is not None:
            promoted.append({"candidate": V4, "E": experts, "T_min": selected_promotion["T_min"]})
    return {"attempts": attempts, "promoted_intervals": promoted}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True, type=pathlib.Path)
    parser.add_argument("--comparison", required=True, type=pathlib.Path)
    parser.add_argument("--raw", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    aggregate = read_json(args.aggregate)
    comparisons = read_json(args.comparison)["comparisons"]
    groups = aggregate["groups"]
    raw_samples: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    with args.raw.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = (
                record["case_id"],
                record["variant"],
                record["measurement_level"],
                record["cache_mode"],
            )
            raw_samples[key].extend(record["timing"]["raw_batch_mean_samples_us"])
    config = {group["case_id"]: group["case_config"] for group in groups}
    main_rows = [
        row
        for row in comparisons
        if row["case_id"].startswith("topk_gate.release.main.")
        and row["candidate_variant"] in CANDIDATES
        and row["cache_mode"] == "warm"
    ]

    intervals: list[dict[str, Any]] = []
    for level in LEVELS:
        for candidate in CANDIDATES:
            candidate_rows = [
                row
                for row in main_rows
                if row["measurement_level"] == level
                and row["candidate_variant"] == candidate
            ]
            for bucket, predicate in BUCKETS:
                attempts = []
                promoted = None
                for threshold in THRESHOLDS:
                    selected = [
                        row
                        for row in candidate_rows
                        if predicate(int(config[row["case_id"]]["E"]))
                        and int(config[row["case_id"]]["T"]) >= threshold
                    ]
                    if not selected:
                        continue
                    metrics = interval_metrics(selected)
                    attempt = {"T_min": threshold, **metrics}
                    attempts.append(attempt)
                    if promoted is None and metrics["passed"]:
                        promoted = attempt
                intervals.append(
                    {
                        "measurement_level": level,
                        "candidate": candidate,
                        "bucket": bucket,
                        "promoted_interval": promoted,
                        "best_ratio_attempt": max(
                            attempts, key=lambda item: item["ratio_of_sums_p50_speedup"]
                        ),
                    }
                )
            if candidate in ("cuda_vector_pair_top2_v3", V4):
                attempts = []
                promoted_interval = None
                for threshold in THRESHOLDS:
                    selected = [
                        row
                        for row in candidate_rows
                        if int(config[row["case_id"]]["E"]) in {8, 16, 32, 64}
                        and int(config[row["case_id"]]["T"]) >= threshold
                    ]
                    if not selected:
                        continue
                    metrics = interval_metrics(selected)
                    attempt = {"T_min": threshold, **metrics}
                    attempts.append(attempt)
                    if promoted_interval is None and metrics["passed"]:
                        promoted_interval = attempt
                intervals.append(
                    {
                        "measurement_level": level,
                        "candidate": candidate,
                        "bucket": "aligned E={8,16,32,64}",
                        "promoted_interval": promoted_interval,
                        "best_ratio_attempt": max(
                            attempts, key=lambda item: item["ratio_of_sums_p50_speedup"]
                        ),
                    }
                )

    # The public policy requires an interval to pass at both L1 and L2.
    promoted = []
    for candidate in CANDIDATES:
        bucket_names = [bucket for bucket, _ in BUCKETS]
        if candidate in ("cuda_vector_pair_top2_v3", V4):
            bucket_names.append("aligned E={8,16,32,64}")
        for bucket in bucket_names:
            matching = [
                item
                for item in intervals
                if item["candidate"] == candidate and item["bucket"] == bucket
            ]
            if len(matching) == 2 and all(item["promoted_interval"] for item in matching):
                threshold = max(item["promoted_interval"]["T_min"] for item in matching)
                promoted.append({"candidate": candidate, "bucket": bucket, "T_min": threshold})

    external = [
        arbitrary_pair_summary(groups, "topk_gate.release.cub_pair.", candidate, "cub_block_radix_top2")
        for candidate in ("cuda_subwarp_pair_top2_v2", "cuda_vector_pair_top2_v3", V4)
    ]
    external.extend(
        arbitrary_pair_summary(groups, "topk_gate.release.vllm_pair.", candidate, "vllm_row_packed_top2")
        for candidate in ("cuda_subwarp_pair_top2_v2", "cuda_vector_pair_top2_v3", V4)
    )
    shape_dispatch = evaluate_v4_shape_dispatch(groups)

    representative = []
    wanted = {(32, 64), (2048, 8), (2048, 33), (2048, 64)}
    for group in groups:
        shape = (int(group["case_config"].get("T", -1)), int(group["case_config"].get("E", -1)))
        if (
            group["case_id"].startswith("topk_gate.release.main.")
            and group["measurement_level"] == "L1_kernel_body"
            and group["variant"] in ("cuda_naive", *CANDIDATES)
            and shape in wanted
        ):
            p50 = group["all_samples_p50_us"]
            representative.append(
                {
                    "T": shape[0],
                    "E": shape[1],
                    "variant": group["variant"],
                    "p50_us": p50,
                    "p90_us": percentile(
                        raw_samples[
                            (
                                group["case_id"],
                                group["variant"],
                                group["measurement_level"],
                                group["cache_mode"],
                            )
                        ],
                        0.90,
                    ),
                    "p95_us": group["all_samples_p95_us"],
                    "cv": group["all_samples_cv"],
                    "rows_per_second": shape[0] * 1.0e6 / p50,
                    "effective_logical_gbps": group["work"]["logical_bytes"] / p50 / 1000.0,
                }
            )
    representative.sort(key=lambda row: (row["T"], row["E"], row["variant"]))

    result = {
        "schema_version": "raggedroute.topk_promotion.v1",
        "source_suite": aggregate.get("suite_id"),
        "records": sum(group["records"] for group in groups),
        "aggregate_groups": len(groups),
        "gpu_uuids": sorted({group["environment"]["gpu_uuid"] for group in groups}),
        "correctness": "all benchmark post-measurement validations passed",
        "promotion_policy": {
            "ratio_of_sums_p50_speedup": 1.05,
            "max_single_shape_p50_regression": 0.03,
            "max_single_shape_p95_regression": 0.05,
            "max_all_samples_cv": 0.10,
            "minimum_process_runs": 5,
            "workspace_bytes": 0,
            "launch_count": 1,
        },
        "intervals": intervals,
        "legacy_bucket_promotions": promoted,
        "shape_dispatch": shape_dispatch,
        "promoted_intervals": shape_dispatch["promoted_intervals"],
        "auto_policy": "shape_dispatched_v4" if shape_dispatch["promoted_intervals"] else "cuda_naive",
        "external_baseline_comparisons": external,
        "representative_release_metrics": representative,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "promotion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "representative_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=representative[0].keys())
        writer.writeheader()
        writer.writerows(representative)

    lines = [
        "# RTX 3080 Top-K Gate Release evidence",
        "",
        "> Latency and speedup below come from unprofiled Release benchmarks. Nsight duration is diagnostic only.",
        "",
        f"- Raw records: {result['records']}; aggregate groups: {len(groups)}; independent process runs/group: 5.",
        f"- GPU UUIDs: {', '.join(result['gpu_uuids'])}.",
        f"- Auto decision: **{result['auto_policy']}**; promoted intervals: {len(shape_dispatch['promoted_intervals'])}.",
        "",
        "## Promotion result",
        "",
        (
            "V4 passed the shape-specific in-tree and strong-library envelope gates."
            if shape_dispatch["promoted_intervals"]
            else "V4 passed no shape-specific interval at both L1/L2 and the strong-library envelope; Auto remains `cuda_naive`."
        ),
        "",
        "| Level | Candidate | Bucket | Best T min | Ratio-of-sums | Max p50 regression | Max p95 regression | Max CV | Passed |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in intervals:
        best = item["best_ratio_attempt"]
        lines.append(
            f"| {item['measurement_level']} | `{item['candidate']}` | {item['bucket']} | "
            f"{best['T_min']} | {best['ratio_of_sums_p50_speedup']:.3f} | "
            f"{best['max_single_shape_p50_regression']:.3f} | "
            f"{best['max_single_shape_p95_regression']:.3f} | {best['max_all_samples_cv']:.3f} | no |"
        )
    lines.extend(
        [
            "",
            "## Strict external pairing",
            "",
            "| Candidate | Denominator | Shapes | Ratio-of-sums | Max regression | Overall claim |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for item in external:
        lines.append(
            f"| `{item['candidate']}` | `{item['denominator']}` | {item['paired_shapes']} | "
            f"{item['ratio_of_sums_speedup']:.3f} | {item['maximum_single_shape_regression']:.3f} | "
            f"{'yes' if item['overall_claim_allowed'] else 'mixed'} |"
        )
    lines.extend(
        [
            "",
            "## Representative L1 metrics",
            "",
            "| T | E | Variant | p50 us | p90 us | p95 us | CV | Mrows/s | Effective GB/s |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in representative:
        lines.append(
            f"| {row['T']} | {row['E']} | `{row['variant']}` | {row['p50_us']:.4f} | "
            f"{row['p90_us']:.4f} | {row['p95_us']:.4f} | {row['cv']:.3f} | "
            f"{row['rows_per_second'] / 1.0e6:.3f} | {row['effective_logical_gbps']:.3f} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
