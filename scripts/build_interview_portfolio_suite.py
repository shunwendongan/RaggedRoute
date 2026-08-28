#!/usr/bin/env python3
"""Build the deterministic seven-operator interview portfolio Release suite."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def cloned_cases(relative: str) -> list[dict]:
    return copy.deepcopy(load(relative)["cases"])


def set_repeats(cases: list[dict], repeats: int) -> None:
    for case in cases:
        common = dict(case.get("common", {}))
        common["kernel_repeats"] = repeats
        case["common"] = common


def variants(names: list[str], baseline: str) -> list[dict]:
    return [
        {"name": name, "promotion_baseline": name == baseline}
        for name in names
    ]


def build_suite() -> dict:
    cases: list[dict] = []

    dense = cloned_cases(
        "configs/operators/dense_gemm/benchmark/candidate_v3_release.json"
    )
    for case in dense:
        case["levels"] = ["l2"]
        case["variants"] = variants(
            ["cublaslt", "cublas", "cuda_register_tiled_v3_64x32_async"],
            "cublaslt",
        )
    set_repeats(dense, 10)
    cases.extend(dense)

    topk = cloned_cases(
        "configs/operators/topk_gate/benchmark/v4_preflight.json"
    )
    set_repeats(topk, 100)
    cases.extend(topk)

    histogram = cloned_cases(
        "configs/operators/histogram/benchmark/candidate_v2_final_l2.json"
    )
    set_repeats(histogram, 20)
    cases.extend(histogram)

    scan = cloned_cases("configs/operators/scan/benchmark/library_release.json")
    set_repeats(scan, 100)
    cases.extend(scan)

    permute = cloned_cases(
        "configs/operators/permute/benchmark/candidate_v3_release.json"
    )
    set_repeats(permute, 100)
    cases.extend(permute)

    permute_library = cloned_cases(
        "configs/operators/permute/benchmark/candidate_v3_library_release.json"
    )
    set_repeats(permute_library, 50)
    cases.extend(permute_library)

    grouped = cloned_cases(
        "configs/operators/grouped_gemm/benchmark/candidate_v3_release.json"
    )
    for case in grouped:
        case["variants"] = variants(
            [
                "cutlass_grouped",
                "cublas_per_expert",
                "cuda_grouped_sm86_fp32_v1",
                "cuda_grouped_sm86_fp32_v2",
                "cuda_grouped_sm86_fp32_v3",
                "cuda_grouped_sm86_fp32_v4a_desc_queue_t1024",
            ],
            "cutlass_grouped",
        )
    set_repeats(grouped, 50)
    cases.extend(grouped)

    unpermute = [
        case
        for case in cloned_cases(
            "configs/operators/unpermute/benchmark/candidate_release.json"
        )
        if case.get("cache_mode", "warm") == "warm"
    ]
    for case in unpermute:
        case["levels"] = ["l2"]
    set_repeats(unpermute, 50)
    cases.extend(unpermute)

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("portfolio source case IDs are not unique")
    return {
        "schema_version": "raggedroute.suite.v2",
        "suite_id": "rtx3080_interview_portfolio_release_v1",
        "protocol": "release",
        "process_runs": 5,
        "common": {
            "warmup": 20,
            "samples": 30,
            "kernel_repeats": 20,
            "seed": 20260828,
            "cache_mode": "warm",
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(build_suite(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
