#!/usr/bin/env python3
"""Reproducible bounded Tensor Core screening, frozen release runs and analysis.

Examples (run from a clean, committed Release build):
  python scripts/tensorcore_experiment.py run --binary out/build/rtx3080-sm86-release/raggedroute_benchmark.exe --config configs/project/benchmark/tensorcore/screening.json --output out/tensorcore/screening.jsonl
  python scripts/tensorcore_experiment.py select --config configs/project/benchmark/tensorcore/screening.json --input out/tensorcore/screening.jsonl --output out/tensorcore/frozen.json
  python scripts/tensorcore_experiment.py run --binary out/build/rtx3080-sm86-release/raggedroute_benchmark.exe --config configs/project/benchmark/tensorcore/full.json --selection out/tensorcore/frozen.json --output out/tensorcore/full.jsonl
  python scripts/tensorcore_experiment.py analyze --config configs/project/benchmark/tensorcore/full.json --selection out/tensorcore/frozen.json --input out/tensorcore/full.jsonl --output-dir out/tensorcore/analysis

Only an explicitly declared library's literal ``unsupported_baseline:`` failure
may continue. All other process, correctness, provenance and completeness errors
abort. Unsupported entries have no synthetic timing or validation result.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import itertools
import json
import math
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any

import run_benchmarks as runner


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/project/benchmark/tensorcore"
SCHEMA = "raggedroute.tensorcore.experiment.v1"
OPERATORS = ("dense_gemm_tc", "grouped_gemm_tc")
LIBRARIES = {"dense_gemm_tc": ("cublas", "cublaslt"),
             "grouped_gemm_tc": ("cublas", "cutlass_grouped")}
DTYPES = ("fp16", "bf16")
ERROR_FIELDS = (
    "quantized_reference_max_abs_error", "quantized_reference_max_rel_error",
    "quantized_reference_normalized_l2_error", "original_fp32_reference_max_abs_error",
    "original_fp32_reference_max_rel_error", "original_fp32_reference_normalized_l2_error",
)
PROMOTION_POLICY = {
    "minimum_ratio_of_sums_p50_speedup": 1.03,
    "minimum_speedup_shape_coverage": 0.80,
    "maximum_single_shape_p50_regression_fraction": 0.05,
    "maximum_single_shape_p95_regression_fraction": 0.03,
    "maximum_all_samples_cv": 0.50,
    "maximum_workspace_growth_bytes": 0,
    "minimum_independent_process_runs": 5,
    "require_same_math_semantics": True,
}


def write_json(path: pathlib.Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: expected nonempty JSONL records")
    return records


def configurations(operator: str) -> list[dict[str, Any]]:
    if operator not in OPERATORS:
        raise ValueError(f"unknown Tensor Core operator {operator}")
    result = []
    shapes = ((64, 32), (64, 64)) if operator == "dense_gemm_tc" else ((16, 64), (32, 64))
    for (m, n), k, asynchronous in itertools.product(shapes, (32, 64), (False, True)):
        result.append({"tile_m": m, "tile_n": n, "tile_k": k,
                       "asynchronous": asynchronous})
    return result


def configuration_id(config: dict[str, Any]) -> str:
    return (f"m{config['tile_m']}_n{config['tile_n']}_k{config['tile_k']}_"
            + ("async" if config["asynchronous"] else "sync"))


def native_variant(config: dict[str, Any]) -> str:
    return "cuda_mma_async" if config["asynchronous"] else "cuda_mma_sync"


def generate_configs(directory: pathlib.Path) -> None:
    dense_sources = [ROOT / "configs/operators/dense_gemm/benchmark/candidate_v3_release.json",
                     ROOT / "configs/project/benchmark/rtx3080_dense_portfolio_region.json"]
    grouped_source = ROOT / "configs/operators/grouped_gemm/benchmark/candidate_v10_release.json"
    dense_shapes, seen = [], set()
    for source in dense_sources:
        suite = runner.load_suite(source)
        for case in suite["cases"]:
            params = case["params"]
            key = tuple(params[name] for name in ("M", "N", "K"))
            if key not in seen:
                seen.add(key)
                dense_shapes.append({"id": f"dense.m{key[0]}_n{key[1]}_k{key[2]}",
                                     "operator": "dense_gemm_tc", "params": params,
                                     "seed": suite["common"]["seed"]})
    grouped_suite = runner.load_suite(grouped_source)
    grouped_shapes = [{"id": case["id"].replace("grouped.v10.", "grouped."),
                       "operator": "grouped_gemm_tc", "params": case["params"],
                       "seed": grouped_suite["common"]["seed"]}
                      for case in grouped_suite["cases"]]
    if len(dense_shapes) != 32 or len(grouped_shapes) != 15:
        raise ValueError("source matrices changed: expected32 unique dense and15 grouped cases")
    screen_shapes = [shape for shape in dense_shapes if shape["params"] in (
        {"M": 1024, "N": 1024, "K": 1024}, {"M": 2048, "N": 64, "K": 128})]
    for distribution, n in itertools.product(("uniform", "zipf"), (64, 128)):
        screen_shapes.append({"id": f"grouped.{distribution}.t2048_e64_k128_n{n}",
            "operator": "grouped_gemm_tc", "seed": grouped_suite["common"]["seed"],
            "params": {"T": 2048, "E": 64, "top_k": 2, "K": 128, "N": n,
                       "distribution": distribution, "zipf_s": 1.4 if distribution == "zipf" else 0.0}})
    shared = {"schema_version": SCHEMA, "protocol": "release", "process_runs": 5,
              "dtypes": list(DTYPES),
              "common": {"warmup": 20, "samples": 30, "cache_mode": "warm", "seed": 20260926},
              "kernel_repeats": {"dense_gemm_tc": 30, "grouped_gemm_tc": 50},
              "configurations": {operator: configurations(operator) for operator in OPERATORS},
              "source_matrices": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                   "sha256": runner.file_sha256(path)}
                                  for path in [*dense_sources, grouped_source]],
              "promotion_policy": PROMOTION_POLICY}
    write_json(directory / "screening.json", shared | {
        "stage": "screening", "suite_id": "sm86_tensorcore_bounded_screening_v1",
        "selection_rule": "lowest equal-shape geometric mean of median-of-five-process p50; configuration_id breaks exact ties",
        "shapes": screen_shapes})
    write_json(directory / "full.json", shared | {
        "stage": "full", "suite_id": "sm86_tensorcore_frozen_full_v1",
        "variants": {operator: ["selected_native", "selected_convert", "fp32_parent", *LIBRARIES[operator]]
                     for operator in OPERATORS},
        "shapes": dense_shapes + grouped_shapes})


def load_config(path: pathlib.Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("stage") not in ("screening", "full"):
        raise ValueError("unsupported Tensor Core experiment schema/stage")
    if config.get("protocol") != "release" or config.get("process_runs") != 5:
        raise ValueError("Tensor Core experiments require release protocol and five processes")
    common = config.get("common", {})
    if any(common.get(key) != expected for key, expected in
           (("warmup", 20), ("samples", 30), ("cache_mode", "warm"))):
        raise ValueError("experiment protocol requires20 warmup,30 samples,warm cache")
    if tuple(config.get("dtypes", [])) != DTYPES:
        raise ValueError("experiment must independently cover FP16 and BF16")
    shapes = config.get("shapes", [])
    if len(shapes) != (6 if config["stage"] == "screening" else 47):
        raise ValueError("expected6 screening or47 full shapes")
    if len({shape["id"] for shape in shapes}) != len(shapes):
        raise ValueError("shape ids must be unique")
    for operator in OPERATORS:
        if config.get("configurations", {}).get(operator) != configurations(operator):
            raise ValueError(f"{operator}: configuration matrix differs from the bounded eight")
        if config.get("kernel_repeats", {}).get(operator) != (30 if operator == OPERATORS[0] else 50):
            raise ValueError(f"{operator}: unexpected kernel repeats")
        expected_shapes = ((2 if operator == OPERATORS[0] else 4) if config["stage"] == "screening"
                           else (32 if operator == OPERATORS[0] else 15))
        if sum(shape["operator"] == operator for shape in shapes) != expected_shapes:
            raise ValueError(f"{operator}: wrong shape count")
    for source in config.get("source_matrices", []):
        if runner.file_sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"source matrix changed: {source['path']}")
    return config


def load_selection(path: pathlib.Path, config: dict[str, Any]) -> dict[str, Any]:
    selection = read_json(path)
    if selection.get("schema_version") != "raggedroute.tensorcore.selection.v1":
        raise ValueError("invalid frozen selection schema")
    for operator, dtype in itertools.product(OPERATORS, DTYPES):
        key = f"{operator}/{dtype}"
        selected = selection.get("selections", {}).get(key, {})
        if selected.get("config") not in config["configurations"][operator]:
            raise ValueError(f"invalid or missing frozen choice for {key}")
    return selection


def work_items(config: dict[str, Any], selection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    items = []
    for shape, dtype in itertools.product(config["shapes"], DTYPES):
        operator = shape["operator"]
        base = {"shape_id": shape["id"], "operator": operator, "dtype": dtype}
        if config["stage"] == "screening":
            for tile in config["configurations"][operator]:
                case_id = f"{shape['id']}.{dtype}.{configuration_id(tile)}"
                items.append(base | {"case_id": case_id, "variant": native_variant(tile), "view": "screening",
                                     "config": tile, "shape": shape})
        else:
            if selection is None:
                raise ValueError("full runs require --selection from completed screening")
            tile = selection["selections"][f"{operator}/{dtype}"]["config"]
            for view in config["variants"][operator]:
                variant = (native_variant(tile) if view == "selected_native" else
                           native_variant(tile) + "_convert" if view == "selected_convert" else view)
                items.append(base | {"case_id": f"{shape['id']}.{dtype}", "variant": variant,
                                     "view": view, "config": tile, "shape": shape})
    return items


def command_case(config: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    tile = item["config"]
    return {"id": item["case_id"], "operator": item["operator"], "levels": ["l2"],
            "common": {"seed": item["shape"]["seed"],
                       "kernel_repeats": config["kernel_repeats"][item["operator"]]},
            "params": item["shape"]["params"] | {"dtype": item["dtype"]} |
                      {name: tile[name] for name in ("tile_m", "tile_n", "tile_k")}}


def is_unsupported_baseline(item: dict[str, Any], completed: subprocess.CompletedProcess[str]) -> bool:
    return (completed.returncode != 0 and item["variant"] in LIBRARIES[item["operator"]]
            and "unsupported_baseline:" in (completed.stdout or ""))


def validate_record(record: dict[str, Any], item: dict[str, Any], process: int,
                    sha: str, run_id: str) -> None:
    env = record.get("environment", {})
    if (record.get("case_id") != item["case_id"] or record.get("variant") != item["variant"]
            or record.get("process_run") != process or record.get("protocol") != "release"
            or record.get("run_id") != run_id):
        raise ValueError("benchmark record does not match the requested command")
    build_sha = env.get("build_git_sha", "")
    if (not isinstance(build_sha, str) or not build_sha or not sha.startswith(build_sha)
            or str(env.get("build_git_dirty")) != "false" or env.get("build_type") != "Release"):
        raise ValueError("benchmark record has dirty/missing/mismatched Release build provenance")
    if not record.get("validation", {}).get("ok"):
        raise ValueError("benchmark correctness validation failed")
    for field in ("batch_mean_us_p50", "batch_mean_us_p95"):
        number = record.get("timing", {}).get(field)
        if not isinstance(number, (float, int)) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"invalid timing.{field}")


def run_experiment(args: argparse.Namespace) -> None:
    config = load_config(args.config.resolve())
    selection = load_selection(args.selection, config) if args.selection else None
    items = work_items(config, selection)
    binary, output = args.binary.resolve(), args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to append/overwrite an existing experiment: {output}")
    if not args.dry_run and not binary.is_file():
        raise FileNotFoundError(binary)
    sha, dirty, branch = runner.git_state(ROOT)
    # Use the existing release gate, plus the stricter fixed five-process contract.
    runner.validate_release(config | {"cases": [{"id": item["case_id"], "levels": ["l2"]}
                                               for item in items]}, dirty)
    if selection and selection.get("repo_commit") != sha:
        raise ValueError("frozen screening selection was produced from a different source commit")
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + config["suite_id"]
    manifest = {"schema_version": "raggedroute.tensorcore.run_manifest.v1", "stage": config["stage"],
                "run_id": run_id, "repo_commit": sha, "repo_dirty": dirty, "repo_branch": branch,
                "binary": str(binary), "binary_sha256": runner.file_sha256(binary),
                "config": config, "config_path": str(args.config.resolve()),
                "config_sha256": runner.file_sha256(args.config.resolve()),
                "selection": selection, "selection_sha256": runner.file_sha256(args.selection) if args.selection else None,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "gpu_snapshot_start": runner.command_output(["nvidia-smi"], ROOT),
                "commands": [], "status": "running"}
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / (output.stem + "-process-records")
    if not args.dry_run:
        scratch.mkdir(exist_ok=False)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    work = [(process, item) for process in range(1, 6) for item in items]
    # Pairing uses the same seeds; shuffle order within each independent-process round.
    ordered = []
    for process in range(1, 6):
        current = [(p, item) for p, item in work if p == process]
        random.Random(config["common"]["seed"] + process).shuffle(current)
        ordered.extend(current)
    try:
        for index, (process, item) in enumerate(ordered, 1):
            record_path = scratch / f"{index:05d}.jsonl"
            case = command_case(config, item)
            command = runner.case_command(binary, config, case, "l2", process, run_id,
                                          record_path, sha, {"name": item["variant"]})
            command_entry = {"index": index, "process_run": process, "case_id": item["case_id"],
                             "variant": item["variant"], "command": command}
            manifest["commands"].append(command_entry)
            if args.dry_run:
                continue
            print(f"[{index}/{len(ordered)}] process{process} {item['case_id']} {item['variant']}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, check=False)
            command_entry["exit_code"] = completed.returncode
            command_entry["output"] = completed.stdout
            (scratch / f"{index:05d}.log").write_text(completed.stdout or "", encoding="utf-8")
            metadata = {"stage": config["stage"], "shape_id": item["shape_id"],
                        "input_dtype": item["dtype"], "view": item["view"],
                        "configuration": item["config"], "configuration_id": configuration_id(item["config"])}
            if is_unsupported_baseline(item, completed):
                if record_path.exists() and record_path.stat().st_size:
                    raise ValueError("unsupported baseline unexpectedly emitted a benchmark timing record")
                record = {"schema_version": "raggedroute.tensorcore.unsupported.v1",
                          "status": "unsupported_baseline", "case_id": item["case_id"],
                          "operator": item["operator"], "variant": item["variant"],
                          "process_run": process, "run_id": run_id, "protocol": "release",
                          "reason": completed.stdout, "exit_code": completed.returncode,
                          "expected_git_sha": sha}
            else:
                if completed.returncode:
                    raise RuntimeError(f"benchmark failed ({completed.returncode}): {completed.stdout}")
                raw = read_records(record_path)
                if len(raw) != 1:
                    raise ValueError("each benchmark process must emit exactly one result")
                record = raw[0]
                validate_record(record, item, process, sha, run_id)
            record["tensorcore_experiment"] = metadata
            with output.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if not args.dry_run:
            end_sha, end_dirty, _ = runner.git_state(ROOT)
            if end_sha != sha or end_dirty:
                raise ValueError("source commit/worktree changed during the release experiment")
        manifest["status"] = "dry_run" if args.dry_run else "completed"
    except BaseException as error:
        manifest["status"] = "aborted"
        manifest["failure"] = str(error)
        raise
    finally:
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if args.dry_run:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
        else:
            manifest["gpu_snapshot_end"] = runner.command_output(["nvidia-smi"], ROOT)
            write_json(manifest_path, manifest)


def group_complete(records: list[dict[str, Any]], config: dict[str, Any],
                   selection: dict[str, Any] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {(item["case_id"], item["variant"]): item for item in work_items(config, selection)}
    members: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record.get("case_id"), record.get("variant"))
        if key not in expected:
            raise ValueError(f"unexpected experiment record: {key}")
        members[key].append(record)
    if set(members) != set(expected):
        raise ValueError(f"missing experiment cells: {sorted(set(expected) - set(members))}")
    result = {}
    provenance = set()
    for key, item in expected.items():
        current = members[key]
        if sorted(record.get("process_run") for record in current) != [1, 2, 3, 4, 5]:
            raise ValueError(f"{key}: expected exactly five distinct process runs")
        for record in current:
            meta = record.get("tensorcore_experiment", {})
            for field, value in (("stage", config["stage"]), ("shape_id", item["shape_id"]),
                                 ("input_dtype", item["dtype"]), ("view", item["view"]),
                                 ("configuration", item["config"])):
                if meta.get(field) != value:
                    raise ValueError(f"{key}: experiment metadata mismatch: {field}")
        unsupported = [record.get("status") == "unsupported_baseline" for record in current]
        if any(unsupported):
            if not all(unsupported) or item["variant"] not in LIBRARIES[item["operator"]]:
                raise ValueError(f"{key}: inconsistent/invalid unsupported baseline status")
            if any("unsupported_baseline:" not in record.get("reason", "")
                   or record.get("protocol") != "release" or record.get("timing") is not None
                   for record in current):
                raise ValueError(f"{key}: unsupported record lacks an explicit reason or contains timing")
            result[key] = {"status": "unsupported_baseline", "item": item, "records": current}
            continue
        for record in current:
            env = record.get("environment", {})
            sha = env.get("build_git_sha", "")
            validate_record(record, item, record["process_run"], sha, record.get("run_id"))
            if not env.get("gpu_uuid"):
                raise ValueError(f"{key}: missing GPU UUID")
            provenance.add((sha, env["gpu_uuid"], record["run_id"]))
            if (record.get("warmup") != 20 or record.get("samples") != 30
                    or record.get("kernel_repeats") != config["kernel_repeats"][item["operator"]]
                    or record.get("seed") != item["shape"]["seed"]
                    or record.get("cache_mode") != "warm"
                    or record.get("measurement_level") != "L2_operator_steady"):
                raise ValueError(f"{key}: timing boundary/protocol differs from experiment config")
            cv = record.get("timing", {}).get("cv")
            if not isinstance(cv, (float, int)) or not math.isfinite(cv) or cv < 0.0:
                raise ValueError(f"{key}: invalid timing.cv")
            raw_samples = record["timing"].get("raw_batch_mean_samples_us", [])
            if len(raw_samples) != 30 or any(not isinstance(value, (float, int)) or
                                            not math.isfinite(value) or value <= 0.0
                                            for value in raw_samples):
                raise ValueError(f"{key}: expected30 finite positive raw timing samples")
        case_configs = {json.dumps(record.get("case_config", {}), sort_keys=True) for record in current}
        modes = {record.get("variant_config", {}).get("math_mode") for record in current}
        if len(case_configs) != 1 or len(modes) != 1:
            raise ValueError(f"{key}: inconsistent case/numerical contract across processes")
        actual_dtype = "fp32" if item["view"] == "fp32_parent" else item["dtype"]
        expected_mode = "strict_fp32" if item["view"] == "fp32_parent" else item["dtype"] + "_acc_fp32"
        case_config = current[0]["case_config"]
        if (case_config.get("dtype") != actual_dtype or case_config.get("output_dtype") != "fp32"
                or case_config.get("accumulator_dtype") != "fp32" or modes != {expected_mode}):
            raise ValueError(f"{key}: unexpected numerical contract")
        if any(case_config.get(name) != value for name, value in item["shape"]["params"].items()):
            raise ValueError(f"{key}: recorded shape differs from predeclared shape")
        if item["variant"].startswith("cuda_mma_"):
            for record in current:
                if any(record["variant_config"].get(name) != item["config"][name]
                       for name in ("tile_m", "tile_n", "tile_k")):
                    raise ValueError(f"{key}: recorded tile differs from frozen configuration")
        process_p50 = {record["process_run"]: record["timing"]["batch_mean_us_p50"] for record in current}
        result[key] = {"status": "collected", "item": item, "records": current,
                       "case_config": case_config, "math_mode": expected_mode,
                       "process_p50": process_p50,
                       "p50": statistics.median(process_p50.values()),
                       "p95": statistics.median(record["timing"]["batch_mean_us_p95"] for record in current),
                       "cv": max(float(record["timing"]["cv"]) for record in current),
                       "workspace_bytes": max(int(record.get("workspace_bytes", 0)) for record in current),
                       "errors": {field: max(float(record["variant_config"][field]) for record in current)
                                  for field in ERROR_FIELDS}}
    if len(provenance) != 1:
        raise ValueError("experiment mixes source commits, GPU UUIDs or run_ids")
    build_sha, gpu_uuid, run_id = next(iter(provenance))
    for group in result.values():
        if group["status"] == "unsupported_baseline" and any(
                record.get("run_id") != run_id or not record.get("expected_git_sha", "").startswith(build_sha)
                for record in group["records"]):
            raise ValueError("unsupported baseline provenance differs from collected results")
    if selection is not None:
        if not selection.get("repo_commit", "").startswith(build_sha):
            raise ValueError("full release build differs from frozen screening source commit")
        if selection.get("gpu_uuid") != gpu_uuid:
            raise ValueError("full release GPU differs from the frozen screening GPU")
    return result


def select_configurations(config: dict[str, Any], records: list[dict[str, Any]],
                          input_path: pathlib.Path) -> dict[str, Any]:
    if config["stage"] != "screening":
        raise ValueError("selection requires the screening config")
    groups = group_complete(records, config)
    rankings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for operator, dtype in itertools.product(OPERATORS, DTYPES):
        for tile in config["configurations"][operator]:
            relevant = [group for group in groups.values() if group["item"]["operator"] == operator
                        and group["item"]["dtype"] == dtype and group["item"]["config"] == tile]
            if any(not math.isfinite(group["cv"]) or group["cv"] > 0.50 for group in relevant):
                raise ValueError("screening CV exceeds0.50; preserve samples and rerun screening as a new experiment")
            score = math.exp(statistics.fmean(math.log(group["p50"]) for group in relevant))
            rankings[f"{operator}/{dtype}"].append({"config": tile,
                "configuration_id": configuration_id(tile), "geomean_p50_us": score,
                "maximum_process_cv": max(group["cv"] for group in relevant),
                "shape_latencies_us": {group["item"]["shape_id"]: group["p50"] for group in relevant}})
    selections = {}
    for key, rows in rankings.items():
        rows.sort(key=lambda row: (row["geomean_p50_us"], row["configuration_id"]))
        selections[key] = rows[0]
    first = records[0]
    short_sha = first["environment"]["build_git_sha"]
    full_sha, _, _ = runner.git_state(ROOT)
    if not full_sha.startswith(short_sha):
        raise ValueError("current source commit differs from screening build")
    return {"schema_version": "raggedroute.tensorcore.selection.v1", "repo_commit": full_sha,
            "screening_input": str(input_path.resolve()), "screening_sha256": runner.file_sha256(input_path),
            "screening_run_id": first["run_id"], "gpu_uuid": first["environment"]["gpu_uuid"],
            "selection_rule": config["selection_rule"], "selections": selections, "rankings": rankings}


def compare(case_id: str, baseline: dict[str, Any], candidate: dict[str, Any],
            view: str, *, same_precision: bool) -> dict[str, Any]:
    if same_precision and (baseline["case_config"] != candidate["case_config"]
                           or baseline["math_mode"] != candidate["math_mode"]):
        raise ValueError(f"{case_id}: same-precision comparison has different case/math contracts")
    process_ratios = {process: baseline["process_p50"][process] / candidate["process_p50"][process]
                      for process in range(1, 6)}
    item = candidate["item"]
    row = {"case_id": case_id, "shape_id": item["shape_id"], "operator": item["operator"],
           "dtype": item["dtype"], "comparison": view,
           "same_precision": same_precision, "baseline": baseline["item"]["variant"],
           "candidate": item["variant"], "configuration_id": configuration_id(item["config"]),
           "baseline_p50_us": baseline["p50"], "candidate_p50_us": candidate["p50"],
           "baseline_p95_us": baseline["p95"], "candidate_p95_us": candidate["p95"],
           "p50_speedup": baseline["p50"] / candidate["p50"],
           "p95_speedup": baseline["p95"] / candidate["p95"],
           "p50_regression_fraction": candidate["p50"] / baseline["p50"] - 1.0,
           "p95_regression_fraction": candidate["p95"] / baseline["p95"] - 1.0,
           "candidate_faster_processes": sum(value > 1.0 for value in process_ratios.values()),
           "paired_processes": 5, "baseline_max_cv": baseline["cv"], "candidate_max_cv": candidate["cv"],
           "cv_risk_above_0_10": max(baseline["cv"], candidate["cv"]) > 0.10,
           "evidence_insufficient_cv_above_0_50": max(baseline["cv"], candidate["cv"]) > 0.50,
           "baseline_workspace_bytes": baseline["workspace_bytes"],
           "candidate_workspace_bytes": candidate["workspace_bytes"],
           "workspace_growth_bytes": candidate["workspace_bytes"] - baseline["workspace_bytes"]}
    row.update({f"process{process}_speedup": value for process, value in process_ratios.items()})
    row.update(candidate["errors"])
    row.update({name: value for name, value in item["shape"]["params"].items()})
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [row["p50_speedup"] for row in rows]
    result = {"paired_case_dtype_combinations": len(rows),
              "ratio_of_sums_p50_speedup": sum(row["baseline_p50_us"] for row in rows) /
                                           sum(row["candidate_p50_us"] for row in rows),
              "geometric_mean_p50_speedup": math.exp(statistics.fmean(math.log(value) for value in ratios)),
              "shape_win_fraction": sum(value > 1.0 for value in ratios) / len(rows),
              "maximum_p50_regression_fraction": max(row["p50_regression_fraction"] for row in rows),
              "maximum_p95_regression_fraction": max(row["p95_regression_fraction"] for row in rows),
              "candidate_faster_process_pairs": sum(row["candidate_faster_processes"] for row in rows),
              "paired_process_pairs": len(rows) * 5,
              "all_five_processes_faster_shapes": sum(row["candidate_faster_processes"] == 5 for row in rows),
              "cv_above_0_10_shapes": [row["case_id"] for row in rows if row["cv_risk_above_0_10"]],
              "cv_above_0_50_shapes": [row["case_id"] for row in rows if row["evidence_insufficient_cv_above_0_50"]],
              "maximum_workspace_growth_bytes": max(row["workspace_growth_bytes"] for row in rows)}
    passed = (result["ratio_of_sums_p50_speedup"] >= 1.03 and result["shape_win_fraction"] >= 0.80
              and result["maximum_p50_regression_fraction"] <= 0.05
              and result["maximum_p95_regression_fraction"] <= 0.03
              and result["maximum_workspace_growth_bytes"] <= 0)
    result["performance_thresholds_passed"] = passed
    result["decision"] = ("insufficient_evidence" if result["cv_above_0_50_shapes"] else
                           "thresholds_passed" if passed else "thresholds_not_met")
    result["public_dispatch_promotion"] = "not_requested_benchmark_only"
    return result


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(config: dict[str, Any], selection: dict[str, Any], records: list[dict[str, Any]],
            output: pathlib.Path) -> None:
    if config["stage"] != "full":
        raise ValueError("analysis requires the full experiment config")
    groups = group_complete(records, config, selection)
    output.mkdir(parents=True, exist_ok=False)
    rows, unsupported, variant_rows = [], [], []
    for group in groups.values():
        item = group["item"]
        if group["status"] == "unsupported_baseline":
            unsupported.append({"case_id": item["case_id"], "variant": item["variant"],
                                "reason": group["records"][0]["reason"]})
        else:
            variant_rows.append({"case_id": item["case_id"], "shape_id": item["shape_id"],
                "operator": item["operator"], "dtype": item["dtype"], "view": item["view"],
                "variant": item["variant"], "p50_us": group["p50"], "p95_us": group["p95"],
                "max_cv": group["cv"], "workspace_bytes": group["workspace_bytes"],
                "actual_path": group["records"][0]["variant_config"]["actual_path"],
                "configuration_id": configuration_id(item["config"])} | group["errors"])
    for shape, dtype in itertools.product(config["shapes"], DTYPES):
        case_id = f"{shape['id']}.{dtype}"
        tile = selection["selections"][f"{shape['operator']}/{dtype}"]["config"]
        native = groups[(case_id, native_variant(tile))]
        converted = groups[(case_id, native_variant(tile) + "_convert")]
        parent = groups[(case_id, "fp32_parent")]
        libraries = [groups[(case_id, variant)] for variant in LIBRARIES[shape["operator"]]
                     if groups[(case_id, variant)]["status"] == "collected"]
        if not libraries:
            raise ValueError(f"{case_id}: no supported same-precision library baseline")
        envelope = min(libraries, key=lambda group: (group["p50"], group["item"]["variant"]))
        rows.extend((compare(case_id, envelope, native, "same_precision_library_envelope", same_precision=True),
                     compare(case_id, parent, native, "native_vs_strict_fp32_parent", same_precision=False),
                     compare(case_id, parent, converted, "conversion_plus_mma_vs_strict_fp32_parent", same_precision=False)))
    summaries = {}
    for operator, dtype in itertools.product(OPERATORS, DTYPES):
        for view in ("same_precision_library_envelope", "native_vs_strict_fp32_parent",
                     "conversion_plus_mma_vs_strict_fp32_parent"):
            current = [row for row in rows if row["operator"] == operator and row["dtype"] == dtype
                       and row["comparison"] == view]
            summaries[f"{operator}/{dtype}/{view}"] = summarize(current)
    report = {"schema_version": "raggedroute.tensorcore.analysis.v1", "selections": selection["selections"],
              "case_dtype_combinations": len(config["shapes"]) * 2,
              "comparison_contract": {"library": "same input/output/accumulation dtype and math mode",
                "parent": "cross-precision path comparison; error versus original FP32 inputs is reported separately",
                "conversion": "both FP32-to-lowp conversions plus GEMM included; buffers preallocated"},
              "promotion_policy": config["promotion_policy"], "summaries": summaries,
              "unsupported_baselines": unsupported, "rows": rows}
    write_json(output / "summary.json", report)
    write_csv(output / "per_shape.csv", rows)
    write_csv(output / "per_variant.csv", variant_rows)
    write_json(output / "unsupported_baselines.json", {"records": unsupported})
    lines = ["# Tensor Core GEMM 正式实验", "", "## 数值与计时合同", "",
             "FP16/BF16 输入，FP32 累加与输出。库比较使用相同精度。与 strict FP32 的比较同时改变输入精度，另列原始输入误差。转换视图包含两输入转换与 GEMM。", "",
             "## 汇总", "", "| 算子/精度/视图 | 总耗时比 | 几何平均 | 获益覆盖 | 最差 p50 回退 | 最差 p95 回退 | 结论 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for key, value in summaries.items():
        lines.append(f"| {key} | {value['ratio_of_sums_p50_speedup']:.4f}× | "
                     f"{value['geometric_mean_p50_speedup']:.4f}× | {value['shape_win_fraction']:.1%} | "
                     f"{value['maximum_p50_regression_fraction']:.1%} | "
                     f"{value['maximum_p95_regression_fraction']:.1%} | {value['decision']} |")
    lines.extend(["", "CV>0.10 的形状在 summary.json 中逐项披露；CV>0.50 时证据不足。全部原始样本保留。",
                  f"明确不支持的库 case–dtype 组合：{len(unsupported)}；详情见 unsupported_baselines.json。",
                  "候选保持 benchmark_only，未修改公共 API 或 Auto 分发。", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    screen, full = load_config(CONFIG_ROOT / "screening.json"), load_config(CONFIG_ROOT / "full.json")
    assert len(work_items(screen)) == 96
    selection = {"repo_commit": "synthetic-test-sha", "gpu_uuid": "synthetic-test-gpu",
                 "selections": {f"{operator}/{dtype}": {"config": configurations(operator)[0]}
                                for operator, dtype in itertools.product(OPERATORS, DTYPES)}}
    assert len(work_items(full, selection)) == 470
    sample = work_items(screen)[0]
    failure = subprocess.CompletedProcess([], 2, "error: unsupported_baseline: shape\n")
    assert not is_unsupported_baseline(sample, failure)
    assert is_unsupported_baseline(sample | {"variant": "cublas"}, failure)
    assert not is_unsupported_baseline(sample | {"variant": "cublas"},
                                       subprocess.CompletedProcess([], 2, "CUDA launch failed"))
    values = {"baseline_p50_us": 100.0, "candidate_p50_us": 80.0, "p50_speedup": 1.25,
              "p50_regression_fraction": -0.2, "p95_regression_fraction": -0.1,
              "candidate_faster_processes": 5, "cv_risk_above_0_10": False,
              "evidence_insufficient_cv_above_0_50": False, "workspace_growth_bytes": 0}
    summary = summarize([values | {"case_id": "synthetic_unit_test"}])
    assert summary["decision"] == "thresholds_passed" and summary["shape_win_fraction"] == 1.0
    synthetic = []
    for item in work_items(full, selection):
        for process in range(1, 6):
            dtype = "fp32" if item["view"] == "fp32_parent" else item["dtype"]
            mode = "strict_fp32" if item["view"] == "fp32_parent" else dtype + "_acc_fp32"
            synthetic.append({"case_id": item["case_id"], "variant": item["variant"],
                "process_run": process, "protocol": "release", "run_id": "synthetic-test",
                "environment": {"build_git_sha": "synthetic-test-sha", "build_git_dirty": "false",
                                "build_type": "Release", "gpu_uuid": "synthetic-test-gpu"},
                "validation": {"ok": True}, "warmup": 20, "samples": 30,
                "kernel_repeats": full["kernel_repeats"][item["operator"]],
                "seed": item["shape"]["seed"], "cache_mode": "warm",
                "measurement_level": "L2_operator_steady", "workspace_bytes": 0,
                "case_config": item["shape"]["params"] | {"dtype": dtype, "output_dtype": "fp32",
                    "accumulator_dtype": "fp32", "input_boundary": "fp32_master" if item["view"] in
                    ("selected_convert", "fp32_parent") else "native_low_precision"},
                "variant_config": {"math_mode": mode, "actual_path": "synthetic_unit_test",
                                   **{name: item["config"][name] for name in ("tile_m", "tile_n", "tile_k")},
                                   **{field: 0.0 for field in ERROR_FIELDS}},
                "timing": {"batch_mean_us_p50": 80.0 if item["view"] == "selected_native" else 100.0,
                           "batch_mean_us_p95": 82.0 if item["view"] == "selected_native" else 102.0,
                           "cv": 0.01, "raw_batch_mean_samples_us": [80.0] * 30},
                "tensorcore_experiment": {"stage": "full", "shape_id": item["shape_id"],
                    "input_dtype": item["dtype"], "view": item["view"], "configuration": item["config"]}})
    grouped = group_complete(synthetic, full, selection)
    assert len(grouped) == 470
    bad = copy.deepcopy(synthetic)
    bad.pop()
    try:
        group_complete(bad, full, selection)
    except ValueError as error:
        assert "five distinct process runs" in str(error)
    else:
        raise AssertionError("incomplete evidence must fail")
    with tempfile.TemporaryDirectory() as temporary:
        destination = pathlib.Path(temporary) / "synthetic-analysis"
        analyze(full, selection, synthetic, destination)
        document = read_json(destination / "summary.json")
        assert document["case_dtype_combinations"] == 94 and len(document["rows"]) == 282
        assert len(document["summaries"]) == 12
    print("Tensor Core experiment self-test passed:8 configs/op/type,96 screening cells,94 full case-dtype combinations,470 full cells; unsupported signature, completeness and282-row analysis checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    generate = commands.add_parser("generate", help="generate committed bounded experiment configs")
    generate.add_argument("--output-dir", type=pathlib.Path, default=CONFIG_ROOT)
    run = commands.add_parser("run", help="execute one complete release stage")
    run.add_argument("--binary", required=True, type=pathlib.Path)
    run.add_argument("--config", required=True, type=pathlib.Path)
    run.add_argument("--selection", type=pathlib.Path)
    run.add_argument("--output", required=True, type=pathlib.Path)
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    select = commands.add_parser("select", help="freeze four choices from complete screening evidence")
    select.add_argument("--config", type=pathlib.Path, default=CONFIG_ROOT / "screening.json")
    select.add_argument("--input", required=True, type=pathlib.Path)
    select.add_argument("--output", required=True, type=pathlib.Path)
    analysis = commands.add_parser("analyze", help="compare full evidence and export CSV/JSON/Markdown")
    analysis.add_argument("--config", type=pathlib.Path, default=CONFIG_ROOT / "full.json")
    analysis.add_argument("--selection", required=True, type=pathlib.Path)
    analysis.add_argument("--input", required=True, type=pathlib.Path)
    analysis.add_argument("--output-dir", required=True, type=pathlib.Path)
    commands.add_parser("self-test", help="validate configs/runner decisions without using the GPU")
    args = parser.parse_args()
    if args.action == "generate":
        generate_configs(args.output_dir)
    elif args.action == "run":
        run_experiment(args)
    elif args.action == "select":
        config = load_config(args.config)
        result = select_configurations(config, read_records(args.input), args.input)
        write_json(args.output, result)
        write_csv(args.output.with_suffix(".rankings.csv"), [
            {"operator_dtype": key, "configuration_id": row["configuration_id"],
             "geomean_p50_us": row["geomean_p50_us"], "maximum_process_cv": row["maximum_process_cv"]}
            for key, rows in result["rankings"].items() for row in rows])
        print(json.dumps(result["selections"], indent=2))
    elif args.action == "analyze":
        config = load_config(args.config)
        analyze(config, load_selection(args.selection, config), read_records(args.input), args.output_dir)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
