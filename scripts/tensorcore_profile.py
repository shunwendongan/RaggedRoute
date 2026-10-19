#!/usr/bin/env python3
"""NSYS then filtered NCU evidence for four frozen Tensor Core paths.

No profiler time is used as release performance. Use --dry-run to generate
commands without querying or launching the GPU. A new output directory is
required; partial reports and complete command output survive a failure.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import re
import sys
from typing import Any

import profile_benchmarks as common

OPERATORS = ("dense_gemm_tc", "grouped_gemm_tc")
DTYPES = ("fp16", "bf16")
WANTED = ("tensor_active_pct", "registers_per_thread", "shared_mem_per_block",
          "waves_per_sm", "occupancy_achieved_pct", "local_load_instructions",
          "local_store_instructions")


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_tool(name: str, explicit: pathlib.Path | None) -> str:
    value = str(explicit.resolve()) if explicit else common.find_tool(name)
    if not value:
        raise FileNotFoundError(f"{name} missing; pass --{name} with its audited absolute path")
    path = pathlib.Path(value).resolve()
    # The Windows ncu launcher is a batch wrapper. Prefer its installed binary
    # so command capture never depends on cmd.exe quoting or another PATH entry.
    if name == "ncu" and path.suffix.lower() in (".bat", ".cmd"):
        native = path.parent / "target/windows-desktop-win7-x64/ncu.exe"
        if native.is_file():
            path = native
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def cases(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("schema_version") != "raggedroute.tensorcore.selection.v1":
        raise ValueError("expected frozen Tensor Core selection v1")
    result = []
    for operator in OPERATORS:
        for dtype in DTYPES:
            tile = selection["selections"][f"{operator}/{dtype}"]["config"]
            if not isinstance(tile["asynchronous"], bool):
                raise ValueError("selection asynchronous must be boolean")
            params = ({"M": 1024, "N": 1024, "K": 1024} if operator == "dense_gemm_tc"
                      else {"T": 2048, "E": 64, "top_k": 2, "K": 128, "N": 128,
                            "distribution": "uniform", "zipf_s": 0.0})
            params.update(dtype=dtype, **{key: tile[key] for key in ("tile_m", "tile_n", "tile_k")})
            result.append({"id": f"{operator}.{dtype}", "operator": operator,
                "variant": "cuda_mma_async" if tile["asynchronous"] else "cuda_mma_sync",
                "level": "l2", "warmup": 20, "seed": 20260926,
                "kernel_pattern": "dense_gemm_mma_kernel" if operator == "dense_gemm_tc"
                                  else "grouped_gemm_mma_kernel", "params": params})
    return result


def report_module(ncu: str, explicit: pathlib.Path | None):
    if explicit:
        directory = explicit.resolve()
    else:
        binary = pathlib.Path(ncu)
        candidates = (binary.parent / "extras/python", binary.parent.parent / "extras/python",
                      binary.parent.parent.parent / "extras/python")
        directory = next((path for path in candidates if (path / "ncu_report.py").is_file()), None)
        if directory is None:
            directory = common.find_ncu_report_path()
    if directory is None or not (directory / "ncu_report.py").is_file():
        raise FileNotFoundError("ncu_report.py missing; pass --ncu-report-python")
    sys.path.insert(0, str(directory))
    return importlib.import_module("ncu_report"), str(directory)


def read_actions(module: Any, report: pathlib.Path, case: dict[str, Any],
                 profile_set: str, supported: set[str]) -> list[dict[str, Any]]:
    actions = list(common.iter_actions(module.load_report(str(report))))
    if len(actions) != 1:
        raise ValueError(f"{case['id']}: expected one filtered NCU launch, found {len(actions)}")
    kernel = common.action_name(actions[0])
    if case["kernel_pattern"] not in kernel:
        raise ValueError(f"{case['id']}: unexpected NCU kernel {kernel}")
    return [{"case_id": case["id"], "profile_set": profile_set, "kernel": kernel,
             "metrics": common.metric_snapshot(actions[0], supported)}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=pathlib.Path)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--ncu", type=pathlib.Path)
    parser.add_argument("--nsys", type=pathlib.Path)
    parser.add_argument("--ncu-report-python", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_cases = cases(selection)
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    ncu, nsys = resolve_tool("ncu", args.ncu), resolve_tool("nsys", args.nsys)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw, reports, analysis = root / "raw", root / "reports", root / "analysis"
    for path in (raw, reports, analysis):
        path.mkdir()
    manifest = {"schema_version": "raggedroute.tensorcore.profile.v1", "created_utc": common.now_utc(),
        "selection": str(args.selection.resolve()), "selection_sha256": common.sha256_file(args.selection),
        "selection_commit": selection["repo_commit"], "binary": str(binary),
        "binary_sha256": common.sha256_file(binary), "tools": {"ncu": ncu, "nsys": nsys},
        "dry_run": args.dry_run, "cases": selected_cases, "commands": [], "complete": False,
        "sass_hmma_evidence": "external_cuobjdump_required"}

    def run(label: str, command: list[str]):
        entry = {"label": label, "command": command}
        manifest["commands"].append(entry)
        write_json(root / "manifest.json", manifest)
        if args.dry_run:
            return None
        outcome = common.run_command(command, capture=True, check=False)
        (raw / f"{label}.stdout.txt").write_text(outcome.stdout, encoding="utf-8")
        (raw / f"{label}.stderr.txt").write_text(outcome.stderr, encoding="utf-8")
        entry["returncode"] = outcome.returncode
        write_json(root / "manifest.json", manifest)
        if outcome.returncode:
            raise RuntimeError(f"{label} failed with exit {outcome.returncode}; see {raw}")
        return outcome

    if not args.dry_run:
        module, module_path = report_module(ncu, args.ncu_report_python)
        manifest["tools"]["ncu_report_python"] = module_path
        run("ncu.version", [ncu, "--version"])
        run("nsys.version", [nsys, "--version"])
    supported: set[str] | None = None
    summaries, all_actions = [], []
    for case in selected_cases:
        case_id = case["id"]
        target = common.target_command(binary, case)
        base = reports / f"{case_id}.system"
        run(f"{case_id}.nsys", [nsys, "profile", "--trace=cuda,nvtx", "--sample=none",
            "--cpuctxsw=none", "--force-overwrite=false", f"--output={base}", *target])
        kernel_csv = analysis / f"{case_id}.nsys_kernels.csv"
        for report_name in common.NSYS_STATS_REPORTS:
            outcome = run(f"{case_id}.nsys.{report_name}", [nsys, "stats", "--force-export=true",
                "--report", report_name, "--format", "csv", str(base) + ".nsys-rep"])
            if outcome is not None:
                output = kernel_csv if report_name == "cuda_gpu_kern_sum" else analysis / f"{case_id}.{report_name}.csv"
                output.write_text(common.extract_csv(outcome.stdout), encoding="utf-8")
        if not args.dry_run:
            kernels = common.rank_nsys_kernels(kernel_csv)
            matched = [row for row in kernels if case["kernel_pattern"] in row["name"]]
            if not matched or any("lowp_simt" in row["name"] for row in kernels):
                raise ValueError(f"{case_id}: NSYS did not confirm the intended MMA kernel")
            write_json(analysis / f"{case_id}.nsys_kernels.json", kernels)
        _, command, basic_report = common.compute_commands(binary, {"cases": [case]}, reports, ncu)[0]
        run(f"{case_id}.ncu.basic", command)
        if args.dry_run:
            continue
        actions = read_actions(module, basic_report, case, "basic", supported or set())
        metrics = actions[0]["metrics"]
        missing = [name for name in WANTED if metrics[name]["status"] != "collected"]
        if missing:
            if supported is None:
                outcome = run("ncu.supported_metrics", [ncu, "--query-metrics", "--query-metrics-mode", "all", "--devices", "0"])
                supported = {line.split(maxsplit=1)[0] for line in outcome.stdout.splitlines() if line.strip()}
                (analysis / "supported_metrics.txt").write_text("\n".join(sorted(supported)) + "\n", encoding="utf-8")
            wanted = [next((metric for metric in common.METRIC_CONCEPTS[name] if metric in supported), None)
                      for name in missing]
            wanted = [metric for metric in wanted if metric]
            if wanted:
                targeted = list(command)
                index = targeted.index("--set")
                targeted[index:index + 2] = ["--metrics", ",".join(wanted)]
                report_base = reports / f"{case_id}.targeted"
                targeted[targeted.index("--export") + 1] = str(report_base)
                run(f"{case_id}.ncu.targeted", targeted)
                actions += read_actions(module, pathlib.Path(str(report_base) + ".ncu-rep"), case,
                                        "targeted", supported)
            actions[0] = read_actions(module, basic_report, case, "basic", supported)[0]
        merged = {name: next((action["metrics"][name] for action in reversed(actions)
            if action["metrics"][name]["status"] == "collected"), actions[-1]["metrics"][name])
            for name in WANTED}
        tensor = merged["tensor_active_pct"]
        value = common.number(str(tensor["value"])) if tensor["status"] == "collected" else None
        verified = value is not None and value > 0.0
        summaries.append({"case_id": case_id, "nsys_mma_kernel_observed": True,
            "ncu_kernel": actions[-1]["kernel"], "tensor_pipe_nonzero_verified": verified,
            "metrics": merged, "sass_hmma_evidence": "external_cuobjdump_required"})
        all_actions.extend(actions)
        write_json(analysis / "ncu_metrics.json", all_actions)
        common.write_ncu_csv(analysis / "ncu_metrics.csv", all_actions)
        write_json(analysis / "summary.json", summaries)
        if not verified:
            raise ValueError(f"{case_id}: nonzero Tensor pipe evidence missing; values/status retained")
    manifest["complete"] = not args.dry_run and len(summaries) == 4
    manifest["profiled_case_count"] = len(summaries)
    write_json(root / "manifest.json", manifest)
    print(root / "manifest.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
