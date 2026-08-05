#!/usr/bin/env python3
"""Package the realistic seven-operator MoE L3 evidence as a traceable report.

Release latency comes only from the three-process, unprofiled JSONL.  NSYS and
NCU artifacts are copied unchanged and used solely to diagnose the steady-state
chain.  The script intentionally treats absent NCU basic metrics as
``not_collected`` instead of assigning a numerical value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import pathlib
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "out/research/l3_realistic_20260805"
REPORT_NAME = "RaggedRoute_L3_realistic_MoE_3way_report.md"
SCENARIOS = (
    ("mixtral_exact", "moe.mixtral_decode_exact.t8_e8_k4096_n14336", "Mixtral 原尺寸单投影 decode"),
    ("chunked_prefill", "moe.chunked_prefill_proxy.t512_e8_k512_n1792", "Chunked prefill 代理"),
    ("finegrained_prefill", "moe.finegrained_prefill_proxy.t1024_e64_k256_n512", "E64 细粒度专家压力"),
)
ALL_CASES = {
    "moe.mixtral_decode_exact.t8_e8_k4096_n14336": "Mixtral 原尺寸单投影 decode",
    "moe.continuous_decode_proxy.t128_e8_k1024_n3584": "连续 decode 代理",
    "moe.chunked_prefill_proxy.t512_e8_k512_n1792": "Chunked prefill 代理",
    "moe.finegrained_prefill_proxy.t1024_e64_k256_n512": "E64 细粒度专家压力",
}
VARIANTS = (
    ("cuda_all_candidates_chain", "CUDA candidate"),
    ("library_all_baselines_chain", "Repository library baseline"),
    ("triton_reference", "Triton reference"),
)
PIPELINES = (
    ("cuda_all_candidates_chain", "CUDA candidate", "cuda_all_candidates_chain_nsys", "cuda_all_candidates_chain_ncu"),
    ("library_all_baselines_chain", "Library baseline", "library_all_baselines_chain_nsys", "library_all_baselines_chain_ncu"),
    ("triton_reference", "Triton reference", "triton_nsys_2026", "triton_reference_ncu"),
)
METRICS = {
    "duration_us": ("gpu__time_duration.sum", "gpu__time_duration.avg"),
    "sm_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "memory_pct": ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",),
    "dram_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    "l1_pct": ("l1tex__throughput.avg.pct_of_peak_sustained_active",),
    "l2_pct": ("lts__throughput.avg.pct_of_peak_sustained_elapsed",),
    "registers_per_thread": ("launch__registers_per_thread",),
    "shared_mem_per_block": ("launch__shared_mem_per_block",),
    "waves_per_sm": ("launch__waves_per_multiprocessor",),
    "theoretical_occupancy_pct": ("sm__maximum_warps_per_active_cycle_pct",),
    "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    "grid_size": ("launch__grid_size",),
    "block_size": ("launch__block_size",),
}


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ncu_action(report: pathlib.Path) -> dict[str, Any]:
    extras = pathlib.Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\extras\python")
    if str(extras) not in sys.path:
        sys.path.insert(0, str(extras))
    import ncu_report  # type: ignore

    context = ncu_report.load_report(str(report))
    actions = []
    for range_index in range(context.num_ranges()):
        current_range = context.range_by_idx(range_index)
        for action_index in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_index)
            if action is not None:
                actions.append(action)
    if len(actions) != 1:
        raise ValueError(f"{report}: expected exactly one NCU action, found {len(actions)}")
    action = actions[0]
    names = set(action.metric_names())
    values: dict[str, dict[str, Any]] = {}
    for concept, candidates in METRICS.items():
        metric_name = next((candidate for candidate in candidates if candidate in names), None)
        if metric_name is None:
            values[concept] = {"status": "not_collected", "metric": None, "value": None, "unit": None}
            continue
        metric = action[metric_name]
        values[concept] = {
            "status": "collected", "metric": metric_name,
            "value": metric.value(), "unit": metric.unit(),
        }
    return {"kernel": str(action.name()), "metrics": values, "report": str(report)}


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_nsys_summary(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        name = row.get("Name", "")
        total = float_or_none(row.get("Total Time (ns)")) or 0.0
        result.append({"name": name, "total_ns": total, "pct": float_or_none(row.get("Time (%)")) or 0.0})
    return sorted(result, key=lambda item: item["total_ns"], reverse=True)


def stage_for_kernel(name: str) -> str:
    lower = name.lower()
    if "grouped" in lower or "cutlass::kernel" in lower:
        return "Grouped GEMM"
    if "dense_gemm" in lower or "gemmsn" in lower:
        return "Dense GEMM"
    if "topk" in lower or "top2" in lower:
        return "Top-K gate"
    if "histogram_exclusive" in lower:
        return "Histogram + scan (fused)"
    if "histogram" in lower:
        return "Histogram"
    if "scan" in lower or "expert_offsets" in lower:
        return "Exclusive scan"
    if "permute" in lower or "expand_rows" in lower or "radix_sort" in lower or "route_map" in lower:
        return "Token permute"
    if "unpermute" in lower or "finalize_routing" in lower:
        return "Unpermute"
    return "Other / runtime"


def trace_events(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    events = []
    for row in rows:
        name = row.get("Name", "")
        start = float_or_none(row.get("Start (ns)"))
        duration = float_or_none(row.get("Duration (ns)"))
        if start is None or duration is None or not row.get("GrdX") or name.startswith("["):
            continue
        events.append({"name": name, "start_ns": start, "duration_ns": duration, "stage": stage_for_kernel(name)})
    return events


def short_name(name: str, width: int = 48) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def draw_timeline(scenario: str, events_by_pipeline: list[tuple[str, list[dict[str, Any]]]], output: pathlib.Path) -> None:
    colors = {
        "Dense GEMM": "#4E79A7", "Top-K gate": "#F28E2B", "Histogram": "#E15759",
        "Histogram + scan (fused)": "#E15759", "Exclusive scan": "#76B7B2",
        "Token permute": "#59A14F", "Grouped GEMM": "#B07AA1", "Unpermute": "#EDC948", "Other / runtime": "#9C9C9C",
    }
    height = 86 + 72 * len(events_by_pipeline)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
             '<rect width="1200" height="100%" fill="white"/>',
             f'<text x="20" y="26" font-family="Segoe UI,Arial" font-size="19" font-weight="bold">NSYS steady-chain swimlanes — {html.escape(scenario)}</text>',
             '<text x="20" y="48" font-family="Segoe UI,Arial" font-size="12" fill="#555">Last captured GPU kernels; rectangle width is NSYS duration. Profiler timeline is diagnostic only.</text>']
    legend_x = 20
    for stage, color in colors.items():
        parts.append(f'<rect x="{legend_x}" y="58" width="11" height="11" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 15}" y="68" font-family="Segoe UI,Arial" font-size="10">{stage}</text>')
        legend_x += 15 + len(stage) * 6.1
        if legend_x > 1100:
            break
    for index, (pipeline, events) in enumerate(events_by_pipeline):
        # A captured chain has 6–10 compute kernels; use its final 12 to exclude profiler warmup.
        selected = events[-12:]
        y = 94 + 72 * index
        parts.append(f'<text x="20" y="{y + 24}" font-family="Segoe UI,Arial" font-size="14" font-weight="bold">{html.escape(pipeline)}</text>')
        parts.append(f'<line x1="190" y1="{y + 19}" x2="1170" y2="{y + 19}" stroke="#d0d0d0"/>')
        if not selected:
            parts.append(f'<text x="205" y="{y + 24}" font-family="Segoe UI,Arial" font-size="12" fill="#a00">not_collected</text>')
            continue
        origin = min(event["start_ns"] for event in selected)
        span = max(event["start_ns"] + event["duration_ns"] for event in selected) - origin
        span = max(span, 1.0)
        for event in selected:
            x = 190 + 980 * (event["start_ns"] - origin) / span
            width = max(1.0, 980 * event["duration_ns"] / span)
            label = event["stage"]
            color = colors.get(label, colors["Other / runtime"])
            parts.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="38" rx="2" fill="{color}"><title>{html.escape(short_name(event["name"], 180))} — {event["duration_ns"] / 1000:.3f} µs</title></rect>')
            if width > 75:
                parts.append(f'<text x="{x + 3:.2f}" y="{y + 23}" font-family="Segoe UI,Arial" font-size="10" fill="white">{html.escape(label)}</text>')
        parts.append(f'<text x="190" y="{y + 57}" font-family="Segoe UI,Arial" font-size="10" fill="#555">trace span {span / 1000:.2f} µs</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def f(value: Any, digits: int = 2) -> str:
    number = float_or_none(value)
    return "not_collected" if number is None else f"{number:.{digits}f}"


def metric_display(metrics: dict[str, dict[str, Any]], name: str, digits: int = 2) -> str:
    """Format one NCU metric while preserving its raw unit in JSON evidence."""
    record = metrics[name]
    value = float_or_none(record.get("value"))
    if value is None:
        return "not_collected"
    if name == "duration_us" and record.get("unit") == "ns":
        value /= 1000.0
    return f"{value:.{digits}f}"


def geom(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def copy_artifacts(destination: pathlib.Path) -> list[dict[str, Any]]:
    artifacts = destination / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    for name in ("release_20260805_2240.jsonl", "release_20260805_2240.jsonl.manifest.json", "comparison_20260805_2240.json"):
        shutil.copy2(RUN_ROOT / name, artifacts / name)
    shutil.copy2(ROOT / "configs/cross_backend/benchmark/l3_realistic_moe_release.json", artifacts / "l3_realistic_moe_release.json")
    for scenario, _, _ in SCENARIOS:
        source = RUN_ROOT / "profile" / scenario
        target = artifacts / "profile" / scenario
        target.mkdir(parents=True)
        for _, _, nsys_dir, ncu_dir in PIPELINES:
            for dirname in (nsys_dir, ncu_dir):
                current = source / dirname
                if not current.is_dir():
                    raise FileNotFoundError(current)
                shutil.copytree(current, target / dirname)
    manifest = []
    for path in sorted(artifacts.rglob("*")):
        if path.is_file():
            manifest.append({"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "docs/reports/l3_realistic_moe_20260805_verified")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    records = read_jsonl(RUN_ROOT / "release_20260805_2240.jsonl")
    if len(records) != 36 or any(not record.get("validation", {}).get("ok") for record in records):
        raise ValueError("release source must contain 36 successful records")
    if {record["environment"]["build_git_sha"] for record in records} != {"af1b71be1629"}:
        raise ValueError("release source SHA is inconsistent")
    comparison = read_json(RUN_ROOT / "comparison_20260805_2240.json")
    if comparison.get("promotion_eligible") is not False or len(comparison.get("comparisons", [])) != 8:
        raise ValueError("invalid cross-backend comparison")
    output.mkdir(parents=True)
    artifacts = copy_artifacts(output)

    by_pair = {(item["case_id"], item["candidate_variant"]): item for item in comparison["comparisons"]}
    by_case_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case_variant[(record["case_id"], record["variant"])].append(record)
    for values in by_case_variant.values():
        if {item["process_run"] for item in values} != {1, 2, 3}:
            raise ValueError("each case/variant requires three process records")

    ncu_data: dict[str, dict[str, dict[str, Any]]] = {}
    nsys_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    timeline_paths = []
    for scenario, _, title in SCENARIOS:
        ncu_data[scenario] = {}
        nsys_data[scenario] = {}
        visual = []
        for variant, label, nsys_dir, ncu_dir in PIPELINES:
            ncu_report_path = RUN_ROOT / "profile" / scenario / ncu_dir / "reports/ncu_basic.ncu-rep"
            ncu_data[scenario][variant] = ncu_action(ncu_report_path)
            summary_path = RUN_ROOT / "profile" / scenario / nsys_dir / ("analysis/nsys_cuda_gpu_kern_sum.csv" if variant != "triton_reference" else "system_cuda_gpu_kern_sum.csv")
            trace_path = RUN_ROOT / "profile" / scenario / nsys_dir / ("analysis/nsys_cuda_gpu_trace.csv" if variant != "triton_reference" else "system_cuda_gpu_trace.csv")
            nsys_data[scenario][variant] = read_nsys_summary(summary_path)
            visual.append((label, trace_events(trace_path)))
        image = output / f"timeline_{scenario}.svg"
        draw_timeline(title, visual, image)
        timeline_paths.append(image)

    (output / "analysis").mkdir()
    (output / "analysis/ncu_basic.json").write_text(json.dumps(ncu_data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "analysis/nsys_hotspots.json").write_text(json.dumps(nsys_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    collection_manifest = {
        "schema_version": "raggedroute.l3_realistic_moe_profile_manifest.v1",
        "release_source": "out/research/l3_realistic_20260805/release_20260805_2240.jsonl",
        "release_contract": {"processes": 3, "warmup": 20, "samples": 30, "kernel_repeats": 3, "seed": 20260805},
        "profiler_contract": {"profile_once": True, "warmup": 20, "ncu_set": "basic", "ncu_clock_control": "none", "ncu_cache_control": "none", "profiler_duration_is_release_metric": False},
        "toolchains": {"native": "NSYS 2026.1.3; NCU 2026.2.1", "triton": "vLLM image + WSL-mounted NSYS/NCU 2026.1.x/2026.2.1"},
        "scenarios": {},
    }
    for scenario, case_id, _ in SCENARIOS:
        params = by_case_variant[(case_id, "cuda_all_candidates_chain")][0]["case_config"]
        collection_manifest["scenarios"][scenario] = {
            "params": {key: params[key] for key in ("T", "E", "K", "N", "R", "top_k")},
            "pipelines": {
                variant: {
                    "nsys_source": str((RUN_ROOT / "profile" / scenario / nsys_dir).relative_to(ROOT)),
                    "ncu_source": str((RUN_ROOT / "profile" / scenario / ncu_dir / "reports/ncu_basic.ncu-rep").relative_to(ROOT)),
                    "ncu_kernel": ncu_data[scenario][variant]["kernel"],
                    "ncu_filter": "grouped_gemm_register16x32_async_v3_kernel" if variant == "cuda_all_candidates_chain" else ("cutlass::Kernel.*GemmGrouped" if variant == "library_all_baselines_chain" else "_grouped_gemm_kernel"),
                }
                for variant, _, nsys_dir, ncu_dir in PIPELINES
            },
        }
    (output / "analysis/collection_manifest.json").write_text(json.dumps(collection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_rows = []
    speed_ratios: dict[str, list[float]] = defaultdict(list)
    for case_id, case_title in ALL_CASES.items():
        reference = by_pair[(case_id, "cuda_all_candidates_chain")]
        triton_p50 = reference["reference_latency_us"]
        for variant, label in VARIANTS:
            if variant == "triton_reference":
                p50, p95, cv = triton_p50, reference["reference_p95_us"], reference["reference_cv"]
            else:
                item = by_pair[(case_id, variant)]
                p50, p95, cv = item["candidate_latency_us"], item["candidate_p95_us"], item["candidate_cv"]
                speed_ratios[variant].append(triton_p50 / p50)
            record = by_case_variant[(case_id, variant)][0]
            config, work = record["case_config"], record["work"]
            result_rows.append({"case_id": case_id, "case": case_title, "variant": variant, "label": label, "p50": p50, "p95": p95, "cv": cv, "tokens_s": config["T"] * 1e6 / p50, "routes_s": config["R"] * 1e6 / p50, "gbps": work["logical_bytes"] / p50 / 1e3, "tflops": work["flops"] / p50 / 1e6, "speed_vs_triton": triton_p50 / p50, "shape": f"T={config['T']}, E={config['E']}, K={config['K']}, N={config['N']}, R={config['R']}"})

    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        rows_by_case[row["case_id"]].append(row)
    release_table = ["| 场景 | 线路 | p50 (µs) | p95 (µs) | CV | tokens/s | routes/s | Effective GB/s | TFLOP/s | vs Triton |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case_id in ALL_CASES:
        for row in rows_by_case[case_id]:
            release_table.append(f"| {row['case']} (`{row['shape']}`) | {row['label']} | {row['p50']:.3f} | {row['p95']:.3f} | {row['cv']:.4f} | {row['tokens_s']:,.0f} | {row['routes_s']:,.0f} | {row['gbps']:.2f} | {row['tflops']:.3f} | {row['speed_vs_triton']:.3f}× |")

    ncu_sections = []
    for scenario, case_id, title in SCENARIOS:
        table = ["| 线路 | NCU kernel | Duration (µs) | SM % | Memory % | DRAM % | L1 % | L2 % | Reg/thread | Shared B/block | Waves/SM | Theoretical / achieved occupancy |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant, label, _, _ in PIPELINES:
            data = ncu_data[scenario][variant]
            metrics = data["metrics"]
            value = lambda name, digits=2: metric_display(metrics, name, digits)
            table.append(f"| {label} | `{short_name(data['kernel'], 52)}` | {value('duration_us')} | {value('sm_pct')} | {value('memory_pct')} | {value('dram_pct')} | {value('l1_pct')} | {value('l2_pct')} | {value('registers_per_thread', 0)} | {value('shared_mem_per_block', 0)} | {value('waves_per_sm')} | {value('theoretical_occupancy_pct')}% / {value('achieved_occupancy_pct')}% |")
        dominant = []
        for variant, label, _, _ in PIPELINES:
            entries = nsys_data[scenario][variant]
            total = sum(item["total_ns"] for item in entries)
            first = entries[0] if entries else {"name": "not_collected", "pct": 0.0}
            dominant.append(f"{label}: `{short_name(first['name'], 72)}` ({first['pct']:.1f}% NSYS GPU kernel time)")
        ncu_sections.append(f"### {title}\n\n" + "\n".join(table) + "\n\nNSYS 主导热点：" + "；".join(dominant) + ".")

    evidence_notes = []
    for scenario, _, title in SCENARIOS:
        cuda = ncu_data[scenario]["cuda_all_candidates_chain"]["metrics"]
        library = ncu_data[scenario]["library_all_baselines_chain"]["metrics"]
        triton = ncu_data[scenario]["triton_reference"]["metrics"]
        evidence_notes.append(
            f"- **{title}**：CUDA candidate 的 grouped GEMM 为 {f(cuda['sm_pct']['value'])}% SM / {f(cuda['dram_pct']['value'])}% DRAM，"
            f"library/CUTLASS 为 {f(library['sm_pct']['value'])}% / {f(library['dram_pct']['value'])}%，"
            f"Triton 为 {f(triton['sm_pct']['value'])}% / {f(triton['dram_pct']['value'])}%。"
            "这些是一次 NCU replay 的机制诊断，不能替代 release p50。"
        )

    geomean_cuda = geom(speed_ratios["cuda_all_candidates_chain"])
    geomean_library = geom(speed_ratios["library_all_baselines_chain"])
    report = f"""# RaggedRoute 真实 MoE L3 三线路实测与 Nsight 分析

> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}。所有发布性能数字来自未 profile 的 3 个独立进程 release JSONL；NSYS/NCU 只用于定位机制，绝不用于发布延迟结论。

## 结论摘要

- **小 shape 的 CUDA 优势不能外推。** Mixtral 原尺寸单投影 decode (`T=8,E=8,K=4096,N=14336`) 中 Triton p50 **2843.136 µs**，快于 CUDA candidate **3656.192 µs** 和 library **5593.429 µs**；此时 L3 时间主要由 Grouped GEMM 决定。
- **中等/预填充代理中 library 更强。** 连续 decode 代理中 library 比 CUDA 快约 **1.12×**；chunked prefill 中 library 为 **250.539 µs**、CUDA 为 **452.267 µs**、Triton 为 **1461.589 µs**。
- **E64 细粒度压力下 CUDA 与 library 接近，Triton 明显退化。** CUDA **214.699 µs**，library **212.992 µs**（差异小于 1%，应结合 library 的 p95/CV 视为接近）；Triton **3286.869 µs**。该实现以 `R` 作为每 expert 的 worst-case Grouped GEMM launch bound，E=64 时产生大量无效 tile 工作。
- 跨四场景相对 Triton 的 p50 几何平均为：CUDA candidate **{geomean_cuda:.3f}×**，library baseline **{geomean_library:.3f}×**。这不是 promotion 结论：三线路的 compiler/runtime 不同，library Top-K/offset 合同也不严格等价。

## 实验合同与边界

| 项目 | 固定值 |
|---|---|
| 源码 / 分支 | `af1b71be1629` / `codex/l3-realistic-moe-eval`，release records 均为 clean build |
| GPU | NVIDIA GeForce RTX 3080，sm_86，68 SM，10 GiB |
| native 工具链 | CUDA 13.3.73，NSYS 2026.1.3，NCU 2026.2.1 |
| Triton 工具链 | `vllm/vllm-openai:latest`，PyTorch 2.11.0+cu130，Triton 3.6.0；WSL 挂载 NSYS/NCU 2026.1.3/2026.2.1 |
| Release 协议 | warm cache；20 warmups；30 samples；3 kernel repeats/sample；3 个独立进程；seed `20260805` |
| L3 边界 | 从 tokens 到 output 的七阶段链；排除输入生成、CPU oracle、H2D、allocation；Triton 另排除 JIT/autotune |
| 路由 | `router_projection_random`；仅 Top-2、`E≤64`，**不是** Zipf/skew trace，也不是 DeepSeek-V3 256-expert/Top-8 复现 |

Library 链使用仓库内 `library_baseline`：Top-K 为 benchmark-only 合同，CUTLASS grouped GEMM 使用固定 host offsets；因此 `strict_comparison_eligible=false`。CUDA/Triton 也因 runtime/compiler stack 不同而保持 `promotion_eligible=false`。

## 未 profile 的 Release 性能

{chr(10).join(release_table)}

`Effective GB/s` 和 `TFLOP/s` 以记录的 L3 logical bytes/FLOPs 除 p50 推导；它们是端到端工作量归一化指标，非硬件总线或峰值算力利用率。

## NSYS 整链泳道图

每张图由对应 `.nsys-rep` 的 `cuda_gpu_trace.csv` 直接绘制，选择 trace 的末尾稳定 GPU kernels；矩形宽度是实际 NSYS duration。图只服务于阶段/launch 诊断，不能与 release p50 混用。

### Mixtral 原尺寸 decode

![Mixtral NSYS swimlanes](timeline_mixtral_exact.svg)

### Chunked prefill

![Chunked prefill NSYS swimlanes](timeline_chunked_prefill.svg)

### E64 细粒度专家压力

![E64 NSYS swimlanes](timeline_finegrained_prefill.svg)

## NCU basic：主导 Grouped GEMM

{chr(10).join(ncu_sections)}

`basic` 不包含逐 warp stall、local load/store spill 指令、source correlation 或 PM sampling；这些字段均为 `not_collected`，而不是零。现有 occupancy、SM/DRAM/L1/L2、registers、shared memory 与 waves 已足以判定三组工作负载均由 Grouped GEMM 的 tile/scheduling 效率主导，因此没有虚构 detailed 结果。

## 基于证据的解释

{chr(10).join(evidence_notes)}

1. **真实模型尺寸改变热点的相对权重。** Dense GEMM、Top-K、histogram/scan、permute、unpermute 的 launch 与融合优势在短小工作负载里可见；当 `K/N` 或每 expert 的矩阵工作增大时，Grouped GEMM 占据 NSYS 大部分 GPU time，最终由其 kernel 的 scheduling/tile 效率决定。
2. **强库并不会天然输给手写 CUDA。** CUTLASS 和 Triton 都采用各自的 schedule、寄存器/共享内存权衡与 grid 几何。当前 candidate 走 strict FP32 SIMT 路径，不能把 Tensor Core/TF32 优化直接当作等价替换；比较必须先冻结数值合同。
3. **Triton 的 E64 退化是实现策略，不是“Trition 一定慢”。** 该 reference 的 `worst_case_R` launch 上界让许多 expert 的空/尾 tile 仍被调度。下一轮的单变量改动应先引入基于实际 `offsets` 的有效 M/tile 上界，随后重新做 full correctness、三进程 unprofiled release 与同一 profile contract。

## 工程限制与后续验证门槛

- Mixtral 场景仅模拟一个 expert projection；完整 MoE FFN 的三投影、通信、KV/attention 和服务调度不在该 L3 边界内。
- Release 未锁 GPU clocks；逐进程 CV 已完整报告。NCU 使用 `--clock-control none` 和 `--cache-control none`，其 duration 只作诊断。
- 任何后续优化必须保持：相同 seed/shape/math/cache/时间边界，所有 oracle PASS，3 个独立 unprofiled 进程 p50/p95 通过门槛；否则只记录为诊断候选。

## 原始工件与可复核性

- [Release JSONL](artifacts/release_20260805_2240.jsonl) · [Release manifest](artifacts/release_20260805_2240.jsonl.manifest.json) · [Comparison JSON](artifacts/comparison_20260805_2240.json)
- [Release suite config](artifacts/l3_realistic_moe_release.json) · [Collection manifest / commands](analysis/collection_manifest.json) · [NCU parsed metrics](analysis/ncu_basic.json) · [NSYS parsed hotspots](analysis/nsys_hotspots.json)
- 所有 3 场景 × 3 线路的原始 `.nsys-rep`、`.sqlite`、NSYS CSV、`.ncu-rep` 与 NCU manifests 位于 [profile artifacts](artifacts/profile/)，各文件的 SHA-256 见 [artifact manifest](artifact_manifest.json)。
"""
    report_path = output / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    artifacts.extend({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in timeline_paths)
    artifacts.extend([
        {"path": "analysis/ncu_basic.json", "bytes": (output / "analysis/ncu_basic.json").stat().st_size, "sha256": sha256(output / "analysis/ncu_basic.json")},
        {"path": "analysis/nsys_hotspots.json", "bytes": (output / "analysis/nsys_hotspots.json").stat().st_size, "sha256": sha256(output / "analysis/nsys_hotspots.json")},
        {"path": "analysis/collection_manifest.json", "bytes": (output / "analysis/collection_manifest.json").stat().st_size, "sha256": sha256(output / "analysis/collection_manifest.json")},
        {"path": REPORT_NAME, "bytes": report_path.stat().st_size, "sha256": sha256(report_path)},
    ])
    (output / "artifact_manifest.json").write_text(json.dumps({"schema_version": "raggedroute.artifact_manifest.v1", "artifacts": artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
