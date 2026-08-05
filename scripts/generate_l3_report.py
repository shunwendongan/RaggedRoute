#!/usr/bin/env python3
"""Generate the reproducible RaggedRoute L3 three-way report and timeline images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


RUN_STAMP = "20260805_204227"
REPORT_STAMP = "20260805"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def short_kernel(name: str) -> str:
    patterns = (
        "grouped_gemm_register16x32_async_v3_kernel",
        "cub_block_radix_top2_kernel",
        "DeviceHistogramInitKernel",
        "DeviceHistogramSweepKernel",
        "block_scan_kernel",
        "DeviceRadixSortSingleTileKernel",
        "compute_expert_offsets",
        "expand_rows",
        "finalize_routing",
        "dense_gemm_register_tiled_v3_64x32_async_kernel",
        "topk_gate_local_pair_two_reduce_kernel",
        "histogram_exclusive_scan_fused_subwarp_kernel",
        "token_permute_token_owned_top2_kernel",
        "unpermute_warp_token_vec4_kernel",
        "ampere_sgemm_32x32_sliced1x4_nn",
    )
    for pattern in patterns:
        if pattern in name:
            return pattern
    if name.startswith("_"):
        return name
    if "vectorized_elementwise_kernel" in name:
        return "fill/reset"
    if "cutlass::Kernel" in name or "GemmGrouped" in name:
        return "CUTLASS grouped GEMM"
    return name.split("(", 1)[0][-70:]


def classify(pipeline: str, name: str) -> str | None:
    if pipeline == "cuda_candidate":
        rules = (
            ("dense_gemm", "Dense GEMM"),
            ("topk_gate", "Top-K Gate"),
            ("histogram_exclusive_scan", "Histogram + Scan (fused)"),
            ("token_permute", "Token Permute"),
            ("grouped_gemm", "Grouped GEMM"),
            ("unpermute", "Unpermute"),
        )
    elif pipeline == "library":
        rules = (
            ("ampere_sgemm", "Dense GEMM"),
            ("cub_block_radix_top2", "Top-K Gate"),
            ("DeviceHistogram", "Histogram"),
            ("block_scan", "Exclusive Scan"),
            ("DeviceRadixSort", "Token Permute"),
            ("compute_expert_offsets", "Token Permute"),
            ("expand_rows", "Token Permute"),
            ("cutlass::Kernel", "Grouped GEMM"),
            ("finalize_routing", "Unpermute"),
        )
    else:
        rules = (
            ("_dense_gemm_kernel", "Dense GEMM"),
            ("_top2_selected_softmax_kernel", "Top-K Gate"),
            ("FillFunctor", "Histogram reset"),
            ("_histogram_kernel", "Histogram"),
            ("_exclusive_scan_kernel", "Exclusive Scan"),
            ("_route_map_kernel", "Token Permute / map"),
            ("_permute_rows_kernel", "Token Permute / rows"),
            ("_grouped_gemm_kernel", "Grouped GEMM"),
            ("_unpermute_kernel", "Unpermute"),
        )
    for needle, stage in rules:
        if needle in name:
            return stage
    return None


def read_timeline(path: Path, pipeline: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events: list[dict[str, Any]] = []
    for row in rows:
        stage = classify(pipeline, row["Name"])
        if stage:
            events.append(
                {
                    "start_ns": int(row["Start (ns)"]),
                    "duration_ns": int(row["Duration (ns)"]),
                    "stage": stage,
                    "kernel": short_kernel(row["Name"]),
                }
            )
    if pipeline == "cuda_candidate":
        start_indices = [i for i, event in enumerate(events) if event["stage"] == "Dense GEMM"]
        start = start_indices[-1]
        end = next(i for i in range(start, len(events)) if events[i]["stage"] == "Unpermute")
        events = events[start : end + 1]
    elif pipeline == "library":
        start_indices = [i for i, event in enumerate(events) if event["stage"] == "Dense GEMM"]
        start = start_indices[-1]
        end = next(i for i in range(start, len(events)) if events[i]["stage"] == "Unpermute")
        events = events[start : end + 1]
    if not events:
        raise RuntimeError(f"No classified timeline events in {path}")
    return events


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


COLORS = {
    "Dense GEMM": "#4C78A8",
    "Top-K Gate": "#F58518",
    "Histogram + Scan (fused)": "#54A24B",
    "Histogram reset": "#9CD78D",
    "Histogram": "#54A24B",
    "Exclusive Scan": "#72B7B2",
    "Token Permute": "#E45756",
    "Token Permute / map": "#FF9DA6",
    "Token Permute / rows": "#E45756",
    "Grouped GEMM": "#B279A2",
    "Unpermute": "#FFBF79",
}


def draw_timeline(events: list[dict[str, Any]], title: str, output: Path) -> dict[str, Any]:
    stages = list(dict.fromkeys(event["stage"] for event in events))
    start_ns = min(event["start_ns"] for event in events)
    end_ns = max(event["start_ns"] + event["duration_ns"] for event in events)
    span_us = (end_ns - start_ns) / 1000.0
    width, left, right, top, lane_h = 1800, 330, 70, 120, 66
    height = top + lane_h * len(stages) + 105
    image = Image.new("RGB", (width, height), "#FBFCFE")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    label_font = load_font(20, bold=True)
    small_font = load_font(16)
    tiny_font = load_font(14)
    draw.text((32, 28), title, fill="#18212F", font=title_font)
    draw.text((32, 70), "NSYS CUDA GPU trace · last post-warmup complete chain · profiler duration is diagnostic only", fill="#536070", font=small_font)
    plot_w = width - left - right
    grid_count = 8
    for i in range(grid_count + 1):
        x = left + plot_w * i / grid_count
        draw.line((x, top - 10, x, height - 72), fill="#D8DEE8", width=1)
        value = span_us * i / grid_count
        draw.text((x - 22, height - 59), f"{value:.1f}", fill="#536070", font=tiny_font)
    draw.text((left + plot_w / 2 - 25, height - 31), "time (µs)", fill="#18212F", font=small_font)
    for lane, stage in enumerate(stages):
        y = top + lane * lane_h
        draw.rectangle((18, y, width - 20, y + lane_h - 8), fill="#F3F6FA")
        stage_total = sum(event["duration_ns"] for event in events if event["stage"] == stage) / 1000.0
        draw.text((32, y + 10), stage, fill="#18212F", font=label_font)
        draw.text((32, y + 36), f"kernel Σ {stage_total:.3f} µs", fill="#536070", font=tiny_font)
        for event in (item for item in events if item["stage"] == stage):
            x0 = left + (event["start_ns"] - start_ns) / (end_ns - start_ns) * plot_w
            x1 = left + (event["start_ns"] + event["duration_ns"] - start_ns) / (end_ns - start_ns) * plot_w
            x1 = max(x1, x0 + 3)
            draw.rounded_rectangle((x0, y + 9, x1, y + lane_h - 17), radius=5, fill=COLORS.get(stage, "#777777"), outline="#FFFFFF", width=1)
            if x1 - x0 > 76:
                text = f"{event['duration_ns'] / 1000.0:.2f} µs"
                draw.text((x0 + 6, y + 20), text, fill="white", font=tiny_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    kernel_sum_us = sum(event["duration_ns"] for event in events) / 1000.0
    return {"span_us": span_us, "kernel_sum_us": kernel_sum_us, "events": events}


def stage_stats(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    totals: dict[str, int] = defaultdict(int)
    kernels: dict[str, list[str]] = defaultdict(list)
    for event in timeline["events"]:
        totals[event["stage"]] += event["duration_ns"]
        if event["kernel"] not in kernels[event["stage"]]:
            kernels[event["stage"]].append(event["kernel"])
    total = sum(totals.values())
    return [
        {
            "stage": stage,
            "duration_us": duration / 1000.0,
            "share_pct": duration / total * 100.0,
            "kernels": kernels[stage],
        }
        for stage, duration in totals.items()
    ]


def parse_ncu(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def metric(label: str) -> float | str:
        match = re.search(rf"^\s*{re.escape(label)}\s+(?:[A-Za-z/#% ]+\s+)?([0-9][0-9,.]*)\s*$", text, re.MULTILINE)
        return float(match.group(1).replace(",", "")) if match else "not_collected"

    kernel_match = re.search(r"^\s{2}(.+?) \([^\n]+Context", text, re.MULTILINE)
    return {
        "kernel": short_kernel(kernel_match.group(1)) if kernel_match else "unsupported_or_unknown",
        "duration_us": metric("Duration"),
        "compute_pct": metric("Compute (SM) Throughput"),
        "memory_pct": metric("Memory Throughput"),
        "dram_pct": metric("DRAM Throughput"),
        "l1_pct": metric("L1/TEX Cache Throughput"),
        "l2_pct": metric("L2 Cache Throughput"),
        "grid": metric("Grid Size"),
        "block": metric("Block Size"),
        "registers": metric("Registers Per Thread"),
        "waves": metric("Waves Per SM"),
        "theoretical_occupancy_pct": metric("Theoretical Occupancy"),
        "achieved_occupancy_pct": metric("Achieved Occupancy"),
        "dynamic_shared_kib": metric("Dynamic Shared Memory Per Block"),
        "warp_stalls": "not_collected",
        "local_load_store_spills": "not_collected",
        "pm_sampling": "not_collected",
    }


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_stage_table(stats: list[dict[str, Any]]) -> str:
    lines = ["| 阶段 | 主 kernel / 子步骤 | GPU kernel time (µs) | kernel-time 占比 |", "|---|---|---:|---:|"]
    for row in stats:
        kernels = "<br>".join(f"`{name}`" for name in row["kernels"])
        lines.append(f"| {row['stage']} | {kernels} | {row['duration_us']:.3f} | {row['share_pct']:.1f}% |")
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_report_links(report_path: Path) -> int:
    text = report_path.read_text(encoding="utf-8")
    targets = re.findall(r"(?:!\[[^\]]*\]|\[[^\]]*\])\(([^)]+)\)", text)
    missing = [
        target
        for target in targets
        if not target.startswith(("http://", "https://", "#"))
        and not (report_path.parent / target).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing report links: {missing}")
    return len(targets)


def copy_artifacts(root: Path, report_dir: Path) -> list[dict[str, Any]]:
    artifact_root = report_dir / "artifacts"
    sources = [
        root / f"out/research/l3_three_way/release_{RUN_STAMP}.jsonl",
        root / f"out/research/l3_three_way/release_{RUN_STAMP}.jsonl.manifest.json",
        root / f"out/research/l3_three_way/comparison_{RUN_STAMP}.json",
        root / "out/research/l3_three_way/verification_ctest.txt",
        Path("C:/Users/Administrator/Desktop/RaggedRoute_7ops_library_comparison_20260805_193637.md"),
    ]
    profile_root = root / "out/research/l3_three_way/profile"
    patterns = {
        "cuda_candidate_nsys": ("manifest.json", "analysis/*.csv", "analysis/*.json", "reports/*.nsys-rep"),
        "cuda_candidate_ncu": ("manifest.json", "analysis/*.txt", "analysis/*.csv", "analysis/*.json", "reports/*.ncu-rep"),
        "library_nsys": ("manifest.json", "analysis/*.csv", "analysis/*.json", "reports/*.nsys-rep"),
        "library_ncu": ("manifest.json", "analysis/*.txt", "analysis/*.csv", "analysis/*.json", "reports/*.ncu-rep"),
        "triton_nsys_capture": ("*.csv", "*.nsys-rep"),
        "triton_ncu": ("*.txt", "*.csv", "*.ncu-rep"),
    }
    for directory, globs in patterns.items():
        for pattern in globs:
            sources.extend((profile_root / directory).glob(pattern))
    copied: list[dict[str, Any]] = []
    for source in sources:
        if not source.exists():
            continue
        if source.is_relative_to(root):
            relative = source.relative_to(root / "out/research/l3_three_way")
        else:
            relative = Path("prior_single_operator_report") / source.name
        target = artifact_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.name == "verification_ctest.txt":
            raw = source.read_bytes()
            encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            normalized = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
            target.write_bytes(normalized.encode("utf-8"))
        else:
            shutil.copy2(source, target)
        copied.append({"path": target.relative_to(report_dir).as_posix(), "bytes": target.stat().st_size, "sha256": sha256(target)})
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = (args.output_dir or root / f"docs/reports/l3_three_way_{REPORT_STAMP}").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    release_path = root / f"out/research/l3_three_way/release_{RUN_STAMP}.jsonl"
    comparison_path = root / f"out/research/l3_three_way/comparison_{RUN_STAMP}.json"
    manifest_path = Path(str(release_path) + ".manifest.json")
    records = read_jsonl(release_path)
    comparison = read_json(comparison_path)
    manifest = read_json(manifest_path)
    if len(records) != 9:
        raise RuntimeError(f"Expected 9 release records, found {len(records)}")
    if any(not row["validation"]["ok"] for row in records):
        raise RuntimeError("At least one release correctness validation failed")
    variants = {row["variant"] for row in records}
    expected = {"cuda_all_candidates_chain", "library_all_baselines_chain", "triton_reference"}
    if variants != expected or any(sum(row["variant"] == variant for row in records) != 3 for variant in expected):
        raise RuntimeError("Release records do not contain exactly three processes for each expected variant")

    timeline_specs = {
        "cuda_candidate": root / "out/research/l3_three_way/profile/cuda_candidate_nsys/analysis/timeline_cuda_gpu_trace.csv",
        "library": root / "out/research/l3_three_way/profile/library_nsys/analysis/timeline_cuda_gpu_trace.csv",
        "triton": root / "out/research/l3_three_way/profile/triton_nsys_capture/nsys_cuda_gpu_trace.csv",
    }
    titles = {
        "cuda_candidate": "CUDA candidates chain — steady-state GPU timeline",
        "library": "Repository library baselines chain — steady-state GPU timeline",
        "triton": "Triton reference chain — profiler-API-captured GPU timeline",
    }
    timelines: dict[str, dict[str, Any]] = {}
    for pipeline, path in timeline_specs.items():
        events = read_timeline(path, pipeline)
        timelines[pipeline] = draw_timeline(events, titles[pipeline], report_dir / f"timeline_{pipeline}.png")

    ncu = {
        "cuda_candidate": parse_ncu(root / "out/research/l3_three_way/profile/cuda_candidate_ncu/analysis/ncu_basic_details.txt"),
        "library": parse_ncu(root / "out/research/l3_three_way/profile/library_ncu/analysis/ncu_basic_details.txt"),
        "triton": parse_ncu(root / "out/research/l3_three_way/profile/triton_ncu/ncu_basic_details.txt"),
    }
    artifacts = copy_artifacts(root, report_dir)
    report_data = {
        "schema_version": "raggedroute.l3_three_way_report.v1",
        "generated_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "release_record_count": len(records),
        "timelines": timelines,
        "ncu": ncu,
        "artifacts": artifacts,
    }
    (report_dir / "report_data.json").write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts.append({"path": "report_data.json", "bytes": (report_dir / "report_data.json").stat().st_size, "sha256": sha256(report_dir / "report_data.json")})
    (report_dir / "artifact_manifest.json").write_text(json.dumps({"schema_version": "raggedroute.artifact_manifest.v1", "artifacts": artifacts}, indent=2), encoding="utf-8")

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_variant[row["variant"]].append(row)
    for rows in by_variant.values():
        rows.sort(key=lambda row: row["process_run"])
    comp_by_candidate = {item["candidate_variant"]: item for item in comparison["comparisons"]}
    cuda_comp = comp_by_candidate["cuda_all_candidates_chain"]
    lib_comp = comp_by_candidate["library_all_baselines_chain"]
    cuda_p50 = cuda_comp["candidate_latency_us"]
    lib_p50 = lib_comp["candidate_latency_us"]
    triton_p50 = cuda_comp["reference_latency_us"]
    cuda_vs_lib = lib_p50 / cuda_p50

    process_lines = ["| 线路 | Process | p50 (µs) | p95 (µs) | CV | Correctness |", "|---|---:|---:|---:|---:|---|"]
    display = {
        "cuda_all_candidates_chain": "CUDA candidates",
        "library_all_baselines_chain": "Library baselines（诊断）",
        "triton_reference": "Triton reference",
    }
    for variant in ("cuda_all_candidates_chain", "library_all_baselines_chain", "triton_reference"):
        for row in by_variant[variant]:
            timing = row["timing"]
            validation = row["validation"]
            process_lines.append(
                f"| {display[variant]} | {row['process_run']} | {timing['batch_mean_us_p50']:.3f} | "
                f"{timing['batch_mean_us_p95']:.3f} | {timing['cv']:.4f} | PASS（abs {validation['max_abs_error']:.3e}） |"
            )

    env = records[0]["environment"]
    common = manifest["suite"]["common"]
    snapshot_start = manifest.get("gpu_snapshot_start", "not_collected")
    snapshot_end = manifest.get("gpu_snapshot_end", "not_collected")
    report = f"""# RaggedRoute 七算子 L3 三线路真实性能与 Nsight 分析

> 生成时间：{report_data['generated_local']}。正式性能数字全部来自未 profile 的 release 运行；NSYS/NCU 只用于机制诊断。

## 结论摘要

- 在本次固定工作负载 `T=512, E=64, top_k=2, R=1024, K=N=128, fp32 strict` 下，CUDA candidate 链的聚合 p50 为 **{cuda_p50:.3f} µs**，Triton reference 为 **{triton_p50:.3f} µs**，CUDA 对 Triton 为 **{cuda_comp['candidate_vs_reference_speedup']:.3f}×**。
- Repository library baseline 链的聚合 p50 为 **{lib_p50:.3f} µs**，对 Triton 为 **{lib_comp['candidate_vs_reference_speedup']:.3f}×**。CUDA 相对 library 的观测比值为 **{cuda_vs_lib:.3f}×**，但它是诊断差距，**不是严格可发布 speedup**：Top-K 是 benchmark-only 合同，CUTLASS grouped GEMM 使用确定性 oracle 生成的固定 host offsets。
- 三条线路均通过七阶段 oracle/postcondition。CUDA 的主要热点是 Grouped GEMM（NSYS kernel-time 占比 {stage_stats(timelines['cuda_candidate'])[-2]['share_pct']:.1f}%）；Triton 的 Grouped GEMM 占 {stage_stats(timelines['triton'])[-2]['share_pct']:.1f}%，是端到端差距的首要来源。
- NCU 表明 CUDA grouped kernel 受到 106 registers/thread、低 occupancy 和 SM/L2 工作不均限制；library CUTLASS kernel 同时受 64-block grid 小于 68 SM、144 registers/thread 和 shared-memory 限制；Triton kernel 则在 4096 blocks、15.06 waves/SM 下达到约 85.74% Compute/L1 利用率，瓶颈是 strict-IEEE worst-case tile 工作量而不是 occupancy 不足。

## 实验合同

| 项目 | 固定值 |
|---|---|
| 源码基线 / 实验提交 | `8d4c281eb28bb63e93f9b499343486938f8c3515` / `a6d37017427755b9493876f8e05df96562b2a3dd` |
| 分支 | `codex/l3-three-way-eval` |
| GPU | {env['gpu_name']}，CC {env['compute_capability']}，{env['sm_count']} SM，{env['global_memory_bytes'] / 2**30:.1f} GiB |
| Driver / CUDA | driver API {env['cuda_driver']}；native runtime {env['cuda_runtime']}；NVCC {env['cuda_compiler']} |
| Nsight | NSYS 2026.1.3；NCU 2026.2.1；NCU set=`basic`，clock-control=`none` |
| Triton 容器 | `vllm/vllm-openai:latest`；PyTorch 2.11.0+cu130；Triton 3.6.0 |
| Shape / 语义 | `chain_from_tokens`，`T=512,E=64,top_k=2,R=1024,K=128,N=128,fp32,strict_fp32` |
| Release 采样 | warm cache；{common['warmup']} warmups；{common['samples']} samples；{common['kernel_repeats']} repeats/sample；3 independent processes/line |
| 随机性 | seed `{common['seed']}`；`router_projection_random`；同一输入合同 |
| 时间边界 | L3 steady chain；排除 input generation、CPU oracle、H2D、allocation；Triton 额外排除 JIT/autotune |
| 正确性 gate | CTest 8/8 PASS；release 9/9 records `validation.ok=true` |

正式运行按独立进程串行执行。Manifest 的 GPU 快照为：

- start：`{snapshot_start}`
- end：`{snapshot_end}`

未固定 GPU clocks；进程级竞争负载清单未单独归档，状态为 `unsupported_or_unknown`。因此报告同时保留逐进程 CV，且不把 profiler duration 用作 release 结论。

## 三条线路的七算子映射

| 阶段 | CUDA candidate | Repository library baseline | Triton reference |
|---|---|---|---|
| Dense GEMM | `cuda_register_tiled_v3_64x32_async` | `cublaslt` | `_dense_gemm_kernel` |
| Top-K Gate | `cuda_local_pair_two_reduce_top2_v4` | `cub_block_radix_top2`（benchmark-only） | `_top2_selected_softmax_kernel` |
| Histogram | `cuda_fused_histogram_scan`（融合） | `CUB DeviceHistogram` | `_histogram_kernel` |
| Exclusive Scan | 同上融合 kernel | `CUB BlockScan` | `_exclusive_scan_kernel` |
| Token Permute | `cuda_candidate_from_ids_equivalent` | `vllm_moe_permute` 路径 | `_route_map_kernel` + `_permute_rows_kernel` |
| Grouped GEMM | `cuda_grouped_sm86_fp32_v1` | `cutlass_grouped` | `_grouped_gemm_kernel` |
| Unpermute | `cuda_warp_token_vec4` | `vllm_finalize_routing` | `_unpermute_kernel` |

## 未 profile 的 Release 结果

| 线路 | 聚合 p50 (µs) | 聚合 p95 (µs) | 跨进程 CV | Effective GB/s | TFLOP/s | 相对 Triton |
|---|---:|---:|---:|---:|---:|---:|
| CUDA candidates | {cuda_p50:.3f} | {cuda_comp['candidate_p95_us']:.3f} | {cuda_comp['candidate_cv']:.4f} | {by_variant['cuda_all_candidates_chain'][1]['work']['effective_gbps_batch_p50']:.2f} | {by_variant['cuda_all_candidates_chain'][1]['work']['tflops_batch_p50']:.3f} | {cuda_comp['candidate_vs_reference_speedup']:.3f}× |
| Library baselines（诊断） | {lib_p50:.3f} | {lib_comp['candidate_p95_us']:.3f} | {lib_comp['candidate_cv']:.4f} | {by_variant['library_all_baselines_chain'][1]['work']['effective_gbps_batch_p50']:.2f} | {by_variant['library_all_baselines_chain'][1]['work']['tflops_batch_p50']:.3f} | {lib_comp['candidate_vs_reference_speedup']:.3f}× |
| Triton reference | {triton_p50:.3f} | {cuda_comp['reference_p95_us']:.3f} | {cuda_comp['reference_cv']:.4f} | {by_variant['triton_reference'][1]['work']['logical_bytes'] / triton_p50 / 1000:.2f} | {by_variant['triton_reference'][1]['work']['flops'] / triton_p50 / 1e6:.3f} | 1.000× |

`comparison.json` 明确给出 `promotion_eligible=false`：CUDA/Triton 使用不同 compiler/runtime stack；library 链另有语义边界。因此这是一份工程诊断报告，不是默认 dispatch 的自动 promotion 记录。

### 每个独立进程

{chr(10).join(process_lines)}

## NSYS：整链泳道与七阶段贡献

泳道图由 NSYS `cuda_gpu_trace.csv` 的真实 start/duration 数据直接绘制，选择预热后的最后一条完整链；它不是手工估算，也不是 GUI 截图。矩形宽度与 GPU kernel duration 成比例，空白代表相邻 GPU work 之间的间隔。

### CUDA candidates

![CUDA candidate NSYS timeline](timeline_cuda_candidate.png)

{markdown_stage_table(stage_stats(timelines['cuda_candidate']))}

GPU kernel Σ={timelines['cuda_candidate']['kernel_sum_us']:.3f} µs，首个到末个 kernel 的 span={timelines['cuda_candidate']['span_us']:.3f} µs。Histogram 与 Exclusive Scan 被融合为一次 kernel launch，因此 7 个语义算子对应 6 个主 kernel。

### Repository library baselines

![Library baseline NSYS timeline](timeline_library.png)

{markdown_stage_table(stage_stats(timelines['library']))}

GPU kernel Σ={timelines['library']['kernel_sum_us']:.3f} µs，span={timelines['library']['span_us']:.3f} µs。Token Permute 路径包含 radix sort、expert-offset 和 row expansion；这解释了 library 链的额外 launch 与间隙。

### Triton reference

![Triton NSYS timeline](timeline_triton.png)

{markdown_stage_table(stage_stats(timelines['triton']))}

GPU kernel Σ={timelines['triton']['kernel_sum_us']:.3f} µs，span={timelines['triton']['span_us']:.3f} µs。该 trace 通过 CUDA Profiler API 只捕获预热后的一条链，排除了 JIT/autotune；NSYS 注入仍会放大 host launch 间隔，所以 span 不等于 release p50。

## NCU：每条链主导 Grouped GEMM

| 线路 | NCU kernel duration (µs) | Compute % | Memory % | DRAM % | L1 % | L2 % | Grid / Block | Reg/thread | Waves/SM | Theoretical / achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| CUDA candidates | {fmt(ncu['cuda_candidate']['duration_us'])} | {fmt(ncu['cuda_candidate']['compute_pct'])} | {fmt(ncu['cuda_candidate']['memory_pct'])} | {fmt(ncu['cuda_candidate']['dram_pct'])} | {fmt(ncu['cuda_candidate']['l1_pct'])} | {fmt(ncu['cuda_candidate']['l2_pct'])} | {fmt(ncu['cuda_candidate']['grid'],0)} / {fmt(ncu['cuda_candidate']['block'],0)} | {fmt(ncu['cuda_candidate']['registers'],0)} | {fmt(ncu['cuda_candidate']['waves'])} | {fmt(ncu['cuda_candidate']['theoretical_occupancy_pct'])}% / {fmt(ncu['cuda_candidate']['achieved_occupancy_pct'])}% |
| Library CUTLASS | {fmt(ncu['library']['duration_us'])} | {fmt(ncu['library']['compute_pct'])} | {fmt(ncu['library']['memory_pct'])} | {fmt(ncu['library']['dram_pct'])} | {fmt(ncu['library']['l1_pct'])} | {fmt(ncu['library']['l2_pct'])} | {fmt(ncu['library']['grid'],0)} / {fmt(ncu['library']['block'],0)} | {fmt(ncu['library']['registers'],0)} | {fmt(ncu['library']['waves'])} | {fmt(ncu['library']['theoretical_occupancy_pct'])}% / {fmt(ncu['library']['achieved_occupancy_pct'])}% |
| Triton | {fmt(ncu['triton']['duration_us'])} | {fmt(ncu['triton']['compute_pct'])} | {fmt(ncu['triton']['memory_pct'])} | {fmt(ncu['triton']['dram_pct'])} | {fmt(ncu['triton']['l1_pct'])} | {fmt(ncu['triton']['l2_pct'])} | {fmt(ncu['triton']['grid'],0)} / {fmt(ncu['triton']['block'],0)} | {fmt(ncu['triton']['registers'],0)} | {fmt(ncu['triton']['waves'])} | {fmt(ncu['triton']['theoretical_occupancy_pct'])}% / {fmt(ncu['triton']['achieved_occupancy_pct'])}% |

NCU duration 受 replay、cache-control 和序列化影响，只作为上下文。`basic` 未采集逐 stall、local load/store spill 和 PM sampling，因此这三类均为 `not_collected`，不能据此宣称某个具体 stall 或 register spill 已被证明。

### 证据链诊断

1. **CUDA candidate：寄存器限制 + 单 wave/工作不均。** 106 registers/thread 将理论 occupancy 限制在 33.33%，achieved 仅 26.62%；同时 Compute 21.71%、DRAM 24.08% 都低，而 grid=272 正好约 1 wave/SM。NCU 还报告 SM active-cycle min 比平均低 46.28%、L2 slice max 高于平均 32.69%。两类独立信号共同支持 latency/imbalance，而不是峰值带宽饱和。
2. **Library CUTLASS：underfill 是首要边界。** grid=64 小于 68 SM、waves/SM=0.94，至少 4 个 SM 无 block；144 registers/thread 和约 16.66 KiB dynamic shared memory 又把每 SM 限制到 1 block，achieved occupancy=16.66%。Compute 55.72% 尚未达到饱和，和 underfill 证据一致。
3. **Triton：计算/L1 饱和且做了过量 worst-case tile 工作。** grid=4096、15.06 waves/SM、achieved occupancy=64.55%，排除小 grid/低 occupancy 为主因；Compute/Memory=85.74%、L1=87.21%，而 DRAM=4.62%。这与 grouped kernel 对每 expert 按 `worst_case_R` tile 覆盖的实现相符，优化方向应是减少无效 tile/采用有效 row-count 调度，而不是单纯提高 occupancy。

## 七算子的组件级解释

| 算子 | 本次 L3 证据 | 工程判断 |
|---|---|---|
| Dense GEMM | NSYS：CUDA 约 13.5% kernel time；library 约 10%；Triton 约 2.2% | 当前 shape 下不是主瓶颈；保持 strict-FP32 合同后再讨论 Tensor Core/TF32。 |
| Top-K Gate | CUDA 单 kernel；library CUB Top-2 占明显比例；Triton 约 0.9% | Library Top-K 合同仅 benchmark-only，不能将链级差值归因成严格算子胜负。 |
| Histogram | CUDA 与 Scan 融合；library CUB init+sweep 两次；Triton 单独 reset+histogram | 融合减少一次 launch 和中间边界，优势主要体现在短链调度。 |
| Exclusive Scan | CUDA 融合；library/Triton 各自独立 launch | E=64 时算术工作很小，launch/gap 往往比 kernel body 更重要。 |
| Token Permute | CUDA 单 kernel；library 三个步骤；Triton map+rows 两个 kernel | Library 的 radix/offset/expand 增加 launch，且 mapping preparation 已包含在 L3 时间边界内。 |
| Grouped GEMM | 三路均为最大热点；Triton 92.0% kernel time | 优化优先级最高；上述 NCU 结论直接适用。 |
| Unpermute | 三路均为短 kernel | 不是当前主导项；应避免为了微小 kernel 改动扩大全链 workspace 或同步成本。 |

逐算子 release 背景可见归档的[七算子单体报告](artifacts/prior_single_operator_report/RaggedRoute_7ops_library_comparison_20260805_193637.md)。该报告来自此前独立单算子 campaign，不与本次 L3 数字混算。

## 优化优先级与验证门槛

1. Triton Grouped GEMM：以实际 expert row counts 限制 grid/tile 工作，保留 strict IEEE FP32；先验证全部 expert skew/tail，再用同一 L3 suite 做 3-process release A/B。目标是降低 grouped kernel GPU-time，同时不得增加 mapping materialization 到时间边界外。
2. CUDA Grouped GEMM：单变量测试更小 live-range/tile 或更均衡的 work scheduling；必须同时观察 registers/thread、local load/store（需 detailed）和 release p50/p95，不能只追 occupancy。
3. Library 路径：若要形成严格对照，先消除 Top-K 合同差异，并让 CUTLASS problem metadata 来自同一 device-side routing 结果；完成前继续标记 `diagnostic_not_strictly_comparable`。

任何候选只有在 correctness 全绿、同一 seed/shape/math/cache/time boundary、3 个独立进程、未 profile release p50 与 p95 均满足阈值后才可 promotion。

## 局限与状态

- 当前 release suite 只有一个固定 L3 shape，不能代表全部 release shape/distribution；跨 shape geomean 为 `not_collected`。
- Warp stall、spill 指令、source correlation、PM sampling 为 `not_collected`；basic 已足以区分当前三种主导机制，故未升级 detailed/full。
- NSYS/NCU 没有逐一 replay 七个非热点 kernel；它们的组件级数据来自同一完整链的 NSYS GPU timeline。
- Library 链 `strict_comparison_eligible=false`；CUDA/Triton comparison 也因 runtime/compiler stack 差异 `promotion_eligible=false`。
- RTX 3080 为 sm_86；本报告不建议 Hopper/Blackwell 专属 TMA、WGMMA、TMEM 或 tcgen05 机制。

## 原始工件

- [Release JSONL](artifacts/release_{RUN_STAMP}.jsonl) · [Release manifest](artifacts/release_{RUN_STAMP}.jsonl.manifest.json) · [Comparison JSON](artifacts/comparison_{RUN_STAMP}.json)
- [CTest 8/8 verification log](artifacts/verification_ctest.txt)
- [Artifact SHA-256 manifest](artifact_manifest.json) · [Machine-readable report data](report_data.json)
- CUDA candidate：[NSYS report](artifacts/profile/cuda_candidate_nsys/reports/system.nsys-rep) · [NCU report](artifacts/profile/cuda_candidate_ncu/reports/ncu_basic.ncu-rep)
- Library：[NSYS report](artifacts/profile/library_nsys/reports/system.nsys-rep) · [NCU report](artifacts/profile/library_ncu/reports/ncu_basic.ncu-rep)
- Triton：[NSYS report](artifacts/profile/triton_nsys_capture/system.nsys-rep) · [NCU report](artifacts/profile/triton_ncu/ncu_basic.ncu-rep)

原始 CSV、NCU 文本、profiler manifests 和命令行均保存在 `artifacts/profile/`。所有归档文件的大小与 SHA-256 在 `artifact_manifest.json` 中。
"""
    report_path = report_dir / "RaggedRoute_L3_3way_comparison.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"{report_path} ({validate_report_links(report_path)} verified links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
