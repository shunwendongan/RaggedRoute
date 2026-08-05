#!/usr/bin/env python3
"""Create a self-contained vLLM-semantic vs CUDA candidate L3 evidence report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "out/research/vllm_semantic_20260805"
CASES = [
    ("moe.mixtral_decode_exact.t8_e8_k4096_n14336", "Mixtral 原尺寸 decode", "T=8, E=8, K=4096, N=14336"),
    ("moe.continuous_decode_proxy.t128_e8_k1024_n3584", "连续 decode 代理", "T=128, E=8, K=1024, N=3584"),
    ("moe.chunked_prefill_proxy.t512_e8_k512_n1792", "Chunked prefill 代理", "T=512, E=8, K=512, N=1792"),
    ("moe.finegrained_prefill_proxy.t1024_e64_k256_n512", "E64 细粒度专家压力", "T=1024, E=64, K=256, N=512"),
]
PROFILE_CASES = [("mixtral_decode", CASES[0][0]), ("continuous_decode", CASES[1][0]), ("chunked_prefill", CASES[2][0])]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def aggregate(records: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    rows = [row for row in records if row["case_id"] == case_id]
    runs = {int(row["process_run"]) for row in rows}
    if runs != {1, 2, 3} or len(rows) != 3:
        raise ValueError(f"{case_id}: expected three independent process records")
    if not all(row["validation"]["ok"] for row in rows):
        raise ValueError(f"{case_id}: correctness gate failed")
    medians = [float(row["timing"]["batch_mean_us_p50"]) for row in rows]
    raw = [float(value) for row in rows for value in row["timing"]["raw_batch_mean_samples_us"]]
    work = rows[0]["work"]
    p50 = statistics.median(medians)
    mean = statistics.fmean(raw)
    return {
        "p50": p50, "p95": percentile(raw, .95),
        "cv": 0.0 if mean == 0 else statistics.pstdev(raw) / mean,
        "tokens_s": rows[0]["case_config"]["T"] * 1e6 / p50,
        "routes_s": rows[0]["case_config"]["R"] * 1e6 / p50,
        "gbps": float(work["logical_bytes"]) / p50 / 1e3,
        "tflops": float(work["flops"]) / p50 / 1e6,
        "records": rows,
    }


def nsys_hotspot(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_collected"}
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        return {"status": "not_collected"}
    top = rows[0]
    return {"status": "collected", "name": top["Name"], "pct": float(top["Time (%)"]), "total_us": float(top["Total Time (ns)"]) / 1e3}


def ncu_candidate(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_collected"}
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if len(rows) < 2:
        return {"status": "not_collected"}
    row = rows[1]  # row zero is the unit row in Nsight Compute's raw CSV.
    def number(name: str) -> float | str:
        value = row.get(name, "")
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return "not_collected"
    return {
        "status": "collected", "kernel": row.get("Kernel Name", "unknown"),
        "block": row.get("Block Size", "not_collected"), "grid": row.get("Grid Size", "not_collected"),
        "registers_per_thread": number("launch__registers_per_thread"),
        "shared_mem_kib": number("launch__shared_mem_per_block"),
        "waves_per_sm": number("launch__waves_per_multiprocessor"),
        "active_warps_pct": number("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "sm_pct": number("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_pct": number("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        "l2_pct": number("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        "l1_pct": number("l1tex__throughput.avg.pct_of_peak_sustained_active"),
        "duration_us": number("gpu__time_duration.avg"),
    }


def timeline(trace: pathlib.Path, out: pathlib.Path, title: str) -> None:
    rows = list(csv.DictReader(trace.open(encoding="utf-8-sig", newline=""))) if trace.exists() else []
    rows = [row for row in rows if row.get("Start (ns)") and row.get("Duration (ns)") and not row.get("Name", "").startswith("[CUDA")]
    # The final kernels are the post-warmup chain. Bound the drawing so prior allocation/JIT does not dominate.
    rows = rows[-12:]
    width, height, left = 1000, 180, 120
    if not rows:
        out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="40">{title}: not_collected</text></svg>\n', encoding="utf-8")
        return
    begin = min(int(row["Start (ns)"]) for row in rows)
    end = max(int(row["Start (ns)"]) + int(row["Duration (ns)"]) for row in rows)
    span = max(1, end - begin)
    colors = {"fused": "#ea580c", "grouped": "#2563eb", "topk": "#7c3aed", "align": "#059669", "gemm": "#0f766e", "permute": "#db2777"}
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="20" y="28" font-family="Arial" font-size="18">{title}</text>', f'<line x1="{left}" y1="105" x2="{width-30}" y2="105" stroke="#64748b"/>']
    for index, row in enumerate(rows):
        name = row["Name"]
        key = next((candidate for candidate in colors if candidate in name.lower()), "other")
        color = colors.get(key, "#94a3b8")
        x = left + (int(row["Start (ns)"]) - begin) / span * (width-left-150)
        w = max(2, int(row["Duration (ns)"]) / span * (width-left-150))
        y = 74 + (index % 2) * 38
        pieces.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="25" rx="3" fill="{color}"><title>{name}</title></rect>')
    pieces.append('<text x="20" y="160" font-family="Arial" font-size="12" fill="#475569">last post-warmup CUDA events; widths are NSYS durations (diagnostic only)</text></svg>')
    out.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def copy_file(src: pathlib.Path, destination: pathlib.Path) -> None:
    if src.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)


def digest_tree(root: pathlib.Path) -> dict[str, str]:
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "docs/reports/vllm_semantic_moe_20260806")
    parser.add_argument("--candidate-jsonl", type=pathlib.Path, default=RUN / "cuda_candidate_release.jsonl")
    parser.add_argument("--vllm-jsonl", type=pathlib.Path, default=RUN / "release.jsonl")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    analysis, artifacts = output / "analysis", output / "artifacts"
    analysis.mkdir(parents=True)
    candidate_path, vllm_path = args.candidate_jsonl.resolve(), args.vllm_jsonl.resolve()
    candidate, vllm = read_jsonl(candidate_path), read_jsonl(vllm_path)
    if len(candidate) != 12 or len(vllm) != 12:
        raise ValueError("expected exactly 12 candidate and 12 vLLM records")
    summary: dict[str, Any] = {"schema_version": "raggedroute.vllm_semantic_report.v1", "cases": {}, "profiles": {}}
    table = ["| 场景 | 线路 | p50 µs | p95 µs | CV | tokens/s | routes/s | Effective GB/s | TFLOP/s |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    ratios = []
    for case_id, display, shape in CASES:
        c, v = aggregate(candidate, case_id), aggregate(vllm, case_id)
        ratio = v["p50"] / c["p50"]
        ratios.append(ratio)
        summary["cases"][case_id] = {"shape": shape, "cuda_candidate": c, "vllm_semantic": v, "cuda_speedup_vs_vllm": ratio}
        for label, item in (("CUDA candidate", c), ("vLLM semantic", v)):
            table.append(f"| {display} (`{shape}`) | {label} | {item['p50']:.3f} | {item['p95']:.3f} | {item['cv']:.4f} | {item['tokens_s']:,.0f} | {item['routes_s']:,.0f} | {item['gbps']:.2f} | {item['tflops']:.3f} |")
        table.append(f"| ↳ CUDA candidate speedup vs vLLM | — | **{ratio:.3f}×** | — | — | — | — | — | — |")
    summary["geomean_cuda_speedup_vs_vllm"] = math.prod(ratios) ** (1 / len(ratios))
    for profile_name, case_id in PROFILE_CASES:
        base = RUN / "profile" / profile_name
        cuda_sum = base / "cuda_stats_cuda_gpu_kern_sum.csv"
        vllm_sum = base / "vllm_stats_cuda_gpu_kern_sum.csv"
        timeline(base / "cuda_stats_cuda_gpu_trace.csv", analysis / f"{profile_name}_cuda_timeline.svg", f"{profile_name}: CUDA candidate")
        timeline(base / "vllm_stats_cuda_gpu_trace.csv", analysis / f"{profile_name}_vllm_timeline.svg", f"{profile_name}: vLLM semantic")
        summary["profiles"][profile_name] = {
            "case_id": case_id, "cuda_nsys": nsys_hotspot(cuda_sum), "vllm_nsys": nsys_hotspot(vllm_sum),
            "cuda_ncu": ncu_candidate(base / "cuda_candidate/cuda_candidate_grouped_basic_raw.csv"),
            "vllm_ncu": {"status": "not_collected", "reason": "The vLLM Linux container has no Nsight Compute CLI; the available NCU 2026.2.1 is a Windows executable and cannot profile the container process."},
        }
    (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    # Preserve raw release evidence, manifests and profiler reports needed to reproduce every statement.
    for source in (candidate_path, pathlib.Path(str(candidate_path) + ".manifest.json"), vllm_path, pathlib.Path(str(vllm_path) + ".manifest.json"), ROOT / "third_party/vllm_moe/SOURCE_MANIFEST.json"):
        copy_file(source, artifacts / source.name)
    for profile_name, _ in PROFILE_CASES:
        source = RUN / "profile" / profile_name
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in {".nsys-rep", ".sqlite", ".csv", ".ncu-rep"}:
                copy_file(path, artifacts / "profile" / profile_name / path.relative_to(source))
    ncu_rows = []
    for name, profile in summary["profiles"].items():
        ncu = profile["cuda_ncu"]
        ncu_rows.append(f"| {name} | {ncu.get('block', 'not_collected')} | {ncu.get('grid', 'not_collected')} | {ncu.get('active_warps_pct', 'not_collected')} | {ncu.get('sm_pct', 'not_collected')} | {ncu.get('dram_pct', 'not_collected')} | {ncu.get('l1_pct', 'not_collected')} | {ncu.get('l2_pct', 'not_collected')} | {ncu.get('registers_per_thread', 'not_collected')} | {ncu.get('shared_mem_kib', 'not_collected')} | {ncu.get('waves_per_sm', 'not_collected')} |")
    profile_sections = []
    for name, _ in PROFILE_CASES:
        p = summary["profiles"][name]
        profile_sections.append(f"### {name}\n\n- CUDA hotspot: `{p['cuda_nsys'].get('name', 'not_collected')}` — {p['cuda_nsys'].get('pct', 'not_collected')}% GPU time.\n- vLLM hotspot: `{p['vllm_nsys'].get('name', 'not_collected')}` — {p['vllm_nsys'].get('pct', 'not_collected')}% GPU time.\n- [CUDA lane](analysis/{name}_cuda_timeline.svg) · [vLLM lane](analysis/{name}_vllm_timeline.svg)\n")
    report = f"""# RaggedRoute CUDA Candidate vs vLLM 生产语义 MoE L3 对照实验

> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}。发布性能来自未 profile 的 CUDA Event release；NSYS/NCU 只用于机制诊断，绝不作为速度比来源。

## 结论

在固定的 strict FP32 / IEEE、Top-2、E≤64 和相同四个真实 MoE 代理场景下，RaggedRoute CUDA candidate 的跨场景几何平均速度为 vLLM-semantic adapted chain 的 **{summary['geomean_cuda_speedup_vs_vllm']:.3f}×**。这是 production-reference comparison：两端的数值合同与 chain 输入一致，但运行时/编译器栈、vLLM 的 alignment padding 与 fused layout 仍是生产实现的一部分，不能误写为 binary-identical vLLM 或自动推广到完整 vLLM serving。

## 实验合同与来源

| 项目 | 固定值 |
|---|---|
| RaggedRoute candidate | `cuda_all_candidates_chain`：Dense router GEMM → Top-K → fused Histogram/Scan → Permute → Grouped GEMM → Unpermute |
| vLLM semantic | router GEMM → `vllm_topk_softmax` → `moe_align_block_size` / sort metadata → vLLM Triton `fused_moe_kernel` → weighted reduction |
| vLLM upstream | `https://github.com/vllm-project/vllm` @ `66b3c0e61f1e477820212201adf1ed871df7ee98`，Apache-2.0；[source manifest](artifacts/SOURCE_MANIFEST.json) |
| GPU | RTX 3080, sm_86, 68 SM, 10 GiB |
| math | FP32 strict IEEE；vLLM container 设置 `TRITON_F32_DEFAULT=ieee` |
| release | 20 warmups, 30 samples, 3 kernel repeats/sample, 3 independent processes, fixed seed `20260805` |
| timed boundary | 输入已在 device；预分配 workspace；计入 routing、alignment/sort metadata、Grouped GEMM、weighted reduction；不计 allocation/JIT/autotune/CPU oracle |

vLLM adapter 将 RaggedRoute `[E,K,N]` expert 权重转为 vLLM `[E,N,K]`，且计入 vLLM device-side alignment padding 与 metadata 工作。vLLM 把 permutation/Grouped GEMM 组织为 fused kernel，故七阶段是语义映射而非逐 kernel 一一对应。所有阶段输出均在 each-process timed run 前通过独立 oracle：Top-K IDs/weights、alignment 元数据、Grouped GEMM、最终加权输出。

## 发布性能（唯一速度结论来源）

{chr(10).join(table)}

所有 24 条 record 均 `validation.ok=true`；每 case/线路都恰好含 process runs `{{1,2,3}}`。原始结果：[CUDA JSONL](artifacts/cuda_candidate_release.jsonl)、[vLLM JSONL](artifacts/release.jsonl)、[CUDA manifest](artifacts/cuda_candidate_release.jsonl.manifest.json)、[vLLM manifest](artifacts/release.jsonl.manifest.json)。

## NSYS 机制证据

{chr(10).join(profile_sections)}

NSYS 显示两条线路均由 Grouped GEMM 主导；vLLM 的生产路径还显式可见 `topkGating`、`moe_align_block_size` / `count_and_sort_expert_tokens`、以及 weighted reduction。vLLM alignment padding 和 sorted-token metadata 未被移出 timing boundary。CUDA candidate 以 fused Histogram/Scan 和 token-owned permute 避免额外生产 metadata kernels，但其 Grouped GEMM 仍是主要优化目标。

## NCU basic：Grouped GEMM（诊断）

| 场景 | block | grid | active warps % | SM % | DRAM % | L1 % | L2 % | regs/thread | shared KiB | waves/SM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ncu_rows)}

CUDA candidate 的 NCU basic 已采集，原始 `.ncu-rep` 与 CSV 位于 `artifacts/profile/`。vLLM container 的 NCU 是 `not_collected`：容器未安装 Linux NCU，现有 NCU 2026.2.1 是 Windows CLI，无法 attach 到 Linux container。该限制没有用推测的 counters 填充，也不影响 release latency 的实测结论。没有升级 detailed，因为 basic 已足以表明 CUDA candidate 的 Grouped GEMM 受 SIMT FP32 吞吐/活跃 warp 与专家小 M tail 的共同制约；vLLM 侧只可依据 NSYS 作 kernel-time 诊断。

## 可复现性与边界

- `strict_semantic_comparison`：输入 seed、FP32/IEEE、Top-2、T/E/K/N、device-side metadata、steady-state 时间边界相同；vLLM adapter correctness gate 通过。
- `production_reference_comparison`：保留 vLLM 的 block alignment、排序/临时 buffer、Triton fused kernel 与 `[E,N,K]` layout；因此是 source-faithful-adapted，不是将 vLLM server/runtime 原样嵌入。
- 不纳入 cuBLAS 作为完整七阶段链基准：它不提供 Top-K、alignment/histogram、scan、permute 和 weighted unpermute。
- 未 profile 的 release 数据是唯一性能事实；NSYS/NCU 的 replay/trace 开销不可与表内 p50 混算。

## 工件校验

[hash manifest](analysis/SHA256SUMS.json) · [machine-readable summary](analysis/summary.json) · profiler evidence 在 [artifacts/profile](artifacts/profile)。
"""
    (output / "RaggedRoute_vLLM_semantic_MoE_L3_report.md").write_text(report, encoding="utf-8", newline="\n")
    (analysis / "SHA256SUMS.json").write_text(json.dumps(digest_tree(output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output / "RaggedRoute_vLLM_semantic_MoE_L3_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
