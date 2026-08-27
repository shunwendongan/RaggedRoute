#!/usr/bin/env python3
"""Build compact, Git-friendly v3 evidence from promotion decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import pathlib
import shutil
import sys
import zipfile
from typing import Any


ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decision(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "raggedroute.promotion_decision.v1":
        raise ValueError(f"not a promotion decision: {path}")
    if document.get("decision") not in {"promote", "reject", "insufficient_evidence"}:
        raise ValueError(f"invalid decision state: {path}")
    return document


def color(speedup: float) -> str:
    if speedup >= 1.03:
        return "#3ca370"
    if speedup >= 1.0:
        return "#9ac46a"
    if speedup >= 0.97:
        return "#e5bd55"
    return "#d7655b"


def svg(decisions: list[tuple[str, dict[str, Any]]]) -> str:
    rows = [
        (scope, row["case_id"], float(row["p50_speedup"]))
        for scope, decision in decisions
        for row in decision.get("rows", [])
    ]
    width = 1160
    row_height = 24
    height = 74 + max(1, len(rows)) * row_height
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#f3f4f6}.muted{fill:#aeb6c4}</style>',
        '<text x="20" y="30" font-size="20" font-weight="600">RaggedRoute CUDA v3 paired p50 speedup</text>',
        '<text x="20" y="52" font-size="12" class="muted">green >=1.03x; yellow 0.97-1.00x; red &lt;0.97x</text>',
    ]
    if not rows:
        body.append('<text x="20" y="82" font-size="14" class="muted">No paired timing rows collected.</text>')
    for index, (scope, case, speedup) in enumerate(rows):
        y = 68 + index * row_height
        body.append(f'<text x="20" y="{y + 16}" font-size="12">{html.escape(scope)}</text>')
        body.append(f'<text x="160" y="{y + 16}" font-size="12">{html.escape(case)}</text>')
        body.append(f'<rect x="1010" y="{y + 3}" width="120" height="18" rx="3" fill="{color(speedup)}"/>')
        body.append(f'<text x="1070" y="{y + 16}" font-size="12" text-anchor="middle">{speedup:.4f}x</text>')
    body.append("</svg>")
    return "\n".join(body) + "\n"


def write_archive(output: pathlib.Path, archive: pathlib.Path, files: list[pathlib.Path]) -> None:
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite archive: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for path in sorted(files, key=lambda item: item.relative_to(output).as_posix()):
            info = zipfile.ZipInfo(path.relative_to(output).as_posix(), ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, path.read_bytes())
    pathlib.Path(str(archive) + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="utf-8", newline="\n"
    )


def build(inputs: list[tuple[str, pathlib.Path]], output: pathlib.Path,
          archive: pathlib.Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True)
    decisions_dir = output / "decisions"
    decisions_dir.mkdir()
    decisions: list[tuple[str, dict[str, Any]]] = []
    copied: list[pathlib.Path] = []
    for scope, source in inputs:
        decision = load_decision(source)
        target = decisions_dir / f"{scope}.json"
        shutil.copyfile(source, target)
        decisions.append((scope, decision))
        copied.append(target)

    csv_path = output / "promotion_summary.csv"
    fields = [
        "scope", "decision", "baseline", "candidate", "paired_shapes",
        "ratio_of_sums_p50_speedup", "shape_balanced_geometric_mean_speedup",
        "shape_win_fraction", "maximum_p50_regression_fraction",
        "maximum_p95_regression_fraction", "maximum_workspace_growth_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for scope, decision in decisions:
            writer.writerow({
                "scope": scope,
                "decision": decision["decision"],
                "baseline": decision["baseline"],
                "candidate": decision["candidate"],
                **decision["summary"],
            })

    heatmap_path = output / "shape_heatmap.svg"
    heatmap_path.write_text(svg(decisions), encoding="utf-8", newline="\n")
    report_path = output / "REPORT.md"
    report_lines = [
        "# RaggedRoute CUDA v3 compact evidence",
        "",
        "> Promotion decisions use clean, uninstrumented Release timing. NSYS/NCU durations are diagnostic only and raw profiler reports remain outside this compact bundle.",
        "",
        "| Scope | Decision | Baseline | Candidate | Paired shapes | Ratio-of-sums | Geomean | Win coverage |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for scope, decision in decisions:
        summary = decision["summary"]
        def fmt(name: str, suffix: str = "") -> str:
            value = summary.get(name)
            return "not_collected" if value is None else f"{float(value):.4f}{suffix}"
        report_lines.append(
            f"| {scope} | `{decision['decision']}` | `{decision['baseline']}` | "
            f"`{decision['candidate']}` | {summary['paired_shapes']} | "
            f"{fmt('ratio_of_sums_p50_speedup', 'x')} | "
            f"{fmt('shape_balanced_geometric_mean_speedup', 'x')} | "
            f"{fmt('shape_win_fraction')} |"
        )
    report_lines.extend([
        "",
        "![Per-shape p50 heatmap](shape_heatmap.svg)",
        "",
        "A `synthetic_fixture` route trace can validate the input and evaluator pipeline, but cannot satisfy the real-trace policy. Such a result must remain `insufficient_evidence`.",
        "",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    evidence_files = copied + [csv_path, heatmap_path, report_path]
    manifest = {
        "schema_version": "raggedroute.compact_v3_evidence.v1",
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(evidence_files, key=lambda item: item.relative_to(output).as_posix())
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    evidence_files.append(manifest_path)
    sums_path = output / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in sorted(evidence_files, key=lambda item: item.relative_to(output).as_posix())
        ),
        encoding="utf-8",
        newline="\n",
    )
    evidence_files.append(sums_path)
    if archive:
        write_archive(output, archive, evidence_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", action="append", nargs=2, metavar=("SCOPE", "PATH"),
                        required=True)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--archive", type=pathlib.Path)
    args = parser.parse_args()
    inputs = [(scope, pathlib.Path(path)) for scope, path in args.decision]
    scopes = [scope for scope, _ in inputs]
    if len(scopes) != len(set(scopes)):
        raise ValueError("decision scopes must be unique")
    build(inputs, args.output_dir, args.archive)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
