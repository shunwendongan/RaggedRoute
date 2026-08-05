#!/usr/bin/env python3
"""Repository-wide, dependency-free CI quality gates."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import unquote

import validate_evidence


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_PUBLISHED_BYTES = 5 * 1024 * 1024
OPERATORS = {
    "dense_gemm", "topk_gate", "histogram", "scan", "permute",
    "grouped_gemm", "unpermute",
}
SOURCE_ROLES = {"cuda_naive", "cuda_candidate", "library_baseline", "cpu_reference", "triton"}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:(?:\\\\|/)")


def tracked_files() -> list[pathlib.Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def check_json(files: list[pathlib.Path]) -> list[str]:
    errors = []
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON: {path.relative_to(ROOT).as_posix()}: {exc}")
    return errors


def check_markdown_links(files: list[pathlib.Path]) -> list[str]:
    errors = []
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            target = unquote(target)
            if not (path.parent / target).resolve().exists():
                errors.append(f"broken Markdown link: {path.relative_to(ROOT).as_posix()} -> {target}")
    return errors


def check_layout(files: list[pathlib.Path]) -> list[str]:
    errors = []
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    for operator in OPERATORS:
        old = f"src/{operator}/baseline.cu"
        if old in relative:
            errors.append(f"legacy baseline path is tracked: {old}")
        baseline = f"src/{operator}/cuda_naive/baseline.cu"
        if baseline not in relative:
            errors.append(f"missing normalized baseline: {baseline}")
    for item in sorted(relative):
        parts = pathlib.PurePosixPath(item).parts
        if len(parts) >= 3 and parts[0] == "src" and parts[1] in OPERATORS and item.endswith((".cu", ".cpp", ".py")):
            if parts[2] not in SOURCE_ROLES and not (len(parts) == 3 and parts[2] == "operator.cpp"):
                errors.append(f"operator source has no recognized role directory: {item}")
    # A measured fused Histogram->Scan implementation is now a first-class
    # SM86 candidate. Its evidence and promotion decision live in docs/reports;
    # the normalized cuda_candidate role remains valid for the source kernel.
    direct_configs = sorted(
        item for item in relative
        if pathlib.PurePosixPath(item).parent.as_posix() == "configs" and item.endswith(".json")
    )
    errors.extend(f"legacy flat config is tracked: {item}" for item in direct_configs)
    legacy_reference = re.compile(
        r"configs[\\/](?:benchmark_|profile_)|"
        r"src[\\/](?:dense_gemm|topk_gate|histogram|scan|permute|grouped_gemm|unpermute)[\\/]baseline\.cu"
    )
    for path in files:
        item = path.relative_to(ROOT).as_posix()
        if item.startswith("docs/reports/artifacts/") or path.suffix.lower() not in {".md", ".py", ".json", ".txt", ".cmake"}:
            continue
        if legacy_reference.search(path.read_text(encoding="utf-8")):
            errors.append(f"active file references a legacy source/config path: {item}")
    return errors


def check_gitkeep(files: list[pathlib.Path]) -> list[str]:
    errors = []
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    for item in sorted(path for path in relative if pathlib.PurePosixPath(path).name == ".gitkeep"):
        parent = pathlib.PurePosixPath(item).parent
        siblings = [path for path in relative if pathlib.PurePosixPath(path).parent == parent and path != item]
        if siblings:
            errors.append(f"meaningless .gitkeep in non-empty tracked directory: {item}")
    return errors


def check_large_files(files: list[pathlib.Path]) -> list[str]:
    errors = []
    for path in files:
        size = path.stat().st_size
        if size > MAX_PUBLISHED_BYTES:
            errors.append(
                f"unregistered large tracked file ({size} bytes > {MAX_PUBLISHED_BYTES}): "
                f"{path.relative_to(ROOT).as_posix()}"
            )
    return errors


def check_public_manifests(files: list[pathlib.Path]) -> list[str]:
    errors = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative == "docs/reports/artifacts/index.json" or (
            relative.startswith("docs/reports/artifacts/") and relative.endswith("/manifest.json")
        ):
            text = path.read_text(encoding="utf-8")
            if ABSOLUTE_WINDOWS_PATH.search(text):
                errors.append(f"public evidence manifest contains a developer-machine absolute path: {relative}")
    return errors


def run_checks(selected: set[str]) -> list[str]:
    files = tracked_files()
    checks = {
        "json": lambda: check_json(files),
        "links": lambda: check_markdown_links(files),
        "layout": lambda: check_layout(files),
        "gitkeep": lambda: check_gitkeep(files),
        "large-files": lambda: check_large_files(files),
        "public-manifests": lambda: check_public_manifests(files),
        "evidence": lambda: validate_evidence.validate_root(ROOT / "docs" / "reports" / "artifacts"),
    }
    errors = []
    for name, function in checks.items():
        if "all" not in selected and name not in selected:
            continue
        found = function()
        print(f"{name}: {'PASS' if not found else 'FAIL'}")
        errors.extend(found)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="append",
        choices=["all", "json", "links", "layout", "gitkeep", "large-files", "public-manifests", "evidence"],
        default=[],
    )
    args = parser.parse_args()
    errors = run_checks(set(args.check or ["all"]))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"repository checks failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("repository checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
