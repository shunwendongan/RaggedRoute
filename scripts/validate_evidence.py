#!/usr/bin/env python3
"""Read-only validation for raggedroute.evidence_bundle.v2 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
import zipfile


SCHEMA_VERSION = "raggedroute.evidence_bundle.v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_FILES = {"manifest.json", "SHA256SUMS"}
REQUIRED = {
    "schema_version", "run_id", "kind", "provenance", "hardware",
    "toolchain", "semantic_contract", "commands", "validation", "decision",
    "release_asset", "files",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_manifest(manifest: dict, bundle: pathlib.Path) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED - set(manifest)
    if missing:
        fail(errors, f"{bundle.name}: missing keys: {sorted(missing)}")
        return errors
    if manifest["schema_version"] != SCHEMA_VERSION:
        fail(errors, f"{bundle.name}: unsupported schema_version")
    provenance = manifest.get("provenance", {})
    if not isinstance(provenance.get("source_git_sha"), str) or len(provenance["source_git_sha"]) < 7:
        fail(errors, f"{bundle.name}: invalid source_git_sha")
    if provenance.get("git_clean") not in (True, False, None):
        fail(errors, f"{bundle.name}: invalid git_clean")
    commands = manifest.get("commands")
    if not isinstance(commands, list) or not commands:
        fail(errors, f"{bundle.name}: commands must be non-empty")
    asset = manifest.get("release_asset", {})
    for key in ("tag", "name", "url", "bytes", "sha256", "immutable"):
        if key not in asset:
            fail(errors, f"{bundle.name}: release_asset missing {key}")
    if asset.get("immutable") is not True:
        fail(errors, f"{bundle.name}: release asset must be immutable")
    if not SHA256_RE.fullmatch(str(asset.get("sha256", ""))):
        fail(errors, f"{bundle.name}: invalid release archive sha256")
    if asset.get("name") and asset["name"] not in str(asset.get("url", "")):
        fail(errors, f"{bundle.name}: release URL does not name archive")

    declared_git: set[str] = set()
    declared_archive: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(manifest.get("files", [])):
        label = f"{bundle.name}: files[{index}]"
        if not isinstance(item, dict):
            fail(errors, f"{label} is not an object")
            continue
        for key in ("path", "role", "bytes", "sha256", "storage"):
            if key not in item:
                fail(errors, f"{label} missing {key}")
        storage = item.get("storage")
        path = str(item.get("path", "")).replace("\\", "/")
        key = (storage, path)
        if key in seen:
            fail(errors, f"{label} duplicate {storage}:{path}")
        seen.add(key)
        if storage not in {"git", "release", "unavailable"}:
            fail(errors, f"{label} invalid storage")
            continue
        if not SHA256_RE.fullmatch(str(item.get("sha256", ""))):
            fail(errors, f"{label} invalid sha256")
        if not isinstance(item.get("bytes"), int) or item.get("bytes", -1) < 0:
            fail(errors, f"{label} invalid bytes")
        if storage == "git":
            declared_git.add(path)
            target = bundle / pathlib.PurePosixPath(path)
            if not target.is_file():
                fail(errors, f"{bundle.name}: missing git file {path}")
            elif target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]:
                fail(errors, f"{bundle.name}: git file mismatch {path}")
        elif storage == "release":
            archive_path = item.get("archive_path")
            if not archive_path:
                fail(errors, f"{label} release item missing archive_path")
            else:
                declared_archive.add(str(archive_path).replace("\\", "/"))
        elif not item.get("reason"):
            fail(errors, f"{label} unavailable item missing reason")

    actual_git = {
        os.path.relpath(path, bundle).replace("\\", "/")
        for path in bundle.rglob("*")
        if path.is_file() and os.path.relpath(path, bundle).replace("\\", "/") not in CONTROL_FILES
    }
    for path in sorted(actual_git - declared_git):
        fail(errors, f"{bundle.name}: undeclared git file {path}")
    for path in sorted(declared_git - actual_git):
        fail(errors, f"{bundle.name}: declared git file absent {path}")
    return errors


def validate_sums(bundle: pathlib.Path) -> list[str]:
    errors: list[str] = []
    sums = bundle / "SHA256SUMS"
    if not sums.is_file():
        return [f"{bundle.name}: missing SHA256SUMS"]
    declared: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            fail(errors, f"{bundle.name}: malformed SHA256SUMS line")
            continue
        target = bundle / pathlib.PurePosixPath(relative)
        declared.add(relative)
        if not target.is_file():
            fail(errors, f"{bundle.name}: checksum target missing {relative}")
        elif sha256_file(target) != expected:
            fail(errors, f"{bundle.name}: checksum mismatch {relative}")
    actual = {
        os.path.relpath(path, bundle).replace("\\", "/")
        for path in bundle.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    }
    for relative in sorted(actual - declared):
        fail(errors, f"{bundle.name}: checksum missing for {relative}")
    return errors


def validate_unpermute_derivations(bundle: pathlib.Path, zipped: zipfile.ZipFile) -> list[str]:
    errors: list[str] = []
    for run_name in ("formal-run1", "formal-rerun"):
        aggregate_name = next((
            name for name in zipped.namelist()
            if f"/{run_name}/" in name and name.endswith("aggregate.json")
        ), None)
        comparison_path = next((bundle / "benchmark" / run_name).glob("*.comparison.json"), None)
        csv_path = bundle / "benchmark" / run_name / "throughput-comparison.csv"
        if not aggregate_name or comparison_path is None or not csv_path.is_file():
            errors.append(f"{bundle.name}: incomplete Unpermute derivation inputs for {run_name}")
            continue
        aggregate = json.loads(zipped.read(aggregate_name))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        selected = {
            (item["case_id"], item["measurement_level"], item["cache_mode"]): item
            for item in comparison["comparisons"]
            if item["candidate_variant"] == "cuda_warp_token_vec4"
        }
        groups = {
            (item["case_id"], item["measurement_level"], item["cache_mode"], item["variant"]): item
            for item in aggregate["groups"]
        }
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 72 or len(selected) != 72:
            errors.append(f"{bundle.name}: {run_name} must contain 72 selected-candidate rows")
            continue
        for row in rows:
            key = (row["case_id"], row["measurement_level"], row["cache_mode"])
            item = selected.get(key)
            if item is None:
                errors.append(f"{bundle.name}: {run_name} CSV contains unknown row {key}")
                continue
            baseline = groups[key + (item["baseline_variant"],)]
            candidate = groups[key + (item["candidate_variant"],)]
            exact = {
                "baseline_p50_us": item["baseline_latency_us"],
                "candidate_p50_us": item["candidate_latency_us"],
                "p50_speedup": item["speedup"],
                "baseline_p95_us": item["baseline_p95_us"],
                "candidate_p95_us": item["candidate_p95_us"],
                "p95_ratio": item["p95_ratio"],
                "baseline_cv": item["baseline_all_samples_cv"],
                "candidate_cv": item["candidate_all_samples_cv"],
            }
            if any(abs(float(row[name]) - float(value)) > 1e-12 for name, value in exact.items()):
                errors.append(f"{bundle.name}: {run_name} derived CSV mismatch for {key}")
                break
            if int(row["logical_bytes"]) != int(candidate["work"]["logical_bytes"]):
                errors.append(f"{bundle.name}: {run_name} logical_bytes mismatch for {key}")
                break
            if baseline["variant"] != "vllm_finalize_routing" or candidate["variant"] != "cuda_warp_token_vec4":
                errors.append(f"{bundle.name}: {run_name} derivation used the wrong variant pair")
                break
    return errors


def validate_archive(manifest: dict, bundle: pathlib.Path, archive: pathlib.Path) -> list[str]:
    errors: list[str] = []
    asset = manifest["release_asset"]
    if not archive.is_file():
        return [f"archive missing: {archive}"]
    if archive.stat().st_size != asset["bytes"] or sha256_file(archive) != asset["sha256"]:
        return [f"archive size/hash mismatch: {archive}"]
    expected = {
        item["archive_path"]: item
        for item in manifest["files"] if item["storage"] == "release"
    }
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist()) - {"ARCHIVE-MANIFEST.json", "ARCHIVE-SHA256SUMS"}
        if names != set(expected):
            fail(errors, f"{archive.name}: archive inventory mismatch")
        for name, item in expected.items():
            data = zipped.read(name)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                fail(errors, f"{archive.name}: member mismatch {name}")
        if "unpermute" in manifest["run_id"]:
            errors.extend(validate_unpermute_derivations(bundle, zipped))
    return errors


def validate_root(root: pathlib.Path, archives: pathlib.Path | None = None) -> list[str]:
    errors: list[str] = []
    bundles = sorted(path for path in root.iterdir() if path.is_dir())
    for bundle in bundles:
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            fail(errors, f"{bundle.name}: missing manifest.json")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(errors, f"{bundle.name}: invalid manifest: {exc}")
            continue
        errors.extend(validate_manifest(manifest, bundle))
        errors.extend(validate_sums(bundle))
        if archives is not None:
            errors.extend(validate_archive(manifest, bundle, archives / manifest["release_asset"]["name"]))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=pathlib.Path, default=pathlib.Path("docs/reports/artifacts"))
    parser.add_argument("--archives", type=pathlib.Path)
    args = parser.parse_args()
    errors = validate_root(args.artifacts_root, args.archives)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"evidence validation failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    count = sum(1 for path in args.artifacts_root.iterdir() if path.is_dir())
    print(f"evidence validation passed: {count} bundle(s), 0 mismatch, 0 undeclared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
