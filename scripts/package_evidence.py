#!/usr/bin/env python3
"""Create immutable raw archives and migrate legacy evidence bundles to v2.

The migration is deliberately fail-closed: an existing archive or v2 bundle is
never overwritten, and every recovered legacy file must match its recorded hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import shutil
import sys
import tempfile
import zipfile
from typing import Any, Iterable, NamedTuple


SCHEMA_VERSION = "raggedroute.evidence_bundle.v2"
RAW_SUFFIXES = (".jsonl", ".jsonl.manifest.json", ".ncu-rep", ".nsys-rep", ".sqlite")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def normalize_text_file(path: pathlib.Path) -> None:
    data = path.read_bytes()
    if b"\x00" not in data and b"\r\n" in data:
        path.write_bytes(data.replace(b"\r\n", b"\n"))


def is_raw(relative: str) -> bool:
    lower = relative.lower()
    name = pathlib.PurePosixPath(lower).name
    return (
        lower.endswith(RAW_SUFFIXES)
        or name.endswith("aggregate.json")
        or name.endswith(".aggregate-v2.json")
        or "manifest" in name
    )


def role_for(relative: str) -> str:
    lower = relative.lower()
    if lower.endswith(".jsonl"):
        return "raw_benchmark_records"
    if "aggregate" in lower and lower.endswith(".json"):
        return "full_benchmark_aggregate"
    if lower.endswith(".ncu-rep"):
        return "ncu_report"
    if lower.endswith(".nsys-rep"):
        return "nsys_report"
    if lower.endswith(".sqlite"):
        return "profiler_database"
    if "comparison" in lower or "summary" in lower or "promotion" in lower:
        return "comparison_or_decision"
    if lower.endswith(".csv") and ("ncu" in lower or "nsys" in lower or "metric" in lower):
        return "normalized_profiler_metrics"
    if lower.endswith(".csv"):
        return "compact_benchmark_summary"
    if lower.endswith(".md"):
        return "report"
    if "manifest" in lower:
        return "run_manifest"
    if "sanitizer" in lower or lower.endswith("check.log"):
        return "correctness_validation"
    return "supporting_evidence"


def safe_path(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"^[A-Za-z]:/", "", value).lstrip("/")
    parts = [part for part in value.split("/") if part not in ("", ".", "..")]
    return "/".join(parts)


def normalized_external_path(value: str) -> str:
    relative = safe_path(value)
    if relative.startswith("local/"):
        relative = relative[6:]
    if relative.startswith("out/"):
        relative = relative[4:]
    if relative.startswith("profile/dense-gemm-"):
        relative = "benchmark/" + pathlib.PurePosixPath(relative).name
    return relative


def parse_sums(path: pathlib.Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            result.append((digest, relative.replace("\\", "/")))
    return result


def nested_values(value: Any, key_names: set[str]) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in key_names:
                yield key, item
            yield from nested_values(item, key_names)
    elif isinstance(value, list):
        for item in value:
            yield from nested_values(item, key_names)


def first_value(documents: list[Any], keys: list[str]) -> Any:
    wanted = set(keys)
    found: dict[str, Any] = {}
    for document in documents:
        for key, value in nested_values(document, wanted):
            found.setdefault(key, value)
    for key in keys:
        if key in found:
            return found[key]
    return None


def load_json(path: pathlib.Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def metadata_documents(bundle: pathlib.Path, legacy: dict) -> list[Any]:
    documents: list[Any] = [legacy]
    candidates = sorted(
        path for path in bundle.rglob("*.json")
        if ("aggregate" in path.name or path.name in {"environment.json", "summary.json"})
    )
    for path in candidates:
        document = load_json(path)
        if document is not None:
            documents.append(document)
        if len(documents) >= 6:
            break
    return documents


def bool_clean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "false"
    return None


def public_tool_value(value: Any) -> Any:
    if not isinstance(value, str) or not re.search(r"[A-Za-z]:[\\/]", value):
        return value
    match = re.search(r"Nsight (?:Compute|Systems) ([0-9.]+)", value)
    if match:
        return match.group(1)
    return pathlib.PureWindowsPath(value).name


def infer_operator(run_id: str, legacy: dict) -> str:
    if legacy.get("operator"):
        return str(legacy["operator"])
    for name in ("dense-gemm", "grouped-gemm", "topk", "histogram", "unpermute", "permute", "scan", "triton"):
        if name in run_id:
            return name.replace("-", "_")
    return "multi_operator"


def known_decision(run_id: str, fallback: str) -> str:
    if "histogram-candidate" in run_id:
        return "promoted_shape_dispatch"
    if "topk-gate" in run_id or "scan-sm86-research" in run_id:
        return "rejected_not_promoted"
    if "grouped-gemm" in run_id:
        return "benchmark_only_not_promoted"
    if "unpermute" in run_id:
        return "rejected_not_promoted"
    if "permute-selected" in run_id:
        return "research_candidate_selected_not_promoted"
    if "permute-sm86-candidates" in run_id:
        return "research_candidates_not_promoted"
    if "dense-gemm-opt" in run_id or "dense-gemm-v3" in run_id:
        return "rejected_not_promoted"
    if "dense-gemm-levels" in run_id:
        return "benchmark_only_not_promoted"
    if "triton-three-levels" in run_id:
        return "reference_only"
    if "naive-profile" in run_id or "scan-library-main" in run_id:
        return "baseline_only"
    return fallback


def build_metadata(run_id: str, legacy: dict, documents: list[Any]) -> dict[str, Any]:
    sha = first_value(documents, [
        "source_git_sha", "release_git_sha", "git_sha", "candidate_checkpoint_git_sha",
        "base_git_sha", "build_git_sha", "implementation_revision",
    ])
    if not isinstance(sha, str) or len(sha) < 7:
        match = re.search(r"(?:Z-|main-)([0-9a-f]{7,40})", run_id)
        sha = match.group(1) if match else "unavailable"
    dirty = first_value(documents, ["build_git_dirty", "source_git_dirty", "git_dirty"])
    clean = bool_clean(dirty)
    gpu_name = first_value(documents, ["gpu_name"])
    compute = first_value(documents, ["compute_capability"])
    sm_count = first_value(documents, ["sm_count"])
    gpu_uuid = first_value(documents, ["gpu_uuid"])
    legacy_gpu = legacy.get("gpu")
    if isinstance(legacy_gpu, dict):
        gpu_name = gpu_name or legacy_gpu.get("name")
        gpu_uuid = gpu_uuid or legacy_gpu.get("uuid")
        compute = compute or legacy_gpu.get("compute_capability")
        sm_count = sm_count or legacy_gpu.get("sm_count")
    target = legacy.get("target")
    if isinstance(target, dict):
        gpu_name = gpu_name or target.get("gpu")
        compute = compute or target.get("compute_capability")
        sm_count = sm_count or target.get("sm_count")
    cuda = first_value(documents, ["cuda_compiler", "cuda"])
    driver = first_value(documents, ["cuda_driver", "driver"])
    ncu = first_value(documents, ["ncu"])
    nsys = first_value(documents, ["nsys"])
    build_type = first_value(documents, ["build_type"])
    operator = infer_operator(run_id, legacy)
    dtype = first_value(documents, ["dtype"])
    accumulator = first_value(documents, ["accumulator_dtype"])
    math_mode = first_value(documents, ["math_mode"])
    levels = sorted({
        str(value) for document in documents
        for _, value in nested_values(document, {"measurement_level"})
    })

    command_contract = legacy.get("command_contract")
    commands = []
    if isinstance(command_contract, dict):
        for role, command in sorted(command_contract.items()):
            commands.append({"role": role, "status": "recorded", "command": str(command)})
    if not commands:
        commands = [{
            "role": "historical_collection",
            "status": "unavailable",
            "reason": "The legacy bundle did not preserve an exact command line.",
        }]

    validation = legacy.get("validation")
    if isinstance(validation, dict):
        validation_doc = {"status": "passed" if validation.get("all_passed") else "historical", "details": validation}
    else:
        validation_doc = {
            "status": "historical",
            "reason": "Validation outputs are inventoried as files; the legacy manifest had no normalized result object.",
        }
    decision = legacy.get("decision")
    if decision is None and "promotion_eligible" in legacy:
        decision = "reference_only" if not legacy["promotion_eligible"] else "eligible"
    if decision is None:
        policy = legacy.get("policy", {})
        decision = policy.get("decision") if isinstance(policy, dict) else None
    normalized_decision = known_decision(run_id, str(decision or "no_dispatch_decision_recorded"))
    decision_doc = {
        "status": "recorded",
        "value": normalized_decision,
        "profiler_duration_used_as_speedup": False,
    }
    return {
        "kind": f"{operator}_performance_evidence",
        "provenance": {
            "source_git_sha": sha,
            "git_clean": clean,
            "git_clean_status": "recorded" if clean is not None else "unavailable",
        },
        "hardware": {
            "status": "recorded" if gpu_name or compute else "unavailable",
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "compute_capability": compute,
            "sm_count": sm_count,
        },
        "toolchain": {
            "status": "recorded" if cuda or driver or ncu or nsys else "unavailable",
            "cuda": cuda,
            "driver": driver,
            "ncu": public_tool_value(ncu),
            "nsys": public_tool_value(nsys),
            "build_type": build_type,
        },
        "semantic_contract": {
            "status": "recorded" if dtype or math_mode or levels else "historical",
            "operator": operator,
            "dtype": dtype,
            "accumulator_dtype": accumulator,
            "math_mode": math_mode,
            "measurement_levels": levels,
            "measurement_rule": "Release A/B timings and profiler durations are separate evidence classes.",
        },
        "commands": commands,
        "validation": validation_doc,
        "decision": decision_doc,
    }


def resolve_recorded_path(relative: str, bundle: pathlib.Path, roots: list[pathlib.Path]) -> pathlib.Path | None:
    candidates = [bundle / pathlib.PurePosixPath(relative)]
    clean = relative[6:] if relative.startswith("local/") else relative
    for root in roots:
        candidates.extend([root / pathlib.PurePosixPath(relative), root / pathlib.PurePosixPath(clean)])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_inventory_item(item: dict, legacy: dict, roots: list[pathlib.Path]) -> pathlib.Path | None:
    raw_path = item.get("path")
    if isinstance(raw_path, str) and pathlib.Path(raw_path).is_file():
        return pathlib.Path(raw_path)
    name = item.get("name")
    if not isinstance(name, str):
        return None
    sources = [legacy.get("benchmark_source"), legacy.get("profile_source")]
    for root in roots:
        for source in sources:
            if isinstance(source, str):
                candidate = root / pathlib.PurePosixPath(source.replace("\\", "/")) / pathlib.PurePosixPath(name.replace("\\", "/"))
                if candidate.is_file():
                    return candidate
    return None


class ArchiveItem(NamedTuple):
    source: pathlib.Path
    logical_path: str
    archive_path: str
    role: str
    bytes: int
    sha256: str


def unique_archive_path(desired: str, used: set[str]) -> str:
    candidate = "raw/" + safe_path(desired)
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = pathlib.PurePosixPath(candidate).stem
    suffix = pathlib.PurePosixPath(candidate).suffix
    parent = pathlib.PurePosixPath(candidate).parent
    index = 2
    while True:
        alternate = (parent / f"{stem}-{index}{suffix}").as_posix()
        if alternate not in used:
            used.add(alternate)
            return alternate
        index += 1


def deterministic_zip(path: pathlib.Path, items: list[ArchiveItem], run_id: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory = [
        {"path": item.archive_path, "role": item.role, "bytes": item.bytes, "sha256": item.sha256}
        for item in sorted(items, key=lambda item: item.archive_path)
    ]
    archive_manifest = json_dump({
        "schema_version": "raggedroute.evidence_archive.v1",
        "run_id": run_id,
        "files": inventory,
    }).encode("utf-8")
    sums = "".join(f"{item['sha256']}  {item['path']}\n" for item in inventory).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for name, data in (("ARCHIVE-MANIFEST.json", archive_manifest), ("ARCHIVE-SHA256SUMS", sums)):
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, data)
        for item in sorted(items, key=lambda item: item.archive_path):
            info = zipfile.ZipInfo(item.archive_path, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, item.source.read_bytes())


UNPERMUTE_FIELDS = [
    "case_id", "measurement_level", "cache_mode", "T", "N", "logical_bytes",
    "baseline_p50_us", "candidate_p50_us", "p50_speedup",
    "baseline_effective_gbps", "candidate_effective_gbps",
    "baseline_tokens_per_s", "candidate_tokens_per_s",
    "baseline_elements_per_s", "candidate_elements_per_s",
    "baseline_p95_us", "candidate_p95_us", "p95_ratio", "baseline_cv", "candidate_cv",
]


def regenerate_unpermute_csv(directory: pathlib.Path) -> int:
    aggregate_path = next(directory.glob("*.aggregate.json"))
    comparison_path = next(directory.glob("*.comparison.json"))
    aggregate = load_json(aggregate_path)
    comparison = load_json(comparison_path)
    groups = {
        (group["case_id"], group["measurement_level"], group["cache_mode"], group["variant"]): group
        for group in aggregate["groups"]
    }
    rows = []
    for item in comparison["comparisons"]:
        if item["candidate_variant"] != "cuda_warp_token_vec4":
            continue
        key = (item["case_id"], item["measurement_level"], item["cache_mode"])
        baseline = groups[key + (item["baseline_variant"],)]
        candidate = groups[key + (item["candidate_variant"],)]
        config = candidate["case_config"]
        logical = int(candidate["work"]["logical_bytes"])
        baseline_us = item["baseline_latency_us"]
        candidate_us = item["candidate_latency_us"]
        tokens = int(config["T"])
        elements = tokens * int(config["N"])
        rows.append({
            "case_id": item["case_id"], "measurement_level": item["measurement_level"],
            "cache_mode": item["cache_mode"], "T": tokens, "N": int(config["N"]),
            "logical_bytes": logical, "baseline_p50_us": baseline_us,
            "candidate_p50_us": candidate_us, "p50_speedup": item["speedup"],
            "baseline_effective_gbps": round(logical / baseline_us / 1000.0, 6),
            "candidate_effective_gbps": round(logical / candidate_us / 1000.0, 6),
            "baseline_tokens_per_s": round(tokens * 1_000_000.0 / baseline_us, 2),
            "candidate_tokens_per_s": round(tokens * 1_000_000.0 / candidate_us, 2),
            "baseline_elements_per_s": round(elements * 1_000_000.0 / baseline_us, 2),
            "candidate_elements_per_s": round(elements * 1_000_000.0 / candidate_us, 2),
            "baseline_p95_us": item["baseline_p95_us"], "candidate_p95_us": item["candidate_p95_us"],
            "p95_ratio": item["p95_ratio"], "baseline_cv": item["baseline_all_samples_cv"],
            "candidate_cv": item["candidate_all_samples_cv"],
        })
    rows.sort(key=lambda row: (row["case_id"], row["measurement_level"], row["cache_mode"]))
    output = directory / "throughput-comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNPERMUTE_FIELDS, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != 72:
        raise ValueError(f"expected 72 selected-candidate rows in {directory}, got {len(rows)}")
    return len(rows)


def migrate_bundle(
    bundle: pathlib.Path, output: pathlib.Path, tag: str, repository: str,
    roots: list[pathlib.Path], repair_unpermute: bool,
) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        legacy: dict[str, Any] = {"schema_version": "missing"}
    else:
        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
        if legacy.get("schema_version") == SCHEMA_VERSION:
            raise FileExistsError(f"refusing to overwrite migrated bundle: {bundle}")
    if repair_unpermute and "unpermute" in bundle.name:
        for name in ("formal-run1", "formal-rerun"):
            regenerate_unpermute_csv(bundle / "benchmark" / name)

    documents = metadata_documents(bundle, legacy)

    old_sums = parse_sums(bundle / "SHA256SUMS")
    known_checksum_mismatches = {
        "benchmark/formal-run1/throughput-comparison.csv",
        "benchmark/formal-rerun/throughput-comparison.csv",
    }
    archive_items: list[ArchiveItem] = []
    unavailable: list[dict[str, Any]] = []
    used_archive_paths: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()

    def add_archive(source: pathlib.Path, logical: str, expected: str | None = None) -> None:
        digest = sha256_file(source)
        if expected and digest != expected:
            raise ValueError(f"recorded SHA mismatch for {logical}: {source}")
        logical_clean = normalized_external_path(logical)
        identity = (digest, logical_clean)
        if identity in seen_sources:
            return
        seen_sources.add(identity)
        archive_path = unique_archive_path(logical_clean, used_archive_paths)
        archive_items.append(ArchiveItem(source, logical_clean, archive_path, role_for(logical_clean), source.stat().st_size, digest))

    if manifest_path.is_file():
        add_archive(manifest_path, "legacy/manifest.json")

    for expected, relative in old_sums:
        source = resolve_recorded_path(relative, bundle, roots)
        if source is None:
            unavailable.append({
                "path": normalized_external_path(relative), "role": role_for(relative), "bytes": 0,
                "sha256": expected, "storage": "unavailable",
                "reason": "The file recorded by the legacy checksum inventory could not be recovered.",
            })
            continue
        actual = sha256_file(source)
        if actual != expected and relative not in known_checksum_mismatches:
            raise ValueError(f"legacy checksum mismatch for {bundle.name}/{relative}")
        if is_raw(relative):
            add_archive(source, relative, actual if relative in known_checksum_mismatches else expected)
        elif not source.is_relative_to(bundle):
            destination_relative = normalized_external_path(relative)
            destination = bundle / pathlib.PurePosixPath(destination_relative)
            if destination.exists():
                if sha256_file(destination) != actual:
                    raise FileExistsError(f"refusing to overwrite {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    inventory_groups = (
        legacy.get("raw_benchmark_inventory", []), legacy.get("raw_profiler_inventory", []),
        legacy.get("release_artifacts", []), legacy.get("raw_profiler_reports", []),
    )
    for inventory in inventory_groups:
        if not isinstance(inventory, list):
            continue
        for item in inventory:
            if not isinstance(item, dict) or not item.get("sha256"):
                continue
            logical = item.get("name") or item.get("path") or item["sha256"]
            source = resolve_inventory_item(item, legacy, roots)
            if source is not None and sha256_file(source) == item["sha256"]:
                if is_raw(str(logical)) or not item.get("committed", False):
                    add_archive(source, f"recovered/{safe_path(str(logical))}", item["sha256"])
            elif not any(entry["sha256"] == item["sha256"] for entry in unavailable):
                unavailable.append({
                    "path": f"recovered/{pathlib.PurePosixPath(safe_path(str(logical))).name}",
                    "role": role_for(str(logical)), "bytes": int(item.get("bytes", 0)),
                    "sha256": item["sha256"], "storage": "unavailable",
                    "reason": "The legacy inventory entry was not present in the supplied recovery roots.",
                })

    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path in {manifest_path, bundle / "SHA256SUMS"}:
            continue
        relative = path.relative_to(bundle).as_posix()
        if is_raw(relative):
            add_archive(path, relative)

    archive_name = f"{bundle.name}-raw.zip"
    archive_path = output / archive_name
    deterministic_zip(archive_path, archive_items, bundle.name)

    for path in sorted(bundle.rglob("*"), reverse=True):
        if path.is_file() and path not in {manifest_path, bundle / "SHA256SUMS"} and is_raw(path.relative_to(bundle).as_posix()):
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    git_files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path not in {manifest_path, bundle / "SHA256SUMS"}:
            normalize_text_file(path)
            relative = path.relative_to(bundle).as_posix()
            git_files.append({
                "path": relative, "role": role_for(relative), "bytes": path.stat().st_size,
                "sha256": sha256_file(path), "storage": "git",
            })
    release_files = [{
        "path": item.logical_path, "archive_path": item.archive_path, "role": item.role,
        "bytes": item.bytes, "sha256": item.sha256, "storage": "release",
    } for item in sorted(archive_items, key=lambda item: item.archive_path)]
    metadata = build_metadata(bundle.name, legacy, documents)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": bundle.name,
        **metadata,
        "release_asset": {
            "tag": tag, "name": archive_name,
            "url": f"https://github.com/{repository}/releases/download/{tag}/{archive_name}",
            "bytes": archive_path.stat().st_size, "sha256": sha256_file(archive_path), "immutable": True,
        },
        "files": git_files + release_files + unavailable,
        "migration": {
            "legacy_schema": legacy.get("schema_version", "missing"),
            "legacy_manifest_archive_path": "raw/legacy/manifest.json" if manifest_path.is_file() else None,
            "checksum_repairs": sorted(known_checksum_mismatches & {relative for _, relative in old_sums}),
        },
    }
    manifest_path.write_text(json_dump(manifest), encoding="utf-8", newline="\n")
    checksummed = [path for path in bundle.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    sums_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in sorted(checksummed)
    )
    (bundle / "SHA256SUMS").write_text(sums_text, encoding="utf-8", newline="\n")
    return {
        "run_id": bundle.name, "kind": metadata["kind"],
        "source_git_sha": metadata["provenance"]["source_git_sha"],
        "git_clean": metadata["provenance"]["git_clean"],
        "decision": metadata["decision"]["value"],
        "release_asset": manifest["release_asset"],
        "git_files": len(git_files), "release_files": len(release_files), "unavailable_files": len(unavailable),
    }


def refresh_git_inventory(artifacts_root: pathlib.Path) -> None:
    """Refresh only compact Git entries after a migration-tool correction.

    Release and unavailable entries remain immutable; raw archives are untouched.
    This is intentionally separate from packaging and cannot create or replace a
    bundle or archive.
    """
    for bundle in sorted(path for path in artifacts_root.iterdir() if path.is_dir()):
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"not a v2 bundle: {bundle}")
        retained = [item for item in manifest["files"] if item["storage"] != "git"]
        manifest["toolchain"]["ncu"] = public_tool_value(manifest["toolchain"].get("ncu"))
        manifest["toolchain"]["nsys"] = public_tool_value(manifest["toolchain"].get("nsys"))
        manifest["decision"]["status"] = "recorded"
        manifest["decision"]["value"] = known_decision(
            manifest["run_id"], manifest["decision"].get("value", "no_dispatch_decision_recorded")
        )
        git_files = []
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and path not in {manifest_path, bundle / "SHA256SUMS"}:
                normalize_text_file(path)
                relative = path.relative_to(bundle).as_posix()
                git_files.append({
                    "path": relative, "role": role_for(relative), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path), "storage": "git",
                })
        manifest["files"] = git_files + retained
        manifest_path.write_text(json_dump(manifest), encoding="utf-8", newline="\n")
        checksummed = [path for path in bundle.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
        (bundle / "SHA256SUMS").write_text("".join(
            f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in sorted(checksummed)
        ), encoding="utf-8", newline="\n")
    rebuild_index(artifacts_root)


def rebuild_index(artifacts_root: pathlib.Path) -> None:
    bundles = []
    release_tag = None
    for bundle in sorted(path for path in artifacts_root.iterdir() if path.is_dir()):
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        asset = manifest["release_asset"]
        release_tag = release_tag or asset["tag"]
        files = manifest["files"]
        bundles.append({
            "run_id": manifest["run_id"], "kind": manifest["kind"],
            "source_git_sha": manifest["provenance"]["source_git_sha"],
            "git_clean": manifest["provenance"]["git_clean"],
            "decision": manifest["decision"].get("value", manifest["decision"]["status"]),
            "release_asset": asset,
            "git_files": sum(item["storage"] == "git" for item in files),
            "release_files": sum(item["storage"] == "release" for item in files),
            "unavailable_files": sum(item["storage"] == "unavailable" for item in files),
        })
    (artifacts_root / "index.json").write_text(json_dump({
        "schema_version": "raggedroute.evidence_index.v1",
        "release_tag": release_tag,
        "bundles": bundles,
    }), encoding="utf-8", newline="\n")


def repack_run_manifests(
    artifacts_root: pathlib.Path, source_dir: pathlib.Path, output_dir: pathlib.Path,
) -> None:
    """Move historical run manifests from Git into newly created raw archives."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("repack output directory must differ from the source directory")
    for bundle in sorted(path for path in artifacts_root.iterdir() if path.is_dir()):
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_archive = source_dir / manifest["release_asset"]["name"]
        destination_archive = output_dir / manifest["release_asset"]["name"]
        raw_git_files = [
            path for path in bundle.rglob("*.json")
            if path != manifest_path and is_raw(path.relative_to(bundle).as_posix())
        ]
        with tempfile.TemporaryDirectory() as directory:
            extracted = pathlib.Path(directory)
            with zipfile.ZipFile(source_archive) as zipped:
                zipped.extractall(extracted)
            release_by_archive = {
                item["archive_path"]: item for item in manifest["files"]
                if item["storage"] == "release"
            }
            items = []
            used = set(release_by_archive)
            for archive_path, item in sorted(release_by_archive.items()):
                source = extracted / pathlib.PurePosixPath(archive_path)
                if not source.is_file() or sha256_file(source) != item["sha256"]:
                    raise ValueError(f"source archive member mismatch: {bundle.name}/{archive_path}")
                items.append(ArchiveItem(
                    source, item["path"], archive_path, item["role"], item["bytes"], item["sha256"]
                ))
            for source in raw_git_files:
                relative = source.relative_to(bundle).as_posix()
                archive_path = unique_archive_path(relative, used)
                items.append(ArchiveItem(
                    source, relative, archive_path, role_for(relative), source.stat().st_size, sha256_file(source)
                ))
            deterministic_zip(destination_archive, items, bundle.name)

        for path in raw_git_files:
            path.unlink()
        for path in sorted(bundle.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        retained = [item for item in manifest["files"] if item["storage"] == "unavailable"]
        git_files = []
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and path not in {manifest_path, bundle / "SHA256SUMS"}:
                normalize_text_file(path)
                relative = path.relative_to(bundle).as_posix()
                git_files.append({
                    "path": relative, "role": role_for(relative), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path), "storage": "git",
                })
        release_files = [{
            "path": item.logical_path, "archive_path": item.archive_path, "role": item.role,
            "bytes": item.bytes, "sha256": item.sha256, "storage": "release",
        } for item in sorted(items, key=lambda item: item.archive_path)]
        manifest["files"] = git_files + release_files + retained
        manifest["release_asset"].update({
            "bytes": destination_archive.stat().st_size,
            "sha256": sha256_file(destination_archive),
        })
        manifest_path.write_text(json_dump(manifest), encoding="utf-8", newline="\n")
        checksummed = [path for path in bundle.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
        (bundle / "SHA256SUMS").write_text("".join(
            f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in sorted(checksummed)
        ), encoding="utf-8", newline="\n")
    rebuild_index(artifacts_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=pathlib.Path, default=pathlib.Path("docs/reports/artifacts"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("out/evidence-release"))
    parser.add_argument("--release-tag", default="evidence-2026-08-03-v1")
    parser.add_argument("--repository", default="shunwendongan/RaggedRoute")
    parser.add_argument("--external-root", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--repair-unpermute", action="store_true")
    parser.add_argument("--refresh-git-inventory", action="store_true")
    parser.add_argument("--repack-source-dir", type=pathlib.Path)
    args = parser.parse_args()
    if args.refresh_git_inventory:
        refresh_git_inventory(args.artifacts_root)
        print("refreshed compact Git inventory; release archives unchanged")
        return 0
    if args.repack_source_dir:
        repack_run_manifests(args.artifacts_root, args.repack_source_dir, args.output_dir)
        print(f"repacked historical run manifests into: {args.output_dir}")
        return 0
    roots = [pathlib.Path.cwd().resolve()] + [path.resolve() for path in args.external_root]
    bundles = sorted(path for path in args.artifacts_root.iterdir() if path.is_dir())
    index = [migrate_bundle(bundle, args.output_dir, args.release_tag, args.repository, roots, args.repair_unpermute) for bundle in bundles]
    index_path = args.artifacts_root / "index.json"
    index_path.write_text(json_dump({"schema_version": "raggedroute.evidence_index.v1", "release_tag": args.release_tag, "bundles": index}), encoding="utf-8", newline="\n")
    print(f"migrated {len(index)} bundles; archives: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
