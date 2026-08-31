#!/usr/bin/env python3
"""Synchronize and verify the vLLM MoE source archive used by benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "third_party" / "vllm_moe"
REPOSITORY = "https://github.com/vllm-project/vllm"
SOURCE_REF = "v0.26.0"
SOURCE_TAG_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
IMAGE = "vllm/vllm-openai@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
IMAGE_BUILD_COMMIT = "ffd46bfab2128bb84146050e98b51a617c6575ab"
FILES = (
    "vllm/model_executor/layers/fused_moe/fused_moe.py",
    "vllm/model_executor/layers/fused_moe/moe_align_block_size.py",
    "vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py",
    "vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py",
    "vllm/model_executor/layers/fused_moe/router/fused_topk_router.py",
)


def destination(source: str) -> pathlib.Path:
    marker = "vllm/model_executor/layers/fused_moe/"
    return DESTINATION / source.removeprefix(marker)


def download(source: str) -> bytes:
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{SOURCE_REF}/{source}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(hashes: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": "raggedroute.vllm_source_manifest.v2",
        "repository": REPOSITORY,
        "source_ref": SOURCE_REF,
        "source_tag_commit": SOURCE_TAG_COMMIT,
        "container_image": IMAGE,
        "container_vllm_version": "0.26.0",
        "container_build_commit": IMAGE_BUILD_COMMIT,
        "license": "Apache-2.0",
        "files": [{"source": source, "sha256": hashes[source]} for source in FILES],
        "runtime_contract": (
            "The runnable workers import APIs from the digest-pinned container. "
            "Archived files are provenance references and must hash-match v0.26.0."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="replace the archive with v0.26.0 files")
    args = parser.parse_args()
    remote = {source: download(source) for source in FILES}
    mismatches: list[str] = []
    for source, data in remote.items():
        path = destination(source)
        if args.sync:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if not path.exists() or path.read_bytes() != data:
            mismatches.append(source)
    hashes = {source: digest(data) for source, data in remote.items()}
    expected = json.dumps(manifest(hashes), ensure_ascii=False, indent=2) + "\n"
    manifest_path = DESTINATION / "SOURCE_MANIFEST.json"
    if args.sync:
        manifest_path.write_text(expected, encoding="utf-8", newline="\n")
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != expected:
        mismatches.append("SOURCE_MANIFEST.json")
    if mismatches:
        raise SystemExit("vLLM source archive mismatch: " + ", ".join(mismatches))
    print(f"verified {len(FILES)} vLLM {SOURCE_REF} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
