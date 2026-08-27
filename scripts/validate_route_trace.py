#!/usr/bin/env python3
"""Validate and normalize anonymized RaggedRoute route traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


SCHEMA_VERSION = "raggedroute.route_trace.v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SOURCE_KINDS = {"production", "captured", "synthetic_fixture"}


def validate_trace(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("trace root must be an object")
    allowed = {
        "schema_version", "trace_id", "source_kind", "description", "anonymized",
        "tokens", "experts", "top_k", "frames",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"unknown trace fields: {', '.join(unknown)}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    trace_id = document.get("trace_id")
    source_kind = document.get("source_kind")
    tokens = document.get("tokens")
    experts = document.get("experts")
    top_k = document.get("top_k")
    frames = document.get("frames")
    if not isinstance(trace_id, str) or not SAFE_ID.fullmatch(trace_id):
        raise ValueError("trace_id must be a non-empty portable identifier")
    if source_kind not in SOURCE_KINDS:
        raise ValueError("source_kind must be production, captured, or synthetic_fixture")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if not isinstance(experts, int) or not 1 <= experts <= 64:
        raise ValueError("experts must be in [1, 64]")
    if not isinstance(top_k, int) or not 1 <= top_k <= experts:
        raise ValueError("top_k must satisfy 1<=top_k<=experts")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty array")
    expected_routes = tokens * top_k
    seen_frames: set[str] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {"frame_id", "expert_ids"}:
            raise ValueError(f"frame {index} must contain only frame_id and expert_ids")
        frame_id = frame["frame_id"]
        ids = frame["expert_ids"]
        if not isinstance(frame_id, str) or not SAFE_ID.fullmatch(frame_id):
            raise ValueError(f"frame {index} has an invalid frame_id")
        if frame_id in seen_frames:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        seen_frames.add(frame_id)
        if not isinstance(ids, list) or len(ids) != expected_routes:
            raise ValueError(f"frame {frame_id} must contain {expected_routes} expert ids")
        if any(not isinstance(value, int) or value < 0 or value >= experts for value in ids):
            raise ValueError(f"frame {frame_id} contains an out-of-range expert id")
        for token in range(tokens):
            selected = ids[token * top_k : (token + 1) * top_k]
            if len(set(selected)) != top_k:
                raise ValueError(f"frame {frame_id} token {token} repeats an expert")
    return document


def normalized_text(document: dict[str, Any]) -> str:
    lines = [
        SCHEMA_VERSION,
        document["trace_id"],
        document["source_kind"],
        f'{document["tokens"]} {document["experts"]} {document["top_k"]} {len(document["frames"])}',
    ]
    for frame in document["frames"]:
        lines.append(frame["frame_id"] + " " + " ".join(map(str, frame["expert_ids"])))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--output-normalized", type=pathlib.Path)
    parser.add_argument("--require-real", action="store_true",
                        help="reject source_kind=synthetic_fixture")
    args = parser.parse_args()
    document = validate_trace(json.loads(args.trace.read_text(encoding="utf-8")))
    if args.require_real and document["source_kind"] == "synthetic_fixture":
        raise ValueError("real-trace evaluation rejects synthetic_fixture input")
    text = normalized_text(document)
    if args.output_normalized:
        args.output_normalized.parent.mkdir(parents=True, exist_ok=True)
        args.output_normalized.write_text(text, encoding="utf-8", newline="\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": document["trace_id"],
        "source_kind": document["source_kind"],
        "tokens": document["tokens"],
        "experts": document["experts"],
        "top_k": document["top_k"],
        "frames": len(document["frames"]),
        "normalized_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

