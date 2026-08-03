#!/usr/bin/env python3
"""Resolve a complete relocatable CUDA Toolkit installation on Windows.

The resolver deliberately validates both the compiler toolchain and the runtime
DLLs used by RaggedRoute's cuBLAS-backed benchmark variants.  It never mutates
machine-wide environment variables or filesystem junctions.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CudaToolkit:
    root: pathlib.Path
    nvcc: pathlib.Path
    runtime_dirs: tuple[pathlib.Path, ...]
    cublas_dll: pathlib.Path
    cublaslt_dll: pathlib.Path

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "nvcc": str(self.nvcc),
            "runtime_dirs": [str(path) for path in self.runtime_dirs],
            "cublas_dll": str(self.cublas_dll),
            "cublaslt_dll": str(self.cublaslt_dll),
        }


def _resolved(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return path.absolute()


def _first_file(directories: Iterable[pathlib.Path], pattern: str) -> pathlib.Path | None:
    for directory in directories:
        try:
            matches = sorted(path for path in directory.glob(pattern) if path.is_file())
        except OSError:
            matches = []
        if matches:
            return _resolved(matches[-1])
    return None


def inspect_toolkit(root: pathlib.Path) -> CudaToolkit | None:
    """Return a validated toolkit or None for incomplete/broken installations."""

    try:
        root = _resolved(root)
        nvcc = root / "bin" / "nvcc.exe"
        required = (
            nvcc,
            root / "include" / "cuda_runtime.h",
            root / "include" / "cublas_v2.h",
            root / "lib" / "x64" / "cublas.lib",
            root / "lib" / "x64" / "cublasLt.lib",
        )
        if not all(path.is_file() for path in required):
            return None
        runtime_dirs = tuple(
            path for path in (root / "bin", root / "bin" / "x64") if path.is_dir()
        )
        cublas = _first_file(runtime_dirs, "cublas64_*.dll")
        cublaslt = _first_file(runtime_dirs, "cublasLt64_*.dll")
        if cublas is None or cublaslt is None:
            return None
        return CudaToolkit(root, _resolved(nvcc), runtime_dirs, cublas, cublaslt)
    except OSError:
        return None


def _version_key(path: pathlib.Path) -> tuple[int, ...]:
    values = tuple(int(item) for item in re.findall(r"\d+", path.name))
    return values or (0,)


def select_toolkit(roots: Iterable[pathlib.Path]) -> CudaToolkit | None:
    unique: dict[str, pathlib.Path] = {}
    for root in roots:
        key = os.path.normcase(str(root))
        unique.setdefault(key, root)
    for root in sorted(unique.values(), key=_version_key, reverse=True):
        toolkit = inspect_toolkit(root)
        if toolkit is not None:
            return toolkit
    return None


def _root_from_nvcc(value: str | None) -> pathlib.Path | None:
    if not value:
        return None
    path = pathlib.Path(value.strip('"'))
    try:
        if path.is_file() and path.name.casefold() == "nvcc.exe":
            return path.parent.parent
    except OSError:
        pass
    return None


def _cache_roots(repo: pathlib.Path) -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for cache in (repo / "out" / "build").glob("*/CMakeCache.txt"):
        try:
            for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("CMAKE_CUDA_COMPILER:") and "=" in line:
                    root = _root_from_nvcc(line.split("=", 1)[1])
                    if root is not None:
                        roots.append(root)
        except OSError:
            continue
    return roots


def _registry_roots() -> list[pathlib.Path]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []
    roots: list[pathlib.Path] = []
    base = r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for access in (winreg.KEY_READ, winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0)):
            try:
                with winreg.OpenKey(hive, base, 0, access) as key:
                    index = 0
                    while True:
                        try:
                            version = winreg.EnumKey(key, index)
                            index += 1
                        except OSError:
                            break
                        try:
                            with winreg.OpenKey(key, version) as version_key:
                                install_dir, _ = winreg.QueryValueEx(version_key, "InstallDir")
                                roots.append(pathlib.Path(install_dir))
                        except OSError:
                            continue
            except OSError:
                continue
    return roots


def _filesystem_roots(environ: Mapping[str, str]) -> list[pathlib.Path]:
    parents: list[pathlib.Path] = []
    for variable in ("ProgramFiles", "ProgramW6432"):
        value = environ.get(variable)
        if value:
            parents.append(pathlib.Path(value) / "NVIDIA GPU Computing Toolkit" / "CUDA")
    if os.name == "nt":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = pathlib.Path(f"{letter}:\\")
            try:
                if not drive.exists():
                    continue
            except OSError:
                continue
            parents.extend(
                (
                    drive / "Dev" / "toolchains" / "CUDA",
                    drive / "toolchains" / "CUDA",
                    drive / "DevTools" / "CUDA",
                )
            )
    roots: list[pathlib.Path] = []
    for parent in parents:
        try:
            roots.extend(path for path in parent.glob("v*") if path.is_dir())
        except OSError:
            continue
    return roots


def resolve_toolkit(
    repo: pathlib.Path,
    environ: Mapping[str, str] | None = None,
) -> CudaToolkit:
    environ = os.environ if environ is None else environ
    explicit = environ.get("RAGGEDROUTE_CUDA_ROOT")
    if explicit:
        toolkit = inspect_toolkit(pathlib.Path(explicit))
        if toolkit is None:
            raise FileNotFoundError(
                f"RAGGEDROUTE_CUDA_ROOT is not a complete CUDA/cuBLAS toolkit: {explicit}"
            )
        return toolkit

    priority_groups: list[list[pathlib.Path]] = []
    cudacxx = _root_from_nvcc(environ.get("CUDACXX"))
    if cudacxx is not None:
        priority_groups.append([cudacxx])
    cuda_path = environ.get("CUDA_PATH")
    if cuda_path:
        priority_groups.append([pathlib.Path(cuda_path)])
    cache = _cache_roots(repo)
    if cache:
        priority_groups.append(cache)
    path_nvcc = _root_from_nvcc(shutil.which("nvcc.exe", path=environ.get("PATH")))
    if path_nvcc is not None:
        priority_groups.append([path_nvcc])
    discovered = _registry_roots() + _filesystem_roots(environ)
    if discovered:
        priority_groups.append(discovered)

    for roots in priority_groups:
        toolkit = select_toolkit(roots)
        if toolkit is not None:
            return toolkit
    raise FileNotFoundError(
        "no complete CUDA Toolkit with nvcc, headers, cuBLAS import libraries, "
        "and cuBLAS/cuBLASLt runtime DLLs was found"
    )


def runtime_environment(
    repo: pathlib.Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], CudaToolkit]:
    source = os.environ if environ is None else environ
    toolkit = resolve_toolkit(repo, source)
    result = dict(source)
    result.update(
        {
            "CUDA_PATH": str(toolkit.root),
            "CUDA_HOME": str(toolkit.root),
            "CUDA_BIN_PATH": str(toolkit.root / "bin"),
            "CUDACXX": str(toolkit.nvcc),
        }
    )
    current = [item for item in result.get("PATH", "").split(os.pathsep) if item]
    prefixes = [str(path) for path in toolkit.runtime_dirs]
    seen: set[str] = set()
    result["PATH"] = os.pathsep.join(
        item for item in prefixes + current if not (os.path.normcase(item) in seen or seen.add(os.path.normcase(item)))
    )
    return result, toolkit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    toolkit = resolve_toolkit(args.repo.resolve())
    if args.json:
        print(json.dumps(toolkit.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(toolkit.root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
