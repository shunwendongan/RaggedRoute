from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_benchmarks = load_module(
    "run_benchmarks", ROOT / "scripts" / "run_benchmarks.py"
)
aggregate_results = load_module(
    "aggregate_results", ROOT / "scripts" / "aggregate_results.py"
)
compare_results = load_module(
    "compare_results", ROOT / "scripts" / "compare_results.py"
)
profile_benchmarks = load_module(
    "profile_benchmarks", ROOT / "scripts" / "profile_benchmarks.py"
)
freeze_results = load_module(
    "freeze_results", ROOT / "scripts" / "freeze_results.py"
)
run_cross_backend = load_module(
    "run_cross_backend", ROOT / "scripts" / "run_cross_backend_benchmarks.py"
)
compare_cross_backend = load_module(
    "compare_cross_backend", ROOT / "scripts" / "compare_cross_backend.py"
)
package_triton_evidence = load_module(
    "package_triton_evidence", ROOT / "scripts" / "package_triton_evidence.py"
)
package_evidence = load_module(
    "package_evidence", ROOT / "scripts" / "package_evidence.py"
)
resolve_cuda_toolkit = load_module(
    "resolve_cuda_toolkit", ROOT / "scripts" / "resolve_cuda_toolkit.py"
)
validate_evidence = load_module(
    "validate_evidence", ROOT / "scripts" / "validate_evidence.py"
)


def make_suite_v2() -> dict:
    return {
        "schema_version": "raggedroute.suite.v2",
        "suite_id": "paired",
        "protocol": "smoke",
        "process_runs": 1,
        "common": {
            "warmup": 2,
            "samples": 3,
            "kernel_repeats": 2,
            "seed": 123,
            "cache_mode": "warm",
        },
        "cases": [
            {
                "id": "case",
                "operator": "histogram",
                "levels": ["l2"],
                "params": {"T": 4, "E": 8, "top_k": 2},
                "workload": {"trace_weight": 2.0},
                "variants": [
                    {"name": "cuda_naive", "promotion_baseline": True},
                    {"name": "cub_device_histogram", "promotion_baseline": False},
                ],
            }
        ],
    }


class SuiteTests(unittest.TestCase):
    @staticmethod
    def make_fake_cuda(root: pathlib.Path, *, with_cublaslt: bool = True) -> None:
        for relative in (
            "bin/nvcc.exe", "include/cuda_runtime.h", "include/cublas_v2.h",
            "lib/x64/cublas.lib", "lib/x64/cublasLt.lib",
            "bin/x64/cublas64_13.dll",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        if with_cublaslt:
            (root / "bin/x64/cublasLt64_13.dll").write_bytes(b"x")

    def test_cuda_resolver_rejects_incomplete_and_selects_newest_complete_toolkit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            old = root / "v12.8"
            incomplete = root / "v13.3"
            self.make_fake_cuda(old)
            self.make_fake_cuda(incomplete, with_cublaslt=False)
            selected = resolve_cuda_toolkit.select_toolkit([old, incomplete])
            self.assertIsNotNone(selected)
            self.assertEqual(selected.root, old.resolve())
            self.assertEqual(selected.runtime_dirs[-1], (old / "bin/x64").resolve())

    def test_cuda_resolver_explicit_override_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "RAGGEDROUTE_CUDA_ROOT"):
                resolve_cuda_toolkit.resolve_toolkit(
                    ROOT, {"RAGGEDROUTE_CUDA_ROOT": str(root / "broken"), "PATH": ""}
                )

    def test_ci_quality_gates_and_sm86_manual_workflow_are_present(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        gpu = (ROOT / ".github" / "workflows" / "gpu-sm86.yml").read_text(encoding="utf-8")
        template = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
        self.assertIn("python scripts/repository_checks.py", ci)
        self.assertIn("cpu-release", ci)
        self.assertIn("ubuntu-latest", ci)
        self.assertIn("windows-latest", ci)
        self.assertIn("runs-on: [self-hosted, Windows, X64, gpu-sm86]", gpu)
        self.assertIn("profile_benchmarks.py system", gpu)
        self.assertIn("profile_benchmarks.py compute", gpu)
        self.assertIn("run_sanitizers.py", gpu)
        self.assertIn("Profiler duration is not used as benchmark speedup", template)
        for relative in (
            "benchmarks/.gitkeep", "cmake/.gitkeep", "include/raggedroute/.gitkeep",
            "scripts/.gitkeep", "tests/.gitkeep",
        ):
            self.assertFalse((ROOT / relative).is_file())

    def test_evidence_v2_schema_and_all_published_bundles_validate(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "evidence-bundle-v2.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            "raggedroute.evidence_bundle.v2",
        )
        errors = validate_evidence.validate_root(ROOT / "docs" / "reports" / "artifacts")
        self.assertEqual(errors, [])

    def test_evidence_archive_is_deterministic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "raw.jsonl"
            source.write_text('{"ok": true}\n', encoding="utf-8")
            item = package_evidence.ArchiveItem(
                source=source,
                logical_path="benchmark/raw.jsonl",
                archive_path="raw/benchmark/raw.jsonl",
                role="raw_benchmark_records",
                bytes=source.stat().st_size,
                sha256=package_evidence.sha256_file(source),
            )
            first = root / "first.zip"
            second = root / "second.zip"
            package_evidence.deterministic_zip(first, [item], "run")
            package_evidence.deterministic_zip(second, [item], "run")
            self.assertEqual(
                package_evidence.sha256_file(first),
                package_evidence.sha256_file(second),
            )
            with self.assertRaises(FileExistsError):
                package_evidence.deterministic_zip(first, [item], "run")

    def test_unavailable_evidence_requires_a_reason(self) -> None:
        bundle = pathlib.Path("bundle")
        manifest = {
            "schema_version": "raggedroute.evidence_bundle.v2",
            "run_id": "run",
            "kind": "test",
            "provenance": {"source_git_sha": "1234567", "git_clean": None},
            "hardware": {"status": "unavailable"},
            "toolchain": {"status": "unavailable"},
            "semantic_contract": {"status": "unavailable"},
            "commands": [{"role": "test", "status": "unavailable", "reason": "missing"}],
            "validation": {"status": "unavailable"},
            "decision": {"status": "historical"},
            "release_asset": {
                "tag": "tag", "name": "run-raw.zip",
                "url": "https://github.com/example/repo/releases/download/tag/run-raw.zip",
                "bytes": 1, "sha256": "0" * 64, "immutable": True,
            },
            "files": [{
                "path": "missing.jsonl", "role": "raw_benchmark_records", "bytes": 1,
                "sha256": "0" * 64, "storage": "unavailable",
            }],
        }
        errors = validate_evidence.validate_manifest(manifest, bundle)
        self.assertTrue(any("missing reason" in error for error in errors))

    def test_windows_entrypoints_share_dynamic_msvc_discovery(self) -> None:
        setup = (ROOT / "scripts" / "setup_msvc_env.bat").read_text(encoding="utf-8")
        configure = (ROOT / "scripts" / "configure_windows.bat").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
        cuda_setup = (ROOT / "scripts" / "setup_cuda_env.bat").read_text(encoding="utf-8")
        test = (ROOT / "scripts" / "test_windows.bat").read_text(encoding="utf-8")

        self.assertIn("VSDEVCMD", setup)
        self.assertIn("vswhere", setup.lower())
        self.assertIn("RAGGEDROUTE_VS_INSTALL_ROOT", setup)
        self.assertIn("VSINSTALLDIR", setup)
        self.assertIn("VS2022_HOME", setup)
        self.assertIn("where cl.exe", setup)
        self.assertIn("setup_msvc_env.bat", configure)
        self.assertIn("setup_msvc_env.bat", build)
        self.assertIn("resolve_cuda_toolkit.py", cuda_setup)
        self.assertIn("bin\\x64", cuda_setup)
        self.assertIn("setup_cuda_env.bat", configure)
        self.assertIn("setup_cuda_env.bat", build)
        self.assertIn("setup_cuda_env.bat", test)
        self.assertIn("cmake --fresh --preset", configure)
        self.assertIn("-DCMAKE_CXX_COMPILER=%RAGGEDROUTE_MSVC_CL%", configure)
        self.assertIn("-DCMAKE_CUDA_COMPILER=%RAGGEDROUTE_NVCC%", configure)
        self.assertIn("-DCMAKE_CUDA_HOST_COMPILER=%RAGGEDROUTE_MSVC_CL%", configure)
        self.assertIn("RAGGEDROUTE_FETCH_REFERENCES", configure)
        self.assertIn("CMAKE_CXX_COMPILER:.*=", build)
        self.assertIn("CMAKE_CUDA_COMPILER", build)
        self.assertIn("CUDA_PATH", cuda_setup)
        self.assertIn("configure_windows.bat", build)
        self.assertNotIn("vswhere", configure.lower())
        self.assertNotIn("vswhere", build.lower())

    def test_fetchcontent_names_remain_windows_path_safe(self) -> None:
        dependencies = (ROOT / "cmake" / "RaggedRouteDependencies.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("FetchContent_Declare(rr_cccl", dependencies)
        self.assertIn("FetchContent_Declare(rr_cutlass", dependencies)
        self.assertIn("CUTLASS_ENABLE_HEADERS_ONLY ON", dependencies)
        self.assertIn("CUTLASS_ENABLE_TOOLS OFF", dependencies)
        self.assertNotIn("FetchContent_Declare(raggedroute_cccl_source", dependencies)
        self.assertNotIn("FetchContent_Declare(raggedroute_cutlass_source", dependencies)

    def test_smoke_suite_is_versioned_and_unique(self) -> None:
        suite = run_benchmarks.load_suite(
            ROOT / "configs" / "project" / "benchmark" / "smoke.json"
        )
        self.assertEqual(suite["schema_version"], "raggedroute.suite.v1")
        self.assertEqual(len(suite["cases"]), 9)

    def test_library_smoke_suite_has_strong_promotion_baselines(self) -> None:
        suite = run_benchmarks.load_suite(
            ROOT / "configs" / "project" / "benchmark" / "library_smoke.json"
        )
        self.assertEqual(suite["schema_version"], "raggedroute.suite.v2")
        self.assertEqual(len(suite["cases"]), 6)
        for case in suite["cases"]:
            baseline = next(
                variant
                for variant in case["variants"]
                if variant["promotion_baseline"]
            )
            self.assertNotIn(baseline["name"], {"cuda_naive", "cuda_naive_from_ids"})

    def test_library_release_suite_is_strict_and_uses_strong_baselines(self) -> None:
        suite = run_benchmarks.load_suite(
            ROOT / "configs" / "project" / "benchmark" / "rtx3080_library_release.json"
        )
        self.assertEqual(suite["schema_version"], "raggedroute.suite.v2")
        self.assertEqual(suite["protocol"], "release")
        self.assertEqual(suite["process_runs"], 3)
        self.assertEqual(len(suite["cases"]), 6)
        for case in suite["cases"]:
            baseline = next(
                variant for variant in case["variants"] if variant["promotion_baseline"]
            )
            self.assertNotIn(baseline["name"], {"cuda_naive", "cuda_naive_from_ids"})

    def test_profile_v2_covers_all_operators_and_skips_warmups(self) -> None:
        config = profile_benchmarks.load_config(
            ROOT / "configs" / "project" / "profile" / "representative.json"
        )
        self.assertEqual(config["schema_version"], "raggedroute.profile_suite.v2")
        self.assertEqual(
            {case["operator"] for case in config["cases"]},
            set(profile_benchmarks.KERNEL_PATTERNS),
        )
        commands = profile_benchmarks.compute_commands(
            pathlib.Path("bench.exe"), config, pathlib.Path("reports"), "ncu"
        )
        self.assertEqual(len(commands), 7)
        for case_id, command, report in commands:
            self.assertIn("--clock-control", command)
            self.assertEqual(command[command.index("--clock-control") + 1], "none")
            self.assertEqual(command[command.index("--launch-skip") + 1], "5")
            self.assertEqual(command[command.index("--launch-count") + 1], "1")
            self.assertEqual(command[command.index("--warmup") + 1], "5")
            self.assertEqual(report.name, f"{case_id}.basic.ncu-rep")
        system = profile_benchmarks.target_command(
            pathlib.Path("bench.exe"), config["system_case"], system=True
        )
        self.assertEqual(system[system.index("--suite") + 1], "chain_from_tokens")
        benchmark_main = (ROOT / "benchmarks" / "benchmark_main.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("iteration < run.warmup", benchmark_main)
        self.assertIn("profile warmup synchronization", benchmark_main)

    def test_profile_v1_remains_readable(self) -> None:
        config = {
            "schema_version": "raggedroute.profile_suite.v1",
            "suite_id": "legacy",
            "cases": [
                {"id": "dense", "operator": "dense_gemm", "params": {"M": 1, "N": 1, "K": 1}}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profile.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(profile_benchmarks.load_config(path)["suite_id"], "legacy")

    def test_profile_metric_statuses_never_turn_missing_into_zero(self) -> None:
        class Metric:
            def value(self):
                return 42

            def unit(self):
                return "%"

        class Action:
            def metric_names(self):
                return [
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                    "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
                    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
                ]

            def __getitem__(self, name):
                if name not in self.metric_names():
                    raise KeyError(name)
                return Metric()

        supported = {"dram__throughput.avg.pct_of_peak_sustained_elapsed"}
        snapshot = profile_benchmarks.metric_snapshot(Action(), supported)
        self.assertEqual(snapshot["sm_throughput_pct"]["status"], "collected")
        self.assertEqual(snapshot["dram_throughput_pct"]["status"], "collected")
        self.assertEqual(snapshot["issue_active_pct"]["status"], "collected")
        self.assertEqual(snapshot["stall_long_scoreboard"]["status"], "collected")
        self.assertEqual(
            snapshot["registers_per_thread"]["status"], "unsupported_or_unknown"
        )
        self.assertIsNone(snapshot["registers_per_thread"]["value"])

    def test_nsys_stats_force_export_for_every_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            report = root / "system.nsys-rep"
            report.write_bytes(b"report")
            analysis = root / "analysis"
            analysis.mkdir()
            result = mock.Mock(
                returncode=0,
                stdout='"Time (%)","Total Time (ns)","Name"\n100.0,1,"kernel"\n',
                stderr="",
            )
            with mock.patch.object(
                profile_benchmarks, "run_command", return_value=result
            ) as run:
                stats = profile_benchmarks.export_nsys_stats(
                    "nsys", report, analysis
                )

            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[:3], ["nsys", "stats", "--force-export=true"])
            self.assertEqual(set(stats), set(profile_benchmarks.NSYS_STATS_REPORTS))
            self.assertTrue(
                all(item["status"] == "collected" for item in stats.values())
            )
            for report_name in profile_benchmarks.NSYS_STATS_REPORTS:
                self.assertTrue(
                    (analysis / f"nsys_{report_name}.csv").is_file()
                )

    def test_freeze_bundle_excludes_profiler_binaries_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            profile = root / "profile"
            benchmark = root / "benchmark"
            output = root / "artifacts"
            (profile / "analysis").mkdir(parents=True)
            (profile / "reports").mkdir()
            benchmark.mkdir()
            (profile / "analysis" / "ncu_metrics.json").write_text("[]\n", encoding="utf-8")
            (profile / "analysis" / "REPORT.md").write_bytes(b"# report\r\n")
            (profile / "reports" / "dense.basic.ncu-rep").write_bytes(b"report")
            (profile / "manifest.json").write_text("{}\n", encoding="utf-8")
            (benchmark / "naive.jsonl").write_text("{}\n", encoding="utf-8")
            (benchmark / "naive.aggregate.json").write_text("{}\n", encoding="utf-8")
            destination = freeze_results.freeze_bundle(
                "run", profile, benchmark, output
            )
            self.assertTrue((destination / "SHA256SUMS").is_file())
            self.assertFalse(list(destination.rglob("*.ncu-rep")))
            self.assertNotIn(b"\r", (destination / "profile" / "REPORT.md").read_bytes())
            checksums = {
                relative: digest
                for line in (destination / "SHA256SUMS")
                .read_text(encoding="utf-8")
                .splitlines()
                for digest, relative in (line.split("  ", 1),)
            }
            for relative, expected in checksums.items():
                self.assertEqual(
                    freeze_results.sha256_file(destination / relative), expected
                )
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["raw_profiler_reports"][0]["committed"])
            with self.assertRaises(FileExistsError):
                freeze_results.freeze_bundle("run", profile, benchmark, output)

    def test_suite_v2_expands_variants_without_changing_logical_case(self) -> None:
        suite = make_suite_v2()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "suite.json"
            path.write_text(json.dumps(suite), encoding="utf-8")
            loaded = run_benchmarks.load_suite(path)
        case = loaded["cases"][0]
        variants = run_benchmarks.case_variants(loaded, case)
        commands = [
            run_benchmarks.case_command(
                pathlib.Path("bench"),
                loaded,
                case,
                "l2",
                1,
                "run",
                pathlib.Path("out"),
                variant=variant,
            )
            for variant in variants
        ]
        self.assertEqual(
            [command[command.index("--variant") + 1] for command in commands],
            ["cuda_naive", "cub_device_histogram"],
        )
        self.assertEqual(
            {command[command.index("--case-id") + 1] for command in commands},
            {"case"},
        )
        self.assertEqual(
            {command[command.index("--seed") + 1] for command in commands}, {"123"}
        )

    def test_suite_v2_expands_parameter_matrix_deterministically(self) -> None:
        suite = make_suite_v2()
        suite["cases"][0]["matrix"] = {"T": [1, 8], "E": [2, 3]}
        suite["cases"][0]["params"] = {"input_mode": "random"}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "suite.json"
            path.write_text(json.dumps(suite), encoding="utf-8")
            loaded = run_benchmarks.load_suite(path)
        self.assertEqual(
            [case["id"] for case in loaded["cases"]],
            ["case.e2_t1", "case.e2_t8", "case.e3_t1", "case.e3_t8"],
        )
        self.assertEqual(
            loaded["cases"][2]["params"], {"input_mode": "random", "E": 3, "T": 1}
        )

    def test_suite_v2_rejects_empty_parameter_matrix_axis(self) -> None:
        suite = make_suite_v2()
        suite["cases"][0]["matrix"] = {"T": []}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "suite.json"
            path.write_text(json.dumps(suite), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_benchmarks.load_suite(path)

    def test_suite_v2_rejects_invalid_baseline_or_duplicate_variants(self) -> None:
        mutations = []
        no_baseline = make_suite_v2()
        no_baseline["cases"][0]["variants"][0]["promotion_baseline"] = False
        mutations.append(no_baseline)
        two_baselines = make_suite_v2()
        two_baselines["cases"][0]["variants"][1]["promotion_baseline"] = True
        mutations.append(two_baselines)
        duplicate = make_suite_v2()
        duplicate["cases"][0]["variants"][1]["name"] = "cuda_naive"
        mutations.append(duplicate)
        one_variant = make_suite_v2()
        one_variant["cases"][0]["variants"] = one_variant["cases"][0]["variants"][:1]
        mutations.append(one_variant)
        for index, suite in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "suite.json"
                path.write_text(json.dumps(suite), encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_benchmarks.load_suite(path)

    def test_stateful_repeat_override_is_applied(self) -> None:
        suite = run_benchmarks.load_suite(
            ROOT / "configs" / "project" / "benchmark" / "smoke.json"
        )
        case = next(case for case in suite["cases"] if case["operator"] == "token_permute")
        command = run_benchmarks.case_command(
            pathlib.Path("bench"), suite, case, "l1", 1, "run", pathlib.Path("out.jsonl")
        )
        repeats = command[command.index("--kernel-repeats") + 1]
        self.assertEqual(repeats, "1")

    def test_suite_uses_suite_flag(self) -> None:
        suite = {
            "common": {},
            "protocol": "smoke",
        }
        case = {
            "id": "chain",
            "suite": "chain_from_tokens",
            "levels": ["l3"],
            "params": {"T": 4, "E": 4, "K": 4, "N": 4},
        }
        command = run_benchmarks.case_command(
            pathlib.Path("bench"), suite, case, "l3", 1, "run", pathlib.Path("out")
        )
        self.assertEqual(command[1:3], ["--suite", "chain_from_tokens"])

    def test_process_runs_replay_the_same_case_seed(self) -> None:
        suite = {
            "common": {"seed": 12345},
            "protocol": "smoke",
        }
        case = {"id": "dense", "operator": "dense_gemm", "levels": ["l1"]}
        first = run_benchmarks.case_command(
            pathlib.Path("bench"), suite, case, "l1", 1, "run", pathlib.Path("out")
        )
        third = run_benchmarks.case_command(
            pathlib.Path("bench"), suite, case, "l1", 3, "run", pathlib.Path("out")
        )
        self.assertEqual(first[first.index("--seed") + 1], "12345")
        self.assertEqual(third[third.index("--seed") + 1], "12345")

    def test_release_gate_rejects_dirty_or_too_few_runs(self) -> None:
        suite = run_benchmarks.load_suite(
            ROOT / "configs" / "project" / "benchmark" / "rtx3080_release.json"
        )
        with self.assertRaises(ValueError):
            run_benchmarks.validate_release(suite, True)
        bad = dict(suite)
        bad["process_runs"] = 2
        with self.assertRaises(ValueError):
            run_benchmarks.validate_release(bad, False)

    def test_git_state_fails_closed_when_provenance_is_unavailable(self) -> None:
        with mock.patch.object(
            run_benchmarks,
            "command_output",
            side_effect=["unavailable: git failed", "", "branch"],
        ):
            sha, dirty, branch = run_benchmarks.git_state(ROOT)
        self.assertTrue(dirty)
        self.assertTrue(sha.startswith("unavailable:"))
        self.assertEqual(branch, "branch")

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(aggregate_results.percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    @staticmethod
    def make_benchmark_record(process_run: int) -> dict:
        return {
            "run_id": "run",
            "case_id": "case",
            "operator": "histogram",
            "variant": "cuda_naive",
            "measurement_level": "L2_operator_steady",
            "cache_mode": "warm",
            "protocol": "smoke",
            "warmup": 2,
            "kernel_repeats": 2,
            "samples": 3,
            "seed": 123,
            "process_run": process_run,
            "excluded_steps": ["h2d_copy"],
            "workspace_bytes": 0,
            "case_config": {"E": 8, "dtype": "int32", "layout": "contiguous"},
            "variant_config": {
                "implementation_category": "in_tree_cuda",
                "implementation_version": "raggedroute.cuda_naive.v1",
                "implementation_revision": "abc",
                "dependency_revision": "not_applicable",
                "algorithm_id": "global_atomic",
                "math_mode": "strict_fp32",
            },
            "work": {
                "logical_bytes": 96.0,
                "flops": 0.0,
                "effective_gbps_batch_p50": 1.0 / process_run,
                "operator_metrics": {"counts_reset_bytes": 32},
            },
            "environment": {
                "gpu_uuid": "gpu",
                "gpu_name": "gpu-name",
                "build_git_sha": "abc",
                "build_type": "Release",
                "compute_capability": "8.6",
                "cuda_compiler": "13.3",
                "cuda_runtime": 13030,
                "cuda_driver": 13030,
            },
            "timing": {
                "batch_mean_us_p50": float(process_run),
                "raw_batch_mean_samples_us": [float(process_run)],
            },
        }

    def test_aggregate_accepts_compatible_process_records(self) -> None:
        records = [self.make_benchmark_record(1), self.make_benchmark_record(2)]
        summary = aggregate_results.aggregate_records(records)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["process_runs"], 2)

    def test_aggregate_rejects_mixed_measurement_conditions(self) -> None:
        mutations = (
            ("kernel_repeats", 3),
            ("seed", 456),
            ("workspace_bytes", 32),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                first = self.make_benchmark_record(1)
                second = copy.deepcopy(self.make_benchmark_record(2))
                second[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    aggregate_results.aggregate_records([first, second])

        first = self.make_benchmark_record(1)
        second = copy.deepcopy(self.make_benchmark_record(2))
        second["environment"]["cuda_driver"] = 13040
        with self.assertRaisesRegex(ValueError, "cuda_driver"):
            aggregate_results.aggregate_records([first, second])

    def make_v2_groups(self) -> list[dict]:
        baseline = self.make_benchmark_record(1)
        candidate = copy.deepcopy(baseline)
        candidate["variant"] = "cub_device_histogram"
        candidate["variant_config"].update(
            {
                "implementation_category": "nvidia_cccl",
                "implementation_version": "cub.device_histogram.v1",
                "dependency_revision": "cccl-v3.4.0",
                "algorithm_id": "cub_device_histogram",
            }
        )
        candidate["timing"]["batch_mean_us_p50"] = 0.5
        candidate["timing"]["raw_batch_mean_samples_us"] = [0.5]
        return aggregate_results.aggregate_records_v2(
            [baseline, candidate], make_suite_v2()
        )

    def test_aggregate_v2_preserves_pairing_fields_and_baseline(self) -> None:
        groups = self.make_v2_groups()
        self.assertEqual(len(groups), 2)
        baseline = next(group for group in groups if group["promotion_baseline"])
        candidate = next(group for group in groups if not group["promotion_baseline"])
        for field in (
            "environment",
            "protocol",
            "seed",
            "excluded_steps",
            "workspace_bytes",
            "case_config",
            "variant_config",
            "workload_config",
            "suite_case",
        ):
            self.assertIn(field, baseline)
            self.assertIn(field, candidate)
        self.assertEqual(baseline["variant"], "cuda_naive")

    def test_aggregate_v2_rejects_missing_declared_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing declared"):
            aggregate_results.aggregate_records_v2(
                [self.make_benchmark_record(1)], make_suite_v2()
            )

    def test_comparison_v1_uses_paired_speedup_and_ratio_of_sums(self) -> None:
        document = compare_results.compare_document(
            {
                "schema_version": "raggedroute.aggregate.v2",
                "suite_id": "paired",
                "groups": self.make_v2_groups(),
            }
        )
        self.assertEqual(document["schema_version"], "raggedroute.comparison.v1")
        self.assertEqual(document["comparisons"][0]["speedup"], 2.0)
        self.assertIn("baseline_p95_us", document["comparisons"][0])
        self.assertIn("candidate_p95_us", document["comparisons"][0])
        self.assertIn("p95_ratio", document["comparisons"][0])
        self.assertIn("baseline_all_samples_cv", document["comparisons"][0])
        self.assertIn("candidate_all_samples_cv", document["comparisons"][0])
        self.assertIn("baseline_process_runs", document["comparisons"][0])
        candidate_summary = document["summary"]["candidate_variants"][0]
        self.assertEqual(
            candidate_summary["shape_balanced_geometric_mean_speedup"], 2.0
        )
        self.assertEqual(candidate_summary["trace_ratio_of_sums_speedup"], 2.0)

    def test_comparison_fairness_matrix_fails_closed(self) -> None:
        def mutate(groups: list[dict], path: tuple[str, ...], value) -> None:
            candidate = next(
                group for group in groups if not group["promotion_baseline"]
            )
            target = candidate
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = value

        mutations = (
            (("environment", "gpu_uuid"), "other-gpu"),
            (("environment", "build_git_sha"), "other-build"),
            (("case_config", "dtype"), "fp16"),
            (("variant_config", "math_mode"), "tf32"),
            (("seed",), 999),
            (("measurement_level",), "L1_kernel_body"),
            (("cache_mode",), "cold_scrub"),
            (("kernel_repeats",), 7),
            (("excluded_steps",), ["different"]),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                groups = copy.deepcopy(self.make_v2_groups())
                mutate(groups, path, value)
                with self.assertRaises(ValueError):
                    compare_results.compare_groups(groups)

    def test_comparison_rejects_missing_candidate_and_duplicate_baseline(self) -> None:
        groups = self.make_v2_groups()
        baseline = next(group for group in groups if group["promotion_baseline"])
        with self.assertRaisesRegex(ValueError, "no candidate"):
            compare_results.compare_groups([baseline])
        duplicate = copy.deepcopy(groups)
        for group in duplicate:
            group["promotion_baseline"] = True
        with self.assertRaisesRegex(ValueError, "exactly one"):
            compare_results.compare_groups(duplicate)

    def test_triton_source_provenance_and_windows_setup_are_present(self) -> None:
        operators = (
            "dense_gemm", "topk_gate", "histogram", "scan", "permute", "grouped_gemm", "unpermute"
        )
        for operator in operators:
            directory = ROOT / "src" / operator / "triton"
            self.assertTrue((directory / "baseline.py").is_file())
            provenance = (directory / "UPSTREAM.md").read_text(encoding="utf-8")
            self.assertIn("Revision:", provenance)
            self.assertIn("benchmark-only", provenance)
        setup = (ROOT / "scripts" / "setup_triton_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("torch==2.12.1+cu130", setup)
        self.assertIn("triton-windows==3.7.1.post27", setup)
        self.assertTrue((ROOT / "third_party" / "licenses" / "Triton-MIT.txt").is_file())

    def test_cross_backend_suite_and_pairing_remain_reference_only(self) -> None:
        suite = run_cross_backend.load_suite(
            ROOT / "configs" / "cross_backend" / "benchmark" / "l1_l2_smoke.json"
        )
        self.assertEqual(suite["schema_version"], "raggedroute.cross_backend_suite.v1")
        self.assertEqual({case["operator"] for case in suite["cases"]}, {
            "dense_gemm", "topk_gate", "histogram", "exclusive_scan", "token_permute", "grouped_gemm", "unpermute"
        })
        for case in suite["cases"]:
            reference = next(item for item in case["variants"] if item.get("reference_baseline"))
            self.assertEqual(reference["backend"], "triton")

        reference = self.make_benchmark_record(1)
        reference["variant"] = "triton_reference"
        reference["variant_config"].update({
            "implementation_category": "in_tree_triton_reference",
            "implementation_version": "raggedroute.triton_reference.v1",
            "compiler_stack": "Triton 3.7.1",
        })
        reference["environment"]["cuda_compiler"] = "Triton 3.7.1"
        reference["environment"]["cuda_runtime"] = 13000
        candidate = copy.deepcopy(self.make_benchmark_record(1))
        reference["excluded_steps"] = ["input_generation", "cpu_reference", "h2d_copy", "workspace_allocation"]
        candidate["excluded_steps"] = list(reference["excluded_steps"])
        case = {
            "id": "case", "operator": "histogram", "levels": ["l2"],
            "params": {"T": 4, "E": 8, "top_k": 2},
        }
        reference["case_config"].update(case["params"])
        candidate["case_config"].update(case["params"])
        toolchains = compare_cross_backend.verify_pair(
            compare_cross_backend.summarize([reference]),
            compare_cross_backend.summarize([candidate]),
            case,
        )
        self.assertIn("reference_toolchain", toolchains)
        self.assertNotEqual(
            toolchains["reference_toolchain"]["cuda_runtime"],
            toolchains["candidate_toolchain"]["cuda_runtime"],
        )

    def test_cross_backend_comparison_rejects_missing_records_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing benchmark records"):
            compare_cross_backend.summarize([])

    def test_cross_backend_l3_is_full_chain_only(self) -> None:
        suite = run_cross_backend.load_suite(
            ROOT / "configs" / "cross_backend" / "benchmark" / "l3_smoke.json"
        )
        case = suite["cases"][0]
        self.assertEqual(case["suite"], "chain_from_tokens")
        self.assertEqual(case["levels"], ["l3"])
        cpp = run_cross_backend.command_for_cpp(
            pathlib.Path("bench.exe"), suite, case, case["variants"][1],
            "l3", 1, "run", pathlib.Path("out.jsonl"), "",
        )
        triton_command = run_cross_backend.command_for_triton(
            pathlib.Path("python"), ROOT, suite, case, "l3", 1, "run",
            pathlib.Path("out.jsonl"),
        )
        self.assertEqual(cpp[1:3], ["--suite", "chain_from_tokens"])
        self.assertIn("--suite", triton_command)
        self.assertNotIn("--operator", triton_command)

    def test_cross_backend_rejects_invalid_l3_targets(self) -> None:
        base = {
            "schema_version": "raggedroute.cross_backend_suite.v1",
            "protocol": "smoke",
            "process_runs": 1,
            "common": {"warmup": 1, "samples": 1},
            "cases": [{
                "id": "bad", "operator": "dense_gemm", "levels": ["l3"],
                "variants": [
                    {"name": "triton_reference", "backend": "triton", "reference_baseline": True},
                    {"name": "cuda_naive", "backend": "cpp"},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not support l3"):
                run_cross_backend.load_suite(path)
            base["cases"][0].pop("operator")
            base["cases"][0]["suite"] = "chain_from_tokens"
            base["cases"][0]["levels"] = ["l2"]
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires exactly level l3"):
                run_cross_backend.load_suite(path)

    def test_triton_profile_config_covers_three_levels(self) -> None:
        config = json.loads(
            (
                ROOT
                / "configs"
                / "cross_backend"
                / "profile"
                / "triton_three_levels.json"
            ).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema_version"], "raggedroute.triton_profile_suite.v1")
        self.assertEqual(len(config["operators"]), 7)
        self.assertEqual(
            {item["operator"] for item in config["operators"]},
            {"dense_gemm", "topk_gate", "histogram", "exclusive_scan", "token_permute", "grouped_gemm", "unpermute"},
        )
        self.assertTrue(all(set(item["kernels"]) == {"l1", "l2"} for item in config["operators"]))
        self.assertEqual(config["l3"]["suite"], "chain_from_tokens")
        self.assertEqual(len(config["l3"]["kernels"]), 7)

    def test_triton_evidence_packager_rejects_failed_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "raw.jsonl"
            record = self.make_benchmark_record(1)
            record["schema_version"] = "raggedroute.benchmark.v1"
            record["validation"] = {"ok": False}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid or failed"):
                package_triton_evidence.records(path)


if __name__ == "__main__":
    unittest.main()
