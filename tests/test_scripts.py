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
    def test_windows_entrypoints_share_dynamic_msvc_discovery(self) -> None:
        setup = (ROOT / "scripts" / "setup_msvc_env.bat").read_text(encoding="utf-8")
        configure = (ROOT / "scripts" / "configure_windows.bat").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")

        self.assertIn("VSDEVCMD", setup)
        self.assertIn("vswhere", setup.lower())
        self.assertIn("VSCMD_VER", setup)
        self.assertIn("where cl.exe", setup)
        self.assertIn("setup_msvc_env.bat", configure)
        self.assertIn("setup_msvc_env.bat", build)
        self.assertNotIn("vswhere", configure.lower())
        self.assertNotIn("vswhere", build.lower())

    def test_smoke_suite_is_versioned_and_unique(self) -> None:
        suite = run_benchmarks.load_suite(ROOT / "configs" / "benchmark_smoke.json")
        self.assertEqual(suite["schema_version"], "raggedroute.suite.v1")
        self.assertEqual(len(suite["cases"]), 9)

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
        suite = run_benchmarks.load_suite(ROOT / "configs" / "benchmark_smoke.json")
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
            ROOT / "configs" / "benchmark_rtx3080_release.json"
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


if __name__ == "__main__":
    unittest.main()
