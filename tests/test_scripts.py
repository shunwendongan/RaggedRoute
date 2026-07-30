from __future__ import annotations

import copy
import importlib.util
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
            "case_config": {"E": 8},
            "variant_config": {"algorithm": "global_atomic"},
            "work": {
                "logical_bytes": 96.0,
                "flops": 0.0,
                "effective_gbps_batch_p50": 1.0 / process_run,
                "operator_metrics": {"counts_reset_bytes": 32},
            },
            "environment": {
                "gpu_uuid": "gpu",
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


if __name__ == "__main__":
    unittest.main()
