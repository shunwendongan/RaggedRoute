from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
