from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promotion = load_module("evaluate_promotion", ROOT / "scripts" / "evaluate_promotion.py")
candidate_matrix = load_module(
    "summarize_candidate_matrix", ROOT / "scripts" / "summarize_candidate_matrix.py"
)
portfolio = load_module(
    "build_v3_portfolio_evidence", ROOT / "scripts" / "build_v3_portfolio_evidence.py"
)
route_trace = load_module("validate_route_trace", ROOT / "scripts" / "validate_route_trace.py")
run_benchmarks = load_module("run_benchmarks_trace", ROOT / "scripts" / "run_benchmarks.py")


class PromotionAndTraceTests(unittest.TestCase):
    @staticmethod
    def record(case: str, variant: str, process: int, p50: float, p95: float) -> dict:
        return {
            "run_id": "release-run",
            "case_id": case,
            "variant": variant,
            "process_run": process,
            "protocol": "release",
            "measurement_level": "L2_operator_steady",
            "workspace_bytes": 0,
            "case_config": {"workload_source": "captured", "T": 8},
            "environment": {
                "build_git_dirty": "false",
                "build_git_sha": "deadbeef",
                "build_type": "Release",
                "gpu_uuid": "gpu",
            },
            "validation": {"ok": True},
            "timing": {"batch_mean_us_p50": p50, "batch_mean_us_p95": p95, "cv": 0.01},
        }

    @staticmethod
    def policy() -> dict:
        return {
            "minimum_independent_process_runs": 5,
            "minimum_paired_shapes": 1,
            "maximum_all_samples_cv": 0.10,
            "minimum_ratio_of_sums_p50_speedup": 1.03,
            "minimum_speedup_shape_coverage": 0.80,
            "maximum_single_shape_p50_regression_fraction": 0.03,
            "maximum_single_shape_p95_regression_fraction": 0.03,
            "maximum_workspace_growth_bytes": 0,
            "require_same_gpu_uuid": True,
            "require_same_case_config": True,
            "require_same_build_git_sha": True,
            "require_same_run_id": True,
            "require_real_route_trace": True,
        }

    def paired(self, candidate_p50: float = 90.0) -> list[dict]:
        records = []
        for process in range(1, 6):
            records.append(self.record("shape", "base", process, 100.0, 105.0))
            records.append(self.record("shape", "candidate", process, candidate_p50, 94.0))
        return records

    def test_three_state_promotion(self) -> None:
        promoted = promotion.evaluate(self.paired(), "base", "candidate", self.policy())
        self.assertEqual(promoted["decision"], "promote")
        rejected = promotion.evaluate(self.paired(110.0), "base", "candidate", self.policy())
        self.assertEqual(rejected["decision"], "reject")
        unstable = self.paired()
        unstable[-1]["timing"]["cv"] = 0.5
        insufficient = promotion.evaluate(unstable, "base", "candidate", self.policy())
        self.assertEqual(insufficient["decision"], "insufficient_evidence")

    def test_default_cv_policy_discloses_point_one_and_caps_at_point_five(self) -> None:
        policy = self.policy()
        policy.pop("maximum_all_samples_cv")
        disclosed = self.paired()
        disclosed[-1]["timing"]["cv"] = 0.20
        decision = promotion.evaluate(disclosed, "base", "candidate", policy)
        self.assertEqual(decision["decision"], "promote")
        unstable = self.paired()
        unstable[-1]["timing"]["cv"] = 0.5001
        decision = promotion.evaluate(unstable, "base", "candidate", policy)
        self.assertEqual(decision["decision"], "insufficient_evidence")
        self.assertTrue(any("0.5000" in reason for reason in decision["reasons"]))

    def test_candidate_matrix_selects_envelope_and_counts_process_direction(self) -> None:
        records = []
        timings = {
            "shape-a": {"candidate": (8.0, 8.5), "lib-a": (10.0, 10.5), "lib-b": (9.0, 9.5)},
            "shape-b": {"candidate": (11.0, 11.5), "lib-a": (10.0, 10.5), "lib-b": (12.0, 12.5)},
        }
        for case_id, variants in timings.items():
            for variant, (p50, p95) in variants.items():
                for process in (1, 2):
                    record = self.record(case_id, variant, process, p50, p95)
                    record["timing"]["cv"] = 0.20
                    records.append(record)
        groups = candidate_matrix.group_records(records)
        rows = candidate_matrix.library_envelope_rows(
            groups, ["shape-a", "shape-b"], ["lib-a", "lib-b"], "candidate"
        )
        self.assertEqual([row["envelope_winner"] for row in rows], ["lib-b", "lib-a"])
        self.assertEqual(rows[0]["candidate_faster_processes"], 2)
        self.assertEqual(rows[1]["candidate_faster_processes"], 0)
        summary = candidate_matrix.summarize(rows)
        self.assertEqual(summary["candidate_faster_process_pairs"], 2)
        self.assertEqual(summary["paired_process_pairs"], 4)
        self.assertAlmostEqual(summary["shape_win_fraction"], 0.5)

    def test_real_trace_gate_rejects_synthetic_as_insufficient(self) -> None:
        records = self.paired()
        for record in records:
            record["case_config"]["workload_source"] = "synthetic_fixture"
        decision = promotion.evaluate(records, "base", "candidate", self.policy())
        self.assertEqual(decision["decision"], "insufficient_evidence")
        self.assertTrue(any("real route trace" in reason for reason in decision["reasons"]))

    def test_missing_release_provenance_is_insufficient(self) -> None:
        records = self.paired()
        records[-1]["environment"].pop("build_git_sha")
        decision = promotion.evaluate(records, "base", "candidate", self.policy())
        self.assertEqual(decision["decision"], "insufficient_evidence")
        self.assertTrue(any("build Git SHA" in reason for reason in decision["reasons"]))

    def test_correctness_failure_has_priority_over_missing_evidence(self) -> None:
        records = self.paired()
        records[-1]["validation"]["ok"] = False
        records[-1]["timing"]["cv"] = 0.5
        decision = promotion.evaluate(records, "base", "candidate", self.policy())
        self.assertEqual(decision["decision"], "reject")

    def test_invalid_timing_is_insufficient_instead_of_crashing(self) -> None:
        records = self.paired()
        records[-1]["timing"]["batch_mean_us_p50"] = 0.0
        decision = promotion.evaluate(records, "base", "candidate", self.policy())
        self.assertEqual(decision["decision"], "insufficient_evidence")
        self.assertTrue(any("invalid timing.batch_mean_us_p50" in reason
                            for reason in decision["reasons"]))

    def test_graph_policy_can_explicitly_use_host_boundary_with_a_cv_exception(self) -> None:
        records = self.paired()
        for record in records:
            record["measurement_level"] = "L4_host_call"
            record["timing"]["cv"] = 0.9
            record["host_time_to_solution_timing"] = {
                "p50_us": record["timing"]["batch_mean_us_p50"],
                "p95_us": record["timing"]["batch_mean_us_p95"],
                "cv": 0.01,
            }
            if record["variant"] == "candidate":
                record.setdefault("variant_config", {}).update({
                    "graph_mode": "fixed_capture_upload_replay",
                    "graph_setup_excluded": True,
                    "mixed_shape_cache_trace": False,
                })
        policy = self.policy() | {
            "required_measurement_level": "L4_host_call",
            "timing_source": "host_time_to_solution_timing",
            "ignore_timing_cv": True,
            "stability_exception": "test-only WDDM exception",
            "required_candidate_variant_config": {
                "graph_mode": "fixed_capture_upload_replay",
                "graph_setup_excluded": True,
                "mixed_shape_cache_trace": False,
            },
        }
        decision = promotion.evaluate(records, "base", "candidate", policy)
        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["measurement"]["timing_source"], "host_time_to_solution_timing")
        self.assertFalse(decision["measurement"]["timing_cv_enforced"])

    def test_empty_decision_uses_json_null_for_uncollected_metrics(self) -> None:
        decision = promotion.evaluate([], "base", "candidate", self.policy())
        self.assertEqual(decision["decision"], "insufficient_evidence")
        self.assertIsNone(decision["summary"]["maximum_p50_regression_fraction"])
        json.dumps(decision, allow_nan=False)
        self.assertIn("not_collected", promotion.markdown(decision))

    def test_compact_portfolio_evidence_has_checksums_and_archive(self) -> None:
        decision = promotion.evaluate(self.paired(), "base", "candidate", self.policy())
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "decision.json"
            source.write_text(json.dumps(decision), encoding="utf-8")
            output = root / "compact"
            archive = root / "compact.zip"
            portfolio.build([("permute", source)], output, archive)
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertTrue((output / "shape_heatmap.svg").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertTrue(archive.is_file())
            self.assertTrue(pathlib.Path(str(archive) + ".sha256").is_file())

    def test_trace_fixture_validates_and_normalizes(self) -> None:
        fixture = json.loads(
            (ROOT / "configs" / "workloads" / "synthetic_fixture.route_trace.json").read_text(
                encoding="utf-8"
            )
        )
        validated = route_trace.validate_trace(fixture)
        normalized = route_trace.normalized_text(validated)
        self.assertEqual(
            normalized,
            (ROOT / "configs" / "workloads" / "synthetic_fixture.rrtrace").read_text(
                encoding="utf-8"
            ),
        )
        bad = copy.deepcopy(fixture)
        bad["frames"][0]["expert_ids"][1] = bad["frames"][0]["expert_ids"][0]
        with self.assertRaisesRegex(ValueError, "repeats an expert"):
            route_trace.validate_trace(bad)

    def test_trace_working_set_expands_to_all_frames(self) -> None:
        suite = run_benchmarks.load_suite(
            ROOT / "configs" / "project" / "benchmark" / "route_trace_smoke.json"
        )
        self.assertEqual(len(suite["cases"]), 9)
        self.assertEqual(
            {case["params"]["route_trace_frame"] for case in suite["cases"]},
            {0, 1, 2},
        )
        self.assertTrue(
            all(case["params"]["route_trace_path"].endswith("synthetic_fixture.rrtrace")
                for case in suite["cases"])
        )


if __name__ == "__main__":
    unittest.main()
