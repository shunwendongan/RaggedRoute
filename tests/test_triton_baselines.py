from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
else:  # pragma: no cover - only used on dependency-free hosts
    torch = None

GPU_READY = bool(HAS_TRITON and HAS_TORCH and torch.cuda.is_available())


def _top2_reference(row: list[float]) -> tuple[list[int], list[float]]:
    ordered = sorted(
        ((-math.inf if math.isnan(value) else value, index) for index, value in enumerate(row)),
        key=lambda item: (-item[0], item[1]),
    )
    saw_non_nan = any(not math.isnan(value) for value in row)
    first, second = ordered[:2]
    ids = [first[1], second[1]] if saw_non_nan else [0, 1]
    if not saw_non_nan or first[0] == second[0]:
        return ids, [0.5, 0.5]
    if first[0] == math.inf or second[0] == -math.inf:
        return ids, [1.0, 0.0]
    relative = math.exp(second[0] - first[0])
    first_weight = 1.0 / (1.0 + relative)
    return ids, [first_weight, relative * first_weight]


@unittest.skipUnless(GPU_READY, "requires CUDA PyTorch and Triton")
class TritonBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from dense_gemm.triton import launch_dense_gemm
        from grouped_gemm.triton import launch_grouped_gemm
        from histogram.triton import launch_histogram
        from permute.triton import launch_token_permute
        from scan.triton import launch_exclusive_scan
        from topk_gate.triton import launch_topk_gate
        from unpermute.triton import launch_unpermute

        cls.dense = staticmethod(launch_dense_gemm)
        cls.grouped = staticmethod(launch_grouped_gemm)
        cls.histogram = staticmethod(launch_histogram)
        cls.permute = staticmethod(launch_token_permute)
        cls.scan = staticmethod(launch_exclusive_scan)
        cls.topk = staticmethod(launch_topk_gate)
        cls.unpermute = staticmethod(launch_unpermute)
        benchmark_spec = importlib.util.spec_from_file_location(
            "raggedroute_triton_benchmark", ROOT / "scripts" / "triton_benchmark.py"
        )
        assert benchmark_spec and benchmark_spec.loader
        benchmark_module = importlib.util.module_from_spec(benchmark_spec)
        sys.modules[benchmark_spec.name] = benchmark_module
        benchmark_spec.loader.exec_module(benchmark_module)
        cls.prepare_chain = staticmethod(benchmark_module.prepare_chain_from_tokens)
        torch.manual_seed(20260729)

    def test_dense_ieee_fp32_irregular_stream_and_redzone(self) -> None:
        a = torch.randn((17, 23), device="cuda", dtype=torch.float32)
        b = torch.randn((23, 19), device="cuda", dtype=torch.float32)
        guard = 32
        storage = torch.full((17 * 19 + 2 * guard,), 12345.0, device="cuda")
        out = storage[guard:-guard].view(17, 19)
        stream = torch.cuda.Stream()
        self.dense(a, b, out, stream=stream)
        stream.synchronize()
        reference = a.double().cpu() @ b.double().cpu()
        torch.testing.assert_close(out.cpu(), reference.float(), rtol=2e-5, atol=2e-5)
        self.assertTrue(torch.all(storage[:guard] == 12345.0).item())
        self.assertTrue(torch.all(storage[-guard:] == 12345.0).item())

    def test_topk_exact_edge_contract(self) -> None:
        rows = [
            [1.0, 1.0, 1.0, 1.0],
            [math.nan, math.nan, math.nan, math.nan],
            [math.inf, 2.0, -1.0, -math.inf],
            [math.inf, math.inf, 0.0, math.nan],
            [3.0, math.nan, 2.0, 1.0],
        ]
        logits = torch.tensor(rows, device="cuda", dtype=torch.float32)
        ids = torch.empty((len(rows), 2), device="cuda", dtype=torch.int32)
        weights = torch.empty((len(rows), 2), device="cuda", dtype=torch.float32)
        self.topk(logits, ids, weights)
        expected = [_top2_reference(row) for row in rows]
        self.assertEqual(ids.cpu().tolist(), [item[0] for item in expected])
        torch.testing.assert_close(
            weights.cpu(), torch.tensor([item[1] for item in expected]), rtol=1e-6, atol=1e-6
        )

    def test_histogram_and_terminal_exclusive_scan(self) -> None:
        ids = torch.tensor([0, 3, 1, 3, 3, 7, 1, 0], device="cuda", dtype=torch.int32)
        counts = torch.full((8,), -1, device="cuda", dtype=torch.int32)
        offsets = torch.empty((9,), device="cuda", dtype=torch.int32)
        self.histogram(ids, counts, reset=True)
        self.scan(counts, offsets)
        self.assertEqual(counts.cpu().tolist(), [2, 2, 0, 3, 0, 0, 0, 1])
        self.assertEqual(offsets.cpu().tolist(), [0, 2, 4, 4, 7, 7, 7, 7, 8])

    def test_permute_mapping_is_bijective_and_rows_match(self) -> None:
        x = torch.arange(5 * 7, device="cuda", dtype=torch.float32).view(5, 7)
        ids = torch.tensor([[2, 0], [1, 2], [0, 3], [2, 1], [3, 0]], device="cuda", dtype=torch.int32)
        counts = torch.bincount(ids.flatten().to(torch.int64), minlength=4).to(torch.int32)
        offsets = torch.cat(
            (torch.zeros(1, device="cuda", dtype=torch.int32), counts.cumsum(0).to(torch.int32))
        )
        routes = ids.numel()
        out = torch.empty((routes, x.shape[1]), device="cuda", dtype=torch.float32)
        route_pos = torch.empty((routes,), device="cuda", dtype=torch.int32)
        sorted_route = torch.empty_like(route_pos)
        self.permute(x, ids, offsets, out, route_pos, sorted_route)
        positions = route_pos.cpu().tolist()
        self.assertEqual(sorted(positions), list(range(routes)))
        self.assertEqual(sorted_route[route_pos].cpu().tolist(), list(range(routes)))
        torch.testing.assert_close(out[route_pos].cpu(), x.repeat_interleave(2, dim=0).cpu())
        flat_ids = ids.flatten().cpu().tolist()
        host_offsets = offsets.cpu().tolist()
        for route, position in enumerate(positions):
            expert = flat_ids[route]
            self.assertLessEqual(host_offsets[expert], position)
            self.assertLess(position, host_offsets[expert + 1])

    def test_grouped_gemm_handles_empty_and_ragged_experts(self) -> None:
        offsets = torch.tensor([0, 2, 2, 5], device="cuda", dtype=torch.int32)
        x = torch.randn((5, 17), device="cuda", dtype=torch.float32)
        weights = torch.randn((3, 17, 13), device="cuda", dtype=torch.float32)
        out = torch.empty((5, 13), device="cuda", dtype=torch.float32)
        self.grouped(x, weights, offsets, out, 3)
        expected = torch.empty_like(out)
        expected[0:2] = x[0:2].double().cpu().matmul(weights[0].double().cpu()).float().cuda()
        expected[2:5] = x[2:5].double().cpu().matmul(weights[2].double().cpu()).float().cuda()
        torch.testing.assert_close(out, expected, rtol=3e-5, atol=3e-5)

    def test_unpermute_indirection_and_weighted_reduce(self) -> None:
        tokens, top_k, output = 6, 2, 11
        routes = tokens * top_k
        permuted = torch.randn((routes, output), device="cuda", dtype=torch.float32)
        route_pos = torch.randperm(routes, device="cuda", dtype=torch.int64).to(torch.int32)
        weights = torch.rand((tokens, top_k), device="cuda", dtype=torch.float32)
        weights /= weights.sum(dim=1, keepdim=True)
        out = torch.empty((tokens, output), device="cuda", dtype=torch.float32)
        self.unpermute(permuted, route_pos, weights, out, top_k)
        expected = torch.empty_like(out)
        for token in range(tokens):
            expected[token] = sum(
                permuted[route_pos[token * top_k + rank].long()] * weights[token, rank]
                for rank in range(top_k)
            )
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_chain_from_tokens_matches_all_independent_stage_oracles(self) -> None:
        prepared = self.prepare_chain(
            {"T": 48, "E": 8, "K": 19, "N": 13}, 20260729
        )
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            prepared.launch("l3")
        stream.synchronize()
        validation = prepared.validate()
        self.assertTrue(validation["ok"], validation["message"])
        self.assertEqual(prepared.operator, "chain_from_tokens")
        self.assertEqual(prepared.case_config["included_operator_count"], 7)
        self.assertEqual(prepared.variant_details["grouped_max_m_policy"], "worst_case_R")


class TritonSourceContractTests(unittest.TestCase):
    def test_every_operator_has_source_and_pinned_provenance(self) -> None:
        operators = ("dense_gemm", "topk_gate", "histogram", "scan", "permute", "grouped_gemm", "unpermute")
        for operator in operators:
            directory = SRC / operator / "triton"
            self.assertTrue((directory / "baseline.py").is_file())
            upstream = (directory / "UPSTREAM.md").read_text(encoding="utf-8")
            self.assertIn("Revision:", upstream)
            self.assertIn("benchmark-only", upstream)


if __name__ == "__main__":
    unittest.main()
