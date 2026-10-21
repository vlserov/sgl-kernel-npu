"""Performance tests for the Ascend LoRA kernels.

The tests measure the latency of every LoRA operator on realistic decode and
prefill shapes, compare it against a pure-PyTorch grouped-matmul baseline and
assert the regression bounds of the suite so that a pathologically slow kernel
is caught in CI.

The suite is described by a YAML file under ``benchmark/lora/configs/``, not
by environment variables: the shapes, the timing loop and the bounds live
next to each other and are diffed like code.  The bounds are intentionally
generous -- they only catch a kernel that became orders of magnitude slower
(or stopped making progress), not small performance changes.

Each test method runs the scenarios of the suite that belong to its group:
``default`` scenarios run for every operator, and the specialised groups
(``fused-slices``, ``mixed-ranks``) run for the operators the suite selects.

The measured numbers are printed so that the CI log contains the performance
report.  The full sweep lives in ``benchmark/lora/bench_lora_kernels.py``.

Usage:
    python tests/python/sgl_kernel_npu/test_lora_kernels_perf.py
    python -m pytest tests/python/sgl_kernel_npu/test_lora_kernels_perf.py -q -s
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "benchmark" / "lora"
sys.path.insert(0, str(BENCHMARK_DIR))

import perf_config
import perf_kit
import sgl_kernel_npu  # noqa: F401  (registers the torch.ops.npu operators)
import torch
import torch_npu  # noqa: F401


def _load_benchmark():
    """Import the LoRA benchmark module that lives next to the sources."""
    spec = importlib.util.spec_from_file_location(
        "bench_lora_kernels", BENCHMARK_DIR / "bench_lora_kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark()

# The suite the CI runs.  Its bounds are the regression contract.
SUITE_NAME = "lora_perf_ci"
SUITE = perf_config.load_suite_by_name(SUITE_NAME)

# The operators that support a per-adapter rank.
RANKED_OPS = bench.RANKED_OPS


def op_available(op: str) -> bool:
    """Some operators are not registered in the reduced (A5) build."""
    return hasattr(torch.ops.npu, op)


def requires_op(op: str):
    return unittest.skipUnless(
        op_available(op), f"npu op '{op}' not registered (A5 reduced build)"
    )


class TestLoraKernelPerformance(unittest.TestCase):
    """Latency and sanity checks for every LoRA operator."""

    def _measure(self, op, workload, dtype=None, seed=None):
        """Build, validate and time one operator, returning the measurement."""
        dtype = dtype if dtype is not None else workload.torch_dtype
        seed = seed if seed is not None else workload.seed
        uniform = op not in RANKED_OPS
        meta = bench.build_meta(workload, uniform)
        case = bench.build_case(op, workload, meta, dtype, seed)

        # The kernel must be correct before its speed is meaningful.
        bench.check_case(case, strict=SUITE.bounds.strict_check)

        if case.reset is not None:
            case.reset()
        kernel_samples = bench.time_samples_ms(
            case.run, workload.warmup, workload.iters, workload.repeat
        )
        kernel = bench.summarize(kernel_samples)

        if case.reset is not None:
            case.reset()
        baseline_samples = bench.time_samples_ms(
            case.reference, workload.warmup, workload.iters, workload.repeat
        )
        baseline = bench.summarize(baseline_samples)

        return case, kernel, baseline

    def _assert_healthy(self, op, workload, kernel, baseline):
        """Fail when the kernel made no progress at all.

        The latency and slowdown bounds are carried by the metrics of
        :meth:`_report`, which enforces them in CI.
        """
        median = kernel["median"]
        self.assertGreater(median, 0.0, f"{op}/{workload.name}: no progress measured")

    def _report(self, op, workload, case, kernel, baseline):
        """Report the measurement and enforce the bounds of the suite.

        The bounds are the metrics' own bounds: they are printed everywhere
        and enforced by ``check_perf`` when the run happens in CI.
        """
        gflops = case.flops / (kernel["median"] / 1000.0) / 1e9
        tokens_per_s = workload.tokens / (kernel["median"] / 1000.0)
        bounds = SUITE.bounds
        slowdown = (
            kernel["median"] / baseline["median"] if baseline["median"] > 0 else 0.0
        )
        perf_kit.check_perf(
            self,
            perf_kit.at_most(
                "kernel_latency_ms",
                kernel["median"],
                bounds.max_latency_ms,
                unit="ms",
            ),
            perf_kit.at_most("slowdown_vs_torch", slowdown, bounds.max_slowdown),
            perf_kit.reported("torch_latency_ms", baseline["median"], unit="ms"),
            perf_kit.reported("speedup", baseline["median"] / kernel["median"]),
            perf_kit.reported("gflops", gflops),
            perf_kit.reported("tokens_per_s", tokens_per_s),
            suffix=f"_{op}_{workload.name}",
        )

    def _run_group(self, op, group):
        """Run every scenario of ``group`` for ``op``."""
        for scenario in SUITE.scenarios:
            if scenario.group != group:
                continue
            if not scenario.applies_to(op):
                continue
            with self.subTest(op=op, scenario=scenario.name):
                workload = bench.to_workload(scenario)
                case, kernel, baseline = self._measure(op, workload)
                self._assert_healthy(op, workload, kernel, baseline)
                self._report(op, workload, case, kernel, baseline)

    def _run_op(self, op):
        """Run the default scenarios of the suite for ``op``."""
        self._run_group(op, "default")

    @requires_op("sgmv_shrink")
    def test_sgmv_shrink_performance(self):
        self._run_op("sgmv_shrink")

    @requires_op("sgmv_expand")
    def test_sgmv_expand_performance(self):
        self._run_op("sgmv_expand")

    @requires_op("bgmv_shrink")
    def test_bgmv_shrink_performance(self):
        self._run_op("bgmv_shrink")

    @requires_op("bgmv_expand")
    def test_bgmv_expand_performance(self):
        self._run_op("bgmv_expand")

    @requires_op("sgemmv_shrink")
    def test_sgemmv_shrink_performance(self):
        self._run_op("sgemmv_shrink")

    @requires_op("sgemmv_expand")
    def test_sgemmv_expand_performance(self):
        self._run_op("sgemmv_expand")

    @requires_op("sgemmc_shrink")
    def test_sgemmc_shrink_performance(self):
        self._run_op("sgemmc_shrink")

    @requires_op("sgemmc_expand")
    def test_sgemmc_expand_performance(self):
        self._run_op("sgemmc_expand")

    def test_fused_slices_performance(self):
        """The grouped operators must also perform well with fused QKV slices."""
        for op in perf_config.GROUPED_OPS:
            if not op_available(op):
                continue
            self._run_group(op, "fused-slices")

    def test_mixed_ranks_performance(self):
        """Adapters with different ranks must not break the grouped path."""
        for op in perf_config.GROUPED_OPS:
            if not op_available(op):
                continue
            self._run_group(op, "mixed-ranks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
