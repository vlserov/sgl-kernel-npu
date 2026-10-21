"""Unit tests for the LoRA performance suite configuration.

The suites under ``benchmark/lora/configs/`` are the single source of truth
for the LoRA performance test, so they need the same protection as code: a
malformed suite must fail loudly instead of silently skipping a bound.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "benchmark" / "lora"
sys.path.insert(0, str(BENCHMARK_DIR))

import perf_config
import perf_kit


class TestSuiteLoading(unittest.TestCase):
    def test_shipped_suites_load_and_validate(self):
        suites = perf_config.available_suites()
        self.assertTrue(suites, "no suite found under benchmark/lora/configs")
        for path in suites:
            with self.subTest(suite=path.name):
                suite = perf_config.load_suite(path)
                self.assertTrue(suite.name)
                self.assertTrue(suite.scenarios)
                self.assertGreater(suite.bounds.max_latency_ms, 0.0)
                self.assertGreater(suite.bounds.max_slowdown, 0.0)

    def test_ci_suite_covers_every_operator(self):
        suite = perf_config.load_suite_by_name("lora_perf_ci")
        self.assertEqual(set(suite.ops), set(perf_config.ALL_OPS))

    def test_ci_suite_has_a_default_group_for_every_operator(self):
        """Every operator needs at least one scenario it can run."""
        suite = perf_config.load_suite_by_name("lora_perf_ci")
        for op in suite.ops:
            with self.subTest(op=op):
                applicable = [
                    scenario
                    for scenario in suite.scenarios
                    if scenario.group == "default" and scenario.applies_to(op)
                ]
                self.assertTrue(applicable, f"no default scenario applies to {op}")

    def test_fixed_rank_operators_never_see_mixed_ranks(self):
        """sgmv/bgmv share one rank, so a mixed-rank scenario must skip them."""
        suite = perf_config.load_suite_by_name("lora_perf_ci")
        mixed = [
            scenario for scenario in suite.scenarios if len(set(scenario.ranks)) > 1
        ]
        self.assertTrue(mixed, "the CI suite no longer covers mixed ranks")
        for scenario in mixed:
            for op in ("sgmv_shrink", "sgmv_expand", "bgmv_shrink", "bgmv_expand"):
                with self.subTest(op=op, scenario=scenario.name):
                    self.assertFalse(scenario.applies_to(op))


class TestSuiteValidation(unittest.TestCase):
    def _suite(self, **overrides):
        data = {
            "name": "test-suite",
            "scenarios": [{"name": "s1", "tokens": 8, "seq_len": 1}],
        }
        data.update(overrides)
        return data

    def test_defaults_are_applied_to_the_scenarios(self):
        suite = perf_config.parse_suite(
            self._suite(defaults={"hidden": 1024, "ranks": [8, 32]})
        )
        scenario = suite.scenarios[0]
        self.assertEqual(scenario.hidden, 1024)
        self.assertEqual(scenario.ranks, [8, 32])
        self.assertEqual(scenario.dtype, perf_config.DEFAULT_DTYPE)

    def test_scenario_overrides_the_defaults(self):
        suite = perf_config.parse_suite(
            self._suite(
                defaults={"hidden": 1024},
                scenarios=[{"name": "s1", "tokens": 8, "seq_len": 1, "hidden": 2048}],
            )
        )
        self.assertEqual(suite.scenarios[0].hidden, 2048)

    def test_missing_name_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.parse_suite(
                {"name": "x", "scenarios": [{"tokens": 8, "seq_len": 1}]}
            )

    def test_missing_scenarios_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.parse_suite({"name": "x", "scenarios": []})

    def test_unsupported_rank_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.parse_suite(
                self._suite(
                    scenarios=[{"name": "s1", "tokens": 8, "seq_len": 1, "ranks": [7]}]
                )
            )

    def test_unknown_op_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.parse_suite(
                self._suite(
                    scenarios=[
                        {"name": "s1", "tokens": 8, "seq_len": 1, "ops": ["nope"]}
                    ]
                )
            )

    def test_duplicate_scenario_name_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.parse_suite(
                self._suite(
                    scenarios=[
                        {"name": "s1", "tokens": 8, "seq_len": 1},
                        {"name": "s1", "tokens": 16, "seq_len": 1},
                    ]
                )
            )

    def test_non_positive_iters_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.parse_suite(
                self._suite(
                    scenarios=[
                        {
                            "name": "s1",
                            "tokens": 8,
                            "seq_len": 1,
                            "timing": {"iters": 0},
                        }
                    ]
                )
            )

    def test_unknown_suite_is_rejected(self):
        with self.assertRaises(perf_config.ConfigError):
            perf_config.load_suite_by_name("does-not-exist")


class TestPerfKit(unittest.TestCase):
    def test_at_most_uses_the_less_equal_assertion(self):
        metric = perf_kit.at_most("latency", 1.0, 2.0, unit="ms")
        self.assertEqual(metric.assertion, "assertLessEqual")
        self.assertIn("bound 2.0000", metric.line())

    def test_at_least_uses_the_greater_equal_assertion(self):
        metric = perf_kit.at_least("throughput", 1.0, 2.0)
        self.assertEqual(metric.assertion, "assertGreaterEqual")

    def test_reported_metric_has_no_bound(self):
        metric = perf_kit.reported("gflops", 1.0)
        self.assertIsNone(metric.bound)
        self.assertIsNone(metric.assertion)

    def test_metric_check_enforces_the_bound(self):
        class Recorder:
            def __init__(self):
                self.calls = []

            def assertLessEqual(self, value, bound):
                self.calls.append((value, bound))

        recorder = Recorder()
        perf_kit.at_most("latency", 1.0, 2.0).check(recorder)
        self.assertEqual(recorder.calls, [(1.0, 2.0)])

    def test_metric_check_is_a_no_op_without_a_bound(self):
        class Recorder:
            def assertLessEqual(self, value, bound):  # pragma: no cover
                raise AssertionError("must not be called")

        perf_kit.reported("gflops", 1.0).check(Recorder())


if __name__ == "__main__":
    unittest.main(verbosity=2)
