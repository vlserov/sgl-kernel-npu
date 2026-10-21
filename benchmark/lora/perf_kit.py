"""Report and bound the numbers a LoRA performance run produces.

Mirrors the SGLang metric contract
(https://github.com/sgl-project/sglang,
``python/sglang/test/kits/perf_bench_kit.py``): a metric carries its own
bound, the report is printed for the CI log, and the bound is only enforced
when the run happens in CI.  A local run therefore prints the bounds without
asserting them, which keeps a slow workstation from failing the suite.

The bounds themselves come from the suite config
(``benchmark/lora/configs/*.yaml``), not from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Environment variable that marks a CI run.  It is set by the CI workflow,
# not by the developer, and it is the only environment variable the perf
# suites look at.
CI_ENV_VAR = "SGL_KERNEL_NPU_IS_IN_CI"

# Environment variable pointing at the file the CI summary is appended to.
SUMMARY_ENV_VAR = "GITHUB_STEP_SUMMARY"


def is_in_ci() -> bool:
    """Return whether the current run is a CI run."""
    return os.environ.get(CI_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def write_github_step_summary(content: str) -> None:
    """Append ``content`` to the CI step summary, if one is configured."""
    path = os.environ.get(SUMMARY_ENV_VAR)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(content)
    except OSError:
        # A summary is a nicety; never fail a performance run because of it.
        pass


@dataclass
class Metric:
    """One measured number and the bound it must respect."""

    name: str
    value: float
    unit: str = ""
    bound: Optional[float] = None
    # unittest method that enforces `bound`, e.g. "assertLessEqual".
    assertion: Optional[str] = None

    def line(self) -> str:
        text = f"{self.name}: {self.value:.4f}"
        if self.unit:
            text += f" {self.unit}"
        if self.bound is not None:
            text += f" (bound {self.bound:.4f})"
        return text

    def check(self, test_case) -> None:
        if self.bound is None:
            return
        getattr(test_case, self.assertion)(self.value, self.bound)


def at_least(name: str, value: float, bound: float, *, unit: str = "") -> Metric:
    """A throughput-like number: the run passes when it reaches ``bound``."""
    return Metric(name, value, unit, bound, "assertGreaterEqual")


def at_most(name: str, value: float, bound: float, *, unit: str = "") -> Metric:
    """A latency-like number: the run passes when it stays under ``bound``."""
    return Metric(name, value, unit, bound, "assertLessEqual")


def reported(name: str, value: float, *, unit: str = "") -> Metric:
    """A number worth printing that no threshold is attached to."""
    return Metric(name, value, unit)


def check_perf(test_case, *metrics: Metric, suffix: str = "") -> None:
    """Report every metric under the caller's test name, then enforce in CI.

    Outside CI the bounds are printed but not asserted, so a local run stays
    informative without gating on the speed of the machine it runs on.
    """
    label = getattr(test_case, "_testMethodName", "performance") + suffix
    report = f"### {label}\n" + "".join(metric.line() + "\n" for metric in metrics)
    print(report, end="", flush=True)
    if not is_in_ci():
        return
    write_github_step_summary(report)
    for metric in metrics:
        metric.check(test_case)


__all__ = (
    "CI_ENV_VAR",
    "Metric",
    "SUMMARY_ENV_VAR",
    "at_least",
    "at_most",
    "check_perf",
    "is_in_ci",
    "reported",
    "write_github_step_summary",
)
