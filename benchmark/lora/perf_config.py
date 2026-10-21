"""Configuration model for the LoRA performance suites.

The performance suites are described by YAML files under
``benchmark/lora/configs/`` instead of environment variables, so that a
suite is a reviewable artifact: the shapes, the timing loop and the
regression bounds live next to each other and are diffed like code.

The layout mirrors the SGLang approach
(https://github.com/sgl-project/sglang):

* ``scripts/ci/runner_configs.yml`` is the single source of truth for the
  CI runner setup, parsed by ``scripts/ci/runner_configs.py``.
* ``python/sglang/test/kits/perf_bench_kit.py`` carries the metric contract
  (``at_most`` / ``at_least`` / ``check_perf``) used by the perf tests.

The two counterparts here are ``configs/*.yaml`` and ``perf_kit.py``.

Config schema
-------------

```yaml
name: lora-perf-ci              # suite name, used in the report
description: ...                # free-form, printed in the report
defaults:                       # optional, applies to every scenario
  dtype: float16
  hidden: 4096
  num_loras: 8
  ranks: [16]
  slice_widths: [4096]
  index_mode: random
  warmup: 5
  iters: 20
  repeat: 5
  seed: 20260620
  baseline: true
  check: true
scenarios:                       # at least one
  - name: decode-b64
    tokens: 64
    seq_len: 1
  - name: prefill-t2048
    tokens: 2048
    seq_len: 128
    slice_widths: [1024, 1024, 1024]   # overrides the default
    ranks: [8, 16, 32, 64]            # overrides the default
bounds:                         # optional, enforced only in CI
  max_latency_ms: 100.0         # hard upper bound per iteration
  max_slowdown: 50.0            # kernel latency / torch baseline latency
  strict_check: false           # fail the test on a numerical mismatch
```

Every field of ``defaults`` can be overridden per scenario, and every
``defaults`` field is itself optional: the loader fills in the values from
``DEFAULT_*`` below when a key is missing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

# Directory that holds the YAML suites.
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Operators the suites may reference.
ALL_OPS = (
    "sgmv_shrink",
    "sgmv_expand",
    "sgemmv_shrink",
    "sgemmv_expand",
    "sgemmc_shrink",
    "sgemmc_expand",
    "bgmv_shrink",
    "bgmv_expand",
)

# The ranks the expand kernels have a dedicated reduction chain for.
SUPPORTED_RANKS = (8, 16, 32, 64)

DTYPES = ("float16", "bfloat16")

INDEX_MODES = ("random", "single")

# The group a scenario belongs to.  The CI test routes a scenario to a test
# method by its group.
DEFAULT_GROUP = "default"

# The operators that support a per-adapter rank.  A scenario with mixed
# ranks can only be measured with these.
RANKED_OPS = ("sgemmv_shrink", "sgemmv_expand", "sgemmc_shrink", "sgemmc_expand")

# The operators that can fuse several output projections into one call.
GROUPED_OPS = ("sgemmv_shrink", "sgemmv_expand", "sgemmc_shrink", "sgemmc_expand")

# Values used when the config omits a key.
DEFAULT_DTYPE = "float16"
DEFAULT_HIDDEN = 4096
DEFAULT_NUM_LORAS = 8
DEFAULT_RANKS = (16,)
DEFAULT_SLICE_WIDTHS = (4096,)
DEFAULT_INDEX_MODE = "random"
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_REPEAT = 5
DEFAULT_SEED = 20260620
DEFAULT_BASELINE = True
DEFAULT_CHECK = True

# Values used when the config omits a bound.
DEFAULT_MAX_LATENCY_MS = 100.0
DEFAULT_MAX_SLOWDOWN = 50.0
DEFAULT_STRICT_CHECK = False


class ConfigError(ValueError):
    """Raised when a suite file is malformed."""


@dataclass
class TimingConfig:
    """How a single measurement is taken."""

    warmup: int = DEFAULT_WARMUP
    iters: int = DEFAULT_ITERS
    repeat: int = DEFAULT_REPEAT

    def validate(self, where: str) -> None:
        if self.warmup < 0:
            raise ConfigError(f"{where}: warmup must be non-negative")
        if self.iters <= 0:
            raise ConfigError(f"{where}: iters must be positive")
        if self.repeat <= 0:
            raise ConfigError(f"{where}: repeat must be positive")


@dataclass
class ScenarioConfig:
    """One shape configuration to measure."""

    name: str
    tokens: int
    seq_len: int
    dtype: str = DEFAULT_DTYPE
    hidden: int = DEFAULT_HIDDEN
    num_loras: int = DEFAULT_NUM_LORAS
    ranks: List[int] = field(default_factory=lambda: list(DEFAULT_RANKS))
    slice_widths: List[int] = field(default_factory=lambda: list(DEFAULT_SLICE_WIDTHS))
    index_mode: str = DEFAULT_INDEX_MODE
    timing: TimingConfig = field(default_factory=TimingConfig)
    baseline: bool = DEFAULT_BASELINE
    check: bool = DEFAULT_CHECK
    seed: int = DEFAULT_SEED
    # The operators the scenario applies to.  ``None`` means every operator.
    # The fixed-rank operators (sgmv/bgmv) cannot run a mixed-rank scenario,
    # and the fused-slice scenario only makes sense for the grouped ones.
    ops: Optional[List[str]] = None
    # The test group the scenario belongs to.  The CI test routes a scenario
    # to a test method by its group, so the suite stays the single source of
    # truth for what is measured.
    group: str = DEFAULT_GROUP

    @property
    def slices(self) -> int:
        return len(self.slice_widths)

    @property
    def total_out(self) -> int:
        return sum(self.slice_widths)

    def applies_to(self, op: str) -> bool:
        """Whether the scenario should be measured for ``op``."""
        return self.ops is None or op in self.ops

    def validate(self, where: str) -> None:
        if not self.name:
            raise ConfigError(f"{where}: scenario name must not be empty")
        if self.tokens <= 0:
            raise ConfigError(f"{where}: tokens must be positive")
        if self.seq_len <= 0:
            raise ConfigError(f"{where}: seq_len must be positive")
        if self.dtype not in DTYPES:
            raise ConfigError(f"{where}: unknown dtype {self.dtype!r}")
        if self.hidden <= 0:
            raise ConfigError(f"{where}: hidden must be positive")
        if self.num_loras <= 0:
            raise ConfigError(f"{where}: num_loras must be positive")
        if not self.ranks:
            raise ConfigError(f"{where}: ranks must not be empty")
        for rank in self.ranks:
            if rank not in SUPPORTED_RANKS:
                raise ConfigError(
                    f"{where}: rank {rank} is not supported, expected one of "
                    f"{list(SUPPORTED_RANKS)}"
                )
        if not self.slice_widths:
            raise ConfigError(f"{where}: slice_widths must not be empty")
        for width in self.slice_widths:
            if width <= 0:
                raise ConfigError(f"{where}: slice_widths must be positive")
        if self.index_mode not in INDEX_MODES:
            raise ConfigError(f"{where}: unknown index_mode {self.index_mode!r}")
        if self.ops is not None:
            if not self.ops:
                raise ConfigError(f"{where}: ops must not be empty when given")
            for op in self.ops:
                if op not in ALL_OPS:
                    raise ConfigError(f"{where}: unknown op {op!r}")
        if not self.group:
            raise ConfigError(f"{where}: group must not be empty")
        self.timing.validate(f"{where}")


@dataclass
class BoundsConfig:
    """The regression contract enforced by the CI test.

    The bounds are intentionally generous: they only catch a kernel that
    became orders of magnitude slower (or stopped making progress), not
    small performance changes.
    """

    # Hard upper bound per iteration, in milliseconds.
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS
    # Maximum kernel latency / torch baseline latency ratio.
    max_slowdown: float = DEFAULT_MAX_SLOWDOWN
    # Fail the test on a numerical mismatch instead of warning.
    strict_check: bool = DEFAULT_STRICT_CHECK

    def validate(self, where: str) -> None:
        if self.max_latency_ms <= 0:
            raise ConfigError(f"{where}: max_latency_ms must be positive")
        if self.max_slowdown <= 0:
            raise ConfigError(f"{where}: max_slowdown must be positive")


@dataclass
class SuiteConfig:
    """A complete performance suite."""

    name: str
    description: str = ""
    ops: List[str] = field(default_factory=lambda: list(ALL_OPS))
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    source: Optional[Path] = None

    def validate(self) -> None:
        where = str(self.source) if self.source else f"suite {self.name!r}"
        if not self.name:
            raise ConfigError("suite name must not be empty")
        if not self.ops:
            raise ConfigError(f"{where}: ops must not be empty")
        for op in self.ops:
            if op not in ALL_OPS:
                raise ConfigError(f"{where}: unknown op {op!r}")
        if not self.scenarios:
            raise ConfigError(f"{where}: at least one scenario is required")
        seen = set()
        for scenario in self.scenarios:
            scenario.validate(where)
            if scenario.name in seen:
                raise ConfigError(f"{where}: duplicate scenario {scenario.name!r}")
            seen.add(scenario.name)
        self.bounds.validate(where)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, where: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    return value


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _as_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    return value


def _as_int_list(value: Any, where: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where}: expected a non-empty list, got {value!r}")
    return [_as_int(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _as_str_list(value: Any, where: str) -> List[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where}: expected a non-empty list, got {value!r}")
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{where}: expected strings, got {item!r}")
    return list(value)


def _parse_timing(raw: Dict[str, Any], where: str) -> TimingConfig:
    timing = TimingConfig()
    for key in ("warmup", "iters", "repeat"):
        if key in raw:
            setattr(timing, key, _as_int(raw[key], f"{where}.{key}"))
    return timing


def _parse_bounds(raw: Dict[str, Any], where: str) -> BoundsConfig:
    bounds = BoundsConfig()
    if "max_latency_ms" in raw:
        bounds.max_latency_ms = _as_float(
            raw["max_latency_ms"], f"{where}.max_latency_ms"
        )
    if "max_slowdown" in raw:
        bounds.max_slowdown = _as_float(raw["max_slowdown"], f"{where}.max_slowdown")
    if "strict_check" in raw:
        bounds.strict_check = _as_bool(raw["strict_check"], f"{where}.strict_check")
    return bounds


def _parse_scenario(
    raw: Dict[str, Any], defaults: Dict[str, Any], where: str
) -> ScenarioConfig:
    merged = copy.deepcopy(defaults)
    merged.update(raw)

    name = merged.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: 'name' is required and must be a string")

    for key in ("tokens", "seq_len"):
        if key not in merged:
            raise ConfigError(f"{where}: '{key}' is required")

    timing_raw = _require_mapping(merged.get("timing"), f"{where}.timing")
    timing = _parse_timing(timing_raw, f"{where}.timing")

    scenario = ScenarioConfig(
        name=name,
        tokens=_as_int(merged["tokens"], f"{where}.tokens"),
        seq_len=_as_int(merged["seq_len"], f"{where}.seq_len"),
        dtype=merged.get("dtype", DEFAULT_DTYPE),
        hidden=_as_int(merged.get("hidden", DEFAULT_HIDDEN), f"{where}.hidden"),
        num_loras=_as_int(
            merged.get("num_loras", DEFAULT_NUM_LORAS), f"{where}.num_loras"
        ),
        ranks=(
            _as_int_list(merged["ranks"], f"{where}.ranks")
            if "ranks" in merged
            else list(DEFAULT_RANKS)
        ),
        slice_widths=(
            _as_int_list(merged["slice_widths"], f"{where}.slice_widths")
            if "slice_widths" in merged
            else list(DEFAULT_SLICE_WIDTHS)
        ),
        index_mode=merged.get("index_mode", DEFAULT_INDEX_MODE),
        timing=timing,
        baseline=_as_bool(
            merged.get("baseline", DEFAULT_BASELINE), f"{where}.baseline"
        ),
        check=_as_bool(merged.get("check", DEFAULT_CHECK), f"{where}.check"),
        seed=_as_int(merged.get("seed", DEFAULT_SEED), f"{where}.seed"),
        ops=(
            _as_str_list(merged["ops"], f"{where}.ops")
            if merged.get("ops") is not None
            else None
        ),
        group=merged.get("group", DEFAULT_GROUP),
    )
    return scenario


def parse_suite(data: Any, source: Optional[Path] = None) -> SuiteConfig:
    """Build a :class:`SuiteConfig` from a parsed YAML document."""
    where = str(source) if source else "<suite>"
    raw = _require_mapping(data, where)

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: 'name' is required and must be a string")

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ConfigError(f"{where}: 'description' must be a string")

    ops = _as_str_list(raw["ops"], f"{where}.ops") if "ops" in raw else list(ALL_OPS)

    defaults = _require_mapping(raw.get("defaults"), f"{where}.defaults")

    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, (list, tuple)) or not raw_scenarios:
        raise ConfigError(f"{where}: 'scenarios' must be a non-empty list")

    scenarios = [
        _parse_scenario(
            _require_mapping(item, f"{where}.scenarios[{index}]"),
            defaults,
            f"{where}.scenarios[{index}]",
        )
        for index, item in enumerate(raw_scenarios)
    ]

    bounds = _parse_bounds(
        _require_mapping(raw.get("bounds"), f"{where}.bounds"), f"{where}.bounds"
    )

    suite = SuiteConfig(
        name=name,
        description=description,
        ops=ops,
        scenarios=scenarios,
        bounds=bounds,
        source=source,
    )
    suite.validate()
    return suite


def load_suite(path: Path) -> SuiteConfig:
    """Load and validate a suite from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"suite file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_suite(data, source=path)


def available_suites() -> List[Path]:
    """Return the YAML suites shipped next to this module, sorted by name."""
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(CONFIG_DIR.glob("*.yaml"))


def load_suite_by_name(name: str) -> SuiteConfig:
    """Load a suite by file name (with or without the ``.yaml`` suffix)."""
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = CONFIG_DIR / name
    if not path.is_file():
        known = ", ".join(p.name for p in available_suites())
        raise ConfigError(f"unknown suite {name!r}; available suites: {known}")
    return load_suite(path)


def scenario_overrides(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Return the per-scenario overrides of the suite ``defaults``."""
    return {
        "dtype": scenario.dtype,
        "hidden": scenario.hidden,
        "num_loras": scenario.num_loras,
        "ranks": list(scenario.ranks),
        "slice_widths": list(scenario.slice_widths),
        "index_mode": scenario.index_mode,
        "seed": scenario.seed,
        "ops": list(scenario.ops) if scenario.ops else None,
        "group": scenario.group,
    }


__all__: Sequence[str] = (
    "ALL_OPS",
    "GROUPED_OPS",
    "RANKED_OPS",
    "SUPPORTED_RANKS",
    "CONFIG_DIR",
    "BoundsConfig",
    "ConfigError",
    "ScenarioConfig",
    "SuiteConfig",
    "TimingConfig",
    "available_suites",
    "load_suite",
    "load_suite_by_name",
    "parse_suite",
    "scenario_overrides",
)
