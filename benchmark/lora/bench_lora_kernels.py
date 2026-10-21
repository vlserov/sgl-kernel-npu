"""Performance benchmark for the Ascend LoRA kernels.

Covers the eight LoRA operators exported by sgl-kernel-npu:

    shrink (hidden -> rank)            expand (rank -> hidden)
    ------------------------------      ------------------------------
    sgmv_shrink                        sgmv_expand
    sgemmv_shrink                      sgemmv_expand
    sgemmc_shrink                      sgemmc_expand
    bgmv_shrink                        bgmv_expand

Every operator is measured against a pure-PyTorch grouped-matmul baseline
that performs exactly the same arithmetic, so the reported speedup is the
kernel speedup and not a difference in work.

The benchmark also validates the kernel output against the same reference
before timing it, so a broken kernel cannot silently look fast.

The sweep is described by a YAML suite under ``benchmark/lora/configs/``,
not by environment variables.  The CLI flags below are overrides on top of
the selected suite, so the common case is "run the suite" and the rare case
is "run the suite with one knob changed".

Usage:
    # Default sweep (the lora_perf suite: decode + prefill, float16, all ops)
    python benchmark/lora/bench_lora_kernels.py

    # Another suite
    python benchmark/lora/bench_lora_kernels.py --suite lora_perf_ci

    # Only the ranked grouped GEMV path
    python benchmark/lora/bench_lora_kernels.py --ops sgemmv_shrink sgemmv_expand

    # Custom shape: 8 LoRA adapters with mixed ranks, bfloat16
    python benchmark/lora/bench_lora_kernels.py --dtype bfloat16 \
        --ranks 8 16 32 64 --num-loras 8 --hidden 4096

    # QKV-style fused slices
    python benchmark/lora/bench_lora_kernels.py --slice-widths 1024 1024 1024

    # Machine readable results
    python benchmark/lora/bench_lora_kernels.py --json lora_perf.json
"""

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# The suite helpers live next to this script; make them importable no matter
# how the script is launched (directly, or through importlib from a test).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import perf_config  # noqa: E402
import perf_kit  # noqa: E402
import sgl_kernel_npu  # noqa: F401  (registers the torch.ops.npu operators)
import torch
import torch_npu  # noqa: F401

DEVICE = "npu"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Ranks the expand kernels have a dedicated reduction chain for.
SUPPORTED_RANKS = perf_config.SUPPORTED_RANKS

ALL_OPS = perf_config.ALL_OPS

# Ops that read a per-adapter rank tensor.
RANKED_OPS = perf_config.RANKED_OPS

# Tolerance of the correctness check that runs before the timing loop.
SHRINK_ATOL = 5e-3
EXPAND_ATOL = 2e-2
RTOL = 2e-2


def log(message: str) -> None:
    print(message, flush=True)


def synchronize() -> None:
    torch.npu.synchronize()


@dataclass
class Workload:
    """A single shape configuration to benchmark.

    This is the resolved form of a ``perf_config.ScenarioConfig``: every knob
    the sweep can turn is already spelled out.
    """

    name: str
    tokens: int
    seq_len: int
    num_loras: int
    hidden: int
    slice_widths: List[int]
    dtype: str = perf_config.DEFAULT_DTYPE
    ranks: List[int] = field(default_factory=lambda: list(perf_config.DEFAULT_RANKS))
    index_mode: str = perf_config.DEFAULT_INDEX_MODE
    seed: int = perf_config.DEFAULT_SEED
    warmup: int = perf_config.DEFAULT_WARMUP
    iters: int = perf_config.DEFAULT_ITERS
    repeat: int = perf_config.DEFAULT_REPEAT
    baseline: bool = perf_config.DEFAULT_BASELINE
    check: bool = perf_config.DEFAULT_CHECK
    ops: Optional[List[str]] = None
    group: str = perf_config.DEFAULT_GROUP

    @property
    def slices(self) -> int:
        return len(self.slice_widths)

    @property
    def total_out(self) -> int:
        return sum(self.slice_widths)

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype]

    def applies_to(self, op: str) -> bool:
        """Whether this workload should be measured for ``op``."""
        return self.ops is None or op in self.ops


@dataclass
class LoraMeta:
    """Per-adapter metadata shared by all operators of one workload."""

    ranks: List[int]
    scales: List[float]
    seq_lens: List[int]
    seq_adapters: List[int]
    token_adapters: List[int] = field(default_factory=list)

    @property
    def max_rank(self) -> int:
        return max(self.ranks)


@dataclass
class Case:
    """Everything needed to run, validate and time one operator."""

    op: str
    workload: Workload
    meta: LoraMeta
    run: Callable[[], None]
    reference: Callable[[], torch.Tensor]
    reset: Optional[Callable[[], None]]
    out_dtype: torch.dtype
    flops: float
    output: Optional[torch.Tensor] = None
    detail: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Suite -> workload
# ---------------------------------------------------------------------------


def to_workload(scenario: perf_config.ScenarioConfig) -> Workload:
    """Convert a suite scenario into the resolved workload the sweep runs."""
    return Workload(
        name=scenario.name,
        tokens=scenario.tokens,
        seq_len=scenario.seq_len,
        num_loras=scenario.num_loras,
        hidden=scenario.hidden,
        slice_widths=list(scenario.slice_widths),
        dtype=scenario.dtype,
        ranks=list(scenario.ranks),
        index_mode=scenario.index_mode,
        seed=scenario.seed,
        warmup=scenario.timing.warmup,
        iters=scenario.timing.iters,
        repeat=scenario.timing.repeat,
        baseline=scenario.baseline,
        check=scenario.check,
        ops=list(scenario.ops) if scenario.ops else None,
        group=scenario.group,
    )


def build_workloads(suite: perf_config.SuiteConfig) -> List[Workload]:
    """Resolve every scenario of the suite into a workload."""
    return [to_workload(scenario) for scenario in suite.scenarios]


# ---------------------------------------------------------------------------
# Workload / metadata construction
# ---------------------------------------------------------------------------


def build_meta(
    workload: Workload,
    uniform: bool = False,
) -> LoraMeta:
    """Create the per-adapter ranks/scales and the token -> adapter mapping.

    ``uniform`` mirrors the fixed-rank operators (sgmv/bgmv) that share a
    single rank and a single scale across all adapters.
    """
    ranks = list(workload.ranks)
    seed = workload.seed
    rng = random.Random(seed)
    if uniform:
        adapter_ranks = [max(ranks)] * workload.num_loras
    else:
        adapter_ranks = [ranks[i % len(ranks)] for i in range(workload.num_loras)]
    adapter_scales = [round(rng.uniform(0.25, 2.0), 3) for _ in adapter_ranks]
    if uniform:
        adapter_scales = [adapter_scales[0]] * workload.num_loras

    num_sequences = max(1, math.ceil(workload.tokens / max(1, workload.seq_len)))
    seq_lens = []
    remaining = workload.tokens
    for _ in range(num_sequences):
        length = min(workload.seq_len, remaining)
        seq_lens.append(length)
        remaining -= length
    seq_lens = [length for length in seq_lens if length > 0]

    if workload.index_mode == "single":
        seq_adapters = [0] * len(seq_lens)
    else:
        seq_adapters = [rng.randrange(workload.num_loras) for _ in seq_lens]

    token_adapters = []
    for adapter, length in zip(seq_adapters, seq_lens):
        token_adapters.extend([adapter] * length)

    return LoraMeta(
        ranks=adapter_ranks,
        scales=adapter_scales,
        seq_lens=seq_lens,
        seq_adapters=seq_adapters,
        token_adapters=token_adapters,
    )


def build_shrink_weight_a(
    meta: LoraMeta,
    slices: int,
    max_rank: int,
    hidden: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build the packed LoRA A weight: [num_loras, slices * max_rank, hidden].

    The shrink kernels address slice ``s`` of an adapter at row ``s * rank``
    where ``rank`` is the *adapter's* rank, so every adapter stores its slices
    back to back and the remaining rows are padding.
    """
    weight = torch.zeros(len(meta.ranks), slices * max_rank, hidden, dtype=dtype)
    for slot, rank in enumerate(meta.ranks):
        if rank == 0:
            continue
        for slice_idx in range(slices):
            block = torch.randn(rank, hidden, generator=generator) * 0.02
            weight[slot, slice_idx * rank : (slice_idx + 1) * rank, :] = block.to(dtype)
    return weight.to(DEVICE)


def build_expand_weight_b(
    meta: LoraMeta,
    total_out: int,
    max_rank: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build the LoRA B weight: [num_loras, total_out, max_rank].

    Only the first ``rank`` columns of an adapter are meaningful, the rest is
    padding for adapters with a smaller rank.
    """
    weight = torch.zeros(len(meta.ranks), total_out, max_rank, dtype=dtype)
    for slot, rank in enumerate(meta.ranks):
        if rank == 0:
            continue
        block = torch.randn(total_out, rank, generator=generator) * 0.02
        weight[slot, :, :rank] = block.to(dtype)
    return weight.to(DEVICE)


# ---------------------------------------------------------------------------
# Pure PyTorch baselines (also used as the correctness reference)
# ---------------------------------------------------------------------------


def torch_grouped_shrink(
    x: torch.Tensor,
    weight_a: torch.Tensor,
    meta: LoraMeta,
    slices: int,
    max_rank: int,
    out_dtype: torch.dtype,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Grouped per-adapter matmul reference for the shrink operators.

    ``scale`` overrides the per-adapter scales, which is what the fixed-rank
    operators (sgmv/bgmv) do: they apply a single scale to every adapter.
    """
    out = torch.zeros(
        x.size(0), slices * max_rank, dtype=torch.float32, device=x.device
    )
    for slot, rank in enumerate(meta.ranks):
        if rank == 0:
            continue
        token_index = torch.nonzero(
            torch.tensor(meta.token_adapters, device=x.device) == slot, as_tuple=True
        )[0]
        if token_index.numel() == 0:
            continue
        xs = x.index_select(0, token_index).float()
        ws = weight_a[slot, : slices * rank].float()
        factor = meta.scales[slot] if scale is None else scale
        out[token_index, : slices * rank] = (xs @ ws.t()) * factor
    return out.to(out_dtype)


def torch_grouped_expand(
    x: torch.Tensor,
    weight_b: torch.Tensor,
    meta: LoraMeta,
    slice_offsets: List[int],
    packed: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped per-adapter matmul reference for the expand operators."""
    out = torch.zeros(x.size(0), weight_b.size(1), dtype=torch.float32, device=x.device)
    for slot, rank in enumerate(meta.ranks):
        if rank == 0:
            continue
        token_index = torch.nonzero(
            torch.tensor(meta.token_adapters, device=x.device) == slot, as_tuple=True
        )[0]
        if token_index.numel() == 0:
            continue
        xs = x.index_select(0, token_index).float()
        for slice_idx in range(len(slice_offsets) - 1):
            left = slice_offsets[slice_idx]
            right = slice_offsets[slice_idx + 1]
            if packed:
                x_slice = xs[:, slice_idx * rank : (slice_idx + 1) * rank]
            else:
                x_slice = xs
            w_slice = weight_b[slot, left:right, :rank].float()
            out[token_index, left:right] = (
                out[token_index, left:right] + x_slice @ w_slice.t()
            )
    return out.to(out_dtype)


def torch_fixed_expand(
    x: torch.Tensor,
    weight_slices: Sequence[torch.Tensor],
    meta: LoraMeta,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference for the fixed-rank expand operators (one weight per slice)."""
    total_out = sum(weight.size(1) for weight in weight_slices)
    out = torch.zeros(x.size(0), total_out, dtype=torch.float32, device=x.device)
    for slot, rank in enumerate(meta.ranks):
        if rank == 0:
            continue
        token_index = torch.nonzero(
            torch.tensor(meta.token_adapters, device=x.device) == slot, as_tuple=True
        )[0]
        if token_index.numel() == 0:
            continue
        xs = x.index_select(0, token_index).float()
        offset = 0
        for weight in weight_slices:
            width = weight.size(1)
            w_slice = weight[slot, :, :rank].float()
            out[token_index, offset : offset + width] = (
                out[token_index, offset : offset + width] + xs @ w_slice.t()
            )
            offset += width
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def build_grouped_shrink_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    """sgemmv_shrink / sgemmc_shrink: grouped GEMV with per-adapter rank+scale."""
    slices = workload.slices
    max_rank = meta.max_rank
    generator = torch.Generator().manual_seed(seed)

    x = (torch.randn(workload.tokens, workload.hidden, generator=generator)).to(
        device=DEVICE, dtype=dtype
    )
    weight_a = build_shrink_weight_a(
        meta, slices, max_rank, workload.hidden, dtype, generator
    )

    lora_indices = torch.tensor(meta.seq_adapters, dtype=torch.int32, device=DEVICE)
    seq_len = torch.tensor(meta.seq_lens, dtype=torch.int32, device=DEVICE)
    lora_ranks = torch.tensor(meta.ranks, dtype=torch.int32, device=DEVICE)
    lora_scales = torch.tensor(meta.scales, dtype=torch.float16, device=DEVICE)

    out_dtype = torch.float32 if op == "sgemmv_shrink" else dtype
    y = torch.zeros(workload.tokens, slices * max_rank, dtype=out_dtype, device=DEVICE)

    if op == "sgemmv_shrink":

        def run():
            torch.ops.npu.sgemmv_shrink(
                x, weight_a, lora_indices, seq_len, lora_ranks, lora_scales, y, slices
            )

    else:

        def run():
            torch.ops.npu.sgemmc_shrink(
                x, weight_a, lora_indices, seq_len, lora_ranks, lora_scales, y, slices
            )

    def reference():
        return torch_grouped_shrink(x, weight_a, meta, slices, max_rank, out_dtype)

    def reset():
        y.zero_()

    rank_sum = sum(meta.ranks[a] for a in meta.token_adapters)
    flops = 2.0 * rank_sum * workload.hidden * slices
    return Case(op, workload, meta, run, reference, reset, out_dtype, flops, output=y)


def build_fixed_shrink_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    """sgmv_shrink: grouped GEMV with a single rank shared by all adapters."""
    max_rank = meta.max_rank
    generator = torch.Generator().manual_seed(seed)

    x = (torch.randn(workload.tokens, workload.hidden, generator=generator)).to(
        device=DEVICE, dtype=dtype
    )
    weight_a = build_shrink_weight_a(
        meta, 1, max_rank, workload.hidden, dtype, generator
    )

    lora_indices = torch.tensor(meta.seq_adapters, dtype=torch.int64, device=DEVICE)
    seq_len = torch.tensor(meta.seq_lens, dtype=torch.int64, device=DEVICE)

    y = torch.zeros(workload.tokens, max_rank, dtype=torch.float32, device=DEVICE)
    scale = float(meta.scales[0])

    def run():
        torch.ops.npu.sgmv_shrink(x, weight_a, lora_indices, seq_len, y, scale)

    def reference():
        return torch_grouped_shrink(
            x, weight_a, meta, 1, max_rank, torch.float32, scale=scale
        )

    def reset():
        y.zero_()

    rank_sum = sum(meta.ranks[a] for a in meta.token_adapters)
    flops = 2.0 * rank_sum * workload.hidden
    return Case(
        op, workload, meta, run, reference, reset, torch.float32, flops, output=y
    )


def build_pertoken_shrink_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    """bgmv_shrink: one adapter index per token."""
    max_rank = meta.max_rank
    generator = torch.Generator().manual_seed(seed)

    x = (torch.randn(workload.tokens, workload.hidden, generator=generator)).to(
        device=DEVICE, dtype=dtype
    )
    weight_a = build_shrink_weight_a(
        meta, 1, max_rank, workload.hidden, dtype, generator
    )

    indices = torch.tensor(meta.token_adapters, dtype=torch.int64, device=DEVICE)
    y = torch.zeros(workload.tokens, max_rank, dtype=torch.float32, device=DEVICE)
    scale = float(meta.scales[0])

    def run():
        torch.ops.npu.bgmv_shrink(x, weight_a, indices, y, scale)

    def reference():
        return torch_grouped_shrink(
            x, weight_a, meta, 1, max_rank, torch.float32, scale=scale
        )

    def reset():
        y.zero_()

    rank_sum = sum(meta.ranks[a] for a in meta.token_adapters)
    flops = 2.0 * rank_sum * workload.hidden
    return Case(
        op, workload, meta, run, reference, reset, torch.float32, flops, output=y
    )


def build_grouped_expand_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    """sgemmv_expand / sgemmc_expand: grouped GEMV with per-adapter rank."""
    slices = workload.slices
    max_rank = meta.max_rank
    total_out = workload.total_out
    generator = torch.Generator().manual_seed(seed)

    # The expand kernels read the intermediate buffer as float32 (sgemmv) or
    # in the activation dtype (sgemmc), matching what the shrink kernel writes.
    x_dtype = torch.float32 if op == "sgemmv_expand" else dtype
    x = (torch.randn(workload.tokens, slices * max_rank, generator=generator)).to(
        device=DEVICE, dtype=x_dtype
    )
    weight_b = build_expand_weight_b(meta, total_out, max_rank, dtype, generator)

    lora_indices = torch.tensor(meta.seq_adapters, dtype=torch.int32, device=DEVICE)
    seq_len = torch.tensor(meta.seq_lens, dtype=torch.int32, device=DEVICE)
    lora_ranks = torch.tensor(meta.ranks, dtype=torch.int32, device=DEVICE)

    slice_offsets = [0]
    for width in workload.slice_widths:
        slice_offsets.append(slice_offsets[-1] + width)
    slice_offsets_tensor = torch.tensor(slice_offsets, dtype=torch.int32, device=DEVICE)

    y = torch.zeros(workload.tokens, total_out, dtype=dtype, device=DEVICE)

    if op == "sgemmv_expand":

        def run():
            torch.ops.npu.sgemmv_expand(
                x, weight_b, lora_indices, seq_len, lora_ranks, slice_offsets_tensor, y
            )

    else:

        def run():
            torch.ops.npu.sgemmc_expand(
                x, weight_b, lora_indices, seq_len, lora_ranks, slice_offsets_tensor, y
            )

    def reference():
        return torch_grouped_expand(
            x, weight_b, meta, slice_offsets, packed=True, out_dtype=dtype
        )

    def reset():
        y.zero_()

    rank_sum = sum(meta.ranks[a] for a in meta.token_adapters)
    flops = 2.0 * rank_sum * total_out
    return Case(op, workload, meta, run, reference, reset, dtype, flops, output=y)


def build_fixed_expand_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    """sgmv_expand: grouped GEMV with a single rank, one call per output slice."""
    max_rank = meta.max_rank
    total_out = workload.total_out
    generator = torch.Generator().manual_seed(seed)

    x = (torch.randn(workload.tokens, max_rank, generator=generator)).to(
        device=DEVICE, dtype=torch.float32
    )
    # The fixed-rank expand operators address the weight of an adapter from
    # its first row, so every output slice owns its own weight tensor.
    weight_slices = [
        build_expand_weight_b(meta, width, max_rank, dtype, generator)
        for width in workload.slice_widths
    ]

    lora_indices = torch.tensor(meta.seq_adapters, dtype=torch.int64, device=DEVICE)
    seq_len = torch.tensor(meta.seq_lens, dtype=torch.int64, device=DEVICE)

    y = torch.zeros(workload.tokens, total_out, dtype=dtype, device=DEVICE)

    def run():
        offset = 0
        for width, weight in zip(workload.slice_widths, weight_slices):
            torch.ops.npu.sgmv_expand(
                x, weight, lora_indices, seq_len, y, offset, width
            )
            offset += width

    def reference():
        return torch_fixed_expand(x, weight_slices, meta, dtype)

    def reset():
        y.zero_()

    rank_sum = sum(meta.ranks[a] for a in meta.token_adapters)
    flops = 2.0 * rank_sum * total_out
    return Case(op, workload, meta, run, reference, reset, dtype, flops, output=y)


def build_pertoken_expand_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    """bgmv_expand: one adapter index per token, one call per output slice."""
    max_rank = meta.max_rank
    total_out = workload.total_out
    generator = torch.Generator().manual_seed(seed)

    x = (torch.randn(workload.tokens, max_rank, generator=generator)).to(
        device=DEVICE, dtype=torch.float32
    )
    weight_slices = [
        build_expand_weight_b(meta, width, max_rank, dtype, generator)
        for width in workload.slice_widths
    ]

    indices = torch.tensor(meta.token_adapters, dtype=torch.int64, device=DEVICE)
    y = torch.zeros(workload.tokens, total_out, dtype=dtype, device=DEVICE)

    def run():
        offset = 0
        for width, weight in zip(workload.slice_widths, weight_slices):
            torch.ops.npu.bgmv_expand(x, weight, indices, y, offset, width)
            offset += width

    def reference():
        return torch_fixed_expand(x, weight_slices, meta, dtype)

    def reset():
        y.zero_()

    rank_sum = sum(meta.ranks[a] for a in meta.token_adapters)
    flops = 2.0 * rank_sum * total_out
    return Case(op, workload, meta, run, reference, reset, dtype, flops, output=y)


def build_case(
    op: str, workload: Workload, meta: LoraMeta, dtype: torch.dtype, seed: int
) -> Case:
    if op not in RANKED_OPS and len(set(meta.ranks)) != 1:
        raise ValueError(
            f"{op} uses a single rank and scale for all adapters, "
            f"got ranks {sorted(set(meta.ranks))}"
        )
    if op in ("sgemmv_shrink", "sgemmc_shrink"):
        return build_grouped_shrink_case(op, workload, meta, dtype, seed)
    if op == "sgmv_shrink":
        return build_fixed_shrink_case(op, workload, meta, dtype, seed)
    if op == "bgmv_shrink":
        return build_pertoken_shrink_case(op, workload, meta, dtype, seed)
    if op in ("sgemmv_expand", "sgemmc_expand"):
        return build_grouped_expand_case(op, workload, meta, dtype, seed)
    if op == "sgmv_expand":
        return build_fixed_expand_case(op, workload, meta, dtype, seed)
    if op == "bgmv_expand":
        return build_pertoken_expand_case(op, workload, meta, dtype, seed)
    raise ValueError(f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Validation and timing
# ---------------------------------------------------------------------------


def check_case(case: Case, strict: bool = True) -> bool:
    """Validate the kernel output against the PyTorch reference.

    With ``strict`` the mismatch raises, otherwise it is only reported and
    ``False`` is returned.
    """
    if case.reset is not None:
        case.reset()
    case.run()
    synchronize()
    actual = case.output
    assert actual is not None
    expected = case.reference()
    atol = SHRINK_ATOL if "shrink" in case.op else EXPAND_ATOL
    try:
        torch.testing.assert_close(
            actual.float().cpu(),
            expected.float().cpu(),
            atol=atol,
            rtol=RTOL,
        )
    except AssertionError as error:
        if strict:
            raise AssertionError(
                f"{case.op} / {case.workload.name}: {error}"
            ) from error
        max_diff = (actual.float().cpu() - expected.float().cpu()).abs().max().item()
        log(
            f"WARNING: {case.op} / {case.workload.name}: output mismatch "
            f"(max abs diff {max_diff:.3e}, tolerance atol={atol} rtol={RTOL})"
        )
        return False
    return True


def time_samples_ms(
    fn: Callable[[], None], warmup: int, iters: int, repeat: int
) -> List[float]:
    """Average latency per iteration in milliseconds, one sample per repeat."""
    for _ in range(warmup):
        fn()
    synchronize()

    samples = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return samples


def summarize(samples: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark_case(case: Case) -> Dict[str, object]:
    """Validate, time the kernel and time the PyTorch baseline."""
    workload = case.workload

    if workload.check:
        check_case(case)

    if case.reset is not None:
        case.reset()
    kernel_samples = time_samples_ms(
        case.run, workload.warmup, workload.iters, workload.repeat
    )

    baseline_samples = []
    if workload.baseline:
        if case.reset is not None:
            case.reset()
        baseline_samples = time_samples_ms(
            case.reference, workload.warmup, workload.iters, workload.repeat
        )

    kernel = summarize(kernel_samples)
    result: Dict[str, object] = {
        "op": case.op,
        "scenario": workload.name,
        "tokens": workload.tokens,
        "seq_len": workload.seq_len,
        "num_loras": workload.num_loras,
        "rank": case.meta.max_rank,
        "hidden": workload.hidden,
        "slices": workload.slices,
        "kernel_ms": kernel,
        "tokens_per_s": workload.tokens / (kernel["median"] / 1000.0),
        "gflops": case.flops / (kernel["median"] / 1000.0) / 1e9,
    }
    if baseline_samples:
        baseline = summarize(baseline_samples)
        result["baseline_ms"] = baseline
        result["speedup"] = baseline["median"] / kernel["median"]
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_result(result: Dict[str, object]) -> None:
    kernel = result["kernel_ms"]  # type: ignore[index]
    log("")
    log(
        f"[{result['op']}] scenario={result['scenario']} tokens={result['tokens']} "
        f"seq_len={result['seq_len']} rank={result['rank']} hidden={result['hidden']} "
        f"slices={result['slices']} num_loras={result['num_loras']}"
    )
    log(
        f"  kernel   : mean={kernel['mean']:.4f} ms  median={kernel['median']:.4f} ms  "
        f"min={kernel['min']:.4f} ms  max={kernel['max']:.4f} ms  std={kernel['std']:.4f} ms"
    )
    if "baseline_ms" in result:
        baseline = result["baseline_ms"]  # type: ignore[index]
        log(
            f"  torch    : mean={baseline['mean']:.4f} ms  median={baseline['median']:.4f} ms  "
            f"min={baseline['min']:.4f} ms  max={baseline['max']:.4f} ms  std={baseline['std']:.4f} ms"
        )
        log(f"  speedup  : {result['speedup']:.2f}x (torch / kernel)")
    log(
        f"  throughput: {result['tokens_per_s']:.1f} tokens/s, "
        f"{result['gflops']:.2f} GFLOP/s"
    )


def print_summary(results: List[Dict[str, object]]) -> None:
    log("")
    log("=" * 108)
    log(
        f"{'op':<16}{'scenario':<24}{'tokens':>8}{'kernel_ms':>12}{'torch_ms':>12}"
        f"{'speedup':>10}{'GFLOP/s':>12}{'tokens/s':>12}"
    )
    log("-" * 108)
    for result in results:
        kernel = result["kernel_ms"]["median"]  # type: ignore[index]
        baseline = (
            result["baseline_ms"]["median"] if "baseline_ms" in result else float("nan")
        )
        speedup = result.get("speedup", float("nan"))
        log(
            f"{result['op']:<16}{result['scenario']:<24}{result['tokens']:>8}"
            f"{kernel:>12.4f}{baseline:>12.4f}{speedup:>10.2f}"
            f"{result['gflops']:>12.2f}{result['tokens_per_s']:>12.1f}"
        )
    log("=" * 108)


def print_metrics(result: Dict[str, object], bounds: perf_config.BoundsConfig) -> None:
    """Print the SGLang style metric report for one measured case."""
    kernel = result["kernel_ms"]  # type: ignore[index]
    metrics = [
        perf_kit.at_most(
            "kernel_latency_ms", kernel["median"], bounds.max_latency_ms, unit="ms"
        ),
        perf_kit.reported("kernel_mean_ms", kernel["mean"], unit="ms"),
        perf_kit.reported("kernel_std_ms", kernel["std"], unit="ms"),
    ]
    if "baseline_ms" in result:
        baseline = result["baseline_ms"]  # type: ignore[index]
        metrics.append(
            perf_kit.reported("torch_latency_ms", baseline["median"], unit="ms")
        )
        metrics.append(perf_kit.reported("speedup", result["speedup"]))
    metrics.append(perf_kit.reported("tokens_per_s", result["tokens_per_s"]))
    metrics.append(perf_kit.reported("gflops", result["gflops"]))

    report = f"### {result['op']}/{result['scenario']}\n" + "".join(
        metric.line() + "\n" for metric in metrics
    )
    log("")
    log(report, end="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Ascend LoRA kernels against a PyTorch baseline."
    )
    parser.add_argument(
        "--suite",
        default="lora_perf",
        help="name of the YAML suite under benchmark/lora/configs",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="list the available suites and exit",
    )
    parser.add_argument(
        "--ops", nargs="+", choices=ALL_OPS, default=None, help="override the suite ops"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="run only the named scenarios of the suite",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP.keys()),
        default=None,
        help="override the dtype",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=None,
        help="override the adapter ranks, assigned round-robin over the adapters",
    )
    parser.add_argument(
        "--num-loras", type=int, default=None, help="override the number of adapters"
    )
    parser.add_argument(
        "--hidden", type=int, default=None, help="override the input hidden size"
    )
    parser.add_argument(
        "--slice-widths",
        nargs="+",
        type=int,
        default=None,
        help="override the output widths of the expanded projections",
    )
    parser.add_argument(
        "--index-mode",
        choices=("random", "single"),
        default=None,
        help="'single' sends every token to adapter 0",
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
        default=None,
        help="skip the PyTorch baseline measurement",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        default=None,
        help="skip the correctness validation that runs before the timing",
    )
    parser.add_argument("--json", type=str, default=None, help="write results as JSON")
    parser.add_argument("--device-id", type=int, default=None)
    return parser.parse_args(argv)


def apply_overrides(
    suite: perf_config.SuiteConfig, args: argparse.Namespace
) -> perf_config.SuiteConfig:
    """Apply the CLI overrides on top of the suite."""
    from dataclasses import replace as dataclass_replace

    if args.ops:
        suite = dataclass_replace(suite, ops=list(args.ops))

    scenarios = suite.scenarios
    if args.scenarios:
        known = {scenario.name for scenario in scenarios}
        unknown = [name for name in args.scenarios if name not in known]
        if unknown:
            raise perf_config.ConfigError(
                f"unknown scenario(s) {unknown}; the suite defines {sorted(known)}"
            )
        wanted = set(args.scenarios)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]

    updated = []
    for scenario in scenarios:
        changes = {}
        if args.dtype:
            changes["dtype"] = args.dtype
        if args.ranks:
            changes["ranks"] = list(args.ranks)
        if args.num_loras is not None:
            changes["num_loras"] = args.num_loras
        if args.hidden is not None:
            changes["hidden"] = args.hidden
        if args.slice_widths:
            changes["slice_widths"] = list(args.slice_widths)
        if args.index_mode:
            changes["index_mode"] = args.index_mode
        if args.seed is not None:
            changes["seed"] = args.seed
        if args.baseline is not None:
            changes["baseline"] = args.baseline
        if args.check is not None:
            changes["check"] = args.check

        timing_changes = {}
        if args.warmup is not None:
            timing_changes["warmup"] = args.warmup
        if args.iters is not None:
            timing_changes["iters"] = args.iters
        if args.repeat is not None:
            timing_changes["repeat"] = args.repeat
        if timing_changes:
            changes["timing"] = dataclass_replace(scenario.timing, **timing_changes)

        if changes:
            scenario = dataclass_replace(scenario, **changes)
        updated.append(scenario)

    suite = dataclass_replace(suite, scenarios=updated)
    suite.validate()
    return suite


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters is not None and args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.repeat is not None and args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.num_loras is not None and args.num_loras <= 0:
        raise ValueError("--num-loras must be positive")
    if args.hidden is not None and args.hidden <= 0:
        raise ValueError("--hidden must be positive")
    if args.ranks:
        for rank in args.ranks:
            if rank not in SUPPORTED_RANKS:
                raise ValueError(
                    f"rank {rank} is not supported, expected one of {SUPPORTED_RANKS}"
                )
    if args.slice_widths and any(width <= 0 for width in args.slice_widths):
        raise ValueError("--slice-widths must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    if args.list_suites:
        for path in perf_config.available_suites():
            log(path.stem)
        return 0

    suite = perf_config.load_suite_by_name(args.suite)
    suite = apply_overrides(suite, args)

    if args.device_id is not None:
        torch.npu.set_device(args.device_id)

    workloads = build_workloads(suite)
    results: List[Dict[str, object]] = []

    log(
        f"LoRA kernel benchmark: suite={suite.name} ops={suite.ops} "
        f"scenarios={[workload.name for workload in workloads]}"
    )
    if suite.description:
        log(f"  {suite.description}")
    log(
        f"  dtype={workloads[0].dtype if workloads else '-'} "
        f"hidden={workloads[0].hidden if workloads else '-'} "
        f"num_loras={workloads[0].num_loras if workloads else '-'}"
    )

    for op in suite.ops:
        if not hasattr(torch.ops.npu, op):
            log(f"skip {op}: operator is not registered in this build")
            continue
        for workload in workloads:
            if not workload.applies_to(op):
                continue
            uniform = op not in RANKED_OPS
            meta = build_meta(workload, uniform)
            case = build_case(op, workload, meta, workload.torch_dtype, workload.seed)
            result = benchmark_case(case)
            results.append(result)
            print_result(result)
            print_metrics(result, suite.bounds)

    print_summary(results)

    if args.json:
        payload = {
            "config": {
                "suite": suite.name,
                "description": suite.description,
                "source": str(suite.source) if suite.source else None,
                "ops": suite.ops,
                "scenarios": [
                    perf_config.scenario_overrides(scenario)
                    for scenario in suite.scenarios
                ],
                "bounds": {
                    "max_latency_ms": suite.bounds.max_latency_ms,
                    "max_slowdown": suite.bounds.max_slowdown,
                    "strict_check": suite.bounds.strict_check,
                },
            },
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        log(f"\nresults written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
