# LoRA kernel benchmarks

Performance benchmark for the eight LoRA operators of `sgl-kernel-npu`:

| shrink (hidden → rank) | expand (rank → hidden) |
| ---------------------- | ---------------------- |
| `sgmv_shrink`          | `sgmv_expand`          |
| `sgemmv_shrink`        | `sgemmv_expand`        |
| `sgemmc_shrink`        | `sgemmc_expand`        |
| `bgmv_shrink`          | `bgmv_expand`          |

* `sgmv` / `sgemmv` group the tokens by request, `bgmv` uses one adapter
  index per token.
* `sgemmv` / `sgemmc` support a per-adapter rank (8/16/32/64) and, for the
  shrink direction, a per-adapter scale; the other two share one rank.
* `sgemmv` / `sgemmc` can fuse several output projections into a single call
  (`slice_widths: [1024, 1024, 1024]` models a QKV projection).

Every operator is validated against a pure-PyTorch grouped-matmul reference
before it is timed, so a broken kernel cannot silently look fast, and the
reported speedup is measured against a baseline that performs exactly the
same arithmetic.

## Configuration

The sweep is described by a YAML suite under [`configs/`](configs), not by
environment variables. A suite is a reviewable artifact: the shapes, the
timing loop and the regression bounds live next to each other and are
diffed like code.

```yaml
name: lora-perf-ci
description: Compact LoRA kernel suite for CI (decode + prefill, float16)

defaults:            # applies to every scenario
  dtype: float16
  hidden: 4096
  num_loras: 8
  ranks: [16]
  slice_widths: [4096]
  index_mode: random
  baseline: true
  check: true
  timing:
    warmup: 5
    iters: 20
    repeat: 3

scenarios:           # at least one
  - name: decode-b64
    tokens: 64
    seq_len: 1
  - name: mixed-rank-prefill-t2048
    tokens: 2048
    seq_len: 128
    ranks: [8, 16, 32, 64]
    group: mixed-ranks
    ops: [sgemmv_shrink, sgemmv_expand, sgemmc_shrink, sgemmc_expand]

bounds:              # enforced only in CI
  max_latency_ms: 100.0
  max_slowdown: 50.0
  strict_check: false
```

Two shipped suites:

| suite          | file                                | purpose                              |
| -------------- | ----------------------------------- | ------------------------------------ |
| `lora_perf`    | `configs/lora_perf.yaml`            | the full sweep (decode + prefill)    |
| `lora_perf_ci` | `configs/lora_perf_ci.yaml`         | the compact suite the CI test runs   |

Scenario fields:

| field           | default     | meaning                                          |
| --------------- | ----------- | ------------------------------------------------ |
| `name`          | required    | the scenario name, unique inside the suite       |
| `tokens`        | required    | number of tokens of the scenario                 |
| `seq_len`       | required    | tokens per sequence                              |
| `dtype`         | `float16`   | `float16` or `bfloat16`                         |
| `hidden`        | `4096`      | input hidden size                               |
| `num_loras`     | `8`         | number of adapters                               |
| `ranks`         | `[16]`      | adapter ranks, assigned round-robin              |
| `slice_widths`  | `[4096]`    | output widths of the expanded projections        |
| `index_mode`    | `random`    | `single` sends every token to adapter 0          |
| `baseline`      | `true`      | measure the PyTorch baseline as well             |
| `check`         | `true`      | validate the output before timing                |
| `seed`          | `20260620`  | seed of the weight and index generation          |
| `group`         | `default`   | the test group the scenario belongs to           |
| `ops`           | all         | the operators the scenario applies to            |
| `timing`        | see above   | `warmup`, `iters` and `repeat` of the loop       |

The `group` and `ops` fields let a scenario target a subset of the
operators. The fixed-rank operators (`sgmv`/`bgmv`) share one rank across
adapters, so a mixed-rank scenario must not run for them; the suite
declares that with `ops:`.

## Running

```bash
# Full sweep: decode + prefill, float16, all operators
python benchmark/lora/bench_lora_kernels.py

# The compact suite
python benchmark/lora/bench_lora_kernels.py --suite lora_perf_ci

# List the available suites
python benchmark/lora/bench_lora_kernels.py --list-suites

# A single operator and dtype
python benchmark/lora/bench_lora_kernels.py --ops sgemmv_shrink sgemmv_expand \
    --dtype bfloat16

# A single scenario
python benchmark/lora/bench_lora_kernels.py --scenarios decode-b64

# Custom shapes (CLI flags override the suite)
python benchmark/lora/bench_lora_kernels.py --hidden 4096 --num-loras 16 \
    --ranks 8 16 32 64 --slice-widths 4096

# QKV-style fused slices
python benchmark/lora/bench_lora_kernels.py --slice-widths 1024 1024 1024

# Machine readable results
python benchmark/lora/bench_lora_kernels.py --json lora_perf.json
```

The report contains the kernel latency (mean/median/min/max/std), the
PyTorch baseline latency, the speedup, the achieved throughput in tokens/s and
the achieved GFLOP/s.

## Regression test

`tests/python/sgl_kernel_npu/test_lora_kernels_perf.py` runs the
`lora_perf_ci` suite in CI. Each test method runs the scenarios of the suite
that belong to its group, and the bounds of the suite are the regression
contract:

| bound            | default | meaning                                    |
| ---------------- | ------- | ------------------------------------------ |
| `max_latency_ms` | `100`   | hard upper bound per iteration (ms)        |
| `max_slowdown`   | `50`    | kernel latency / torch baseline            |
| `strict_check`   | `false` | fail the test on a numerical mismatch      |

The numerical validation of the test is a warning by default, so a
pre-existing kernel bug does not block unrelated pull requests.

The bounds are only *enforced* in CI: a local run prints them next to every
measurement but does not assert them, so a slow workstation does not fail the
suite. CI is detected through the `SGL_KERNEL_NPU_IS_IN_CI` environment
variable, which the CI workflow sets — it is a marker for the CI run, not a
tuning knob.

`tests/python/sgl_kernel_npu/test_lora_perf_config.py` covers the config
loader itself: a malformed suite must fail loudly instead of silently
skipping a bound.
