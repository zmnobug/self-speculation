# Qwen3.8 SPORK V2 scripted runner

This directory turns `REPRODUCTION_QWEN38_27B_V2.md` into a fail-closed server
workflow. The runner never interprets prose results. It accepts only locked
JSONL records and machine-readable gate evidence.

## Current Quick10 timing run

The current experiment is intentionally limited to 10 frozen tau2-bench
`airline` tasks and one matched comparison:

```text
baseline-ngram vs d1-d2-d3
10 unique tasks x 2 configs x 1 repeat = 20 measured task runs
```

It requires only the composite launcher and the server adapter's `v2-e2e`
subcommand. Fill the common paths plus `V2_MANIFEST_QUICK10`,
`V2_COMPOSITE_SERVER_LAUNCHER`, `V2_RUNS_ROOT`, and `V2_STATE_DIR` in
`server.env`. The simplest setup uses the Quick10-only template:

```bash
cp scripts/qwen38_v2/server.quick10.env.example scripts/qwen38_v2/server.env
# Replace every /ABS/... path, then run:
scripts/qwen38_v2/run_v2.sh quick
```

The unused full-experiment paths are not required. The equivalent run command
is:

```bash
scripts/qwen38_v2/run_v2.sh quick
```

Build the manifest by sorting the existing airline-43 task IDs and sampling 10
without replacement using seed 42. Freeze those IDs before inference and never
replace a task based on observed latency, probe accuracy, or task outcome. Each
task should retain the normal airline agent/tool workflow; only read-only calls
may execute speculatively. Output is written to the new
`runs/v2/quick10/<timestamp>/analysis/QUICK10_{SUMMARY.json,REPORT.md}` files.
This run is diagnostic and cannot by itself establish paper-level significance.

## One-time server integration

The experiment server's existing `experiments.qwen38` package must expose four
subcommands. `run_v2.sh preflight` refuses to continue until all four appear in
`python -m experiments.qwen38 --help`.

### `v2-e2e`

Required arguments:

```text
--base-url --model-name --tokenizer-path --protocol --task-manifest
--config --server-mode --tool-floor-s --repeat --stage --run-id
--warmup-requests --task-timeout-s --task-jsonl --turn-jsonl
--event-jsonl --warmup-jsonl
```

It executes exactly one config block and appends one task record and all turn
records to the requested JSONL files. Records must satisfy
`TASK_FIELDS`, `TURN_FIELDS`, `EVENT_FIELDS`, and `WARMUP_FIELDS` in
`experiments.qwen38_v2.contract`. The driver must open files in append mode and
flush after every task. Each block must record exactly `--warmup-requests`
successful discarded warmups; those requests must not appear as measured tasks.

Supported config names are locked:

```text
baseline-serial  d1  d1-d2  baseline-ngram  d1-d2-d3
```

### `v2-cache-gate`

It runs the main-only/cold-concurrent/first-token-cached microbenchmark and
writes `spork-gate-v2` JSON containing these passing checks:

```text
shared_prefix_verified
cached_prefill_le_half_cold
main_tpot_regression_le_1pct
```

### `v2-d2-gate`

It writes confidence/retry evidence with these checks:

```text
confidence_recorded
threshold_locked_before_formal
retry_behavior_recorded
identifiability_classified
```

`identifiability_classified` may pass with a result such as
`not_identifiable_on_tau2`; it must not falsely claim a D2 benefit.

### `v2-d3-gate`

It runs stock-ngram/composite-no-draft equivalence, fixed-request greedy
equivalence, pure boundary acceptance, and leak checks. Required checks:

```text
stock_composite_no_draft_equivalent
source_specific_counters
spork_boundary_accepted_positive
fixed_request_greedy_equivalent
active_requests_zero
```

All gate evidence uses:

```json
{
  "schema_version": "spork-gate-v2",
  "gate": "cache",
  "passed": true,
  "checks": {
    "shared_prefix_verified": {
      "passed": true,
      "observed": {"cached_tokens": 12000},
      "requirement": "shared prefix is exact"
    }
  },
  "metrics": {}
}
```

The `gate` value is `cache`, `d2`, or `d3` as appropriate.

## Required correctness test

Gate 1 requires `tests/test_core_e2e_v2_correctness.py` in the server checkout.
That file must exercise the real server adapter, not mock only the JSON writer.
At minimum it must lock these cases:

1. Wrong fork tool name never commits state and its execution is discarded.
2. Same name with different canonical arguments is rejected; main executes once.
3. Exact read-only match reuses the speculative result with no second execution.
4. Fork call with no authoritative main call never commits agent state.
5. Write/non-idempotent tools always take the serial path.
6. Main completion cancels an obsolete fork and does not drain its tail.
7. D3 submit/clear is request-scoped and clear occurs at most once.
8. A missing real tau2 grader cannot be serialized as a successful quality row.

The runner executes the entire test suite with a 600-second default timeout.
Failure or timeout blocks every online stage.

## Gate 0 normalization

The existing Phase 1/R1 artifacts must be normalized once into
`spork-gate-v2` JSON:

```json
{
  "schema_version": "spork-gate-v2",
  "checks": {
    "forced_prefix_bytes_verified": true,
    "forced_prefix_only_in_fork": true,
    "stop_sequence_complete": true,
    "main_first_output_single_token": true,
    "json_main_parse_pass": true,
    "json_fork_parse_pass": true,
    "manifest_no_leakage": true,
    "r1_raw_recomputed": true
  },
  "artifacts": {
    "phase1_audit": {
      "path": "/absolute/path/audit-summary.json",
      "sha256": "replace-with-sha256"
    },
    "r1_raw": {
      "path": "/absolute/path/probes.jsonl",
      "sha256": "replace-with-sha256"
    },
    "r1_bootstrap": {
      "path": "/absolute/path/bootstrap.json",
      "sha256": "replace-with-sha256"
    }
  }
}
```

Each boolean must be derived from the referenced raw artifact. Do not type
`true` merely because an old Markdown report says PASS.

Each artifact entry is hash-locked and must use this form:

```json
"phase1_audit": {
  "path": "/absolute/path/audit-summary.json",
  "sha256": "64-lowercase-hex-characters"
}
```

The validator rereads the raw files and verifies each SHA256 whenever a later
stage depends on Gate 0.

## Server launchers

Create one foreground launcher for each serving mode. The launcher must end in
`exec`, for example:

```bash
#!/usr/bin/env bash
set -euo pipefail
exec vllm serve /data2/Qwen3.8-27B \
  --host 127.0.0.1 --port 8000 --served-model-name qwen3.8-27b \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching
```

The composite launcher must add only the locked composite ngram+boundary
proposer configuration. Record the exact command in the result directory.

## Full experiment execution (not used by the current Quick10 run)

The normal server-agent path is only three commands:

```bash
scripts/qwen38_v2/run_v2.sh qualify
# Human reviews runs/v2 gate/raw artifacts here.
scripts/qwen38_v2/run_v2.sh unlock
scripts/qwen38_v2/run_v2.sh formal
```

`qualify` and `formal` stop on the first nonzero gate. For diagnosis or a
single-stage retry, use the expanded sequence below. `formal` also runs the
machine report generator; it writes the primary estimator and verdict directly
from locked JSON rather than asking an agent to recalculate them.

```bash
cd /path/to/self-speculation
cp scripts/qwen38_v2/server.env.example scripts/qwen38_v2/server.env
# Edit every absolute path.

scripts/qwen38_v2/run_v2.sh preflight
scripts/qwen38_v2/run_v2.sh gate0
scripts/qwen38_v2/run_v2.sh gate1
scripts/qwen38_v2/run_v2.sh smoke
scripts/qwen38_v2/run_v2.sh cache
scripts/qwen38_v2/run_v2.sh d2
scripts/qwen38_v2/run_v2.sh d3
scripts/qwen38_v2/run_v2.sh precheck
```

Inspect every artifact under `runs/v2/` before formal runs. Then explicitly
create the hash-bound authorization marker and run:

```bash
scripts/qwen38_v2/run_v2.sh unlock
scripts/qwen38_v2/run_v2.sh floor
scripts/qwen38_v2/run_v2.sh full
scripts/qwen38_v2/run_v2.sh report
```

Each formal config block starts from a fresh server process and identical
warmup. Repeat 2 reverses config order. A stage creates its `.state` marker only
after contract and gate validation returns success. Markers bind the complete
stage directory to SHA256 fingerprints of `src/`, tests, the server adapter,
protocol, manifests, environment file, and both server launchers. Any change
invalidates all downstream markers and requires the affected stages to be run
again.

## What the runner deliberately does not guess

This checkout does not contain the server's tau2 harness. Therefore the runner
does not synthesize an imitation benchmark driver. The server adapter must
implement the four CLI entry points above against the real tau2 environment and
grader. Until it does, `preflight` or Gate 1 fails closed and no GPU-scale stage
can start.
