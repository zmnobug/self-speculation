#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${V2_ENV_FILE:-$SCRIPT_DIR/server.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

V2_PYTHON="${V2_PYTHON:-python3}"
V2_DRIVER_MODULE="${V2_DRIVER_MODULE:-experiments.qwen38}"
V2_RUNS_ROOT="${V2_RUNS_ROOT:-$PROJECT_ROOT/runs/v2}"
V2_STATE_DIR="${V2_STATE_DIR:-$V2_RUNS_ROOT/.state}"
V2_RESULTS_ROOT="${V2_RESULTS_ROOT:-$PROJECT_ROOT/results/v2}"
V2_WARMUP_REQUESTS="${V2_WARMUP_REQUESTS:-10}"
V2_TASK_TIMEOUT_S="${V2_TASK_TIMEOUT_S:-900}"
V2_BOOTSTRAP_ITERS="${V2_BOOTSTRAP_ITERS:-10000}"
V2_SERVER_START_TIMEOUT_S="${V2_SERVER_START_TIMEOUT_S:-300}"
V2_UNITTEST_TIMEOUT_S="${V2_UNITTEST_TIMEOUT_S:-600}"
V2_PLAIN_TREATMENT="${V2_PLAIN_TREATMENT:-d1-d2}"
V2_QUICK_REPEATS="${V2_QUICK_REPEATS:-1}"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SERVER_PID=""
GPU_MONITOR_PID=""
CRITICAL_PATHS=()
V2_QUICK_MODE=0

die() {
  printf 'V2 RUNNER ERROR: %s\n' "$*" >&2
  exit 2
}

require_var() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || die "required variable $name is not set; configure $ENV_FILE"
  [[ "$value" != *'/ABS/'* ]] || die "required variable $name still contains /ABS/ placeholder"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "required file does not exist: $path"
}

require_path() {
  local path="$1"
  [[ -e "$path" ]] || die "required path does not exist: $path"
}

require_executable() {
  local path="$1"
  [[ -x "$path" ]] || die "required executable does not exist or is not executable: $path"
}

health_url() {
  printf '%s/models' "${V2_BASE_URL%/}"
}

server_is_ready() {
  curl --noproxy '*' --fail --silent --show-error "$(health_url)" >/dev/null 2>&1
}

stop_server() {
  stop_gpu_monitor
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$SERVER_PID" 2>/dev/null && (( waited < 30 )); do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

stop_gpu_monitor() {
  if [[ -z "$GPU_MONITOR_PID" ]]; then
    return
  fi
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill -TERM "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=""
}

validate_gpu_csv() {
  local output="$1"
  [[ -s "$output" ]] || die "GPU telemetry is empty: $output"
  [[ "$(wc -l <"$output")" -ge 2 ]] || \
    die "GPU telemetry contains no samples: $output"
}

start_gpu_monitor() {
  local output="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu \
    --format=csv -l 1 >"$output" 2>&1 &
  GPU_MONITOR_PID=$!
}

metrics_url() {
  local root="${V2_BASE_URL%/}"
  root="${root%/v1}"
  printf '%s/metrics' "$root"
}

capture_server_metrics() {
  local output="$1"
  curl --noproxy '*' --fail --silent --show-error "$(metrics_url)" >"$output" || \
    die "server metrics endpoint is unavailable: $(metrics_url)"
}

trap stop_server EXIT INT TERM

start_server() {
  local mode="$1"
  local log_path="$2"
  local launcher
  case "$mode" in
    plain) launcher="$V2_PLAIN_SERVER_LAUNCHER" ;;
    composite) launcher="$V2_COMPOSITE_SERVER_LAUNCHER" ;;
    *) die "unsupported server mode: $mode" ;;
  esac
  require_executable "$launcher"
  if server_is_ready; then
    die "$(health_url) is already serving; stop the unrelated server before this block"
  fi
  mkdir -p "$(dirname -- "$log_path")"
  "$launcher" >"$log_path" 2>&1 &
  SERVER_PID=$!
  local waited=0
  while ! server_is_ready; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      wait "$SERVER_PID" || true
      SERVER_PID=""
      die "$mode server exited before readiness; inspect $log_path"
    fi
    if (( waited >= V2_SERVER_START_TIMEOUT_S )); then
      die "$mode server did not become ready within ${V2_SERVER_START_TIMEOUT_S}s"
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

timestamp() {
  date -u '+%Y%m%dT%H%M%SZ'
}

marker_path() {
  printf '%s/%s.pass.json' "$V2_STATE_DIR" "$1"
}

run_with_critical_paths() {
  local command=("$@")
  local path
  for path in "${CRITICAL_PATHS[@]}"; do
    command+=(--path "$path")
  done
  "${command[@]}"
}

require_marker() {
  local stage="$1"
  local marker
  marker="$(marker_path "$stage")"
  [[ -f "$marker" ]] || die "stage $stage has not passed"
  run_with_critical_paths \
    "$V2_PYTHON" -m experiments.qwen38_v2 verify-stage \
    --stage "$stage" --marker "$marker" >/dev/null
  if [[ "$stage" == "gate0" ]]; then
    "$V2_PYTHON" -m experiments.qwen38_v2 gate0 \
      --input "$V2_GATE0_EVIDENCE" >/dev/null
  fi
}

write_marker() {
  local stage="$1"
  local artifact="$2"
  mkdir -p "$V2_STATE_DIR"
  run_with_critical_paths \
    "$V2_PYTHON" -m experiments.qwen38_v2 mark-stage \
    --stage "$stage" --artifact "$artifact" --marker "$(marker_path "$stage")" \
    >/dev/null
}

capture_provenance() {
  local run_dir="$1"
  local provenance="$run_dir/provenance"
  mkdir -p "$provenance"
  run_with_critical_paths \
    "$V2_PYTHON" -m experiments.qwen38_v2 fingerprint \
    --output "$provenance/input-fingerprint.json" >/dev/null
  cp -- "$V2_COMPOSITE_SERVER_LAUNCHER" "$provenance/composite-server-launcher.sh"
  cp -- "$V2_PROTOCOL" "$provenance/locked-protocol.json"
  cp -- "$V2_MANIFEST_QUICK10" "$provenance/manifest-quick10.json"
  if (( V2_QUICK_MODE == 0 )); then
    cp -- "$V2_PLAIN_SERVER_LAUNCHER" "$provenance/plain-server-launcher.sh"
    cp -- "$V2_MANIFEST_SMOKE" "$provenance/manifest-smoke.json"
    cp -- "$V2_MANIFEST_AIRLINE" "$provenance/manifest-airline.json"
    cp -- "$V2_MANIFEST_FULL" "$provenance/manifest-full.json"
    cp -- "$V2_D2_MANIFEST" "$provenance/manifest-d2.json"
  fi
  "$V2_PYTHON" --version >"$provenance/python-version.txt" 2>&1
  git -C "$PROJECT_ROOT" rev-parse HEAD >"$provenance/git-head.txt" 2>&1 || true
  git -C "$PROJECT_ROOT" status --short >"$provenance/git-status.txt" 2>&1 || true
  nvidia-smi >"$provenance/nvidia-smi.txt" 2>&1 || true
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"$provenance/created-at-utc.txt"
}

initialize_run_dir() {
  local run_dir="$1"
  mkdir -p "$run_dir/raw" "$run_dir/logs" "$run_dir/analysis"
  capture_provenance "$run_dir"
}

new_run_dir() {
  local stage="$1"
  local suffix="${2:-}"
  local run_dir="$V2_RUNS_ROOT/$stage/$(timestamp)-$$${suffix}"
  initialize_run_dir "$run_dir"
  printf '%s' "$run_dir"
}

run_id_for() {
  local run_dir="$1"
  local relative="${run_dir#"$V2_RUNS_ROOT"/}"
  printf '%s' "${relative//\//-}"
}

check_common_environment() {
  local requested_stage="${1:-preflight}"
  if [[ "$requested_stage" == "quick" || "$requested_stage" == "quick10" ]]; then
    V2_QUICK_MODE=1
  fi
  local required_variables=(
    V2_MODEL_NAME V2_TOKENIZER_PATH V2_BASE_URL V2_PROTOCOL
    V2_DRIVER_SOURCE V2_GATE0_EVIDENCE V2_COMPOSITE_SERVER_LAUNCHER
    V2_MANIFEST_QUICK10 V2_RUNS_ROOT V2_STATE_DIR
  )
  if (( V2_QUICK_MODE == 0 )); then
    required_variables+=(
      V2_PLAIN_SERVER_LAUNCHER V2_MANIFEST_SMOKE V2_MANIFEST_AIRLINE
      V2_MANIFEST_FULL V2_D2_MANIFEST V2_RESULTS_ROOT
    )
  fi
  local name
  for name in "${required_variables[@]}"; do
    require_var "$name"
  done
  require_executable "$V2_PYTHON"
  require_executable "$V2_COMPOSITE_SERVER_LAUNCHER"
  if (( V2_QUICK_MODE == 0 )); then
    require_executable "$V2_PLAIN_SERVER_LAUNCHER"
  fi
  require_file "$ENV_FILE"
  require_path "$V2_DRIVER_SOURCE"
  require_file "$V2_TOKENIZER_PATH/tokenizer_config.json"
  require_file "$V2_PROTOCOL"
  require_file "$V2_GATE0_EVIDENCE"
  require_file "$V2_MANIFEST_QUICK10"
  if (( V2_QUICK_MODE == 0 )); then
    require_file "$V2_MANIFEST_SMOKE"
    require_file "$V2_MANIFEST_AIRLINE"
    require_file "$V2_MANIFEST_FULL"
    require_file "$V2_D2_MANIFEST"
  fi
  command -v curl >/dev/null || die "curl is required"
  command -v git >/dev/null || die "git is required"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
  command -v timeout >/dev/null || die "GNU timeout is required"
  "$V2_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || \
    die "V2_PYTHON must be Python 3.10 or newer"
  CRITICAL_PATHS=(
    "$SCRIPT_DIR/run_v2.sh"
    "$ENV_FILE"
    "$PROJECT_ROOT/src"
    "$PROJECT_ROOT/experiments/qwen38_v2"
    "$PROJECT_ROOT/tests"
    "$V2_DRIVER_SOURCE"
    "$V2_COMPOSITE_SERVER_LAUNCHER"
    "$V2_PROTOCOL"
    "$V2_GATE0_EVIDENCE"
    "$V2_MANIFEST_QUICK10"
  )
  if (( V2_QUICK_MODE == 0 )); then
    CRITICAL_PATHS+=(
      "$V2_PLAIN_SERVER_LAUNCHER"
      "$V2_MANIFEST_SMOKE"
      "$V2_MANIFEST_AIRLINE"
      "$V2_MANIFEST_FULL"
      "$V2_D2_MANIFEST"
    )
  fi
  "$V2_PYTHON" -m experiments.qwen38_v2 --help >/dev/null
  local driver_help
  driver_help="$("$V2_PYTHON" -m "$V2_DRIVER_MODULE" --help 2>&1)" || \
    die "cannot load server driver module $V2_DRIVER_MODULE"
  local required_driver_commands=(v2-e2e)
  if [[ "$requested_stage" != "quick" && "$requested_stage" != "quick10" ]]; then
    required_driver_commands+=(v2-cache-gate v2-d2-gate v2-d3-gate)
  fi
  for command in "${required_driver_commands[@]}"; do
    [[ "$driver_help" == *"$command"* ]] || \
      die "$V2_DRIVER_MODULE does not expose required subcommand $command"
  done
  local e2e_help
  e2e_help="$("$V2_PYTHON" -m "$V2_DRIVER_MODULE" v2-e2e --help 2>&1)" || \
    die "cannot inspect $V2_DRIVER_MODULE v2-e2e"
  local option
  for option in \
    --base-url --model-name --tokenizer-path --protocol --task-manifest \
    --config --server-mode --tool-floor-s --repeat --stage --run-id \
    --warmup-requests --task-timeout-s --task-jsonl --turn-jsonl \
    --event-jsonl --warmup-jsonl; do
    [[ "$e2e_help" == *"$option"* ]] || \
      die "$V2_DRIVER_MODULE v2-e2e does not expose required option $option"
  done
  mkdir -p "$V2_RUNS_ROOT" "$V2_STATE_DIR"
  if (( V2_QUICK_MODE == 0 )); then
    mkdir -p "$V2_RESULTS_ROOT"
  fi
}

run_e2e_block() {
  local mode="$1"
  local config="$2"
  local manifest="$3"
  local floor="$4"
  local repeat="$5"
  local stage="$6"
  local run_dir="$7"
  local block="${mode}-${config}-r${repeat}"
  start_server "$mode" "$run_dir/logs/server-$block.log"
  capture_server_metrics "$run_dir/metrics-${block}-before.prom"
  start_gpu_monitor "$run_dir/gpu-${block}.csv"
  local command=(
    "$V2_PYTHON" -m "$V2_DRIVER_MODULE" v2-e2e
    --base-url "$V2_BASE_URL"
    --model-name "$V2_MODEL_NAME"
    --tokenizer-path "$V2_TOKENIZER_PATH"
    --protocol "$V2_PROTOCOL"
    --task-manifest "$manifest"
    --config "$config"
    --server-mode "$mode"
    --tool-floor-s "$floor"
    --repeat "$repeat"
    --stage "$stage"
    --run-id "$(run_id_for "$run_dir")"
    --warmup-requests "$V2_WARMUP_REQUESTS"
    --task-timeout-s "$V2_TASK_TIMEOUT_S"
    --task-jsonl "$run_dir/raw/tasks.jsonl"
    --turn-jsonl "$run_dir/raw/turns.jsonl"
    --event-jsonl "$run_dir/raw/events.jsonl"
    --warmup-jsonl "$run_dir/raw/warmups.jsonl"
  )
  printf '%q ' "${command[@]}" >>"$run_dir/commands.txt"
  printf '\n' >>"$run_dir/commands.txt"
  "${command[@]}" >"$run_dir/logs/driver-$block.log" 2>&1
  stop_gpu_monitor
  validate_gpu_csv "$run_dir/gpu-${block}.csv"
  capture_server_metrics "$run_dir/metrics-${block}-after.prom"
  stop_server
}

contract_check() {
  local run_dir="$1"
  local expected_tasks="${2:-}"
  local command=(
    "$V2_PYTHON" -m experiments.qwen38_v2 contract
    --tasks "$run_dir/raw/tasks.jsonl" \
    --turns "$run_dir/raw/turns.jsonl" \
    --events "$run_dir/raw/events.jsonl" \
    --warmups "$run_dir/raw/warmups.jsonl" \
    --expected-warmups "$V2_WARMUP_REQUESTS"
    --output "$run_dir/analysis/contract.json"
  )
  if [[ -n "$expected_tasks" ]]; then
    command+=(--expected-tasks "$expected_tasks")
  fi
  "${command[@]}" >/dev/null
}

run_gate() {
  local run_dir="$1"
  local baseline="$2"
  local treatment="$3"
  local output_name="$4"
  shift 4
  "$V2_PYTHON" -m experiments.qwen38_v2 gate \
    --tasks "$run_dir/raw/tasks.jsonl" \
    --turns "$run_dir/raw/turns.jsonl" \
    --baseline "$baseline" \
    --treatment "$treatment" \
    --output "$run_dir/analysis/$output_name.json" \
    "$@" >/dev/null
}

run_analysis() {
  local run_dir="$1"
  local baseline="$2"
  local treatment="$3"
  local output_name="$4"
  "$V2_PYTHON" -m experiments.qwen38_v2 analyze \
    --tasks "$run_dir/raw/tasks.jsonl" \
    --baseline "$baseline" \
    --treatment "$treatment" \
    --bootstrap-iters "$V2_BOOTSTRAP_ITERS" \
    --require-quality-pass \
    --output "$run_dir/analysis/$output_name.json" >/dev/null
}

run_diagnostic_analysis() {
  local run_dir="$1"
  local baseline="$2"
  local treatment="$3"
  local output_name="$4"
  "$V2_PYTHON" -m experiments.qwen38_v2 analyze \
    --tasks "$run_dir/raw/tasks.jsonl" \
    --baseline "$baseline" \
    --treatment "$treatment" \
    --bootstrap-iters "$V2_BOOTSTRAP_ITERS" \
    --output "$run_dir/analysis/$output_name.json" >/dev/null
}

run_gate0_stage() {
  require_var V2_GATE0_EVIDENCE
  require_file "$V2_GATE0_EVIDENCE"
  local run_dir
  run_dir="$(new_run_dir gate0)"
  "$V2_PYTHON" -m experiments.qwen38_v2 gate0 \
    --input "$V2_GATE0_EVIDENCE" \
    --output "$run_dir/analysis/gate0.json" >/dev/null
  write_marker gate0 "$run_dir"
  printf 'Gate 0 PASS: %s\n' "$run_dir"
}

run_gate1_stage() {
  require_marker gate0
  local run_dir
  run_dir="$(new_run_dir gate1)"
  local correctness_test="$PROJECT_ROOT/tests/test_core_e2e_v2_correctness.py"
  require_file "$correctness_test"
  timeout --signal=TERM --kill-after=30s "${V2_UNITTEST_TIMEOUT_S}s" \
    "$V2_PYTHON" -m unittest discover -s tests -v \
    >"$run_dir/logs/unittest.log" 2>&1
  "$V2_PYTHON" -m compileall -q src examples experiments tests
  printf '{"schema_version":"spork-gate-v2","gate":"correctness","passed":true}\n' \
    >"$run_dir/analysis/gate1.json"
  write_marker gate1 "$run_dir"
  printf 'Gate 1 PASS: %s\n' "$run_dir"
}

run_smoke_stage() {
  require_marker gate1
  local run_dir
  run_dir="$(new_run_dir smoke)"
  for config in baseline-serial d1 d1-d2; do
    run_e2e_block plain "$config" "$V2_MANIFEST_SMOKE" 2 1 smoke "$run_dir"
  done
  contract_check "$run_dir"
  run_gate "$run_dir" baseline-serial d1 gate-d1 \
    --require-mechanism --require-quality
  run_gate "$run_dir" baseline-serial d1-d2 gate-d1-d2 \
    --require-mechanism --require-quality
  write_marker smoke "$run_dir"
  printf 'Timeline smoke PASS: %s\n' "$run_dir"
}

run_evidence_stage() {
  local gate="$1"
  local mode="$2"
  local prerequisite="$3"
  local subcommand="v2-${gate}-gate"
  require_marker "$prerequisite"
  local run_dir
  run_dir="$(new_run_dir "$gate")"
  local manifest="$V2_MANIFEST_SMOKE"
  if [[ "$gate" == "d2" ]]; then
    manifest="$V2_D2_MANIFEST"
  fi
  start_server "$mode" "$run_dir/logs/server.log"
  capture_server_metrics "$run_dir/metrics-before.prom"
  start_gpu_monitor "$run_dir/gpu.csv"
  "$V2_PYTHON" -m "$V2_DRIVER_MODULE" "$subcommand" \
    --base-url "$V2_BASE_URL" \
    --model-name "$V2_MODEL_NAME" \
    --tokenizer-path "$V2_TOKENIZER_PATH" \
    --protocol "$V2_PROTOCOL" \
    --task-manifest "$manifest" \
    --output "$run_dir/${gate}-evidence.json" \
    >"$run_dir/logs/driver.log" 2>&1
  stop_gpu_monitor
  validate_gpu_csv "$run_dir/gpu.csv"
  capture_server_metrics "$run_dir/metrics-after.prom"
  stop_server
  "$V2_PYTHON" -m experiments.qwen38_v2 evidence \
    --gate "$gate" \
    --input "$run_dir/${gate}-evidence.json" \
    --output "$run_dir/analysis/${gate}.json" >/dev/null
  write_marker "$gate" "$run_dir"
  printf '%s gate PASS: %s\n' "$gate" "$run_dir"
}

ordered_configs() {
  local repeat="$1"
  shift
  local configs=("$@")
  if (( repeat % 2 == 0 )); then
    local index
    for ((index=${#configs[@]}-1; index>=0; index--)); do
      printf '%s\n' "${configs[index]}"
    done
  else
    printf '%s\n' "${configs[@]}"
  fi
}

run_suite() {
  local run_dir="$1"
  local manifest="$2"
  local floor="$3"
  local repeats="$4"
  local stage="$5"
  local repeat config
  for ((repeat=1; repeat<=repeats; repeat++)); do
    while IFS= read -r config; do
      run_e2e_block plain "$config" "$manifest" "$floor" "$repeat" "$stage" "$run_dir"
    done < <(ordered_configs "$repeat" baseline-serial d1 d1-d2)
    while IFS= read -r config; do
      run_e2e_block composite "$config" "$manifest" "$floor" "$repeat" "$stage" "$run_dir"
    done < <(ordered_configs "$repeat" baseline-ngram d1-d2-d3)
  done
}

validate_suite() {
  local run_dir="$1"
  local formal="$2"
  contract_check "$run_dir"
  run_gate "$run_dir" baseline-serial "$V2_PLAIN_TREATMENT" plain-gate \
    --require-mechanism --require-quality
  run_gate "$run_dir" baseline-ngram d1-d2-d3 composite-gate \
    --require-mechanism --require-d3 --require-quality
  if [[ "$formal" == "yes" ]]; then
    run_analysis "$run_dir" baseline-serial "$V2_PLAIN_TREATMENT" plain-analysis
    run_analysis "$run_dir" baseline-ngram d1-d2-d3 composite-analysis
  fi
}

run_precheck_stage() {
  require_marker cache
  require_marker d2
  require_marker d3
  local run_dir
  run_dir="$(new_run_dir precheck)"
  run_suite "$run_dir" "$V2_MANIFEST_AIRLINE" 2 1 precheck
  validate_suite "$run_dir" no
  write_marker precheck "$run_dir"
  printf 'Airline precheck PASS: %s\n' "$run_dir"
}

run_unlock_stage() {
  local stage
  for stage in gate0 gate1 smoke cache d2 d3 precheck; do
    require_marker "$stage"
  done
  local run_dir
  run_dir="$(new_run_dir unlock)"
  printf '{"schema_version":"spork-formal-unlock-v2","passed":true}\n' \
    >"$run_dir/analysis/formal-unlock.json"
  write_marker unlock "$run_dir"
  printf 'Formal stages unlocked: %s\n' "$run_dir"
}

run_floor_stage() {
  require_marker precheck
  require_marker unlock
  local stage_root="$V2_RUNS_ROOT/floor/$(timestamp)-$$"
  mkdir -p "$stage_root"
  local floor run_dir
  for floor in 0.5 1 2 5; do
    run_dir="$stage_root/floor-$floor"
    initialize_run_dir "$run_dir"
    run_suite "$run_dir" "$V2_MANIFEST_AIRLINE" "$floor" 3 "floor-$floor"
    validate_suite "$run_dir" yes
  done
  write_marker floor "$stage_root"
  printf 'Floor sweep complete: %s\n' "$stage_root"
}

run_full_stage() {
  require_marker floor
  require_marker unlock
  local run_dir
  run_dir="$(new_run_dir full155)"
  run_suite "$run_dir" "$V2_MANIFEST_FULL" 2 3 full155
  validate_suite "$run_dir" yes
  write_marker full155 "$run_dir"
  printf 'Full-155 complete: %s\n' "$run_dir"
}

run_report_stage() {
  require_marker unlock
  require_marker floor
  require_marker full155
  "$V2_PYTHON" -m experiments.qwen38_v2 report \
    --state-dir "$V2_STATE_DIR" \
    --output-dir "$V2_RESULTS_ROOT" \
    --output "$V2_RESULTS_ROOT/report-command-result.json" >/dev/null
  write_marker report "$V2_RESULTS_ROOT"
  printf 'V2 report complete: %s\n' "$V2_RESULTS_ROOT"
}

run_quick10_stage() {
  require_marker gate1
  local run_dir
  run_dir="$(new_run_dir quick10)"
  local repeat config
  for ((repeat=1; repeat<=V2_QUICK_REPEATS; repeat++)); do
    while IFS= read -r config; do
      run_e2e_block composite "$config" "$V2_MANIFEST_QUICK10" 2 "$repeat" \
        quick10 "$run_dir"
    done < <(ordered_configs "$repeat" baseline-ngram d1-d2-d3)
  done
  contract_check "$run_dir" 10
  local gate_failed=0
  if ! run_gate "$run_dir" baseline-ngram d1-d2-d3 composite-gate \
    --require-mechanism --require-d3 --require-quality; then
    gate_failed=1
  fi
  run_diagnostic_analysis "$run_dir" baseline-ngram d1-d2-d3 composite-analysis
  "$V2_PYTHON" -m experiments.qwen38_v2 quick-report \
    --run-dir "$run_dir" \
    --output-dir "$run_dir/analysis" \
    --output "$run_dir/analysis/quick-report-command.json" >/dev/null
  if (( gate_failed != 0 )); then
    die "Quick10 produced a report but failed a mechanism/quality gate: $run_dir"
  fi
  write_marker quick10 "$run_dir"
  printf 'Quick10 timing diagnostic complete: %s\n' "$run_dir"
}

run_quick_stage() {
  run_gate0_stage
  run_gate1_stage
  run_quick10_stage
}

run_qualify_stage() {
  run_gate0_stage
  run_gate1_stage
  run_smoke_stage
  run_evidence_stage cache plain smoke
  run_evidence_stage d2 plain cache
  run_evidence_stage d3 composite d2
  run_precheck_stage
  printf 'Qualification sequence complete; inspect artifacts before unlock.\n'
}

run_formal_stage() {
  require_marker unlock
  run_floor_stage
  run_full_stage
  run_report_stage
  printf 'Formal tau2 sequence complete.\n'
}

usage() {
  cat <<'EOF'
Usage: scripts/qwen38_v2/run_v2.sh STAGE

Stages:
  preflight  Validate paths, launchers, manifests, and server driver CLI.
  quick      Run Gate 0, Gate 1, then the complete 10-case timing diagnostic.
  quick10    Run only the 10-case timing diagnostic (requires Gate 1).
  qualify    Run Gate 0 through airline precheck, stopping on first failure.
  gate0      Validate normalized Phase1/R1 evidence.
  gate1      Run correctness tests and compileall.
  smoke      Run plain baseline/D1/D1-D2 on smoke10 and enforce real reuse.
  cache      Run the D1 prefix-cache evidence gate.
  d2         Run the D2 confidence/retry evidence gate.
  d3         Run the matched-ngram D3 evidence gate.
  precheck   Run one airline-43 pass in plain and composite modes.
  unlock     Revalidate all gates and explicitly unlock formal inference.
  floor      Run formal 0.5/1/2/5s airline sweep (requires unlock).
  full       Run formal full-155 at 2s (requires floor completion and unlock).
  report     Generate locked JSON/Markdown reports from floor and full.
  formal     Run floor, full, and report; stop on first failure (requires unlock).
EOF
}

main() {
  local stage="${1:-}"
  [[ -n "$stage" ]] || { usage; exit 2; }
  if [[ "$stage" == "-h" || "$stage" == "--help" || "$stage" == "help" ]]; then
    usage
    return
  fi
  check_common_environment "$stage"
  case "$stage" in
    preflight) printf 'V2 preflight PASS\n' ;;
    quick) run_quick_stage ;;
    quick10) run_quick10_stage ;;
    qualify) run_qualify_stage ;;
    gate0) run_gate0_stage ;;
    gate1) run_gate1_stage ;;
    smoke) run_smoke_stage ;;
    cache) run_evidence_stage cache plain smoke ;;
    d2) run_evidence_stage d2 plain cache ;;
    d3) run_evidence_stage d3 composite d2 ;;
    precheck) run_precheck_stage ;;
    unlock) run_unlock_stage ;;
    floor) run_floor_stage ;;
    full) run_full_stage ;;
    report) run_report_stage ;;
    formal) run_formal_stage ;;
    *) usage; die "unknown stage: $stage" ;;
  esac
}

main "$@"
