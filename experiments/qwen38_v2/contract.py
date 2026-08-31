from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "spork-e2e-v2"
GATE_SCHEMA_VERSION = "spork-gate-v2"

TASK_FIELDS = (
    "schema_version",
    "record_type",
    "run_id",
    "stage",
    "config",
    "repeat",
    "task_id",
    "domain",
    "status",
    "error",
    "task_wall_s",
    "resource_drained_s",
    "quality_reward",
    "action_match",
    "n_turns",
    "n_tool_turns",
)

TURN_FIELDS = (
    "schema_version",
    "record_type",
    "run_id",
    "stage",
    "config",
    "repeat",
    "task_id",
    "domain",
    "turn_index",
    "main_start_ms",
    "main_first_token_ms",
    "main_tool_call_decoded_ms",
    "main_completed_ms",
    "fork_start_ms",
    "fork_tool_call_decoded_ms",
    "fork_completed_ms",
    "fork_cancel_requested_ms",
    "main_tool_call",
    "fork_tool_call",
    "probe_confidence",
    "probe_attempt",
    "d2_threshold",
    "tool_read_only",
    "speculative_tool_dispatched",
    "speculative_tool_start_ms",
    "speculative_tool_end_ms",
    "canonical_tool_start_ms",
    "canonical_tool_end_ms",
    "strict_match",
    "speculative_result_reused",
    "speculative_execution_wasted",
    "duplicate_tool_execution",
    "authority_violation",
    "write_tool_speculated",
    "real_overlap_ms",
    "residual_tool_wait_ms",
    "obsolete_fork_tail_ms",
    "draft_submit_start_ms",
    "draft_submit_end_ms",
    "draft_boundary_hit",
    "d3_proposed_tokens",
    "d3_accepted_tokens",
    "ngram_proposed_tokens",
    "ngram_accepted_tokens",
    "draft_clear_start_ms",
    "draft_clear_end_ms",
    "draft_clear_count",
)

WARMUP_FIELDS = (
    "schema_version",
    "record_type",
    "run_id",
    "stage",
    "config",
    "repeat",
    "server_mode",
    "request_index",
    "status",
    "error",
    "wall_s",
)

EVENT_FIELDS = (
    "schema_version",
    "record_type",
    "run_id",
    "stage",
    "config",
    "repeat",
    "task_id",
    "domain",
    "turn_index",
    "event_index",
    "source",
    "event",
    "monotonic_ms",
    "payload",
)

GATE0_CHECKS = (
    "forced_prefix_bytes_verified",
    "forced_prefix_only_in_fork",
    "stop_sequence_complete",
    "main_first_output_single_token",
    "json_main_parse_pass",
    "json_fork_parse_pass",
    "manifest_no_leakage",
    "r1_raw_recomputed",
)

EVIDENCE_CHECKS = {
    "cache": (
        "shared_prefix_verified",
        "cached_prefill_le_half_cold",
        "main_tpot_regression_le_1pct",
    ),
    "d2": (
        "confidence_recorded",
        "threshold_locked_before_formal",
        "retry_behavior_recorded",
        "identifiability_classified",
    ),
    "d3": (
        "stock_composite_no_draft_equivalent",
        "source_specific_counters",
        "spork_boundary_accepted_positive",
        "fixed_request_greedy_equivalent",
        "active_requests_zero",
    ),
}

_TASK_STATUSES = {"completed", "max_turns", "timeout", "error"}


class ContractError(ValueError):
    """Raised when a V2 experiment artifact violates the locked contract."""


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_fields(record: Mapping[str, Any], fields: Iterable[str], source: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ContractError(f"{source}: missing required fields: {', '.join(missing)}")


def _require_string(record: Mapping[str, Any], field: str, source: str) -> None:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{source}: {field} must be a non-empty string")


def _require_bool(record: Mapping[str, Any], field: str, source: str) -> None:
    if not isinstance(record[field], bool):
        raise ContractError(f"{source}: {field} must be boolean")


def _optional_number(
    record: Mapping[str, Any], field: str, source: str, *, minimum: float = 0.0
) -> None:
    value = record[field]
    if value is None:
        return
    if not _is_number(value) or float(value) < minimum:
        raise ContractError(f"{source}: {field} must be null or >= {minimum}")


def _optional_bool(record: Mapping[str, Any], field: str, source: str) -> None:
    value = record[field]
    if value is not None and not isinstance(value, bool):
        raise ContractError(f"{source}: {field} must be null or boolean")


def _validate_run_identity(record: Mapping[str, Any], source: str) -> None:
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError(
            f"{source}: schema_version must be {SCHEMA_VERSION!r}, "
            f"got {record['schema_version']!r}"
        )
    for field in ("run_id", "stage", "config"):
        _require_string(record, field, source)
    if not isinstance(record["repeat"], int) or isinstance(record["repeat"], bool):
        raise ContractError(f"{source}: repeat must be an integer")
    if record["repeat"] <= 0:
        raise ContractError(f"{source}: repeat must be positive")


def _validate_identity(record: Mapping[str, Any], source: str) -> None:
    _validate_run_identity(record, source)
    for field in ("task_id", "domain"):
        _require_string(record, field, source)


def validate_task(record: Mapping[str, Any], source: str = "task") -> None:
    _require_fields(record, TASK_FIELDS, source)
    _validate_identity(record, source)
    if record["record_type"] != "task":
        raise ContractError(f"{source}: record_type must be 'task'")
    if record["status"] not in _TASK_STATUSES:
        raise ContractError(f"{source}: unsupported status {record['status']!r}")
    if record["error"] is not None and not isinstance(record["error"], str):
        raise ContractError(f"{source}: error must be null or string")
    for field in ("task_wall_s", "resource_drained_s"):
        _optional_number(record, field, source)
    if record["status"] == "completed" and record["task_wall_s"] is None:
        raise ContractError(f"{source}: completed task requires task_wall_s")
    if record["status"] == "completed" and float(record["task_wall_s"]) <= 0:
        raise ContractError(f"{source}: completed task requires positive task_wall_s")
    if (
        record["task_wall_s"] is not None
        and record["resource_drained_s"] is not None
        and float(record["resource_drained_s"]) + 1e-9 < float(record["task_wall_s"])
    ):
        raise ContractError(f"{source}: resource_drained_s cannot be below task_wall_s")
    _optional_number(record, "quality_reward", source)
    if record["quality_reward"] is not None and float(record["quality_reward"]) > 1.0:
        raise ContractError(f"{source}: quality_reward must be in [0, 1]")
    _optional_bool(record, "action_match", source)
    for field in ("n_turns", "n_tool_turns"):
        if (
            not isinstance(record[field], int)
            or isinstance(record[field], bool)
            or record[field] < 0
        ):
            raise ContractError(f"{source}: {field} must be a non-negative integer")
    if record["n_tool_turns"] > record["n_turns"]:
        raise ContractError(f"{source}: n_tool_turns cannot exceed n_turns")


def _validate_time_order(
    record: Mapping[str, Any], start: str, end: str, source: str
) -> None:
    left = record[start]
    right = record[end]
    if left is not None and right is not None and float(right) + 1e-9 < float(left):
        raise ContractError(f"{source}: {end} cannot precede {start}")


def validate_turn(record: Mapping[str, Any], source: str = "turn") -> None:
    _require_fields(record, TURN_FIELDS, source)
    _validate_identity(record, source)
    if record["record_type"] != "turn":
        raise ContractError(f"{source}: record_type must be 'turn'")
    if (
        not isinstance(record["turn_index"], int)
        or isinstance(record["turn_index"], bool)
        or record["turn_index"] < 0
    ):
        raise ContractError(f"{source}: turn_index must be a non-negative integer")

    time_fields = tuple(field for field in TURN_FIELDS if field.endswith("_ms"))
    for field in time_fields:
        _optional_number(record, field, source)
    if record["main_start_ms"] is None or record["main_completed_ms"] is None:
        raise ContractError(f"{source}: main_start_ms and main_completed_ms are required")
    for field in ("real_overlap_ms", "residual_tool_wait_ms", "obsolete_fork_tail_ms"):
        if record[field] is None:
            raise ContractError(f"{source}: {field} is required")

    for start, end in (
        ("main_start_ms", "main_first_token_ms"),
        ("main_start_ms", "main_tool_call_decoded_ms"),
        ("main_start_ms", "main_completed_ms"),
        ("fork_start_ms", "fork_tool_call_decoded_ms"),
        ("fork_start_ms", "fork_completed_ms"),
        ("speculative_tool_start_ms", "speculative_tool_end_ms"),
        ("canonical_tool_start_ms", "canonical_tool_end_ms"),
        ("draft_submit_start_ms", "draft_submit_end_ms"),
        ("draft_clear_start_ms", "draft_clear_end_ms"),
    ):
        _validate_time_order(record, start, end, source)
    if (
        record["fork_start_ms"] is not None
        and record["main_first_token_ms"] is not None
        and float(record["fork_start_ms"]) + 1e-9 < float(record["main_first_token_ms"])
    ):
        raise ContractError(f"{source}: fork started before the main first-token trigger")

    for field in (
        "tool_read_only",
        "speculative_tool_dispatched",
        "strict_match",
        "speculative_result_reused",
        "speculative_execution_wasted",
        "duplicate_tool_execution",
        "authority_violation",
        "write_tool_speculated",
        "draft_boundary_hit",
    ):
        _require_bool(record, field, source)

    for field in ("main_tool_call", "fork_tool_call"):
        value = record[field]
        if value is not None and not isinstance(value, Mapping):
            raise ContractError(f"{source}: {field} must be null or an object")
    _optional_number(record, "probe_confidence", source)
    _optional_number(record, "d2_threshold", source)
    if record["probe_attempt"] is not None and (
        not isinstance(record["probe_attempt"], int)
        or isinstance(record["probe_attempt"], bool)
        or record["probe_attempt"] < 0
    ):
        raise ContractError(f"{source}: probe_attempt must be null or non-negative integer")

    for field in (
        "d3_proposed_tokens",
        "d3_accepted_tokens",
        "ngram_proposed_tokens",
        "ngram_accepted_tokens",
        "draft_clear_count",
    ):
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"{source}: {field} must be a non-negative integer")
    if record["d3_accepted_tokens"] > record["d3_proposed_tokens"]:
        raise ContractError(f"{source}: D3 accepted tokens exceed proposed tokens")
    if record["ngram_accepted_tokens"] > record["ngram_proposed_tokens"]:
        raise ContractError(f"{source}: ngram accepted tokens exceed proposed tokens")
    if record["draft_clear_count"] > 1:
        raise ContractError(f"{source}: draft_clear_count exceeds one")

    if record["speculative_result_reused"]:
        if not record["speculative_tool_dispatched"] or not record["strict_match"]:
            raise ContractError(f"{source}: reused result requires dispatch and strict match")
        if not record["tool_read_only"]:
            raise ContractError(f"{source}: reused result came from a non-read-only tool")
        if record["real_overlap_ms"] <= 0:
            raise ContractError(f"{source}: reused result requires positive real overlap")
        if record["duplicate_tool_execution"]:
            raise ContractError(f"{source}: reused result was executed a second time")
        if record["canonical_tool_start_ms"] is not None:
            raise ContractError(
                f"{source}: reused result cannot have a second canonical execution"
            )
    if record["write_tool_speculated"]:
        raise ContractError(f"{source}: write tool was speculatively executed")

    if record["speculative_tool_dispatched"]:
        if not record["tool_read_only"]:
            raise ContractError(f"{source}: speculative execution requires a read-only tool")
        for field in (
            "fork_tool_call",
            "fork_tool_call_decoded_ms",
            "speculative_tool_start_ms",
            "speculative_tool_end_ms",
        ):
            if record[field] is None:
                raise ContractError(
                    f"{source}: speculative dispatch requires {field}"
                )
    elif any(
        record[field] is not None
        for field in ("speculative_tool_start_ms", "speculative_tool_end_ms")
    ):
        raise ContractError(
            f"{source}: speculative tool timestamps exist without a dispatch"
        )

    if record["strict_match"]:
        if record["main_tool_call"] is None or record["fork_tool_call"] is None:
            raise ContractError(f"{source}: strict match requires both tool calls")
        if record["main_tool_call"] != record["fork_tool_call"]:
            raise ContractError(
                f"{source}: strict match is true but canonical tool calls differ"
            )
    if record["speculative_execution_wasted"] != (
        record["speculative_tool_dispatched"]
        and not record["speculative_result_reused"]
    ):
        raise ContractError(
            f"{source}: speculative_execution_wasted must equal dispatch AND NOT reuse"
        )

    expected_overlap = 0.0
    spec_start = record["speculative_tool_start_ms"]
    spec_end = record["speculative_tool_end_ms"]
    main_call = record["main_tool_call_decoded_ms"]
    if spec_start is not None and spec_end is not None and main_call is not None:
        expected_overlap = max(0.0, min(float(spec_end), float(main_call)) - float(spec_start))
    if abs(float(record["real_overlap_ms"]) - expected_overlap) > 1.0:
        raise ContractError(
            f"{source}: real_overlap_ms={record['real_overlap_ms']} does not match "
            f"the timestamp intersection {expected_overlap:.3f}"
        )
    expected_residual = 0.0
    if record["speculative_result_reused"]:
        expected_residual = max(0.0, float(spec_end) - float(main_call))
    if abs(float(record["residual_tool_wait_ms"]) - expected_residual) > 1.0:
        raise ContractError(
            f"{source}: residual_tool_wait_ms={record['residual_tool_wait_ms']} does "
            f"not match the timestamp remainder {expected_residual:.3f}"
        )


def validate_warmup(record: Mapping[str, Any], source: str = "warmup") -> None:
    _require_fields(record, WARMUP_FIELDS, source)
    _validate_run_identity(record, source)
    if record["record_type"] != "warmup":
        raise ContractError(f"{source}: record_type must be 'warmup'")
    if record["server_mode"] not in {"plain", "composite"}:
        raise ContractError(f"{source}: server_mode must be plain or composite")
    if (
        not isinstance(record["request_index"], int)
        or isinstance(record["request_index"], bool)
        or record["request_index"] <= 0
    ):
        raise ContractError(f"{source}: request_index must be a positive integer")
    if record["status"] not in {"completed", "error"}:
        raise ContractError(f"{source}: warmup status must be completed or error")
    if record["error"] is not None and not isinstance(record["error"], str):
        raise ContractError(f"{source}: error must be null or string")
    _optional_number(record, "wall_s", source)
    if record["status"] == "completed" and record["wall_s"] is None:
        raise ContractError(f"{source}: completed warmup requires wall_s")
    if record["status"] == "completed" and float(record["wall_s"]) <= 0:
        raise ContractError(f"{source}: completed warmup requires positive wall_s")


def validate_event(record: Mapping[str, Any], source: str = "event") -> None:
    _require_fields(record, EVENT_FIELDS, source)
    _validate_identity(record, source)
    if record["record_type"] != "event":
        raise ContractError(f"{source}: record_type must be 'event'")
    for field in ("turn_index", "event_index"):
        if (
            not isinstance(record[field], int)
            or isinstance(record[field], bool)
            or record[field] < 0
        ):
            raise ContractError(f"{source}: {field} must be a non-negative integer")
    if record["source"] not in {"main", "fork", "tool", "draft", "driver"}:
        raise ContractError(f"{source}: unsupported event source {record['source']!r}")
    _require_string(record, "event", source)
    _optional_number(record, "monotonic_ms", source)
    if record["monotonic_ms"] is None:
        raise ContractError(f"{source}: monotonic_ms is required")
    if not isinstance(record["payload"], Mapping):
        raise ContractError(f"{source}: payload must be an object")


def load_records(
    path: str | Path,
    record_type: str,
    *,
    validator: Callable[[Mapping[str, Any], str], None] | None = None,
) -> list[dict[str, Any]]:
    artifact = Path(path)
    if not artifact.is_file():
        raise ContractError(f"missing artifact: {artifact}")
    validators = {
        "task": validate_task,
        "turn": validate_turn,
        "warmup": validate_warmup,
        "event": validate_event,
    }
    if record_type not in validators:
        raise ContractError(f"unsupported record type: {record_type}")
    validate = validator or validators[record_type]
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for line_number, raw_line in enumerate(
        artifact.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        source = f"{artifact}:{line_number}"
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ContractError(f"{source}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ContractError(f"{source}: record must be a JSON object")
        validate(value, source)
        key = (value["run_id"], value["config"], value["repeat"])
        if record_type == "warmup":
            key += (value["request_index"],)
        else:
            key += (value["task_id"],)
            if record_type == "turn":
                key += (value["turn_index"],)
            elif record_type == "event":
                key += (value["turn_index"], value["event_index"])
        if key in seen:
            raise ContractError(f"{source}: duplicate record key {key!r}")
        seen.add(key)
        records.append(value)
    if not records:
        raise ContractError(f"artifact has no {record_type} records: {artifact}")
    return records


def validate_artifact_relationships(
    tasks: Iterable[Mapping[str, Any]],
    turns: Iterable[Mapping[str, Any]],
    warmups: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    *,
    expected_warmups: int,
    expected_tasks: int | None = None,
) -> dict[str, int]:
    if expected_warmups <= 0:
        raise ContractError("expected_warmups must be positive")
    task_rows = list(tasks)
    turn_rows = list(turns)
    warmup_rows = list(warmups)
    event_rows = list(events)
    task_keys = {
        (
            row["run_id"],
            row["stage"],
            row["config"],
            row["repeat"],
            row["task_id"],
            row["domain"],
        )
        for row in task_rows
    }
    task_blocks = {key[:4] for key in task_keys}
    if expected_tasks is not None:
        if expected_tasks <= 0:
            raise ContractError("expected_tasks must be positive")
        tasks_per_block: dict[tuple[Any, ...], set[str]] = {}
        for row in task_rows:
            block = (row["run_id"], row["stage"], row["config"], row["repeat"])
            tasks_per_block.setdefault(block, set()).add(str(row["task_id"]))
        invalid_task_counts = {
            block: len(task_ids)
            for block, task_ids in tasks_per_block.items()
            if len(task_ids) != expected_tasks
        }
        if invalid_task_counts:
            raise ContractError(
                f"config blocks do not contain exactly {expected_tasks} tasks: "
                f"{invalid_task_counts!r}"
            )
    for label, rows in (("turn", turn_rows), ("event", event_rows)):
        orphaned = [
            row
            for row in rows
            if (
                row["run_id"],
                row["stage"],
                row["config"],
                row["repeat"],
                row["task_id"],
                row["domain"],
            )
            not in task_keys
        ]
        if orphaned:
            raise ContractError(f"{label} artifact contains {len(orphaned)} orphan rows")
    turn_indices: dict[tuple[Any, ...], set[int]] = {}
    for row in turn_rows:
        key = (
            row["run_id"],
            row["stage"],
            row["config"],
            row["repeat"],
            row["task_id"],
            row["domain"],
        )
        turn_indices.setdefault(key, set()).add(int(row["turn_index"]))
    mismatches = [
        row
        for row in task_rows
        if turn_indices.get(
            (
                row["run_id"],
                row["stage"],
                row["config"],
                row["repeat"],
                row["task_id"],
                row["domain"],
            ),
            set(),
        )
        != set(range(int(row["n_turns"])))
    ]
    if mismatches:
        raise ContractError(
            f"{len(mismatches)} task rows disagree with their turn record counts"
        )
    event_indices: dict[tuple[Any, ...], list[tuple[int, float]]] = {}
    for row in event_rows:
        turn_key = (
            row["run_id"],
            row["stage"],
            row["config"],
            row["repeat"],
            row["task_id"],
            row["domain"],
            row["turn_index"],
        )
        event_indices.setdefault(turn_key, []).append(
            (int(row["event_index"]), float(row["monotonic_ms"]))
        )
    expected_turn_keys = {
        (
            row["run_id"],
            row["stage"],
            row["config"],
            row["repeat"],
            row["task_id"],
            row["domain"],
            row["turn_index"],
        )
        for row in turn_rows
    }
    if set(event_indices) != expected_turn_keys:
        raise ContractError("every turn must have raw events and no event-only turns")
    for turn_key, indexed_times in event_indices.items():
        ordered = sorted(indexed_times)
        if [index for index, _ in ordered] != list(range(len(ordered))):
            raise ContractError(f"event indices are not contiguous for {turn_key!r}")
        times = [event_time for _, event_time in ordered]
        if times != sorted(times):
            raise ContractError(f"event timestamps regress for {turn_key!r}")

    warmup_indices: dict[tuple[Any, ...], set[int]] = {}
    for row in warmup_rows:
        block = (row["run_id"], row["stage"], row["config"], row["repeat"])
        if block not in task_blocks:
            raise ContractError(f"warmup artifact contains an orphan config block {block!r}")
        if row["status"] != "completed":
            raise ContractError(f"warmup request failed in config block {block!r}")
        warmup_indices.setdefault(block, set()).add(int(row["request_index"]))
    invalid_warmups = {
        block: sorted(warmup_indices.get(block, set()))
        for block in task_blocks
        if warmup_indices.get(block, set()) != set(range(1, expected_warmups + 1))
    }
    if invalid_warmups:
        raise ContractError(
            f"config blocks do not contain exactly {expected_warmups} warmups: "
            f"{invalid_warmups!r}"
        )
    return {
        "task_records": len(task_rows),
        "turn_records": len(turn_rows),
        "warmup_records": len(warmup_rows),
        "event_records": len(event_rows),
        "config_blocks": len(task_blocks),
    }


def validate_gate0(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise ContractError(f"missing Gate 0 artifact: {artifact}")
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"{artifact}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{artifact}: Gate 0 artifact must be an object")
    if value.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ContractError(
            f"{artifact}: schema_version must be {GATE_SCHEMA_VERSION!r}"
        )
    checks = value.get("checks")
    if not isinstance(checks, dict):
        raise ContractError(f"{artifact}: checks must be an object")
    missing = [name for name in GATE0_CHECKS if name not in checks]
    if missing:
        raise ContractError(f"{artifact}: missing Gate 0 checks: {', '.join(missing)}")
    failed = [name for name in GATE0_CHECKS if checks.get(name) is not True]
    if failed:
        raise ContractError(f"{artifact}: failed Gate 0 checks: {', '.join(failed)}")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ContractError(f"{artifact}: artifacts provenance must be a non-empty object")
    for name, provenance in artifacts.items():
        if not isinstance(provenance, dict):
            raise ContractError(
                f"{artifact}: artifact {name!r} must contain path and sha256"
            )
        raw_path = provenance.get("path")
        expected_sha = provenance.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ContractError(
                f"{artifact}: artifact {name!r} path must be absolute"
            )
        raw_artifact = Path(raw_path)
        if not raw_artifact.is_file():
            raise ContractError(
                f"{artifact}: artifact {name!r} does not exist: {raw_artifact}"
            )
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise ContractError(
                f"{artifact}: artifact {name!r} has an invalid sha256"
            )
        digest = hashlib.sha256(raw_artifact.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise ContractError(
                f"{artifact}: artifact {name!r} sha256 mismatch"
            )
    return value


def validate_gate_evidence(path: str | Path, gate: str) -> dict[str, Any]:
    if gate not in EVIDENCE_CHECKS:
        raise ContractError(f"unsupported evidence gate: {gate}")
    artifact = Path(path)
    if not artifact.is_file():
        raise ContractError(f"missing {gate} evidence artifact: {artifact}")
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"{artifact}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{artifact}: gate evidence must be an object")
    if value.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ContractError(
            f"{artifact}: schema_version must be {GATE_SCHEMA_VERSION!r}"
        )
    if value.get("gate") != gate:
        raise ContractError(f"{artifact}: gate must be {gate!r}")
    checks = value.get("checks")
    if not isinstance(checks, dict):
        raise ContractError(f"{artifact}: checks must be an object")
    required = EVIDENCE_CHECKS[gate]
    missing = [name for name in required if name not in checks]
    if missing:
        raise ContractError(f"{artifact}: missing {gate} checks: {', '.join(missing)}")
    failed = []
    for name in required:
        check = checks[name]
        if not isinstance(check, dict) or check.get("passed") is not True:
            failed.append(name)
    if failed:
        raise ContractError(f"{artifact}: failed {gate} checks: {', '.join(failed)}")
    if value.get("passed") is not True:
        raise ContractError(f"{artifact}: top-level passed must be true")
    return value
