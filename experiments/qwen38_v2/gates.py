from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract import ContractError


def _task_key(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["repeat"]), str(record["task_id"])


def evaluate_run_gate(
    tasks: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    treatment: str,
    require_mechanism: bool,
    require_d3: bool,
    require_quality: bool,
    quality_margin: float = 0.01,
    max_obsolete_fork_tail_ms: float = 1.0,
) -> dict[str, Any]:
    baseline_tasks = [record for record in tasks if record["config"] == baseline]
    treatment_tasks = [record for record in tasks if record["config"] == treatment]
    if not baseline_tasks:
        raise ContractError(f"no task records found for baseline {baseline!r}")
    if not treatment_tasks:
        raise ContractError(f"no task records found for treatment {treatment!r}")
    treatment_turns = [record for record in turns if record["config"] == treatment]
    if not treatment_turns:
        raise ContractError(f"no turn records found for treatment {treatment!r}")

    baseline_keys = {_task_key(record) for record in baseline_tasks}
    treatment_keys = {_task_key(record) for record in treatment_tasks}
    baseline_status = {_task_key(record): record["status"] for record in baseline_tasks}
    treatment_only_failures = [
        {
            "repeat": key[0],
            "task_id": key[1],
            "baseline_status": baseline_status.get(key),
            "treatment_status": record["status"],
        }
        for record in treatment_tasks
        for key in [_task_key(record)]
        if record["status"] != "completed"
        and baseline_status.get(key) != record["status"]
    ]
    baseline_quality = [
        float(record["quality_reward"])
        for record in baseline_tasks
        if record["quality_reward"] is not None
    ]
    treatment_quality = [
        float(record["quality_reward"])
        for record in treatment_tasks
        if record["quality_reward"] is not None
    ]
    quality_difference = None
    if baseline_quality and len(baseline_quality) == len(baseline_tasks):
        if treatment_quality and len(treatment_quality) == len(treatment_tasks):
            quality_difference = sum(treatment_quality) / len(treatment_quality) - sum(
                baseline_quality
            ) / len(baseline_quality)

    dispatches = sum(bool(record["speculative_tool_dispatched"]) for record in treatment_turns)
    strict_matches = sum(bool(record["strict_match"]) for record in treatment_turns)
    reused_results = sum(bool(record["speculative_result_reused"]) for record in treatment_turns)
    total_overlap_ms = sum(float(record["real_overlap_ms"]) for record in treatment_turns)
    duplicate_executions = sum(
        bool(record["duplicate_tool_execution"]) for record in treatment_turns
    )
    authority_violations = sum(bool(record["authority_violation"]) for record in treatment_turns)
    write_speculations = sum(bool(record["write_tool_speculated"]) for record in treatment_turns)
    excess_clears = sum(int(record["draft_clear_count"]) > 1 for record in treatment_turns)
    obsolete_tail_violations = sum(
        float(record["obsolete_fork_tail_ms"]) > max_obsolete_fork_tail_ms
        for record in treatment_turns
    )
    d3_proposed = sum(int(record["d3_proposed_tokens"]) for record in treatment_turns)
    d3_accepted = sum(int(record["d3_accepted_tokens"]) for record in treatment_turns)
    ngram_proposed = sum(int(record["ngram_proposed_tokens"]) for record in treatment_turns)
    ngram_accepted = sum(int(record["ngram_accepted_tokens"]) for record in treatment_turns)

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, observed: Any, requirement: str) -> None:
        checks[name] = {
            "passed": bool(passed),
            "observed": observed,
            "requirement": requirement,
        }

    add(
        "task_sets_equal",
        baseline_keys == treatment_keys,
        {
            "baseline": len(baseline_keys),
            "treatment": len(treatment_keys),
            "baseline_only": sorted(baseline_keys - treatment_keys),
            "treatment_only": sorted(treatment_keys - baseline_keys),
        },
        "baseline and treatment must contain identical (repeat, task_id) keys",
    )
    add(
        "no_treatment_only_failures",
        not treatment_only_failures,
        {
            "baseline": {
                status: sum(record["status"] == status for record in baseline_tasks)
                for status in ("completed", "max_turns", "timeout", "error")
            },
            "treatment": {
                status: sum(record["status"] == status for record in treatment_tasks)
                for status in ("completed", "max_turns", "timeout", "error")
            },
            "treatment_only_failures": treatment_only_failures,
        },
        "shared failures are retained, but treatment may not add a failure",
    )
    add(
        "task_counts_nonzero",
        bool(baseline_keys and treatment_keys),
        {
            "baseline": len(baseline_keys),
            "treatment": len(treatment_keys),
        },
        "both sides of the comparison must contain tasks",
    )
    add(
        "treatment_turns_nonzero",
        bool(treatment_turns),
        len(treatment_turns),
        "the treatment must emit turn-level evidence",
    )
    add(
        "quality_counts_match",
        not require_quality
        or (
            len(baseline_quality) == len(baseline_tasks)
            and len(treatment_quality) == len(treatment_tasks)
        ),
        {
            "baseline_quality": len(baseline_quality),
            "baseline_tasks": len(baseline_tasks),
            "treatment_quality": len(treatment_quality),
            "treatment_tasks": len(treatment_tasks),
        },
        "every paired task must have a grader result",
    )
    add(
        "quality_present",
        not require_quality or quality_difference is not None,
        quality_difference,
        "every task must contain a real quality_reward",
    )
    add(
        "quality_noninferior",
        not require_quality
        or (quality_difference is not None and quality_difference >= -quality_margin),
        quality_difference,
        f"treatment - baseline quality must be >= {-quality_margin}",
    )
    add(
        "speculative_tool_dispatched",
        not require_mechanism or dispatches > 0,
        dispatches,
        "at least one read-only speculative tool must be dispatched",
    )
    add(
        "strict_match_observed",
        not require_mechanism or strict_matches > 0,
        strict_matches,
        "at least one full name+arguments strict match is required",
    )
    add(
        "speculative_result_reused",
        not require_mechanism or reused_results > 0,
        reused_results,
        "at least one speculative tool result must be reused",
    )
    add(
        "positive_real_overlap",
        not require_mechanism or total_overlap_ms > 0,
        total_overlap_ms,
        "accepted execution must overlap main decode in wall-clock time",
    )
    add(
        "no_duplicate_tool_execution",
        duplicate_executions == 0,
        duplicate_executions,
        "an accepted speculative result must not be executed again",
    )
    add(
        "main_authority_preserved",
        authority_violations == 0,
        authority_violations,
        "fork output must never drive authoritative agent state",
    )
    add(
        "no_write_tool_speculation",
        write_speculations == 0,
        write_speculations,
        "write and non-idempotent tools always use serial fallback",
    )
    add(
        "draft_cleared_at_most_once",
        excess_clears == 0,
        excess_clears,
        "each request may clear its draft at most once",
    )
    add(
        "obsolete_fork_not_drained",
        obsolete_tail_violations == 0,
        {
            "violations": obsolete_tail_violations,
            "max_allowed_ms": max_obsolete_fork_tail_ms,
        },
        "the critical path must not wait for an obsolete fork",
    )
    add(
        "spork_d3_proposed",
        not require_d3 or d3_proposed > 0,
        d3_proposed,
        "SPORK-boundary proposed tokens must be source-attributed and positive",
    )
    add(
        "spork_d3_accepted",
        not require_d3 or d3_accepted > 0,
        d3_accepted,
        "SPORK-boundary accepted tokens must be source-attributed and positive",
    )
    add(
        "d3_counts_valid",
        d3_accepted <= d3_proposed and ngram_accepted <= ngram_proposed,
        {
            "d3_proposed": d3_proposed,
            "d3_accepted": d3_accepted,
            "ngram_proposed": ngram_proposed,
            "ngram_accepted": ngram_accepted,
        },
        "accepted token counts cannot exceed their same-source proposals",
    )

    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema_version": "spork-run-gate-v2",
        "baseline": baseline,
        "treatment": treatment,
        "requirements": {
            "mechanism": require_mechanism,
            "d3": require_d3,
            "quality": require_quality,
        },
        "metrics": {
            "dispatches": dispatches,
            "strict_matches": strict_matches,
            "reused_results": reused_results,
            "real_overlap_ms": total_overlap_ms,
            "d3_proposed_tokens": d3_proposed,
            "d3_accepted_tokens": d3_accepted,
            "ngram_proposed_tokens": ngram_proposed,
            "ngram_accepted_tokens": ngram_accepted,
            "quality_difference": quality_difference,
        },
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }
