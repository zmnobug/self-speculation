from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.qwen38_v2.contract import (
    ContractError,
    load_records,
    validate_artifact_relationships,
    validate_gate0,
    validate_gate_evidence,
    validate_task,
    validate_turn,
)
from experiments.qwen38_v2.provenance import create_stage_marker, verify_stage_marker


def task_record(
    *,
    config: str = "d1",
    task_id: str = "airline:1",
    repeat: int = 1,
    wall: float = 2.0,
    quality: float | None = 1.0,
) -> dict[str, object]:
    return {
        "schema_version": "spork-e2e-v2",
        "record_type": "task",
        "run_id": "run",
        "stage": "smoke",
        "config": config,
        "repeat": repeat,
        "task_id": task_id,
        "domain": "airline",
        "status": "completed",
        "error": None,
        "task_wall_s": wall,
        "resource_drained_s": wall + 0.1,
        "quality_reward": quality,
        "action_match": True,
        "n_turns": 2,
        "n_tool_turns": 1,
    }


def turn_record(
    *,
    config: str = "d1",
    task_id: str = "airline:1",
    repeat: int = 1,
    d3: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "spork-e2e-v2",
        "record_type": "turn",
        "run_id": "run",
        "stage": "smoke",
        "config": config,
        "repeat": repeat,
        "task_id": task_id,
        "domain": "airline",
        "turn_index": 0,
        "main_start_ms": 0.0,
        "main_first_token_ms": 10.0,
        "main_tool_call_decoded_ms": 100.0,
        "main_completed_ms": 105.0,
        "fork_start_ms": 10.0,
        "fork_tool_call_decoded_ms": 30.0,
        "fork_completed_ms": 31.0,
        "fork_cancel_requested_ms": None,
        "main_tool_call": {"name": "lookup", "arguments": {"id": 1}},
        "fork_tool_call": {"name": "lookup", "arguments": {"id": 1}},
        "probe_confidence": 0.99,
        "probe_attempt": 0,
        "d2_threshold": 0.9,
        "tool_read_only": True,
        "speculative_tool_dispatched": True,
        "speculative_tool_start_ms": 30.0,
        "speculative_tool_end_ms": 80.0,
        "canonical_tool_start_ms": None,
        "canonical_tool_end_ms": None,
        "strict_match": True,
        "speculative_result_reused": True,
        "speculative_execution_wasted": False,
        "duplicate_tool_execution": False,
        "authority_violation": False,
        "write_tool_speculated": False,
        "real_overlap_ms": 50.0,
        "residual_tool_wait_ms": 0.0,
        "obsolete_fork_tail_ms": 0.0,
        "draft_submit_start_ms": 32.0 if d3 else None,
        "draft_submit_end_ms": 34.0 if d3 else None,
        "draft_boundary_hit": d3,
        "d3_proposed_tokens": 8 if d3 else 0,
        "d3_accepted_tokens": 6 if d3 else 0,
        "ngram_proposed_tokens": 10 if d3 else 0,
        "ngram_accepted_tokens": 5 if d3 else 0,
        "draft_clear_start_ms": 106.0 if d3 else None,
        "draft_clear_end_ms": 107.0 if d3 else None,
        "draft_clear_count": 1 if d3 else 0,
    }


def warmup_record(*, config: str = "d1", request_index: int = 1) -> dict[str, object]:
    return {
        "schema_version": "spork-e2e-v2",
        "record_type": "warmup",
        "run_id": "run",
        "stage": "smoke",
        "config": config,
        "repeat": 1,
        "server_mode": "plain",
        "request_index": request_index,
        "status": "completed",
        "error": None,
        "wall_s": 0.5,
    }


def event_record(*, config: str = "d1") -> dict[str, object]:
    return {
        "schema_version": "spork-e2e-v2",
        "record_type": "event",
        "run_id": "run",
        "stage": "smoke",
        "config": config,
        "repeat": 1,
        "task_id": "airline:1",
        "domain": "airline",
        "turn_index": 0,
        "event_index": 0,
        "source": "main",
        "event": "started",
        "monotonic_ms": 0.0,
        "payload": {},
    }


class ContractTest(unittest.TestCase):
    def test_accepts_complete_task_and_turn_records(self) -> None:
        validate_task(task_record())
        validate_turn(turn_record(d3=True))

    def test_rejects_reuse_without_strict_match(self) -> None:
        record = turn_record()
        record["strict_match"] = False
        with self.assertRaisesRegex(ContractError, "requires dispatch and strict match"):
            validate_turn(record)

    def test_recomputes_real_overlap_from_timestamps(self) -> None:
        record = turn_record()
        record["real_overlap_ms"] = 49.0
        validate_turn(record)
        record["real_overlap_ms"] = 40.0
        with self.assertRaisesRegex(ContractError, "does not match"):
            validate_turn(record)

    def test_jsonl_loader_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            line = json.dumps(task_record())
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "duplicate record key"):
                load_records(path, "task")

    def test_relationships_require_exact_warmups_and_turn_counts(self) -> None:
        task = task_record()
        task["n_turns"] = 1
        warmups = [warmup_record(request_index=index) for index in (1, 2)]
        counts = validate_artifact_relationships(
            [task],
            [turn_record()],
            warmups,
            [event_record()],
            expected_warmups=2,
            expected_tasks=1,
        )
        self.assertEqual(counts["config_blocks"], 1)

        with self.assertRaisesRegex(ContractError, "exactly 3 warmups"):
            validate_artifact_relationships(
                [task], [turn_record()], warmups, [event_record()], expected_warmups=3
            )
        with self.assertRaisesRegex(ContractError, "exactly 2 tasks"):
            validate_artifact_relationships(
                [task],
                [turn_record()],
                warmups,
                [event_record()],
                expected_warmups=2,
                expected_tasks=2,
            )

    def test_gate_evidence_requires_every_locked_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.jsonl"
            raw_path.write_text("raw\n", encoding="utf-8")
            gate0_path = root / "gate0.json"
            gate0_path.write_text(
                json.dumps(
                    {
                        "schema_version": "spork-gate-v2",
                        "checks": {
                            "forced_prefix_bytes_verified": True,
                            "forced_prefix_only_in_fork": True,
                            "stop_sequence_complete": True,
                            "main_first_output_single_token": True,
                            "json_main_parse_pass": True,
                            "json_fork_parse_pass": True,
                            "manifest_no_leakage": True,
                            "r1_raw_recomputed": True,
                        },
                        "artifacts": {
                            "raw": {
                                "path": str(raw_path),
                                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(validate_gate0(gate0_path)["checks"]["r1_raw_recomputed"])

            cache_path = root / "cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "schema_version": "spork-gate-v2",
                        "gate": "cache",
                        "passed": True,
                        "checks": {
                            "shared_prefix_verified": {"passed": True},
                            "cached_prefill_le_half_cold": {"passed": True},
                            "main_tpot_regression_le_1pct": {"passed": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(validate_gate_evidence(cache_path, "cache")["passed"])

    def test_stage_marker_rejects_changed_code_or_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            critical = root / "driver.py"
            artifact = root / "gate.json"
            marker_path = root / "marker.json"
            critical.write_text("version = 1\n", encoding="utf-8")
            artifact.write_text("{}\n", encoding="utf-8")
            marker = create_stage_marker(
                stage="smoke", artifact=artifact, critical_paths=[critical]
            )
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertTrue(
                verify_stage_marker(
                    marker_path, stage="smoke", critical_paths=[critical]
                )["passed"]
            )

            critical.write_text("version = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "fingerprint changed"):
                verify_stage_marker(
                    marker_path, stage="smoke", critical_paths=[critical]
                )


if __name__ == "__main__":
    unittest.main()
