from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import analyze
from .contract import (
    ContractError,
    load_records,
    validate_artifact_relationships,
    validate_gate0,
    validate_gate_evidence,
)
from .gates import evaluate_run_gate
from .provenance import create_stage_marker, fingerprint, verify_stage_marker
from .report import build_quick_report, build_report


def _write_json(value: Any, output: str | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


def _add_artifacts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", required=True, help="V2 task JSONL artifact")
    parser.add_argument("--turns", required=True, help="V2 turn JSONL artifact")


def _add_pair(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and analyze Qwen3.8 SPORK V2 experiment artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("contract", help="validate JSONL schemas only")
    _add_artifacts(contract)
    contract.add_argument("--warmups", required=True, help="discarded warmup JSONL")
    contract.add_argument("--events", required=True, help="raw timeline event JSONL")
    contract.add_argument("--expected-warmups", type=int, default=10)
    contract.add_argument("--expected-tasks", type=int)
    contract.add_argument("--output")

    gate0 = subparsers.add_parser("gate0", help="validate normalized Gate 0 evidence")
    gate0.add_argument("--input", required=True)
    gate0.add_argument("--output")

    evidence = subparsers.add_parser(
        "evidence", help="validate a normalized cache, D2, or D3 gate artifact"
    )
    evidence.add_argument("--input", required=True)
    evidence.add_argument("--gate", required=True, choices=("cache", "d2", "d3"))
    evidence.add_argument("--output")

    gate = subparsers.add_parser("gate", help="evaluate mechanism and safety gates")
    _add_artifacts(gate)
    _add_pair(gate)
    gate.add_argument("--require-mechanism", action="store_true")
    gate.add_argument("--require-d3", action="store_true")
    gate.add_argument("--require-quality", action="store_true")
    gate.add_argument("--quality-margin-pp", type=float, default=1.0)
    gate.add_argument("--max-obsolete-fork-tail-ms", type=float, default=1.0)
    gate.add_argument("--output")

    stats = subparsers.add_parser("analyze", help="calculate locked paired statistics")
    stats.add_argument("--tasks", required=True)
    _add_pair(stats)
    stats.add_argument("--bootstrap-iters", type=int, default=10_000)
    stats.add_argument("--seed", type=int, default=20260831)
    stats.add_argument("--quality-margin-pp", type=float, default=1.0)
    stats.add_argument(
        "--require-quality-pass",
        action="store_true",
        help="return nonzero unless the bootstrap quality gate passes",
    )
    stats.add_argument("--output")

    fingerprints = subparsers.add_parser(
        "fingerprint", help="hash locked code, protocol, launcher, and manifest inputs"
    )
    fingerprints.add_argument("--path", action="append", required=True)
    fingerprints.add_argument("--output")

    marker = subparsers.add_parser(
        "mark-stage", help="create a stage marker bound to inputs and artifacts"
    )
    marker.add_argument("--stage", required=True)
    marker.add_argument("--artifact", required=True)
    marker.add_argument("--marker", required=True)
    marker.add_argument("--path", action="append", required=True)

    verify_marker = subparsers.add_parser(
        "verify-stage", help="reject a stale or modified stage marker"
    )
    verify_marker.add_argument("--stage", required=True)
    verify_marker.add_argument("--marker", required=True)
    verify_marker.add_argument("--path", action="append", required=True)
    verify_marker.add_argument("--output")

    report = subparsers.add_parser(
        "report", help="render locked floor/full results without recomputing metrics"
    )
    report.add_argument("--state-dir", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--output")

    quick_report = subparsers.add_parser(
        "quick-report", help="render the 10-case timing diagnostic"
    )
    quick_report.add_argument("--run-dir", required=True)
    quick_report.add_argument("--output-dir", required=True)
    quick_report.add_argument("--output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "contract":
            tasks = load_records(args.tasks, "task")
            turns = load_records(args.turns, "turn")
            warmups = load_records(args.warmups, "warmup")
            events = load_records(args.events, "event")
            counts = validate_artifact_relationships(
                tasks,
                turns,
                warmups,
                events,
                expected_warmups=args.expected_warmups,
                expected_tasks=args.expected_tasks,
            )
            result = {
                "schema_version": "spork-contract-check-v2",
                "passed": True,
                **counts,
            }
        elif args.command == "gate0":
            evidence = validate_gate0(args.input)
            result = {
                "schema_version": "spork-gate0-result-v2",
                "passed": True,
                "input": str(Path(args.input).resolve()),
                "checks": evidence["checks"],
                "artifacts": evidence["artifacts"],
            }
        elif args.command == "evidence":
            artifact = validate_gate_evidence(args.input, args.gate)
            result = {
                "schema_version": "spork-evidence-result-v2",
                "gate": args.gate,
                "passed": True,
                "input": str(Path(args.input).resolve()),
                "checks": artifact["checks"],
                "metrics": artifact.get("metrics", {}),
            }
        elif args.command == "gate":
            tasks = load_records(args.tasks, "task")
            turns = load_records(args.turns, "turn")
            result = evaluate_run_gate(
                tasks,
                turns,
                baseline=args.baseline,
                treatment=args.treatment,
                require_mechanism=args.require_mechanism,
                require_d3=args.require_d3,
                require_quality=args.require_quality,
                quality_margin=args.quality_margin_pp / 100.0,
                max_obsolete_fork_tail_ms=args.max_obsolete_fork_tail_ms,
            )
        elif args.command == "analyze":
            tasks = load_records(args.tasks, "task")
            result = analyze(
                tasks,
                args.baseline,
                args.treatment,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed,
                quality_margin=args.quality_margin_pp / 100.0,
            )
            if args.require_quality_pass:
                result["passed"] = result["verdicts"]["quality"] == "PASS"
        elif args.command == "fingerprint":
            result = fingerprint(args.path)
        elif args.command == "mark-stage":
            result = create_stage_marker(
                stage=args.stage,
                artifact=args.artifact,
                critical_paths=args.path,
            )
            _write_json(result, args.marker)
            return 0
        elif args.command == "verify-stage":
            result = verify_stage_marker(
                args.marker,
                stage=args.stage,
                critical_paths=args.path,
            )
        elif args.command == "report":
            result = build_report(args.state_dir, args.output_dir)
        elif args.command == "quick-report":
            result = build_quick_report(args.run_dir, args.output_dir)
        else:  # pragma: no cover
            raise AssertionError(f"unhandled command: {args.command}")
    except (ContractError, ValueError) as error:
        sys.stderr.write(f"V2 VALIDATION FAILED: {error}\n")
        return 2

    _write_json(result, getattr(args, "output", None))
    if result.get("passed") is False:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
