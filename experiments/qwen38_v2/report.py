from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract import ContractError, load_records


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"report input does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _artifact(state_dir: Path, stage: str) -> Path:
    marker = _json(state_dir / f"{stage}.pass.json")
    artifact = marker.get("artifact")
    if not isinstance(artifact, str):
        raise ContractError(f"{stage} marker has no artifact path")
    path = Path(artifact)
    if not path.exists():
        raise ContractError(f"{stage} artifact no longer exists: {path}")
    return path


def _comparison(analysis: Mapping[str, Any]) -> dict[str, Any]:
    observed = analysis["observed"]
    intervals = analysis["confidence_intervals_95"]
    verdicts = analysis["verdicts"]
    return {
        "baseline": analysis["baseline"],
        "treatment": analysis["treatment"],
        "n_task_clusters": analysis["n_task_clusters"],
        "n_paired_rows": analysis["n_paired_rows"],
        "baseline_mean_s": observed["baseline_mean_s"],
        "treatment_mean_s": observed["treatment_mean_s"],
        "ratio_of_means": observed["ratio_of_means"],
        "ratio_of_means_ci95": intervals["ratio_of_means"],
        "p95_speedup": observed["p95_speedup"],
        "p95_speedup_ci95": intervals["p95_speedup"],
        "mean_paired_ratio_secondary": observed["mean_paired_ratio"],
        "geometric_mean_ratio_secondary": observed["geometric_mean_ratio"],
        "quality_difference": observed["quality_difference"],
        "quality_difference_ci95": intervals.get("quality_difference"),
        "verdicts": verdicts,
        "estimator_direction_flip": (
            (float(observed["ratio_of_means"]) - 1.0)
            * (float(observed["mean_paired_ratio"]) - 1.0)
            < 0.0
        ),
        "per_repeat": analysis["per_repeat"],
        "diagnostics": analysis["diagnostics"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SPORK Qwen3.8-27B V2 Core E2E Report",
        "",
        f"- Reproduction level: `{summary['reproduction_level']}`",
        f"- Core verdict: `{summary['core_verdict']}`",
        f"- Mechanism verdict: `{summary['mechanism_verdict']}`",
        f"- D3 verdict: `{summary['d3_verdict']}`",
        "",
        "## Full-155",
        "",
        "| Comparison | Mean B/T | 95% CI | P95 B/T | 95% CI | Quality | Mean verdict | Tail verdict |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name in ("plain", "composite"):
        comparison = summary["full155"][name]
        mean_ci = comparison["ratio_of_means_ci95"]
        p95_ci = comparison["p95_speedup_ci95"]
        lines.append(
            "| {name} | {mean:.3f} | [{mean_lo:.3f}, {mean_hi:.3f}] | "
            "{p95:.3f} | [{p95_lo:.3f}, {p95_hi:.3f}] | {quality} | {mean_v} | "
            "{tail_v} |".format(
                name=name,
                mean=comparison["ratio_of_means"],
                mean_lo=mean_ci[0],
                mean_hi=mean_ci[1],
                p95=comparison["p95_speedup"],
                p95_lo=p95_ci[0],
                p95_hi=p95_ci[1],
                quality=comparison["verdicts"]["quality"],
                mean_v=comparison["verdicts"]["mean_latency"],
                tail_v=comparison["verdicts"]["tail_latency"],
            )
        )
    lines.extend(
        [
            "",
            "`ratio_of_means = mean(baseline) / mean(treatment)` is the primary mean "
            "estimator. `mean_paired_ratio` is secondary and never overrides it.",
            "",
            "## Floor Sweep",
            "",
            "| Floor (s) | Plain mean B/T | Composite mean B/T | Plain verdict | Composite verdict |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for floor, values in summary["floor_sweep"].items():
        lines.append(
            f"| {floor} | {values['plain']['ratio_of_means']:.3f} | "
            f"{values['composite']['ratio_of_means']:.3f} | "
            f"{values['plain']['verdicts']['mean_latency']} | "
            f"{values['composite']['verdicts']['mean_latency']} |"
        )
    lines.extend(["", "## Failures", ""])
    failures = summary["full155_failures"]
    if failures:
        lines.extend(
            f"- `{item['config']}` repeat {item['repeat']} `{item['task_id']}`: "
            f"`{item['status']}`"
            for item in failures
        )
    else:
        lines.append("No non-completed full-155 task rows.")
    lines.append("")
    flips = [
        name
        for name, comparison in summary["full155"].items()
        if comparison["estimator_direction_flip"]
    ]
    if flips:
        lines.append(
            "Estimator direction flip detected for: " + ", ".join(sorted(flips)) + "."
        )
        lines.append("")
    return "\n".join(lines)


def build_report(state_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    state = Path(state_dir)
    destination = Path(output_dir)
    full_root = _artifact(state, "full155")
    floor_root = _artifact(state, "floor")
    d3_root = _artifact(state, "d3")
    d3_evidence = _json(d3_root / "analysis/d3.json")
    plain_gate = _json(full_root / "analysis/plain-gate.json")
    composite_gate = _json(full_root / "analysis/composite-gate.json")
    plain = _comparison(_json(full_root / "analysis/plain-analysis.json"))
    composite = _comparison(_json(full_root / "analysis/composite-analysis.json"))

    floor_sweep: dict[str, Any] = {}
    for floor_dir in sorted(floor_root.glob("floor-*")):
        floor = floor_dir.name.removeprefix("floor-")
        floor_sweep[floor] = {
            "plain": _comparison(_json(floor_dir / "analysis/plain-analysis.json")),
            "composite": _comparison(
                _json(floor_dir / "analysis/composite-analysis.json")
            ),
        }
    if set(floor_sweep) != {"0.5", "1", "2", "5"}:
        raise ContractError(f"incomplete floor sweep: {sorted(floor_sweep)}")

    task_rows = load_records(full_root / "raw/tasks.jsonl", "task")
    failures = [
        {
            "config": row["config"],
            "repeat": row["repeat"],
            "task_id": row["task_id"],
            "status": row["status"],
            "error": row["error"],
        }
        for row in task_rows
        if row["status"] != "completed"
    ]
    mechanism_pass = bool(plain_gate.get("passed") and composite_gate.get("passed"))
    d3_mechanism_pass = (
        mechanism_pass
        and d3_evidence.get("passed") is True
        and composite_gate.get("metrics", {}).get("d3_accepted_tokens", 0) > 0
    )
    d3_speed_pass = (
        composite["verdicts"]["quality"] == "PASS"
        and (
            composite["verdicts"]["mean_latency"] == "PASS"
            or composite["verdicts"]["tail_latency"] == "PASS"
        )
    )
    d3_verdict = "PASS" if d3_mechanism_pass and d3_speed_pass else (
        "MECHANISM_ONLY" if d3_mechanism_pass else "BLOCKED"
    )
    if d3_verdict == "PASS":
        level = "L4"
        core_verdict = "PASS"
    elif d3_mechanism_pass:
        level = "L3"
        mean_tail = {
            composite["verdicts"]["mean_latency"],
            composite["verdicts"]["tail_latency"],
        }
        core_verdict = "NEGATIVE" if mean_tail == {"NEGATIVE"} else "INCONCLUSIVE"
    elif mechanism_pass:
        level = "L2"
        core_verdict = "BLOCKED"
    else:
        level = "L1"
        core_verdict = "BLOCKED"

    summary = {
        "schema_version": "spork-core-report-v2",
        "reproduction_level": level,
        "core_verdict": core_verdict,
        "mechanism_verdict": "PASS" if mechanism_pass else "BLOCKED",
        "d3_verdict": d3_verdict,
        "full155": {"plain": plain, "composite": composite},
        "floor_sweep": floor_sweep,
        "full155_failures": failures,
        "artifacts": {
            "full155": str(full_root),
            "floor": str(floor_root),
            "d3": str(d3_root),
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "CORE_E2E_V2_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "CORE_E2E_V2_REPORT.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    deviations = {
        "schema_version": "spork-deviations-v2",
        "full155_failures": failures,
        "estimator_direction_flips": [
            name
            for name, comparison in summary["full155"].items()
            if comparison["estimator_direction_flip"]
        ],
        "diagnostics": {
            name: comparison["diagnostics"]
            for name, comparison in summary["full155"].items()
        },
    }
    (destination / "FAILURES_AND_DEVIATIONS.json").write_text(
        json.dumps(deviations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    deviation_lines = ["# Failures And Deviations", ""]
    if failures:
        deviation_lines.extend(
            f"- `{item['config']}` repeat {item['repeat']} `{item['task_id']}`: "
            f"`{item['status']}`; error={item['error']!r}"
            for item in failures
        )
    else:
        deviation_lines.append("No non-completed full-155 task rows.")
    deviation_lines.extend(["", "## Estimator Direction Flips", ""])
    flips = deviations["estimator_direction_flips"]
    deviation_lines.append(", ".join(flips) if flips else "None.")
    deviation_lines.append("")
    (destination / "FAILURES_AND_DEVIATIONS.md").write_text(
        "\n".join(deviation_lines), encoding="utf-8"
    )
    return summary


def build_quick_report(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir)
    destination = Path(output_dir)
    composite_gate = _json(run_root / "analysis/composite-gate.json")
    composite = _comparison(_json(run_root / "analysis/composite-analysis.json"))
    task_rows = load_records(run_root / "raw/tasks.jsonl", "task")
    task_ids = sorted({str(row["task_id"]) for row in task_rows})
    failures = [
        {
            "config": row["config"],
            "repeat": row["repeat"],
            "task_id": row["task_id"],
            "status": row["status"],
            "error": row["error"],
        }
        for row in task_rows
        if row["status"] != "completed"
    ]
    mechanism_pass = bool(composite_gate.get("passed"))
    quality_pass = composite["verdicts"]["quality"] == "PASS"
    ratio = float(composite["ratio_of_means"])
    if not mechanism_pass:
        timing_signal = "MECHANISM_BLOCKED"
        recommendation = "fix mechanism gates; do not run more cases"
    elif not quality_pass:
        timing_signal = "QUALITY_NEGATIVE"
        recommendation = "audit task trajectories and grader before scaling"
    elif ratio > 1.0:
        timing_signal = "FASTER_POINT_ESTIMATE"
        recommendation = "repeat or expand only if the confidence intervals justify it"
    else:
        timing_signal = "SLOWER_POINT_ESTIMATE"
        recommendation = "do not scale; use the timeline to remove overhead first"
    summary = {
        "schema_version": "spork-quick10-report-v2",
        "scope": "10-case timing diagnostic; not a paper-reproduction verdict",
        "run_dir": str(run_root.resolve()),
        "n_unique_tasks": len(task_ids),
        "task_ids": task_ids,
        "mechanism_pass": mechanism_pass,
        "quality_pass": quality_pass,
        "timing_signal": timing_signal,
        "recommendation": recommendation,
        "composite": composite,
        "composite_gate": composite_gate,
        "failures": failures,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "QUICK10_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SPORK Quick10 Timing Diagnostic",
        "",
        "This is a 10-case diagnostic, not a paper-reproduction verdict.",
        "",
        f"- Timing signal: `{timing_signal}`",
        f"- Mechanism gate: `{'PASS' if mechanism_pass else 'FAIL'}`",
        f"- Quality gate: `{'PASS' if quality_pass else 'FAIL'}`",
        f"- Recommendation: {recommendation}",
        f"- Speculative dispatches: {composite_gate.get('metrics', {}).get('dispatches', 0)}",
        f"- Reused tool results: {composite_gate.get('metrics', {}).get('reused_results', 0)}",
        f"- Real tool overlap: {composite_gate.get('metrics', {}).get('real_overlap_ms', 0):.1f} ms",
        f"- D3 accepted/proposed tokens: "
        f"{composite_gate.get('metrics', {}).get('d3_accepted_tokens', 0)}/"
        f"{composite_gate.get('metrics', {}).get('d3_proposed_tokens', 0)}",
        "",
        "| Comparison | Mean B/T | 95% CI | Mean B (s) | Mean T (s) | P95 B/T | Quality |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    interval = composite["ratio_of_means_ci95"]
    lines.append(
        f"| matched ngram vs D1/D2/D3 | {composite['ratio_of_means']:.3f} | "
        f"[{interval[0]:.3f}, {interval[1]:.3f}] | "
        f"{composite['baseline_mean_s']:.3f} | "
        f"{composite['treatment_mean_s']:.3f} | "
        f"{composite['p95_speedup']:.3f} | "
        f"{composite['verdicts']['quality']} |"
    )
    lines.extend(
        [
            "",
            "Primary estimator: `mean(baseline) / mean(treatment)`. Values above 1 "
            "mean faster treatment.",
            "",
        ]
    )
    if failures:
        lines.append("## Non-completed Tasks")
        lines.append("")
        lines.extend(
            f"- `{item['config']}` repeat {item['repeat']} `{item['task_id']}`: "
            f"`{item['status']}`"
            for item in failures
        )
        lines.append("")
    (destination / "QUICK10_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary
