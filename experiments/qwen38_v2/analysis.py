from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contract import ContractError


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ContractError("cannot calculate a percentile from an empty sample")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ContractError("cannot calculate a percentile from an empty sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ContractError("cannot calculate a mean from an empty sample")
    return statistics.fmean(values)


def _task_rows(
    records: Iterable[Mapping[str, Any]], config: str
) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["config"] != config:
            continue
        task_id = str(record["task_id"])
        repeat = int(record["repeat"])
        if repeat in grouped[task_id]:
            raise ContractError(
                f"duplicate task/config/repeat record: {config}/{task_id}/{repeat}"
            )
        grouped[task_id][repeat] = record
    if not grouped:
        raise ContractError(f"no task records found for config {config!r}")
    return dict(grouped)


def paired_records(
    records: Sequence[Mapping[str, Any]], baseline: str, treatment: str
) -> tuple[
    dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    dict[str, Any],
]:
    base = _task_rows(records, baseline)
    treat = _task_rows(records, treatment)
    base_tasks = set(base)
    treat_tasks = set(treat)
    common_tasks = sorted(base_tasks & treat_tasks)
    pairs: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    missing_repeats: list[dict[str, Any]] = []
    for task_id in common_tasks:
        base_repeats = set(base[task_id])
        treat_repeats = set(treat[task_id])
        common_repeats = sorted(base_repeats & treat_repeats)
        if base_repeats != treat_repeats:
            missing_repeats.append(
                {
                    "task_id": task_id,
                    "baseline_only": sorted(base_repeats - treat_repeats),
                    "treatment_only": sorted(treat_repeats - base_repeats),
                }
            )
        usable = []
        for repeat in common_repeats:
            baseline_row = base[task_id][repeat]
            treatment_row = treat[task_id][repeat]
            if (
                baseline_row["status"] == "completed"
                and treatment_row["status"] == "completed"
                and baseline_row["task_wall_s"] is not None
                and treatment_row["task_wall_s"] is not None
            ):
                usable.append((baseline_row, treatment_row))
        if usable:
            pairs[task_id] = usable
    diagnostics = {
        "baseline_tasks": len(base_tasks),
        "treatment_tasks": len(treat_tasks),
        "common_tasks": len(common_tasks),
        "paired_completed_tasks": len(pairs),
        "baseline_only_tasks": sorted(base_tasks - treat_tasks),
        "treatment_only_tasks": sorted(treat_tasks - base_tasks),
        "repeat_mismatches": missing_repeats,
        "baseline_noncompleted": sum(
            row["status"] != "completed" for values in base.values() for row in values.values()
        ),
        "treatment_noncompleted": sum(
            row["status"] != "completed"
            for values in treat.values()
            for row in values.values()
        ),
    }
    if not pairs:
        raise ContractError("no completed baseline/treatment task pairs")
    return pairs, diagnostics


def _flatten(
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    task_ids: Sequence[str],
) -> tuple[list[float], list[float]]:
    baseline_walls: list[float] = []
    treatment_walls: list[float] = []
    for task_id in task_ids:
        for baseline, treatment in pairs[task_id]:
            baseline_walls.append(float(baseline["task_wall_s"]))
            treatment_walls.append(float(treatment["task_wall_s"]))
    return baseline_walls, treatment_walls


def _quality_pairs(
    records: Sequence[Mapping[str, Any]], baseline: str, treatment: str
) -> tuple[
    dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    bool,
]:
    base = _task_rows(records, baseline)
    treat = _task_rows(records, treatment)
    complete = set(base) == set(treat)
    pairs: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for task_id in sorted(set(base) & set(treat)):
        base_repeats = set(base[task_id])
        treat_repeats = set(treat[task_id])
        complete = complete and base_repeats == treat_repeats
        task_pairs = []
        for repeat in sorted(base_repeats & treat_repeats):
            baseline_row = base[task_id][repeat]
            treatment_row = treat[task_id][repeat]
            if (
                baseline_row["quality_reward"] is None
                or treatment_row["quality_reward"] is None
            ):
                complete = False
                continue
            task_pairs.append((baseline_row, treatment_row))
        if task_pairs:
            pairs[task_id] = task_pairs
        else:
            complete = False
    return pairs, complete


def _quality_difference(
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    task_ids: Sequence[str],
) -> float:
    differences = [
        float(treatment["quality_reward"]) - float(baseline["quality_reward"])
        for task_id in task_ids
        for baseline, treatment in pairs[task_id]
    ]
    return _mean(differences)


def _statistics_for_sample(
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    task_ids: Sequence[str],
) -> dict[str, float | None]:
    baseline, treatment = _flatten(pairs, task_ids)
    baseline_mean = _mean(baseline)
    treatment_mean = _mean(treatment)
    task_ratios = []
    for task_id in task_ids:
        task_baseline = [float(pair[0]["task_wall_s"]) for pair in pairs[task_id]]
        task_treatment = [float(pair[1]["task_wall_s"]) for pair in pairs[task_id]]
        task_ratios.append(_mean(task_baseline) / _mean(task_treatment))
    baseline_p95 = nearest_rank(baseline, 0.95)
    treatment_p95 = nearest_rank(treatment, 0.95)
    return {
        "baseline_mean_s": baseline_mean,
        "treatment_mean_s": treatment_mean,
        "baseline_p50_s": nearest_rank(baseline, 0.50),
        "treatment_p50_s": nearest_rank(treatment, 0.50),
        "baseline_p95_s": baseline_p95,
        "treatment_p95_s": treatment_p95,
        "ratio_of_means": baseline_mean / treatment_mean,
        "mean_latency_difference_s": _mean(
            [treatment[index] - baseline[index] for index in range(len(baseline))]
        ),
        "p95_speedup": baseline_p95 / treatment_p95,
        "p95_reduction": 1.0 - treatment_p95 / baseline_p95,
        "mean_paired_ratio": _mean(task_ratios),
        "geometric_mean_ratio": math.exp(_mean([math.log(value) for value in task_ratios])),
    }


def _per_repeat_statistics(
    pairs: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
) -> dict[str, dict[str, float]]:
    grouped: dict[int, tuple[list[float], list[float]]] = {}
    for task_pairs in pairs.values():
        for baseline, treatment in task_pairs:
            repeat = int(baseline["repeat"])
            baseline_values, treatment_values = grouped.setdefault(repeat, ([], []))
            baseline_values.append(float(baseline["task_wall_s"]))
            treatment_values.append(float(treatment["task_wall_s"]))
    result: dict[str, dict[str, float]] = {}
    for repeat, (baseline, treatment) in sorted(grouped.items()):
        result[str(repeat)] = {
            "baseline_mean_s": _mean(baseline),
            "treatment_mean_s": _mean(treatment),
            "baseline_p50_s": nearest_rank(baseline, 0.50),
            "treatment_p50_s": nearest_rank(treatment, 0.50),
            "baseline_p95_s": nearest_rank(baseline, 0.95),
            "treatment_p95_s": nearest_rank(treatment, 0.95),
            "ratio_of_means": _mean(baseline) / _mean(treatment),
        }
    return result


def analyze(
    records: Sequence[Mapping[str, Any]],
    baseline: str,
    treatment: str,
    *,
    bootstrap_iters: int = 10_000,
    seed: int = 20260831,
    quality_margin: float = 0.01,
) -> dict[str, Any]:
    if bootstrap_iters <= 0:
        raise ValueError("bootstrap_iters must be positive")
    pairs, diagnostics = paired_records(records, baseline, treatment)
    task_ids = sorted(pairs)
    observed = _statistics_for_sample(pairs, task_ids)
    quality_pairs, quality_complete = _quality_pairs(records, baseline, treatment)
    quality_task_ids = sorted(quality_pairs)
    observed["quality_difference"] = (
        _quality_difference(quality_pairs, quality_task_ids)
        if quality_complete and quality_task_ids
        else None
    )
    rng = random.Random(seed)
    quality_rng = random.Random(seed + 1)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_iters):
        sampled_ids = [rng.choice(task_ids) for _ in task_ids]
        result = _statistics_for_sample(pairs, sampled_ids)
        for field in (
            "ratio_of_means",
            "mean_latency_difference_s",
            "p95_speedup",
        ):
            value = result[field]
            if value is not None:
                samples[field].append(float(value))
        if quality_complete and quality_task_ids:
            sampled_quality_ids = [
                quality_rng.choice(quality_task_ids) for _ in quality_task_ids
            ]
            samples["quality_difference"].append(
                _quality_difference(quality_pairs, sampled_quality_ids)
            )

    intervals = {
        field: [percentile(values, 0.025), percentile(values, 0.975)]
        for field, values in samples.items()
    }
    quality_value = observed["quality_difference"]
    quality_interval = intervals.get("quality_difference")
    if quality_value is None or quality_interval is None:
        quality_verdict = "BLOCKED"
    elif quality_value < -quality_margin or quality_interval[0] < -quality_margin:
        quality_verdict = "NEGATIVE"
    else:
        quality_verdict = "PASS"

    ratio = float(observed["ratio_of_means"])
    ratio_interval = intervals["ratio_of_means"]
    if quality_verdict != "PASS":
        mean_verdict = "NEGATIVE" if quality_verdict == "NEGATIVE" else "BLOCKED"
    elif ratio > 1.0 and ratio_interval[0] > 1.0:
        mean_verdict = "PASS"
    elif ratio > 1.0 and ratio_interval[0] <= 1.0 <= ratio_interval[1]:
        mean_verdict = "INCONCLUSIVE"
    else:
        mean_verdict = "NEGATIVE"

    p95 = float(observed["p95_speedup"])
    p95_interval = intervals["p95_speedup"]
    if quality_verdict != "PASS":
        tail_verdict = "NEGATIVE" if quality_verdict == "NEGATIVE" else "BLOCKED"
    elif p95 > 1.0 and p95_interval[0] > 1.0:
        tail_verdict = "PASS"
    elif p95 > 1.0 and p95_interval[0] <= 1.0 <= p95_interval[1]:
        tail_verdict = "INCONCLUSIVE"
    else:
        tail_verdict = "NEGATIVE"

    return {
        "schema_version": "spork-analysis-v2",
        "baseline": baseline,
        "treatment": treatment,
        "bootstrap_iters": bootstrap_iters,
        "bootstrap_seed": seed,
        "quality_margin": quality_margin,
        "n_task_clusters": len(task_ids),
        "n_quality_task_clusters": len(quality_task_ids),
        "n_paired_rows": sum(len(pairs[task_id]) for task_id in task_ids),
        "diagnostics": diagnostics,
        "per_repeat": _per_repeat_statistics(pairs),
        "observed": observed,
        "confidence_intervals_95": intervals,
        "verdicts": {
            "quality": quality_verdict,
            "mean_latency": mean_verdict,
            "tail_latency": tail_verdict,
        },
    }
