from __future__ import annotations

import unittest

from experiments.qwen38_v2.analysis import analyze, nearest_rank
from experiments.qwen38_v2.gates import evaluate_run_gate
from tests.test_qwen38_v2_contract import task_record, turn_record


class AnalysisTest(unittest.TestCase):
    def test_nearest_rank(self) -> None:
        self.assertEqual(nearest_rank([1, 2, 3, 4], 0.50), 2.0)
        self.assertEqual(nearest_rank([1, 2, 3, 4], 0.95), 4.0)

    def test_reports_estimator_direction_flip_without_changing_primary_verdict(self) -> None:
        records = [
            task_record(config="baseline", task_id="short", wall=1.0),
            task_record(config="treatment", task_id="short", wall=0.1),
            task_record(config="baseline", task_id="long", wall=100.0),
            task_record(config="treatment", task_id="long", wall=200.0),
        ]

        result = analyze(records, "baseline", "treatment", bootstrap_iters=200, seed=7)

        self.assertLess(result["observed"]["ratio_of_means"], 1.0)
        self.assertGreater(result["observed"]["mean_paired_ratio"], 1.0)
        self.assertEqual(result["verdicts"]["mean_latency"], "NEGATIVE")

    def test_bootstrap_clusters_repeats_by_task(self) -> None:
        records = []
        for repeat in (1, 2, 3):
            records.extend(
                (
                    task_record(
                        config="baseline", task_id="one", repeat=repeat, wall=10.0
                    ),
                    task_record(
                        config="treatment", task_id="one", repeat=repeat, wall=5.0
                    ),
                    task_record(
                        config="baseline", task_id="two", repeat=repeat, wall=20.0
                    ),
                    task_record(
                        config="treatment", task_id="two", repeat=repeat, wall=10.0
                    ),
                )
            )

        result = analyze(records, "baseline", "treatment", bootstrap_iters=100, seed=11)

        self.assertEqual(result["n_task_clusters"], 2)
        self.assertEqual(result["n_paired_rows"], 6)
        self.assertEqual(result["observed"]["ratio_of_means"], 2.0)
        self.assertEqual(result["confidence_intervals_95"]["ratio_of_means"], [2.0, 2.0])

    def test_quality_includes_shared_noncompleted_tasks(self) -> None:
        records = [
            task_record(config="baseline", task_id="easy", wall=10.0, quality=1.0),
            task_record(config="treatment", task_id="easy", wall=8.0, quality=1.0),
            task_record(config="baseline", task_id="hard", wall=20.0, quality=1.0),
            task_record(config="treatment", task_id="hard", wall=20.0, quality=0.0),
        ]
        for row in records[2:]:
            row["status"] = "max_turns"

        result = analyze(records, "baseline", "treatment", bootstrap_iters=100, seed=5)

        self.assertEqual(result["n_task_clusters"], 1)
        self.assertEqual(result["n_quality_task_clusters"], 2)
        self.assertEqual(result["observed"]["quality_difference"], -0.5)
        self.assertEqual(result["verdicts"]["quality"], "NEGATIVE")


class GateTest(unittest.TestCase):
    def test_mechanism_and_d3_gate_passes_only_with_real_reuse(self) -> None:
        tasks = [
            task_record(config="baseline-ngram"),
            task_record(config="d1-d2-d3"),
        ]
        turns = [turn_record(config="d1-d2-d3", d3=True)]

        result = evaluate_run_gate(
            tasks,
            turns,
            baseline="baseline-ngram",
            treatment="d1-d2-d3",
            require_mechanism=True,
            require_d3=True,
            require_quality=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["reused_results"], 1)
        self.assertEqual(result["metrics"]["d3_accepted_tokens"], 6)

    def test_gate_rejects_cost_only_fork(self) -> None:
        tasks = [task_record(config="baseline"), task_record(config="d1")]
        turn = turn_record(config="d1")
        turn["speculative_tool_dispatched"] = False
        turn["speculative_tool_start_ms"] = None
        turn["speculative_tool_end_ms"] = None
        turn["strict_match"] = False
        turn["speculative_result_reused"] = False
        turn["real_overlap_ms"] = 0.0

        result = evaluate_run_gate(
            tasks,
            [turn],
            baseline="baseline",
            treatment="d1",
            require_mechanism=True,
            require_d3=False,
            require_quality=True,
        )

        self.assertFalse(result["passed"])
        self.assertIn("speculative_result_reused", result["failed_checks"])
        self.assertIn("positive_real_overlap", result["failed_checks"])

    def test_gate_allows_shared_failure_but_rejects_treatment_only_failure(self) -> None:
        shared_baseline = task_record(config="baseline", task_id="hard")
        shared_treatment = task_record(config="d1", task_id="hard")
        for row in (shared_baseline, shared_treatment):
            row["status"] = "max_turns"
            row["task_wall_s"] = 20.0
            row["quality_reward"] = 0.0
        shared = evaluate_run_gate(
            [shared_baseline, shared_treatment],
            [turn_record(config="d1", task_id="hard")],
            baseline="baseline",
            treatment="d1",
            require_mechanism=False,
            require_d3=False,
            require_quality=True,
        )
        self.assertTrue(shared["checks"]["no_treatment_only_failures"]["passed"])

        shared_baseline["status"] = "completed"
        treatment_only = evaluate_run_gate(
            [shared_baseline, shared_treatment],
            [turn_record(config="d1", task_id="hard")],
            baseline="baseline",
            treatment="d1",
            require_mechanism=False,
            require_d3=False,
            require_quality=True,
        )
        self.assertFalse(treatment_only["passed"])
        self.assertIn("no_treatment_only_failures", treatment_only["failed_checks"])


if __name__ == "__main__":
    unittest.main()
