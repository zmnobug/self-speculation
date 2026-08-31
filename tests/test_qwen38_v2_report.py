from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.qwen38_v2.analysis import analyze
from experiments.qwen38_v2.report import build_quick_report, build_report
from tests.test_qwen38_v2_contract import task_record


class ReportTest(unittest.TestCase):
    def test_builds_locked_primary_summary(self) -> None:
        records = [
            task_record(config="baseline-serial", wall=10.0),
            task_record(config="d1-d2", wall=5.0),
            task_record(config="baseline-ngram", wall=10.0),
            task_record(config="d1-d2-d3", wall=5.0),
        ]
        plain = analyze(
            records, "baseline-serial", "d1-d2", bootstrap_iters=20, seed=1
        )
        composite = analyze(
            records, "baseline-ngram", "d1-d2-d3", bootstrap_iters=20, seed=1
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            full = root / "full"
            floors = root / "floors"
            d3 = root / "d3"
            output = root / "results"
            state.mkdir()
            (full / "analysis").mkdir(parents=True)
            (full / "raw").mkdir()
            (d3 / "analysis").mkdir(parents=True)
            (d3 / "analysis/d3.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            (full / "analysis/plain-analysis.json").write_text(
                json.dumps(plain), encoding="utf-8"
            )
            (full / "analysis/composite-analysis.json").write_text(
                json.dumps(composite), encoding="utf-8"
            )
            (full / "analysis/plain-gate.json").write_text(
                json.dumps({"passed": True, "metrics": {}}), encoding="utf-8"
            )
            (full / "analysis/composite-gate.json").write_text(
                json.dumps({"passed": True, "metrics": {"d3_accepted_tokens": 5}}),
                encoding="utf-8",
            )
            (full / "raw/tasks.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            for floor in ("0.5", "1", "2", "5"):
                analysis = floors / f"floor-{floor}" / "analysis"
                analysis.mkdir(parents=True)
                (analysis / "plain-analysis.json").write_text(
                    json.dumps(plain), encoding="utf-8"
                )
                (analysis / "composite-analysis.json").write_text(
                    json.dumps(composite), encoding="utf-8"
                )
            (state / "full155.pass.json").write_text(
                json.dumps({"artifact": str(full)}), encoding="utf-8"
            )
            (state / "floor.pass.json").write_text(
                json.dumps({"artifact": str(floors)}), encoding="utf-8"
            )
            (state / "d3.pass.json").write_text(
                json.dumps({"artifact": str(d3)}), encoding="utf-8"
            )

            summary = build_report(state, output)

            self.assertEqual(summary["reproduction_level"], "L4")
            self.assertEqual(summary["core_verdict"], "PASS")
            self.assertEqual(
                summary["full155"]["composite"]["ratio_of_means"], 2.0
            )
            self.assertTrue((output / "CORE_E2E_V2_REPORT.md").is_file())
            self.assertTrue((output / "FAILURES_AND_DEVIATIONS.md").is_file())

            quick_output = root / "quick-results"
            quick = build_quick_report(full, quick_output)
            self.assertEqual(quick["timing_signal"], "FASTER_POINT_ESTIMATE")
            self.assertEqual(quick["composite"]["ratio_of_means"], 2.0)
            self.assertTrue((quick_output / "QUICK10_REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
