from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_qwen38_v2_contract import (
    event_record,
    task_record,
    turn_record,
    warmup_record,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class CliIntegrationTest(unittest.TestCase):
    def test_contract_gate_and_analysis_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_task = task_record(config="baseline", wall=10.0)
            treatment_task = task_record(config="d1", wall=8.0)
            baseline_task["n_turns"] = 1
            treatment_task["n_turns"] = 1
            tasks = root / "tasks.jsonl"
            turns = root / "turns.jsonl"
            events = root / "events.jsonl"
            warmups = root / "warmups.jsonl"
            _write_jsonl(tasks, [baseline_task, treatment_task])
            _write_jsonl(
                turns,
                [turn_record(config="baseline"), turn_record(config="d1")],
            )
            _write_jsonl(
                events,
                [event_record(config="baseline"), event_record(config="d1")],
            )
            _write_jsonl(
                warmups,
                [warmup_record(config="baseline"), warmup_record(config="d1")],
            )

            base = [sys.executable, "-m", "experiments.qwen38_v2"]
            contract = subprocess.run(
                base
                + [
                    "contract",
                    "--tasks",
                    str(tasks),
                    "--turns",
                    str(turns),
                    "--events",
                    str(events),
                    "--warmups",
                    str(warmups),
                    "--expected-warmups",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(json.loads(contract.stdout)["passed"])

            gate = subprocess.run(
                base
                + [
                    "gate",
                    "--tasks",
                    str(tasks),
                    "--turns",
                    str(turns),
                    "--baseline",
                    "baseline",
                    "--treatment",
                    "d1",
                    "--require-mechanism",
                    "--require-quality",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(json.loads(gate.stdout)["passed"])

            analysis = subprocess.run(
                base
                + [
                    "analyze",
                    "--tasks",
                    str(tasks),
                    "--baseline",
                    "baseline",
                    "--treatment",
                    "d1",
                    "--bootstrap-iters",
                    "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(analysis.stdout)
            self.assertEqual(result["observed"]["ratio_of_means"], 1.25)
            self.assertEqual(result["verdicts"]["mean_latency"], "PASS")


if __name__ == "__main__":
    unittest.main()
