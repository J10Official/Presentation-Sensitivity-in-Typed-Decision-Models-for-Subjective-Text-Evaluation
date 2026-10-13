"""Offline checks for the interface-matched comparison launcher."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_decision_model_comparisons import (  # noqa: E402
    MODEL_ROUTES,
    SCHEMA_SMOKE_DATASETS,
    baseline_command,
    intervention_command,
    run_process_until_success,
)


def arguments():
    return argparse.Namespace(
        live=True,
        preview=False,
        batch_size=20,
        max_attempts=3,
        timeout=60,
        limit_per_dataset=None,
        supervisor_attempts=0,
        resume_delay=0,
    )


class ComparisonLauncherChecks(unittest.TestCase):
    def test_routes_and_schema_smoke_cover_choice_and_score(self):
        self.assertEqual(MODEL_ROUTES, {
            "clef_flash": "cloudflare/clef-flash",
            "pplx_decider_v1_1_27b": "perplexity/pplx-decider-v1.1-27b",
        })
        self.assertEqual(SCHEMA_SMOKE_DATASETS,
                         ("ukp_humor_pairwise.jsonl", "humicroedit_scoring.jsonl"))

    def test_commands_preserve_route_and_isolate_output(self):
        args = arguments()
        root = Path("separate-output")
        for route in MODEL_ROUTES.values():
            baseline = baseline_command(args, route, root / "baseline", SCHEMA_SMOKE_DATASETS, 1)
            intervention = intervention_command(args, route, root / "interventions")
            self.assertEqual(baseline[baseline.index("--model") + 1], route)
            self.assertEqual(intervention[intervention.index("--model") + 1], route)
            self.assertEqual(Path(baseline[baseline.index("--output-root") + 1]), root / "baseline")
            self.assertEqual(Path(intervention[intervention.index("--output-root") + 1]),
                             root / "interventions")
            self.assertIn("--limit-per-dataset", baseline)
            self.assertNotIn("--limit-per-dataset", intervention)

    def test_supervisor_restarts_only_temporary_exit_and_logs_attempts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            with patch("run_decision_model_comparisons.run_process", side_effect=[75, 75, 0]) as run:
                code = run_process_until_success(
                    ["worker"], root, root / "out.log", root / "err.log",
                    root / "supervisor.jsonl", attempts=0, resume_delay=0, label="test")
            self.assertEqual(code, 0)
            self.assertEqual(run.call_count, 3)
            records = [json.loads(line) for line in
                       (root / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(record["event"] == "process_resume" for record in records), 2)

    def test_supervisor_does_not_loop_on_permanent_exit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            with patch("run_decision_model_comparisons.run_process", return_value=1) as run:
                code = run_process_until_success(
                    ["worker"], root, root / "out.log", root / "err.log",
                    root / "supervisor.jsonl", attempts=0, resume_delay=0, label="test")
            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
