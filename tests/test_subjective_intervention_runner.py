"""Offline checks for the 50-condition subjective intervention runner."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_subjective_interventions import WAVES  # noqa: E402
from subjective_intervention_runner import (  # noqa: E402
    CONDITIONS,
    PAIRWISE_CONDITIONS,
    SCORING_CONDITIONS,
    parse_choice,
    parse_score,
    read_jsonl,
    request_for,
    run_condition,
)


PAIR_FILE = "ukp_humor_pairwise.jsonl"
SCORE_FILE = "summeval_relevance_scoring.jsonl"


def pair_row():
    return {
        "id": "pair-1", "dataset": "ukp_humor", "task": "humor_pairwise",
        "input": {"instruction": "Which option contains the funnier text?", "context": "",
                  "options": [{"label": "A", "text": "First"}, {"label": "B", "text": "Second"}]},
        "reference": {"choice": "A", "winner_position": 1,
                      "individual_choices": ["A", "A", "A", "A", "B"]},
        "metadata": {"agreement_group": "clear", "annotation_depth": 5},
    }


def score_row():
    return {
        "id": "score-1", "dataset": "summeval", "task": "summary_relevance_scoring",
        "input": {"instruction": "How relevant is the summary?", "context": "Article", "text": "Summary",
                  "scale": {"minimum": 1, "maximum": 5, "direction": "higher_is_more_relevant"}},
        "reference": {"score": 4.0, "individual_scores": [4] * 8},
        "metadata": {"agreement_group": "clear", "annotation_depth": 8},
    }


def response_for(payload):
    question = payload["questions"]["judgment"]
    if question["type"] == "choice":
        labels = list(question["criteria"])
        remaining = 0.2 / (len(labels) - 1)
        probabilities = {label: remaining for label in labels}
        probabilities[labels[0]] = 0.8
        confidence = (0.8 - 1 / len(labels)) / (1 - 1 / len(labels))
        answer = {"type": "choice", "choice": labels[0],
                  "probabilities": probabilities, "confidence": confidence}
    else:
        count = len(question["criteria"])
        index = count // 2
        probabilities = {str(i): float(i == index) for i in range(count)}
        answer = {"type": "score", "score": float(index),
                  "legend": {str(i): value for i, value in enumerate(question["criteria"])},
                  "probabilities": probabilities, "confidence": 1.0}
    return {"model": "jev-test", "answers": {"judgment": answer},
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "id": "response-id", "provider": "test"}


class InterventionRunnerChecks(unittest.TestCase):
    def test_manifest_has_exactly_50_conditions_and_semantic_waves(self):
        self.assertEqual(len(PAIRWISE_CONDITIONS), 27)
        self.assertEqual(len(SCORING_CONDITIONS), 23)
        self.assertEqual(len(CONDITIONS), 50)
        self.assertEqual(len(WAVES), 32)
        self.assertEqual(sum(value is not None for wave in WAVES for value in wave), 50)
        for pairwise, scoring in WAVES:
            if pairwise and scoring:
                self.assertEqual(pairwise.split("_", 2)[1], scoring.split("_", 2)[1])

    def test_pairwise_factorial_and_tie_order_are_exact(self):
        row = pair_row()
        expected = {
            "pair_exchange_labels": [("B", "First"), ("A", "Second")],
            "pair_exchange_positions": [("B", "Second"), ("A", "First")],
            "pair_exchange_both": [("A", "Second"), ("B", "First")],
        }
        for condition, entries in expected.items():
            payload, context = request_for(PAIR_FILE, row, condition)
            self.assertEqual(list(payload["questions"]["judgment"]["criteria"].items()), entries)
            self.assertNotIn("reference", json.dumps(payload))
            self.assertEqual(set(context["option_to_original"].values()), {"A", "B"})
        for condition, order in {
            "pair_tie_first": ["X", "A", "B"],
            "pair_tie_middle": ["A", "X", "B"],
            "pair_tie_last": ["A", "B", "X"],
        }.items():
            payload, _ = request_for(PAIR_FILE, row, condition)
            self.assertEqual(list(payload["questions"]["judgment"]["criteria"]), order)

    def test_added_label_interventions_have_exact_mappings(self):
        row = pair_row()
        for condition, expected in {
            "pair_numeric_first_1": [("1", "First"), ("2", "Second")],
            "pair_numeric_first_2": [("2", "First"), ("1", "Second")],
        }.items():
            payload, context = request_for(PAIR_FILE, row, condition)
            self.assertEqual(list(payload["questions"]["judgment"]["criteria"].items()), expected)
            self.assertEqual(set(context["option_to_original"].values()), {"A", "B"})

        first, _ = request_for(PAIR_FILE, row, "pair_prior_judge_first_selected")
        second, _ = request_for(PAIR_FILE, row, "pair_prior_judge_second_selected")
        first_options = list(first["questions"]["judgment"]["criteria"].values())
        second_options = list(second["questions"]["judgment"]["criteria"].values())
        self.assertIn("selected this option", first_options[0])
        self.assertIn("did not select this option", first_options[1])
        self.assertIn("did not select this option", second_options[0])
        self.assertIn("selected this option", second_options[1])
        self.assertNotIn("reference", json.dumps(first))

        for condition, identifier in (("score_numeric_identifier_1", "1"),
                                      ("score_numeric_identifier_2", "2")):
            payload, _ = request_for(SCORE_FILE, score_row(), condition)
            self.assertEqual(payload["state"]["candidate_identifier"], identifier)
        for condition, phrase in (("score_prior_judge_high", "top"),
                                  ("score_prior_judge_low", "bottom")):
            payload, _ = request_for(SCORE_FILE, score_row(), condition)
            self.assertIn(phrase, payload["state"]["previous_automated_evaluation"])

    def test_scoring_reverse_and_granularity_mapping(self):
        row = score_row()
        payload, context = request_for(SCORE_FILE, row, "score_anchor_reverse")
        self.assertEqual(context["quality_by_index"], [1.0, 0.75, 0.5, 0.25, 0.0])
        answer = response_for(payload)["answers"]["judgment"]
        parsed, error = parse_score(payload, answer, context)
        self.assertIsNone(error)
        self.assertEqual(parsed["normalized_quality"], 0.5)
        for condition, count in (("score_granularity_3", 3), ("score_granularity_7", 7)):
            payload, context = request_for(SCORE_FILE, row, condition)
            self.assertEqual(len(payload["questions"]["judgment"]["criteria"]), count)
            self.assertEqual(context["quality_by_index"][0], 0.0)
            self.assertEqual(context["quality_by_index"][-1], 1.0)

    def test_choice_and_scale_dependent_score_consistency_tolerances(self):
        payload, _ = request_for(PAIR_FILE, pair_row(), "pair_exchange_labels")
        for gap, valid in ((0.01, True), (0.02, False)):
            answer = {"type": "choice", "choice": "A", "confidence": 0.0,
                      "probabilities": {"A": 0.49, "B": 0.49 + gap}}
            parsed, error = parse_choice(payload, answer)
            self.assertEqual(error is None, valid)
        for condition in ("score_identifier_a", "score_granularity_3", "score_granularity_7"):
            payload, context = request_for(SCORE_FILE, score_row(), condition)
            answer = response_for(payload)["answers"]["judgment"]
            tolerance = 0.01 * len(answer["probabilities"])
            for direction in (-1, 1):
                score = answer["score"] + direction * tolerance
                parsed, error = parse_score(payload, {**answer, "score": score}, context)
                self.assertIsNone(error)
                parsed, error = parse_score(payload, {
                    **answer, "score": score + direction * 0.001,
                }, context)
                self.assertIsNone(parsed)
                self.assertEqual(error["kind"], "invalid_answer")

    def test_condition_output_resumes_without_repeating_success(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source, output = root / "input", root / "output"
            source.mkdir()
            (source / PAIR_FILE).write_text(json.dumps(pair_row()) + "\n", encoding="utf-8")
            calls = []

            def transport(request, timeout):
                payload = json.loads(request.data)
                calls.append(payload)
                return 200, json.dumps(response_for(payload)).encode(), {}

            summary = run_condition(source, output, PAIR_FILE, "pair_exchange_labels",
                                    "test-key", batch_size=1, transport=transport)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(len(calls), 1)
            simple = read_jsonl(output / "pair_exchange_labels" /
                                "ukp_humor_pairwise" / "simple.jsonl")
            self.assertEqual(len(simple), 1)
            self.assertEqual(simple[0]["condition"], "pair_exchange_labels")

            resumed = run_condition(source, output, PAIR_FILE, "pair_exchange_labels",
                                    "test-key", batch_size=1, transport=transport)
            self.assertEqual(resumed["skipped_successful"], 1)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
