"""Offline contract checks for the unified subjective baseline runner."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subjective_baseline_runner import (  # noqa: E402
    DATASETS,
    digest,
    parse_choice,
    parse_score,
    read_jsonl,
    request_for,
    run_all,
    valid_distribution,
)


CHOICE_FILE = "ukp_humor_pairwise.jsonl"
SCORE_FILE = "summeval_relevance_scoring.jsonl"


def choice_row(ident="choice-1", first="First joke"):
    return {
        "id": ident, "dataset": "ukp_humor", "task": "humor_pairwise",
        "input": {"instruction": "Which option contains the funnier text?", "context": "",
                  "options": [{"label": "A", "text": first}, {"label": "B", "text": "Second joke"}]},
        "reference": {"choice": "A", "winner_position": 1, "individual_choices": ["A"] * 4 + ["B"]},
        "metadata": {"agreement_group": "clear", "annotation_depth": 5},
    }


def score_row(ident="score-1"):
    return {
        "id": ident, "dataset": "summeval", "task": "summary_relevance_scoring",
        "input": {"instruction": "How relevant is the summary to the source article?",
                  "context": "Source article.", "text": "Summary.",
                  "scale": {"minimum": 1, "maximum": 5, "direction": "higher_is_more_relevant"}},
        "reference": {"score": 4.0, "individual_scores": [4] * 8},
        "metadata": {"agreement_group": "clear", "annotation_depth": 8},
    }


def response_for(payload):
    question = payload["questions"]["judgment"]
    if question["type"] == "choice":
        labels = list(question["criteria"])
        answer = {"type": "choice", "choice": labels[0],
                  "probabilities": {labels[0]: 0.8, labels[1]: 0.2}, "confidence": 0.6}
    else:
        legend = {str(index): value for index, value in enumerate(question["criteria"])}
        probabilities = {str(index): 0.0 for index in range(len(legend))}
        probabilities["3"] = 1.0
        answer = {"type": "score", "score": 3.0, "legend": legend,
                  "probabilities": probabilities, "confidence": 1.0}
    return {"model": "jev-test", "answers": {"judgment": answer},
            "usage": {"input_tokens": 10, "output_tokens": 5}, "id": "response-id", "provider": "test"}


class SubjectiveRunnerChecks(unittest.TestCase):
    def write_inputs(self, root, rows):
        for name, values in rows.items():
            (root / name).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def test_requests_contain_only_judge_input_and_score_maps_to_native_scale(self):
        choice = choice_row()
        payload = request_for(CHOICE_FILE, choice)
        serialized = json.dumps(payload)
        self.assertNotIn("reference", serialized)
        self.assertNotIn("agreement_group", serialized)
        self.assertEqual(payload["questions"]["judgment"]["criteria"], {"A": "First joke", "B": "Second joke"})

        score = score_row()
        payload = request_for(SCORE_FILE, score)
        self.assertEqual(payload["questions"]["judgment"]["type"], "score")
        self.assertEqual(len(payload["questions"]["judgment"]["criteria"]), 5)
        self.assertEqual(DATASETS[SCORE_FILE]["native_minimum"], 1)

    def test_parsers_reject_internally_inconsistent_answers(self):
        choice_payload = request_for(CHOICE_FILE, choice_row())
        parsed, error = parse_choice(choice_payload, {
            "type": "choice", "choice": "A", "probabilities": {"A": 0.2, "B": 0.8},
            "confidence": 0.6,
        })
        self.assertIsNone(parsed)
        self.assertEqual(error["kind"], "invalid_answer")

    def test_parsers_accept_provider_specific_confidence_semantics(self):
        choice_payload = request_for(CHOICE_FILE, choice_row())
        parsed, error = parse_choice(choice_payload, {
            "type": "choice", "choice": "A", "probabilities": {"A": 0.7119, "B": 0.2881},
            "confidence": 0.1797,
        })
        self.assertIsNone(error)
        self.assertEqual(parsed["confidence"], 0.1797)
        self.assertAlmostEqual(parsed["probability_derived_confidence"], 0.4238)

        score_payload = request_for(SCORE_FILE, score_row())
        criteria = score_payload["questions"]["judgment"]["criteria"]
        probabilities = {"0": 0.4278, "1": 0.2437, "2": 0.1963, "3": 0.1322, "4": 0.0}
        score = sum(index * probabilities[str(index)] for index in range(5))
        parsed, error = parse_score(SCORE_FILE, score_payload, {
            "type": "score", "score": score,
            "legend": {str(index): value for index, value in enumerate(criteria)},
            "probabilities": probabilities, "confidence": 0.0645,
        })
        self.assertIsNone(error)
        self.assertEqual(parsed["confidence"], 0.0645)

        score_payload = request_for(SCORE_FILE, score_row())
        criteria = score_payload["questions"]["judgment"]["criteria"]
        parsed, error = parse_score(SCORE_FILE, score_payload, {
            "type": "score", "score": 2.0,
            "legend": {str(index): value for index, value in enumerate(criteria)},
            "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0},
            "confidence": 0.0,
        })
        self.assertIsNone(parsed)
        self.assertEqual(error["kind"], "invalid_answer")

    def test_probability_mass_tolerance_is_point_zero_one(self):
        self.assertTrue(valid_distribution({"A": 0.45, "B": 0.54}, ["A", "B"]))
        self.assertTrue(valid_distribution({"A": 0.46, "B": 0.55}, ["A", "B"]))
        self.assertFalse(valid_distribution({"A": 0.45, "B": 0.539}, ["A", "B"]))
        self.assertFalse(valid_distribution({"A": 0.46, "B": 0.551}, ["A", "B"]))

    def test_choice_consistency_tolerance_is_point_zero_one(self):
        choice_payload = request_for(CHOICE_FILE, choice_row())
        parsed, error = parse_choice(choice_payload, {
            "type": "choice", "choice": "A", "probabilities": {"A": 0.49, "B": 0.50},
            "confidence": 0.0,
        })
        self.assertIsNone(error)
        self.assertEqual(parsed["choice"], "A")
        parsed, error = parse_choice(choice_payload, {
            "type": "choice", "choice": "A", "probabilities": {"A": 0.49, "B": 0.51},
            "confidence": 0.0,
        })
        self.assertIsNone(parsed)
        self.assertEqual(error["kind"], "invalid_answer")

    def test_score_consistency_tolerance_is_point_zero_one_per_level(self):
        score_payload = request_for(SCORE_FILE, score_row())
        for count in (3, 4, 5, 7):
            with self.subTest(levels=count):
                criteria = [f"Level {index}" for index in range(count)]
                score_payload["questions"]["judgment"]["criteria"] = criteria
                common = {
                    "type": "score",
                    "legend": {str(index): value for index, value in enumerate(criteria)},
                    "probabilities": {str(index): float(index == 1) for index in range(count)},
                    "confidence": 1.0,
                }
                for direction in (-1, 1):
                    score = 1 + direction * 0.01 * count
                    parsed, error = parse_score(SCORE_FILE, score_payload, {**common, "score": score})
                    self.assertIsNone(error)
                    self.assertAlmostEqual(parsed["api_score"], score)
                    parsed, error = parse_score(SCORE_FILE, score_payload, {
                        **common, "score": score + direction * 0.001,
                    })
                    self.assertIsNone(parsed)
                    self.assertEqual(error["kind"], "invalid_answer")

    def test_separate_full_exact_and_simple_logs_with_resume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source, output = root / "input", root / "output"
            source.mkdir()
            rows = {CHOICE_FILE: [choice_row()], SCORE_FILE: [score_row()]}
            self.write_inputs(source, rows)
            calls = []

            def transport(request, timeout):
                payload = json.loads(request.data)
                calls.append(payload)
                body = json.dumps(response_for(payload)).encode("utf-8")
                return 200, body, {"X-Test": "complete"}

            summary = run_all(source, output, "test-key", (CHOICE_FILE, SCORE_FILE),
                              batch_size=2, transport=transport)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(len(calls), 2)
            for name in (CHOICE_FILE, SCORE_FILE):
                directory = output / Path(name).stem
                full = read_jsonl(directory / "full_results.jsonl")
                simple = read_jsonl(directory / "simple.jsonl")
                events = read_jsonl(directory / "events.jsonl")
                self.assertEqual(len(full), 1)
                self.assertEqual(len(simple), 1)
                self.assertEqual([event["event"] for event in events], ["request", "response"])
                self.assertEqual(json.loads(events[0]["body_utf8"]), full[0]["request"])
                self.assertEqual(full[0]["request_sha256"], digest(full[0]["request"]))
                self.assertEqual(events[1]["headers"], {"X-Test": "complete"})
                self.assertNotIn("raw_response", simple[0])
            score_simple = read_jsonl(output / Path(SCORE_FILE).stem / "simple.jsonl")[0]
            self.assertEqual(score_simple["api_score"], 3.0)
            self.assertEqual(score_simple["native_score"], 4.0)
            self.assertEqual(score_simple["signed_error"], 0.0)

            resumed = run_all(source, output, "test-key", (CHOICE_FILE, SCORE_FILE),
                              batch_size=2, transport=transport)
            self.assertEqual(resumed["skipped_successful"], 2)
            self.assertEqual(len(calls), 2)

            rows[SCORE_FILE][0]["reference"]["score"] = 3.5
            self.write_inputs(source, rows)
            reanalyzed = run_all(source, output, "test-key", (CHOICE_FILE, SCORE_FILE),
                                 batch_size=2, transport=transport)
            self.assertEqual(reanalyzed["skipped_successful"], 2)
            self.assertEqual(len(calls), 2)
            score_simple = read_jsonl(output / Path(SCORE_FILE).stem / "simple.jsonl")[0]
            self.assertEqual(score_simple["reference_score"], 3.5)
            self.assertEqual(score_simple["signed_error"], 0.5)
            self.assertTrue(read_jsonl(output / Path(SCORE_FILE).stem / "full_results.jsonl")[-1]["reanalysis_only"])

            rows[CHOICE_FILE][0] = choice_row(first="Edited joke")
            self.write_inputs(source, rows)
            changed = run_all(source, output, "test-key", (CHOICE_FILE, SCORE_FILE),
                              batch_size=2, transport=transport)
            self.assertEqual(changed["completed_now"], 1)
            self.assertEqual(changed["skipped_successful"], 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(read_jsonl(output / Path(CHOICE_FILE).stem / "full_results.jsonl")), 2)

    def test_failure_does_not_discard_other_items_and_resume_retries_only_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source, output = root / "input", root / "output"
            source.mkdir()
            rows = [choice_row(f"choice-{index}", str(index)) for index in range(3)]
            self.write_inputs(source, {CHOICE_FILE: rows})
            seen = []

            def failing(request, timeout):
                payload = json.loads(request.data)
                text = payload["questions"]["judgment"]["criteria"]["A"]
                seen.append(text)
                if text == "1":
                    return 400, b'{"error":"bad request"}', {}
                return 200, json.dumps(response_for(payload)).encode("utf-8"), {}

            with self.assertRaises(RuntimeError):
                run_all(source, output, "test-key", (CHOICE_FILE,), batch_size=3,
                        max_attempts=1, transport=failing)
            self.assertCountEqual(seen, ["0", "1", "2"])
            self.assertEqual(len(read_jsonl(output / Path(CHOICE_FILE).stem / "full_results.jsonl")), 3)

            retried = []

            def success(request, timeout):
                payload = json.loads(request.data)
                retried.append(payload["questions"]["judgment"]["criteria"]["A"])
                return 200, json.dumps(response_for(payload)).encode("utf-8"), {}

            summary = run_all(source, output, "test-key", (CHOICE_FILE,), batch_size=3,
                              max_attempts=1, transport=success)
            self.assertEqual(retried, ["1"])
            self.assertEqual(summary["skipped_successful"], 2)
            self.assertEqual(summary["failed"], 0)


if __name__ == "__main__":
    unittest.main()
