"""Offline release-schema, source-integrity, and reconstruction contracts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_subjective_study_datasets as prep
import recorded_results as release
from test_subjective_baseline_runner import choice_row, score_row, CHOICE_FILE, SCORE_FILE


class RecordedReleaseChecks(unittest.TestCase):
    def observation(self, score=False):
        pred = ({"api_score": 3.0, "probabilities": {str(i): float(i == 3) for i in range(5)}, "confidence": 1.0}
                if score else {"choice": "A", "probabilities": {"A": .8, "B": .2}, "confidence": .6})
        return {"id": "a" * 64, "request_sha256": "b" * 64, "prediction": pred,
                "response_model": "test-model", "usage": {"input_tokens": 10, "cost": .001}}

    def test_allowlist_discards_content_annotations_and_credentials(self):
        raw = {**self.observation()["prediction"], "response_model": "test-model",
               "source_record": {"text": "private", "reference": "A"},
               "error": "private", "headers": {"Authorization": "private"},
               "usage": {"cost": .001, "account": "private"}}
        safe = release.safe_observation(raw)
        self.assertEqual(set(safe), {"prediction", "usage", "response_model"})
        self.assertEqual(safe["usage"], {"cost": .001})
        self.assertNotIn("private", json.dumps(safe))

    def test_rejects_extra_record_and_snapshot_fields(self):
        for target in ("record", "snapshot"):
            record = self.observation()
            record["baseline_snapshot"] = {
                key: record[key] for key in ("prediction", "usage", "response_model")}
            (record if target == "record" else record["baseline_snapshot"])["source_text"] = "private"
            with self.assertRaises(ValueError):
                release.validate_record(record, "choice", ["test-model"], True)

    def test_rejects_nonnumeric_usage_and_unexpected_alias(self):
        record = self.observation()
        record["usage"]["cost"] = "private"
        with self.assertRaises(ValueError):
            release.validate_record(record, "choice", ["test-model"], False)
        with self.assertRaises(ValueError):
            release.validate_record(self.observation(), "choice", ["other-model"], False)

    def test_baseline_and_position_swap_round_trip(self):
        row = choice_row("a" * 64)
        observation = self.observation()
        for condition in ("baseline", "pair_exchange_positions", "pair_exchange_labels"):
            payload, _ = release.payload_for("Jev", CHOICE_FILE, row, condition)
            record, rebuilt = release.restore_record("Jev", CHOICE_FILE, condition, row, observation, release.baseline.digest(payload))
            self.assertEqual(rebuilt, payload)
            self.assertEqual(release.prediction(record["parsed"]), observation["prediction"])
            self.assertEqual(record["source_record"]["reference"], row["reference"])
            self.assertNotIn("raw_response", record)

    def test_score_and_reverse_anchor_round_trip(self):
        row, observation = score_row("a" * 64), self.observation(True)
        for condition in ("baseline", "score_anchor_reverse"):
            payload, _ = release.payload_for("Jev", SCORE_FILE, row, condition)
            record, _ = release.restore_record("Jev", SCORE_FILE, condition, row, observation, release.baseline.digest(payload))
            self.assertEqual(release.prediction(record["parsed"]), observation["prediction"])
            if condition == "score_anchor_reverse":
                self.assertEqual(record["parsed"]["normalized_quality"], .25)

    def test_changed_reconstructed_request_is_rejected(self):
        with self.assertRaises(ValueError):
            release.restore_record("Jev", CHOICE_FILE, "baseline", choice_row("a" * 64), self.observation(), "b" * 64)

    def test_existing_source_is_checked_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance").mkdir()
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            frozen_hash = prep.sha256_file(source)
            (root / "provenance/dataset_manifest.json").write_text(json.dumps({"source_manifest": {"files": {"source.txt": frozen_hash}}}), encoding="utf-8")
            with patch.object(prep, "ROOT", root):
                prep.ensure_download("unused", source, False)
                source.write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    prep.ensure_download("unused", source, False)

    def test_recovered_article_set_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance").mkdir()
            story_dir = root / "stories/cnndm/cnn/stories"
            story_dir.mkdir(parents=True)
            story = story_dir / "example.story"
            story.write_text("frozen", encoding="utf-8")
            manifest = {"source_manifest": {"files": {}, "summeval_story_files": 1,
                        "summeval_story_set_sha256": prep.stable_id(prep.sha256_file(story))}}
            (root / "provenance/dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(prep, "ROOT", root), patch.object(prep, "SUMMEVAL_STORY_DIR", root / "stories"):
                prep.verify_frozen_sources()
                story.write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    prep.verify_frozen_sources()

    def test_comparison_fails_for_real_numeric_or_structural_changes(self):
        release.compare_numbers({"x": [1.0]}, {"x": [1.0 + 1e-13]})
        for changed in ({"x": [1.01]}, {"y": [1.]}, {"x": []}):
            with self.assertRaises(ValueError):
                release.compare_numbers({"x": [1.]}, changed)


if __name__ == "__main__":
    unittest.main()
