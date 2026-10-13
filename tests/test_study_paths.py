"""Regression checks for relocated scripts and portable relative I/O paths."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper
import prepare_subjective_study_datasets as prep
import recorded_results
from study_paths import repo_path, relative_path
from run_subjective_interventions import worker_command
from test_subjective_baseline_runner import choice_row, score_row, CHOICE_FILE, SCORE_FILE


class PortablePaths(unittest.TestCase):
    def test_repository_paths_ignore_caller_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("os.getcwd", return_value=directory):
                self.assertEqual(repo_path("reproduction/prepared_data"), ROOT / "reproduction/prepared_data")
                self.assertEqual(repo_path(r"reproduction\prepared_data"), ROOT / "reproduction/prepared_data")
        self.assertEqual(relative_path(ROOT / "reports/validated_analysis"), "reports/validated_analysis")

    def test_absolute_cli_data_paths_are_rejected(self):
        for value in ("/tmp/data", r"Q:\absolute\data", r"\\host\share\data", "Q:relative-to-drive"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                repo_path(value)

    def test_worker_commands_use_relative_script_input_output_paths(self):
        args = argparse.Namespace(live=False, input_root=ROOT / "prepared_data/example",
            output_root=ROOT / "reproduction/previews", model="test", batch_size=2,
            max_attempts=1, timeout=3, limit_per_dataset=1)
        command = worker_command(args, "pair_exchange_labels", CHOICE_FILE)
        for value in (command[1], command[command.index("--input-root") + 1], command[command.index("--output-root") + 1]):
            self.assertFalse(Path(value).is_absolute() or PureWindowsPath(value).anchor)
        self.assertTrue((ROOT / command[1]).is_file())

    def test_real_preview_clis_work_from_foreign_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            for name, row in ((CHOICE_FILE, choice_row()), (SCORE_FILE, score_row())):
                (inputs / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
            base = [sys.executable, "-B", str(ROOT / "scripts/subjective_baseline_runner.py"), "--preview",
                    "--input-root", relative_path(inputs), "--output-root", relative_path(root / "baseline"),
                    "--datasets", CHOICE_FILE + "," + SCORE_FILE]
            intervention = [sys.executable, "-B", str(ROOT / "scripts/subjective_intervention_runner.py"), "--preview",
                    "--condition", "pair_exchange_labels", "--dataset", CHOICE_FILE,
                    "--input-root", relative_path(inputs), "--output-root", relative_path(root / "intervention")]
            for command in (base, intervention):
                result = subprocess.run(command, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn(str(ROOT), result.stdout)
            self.assertTrue(list((root / "baseline").rglob("*.jsonl")))
            self.assertTrue((root / "intervention/pair_exchange_labels/ukp_humor_pairwise/request_preview.json").is_file())

    def test_reconstruction_work_argument_is_repository_relative(self):
        with patch.object(sys, "argv", ["recorded_results.py", "reconstruct", "--work-dir", "reproduction/test-paths"]), patch.object(recorded_results, "reconstruct") as run:
            recorded_results.main()
            run.assert_called_once_with(ROOT / "reproduction/test-paths")

    def test_preparation_output_argument_is_repository_relative(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            target = Path(directory) / "prepared"
            with patch.object(sys, "argv", ["prepare.py", "--no-download", "--output-dir", relative_path(target)]), patch.object(prep, "download_sources", side_effect=RuntimeError("stop before downloading")):
                with self.assertRaisesRegex(RuntimeError, "stop before downloading"):
                    prep.main()
            self.assertTrue(target.is_dir())

    def test_paper_build_commands_use_paper_relative_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "paper-output"
            output.mkdir()
            (output / "manuscript.log").write_text("Clean build", encoding="utf-8")
            value = relative_path(output, ROOT / "paper")
            with patch.object(sys, "argv", ["build.py", "--output-directory", value]), patch.object(build_paper.shutil, "which", side_effect=lambda value: value), patch.object(build_paper.subprocess, "run", return_value=argparse.Namespace(stdout="TeX Live")), patch.object(build_paper, "run") as run:
                build_paper.main()
            calls = [call.args[0] for call in run.call_args_list]
            self.assertIn("-output-directory=" + value, calls[0])
            self.assertEqual(calls[1][1], value + "/manuscript")

    def test_active_artifact_manifest_paths_resolve_and_match(self):
        manifest = json.loads((ROOT / "reports/validated_analysis/manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["new_artifact_sha256"].items():
            self.assertFalse(Path(name).is_absolute() or PureWindowsPath(name).anchor)
            self.assertNotIn("\\", name)
            self.assertEqual(recorded_results.sha(ROOT / name), expected, name)

    def test_level2_audit_only_needs_no_private_files_or_writes(self):
        import analysis_level2 as level2
        with patch.object(sys, "argv", ["analysis_level2.py", "--audit-only"]), patch.object(level2, "write_manifest", side_effect=AssertionError("Unexpected manifest write")), patch.object(level2, "write_report", side_effect=AssertionError("Unexpected report write")), patch.object(level2.l1, "save_json", side_effect=AssertionError("Unexpected output write")):
            level2.main()
