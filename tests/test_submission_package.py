"""Offline checks for release paths, exclusions, and fresh-clone PDF auditing."""
import importlib.util
import json
import re
import unittest
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


class SubmissionPackageChecks(unittest.TestCase):
    def test_release_inventory_uses_contained_relative_paths(self):
        manifest = json.loads((ROOT / "provenance/package_manifest.json").read_text())
        for section in ("public_files", "local_files", "frozen_numeric_sha256"):
            for name in manifest[section]:
                posix = PurePosixPath(name.replace("\\", "/"))
                self.assertFalse(posix.is_absolute() or re.match(r"^[A-Za-z]:", name), name)
                self.assertNotIn("..", posix.parts, name)
                if section != "local_files":
                    self.assertTrue((ROOT / name).is_file(), name)

    def test_private_work_is_excluded_from_release(self):
        manifest = json.loads((ROOT / "provenance/package_manifest.json").read_text())
        private = {"scratch", "tmp", "pre-review", "old_results", "TablesExtended",
                   "datasets", "prepared_data", "experiment_results", "reproduction", ".git"}
        for name in manifest["public_files"]:
            self.assertNotIn(PurePosixPath(name).parts[0], private, name)
            self.assertNotIn(Path(name).name, {".env", "reviews.md", "AGENT_CONTEXT.md"})

    def test_conclusion_page_with_and_without_tex_auxiliary(self):
        spec = importlib.util.spec_from_file_location("paper_verify", ROOT / "scripts/verify_paper.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = "Earlier paragraph.\n\nA concluding sentence. Final sentence.\n\n\\label{sec:main_end}"
        pages = ["Introduction", "Final sentence."]
        self.assertEqual(module.conclusion_page(source, pages), 2)
        self.assertEqual(module.conclusion_page(source, pages, r"\newlabel{sec:main_end}{{7}{2}}"), 2)
        with self.assertRaises(AssertionError):
            module.conclusion_page(source, pages, r"\newlabel{sec:main_end}{{7}{3}}")
        with self.assertRaises(AssertionError):
            module.conclusion_page(source, ["Missing ending"])

    def test_analysis_scripts_and_paper_assets_have_clear_locations(self):
        for name in ("analysis_level1.py", "analysis_level2.py"):
            self.assertFalse((ROOT / name).exists())
            self.assertTrue((ROOT / "scripts" / name).is_file())
        self.assertFalse((ROOT / "VALIDATION_REPORT.md").exists())
        for directory in ("full_study_analysis", "validated_analysis"):
            self.assertTrue((ROOT / "reports" / directory).is_dir())
        manifest = json.loads((ROOT / "provenance/package_manifest.json").read_text())
        for name in manifest["public_files"]:
            if name.startswith("paper/"):
                self.assertIn(Path(name).suffix, {".tex", ".bib", ".sty", ".bst", ".pdf"}, name)

    def test_all_latex_inputs_and_figures_exist(self):
        paper = ROOT / "paper"
        imported = set()
        for source in paper.glob("*.tex"):
            text = source.read_text(encoding="utf-8")
            for name in re.findall(r"\\input\{([^}]+)\}", text):
                target = paper / (name if name.endswith(".tex") else name + ".tex")
                self.assertTrue(target.is_file(), name)
                imported.add(target.resolve())
            for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
                target = paper / "figures" / (name if name.endswith(".pdf") else name + ".pdf")
                self.assertTrue(target.is_file(), name)
        self.assertEqual({p.resolve() for p in (paper / "generated").glob("*.tex")},
                         {p for p in imported if p.parent == (paper / "generated").resolve()})


if __name__ == "__main__":
    unittest.main()
