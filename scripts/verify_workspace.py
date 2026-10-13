"""Verify release hashes and frozen result inventories without network or model calls."""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_hashes(expected):
    for relative, value in expected.items():
        assert not PurePosixPath(relative.replace("\\", "/")).is_absolute() and not re.match(r"^[A-Za-z]:", relative), f"Absolute manifest path: {relative}"
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), f"Manifest path escapes repository: {relative}"
        assert path.is_file(), f"Missing: {relative}"
        assert digest(path) == value, f"Hash mismatch: {relative}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also verify the local, untracked data and recorded-response archive")
    args = parser.parse_args()
    manifest = read("provenance/package_manifest.json")
    check_hashes(manifest["public_files"])
    check_hashes(manifest["frozen_numeric_sha256"])
    effects = read("reports/validated_analysis/level1_effects.json")
    results = read("reports/validated_analysis/level2_results.json")
    assert len(effects) == 369
    assert len({tuple(r[k] for k in ("model", "dataset", "family", "contrast")) for r in effects}) == 369
    assert len(read("reports/full_study_analysis/tables.json")) == 55
    assert len(results["practical_impact"]) == 450
    assert len(results["shared_vulnerability"]["associations"]) == 648
    assert len(results["cross_task_meta"]) == 45
    for name in ("level1_audit.json", "level2_audit.json"):
        assert not read("reports/validated_analysis/" + name)["failures"], name
    assert not read("reports/validated_analysis/corrections.json")
    assert sum(r["q_by_validated"] < .05 for r in effects) == 296
    assert sum(r["pooled"]["q_by_level2"] < .05 for r in results["cross_task_meta"]) == 1
    source = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    assert "\\externaldocument" not in source
    assert source.index("\\bibliography{") < source.index("\\appendix") < source.index("\\input{appendix_results}")
    # Do not print any suspected secret values in an error message.
    for relative in manifest["public_files"]:
        path = ROOT / relative
        if path.suffix in {".py", ".md", ".json", ".txt", ".tex", ".bib"}:
            assert not re.search(rb"\bsk-or-v1-[A-Za-z0-9]{20,}", path.read_bytes()), f"Possible credential in {relative}"
            assert not re.search(rb"[A-Za-z]:[\\/](?:Users|home)[\\/]", path.read_bytes(), re.I), f"Machine-specific home path in {relative}"
    if args.full:
        check_hashes(manifest["local_files"])
        prepared = ROOT / "prepared_data/subjective_study_1500"
        for name, expected in read("provenance/dataset_manifest.json")["output_sha256"].items():
            assert digest(prepared / name) == expected, name
            if name.endswith(".jsonl"):
                rows = [json.loads(line) for line in (prepared / name).read_text(encoding="utf-8").splitlines()]
                assert len(rows) == len({r["id"] for r in rows}) == 1500, name
                assert sum(r["metadata"]["agreement_group"] == "clear" for r in rows) == 750, name
    print(f"PASS: {len(manifest['public_files'])} public-file hashes; frozen 369/450/648/45 inventories; zero corrected effects.")
    if args.full:
        print(f"PASS: {len(manifest['local_files'])} local archive hashes and six 1,500-item datasets.")


if __name__ == "__main__":
    main()
