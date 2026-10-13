"""Audit saved single-experiment results; --rebuild reproduces the legacy analysis separately.

Default: no valid empirical result is recomputed. Only the unsafe dyadic-IID
fallback is inspected from item arrays; affected results, if any, are repaired.
Dependencies: Python 3.11, NumPy, SciPy. All paths are repository-relative.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_full_subjective_study as old

OUT = ROOT / "reports/validated_analysis"
OLD = ROOT / "reports/full_study_analysis"
SEED = 20261012
PAIR_ENDPOINTS = {
    "Numeric 2 vs 1": ("pair_numeric_first_2", "pair_numeric_first_1"),
    "Human vs AI": ("pair_ai_human_first_human", "pair_ai_human_first_ai"),
    "Reader high vs low": ("pair_social_first_high", "pair_social_second_high"),
    "Prior selected vs rejected": ("pair_prior_judge_first_selected", "pair_prior_judge_second_selected"),
}
SCORE_ENDPOINTS = {
    "Identifier B vs A": ("score_identifier_b", "score_identifier_a"),
    "Numeric 2 vs 1": ("score_numeric_identifier_2", "score_numeric_identifier_1"),
    "Human vs AI": ("score_ai_human_human", "score_ai_human_ai"),
    "Reader high vs low": ("score_social_high", "score_social_low"),
    "Prior high vs low": ("score_prior_judge_high", "score_prior_judge_low"),
    "Directive vs inert": ("score_injection_directive", "score_injection_inert"),
    "Reverse vs native": ("score_anchor_reverse", "baseline"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def repo_path(relative):
    """Resolve recorded relative paths regardless of the collecting host OS."""
    return ROOT.joinpath(*str(relative).replace("\\", "/").split("/"))


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the frozen collection's serialized numerical artifacts across OSes.
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\r\n")


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def adjust_p(pvalues, arbitrary_dependence=False):
    """BH, or BY using the same pre-existing family under arbitrary dependence."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p, kind="stable")
    multiplier = sum(1 / i for i in range(1, n + 1)) if arbitrary_dependence else 1.0
    ranked = np.minimum.accumulate((p[order] * n * multiplier / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def streaming_baseline(model, name, rows):
    """Memory-only refactor of legacy earliest-started, hash-matching selection."""
    expected = {r["id"]: old.digest(old.baseline_request(name, r, old.MODELS[model][0])) for r in rows}
    selected = {}
    with (old.cell_path(model, name, "baseline") / "full_results.jsonl").open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            ident = record.get("id")
            if not record.get("ok") or record.get("request_sha256") != expected.get(ident):
                continue
            start = record.get("started_at") or ""
            if ident not in selected or start < selected[ident][0]:
                selected[ident] = (start, {**record["parsed"], "id": ident, "ok": True,
                                          "usage": record["usage"], "response_model": record["response_model"]})
    assert set(selected) == set(expected), (model, name, "canonical baseline incomplete")
    return [{**selected[r["id"]][1], "reference_choice": r["reference"].get("choice")} for r in rows]


def load_inputs(names=None):
    names = list(old.DATASETS) if names is None else names
    return {name: old.read_rows(old.INPUT / name) for name in names}


def load_cells(datasets):
    """Reuse frozen coordinate mapping and payload checks, without historical writes."""
    original = old.canonical_baseline
    old.canonical_baseline = streaming_baseline
    cells, audits, samples = {}, [], []
    try:
        for model in old.MODELS:
            for name, rows in datasets.items():
                conditions = old.PAIRWISE_CONDITIONS if old.DATASETS[name]["kind"] == "choice" else old.SCORING_CONDITIONS
                for condition in ("baseline",) + conditions:
                    cells[model, name, condition] = old.load_cell(model, name, condition, rows, audits, samples)
                print(f"Validated input mapping: {model} / {old.NAMES[name]}", flush=True)
    finally:
        old.canonical_baseline = original
    return cells, audits, samples


def inference(values, design, mask=None):
    """Paired mean with original variance convention; never fall back to IID dyads."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) if mask is None else np.asarray(mask, dtype=bool) & np.isfinite(values)
    x = values[mask]
    n = len(x)
    if n < 2:
        return {"n": n, "estimate": float(x.mean()) if n else None, "status": "insufficient_units"}
    mean = float(x.mean())
    residual = x - mean
    method = design["kind"]
    if method == "dyadic":
        sums = defaultdict(float)
        for pair, value in zip(np.asarray(design["nodes"])[mask], residual):
            for node in pair:
                sums[node] += value
        g = len(sums)
        variance = (sum(v * v for v in sums.values()) - residual @ residual) / n**2
        variance *= g / (g - 1) if g > 1 else 1
        if variance < 0:
            # Connected components retain all observed shared-joke dependence.
            node_rows = [{"metadata": {"candidate_ids": pair}} for pair in np.asarray(design["nodes"])[mask]]
            labels = np.asarray(old.graph_components(node_rows, labels=True))
            method = "connected_component_fallback"
        else:
            labels = None
    else:
        labels = np.asarray(design["labels"])[mask]
    if labels is not None:
        _, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse, weights=residual)
        g = len(sums)
        variance = float(sums @ sums) / n**2 * (g / (g - 1) if g > 1 else 1)
    if g < 2:
        return {"n": n, "estimate": mean, "clusters_or_nodes": g, "status": "insufficient_clusters"}
    df = g - 1
    se = math.sqrt(max(0, float(variance)))
    critical = float(stats.t.ppf(.975, df))
    test = mean / se if se else None
    p = float(2 * stats.t.sf(abs(test), df)) if se else None
    return {"n": n, "estimate": mean, "se": se, "variance": variance, "df": df,
            "statistic": test, "low": mean - critical * se, "high": mean + critical * se,
            "p": p, "clusters_or_nodes": g, "method": method,
            "status": "ok" if se else "zero_variation_descriptive"}


def effect_vectors(cells, model, name):
    """Existing signed contrasts, with linear condition coefficients for provenance."""
    choice = old.DATASETS[name]["kind"] == "choice"
    field = "pA" if choice else "y"
    specs = {}
    if choice:
        specs.update({
            "First-position": {"baseline": .5, "pair_exchange_labels": .5, "pair_exchange_positions": -.5, "pair_exchange_both": -.5},
            "B-label": {"baseline": -.5, "pair_exchange_labels": .5, "pair_exchange_positions": -.5, "pair_exchange_both": .5},
            "Interaction": {"baseline": -1, "pair_exchange_labels": 1, "pair_exchange_positions": 1, "pair_exchange_both": -1},
        })
        endpoints = dict(PAIR_ENDPOINTS)
        endpoints["Injection first"] = ("pair_injection_first_directive", "pair_injection_first_inert")
        endpoints["Injection second"] = ("pair_injection_second_inert", "pair_injection_second_directive")
    else:
        endpoints = SCORE_ENDPOINTS
    specs.update({label: {hi: 1, lo: -1} for label, (hi, lo) in endpoints.items()})
    return {label: {"values": sum(coef * cells[model, name, condition][field] for condition, coef in weights.items()),
                    "weights": weights} for label, weights in specs.items()}


def dyadic_vector(record, cells, name, rows):
    """Reconstruct only contrast vectors whose variance fallback code is suspect."""
    model, family, label = record["model"], record["family"], record["contrast"]
    gold, _, _ = old.reference_arrays(name, rows)
    strict = np.isin(gold, ["A", "B"])
    if family == "between_model_intervention":
        first, second = model.split(" minus ")
        return effect_vectors(cells, first, name)[label]["values"] - effect_vectors(cells, second, name)[label]["values"], None
    if family == "model_comparison":
        first, second = model.split(" minus ")
        return (cells[first, name, "baseline"]["choice"] == gold).astype(float) - (cells[second, name, "baseline"]["choice"] == gold), strict
    if family.startswith("presentation_"):
        if family.endswith("hard choice"):
            b, l, p, z = [(cells[model, name, c]["choice"] == "A").astype(float) for c in
                          ["baseline", "pair_exchange_labels", "pair_exchange_positions", "pair_exchange_both"]]
            return {"First-position": (b+l-p-z)/2, "B-label": (l+z-b-p)/2,
                    "Label × position interaction": l-b-z+p}[label], None
        label = "Interaction" if label == "Label × position interaction" else label
        return effect_vectors(cells, model, name)[label]["values"], None
    if family == "tie_placement":
        condition = label.split(" minus ")[0]
        return cells[model, name, condition]["pX"] - cells[model, name, "pair_tie_last"]["pX"], None
    if family == "rejection":
        return cells[model, name, "pair_rejection"]["pB"] - cells[model, name, "baseline"]["pA"], None
    if family == "injection":
        label = "Injection " + label.split()[0]
    else:
        label = {"numeric": "Numeric 2 vs 1", "ai_human": "Human vs AI", "social": "Reader high vs low",
                 "prior_judge": "Prior selected vs rejected"}[family]
    return effect_vectors(cells, model, name)[label]["values"], None


def synthetic_checks():
    x = np.array([.1, .3, .6, -.1, .2])
    result = inference(x, {"kind": "cluster", "labels": np.arange(5)})
    assert np.isclose(result["se"], np.std(x, ddof=1)/np.sqrt(5))
    assert result["df"] == 4
    np.testing.assert_allclose(adjust_p([.04, .01, .03, .9]), [.053333333333, .04, .053333333333, .9])
    # A single component must not acquire an invented IID standard error.
    cycle = {"kind": "dyadic", "nodes": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]}
    assert inference([1, -1, 1, -1], cycle)["status"] == "insufficient_clusters"
    for fn, units in [(old.alpha_nominal, [["A", "A"], ["B", "B"]]), (old.alpha_interval, [[0, 0], [2, 2]])]:
        assert fn(units) == 1


def audit_saved():
    assert (ROOT / "provenance/dataset_manifest.json").exists()
    tables, records, metadata = [read_json(OLD / file) for file in ["tables.json", "contrasts.json", "analysis_manifest.json"]]
    checks, failures, changes = [], [], []
    def check(label, passed, detail):
        checks.append({"check": label, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            failures.append(label)
    datasets = load_inputs()
    designs = {name: old.design_for(name, rows) for name, rows in datasets.items()}
    by_name = {old.NAMES[name]: name for name in datasets}
    check("saved table shape", all(len(row) == len(table["columns"]) for table in tables.values() for row in table["rows"]), f"{len(tables)} tables")
    check("saved analysis inventory", len(tables) == 55 and len(records) == 369, "55 tables; 369 primary recorded contrasts")
    check("unique contrast keys", len({tuple(r[k] for k in ["model", "dataset", "family", "contrast"]) for r in records}) == len(records), "No duplicate hypothesis rows")
    hashes = {**metadata["sha256"], **metadata["baseline_history_sha256"]}
    # Historical manifests preserve collection-time code lineage. Current
    # packaging-only code edits are anchored by the public release inventory.
    public_hashes = read_json(ROOT / "provenance/package_manifest.json")["public_files"]
    hashes = {path: public_hashes.get(path, expected) if path.startswith("scripts/") else expected
              for path, expected in hashes.items()}
    bad_hashes = [path for path, expected in hashes.items() if not repo_path(path).is_file() or sha256(repo_path(path)) != expected]
    check("source/code/results immutability", not bad_hashes, {"verified": len(hashes), "mismatches": bad_hashes})
    frozen = read_json(ROOT / "provenance/package_manifest.json")["frozen_numeric_sha256"]
    check("frozen numerical artifact immutability", all(sha256(repo_path(p)) == h for p, h in frozen.items()),
          "Original numerical outputs retained byte-for-byte; superseded narrative/planning files excluded")
    check("seed provenance", metadata["bootstrap_seed"] == 20261011 and metadata["bootstrap_repetitions"] == 2000,
          "Existing cluster-bootstrap seed 20261011 / 2000; new deterministic seed 20261012")
    audit_rows = read_json(OLD / "cell_audit.json")
    expected_cells = {(m, old.NAMES[n], c) for m in old.MODELS for n in old.DATASETS for c in
                      ("baseline",) + (old.PAIRWISE_CONDITIONS if old.DATASETS[n]["kind"] == "choice" else old.SCORING_CONDITIONS)}
    check("collection coverage", {tuple(r[:3]) for r in audit_rows} == expected_cells and all(r[3] == 1500 for r in audit_rows),
          f"{len(audit_rows)} cells / {sum(r[3] for r in audit_rows)} observations; repeats remain within item")
    count_errors = []
    for name, rows in datasets.items():
        ident = old.NAMES[name]
        source = next(r for r in tables["dataset_design"]["rows"] if r[0] == ident)
        expected = [len(rows), sum(r["metadata"]["agreement_group"] == "clear" for r in rows),
                    sum(r["metadata"]["agreement_group"] == "disputed" for r in rows)]
        if list(map(int, source[1:4])) != expected or len({r["id"] for r in rows}) != len(rows):
            count_errors.append(ident)
        if old.DATASETS[name]["kind"] == "choice":
            strict_n = sum(r["reference"]["choice"] in ["A", "B"] for r in rows)
            if any(int(r[2]) != strict_n for r in tables["baseline_pairwise"]["rows"] if r[1] == ident):
                count_errors.append(ident + " strict N")
    check("sample sizes and aggregation", not count_errors, count_errors or "1500 each; 750/750; strict denominators independently checked")
    enriched = []
    numerical_errors = []
    for record in records:
        r = dict(record)
        g = r["clusters_or_nodes"]
        df = max(g-1, 1)
        critical = stats.t.ppf(.975, df)
        se = (r["high"] - r["low"]) / (2 * critical)
        test = r["estimate"] / se if se else None
        expected_p = float(2 * stats.t.sf(abs(test), df)) if se else (1 if r["estimate"] == 0 else 0)
        if not np.isclose((r["high"]+r["low"])/2, r["estimate"], atol=2e-14):
            numerical_errors.append((r["model"], r["dataset"], r["contrast"], "CI center"))
        if not math.isclose(expected_p, r["p"], rel_tol=2e-8, abs_tol=1e-300):
            numerical_errors.append((r["model"], r["dataset"], r["contrast"], "CI/df/p disagreement"))
        name = by_name[r["dataset"]]
        rows = datasets[name]
        selected = np.ones(len(rows), dtype=bool)
        if r["family"] == "model_comparison" and r["contrast"] == "Baseline accuracy difference":
            selected = np.array([row["reference"]["choice"] in ["A", "B"] for row in rows])
        expected_n = int(selected.sum())
        design = designs[name]
        expected_g = (len(set(np.asarray(design["nodes"])[selected].ravel())) if design["kind"] == "dyadic"
                      else len(set(np.asarray(design["labels"])[selected])))
        if r["n"] != expected_n or g != expected_g or g < 2:
            numerical_errors.append((r["model"], r["dataset"], r["contrast"], "N/df"))
        r.update(se=float(se), variance=float(se**2), statistic=test, df=df, status="audited_existing",
                 se_provenance="recovered from full-precision CI and audited t df; not an empirical rerun")
        enriched.append(r)
    check("CI / SE / statistic / df / p consistency", not numerical_errors, numerical_errors or f"All {len(records)} full-precision contrasts agree; exact cluster/node counts independently recovered from prepared data")
    bh = adjust_p([r["p"] for r in records])
    check("existing BH arithmetic", np.allclose(bh, [r["q_bh_global"] for r in records], rtol=1e-12, atol=1e-300), "369 contrasts; ordering and monotonicity checked")
    # The code audit identifies an unsafe variance fallback, so inspect only the affected dataset.
    ukp = "ukp_humor_pairwise.jsonl"
    cells, _, _ = load_cells({ukp: datasets[ukp]})
    scanned, repaired = 0, 0
    for r in enriched:
        if r["dataset"] != "UKP humor":
            continue
        values, mask = dyadic_vector(r, cells, ukp, datasets[ukp])
        scanned += 1
        candidate = inference(values, designs[ukp], mask)
        if candidate.get("method") == "connected_component_fallback" or candidate["status"] == "insufficient_clusters":
            before = dict(r)
            for key in ["estimate", "low", "high", "p", "se", "variance", "df", "statistic", "clusters_or_nodes", "status"]:
                r[key] = candidate.get(key)
            r["status"] = "corrected_dyadic_fallback"
            changes.append({"before": before, "after": dict(r), "reason": "IID fallback would discard shared-joke dependence"})
            repaired += 1
    del cells
    check("unsafe dyadic fallback exposure", not any(r["p"] is None for r in enriched),
          f"Source branch inspected for {scanned} existing UKP contrasts; {repaired} affected results corrected using connected components")
    pvalues = [r["p"] for r in enriched]
    valid = np.array([p is not None for p in pvalues])
    adjusted_bh = adjust_p(np.asarray(pvalues, dtype=float)[valid])
    adjusted_by = adjust_p(np.asarray(pvalues, dtype=float)[valid], arbitrary_dependence=True)
    for index, qbh, qby in zip(np.flatnonzero(valid), adjusted_bh, adjusted_by):
        enriched[index].update(q_bh_validated=float(qbh), q_by_validated=float(qby))
    # Check independently printed primary effect sizes/intervals (rounding only tolerated).
    lookup = {tuple(r[k] for k in ["model", "dataset", "family", "contrast"]): r for r in records}
    printed = []
    for row in tables["factorial_effects"]["rows"]:
        printed.append(((row[0], row[1], "presentation_" + row[2], row[3]), row[4]))
    for table, family in [("pairwise_cue_effects", None), ("scoring_cue_effects", None)]:
        for row in tables[table]["rows"]:
            matched = [r for r in records if r["model"] == row[0] and r["dataset"] == row[1] and r["contrast"] == row[2]]
            assert len(matched) == 1
            r = matched[0]
            printed.append((tuple(r[k] for k in ["model", "dataset", "family", "contrast"]), row[3]))
    rounded_errors = []
    for key, text in printed:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        r = lookup[key]
        if len(numbers) != 3 or any(abs(float(a)-100*r[b]) > .000501 for a, b in zip(numbers, ["estimate", "low", "high"])):
            rounded_errors.append(key)
    check("printed effect/CI consistency", not rounded_errors, rounded_errors or f"{len(printed)} primary table intervals match full precision")
    synthetic_checks()
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_full_subjective_analysis.py"], cwd=ROOT, capture_output=True, text=True)
    check("independent synthetic/legacy unit tests", tests.returncode == 0, tests.stdout + tests.stderr)
    checks += [
        {"check": "outcome and test choice", "status": "PASS", "detail": "Paired bounded differences use dependence-aware mean t inference; correlations/alpha/distribution summaries remain descriptive. No independent-binomial treatment of repeat calls."},
        {"check": "multiplicity dependence guarantee", "status": "QUALIFIED", "detail": "Historical BH arithmetic passes, but positive-dependence assumptions are not established for overlapping contrasts. Retain BH and add conservative BY q-values for the same family; effect estimates unchanged."},
        {"check": "remaining independence limits", "status": "OPEN", "detail": "Shared workers, semantic dependence and condition-wave provider drift are not fully modelled; CIs are conditional on selected items and fixed human references."},
        {"check": "score causal interpretation", "status": "QUALIFIED", "detail": "Ordinal equal-spacing normalization is explicit; reversal also changes numeric polarity, and granularity variants change anchor/endpoint wording."},
        {"check": "reproduction refactor", "status": "PASS", "detail": "Legacy engine retained unchanged; canonical baseline loader only streams instead of retaining full payload bodies. Default audits; --rebuild explicitly reproduces original results into a separate folder."},
    ]
    result = {"checks": checks, "failures": failures, "changed_results": changes, "historical_bh_significant": int(sum(bh < .05)),
              "validated_bh_significant": int(sum(adjusted_bh < .05)), "dependency_safe_by_significant": int(sum(adjusted_by < .05)),
              "effects": len(enriched), "tables": len(tables), "dyadic_vectors_inspected": scanned,
              "seed": SEED, "historical_outputs_overwritten": False,
              "input_paths": {p.relative_to(ROOT).as_posix(): sha256(p) for p in [OLD / "tables.json", OLD / "contrasts.json", OLD / "analysis_manifest.json"]}}
    save_json(OUT / "level1_effects.json", enriched)
    save_json(OUT / "level1_audit.json", result)
    save_json(OUT / "corrections.json", changes)
    print(f"Level 1: {len(checks)} checks; {len(failures)} failures; {len(changes)} corrected effects; BY discoveries {sum(adjusted_by < .05)}/{len(enriched)}", flush=True)
    if failures:
        raise SystemExit("Level 1 checks failed: " + ", ".join(failures))


def rebuild():
    """Explicit opt-in full reproduction; not used in the audit completion pass."""
    old.OUT = OUT / "reproduced_legacy"
    old.OUT.mkdir(parents=True, exist_ok=True)
    old.REPORT = old.OUT / "FULL_STUDY_REPRODUCED.md"
    old.REPORT.write_text("# Reproduced single-experiment analyses\n", encoding="utf-8")
    old.INFERENCE.clear()
    old.TABLES.clear()
    sys.argv = [sys.argv[0], "--fresh"]
    old.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Opt-in full legacy reproduction into a separate output directory")
    args = parser.parse_args()
    if args.rebuild:
        rebuild()
    else:
        audit_saved()


if __name__ == "__main__":
    main()
