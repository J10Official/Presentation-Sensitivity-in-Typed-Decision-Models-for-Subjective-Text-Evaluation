"""Export sanitized judgments, reconstruct analysis inputs, and reproduce results offline.

Run from the repository root. Source downloads belong to the preparation script;
this module never calls a model. Reconstructed checkpoints are compatibility
inputs, not original HTTP logs. Private histories are never overwritten.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import subjective_baseline_runner as baseline
import subjective_intervention_runner as intervention
from study_paths import repo_path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "recorded_results"
MODELS = {
    "Jev": ("typesafe/jev-1.13", "subjective_baseline", "subjective_interventions"),
    "Clef": ("cloudflare/clef-flash", "decision_model_comparisons/clef_flash/baseline",
             "decision_model_comparisons/clef_flash/interventions"),
    "Perplexity": ("perplexity/pplx-decider-v1.1-27b", "decision_model_comparisons/pplx_decider_v1_1_27b/baseline",
                  "decision_model_comparisons/pplx_decider_v1_1_27b/interventions"),
}
USAGE_FIELDS = {"input_tokens", "output_tokens", "cost"}
HASH = re.compile(r"[0-9a-f]{64}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def conditions(name):
    return ("baseline",) + (intervention.PAIRWISE_CONDITIONS if baseline.DATASETS[name]["kind"] == "choice"
                             else intervention.SCORING_CONDITIONS)


def payload_for(model, name, row, condition):
    route = MODELS[model][0]
    return (baseline.request_for(name, row, route), None) if condition == "baseline" else intervention.request_for(name, row, condition, route)


def cell_path(root, model, name, condition):
    _, base, inter = MODELS[model]
    return root / (base if condition == "baseline" else inter + "/" + condition) / Path(name).stem


def prediction(record):
    """Explicit allowlist: never copy text, gold, errors, headers or free-form notes."""
    keys = ("probabilities", "confidence", "choice" if "choice" in record else "api_score")
    return {key: record[key] for key in keys}


def usage(record):
    return {key: value for key, value in (record.get("usage") or {}).items()
            if key in USAGE_FIELDS and type(value) in (int, float) and math.isfinite(value)}


def safe_observation(record):
    return {"prediction": prediction(record), "usage": usage(record), "response_model": record["response_model"]}


def parse_prediction(name, condition, payload, context, pred):
    kind = baseline.DATASETS[name]["kind"]
    answer = {"type": kind, "confidence": pred["confidence"], "probabilities": pred["probabilities"]}
    if kind == "choice":
        answer["choice"] = pred["choice"]
        parsed, error = (baseline.parse_choice(payload, answer) if condition == "baseline"
                         else intervention.parse_choice(payload, answer))
    else:
        criteria = payload["questions"][baseline.QUESTION_ID]["criteria"]
        answer.update(score=pred["api_score"], legend={str(i): text for i, text in enumerate(criteria)})
        parsed, error = (baseline.parse_score(name, payload, answer) if condition == "baseline"
                         else intervention.parse_score(payload, answer, context))
    if error:
        raise ValueError(f"Invalid recorded prediction: {name}/{condition}")
    return parsed, answer


def export_archive():
    import prepare_subjective_study_datasets as prep
    prep.verify_frozen_sources()
    frozen = read(ROOT / "provenance/dataset_manifest.json")
    prepared = ROOT / "prepared_data/subjective_study_1500"
    for name, expected in frozen["output_sha256"].items():
        if sha(prepared / name) != expected:
            raise ValueError(f"Prepared source differs: {name}")
    cells, aliases = [], {model: set() for model in MODELS}
    for model in MODELS:
        for name in baseline.DATASETS:
            items = list(rows(prepared / name))
            for condition in conditions(name):
                directory = cell_path(ROOT / "experiment_results", model, name, condition)
                snapshots = list(rows(directory / "simple.jsonl"))
                if [r["id"] for r in snapshots] != [r["id"] for r in items] or not all(r["ok"] for r in snapshots):
                    raise ValueError("Incomplete or unordered snapshot")
                expected = {r["id"]: baseline.digest(payload_for(model, name, r, condition)[0]) for r in items}
                snapshot_by_id = {r["id"]: r for r in snapshots}
                selected = {}
                for record in rows(directory / "full_results.jsonl"):
                    ident = record.get("id")
                    if not record.get("ok") or record.get("request_sha256") != expected.get(ident):
                        continue
                    if condition == "baseline":
                        if ident in selected and (record.get("started_at") or "") >= selected[ident][0]:
                            continue
                    elif prediction(record["parsed"]) != prediction(snapshot_by_id[ident]):
                        continue
                    if baseline.digest(record["request"]) != expected[ident]:
                        raise ValueError("Stored request differs from its hash")
                    selected[ident] = (record.get("started_at") or "", {
                        **safe_observation({**record["parsed"], "usage": record.get("usage"), "response_model": record["response_model"]}),
                        "id": ident, "request_sha256": expected[ident],
                    })
                if set(selected) != set(expected):
                    raise ValueError(f"Missing matching history: {model}/{name}/{condition}")
                exported = []
                for item, snapshot in zip(items, snapshots):
                    record = selected[item["id"]][1]
                    # Non-baselines use the exact snapshot analyzed by the study.
                    if condition != "baseline":
                        record.update(safe_observation(snapshot))
                    else:
                        record["baseline_snapshot"] = safe_observation(snapshot)
                    payload, context = payload_for(model, name, item, condition)
                    parse_prediction(name, condition, payload, context, record["prediction"])
                    aliases[model].add(record["response_model"])
                    if condition == "baseline":
                        aliases[model].add(record["baseline_snapshot"]["response_model"])
                    exported.append(record)
                relative = f"{model.lower()}/{Path(name).stem}/{condition}.jsonl.gz"
                path = ARCHIVE / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                # No original filename or timestamp in the gzip header.
                with path.open("wb") as target:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                        for record in exported:
                            compressed.write((json.dumps(record, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))
                cells.append({"model": model, "dataset": name, "condition": condition,
                              "file": relative, "records": len(exported), "sha256": sha(path)})
            print(f"Exported {model} / {name}", flush=True)
    if any(len(values) != 1 for values in aliases.values()):
        raise ValueError("Unexpected served-model identifiers; review before release")
    save(ARCHIVE / "manifest.json", {
        "schema_version": 1, "cells": cells, "judgments": sum(c["records"] for c in cells),
        "served_models": {m: sorted(v) for m, v in aliases.items()},
        "dataset_manifest_sha256": sha(ROOT / "provenance/dataset_manifest.json"),
        "selection": "baseline earliest-started valid matching checkpoint; other conditions analyzed snapshot verified against matching history",
        "baseline_snapshots": "Separate later baseline snapshots preserve existing recovery-sensitivity diagnostics; not extra planned judgments",
        "omitted": ["source text", "human annotations", "timestamps", "headers", "errors", "provider response IDs", "run IDs", "local paths", "credentials"],
    })
    verify_archive()


def validate_record(record, kind, aliases, baseline_cell):
    required = {"id", "request_sha256", "response_model", "prediction", "usage"}
    if baseline_cell:
        required.add("baseline_snapshot")
    if set(record) != required or not HASH.fullmatch(record["id"]) or not HASH.fullmatch(record["request_sha256"]):
        raise ValueError("Unexpected archive schema or identifiers")
    for observation in [record] + ([record["baseline_snapshot"]] if baseline_cell else []):
        if observation is not record and set(observation) != {"response_model", "prediction", "usage"}:
            raise ValueError("Unexpected snapshot fields")
        pred = observation["prediction"]
        field = "choice" if kind == "choice" else "api_score"
        if set(pred) != {field, "probabilities", "confidence"} or observation["response_model"] not in aliases:
            raise ValueError("Unexpected prediction fields or model")
        if set(observation["usage"]) - USAGE_FIELDS:
            raise ValueError("Unexpected usage fields")
        values = [pred["confidence"], *pred["probabilities"].values(), *observation["usage"].values()]
        if kind == "score":
            values.append(pred["api_score"])
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Non-numeric recorded value")
        probabilities = pred["probabilities"]
        if (not 0 <= pred["confidence"] <= 1 or not probabilities or
                any(not 0 <= value <= 1 for value in probabilities.values()) or
                abs(sum(probabilities.values()) - 1) > baseline.PROBABILITY_MASS_TOLERANCE + 1e-9):
            raise ValueError("Invalid probability or confidence")
        if any(k not in {"A", "B", "X", "1", "2", "0", "3", "4", "5", "6"} for k in pred["probabilities"]):
            raise ValueError("Unexpected probability label")
        if kind == "choice" and pred["choice"] not in pred["probabilities"]:
            raise ValueError("Invalid choice")


def verify_archive():
    manifest = read(ARCHIVE / "manifest.json")
    expected = {(m, n, c) for m in MODELS for n in baseline.DATASETS for c in conditions(n)}
    if manifest["dataset_manifest_sha256"] != sha(ROOT / "provenance/dataset_manifest.json"):
        raise ValueError("Dataset manifest differs from recorded release")
    actual, files, total = set(), set(), 0
    for cell in manifest["cells"]:
        key = cell["model"], cell["dataset"], cell["condition"]
        relative = f"{cell['model'].lower()}/{Path(cell['dataset']).stem}/{cell['condition']}.jsonl.gz"
        if key not in expected or key in actual or cell["file"] != relative:
            raise ValueError("Invalid or duplicate archive cell")
        actual.add(key)
        path = ARCHIVE / relative
        if sha(path) != cell["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {relative}")
        ids = set()
        for record in rows(path):
            validate_record(record, baseline.DATASETS[cell["dataset"]]["kind"], manifest["served_models"][cell["model"]], cell["condition"] == "baseline")
            if record["id"] in ids:
                raise ValueError("Duplicate archived item")
            ids.add(record["id"])
        if len(ids) != cell["records"] or len(ids) != 1500:
            raise ValueError("Incomplete archive cell")
        total += len(ids)
        files.add(path.resolve())
    if actual != expected or total != manifest["judgments"] or total != 702000:
        raise ValueError("Incomplete study archive")
    if set(p.resolve() for p in ARCHIVE.rglob("*.gz")) != files:
        raise ValueError("Unlisted files in recorded archive")
    print(f"PASS: {len(actual)} sanitized cells / {total} judgments; schema and compressed hashes", flush=True)
    return manifest


def restore_record(model, name, condition, item, observation, request_sha):
    payload, context = payload_for(model, name, item, condition)
    if baseline.digest(payload) != request_sha:
        raise ValueError(f"Reconstructed request hash differs: {model}/{name}/{condition}")
    parsed, answer = parse_prediction(name, condition, payload, context, observation["prediction"])
    result = {"ok": True, "response": {"answers": {baseline.QUESTION_ID: answer},
              "model": observation["response_model"], "usage": observation["usage"]}}
    record = (baseline.result_for(name, item, payload, result) if condition == "baseline" else
              intervention.result_for(name, condition, item, payload, context, result))
    if not record["ok"] or prediction(record["parsed"]) != observation["prediction"]:
        raise ValueError("Prediction round-trip failed")
    # Reconstructed fields, not original HTTP response/metadata.
    record.pop("raw_response", None)
    record.update(id=item["id"], source_record=item, request_sha256=request_sha, parsed=parsed)
    return record, payload


def reconstruct(work):
    manifest = verify_archive()
    prepared = work / "prepared_data"
    frozen = read(ROOT / "provenance/dataset_manifest.json")["output_sha256"]
    for name, expected in frozen.items():
        if sha(prepared / name) != expected:
            raise ValueError(f"Prepared data not identical to study: {name}")
    datasets = {n: list(rows(prepared / n)) for n in baseline.DATASETS}
    for cell in manifest["cells"]:
        model, name, condition = (cell[k] for k in ("model", "dataset", "condition"))
        observations = list(rows(ARCHIVE / cell["file"]))
        if [r["id"] for r in observations] != [r["id"] for r in datasets[name]]:
            raise ValueError("Reconstructed sample IDs/order differ")
        simple, checkpoints = [], []
        for i, (item, observation) in enumerate(zip(datasets[name], observations)):
            record, payload = restore_record(model, name, condition, item, observation, observation["request_sha256"])
            if condition == "baseline" or i == 0:
                checkpoint = {k: record[k] for k in ("id", "ok", "parsed", "usage", "response_model", "request_sha256")}
                if i == 0:
                    checkpoint.update(request=payload, request_body_utf8=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                checkpoints.append(checkpoint)
            snapshot = record
            if condition == "baseline":
                snapshot, _ = restore_record(model, name, condition, item, observation["baseline_snapshot"], observation["request_sha256"])
            simple.append((baseline.simple_result if condition == "baseline" else intervention.simple_result)(snapshot))
        directory = cell_path(work / "experiment_results", model, name, condition)
        write_rows(directory / "simple.jsonl", simple)
        write_rows(directory / "full_results.jsonl", checkpoints)
    print("PASS: all 702,000 request hashes and prediction round-trips; references restored from official sources", flush=True)


def compare_numbers(left, right, path="root"):
    """Exact structure/strings; full-precision numeric comparison with tight tolerance."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"Result keys differ at {path}")
        for key in left:
            compare_numbers(left[key], right[key], path + "/" + key)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"Result length differs at {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            compare_numbers(a, b, path + f"/{i}")
    elif type(left) in (int, float) and type(right) in (int, float):
        if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Numerical result differs at {path}")
    elif left != right:
        raise ValueError(f"Result differs at {path}")


def reproduce(work):
    import analysis_level1 as l1
    import analysis_level2 as l2
    old = l1.old
    old.INPUT = work / "prepared_data"
    old.MODELS = {m: (route, work / "experiment_results" / base, work / "experiment_results" / inter)
                  for m, (route, base, inter) in MODELS.items()}
    old.OUT = work / "single_experiment"
    old.OUT.mkdir(parents=True, exist_ok=True)
    old.REPORT = old.OUT / "REPRODUCED.md"
    old.REPORT.write_text("# Reconstructed-output reproduction\n", encoding="utf-8")
    old.INFERENCE.clear()
    old.TABLES.clear()
    old.VARIANCE_FALLBACKS.clear()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        old.main()
    finally:
        sys.argv = original_argv
    compare_numbers(old.TABLES, read(ROOT / "reports/full_study_analysis/tables.json"))
    compare_numbers(old.INFERENCE, read(ROOT / "reports/full_study_analysis/contrasts.json"))
    print("PASS: all 55 tables and 369 primary contrasts reproduce", flush=True)
    # Frozen enriched SE/df/BY fields have an audited deterministic relationship
    # to the now-reproduced CI/p values. Check it again before pooling.
    effects = read(ROOT / "reports/validated_analysis/level1_effects.json")
    by_key = {tuple(r[k] for k in ("model", "dataset", "family", "contrast")): r for r in old.INFERENCE}
    for effect in effects:
        source = by_key[tuple(effect[k] for k in ("model", "dataset", "family", "contrast"))]
        for key in ("estimate", "low", "high", "p", "n", "clusters_or_nodes"):
            compare_numbers(effect[key], source[key], key)
        se = (source["high"] - source["low"]) / (2 * l1.stats.t.ppf(.975, effect["df"]))
        compare_numbers(effect["variance"], se * se, "effect variance")
    datasets = l1.load_inputs()
    cells, audits, samples = l1.load_cells(datasets)
    l2.synthetic_checks()
    print("Reproducing 450 impacts and 648 susceptibility associations", flush=True)
    impact = l2.practical_impact(datasets, cells)
    vulnerability = l2.shared_vulnerability(datasets, cells)
    pools = l2.cross_task_meta(effects)
    audit = l2.audit_results(impact, vulnerability, pools, effects)
    if audit["failures"]:
        raise ValueError("Reproduced level-2 audit failed")
    results = {"practical_impact": impact, "shared_vulnerability": vulnerability, "cross_task_meta": pools}
    # Use the analysis serializer to normalize NumPy/nonfinite values exactly as
    # in the original analysis, then compare against the frozen publication data.
    l1.save_json(work / "level2_results.json", results)
    compare_numbers(read(work / "level2_results.json"), read(ROOT / "reports/validated_analysis/level2_results.json"))
    save(work / "REPRODUCTION_VALIDATION.json", {
        "status": "PASS", "cells": len(cells), "judgments": len(cells) * 1500,
        "tables": len(old.TABLES), "primary_contrasts": len(old.INFERENCE),
        "practical_impacts": len(impact), "susceptibility_associations": len(vulnerability["associations"]),
        "meta_analyses": len(pools), "level2_audit": audit,
        "numeric_tolerance": {"relative": 1e-10, "absolute": 1e-12},
        "new_model_calls": 0, "private_history_required": False,
        "archive_manifest_sha256": sha(ARCHIVE / "manifest.json"),
    })
    print("PASS: complete offline reproduction, including 450/648/45 secondary results", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify", "reconstruct", "reproduce"))
    parser.add_argument("--work-dir", type=repo_path, default="reproduction")
    args = parser.parse_args()
    if args.action == "export":
        export_archive()
    elif args.action == "verify":
        verify_archive()
    else:
        work = args.work_dir.resolve()
        if not work.is_relative_to(ROOT / "reproduction"):
            raise ValueError("Use the ignored reproduction/ directory or one of its subdirectories")
        reconstruct(work)
        if args.action == "reproduce":
            reproduce(work)


if __name__ == "__main__":
    main()
