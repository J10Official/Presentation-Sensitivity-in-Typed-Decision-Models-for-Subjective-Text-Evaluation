"""Run the six subjective-study baselines through Jev Choice or Score."""

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from threading import Lock
from uuid import uuid4

from openrouter_api import read_key, request_body, send_decision, utc_now
from study_paths import repo_path, relative_path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "typesafe/jev-1.13"
QUESTION_ID = "judgment"
INPUT_ROOT = ROOT / "prepared_data" / "subjective_study_1500"
OUTPUT_ROOT = ROOT / "experiment_results" / "subjective_baseline"
SIMPLE_SNAPSHOT_EVERY = 100
PROBABILITY_MASS_TOLERANCE = 0.01
CHOICE_CONSISTENCY_TOLERANCE = 0.01
SCORE_TOLERANCE_PER_LEVEL = 0.01
PERMANENT_HTTP_STATUSES = {400, 401, 402, 403, 404, 405, 422}


class PermanentRunError(RuntimeError):
    pass

DATASETS = {
    "ukp_humor_pairwise.jsonl": {
        "kind": "choice",
        "state": lambda x: "Two standalone short texts are being compared.",
        "instructions": lambda x: x["instruction"],
    },
    "humicroedit_scoring.jsonl": {
        "kind": "score",
        "state": lambda x: {"original_headline": x["context"], "edited_headline": x["text"]},
        "instructions": lambda x: "How funny is `edited_headline` as an edit of `original_headline`?",
        "criteria": ["Not funny.", "Slightly funny.", "Moderately funny.", "Funny."],
        "native_minimum": 0,
    },
    "helpsteer_pairwise.jsonl": {
        "kind": "choice",
        "state": lambda x: {"user_request": x["context"]},
        "instructions": lambda x: "Which option contains the more helpful response to `user_request`?",
    },
    "helpsteer_scoring.jsonl": {
        "kind": "score",
        "state": lambda x: {"user_request": x["context"], "response": x["text"]},
        "instructions": lambda x: "How helpful is `response` overall for `user_request`?",
        "criteria": [
            "Not useful or helpful at all; completely misses the essence of the user's request.",
            "Borderline unhelpful; mostly does not capture what the user wanted, although it may still be usable in a small way.",
            "Partially helpful; addresses the request but misses the overall goal in some way, so it does not fully satisfy the user.",
            "Mostly helpful; mainly aligned with the user's intent and largely satisfies the request, with some room for improvement.",
            "Extremely helpful; completely aligned with the spirit of the user's request.",
        ],
        "native_minimum": 0,
    },
    "openai_summary_pairwise.jsonl": {
        "kind": "choice",
        "state": lambda x: {"source_text": x["context"]},
        "instructions": lambda x: "Which option contains the better summary of `source_text`?",
    },
    "summeval_relevance_scoring.jsonl": {
        "kind": "score",
        "state": lambda x: {"source_article": x["context"], "summary": x["text"]},
        "instructions": lambda x: (
            "How relevant is `summary` to `source_article`? Relevance means selecting important content "
            "from the source. Penalize redundant or excess information."
        ),
        "criteria": [
            "Very low relevance.",
            "Low relevance.",
            "Moderate relevance.",
            "High relevance.",
            "Very high relevance.",
        ],
        "native_minimum": 1,
    },
}


def identity(value):
    return value


# Future interventions belong here as pure input-to-input functions. The runner's
# resume key includes the resulting request hash, so edited records are rerun while
# unchanged successful records are retained.
TRANSFORMS = {name: identity for name in DATASETS}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(json_bytes(value)).hexdigest()


def request_for(dataset_name, row, model=MODEL):
    config = DATASETS[dataset_name]
    item = TRANSFORMS[dataset_name](copy.deepcopy(row["input"]))
    if not isinstance(item, dict):
        raise ValueError(f"{dataset_name} transform must return an object")
    question = {"type": config["kind"], "instructions": config["instructions"](item)}
    if config["kind"] == "choice":
        options = item.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"{dataset_name}:{row.get('id')} needs at least two options")
        labels = [option.get("label") for option in options]
        texts = [option.get("text") for option in options]
        if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"{dataset_name}:{row.get('id')} has invalid or duplicate labels")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError(f"{dataset_name}:{row.get('id')} has an empty option")
        question["criteria"] = {option["label"]: option["text"] for option in options}
    else:
        scale = item.get("scale", {})
        minimum, maximum = scale.get("minimum"), scale.get("maximum")
        if minimum != config["native_minimum"] or maximum - minimum + 1 != len(config["criteria"]):
            raise ValueError(f"{dataset_name}:{row.get('id')} scale does not match its frozen rubric")
        question["criteria"] = list(config["criteria"])
    return {"model": model, "state": config["state"](item), "questions": {QUESTION_ID: question}}


def finite_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def valid_distribution(probabilities, keys):
    return (
        isinstance(probabilities, dict)
        and set(probabilities) == set(keys)
        and all(finite_probability(value) for value in probabilities.values())
        and abs(sum(probabilities.values()) - 1) <= PROBABILITY_MASS_TOLERANCE + 1e-9
    )


def choice_confidence(probabilities, labels):
    count = len(labels)
    return (max(probabilities[label] for label in labels) - 1 / count) / (1 - 1 / count)


def score_confidence(probabilities, level_count):
    values = [probabilities[str(index)] for index in range(level_count)]
    mode = values.index(max(values))
    spread = sum(probability * abs(index - mode) for index, probability in enumerate(values))
    uniform_spread = sum(abs(index - (level_count - 1) / 2) for index in range(level_count)) / level_count
    return max(0.0, 1 - spread / uniform_spread)


def parse_choice(payload, answer):
    labels = list(payload["questions"][QUESTION_ID]["criteria"])
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    distribution_valid = valid_distribution(probabilities, labels)
    valid = (
        answer.get("type") == "choice"
        and answer.get("choice") in labels
        and distribution_valid
        and probabilities[answer["choice"]] >= max(probabilities.values()) - CHOICE_CONSISTENCY_TOLERANCE - 1e-9
        and finite_probability(confidence)
    )
    if not valid:
        return None, {"kind": "invalid_answer", "message": "Expected a valid Choice answer, full distribution, and confidence."}
    return {
        "choice": answer["choice"],
        "probabilities": probabilities,
        "confidence": confidence,
        "probability_derived_confidence": choice_confidence(probabilities, labels),
        "option_order": labels,
        "selected_position": labels.index(answer["choice"]) + 1,
    }, None


def parse_score(dataset_name, payload, answer):
    config = DATASETS[dataset_name]
    criteria = payload["questions"][QUESTION_ID]["criteria"]
    levels = [str(index) for index in range(len(criteria))]
    probabilities, legend, score = answer.get("probabilities"), answer.get("legend"), answer.get("score")
    distribution_valid = valid_distribution(probabilities, levels)
    expected_score = (sum(index * probabilities[str(index)] for index in range(len(criteria)))
                      if distribution_valid else None)
    valid = (
        answer.get("type") == "score"
        and type(score) in (int, float)
        and math.isfinite(score)
        and 0 <= score <= len(criteria) - 1
        and distribution_valid
        and abs(score - expected_score) <= SCORE_TOLERANCE_PER_LEVEL * len(criteria) + 1e-9
        and isinstance(legend, dict)
        and legend == {str(index): text for index, text in enumerate(criteria)}
        and finite_probability(answer.get("confidence"))
    )
    if not valid:
        return None, {"kind": "invalid_answer", "message": "Expected a valid Score answer, matching legend, full distribution, and confidence."}
    offset = config["native_minimum"]
    return {
        "api_score": score,
        "native_score": score + offset,
        "legend": legend,
        "probabilities": probabilities,
        "native_probabilities": {str(index + offset): probabilities[str(index)] for index in range(len(criteria))},
        "confidence": answer["confidence"],
        "probability_derived_confidence": score_confidence(probabilities, len(criteria)),
    }, None


def result_for(dataset_name, row, payload, result):
    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    answers = response.get("answers") if isinstance(response.get("answers"), dict) else {}
    answer = answers.get(QUESTION_ID) if isinstance(answers.get(QUESTION_ID), dict) else {}
    kind = DATASETS[dataset_name]["kind"]
    parsed, parse_error = (parse_choice(payload, answer) if kind == "choice"
                           else parse_score(dataset_name, payload, answer))
    error = result.get("error") if not result.get("ok") else parse_error
    reference = row["reference"]
    analysis = {}
    if parsed and kind == "choice":
        reference_choice = reference.get("choice")
        analysis = {
            "reference_choice": reference_choice,
            "agrees_with_reference": parsed["choice"] == reference_choice if reference_choice in parsed["option_order"] else None,
        }
    elif parsed:
        reference_score = reference.get("score")
        difference = (round(parsed["native_score"] - reference_score, 12)
                      if type(reference_score) in (int, float) else None)
        analysis = {
            "reference_score": reference_score,
            "signed_error": difference,
            "absolute_error": abs(difference) if difference is not None else None,
        }
    return {
        "ok": bool(result.get("ok") and parsed),
        "kind": kind,
        "model": payload["model"],
        "response_model": response.get("model"),
        "parsed": parsed,
        "analysis": analysis,
        "usage": response.get("usage"),
        "provider_response_id": response.get("id"),
        "provider": response.get("provider"),
        "status_code": result.get("status_code"),
        "http_attempts": result.get("attempts", 0),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "latency_seconds": result.get("latency_seconds"),
        "error": error,
        "raw_response": result.get("response"),
    }


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def read_jsonl(path, repair=False):
    if not path.exists():
        return []
    records, offset = [], 0
    mode = "rb+" if repair else "rb"
    with path.open(mode) as source:
        lines = source.readlines()
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                if not repair or index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError(f"Corrupt JSONL: {path}, line {index + 1}")
                source.truncate(offset)
                print(f"Recovered interrupted final write in {path}", flush=True)
                break
            offset += len(line)
        else:
            if repair and lines and not lines[-1].endswith(b"\n"):
                source.seek(0, 2)
                source.write(b"\n")
        if repair:
            source.flush()
            os.fsync(source.fileno())
    return records


def load_inputs(input_root, dataset_names, limit_per_dataset=None):
    loaded = {}
    for name in dataset_names:
        path = input_root / name
        rows = read_jsonl(path)
        if limit_per_dataset is not None:
            rows = rows[:limit_per_dataset]
        ids = [row.get("id") for row in rows]
        if any(not isinstance(ident, str) or not ident for ident in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Missing or duplicate IDs in {path}")
        for row in rows:
            if not all(key in row for key in ("dataset", "task", "input", "reference", "metadata")):
                raise ValueError(f"Incomplete prepared record {name}:{row.get('id')}")
            request_for(name, row)
        loaded[name] = rows
    return loaded


def dataset_directory(output_root, dataset_name):
    return output_root / Path(dataset_name).stem


def current_history(output_root, dataset_name, rows, model):
    path = dataset_directory(output_root, dataset_name) / "full_results.jsonl"
    history = read_jsonl(path, repair=True)
    current = {}
    for record in history:
        if not all(key in record for key in ("id", "request_sha256", "item_attempt", "ok")):
            raise ValueError(f"Invalid checkpoint in {path}")
        current[(record["id"], record["request_sha256"])] = record
    requests = {row["id"]: request_for(dataset_name, row, model) for row in rows}
    latest = {}
    for row in rows:
        payload = requests[row["id"]]
        record = current.get((row["id"], digest(payload)))
        source_sha = digest(row)
        if record is not None and record.get("source_record_sha256") != source_sha and record.get("raw_response"):
            replay = {
                "ok": 200 <= (record.get("status_code") or 0) < 300,
                "response": record["raw_response"],
                "status_code": record.get("status_code"),
                "attempts": record.get("http_attempts", 0),
                "error": record.get("error"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "latency_seconds": record.get("latency_seconds"),
            }
            refreshed = {
                **record,
                "source_record_sha256": source_sha,
                "source_record": row,
                **result_for(dataset_name, row, payload, replay),
                "reanalysis_only": True,
                "reanalyzed_at": utc_now(),
            }
            append_jsonl(path, refreshed)
            history.append(refreshed)
            record = refreshed
        if record is not None:
            latest[row["id"]] = record
    return history, latest, requests


def simple_result(record):
    row, parsed = record["source_record"], record.get("parsed") or {}
    simple = {
        "id": record["id"],
        "dataset": row["dataset"],
        "task": row["task"],
        "kind": record["kind"],
        "ok": record["ok"],
        "response_model": record["response_model"],
        "confidence": parsed.get("confidence"),
        "probability_derived_confidence": parsed.get("probability_derived_confidence"),
        "agreement_group": row["metadata"].get("agreement_group"),
        "annotation_depth": row["metadata"].get("annotation_depth"),
        "usage": record.get("usage"),
        "error": record.get("error"),
    }
    if record["kind"] == "choice":
        simple.update({key: parsed.get(key) for key in
                       ("choice", "probabilities", "option_order", "selected_position")})
        simple.update(record.get("analysis", {}))
    else:
        simple.update({key: parsed.get(key) for key in
                       ("api_score", "native_score", "probabilities", "native_probabilities")})
        simple.update(record.get("analysis", {}))
    return simple


def write_ordered_simple(output_root, dataset_name, rows, latest):
    path = dataset_directory(output_root, dataset_name) / "simple.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            if row["id"] in latest:
                target.write(json.dumps(simple_result(latest[row["id"]]), ensure_ascii=False, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def write_previews(input_root, output_root, dataset_names, model):
    rows_by_dataset = load_inputs(input_root, dataset_names, 1)
    path = output_root / "request_previews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for name in dataset_names:
            row = rows_by_dataset[name][0]
            payload = request_for(name, row, model)
            target.write(json.dumps({
                "dataset_file": name,
                "id": row["id"],
                "request_sha256": digest(payload),
                "request_body_utf8": request_body(payload).decode("utf-8"),
                "request": payload,
            }, ensure_ascii=False, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    return path


def run_all(input_root, output_root, api_key, dataset_names=tuple(DATASETS), model=MODEL,
            limit_per_dataset=None, batch_size=10, max_attempts=6, timeout=60, transport=None):
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY; no requests sent.")
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    if max_attempts < 1 or timeout <= 0:
        raise ValueError("max_attempts and timeout must be positive")
    rows_by_dataset = load_inputs(input_root, dataset_names, limit_per_dataset)
    histories, latest, requests = {}, {}, {}
    for name, rows in rows_by_dataset.items():
        histories[name], latest[name], requests[name] = current_history(output_root, name, rows, model)
        write_ordered_simple(output_root, name, rows, latest[name])

    run_id, lock = str(uuid4()), Lock()
    per_dataset = [[(name, row) for row in rows_by_dataset[name]] for name in dataset_names]
    all_items = [item for group in zip_longest(*per_dataset) for item in group if item is not None]
    pending = [(name, row) for name, row in all_items
               if not latest[name].get(row["id"], {}).get("ok")]
    start_event = {
        "event": "run_start", "timestamp": utc_now(), "run_id": run_id, "model": model,
        "dataset_files": list(dataset_names), "limit_per_dataset": limit_per_dataset,
        "batch_size": batch_size, "max_attempts": max_attempts, "timeout_seconds": timeout,
        "selected_items": len(all_items), "pending_items": len(pending),
    }
    run_log = (dataset_directory(output_root, dataset_names[0]) / "runs.jsonl"
               if len(dataset_names) == 1 else output_root / "runs.jsonl")
    append_jsonl(run_log, start_event)

    attempts_by_item = {}
    for name in dataset_names:
        for record in histories[name]:
            attempts_by_item[(name, record["id"])] = max(
                attempts_by_item.get((name, record["id"]), 0), record["item_attempt"])

    def judge(item):
        name, row = item
        payload = requests[name][row["id"]]
        request_sha = digest(payload)
        context = {
            "run_id": run_id,
            "dataset_file": name,
            "id": row["id"],
            "item_attempt": attempts_by_item.get((name, row["id"]), 0) + 1,
            "source_record_sha256": digest(row),
            "request_sha256": request_sha,
        }
        event_path = dataset_directory(output_root, name) / "events.jsonl"

        def log_event(event):
            with lock:
                append_jsonl(event_path, {**context, **event})

        kwargs = {"max_attempts": max_attempts, "timeout": timeout, "log_event": log_event}
        if transport is not None:
            kwargs["transport"] = transport
        try:
            result = send_decision(payload, api_key, **kwargs)
        except Exception as exc:
            result = {"ok": False, "response": None, "status_code": None, "attempts": 0,
                      "error": {"kind": "worker_error", "message": repr(exc)},
                      "started_at": None, "finished_at": utc_now(), "latency_seconds": None}
            log_event({"event": "worker_error", "timestamp": utc_now(), "error": result["error"]})
        record = {
            **context,
            "request_body_utf8": request_body(payload).decode("utf-8"),
            "request": payload,
            "source_record": row,
            **result_for(name, row, payload, result),
        }
        with lock:
            append_jsonl(dataset_directory(output_root, name) / "full_results.jsonl", record)
        return name, record

    completed_now, dirty_datasets = 0, set()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for start in range(0, len(pending), batch_size):
            futures = [pool.submit(judge, item) for item in pending[start:start + batch_size]]
            batch_records = [future.result() for future in as_completed(futures)]
            for name, record in batch_records:
                latest[name][record["id"]] = record
                completed_now += int(record["ok"])
                dirty_datasets.add(name)
            done = min(start + batch_size, len(pending))
            if done % SIMPLE_SNAPSHOT_EVERY < batch_size or done == len(pending):
                for name in dirty_datasets:
                    write_ordered_simple(output_root, name, rows_by_dataset[name], latest[name])
                dirty_datasets.clear()
            failures = sum(not record["ok"] for _, record in batch_records)
            print(f"Completed {done}/{len(pending)} pending requests; batch failures: {failures}", flush=True)

    failed = [(name, row["id"]) for name, row in all_items
              if not latest[name].get(row["id"], {}).get("ok")]
    summary = {
        "event": "run_end", "timestamp": utc_now(), "run_id": run_id,
        "selected": len(all_items), "completed_now": completed_now,
        "skipped_successful": len(all_items) - len(pending), "failed": len(failed),
        "failed_items": [{"dataset_file": name, "id": ident} for name, ident in failed],
    }
    append_jsonl(run_log, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if failed:
        permanent = [
            (name, ident) for name, ident in failed
            if latest[name][ident].get("status_code") in PERMANENT_HTTP_STATUSES
        ]
        if permanent:
            raise PermanentRunError(
                f"{len(permanent)} items received permanent HTTP errors; check route, key, credits, and payload."
            )
        raise RuntimeError(f"{len(failed)} items failed; saved all results. Rerun the same command to retry them.")
    return summary


def parse_dataset_names(value):
    if value == "all":
        return tuple(DATASETS)
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in names if name not in DATASETS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"Unknown dataset file(s): {', '.join(unknown) or value}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Allow billable API requests")
    mode.add_argument("--preview", action="store_true", help="Write one exact request preview per selected dataset; send nothing")
    parser.add_argument("--input-root", type=repo_path, default=relative_path(INPUT_ROOT))
    parser.add_argument("--output-root", type=repo_path, default=relative_path(OUTPUT_ROOT))
    parser.add_argument("--datasets", type=parse_dataset_names, default=tuple(DATASETS),
                        help="all or comma-separated prepared JSONL filenames")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Maximum parallel requests and checkpoint batch size (default: 10)")
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if args.preview:
        path = write_previews(args.input_root, args.output_root, args.datasets, args.model)
        print(relative_path(path), flush=True)
        return
    try:
        run_all(args.input_root, args.output_root, read_key(ROOT / ".env"), args.datasets, args.model,
                args.limit_per_dataset, args.batch_size, args.max_attempts, args.timeout)
    except PermanentRunError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
