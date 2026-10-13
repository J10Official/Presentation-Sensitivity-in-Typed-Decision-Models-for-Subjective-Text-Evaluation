"""Audit source-to-request-to-result mappings for the six subjective baselines.

This script makes no network or model calls. It independently rebuilds the human
references from the downloaded source files, checks all prepared records and all
outbound requests, exercises the response parsers, and writes a human-readable
report containing one complete real example per dataset.
"""

from __future__ import annotations

import copy
import csv
import gzip
import io
import json
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_subjective_study_datasets as prep  # noqa: E402
import subjective_baseline_runner as runner  # noqa: E402


INPUT_ROOT = ROOT / "prepared_data" / "subjective_study_1500"
REPORT_PATH = ROOT / "reports" / "SUBJECTIVE_BASELINE_MAPPING_AUDIT_2026-10-11.md"
JSON_PATH = ROOT / "reports" / "subjective_baseline_mapping_audit.json"

SPECS = {
    "ukp_humor_pairwise.jsonl": {
        "kind": "choice",
        "state": lambda item: "Two standalone short texts are being compared.",
        "instructions": lambda item: item["instruction"],
    },
    "humicroedit_scoring.jsonl": {
        "kind": "score",
        "state": lambda item: {"original_headline": item["context"], "edited_headline": item["text"]},
        "instructions": lambda item: "How funny is `edited_headline` as an edit of `original_headline`?",
        "criteria": ["Not funny.", "Slightly funny.", "Moderately funny.", "Funny."],
        "native_minimum": 0,
    },
    "helpsteer_pairwise.jsonl": {
        "kind": "choice",
        "state": lambda item: {"user_request": item["context"]},
        "instructions": lambda item: "Which option contains the more helpful response to `user_request`?",
    },
    "helpsteer_scoring.jsonl": {
        "kind": "score",
        "state": lambda item: {"user_request": item["context"], "response": item["text"]},
        "instructions": lambda item: "How helpful is `response` overall for `user_request`?",
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
        "state": lambda item: {"source_text": item["context"]},
        "instructions": lambda item: "Which option contains the better summary of `source_text`?",
    },
    "summeval_relevance_scoring.jsonl": {
        "kind": "score",
        "state": lambda item: {"source_article": item["context"], "summary": item["text"]},
        "instructions": lambda item: (
            "How relevant is `summary` to `source_article`? Relevance means selecting important content "
            "from the source. Penalize redundant or excess information."
        ),
        "criteria": [
            "Very low relevance.", "Low relevance.", "Moderate relevance.",
            "High relevance.", "Very high relevance.",
        ],
        "native_minimum": 1,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def expected_request(name: str, row: dict) -> dict:
    spec, item = SPECS[name], row["input"]
    question = {"type": spec["kind"], "instructions": spec["instructions"](item)}
    if spec["kind"] == "choice":
        question["criteria"] = {option["label"]: option["text"] for option in item["options"]}
    else:
        question["criteria"] = list(spec["criteria"])
    return {"model": runner.MODEL, "state": spec["state"](item),
            "questions": {runner.QUESTION_ID: question}}


def displayed_index(raw_candidates: list[str], displayed_text: str) -> int:
    matches = [index for index, text in enumerate(raw_candidates) if text == displayed_text]
    assert len(matches) == 1
    return matches[0]


def display_label(raw_index: int, order: list[int]) -> str:
    return "A" if order.index(raw_index) == 0 else "B"


def raw_ukp() -> dict[str, dict]:
    lines = prep.UKP_RESULTS.read_text(encoding="utf-8").splitlines()
    columns, groups = lines[0].split("\t"), defaultdict(list)
    for line in lines[1:]:
        row = dict(zip(columns, line.split("\t")))
        first, second = sorted((row["Text A"], row["Text B"]))
        answer = row["Answer"]
        choice = "X" if answer == "X" else int((row["Text A"] if answer == "A" else row["Text B"]) == second)
        groups[(first, second)].append((row["Worker ID"], choice))
    output = {}
    for candidates, annotations in groups.items():
        if len(annotations) == 5 and len({worker for worker, _ in annotations}) == 5:
            output[prep.stable_id("ukp", *candidates)] = {
                "candidates": list(candidates), "choices": [choice for _, choice in annotations]
            }
    return output


def raw_humicroedit() -> dict[str, dict]:
    output = {}
    with zipfile.ZipFile(prep.HUMICRO_ZIP) as archive:
        for split in ("train", "dev", "test"):
            member = f"semeval-2020-task-7-dataset/subtask-1/{split}.csv"
            for row in csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))):
                original = prep.re.sub(r"<(.*?)/>", r"\1", row["original"])
                edited = prep.re.sub(r"<.*?/>", lambda _: row["edit"], row["original"])
                ident = prep.stable_id("humicroedit", row["id"], edited)
                output[ident] = {"context": original, "text": edited,
                                 "ratings": [int(value) for value in row["grades"]]}
    return output


def raw_helpsteer_pairs() -> dict[str, dict]:
    output = {}
    for row in prep.jsonl(prep.HELPSTEER_DIR / "preference/preference.jsonl.gz"):
        ident = prep.stable_id("helpsteer-pair", row["prompt"], row["response_1"], row["response_2"])
        output[ident] = {
            "context": row["prompt"], "candidates": [row["response_1"], row["response_2"]],
            "strengths": [value["strength"] for value in row["all_preferences_unprocessed"]],
            "published": row["preference_strength"],
        }
    return output


def raw_helpsteer_scores() -> dict[str, dict]:
    disagreements = {(row["prompt"], row["response"]): row
                     for row in prep.jsonl(prep.HELPSTEER_DIR / "disagreements/disagreements.jsonl.gz")}
    output = {}
    for split in ("train", "validation"):
        for row in prep.jsonl(prep.HELPSTEER_DIR / f"{split}.jsonl.gz"):
            raw = disagreements.get((row["prompt"], row["response"]))
            if raw:
                ident = prep.stable_id("helpsteer-score", row["prompt"], row["response"])
                output[ident] = {"context": row["prompt"], "text": row["response"],
                                 "score": row["helpfulness"], "ratings": raw["helpfulness"]}
    return output


def raw_openai_pairs() -> dict[str, dict]:
    groups, contexts, candidates = {}, {}, {}
    for path in sorted(prep.OPENAI_DIR.glob("*.json")):
        for row in prep.jsonl(path):
            info = row["info"]
            context = info.get("post") or info.get("article")
            if not info.get("post") or not context or row.get("choice") not in (0, 1):
                continue
            first, second = [summary["text"] for summary in row["summaries"]]
            if not first.strip() or not second.strip() or first == second:
                continue
            low, high = sorted((first, second))
            context_id = prep.stable_id("openai-context", context)
            low_id, high_id = prep.stable_id("openai-summary", low), prep.stable_id("openai-summary", high)
            key = (context_id, low_id, high_id)
            contexts[context_id], candidates[low_id], candidates[high_id] = context, low, high
            group = groups.setdefault(key, {"workers": defaultdict(set), "titles": set()})
            chosen = (first, second)[row["choice"]]
            group["workers"][row["worker"]].add(int(chosen == high))
            group["titles"].add(info.get("title") or "")
    output = {}
    for (context_id, low_id, high_id), group in groups.items():
        votes = [next(iter(values)) for values in group["workers"].values() if len(values) == 1]
        title = sorted(group["titles"])[0]
        context = (title + "\n\n" if title else "") + contexts[context_id]
        ident = prep.stable_id("openai-pair", context_id, low_id, high_id)
        output[ident] = {"context": context, "candidates": [candidates[low_id], candidates[high_id]],
                         "choices": votes}
    return output


def raw_summeval() -> dict[str, dict]:
    rows = sorted(prep.jsonl(prep.SUMMEVAL_ANNOTATIONS), key=lambda row: (str(row["id"]), row["model_id"]))
    canonical = {}
    for row in rows:
        canonical.setdefault((str(row["id"]), row["decoded"]), row)
    output = {}
    for (article_id, decoded), row in canonical.items():
        article = prep.parse_story((prep.SUMMEVAL_STORY_DIR / row["filepath"]).read_text(encoding="utf-8"))
        expert = [annotation["relevance"] for annotation in row["expert_annotations"]]
        crowd = [annotation["relevance"] for annotation in row["turker_annotations"]]
        output[prep.stable_id("summeval", article_id, decoded)] = {
            "context": article, "text": decoded, "expert": expert, "crowd": crowd,
        }
    return output


def reference_from_votes(displayed: list[str]) -> str:
    counts = Counter(displayed)
    for label in ("A", "B"):
        if counts[label] > len(displayed) / 2:
            return label
    return "SPLIT"


def verify_choice(name: str, row: dict, raw: dict) -> None:
    options = row["input"]["options"]
    assert [option["label"] for option in options] == ["A", "B"]
    order = [displayed_index(raw["candidates"], option["text"]) for option in options]
    assert sorted(order) == [0, 1]
    assert row["metadata"]["orientation_swapped"] == (order == [1, 0])

    if name == "ukp_humor_pairwise.jsonl":
        displayed = [display_label(choice, order) if choice in (0, 1) else "X" for choice in raw["choices"]]
        counts = Counter(displayed)
        expected = "A" if counts["A"] >= 3 else "B" if counts["B"] >= 3 else "X" if counts["X"] >= 3 else "NO_MAJORITY"
    elif name == "helpsteer_pairwise.jsonl":
        source_choices = [1 if strength > 0 else 0 for strength in raw["strengths"]]
        displayed = [display_label(choice, order) for choice in source_choices]
        expected = (display_label(1, order) if raw["published"] > 0 else
                    display_label(0, order) if raw["published"] < 0 else "ABOUT_THE_SAME")
    else:
        assert row["input"]["context"] == raw["context"]
        displayed = [display_label(choice, order) for choice in raw["choices"]]
        expected = reference_from_votes(displayed)

    assert displayed == row["reference"]["individual_choices"]
    assert expected == row["reference"]["choice"]
    position = None if expected not in ("A", "B") else ["A", "B"].index(expected) + 1
    assert position == row["reference"]["winner_position"]


def verify_score(name: str, row: dict, raw: dict) -> None:
    assert row["input"]["context"] == raw["context"] and row["input"]["text"] == raw["text"]
    if name == "humicroedit_scoring.jsonl":
        ratings, score = raw["ratings"], statistics.mean(raw["ratings"])
    elif name == "helpsteer_scoring.jsonl":
        ratings, score = raw["ratings"], statistics.mean(raw["ratings"])
        assert row["metadata"]["published_helpfulness"] == raw["score"]
    else:
        ratings = raw["expert"] + raw["crowd"]
        score = statistics.mean(ratings)
        assert row["reference"]["expert_scores"] == raw["expert"]
        assert row["reference"]["crowd_scores"] == raw["crowd"]
    assert row["reference"]["individual_scores"] == ratings
    assert abs(row["reference"]["score"] - score) < 1e-12


def fake_result(response: dict) -> dict:
    return {"ok": True, "response": response, "status_code": 200, "attempts": 1,
            "error": None, "started_at": "SIMULATED", "finished_at": "SIMULATED",
            "latency_seconds": 0.0}


def simulated_response(name: str, payload: dict) -> dict:
    if name == "ukp_humor_pairwise.jsonl":
        answer = {"type": "choice", "choice": "A", "probabilities": {"A": 0.8, "B": 0.2}, "confidence": 0.6}
    elif name == "helpsteer_pairwise.jsonl":
        answer = {"type": "choice", "choice": "A", "probabilities": {"A": 0.65, "B": 0.35}, "confidence": 0.3}
    elif name == "openai_summary_pairwise.jsonl":
        answer = {"type": "choice", "choice": "B", "probabilities": {"A": 0.3, "B": 0.7}, "confidence": 0.4}
    else:
        criteria = payload["questions"][runner.QUESTION_ID]["criteria"]
        legend = {str(index): text for index, text in enumerate(criteria)}
        if name == "humicroedit_scoring.jsonl":
            probabilities, score = {"0": 0.8, "1": 0.2, "2": 0.0, "3": 0.0}, 0.2
        elif name == "helpsteer_scoring.jsonl":
            probabilities, score = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.2, "4": 0.8}, 3.8
        else:
            probabilities, score = {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.6, "4": 0.3}, 3.2
        confidence = runner.score_confidence(probabilities, len(criteria))
        answer = {"type": "score", "score": score, "legend": legend,
                  "probabilities": probabilities, "confidence": confidence}
    return {"id": "SIMULATED-NOT-SENT", "model": runner.MODEL, "provider": "illustrative-only",
            "answers": {runner.QUESTION_ID: answer},
            "usage": {"input_tokens": None, "output_tokens": None}}


def json_block(value: dict) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n```"


def mapping_note(name: str, row: dict) -> str:
    if name == "ukp_humor_pairwise.jsonl":
        return "Five source votes are remapped to displayed A/B (X is preserved); a candidate or X needs at least 3/5 votes, otherwise the reference is `NO_MAJORITY`."
    if name == "humicroedit_scoring.jsonl":
        return "Gold is the arithmetic mean of five native 0–3 humor grades. Jev API levels are also 0–3, so no offset is applied."
    if name == "helpsteer_pairwise.jsonl":
        return "The published preference-strength sign (aggregated upstream from the three most similar judgments) selects source response 1 or 2; that source response is then mapped to its displayed A/B position. Five unprocessed signs are retained as supporting individual choices."
    if name == "helpsteer_scoring.jsonl":
        return "Gold is the arithmetic mean of all five released helpfulness ratings on 0–4. The source's published integer—derived upstream from the three most-agreeing annotations and rounded—is retained separately in metadata."
    if name == "openai_summary_pairwise.jsonl":
        return "Gold is the strict majority of all retained unique-worker votes for the identical source/summary pair. Exact ties are `SPLIT`. Votes are remapped after display-order balancing."
    return "Gold is the equal-weight arithmetic mean of 3 expert and 5 crowd relevance scores on the native 1–5 scale. Jev's API index 0–4 is converted to native 1–5 by adding 1."


def main() -> None:
    rows_by_name = {name: read_jsonl(INPUT_ROOT / name) for name in SPECS}
    raw_by_name = {
        "ukp_humor_pairwise.jsonl": raw_ukp(),
        "humicroedit_scoring.jsonl": raw_humicroedit(),
        "helpsteer_pairwise.jsonl": raw_helpsteer_pairs(),
        "helpsteer_scoring.jsonl": raw_helpsteer_scores(),
        "openai_summary_pairwise.jsonl": raw_openai_pairs(),
        "summeval_relevance_scoring.jsonl": raw_summeval(),
    }
    summaries, examples = {}, {}
    for name, rows in rows_by_name.items():
        assert len(rows) == 1500 and len({row["id"] for row in rows}) == 1500
        spec, raw_map = SPECS[name], raw_by_name[name]
        for row in rows:
            assert row["id"] in raw_map
            payload = runner.request_for(name, row)
            assert payload == expected_request(name, row)
            changed = copy.deepcopy(row)
            changed["reference"] = {"deliberately": "changed"}
            changed["metadata"] = {"deliberately": "changed"}
            assert runner.request_body(runner.request_for(name, changed)) == runner.request_body(payload)
            if spec["kind"] == "choice":
                verify_choice(name, row, raw_map[row["id"]])
            else:
                verify_score(name, row, raw_map[row["id"]])
                scale = row["input"]["scale"]
                assert scale["minimum"] == spec["native_minimum"]
                assert scale["maximum"] - scale["minimum"] + 1 == len(spec["criteria"])

        if spec["kind"] == "choice":
            references = Counter(row["reference"]["choice"] for row in rows)
            strict = references["A"] + references["B"]
            detail = {"kind": "choice", "records": len(rows), "options": 2,
                      "reference_counts": dict(sorted(references.items())),
                      "binary_comparable": strict, "no_binary_gold": len(rows) - strict}
            eligible = [row for row in rows if row["reference"]["choice"] in ("A", "B")]
        else:
            scores = [row["reference"]["score"] for row in rows]
            detail = {"kind": "score", "records": len(rows), "levels": len(spec["criteria"]),
                      "api_scale": [0, len(spec["criteria"]) - 1],
                      "native_scale": [spec["native_minimum"], spec["native_minimum"] + len(spec["criteria"]) - 1],
                      "reference_min": min(scores), "reference_max": max(scores)}
            eligible = rows
        summaries[name] = detail
        examples[name] = min(eligible, key=lambda row: len(json.dumps(runner.request_for(name, row), ensure_ascii=False)))

    # Exercise every possible parser label/level independently of gold.
    for name, rows in rows_by_name.items():
        payload = runner.request_for(name, rows[0])
        question = payload["questions"][runner.QUESTION_ID]
        if SPECS[name]["kind"] == "choice":
            for label in question["criteria"]:
                probabilities = {candidate: float(candidate == label) for candidate in question["criteria"]}
                parsed, error = runner.parse_choice(payload, {"type": "choice", "choice": label,
                                                               "probabilities": probabilities, "confidence": 1.0})
                assert not error and parsed["choice"] == label
        else:
            for index in range(len(question["criteria"])):
                probabilities = {str(level): float(level == index) for level in range(len(question["criteria"]))}
                answer = {"type": "score", "score": float(index),
                          "legend": {str(level): text for level, text in enumerate(question["criteria"])},
                          "probabilities": probabilities, "confidence": 1.0}
                parsed, error = runner.parse_score(name, payload, answer)
                assert not error and parsed["native_score"] == index + SPECS[name]["native_minimum"]

    sections = []
    for name, row in examples.items():
        payload = runner.request_for(name, row)
        response = simulated_response(name, payload)
        compared = runner.result_for(name, row, payload, fake_result(response))
        comparison = {"parsed": compared["parsed"], "comparison_against_human_reference": compared["analysis"]}
        sections.append(
            f"## {name}\n\n"
            f"Mapping rule: {mapping_note(name, row)}\n\n"
            "### Complete prepared record (analysis side)\n\n"
            "Only its `input` object is used to form the request; `reference` and `metadata` are never sent.\n\n"
            f"{json_block(row)}\n\n"
            "### Exact outbound Jev request\n\n"
            "The object is shown pretty-printed; field and option order are preserved in the compact UTF-8 request body.\n\n"
            f"{json_block(payload)}\n\n"
            "### Illustrative response filled in for this audit\n\n"
            "This is schema-valid simulated output, **not an actual Jev response and not an API call**.\n\n"
            f"{json_block(response)}\n\n"
            "### Parsed result and comparison with gold\n\n"
            f"{json_block(comparison)}\n"
        )

    table = [
        "| Dataset file | Interface | Sent choices/levels | Native scale | Human comparator | No binary gold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, info in summaries.items():
        if info["kind"] == "choice":
            table.append(f"| `{name}` | Choice | {info['options']} | A/B | {info['binary_comparable']:,} | {info['no_binary_gold']:,} |")
        else:
            table.append(f"| `{name}` | Score | {info['levels']} | {info['native_scale'][0]}–{info['native_scale'][1]} | {info['records']:,} | n/a |")

    report = f"""# Subjective baseline mapping audit

**Audit date:** 11 October 2026  
**Scope:** all 9,000 prepared records; offline only; no OpenRouter/Jev requests made.

## Verdict

The current implementation passes a full source-to-request-to-result mapping audit. Every prepared record was matched back to the downloaded source annotation, every exact request was rebuilt independently, and every possible Choice label and Score level was round-tripped through the parser. Gold fields are excluded by construction: mutating all `reference` and `metadata` fields leaves every request byte-equivalent.

One issue was fixed during review: the HelpSteer Score rubric had improvised wording that mixed helpfulness with correctness/harmlessness. It now uses the official HelpSteer2 Appendix G 0–4 helpfulness anchors. Humicroedit uses its published 0–3 humor anchors. SummEval supplies the relevance definition and a 1–5 Likert scale but not separate prose for all five points, so the five ordered intensity labels are explicitly treated as a frozen API adaptation.

| Audit check | Result |
|---|---|
| Prepared record matches downloaded source | Pass, 9,000/9,000 |
| Request independently reconstructed from judge-visible input | Pass, 9,000/9,000 |
| Gold/metadata mutation changes request bytes | Pass: 0/9,000 changed |
| Choice source identity → displayed A/B → returned label | Pass for every pair and every parser label |
| Score API index → native score offset | Pass for every possible level in all three Score tasks |
| Returned choice, probability distribution, weighted score, legend, and confidence internally consistent | Enforced by parser and unit-tested |
| Interrupted-write recovery, per-item checkpoints, ordered simplified output, and resume-by-request-hash | Unit-tested |

{chr(10).join(table)}

For Choice tasks, A and B are **display labels**, not source candidate identities. The preparation script first balances winner position, then remaps every source vote and gold winner into the displayed labels. For Score tasks, Jev always returns zero-indexed criterion levels: native score = API score + 0 for Humicroedit/HelpSteer and API score + 1 for SummEval. A probabilistic Jev score can be continuous because it is the expectation over level probabilities.

Items whose source reference is `X`, `NO_MAJORITY`, `ABOUT_THE_SAME`, or `SPLIT` have no A/B gold under this two-option baseline. The runner correctly records `agrees_with_reference: null`; these items may inform choice proportions/uncertainty but must be excluded from binary human-alignment accuracy unless an explicit tie protocol is run.

Every call is an HTTP `POST` to `https://openrouter.ai/api/alpha/decisions` with `Content-Type: application/json` and a bearer token (redacted in logs). The body contains only `model`, `state`, and `questions`; no temperature, sampling, gold, reference, metadata, or analysis field is added. JSON bodies below are pretty-printed for inspection; the transport sends the same object as compact UTF-8 JSON. Exact compact bodies are also written by the runner to `request_body_utf8`.

## Prompt/rubric provenance

| Dataset | Instruction status | Criteria status |
|---|---|---|
| UKP Humor | Frozen study wording requested for Jev: “Which option contains the funnier text?” | Candidate texts are the two Choice criteria |
| Humicroedit | Frozen study wording identifies original versus edited headline | Published 0–3 humor anchors |
| HelpSteer2 pairwise | Frozen Jev adaptation of the holistic response-preference task | The two responses are the Choice criteria |
| HelpSteer2 Score | Frozen study wording for overall helpfulness | Official Appendix G 0–4 helpfulness constructs, ordered low to high |
| OpenAI summaries | Frozen generic adaptation; exact worker UI prompt is not stored in each released record | The two summaries are the Choice criteria |
| SummEval relevance | Frozen adaptation of the published relevance definition | Five ordered intensity labels are a disclosed API adaptation; native scale remains 1–5 |

## Gold construction and provenance

| Dataset | Gold rule | Annotation depth |
|---|---|---:|
| UKP Humor pairwise | Candidate/X with at least 3 of 5 votes; otherwise `NO_MAJORITY`; remapped to display order | 5 |
| Humicroedit Score | Arithmetic mean of five published integer 0–3 grades | 5 |
| HelpSteer2 pairwise | Sign of published aggregate `preference_strength` (based upstream on three most-similar judgments); five unprocessed signs retained | 5 released supporting judgments |
| HelpSteer2 Score | Arithmetic mean of all five released helpfulness ratings; published upstream integer retained separately | 5 |
| OpenAI summaries pairwise | Strict majority after deduplicating workers for an identical context/candidate pair; exact ties are `SPLIT` | 7 or more |
| SummEval relevance Score | Equal-weight mean of 3 expert + 5 crowd relevance ratings | 8 |

Primary references: [UKP Humor release](https://github.com/UKPLab/acl2019-GPPL-humour-metaphor/tree/{prep.UKP_REVISION}), [Humicroedit release](https://zenodo.org/records/3969509), [HelpSteer2 release](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/{prep.HELPSTEER_REVISION}), [OpenAI summarize-from-feedback data](https://github.com/openai/summarize-from-feedback/tree/master/dataset), and [SummEval paper/data description](https://aclanthology.org/2021.tacl-1.24/). The exact OpenRouter endpoint, model id, Choice/Score schemas, and zero-indexed Score behavior are documented in [OpenRouter's Jev guide](https://openrouter.ai/blog/insights/what-is-jev/) and the underlying primitives in [TypeSafe's API reference](https://docs.typesafe.ai/api).

## Complete real examples

The examples below are actual selected source records. For readability, the shortest strict-gold request in each prepared file is shown, without truncating any field. Responses are deliberately filled-in examples so that the full parsing and comparison behavior is visible before any paid calls.

{chr(10).join(sections)}

## Methodological guardrails established by this audit

1. Report human **alignment**, not objective accuracy, for subjective references.
2. Compute pairwise alignment only where the source reference is A or B; report tie/ambiguous strata separately.
3. Preserve displayed labels, option order, and full probability vectors in results; never infer source identity from A/B alone.
4. Compare continuous Jev Score expectations to human aggregate scores on the native scale after the fixed offset.
5. For HelpSteer Score, use the five-rating mean as the human reference and retain the published rounded integer for sensitivity analysis.
6. Freeze prompts, rubric arrays, scale offsets, and dataset hashes before the baseline run. Any later wording change creates a new condition/request hash.
7. Treat the SummEval level descriptions as a disclosed adaptation, not wording quoted from the original annotation instrument.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    JSON_PATH.write_text(json.dumps({"status": "passed", "records_checked": 9000,
                                    "datasets": summaries}, ensure_ascii=False, indent=2),
                         encoding="utf-8", newline="\n")
    print(json.dumps({"status": "passed", "records_checked": 9000,
                      "report": REPORT_PATH.relative_to(ROOT).as_posix(), "summary": JSON_PATH.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
