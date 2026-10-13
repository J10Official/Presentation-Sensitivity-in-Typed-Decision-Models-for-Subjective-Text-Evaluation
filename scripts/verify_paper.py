"""Validate presentation sources and final PDFs; no statistical recomputation."""
from pathlib import Path
from hashlib import sha256
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper"
EXPORT = ROOT / "reports/paper_exports"
PDF_PATH = HERE / "output/current/manuscript.pdf"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def one(rows, selector):
    matches = [r for r in rows if all(r.get(k) == v for k, v in selector.items())]
    assert len(matches) == 1, (selector, len(matches))
    return matches[0]


def normalized(text):
    return re.sub(r"\s+", "", text).replace("--", "-").replace("–", "-").replace("−", "-").replace("∗", "*")


def conclusion_page(source, page_texts, aux_text=None):
    """Locate the conclusion in the PDF, including in clones without TeX caches."""
    conclusion = source.split(r"\label{sec:main_end}")[0].rstrip().split("\n\n")[-1]
    final_sentence = conclusion.rsplit(". ", 1)[-1]
    assert "\\" not in final_sentence, "Rebuild TeX after changing the conclusion marker"
    # ACL review line numbers can split an otherwise contiguous sentence when
    # PDF text is extracted from a two-column page.
    def without_review_line_numbers(text):
        text = re.sub(r"(?m)^\s*\d+\s+|\s+\d+\s*$", "", text)
        return re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    pages = [i for i, text in enumerate(page_texts, 1)
             if normalized(final_sentence) in normalized(without_review_line_numbers(text))]
    assert len(pages) == 1, ("Conclusion ending must appear on exactly one page", pages)
    if aux_text is not None:
        label = re.search(r"\\newlabel\{sec:main_end\}\{\{[^}]*\}\{(\d+)\}", aux_text)
        assert label and int(label[1]) == pages[0], "TeX auxiliary/PDF page mismatch"
    return pages[0]


def overview_checks(effects, results, tables):
    """Independently verify every revised table cell and literal request."""
    ledger = read(EXPORT / "review_display_evidence.json")
    assert len(ledger) == 135, len(ledger)
    expected_rows = {}
    for cell in ledger:
        if cell["source"] == "level1_effects.json":
            row = one(effects, cell["selector"])
            value = {k: row[k] for k in ("estimate", "low", "high")}
            q = row["q_by_validated"]
        elif cell["source"] == "level2_results.json":
            row = one(results["practical_impact"], cell["selector"])[cell["field"]]
            value = {k: row[k] for k in ("estimate", "low", "high")}
            q = row.get("q_by_level2")
        else:
            table = tables[cell["selector"]["table"]]
            value = table["rows"][cell["selector"]["row"]][table["columns"].index(cell["field"])]
            q = None
        assert value == cell["value"] and q == cell["q"], cell
        def number(v):
            rounded = round(float(v) * cell["scale"], cell["digits"])
            return f"{0.0 if rounded == 0 else rounded:.{cell['digits']}f}"
        if cell["kind"] == "ci":
            text = number(value["estimate"]) + " [" + number(value["low"]) + ", " + number(value["high"]) + "]"
            text += "*" if q is not None and q < .05 else ""
        else:
            text = number(value)
        assert text == cell["rendered"]
        expected_rows.setdefault(cell["table"], {}).setdefault(cell["row"], []).append(text)
    counts = {"results_choice_effects": 12, "results_choice_consequences": 9,
              "results_score_effects": 12, "appendix_reader_alignment": 9, "appendix_reader_error": 9}
    for name, count in counts.items():
        with (EXPORT / (name + ".csv")).open(encoding="utf8", newline="") as stream:
            rows = list(csv.reader(stream))[1:]
        assert len(rows) == count and len(expected_rows[name]) == count
        for row in rows:
            assert row[1:] == expected_rows[name][row[0]], (name, row)
    diagnostics = read(EXPORT / "review_diagnostic_evidence.json")
    with (EXPORT / "appendix_robustness_summary.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    assert len(rows) == len(diagnostics) == 6
    for row, cell in zip(rows, diagnostics):
        value = tables[cell["source_table"]]["rows"][cell["source_row"]][cell["source_column"]]
        assert value == cell["original"] and row[0] == cell["row"] and row[-1] == cell["rendered"]
        if "[" in value:
            values = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
            expected = f"{float(values[0]):.2f} [{float(values[1]):.2f}, {float(values[2]):.2f}]"
        else:
            expected = f"{float(value)*(100 if cell['source_table']=='score_weighted_sensitivity' else 1):.2f}"
        assert expected == row[-1]
    # New main-body claims, checked directly against their actual comparators.
    for condition, expected in (("pair_social_first_high", -4.6), ("pair_social_second_high", -6.0)):
        row = one(results["practical_impact"], {"model":"Perplexity","dataset":"UKP humor","condition":condition})
        assert row["control"] == "pair_social_none"
        assert round(100*row["human_alignment_change"]["estimate"],1) == expected
        assert row["human_alignment_change"]["q_by_level2"] < .05
    row = one(results["practical_impact"], {"model":"Perplexity","dataset":"Humicroedit","condition":"score_social_high"})
    assert row["control"] == "score_social_none" and round(100*row["human_MAE_change"]["estimate"],1) == 4.8
    tie = tables["tie_by_reference"]
    native = [dict(zip(tie["columns"],r)) for r in tie["rows"]]
    for model, expected in (("Jev",43),("Clef",19),("Perplexity",88)):
        row = one(native, {"Model":model,"Dataset":"UKP humor","Condition":"pair_tie_last","Human reference":"X"})
        assert int(row["N"]) == 187 and round(float(row["X selected %"])*187/100) == expected
    assert max(abs(float(r[5])) for r in tables["score_representation_contrasts"]["rows"]) < .05
    assert round(max(abs(float(r[5])-float(r[4]))*100 for r in tables["score_weighted_sensitivity"]["rows"]),2) == .37
    assert len(tables["long_item_sensitivity"]["rows"]) == 45
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from subjective_baseline_runner import INPUT_ROOT
    from subjective_intervention_runner import request_for
    examples = read(EXPORT / "request_examples.json")
    assert len(examples) == 2
    for item in examples:
        source = item["source_input"]
        if (INPUT_ROOT / item["dataset_file"]).exists():
            with (INPUT_ROOT / item["dataset_file"]).open(encoding="utf8") as stream:
                local = json.loads(next(stream))
            assert local["id"] == source["id"] and local["input"] == source["input"]
        payload, mapping = request_for(item["dataset_file"], source, item["condition"])
        assert source["id"] == item["item_id"] and payload == item["payload"]
        assert "reference" not in json.dumps(payload) and "worker" not in json.dumps(payload)
    # Appendix collection context and human-reference diagnostics use saved summaries.
    cells = read(ROOT / "reports/full_study_analysis/cell_audit.json")
    assert len(cells) == 468 and sum(row[3] for row in cells) == 702000
    assert all(len({row[4] for row in cells if row[0] == model}) == 1
               for model in ("Jev", "Clef", "Perplexity"))
    drift = tables["repeat_drift"]["rows"]
    assert len(drift) == 72
    largest = max(drift, key=lambda row: abs(float(row[3].split()[0])))
    assert largest[:4] == ["Clef", "HelpSteer pairs", "3", "0.159 [0.002, 0.316]"]
    for dataset, expected in (("UKP humor", 85.32), ("HelpSteer pairs", 82.36),
                              ("OpenAI summaries", 90.97)):
        row = one([dict(zip(tables["finite_vote_stability"]["columns"], r))
                   for r in tables["finite_vote_stability"]["rows"]],
                  {"Dataset": dataset, "Group": "All strict"})
        assert round(100 * float(row["Mean empirical majority-reproduction probability"]), 2) == expected
    return {"result":"PASS","source_linked_cells":len(ledger),"diagnostics":6,"literal_requests":2,
            "consistent_model_identifier_cells":468,"repeat_drift_comparisons":72,
            "reference_stability_diagnostics":3}


def source_checks():
    inputs = read(ROOT / "provenance/package_manifest.json")["frozen_numeric_sha256"]
    for relative, expected in inputs.items():
        assert digest(ROOT / relative) == expected, relative
    old = read(EXPORT / "presentation_manifest.json")
    for relative, value in old["input_sha256"].items():
        assert digest(ROOT / relative) == value, relative
    for relative in ("level1_audit.json", "level2_audit.json"):
        assert not read(ROOT / "reports/validated_analysis" / relative)["failures"]

    effects = read(ROOT / "reports/validated_analysis/level1_effects.json")
    results = read(ROOT / "reports/validated_analysis/level2_results.json")
    tables = read(ROOT / "reports/full_study_analysis/tables.json")
    claims = read(EXPORT / "claim_evidence.json")
    checked = []
    for key, claim in claims.items():
        ref = claim["sources"][0]
        selector, field = ref["selector"], ref["field"]
        name = Path(ref["file"]).name
        # Also support backslash paths when run on a different host.
        name = name.split("\\")[-1]
        if name == "tables.json" and "row" in selector:
            table = tables[selector["table"]]
            value = table["rows"][selector["row"]][table["columns"].index(field)]
        elif name == "level1_effects.json" and isinstance(claim["value"], dict):
            row = one(effects, selector)
            value = {k: row[k] for k in claim["value"]}
        elif key == "jev_prior_sum_q":
            value = one(effects, {**selector, "contrast": "Selected vs rejected prior judge cue"})[field]
        elif key.startswith("meta_"):
            pooled = one(results["cross_task_meta"], selector)["pooled"]
            value = pooled if field == "pooled" else pooled[field]
        elif key in {"clef_reject_hs", "clef_reject_sum", "jev_reject_sum", "clef_rejection_same_hs"}:
            value = one(results["practical_impact"], selector)[field]
        elif key == "jev_format_rank_range":
            table = tables["score_formats"]
            col = table["columns"].index("Rank correlation vs native")
            values = [float(r[col]) for r in table["rows"] if r[0] == "Jev"]
            value = [min(values), max(values)]
        elif key == "jev_score_id_max":
            value = max(abs(r["estimate"] * 100) for r in effects
                        if r["model"] == "Jev" and r["family"] in {"alphabetic", "numeric"}
                        and r["dataset"] in {"Humicroedit", "HelpSteer scores", "SummEval relevance"})
        elif key == "reader_prior_range":
            rows = [r for r in results["shared_vulnerability"]["associations"]
                    if r["first"] == "Reader high vs low"
                    and r["second"] in {"Prior selected vs rejected", "Prior high vs low"}]
            assert len(rows) == 18
            assert all(r["magnitude_correlation"]["q_by_level2"] < .05 for r in rows)
            values = [r["magnitude_correlation"]["estimate"] for r in rows]
            value = [min(values), max(values)]
        else:
            raise AssertionError(f"Unhandled source claim: {key}")
        assert value == claim["value"], key
        checked.append(key)

    for name, count in (("figure1_position_identifier", 18),
                        ("figure2_endorsement_cues", 36),
                        ("figure3_score_levels_ranks", 27)):
        points = read(EXPORT / (name + "_data.json"))
        assert len(points) == count
        for point in points:
            row = one(effects, {k: point[k] for k in ("model", "dataset", "family", "contrast")})
            assert all(point[k] == v for k, v in row.items())
    assert len(effects) == 369 and len(results["practical_impact"]) == 450
    assert len(results["shared_vulnerability"]["associations"]) == 648
    assert len(results["cross_task_meta"]) == 45
    assert sum(r["pooled"]["q_by_level2"] < .05 for r in results["cross_task_meta"]) == 1
    assert all(r["pooled"]["df"] == 2 and r["pooled"]["k"] == 3 for r in results["cross_task_meta"])
    return {"result": "PASS", "immutable_inputs": len(inputs), "claims": checked,
            "overview_checks": overview_checks(effects, results, tables),
            "figure_source_points": 81,
            "copied_table_checks": read(EXPORT / "presentation_checks.json")}


def pdf_checks(name, expected_tables, expected_figures, max_pages=None):
    import pdfplumber

    path = PDF_PATH
    log_path = path.with_suffix(".log")
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"Overfull|undefined|multiply defined|^!|Fatal error|Label\(s\) may have changed|Line number reference failed", log, re.M)
    inventory, captions_t, captions_f, texts = [], [], [], []
    with pdfplumber.open(path) as pdf:
        assert pdf.metadata.get("Author") == "Anonymous", "PDF author metadata must be anonymous"
        if max_pages:
            assert 8 <= len(pdf.pages) <= max_pages, len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            assert not any(token in text for token in ("??", "\ufffd", "{{", "<!--")), (name, i)
            assert not any(c["x0"] < -.5 or c["x1"] > page.width + .5 or c["top"] < -.5
                           or c["bottom"] > page.height + .5 for c in page.chars), (name, i)
            tables = re.findall(r"Table\s*(S?\d+):", text)
            figures = re.findall(r"Figure\s*(S?\d+):", text)
            captions_t.extend(tables)
            captions_f.extend(figures)
            inventory.append({"page": i, "words": len(text.split()), "tables": tables, "figures": figures})
            # Preserve column reading order and omit review line numbers for prose matching.
            # Full-page text remains necessary for full-width table/figure captions.
            left = page.crop((60, 0, page.width / 2, page.height)).extract_text() or ""
            right = page.crop((page.width / 2, 0, page.width - 60, page.height)).extract_text() or ""
            texts.append(text + "\n" + left + "\n" + right)
    assert sorted(captions_t) == sorted(expected_tables), (name, captions_t)
    assert sorted(captions_f) == sorted(expected_figures), (name, captions_f)
    return {"pages": len(inventory), "sha256": digest(path), "inventory": inventory}, texts


def main():
    import pdfplumber

    report = {"source_checks": source_checks()}
    report["manuscript"], texts = pdf_checks("manuscript", [str(n) for n in range(1, 35)], ["1"])
    starts = [i for i, text in enumerate(texts, 1) if "ADataSourcesandSelectionDetails" in normalized(text)]
    assert len(starts) == 1, "Expected one lettered Appendix A after references"
    appendix_start = starts[0]
    aux_path = PDF_PATH.with_suffix(".aux")
    if aux_path.exists():
        label = re.search(r"\\newlabel\{app:sources\}\{\{([^{}]+)\}\{(\d+)\}", aux_path.read_text(encoding="utf-8"))
        assert label and label[1] == "A" and int(label[2]) == appendix_start
        aux = aux_path.read_text(encoding="utf-8")
        for name, number in (("tab:choice_conditions", "2"), ("tab:score_conditions", "3")):
            table_label = re.search(r"\\newlabel\{" + name + r"\}\{\{(\d+)\}", aux)
            assert table_label and table_label[1] == number, (name, number)
    assert appendix_start > 1, "Appendices must follow the main paper and references"
    assert "\\externaldocument" not in (HERE / "main.tex").read_text(encoding="utf-8")
    assert "Supplement~" not in (HERE / "results.tex").read_text(encoding="utf-8")
    report["main_and_references_pages"] = appendix_start - 1
    report["appendix_start_page"] = appendix_start
    report["appendix_pages"] = report["manuscript"]["pages"] - appendix_start + 1
    report["packaging"] = "Single PDF; references followed by lettered appendices; internal cross-references"
    all_text = normalized("\n".join(texts[:appendix_start - 1]))
    appendix_text = normalized("\n".join(texts[appendix_start - 1:]))
    assert all(value in appendix_text for value in ("0.159[0.002,0.316]", "85.32", "82.36", "90.97"))
    # Also follow uninterrupted column order across page breaks. Interleaving
    # full-page and cropped text can otherwise split a printed numerical claim.
    with pdfplumber.open(PDF_PATH) as pdf:
        columns = []
        body_columns = []
        for page in pdf.pages[:appendix_start - 1]:
            columns.extend(re.sub(r"\n\d+\s*$", "", page.crop(box).extract_text() or "") for box in (
                (60, 0, page.width / 2, page.height),
                (page.width / 2, 0, page.width - 60, page.height)))
            # Full-width floats otherwise interrupt a claim spanning columns.
            # Review line numbers locate the prose below those floats.
            words = page.extract_words()
            for left in (True, False):
                margins = [w for w in words if w["text"].isdigit() and
                           (w["x1"] < 60 if left else w["x0"] > page.width - 60)]
                if margins:
                    top = max(0, min(w["top"] for w in margins) - 2)
                    box = (60, top, page.width / 2, page.height - 60) if left else (
                        page.width / 2, top, page.width - 60, page.height - 60)
                    body_columns.append(page.crop(box).extract_text() or "")
    all_text += normalized("\n".join(columns))
    all_text += normalized("\n".join(body_columns))
    claims = read(EXPORT / "claim_evidence.json")
    # The restructured Results promotes a declared subset; all original source
    # facts remain audited above, including details moved into the appendix.
    main_claims = [
        "jev_pos_sum", "pplx_reader_flip_ukp",
        "jev_reader_sum", "jev_reader_hs", "jev_reader_score", "jev_prior_sum", "jev_prior_sum_q",
        "clef_source_sum", "clef_source_flip_sum", "pplx_inject_sum_second", "clef_inject_score",
        "jev_humor_sd_ratio", "jev_sum_sd_ratio",
        "jev_expert_r", "jev_crowd_r", "jev_pool_r", "jev_numeric_ukp",
        "clef_reject_sum", "clef_recovered_hs", "clef_baseline_hs",
        "jev_format_rank_range",
        "jev_pos_selection_sum", "clef_b_selection_sum", "jev_wrapper_source_sum_ci", "clef_rejection_same_hs",
    ]
    for key in main_claims:
        text = claims[key]["rendered"]
        if key == "reader_prior_range":
            assert "0.155" in all_text and "0.889" in all_text
        else:
            assert normalized(text) in all_text, ("Main fact not printed", key, text)
    for heading in ("Introduction", "RelatedWork", "StudyDesignandHumanReferences",
                    "RequestsandInterventions", "MeasurementandInference", "Results",
                    "Discussion", "Conclusion", "Limitations", "EthicalConsiderations", "References"):
        assert heading in all_text, heading
    report["main_printed_claims"] = len(main_claims)
    # Check every explicit task/model cell as printed, including intervals and stars.
    for key in ("results_choice_effects", "results_score_effects", "results_choice_consequences"):
        with (EXPORT / (key + ".csv")).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))[1:]
        for row in rows:
            assert all(normalized(cell) in all_text for cell in row[1:4]), (key, row)
    assert "Choicejudgments" in all_text and "Scoringjudgments" in all_text
    with pdfplumber.open(PDF_PATH) as pdf:
        result_start = next(i for i,p in enumerate(pdf.pages,1) if "6Results" in normalized(p.extract_text() or ""))
        discussion_start = next(i for i,p in enumerate(pdf.pages,1) if "7Discussion" in normalized(p.extract_text() or ""))
        references_start = next(i for i,p in enumerate(pdf.pages,1) if "References" in (p.extract_text() or "") and i > discussion_start - 1)
    # ARR excludes Limitations/Ethics and references from the content limit.
    aux_path = PDF_PATH.with_suffix(".aux")
    aux_text = aux_path.read_text(encoding="utf-8") if aux_path.exists() else None
    main_end_page = conclusion_page((HERE / "main.tex").read_text(encoding="utf-8"), texts, aux_text)
    assert main_end_page <= 8, main_end_page
    assert "Acknowledgments" not in "\n".join(texts)
    report["main_text_pages"] = main_end_page
    report["results_start_page"] = result_start
    report["discussion_start_page"] = discussion_start
    report["references_start_page"] = references_start
    assert "Sequentialcollection" not in all_text and "Unexposedpreprocessing" not in all_text
    assert all(not p["figures"] for p in report["manuscript"]["inventory"][:appendix_start - 1])
    report["result"] = "PASS"
    release_check = ROOT / "provenance/workspace_validation.json"
    release_pdf = read(release_check).get("paper", {}).get("sha256") if release_check.exists() else None
    report["visual_review"] = (read(release_check).get("paper", {}).get("visual_review", "See recorded release visual QA.")
                               if release_pdf == digest(PDF_PATH) else
                               "Not performed by this text-check script; rerender and inspect after layout edits.")
    report["build_log_checked"] = PDF_PATH.with_suffix(".log").exists()
    (EXPORT / "package_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: {len(claims)} source claims; 81 figure points; {len(main_claims)} printed main facts; "
          f"{report['manuscript']['pages']}-page combined PDF "
          f"({report['main_and_references_pages']} main/reference pages + {report['appendix_pages']} appendix pages).")


if __name__ == "__main__":
    main()
