"""Readable manuscript displays sourced only from existing audited results."""
import json


def build_tables(b, tables, effects, results):
    ledger = []
    def make(*args, **kwargs):
        b.make_table(*args, raw_caption=True, **kwargs)

    def num(value, digits):
        value = round(float(value), digits)
        return f"{0.0 if value == 0 else value:.{digits}f}"

    def cell(table, row, model, source, selector, field, value, kind="ci", scale=100, digits=1, q=None):
        if kind == "ci":
            text = num(value["estimate"] * scale, digits) + " [" + num(value["low"] * scale, digits) + ", " + num(value["high"] * scale, digits) + "]"
            text += "*" if q is not None and q < .05 else ""
        else:
            text = num(float(value) * scale, digits)
        ledger.append(dict(table=table, row=row, model=model, source=source, selector=selector,
                           field=field, value=value, kind=kind, scale=scale, digits=digits, q=q, rendered=text))
        return text

    def effect_cell(key, label, model, dataset, family, contrast):
        r = b.effect(effects, model, dataset, family, contrast)
        selector = {k: r[k] for k in ("model", "dataset", "family", "contrast")}
        return cell(key, label, model, "level1_effects.json", selector, None,
                    {k: r[k] for k in ("estimate", "low", "high")}, q=b.qvalue(r))

    def table_cell(key, label, model, table, prefix, field, digits=2, scale=1):
        index, row = b.find_row(tables, table, *prefix)
        value = row[tables[table]["columns"].index(field)]
        return cell(key, label, model, "tables.json", {"table": table, "row": index}, field,
                    value, kind="number", digits=digits, scale=scale)

    def impact_cell(key, label, model, dataset, condition, field):
        r = b.impact(results, model, dataset, condition)[field]
        return cell(key, label, model, "level2_results.json",
                    dict(model=model, dataset=dataset, condition=condition), field,
                    {k: r[k] for k in ("estimate", "low", "high")}, q=b.qvalue(r))

    columns = ["Task / contrast", "Jev / pp [95% CI]", "Clef Flash / pp [95% CI]", "Perplexity / pp [95% CI]"]
    choice = [
        ("First position", "presentation_probability", "First-position"),
        ("B versus A", "presentation_probability", "B-label"),
        ("Human versus AI", "ai_human", "Human vs AI attribution"),
        ("Reader feedback", "social", "High vs low reader cue"),
    ]
    rows = []
    for task, dataset in zip(b.TASKS, b.PAIRS):
        for name, family, contrast in choice:
            label = task + ": " + name
            rows.append([label] + [effect_cell("results_choice_effects", label, m, dataset, family, contrast) for m in b.MODELS])
    make("results_choice_effects", columns, rows,
                 r"Choice probability effects on unchanged texts, $n=1{,}500$ per task/model. Positive values favor first position, label B, Human attribution, or favorable reader feedback. Position and B effects average over the other factorial factor; cue effects reverse both assignments within the same format. Intervals are pointwise and dependence-aware; * marks primary-family BY $q<.05$. Complete identifiers and cue controls: Appendices~\ref{app:results_identifiers}--\ref{app:results_cues}.",
                 "tab:results_choice_effects")
    rows = []
    for task, dataset in zip(b.TASKS, b.PAIRS):
        label = task + ": reader reversal (%)"
        rows.append([label] + [table_cell("results_choice_consequences", label, m, "pairwise_cue_effects",
                    (m, dataset, "High vs low reader cue"), "Any reversal flip %") for m in b.MODELS])
        label = task + ": unchanged repeat (%)"
        rows.append([label] + [table_cell("results_choice_consequences", label, m, "repeatability",
                    (m, dataset), "Baseline/repeat flip % (pairs) / mean abs delta (scores)") for m in b.MODELS])
        label = task + ": rejection agreement (pp)"
        rows.append([label] + [impact_cell("results_choice_consequences", label, m, dataset, "pair_rejection", "human_alignment_change") for m in b.MODELS])
    make("results_choice_consequences", ["Task / outcome", "Jev", "Clef Flash", "Perplexity"], rows,
                 r"Practical choice consequences. Reader rows give selected-text changes under reversed endorsements; repeat rows average four unchanged-call comparisons with baseline (both descriptive \%, all 1,500 pairs). Rejection rows give recovered-rejection minus baseline human agreement in pp, with pointwise 95\% CIs; * marks secondary-family BY $q<.05$. Strict reference counts: 1,063/1,447/1,451 for humor/helpfulness/summary. Cue-versus-control agreement changes: Table~\ref{tab:appendix_reader_alignment}.",
                 "tab:results_choice_consequences")
    score = [
        ("Reader feedback", "social", "High vs low reader cue"),
        ("Previous judge", "prior_judge", "High vs low prior judge cue"),
        ("Three levels", "score_format", "score_granularity_3 minus native"),
        ("Seven levels", "score_format", "score_granularity_7 minus native"),
    ]
    rows = []
    for task, dataset in zip(("Humor", "Helpfulness", "Relevance"), b.SCORES):
        for name, family, contrast in score:
            label = task + ": " + name
            rows.append([label] + [effect_cell("results_score_effects", label, m, dataset, family, contrast) for m in b.MODELS])
    make("results_score_effects", columns, rows,
                 r"Score effects in normalized-score pp, $n=1{,}500$ per task/model. Feedback compares favorable with unfavorable claims; scale variants compare with the native rubric. Positive values indicate higher ratings. Alternative formats change descriptions as well as level count. Pointwise 95\% CIs; * marks primary-family BY $q<.05$. Rank associations, anchor reversal, and reference-error changes: Appendix~\ref{app:results_formats}.",
                 "tab:results_score_effects")
    for datasets, key, field, conditions, title in (
        (b.PAIRS, "appendix_reader_alignment", "human_alignment_change",
         ("pair_social_first_high", "pair_social_second_high"), "Agreement change / pp [95% CI]"),
        (b.SCORES, "appendix_reader_error", "human_MAE_change",
         ("score_social_high", "score_social_low"), "MAE change / pp [95% CI]"),
    ):
        rows = []
        for dataset in datasets:
            for m in b.MODELS:
                label = b.SHORT[dataset] + " / " + m
                rows.append([label] + [impact_cell(key, label, m, dataset, c, field) for c in conditions])
        pair = datasets == b.PAIRS
        names = ("First high / " + title, "Second high / " + title) if pair else ("High feedback / " + title, "Low feedback / " + title)
        caption = (r"Reader-feedback agreement changes versus the unspecified-feedback control, on strict references only (UKP 1,063; HelpSteer2 1,447; OpenAI 1,451). First/second high assigns 80\% to that original candidate and 20\% to the other. Positive changes mean higher agreement. These comparators differ from the reversal contrast in Table~\ref{tab:results_choice_effects}." if pair else
                   r"Reader-feedback changes in MAE against the fixed human mean, relative to unspecified feedback; $n=1{,}500$. Negative changes indicate smaller error. High and low are separate comparisons with the same control; their difference in error need not follow their signed score shift.")
        make(key, ["Dataset / model", *names], rows, caption + r" Pointwise dependence-aware 95\% CIs; * marks secondary-family BY $q<.05$.", "tab:" + key)

    # Every row is an existing diagnostic, displayed without new tests or pooling.
    diagnostic_rows = []
    diagnostic_sources = []
    cases = [
        ("Baseline snapshot", "baseline_definition_sensitivity", ("Jev", b.PAIRS[0], "Stored snapshot"), 4, "UKP position (pp)"),
        ("Five-call mean", "baseline_definition_sensitivity", ("Jev", b.PAIRS[0], "Mean of five unchanged calls"), 4, "UKP position (pp)"),
        ("Score reconstruction", "score_representation_contrasts", ("Jev", b.SCORES[1], "Reader high vs low"), 4, "Helpfulness reader (pp)"),
        ("Band reweighting", "score_weighted_sensitivity", ("Jev", b.SCORES[0]), 5, "Humor MAE (pp)"),
        ("Lower-length subset", "long_item_sensitivity", ("Jev", b.PAIRS[0], "First-position"), 4, "UKP position (pp)"),
        ("Component variance", "dependence_robustness", ("Jev", b.PAIRS[0], "First-position"), 4, "UKP position (pp)"),
    ]
    for name, table, prefix, column, scope in cases:
        index, row = b.find_row(tables, table, *prefix)
        value = row[column]
        if table in ("long_item_sensitivity", "dependence_robustness"):
            text = b.printed_ci(value)
        else:
            text = num(float(value) * (100 if table == "score_weighted_sensitivity" else 1), 2)
        diagnostic_rows.append([name, "Jev: " + scope, text])
        diagnostic_sources.append(dict(row=name, source_table=table, source_row=index, source_column=column, original=value, rendered=text))
    make("appendix_robustness_summary", ["Check", "Quantity", "Alternative estimate"], diagnostic_rows,
                 r"Representative numerical sensitivity checks, all from existing diagnostics. Jev's primary UKP position effect is -2.25 [-2.98, -1.52] pp; primary helpfulness reader effect is 16.47 [16.12, 16.81] pp; primary humor MAE is 25.56 pp. Baseline alternatives address the factorial; reweighting addresses baseline score error. Length and component intervals are sensitivity diagnostics. Full cases and exceptions are summarized in the accompanying text and retained in the electronic tables.",
                 "tab:appendix_robustness_summary")
    b.write_json(b.GEN / "review_display_evidence.json", ledger)
    b.write_json(b.GEN / "review_diagnostic_evidence.json", diagnostic_sources)


def request_examples(b):
    """Materialize literal request examples from the frozen prepared rows offline."""
    import sys
    sys.path.insert(0, str(b.ROOT / "scripts"))
    from subjective_baseline_runner import INPUT_ROOT
    from subjective_intervention_runner import request_for
    examples = []
    existing_path = b.GEN / "request_examples.json"
    existing = b.read(existing_path) if existing_path.exists() else []
    for filename, condition in (("ukp_humor_pairwise.jsonl", "pair_social_first_high"),
                                ("humicroedit_scoring.jsonl", "score_social_high")):
        if (INPUT_ROOT / filename).exists():
            with (INPUT_ROOT / filename).open(encoding="utf-8") as stream:
                row = json.loads(next(stream))
        else:
            row = next(e["source_input"] for e in existing if e["dataset_file"] == filename)
        payload, mapping = request_for(filename, row, condition)
        examples.append(dict(dataset_file=filename, item_id=row["id"], condition=condition,
                             source_input={"id": row["id"], "input": row["input"]}, payload=payload))
    b.write_json(b.GEN / "request_examples.json", examples)
    lines = [r"\subsection{Complete feedback-request examples}", r"\label{app:request_examples}",
             "These literal payloads are reconstructed offline from selected records using the collection runner. They include all submitted fields; the first assigns high feedback to original A and low feedback to B. Scoring adds feedback as shared-state metadata. References and internal analysis mappings are withheld."]
    # Render every payload character with wrapping inside the appendix column.
    for example in examples:
        lines += [r"\paragraph{" + b.latex_escape(example["dataset_file"]) + "}", r"\begin{quote}\scriptsize\ttfamily\raggedright"]
        text = json.dumps(example["payload"], indent=2, ensure_ascii=False)
        for line in text.splitlines():
            # Allow line breaks after each space while retaining all characters.
            escaped = b.latex_escape(line.lstrip()).replace(" ", r"\allowbreak{} ")
            lines.append(escaped + r"\par")
        lines.append(r"\end{quote}")
    (b.TEX / "request_examples.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
