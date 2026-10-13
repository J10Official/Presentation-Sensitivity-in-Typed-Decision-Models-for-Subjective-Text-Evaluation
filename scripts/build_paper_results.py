"""Rebuild current figures/tables from validated outputs; never overwrite manuscript prose.

Python 3.11 + NumPy + Matplotlib. No model calls, hypothesis tests, or meta-analysis.
Run from any directory. LaTeX compilation is a separate, optional presentation step.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper"
VALID = ROOT / "reports/validated_analysis"
LEGACY = ROOT / "reports/full_study_analysis"
GEN = ROOT / "reports/paper_exports"
TEX = HERE / "generated"
PDF = HERE / "figures"
MODELS = ["Jev", "Clef", "Perplexity"]
PAIRS = ["UKP humor", "HelpSteer pairs", "OpenAI summaries"]
SCORES = ["Humicroedit", "HelpSteer scores", "SummEval relevance"]
TASKS = ["Humor", "Helpfulness", "Summary"]
SHORT = dict(zip(PAIRS + SCORES, ["UKP", "HS pair", "OpenAI", "Humicroedit", "HS score", "SummEval"]))
COLORS = {"Jev": "#0072B2", "Clef": "#D55E00", "Perplexity": "#009E73"}
MARKERS = {"Jev": "o", "Clef": "s", "Perplexity": "^"}
CLAIMS, DISPLAYS = {}, []


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sources():
    manifest = read(VALID / "manifest.json")
    hashes = {p.replace("\\", "/"): h for p, h in manifest["new_artifact_sha256"].items()}
    for name in ["level1_effects.json", "level1_audit.json", "level2_results.json", "level2_audit.json"]:
        path = VALID / name
        assert digest(path) == hashes[path.relative_to(ROOT).as_posix()]
    for name in ["level1_audit.json", "level2_audit.json"]:
        assert not read(VALID / name)["failures"]
    # Historical descriptive tables are also anchored by their audited input hash.
    table_hashes = {p.replace("\\", "/"): h for p, h in read(VALID / "level1_audit.json")["input_paths"].items()}
    assert digest(LEGACY / "tables.json") == table_hashes[(LEGACY / "tables.json").relative_to(ROOT).as_posix()]
    return read(LEGACY / "tables.json"), read(VALID / "level1_effects.json"), read(VALID / "level2_results.json")


def source_ref(file, selector, field=None):
    return {"file": file.relative_to(ROOT).as_posix(), "selector": selector, "field": field}


def find_row(tables, table, *prefix):
    matches = [(i, r) for i, r in enumerate(tables[table]["rows"]) if r[:len(prefix)] == list(prefix)]
    assert len(matches) == 1, (table, prefix, len(matches))
    return matches[0]


def effect(effects, model, dataset, family, contrast):
    matches = [r for r in effects if (r["model"], r["dataset"], r["family"], r["contrast"]) == (model, dataset, family, contrast)]
    assert len(matches) == 1, (model, dataset, family, contrast)
    return matches[0]


def impact(results, model, dataset, condition):
    matches = [r for r in results["practical_impact"] if (r["model"], r["dataset"], r["condition"]) == (model, dataset, condition)]
    assert len(matches) == 1
    return matches[0]


def qvalue(r):
    return r.get("q_by_validated", r.get("q_by_level2"))


def ci(r, scale=100, stars=False, digits=2):
    text = f"{r['estimate']*scale:.{digits}f} [{r['low']*scale:.{digits}f}, {r['high']*scale:.{digits}f}]"
    return text + ("*" if stars and qvalue(r) is not None and qvalue(r) < .05 else "")


def printed_ci(text, digits=2):
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    assert len(vals) == 3, text
    return f"{float(vals[0]):.{digits}f} [{float(vals[1]):.{digits}f}, {float(vals[2]):.{digits}f}]"


def register(name, value, refs, text=None):
    assert name not in CLAIMS, name
    CLAIMS[name] = {"value": value, "rendered": str(value) if text is None else text, "sources": refs}
    return CLAIMS[name]["rendered"]


def register_effect(name, effects, model, dataset, family, contrast, full_ci=True):
    r = effect(effects, model, dataset, family, contrast)
    return register(name, {k: r[k] for k in ["estimate", "low", "high", "n", "q_by_validated"]},
                    [source_ref(VALID / "level1_effects.json", {k: r[k] for k in ["model", "dataset", "family", "contrast"]})],
                    ci(r) if full_ci else f"{r['estimate']*100:.2f}")


def register_table(name, tables, table, prefix, column, digits=2, scale=1, point_only=False):
    index, row = find_row(tables, table, *prefix)
    value = row[column]
    text = (f"{float(str(value).split()[0]):.{digits}f}" if point_only else
            printed_ci(value, digits) if "[" in str(value) else f"{float(value)*scale:.{digits}f}")
    return register(name, value, [source_ref(LEGACY / "tables.json", {"table": table, "row": index}, tables[table]["columns"][column])], text)


def make_claims(tables, effects, results):
    for name, dataset, contrast in [
        ("jev_pos_ukp", PAIRS[0], "First-position"), ("jev_pos_hs", PAIRS[1], "First-position"),
        ("jev_pos_sum", PAIRS[2], "First-position"), ("jev_b_ukp", PAIRS[0], "B-label"),
        ("jev_b_hs", PAIRS[1], "B-label"), ("jev_b_sum", PAIRS[2], "B-label")]:
        register_effect(name, effects, "Jev", dataset, "presentation_probability", contrast)
    for name, model, dataset, family, contrast, full in [
        ("clef_b_ukp", "Clef", PAIRS[0], "presentation_probability", "B-label", False),
        ("clef_b_sum", "Clef", PAIRS[2], "presentation_probability", "B-label", False),
        ("jev_pos_selection_sum", "Jev", PAIRS[2], "presentation_hard choice", "First-position", True),
        ("clef_b_selection_sum", "Clef", PAIRS[2], "presentation_hard choice", "B-label", True),
        ("pplx_pos_ukp", "Perplexity", PAIRS[0], "presentation_probability", "First-position", False),
        ("pplx_pos_hs", "Perplexity", PAIRS[1], "presentation_probability", "First-position", False),
        ("jev_numeric_ukp", "Jev", PAIRS[0], "numeric", "Numeric identifier 2 vs 1", False),
        ("clef_numeric_ukp", "Clef", PAIRS[0], "numeric", "Numeric identifier 2 vs 1", False),
        ("jev_reader_sum", "Jev", PAIRS[2], "social", "High vs low reader cue", False),
        ("jev_reader_hs", "Jev", PAIRS[1], "social", "High vs low reader cue", False),
        ("jev_reader_score", "Jev", SCORES[1], "social", "High vs low reader cue", True),
        ("clef_source_sum", "Clef", PAIRS[2], "ai_human", "Human vs AI attribution", True),
        ("jev_source_score_humor", "Jev", SCORES[0], "ai_human", "Human vs AI attribution", False),
        ("jev_prior_sum", "Jev", PAIRS[2], "prior_judge", "Selected vs rejected prior judge cue", False),
        ("pplx_inject_sum_second", "Perplexity", PAIRS[2], "injection", "second directive minus inert", True),
        ("clef_inject_score", "Clef", SCORES[2], "injection", "Highest-rating directive minus inert", True),
        ("jev_reverse_humor", "Jev", SCORES[0], "score_format", "score_anchor_reverse minus native", True),
        ("jev_seven_humor", "Jev", SCORES[0], "score_format", "score_granularity_7 minus native", True),
        ("jev_three_sum", "Jev", SCORES[2], "score_format", "score_granularity_3 minus native", True),
    ]:
        register_effect(name, effects, model, dataset, family, contrast, full)
    for name, table, prefix, column, digits, scale in [
        ("pplx_reader_flip_ukp", "pairwise_cue_effects", ("Perplexity", PAIRS[0], "High vs low reader cue"), 5, 2, 1),
        ("clef_source_flip_sum", "pairwise_cue_effects", ("Clef", PAIRS[2], "Human vs AI attribution"), 5, 2, 1),
        ("jev_tie_native", "tie_by_reference", ("Jev", PAIRS[0], "pair_tie_last", "X"), 5, 2, 1),
        ("jev_tie_strict", "tie_overview", ("Jev", PAIRS[0], "pair_tie_last"), 5, 2, 1),
        ("jev_tie_first", "tie_overview", ("Jev", PAIRS[0], "pair_tie_first"), 3, 2, 1),
        ("jev_tie_middle", "tie_overview", ("Jev", PAIRS[0], "pair_tie_middle"), 3, 2, 1),
        ("jev_tie_last", "tie_overview", ("Jev", PAIRS[0], "pair_tie_last"), 3, 2, 1),
        ("pplx_tie_native", "tie_by_reference", ("Perplexity", PAIRS[0], "pair_tie_last", "X"), 5, 2, 1),
        ("jev_summary_clear", "baseline_agreement_groups", ("Jev", PAIRS[2], "clear"), 5, 2, 1),
        ("jev_summary_disputed", "baseline_agreement_groups", ("Jev", PAIRS[2], "disputed"), 5, 2, 1),
        ("jev_summary_sens_clear", "disagreement_correlations", ("Jev", PAIRS[2]), 5, 2, 100),
        ("jev_summary_sens_disputed", "disagreement_correlations", ("Jev", PAIRS[2]), 6, 2, 100),
        ("jev_humor_sd_ratio", "score_compression", ("Jev", SCORES[0]), 4, 2, 1),
        ("jev_sum_sd_ratio", "score_compression", ("Jev", SCORES[2]), 4, 2, 1),
        ("jev_expert_r", "summeval_reference_sensitivity", ("Jev", "Experts only"), 5, 3, 1),
        ("jev_crowd_r", "summeval_reference_sensitivity", ("Jev", "Crowd only"), 5, 3, 1),
        ("jev_pool_r", "summeval_reference_sensitivity", ("Jev", "3:5 pooled"), 5, 3, 1),
        ("jev_humor_mid", "score_compression", ("Jev", SCORES[0]), 6, 2, 1),
        ("human_humor_mid", "score_compression", ("Jev", SCORES[0]), 5, 2, 1),
        ("jev_humor_slope", "score_compression", ("Jev", SCORES[0]), 2, 3, 1),
        ("jev_len_ukp", "length_associations", ("Jev", PAIRS[0]), 3, 2, 1),
        ("jev_len_hs", "length_associations", ("Jev", PAIRS[1]), 3, 2, 1),
        ("jev_len_score", "length_associations", ("Jev", SCORES[0]), 3, 2, 1),
        ("jev_depend_pos", "dependence_robustness", ("Jev", PAIRS[0], "First-position"), 4, 2, 1),
        ("jev_depend_b", "dependence_robustness", ("Jev", PAIRS[0], "B-label"), 4, 2, 1),
    ]:
        register_table(name, tables, table, prefix, column, digits, scale)
    p = effect(effects, "Jev", PAIRS[2], "prior_judge", "Selected vs rejected prior judge cue")
    register("jev_prior_sum_q", p["q_by_validated"], [source_ref(VALID / "level1_effects.json", {"model": "Jev", "dataset": PAIRS[2], "family": "prior_judge"}, "q_by_validated")], f"{p['q_by_validated']:.3f}")
    for model, dataset, key in [("Clef", PAIRS[1], "clef_reject_hs"), ("Clef", PAIRS[2], "clef_reject_sum"), ("Jev", PAIRS[2], "jev_reject_sum")]:
        r = impact(results, model, dataset, "pair_rejection")["human_alignment_change"]
        register(key, r, [source_ref(VALID / "level2_results.json", {"model": model, "dataset": dataset, "condition": "pair_rejection"}, "human_alignment_change")], ci(r))
    for key, dataset, column in [("clef_recovered_hs", PAIRS[1], 3), ("clef_baseline_hs", PAIRS[1], 3)]:
        table = "baseline_pairwise" if "baseline" in key else "selection_rejection"
        register_table(key, tables, table, ("Clef", dataset), column, point_only=True)
    for key, family in [("jev_wrapper_source_sum", "ai_human"), ("jev_wrapper_reader_sum", "social")]:
        register_table(key, tables, "wrapper_effects", ("Jev", PAIRS[2], family), 3, digits=3, point_only=True)
    register_table("jev_wrapper_source_sum_ci", tables, "wrapper_effects", ("Jev", PAIRS[2], "ai_human"), 3)
    same = impact(results, "Clef", PAIRS[1], "pair_rejection")["selection_flip_rate"]
    register("clef_rejection_same_hs", same,
             [source_ref(VALID / "level2_results.json", {"model": "Clef", "dataset": PAIRS[1], "condition": "pair_rejection"}, "selection_flip_rate")],
             f"{100*same['estimate']:.2f}")
    formats = [r for r in tables["score_formats"]["rows"] if r[0] == "Jev"]
    rank_column = tables["score_formats"]["columns"].index("Rank correlation vs native")
    register("jev_format_rank_range", [min(float(r[rank_column]) for r in formats), max(float(r[rank_column]) for r in formats)],
             [source_ref(LEGACY / "tables.json", {"table": "score_formats", "model": "Jev"}, "Rank correlation vs native")],
             f"{min(float(r[rank_column]) for r in formats):.3f}-{max(float(r[rank_column]) for r in formats):.3f}")
    scored_id = [r for r in effects if r["model"] == "Jev" and r["dataset"] in SCORES and r["family"] in ["alphabetic", "numeric"]]
    maximum = max(abs(r["estimate"]*100) for r in scored_id)
    register("jev_score_id_max", maximum, [source_ref(VALID / "level1_effects.json", {"model": "Jev", "track": "score", "families": ["alphabetic", "numeric"]})], f"{maximum:.2f}")
    assocs = results["shared_vulnerability"]["associations"]
    selected = [r for r in assocs if r["first"] == "Reader high vs low" and r["second"] in ["Prior selected vs rejected", "Prior high vs low"]]
    assert len(selected) == 18
    register("reader_prior_range", [min(r["magnitude_correlation"]["estimate"] for r in selected), max(r["magnitude_correlation"]["estimate"] for r in selected)],
             [source_ref(VALID / "level2_results.json", {"first": "Reader high vs low", "second": "Prior", "all_models_components": True})],
             f"{min(r['magnitude_correlation']['estimate'] for r in selected):.3f}-{max(r['magnitude_correlation']['estimate'] for r in selected):.3f}")
    for label, contrast in [("meta_b", "B-label"), ("meta_pos", "First-position"), ("meta_reader", "High vs low reader cue")]:
        row = next(r for r in results["cross_task_meta"] if r["model"] == "Jev" and r["track"] == "choice" and r["contrast"] == contrast)
        register(label, row["pooled"], [source_ref(VALID / "level2_results.json", {"model": "Jev", "track": "choice", "contrast": contrast}, "pooled")], ci(row["pooled"], scale=1))
        register(label+"_i2", row["pooled"]["I2_percent"], [source_ref(VALID / "level2_results.json", {"model": "Jev", "track": "choice", "contrast": contrast}, "I2_percent")], f"{row['pooled']['I2_percent']:.1f}")
    return selected


def latex_escape(value):
    text = str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}", "²": r"$^2$", "τ": r"$\tau$"}
    return "".join(replacements.get(c, c) for c in text).replace("−", "-").replace("×", r"$\times$").replace("rho", r"$\rho$")


def make_table(key, columns, rows, caption, label, wide=True, raw_caption=False):
    rows = [[str(v) for v in row] for row in rows]
    assert all(len(r) == len(columns) for r in rows), key
    with (GEN / f"{key}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(columns); writer.writerows(rows)
    env = "table*" if wide else "table"
    n = len(columns)
    # Wrapping headers preserves readable body type without shrinking the entire table.
    header = []
    for c in columns:
        words = c.split(" / ")
        header.append(r"\shortstack[l]{" + r"\\{}".join(latex_escape(w) for w in words) + "}")
    source = [f"\\begin{{{env}}}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
              r"\begin{tabular}{@{}" + "l" + "r"*(n-1) + "@{}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    source += [" & ".join(latex_escape(v) for v in row) + r" \\" for row in rows]
    source += [r"\bottomrule", r"\end{tabular}", "\\caption{" + (caption if raw_caption else latex_escape(caption)) + "}", "\\label{"+label+"}", f"\\end{{{env}}}", ""]
    (TEX / f"{key}.tex").write_text("\n".join(source), encoding="utf-8", newline="\n")
    DISPLAYS.append({"id": key, "kind": "table", "label": label, "rows": len(rows), "columns": columns})


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.3, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "subjective-judge-results-2026-10-12",
                         "savefig.facecolor": "white", "axes.linewidth": .6})


def point(ax, x, y, lo, hi, model, significant=True):
    color = COLORS[model]
    ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt=MARKERS[model], color=color,
                markerfacecolor=color if significant else "white", markeredgecolor=color,
                markersize=4.8, elinewidth=1, capsize=1.8, markeredgewidth=1, zorder=3)


def model_legend(fig, y=-.01, shapes=True):
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m] if shapes else None,
                      linestyle="none" if shapes else "-", label=m, markersize=5, linewidth=3) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.54, y), frameon=False, columnspacing=2)


def save_figure(fig, name, data):
    # Only the score-format plot is included in the current manuscript.
    if name == "figure3_score_levels_ranks":
        fig.savefig(PDF / f"{name}.pdf", bbox_inches="tight", pad_inches=.035, metadata={"Title": name, "Author": "Anonymous", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    write_json(GEN / f"{name}_data.json", data)
    DISPLAYS.append({"id": name, "kind": "figure", "points": len(data)})


def figures(tables, effects):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15))
    plotted = []
    for ax, contrast, title, bounds in zip(axes, ["First-position", "B-label"], ["(a) Position effect", "(b) A/B identifier effect"], [(-8, 6), (-12, 11)]):
        for task, dataset in enumerate(PAIRS):
            for j, model in enumerate(MODELS):
                r = effect(effects, model, dataset, "presentation_probability", contrast)
                point(ax, r["estimate"]*100, 2-task + (.18-j*.18), r["low"]*100, r["high"]*100, model, qvalue(r)<.05)
                plotted.append({**r, "plot_scale": 100, "panel": title})
        ax.axvline(0, color="#666666", linewidth=.8, linestyle="--")
        ax.set(yticks=[2, 1, 0], yticklabels=TASKS, ylim=(-.5, 2.5), xlim=bounds, title=title)
        ax.set_xlabel("Probability change (pp); + favors " + ("first" if contrast=="First-position" else "B"))
        ax.grid(axis="x", color="#e5e5e5", linewidth=.5); ax.set_axisbelow(True)
    model_legend(fig, -.025)
    fig.subplots_adjust(left=.12, right=.99, top=.87, bottom=.25, wspace=.42)
    save_figure(fig, "figure1_position_identifier", plotted)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    plotted = []
    for ax, datasets, track, title in zip(axes, [PAIRS, SCORES], ["choice", "score"], ["(a) Pairwise probability", "(b) Normalized score"]):
        labels = []
        for task, dataset in enumerate(datasets):
            for family, cue in [("social", "Readers"), ("prior_judge", "Prior judge")]:
                row = task*2 + (family=="prior_judge")
                labels.append(TASKS[task] + ": " + cue)
                contrast = "High vs low reader cue" if family=="social" else ("Selected vs rejected prior judge cue" if track=="choice" else "High vs low prior judge cue")
                for j, model in enumerate(MODELS):
                    r = effect(effects, model, dataset, family, contrast)
                    point(ax, 100*r["estimate"], 5-row + (.18-j*.18), 100*r["low"], 100*r["high"], model, qvalue(r)<.05)
                    plotted.append({**r, "plot_scale": 100, "panel": title})
        ax.axvline(0, color="#666666", linewidth=.8, linestyle="--")
        ax.set(yticks=list(range(5, -1, -1)), yticklabels=labels, ylim=(-.55, 5.55), title=title)
        ax.set_xlim(-1, 30 if track=="choice" else 23)
        ax.set_xlabel("High - low cue effect (pp)")
        for boundary in [1.5, 3.5]: ax.axhline(boundary, color="#eeeeee", linewidth=.6)
        ax.grid(axis="x", color="#e5e5e5", linewidth=.5); ax.set_axisbelow(True)
    model_legend(fig, -.025)
    fig.subplots_adjust(left=.18, right=.99, top=.89, bottom=.22, wspace=.77)
    save_figure(fig, "figure2_endorsement_cues", plotted)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    variant_marks = {"score_anchor_reverse": "o", "score_granularity_3": "s", "score_granularity_7": "^"}
    plotted = []
    for ax, dataset, title in zip(axes, SCORES, ["Humicroedit", "HelpSteer2", "SummEval relevance"]):
        for model in MODELS:
            for condition, marker in variant_marks.items():
                _, table = find_row(tables, "score_formats", model, dataset, condition)
                r = effect(effects, model, dataset, "score_format", condition+" minus native")
                rank = float(table[tables["score_formats"]["columns"].index("Rank correlation vs native")])
                assert .89 <= rank <= 1, (model, dataset, condition, rank)
                ax.errorbar(100*r["estimate"], rank, xerr=[[100*(r["estimate"]-r["low"])], [100*(r["high"]-r["estimate"])]] ,
                            fmt=marker, color=COLORS[model], markersize=4.8, capsize=1.8, elinewidth=.8)
                plotted.append({**r, "spearman_vs_native": rank, "condition": condition, "plot_scale": 100})
        ax.axvline(0, color="#777777", linewidth=.7, linestyle="--")
        ax.set(xlim=(-8.8, 9), ylim=(.89, 1.006), title=title, xlabel="Mean score shift (pp)")
        ax.grid(color="#eeeeee", linewidth=.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Rank agreement with native rubric")
    model_legend(fig, -.01, shapes=False)
    handles = [Line2D([0], [0], color="#555555", marker=m, linestyle="none", label=l, markersize=4.5) for m,l in [("o", "Reversed"), ("s", "3 levels"), ("^", "7 levels")]]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.54, .94), frameon=False)
    fig.subplots_adjust(left=.105, right=.99, top=.71, bottom=.25, wspace=.18)
    save_figure(fig, "figure3_score_levels_ranks", plotted)


def main_tables(tables, results):
    rows = []
    for task, pair, score in zip(TASKS, PAIRS, SCORES):
        for model in MODELS:
            _, p = find_row(tables, "baseline_pairwise", model, pair)
            _, s = find_row(tables, "baseline_scoring", model, score)
            _, pr = find_row(tables, "repeatability", model, pair)
            _, sr = find_row(tables, "repeatability", model, score)
            rows.append([task+" / "+model, f"{float(p[3].split()[0]):.2f}", f"{float(pr[2]):.2f}", f"{float(s[5]):.3f}", f"{float(s[8]):.3f}", f"{100*float(sr[2]):.2f}"])
    make_table("table1_baseline_repeat", ["Task / model", "Pair alignment / %", "Pair repeat / flip %", "Score / MAE", "Score / Spearman", "Score repeat / abs. delta pp"], rows,
               "Baseline human alignment and unchanged-request variation. Pair alignment uses strict A/B references only (UKP n=1,063; HelpSteer2 n=1,447; OpenAI n=1,451); score MAE uses normalized 0-1 human means (n=1,500). Repeat columns average four comparisons with the canonical baseline on the same items. Choice and score results use the datasets in Table 1. Values are descriptive and rounded.", "tab:results_baseline")
    rows = []
    for task, dataset in zip(TASKS, PAIRS):
        for model in MODELS:
            r = impact(results, model, dataset, "pair_rejection")
            _, tie = find_row(tables, "tie_overview", model, dataset, "pair_tie_last")
            rows.append([task+" / "+model, ci(r["selection_flip_rate"]), ci(r["human_alignment_change"], stars=True), f"{float(tie[3]):.2f}"])
    make_table("table2_framing", ["Task / model", "Rejection inconsistency / % [95% CI]", "Strict alignment change / pp [95% CI]", "X-last use / %"], rows,
               "Decision framing with unchanged texts. Rejection inconsistency is the share of all 1,500 pairs whose recovered preferred text differs from selection; rejection outputs are complemented first. Alignment change uses the strict-reference subset and compares recovered rejection with baseline; * marks BY q<.05 in the 1,656-test secondary family. X-last is a separate condition, shown descriptively on all 1,500 items. Intervals are pointwise and dependence-aware.", "tab:results_framing")


def results_overviews(tables, effects, results):
    import sys
    from paper_tables import build_tables, request_examples
    build_tables(sys.modules[__name__], tables, effects, results)
    request_examples(sys.modules[__name__])


def appendix_tables(tables, effects, results, associations):
    for outcome, key in [("probability", "appendix_factorial_probability"), ("hard choice", "appendix_factorial_choice")]:
        rows=[]
        for dataset in PAIRS:
            for model in MODELS:
                records=[effect(effects,model,dataset,"presentation_"+outcome,c) for c in ["First-position","B-label","Label × position interaction"]]
                rows.append([SHORT[dataset]+" / "+model]+[ci(r,stars=True) for r in records])
        make_table(key,["Dataset / model","First-position / pp [95% CI]","B-label / pp [95% CI]","Interaction / pp [95% CI]"],rows,
                   "Complete factorial "+outcome+" contrasts. Positive main effects favor first position or B. Interaction is (label effect with the original candidate first) minus (label effect with it second). N=1,500 per row; * marks BY q<.05 in the 369-test family. Intervals are pointwise.","tab:"+key)
    for datasets, track, key in [(PAIRS,"choice","appendix_pair_semantic"),(SCORES,"score","appendix_score_semantic")]:
        rows=[]
        for dataset in datasets:
            for model in MODELS:
                definitions=[("ai_human","Human vs AI attribution"),("social","High vs low reader cue"),("prior_judge","Selected vs rejected prior judge cue" if track=="choice" else "High vs low prior judge cue")]
                rows.append([SHORT[dataset]+" / "+model]+[ci(effect(effects,model,dataset,f,c),stars=True) for f,c in definitions])
        make_table(key,["Dataset / model","Human - AI / pp [95% CI]","Reader high - low / pp [95% CI]","Prior high - low / pp [95% CI]"],rows,
                   "Complete "+("pairwise probability" if track=="choice" else "normalized-score")+" semantic-cue contrasts, n=1,500. Cues are synthetic. "+("Pairwise contrasts reverse both assignments." if track=="choice" else "Score contrasts change metadata on one text.")+" * marks BY q<.05 in the 369-test family; pointwise CIs.","tab:"+key)
    rows=[]
    for dataset in PAIRS:
        for model in MODELS:
            _, t = find_row(tables,"pairwise_cue_effects",model,dataset,"Numeric identifier 2 vs 1")
            r=effect(effects,model,dataset,"numeric","Numeric identifier 2 vs 1")
            rows.append([SHORT[dataset]+" / "+model,ci(r,stars=True),t[4],f"{float(t[5]):.2f}"])
    make_table("appendix_numeric_pair",["Dataset / model","2 - 1 probability / pp [95% CI]","2 - 1 selection / pp [95% CI]","Reversal flips / %"],rows,
               "Pairwise numeric assignments reverse identifiers at fixed positions. Selection intervals and flips are descriptive saved summaries; only probability stars use the 369-test BY family. N=1,500.","tab:appendix_numeric_pair")
    rows=[]
    for dataset in SCORES:
        for model in MODELS:
            rows.append([SHORT[dataset]+" / "+model]+[ci(effect(effects,model,dataset,f,c),stars=True) for f,c in [("alphabetic","B vs A identifier"),("numeric","2 vs 1 identifier")]])
    make_table("appendix_score_identifiers",["Dataset / model","B - A / pp [95% CI]","2 - 1 / pp [95% CI]"],rows,
               "Scoring identifiers are shared-state candidate metadata. All effects use normalized score percentage points and n=1,500; * marks BY q<.05 in the 369-test family; pointwise CIs.","tab:appendix_score_identifiers")
    rows=[]
    for dataset in PAIRS:
        for model in MODELS:
            for target in ["first","second"]:
                r=effect(effects,model,dataset,"injection",target+" directive minus inert")
                _, t=find_row(tables,"injection_effects",model,dataset,target)
                hard = t[tables["injection_effects"]["columns"].index("Hard-choice effect pp [95% CI] / score")]
                flips = t[tables["injection_effects"]["columns"].index("Flip % / score shift >.05 %")]
                rows.append([SHORT[dataset]+" / "+model,target,ci(r,stars=True),hard,f"{float(flips):.2f}"])
    make_table("appendix_pair_injection",["Dataset / model","Target","Target probability / pp [95% CI]","Target selection / pp [95% CI]","Flips / %"],rows,
               "Directive minus same-position inert note; positive effects favor the targeted candidate. N=1,500. Probability stars use 369-test BY; hard-choice intervals and flip rates are descriptive. Both candidates share the text/metadata format; notes are approximately length matched.","tab:appendix_pair_injection")
    rows=[]
    for dataset in SCORES:
        for model in MODELS:
            r=effect(effects,model,dataset,"injection","Highest-rating directive minus inert")
            p=impact(results,model,dataset,"score_injection_directive")
            rows.append([SHORT[dataset]+" / "+model,ci(r,stars=True),f"{100*p['mean_absolute_score_change']['estimate']:.2f}",f"{100*p['score_shift_over_005_rate']['estimate']:.2f}",ci(p["human_MAE_change"],stars=True)])
    make_table("appendix_score_injection",["Dataset / model","Score delta / pp [95% CI]","Mean abs. / pp","Abs. >5 pp / %","Human MAE delta / pp [95% CI]"],rows,
               "Score directive minus inert note. Score-delta stars use 369-test BY; MAE-delta stars use 1,656-test BY. Negative MAE change means closer to the fixed human mean. Absolute shifts/rates are descriptive; n=1,500; pointwise CIs.","tab:appendix_score_injection")
    rows=[]
    for dataset in PAIRS:
        for model in MODELS:
            for condition, place in [("pair_tie_last","Last"),("pair_tie_first","First"),("pair_tie_middle","Middle")]:
                _, t=find_row(tables,"tie_overview",model,dataset,condition)
                rows.append([SHORT[dataset]+" / "+model,place,f"{float(t[3]):.2f}",f"{float(t[4]):.3f}",t[8],f"{float(t[9]):.2f}"])
    make_table("appendix_tie_placement",["Dataset / model","X position","X use / %","Mean P(X)","Conditional A delta / pp [95% CI]","Non-X flip / %"],rows,
               "All no-preference placements, n=1,500. Conditional A is P(A)/(P(A)+P(B)) versus baseline; its intervals are descriptive. Non-X flips restrict to items choosing A/B in the X condition. X-first/middle also move the original candidates' absolute positions.","tab:appendix_tie_placement")
    rows=[]
    for dataset in PAIRS:
        for model in MODELS:
            for t in tables["tie_by_reference"]["rows"]:
                if t[:3]==[model,dataset,"pair_tie_last"] and t[3] not in ["A","B"]:
                    rows.append([SHORT[dataset]+" / "+model,t[3],t[4],f"{float(t[5]):.2f}",f"{float(t[6]):.3f}"])
    make_table("appendix_tie_reference",["Dataset / model","Native reference","N","X-last use / %","Mean P(X)"],rows,
               "Non-strict native-reference cases. UKP X is a majority of neither votes; NO_MAJORITY is separate. HelpSteer ABOUT_THE_SAME is published strength zero, and OpenAI SPLIT is equal vote totals. These categories retain their source-specific meanings.","tab:appendix_tie_reference")
    for dataset, suffix in zip(SCORES,["humor","helpfulness","relevance"]):
        rows=[]
        for model in MODELS:
            for condition,label in [("score_anchor_reverse","Reverse"),("score_granularity_3","3 levels"),("score_granularity_7","7 levels")]:
                r=effect(effects,model,dataset,"score_format",condition+" minus native")
                _,t=find_row(tables,"score_formats",model,dataset,condition)
                rank = t[tables["score_formats"]["columns"].index("Rank correlation vs native")]
                mae_change = t[tables["score_formats"]["columns"].index("MAE change vs native")]
                rows.append([model,label,ci(r,stars=True),f"{float(rank):.3f}",f"{float(mae_change)*100:.2f}"])
        make_table("appendix_formats_"+suffix,["Model","Variant","Score delta / pp [95% CI]","Rank rho","Human MAE / delta pp"],rows,
                   dataset+": remapped scores versus native rubric, n=1,500. * marks 369-test BY. Rank association and saved MAE deltas are descriptive. Reversal changes index polarity; 3/7-level variants also change descriptors." + (" Humor variants additionally change the upper endpoint." if suffix == "humor" else ""),"tab:appendix_formats_"+suffix)
    rows=[]
    for dataset in SCORES:
        for model in MODELS:
            _,t=find_row(tables,"score_compression",model,dataset)
            rows.append([SHORT[dataset]+" / "+model]+[f"{float(t[i]):.3f}" for i in [2,4]]+[f"{float(t[i]):.2f}" for i in [5,6]]+[f"{float(t[i]):.3f}" for i in [7,8]])
    make_table("appendix_compression",["Dataset / model","Slope","SD ratio","Human mid / %","Judge mid / %","Human / P90-P10","Judge / P90-P10"],rows,
               "Descriptive score compression. Slope regresses judge on human mean; SD ratio is judge/human. Midpoint band is [0.4,0.6]. Dispersion and slopes compare judge scores with the human reference; n=1,500.","tab:appendix_compression")
    rows=[]
    for dataset in PAIRS+SCORES:
        for model in MODELS:
            _,t=find_row(tables,"length_associations",model,dataset)
            rows.append([SHORT[dataset]+" / "+model,t[2],t[3]])
    make_table("appendix_length",["Dataset / model","Unadjusted rho","Adjusted slope / pp per length SD [95% CI]"],rows,
               "Observational text-length associations, n=1,500. Pair outcome is P(original A), predictor log(1+words A)-log(1+words B); scoring uses normalized quality and log(1+candidate words). Slopes are per within-dataset predictor SD. Adjustment includes human reference, disagreement, agreement strata and applicable context/category/system covariates. CIs are pointwise descriptive intervals.","tab:appendix_length")
    for datasets,key in [(PAIRS,"appendix_disagreement_pair"),(SCORES,"appendix_disagreement_score")]:
        rows=[]
        for dataset in datasets:
            for model in MODELS:
                _,t=find_row(tables,"disagreement_correlations",model,dataset)
                rows.append([SHORT[dataset]+" / "+model,t[2],t[3],t[4],f"{float(t[5])*100:.2f}",f"{float(t[6])*100:.2f}"])
        pair=datasets==PAIRS
        make_table(key,["Dataset / model","Disp. vs / "+("margin rho" if pair else "prob. var. rho"),"Disp. vs / "+("sensitivity rho" if pair else "MAE rho"),"Disp. vs / repeat SD rho","Clear abs. / sensitivity pp","Disputed abs. / sensitivity pp"],rows,
                   "Descriptive Spearman associations, n=1,500 with 750 per stratum. " + ("Dispersion is vote entropy; margin is absolute A/B probability difference. Sensitivity averages absolute changes across three nonbaseline A/B cells." if pair else "Dispersion is rating SD/native range (SummEval: mean within-group SD). Probability variance describes the model's rating distribution. Sensitivity averages absolute A-identifier, B-identifier and remapped reverse-anchor changes."),"tab:"+key)
    rows=[]
    for model in MODELS:
        for ref in ["3:5 pooled","Experts only","Crowd only","Equal group means"]:
            _,t=find_row(tables,"summeval_reference_sensitivity",model,ref)
            rows.append([model,ref]+[f"{float(t[i]):.3f}" for i in [2,3,4,5,6]])
    make_table("appendix_reference_population",["Model","Reference","Mean","Signed error","MAE","Pearson","Spearman"],rows,
               "SummEval reference sensitivity on the same 1,500 summaries. The fixed primary reference is the mean of all eight ratings (3 experts, 5 crowd). Alternative references show alignment sensitivity to annotator group and aggregation.","tab:appendix_reference_population")
    rows=[]
    for dataset in PAIRS+SCORES:
        for model in MODELS:
            a=next(r for r in associations if r["model"]==model and r["dataset"]==dataset)
            r=a["magnitude_correlation"]; o=a["overlap"]
            rows.append([SHORT[dataset]+" / "+model,ci(r,scale=1,stars=True,digits=3),str(o["first_event_n"])+" / "+str(o["second_event_n"]),str(o["intersection_n"])+" / "+str(o["union_n"]),f"{o['jaccard']:.3f}"])
    make_table("appendix_shared_vulnerability",["Dataset / model","Magnitude r / [95% CI]","Reader/prior / event N","Intersect/union / N","Jaccard"],rows,
               "Reader/prior-judge susceptibility across all 18 model/dataset combinations, selected by cue identity. Pearson associations use absolute within-item contrast magnitudes and disjoint source calls. Events are choice reversals (pairs) or absolute normalized score shifts >0.05 (scores). * marks 1,656-test BY; dependence-aware Fisher-z pointwise CIs.","tab:appendix_shared_vulnerability")
    for track,key in [("choice","appendix_meta_choice"),("score","appendix_meta_score")]:
        rows=[]
        for row in results["cross_task_meta"]:
            if row["track"]!=track:continue
            p=row["pooled"]
            label={"Numeric identifier 2 vs 1":"Numeric","2 vs 1 identifier":"Numeric","B vs A identifier":"B-A","Human vs AI attribution":"Human-AI","High vs low reader cue":"Reader","Selected vs rejected prior judge cue":"Prior","High vs low prior judge cue":"Prior","first directive minus inert":"Inject first","second directive minus inert":"Inject second","Highest-rating directive minus inert":"Inject","score_anchor_reverse minus native":"Reverse"}.get(row["contrast"],row["contrast"])
            rows.append([row["model"]+" / "+label,ci(p,scale=1,stars=True),f"{p['Q']:.2f}",f"{p['I2_percent']:.1f}",f"{p['tau2']:.3f}"])
        make_table(key,["Model / contrast","Pooled pp / [95% CI]","Q","I² / %","τ² / pp²"],rows,
                   "All "+track+" cross-task pools: three tasks each, REML between-task variance and modified Knapp-Hartung t inference (df=2); Q df=2. I2 uses REML-consistent typical sampling variance. " + ("* marks 1,656-test BY. " if any(r[1].endswith('*') for r in rows) else "No pooled contrast meets the 1,656-test BY threshold. ") + "CIs are pointwise; pooling summarizes the three selected tasks. Weights and all leave-one-task-out results are retained electronically.","tab:"+key)
    # Small robustness table defined by headline contrasts, not by whichever check looked best.
    rows=[]
    for model in MODELS:
        for contrast in ["First-position","B-label","Human vs AI","Reader high vs low"]:
            _,t=find_row(tables,"dependence_robustness",model,PAIRS[0],contrast)
            rows.append([model+" / "+contrast,t[3],t[4],t[5],f"{float(t[6]):.2f}"])
    make_table("appendix_dependence",["Model / UKP contrast","Dyadic / pp [95% CI]","Component / pp [95% CI]","Bootstrap / pp [95% CI]","Omit largest / pp"],rows,
               "Dependence sensitivity for all models and four UKP headline contrasts; 2,000 bootstrap draws with a fixed seed. The largest connected joke group has 207 pairs. Results are sensitivity diagnostics.","tab:appendix_dependence")


def export_complete(effects, results, tables):
    def csv_rows(path, rows):
        columns=list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,columns);writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    csv_rows(GEN/"all_369_contrasts.csv",effects)
    csv_rows(GEN/"all_450_practical_impacts.csv",results["practical_impact"])
    csv_rows(GEN/"all_648_vulnerability_associations.csv",results["shared_vulnerability"]["associations"])
    csv_rows(GEN/"all_45_meta_analyses.csv",results["cross_task_meta"])




def audit_presentation(tables, effects, results):
    """Check display semantics against named source fields, not just numeric ranges."""
    checks=[]
    checked_cells=0
    def displayed(key):
        with (GEN/(key+".csv")).open(encoding="utf-8",newline="") as stream:
            return list(csv.reader(stream))[1:]
    def row(table, *prefix):
        _, values=find_row(tables,table,*prefix)
        return dict(zip(tables[table]["columns"],values))
    def same(actual, expected):
        nonlocal checked_cells
        assert actual==expected,(actual,expected)
        checked_cells+=len(expected)
    for data, sourceset in [(read(GEN/"figure1_position_identifier_data.json"),None),
                            (read(GEN/"figure2_endorsement_cues_data.json"),None),
                            (read(GEN/"figure3_score_levels_ranks_data.json"),"score_formats")]:
        for pointdata in data:
            src=effect(effects,pointdata['model'],pointdata['dataset'],pointdata['family'],pointdata['contrast'])
            assert all(pointdata[k]==v for k,v in src.items())
            if sourceset:
                s=row(sourceset,pointdata['model'],pointdata['dataset'],pointdata['condition'])
                assert pointdata['spearman_vs_native']==float(s['Rank correlation vs native'])
        checks.append({"check":"Figure source fields", "points":len(data),"result":"PASS"})
    expected=[]
    for task,pair,score in zip(TASKS,PAIRS,SCORES):
        for model in MODELS:
            p=row('baseline_pairwise',model,pair);s=row('baseline_scoring',model,score)
            pr=row('repeatability',model,pair);sr=row('repeatability',model,score)
            field='Baseline/repeat flip % (pairs) / mean abs delta (scores)'
            expected.append([task+' / '+model,f"{float(p['Human alignment % [95% CI]'].split()[0]):.2f}",
                             f"{float(pr[field]):.2f}",f"{float(s['MAE']):.3f}",f"{float(s['Spearman']):.3f}",f"{100*float(sr[field]):.2f}"])
    for a,b in zip(displayed('table1_baseline_repeat'),expected):same(a,b)
    for a,(dataset,model) in zip(displayed('table2_framing'),[(d,m) for d in PAIRS for m in MODELS]):
        r=impact(results,model,dataset,'pair_rejection');t=row('tie_overview',model,dataset,'pair_tie_last')
        same(a,[TASKS[PAIRS.index(dataset)]+' / '+model,ci(r['selection_flip_rate']),ci(r['human_alignment_change'],stars=True),f"{float(t['X selected %']):.2f}"])
    for a,(dataset,model,target) in zip(displayed('appendix_pair_injection'),[(d,m,t) for d in PAIRS for m in MODELS for t in ['first','second']]):
        s=row('injection_effects',model,dataset,target)
        r=effect(effects,model,dataset,'injection',target+' directive minus inert')
        same(a,[SHORT[dataset]+' / '+model,target,ci(r,stars=True),s['Hard-choice effect pp [95% CI] / score'],f"{float(s['Flip % / score shift >.05 %']):.2f}"])
    for dataset,suffix in zip(SCORES,['humor','helpfulness','relevance']):
        definitions=[('score_anchor_reverse','Reverse'),('score_granularity_3','3 levels'),('score_granularity_7','7 levels')]
        for a,(model,condition,label) in zip(displayed('appendix_formats_'+suffix),[(m,c,l) for m in MODELS for c,l in definitions]):
            s=row('score_formats',model,dataset,condition)
            r=effect(effects,model,dataset,'score_format',condition+' minus native')
            same(a,[model,label,ci(r,stars=True),f"{float(s['Rank correlation vs native']):.3f}",f"{float(s['MAE change vs native'])*100:.2f}"])
    checks.append({"check":"Baseline, framing, directive and format named-field checks","cells":checked_cells,"result":"PASS"})
    # Every direct table-derived prose fact must resolve to its declared column.
    claims_checked=0
    for claim in CLAIMS.values():
        for ref in claim['sources']:
            sel=ref['selector']
            if Path(ref['file']).name=='tables.json' and 'row' in sel:
                table=tables[sel['table']]
                value=table['rows'][sel['row']][table['columns'].index(ref['field'])]
                assert value==claim['value']
                claims_checked+=1
            elif Path(ref['file']).name=='level1_effects.json' and isinstance(claim['value'],dict):
                r=effect(effects,sel['model'],sel['dataset'],sel['family'],sel['contrast'])
                assert all(r[k]==v for k,v in claim['value'].items())
                claims_checked+=1
    assert claims_checked >= 50, claims_checked
    checks.append({"check":"Direct prose facts resolve to declared sources","claims":claims_checked,"result":"PASS"})
    # Audit the remaining derived facts using named fields from the frozen ledger.
    prior=effect(effects,'Jev',PAIRS[2],'prior_judge','Selected vs rejected prior judge cue')
    assert CLAIMS['jev_prior_sum_q']['value']==prior['q_by_validated']
    for model,dataset,key in [('Clef',PAIRS[1],'clef_reject_hs'),('Clef',PAIRS[2],'clef_reject_sum'),('Jev',PAIRS[2],'jev_reject_sum')]:
        assert CLAIMS[key]['value']==impact(results,model,dataset,'pair_rejection')['human_alignment_change']
    assert CLAIMS['clef_rejection_same_hs']['value']==impact(results,'Clef',PAIRS[1],'pair_rejection')['selection_flip_rate']
    for key,contrast in [('meta_b','B-label'),('meta_pos','First-position'),('meta_reader','High vs low reader cue')]:
        m=next(r for r in results['cross_task_meta'] if (r['model'],r['track'],r['contrast'])==('Jev','choice',contrast))
        assert CLAIMS[key]['value']==m['pooled'] and CLAIMS[key+'_i2']['value']==m['pooled']['I2_percent']
    ranks=[float(dict(zip(tables['score_formats']['columns'],r))['Rank correlation vs native']) for r in tables['score_formats']['rows'] if r[0]=='Jev']
    assert CLAIMS['jev_format_rank_range']['value']==[min(ranks),max(ranks)]
    identifiers=[abs(r['estimate']*100) for r in effects if r['model']=='Jev' and r['dataset'] in SCORES and r['family'] in ['alphabetic','numeric']]
    assert CLAIMS['jev_score_id_max']['value']==max(identifiers)
    associations=[r for r in results['shared_vulnerability']['associations'] if r['first']=='Reader high vs low' and r['second'] in ['Prior selected vs rejected','Prior high vs low']]
    magnitudes=[r['magnitude_correlation']['estimate'] for r in associations]
    assert len(associations)==18 and all(qvalue(r['magnitude_correlation'])<.05 for r in associations)
    assert CLAIMS['reader_prior_range']['value']==[min(magnitudes),max(magnitudes)]
    assert claims_checked+14==len(CLAIMS)==69
    checks.append({"check":"Remaining derived prose facts: extrema, practical impacts, q and meta summaries","claims":14,"result":"PASS"})
    assert all(qvalue(effect(effects,'Jev',d,'presentation_probability',c))<.05 for d in PAIRS for c in ['First-position','B-label'])
    assert all(qvalue(effect(effects,m,d,'social','High vs low reader cue'))<.05 for m in MODELS for d in PAIRS+SCORES)
    assert all(qvalue(effect(effects,'Jev',d,'presentation_probability','Label × position interaction'))>=.05 for d in PAIRS)
    assert sum(qvalue(r['pooled'])<.05 for r in results['cross_task_meta'])==1
    assert sum(int(row('repeatability','Perplexity',d)['Exact vector matches among 4 repeats'])==4 for d in PAIRS+SCORES)==5
    assert len(effects)==369 and len(results['practical_impact'])==450 and len(results['shared_vulnerability']['associations'])==648 and len(results['cross_task_meta'])==45
    checks.append({"check":"Directional/discovery, exact-repeat and complete-export prose claims","result":"PASS"})
    write_json(GEN/'presentation_checks.json',checks)


def verify_and_manifest(used):
    assert len(CLAIMS)>=40 and used
    for name in used:
        r=CLAIMS[name]
        assert r["sources"] and r["rendered"]
        for s in r["sources"]: assert (ROOT/s["file"]).is_file()
    data_files=list(GEN.glob("figure*_data.json"))
    for path in data_files:
        data=read(path)
        assert data and all(r["n"]==1500 for r in data)
        for r in data: assert r["low"] <= r["estimate"] <= r["high"]
    assert len(read(GEN/"figure1_position_identifier_data.json"))==18
    assert len(read(GEN/"figure2_endorsement_cues_data.json"))==36
    assert len(read(GEN/"figure3_score_levels_ranks_data.json"))==27
    assert len(DISPLAYS)==len({d['id'] for d in DISPLAYS})
    inputs=[VALID/f for f in ["level1_effects.json","level1_audit.json","level2_results.json","level2_audit.json","manifest.json"]]
    inputs += [LEGACY/"tables.json",ROOT/"scripts/analysis_level1.py",ROOT/"scripts/analysis_level2.py"]
    files=[p for p in HERE.rglob("*") if p.is_file() and p.suffix in [".md",".py",".tex",".csv",".json",".pdf",".png",".svg"] and p.name not in ["presentation_manifest.json"] and "qa" not in p.parts and "__pycache__" not in p.parts and ("output" not in p.relative_to(HERE).parts or p == HERE / "output/current/manuscript.pdf")]
    write_json(GEN/"presentation_manifest.json",{
        "purpose":"Deterministic presentation of previously validated results; no new inference or API calls",
        "input_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
        "output_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in files},
        "numeric_claims":len(CLAIMS),"used_prose_claims":sorted(used),"displays":DISPLAYS,
        "checks":["Validated source hashes and zero failed source checks","Unique contrast and row mappings","All plotted point estimates lie within their saved intervals","18 factorial, 36 cue, 27 format display points","Complete electronic exports: 369,450,648,45 rows","No unresolved numerical placeholders","Pointwise CIs distinguished from BY discovery rules"],
        "unresolved":["Human references fixed; shared workers and provider-time covariance not fully modelled","Three-task pools are secondary, not task-population estimates"]})


def main():
    for directory in [GEN,PDF,TEX]:directory.mkdir(parents=True,exist_ok=True)
    tables,effects,results=sources()
    selected=make_claims(tables,effects,results)
    figures(tables,effects)
    main_tables(tables,results)
    results_overviews(tables,effects,results)
    appendix_tables(tables,effects,results,selected)
    export_complete(effects,results,tables)
    # The reconciled manuscript is authored LaTeX, not the superseded draft templates.
    write_json(GEN/"claim_evidence.json",CLAIMS)
    used=set(CLAIMS)
    audit_presentation(tables,effects,results)
    verify_and_manifest(used)
    print(f"Presentation complete: {len(DISPLAYS)} displays, {len(used)} source-linked prose facts; no inference rerun.")


if __name__=="__main__":main()
