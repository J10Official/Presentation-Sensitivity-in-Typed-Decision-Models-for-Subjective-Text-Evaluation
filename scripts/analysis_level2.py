"""Agreed cross-experiment synthesis: practical impact, shared vulnerability, REML meta-analysis.

Run once after analysis_level1.py. No network calls or historical output changes.
Only comparable signed effects are pooled across tasks; models/interfaces stay separate.
"""

import argparse
import itertools
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy import optimize, stats

import analysis_level1 as l1

ROOT, OUT, old = l1.ROOT, l1.OUT, l1.old
POOLS = {
    "choice": [
        ("presentation_probability", "First-position"), ("presentation_probability", "B-label"),
        ("numeric", "Numeric identifier 2 vs 1"), ("ai_human", "Human vs AI attribution"),
        ("social", "High vs low reader cue"), ("prior_judge", "Selected vs rejected prior judge cue"),
        ("injection", "first directive minus inert"), ("injection", "second directive minus inert"),
    ],
    "score": [
        ("alphabetic", "B vs A identifier"), ("numeric", "2 vs 1 identifier"),
        ("ai_human", "Human vs AI attribution"), ("social", "High vs low reader cue"),
        ("prior_judge", "High vs low prior judge cue"),
        ("injection", "Highest-rating directive minus inert"),
        ("score_format", "score_anchor_reverse minus native"),
    ],
}


def control_for(condition):
    """Match the pre-existing experimental control, not an indiscriminate old baseline."""
    if "injection" in condition:
        if condition.endswith("_directive"):
            return condition.replace("_directive", "_inert")
        if condition.endswith("_inert"):
            return condition.split("_injection")[0] + "_injection_clean"
        return "baseline"
    for family, neutral in [("ai_human", "anonymous"), ("social", "none"), ("prior_judge", "none")]:
        if family in condition and not condition.endswith("_" + neutral):
            return condition.split("_" + family)[0] + "_" + family + "_" + neutral
    return "baseline"


def condition_kind(condition):
    if "repeat_" in condition:
        return "reliability_control"
    if condition.endswith(("_anonymous", "_none", "_clean", "_inert")):
        return "formatting_or_inert_control"
    return "intervention"


def practical_impact(datasets, cells):
    results = []
    for model in old.MODELS:
        for name, rows in datasets.items():
            design = old.design_for(name, rows)
            choice = old.DATASETS[name]["kind"] == "choice"
            gold, _, _ = old.reference_arrays(name, rows)
            strict = np.isin(gold, ["A", "B"]) if choice else None
            conditions = old.PAIRWISE_CONDITIONS if choice else old.SCORING_CONDITIONS
            for condition in conditions:
                control = control_for(condition)
                a, b = cells[model, name, condition], cells[model, name, control]
                row = {"model": model, "dataset": old.NAMES[name], "track": "choice" if choice else "score",
                       "condition": condition, "control": control, "role": condition_kind(condition), "n": len(rows)}
                if choice:
                    inverse = condition == "pair_rejection"
                    selected = np.where(a["choice"] == "A", "B", "A") if inverse else a["choice"]
                    value = a["pB"] if inverse else a["pA"]
                    delta = value - b["pA"]
                    row.update(signed_original_A_probability_change=l1.inference(delta, design),
                               mean_absolute_probability_change=l1.inference(abs(delta), design),
                               net_original_A_selection_change=l1.inference((selected == "A").astype(float) - (b["choice"] == "A"), design),
                               selection_flip_rate=l1.inference(selected != b["choice"], design),
                               human_alignment_change=l1.inference((selected == gold).astype(float) - (b["choice"] == gold), design, strict),
                               strict_n=int(strict.sum()),
                               control_human_alignment=float(np.mean((b["choice"] == gold)[strict])),
                               changed_human_alignment=float(np.mean((selected == gold)[strict])))
                    if "tie_" in condition:
                        row.update(tie_selection_rate=float(np.mean(selected == "X")), mean_tie_probability=float(a["pX"].mean()))
                    # Audit against the independently retained selection/rejection table.
                    if inverse:
                        legacy = next(r for r in l1.read_json(l1.OLD / "tables.json")["selection_rejection"]["rows"] if r[:2] == [model, old.NAMES[name]])
                        assert abs(100 * row["human_alignment_change"]["estimate"] - float(legacy[4])) < .000051
                else:
                    delta = a["y"] - b["y"]
                    mae_delta = abs(a["y"] - gold) - abs(b["y"] - gold)
                    row.update(signed_normalized_score_change=l1.inference(delta, design),
                               mean_absolute_score_change=l1.inference(abs(delta), design),
                               human_MAE_change=l1.inference(mae_delta, design),
                               score_shift_over_005_rate=l1.inference(abs(delta) > .05, design),
                               control_human_MAE=float(np.mean(abs(b["y"] - gold))),
                               changed_human_MAE=float(np.mean(abs(a["y"] - gold))))
                results.append(row)
    return results


def vulnerability_specs(cells, model, name):
    specs = l1.effect_vectors(cells, model, name)
    specs.pop("Interaction", None)
    choice = old.DATASETS[name]["kind"] == "choice"
    b = cells[model, name, "baseline"]
    if choice:
        rejection = cells[model, name, "pair_rejection"]
        specs["Selection–rejection"] = {"values": rejection["pB"] - b["pA"],
                                        "weights": {"pair_rejection": -1, "baseline": -1}}
        tie = cells[model, name, "pair_tie_last"]
        conditional = np.divide(tie["pA"], tie["pA"] + tie["pB"], out=np.full(len(b["pA"]), .5), where=tie["pA"] + tie["pB"] > 0)
        specs["Tie conditional A/B"] = {"values": conditional - b["pA"], "weights": {"pair_tie_last": 1, "baseline": -1},
                                        "noise_note": "Repeat-based variance is only a linear-coefficient approximation for the conditional tie contrast."}
    else:
        for levels in [3, 7]:
            condition = "score_granularity_" + str(levels)
            specs[f"{levels}-level format"] = {"values": cells[model, name, condition]["y"] - b["y"],
                                                "weights": {condition: 1, "baseline": -1}}
    for label, spec in specs.items():
        if not choice:
            spec["event"] = abs(spec["values"]) > .05
        elif label in ["First-position", "B-label"]:
            bb, ll, pp, zz = [cells[model, name, c]["choice"] for c in ["baseline", "pair_exchange_labels", "pair_exchange_positions", "pair_exchange_both"]]
            spec["event"] = ((bb != pp) | (ll != zz)) if label == "First-position" else ((bb != ll) | (pp != zz))
        elif label == "Selection–rejection":
            recovered = np.where(cells[model, name, "pair_rejection"]["choice"] == "A", "B", "A")
            spec["event"] = recovered != b["choice"]
        else:
            first, second = list(spec["weights"])
            spec["event"] = cells[model, name, first]["choice"] != cells[model, name, second]["choice"]
    return specs


def magnitude_correlation(x, y, design):
    """Pearson of absolute contrasts; cluster/dyadic delta-method Fisher-z interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.std(), y.std()
    if sx <= 1e-14 or sy <= 1e-14:
        return {"n": len(x), "estimate": None, "status": "constant_magnitude"}
    zx, zy = (x-x.mean()) / sx, (y-y.mean()) / sy
    rho = float(np.clip(np.mean(zx*zy), -1, 1))
    if abs(rho) >= 1-1e-12:
        return {"n": len(x), "estimate": rho, "status": "perfect_correlation_descriptive"}
    influence = zx*zy - rho*(zx*zx+zy*zy)/2
    influence -= influence.mean()
    fisher = l1.inference(np.arctanh(rho) + influence/(1-rho*rho), design)
    if "low" not in fisher:
        return {"n": len(x), "estimate": rho, "status": fisher["status"]}
    return {"n": len(x), "estimate": rho, "low": float(np.tanh(fisher["low"])), "high": float(np.tanh(fisher["high"])),
            "fisher_z": float(np.arctanh(rho)), "se_fisher_z": fisher["se"], "statistic": fisher["statistic"],
            "df": fisher["df"], "p": fisher["p"], "method": fisher["method"], "status": fisher["status"]}


def shared_vulnerability(datasets, cells):
    associations, excluded, noise = [], [], []
    for model in old.MODELS:
        for name, rows in datasets.items():
            design = old.design_for(name, rows)
            specs = vulnerability_specs(cells, model, name)
            choice = old.DATASETS[name]["kind"] == "choice"
            prefix, field = ("pair", "pA") if choice else ("score", "y")
            repeated = np.stack([cells[model, name, "baseline"][field]] + [cells[model, name, f"{prefix}_repeat_{i}"][field] for i in range(1, 5)])
            repeat_var = np.var(repeated, axis=0, ddof=1)
            for label, spec in specs.items():
                noise.append({"model": model, "dataset": old.NAMES[name], "contrast": label,
                              "observed_mean_absolute_effect": float(np.mean(abs(spec["values"]))),
                              "repeat_based_contrast_noise_SD": float(np.sqrt(repeat_var.mean() * sum(v*v for v in spec["weights"].values()))),
                              "note": spec.get("noise_note", "Approximation assumes equal call-noise variance and independent call errors; not a noise-corrected effect.")})
            for first, second in itertools.combinations(specs, 2):
                a, b = specs[first], specs[second]
                common = sorted(set(a["weights"]) & set(b["weights"]))
                identity = {"model": model, "dataset": old.NAMES[name], "track": "choice" if choice else "score",
                            "first": first, "second": second}
                if common:
                    covariance = repeat_var.mean() * sum(a["weights"][c] * b["weights"][c] for c in common)
                    excluded.append({**identity, "shared_cells": common, "estimated_signed_shared_call_noise_covariance": float(covariance),
                                     "reason": "Shared measurement calls can mechanically induce association; excluded from primary correlations."})
                    continue
                correlation = magnitude_correlation(abs(a["values"]), abs(b["values"]), design)
                ea, eb = a["event"], b["event"]
                intersection, union = ea & eb, ea | eb
                overlap = {"first_event_n": int(ea.sum()), "second_event_n": int(eb.sum()),
                           "intersection_n": int(intersection.sum()), "union_n": int(union.sum()),
                           "jaccard": float(intersection.sum()/union.sum()) if union.any() else None,
                           "event_definition": "underlying-choice flip" if choice else "absolute normalized score shift > .05"}
                associations.append({**identity, "n": len(rows), "first_source_calls": sorted(a["weights"]),
                                     "second_source_calls": sorted(b["weights"]),
                                     "magnitude_correlation": correlation, "overlap": overlap})
    return {"associations": associations, "excluded_shared_measurement_pairs": excluded, "repeat_noise_diagnostics": noise,
            "interpretation": "Associations of susceptibility, not shared causal mechanisms. Item quality/margins, bounded scales and temporal/provider variation can also induce association."}


def reml_pool(effects, variances):
    """One pooling method: REML tau² and modified Knapp–Hartung inference."""
    y, v = np.asarray(effects, float), np.asarray(variances, float)
    if len(y) < 2 or np.any(v <= 0) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(v)):
        return {"status": "not_poolable", "reason": "Need at least two finite effects with positive sampling variances."}
    k, df = len(y), len(y)-1
    # Standardization improves numerical conditioning, not substantive effect weighting.
    scale = max(float(np.std(y, ddof=1)), float(np.sqrt(v.max())), 1e-12)
    z, u = (y-y.mean())/scale, v/scale**2
    def objective(tau):
        w = 1/(u+tau)
        mu = np.dot(w, z)/w.sum()
        return .5*(np.log(u+tau).sum() + math.log(w.sum()) + np.dot(w, (z-mu)**2))
    upper = max(10*float(np.var(z, ddof=1)), 10*float(u.max()), 1.0)
    fit = optimize.minimize_scalar(objective, bounds=(0, upper), method="bounded", options={"xatol": 1e-12, "maxiter": 500})
    assert fit.success, fit.message
    tau_scaled = 0.0 if objective(0) <= fit.fun else float(fit.x)
    tau2 = tau_scaled*scale**2
    w = 1/(v+tau2)
    mean = float(np.dot(w, y)/w.sum())
    residual = y-mean
    kh_scale = max(1.0, float(np.dot(w, residual**2)/df))
    se = math.sqrt(kh_scale/w.sum())
    critical = float(stats.t.ppf(.975, df))
    statistic = mean/se
    p = float(2*stats.t.sf(abs(statistic), df))
    wi = 1/v  # Used only to define Q and typical sampling variance, not another pooled result.
    qcenter = np.dot(wi, y)/wi.sum()
    q = float(np.dot(wi, (y-qcenter)**2))
    typical_v = df*wi.sum()/(wi.sum()**2-np.dot(wi, wi))
    i2 = 100*tau2/(tau2+typical_v)
    scaled_w = 1/(u+tau_scaled)
    scaled_mu = np.dot(scaled_w, z)/scaled_w.sum()
    derivative = .5*(scaled_w.sum() - np.dot(scaled_w, scaled_w)/scaled_w.sum()
                     - np.dot(scaled_w**2, (z-scaled_mu)**2))
    derivative_relative = derivative/max(scaled_w.sum(), 1e-12)
    return {"status": "ok", "method": "REML + modified Knapp–Hartung", "k": k, "estimate": mean,
            "se": se, "low": mean-critical*se, "high": mean+critical*se, "df": df, "statistic": statistic, "p": p,
            "weights": (w/w.sum()).tolist(), "tau2": tau2, "Q": q, "Q_df": df,
            "Q_p": float(stats.chi2.sf(q, df)), "I2_percent": float(i2),
            "I2_definition": "100*tau²/(tau²+typical within-task variance), REML-consistent",
            "typical_within_task_variance": float(typical_v), "KH_variance_multiplier": kh_scale,
            "reml_derivative_relative": float(derivative_relative), "tau_at_boundary": tau_scaled == 0}


def cross_task_meta(effects):
    results = []
    for model in old.MODELS:
        for track, definitions in POOLS.items():
            names = {old.NAMES[n] for n in old.DATASETS if old.DATASETS[n]["kind"] == track}
            for family, contrast in definitions:
                sources = [r for r in effects if r["model"] == model and r["dataset"] in names and r["family"] == family and r["contrast"] == contrast]
                assert len(sources) == 3 and len({r["dataset"] for r in sources}) == 3, (model, track, family, contrast)
                # Pool raw differences in probability pp or normalized-score pp, not SMDs.
                y = np.array([r["estimate"]*100 for r in sources])
                v = np.array([r["variance"]*10000 for r in sources])
                pooled = reml_pool(y, v)
                row = {"model": model, "track": track, "family": family, "contrast": contrast,
                       "unit": "probability percentage points" if track == "choice" else "normalized-score percentage points",
                       "tasks": [{"dataset": r["dataset"], "n": r["n"], "effect_pp": float(a), "variance_pp2": float(b),
                                  "weight": pooled.get("weights", [None]*3)[i]} for i, (r, a, b) in enumerate(zip(sources, y, v))],
                       "pooled": pooled}
                if pooled["status"] == "ok":
                    row["leave_one_task_out"] = [{"omitted": r["dataset"], **reml_pool(np.delete(y, i), np.delete(v, i))} for i, r in enumerate(sources)]
                results.append(row)
    return results


def synthetic_checks():
    same = reml_pool([0, 0, 0], [1, 1, 1])
    assert same["tau2"] == 0 and same["df"] == 2 and same["Q"] == 0
    assert np.isclose(same["se"], 1/np.sqrt(3)) and same["p"] == 1
    varying = reml_pool([1, 2, 3], [.1, .1, .1])
    assert np.isclose(varying["tau2"], .9, atol=2e-7)
    assert np.isclose(varying["estimate"], 2) and np.isclose(varying["Q"], 20)
    assert np.isclose(varying["I2_percent"], 90, atol=1e-5)
    assert np.isclose(varying["se"], 1/np.sqrt(3), atol=1e-7)
    moved = reml_pool([10, 20, 30], [10, 10, 10])
    assert np.isclose(moved["estimate"], 10*varying["estimate"])
    assert np.isclose(moved["tau2"], 100*varying["tau2"], atol=2e-5)
    x, y = np.arange(1, 7.), np.array([1, 3, 2, 5, 6, 4.])
    rho = magnitude_correlation(x, y, {"kind": "cluster", "labels": np.arange(6)})
    assert np.isclose(rho["estimate"], np.corrcoef(x, y)[0, 1])
    assert control_for("pair_injection_second_directive") == "pair_injection_second_inert"
    assert control_for("score_ai_human_human") == "score_ai_human_anonymous"
    assert reml_pool([0, 1], [0, 1])["status"] == "not_poolable"


def audit_results(impact, vulnerability, pools, effects):
    checks, failures = [], []
    def check(label, passed, detail):
        checks.append({"check": label, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            failures.append(label)
    check("M1 inventory and denominators", len(impact) == 450 and all(r["n"] == 1500 for r in impact), "450 changed/control comparisons; 18 canonical baseline cells are controls, not additional observations")
    mean_checks = []
    for row in impact:
        if row["track"] == "choice":
            a = row["human_alignment_change"]["estimate"]
            mean_checks.append(np.isclose(a, row["changed_human_alignment"]-row["control_human_alignment"]))
            mean_checks.append(row["human_alignment_change"]["n"] == row["strict_n"])
        else:
            mean_checks.append(np.isclose(row["human_MAE_change"]["estimate"], row["changed_human_MAE"]-row["control_human_MAE"]))
    check("M1 paired practical-impact arithmetic", all(mean_checks), "Difference of means equals mean paired difference; strict-reference denominators retained")
    numerical = []
    for row in impact:
        for value in row.values():
            if not isinstance(value, dict) or value.get("status") != "ok":
                continue
            critical = stats.t.ppf(.975, value["df"])
            numerical.append(value["df"] == value["clusters_or_nodes"]-1 and
                             np.isclose(value["variance"], value["se"]**2) and
                             np.isclose(value["high"], value["estimate"]+critical*value["se"]) and
                             np.isclose(value["low"], value["estimate"]-critical*value["se"]) and
                             np.isclose(value["statistic"], value["estimate"]/value["se"]) and
                             np.isclose(value["p"], 2*stats.t.sf(abs(value["statistic"]), value["df"])))
    check("M1 CI / SE / statistic / df / p", all(numerical), f"All {len(numerical)} non-degenerate paired summaries agree; zero-variation results remain descriptive")
    check("M2 no shared-call primary edges", all(not set(r["first_source_calls"]) & set(r["second_source_calls"]) for r in vulnerability["associations"]),
          "Primary associations exclude any shared source calls; excluded edges and estimated repeat-noise covariance retained separately")
    check("M2 overlap arithmetic", all(0 <= r["overlap"]["intersection_n"] <= min(r["overlap"]["first_event_n"], r["overlap"]["second_event_n"]) <= r["overlap"]["union_n"] <= 1500 and
                                      r["overlap"]["union_n"] == r["overlap"]["first_event_n"]+r["overlap"]["second_event_n"]-r["overlap"]["intersection_n"] and
                                      (r["overlap"]["jaccard"] is None if not r["overlap"]["union_n"] else np.isclose(r["overlap"]["jaccard"], r["overlap"]["intersection_n"]/r["overlap"]["union_n"])) for r in vulnerability["associations"]),
          "Jaccard numerator/denominator counts checked; zero union is missing, not zero overlap")
    correlation_checks = []
    for row in vulnerability["associations"]:
        r = row["magnitude_correlation"]
        if r.get("status") != "ok":
            continue
        critical = stats.t.ppf(.975, r["df"])
        correlation_checks.append(r["n"] == row["n"] and
                                  np.isclose(r["fisher_z"], np.arctanh(r["estimate"])) and
                                  np.isclose(r["low"], np.tanh(r["fisher_z"]-critical*r["se_fisher_z"])) and
                                  np.isclose(r["high"], np.tanh(r["fisher_z"]+critical*r["se_fisher_z"])) and
                                  np.isclose(r["statistic"], r["fisher_z"]/r["se_fisher_z"]) and
                                  np.isclose(r["p"], 2*stats.t.sf(abs(r["statistic"]), r["df"])))
    check("M2 Fisher-z CI / SE / statistic / p", all(correlation_checks), f"All {len(correlation_checks)} non-degenerate correlation intervals/tests agree")
    weight_ok, math_ok, optimum_ok, source_ok = [], [], [], []
    all_inference = []
    for row in pools:
        p = row["pooled"]
        if p["status"] != "ok":
            continue
        y = np.array([r["effect_pp"] for r in row["tasks"]]); v = np.array([r["variance_pp2"] for r in row["tasks"]])
        w = 1/(v+p["tau2"])
        weight_ok.append(np.allclose(w/w.sum(), p["weights"]) and np.isclose(np.dot(p["weights"], y), p["estimate"]))
        q_alternative = np.sum((y-np.average(y, weights=1/v))**2/v)
        wi = 1/v
        typical_v = 2*wi.sum()/(wi.sum()**2-np.dot(wi, wi))
        kh_scale = max(1., float(np.dot(w, (y-p["estimate"])**2)/2))
        math_ok.append(np.isclose(q_alternative, p["Q"]) and p["df"] == 2 and p["Q_df"] == 2 and
                       np.isclose(p["Q_p"], stats.chi2.sf(p["Q"], 2)) and
                       np.isclose(p["typical_within_task_variance"], typical_v) and
                       np.isclose(p["I2_percent"], 100*p["tau2"]/(p["tau2"]+typical_v)) and
                       np.isclose(p["KH_variance_multiplier"], kh_scale) and
                       np.isclose(p["se"], np.sqrt(kh_scale/w.sum())) and p["tau2"] >= 0)
        optimum_ok.append(p["reml_derivative_relative"] >= -1e-6 if p["tau_at_boundary"] else abs(p["reml_derivative_relative"]) < 1e-5)
        source_ok.append(len({r["dataset"] for r in row["tasks"]}) == 3 and all(r["n"] == 1500 for r in row["tasks"]))
        for task in row["tasks"]:
            source = next(r for r in effects if (r["model"], r["dataset"], r["family"], r["contrast"]) ==
                          (row["model"], task["dataset"], row["family"], row["contrast"]))
            source_ok.append(np.isclose(task["effect_pp"], source["estimate"]*100) and
                             np.isclose(task["variance_pp2"], source["variance"]*10000))
        all_inference.append(p)
        all_inference += row["leave_one_task_out"]
    check("M3 source effect-size / task weighting", all(weight_ok) and all(source_ok), "Raw pp differences; inverse (within-task variance + REML tau²) weights sum to one; no weighting by API-call count")
    check("M3 Q / I² / tau² / df", all(math_ok) and len(pools) == 45, "45 separate three-task pools; Q df=2; REML-consistent I²; tau² in pp²")
    check("M3 REML optimum", all(optimum_ok), "Boundary KKT or interior restricted-likelihood derivative checked")
    inference_ok = []
    for r in all_inference:
        critical = stats.t.ppf(.975, r["df"])
        inference_ok.append(np.isclose(r["high"]-r["estimate"], critical*r["se"]) and
                            np.isclose(r["low"], r["estimate"]-critical*r["se"]) and
                            np.isclose(r["statistic"], r["estimate"]/r["se"]) and
                            np.isclose(r["p"], 2*stats.t.sf(abs(r["statistic"]), r["df"])))
    check("M3 CI / SE / statistic / p and leave-one-out", all(inference_ok), "Full pools df=2; leave-one-task-out df=1; same REML/mKH method throughout")
    check("synthetic statistical fixtures", True, "Hand-solvable REML tau²=.9, Q=20, I²=90%; zero heterogeneity; unit scaling; correlation and control definitions passed before empirical execution")
    # Add all newly tested secondary hypotheses to one conservative dependency-safe family.
    secondary = []
    for row in impact:
        if row["role"] == "reliability_control":
            continue
        keys = ["signed_original_A_probability_change", "net_original_A_selection_change", "human_alignment_change"] if row["track"] == "choice" else ["signed_normalized_score_change", "human_MAE_change"]
        for key in keys:
            if row[key].get("p") is not None:
                secondary.append(row[key])
    secondary += [r["magnitude_correlation"] for r in vulnerability["associations"] if r["magnitude_correlation"].get("p") is not None]
    secondary += [r["pooled"] for r in pools if r["pooled"].get("p") is not None]
    for target, q in zip(secondary, l1.adjust_p([r["p"] for r in secondary], arbitrary_dependence=True)):
        target["q_by_level2"] = float(q)
    check("secondary multiple-comparison handling", all(0 <= r["q_by_level2"] <= 1 and r["q_by_level2"]+1e-15 >= r["p"] for r in secondary),
          f"One BY family of {len(secondary)} new signed-effect/correlation/pooling tests; absolute movement and overlap remain descriptive")
    return {"checks": checks, "failures": failures, "secondary_tests": len(secondary),
            "secondary_by_significant": sum(r["q_by_level2"] < .05 for r in secondary),
            "pool_count": len(pools), "meta_by_significant": sum(r["pooled"].get("q_by_level2", 1) < .05 for r in pools),
            "primary_associations": len(vulnerability["associations"]),
            "excluded_shared_call_associations": len(vulnerability["excluded_shared_measurement_pairs"]),
            "seed": l1.SEED, "random_draws": 0,
            "limitations": ["Only three task components per pool; heterogeneity is imprecise and the task sample is not random.",
                            "Provider-wave drift, shared annotators and complete human-reference uncertainty remain unmodelled.",
                            "Correlation intervals use a first-order influence-function approximation; shared-mechanism claims are not supported.",
                            "Repeat-noise covariance assumes independent equal-variance call errors; it is a diagnostic, not subtraction from observed effects.",
                            "No causal pairwise-versus-scoring comparison: datasets, rubrics and cue representations differ."]}


def write_report(level1, audit, pools, vulnerability):
    def line(check):
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        return f"| {check['check']} | {check['status']} | {detail} |"
    text = ["# Analysis validation report", "", "12 October 2026 · Offline completion pass", "",
            "## Verdict", "",
            f"All {level1['effects']} existing full-precision contrasts passed the numerical audit. "
            f"{len(level1['changed_results'])} empirical effect estimates required correction. "
            "All retained investigation families are covered. Historical data, outputs and reports were not overwritten.", "",
            "## Existing analyses", "", "| Check | Result | Evidence |", "| --- | --- | --- |"]
    text += [line(c) for c in level1["checks"]]
    text += ["", "## What changed", "",
             f"- Existing effect estimates, confidence intervals and p-values: {len(level1['changed_results'])} corrected. "
             "Exact before/after records are in `reports/validated_analysis/corrections.json`.",
             "- The legacy negative dyadic-variance IID fallback is unsafe. The validated inference function uses connected joke components instead. "
             f"Inspection of {level1['dyadic_vectors_inspected']} existing UKP contrasts found no exposure, so this defensive correction changes no existing estimate.",
             f"- Historical BH discoveries: {level1['historical_bh_significant']}/369. Dependency-safe BY discoveries: "
             f"{level1['dependency_safe_by_significant']}/369. This changes significance classification, not effect sizes; both q-values are retained.",
             "- Two implementation-only issues were repaired once each: a new synthetic variance fixture and Windows console encoding. Neither changed empirical results.",
             "- New SE, df and test-statistic fields expose the arithmetic previously implicit in the saved CI/p values. "
             "Valid results were audited, not rerun. `--rebuild` is an explicit optional full reproduction into a separate directory.", "",
             "## Agreed cross-experiment analyses", "",
             f"M1: 450 condition/control practical-impact comparisons. M2: {audit['primary_associations']} source-call-disjoint "
             f"susceptibility associations; {audit['excluded_shared_call_associations']} mechanically overlapping comparisons excluded. "
             f"M3: {audit['pool_count']} model/interface/contrast-specific cross-task pools, with leave-one-task-out checks. "
             "Each empirical synthesis was executed once; tiny synthetic fixtures are validation tests, not extra empirical runs.", "",
             "Random effects are chosen because task-specific effects need not be equal. The only pooling method is REML tau² with modified "
             "Knapp–Hartung intervals: weights = 1/(within-task variance + tau²), variance multiplier at least one, t df=k−1. "
             "Q uses within-task inverse-variance weights, Q df=k−1; I²=100×tau²/(tau²+typical within-task variance), consistent with REML. "
             "Effects are probability pp or normalized-score pp; tau² is pp². No mixed-interface, mixed-cue or overall-model pooling is performed. "
             "Method references: [Röver, Knapp & Friede (2015)](https://arxiv.org/abs/1508.01227), "
             "[metafor REML documentation](https://wviechtb.github.io/metafor/reference/rma.uni.html) and "
             "[REML-consistent I² definition](https://www.metafor-project.org/doku.php/faq).", "",
             "| Check | Result | Evidence |", "| --- | --- | --- |"]
    text += [line(c) for c in audit["checks"]]
    text += ["", f"New secondary tests use one BY family of {audit['secondary_tests']} tests; "
             f"{audit['secondary_by_significant']} meet q<.05, including {audit['meta_by_significant']} pooled effect(s). "
             "This family is distinct from the preserved historical 369-test ledger. Leave-one-out estimates and heterogeneity tests are diagnostics, "
             "not another selection of primary discoveries. All 95% confidence intervals are pointwise, not simultaneous; discovery claims use adjusted q-values.", "",
             "### Selected pooled results (all pools are in `level2_results.json`)", "",
             "| Model / track / effect | Pooled pp [95% CI] | Q (df=2) | I² % | tau² pp² |", "| --- | --- | ---: | ---: | ---: |"]
    selected = [r for r in pools if r["model"] == "Jev" and r["contrast"] in ["First-position", "B-label", "High vs low reader cue"]]
    for row in selected:
        p = row["pooled"]
        text.append(f"| {row['model']} / {row['track']} / {row['contrast']} | {p['estimate']:.3f} [{p['low']:.3f}, {p['high']:.3f}] | {p['Q']:.2f} | {p['I2_percent']:.1f} | {p['tau2']:.3f} |")
    text += ["", "## Open issues and boundaries", ""]
    text += ["- " + limitation for limitation in audit["limitations"]]
    text += ["- Scoring reversal changes order and numeric polarity; scale-size variants change wording, including some endpoint semantics. "
             "They support format-sensitivity claims, not isolated causal mechanisms.",
             "- The original table-defined investigations were planned; later secondary analyses were not preregistered. "
             "No outcome-dependent dataset exclusions or silent gold changes were made during this completion pass.", "",
             "## Reproduce", "", "From the repository root, using Python 3.11 with NumPy and SciPy:", "",
             "```sh", 'python scripts/analysis_level1.py',
             'python scripts/analysis_level2.py --audit-only', "```", "",
             "Both scripts resolve paths from their own location. Level 1 defaults to audit-first mode; "
             "`scripts/analysis_level1.py --rebuild` explicitly regenerates the legacy analysis separately. "
             "`reports/validated_analysis/manifest.json` records versions and input/output hashes; `provenance/` records the clean-package lineage. "
             "Machine-readable audits retain exact counts, formulas, SE/df/p and all heterogeneity/leave-one-out results. "
             "Use `scripts/analysis_level2.py --audit-only` to validate the saved synthesis without writing files or recomputing empirical analyses.", "",
             "## Optional follow-ups (not executed)", "",
             "- Presentation-averaged judging using the four existing position/label cells.",
             "- A nested annotation-uncertainty analysis for human-alignment claims, if the paper requires population-reference inference.",
             "- New time-counterbalanced or endpoint-matched scale experiments would require a study change; they are not missing offline computations.", ""]
    (OUT / "validation_report.md").write_text("\n".join(text), encoding="utf-8", newline="\n")


def write_manifest():
    original = l1.read_json(l1.OLD / "analysis_manifest.json")
    previous = l1.read_json(OUT / "manifest.json") if (OUT / "manifest.json").exists() else {}
    files = [OUT / name for name in ["level1_audit.json", "level1_effects.json", "level2_results.json", "level2_audit.json", "corrections.json", "level2_input_mapping_audit.json"]]
    files += [ROOT / "scripts/analysis_level1.py", ROOT / "scripts/analysis_level2.py", OUT / "validation_report.md"]
    current = {p.relative_to(ROOT).as_posix(): l1.sha256(p) for p in files}
    input_hashes = {**original["sha256"], **original["baseline_history_sha256"]}
    public_hashes = l1.read_json(ROOT / "provenance/package_manifest.json")["public_files"]
    input_hashes = {p: public_hashes.get(p, h) if p.startswith("scripts/") else h
                   for p, h in input_hashes.items()}
    mismatches = [p for p, expected in input_hashes.items() if l1.sha256(l1.repo_path(p)) != expected]
    assert not mismatches, mismatches
    l1.save_json(OUT / "manifest.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                       "seed": l1.SEED, "random_draws": 0, "method": "REML + modified Knapp–Hartung only",
                                       "input_sha256": input_hashes, "new_artifact_sha256": current,
                                       "resolved_model_roots": {m: [p.relative_to(ROOT).as_posix() for p in old.MODELS[m][1:]] for m in old.MODELS},
                                       "historical_artifact_sha256": previous.get("historical_artifact_sha256", previous.get("new_artifact_sha256", {})),
                                       "packaging_provenance": "provenance/package_manifest.json", "empirical_level2_executions": 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Validate saved outputs without private histories, writes or empirical recomputation")
    args = parser.parse_args()
    level1 = l1.read_json(OUT / "level1_audit.json")
    assert not level1["failures"], "Run analysis_level1.py successfully first"
    effects = l1.read_json(OUT / "level1_effects.json")
    synthetic_checks()
    if args.audit_only:
        manifest = l1.read_json(OUT / "manifest.json")
        result_path = OUT / "level2_results.json"
        saved_hashes = {p.replace("\\", "/"): h for p, h in manifest["new_artifact_sha256"].items()}
        assert l1.sha256(result_path) == saved_hashes[result_path.relative_to(ROOT).as_posix()], "Saved synthesis changed since execution"
        results = l1.read_json(result_path)
        impact, vulnerability, pools = (results[k] for k in ["practical_impact", "shared_vulnerability", "cross_task_meta"])
        print("Auditing saved M1-M3 outputs; no empirical recomputation", flush=True)
    else:
        datasets = l1.load_inputs()
        cells, cell_audits, samples = l1.load_cells(datasets)
        print("Running M1: practical impact", flush=True)
        impact = practical_impact(datasets, cells)
        print("Running M2: shared vulnerability", flush=True)
        vulnerability = shared_vulnerability(datasets, cells)
        print("Running M3: one REML/mKH synthesis", flush=True)
        pools = cross_task_meta(effects)
        results = {"practical_impact": impact, "shared_vulnerability": vulnerability, "cross_task_meta": pools}
    audit = audit_results(impact, vulnerability, pools, effects)
    if args.audit_only:
        print(f"Level 2: {len(impact)} impacts, {len(vulnerability['associations'])} associations, {len(pools)} pools; {len(audit['failures'])} failures", flush=True)
        if audit["failures"]:
            raise SystemExit("Level 2 audit failures: " + ", ".join(audit["failures"]))
        return
    if not args.audit_only:
        l1.save_json(OUT / "level2_results.json", results)
        l1.save_json(OUT / "level2_input_mapping_audit.json", {"cells": cell_audits, "payload_samples": samples})
    l1.save_json(OUT / "level2_audit.json", audit)
    write_report(level1, audit, pools, vulnerability)
    write_manifest()
    print(f"Level 2: {len(impact)} impacts, {len(vulnerability['associations'])} associations, {len(pools)} pools; {len(audit['failures'])} failures", flush=True)
    if audit["failures"]:
        raise SystemExit("Level 2 audit failures: " + ", ".join(audit["failures"]))


if __name__ == "__main__":
    main()
