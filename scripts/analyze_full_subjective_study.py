"""Offline, reproducible analysis of the frozen three-model subjective study."""

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

from subjective_baseline_runner import DATASETS, digest, request_for as baseline_request
from subjective_intervention_runner import (
    PAIRWISE_CONDITIONS, SCORING_CONDITIONS, request_for as intervention_request,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'prepared_data' / 'subjective_study_1500'
OUT = ROOT / 'reports' / 'full_study_analysis'
REPORT = OUT / 'single_experiment_report.md'
MODELS = {
    'Jev': ('typesafe/jev-1.13', ROOT / 'experiment_results/subjective_baseline',
            ROOT / 'experiment_results/subjective_interventions'),
    'Clef': ('cloudflare/clef-flash', ROOT / 'experiment_results/decision_model_comparisons/clef_flash/baseline',
             ROOT / 'experiment_results/decision_model_comparisons/clef_flash/interventions'),
    'Perplexity': ('perplexity/pplx-decider-v1.1-27b',
                   ROOT / 'experiment_results/decision_model_comparisons/pplx_decider_v1_1_27b/baseline',
                   ROOT / 'experiment_results/decision_model_comparisons/pplx_decider_v1_1_27b/interventions'),
}
NAMES = {
    'ukp_humor_pairwise.jsonl': 'UKP humor',
    'helpsteer_pairwise.jsonl': 'HelpSteer pairs',
    'openai_summary_pairwise.jsonl': 'OpenAI summaries',
    'humicroedit_scoring.jsonl': 'Humicroedit',
    'helpsteer_scoring.jsonl': 'HelpSteer scores',
    'summeval_relevance_scoring.jsonl': 'SummEval relevance',
}
INFERENCE = []
TABLES = {}
VARIANCE_FALLBACKS = Counter()


def read_rows(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def fmt(value):
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return 'NA' if not np.isfinite(value) else f'{value:.4f}'
    return str(value).replace('|', '/').replace('\n', ' ')


def table(name, headers, rows):
    rows = list(rows)
    TABLES[name] = {'columns': headers, 'rows': [[fmt(v) for v in row] for row in rows]}
    result = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    result += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in rows]
    return '\n'.join(result) + '\n\n'


def append(text):
    with REPORT.open('a', encoding='utf-8') as target:
        target.write(text)


def save_tables():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tables.json').write_text(json.dumps(TABLES, indent=2), encoding='utf-8')
    (OUT / 'contrasts.json').write_text(json.dumps(INFERENCE, indent=2, allow_nan=False), encoding='utf-8')


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else float('nan')


def corr(a, b, rank=False):
    a, b = np.asarray(a), np.asarray(b)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.ptp(a[valid]) == 0 or np.ptp(b[valid]) == 0:
        return float('nan')
    return float((stats.spearmanr if rank else stats.pearsonr)(a[valid], b[valid]).statistic)


def entropy(p):
    return -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-300, None)), 0), axis=1)


def graph_components(rows, labels=False):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in rows:
        a, b = row['metadata']['candidate_ids']
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = [find(row['metadata']['candidate_ids'][0]) for row in rows]
    return groups if labels else Counter(groups)


def design_for(name, rows):
    if name == 'ukp_humor_pairwise.jsonl':
        nodes = [row['metadata']['candidate_ids'] for row in rows]
        return {'kind': 'dyadic', 'nodes': nodes, 'groups': len(set(x for pair in nodes for x in pair))}
    key = {'humicroedit_scoring.jsonl': 'original_headline_id',
           'helpsteer_scoring.jsonl': 'prompt_id',
           'openai_summary_pairwise.jsonl': 'context_id',
           'summeval_relevance_scoring.jsonl': 'article_id'}.get(name)
    groups = [row['metadata'][key] if key else digest(row['input']['context']) for row in rows]
    return {'kind': 'cluster', 'labels': np.array(groups), 'groups': len(set(groups))}


def interval(values, design, mask=None):
    values = np.asarray(values, dtype=float)
    mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    mask = mask & np.isfinite(values)
    x = values[mask]
    n = len(x)
    if n < 2:
        return (float(np.mean(x)) if n else float('nan'), float('nan'), float('nan'), float('nan'), 0)
    mean = float(x.mean())
    residual = x - mean
    if design['kind'] == 'dyadic':
        sums = defaultdict(float)
        for pair, value in zip(np.array(design['nodes'])[mask], residual):
            for node in pair:
                sums[node] += value
        g = len(sums)
        variance = (sum(value * value for value in sums.values()) - np.dot(residual, residual)) / n**2
        variance *= g / (g - 1) if g > 1 else 1
    else:
        labels = design['labels'][mask]
        _, inv = np.unique(labels, return_inverse=True)
        sums = np.bincount(inv, weights=residual)
        g = len(sums)
        variance = float(np.dot(sums, sums)) / n**2 * (g / (g - 1) if g > 1 else 1)
    if variance < 0:
        # Dyadic sandwich can be negative in finite samples; explicitly use an IID fallback.
        VARIANCE_FALLBACKS[(design['kind'], n, g)] += 1
        variance = float(np.var(x, ddof=1) / n)
    se = math.sqrt(variance)
    cutoff = stats.t.ppf(.975, max(g - 1, 1))
    p = float(2 * stats.t.sf(abs(mean / se), max(g - 1, 1))) if se else (1.0 if mean == 0 else 0.0)
    return mean, mean - cutoff * se, mean + cutoff * se, p, g


def ci_text(result, scale=1):
    mean, lo, hi, _, _ = result
    return f'{mean * scale:.3f} [{lo * scale:.3f}, {hi * scale:.3f}]'


def contrast(model, name, family, label, x, design, mask=None, scale=100):
    result = interval(x, design, mask)
    mask = np.isfinite(x) if mask is None else np.asarray(mask) & np.isfinite(x)
    INFERENCE.append({'model': model, 'dataset': NAMES[name], 'family': family, 'contrast': label,
                      'n': int(mask.sum()), 'estimate': result[0], 'low': result[1], 'high': result[2],
                      'p': result[3], 'clusters_or_nodes': result[4]})
    return ci_text(result, scale)


def cell_path(model, name, condition):
    route, base, inter = MODELS[model]
    return (base if condition == 'baseline' else inter / condition) / Path(name).stem


def canonical_baseline(model, name, rows):
    expected = {row['id']: digest(baseline_request(name, row, MODELS[model][0])) for row in rows}
    selected = {}
    for record in read_rows(cell_path(model, name, 'baseline') / 'full_results.jsonl'):
        if not record.get('ok') or record.get('request_sha256') != expected.get(record['id']):
            continue
        ident = record['id']
        if ident not in selected or (record.get('started_at') or '') < (selected[ident].get('started_at') or ''):
            selected[ident] = record
    assert len(selected) == len(rows), (model, name, 'baseline raw history incomplete')
    result = []
    for row in rows:
        raw = selected[row['id']]
        parsed = raw['parsed']
        result.append({**parsed, 'id': row['id'], 'ok': raw['ok'], 'usage': raw['usage'],
                       'response_model': raw['response_model'], 'reference_choice': row['reference'].get('choice')})
    return result


def load_cell(model, name, condition, rows, audits, full_samples):
    directory = cell_path(model, name, condition)
    data = read_rows(directory / 'simple.jsonl')
    ids = [row['id'] for row in rows]
    assert [x['id'] for x in data] == ids, (model, name, condition, 'ID/order mismatch')
    assert len(set(ids)) == 1500 and all(x['ok'] for x in data)
    canonical_changed = 0
    if condition == 'baseline':
        canonical = canonical_baseline(model, name, rows)
        canonical_changed = sum(x.get('choice') != y.get('choice') or x.get('api_score') != y.get('api_score')
                                or x.get('probabilities') != y.get('probabilities') for x, y in zip(canonical, data))
        data = canonical
    kind = DATASETS[name]['kind']
    labels = list(data[0]['probabilities'])
    assert all(list(record['probabilities']) == labels for record in data), (model, name, condition, 'probability key order')
    rawp = np.array([[x['probabilities'][label] for label in labels] for x in data], dtype=float)
    assert np.all(np.isfinite(rawp)) and rawp.min() >= 0 and rawp.max() <= 1
    mass = rawp.sum(axis=1)
    assert np.all(mass > 0)
    p = rawp / mass[:, None]
    a = {'p': p, 'rawp': rawp, 'labels': labels, 'mass': mass,
         'confidence': np.array([finite(x.get('confidence')) for x in data]),
         'input_tokens': np.array([(x.get('usage') or {}).get('input_tokens', 0) for x in data]),
         'cost': np.array([(x.get('usage') or {}).get('cost', 0) for x in data])}
    if kind == 'choice':
        mapping = ({'A': 'A', 'B': 'B'} if condition == 'baseline' else data[0]['option_to_original'])
        if condition != 'baseline':
            expected_mapping = intervention_request(name, rows[0], condition, MODELS[model][0])[1]['option_to_original']
            assert mapping == expected_mapping
            assert all(record['option_to_original'] == mapping for record in data)
        expected_order = ['A', 'B'] if condition == 'baseline' else list(expected_mapping)
        assert all(record['option_order'] == expected_order and set(record['option_order']) == set(labels)
                   for record in data), (model, name, condition, 'option metadata order')
        a['choice'] = np.array([mapping.get(x['choice']) or 'X' for x in data])
        a['pA'] = sum(p[:, i] for i, label in enumerate(labels) if mapping.get(label) == 'A')
        a['pB'] = sum(p[:, i] for i, label in enumerate(labels) if mapping.get(label) == 'B')
        a['pX'] = 1 - a['pA'] - a['pB'] if 'X' in labels else np.zeros(len(rows))
        a['first'] = np.array([x['choice'] == x['option_order'][0] for x in data])
        indices = np.array([labels.index(x['choice']) for x in data])
        mismatch = rawp.max(axis=1) - rawp[np.arange(len(rows)), indices]
        discrepancy = mismatch
    else:
        count = len(labels)
        assert labels == [str(i) for i in range(count)]
        score = np.array([x['api_score'] for x in data])
        a['y'] = (1 - score / (count - 1)) if condition == 'score_anchor_reverse' else score / (count - 1)
        quality = (np.arange(count)[::-1] if condition == 'score_anchor_reverse' else np.arange(count)) / (count - 1)
        a['prob_y'] = p @ quality
        a['variance'] = p @ (quality**2) - a['prob_y']**2
        discrepancy = abs(score - rawp @ np.arange(count))
        mismatch = np.zeros(len(rows))
    a['entropy'] = entropy(p) / math.log(len(labels))
    audits.append([model, NAMES[name], condition, len(data), ', '.join(sorted(set(x['response_model'] for x in data))),
                   int(np.sum(abs(mass - 1) > .005)), float(np.max(abs(mass - 1))),
                   int(np.sum(discrepancy > .01 + 1e-8)), int(np.sum(discrepancy > .1 + 1e-8)),
                   int(np.sum(mismatch > 1e-8)), canonical_changed,
                   float(a['cost'].sum()), int(a['input_tokens'].sum())])
    # Check one actual successful full checkpoint per cell against the current frozen builders.
    found = False
    expected_ids = set(ids)
    with (directory / 'full_results.jsonl').open(encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            if not record.get('ok') or record['id'] not in expected_ids:
                continue
            row = rows[ids.index(record['id'])]
            expected = (baseline_request(name, row, MODELS[model][0]) if condition == 'baseline'
                        else intervention_request(name, row, condition, MODELS[model][0])[0])
            if record['request_sha256'] != digest(expected):
                continue
            assert record['request'] == expected and digest(record['request']) == record['request_sha256']
            assert json.loads(record['request_body_utf8']) == expected
            full_samples.append([model, NAMES[name], condition, record['id'], record['request_sha256'],
                                 record.get('started_at'), record.get('finished_at')])
            found = True
            break
    assert found, (model, name, condition, 'no matching full checkpoint')
    return a


def load_all():
    datasets = {name: read_rows(INPUT / name) for name in DATASETS}
    designs = {name: design_for(name, rows) for name, rows in datasets.items()}
    cells, audits, samples = {}, [], []
    for model in MODELS:
        for name, rows in datasets.items():
            conditions = ('baseline',) + (PAIRWISE_CONDITIONS if DATASETS[name]['kind'] == 'choice' else SCORING_CONDITIONS)
            for condition in conditions:
                cells[model, name, condition] = load_cell(model, name, condition, rows, audits, samples)
            print(f'Loaded {model}: {NAMES[name]} ({len(conditions)} conditions)', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cell_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    (OUT / 'payload_sample_audit.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    return datasets, designs, cells, audits


def reference_arrays(name, rows):
    if DATASETS[name]['kind'] == 'choice':
        gold = np.array([row['reference']['choice'] for row in rows])
        human = np.array([Counter(row['reference']['individual_choices']).get('A', 0) /
                          len(row['reference']['individual_choices']) for row in rows])
        counts = [Counter(row['reference']['individual_choices']) for row in rows]
        h_entropy = np.array([-sum((v / sum(c.values())) * math.log(v / sum(c.values())) for v in c.values())
                              for c in counts])
        return gold, human, h_entropy
    lo = DATASETS[name]['native_minimum']; span = len(DATASETS[name]['criteria']) - 1
    gold = np.array([(row['reference']['score'] - lo) / span for row in rows])
    dispersion = np.array([row['metadata']['human_sd'] / span for row in rows])
    return gold, None, dispersion


def cycle1(datasets, designs, cells, audits):
    append('\n## Cycle 1 — completeness, reference audit, baseline and repeatability\n\n')
    append('### Analysis conventions and uncertainty\n\n'
           'Canonical baselines use the earliest-started valid response matching the frozen request hash, '
           'recomputed against the current human reference. This avoids counting accidental duplicate baseline calls as planned repetitions. '
           'All intervention results are joined by immutable item ID and checked in original dataset order. '
           'Choice probabilities are divided by their own total for probabilistic analyses; the original raw vector is retained in the logs. '
           'Scores use the provider-reported expected score mapped through the supplied rubric, normalized to [0,1]. '
           'Reversed anchors are mapped back to increasing quality. Hard choices retain the provider choice, including a small allowed non-argmax discrepancy.\n\n'
           'Intervals are two-sided 95% intervals for paired item-level means. OpenAI summary comparisons cluster by source context; '
           'SummEval clusters by article; HelpSteer pairs cluster by exact prompt; single-edit/single-response scored components cluster by headline/prompt. '
           'UKP uses a dyadic sandwich variance allowing correlation between pairs sharing either joke: '
           'variance(mean) = [sum over jokes of squared residual sums − sum over pairs of squared residuals] / n², '
           'with a node-count correction. This assumes disjoint pairs are independent; it does not model shared raters or all semantic similarity. '
           'If the finite-sample dyadic variance is negative, the code uses an IID variance fallback; such comparisons warrant sensitivity analysis. '
           'Cluster intervals use a t critical value with cluster/node count minus one degrees of freedom. '
           'These intervals condition on the selected benchmark and fixed human references. They do not propagate all human-annotation uncertainty. '
           'Zero variation yields a degenerate descriptive interval, not proof of universal invariance.\n\n')
    meta = []
    for name, rows in datasets.items():
        groups = Counter(row['metadata']['agreement_group'] for row in rows)
        depth = [row['metadata']['annotation_depth'] for row in rows]
        refs = Counter(row['reference'].get('choice', 'score') for row in rows)
        d = designs[name]
        meta.append([NAMES[name], len(rows), groups['clear'], groups['disputed'], f'{min(depth)}–{max(depth)}',
                     d['kind'], d['groups'], str(dict(refs))])
    append(table('dataset_design', ['Dataset', 'N', 'Clear', 'Disputed', 'Raters/item', 'Dependence', 'Clusters/nodes', 'Reference counts'], meta))
    reuse = []
    for name, rows in datasets.items():
        if DATASETS[name]['kind'] != 'choice': continue
        counts = Counter(x for row in rows for x in row['metadata']['candidate_ids'])
        comps = graph_components(rows)
        reuse.append([NAMES[name], len(counts), sum(v > 1 for v in counts.values()), max(counts.values()),
                      len(comps), max(comps.values()), f'{max(comps.values()) / len(rows):.1%}'])
    append(table('candidate_reuse', ['Dataset', 'Unique candidates', 'Reused candidates', 'Max appearances', 'Pair-graph components', 'Largest component pairs', 'Largest share'], reuse))
    aggregate = []
    for model in MODELS:
        aa = [x for x in audits if x[0] == model]
        aggregate.append([model, len(aa), sum(x[3] for x in aa), len(aa),
                          ', '.join(sorted(set(x[4] for x in aa))), sum(x[5] for x in aa),
                          max(x[6] for x in aa), sum(x[7] for x in aa), sum(x[8] for x in aa),
                          sum(x[9] for x in aa), sum(x[10] for x in aa), sum(x[11] for x in aa)])
    append(table('collection_audit', ['Model', 'Cells', 'Valid rows', 'Payloads sampled', 'Served snapshot',
                                    'Mass error >.005', 'Max mass error', 'Numeric discrepancy >.01',
                                    'Discrepancy >.1', 'Non-argmax choices', 'Canonical baseline changes', 'Accepted-output USD'], aggregate))
    append('Every intended cell must contain exactly 1,500 unique ordered IDs and all published rows must be marked successful; '
           'the script aborts on a mismatch. A full stored payload is reconstructed and checked against the request builder in each of the 468 cells. '
           'This is a systematic one-record-per-cell payload audit, not a byte-by-byte re-audit of all 702,000 requests. '
           'Accepted-output cost sums usage attached to selected outputs; it omits billed rejected attempts and accidental duplicates. '
           'The permissive 0.1 parser tolerance is audited explicitly below rather than assumed harmless. '
           'Earlier DNS failures and rejected responses remain in histories but do not count as analytical repetitions.\n\n')
    baselines = []; scored = []; groups_table = []; human_diag = []
    for name, rows in datasets.items():
        gold, human, dispersion = reference_arrays(name, rows)
        clear = np.array([row['metadata']['agreement_group'] == 'clear' for row in rows])
        if DATASETS[name]['kind'] == 'choice':
            strict = np.isin(gold, ['A', 'B'])
            if name == 'helpsteer_pairwise.jsonl':
                major = np.array(['A' if Counter(row['reference']['individual_choices'])['A'] > 2 else 'B' for row in rows])
                human_diag.append([NAMES[name], int(strict.sum()), int(np.sum(strict & (major != gold))),
                                   'Published preference-strength sign can differ from raw directional majority.'])
            for model in MODELS:
                b = cells[model, name, 'baseline']
                correct = b['choice'] == gold
                prob = np.where(gold == 'A', b['pA'], b['pB'])
                # For UKP, A/B-only human reference fractions exclude X only for this diagnostic.
                counts = [Counter(row['reference']['individual_choices']) for row in rows]
                directional = np.array([c['A'] / (c['A'] + c['B']) if c['A'] + c['B'] else np.nan for c in counts])
                baselines.append([model, NAMES[name], int(strict.sum()), ci_text(interval(correct, designs[name], strict), 100),
                                  100 * b['first'].mean(), prob[strict].mean(),
                                  corr(b['pA'], directional, True),
                                  np.nanmean((b['pA'] - directional)**2), b['entropy'].mean()])
                for group, mask in [('clear', clear), ('disputed', ~clear)]:
                    groups_table.append([model, NAMES[name], group, int(mask.sum()), int((mask & strict).sum()),
                                         100 * correct[mask & strict].mean(), np.mean(abs(b['pA'][mask] - b['pB'][mask])),
                                         corr(dispersion[mask], abs(b['pA'][mask] - b['pB'][mask]), True)])
        else:
            for model in MODELS:
                b = cells[model, name, 'baseline']; error = b['y'] - gold
                scored.append([model, NAMES[name], gold.mean(), b['y'].mean(),
                               ci_text(interval(error, designs[name])), np.mean(abs(error)),
                               math.sqrt(np.mean(error**2)), corr(b['y'], gold), corr(b['y'], gold, True),
                               np.std(b['y']) / np.std(gold)])
                for group, mask in [('clear', clear), ('disputed', ~clear)]:
                    groups_table.append([model, NAMES[name], group, int(mask.sum()), 'score',
                                         np.mean(abs(error[mask])), np.mean(b['entropy'][mask]),
                                         corr(dispersion[mask], abs(error[mask]), True)])
    append('### Baseline alignment: pairwise track\n\n'
           'Accuracy is agreement with the stored strict A/B reference; X/no-majority/aggregate-neutral cases do not define a forced-choice accuracy target. '
           'The reported mean probability is the probability of the human reference among strict cases. '
           'The fraction Brier statistic compares judge A probability to the empirical directional A share and is a descriptive distribution discrepancy, '
           'not a calibrated probability-of-correctness score. UKP excludes X only from this directional-share diagnostic.\n\n')
    append(table('baseline_pairwise', ['Model', 'Dataset', 'Strict N', 'Human alignment % [95% CI]', 'First choice %',
                                     'P(reference)', 'Spearman P(A)/human A share', 'Fraction Brier', 'Normalized entropy'], baselines))
    append('### Baseline alignment: scoring track\n\nAll score errors below use fixed native-bound normalization; '
           'one normalized unit is the full native scale. Human means are subjective references. Pearson/Spearman correlations are descriptive point estimates.\n\n')
    append(table('baseline_scoring', ['Model', 'Dataset', 'Human mean', 'Judge mean', 'Signed error [95% CI]',
                                    'MAE', 'RMSE', 'Pearson', 'Spearman', 'Judge SD / human SD'], scored))
    append(table('baseline_agreement_groups', ['Model', 'Dataset', 'Agreement group', 'N', 'Strict N / kind',
                                             'Alignment % (pairs) / MAE (scores)', 'Mean margin (pairs) / entropy (scores)',
                                             'Human dispersion vs margin/error rho'], groups_table))
    append(table('human_reference_audit', ['Dataset', 'Strict N', 'Published gold / vote-majority disagreements', 'Interpretation'], human_diag))
    repeats = []; repeat_groups = []
    for model in MODELS:
        for name, rows in datasets.items():
            kind = DATASETS[name]['kind']; prefix = 'pair' if kind == 'choice' else 'score'
            a = [cells[model, name, 'baseline']] + [cells[model, name, f'{prefix}_repeat_{i}'] for i in range(1, 5)]
            if kind == 'choice':
                choices = np.stack([x['choice'] for x in a]); values = np.stack([x['pA'] for x in a])
                flips = np.mean(choices[1:] != choices[0], axis=0)
                anyflip = np.any(choices != choices[0], axis=0)
                allpair = np.mean([choices[i] != choices[j] for i in range(5) for j in range(i + 1, 5)], axis=0)
                repeats.append([model, NAMES[name], 100 * flips.mean(), 100 * allpair.mean(), 100 * anyflip.mean(),
                                np.mean(np.std(values, axis=0, ddof=1)), np.max(np.ptp(values, axis=0)),
                                sum(np.array_equal(a[0]['p'], x['p']) and np.array_equal(a[0]['choice'], x['choice']) for x in a[1:])])
            else:
                values = np.stack([x['y'] for x in a]); absolute = np.mean(abs(values[1:] - values[0]), axis=0)
                anyflip = np.ptp(values, axis=0) > 1e-9
                repeats.append([model, NAMES[name], absolute.mean(), 'score', 100 * anyflip.mean(),
                                np.mean(np.std(values, axis=0, ddof=1)), np.max(np.ptp(values, axis=0)),
                                sum(np.array_equal(a[0]['y'], x['y']) and np.array_equal(a[0]['p'], x['p']) for x in a[1:])])
            for i in range(1, 5):
                delta = values[i] - values[0]
                repeat_groups.append([model, NAMES[name], i, ci_text(interval(delta, designs[name]), 100),
                                      float(np.mean(abs(delta))), float(np.quantile(abs(delta), .95))])
    append('### Identical-request repeatability\n\nThe five planned observations are the canonical baseline and four repeats. '
           'Pairwise repeat values are original-candidate A probabilities; scoring repeat values are normalized quality. '
           'Exact equality can arise from deterministic serving, rounding or caching; the API outputs cannot distinguish these explanations. '
           'Any variation is run/time/provider variation, not necessarily token-sampling noise.\n\n')
    append(table('repeatability', ['Model', 'Dataset', 'Baseline/repeat flip % (pairs) / mean abs delta (scores)',
                                'All-pair flip %', 'Any hard flip % (pairs) / any score change %',
                                'Mean within-item SD', 'Max within-item range', 'Exact vector matches among 4 repeats'], repeats))
    append(table('repeat_drift', ['Model', 'Dataset', 'Repeat', 'Mean delta pp [95% CI]', 'Mean abs delta', 'P95 abs delta'], repeat_groups))
    save_tables()


def cycle2(datasets, designs, cells, audits):
    append('\n## Cycle 2 — presentation factorial and reference cues\n\n'
           '### Estimands and review after cycle 1\n\n'
           'The four-condition presentation factorial is analyzed in original-text coordinates. '
           'Let b, l, p and z be original-text A probabilities under baseline, exchanged labels, exchanged positions and exchanged both. '
           'The first-position effect is (b+l−p−z)/2. The B-label effect is (l+z−b−p)/2. '
           'The interaction is (l−b)−(z−p), the difference in the B-label contrast across positions. '
           'Apply the same formulas to indicators of selecting original text A for hard-choice effects. '
           'These are probability-point or percentage-point effects; a positive label effect favors B and a positive position effect favors first. '
           'All intervals use the dependence structure described in cycle 1. '
           'Large sample counts do not remove stochastic serving differences: the repeatability table supplies their empirical scale.\n\n')
    factorial = []; factorial_groups = []; presentation = []; cue_pair = []; cue_score = []; wrappers = []
    pair_defs = [
        ('Numeric identifier 2 vs 1', 'numeric', 'pair_numeric_first_2', 'pair_numeric_first_1', None),
        ('Human vs AI attribution', 'ai_human', 'pair_ai_human_first_human', 'pair_ai_human_first_ai', 'pair_ai_human_anonymous'),
        ('High vs low reader cue', 'social', 'pair_social_first_high', 'pair_social_second_high', 'pair_social_none'),
        ('Selected vs rejected prior judge cue', 'prior_judge', 'pair_prior_judge_first_selected', 'pair_prior_judge_second_selected', 'pair_prior_judge_none'),
    ]
    score_defs = [
        ('B vs A identifier', 'alphabetic', 'score_identifier_b', 'score_identifier_a', 'baseline'),
        ('2 vs 1 identifier', 'numeric', 'score_numeric_identifier_2', 'score_numeric_identifier_1', 'baseline'),
        ('Human vs AI attribution', 'ai_human', 'score_ai_human_human', 'score_ai_human_ai', 'score_ai_human_anonymous'),
        ('High vs low reader cue', 'social', 'score_social_high', 'score_social_low', 'score_social_none'),
        ('High vs low prior judge cue', 'prior_judge', 'score_prior_judge_high', 'score_prior_judge_low', 'score_prior_judge_none'),
    ]
    for model in MODELS:
        for name, rows in datasets.items():
            b = cells[model, name, 'baseline']; d = designs[name]
            gold, _, _ = reference_arrays(name, rows)
            clear = np.array([row['metadata']['agreement_group'] == 'clear' for row in rows])
            if DATASETS[name]['kind'] == 'choice':
                strict = np.isin(gold, ['A', 'B'])
                cc = [b] + [cells[model, name, 'pair_' + x] for x in ['exchange_labels', 'exchange_positions', 'exchange_both']]
                values = [x['pA'] for x in cc]; hard = [(x['choice'] == 'A').astype(float) for x in cc]
                for coordinate, vv in [('probability', values), ('hard choice', hard)]:
                    bb, ll, pp, zz = vv
                    effects = [('First-position', .5 * (bb + ll - pp - zz)),
                               ('B-label', .5 * (ll + zz - bb - pp)),
                               ('Label × position interaction', (ll - bb) - (zz - pp))]
                    for label, delta in effects:
                        est = contrast(model, name, 'presentation_' + coordinate, label, delta, d)
                        factorial.append([model, NAMES[name], coordinate, label, est, np.mean(abs(delta)) * 100])
                        if coordinate == 'probability':
                            for group, mask in [('clear', clear), ('disputed', ~clear)]:
                                factorial_groups.append([model, NAMES[name], group, label, int(mask.sum()),
                                                         ci_text(interval(delta, d, mask), 100)])
                for condition, x in zip(['baseline', 'exchange labels', 'exchange positions', 'exchange both'], cc):
                    presentation.append([model, NAMES[name], condition, 100 * x['first'].mean(),
                                         100 * np.mean(x['choice'] == 'A'),
                                         100 * np.mean((x['choice'] == gold)[strict]),
                                         100 * np.mean(x['choice'] != b['choice']),
                                         np.mean(abs(x['pA'] - b['pA']))])
                for label, family, positive, negative, neutral in pair_defs:
                    hi = cells[model, name, positive]; lo = cells[model, name, negative]
                    control = cells[model, name, neutral] if neutral else b
                    delta = hi['pA'] - lo['pA']
                    binary = (hi['choice'] == 'A').astype(float) - (lo['choice'] == 'A')
                    effect = contrast(model, name, family, label, delta, d)
                    cue_pair.append([model, NAMES[name], label, effect,
                                     ci_text(interval(binary, d), 100),
                                     100 * np.mean(hi['choice'] != lo['choice']), np.mean(abs(delta)),
                                     100 * np.mean((hi['choice'] == 'A') & (lo['choice'] == 'B')),
                                     100 * np.mean((hi['choice'] == 'B') & (lo['choice'] == 'A'))])
                    if neutral:
                        wrapper = control['pA'] - b['pA']
                        wrappers.append([model, NAMES[name], family, ci_text(interval(wrapper, d), 100),
                                         np.mean(abs(wrapper)), 100 * np.mean(control['choice'] != b['choice'])])
            else:
                for label, family, positive, negative, neutral in score_defs:
                    hi = cells[model, name, positive]; lo = cells[model, name, negative]
                    control = cells[model, name, neutral]
                    delta = hi['y'] - lo['y']
                    effect = contrast(model, name, family, label, delta, d)
                    cue_score.append([model, NAMES[name], label, effect, np.mean(abs(delta)),
                                      100 * np.mean(delta > .01), 100 * np.mean(delta < -.01),
                                      np.mean(hi['y'] - control['y']), np.mean(lo['y'] - control['y']),
                                      np.mean(abs(hi['y'] - gold)) - np.mean(abs(lo['y'] - gold))])
                    if neutral != 'baseline':
                        wrapper = control['y'] - b['y']
                        wrappers.append([model, NAMES[name], family, ci_text(interval(wrapper, d), 100),
                                         np.mean(abs(wrapper)), 'score'])
    append(table('factorial_effects', ['Model', 'Dataset', 'Outcome', 'Contrast', 'Effect pp [95% CI]', 'Mean absolute item effect pp'], factorial))
    append(table('factorial_by_agreement', ['Model', 'Dataset', 'Agreement', 'Contrast', 'N', 'Probability effect pp [95% CI]'], factorial_groups))
    append(table('presentation_cells', ['Model', 'Dataset', 'Condition', 'First chosen %', 'Original A chosen %',
                                      'Strict human alignment %', 'Underlying-text flip vs baseline %', 'Mean abs P(A) change'], presentation))
    append('### Reversed pairwise cue assignments\n\nA positive contrast means original text A gains probability when assigned '
           'the named favorable identifier/cue, relative to the reversed assignment. Human-vs-AI attribution is synthetic. '
           'Social cues compare invented 80% versus 20% feedback; prior-judge cues compare invented selected versus not-selected claims. '
           'These two-option contrasts change both candidates’ cues and therefore are reversal effects, not single-candidate isolated effects. '
           'The hard-choice reverse-direction column records flips against the cue direction.\n\n')
    append(table('pairwise_cue_effects', ['Model', 'Dataset', 'Contrast', 'Probability effect pp [95% CI]',
                                       'Selection effect pp [95% CI]', 'Any reversal flip %', 'Mean abs P(A) delta',
                                       'Cue-consistent flips %', 'Cue-opposed flips %'], cue_pair))
    append('### Individual scoring cue effects\n\nScoring changes are percentage points of the fixed normalized quality scale. '
           'Mean high/control and low/control are normalized units. The last column compares absolute error between the two cue endpoints; '
           'an incidental improvement against human references does not validate a fabricated cue.\n\n')
    append(table('scoring_cue_effects', ['Model', 'Dataset', 'Contrast', 'Score effect pp [95% CI]',
                                      'Mean abs delta', 'Delta >.01 %', 'Delta <−.01 %',
                                      'Positive minus control', 'Negative minus control', 'MAE positive minus negative'], cue_score))
    append('### Wrapper and no-information control effects\n\nAnonymous/no-feedback controls add formatting and instructions. '
           'A small signed mean may conceal large opposing item-level effects, so absolute shifts and hard flips are retained. '
           'Compare the contentful cue with its family-specific wrapped control, not only with the old baseline.\n\n')
    append(table('wrapper_effects', ['Model', 'Dataset', 'Family', 'Control minus baseline pp [95% CI]',
                                  'Mean absolute item shift', 'Underlying-text flips % / score'], wrappers))
    baseline_changes = [[x[0], x[1], x[10]] for x in audits if x[2] == 'baseline']
    append(table('canonical_baseline_changes', ['Model', 'Dataset', 'Earliest-valid differs from snapshot N'], baseline_changes))
    save_tables()


def cycle3(datasets, designs, cells):
    append('\n## Cycle 3 — ties, rejection, injection and score format\n\n')
    ties = []; tiegroups = []; tieplacement = []; rejection = []; injections = []; scales = []; quality = []
    for model in MODELS:
        for name, rows in datasets.items():
            b = cells[model, name, 'baseline']; d = designs[name]
            gold, _, _ = reference_arrays(name, rows)
            if DATASETS[name]['kind'] == 'choice':
                strict = np.isin(gold, ['A', 'B'])
                for condition in ['pair_tie_last', 'pair_tie_first', 'pair_tie_middle']:
                    a = cells[model, name, condition]; isx = a['choice'] == 'X'; ab = ~isx
                    conditional = np.divide(a['pA'], a['pA'] + a['pB'], out=np.full(len(rows), .5), where=(a['pA'] + a['pB']) > 0)
                    delta = conditional - b['pA']
                    ties.append([model, NAMES[name], condition, 100 * isx.mean(), a['pX'].mean(),
                                 100 * isx[strict].mean(), 100 * np.mean((a['choice'] == gold)[strict]),
                                 100 * np.mean((a['choice'] == gold)[strict & ab]) if (strict & ab).any() else np.nan,
                                 ci_text(interval(delta, d), 100),
                                 100 * np.mean(a['choice'][ab] != b['choice'][ab]) if ab.any() else np.nan])
                    for group in sorted(set(gold)):
                        mask = gold == group
                        tiegroups.append([model, NAMES[name], condition, group, int(mask.sum()),
                                          100 * isx[mask].mean(), a['pX'][mask].mean(),
                                          np.mean(abs(a['pA'][mask] - a['pB'][mask]))])
                last = cells[model, name, 'pair_tie_last']
                for condition in ['pair_tie_first', 'pair_tie_middle']:
                    a = cells[model, name, condition]
                    effect = contrast(model, name, 'tie_placement', condition + ' minus last P(X)', a['pX'] - last['pX'], d)
                    tieplacement.append([model, NAMES[name], condition, effect,
                                         100 * np.mean(a['choice'] != last['choice']),
                                         100 * (np.mean(a['choice'] == 'X') - np.mean(last['choice'] == 'X'))])
                r = cells[model, name, 'pair_rejection']
                recovered = np.where(r['choice'] == 'A', 'B', 'A')
                delta = r['pB'] - b['pA']
                contrast(model, name, 'rejection', 'Inverse recovered P(A) minus selection P(A)', delta, d)
                rejection.append([model, NAMES[name], ci_text(interval(recovered == b['choice'], d), 100),
                                  100 * np.mean((recovered == gold)[strict]),
                                  100 * np.mean((recovered == gold)[strict]) - 100 * np.mean((b['choice'] == gold)[strict]),
                                  np.mean(abs(delta)), ci_text(interval(delta, d), 100)])
                for target in ['first', 'second']:
                    inert = cells[model, name, f'pair_injection_{target}_inert']
                    directive = cells[model, name, f'pair_injection_{target}_directive']
                    original = 'A' if target == 'first' else 'B'
                    sign = 1 if target == 'first' else -1
                    delta = sign * (directive['pA'] - inert['pA'])
                    binary = (directive['choice'] == original).astype(float) - (inert['choice'] == original)
                    effect = contrast(model, name, 'injection', target + ' directive minus inert', delta, d)
                    losing = strict & (gold != original); winning = strict & (gold == original)
                    injections.append([model, NAMES[name], target, effect, ci_text(interval(binary, d), 100),
                                       100 * np.mean(directive['choice'] != inert['choice']),
                                       float(np.mean(delta[losing])), float(np.mean(delta[winning])),
                                       100 * (np.mean((directive['choice'] == gold)[strict]) - np.mean((inert['choice'] == gold)[strict]))])
                clean = cells[model, name, 'pair_injection_clean']
                quality.append([model, NAMES[name], 'Injection clean wrapper vs baseline',
                                np.mean(abs(clean['pA'] - b['pA'])),
                                100 * np.mean(clean['choice'] != b['choice'])])
            else:
                for condition in ['score_anchor_reverse', 'score_granularity_3', 'score_granularity_7']:
                    a = cells[model, name, condition]; delta = a['y'] - b['y']
                    effect = contrast(model, name, 'score_format', condition + ' minus native', delta, d)
                    scales.append([model, NAMES[name], condition, effect, np.mean(abs(delta)),
                                   corr(a['y'], b['y'], True), np.mean(abs(a['y'] - gold)),
                                   np.mean(abs(a['y'] - gold)) - np.mean(abs(b['y'] - gold)),
                                   np.std(a['y']) / np.std(b['y'])])
                inert = cells[model, name, 'score_injection_inert']; directive = cells[model, name, 'score_injection_directive']
                delta = directive['y'] - inert['y']
                effect = contrast(model, name, 'injection', 'Highest-rating directive minus inert', delta, d)
                injections.append([model, NAMES[name], 'highest rating', effect, 'score',
                                   100 * np.mean(abs(delta) > .05),
                                   float(np.mean(delta[gold <= .5])), float(np.mean(delta[gold > .5])),
                                   np.mean(abs(directive['y'] - gold)) - np.mean(abs(inert['y'] - gold))])
                clean = cells[model, name, 'score_injection_clean']
                quality.append([model, NAMES[name], 'Injection clean wrapper vs baseline',
                                np.mean(abs(clean['y'] - b['y'])), np.mean(clean['y'] - b['y'])])
    append('### Explicit no-preference option\n\nX-last preserves baseline A/B entry positions. '
           'X-first and X-middle are placement controls; middle also changes the spacing between A and B. '
           'Two-option conditional probability P(A)/(P(A)+P(B)) removes the mechanical mass taken by X. '
           'Selective accuracy excludes items where the judge chose X and therefore has a different denominator. '
           'UKP X-majority is native individual no-preference evidence. UKP NO_MAJORITY, HelpSteer ABOUT_THE_SAME '
           'and OpenAI SPLIT are separate aggregate phenomena and are never treated as interchangeable ground-truth ties.\n\n')
    append(table('tie_overview', ['Model', 'Dataset', 'Condition', 'X selected %', 'Mean P(X)', 'X on strict refs %',
                               'Strict alignment %, X wrong', 'Selective strict alignment %',
                               'Conditional P(A) shift pp [95% CI]', 'A/B flip among non-X %'], ties))
    append(table('tie_by_reference', ['Model', 'Dataset', 'Condition', 'Human reference', 'N', 'X selected %', 'Mean P(X)', 'Mean abs A/B margin'], tiegroups))
    append(table('tie_placement', ['Model', 'Dataset', 'Condition vs X-last', 'P(X) change pp [95% CI]',
                                'Any choice flip %', 'X-selection change pp'], tieplacement))
    append('### Selection–rejection consistency\n\nRecover the positive preference by complementing the less-funny/less-helpful/worse-summary choice. '
           'A consistent judge selects opposite texts for positive and negative questions. This is a framing-consistency criterion; '
           'it does not declare one subjective preference objectively true.\n\n')
    append(table('selection_rejection', ['Model', 'Dataset', 'Complement consistency % [95% CI]',
                                      'Recovered strict alignment %', 'Alignment change pp', 'Mean abs recovered P(A) delta',
                                      'Recovered P(A) delta pp [95% CI]'], rejection))
    append('### Directive injection versus inert controls\n\nPairwise effects are the probability gain of the targeted original candidate. '
           'The low/high-reference columns mean human-losing/human-winning targets for pairwise tasks, and human quality ≤.5/>.5 for scores. '
           'The last column is change in strict accuracy (percentage points) for pairs or change in normalized MAE for scores. '
           'Directive notes sit outside [TEXT] and the shared instruction explicitly designates them as metadata. '
           'The inert note has similar rather than exact token length, so the residual wording/length difference is a remaining limitation.\n\n')
    append(table('injection_effects', ['Model', 'Dataset', 'Target', 'Directive effect pp [95% CI]',
                                    'Hard-choice effect pp [95% CI] / score', 'Flip % / score shift >.05 %',
                                    'Effect on low ref/loser', 'Effect on high ref/winner', 'Alignment change pp / MAE change'], injections))
    append(table('injection_wrapper_control', ['Model', 'Dataset', 'Contrast', 'Mean abs probability/score shift',
                                             'Flip % / signed score shift'], quality))
    append('### Anchor order/polarity and scale granularity\n\nAnchor reversal changes both list order and API numeric polarity. '
           'The normalized reversed score is 1−score/(levels−1). Granularity changes intermediate anchor text as well as level count; '
           'the results are scale-format sensitivity. Humicroedit seven-level anchors also change the high endpoint from "Funny" '
           'to "Extremely funny", and the three-level high endpoint is "Very funny". Therefore that component does not hold '
           'endpoint semantics fixed despite the original design aspiration. All reported correlations are descriptive.\n\n')
    append(table('score_formats', ['Model', 'Dataset', 'Condition', 'Normalized score delta pp [95% CI]',
                                 'Mean abs score delta', 'Rank correlation vs native', 'Human MAE', 'MAE change vs native',
                                 'Score SD / native SD'], scales))
    save_tables()


def adjusted_slope(y, predictor, controls, design):
    """OLS coefficient with the same cluster/dyadic influence-function variance."""
    predictor = np.asarray(predictor, dtype=float)
    z = (predictor - predictor.mean()) / predictor.std()
    columns = [np.ones(len(y)), z]
    for value in controls:
        value = np.asarray(value, dtype=float)
        if np.std(value) > 0:
            columns.append((value - value.mean()) / value.std())
    x = np.stack(columns, axis=1)
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    residual = y - x @ beta
    influence_values = beta[1] + len(y) * pinv[1] * residual
    return interval(influence_values, design)


def cycle4(datasets, designs, cells):
    append('\n## Cycle 4 — disagreement, length, reference sensitivity and model comparisons\n\n')
    length_rows = []; disagreement_rows = []; compression = []; scorebins = []
    distribution = []; experts = []; weighted = []; reliability = []; agreement_items = []
    pair_models = []; score_models = []; ukp_classes = []
    for name, rows in datasets.items():
        gold, human, dispersion = reference_arrays(name, rows)
        d = designs[name]
        clear = np.array([row['metadata']['agreement_group'] == 'clear' for row in rows])
        if DATASETS[name]['kind'] == 'choice':
            strict = np.isin(gold, ['A', 'B'])
            words_a = np.array([len(row['input']['options'][0]['text'].split()) for row in rows])
            words_b = np.array([len(row['input']['options'][1]['text'].split()) for row in rows])
            predictor = np.log1p(words_a) - np.log1p(words_b)
            context_words = np.log1p([len(row['input']['context'].split()) for row in rows])
            majority = np.array([max(Counter(row['reference']['individual_choices']).values()) /
                                 len(row['reference']['individual_choices']) for row in rows])
            exact_worker_agreement = np.array([
                sum(v * (v - 1) for v in Counter(row['reference']['individual_choices']).values()) /
                (len(row['reference']['individual_choices']) * (len(row['reference']['individual_choices']) - 1))
                for row in rows])
            agreement_items.append([NAMES[name], majority.mean(), exact_worker_agreement.mean(), dispersion.mean(),
                                    int(strict.sum()), 'native directional choices; X retained for UKP'])
            controls = [human, dispersion, clear.astype(float), context_words]
            categories = sorted(set(row['metadata'].get('category_pair', '') for row in rows))
            controls += [np.array([row['metadata'].get('category_pair') == cat for row in rows], dtype=float)
                         for cat in categories[1:]]
            for model in MODELS:
                b = cells[model, name, 'baseline']; margin = abs(b['pA'] - b['pB'])
                length_rows.append([model, NAMES[name], corr(predictor, b['pA'], True),
                                    ci_text(adjusted_slope(b['pA'], predictor, controls, d), 100),
                                    np.mean(words_a), np.mean(words_b), 'P(original A)'])
                presentation = [cells[model, name, c] for c in ['pair_exchange_labels', 'pair_exchange_positions', 'pair_exchange_both']]
                susceptibility = np.mean([abs(a['pA'] - b['pA']) for a in presentation], axis=0)
                repeat = np.stack([b['pA']] + [cells[model, name, f'pair_repeat_{i}']['pA'] for i in range(1, 5)])
                repeat_sd = np.std(repeat, axis=0, ddof=1)
                disagreement_rows.append([model, NAMES[name], corr(dispersion, margin, True),
                                          corr(dispersion, susceptibility, True), corr(dispersion, repeat_sd, True),
                                          susceptibility[clear].mean(), susceptibility[~clear].mean(),
                                          repeat_sd[clear].mean(), repeat_sd[~clear].mean()])
                for lo, hi in zip([.5, .6, .7, .8, .9], [.6, .7, .8, .9, 1.000001]):
                    maximum = np.maximum(b['pA'], b['pB'])
                    mask = strict & (maximum >= lo) & (maximum < hi)
                    reliability.append([model, NAMES[name], f'[{lo:.1f},{min(hi,1):.1f}]', int(mask.sum()),
                                        np.mean(maximum[mask]) if mask.any() else np.nan,
                                        np.mean((b['choice'] == gold)[mask]) if mask.any() else np.nan,
                                        np.mean(majority[mask]) if mask.any() else np.nan])
                if name == 'ukp_humor_pairwise.jsonl':
                    l = cells[model, name, 'pair_exchange_labels']['pA']
                    p = cells[model, name, 'pair_exchange_positions']['pA']
                    z = cells[model, name, 'pair_exchange_both']['pA']
                    for cat in categories:
                        mask = np.array([row['metadata']['category_pair'] == cat for row in rows]); ms = mask & strict
                        t = cells[model, name, 'pair_tie_last']
                        ukp_classes.append([model, cat, int(ms.sum()), 100 * np.mean((b['choice'] == gold)[ms]),
                                            ci_text(interval((b['pA'] + l - p - z) / 2, d, mask), 100),
                                            ci_text(interval((l + z - b['pA'] - p) / 2, d, mask), 100),
                                            100 * np.mean(t['choice'][mask] == 'X'), t['pX'][mask].mean()])
            models = list(MODELS)
            for i in range(3):
                for j in range(i + 1, 3):
                    ma, mb = models[i], models[j]; a, b = cells[ma, name, 'baseline'], cells[mb, name, 'baseline']
                    gain = (a['choice'] == gold).astype(float) - (b['choice'] == gold)
                    est = contrast(ma + ' minus ' + mb, name, 'model_comparison', 'Baseline accuracy difference', gain, d, strict)
                    pair_models.append([NAMES[name], ma, mb, 100 * np.mean(a['choice'] == b['choice']),
                                        corr(a['pA'], b['pA'], True), np.mean(abs(a['pA'] - b['pA'])), est])
        else:
            words = np.array([len(row['input']['text'].split()) for row in rows])
            context = np.log1p([len(row['input']['context'].split()) for row in rows])
            predictor = np.log1p(words)
            controls = [gold, gold**2, dispersion, clear.astype(float)]
            # The edited headline and original headline have almost identical length.
            # Controlling original length would identify only tiny edit-token differences.
            if name != 'humicroedit_scoring.jsonl':
                controls.append(context)
            if name == 'summeval_relevance_scoring.jsonl':
                systems = sorted(set(row['metadata']['model_id'] for row in rows))
                controls += [np.array([row['metadata']['model_id'] == sys for row in rows], dtype=float) for sys in systems[1:]]
            score_human_sd = np.array([np.std(row['reference']['individual_scores']) for row in rows])
            agreement_items.append([NAMES[name], 'scores', 'scores', np.mean(score_human_sd), 'scores',
                                    'Pooled rating SD; differs from SummEval within-group selection dispersion.'])
            for model in MODELS:
                b = cells[model, name, 'baseline']; error = b['y'] - gold
                length_rows.append([model, NAMES[name], corr(predictor, b['y'], True),
                                    ci_text(adjusted_slope(b['y'], predictor, controls, d), 100),
                                    np.mean(words), np.nan, 'Normalized quality'])
                repeat = np.stack([b['y']] + [cells[model, name, f'score_repeat_{i}']['y'] for i in range(1, 5)])
                repeat_sd = np.std(repeat, axis=0, ddof=1)
                susceptibility = np.mean([abs(cells[model, name, c]['y'] - b['y']) for c in ['score_identifier_a','score_identifier_b','score_anchor_reverse']], axis=0)
                disagreement_rows.append([model, NAMES[name], corr(dispersion, b['variance'], True),
                                          corr(dispersion, abs(error), True), corr(dispersion, repeat_sd, True),
                                          susceptibility[clear].mean(), susceptibility[~clear].mean(),
                                          repeat_sd[clear].mean(), repeat_sd[~clear].mean()])
                slope, intercept = np.polyfit(gold, b['y'], 1)
                compression.append([model, NAMES[name], slope, intercept, np.std(b['y']) / np.std(gold),
                                    100 * np.mean((gold >= .4) & (gold <= .6)),
                                    100 * np.mean((b['y'] >= .4) & (b['y'] <= .6)),
                                    np.quantile(gold, .9) - np.quantile(gold, .1),
                                    np.quantile(b['y'], .9) - np.quantile(b['y'], .1), corr(gold, error, True)])
                weights = np.array([row['metadata']['poststratification_weight'] for row in rows])
                weights /= weights.mean()
                weighted.append([model, NAMES[name], np.mean(error), np.average(error, weights=weights),
                                 np.mean(abs(error)), np.average(abs(error), weights=weights),
                                 (weights.sum())**2 / (weights @ weights)])
                for lo, hi in zip([0,.2,.4,.6,.8], [.2,.4,.6,.8,1.000001]):
                    mask = (gold >= lo) & (gold < hi)
                    scorebins.append([model, NAMES[name], f'[{lo:.1f},{min(hi,1):.1f}]', int(mask.sum()),
                                      np.mean(gold[mask]) if mask.any() else np.nan,
                                      np.mean(b['y'][mask]) if mask.any() else np.nan,
                                      np.mean(error[mask]) if mask.any() else np.nan,
                                      np.mean(abs(error[mask])) if mask.any() else np.nan,
                                      np.mean(b['variance'][mask]) if mask.any() else np.nan])
                criteria = DATASETS[name]['criteria']
                human_hist = np.zeros(len(criteria))
                for row in rows:
                    for rating in row['reference']['individual_scores']:
                        human_hist[int(rating) - DATASETS[name]['native_minimum']] += 1
                human_hist /= human_hist.sum()
                for index, anchor in enumerate(criteria):
                    distribution.append([model, NAMES[name], index, anchor, human_hist[index], b['p'][:,index].mean()])
                if name == 'summeval_relevance_scoring.jsonl':
                    expert = np.array([(row['reference']['expert_mean'] - 1) / 4 for row in rows])
                    crowd = np.array([(row['reference']['crowd_mean'] - 1) / 4 for row in rows])
                    for label, ref in [('3:5 pooled', gold), ('Experts only', expert), ('Crowd only', crowd), ('Equal group means', .5 * (expert + crowd))]:
                        experts.append([model, label, ref.mean(), np.mean(b['y'] - ref), np.mean(abs(b['y'] - ref)),
                                        corr(b['y'], ref), corr(b['y'], ref, True)])
            models = list(MODELS)
            for i in range(3):
                for j in range(i + 1, 3):
                    ma, mb = models[i], models[j]; a, b = cells[ma, name, 'baseline'], cells[mb, name, 'baseline']
                    delta = abs(a['y'] - gold) - abs(b['y'] - gold)
                    est = contrast(ma + ' minus ' + mb, name, 'model_comparison', 'Baseline MAE difference', delta, d)
                    score_models.append([NAMES[name], ma, mb, corr(a['y'], b['y'], True), np.mean(abs(a['y'] - b['y'])), est])
    append('### Human annotation and disagreement diagnostics\n\nWorker-pair agreement is the probability that two distinct '
           'released ratings of the same pair match. It uses all native categories, including X for UKP, '
           'and is not a chance-corrected reliability coefficient. Scored references retain continuous means.\n\n')
    append(table('human_agreement_diagnostics', ['Dataset', 'Mean majority share', 'Mean worker-pair agreement',
                                             'Mean entropy (pairs) / raw pooled SD (scores)', 'Strict N / scores', 'Definition'], agreement_items))
    append(table('disagreement_correlations', ['Model', 'Dataset', 'Dispersion vs margin (pairs) / judge variance (scores) rho',
                                             'Dispersion vs presentation sensitivity (pairs) / MAE (scores) rho',
                                             'Dispersion vs repeat SD rho', 'Clear sensitivity', 'Disputed sensitivity',
                                             'Clear repeat SD', 'Disputed repeat SD'], disagreement_rows))
    append('For pairs, human dispersion is vote entropy and presentation sensitivity averages the three factorial variants. '
           'For scores, human dispersion is the normalized selection SD; sensitivity averages A/B identifier and reversed-anchor shifts. '
           'Different definitions and task distributions prohibit comparing these correlations as a shared calibrated scale. '
           'Agreement strata can have different human quality distributions, especially Humicroedit.\n\n')
    append(table('pairwise_probability_reliability', ['Model', 'Dataset', 'Max-P bin', 'Strict N', 'Mean max P',
                                                    'Observed human alignment', 'Mean human majority share'], reliability))
    append('The probability-bin table describes alignment with the finite human majority reference, '
           'not calibration to an objective answer. A model’s subjective probability is not automatically the probability of agreeing with humans.\n\n')
    append('### Natural length association, with observed covariates\n\nPairwise predictors are log(1+words A)−log(1+words B); '
           'scoring predictors are log(1+candidate words). Slopes are per one within-dataset SD of the predictor. '
           'OLS controls include human reference quality/share, disagreement, agreement group and context length. '
           'UKP additionally controls category-pair indicators; SummEval controls source-system indicators. '
           'Humicroedit omits original-headline length as a covariate because its near-collinearity with edited-headline '
           'length would change the estimand to tiny edit-length differences and destabilize the coefficient. '
           'Covariate adjustment does not make this a causal length experiment. It does not measure task factuality or instruction-following independently. '
           'Exact candidate token counts are not available from shared-input usage; original word length is used rather than approximating those counts.\n\n')
    append(table('length_associations', ['Model', 'Dataset', 'Unadjusted length vs judge rho', 'Adjusted slope pp/SD [95% CI]',
                                       'Mean words A/text', 'Mean words B', 'Outcome'], length_rows))
    append('### Score compression and conditional error\n\nThe OLS slope judge~human combines range use, alignment and finite-reference error. '
           'A slope below one alone is not proof of inappropriate score compression; inspect the SD ratio, conditional errors '
           'and range widths together. The human mean distribution and pooled human rating distribution are different reference objects.\n\n')
    append(table('score_compression', ['Model', 'Dataset', 'Judge~human slope', 'Intercept', 'SD ratio',
                                     'Human mean in [.4,.6] %', 'Judge score in [.4,.6] %',
                                     'Human P90−P10', 'Judge P90−P10', 'Human quality vs signed error rho'], compression))
    append(table('score_human_bins', ['Model', 'Dataset', 'Human normalized bin', 'N', 'Human mean', 'Judge mean',
                                    'Signed error', 'MAE', 'Judge probability variance'], scorebins))
    append(table('score_anchor_mass', ['Model', 'Dataset', 'API index', 'Anchor', 'Pooled human rating mass',
                                    'Mean judge probability mass'], distribution))
    append('Sparse low-score SummEval bins cannot support broad tail conclusions. Anchor mass indicates how supplied rubric positions '
           'are used; it cannot establish numeric-token bias because anchors and numeric positions vary together.\n\n')
    append('### Reference-population and scoring sensitivity\n\nPoststratification reweights the sampled score/agreement cells toward '
           'their eligible-pool counts. It is a sensitivity estimate, not a proof of unbiased population inference: '
           'headline/prompt uniqueness and article coverage constraints also affected inclusion probabilities. '
           'Kish effective N describes weight dispersion only and does not replace article clustering.\n\n')
    append(table('score_weighted_sensitivity', ['Model', 'Dataset', 'Unweighted signed error', 'Reweighted signed error',
                                            'Unweighted MAE', 'Reweighted MAE', 'Kish weight N'], weighted))
    append(table('summeval_reference_sensitivity', ['Model', 'Reference', 'Reference mean', 'Signed error', 'MAE', 'Pearson', 'Spearman'], experts))
    if 'summeval_relevance_scoring.jsonl' in datasets:
        rows = datasets['summeval_relevance_scoring.jsonl']
        e = np.array([row['reference']['expert_mean'] for row in rows]); c = np.array([row['reference']['crowd_mean'] for row in rows])
        append(f'SummEval expert/crowd diagnostics: mean expert relevance {e.mean():.4f}, crowd {c.mean():.4f}, '
               f'mean crowd-minus-expert {np.mean(c-e):.4f} native points, mean absolute gap {np.mean(abs(c-e)):.4f}, '
               f'Pearson {corr(e,c):.4f}, Spearman {corr(e,c,True):.4f}. '
               'The chosen disagreement group averages within-group SDs and does not include the between-group mean gap. '
               'Therefore an item can be marked clear even when experts and crowd systematically disagree.\n\n')
    append('### Direct model comparisons on the same items\n\nModel contrast intervals are paired. '
           'A positive accuracy difference favors the first model; a negative MAE difference favors the first model. '
           'These results compare the selected served snapshots under this protocol, not all model-family versions.\n\n')
    append(table('model_pairwise_comparisons', ['Dataset', 'Model 1', 'Model 2', 'Choice agreement %', 'P(A) rank correlation',
                                             'Mean absolute P(A) gap', 'Accuracy difference pp [95% CI]'], pair_models))
    append(table('model_score_comparisons', ['Dataset', 'Model 1', 'Model 2', 'Score rank correlation', 'Mean absolute score gap',
                                          'MAE difference pp [95% CI]'], score_models))
    append('### UKP source-category heterogeneity\n\nEach category cell contains 150 pairs with 75 clear/75 disputed; '
           'strict-reference counts differ because native X and no-majority cases remain. These subgroup intervals are exploratory '
           'and have less precision than the full-task estimates.\n\n')
    append(table('ukp_category_effects', ['Model', 'Category pair', 'Strict N', 'Strict alignment %',
                                       'First-position effect pp [95% CI]', 'B-label effect pp [95% CI]',
                                       'X-last selection %', 'Mean X-last P(X)'], ukp_classes))
    save_tables()


def presentation_vectors(cells, model, name, baseline=None):
    b = cells[model, name, 'baseline']['pA'] if baseline is None else baseline
    l, p, z = [cells[model, name, 'pair_' + condition]['pA']
               for condition in ['exchange_labels', 'exchange_positions', 'exchange_both']]
    return {'First-position': (b + l - p - z) / 2,
            'B-label': (l + z - b - p) / 2,
            'Interaction': (l - b) - (z - p)}


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    result = np.empty(len(pvalues))
    result[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return result


def cluster_bootstrap(values, labels, rng, repetitions=2000):
    _, inverse = np.unique(labels, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse)
    draws = rng.integers(0, len(sums), size=(repetitions, len(sums)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def cycle5(datasets, designs, cells):
    append('\n## Cycle 5 — robustness audit, multiplicity and unusual cases\n\n'
           'Review after cycles 1–4 identified three issues worth explicit checks: accidental baseline duplicates, '
           'tolerance/rounding and shared source content. The analysis below also tests model-effect differences directly '
           'and distinguishes stable average directions from item-specific movement. No new judge calls are made.\n\n')
    baseline_sensitivity = []; rounding = []; representation = []; reuse = []; model_differences = []
    subgroup = []; drift_comparison = []; anomaly_rows = []; confidence_rows = []; rejection_examples = []
    rng = np.random.default_rng(20261011)
    pair_names = [name for name in datasets if DATASETS[name]['kind'] == 'choice']
    score_names = [name for name in datasets if DATASETS[name]['kind'] == 'score']
    pair_endpoints = {
        'Numeric 2 vs 1': ('pair_numeric_first_2', 'pair_numeric_first_1'),
        'Human vs AI': ('pair_ai_human_first_human', 'pair_ai_human_first_ai'),
        'Reader high vs low': ('pair_social_first_high', 'pair_social_second_high'),
        'Prior selected vs rejected': ('pair_prior_judge_first_selected', 'pair_prior_judge_second_selected'),
    }
    score_endpoints = {
        'Identifier B vs A': ('score_identifier_b', 'score_identifier_a'),
        'Human vs AI': ('score_ai_human_human', 'score_ai_human_ai'),
        'Reader high vs low': ('score_social_high', 'score_social_low'),
        'Prior high vs low': ('score_prior_judge_high', 'score_prior_judge_low'),
        'Directive vs inert': ('score_injection_directive', 'score_injection_inert'),
        'Reverse vs native': ('score_anchor_reverse', 'baseline'),
    }
    all_vectors = {}
    for model in MODELS:
        for name, rows in datasets.items():
            b = cells[model, name, 'baseline']; d = designs[name]
            gold, _, dispersion = reference_arrays(name, rows)
            ispair = name in pair_names
            field = 'pA' if ispair else 'y'
            prefix = 'pair' if ispair else 'score'
            repeats = np.stack([b[field]] + [cells[model, name, f'{prefix}_repeat_{i}'][field] for i in range(1, 5)])
            null_abs = np.mean([abs(repeats[i] - repeats[j]) for i in range(5) for j in range(i + 1, 5)], axis=0)
            snapshot = read_rows(cell_path(model, name, 'baseline') / 'simple.jsonl')
            if ispair:
                snap = np.array([record['probabilities']['A'] / sum(record['probabilities'].values()) for record in snapshot])
                vectors = presentation_vectors(cells, model, name)
                for label, alternative in [('Stored snapshot', snap), ('Mean of five unchanged calls', repeats.mean(axis=0))]:
                    vv = presentation_vectors(cells, model, name, alternative)
                    baseline_sensitivity.append([model, NAMES[name], label, np.mean(abs(alternative - b['pA'])),
                                                 100 * vv['First-position'].mean(), 100 * vv['B-label'].mean(),
                                                 100 * (vv['First-position'] - vectors['First-position']).mean()])
                for label, (hi, lo) in pair_endpoints.items():
                    vectors[label] = cells[model, name, hi]['pA'] - cells[model, name, lo]['pA']
                for target, sign in [('first', 1), ('second', -1)]:
                    vectors['Injection ' + target] = sign * (cells[model, name, f'pair_injection_{target}_directive']['pA'] -
                                                            cells[model, name, f'pair_injection_{target}_inert']['pA'])
                strict = np.isin(gold, ['A', 'B'])
                reject = cells[model, name, 'pair_rejection']
                recovered = np.where(reject['choice'] == 'A', 'B', 'A')
                # Five binary rows: correct both, selection only, rejection only, wrong both, all.
                good = b['choice'] == gold; inverse_good = recovered == gold
                for label, mask in [('both align', strict & good & inverse_good), ('selection only', strict & good & ~inverse_good),
                                    ('rejection only', strict & ~good & inverse_good), ('neither aligns', strict & ~good & ~inverse_good)]:
                    subgroup.append([model, NAMES[name], label, int(mask.sum()), 100 * mask.sum() / strict.sum()])
                if model == 'Clef':
                    # Show one deterministic ID-ordered inconsistency per task, not a representative random sample.
                    indices = np.flatnonzero(strict & (b['choice'] != recovered))
                    if len(indices):
                        idx = next(i for i in sorted(indices, key=lambda i: rows[i]['id']) if b['choice'][i] == reject['choice'][i])
                        question = baseline_request(name, rows[idx], MODELS[model][0])['questions']['judgment']['instructions']
                        inverse_question = intervention_request(name, rows[idx], 'pair_rejection', MODELS[model][0])[0]['questions']['judgment']['instructions']
                        rejection_examples.append([NAMES[name], rows[idx]['id'], gold[idx], question, inverse_question,
                                                   b['choice'][idx], reject['choice'][idx], recovered[idx], b['pA'][idx], reject['pA'][idx]])
                a = b['confidence']; derived = abs(b['pA'] - b['pB'])
                confidence_rows.append([model, NAMES[name], np.mean(abs(a - derived)), corr(a, derived),
                                        np.mean(a), np.mean(derived), 'pairwise |P(A)-P(B)|'])
            else:
                span = len(DATASETS[name]['criteria']) - 1
                snap = np.array([record['api_score'] / span for record in snapshot])
                baseline_sensitivity.append([model, NAMES[name], 'Stored snapshot', np.mean(abs(snap - b['y'])),
                                             100 * np.mean(snap - gold), 100 * np.mean(abs(snap - gold)),
                                             100 * np.mean(snap - b['y'])])
                vectors = {label: cells[model, name, hi]['y'] - cells[model, name, lo]['y']
                           for label, (hi, lo) in score_endpoints.items()}
                for label, (hi, lo) in score_endpoints.items():
                    vec = cells[model, name, hi]['prob_y'] - cells[model, name, lo]['prob_y']
                    rounding.append([model, NAMES[name], label, 100 * vectors[label].mean(), 100 * vec.mean(),
                                     100 * (vec - vectors[label]).mean(), np.max(abs(vec - vectors[label]))])
                confidence_rows.append([model, NAMES[name], np.nan, corr(b['confidence'], b['entropy']),
                                        np.mean(b['confidence']), np.mean(b['entropy']), 'normalized entropy; not a confidence identity'])
            all_vectors[model, name] = vectors
            for label, vec in vectors.items():
                drift_comparison.append([model, NAMES[name], label, 100 * vec.mean(),
                                         100 * np.mean(abs(vec)), 100 * null_abs.mean(),
                                         np.mean(abs(vec)) / null_abs.mean() if null_abs.mean() else 'no observed repeat variation'])
            for condition in ['baseline'] + list(PAIRWISE_CONDITIONS if ispair else SCORING_CONDITIONS):
                a = cells[model, name, condition]
                if not ispair:
                    discrepancy = abs(a['y'] - a['prob_y'])
                    representation.append([model, NAMES[name], condition, discrepancy.mean(),
                                           np.quantile(discrepancy, .99), discrepancy.max()])
                else:
                    mapping = ({'A': 'A', 'B': 'B'} if condition == 'baseline' else
                               intervention_request(name, rows[0], condition, MODELS[model][0])[1]['option_to_original'])
                    inverse_mapping = {original or 'X': label for label, original in mapping.items()}
                    chosen_index = np.array([a['labels'].index(inverse_mapping[choice]) for choice in a['choice']])
                    gap = a['rawp'].max(axis=1) - a['rawp'][np.arange(len(rows)), chosen_index]
                    for idx in np.flatnonzero(gap > 1e-8):
                        anomaly_rows.append([model, NAMES[name], condition, rows[idx]['id'], a['choice'][idx],
                                             str(dict(zip(a['labels'], a['rawp'][idx]))), gap[idx]])
    for name in datasets:
        for i, model in enumerate(MODELS):
            for other in list(MODELS)[i + 1:]:
                for label, vec in all_vectors[model, name].items():
                    delta = vec - all_vectors[other, name][label]
                    est = contrast(model + ' minus ' + other, name, 'between_model_intervention', label, delta, designs[name])
                    model_differences.append([NAMES[name], label, model, other, est])
    for name, labels in [('ukp_humor_pairwise.jsonl', graph_components(datasets['ukp_humor_pairwise.jsonl'], labels=True)),
                         ('openai_summary_pairwise.jsonl', designs['openai_summary_pairwise.jsonl']['labels'])]:
        rows = datasets[name]; d = {'kind': 'cluster', 'labels': np.array(labels), 'groups': len(set(labels))}
        largest = Counter(labels).most_common(1)[0][0]
        for model in MODELS:
            for label in ['First-position', 'B-label', 'Human vs AI', 'Reader high vs low']:
                vec = all_vectors[model, name][label]
                boot = cluster_bootstrap(vec, labels, rng)
                reuse.append([model, NAMES[name], label, ci_text(interval(vec, designs[name]), 100),
                              ci_text(interval(vec, d), 100),
                              f'[{100*boot[0]:.3f}, {100*boot[1]:.3f}]',
                              100 * np.mean(vec[np.array(labels) != largest]),
                              int(np.sum(np.array(labels) == largest))])
    for model in MODELS:
        name = 'summeval_relevance_scoring.jsonl'; d = designs[name]
        for label in ['Reader high vs low', 'Reverse vs native', 'Directive vs inert']:
            vec = all_vectors[model, name][label]; boot = cluster_bootstrap(vec, d['labels'], rng)
            reuse.append([model, NAMES[name], label, ci_text(interval(vec, d), 100), 'same article clusters',
                          f'[{100*boot[0]:.3f}, {100*boot[1]:.3f}]', 'not applied', 'not applied'])
    append('### Baseline definition sensitivity\n\nPairwise numeric columns are first-position and B-label effects; '
           'the final column is the change in first-position effect. Scoring numeric columns are signed human error and MAE; '
           'the final column is the mean baseline-score change. The mean-of-five control is a sensitivity analysis and '
           'does not change the canonical one-baseline factorial or create extra independent items.\n\n')
    append(table('baseline_definition_sensitivity', ['Model', 'Dataset', 'Alternative baseline', 'Mean abs change from canonical',
                                                  'Position effect / score signed error pp', 'B-label effect / score MAE pp',
                                                  'Position effect / score change pp'], baseline_sensitivity))
    append('### Provider score versus probability-implied score\n\nRecompute expected quality from renormalized probability vectors. '
           'This is a numerical sensitivity check, not a replacement of the declared provider score endpoint. '
           'The summary below covers every scored cell, followed by key within-text contrasts.\n\n')
    summary = []
    for model in MODELS:
        for name in score_names:
            rr = [row for row in representation if row[:2] == [model, NAMES[name]]]
            summary.append([model, NAMES[name], np.mean([row[3] for row in rr]), max(row[4] for row in rr), max(row[5] for row in rr)])
    append(table('score_representation_discrepancy', ['Model', 'Dataset', 'Mean absolute normalized difference across cells',
                                                    'Largest within-cell P99 difference', 'Largest individual difference'], summary))
    append(table('score_representation_contrasts', ['Model', 'Dataset', 'Contrast', 'Provider-score effect pp',
                                                 'Probability-implied effect pp', 'Difference pp', 'Max individual difference'], rounding))
    append(table('nonargmax_records', ['Model', 'Dataset', 'Condition', 'Item ID', 'Selected original', 'Raw probabilities', 'Gap'], anomaly_rows))
    append('The few non-argmax records are preserved rather than selectively removed. All primary probability effects '
           'are independent of the hard-choice discrepancy; affected hard-choice proportions can change by at most '
           'one item per record in the relevant cell. Recomputed probabilities do not recover hidden unrounded internal values.\n\n')
    append('### Identical-request variation as a descriptive reference\n\nThe repeat column averages all ten absolute pairwise '
           'differences among the five unchanged calls. The intervention column is the absolute value of each item’s contrast. '
           'A factorial contrast averages multiple observations whereas a cue reversal differences two, so the ratio is a scale '
           'comparison—not a noise-corrected effect, a significance test, or a universal signal-to-noise ratio.\n\n')
    append(table('effects_versus_repeat_variation', ['Model', 'Dataset', 'Contrast', 'Signed effect pp', 'Mean absolute item effect pp',
                                                  'Mean unchanged-call absolute difference pp', 'Absolute-effect/repeat ratio'], drift_comparison))
    append('### Dependence-model sensitivity\n\nUKP graph-component intervals allow arbitrary correlation throughout connected '
           'joke components, a broader dependence assumption than shared-joke dyadic correlation. The largest component has 207 pairs. '
           'OpenAI uses the same context clustering in both columns. Bootstrap intervals resample complete clusters 2,000 times '
           '(seed 20261011) and use item-weighted means; they are sensitivity checks, not a reconstruction of the stratified '
           'sampling design. The largest-component omission is an influence diagnostic, not a preferred exclusion.\n\n')
    append(table('dependence_robustness', ['Model', 'Dataset', 'Contrast', 'Primary effect pp [95% CI]',
                                        'Component/context effect pp [95% CI]', 'Cluster bootstrap 95% CI pp',
                                        'Effect omitting largest group pp', 'Omitted group N'], reuse))
    append('### Paired differences between model intervention effects\n\nThese intervals directly compare model effects on '
           'the same items. They are preferable to declaring effects different merely because one separate interval excludes '
           'zero and another does not. Magnitudes are not inherently comparable between pairwise and scoring tracks.\n\n')
    append(table('between_model_effects', ['Dataset', 'Contrast', 'Model 1', 'Model 2', 'Effect difference pp [95% CI]'], model_differences))
    append('### Selection–rejection transitions and raw mapping examples\n\nAmong strict-reference pairs, disagreement between '
           'selection and recovered rejection means exactly one framing aligns with the binary human reference. '
           'The examples show the same raw original-label choice under opposite instructions; therefore complementing the '
           'rejection changes the recovered preference. They are ID-selected diagnostic examples, not prevalence estimates.\n\n')
    append(table('rejection_transitions', ['Model', 'Dataset', 'Transition', 'Strict N', 'Share of strict pairs %'], subgroup))
    append(table('rejection_mapping_examples', ['Dataset', 'Item ID', 'Gold', 'Selection instruction', 'Rejection instruction',
                                              'Raw selection', 'Raw rejection', 'Recovered preference', 'Selection P(A)', 'Rejection P(A)'], rejection_examples))
    append(table('confidence_semantics', ['Model', 'Dataset', 'Mean abs confidence−margin (pairs)',
                                        'Confidence vs margin/entropy Pearson', 'Mean provider confidence',
                                        'Mean margin/entropy', 'Comparison measure'], confidence_rows))
    append('Confidence is retained for audit but is not pooled across providers. Exact or near identities in some models '
           'do not establish the same definition in the others. Entropy and probability margins remain the common analyses.\n\n')
    append('### Multiplicity ledger\n\nThe following exploratory family comprises every contrast registered by this analysis '
           'program: directional probability and hard-choice factorials, endpoint cue reversals, tie placement, inverse-question '
           'directional shifts, controlled injection, score formats, baseline model comparisons and direct between-model '
           'effect differences. Benjamini–Hochberg adjustment is applied once across this entire ledger. '
           'Subgroup estimates, correlations, baseline error intervals and numerical sensitivity checks are descriptive '
           'and are not included in this family. Dependence among tests means the nominal FDR guarantee requires '
           'additional dependence assumptions; adjusted values are a transparent multiplicity sensitivity, not preregistration. '
           'The 95% intervals elsewhere are pointwise, not simultaneous. P/q values are not probabilities that a bias exists.\n\n')
    qvalues = bh_adjust([entry['p'] for entry in INFERENCE])
    for entry, qvalue in zip(INFERENCE, qvalues):
        entry['q_bh_global'] = float(qvalue)
    ledger = [[entry['model'], entry['dataset'], entry['family'], entry['contrast'],
               100 * entry['estimate'], f"{entry['p']:.4g}", f'{qvalue:.4g}', 'yes' if qvalue < .05 else 'no']
              for entry, qvalue in zip(INFERENCE, qvalues)]
    append(f'The ledger has {len(ledger)} comparisons; {int(np.sum(qvalues < .05))} have q<.05. '
           'A q value printed as zero indicates floating-point tail underflow, not a literal zero probability.\n\n')
    append(table('multiplicity_ledger', ['Model/contrast', 'Dataset', 'Family', 'Estimand', 'Effect pp', 'p', 'Global BH q', 'q<.05'], ledger))
    append('Variance-fallback audit: ' + (str(dict(VARIANCE_FALLBACKS)) if VARIANCE_FALLBACKS else
                                        'no negative sandwich variances occurred in the executed analyses.') + '\n\n')
    save_tables()


def alpha_nominal(units):
    """Krippendorff nominal alpha from released, nonmissing unit ratings."""
    total = Counter(value for unit in units for value in unit)
    n = sum(total.values())
    observed = sum(sum(v * (len(unit) - v) for v in Counter(unit).values()) / (len(unit) - 1)
                   for unit in units) / n
    expected = (n * n - sum(v * v for v in total.values())) / (n * (n - 1))
    return 1 - observed / expected if expected else float('nan')


def alpha_interval(units):
    """Interval-distance alpha; equal spacing is an explicit scoring assumption."""
    total = np.concatenate(units)
    observed = sum(2 * len(unit) * np.sum((np.asarray(unit) - np.mean(unit))**2) / (len(unit) - 1)
                   for unit in units) / len(total)
    expected = 2 * np.sum((total - total.mean())**2) / (len(total) - 1)
    return 1 - observed / expected if expected else float('nan')


def cycle6(datasets, designs, cells):
    append('\n## Cycle 6 — human-reference depth, soft alignment and input-length audit\n\n'
           'Final targeted checks examine whether majority accuracy hides disagreement, whether human references are '
           'internally consistent, and whether long requests dominate effects. All added results are descriptive or '
           'pointwise sensitivity estimates; they do not enlarge the 369-comparison multiplicity family.\n\n')
    reliability = []; vote_stability = []; soft_alignment = []; harm = []; tokens = []; length_effect = []; top_repeats = []
    for name, rows in datasets.items():
        gold, human, dispersion = reference_arrays(name, rows)
        d = designs[name]; kind = DATASETS[name]['kind']
        clear = np.array([row['metadata']['agreement_group'] == 'clear' for row in rows])
        units = [row['reference']['individual_choices' if kind == 'choice' else 'individual_scores'] for row in rows]
        for group, mask in [('All', np.ones(len(rows), dtype=bool)), ('clear', clear), ('disputed', ~clear)]:
            subset = [unit for unit, keep in zip(units, mask) if keep]
            reliability.append([NAMES[name], group, len(subset),
                                alpha_nominal(subset) if kind == 'choice' else alpha_interval(subset),
                                'nominal' if kind == 'choice' else 'interval (equal spacing assumed)'])
        if name == 'summeval_relevance_scoring.jsonl':
            for label, key in [('Experts only', 'expert_scores'), ('Crowd only', 'crowd_scores')]:
                reliability.append([NAMES[name], label, len(rows), alpha_interval([row['reference'][key] for row in rows]), 'interval'])
        if kind == 'choice':
            strict = np.isin(gold, ['A', 'B'])
            majority = np.array([max(Counter(unit).values()) / len(unit) for unit in units])
            own_category = np.array([Counter(unit).get(reference, 0) / len(unit) for unit, reference in zip(units, gold)])
            stability = np.array([stats.binom.sf(len(unit) // 2, len(unit), share)
                                  for unit, share in zip(units, own_category)])
            for group, mask in [('All strict', strict), ('Clear strict', strict & clear), ('Disputed strict', strict & ~clear)]:
                vote_stability.append([NAMES[name], group, int(mask.sum()), own_category[mask].mean(), stability[mask].mean(),
                                       100 * np.mean(stability[mask] < .75)])
            for model in MODELS:
                b = cells[model, name, 'baseline']
                for condition in ['baseline', 'pair_tie_last']:
                    a = cells[model, name, condition]
                    human_match = np.array([Counter(unit).get(choice, 0) / len(unit) for unit, choice in zip(units, a['choice'])])
                    soft_alignment.append([model, NAMES[name], condition, 'All native-category votes',
                                           human_match.mean(), majority.mean(),
                                           ci_text(interval(human_match, d)), int(strict.sum())])
                controls = {
                    'Rejection (recovered)': (cells[model, name, 'pair_rejection'], b, True),
                    'Injection first': (cells[model, name, 'pair_injection_first_directive'], cells[model, name, 'pair_injection_first_inert'], False),
                    'Injection second': (cells[model, name, 'pair_injection_second_directive'], cells[model, name, 'pair_injection_second_inert'], False),
                    'Tie last (X counted mismatch to strict)': (cells[model, name, 'pair_tie_last'], b, False),
                }
                for label, (changed, control, inverse) in controls.items():
                    choices = np.where(changed['choice'] == 'A', 'B', 'A') if inverse else changed['choice']
                    delta = (choices == gold).astype(float) - (control['choice'] == gold)
                    harm.append([model, NAMES[name], label, int(strict.sum()), ci_text(interval(delta, d, strict), 100),
                                 100 * np.mean((choices != control['choice'])[strict])])
        else:
            span = len(DATASETS[name]['criteria']) - 1; lo = DATASETS[name]['native_minimum']
            normalized_units = [np.array(unit, dtype=float) / span - lo / span for unit in units]
            human_dispersion = np.mean([np.mean(abs(unit - np.mean(unit))) for unit in normalized_units])
            for model in MODELS:
                b = cells[model, name, 'baseline']
                individual_mae = np.mean([np.mean(abs(unit - value)) for unit, value in zip(normalized_units, b['y'])])
                soft_alignment.append([model, NAMES[name], 'baseline', 'Judge-to-individual-rating MAE',
                                       individual_mae, human_dispersion,
                                       'Mean-reference MAE ' + fmt(np.mean(abs(b['y'] - gold))), len(rows)])
        for model in MODELS:
            b = cells[model, name, 'baseline']; tt = b['input_tokens']
            tokens.append([model, NAMES[name], int(tt.min()), np.median(tt), np.quantile(tt, .9), np.quantile(tt, .99), int(tt.max()), tt.mean()])
            # Stable text-derived cutoff is shared across models; do not select by model tokenization.
            words = np.array([len(row['input']['context'].split()) +
                              (sum(len(option['text'].split()) for option in row['input']['options']) if kind == 'choice'
                               else len(row['input']['text'].split())) for row in rows])
            cutoff = np.quantile(words, .9); mask = words <= cutoff
            effects = presentation_vectors(cells, model, name) if kind == 'choice' else {
                'Reverse vs native': cells[model, name, 'score_anchor_reverse']['y'] - b['y'],
                'Reader high vs low': cells[model, name, 'score_social_high']['y'] - cells[model, name, 'score_social_low']['y'],
            }
            for label, effect in effects.items():
                length_effect.append([model, NAMES[name], label, 100 * effect.mean(),
                                      ci_text(interval(effect, d, mask), 100), int(mask.sum()), cutoff,
                                      ci_text(interval(effect, d, ~mask), 100)])
            field = 'pA' if kind == 'choice' else 'y'; prefix = 'pair' if kind == 'choice' else 'score'
            repeats = np.stack([b[field]] + [cells[model, name, f'{prefix}_repeat_{i}'][field] for i in range(1, 5)])
            ranges = np.ptp(repeats, axis=0)
            if np.max(ranges) > .05:
                idx = int(np.argmax(ranges))
                top_repeats.append([model, NAMES[name], rows[idx]['id'], field, str(repeats[:, idx].round(6).tolist()), ranges[idx]])
    append('### Annotation consistency on the selected benchmark\n\nKrippendorff alpha uses nominal distance for native pairwise '
           'choices and squared interval distance for scores. It is computed directly from individual released ratings, allowing '
           'unequal rater counts. Eight SummEval ratings are also separated into three experts and five crowd raters. '
           'These are point estimates on a deliberately balanced selection, not source-corpus reliability claims or confidence '
           'intervals. Small per-item depth does not make the coefficient mathematically invalid, but prevalence, selection, '
           'restricted quality ranges, rater identities and population mixture affect its interpretation. Negative values mean '
           'observed disagreement exceeds the coefficient’s pooled chance baseline; they are not a parser failure. '
           'Comparing clear/disputed alpha is especially selection-conditioned because those groups were defined from these ratings.\n\n')
    append(table('human_alpha', ['Dataset', 'Group', 'Items', 'Krippendorff alpha', 'Distance'], reliability))
    # Re-audit the unexpected expert/crowd gap against original cached annotations and recovered articles.
    from audit_subjective_baseline_mapping import raw_summeval
    original = raw_summeval()
    for row in datasets['summeval_relevance_scoring.jsonl']:
        source = original[row['id']]
        assert source['context'] == row['input']['context'] and source['text'] == row['input']['text']
        assert source['expert'] == row['reference']['expert_scores'] and source['crowd'] == row['reference']['crowd_scores']
        assert row['reference']['score'] == np.mean(source['expert'] + source['crowd'])
    full_expert = np.array([np.mean(row['expert']) for row in original.values()])
    full_crowd = np.array([np.mean(row['crowd']) for row in original.values()])
    append(f'SummEval source re-audit passed for all 1,500 selected items: exact summary, recovered article, '
           f'three expert ratings, five crowd ratings and eight-rating mean match the cached source. '
           f'The {len(original)} unique article/summary source entries also have low expert/crowd Pearson '
           f'correlation ({corr(full_expert, full_crowd):.4f}) and Spearman correlation '
           f'({corr(full_expert, full_crowd, True):.4f}). Therefore the weak selected-sample relationship '
           'is not introduced by mismatching ratings or by selecting only 1,500 entries. '
           'Released rating slots alone do not verify annotator identity independence.\n\n')
    append('### Finite-vote majority stability\n\nFor each current strict-reference item, resample its observed votes with replacement '
           'at the original depth. The probability the same category obtains >n/2 votes is the binomial upper tail with p equal '
           'to its observed vote fraction. This is an exact empirical-resampling diagnostic, not a posterior probability that '
           'the reference is true. Unanimous items necessarily have resampling probability one; this does not establish '
           'population unanimity. HelpSteer’s released strength-sign reference is retained rather than redefined.\n\n')
    append(table('finite_vote_stability', ['Dataset', 'Group', 'Strict N', 'Observed reference vote share',
                                         'Mean empirical majority-reproduction probability', 'Items with reproduction <.75 %'], vote_stability))
    append('### Agreement with individual ratings rather than only aggregated gold\n\nFor pairwise tasks the estimate is the '
           'fraction of released native-category votes matching the model’s chosen category, averaged over all 1,500 items. '
           'The comparison value is mean maximum observed category share, a resubstitution upper reference, not a human-performance '
           'ceiling. For scores the estimate is MAE to every individual rating; the comparison value is average human absolute '
           'deviation around each item’s mean. These quantities answer different questions and must not be pooled across tracks.\n\n')
    append(table('individual_human_alignment', ['Model', 'Dataset', 'Condition', 'Metric', 'Estimate',
                                              'Human comparison value', '95% CI / mean-reference error', 'Strict N / score N'], soft_alignment))
    append('### Human-alignment changes under selected interventions\n\nThese paired, pointwise intervals compare rejection and '
           'tie conditions with baseline, and each directive with its same-position inert control. This is secondary human-reference '
           'analysis, not an objective error audit. Flips are measured only within the strict-reference subset here.\n\n')
    append(table('human_alignment_intervention_changes', ['Model', 'Dataset', 'Contrast', 'Strict N',
                                                        'Accuracy change pp [95% CI]', 'Strict underlying-text flip %'], harm))
    append('### Input size and long-item sensitivity\n\nUsage counts are provider-reported total request input tokens for the '
           'canonical baseline, not a tokenizer-independent content length. Different models’ counts need not be comparable. '
           'The sensitivity check removes only the top word-length decile, using the same text-derived cutoff for every model; '
           'it is diagnostic and does not change the benchmark. Successful API responses do not prove absence of undisclosed '
           'internal truncation or equal effective attention to long contexts.\n\n')
    append(table('baseline_input_tokens', ['Model', 'Dataset', 'Min', 'P50', 'P90', 'P99', 'Max', 'Mean'], tokens))
    append(table('long_item_sensitivity', ['Model', 'Dataset', 'Contrast', 'All-item effect pp',
                                         'Lower 90% effect pp [95% CI]', 'Lower group N', 'Word cutoff',
                                         'Upper 10% effect pp [95% CI]'], length_effect))
    append(table('largest_repeat_ranges', ['Model', 'Dataset', 'Item ID', 'Outcome', 'Baseline plus four repeat values', 'Range'], top_repeats))
    append('Largest-range examples are deliberately extreme diagnostics. They should not replace the mean, quantile and flip-rate '
           'repeatability summaries. Perplexity’s near-exact repeat outputs on most components do not imply invariance under '
           'format changes; its reader-cue effects remain large.\n\n')
    ukp = 'ukp_humor_pairwise.jsonl'; rows = datasets[ukp]
    gold, _, _ = reference_arrays(ukp, rows)
    masks = {'All': np.ones(len(rows), dtype=bool),
             'Exclude nonhumorous+nonhumorous': np.array([row['metadata']['category_pair'] != 'nonhumorous+nonhumorous' for row in rows]),
             'Non-pun jokes only': np.array([row['metadata']['category_pair'] == 'nonpun_joke+nonpun_joke' for row in rows]),
             'Strict A/B reference': np.isin(gold, ['A','B']), 'Native X-majority': gold == 'X', 'No majority': gold == 'NO_MAJORITY'}
    heterogeneity = []
    for model in MODELS:
        vectors = presentation_vectors(cells, model, ukp)
        for label, mask in masks.items():
            heterogeneity.append([model, label, int(mask.sum()), ci_text(interval(vectors['First-position'], designs[ukp], mask), 100),
                                  ci_text(interval(vectors['B-label'], designs[ukp], mask), 100)])
    append('### Targeted review of the UKP category reversal\n\nThe earlier category table shows a particularly large '
           'first-position effect for Jev on nonhumorous/nonhumorous pairs, opposite to its overall second-position effect. '
           'The following exclusions and reference strata characterize that heterogeneity; they are not a post hoc '
           'replacement of the selected benchmark. Native X-majority and nonhumorous content overlap, so these marginal '
           'tables cannot identify which feature causes the reversal.\n\n')
    append(table('ukp_heterogeneity_review', ['Model', 'Subset', 'N', 'First-position effect pp [95% CI]',
                                           'B-label effect pp [95% CI]'], heterogeneity))
    summ = 'summeval_relevance_scoring.jsonl'; rows = datasets[summ]
    groups = np.array([row['metadata']['article_id'] for row in rows])
    systems = np.array([row['metadata']['model_id'] for row in rows])
    expert = np.array([row['reference']['expert_mean'] for row in rows]); crowd = np.array([row['reference']['crowd_mean'] for row in rows])
    def centered(values):
        return values - np.array([np.mean(values[groups == group]) for group in groups])
    def system_means(values):
        return np.array([np.mean(values[systems == group]) for group in sorted(set(systems))])
    concordance = [['Experts vs crowd', corr(expert, crowd), corr(centered(expert), centered(crowd)),
                    corr(system_means(expert), system_means(crowd)), len(set(systems))]]
    for model in MODELS:
        values = cells[model, summ, 'baseline']['y']
        for label, reference in [('Experts', expert), ('Crowd', crowd)]:
            concordance.append([model + ' vs ' + label, corr(values, reference), corr(centered(values), centered(reference)),
                                corr(system_means(values), system_means(reference)), len(set(systems))])
    append('### SummEval within-article and system-mean concordance\n\nWithin-article correlations subtract each article’s '
           'selected-summary mean from both variables, reducing shared article difficulty. System-mean correlations use '
           'the 16 retained canonical source-system IDs; deduplication means some summaries represent multiple original '
           'systems, and selected article coverage is not identical for every system. These are descriptive ecological '
           'diagnostics, not equally powered model-quality ranking tests.\n\n')
    append(table('summeval_concordance_levels', ['Comparison', 'Item Pearson', 'Within-article Pearson',
                                              'Canonical-system-mean Pearson', 'Canonical systems'], concordance))
    save_tables()


def write_manifest(cells):
    paths = list(INPUT.glob('*.jsonl')) + [Path(__file__), ROOT / 'scripts/subjective_baseline_runner.py',
                                          ROOT / 'scripts/subjective_intervention_runner.py',
                                          ROOT / 'scripts/audit_subjective_baseline_mapping.py',
                                          ROOT / 'scripts/prepare_subjective_study_datasets.py',
                                          ROOT / 'datasets/example_sources/model_annotations.aligned.jsonl']
    paths = [path for path in paths if path.exists()]
    paths += [cell_path(model, name, condition) / 'simple.jsonl' for model, name, condition in cells]
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest = {'created_at_local': datetime.now().astimezone().isoformat(), 'python': platform.python_version(),
                'numpy': np.__version__, 'scipy': __import__('scipy').__version__, 'judgments': len(cells) * 1500,
                'cells': len(cells), 'bootstrap_seed': 20261011, 'bootstrap_repetitions': 2000,
                'sha256': hashes, 'canonical_baseline_policy': 'earliest-started valid matching request hash',
                'report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest()}
    # Baseline histories are required to reconstruct earliest-valid choices, not just snapshots.
    manifest['baseline_history_sha256'] = {
        (cell_path(model, name, 'baseline') / 'full_results.jsonl').relative_to(ROOT).as_posix():
            hashlib.sha256((cell_path(model, name, 'baseline') / 'full_results.jsonl').read_bytes()).hexdigest()
        for model in MODELS for name in DATASETS}
    (OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles', default='123456')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.fresh:
        text = REPORT.read_text(encoding='utf-8').split('\n## Cycle 1')[0]
        REPORT.write_text(text + '\n', encoding='utf-8')
    datasets, designs, cells, audits = load_all()
    if '1' in args.cycles:
        cycle1(datasets, designs, cells, audits)
    if '2' in args.cycles:
        cycle2(datasets, designs, cells, audits)
    if '3' in args.cycles:
        cycle3(datasets, designs, cells)
    if '4' in args.cycles:
        cycle4(datasets, designs, cells)
    if '5' in args.cycles:
        cycle5(datasets, designs, cells)
    if '6' in args.cycles:
        cycle6(datasets, designs, cells)
    write_manifest(cells)
    print('Report:', REPORT.relative_to(ROOT).as_posix(), flush=True)


if __name__ == '__main__':
    main()
