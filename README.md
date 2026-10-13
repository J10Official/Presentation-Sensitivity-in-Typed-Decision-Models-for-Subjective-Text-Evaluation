# Presentation Sensitivity in Typed Decision Models for Subjective Text Evaluation

Code, validated results, and manuscript for an audit of **Jev, Clef Flash, and Perplexity Decider** on humor, helpfulness, and summarization. The central question is whether judgments change when the evaluated text stays the same but its presentation changes.

The study contains **9,000 selected items**, **468 endpoint–component–condition cells**, and **702,000 planned item judgments**. It separates position from identifier assignment, directional probability/score shifts from changed decisions, and human-reference alignment from robustness. Strong repeatability does not necessarily imply presentation robustness, and effects can reverse across tasks.

[Read the anonymous manuscript](paper/output/current/manuscript.pdf). Choice and score interfaces are evaluated separately. The paper reports task-level effects, decision consequences, and human alignment, with complete methods and secondary analyses in its appendix.

[Anonymous code and results repository](https://anonymous.4open.science/r/Presentation-Sensitivity-in-Typed-Decision-Models-for-Subjective-Text-Evaluation-6780/): data-preparation, collection and analysis scripts; aggregate results; **sanitized item-level model predictions**; request specifications; and validation tests. The [recorded-results archive](recorded_results/README.md) supports offline empirical reproduction without new API calls.

## Study and human references

Every component has 1,500 unique selected records: 750 clear-agreement and 750 disputed-agreement records, defined within its own source. Pairwise strict-reference winner positions are balanced; scoring preserves eligible score-band proportions within agreement strata, rather than forcing a uniform score distribution. This is a designed benchmark, not a source-population probability sample.

| Task | Native pairwise component | Native scoring component |
| --- | --- | --- |
| Humor | [UKP Humor](https://github.com/UKPLab/acl2019-GPPL-humour-metaphor): five A/B/neither votes; four content categories and ten category-pair cells | [Humicroedit](https://zenodo.org/records/3969509): mean of five humor ratings, 0–3 |
| Helpfulness | [HelpSteer2 Preference](https://huggingface.co/datasets/nvidia/HelpSteer2): five judgments; sign of the published aggregate preference strength | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2): mean of five helpfulness ratings, 0–4 |
| Summarization | [OpenAI summary comparisons](https://github.com/openai/summarize-from-feedback#human-feedback-data): majority of 7–23 distinct-worker votes per retained pair | [SummEval](https://github.com/Yale-LILY/SummEval): mean of eight relevance ratings, three expert plus five crowd, 1–5 |

Pairwise and scoring components are **parallel native-reference studies**, not a controlled causal comparison of interfaces. Human aggregates are subjective references, not objective truth. See the paper for eligibility, reuse caps, source clustering, native no-preference categories, and rubric definitions. Source revisions, checksums, and first-party URLs are recorded in [the dataset manifest](provenance/dataset_manifest.json) and preparation script; SummEval articles are recovered from the original CNN/Daily Mail story archives.

## Investigations

The collected conditions cover position × A/B identifiers; numeric identifiers; fabricated AI/Human attribution, reader feedback and prior-judge cues; matched-control option injection; selection versus rejection; explicit no-preference placement; score-anchor reversal and three-/seven-level rubric variants; and four unchanged-request repeats beyond baseline. Length association, score compression, human alignment/disagreement, and reference-population sensitivity are offline analyses of these judgments.

The primary ledger has 369 contrasts. Secondary analyses comprise 450 practical-impact comparisons, 648 source-call-disjoint susceptibility associations, and 45 comparable cross-task meta-analyses. Inference accounts for source reuse; the primary discovery rule is BY adjustment, with a separate secondary family. CIs are pointwise. Cross-task synthesis uses REML with modified Knapp–Hartung intervals, Q, I², tau², and leave-one-task-out diagnostics. Only three selected task components enter each pool.

## Repository map

| Path | Purpose |
| --- | --- |
| `scripts/prepare_subjective_study_datasets.py` | First-party downloads, deterministic selection, agreement/category quotas, and winner orientation |
| `scripts/openrouter_api.py` | API transport only: faithful serialization, retry classification, exact-body logging |
| `scripts/subjective_baseline_runner.py` | Choice/score request construction, parsing, ordered checkpoints, and payload-hash resume |
| `scripts/subjective_intervention_runner.py` | Deterministic interventions and original-candidate/score-coordinate mappings |
| `scripts/run_subjective_interventions.py` | Sequential condition waves with parallel dataset workers |
| `scripts/run_decision_model_comparisons.py` | Interface-matched Clef/Perplexity launcher and supervision |
| `scripts/analysis_level1.py` | Audit/enrich saved single-experiment results; optional separate full reconstruction |
| `scripts/analysis_level2.py` | Practical impacts, shared susceptibility, and cross-task synthesis; saved-result audit mode |
| `scripts/analyze_full_subjective_study.py` | Current single-experiment calculation engine used by level 1, not an obsolete alternative |
| `scripts/audit_subjective_baseline_mapping.py` | Independent source/reference/request mapping checks; also supplies the SummEval source loader |
| `reports/full_study_analysis/` | Descriptive tables, primary contrasts, collection/mapping audits and source manifest |
| `reports/validated_analysis/` | Validated primary effects, cross-experiment/meta-analysis results and statistical audits |
| `reports/paper_exports/` | Code-generated CSV tables, claim/figure evidence, request specification and presentation checks |
| `recorded_results/` | Sanitized predictions for all 702,000 analyzed judgments, compressed by cell, with checksums |
| `scripts/recorded_results.py` | Allowlist export, archive verification, source-data reconstruction, and full offline numerical reproduction |
| `paper/` | Manuscript LaTeX, bibliography/template, imported table TeX, used figure and submission PDF |
| `tests/` | Offline request, mapping, parsing, retry/resume, orchestration, and numerical contract tests |
| `provenance/` | Source/preparation hashes, original analysis lineage, release inventory, and current validation summary |

## Quick start: no API calls

Use Python 3.11. Data preparation and API collection use the standard library; analysis and presentation dependencies are pinned here.

Data/input/output CLI paths are relative to the repository root, including when a script is launched from another directory. `build_paper.py --output-directory` is relative to `paper/`. Absolute data/output arguments are rejected; paths to installed TeX executables may be absolute. Published manifests use relative paths with portable `/` separators.

```sh
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/verify_workspace.py
python scripts/verify_paper.py
```

These commands work from a public-only checkout without credentials or private data. The public-file check verifies release hashes; the paper check maps claims, table cells, and figure coordinates to saved results and checks the packaged PDF. Statistical audits are machine-generated JSON in `reports/validated_analysis/`; [the release validation summary](provenance/workspace_validation.json) records packaging checks. `python scripts/recorded_results.py verify` additionally checks every compressed prediction file, its allowlist schema, and full study coverage.

## Data and recorded-output availability

The published `recorded_results/` archive contains model predictions for all 468 cells / 702,000 analyzed item judgments. It excludes source texts, human annotations, credentials, headers, error messages, timestamps, run IDs, provider response IDs, and local paths. Stable item/request hashes support joining and validation. The canonical baselines and later baseline snapshots are retained separately so recovery-sensitivity diagnostics also reproduce. No prediction values are rounded during export.

Original source caches, prepared data and verbose HTTP histories are excluded from the release. Preparation and collection create Git-ignored `datasets/`, `prepared_data/` and `experiment_results/` directories when used. Dataset texts and human references are reconstructed from the first-party releases under their access conditions. Illustrative examples remain in the manuscript and request-example files. Original source hosts must still provide the frozen bytes.

For full offline statistical reproduction, prepare the source data and run:

```sh
python scripts/prepare_subjective_study_datasets.py --output-dir reproduction/prepared_data
python scripts/recorded_results.py reproduce
```

Add `--no-download` to preparation if the official source cache is already present. Every source file and the recovered SummEval article set must match the frozen checksums **before sampling**; UKP and HelpSteer2 additionally use pinned revisions. The sampler uses seed `20261010` and immutable record IDs. JSONL line endings reproduce the original collection bytes across operating systems. Changed downloads are rejected rather than silently accepted. Downloading releases does not grant new redistribution rights.

Reconstruction checks all prepared-file hashes and all 702,000 request hashes. The existing analysis engines then regenerate 55 tables, 369 primary contrasts, 450 practical-impact comparisons, 648 susceptibility associations, and 45 task pools and compare them with the frozen results. Outputs and a validation report go into ignored `reproduction/`; frozen outputs and original logs are untouched. See [archive documentation](recorded_results/README.md) for schema, selection rules, omissions, and numerical tolerances. `reconstruct` runs only the joining step; `reproduce` also runs the analyses. Neither calls a model.

For maintainers who retain the original private histories, the earlier audit commands remain available:

```sh
python scripts/verify_workspace.py --full
python scripts/analysis_level1.py
python scripts/analysis_level2.py --audit-only
```

The full workspace check and level-1 maintainer audit require the original private source/prepared/history folders restored in their documented locations. Level 2 `--audit-only` validates saved numerical outputs without private histories or file writes. Public empirical reproduction uses `scripts/recorded_results.py reproduce`. Level 1 audits and enriches the single-experiment engine in `scripts/analyze_full_subjective_study.py`; level 2 implements the cross-experiment analyses used in the paper. `scripts/analysis_level1.py --rebuild` explicitly rebuilds single-experiment results separately. Neither script is obsolete. An empirical level-2 run can regenerate its Markdown audit report locally under `reports/validated_analysis/`; that optional report is excluded from the release.

## Inspect requests before collection

```sh
python scripts/subjective_baseline_runner.py --preview --output-root ../local-study-work/request-preview
```

[The exact request specification](reports/paper_exports/request_specification.json) shows every component/condition with content placeholders and response-coordinate mappings. Real previews use locally prepared content. Each request has one item/question; full option content is in choice criteria, shared context is in state, and score criteria contain ordered anchors. **Gold labels, individual human votes, worker notes, and reference scores are never sent.**

To make new billable calls, copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`, and explicitly choose `--live`. Route names below are the ones used in the recorded study, not a promise of current provider availability.

```sh
python scripts/subjective_baseline_runner.py --live --batch-size 20
python scripts/run_subjective_interventions.py --live --batch-size 20
python scripts/run_decision_model_comparisons.py --live --phase all --batch-size 20
```

Use a different `--output-root` for independent new runs. Resume matches item ID **and exact request hash**; changed payloads are not silently reused. Checkpoints preserve complete full-result histories and ordered simplified JSONL snapshots. Six parallel dataset workers with batch size 20 can create up to 120 outstanding requests per endpoint. The historical routes are `typesafe/jev-1.13`, `cloudflare/clef-flash`, and `perplexity/pplx-decider-v1.1-27b`, using the OpenRouter Decisions endpoint rather than Chat Completions.

## Rebuild figures, tables, and paper

```sh
python scripts/build_paper_results.py
python scripts/build_paper.py
python scripts/verify_paper.py
```

The display builder reads validated results without model calls or new statistical fits. `scripts/paper_tables.py` supplies table formatting; code-generated CSV/JSON evidence lives in `reports/paper_exports/`. The two core analysis directories are both required: descriptive/reference tables and primary contrasts feed the validated effects and meta-analyses. They are complementary, not old/new copies.

`paper/generated/` contains only imported LaTeX tables and request examples. These are real manuscript dependencies: deleting the directory breaks compilation. The bibliography, template files and `paper/figures/figure3_score_levels_ranks.pdf` are also required. `paper/output/current/manuscript.pdf` is the submission artifact. Unused plots, duplicate exports, build caches and standalone narrative validation reports are excluded.

The paper builder requires installed `pdflatex` and `bibtex`; executable paths can be supplied using `--latex` and `--bibtex`. It compiles one PDF with one bibliography and internal appendix references, without automatic package installation. Render and inspect pages after layout changes.

## Interpretation and release notes

Human references are fixed aggregates, and available worker identifiers leave some shared-rater dependence unresolved. Score normalization assumes equal spacing of ordinal anchors. Authorship and feedback experiments measure sensitivity to synthetic metadata on unchanged content. Exact requests, responses, served-model identifiers, and retries remain in the recorded provenance. The paper describes the evaluated task configurations and separate choice/score studies.

Please cite the manuscript and the original dataset papers listed in its bibliography when reusing this work. A manuscript citation will be added when publication metadata exists. No license for the project's own code has been selected yet; availability alone does not grant a reuse license. Third-party dataset and ACL-template notices remain applicable. Do not link a non-anonymous author repository in a double-anonymous submission.

The release inventory contains the current manuscript, source code, tests, validated aggregate outputs, and sanitized item-level predictions. Older manuscript versions, reviewer notes, planning documents, and build caches are excluded. Local agent context is ignored. Original source data and verbose histories remain ignored. Paths in the package are repository-relative; PDF author metadata is anonymous. Git history and hosting-account identity must also be checked when choosing the anonymous distribution channel.
