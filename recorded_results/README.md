# Sanitized recorded judgments

This archive releases the **702,000 analyzed item judgments in 468 cells**. It
contains model predictions, not dataset texts or human annotations. It permits
offline reconstruction and numerical reproduction without purchasing new API
calls. The original verbose HTTP histories remain private.

Machine-generated validation checks are in
[`../provenance/workspace_validation.json`](../provenance/workspace_validation.json).

## Contents and schema

`manifest.json` lists every compressed cell, its model/dataset/condition, record
count, and SHA-256. Each `model/dataset/condition.jsonl.gz` contains 1,500 ordered
records with these fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable hash identifying the prepared item; joins to regenerated source data. |
| `request_sha256` | Hash of the exact original request; checked against rebuilt requests. |
| `response_model` | Returned model identifier, retained for version provenance. |
| `prediction` | Choice or API score, full reported probability vector, and confidence. |
| `usage` | Available numeric input/output token counts and cost. |
| `baseline_snapshot` | Baseline cells only: the later saved snapshot, used in the original recovery-sensitivity diagnostics. |

Probability labels retain their original order. Values are not rounded or
normalized during export. Scoring indices retain their original polarity;
analysis applies the published mappings. The separate baseline snapshots do not
increase the 702,000 planned-judgment count.

Baselines use the same earliest-started valid, exact-request-matching checkpoint
as the published analyses. Other conditions preserve the analyzed successful
snapshot, checked against a matching full checkpoint. Recovery duplicates,
failed calls and transport events are excluded from this analysis archive.

There are **no source texts, human ratings/votes, annotator identifiers,
credentials, headers, error messages, provider response IDs, timestamps, run IDs,
or local filesystem paths**. Stable item/request hashes are linkage identifiers,
not a claim that the underlying source text cannot be identified by someone who
already has the original dataset. Dataset content and references are obtained
from the original releases under their respective access conditions.

## Reproduce

From the repository root, install `requirements.txt` using Python 3.11. Then:

```sh
python scripts/recorded_results.py verify
python scripts/prepare_subjective_study_datasets.py --output-dir reproduction/prepared_data
python scripts/recorded_results.py reproduce
```

If the verified official source cache is already present, add `--no-download` to
the preparation command. All source files and the recovered article set must
match the frozen hashes before sampling. The reconstruction additionally checks
all 12 prepared JSONL/analysis-file hashes, ordered item IDs, and every recorded
request hash. It refuses changed inputs instead of silently replacing the study
with a new source version. Source hosts must still make those bytes available.

`reproduction/` is Git-ignored. It receives regenerated datasets, compact
analysis-compatible checkpoints, reconstructed references and option mappings,
recomputed results, and `REPRODUCTION_VALIDATION.json`. It never replaces private
histories or frozen published results. Reconstructed `full_results.jsonl` files
are **compatibility inputs**, not original wire logs: their request bodies and
source-dependent fields are rebuilt, and historical transport metadata is absent.

The reproduction runs the existing single-experiment engine and checks all 55
tables and 369 contrasts; then checks all 450 practical-impact comparisons, 648
susceptibility associations, and 45 meta-analyses. Strings/structure must match;
numerical comparisons use relative tolerance `1e-10` and absolute tolerance
`1e-12`. The enriched primary SE/variance fields are independently checked
against the reproduced confidence intervals before cross-task pooling.

`reconstruct` performs only the joining/mapping step; `reproduce` performs that
step and the analyses. Neither contacts a model. Network access is needed only
for source downloading when the local source cache is absent.

## Maintaining this release

`python scripts/recorded_results.py export` is the maintainer-only exporter. It
requires the original local prepared data and full histories, verifies source
and selection hashes, and uses a strict output allowlist. Gzip headers omit
filenames and use a zero timestamp. Tests exercise schema exclusions, hash
failures, and choice/score round trips. Do not publish the ignored original logs
instead of this archive. Code licensing remains as stated in the root README;
this archive does not grant rights to the original datasets or model services.
