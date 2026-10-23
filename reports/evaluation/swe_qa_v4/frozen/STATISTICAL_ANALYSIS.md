# SWE-QA v4 — Frozen Statistical Analysis

This report is derived from the frozen v4 inputs and does not modify any raw
experiment record. The four repositories are analysed as strata: Flask,
Requests, pytest, and Pylint (48 questions per repository; 192 questions in
total). Earlier v2/v3 records are not pooled with v4.

## Retrieval (RQ1)

Metrics are computed at depth 10 against the benchmark's silver gold-source
metadata. Three questions have no usable gold source and are excluded from
retrieval scoring (`n = 189`). Bootstrap intervals are stratified by repository
(20,000 resamples; seed `20260930`).

| method | Recall@5 | MRR@10 | Hit@5 |
|---|---:|---:|---:|
| lexical | 0.4730 | 0.4392 | 0.6984 |
| semantic | 0.7093 | 0.8312 | 0.9418 |
| hybrid | 0.7392 | 0.6972 | 0.9153 |
| B0-k5 | 0.6623 | 0.7419 | 0.8836 |

The pre-registered paired comparison of hybrid against B0-k5 gives a
Recall@5 difference of `+0.0770` (95% bootstrap CI `[+0.0348, +0.1196]`,
Wilcoxon `p = 0.00067`) and an MRR@10 difference of `-0.0446` (95% CI
`[-0.1008, +0.0108]`, `p = 0.1332`). Thus the evidence supports a recall gain,
but not an MRR gain. The full machine-readable output is
`retrieval_metrics.json`.

## Answer quality (RQ2)

The automatic judge used `gemini-2.5-flash`, temperature `0`, and
`max_tokens=2000`. Blinding and order were deterministic (seed `20260930`).
The initial pass contained 12 provider HTTP 429/rate-limit failures. After the
authorized retry, all 576 blinded records are usable. Initial and retry files
remain separate; the summary uses the latest successful record for each blind
ID.

Differences are on the judge's 0–10 score scale and are computed per question.
Intervals are stratified paired bootstraps (20,000 resamples); p-values are
two-sided Wilcoxon signed-rank tests, with Holm correction across the three
planned comparisons.

| comparison | pairs | mean difference | 95% CI | Wilcoxon p | Holm p |
|---|---:|---:|---:|---:|---:|
| A1 − A0 | 192 | −1.773 | [−2.776, −0.729] | 0.00090 | 0.00248 |
| A1 − B0-k5 | 192 | −1.355 | [−2.127, −0.570] | 0.00083 | 0.00248 |
| B0-k5 − A0 | 192 | −0.418 | [−1.389, +0.545] | 0.43427 | 0.43427 |

The protocol's non-inferiority margin is 1.5 points, with the criterion that
the lower CI bound must exceed `−1.5`. A1 is **not non-inferior** to A0
(lower bound −2.776) and **not non-inferior** to B0-k5 (lower bound −2.127).
The observed A1 deficit is significant after Holm correction. B0-k5 is not
detectably different from A0 on this judge outcome.

Repository-stratified means (same score scale) are included in
`judge_summary.json`; they are descriptive and are not treated as separate
confirmatory tests. The largest A1 deficits are in Flask and Requests, while
pytest is near parity with A0.

## Reproducibility and limitations

## Citation structural check (RQ3, automatic portion)

The 192 frozen A1 records contain 190 usable answers (the remaining two are
explicit no-evidence outputs). All 190 usable answers carry citations: 425
citation objects in total. Every citation passed the required schema check
(`chunk_id`, source file, and start/end line fields), and every observed line
range was positive and ordered (`start_line <= end_line`). This is a
structural/provenance check only; semantic citation support still requires the
frozen human-review sample and is not inferred from schema validity.

- Raw answer and judge files are append-only; retries are stored separately.
- Unavailable outputs are excluded from paired quality inference, never scored
  as zero.
- The automatic judge is one model/provider and is not a substitute for the
  frozen human-review protocol (RQ3/RQ5).
- These results support retrieval and conditional answer-quality claims only;
  they do not establish human citation precision or generalisation beyond the
  four pinned repository snapshots.

Machine-readable provenance: `PRIMARY_INPUT_MANIFEST.json`,
`retrieval_metrics.json`, and `judge_summary.json`.

Note (repository publication): `PRIMARY_INPUT_MANIFEST.json` is withheld from
version control because its `raw_sources` paths are local absolute paths. Its
content is unchanged; the raw-source files it lists are committed under
`main_avalai_gemini2/` and `main_avalai_gemini2_flask_requests/`. A sanitized,
repository-relative version will be added separately. The statistics above do
not depend on the manifest file itself.
