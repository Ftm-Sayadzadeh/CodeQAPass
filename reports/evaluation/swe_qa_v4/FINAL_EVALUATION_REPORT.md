# SWE-QA v4 — Final Evaluation Report

## Scope and frozen design

The evaluation covers 192 benchmark questions from four pinned Python
repositories: Flask, Requests, pytest, and Pylint (48 questions per
repository). Retrieval and answer-generation inputs are frozen under the v4
manifest; v2/v3 records are not pooled with these results. The main arms are
A0 (closed-book), B0-k5 (dense top-5 context), and A1 (CodeCompass hybrid
retrieval with deterministic metadata-derived citations). The automatic judge
uses `gemini-2.5-flash` at temperature 0. Human review uses a 30-question
stratified sample and two raters.

## RQ1 — Retrieval

At depth 5, lexical retrieval reaches Recall@5 `0.473`, semantic retrieval
`0.709`, hybrid retrieval `0.739`, and B0-k5 `0.662`. Hybrid minus B0-k5
Recall@5 is `+0.0770` with 95% stratified bootstrap CI `[+0.0348, +0.1196]`
and Wilcoxon `p=0.00067`. The corresponding MRR@10 difference is `−0.0446`,
CI `[-0.1008, +0.0108]`, `p=0.1332`; the experiment therefore supports a
recall improvement without evidence of an MRR improvement.

## RQ2 — Automatic answer quality

All 576 blinded records are usable after the authorized retry of 12 provider
HTTP-429 failures. Retries remain separate raw files; the summary uses the
latest successful record for each blind ID. Differences are on the judge's
1–20 mean score scale, with 20,000 stratified paired bootstrap resamples and
Holm correction across the three planned comparisons.

| comparison | pairs | mean difference | 95% CI | Holm p |
|---|---:|---:|---:|---:|
| A1 − A0 | 192 | −1.773 | [−2.776, −0.729] | 0.00248 |
| A1 − B0-k5 | 192 | −1.355 | [−2.127, −0.570] | 0.00248 |
| B0-k5 − A0 | 192 | −0.418 | [−1.389, +0.545] | 0.43427 |

With the pre-registered non-inferiority margin of 1.5 points, A1 is not
non-inferior to A0 or B0-k5. B0-k5 is not detectably different from A0 on the
automatic judge outcome. These are conditional model/provider measurements,
not human-equivalence claims.

## RQ3/RQ5 — Human review

The independent-rater Task-1 means are A0 `4.158` (CI `[3.575, 4.717]`), A1
`6.683` (CI `[5.867, 7.492]`), and B0-k5 `7.333` (CI `[6.458, 8.150]`). Paired
differences are A1−A0 `+2.525` (CI `[+1.542, +3.508]`), A1−B0-k5 `−0.650`
(CI `[-1.650, +0.308]`), and B0-k5−A0 `+3.175` (CI `[+2.083, +4.258]`).
The human sample therefore rates A1 above A0, while B0-k5 has the highest
point estimate; A1 versus B0-k5 is not conclusive under the frozen interval
rule.

Quadratic weighted kappa between raters is `0.637` for correctness, `0.823`
for completeness, `0.882` for relevance, and `0.189` for groundedness.
Overall-mean Spearman agreement is `0.746`. The low groundedness agreement is
a material measurement limitation and is reported rather than hidden.

For citation support, the blinded labels have strict/lenient precision and
claim coverage of X `0.806/0.881/0.913` and Y `0.788/0.894/0.913`. The two
lists are identical for 19 questions, so X/Y cannot be uniquely mapped back
to A1/A2 for those questions after unblinding. On the 11 uniquely mapped
questions only, A1 has strict/lenient precision `0.553/0.658` and A2
`0.905/1.000`; these arm-specific values are exploratory, while X/Y are the
complete blinded estimands.

Spearman agreement between the independent human overall mean and the
automatic judge mean is A0 `0.781`, A1 `0.963`, and B0-k5 `0.863` (30 paired
answers per arm). This is agreement, not equivalence.

## Interpretation

The evidence separates retrieval from answer quality. Hybrid retrieval improves
recall over B0-k5, but the automatic judge does not show a corresponding A1
quality improvement. Human review places A1 above closed-book A0 and below the
point estimate of B0-k5, with uncertainty covering the A1/B0-k5 difference.
The defensible conclusion is that better evidence recall does not automatically
translate into better final answers under the current context-integration and
generation settings.

## Speed, accuracy, and cost trade-off

Per-question generation records contain latency but no provider token-usage or
price fields, so a monetary cost ranking cannot be claimed from this run. The
observed median answer latency was approximately A0 `7.95 s`, B0-k5 `5.23 s`,
and A1 `5.09 s`; these values exclude one-time indexing/build cost and should
be preferred over the highly skewed A0 mean. All arms used the same generation
token limit, while B0-k5 and A1 additionally depend on retrieval/index state.

The practical choice therefore depends on the objective. B0-k5 is the strongest
candidate when the criterion is the observed human answer score together with
simple retrieval. A1 is preferable when deterministic metadata-derived
citations and the product's full provenance/navigation path are required, but
its current answer-generation configuration needs improvement because the
automatic judge finds a deficit and human review places it below B0-k5's point
estimate. A0 has the lowest infrastructure complexity, yet it is slower in the
recorded median and has the lowest human score. A defensible cost claim needs a
new controlled run that records prompt/completion tokens, embedding calls,
index-build cost, and provider pricing for every arm.

## Limitations and reproducibility

- Automatic judge results use one model/provider and one frozen prompt.
- Human review covers 30 questions; the independent rater is primary and the
  author's scores are sensitivity/agreement data.
- Semantic citation precision is reported with the X/Y ambiguity described
  above; structural citation validity is complete for usable A1 answers.
- RQ4 is deferred because no human-validated unanswerable set was frozen.
- Raw records, retry records, manifests, and machine-readable analyses are
  retained under `reports/evaluation/swe_qa_v4/frozen/`.

Supporting machine-readable artifacts: `retrieval_metrics.json`,
`judge_summary.json`, and `human_review_analysis.json`.
