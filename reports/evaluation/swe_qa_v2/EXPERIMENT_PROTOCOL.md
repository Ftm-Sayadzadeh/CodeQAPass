# SWE-QA evaluation protocol (v3): baseline comparison and human study

Status: design frozen before the B0 run. Anything run before this document (A0-restricted, A0-neutral, A1, A2, retrieval
metrics for lexical/semantic/hybrid, two blinded LLM reviews) keeps its original records. Changes after this point go into
the deviation log at the end.

## 1. Questions and claims

| ID | Question | Evidence | Allowed claim if supported |
|---|---|---|---|
| E1 | Do semantic and hybrid retrieval beat lexical retrieval on repository questions? | Section 4.1 | "better than lexical" |
| E2 | Does retrieval over AST-aware chunks with hybrid ranking (A1) improve answers over closed-book (A0N) and over a function-chunk RAG baseline (B0)? | Sections 4.2 and 4.4 | per-metric, with intervals |
| E3 | Are structurally valid citations also semantically supporting, and does model-selected attribution (A2) change that? | Section 4.3 | structural validity vs semantic precision |
| E4 | Do developers prefer retrieval-grounded answers, and is the interface usable? | Section 4.5 | human preference and usability |

Not claimed: superiority over published systems, generalisation beyond Flask and Requests, a result on non-English questions
before Section 4.6 is run.

## 2. Data and frozen settings

- Questions: SWE-QA-Bench `c13deac7a0d99b0ca2e593e004c4739475785b08`, Flask 48 and Requests 48. Repository commits
  `85c5d93c...` and `46e939b5...` (see `provenance.json`).
- Generation model: GLM-5.3-Flash, temperature 0, 1,200 output tokens, reasoning effort low. Embedding: gemini-embedding-2 (3072).
- Shared generation prompt for every context-based arm: `a1_prompt` in `run_swe_qa_v2.py` (answer from context, no citations).
- Context budget: A1 and B0-k5 use at most 5 chunks and 6,000 source characters; B0-k10 uses 10 chunks without a character cap.

## 3. Arms

| Arm | Retrieval | Context | Citations |
|---|---|---|---|
| A0N | none (project name given) | none | none |
| B0-k5 / B0-k10 | dense top-k over function and class chunks (SWE-QA style, re-implemented, `swe_qa_b0.py`) | file/symbol/lines header + chunk | none attached |
| A1 | hybrid RRF over AST chunks (`contexts.jsonl`) | same header format | deterministic from metadata |
| A2 | A1 context | A1 context | model-selected, fixed A1 answer |

B0 is a controlled re-implementation, not the official code: the official baseline needs Voyage `voyage-code-3` and DeepSeek-V3.
B0 shares embedding model, generation model, prompt, and context format with A1, so differences come from chunking, ranking, and
the extra hybrid signal. The function-chunk count (1,567 and 747) equals the AST chunk count, so the main differences are class
chunk content (name and docstring only), the lack of lexical fusion, and the ranking depth. Report this honestly.

## 4. Experiments

### 4.1 Retrieval (done, extended with B0)
File-level Hit@k, Recall@k, Precision@k (k = 1, 3, 5, 10) and MRR@10 against files named in the reference answer
(`swe_qa_retrieval_metrics.py`; requests-23 has no gold file). Add the B0 ranking (`b0/rankings.jsonl`) as a fourth method.
Paired bootstrap (20,000 resamples, seed 20260930) and Wilcoxon signed-rank for hybrid vs semantic, vs lexical, vs B0.

### 4.2 Answer quality
Primary endpoint: official SWE-QA judge prompt, Gemini 2.5 Flash, mean of five criteria, paired by question (n = 96), for
A1 vs A0N, A1 vs B0-k5, B0-k5 vs A0N. Secondary: each criterion, B0-k10, per-repository effects. Holm correction over the
three primary comparisons. The judge differs from the official one (GPT-5), so absolute values are not comparable with
published numbers; say so.

### 4.3 Citations
A1 vs A2 on the 30-question blinded citation sample: semantic precision, missing-key-source rate, plus structural validity
over all 96 questions. Structural zero errors for A1 is an invariant, reported as such.

### 4.4 LLM reviewers (not human)
Two independent reviewers (different model families), fresh sessions, only the blinded package, rubric 0-10 on correctness,
completeness, relevance, groundedness, overall. Extend the answer sample to three arms (A0N, B0-k5, A1) for the same 30
questions, 90 rows. Report agreement (Spearman, weighted kappa) and keep reviewer outputs unedited.

### 4.5 Online human study (planned)
- Items: the same 30 questions, three arms, blinded and shuffled per participant; each participant rates 10 questions
  (rotating design so every question gets at least 3 ratings).
- Participants: 8 to 15 developers; consent text, no personal data, anonymous IDs; check whether the university requires ethics review.
- Measures: 0-10 correctness, groundedness, usefulness; pairwise preference; five usability items on a short video of the
  interface; free-text comments.
- Analysis: mixed-effects or paired nonparametric tests, Krippendorff's alpha, comparison with LLM reviewers.
- No hosting of the application and no personal data collection beyond the form (project rules forbid cloud deployment without approval).

### 4.6 Persian questions (planned)
Translate the 96 questions into Persian, verified by a native speaker, and rerun 4.1 and the A1/B0/A0N generation; judge the
Persian answers with the Persian-aware reviewers, not with the English-only official judge alone. Report as exploratory
until the translation is verified.

### 4.7 Optional extensions
RepoReaper retrieval-only (full index, same embedding, same top-k) and an ablation without RRF; RepoProbe or CoReQA subset
if time allows. RepoRAG, RepoQA and CodeQA stay in Related Work.

## 5. Statistics and reporting rules
- Paired design everywhere; report effect size with 95% bootstrap intervals and Wilcoxon p; Holm correction for primary families.
- Report unavailable and truncated outputs as such; never score them as zero; retries are separate records.
- Distinguish measured, unavailable, and failed executions; keep raw records append-only.
- Sensitivity: report judge and reviewers separately when they disagree; do not choose the favourable one.
- Report known blinding leaks (answer length, wording like "the context").

## 6. Run order and cost

| Step | Calls | Notes |
|---|---:|---|
| B0 index (2,314 chunks) | about 73 embedding batches | one time |
| B0 retrieve (96 questions) | 96 embedding calls | rankings and contexts |
| B0 generate (k5 and k10) | 192 GLM calls | shared prompt |
| Judge B0 answers | 192 Gemini calls | official prompt |
| Extend reviewer package to 3 arms | 0 | two reviewer sessions by the author |

Order: B0 index and retrieval, retrieval metrics, B0 generation, judge, reviewer package, human study.

## 7. Threats to validity
Two repositories and 96 questions; reference-derived gold files; LLM judge and reviewers share LLM biases; B0 is a
re-implementation; possible memorisation of public repositories; answer length differences between arms; moderate reviewer agreement.

## 8. Deviation log
2026-10-03. Executed scope is final for the current deliverable; nothing below changes any recorded result.

- B0 (sections 3, 4.1 B0 row, 4.2 B0 comparisons): run on 2026-10-04 as a new, separately recorded step; the earlier entry that
  said B0 was not run no longer holds. The three-arm reviewer sample (4.4) was not built.
- B0 retrieval: B0 retrieved 10 chunks per question while lexical, semantic and hybrid retrieved 20, so file-level metrics
  for the four-method comparison are computed at equal chunk depth 10 (`retrieval_metrics/metrics_depth10.json`). The original
  20-chunk `metrics.json` for the three methods is unchanged.
- B0 generation, first attempt (invalid): `CODECOMPASS_LLM_REASONING_EFFORT=low`, used for A0/A1/A2, was not set, so many
  answers were truncated or unavailable (k5: 34 truncated, 26 unavailable of 96 at the time it was stopped). The run was stopped,
  its raw records were kept unmodified as `b0/invalid_run1_no_reasoning_effort_answers_b0_k5.jsonl` and `..._k10.jsonl`, and
  all B0 answers used in analysis come from a second run with the frozen setting. `swe_qa_b0_answers.py` now refuses to
  generate unless the variable equals `low`. One unavailable k5 answer (requests-28) was retried once in `answers_b0_k5_retry.jsonl`.
- B0 judging: the B0 answers were judged in a separate pass from the A0-neutral/A1 judging, with the same judge model
  (gemini-2.5-flash), official prompt, temperature 0 and rubric. The pass was interrupted by connection failures and HTTP 429
  and resumed; failed attempts remain in `b0/judge_scores_b0.jsonl`, a later usable record supersedes an earlier unavailable one,
  and no unavailable record is scored as zero. Because the passes are separate, a judge drift between them cannot be excluded.
- Roles: B0-k5 is the primary B0 comparison (same 5-chunk, 6,000-character budget as A1); B0-k10 is a sensitivity analysis.
- Section 4.4: the reviewer sample stayed two arms (A0-neutral, A1) for answers and A1/A2 for citations; the three-arm extension was not built.
- Sections 4.5 and 4.6 (online human study, Persian questions): not executed; they are future work and are not claimed.
- Human scoring of the blinded CSV files: not done. E3 and E4 rest on LLM reviewers only and are labelled as such.
- Judge: the first Gemini attempt returned HTTP 403 and is kept as superseded; the completed run retried 17 HTTP 429 records.
  Holm correction (three primary comparisons on the mean of five criteria, including B0-k5) is applied in `b0/judge_summary_b0.json`.
- B0 judging, incomplete: of 192 B0 judgements, 134 are usable (k5: 70, k10: 64) and 58 are unresolved. In the resumed pass,
  from blind id B136 onward the judge returned truncated JSON (`finish_reason=length`, `max_tokens=500` consumed by hidden
  reasoning), which repeating the same request does not fix: 16 records failed this way. The pass was then stopped by the author, so
  the remaining 42 records (B150 onward in the shuffled order) were not re-attempted after the first pass's connection failures.
  A separate network-failure retry (`b0/judge_scores_b0_retry.jsonl`, same model, prompt, temperature 0 and `max_tokens` 500) was
  started for those 42; its first 9 records all failed again with truncated JSON, so it was stopped to avoid spending more quota.
  Final state: 134 usable, 58 unresolved (25 truncated/invalid JSON, 33 connection failures never retried successfully), see
  `b0_judging_status` in `b0/judge_summary_b0.json`. Unresolved records are not scored as zero and were not re-judged with a larger `max_tokens`, because
  mixing two output limits in one score set would change the judge settings for a subset. Paired B0 comparisons use only
  questions with a usable judged answer in both arms (A1 vs B0-k5: n=70 of 96). Judging order was shuffled (seed 20260930), so the
  unresolved records are the tail of that order and not a content-selected subset; this is an argument, not a proof, and answer
  length of missing and usable records is similar (median 1,116 vs 1,166 characters for k5). A uniform re-judge of all 192 B0
  answers (and of A0-neutral and A1) with a larger limit would be a new, separately named run.
- `provenance.json` was regenerated on 2026-10-03 to include derived-artifact hashes; earlier versions are kept.
