# SWE-QA v2 controlled evaluation

State of the executed scope (2026-10-03; B0 added 2026-10-06, B0 judging incomplete). Design and rules: `EXPERIMENT_PROTOCOL.md`. Reproducibility record:
`provenance.json` (hashes of inputs, raw records, derived artifacts, prompts, repository commits, dirty-worktree flag) and
`manifest.json`; earlier versions are kept as `*_v1_superseded.*`, `provenance_pre_*.json`.

The run uses 96 questions (48 Flask, 48 Requests) from SWE-QA-Bench master `c13deac7a0d99b0ca2e593e004c4739475785b08`, with
repository snapshots and hybrid-retrieval contexts frozen in `contexts.jsonl`. Raw records are never overwritten; retries are
separate files.

## What was executed and what was not

| Item | Status | Evidence |
|---|---|---|
| A0-restricted, A0-neutral, A1, A2 answer generation | done | 96 usable answers per arm; retries in `*_retry.jsonl` |
| Retrieval metrics (lexical, semantic, hybrid) | done | `retrieval_metrics/` |
| Gemini 2.5 Flash judge, A0-neutral vs A1 | done | `judge_scores.jsonl` (96 usable per arm), `judge_summary.json` |
| Two blinded LLM reviewers (30 questions) | done | `reviewer_runs/`, `llm_review_summary.json` |
| B0 function-chunk baseline: index, retrieval, answers (k5, k10) | done | `b0/`; 96 answers per configuration (`answers_b0_k5.jsonl` + 1 retry, `answers_b0_k10.jsonl`) |
| B0 retrieval metrics at equal chunk depth 10 | done | `retrieval_metrics/metrics_depth10.json` |
| B0 judging (Gemini 2.5 Flash) | **incomplete** | 134 of 192 usable; 58 unresolved (see "B0 baseline") |
| Human scoring of the blinded samples | **not done** | CSV files prepared, no scores |
| Persian question set (protocol 4.6) | **not run** | planned |
| Online human study (protocol 4.5) | **not run** | planned |

## Arms

| Arm | File | Description |
|---|---|---|
| A0-restricted | `a0_closed_book.jsonl` | No context, but the context-only system prompt: measures refusal (93-95 of 96 refuse), not knowledge. Diagnostic only. |
| A0-neutral | `a0_neutral.jsonl` | No context, neutral system prompt, project name given (9/96 refusal-like). Closed-book baseline. |
| A1 | `a1_codeqapass.jsonl` | Hybrid retrieval context, model answer, deterministic citations from metadata. |
| A2 | `a2_model_citations.jsonl` | Fixed A1 answer plus the same context; the model selects citations post hoc. |
| B0-k5 | `b0/answers_b0_k5.jsonl` | SWE-QA-style function/class chunks, dense top-5 (same 5-chunk, 6,000-character budget and prompt as A1); no citations. Primary B0 baseline. |
| B0-k10 | `b0/answers_b0_k10.jsonl` | Same B0 index, top-10 chunks, no character cap. Secondary sensitivity analysis. |

All generation uses GLM-5.3-Flash, reasoning effort low, temperature 0, 1,200 output tokens. The reasoning-effort setting
comes from an uncommitted source change recorded in `provenance.json` (`src_worktree_dirty`).

## Results

**Retrieval (95 questions with a gold file; file-level, gold = files named in the reference answer).**

| Method | Hit@1 | Hit@5 | Recall@5 | Precision@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Lexical | 0.34 | 0.84 | 0.62 | 0.24 | 0.53 |
| Semantic | 0.89 | 1.00 | 0.83 | 0.39 | 0.94 |
| Hybrid | 0.68 | 1.00 | 0.88 | 0.33 | 0.82 |

Semantic and hybrid beat lexical (p<0.001). Hybrid covers more gold files (Recall@5 +0.04 over semantic, p=0.03) but ranks the
first gold file lower (MRR@10 -0.12, p=0.0001).

**Answer quality, automatic judge (Gemini 2.5 Flash, official SWE-QA prompt, 96 pairs, 1-20).** A1 minus A0-neutral: mean +0.44
[-0.76, 1.61], p=0.29; correctness +1.02 (p=0.11); no criterion is significant. Absolute values are not comparable with published
numbers (different judge).

**Answer quality, two blinded LLM reviewers (30 questions, 0-10; not human).** Mean over both reviewers, A1 minus A0-neutral:
groundedness +2.95 [2.17, 3.73], correctness +2.03 [0.85, 3.23], overall +1.62 [0.47, 2.77]. The reviewers differ: for the
stricter reviewer overall is +2.23, for the other +1.00 with an interval that includes zero; groundedness is significant for both.
Agreement is moderate (Spearman 0.42-0.61). The judge and the reviewers disagree on correctness and completeness; both are reported.

**Citations.** A1 structural validity 325/325 holds by construction (the citations are the retrieved blocks). A2 reproduces a
context block range exactly in 227/229 citations. Semantic precision from the LLM reviewers (30 questions): A1 66-71%,
A2 87-99%, with no difference in missing key sources.

## B0 baseline (re-implemented SWE-QA-style function-chunk RAG)

B0 is a controlled re-implementation, not the official code: it shares the embedding model, generation model, prompt and context
format with A1; chunking (function and class chunks), ranking (dense only) and depth differ.

**Retrieval at equal chunk depth 10** (95 questions with a gold file; B0 retrieved only 10 chunks, so every method is cut to its
first 10 chunks; `retrieval_metrics/metrics_depth10.json`; the 20-chunk table above is unchanged).

| Method | Hit@1 | Hit@5 | Recall@5 | Precision@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Lexical | 0.34 | 0.84 | 0.62 | 0.26 | 0.53 |
| Semantic | 0.89 | 0.99 | 0.79 | 0.51 | 0.94 |
| Hybrid | 0.68 | 1.00 | 0.87 | 0.38 | 0.82 |
| B0 | 0.75 | 0.97 | 0.74 | 0.40 | 0.85 |

Hybrid minus B0: Recall@5 +0.13 [0.08, 0.19], MRR@10 -0.02 [-0.09, 0.05] (p=0.55). Semantic minus B0: MRR@10 +0.09 (p=0.0004),
Recall@5 +0.05 (p=0.054). Hybrid is better than B0 in coverage and not in first-hit ranking.

**Answer quality, automatic judge (same Gemini 2.5 Flash judge, official prompt, 1-20, mean of five criteria), usable pairs only.**
Settings of the B0 judging: judge model gemini-2.5-flash, temperature 0, max_tokens 500, same prompt and rubric as for A0-neutral/A1,
scored in a separate pass.

| Comparison (primary, Holm over the three) | pairs | difference | 95% bootstrap interval | p | Holm p |
|---|---:|---:|---|---:|---:|
| A1 - A0-neutral | 96 | +0.44 | [-0.76, 1.61] | 0.29 | 0.88 |
| A1 - B0-k5 | 70 | -0.05 | [-1.26, 1.10] | 0.33 | 0.88 |
| B0-k5 - A0-neutral | 70 | +0.38 | [-0.77, 1.54] | 0.65 | 0.88 |

Under the available paired cases, no statistically significant quality difference was detected between A1 and B0-k5. This does not
establish equivalence. B0-k5 is the primary baseline, while B0-k10 is a secondary sensitivity analysis. The scores were produced
by an LLM judge, not human reviewers.

Secondary (not corrected for multiplicity, exploratory): B0-k10, which receives twice the chunks and no character cap, scored higher
than A1 on the 64 usable pairs (A1 - B0-k10 = -2.10 [-3.29, -0.96], p=0.002) and higher than A0-neutral (+2.15 [0.91, 3.40]) and than
B0-k5 on 44 shared pairs (+1.97 [0.82, 3.20]). The difference is confounded by context size and by which pairs are usable, so it says
that a larger context budget helped this judge's scores; it does not show that B0 chunking beats AST-aware chunking.

**B0 completeness and judging status.**

```
B0-k5 answers: 96
B0-k10 answers: 96

B0 judged usable:
- k5: 70
- k10: 64
- total: 134/192

unresolved:
- latest status length/invalid JSON: 25
- latest status connection/provider: 33
- total unresolved: 58

retry recovery: 0
judge model: gemini-2.5-flash
temperature: 0
max_tokens: 500
```

B0 judging did not finish: the judge returned truncated JSON (`finish_reason=length`; hidden reasoning used the 500-token limit) and the
provider had connection failures. Retries used the same settings and recovered nothing (a retry of the connection failures was
stopped after its first 9 records failed again, to avoid spending more quota). Comparisons use only usable pairs; the 58 unresolved
records are not scored as zero. A first B0 generation run without `CODECOMPASS_LLM_REASONING_EFFORT=low` was invalid (many truncated
or unavailable answers) and is excluded from all analysis; its raw records are kept as `b0/invalid_run1_no_reasoning_effort_*.jsonl`,
and the generation script now refuses to run without the setting.

## Claims supported and not supported

| Claim | Status |
|---|---|
| Semantic and hybrid retrieval beat lexical retrieval on these questions | supported |
| Hybrid improves coverage (Recall) but not first-hit ranking compared with semantic | supported |
| Deterministic citations are structurally valid by construction | supported (design property) |
| Structurally valid citations are not necessarily semantically supporting; model selection raises precision | supported by LLM reviewers on 30 questions |
| Retrieval-grounded answers are more grounded than closed-book answers | supported by both LLM reviewers; not by correctness/completeness of the automatic judge |
| Retrieval improves overall answer quality | **not supported** by the official-prompt judge; mixed across LLM reviewers |
| Superiority over RepoReaper, RepoRAG, CodeQA baselines or published systems | not supported (not run, or protocol differs) |
| Hybrid retrieval over AST chunks covers more gold files than function-chunk dense retrieval (B0) | supported (Recall@5, equal chunk depth 10), not for first-hit ranking |
| Answers from A1 are better than answers from a function-chunk RAG (B0-k5) | **not supported**: no significant difference detected on 70 usable pairs (not evidence of equivalence); B0 judging incomplete |
| A larger context budget raises judge scores | exploratory only (B0-k10, uncorrected, confounded) |
| Human-evaluated quality or usability | not tested |
| Results on Persian questions at SWE-QA scale | not tested |
| Generalisation beyond Flask and Requests | not tested |

## Limitations

Two repositories and 96 questions; gold files derived from reference answers (tests included when named); the judge and
the reviewers are LLMs and share LLM biases, and the judge is not the official GPT-5 judge; both repositories are public so
closed-book answers may reflect memorisation; answer length differs between arms and some answers mention "the context",
which can reveal the arm to reviewers; reasoning effort is set through an environment variable (the source change that supports it is committed in `472de47`). B0 is a re-implementation; B0 judging covers 134 of 192 answers and was done in a separate pass, so a judge drift between passes cannot be excluded; B0 retrieval depth (10 chunks) is lower than the other methods' 20, so the four-method comparison is at depth 10.

## Unavailable and superseded records

Unavailable first-pass answers (A0: 2, A1: 2, A2: 4) were retried once in separate files. The judge's earlier HTTP 403
attempt is kept in `judge_scores_v1_403_superseded.jsonl`; 17 judge records failed with HTTP 429, stay in
`judge_scores.jsonl`, and are superseded by later usable records of the same `blind_id`. No unavailable output is scored as zero.
The first human sample (`*_v1_superseded.*`) was not blind and must not be scored.

B0 records: `b0/answers_b0_k5_retry.jsonl` holds the one retried unavailable answer (requests-28); `b0/judge_scores_b0.jsonl` and
`b0/judge_scores_b0_retry.jsonl` hold all judge attempts (later usable record supersedes earlier unavailable per `blind_id`);
`b0/invalid_run1_no_reasoning_effort_*.jsonl` is the invalid first generation run, kept as evidence only.
