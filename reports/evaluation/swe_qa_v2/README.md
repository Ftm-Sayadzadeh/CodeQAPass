# SWE-QA v2 controlled evaluation

Final state of the executed scope (2026-10-03). Design and rules: `EXPERIMENT_PROTOCOL.md`. Reproducibility record:
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
| B0 function-chunk baseline | **not run** | code only (`scripts/evaluation/swe_qa_b0.py`) |
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
| Superiority over an SWE-QA-style function-chunk RAG (B0) | not tested |
| Human-evaluated quality or usability | not tested |
| Results on Persian questions at SWE-QA scale | not tested |
| Generalisation beyond Flask and Requests | not tested |

## Limitations

Two repositories and 96 questions; gold files derived from reference answers (tests included when named); the judge and
the reviewers are LLMs and share LLM biases, and the judge is not the official GPT-5 judge; both repositories are public so
closed-book answers may reflect memorisation; answer length differs between arms and some answers mention "the context",
which can reveal the arm to reviewers; reasoning effort is set through an uncommitted source change.

## Unavailable and superseded records

Unavailable first-pass answers (A0: 2, A1: 2, A2: 4) were retried once in separate files. The judge's earlier HTTP 403
attempt is kept in `judge_scores_v1_403_superseded.jsonl`; 17 judge records failed with HTTP 429, stay in
`judge_scores.jsonl`, and are superseded by later usable records of the same `blind_id`. No unavailable output is scored as zero.
The first human sample (`*_v1_superseded.*`) was not blind and must not be scored.
