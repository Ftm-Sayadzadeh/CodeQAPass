# SWE-QA evaluation protocol v6 (FROZEN on 2026-10-19, before any generation)

Status: written on 2026-10-18 before any v6 data, index, generation or judging exists. The owner approved running this evaluation
(option 2: fresh repositories plus a preceding protocol draft). The freeze happens after the offline stages (indexing, rankings,
contexts) and before the first generation call; hashes go to `FREEZE_V6.json`. Nothing under `reports/evaluation/swe_qa_v4/` or
`reports/evaluation/swe_qa_v5/` is modified, and v6 records are never pooled with them.

## 1. Question and background

v5 showed that dropping oversized class chunks and using symbol focus only as an ordering (arm `v5a`) improved answer quality
over the production context on 33 unseen conan questions (+2.35, 95% CI [+1.06, +3.85]) and halved the insufficient-evidence rate
(48.5% to 27.3%), but v5a stayed below B0-k5 (−1.58, CI [−3.69, +0.57]). Offline analysis then showed that the remaining gap is not
the refusal count but evidence density (A1 median block 1314 characters against 138 for B0, with more characters and less identifier
coverage) and a strict prompt (19 of 33 v5a answers contain insufficiency language against 12 for B0 and 0 for closed-book).

v6 tests two additions on top of v5a, on questions and repositories that no earlier analysis has seen:

- Idea 1, dense context: chunks above 1200 characters are cut to a verbatim question-relevant excerpt and 8 chunks are kept instead of 5.
- Idea 2, partial-answer prompt: answer the supported parts and name what cannot be established; refuse only when no context is relevant.

Disclosure: both additions were designed after looking at v4 records (offline coverage, 192 questions) and at the v5 conan results
(refusal and evidence statistics). Their constants are fixed below and were not tuned on the fresh set. The fresh set exists to remove
that circularity.

## 2. Data

- Benchmark: SWE-QA-Bench, Hugging Face `swe-qa/SWE-QA-Benchmark`, revision `f048f07da234b39e32937f115d027c5b71acd54e` (as v4). The
  dataset files are fetched from that revision (a refetch of `conan.jsonl` was byte-identical to the local v4 file, validating the URL scheme).
- Selection rule, fixed before any v6 output and before reading any question text of the chosen projects: from the 15 SWE-QA projects
  take the two smallest (official GitHub size in `reports/evaluation/swe_qa_v4/D1_METADATA_REPORT.md`) that were used in neither the v4 main
  set nor the v5 T1 set (flask, requests, pytest, pylint, conan). That gives **streamlink (42,338 KB)** and **sqlfluff (54,239 KB)**: 2 x 48 = 96
  questions. Prespecified substitute if a project exceeds the 50,000-eligible-chunk cap or cannot be indexed under the frozen indexing
  configuration: xarray (55,111 KB), decided before any generation.
- Snapshots: official commits from `repo_commit.txt` of `peng-weihan/SWE-QA-Bench` at commit `744a673b63bab77996096d278572da675defc0f9`
  (short SHAs streamlink `ab1f365`, sqlfluff `db9801b`, xarray `40119bf`), resolved to full SHAs at clone time; on mismatch stop and log a deviation.
- Reference files ("silver evidence") are the repository `.py` files named in the reference answer. For questions whose reference answer names
  none, coverage metrics are undefined; judge scores use all questions.
- Licences: repositories are cloned locally for indexing only and are not redistributed.

## 3. Arms and frozen settings

All arms use the generator and settings of v4/v5 (`temperature=0`, `max_tokens=1200`, `reasoning_effort=low`), the same 6000-character code
budget, the v4 embedding identity and the v4 indexing pipeline. Retrieval embeddings come from the same provider as the indexes.

| arm | context | prompt |
|---|---|---|
| b0_k5_p | dense top-5 function chunks (B0) | partial-answer |
| a1_v5a | v5a: hybrid limit 10, hubs dropped, soft focus, first 5 | production (strict) |
| a1_v6e | v6: v5a plus excerpts and 8 chunks | production (strict) |
| a1_v6 | same context as a1_v6e | partial-answer |

Cost-reduction amendment (made before the freeze, at the owner's request, keeping the test of the v6 idea intact): the first draft listed eight arms
(a0, b0_k5, b0_k5_p, a1_cur, a1_v5a, a1_v5a_p, a1_v6e, a1_v6) and two judge passes. Only the arms needed for the primary hypotheses and for separating the
two ideas are kept. Not run: closed-book `a0`, production `a1_cur`, strict-prompt `b0_k5`, `a1_v5a_p` (prompt effect on the v5a context) and the replication of
v5 on fresh data. Consequences, to be reported as limitations: no closed-book reference on these repositories, no fresh-data replication of v5a versus the
production context, and the effect of the partial prompt on B0 is not isolated (the B0 comparator uses the same partial prompt as v6, so the primary comparison is matched).

Constants (fixed): hub = `class_overview` chunk above 2400 characters; excerpt threshold 1200 characters; signature at most 300 characters;
8 chunks kept after hub removal; budget 6000 characters. The partial-answer prompt replaces only the insufficiency sentence of the production
system prompt (see `scripts/evaluation/swe_qa_v6_assembly.py`; a test asserts that nothing else changes). `b0_k5_p` uses the partial prompt so that the prompt effect
is not credited to retrieval when comparing with B0.

Trust boundary: every excerpt is a verbatim contiguous slice of an indexed function/method chunk; file path, symbol and exact start/end lines come from
chunk metadata and line counting, never from model text (checked offline on 983 excerpts of the v4 questions: 0 differences from the source files).
`class_overview` chunks are not verbatim slices and are never excerpted. If the assembled context is empty the system returns the explicit
insufficient-evidence result. `src/codecompass/` is not modified by this protocol.

## 4. Hypotheses and analysis

Unit: question; strata: repository. Primary endpoint: mean of the five judge criteria (1-20), mean over the available judge passes of a record. Paired
stratified bootstrap (20,000 resamples, fixed seed) and Wilcoxon signed-rank; Holm over the primary family.

Primary family:

- H1 (superiority): `a1_v6` − `a1_v5a` > 0.
- H2 (non-inferiority, margin 1.5 as in v4/v5): `a1_v6` − `b0_k5_p` has a 95% lower bound above −1.5. Superiority only if the whole interval is above 0.

Secondary (exploratory, no multiplicity adjustment, reported with the same intervals):

- decomposition of v6: `a1_v6e` − `a1_v5a` (excerpts and 8 chunks, strict prompt), `a1_v6` − `a1_v6e` (partial prompt on the v6 context);
- `a1_v5a` − `b0_k5_p` and `a1_v6e` − `b0_k5_p` (other arms against the matched B0 comparator);
- judge criteria separately (correctness, completeness), insufficient-evidence rate, hedged rate, delivered coverage, per repository.

Insufficient-evidence rate: an answer counts as a refusal if it starts with "There is not enough retrieved evidence" or "Not enough retrieved evidence"
(case-insensitive), or the system returned the explicit no-evidence result. Answers that merely contain insufficiency language later in the text are
reported separately as hedged.

Safety metric (groundedness proxy, fixed before generation): the share of backticked identifiers in an answer that appear neither in the delivered context
nor in the question (deterministic substring check, case-insensitive; identifiers shorter than 4 characters and common builtins are ignored). It is reported per
arm for every RAG arm. The partial prompt is flagged as unsafe if the `a1_v6` rate exceeds the `a1_v5a` rate by more than 5 percentage points.

Power note, stated in advance: with the v4 paired standard deviation 5.64 (A1 − B0-k5), 96 paired questions give a standard error of about 0.58 (95% interval about
±1.1). A decomposition effect below roughly 1.1 points cannot be separated from noise.

Decision rules: a claim "v6 is better than v5a" requires H1's lower bound above 0 after Holm and the safety flag not raised. A claim about the contribution of a single
idea is exploratory. Production changes in `src/` need separate owner approval. Results, including negative ones, are reported for every arm; unavailable and failed
records are counted separately and never scored as zero.

## 5. Procedure

1. Offline and embedding-only stages: download the two datasets, clone the repositories at the pinned commits, check the chunk cap, build the A1 index through the
   unchanged product indexing pipeline (with the disk embedding cache) and the B0 index, compute semantic, lexical and B0 rankings and the A1-cur service-equivalence capture,
   build and store every arm's context (with a replica check against the production service) and report delivered coverage.
2. Freeze: SHA-256 of this protocol, the runner and assembly scripts, the dataset files, the rankings and the contexts into `FREEZE_V6.json`; owner approval is already given
   for this run.
3. Generation: one attempt per (arm, question) with at most one separate retry file; raw records are append-only and never overwritten.
4. Judging: `gemini-2.5-flash`, temperature 0, the v4 judge prompt, blind ids shuffled with a fixed seed. Pass 1 covers every record; a second pass is run only if the provider
   budget allows and is stored in a separate file (v5: Spearman 0.97 between two passes). The analysis states how many records had one or two passes; when both exist
   the unit score is their mean. Failed requests are retried in small batches with backoff and recorded as `unavailable` until a usable record exists.
5. Analysis only after all files are complete or explicitly unavailable.

## 6. Threats to validity

- One judge model and one provider; a single pass (second pass optional) leaves judge noise unreduced. No human review in v6.
- The partial prompt can increase unsupported statements; the judge may not penalise them fully. The identifier-based proxy is coarse (it checks names, not claims).
- v6 changes retrieval depth, context size and excerpting together; the decomposition is exploratory.
- 96 questions from two repositories; results do not generalise to all repositories. Streamlink is plugin-oriented and sqlfluff is a parser/linter, so domain effects are possible.
- Public benchmark and repositories: closed-book performance may reflect memorisation.
- Silver evidence is reference-derived, not human gold; identifier coverage is a proxy.
- Provider cost, credit exhaustion and network errors can interrupt runs; interrupted runs are resumed without overwriting records (recorded in the deviation log).

## 7. Stop and negative-result conditions

If H1 is not supported, v5a remains the validated improvement and v6 is reported as not shown to help. If the safety flag is raised, the partial prompt is reported as
increasing unsupported content regardless of the judge score. A negative or inconclusive v6 result does not affect the v5 result.

## 8. Freeze record

| item | value |
|---|---|
| freeze date | 2026-10-19 (owner approval in chat; after indexing, rankings and contexts, before any generation) |
| hashes | `FREEZE_V6.json` (this protocol, runner/assembly/gate/environment scripts, datasets, commits, rankings, contexts) |

Offline stage result (computed before generation; no arm was changed because of it). The production-context replica matches the production service for 96/96
questions. Delivered coverage on the fresh questions (gold-file coverage / reference-identifier coverage; gold files exist for 49 of 96 questions, 47 of them
sqlfluff and only 2 streamlink, because most streamlink reference answers name no repository file): B0-k5 0.636/0.419; A1-cur 0.471/0.455; v5a 0.553/0.512;
v6 0.657/0.587. Streamlink therefore contributes mainly through the judge scores, not through the coverage metrics.

## 9. Deviation log

- 2026-10-19 (before freeze): indexing was interrupted twice by provider errors (network `ConnectionError`, then exhausted credit); the disk embedding cache keeps all
  completed batches and the indexing steps are rerun unchanged. Design amended to four arms and one mandatory judge pass (section 3); no generation had been run.
- 2026-10-19 (after freeze, during generation): the four arms were started in parallel and the generator provider returned HTTP 429 `Rate limit reached for requests`
  (not a credit problem). Each arm kept 5-7 usable answers; the remaining attempts were recorded as `unavailable` in the raw files, which are preserved unchanged
  (`generation_<arm>.jsonl`, 77-82 records each; the processes were stopped before all 96 questions were attempted). Continuation: the runner's retry mode now processes
  every question without a usable record in the original file (unavailable or never attempted) into a separate `generation_<arm>_retry.jsonl`, with six attempts and waits of
  15, 30, 60, 120 and 240 s on failure, and the arms are run one after another. The judge default was lowered to 2 workers. No context, prompt, arm, hypothesis or analysis
  rule changed; the amended runner hash is in `FREEZE_V6_ADDENDUM.json`. Raw records of the failed attempts are never converted to scores.
- 2026-10-19 (after freeze, during judging): the account credit was exhausted again after 153 of 384 judge records (all arms interleaved by the blind shuffle); the remaining
  attempts were recorded as `unavailable` (14 records) and the run was stopped. The runner now recognises credit errors and stops the pass after one recorded failure instead
  of attempting every remaining record. No score, context, prompt or analysis rule changed; amended runner hash in `FREEZE_V6_ADDENDUM.json`.
