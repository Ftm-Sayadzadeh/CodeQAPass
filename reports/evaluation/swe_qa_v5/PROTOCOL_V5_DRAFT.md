# SWE-QA evaluation protocol v5 (FROZEN for tier T1 on 2026-10-18)

Status: written before any v5 generation or judging; frozen for tier T1 by the owner's approval on 2026-10-18, after the
offline ranking and context stages and before any generation. Hashes are in `FREEZE_V5.json` (section 10). Nothing under `reports/evaluation/swe_qa_v4/` is modified; v4 records are
never pooled with v5 records.

## 1. Motivation and evidence (all from frozen v4 records, no new model calls)

v4 showed higher Recall@5 for hybrid retrieval but lower automatic answer quality (A1 − B0-k5 = −1.355, 95% CI
[−2.127, −0.570]). An offline replay (`scripts/evaluation/swe_qa_v5_dev_context_gate.py`, output in
`reports/evaluation/swe_qa_v5_dev/offline_gate/`) reproduces the v4 A1 context exactly for 192/192 questions and finds:

| finding | value |
|---|---|
| gold-file coverage of the top-5 ranking vs. of the context actually given to the model (A1) | 0.666 vs 0.527 |
| A1 answers containing an insufficient-evidence statement (regex, approximate) | about 98/192; mean judge score 6.2 vs 15.3 for the rest |
| top-5 slots occupied by class chunks above 2400 characters ("hubs") | 186/945; file-in-gold precision 0.134 vs 0.514 for other chunks (Pylint: 0.05) |
| Spearman rho, delivered identifier coverage vs. judge score | A1 0.59, B0-k5 0.69 |

Offline context metrics for the declared variants (gold-file coverage / reference-identifier coverage, n=192):
baseline 0.527/0.401; `hubless_soft_k5` 0.675/0.522; `hubless_soft_k10` 0.700/0.573; B0-k5 0.597/0.466. These are
diagnostics against ground truth that is never available at run time. They show where evidence is lost; they do not
show that answer quality improves.

Disclosure of researcher degrees of freedom: the diagnosis and the variant choice are post hoc and were made on the
v4 questions. Eight context variants were evaluated offline on those 192 questions (baseline, nofocus, soft_focus,
outline, outline_soft, outline_soft_k10, hubless_soft_k5, hubless_soft_k10); the two arms below were selected after
seeing those results. Earlier v5-dev smoke tests (8 questions, single judge pass, partly unusable because of provider
HTTP 429) were inconclusive and are not evidence for or against any arm.

## 2. Hypotheses

- H1 (primary, superiority): the v5a arm scores higher than the current production A1 on the mean of the five judge
  criteria (1–20 scale), paired by question.
- H2 (primary, non-inferiority): v5a is non-inferior to B0-k5 with margin Delta = 1.5 on the same scale (lower 95%
  bound of v5a − B0-k5 above −1.5). Superiority over B0-k5 is claimed only if the whole interval is above zero.
- H3 (exploratory): v5b (retrieval depth 10) differs from v5a; compared to B0-k10 under the same budget.
- Mechanism check (secondary, deterministic): delivered gold-file coverage and delivered reference-identifier
  coverage of each arm, computed offline from the indexed chunks.

## 3. Arms and frozen settings

All arms use the same generator provider/model, `temperature=0.0`, `max_tokens=1200`, `reasoning_effort=low`, and the
same 6000-character code budget. Retrieval embeddings and indexes are the v4 embedding identity.

| arm | description |
|---|---|
| A0 | closed-book, same prompt as v4 |
| B0-k5, B0-k10 | dense top-k over function chunks, class chunks name and docstring only; both under the 6000-character budget (v4's unlimited-budget k10 is not reused) |
| A1-cur | production configuration: hybrid retrieval limit 5, hard symbol focus, greedy 6000-character context |
| A1-v5a | hybrid retrieval limit 10; drop hub chunks; soft symbol focus (focus matches first, rest after); first 5 remaining chunks; greedy 6000-character context |
| A1-v5b | as v5a but all non-hub chunks of the top 10, greedy 6000-character context |

Definitions (fixed, not tuned): a hub is a `class_overview` chunk whose code exceeds 2400 characters (read from the
index; the value is a pre-declared constant `HUB_CHARS`). Soft focus reuses the production focus rules
(`_focus_on_named_symbol`) but only reorders. File paths, symbols, line ranges, source hashes and citations continue
to come from SQLite chunk metadata; no model text is used for source identity. If the assembled context is empty the
system returns the existing explicit insufficient-evidence result.

The v5 arms are implemented in evaluation scripts only. `src/codecompass/` is not modified by this protocol; any
production change requires separate owner approval after the results.

## 4. Data

| tier | questions | role |
|---|---|---|
| T1 | the 33 conan questions not used in the v4 pilot (the 15 pilot ids in `SAMPLE_FREEZE.json` are excluded): conan-2..48 minus the pilot ids | confirmatory-in-design, underpowered (section 6) |
| T2 (optional, needs owner approval for download and indexing) | one or more SWE-QA repositories never used before (for example sympy, the v4 replacement) | adds power; analysed separately and pooled with T1 only if pre-declared before generation |
| T3 | the 192 v4 questions | development-contaminated; reported as a descriptive replication, never as confirmation |

Questions, reference answers and gold files are frozen before generation. Conan retrieval rankings (semantic, lexical,
B0 dense) and every arm's context are produced once (embedding calls only) and stored before any generation
(`reports/evaluation/swe_qa_v5/t1_conan/`). For 12 of the 33 T1 questions the reference answer names no repository
file, so gold-file and identifier coverage metrics use only the questions where they are defined; judge scores use all 33.

Offline T1 diagnostic, computed before generation and not used to change any arm (gold-file coverage / identifier
coverage of the delivered context): A1-cur 0.371/0.459; v5a 0.515/0.591; v5b 0.555/0.629; B0-k5 0.587/0.629;
B0-k10 0.602/0.691. The production-context replica matches the production service for 33/33 questions. On these
unseen questions the v5 arms recover part of the loss relative to A1-cur but do not exceed B0 on these proxies, unlike on
the v4 development questions.

## 5. Procedure

1. Freeze this protocol and record SHA-256 of this file, the dataset files, the gate script and the arm
   implementations (section 10).
2. Offline stage (no LLM): build rankings and contexts for every arm; compute delivered coverage metrics; store
   contexts so that generation and judging are replayable.
3. Generation: one attempt per (arm, question) plus at most one separate retry file for provider failures. Raw records
   are append-only; the runner refuses to overwrite.
4. Judging: `gemini-2.5-flash`, temperature 0, the v4 judge prompt, blind ids shuffled with a fixed seed, **two
   independent judge passes** stored as separate files. The unit score is the mean of the available passes; a pass
   that failed (including HTTP 429) is `unavailable` and is never scored as zero. Judging is run in small batches with
   backoff, so a rate limit cannot silently drop a whole arm.
5. Analysis only after all files are complete or explicitly marked unavailable.

## 6. Statistics and decision rules

- Unit: question. Paired stratified bootstrap (20,000 resamples, fixed seed) and Wilcoxon signed-rank; Holm
  correction over the primary family (H1, H2).
- Primary endpoint: mean of the five judge criteria (1–20). Secondary: correctness, completeness, insufficient-
  evidence rate (flagged by the judge-independent rule in section 7), citation structural validity.
- Power note, stated in advance: the v4 paired standard deviation of A1 − B0-k5 was 5.64. With 33 questions the
  standard error is about 0.98 (95% interval about ±1.9), with 96 about 0.58, with 144 about 0.47. T1 alone therefore
  cannot demonstrate non-inferiority with Delta = 1.5 unless the true difference is clearly positive. T1-only results
  are labelled "promising" or "not supported", never "established".
- Adoption rule for production: requires owner approval and a lower 95% bound above zero for v5a − A1-cur on the
  pre-declared confirmatory set. A point estimate alone is not sufficient.
- Reporting: every arm, including arms that do not improve; results by repository; unavailable and failed records
  counted separately; a negative result is reported as a negative result.

## 7. Insufficient-evidence rate

An answer is counted as an insufficient-evidence refusal if the first sentence matches a fixed rule (starts with
"There is not enough retrieved evidence" or "Not enough retrieved evidence", case-insensitive) or the system returned
the explicit no-evidence result. The regex is fixed before generation. The v4 figure above used a looser pattern and
is approximate.

## 8. Threats to validity

- One judge model and one provider; two passes reduce but do not remove judge noise.
- Post-hoc design (section 1); T3 cannot confirm it.
- v5a/v5b change the number of blocks and the retrieval depth together with the hub rule; their effects are not
  separable in this protocol. Single-factor ablations are out of scope unless the owner approves them.
- Questions that name a hub class explicitly (7/192 in v4) lose the hub's own chunk under the hub rule; offline,
  delivered coverage still improved for those questions (identifier coverage 0.228 to 0.461), but this is untested
  with a model.
- Pylint retrieval is weak for abstract paraphrased questions (6/48 have no gold file in the top 10 for any method);
  gold files were checked and are literally named in the reference answers, so this is a retrieval limitation, not a
  gold-extraction error.
- SWE-QA is a public benchmark; models may have seen the repositories (closed-book A0 is strong on Flask and Requests).
- Gold-file and identifier coverage are proxies against silver ground truth.

## 9. Stop and negative-result conditions

If v5a does not improve on A1-cur in T1 and the offline coverage gain does not translate into judged quality, the
thesis conclusion remains that better evidence recall does not by itself improve final answers under the current
generation settings, with the context-loss mechanism reported as a measured diagnostic and not as a successful fix.

## 10. Freeze record

| item | value |
|---|---|
| freeze date | 2026-10-18 (owner approval in chat, after offline ranking/context stages, before any generation) |
| hashes | `FREEZE_V5.json` (this protocol, runner, gate script, conan dataset, sample freeze, rankings, contexts) |
| T2 decision | not part of this run; to be decided after T1, with owner approval for any download |

## 11. Deviation log

- 2026-10-18: the first embedding calls failed with HTTP 429 `insufficient_quota` (provider credit exhausted; a second
  attempt right after recharging failed because the recharge had not yet been applied). No records were affected. The
  ranking phase was made resumable and now writes a question's four records together; the rerun completed 33/33.
  The earlier v5-dev judge failures reported as "HTTP 429" may have had the same cause; they were not analysed.
