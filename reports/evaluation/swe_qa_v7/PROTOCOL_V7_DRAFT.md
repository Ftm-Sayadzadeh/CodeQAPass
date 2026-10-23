# SWE-QA head-to-head re-evaluation protocol v7 on a development-contaminated benchmark (frozen by FREEZE_V7.json)

Status: written before any v7 generation, timing run or judging. It becomes binding when `FREEZE_V7.json` (SHA-256 of this protocol, the runner, the analysis script, the preflight outputs, the v4 inputs and the price table) is written, which happens after the runner's dry-run passes; a change after that needs an addendum and a deviation-log entry. Nothing under
`reports/evaluation/swe_qa_v4/`, `swe_qa_v5/` or `swe_qa_v6/` is modified; v7 records are never overwritten, and failed attempts are recorded, never scored as zero.

## 1. Purpose and honest expectations

Purpose: a fully specified re-evaluation (not an independent test) of the selected CodeCompass configuration against the baselines on the four v4 repositories, measuring accuracy
(with statistical tests), token use, cost, latency and provenance, so that every claim of "better" is tied to a measured quantity.

Disclosure and expectations stated in advance:

- The v4 questions were used to design v5a and v6e. This is therefore a re-run on a development-contaminated benchmark ("semi-independent"); independent evidence comes from
  conan (v5) and streamlink/sqlfluff (v6), which are reported next to it, not replaced by it.
- A1 contexts are larger than B0 contexts (v6 data: about 3.6k characters for v5a, 5.5k for v6e, about 1.9k to 2.8k for B0) and A1 retrieval runs two searches (semantic and lexical) where B0
  runs one dense search. We therefore expect A1 to use more input tokens and to be slower per answer than B0. The claim to be tested is a quality and provenance advantage, with
  cost and latency reported as a trade-off, not as an advantage. No efficiency claim is made unless the interval excludes parity.

## 2. Data

- Benchmark: SWE-QA-Bench, dataset revision `f048f07da234b39e32937f115d027c5b71acd54e`; four repositories at the official commits of v4 (flask `85c5d93`, requests `46e939b`, pytest `5989efe`, pylint
  `44740e5`), 48 questions each, 192 in total, same frozen questions, references, indexes (`data/benchmarks/swe_qa_v4_avalai_gemini2`) and embedding identity as v4.
- Silver evidence: repository `.py` files named in the reference answer (3 questions have none; coverage metrics use the others).
- Timing and token subset (Study 2): 48 questions, 12 per repository, drawn with a fixed seed (`20261022`) from the 192 before any run; the list is stored in the freeze.

## 3. Arms

All arms use the v4 generator (`glm-5.3-flash`), `temperature=0`, `max_tokens=1200`, `reasoning_effort=low`, the production strict prompt (A0 uses the v4 closed-book prompt) and the 6000-character code budget.

| arm | description | source of its records |
|---|---|---|
| a0 | closed-book | v4 frozen (quality); re-run in Study 2 (efficiency) |
| b0_k5 | dense top-5 function chunks (our re-implementation of the SWE-QA function-chunk RAG baseline) | v4 frozen (quality); re-run in Study 2 |
| a1_v4 | production A1: hybrid limit 5, hard symbol focus (the v4 system) | v4 frozen (quality); re-run in Study 2 |
| a1_v5a | selected method: hybrid limit 10, class hubs (above 2400 characters) dropped, soft focus (ordering only), first 5, greedy 6000 characters | new in Study 1 and Study 2 |
| a1_v6e | v5a plus verbatim excerpts of chunks above 1200 characters and 8 chunks (strict prompt) | new in Study 1 and Study 2 |

`a1_v5a` is the primary method (pre-registered and validated in v5 on conan). `a1_v6e` is a secondary arm: its fresh-data advantage in v6 (+0.89, 95% CI −0.12 to +1.92) was not
significant and it was chosen after seeing results, so it is tested here as a hypothesis, not assumed. The partial-answer prompt is not used (it did not help in v6).

Considered and not included (owner decision, 2026-10-22): a `method-rescue` arm (class outline plus at most two relevant method bodies at depth 10). Its offline gain over v5a on the v6 questions (gold-file coverage 0.553 to 0.634, identifier coverage 0.512 to 0.561) is smaller than that of v6e
(0.657 and 0.587 at a similar context size), it comes mainly from the larger context, and the corresponding generation test of v6e did not reach significance. It is therefore left to future work and v7 draws no conclusion about it.

Source identity (file, symbol, line range, chunk id, citations) comes from SQLite chunk metadata only; excerpts are verbatim slices with line numbers from counting.

## 4. Study 1: answer quality (192 questions)

- New generations: `a1_v5a` and `a1_v6e` on all 192 questions. Study 1 replays the retrieval: contexts are rebuilt offline from the frozen v4 rankings, so Study 1 measures context assembly plus generation, not live retrieval; Study 2 runs retrieval live. This difference is
  stated in the results and in the threats section. Raw files are append-only.
- Comparators `a0`, `b0_k5`, `a1_v4`: the v4 frozen answers (re-judged in the new batch, see below; their v4 judge scores are kept for the stability check).
- Judge: `gemini-2.5-flash`, temperature 0, v4 judge prompt, scale 1-20 on correctness, completeness, relevance, clarity and reasoning; blind ids shuffled with the fixed seed `20260930` (the second-pass sample uses the same seed). Pass 1 covers at most 960 records (records that are not judgeable are logged, not judged); a second pass covers a stratified random 25% of the judgeable records, drawn separately for every
  arm x repository cell (12 of 48 records per cell, 240 in total), for agreement (Spearman, mean absolute difference).
- Judge input, fixed in advance: the candidate text is prepared as in v4 (file:line citation patterns removed with the v4 regular expression; other text unchanged); the judge sees only the question, the reference answer and the candidate (the official SWE-QA judge prompt, which rates the candidate against the reference),
  never the arm name, order or any metadata; ids are shuffled across arms and repositories. Answer length is reported per arm (characters), with the Spearman correlation between length and score, and the primary contrasts are repeated within length terciles as a sensitivity analysis.
- Records whose status is not `usable` or `no_evidence` (unavailable or truncated) are not judged and are counted per arm; explicit no-evidence answers are judged as in v4.
- No batch drift by design: all 576 old answers (a0, b0_k5, a1_v4) are re-judged inside the same blind batch as the new answers (at most 960 records, shuffled together with a fixed seed). The primary analysis uses these same-batch scores. The
  v4 frozen judge scores of the old answers are compared with their re-judged scores (mean paired difference with interval, Spearman) as a judge-stability result, and the primary contrasts are also reported on the frozen v4 scores as a sensitivity analysis.

## 5. Study 2: efficiency (48 questions x 5 arms x 3 repeats)

Measured per call, retrieval executed live for every retrieval arm (production hybrid search at limit 5 for `a1_v4` and limit 10 for the v5a and v6e arms, followed by the replayed assembly rules; one dense search for `b0_k5`). Tokens and cost are captured by
measurement wrappers around the generation and embedding providers in the evaluation scripts, which read the raw response (`usage`, `estimated_cost`):

- Latency: total wall-clock from question to answer; and by stage: query embedding plus retrieval (semantic search, lexical search, fusion), context assembly, generation. For `a1_v4` (production service) the generation
  time is measured by a timing wrapper around the LLM call and the remainder is retrieval plus assembly.
- Tokens: prompt tokens, completion tokens (including reasoning tokens if the provider reports them) and total, read from the provider `usage` field through a measurement wrapper in the evaluation script (`src/` is not
  modified). A preflight probe on 2026-10-22 confirmed that the provider returns `usage` (prompt, completion and reasoning tokens, cached tokens) and an `estimated_cost` field for the generator, the judge model and the
  embedding model, so tokens are exact, not estimated. If a call lacks `usage`, that record is marked unavailable for token statistics (no estimate is substituted).
- Context: number of blocks, characters, distinct files; embedding calls per question (query embeddings) and embedded query tokens.
- Four cost quantities are stored for every call and kept apart: (1) gross list-price cost = prompt tokens x input price + completion tokens x output price + query-embedding tokens x embedding price (the primary cost); (2) cache-adjusted cost = uncached prompt tokens x input price + cached prompt tokens x cached-input price
  + completion tokens x output price + embedding cost; (3) the provider's `estimated_cost.unit`; (4) the Toman value (`estimated_cost.irt`, or the unit value times the dashboard rate). Prices are read from the provider model list (`/models`, field `pricing`; on 2026-10-22: glm-5.3-flash
  input 0.15, cached input 0.03, output 0.5; gemini-embedding-2 input 0.2; gemini-2.5-flash input 0.3, output 2.5; per million tokens; two probes on 2026-10-22 confirmed that the returned `estimated_cost.unit` equals prompt tokens x input price + completion tokens x output price exactly) and recorded in the freeze. The
  provider `estimated_cost` of each call (`unit` in credit units, `irt` in Toman at its `exchange_rate`) is stored as well and compared with the computed cost. Conversion to Toman uses the dashboard rate of 271,450 Toman per credit unit (2026-10-22) and is reported as a convenience figure beside the USD value. Judge cost is reported
  separately as evaluation spend and is not part of per-answer cost.
- Percentiles: for every arm and stage (total, retrieval plus assembly, generation) the median and the 95th percentile are reported twice, over the 144 individual successful attempts of the arm and over the 48 per-question medians. Stages that do not exist for an arm
  (retrieval and assembly for `a0`) are reported as not applicable, never as zero.
- Quality-adjusted cost (secondary): the arm's list-price cost per answer from Study 2 (mean of the per-question medians) divided by the arm's mean judge score over the 192 questions of Study 1, and the same cost divided by the share of good answers (judge mean >= 15) in Study 1.
- Conditions: calls are sequential (no concurrency; parallel runs triggered provider rate limits in v6); arm order is shuffled per question and repeat with the fixed seed; three untimed warm-up calls per arm precede the run; each (question, arm) is run 3 times and
  the median of the three is the unit; timestamps are stored. Failed or rate-limited attempts (HTTP 429, network error) are logged separately and excluded from latency statistics, and their number is reported per arm.
- Provider prompt caching: repeating a question can hit the provider cache (`cached_tokens` in `usage`, cached input price 0.03 against 0.15), which can lower latency and cost of repeats 2 and 3. Mitigations declared in advance: arm order is shuffled so no arm
  systematically benefits; `cached_tokens` is recorded and its share is reported per arm; the primary cost is the list price (prompt tokens x input price, ignoring the cache), with the cache-adjusted provider cost reported next to it; the primary latency unit is the
  median of the three repeats, with the first repeat alone as a sensitivity analysis.
- Equality check: for each of `a1_v4`, `a1_v5a` and `a1_v6e` and each Study 2 question, the live context must equal the context used in Study 1. Stored and compared per question: context hash, chunk ids, source files, line ranges, number of blocks and number of omitted chunks.
  Mismatches are reported per arm and the affected questions are excluded from the efficiency comparison of that arm.
- Index and build quantities (descriptive, deterministic): chunks, embedded characters and tokens, vector and database size on disk for the A1 and B0 indexes of each repository; build wall-clock time is not measured (the existing build logs include retries and cached
  batches, and a clean rebuild would re-embed every chunk at extra cost); embedded tokens of the indexes are estimated by scaling the embedded characters with a characters-per-token ratio calibrated on a sample of chunk texts whose exact `usage` is read from the
  provider, and are labelled as estimated.

## 6. Study 3: retrieval and delivered evidence (offline, no model call)

Computed from the frozen rankings (semantic, lexical, hybrid, B0) and the stored contexts for all 192 questions: file-level Recall@5, Recall@10, Hit@5, MRR@10 for B0, lexical, semantic, hybrid, reported under two conventions: the v4 convention (the ranking is the list of distinct files
of the first 10 chunks, Recall@5 and Hit@5 use its first 5 files) and an equal-chunk-depth convention (files inside the first 5 chunks), because the production A1 retrieves 5 chunks while the v4 hybrid figure came from a deeper pool. The preflight run reproduces the frozen v4 numbers exactly
(Recall@5: lexical 0.4730, semantic 0.7093, hybrid 0.7392, B0 0.6623; MRR@10: 0.4392, 0.8312, 0.6972, 0.7419; Hit@5: 0.6984, 0.9418, 0.9153, 0.8836) and replays the production A1 context for 192/192 questions (`reports/evaluation/swe_qa_v7/preflight_final/`); and for each arm the delivered context
coverage (gold-file coverage and reference-identifier coverage of the context actually given to the model), block count and characters. Paired stratified bootstrap per comparison.

## 7. Study 4: human validation and citation support

Status: the owner confirmed on 2026-10-22 that two raters are available (the v4 independent rater and the author); it runs. The 30 v4 human-review questions (ids in `SAMPLE_FREEZE.json`) are used with exactly four arms, `a0`, `b0_k5`, `a1_v4` and `a1_v5a`, all rated anew by both raters in one blinded, shuffled package
(120 items per rater) with the v4 rubric; `a1_v6e` is not part of Study 4, so any statement about it rests on the judge only; the earlier v4 ratings are kept for a rater-stability check, not pooled. Outputs: arm means with intervals, weighted kappa, Spearman with the judge, and citation support precision and claim coverage on the cited code. This study cannot establish non-inferiority
(n is small) and is descriptive. If a rater becomes unavailable, the study is reduced to the available rater and the limitation is stated.

## 8. Hypotheses and decision rules

Primary family (Holm over two): on judge mean of the five criteria, paired by question, stratified by repository.

Decision basis, fixed in advance: superiority is decided by the Holm-adjusted two-sided Wilcoxon signed-rank p-value (below 0.05); the ordinary 95% stratified bootstrap interval of the mean difference is reported as the effect size and must also exclude 0 for the statement to be made.
Non-inferiority is decided by the lower limit of the 97.5% stratified bootstrap interval (the conservative level for two primary tests) exceeding the margin.

- H1 (superiority): `a1_v5a` − `a1_v4` > 0.
- H2 (non-inferiority, then superiority): `a1_v5a` − `b0_k5`; non-inferior if the lower 97.5% limit exceeds −1.5; superior if it also meets the superiority rule above.

Secondary family (Holm over two, reported separately, same decision basis):

- H3 (superiority): `a1_v6e` − `a1_v5a` > 0. The statement "v6e is better than v5a" is made only if H3 is supported.
- H4 (superiority): `a1_v6e` − `b0_k5` > 0. If not supported it is reported as exploratory, never as a failed or successful claim.

Exploratory (no multiplicity adjustment, labelled as such): all other pairs among the five arms (including every arm against `a0`); criteria separately; per repository; refusal, hedged and good-answer rates; unsupported-identifier share; efficiency comparisons.

Reporting rule: every measured quantity named in this protocol is reported in the results (main table or supplement) whatever its direction; the choice of emphasis in the text is the authors', but no pre-declared
measurement is dropped after the results are known.

Pre-declared efficiency contrasts (to avoid selecting comparisons after the fact): `a1_v5a` against `b0_k5`, `a1_v4` and `a0`; `a1_v6e` against `a1_v5a` and `b0_k5`. Each is evaluated on total latency, retrieval-plus-assembly latency, generation latency, prompt tokens,
completion tokens, total tokens and list-price cost.

Efficiency claims: for latency, tokens and cost the statistic is the per-question ratio (median-of-repeats of arm X divided by arm Y) summarised by the median ratio with a paired bootstrap interval. "Faster", "fewer tokens" or "cheaper"
is claimed only if the 95% interval of the ratio excludes 1; otherwise "no detectable difference". Where an arm is worse on efficiency, this is stated as such.

Overall conclusion template (fixed in advance): the method is described as better on a dimension only where the pre-declared test supports it; trade-offs are shown with a quality-versus-cost and quality-versus-latency plot (arm means with intervals).

## 9. Statistics

- Unit: question; strata: repository. Paired stratified bootstrap (20,000 resamples, seed `20260930`) for means, medians and ratios; Wilcoxon signed-rank (two-sided); Holm within each declared family.
- Effect sizes: mean paired difference on the 1-20 scale, standardised paired effect (mean difference divided by the SD of the differences), win/tie/loss counts and an exact sign test.
- Binary outcomes (refusal, hedged, good answer): risk difference with Wilson interval and exact McNemar test.
- Heterogeneity: per-repository effects with intervals; between-repository heterogeneity of the primary contrasts may be summarised by I-squared from the four per-repository estimates, but with four strata it is unstable and is reported descriptively only, without conclusions; no claim of effect in a repository whose interval contains 0.
- Judge reliability: pass agreement (Spearman, mean absolute difference), stability of the judge between the v4 batch and the new batch (section 4), and the per-criterion score distributions.
- Power note: the v4 paired standard deviation of A1 − B0 was 5.64; with 192 pairs the standard error is about 0.41 (95% interval about ±0.8), so a mean difference of about 1.2 is detectable with 80% power; smaller effects are not.
  For Study 2 with 48 pairs per contrast the standard error of a latency or token ratio is large for small effects; the p95 estimates are approximate and reported as such.
- Missing data: unavailable records are excluded pairwise and counted; no imputation.

## 10. Procedure

1. Preflight (no generation): check provider `usage`, record the prices, build the question subsets and the freeze lists, verify snapshots and indexes, replay and equality-check contexts offline, compute Study 3.
2. Freeze: SHA-256 of this protocol, scripts, subsets, contexts, rankings and the price table into `FREEZE_V7.json` (owner approval).
3. Study 1 generation, then judging (pass 1 on all 960 records in one blind batch, then the 25% second pass); raw records append-only; retries in separate files with waits for rate limits; credit errors stop the run and are logged.
4. Study 2 timing run in a single sitting, sequential, shuffled, with failures logged.
5. Analysis only after all files are complete or explicitly unavailable; Study 4 if raters are available.

## 11. Cost estimate (measured per-call costs; to be confirmed with the owner before the freeze)

Measured on 2026-10-22 with real prompts from the v6 data (3 questions per arm, 5 judge prompts; the provider reports exact `usage`): `b0` context 922 prompt and 301 completion tokens per call; `a1_v5a` 1025 and 376; the v6e/v6 context 1689 and 349;
judge 1633 prompt and 45 completion tokens (no reasoning tokens reported). With the price list this is about 0.00029, 0.00034 and 0.00043 credit units per generation call and about 0.0006 per judge call. For `a0` and `a1_v4`, which were not probed, 0.00022 and 0.00031 are assumed.
The small samples make these averages approximate (roughly +-30%).

| item | calls | approximate cost (credit units) |
|---|---:|---:|
| Study 1 generation: a1_v5a, a1_v6e on 192 questions | 384 | 0.15 |
| Study 2 generation: 48 questions x 5 arms x 3 repeats + warm-up | 735 | 0.24 |
| query embeddings | about 600 | under 0.01 |
| judge: 960 records pass 1 + 240 second pass | 1200 | 0.72 |
| total | about 1120 generation, 1200 judge | about 1.1 (budget 2 for retries and reruns) |

Unit of account: all costs are reported in the provider's credit units (the unit of `estimated_cost.unit` and of the `pricing` field per million tokens). The dashboard states 1 credit unit = 271,450 Toman on 2026-10-22 (the same value is returned in `estimated_cost.exchange_rate`);
the owner reads one credit unit as one US dollar at that day's rate, and no separate USD figure is computed by this protocol. In Toman the estimate is about 300,000 Toman (budget about 540,000 Toman). The judge is about two thirds of the cost. The actual cost is recorded
per call and compared with this estimate at the end. Wall-clock: Study 2 about 1.5 hours sequential; Study 1 generation about 1 hour; judging about 1.5 hours with two workers. The provider has run out of credit during v5 and v6 and rate-limited a parallel run; the run
stops on a credit error and resumes without rewriting records.

## 12. Threats to validity

- Development contamination of the v4 questions (section 1); mitigated only by reporting the independent datasets beside it.
- Study 1 replays retrieval from frozen rankings and Study 2 runs it live; the quality results therefore do not include live-retrieval variation.
- One judge model and provider; human validation covers only 30 questions (Study 4) and is descriptive.
- B0 is our re-implementation, not the published baseline numbers.
- Latency depends on provider load, the network and provider-side prompt caching (section 5); the numbers describe the run date and setup, not a general property. Token and cost results depend on the provider reporting usage and on the price list of the run date.
- Same generator for all arms; conclusions do not transfer to other models.
- Closed-book performance may reflect memorisation of public repositories.
- Silver evidence and identifier coverage are proxies.

## 13. Freeze record

The freeze date is the `frozen_on` field of `FREEZE_V7.json`; the price table is `prices.json` (read from the provider model list at freeze time; its hash is part of the freeze); the hashes of this protocol, `scripts/evaluation/swe_qa_v7_runner.py`, `swe_qa_v7_analyze.py`,
`swe_qa_v7_preflight.py`, the preflight outputs and the v4 inputs are in the same file. Every model-calling phase of the runner refuses to start unless these hashes still match; a runner change after the freeze needs `FREEZE_V7_ADDENDUM.json`.

## 14. Deviation log

- 2026-10-22 (before freeze): the first offline preflight run lacked the equal-chunk-depth retrieval metrics and used the wrong Recall@5 convention at first; it was corrected and rerun into `preflight_final/`. The earlier `preflight/` folder is superseded and is not used (its deletion was blocked by the
  session safety check and is left to the owner).
