# SWE-QA evaluation protocol v4 (DRAFT, not frozen)

Status: draft. It becomes binding only when every item in section 10 ("Open decisions") is closed, the protocol is committed,
and the freeze commit is recorded in `manifest.json`. Nothing in section 5 may be run before that. The v2/v3 runs
(`reports/evaluation/swe_qa_v2/`, Flask and Requests, 96 questions) are a **pilot** for this protocol; their records stay
unchanged and are not pooled with v4 results.

## 1. Claims and research questions

The central claim is deliberately not "better answers". It is:

> CodeQAPass is a structured repository-QA pipeline whose answers carry metadata-derived, traceable citations, with answer
> quality that is competitive with a closed-book model and a function-chunk RAG baseline.

| ID | Question | Primary endpoint | Allowed claim if supported |
|---|---|---|---|
| RQ1 | Do the retrieval methods find the evidence files? | Recall@5, MRR@10 (silver files) | "finds more/fewer reference-derived files" |
| RQ2 | Is A1 answer quality non-inferior to closed-book (A0) and to the function-chunk baseline (B0-k5)? | mean of five judge criteria | "non-inferior within the pre-set margin, for these repositories and this judge" |
| RQ3 | Are citations traceable and do they support the claims? | structural validity (all); semantic precision (human sample) | structural validity as a design property; semantic precision measured |
| RQ4 | (deferred) behaviour when the repository has no evidence | none | no claim; see 5.4 |
| RQ5 | Do human ratings agree with the automatic judge and with each other? | weighted kappa, Spearman | agreement level, not equivalence |
| RQ6 | Does it work for Persian questions? | Recall@5 drop, judge mean | exploratory |

Not claimed: a reliability claim about refusal or insufficient-evidence handling; superiority of answer quality over every baseline; superiority over published systems; generalisation beyond
the selected repositories; a deterministic abstention mechanism; human-evaluated usability.

## 2. Data

- Benchmark: SWE-QA-Bench (Hugging Face `swe-qa/SWE-QA-Benchmark`), dataset revision `f048f07da234b39e32937f115d027c5b71acd54e`
  (D0 closed; see `D1_METADATA_REPORT.md`). The release lists 15 Python projects with 48 questions each.
- Main set: four repositories, 48 questions each = 192 questions: flask, requests, pytest, pylint (amended before execution). They
  were selected before any model execution to provide domain and size diversity; GitHub stars were used only as a deterministic
  tie-breaker and are not a validated measure of memorisation. Sphinx and Astropy are retained as prespecified optional extensions,
  not part of the primary four-repository analysis.
- Repository snapshots: the official commits in the SWE-QA-Bench file `repo_commit.txt` (read at repository commit
  `744a673b63bab77996096d278572da675defc0f9`); full SHAs are resolved and verified at clone time and recorded in `manifest.json`.
  The same snapshot is used by every arm. Full SHAs resolved through the GitHub API on 2026-10-06 (existence confirmed):

  | Project | Full commit SHA |
  |---|---|
  | flask | `85c5d93cbd049c4bd0679c36fd1ddcae8c37b642` (from v2 provenance) |
  | requests | `46e939b5525d9c72b677340985582b04b128478a` (from v2 provenance) |
  | pytest | `5989efe3efec31a9d14f550d89903b6311dfc828` |
  | pylint | `44740e5e5cdd2b80321a4f0ed4254213d058b9f6` |
  | sphinx | `6c9e3209c4ebc81c7c09f6a30eeaa4651f32f461` |
  | astropy | `0a041d38fbd0014298d4cf340d9f891c4522093f` |
  | sympy (replacement) | `3c817ed8ab551d71e36907ce869afe5e4ca789ff` |
  | conan (pilot) | `52f43d98fe44ab92d01317110488d0670c3e9e64` |

  Clone control (mandatory): clone only at execution time; `git rev-parse HEAD` must equal the SHA above. On any mismatch the run stops, a
  deviation is logged, and nothing is substituted automatically.
- Indexing budget: at most 50,000 eligible chunks/vectors per added repository and 200,000 in total; non-Python and excluded files do not
  count. If a selected repository exceeds 50,000 eligible chunks under the frozen indexing configuration, it is replaced by its prespecified
  substitute before answer generation (astropy -> sympy). No replacement is made after observing model results. Report per repository:
  actual chunks, embeddings produced, chunks excluded, and whether the cap was triggered.
- Pilot repository: the smallest SWE-QA repository not among the four, measured by the official GitHub repository size recorded in
  `D1_METADATA_REPORT.md` before execution. Of the nine remaining projects this is `conan` (36,010 KB; streamlink is 42,338 KB).
  If this metadata is found to be wrong, the real smallest repository replaces it and the change is logged.
- Reference files ("silver evidence"): repository `.py` files named in the reference answer. They are reference-derived, not
  human gold, and are reported as such.

## 3. Systems and frozen settings

| Arm | Retrieval | Context | Generation path |
|---|---|---|---|
| A0 | none (project name given) | none | neutral closed-book prompt (same as A0-neutral of v2) |
| B0-k5 (primary baseline) | dense top-5 over function and class chunks | same header format, 5 chunks, 6,000 characters | same prompt builder as A1 |
| A1 | hybrid RRF over AST chunks, top-5 | 6,000 characters | `GroundedQAService` (product path) |
| B0-k10 | dense top-10, no character cap | | sensitivity only; never in a primary comparison |
| A2 | A1 context | | post-hoc model citations on the fixed A1 answer; used only in RQ3 |

- Generation model: the pinned GLM model used in v2 (`glm-5.3-flash`), temperature 0, **`max_tokens=1200`** for every arm,
  `CODECOMPASS_LLM_REASONING_EFFORT=low`. The generation scripts abort unless the variable equals `low`.
- `max_tokens=1200` is a research-configuration decision made before the pilot. The product default is 180
  (`QARequest.max_tokens`); the v4 results describe the research configuration, and this is stated as a limitation.
- A1 runs through `GroundedQAService.answer` with `retrieval_method="hybrid"`, `retrieval_limit=5`, `max_context_chars=6000`.
  This includes the product's named-symbol focus step, which the v2 harness did not have. B0 does not use that step, so
  the A1-vs-B0 comparison is system-versus-baseline; component effects are read from the retrieval ablation (RQ1).
- B0 is a controlled re-implementation of the function-chunk baseline (the official one needs Voyage and DeepSeek). It must be
  named that way everywhere.
- Embedding: `gemini-embedding-2` (3072) through one fixed Google AI Studio endpoint for the primary run. All four repositories and
  retrieval arms must use this exact provider/model/dimension identity; earlier AvalAI v2 records remain a pilot and are not pooled.
  Retrieval depth for RQ1: 10 chunks for every method.
- Judge: `gemini-2.5-flash`, official SWE-QA prompt unchanged, criteria **correctness, completeness, relevance, clarity, and
  reasoning**, each scored 1-20; the primary endpoint is their mean. Temperature 0, **`max_tokens=2000`**.

## 4. Stage 0: pilot (not analysed)

Repository: the pilot repository of section 2 (`conan`). Sample: the first 15 questions of the conan order in `SAMPLE_FREEZE.json`
(15 questions x 3 arms A0, B0-k5, A1 = at most 45 generation calls; if judged, at most 45 judge calls). The first 3 of them are the rater
calibration items. Purpose, and nothing else: answers complete; `finish_reason`; sufficiency of `max_tokens=1200` for generation; sufficiency
of `max_tokens=2000` for the judge and validity of the judge JSON. The pilot must not be used to choose a prompt, a token limit or any setting
that changes results; a failed check is reported and the protocol is amended through the deviation log, not tuned for a better score. No quality
score from the pilot enters any result. Pilot records are kept and excluded from every analysis. The pilot needs a clone of the pilot repository
(about 36 MB by GitHub size) and model API calls, and is run only after a separate explicit approval.

## 5. Experiments

### 5.1 RQ1 retrieval
Methods: lexical, semantic, hybrid (AST chunks) and B0 (function chunks), depth 10. Metrics per question: Recall@5
(primary), MRR@10 (primary), Precision@5, Hit@1/3/5, latency, index size. Questions without silver files are excluded and
listed. Paired stratified bootstrap and Wilcoxon for hybrid-vs-B0, hybrid-vs-semantic, semantic-vs-B0, semantic-vs-lexical.

### 5.2 RQ2 answer quality
- Judging: all A0, A1 and B0-k5 answers (and B0-k10 for sensitivity) are pooled, shuffled with a fixed seed, blinded, citations
  stripped, and judged in **one pass with one setting**. Existing v2 judge scores are not mixed into the v4 pass.
- Unavailable or truncated judgements are never scored as zero. If more than 5% of judgements are unresolved, the unresolved
  ones are retried once with identical settings in a separate file; remaining ones are excluded and counted by failure type.
- Endpoint: mean of the five criteria. Comparisons: A1-A0, A1-B0-k5, B0-k5-A0 (Holm over these three).
- Non-inferiority: A1 versus B0-k5 and A1 versus A0, margin Delta = 1.5 on the 1-20 scale. Non-inferior only if the lower bound of
  the 95% interval of the difference is above -Delta. This is pre-specified and conditional on the selected repositories and the
  automatic judge; it is not generalised to other repositories or judges. Not stated: "equivalent".

### 5.3 RQ3 citations
Structural validity (file exists, line range valid, contained in the allowed context, overlaps a real symbol) on all answers. Semantic
precision and claim coverage on the human sample (5.5) for A1 and A2. A1's structural validity is reported as a design property.

### 5.4 RQ4 deferred
RQ4 is deferred because a human-validated unanswerable set was not available before protocol freeze. Insufficient-evidence handling is
discussed as a limitation and a future evaluation direction: the deterministic no-evidence answer of `GroundedQAService` fires only when
the context is empty, so refusals in practice depend on the prompt, and no claim about refusal reliability is made. Unsupported claims are
still observable through the groundedness ratings of 5.5.

### 5.5 RQ5 human validation
Rubric: `HUMAN_REVIEW_RUBRIC.md`; package layout: `HUMAN_REVIEW_PACKAGE_SCHEMA.md` (both versioned; the rubric version is recorded in
provenance before scoring). The overall human score is the automatic mean of the four criteria; citation support is `yes`/`partial`/`no`
per citation (`not_applicable` where no claim exists), `missing_key_source` is a diagnostic field only (`not_applicable` /
`source_identified` / `source_missing`); strict precision counts `yes`, lenient precision counts `yes` and `partial`; truncated or unavailable
items receive no score and are counted separately. The frozen sample (seed, all 30 question ids, orders) is `SAMPLE_FREEZE.json`. The blinding
key is encrypted and held by a non-rater custodian until both score files are submitted (schema section 7); a hash alone is not treated as
confidentiality.
30 questions (8, 8, 7, 7 across the four repositories), three arms (A0, B0-k5, A1), 90 answers per rater, blinded and shuffled, rubric 0-10 for correctness,
completeness, relevance, groundedness, plus citation support for the citation sample. The questions are the first five items per repository
in a fixed-seed stratified order frozen before generation; there is no smaller fallback sample.

Human evaluation was conducted by two raters: the system author and an independent Python-competent reviewer who was not involved in the
design or implementation of CodeQAPass. Both raters independently scored anonymized answers using the frozen rubric. Results from the
independent reviewer constitute the primary human analysis, while the author's scores are reported as a sensitivity analysis and used to
calculate inter-rater agreement. The reviewer's identity is not stored in the repository; provenance records only the role, the scoring
dates and the rubric version. The phrase "two independent raters" is not used. Analysis: weighted kappa, Spearman, paired bootstrap,
win/tie/loss. Human results are descriptive; they do not establish non-inferiority (n is too small for the margin).

### 5.6 RQ6 Persian cross-lingual evaluation (D5 closed: will be run)
32 questions: 8 from each of the four repositories (the same stratified order as 5.5, selected with a fixed seed before generation),
translated into Persian by machine and reviewed by a native speaker; code identifiers stay in English; the translations are frozen and
hashed before any generation. Arms: A0 and A1 only. Metrics: retrieval Recall@5 versus the English questions (A1 hybrid retrieval), judge
mean (same rubric, `max_tokens=2000`), citation structural validity. The judge compares Persian answers with English reference answers,
a cross-lingual judgement. Results are reported separately and never averaged with the English results; they are not used for
superiority or non-inferiority. Persian outputs were evaluated by the LLM judge and subjected to an author-led qualitative audit on a
stratified sample. This audit was not treated as an independent human evaluation and gives no statistic.

### 5.7 Secondary: SWE-QA-Pro subset
Pre-specified trigger: run it if the primary A1-A0 difference is smaller than Delta (memorisation concern). 100 questions, repositories
drawn with a fixed seed before generation, arms A0 and A1, reported as external validation of a subset.

## 6. Statistics
Unit: question. Paired bootstrap (20,000 resamples, fixed seed) stratified by repository; Wilcoxon signed-rank; McNemar and Wilson
for binary outcomes; Holm over each primary family. Decision rule for A1 versus B0-k5 on any primary comparison: if the 95% interval of the difference is entirely above zero, superiority is
reported; a positive difference whose interval includes zero is called a "numerical, non-significant improvement" and nothing stronger.
Per-repository results are always reported. Inference is conditional on the
selected repositories. Excluded pairs and answer-length differences are reported.

## 7. Reproducibility rules
Clone only at execution time and verify `git rev-parse HEAD` against section 2; raw records are append-only; retries are separate files; unavailable is never zero; the reasoning-effort guard stays; provenance
records commit, dataset revision, snapshots, prompts, settings and hashes of every artifact; every deviation goes into section 11.

## 8. Cost estimate (calls)

| Item | Calls |
|---|---:|
| Generation A0, B0-k5, A1 (192 each; A0 reuse possible, see note) | 576 |
| B0-k10 sensitivity | 192 |
| Judge (pooled) | 576 to 768 |
| Indexing: 4 new repositories, two indexes each (AST + function chunks) | depends on repository size |
| RQ6 Persian (48 x 2) generation and judge | 96 + 96 |
| Human annotation | 90 answers per rater |

Note: v2 A0-neutral records for Flask and Requests used identical settings and may be reused; regenerating is cleaner and is the default.

## 9. Threats to validity
Four public repositories (possible memorisation); LLM judge and LLM-written reference answers; silver evidence from reference answers; B0 is a
re-implementation; A1 includes a product component that B0 lacks; research `max_tokens` differs from the product default; two
raters (one is the author); Persian translation may be less natural than organic questions; results are conditional on the chosen repositories.

## 10. Decisions
- D0: closed (revision pinned, section 2).
- D1: amended before execution (four primary repositories: flask, requests, pytest, pylint; Sphinx/Astropy are optional extensions; official commits and indexing budget remain recorded in `D1_METADATA_REPORT.md`).
- D2: closed (author plus one independent reviewer; section 5.5). Remaining administrative step: record role, dates and rubric version in
  provenance before scoring.
- D3: closed (`max_tokens` 1200 for generation, 2000 for the judge).
- D4: closed (RQ4 deferred; section 5.4).
- D5: amended (Persian, 32 questions, A0/A1, LLM judge, author audit; section 5.6).
- D6: closed (SWE-QA-Pro stays secondary with the trigger in 5.7).
- D7: closed. The blinding-key custodian is a person who is not a rater and took no part in the project; identity not stored in the repository. The custodian must differ from both raters.
- Rubric and package schema written (draft 0.2, for review). Not yet done before the freeze commit: review of both documents; the package
  builder and blinded files (generated only after the final answers exist); pilot execution (Stage 0, needs a separate approval); freeze
  commit and its SHA in `manifest.json`.

## 11. Deviation log
- **2026-10-11 — Pylint structural fixtures.** The pinned Pylint snapshot contains 23
  intentionally invalid-Python fixture files. The A1 indexing CLI was extended with
  an evaluation-only, explicit allowlist at
  `pylint_allowed_structural_errors.txt`; only parse/chunk errors for those paths are
  non-fatal, while any unexpected structural error still aborts indexing. The same
  fixture set is excluded from the B0 structural candidate. The resulting Pylint A1
  index contains 9,428 valid chunks from 2,310 discovered files, with SQLite/Chroma
  ID sets equal and the pinned commit unchanged.
- **2026-10-11 — Chroma staging batch size.** Chroma upserts use batches of 500
  rather than 5,000 to bound memory for 3,072-dimensional vectors; this changes
  only persistence batching, not retrieval or candidate contents.
