# Human review rubric (protocol v4, DRAFT, version 0.2)

Part of `PROTOCOL_V4_DRAFT.md` section 5.5. It becomes binding at the protocol freeze; the rubric SHA-256 and version go into provenance before any
scoring starts. Package layout: `HUMAN_REVIEW_PACKAGE_SCHEMA.md`. Frozen sample: `SAMPLE_FREEZE.json`.

## 1. What is rated

Exactly 30 questions (8, 8, 7, 7 across the four primary repositories), three answers per question (A0, B0-k5, A1), so exactly 90 answers per rater. There is no
smaller fallback sample. Two raters score independently: Rater 1 is the system author, Rater 2 is an independent Python-competent reviewer who took
no part in the design or implementation of CodeQAPass. The independent rater's scores are the primary human analysis; the author's scores are a
sensitivity analysis and are used for agreement. Task 1 rates answer quality; Task 2 rates citation support (A1 versus A2) on the same 30 questions.

## 2. Question selection (frozen in `SAMPLE_FREEZE.json`, written before any generation)

Per repository the order of the 48 questions is `random.Random(f"20260930:{repo}").sample(range(1, 49), 48)`; the question id is `{repo}-{n}`
(n = 1-based line number in the repository's JSONL). The full orders are stored in `SAMPLE_FREEZE.json`, so nothing depends on re-running the
generator. The frozen human allocation is recorded in `SAMPLE_FREEZE.json`; the first 8 of each order are the Persian sample. No question is removed or replaced after outputs exist,
except through the deviation log.

Human sample, in order (30 ids):

| Repository | Question ids |
|---|---|
| flask | flask-13, flask-41, flask-47, flask-46, flask-15, flask-14, flask-32, flask-30 |
| requests | requests-19, requests-7, requests-43, requests-10, requests-17, requests-13, requests-25, requests-18 |
| pytest | pytest-2, pytest-32, pytest-15, pytest-18, pytest-33, pytest-7, pytest-27 |
| pylint | pylint-22, pylint-40, pylint-14, pylint-19, pylint-27, pylint-15, pylint-23 |

Sphinx and Astropy remain optional extensions and are not included in this human sample.

## 3. What the rater sees

For each item: the question, the reference answer (from the benchmark), and one anonymised answer with all citation strings removed. The rater does
not see the system name, the arm, the item-to-arm key, the automatic judge scores, or any other rater's scores. The rater may open the repository
snapshot at the commit named in the protocol to verify claims. Items are shuffled; the three answers of a question are not adjacent by default.

## 4. Task 1: answer quality (integers 0-10)

Rate each criterion independently. Anchors for 0, 5 and 10 (use the values in between freely):

| Criterion | 0 | 5 | 10 |
|---|---|---|---|
| **Correctness** (are the stated facts right?) | The main claims contradict the reference answer or the code, or are invented. | About half of the key facts are right and some are wrong. | All key facts agree with the reference answer and the code; no errors. |
| **Completeness** (are the points of the reference covered?) | None of the main points of the reference answer appear. | About half of the main points appear; important ones are missing. | All main points of the reference answer appear. |
| **Relevance** (does it answer this question?) | Off topic or answers a different question. | Partly addresses the question, with substantial generic or unrelated content. | Directly answers the question; no filler. |
| **Groundedness** (is every statement verifiable in the repository?) | Names files, symbols or behaviour that do not exist or cannot be verified. | Some statements are verifiable and some are not. | Every stated file, symbol and behaviour can be verified in the repository. |

Rules:
- Judge against the reference answer and the repository, not against how confident or fluent the text sounds. Do not reward length.
- Groundedness is judged against the repository, whether or not the system had a context.
- The overall score is the **arithmetic mean of the four criteria, computed automatically**. Raters do not give an overall score.
- An answer that says there is not enough evidence is scored on the four criteria like any other answer, and the box `refusal` is ticked.
- An item marked `truncated` or `unavailable` in the package (the builder sets this from `finish_reason`/status) receives **no scores**; it is recorded
  separately and counted in the report. It is never scored as zero.

## 5. Task 2: citation support (A1 versus A2)

For each of the 30 questions two blinded citation lists (labels X, Y) are attached to the same fixed answer; the package includes the code at each
cited range. For each citation choose one `support` label:

- `yes`: the cited range supports a claim in the answer;
- `partial`: related to the answer but supports only part of a claim or a minor point;
- `no`: unrelated, or does not support any claim;
- `not_applicable`: the answer makes no claim that a source could support (for example a refusal); such rows are excluded from the precision
  denominators.

Definitions fixed in advance:
- strict precision = yes / (yes + partial + no);
- lenient precision = (yes + partial) / (yes + partial + no).

Per question and label also record `missing_key_source`, a **diagnostic field only**; it never changes the correctness or groundedness scores and
is not confused with an invalid or absent citation:
- `not_applicable`: the answer makes no claim needing a specific source;
- `source_identified`: the cited ranges contain the source(s) needed for the main claims of the answer;
- `source_missing`: a source needed for a main claim is not among the cited ranges.
Claim coverage = source_identified / (source_identified + source_missing).

## 6. Procedure

1. Short training (about 15 minutes) with three calibration items from the pilot repository (`conan`, ids in `SAMPLE_FREEZE.json`); calibration items are
   not analysed.
2. Score independently; no discussion between raters before both have finished; no use of LLM judge outputs or other tools to generate scores.
3. The blinding key is held outside the reviewer package and outside both raters' reach (schema section 7) until both score files are submitted.
4. No adjudication is used for the primary analysis. Disagreements are reported through agreement statistics.
5. Blinding limits are stated: the author may recognise the style of the system's own outputs; this is why the primary human analysis uses the
   independent rater.

## 7. Analysis hooks (details in the protocol)

- Primary human analysis: independent rater. Sensitivity: author. Agreement per criterion and for the overall mean: quadratic weighted kappa and Spearman
  correlation.
- Differences between arms: paired bootstrap on per-question means, win/tie/loss.
- Agreement of the human mean with the automatic judge uses Spearman correlation on per-answer means (scale-free). Human scores (0-10) and judge scores
  (1-20) are not rescaled into each other; effects are reported in each scale's own units.
- Decision rule: if the 95% interval of the difference A1 minus B0-k5 is entirely above zero, superiority is reported. A positive difference whose
  interval includes zero is reported only as a "numerical, non-significant improvement".
- Known blinding leaks (answer length, wording such as "the context", formatting) are checked and reported.

## 8. Rater data entry

One row per item; columns in the schema. Scores are integers 0-10; empty cells are not allowed except for items flagged `truncated`/`unavailable`.
