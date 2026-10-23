# SWE-QA v7 review (Task 1 and Task 2)

You rate answers of code question-answering systems about four Python repositories. 

## What you see
Task 1 (`task1_items.csv`): 120 rows = 30 questions x 4 anonymous answers, shuffled. Each row has the question, the reference answer (from the benchmark) and one answer with all citation strings removed.
You do not see which system wrote an answer. You may open the repository at the snapshot commit to verify claims (commits below).

| repository | snapshot commit |
|---|---|
| flask | `85c5d93cbd049c4bd0679c36fd1ddcae8c37b642` |
| requests | `46e939b5525d9c72b677340985582b04b128478a` |
| pytest | `5989efe3efec31a9d14f550d89903b6311dfc828` |
| pylint | `44740e5e5cdd2b80321a4f0ed4254213d058b9f6` |

## Task 1: answer quality (integers 0-10 for each of four criteria)
| Criterion | 0 | 5 | 10 |
|---|---|---|---|
| **Correctness** (are the stated facts right?) | The main claims contradict the reference answer or the code, or are invented. | About half of the key facts are right and some are wrong. | All key facts agree with the reference answer and the code; no errors. |
| **Completeness** (are the points of the reference covered?) | None of the main points of the reference answer appear. | About half of the main points appear; important ones are missing. | All main points of the reference answer appear. |
| **Relevance** (does it answer this question?) | Off topic or answers a different question. | Partly addresses the question, with substantial generic or unrelated content. | Directly answers the question; no filler. |
| **Groundedness** (is every statement verifiable in the repository?) | Names files, symbols or behaviour that do not exist or cannot be verified. | Some statements are verifiable and some are not. | Every stated file, symbol and behaviour can be verified in the repository. |

Rules:
- Judge against the reference answer and the repository, not against how confident or fluent the text sounds. Do not reward length.
- Groundedness is judged against the repository, whether or not the system had a context.
- Do not give an overall score; the mean of the four criteria is computed afterwards.
- If an answer says there is not enough evidence, score it on the four criteria like any other answer and write `yes` in the `refusal` column (otherwise `no`).
- Rows with `status_flag` other than `ok` get no scores (leave the four score cells empty); they are counted separately and never treated as zero.

## Task 2: citation support (`task2_items.csv`, code in `task2_cited_code.csv`)
For each question two answers (labels X and Y) are shown with the list of sources the system cited for its own answer; one row per citation. For each citation choose one `support` value:
- `yes`: the cited range supports a claim in the answer;
- `partial`: related, but supports only part of a claim or a minor point;
- `no`: unrelated, or does not support any claim;
- `not_applicable`: the answer makes no claim a source could support (for example a refusal); these rows leave the precision denominators.
Use `missing_key_source` once per question and label (put it on the first row of that question and label): `not_applicable`, `source_identified` (a source the answer needed is cited) or `source_missing` (a needed source is not cited). It is diagnostic only.

## Submitting
Edit only your own folder's files, keep the column names and the item order, save as CSV (UTF-8), and do not look for any key. Record the dates you scored (role only, no name).
