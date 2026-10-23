# Human review package schema (protocol v4, DRAFT, version 0.2)

Describes the files that will be generated from the final answers; nothing is generated now. Used with `HUMAN_REVIEW_RUBRIC.md` and
`SAMPLE_FREEZE.json`. No answer, key or reviewer identity is stored in this document.

## 1. Files per rater

| File | Content |
|---|---|
| `task1_items.csv` | 90 rows: one anonymised answer each (Task 1) |
| `task2_items.csv` | 60 rows: one blinded citation list each (30 questions x labels X/Y), one row per citation |
| `task2_cited_code.csv` | code text of each cited range, joined to `task2_items.csv` by `citation_id` |
| `INSTRUCTIONS.md` | the rubric as shown to raters (no key, no arm names) |

## 2. Task 1 columns (`task1_items.csv`)

`item_id` (random, not ordered by arm), `question_id`, `question`, `reference_answer`, `answer` (citation strings removed),
`status_flag` (`ok`, `truncated` or `unavailable`, set by the builder), then the rater's columns: `correctness`, `completeness`, `relevance`,
`groundedness` (integers 0-10), `refusal` (`yes`/`no`), `comment` (optional). The overall mean is computed afterwards, never entered.

## 3. Task 2 columns (`task2_items.csv`)

`item_id`, `question_id`, `question`, `answer_being_attributed` (the fixed A1 answer, citations removed), `system_label` (`X` or `Y`), `citation_id`,
`cited_range` (`path:start-end`), then the rater's columns: `support` (`yes` / `partial` / `no` / `not_applicable`) per citation row, and
`missing_key_source` (`not_applicable` / `source_identified` / `source_missing`) once per question and label (diagnostic only; it never alters Task 1
scores). Strict precision counts `yes`; lenient precision counts `yes` and `partial`; `not_applicable` rows leave the denominators.

## 4. Blinding rules (builder checks, must pass before sending)

- No arm name or system name (`A0`, `A1`, `A2`, `B0`, `CodeQAPass`, `baseline`) in any rater-visible file or column header.
- Item order and the X/Y order come from OS entropy (`secrets`), not from a seed, and the order is stored only in the key (section 7).
- Citation strings matching `path:line` are stripped from Task 1 answers.
- Answer lengths and the phrase "the context" are measured per arm and reported as known leaks; they are not edited out.
- Exactly 30 questions x 3 arms in Task 1; exactly 30 questions x 2 labels in Task 2; counts verified against `SAMPLE_FREEZE.json`.

## 5. Status handling

A Task 1 item with `finish_reason != stop` or an empty answer gets `status_flag = truncated` or `unavailable`, is shown for completeness of the package
but has empty score cells that the validator accepts. Such items are counted per arm and reported; they are never scored as zero and never dropped
silently.

## 6. Provenance record (written before scoring and updated after)

Recorded in provenance, with `SAMPLE_FREEZE.json` as the carrier of the sample itself: the exact seed; the 30 question ids with their repositories and the
final order (all in `SAMPLE_FREEZE.json`, whose SHA-256 is recorded); rubric SHA-256 and version; package SHA-256; SHA-256 of the key ciphertext;
scoring dates per rater (role only: `author`, `independent reviewer`). **The independent reviewer's name is not stored in the repository.**

## 7. Blinding key custody

A hash proves the key was not changed; it does not keep it secret. The author is also a rater, so the key must be out of both raters' reach:

1. The builder generates the item-to-arm mapping and the X/Y mapping, encrypts it with a one-time passphrase (for example `age` or
   `openssl enc -aes-256-cbc -pbkdf2`), and never writes the plaintext key into the workspace or the reviewer package.
2. The passphrase is given only to a **key custodian who is not a rater** (identity not stored in the repository) and the author keeps no copy.
3. SHA-256 of the ciphertext goes into provenance before scoring starts; the ciphertext is stored outside the reviewer package.
4. After **both** score files are submitted and have passed validation, the custodian releases the passphrase and the key is opened for the analysis.
5. If no custodian is available, blinding is reported as incomplete for the author's scores and the primary analysis stays with the independent
   rater (the author's recognition of own outputs is a stated limitation either way).

## 8. Validation after submission

Rows complete; integers in range 0-10; `status_flag` items empty only where flagged; item ids match the package; both raters cover the same 90 + 60 items.
A submission that fails validation is returned to the rater, not corrected by the author.

## 9. Output of the analysis step

Per-answer mean of the four criteria, per-question means per arm, agreement statistics, win/tie/loss, paired bootstrap intervals, citation precision
(strict and lenient), claim coverage, counts of flagged items, and the leak checks.
