# SWE-QA v4 human-review analysis

This report is generated from the completed author and independent-rater CSVs. The independent rater is the primary human analysis; the author's ratings are used for sensitivity and agreement. Bootstrap intervals use 20,000 resamples with seed `20260930`.

## Task 1

Mean overall score is the arithmetic mean of correctness, completeness, relevance, and groundedness.

| arm | n | mean | bootstrap 95% CI |
|---|---:|---:|---:|
| A0 | 30 | 4.158 | [3.575, 4.717] |
| A1 | 30 | 6.683 | [5.867, 7.492] |
| B0-k5 | 30 | 7.333 | [6.458, 8.150] |

Paired differences on the same questions:

| comparison | mean difference | bootstrap 95% CI |
|---|---:|---:|
| A1 − A0 | +2.525 | [+1.542, +3.508] |
| A1 − B0-k5 | −0.650 | [−1.650, +0.308] |
| B0-k5 − A0 | +3.175 | [+2.083, +4.258] |

The independent rater therefore rates A1 above A0, while B0-k5 has the highest
mean. The A1 versus B0-k5 interval includes zero and is reported as a numerical,
non-significant deficit under the frozen decision rule.

Rater agreement (quadratic weighted kappa / Spearman) is:

| criterion | kappa | Spearman |
|---|---:|---:|
| correctness | 0.637 | 0.656 |
| completeness | 0.823 | 0.855 |
| relevance | 0.882 | 0.861 |
| groundedness | 0.189 | 0.420 |

Overall-mean Spearman agreement is `0.746`. Groundedness has substantially
lower agreement and should be discussed as a judgment-sensitive criterion.

## Task 2

Because the two blinded citation lists are sometimes identical, X/Y cannot be
uniquely mapped back to A1/A2 for 19 of the 30 questions after unblinding. The
report therefore treats X and Y as the primary blinded estimands and reports
arm-specific values only for the 11 questions whose citation sets identify the
arm uniquely.

| label | citations | strict precision | lenient precision | claim coverage |
|---|---:|---:|---:|---:|
| X | 67 | 0.806 | 0.881 | 0.913 |
| Y | 66 | 0.788 | 0.894 | 0.913 |

On the 11 uniquely mapped questions, A1 has strict/lenient precision
`0.553/0.658` over 38 citations and A2 has `0.905/1.000` over 21 citations.
These arm-specific figures are exploratory because the mapping is unresolved
for the remaining 19 questions.

Exact agreement between the two raters is 0.692 for citation support and 0.594
for missing-key-source status.

Spearman agreement between the independent human overall mean and the
automatic judge mean is A0 `0.781`, A1 `0.963`, and B0-k5 `0.863` (30 paired
answers per arm). This is agreement, not equivalence.

Machine-readable provenance and all hashes are in `human_review_analysis.json`.
