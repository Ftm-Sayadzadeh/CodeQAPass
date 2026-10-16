# D1 metadata report: repository selection for protocol v4

Checked: 2026-10-06 (UTC). Read-only, metadata only. No question or answer text, no repository code and no experiment output was
downloaded, and no model or embedding API was called. `PROTOCOL_V4_DRAFT.md` and all frozen artifacts are unchanged.

## 1. Sources and dataset revision (closes D0)

| Item | Value | Source |
|---|---|---|
| Dataset | `swe-qa/SWE-QA-Benchmark` | Hugging Face API `/api/datasets/swe-qa/SWE-QA-Benchmark` |
| Revision (sha) | `f048f07da234b39e32937f115d027c5b71acd54e` | same |
| Last modified | 2026-08-10T06:46:41Z | same |
| License | apache-2.0 | same (card data) |
| Structure | one JSONL per project + `default.jsonl` (all 720); features `question`, `answer` only | datasets-server `/info` |
| Questions | 15 projects x 48 = 720 | datasets-server `/info` |

The dataset does not pin a repository commit. Snapshot commits are our decision (section 4).

Consistency with the pilot: the local `flask.jsonl` and `requests.jsonl` used in v2 have the same size and the same Git blob SHA as the
files at revision `f048f07...` (flask `719b8b02f162...`, requests `ec0ac426b800...`). The pilot and v4 therefore use the same
questions for these two repositories.

## 2. Candidate metadata (GitHub API, `repos/{owner}/{name}`)

`size` is the GitHub repository size in KB (it includes history and non-Python files), so it is only a coarse size proxy. Python
file counts and chunk counts at the snapshot were **not** measured (that needs a clone). The domain label is assigned by us from
GitHub topics and the project description, before any v4 output exists.

| Project | Domain label | Size KB | Stars | Created | License | Default branch |
|---|---|---:|---:|---|---|---|
| flask (fixed) | web framework | 12,311 | 74,909 | 2010-04-06 | BSD-3-Clause | main |
| requests (fixed) | HTTP client | 13,631 | 54,478 | 2011-02-13 | Apache-2.0 | main |
| conan | package management | 36,010 | 9,528 | 2015-12-01 | MIT | develop2 |
| pylint | static analysis | 41,952 | 5,730 | 2015-12-09 | GPL-2.0 | main |
| pytest | testing | 41,962 | 14,572 | 2015-06-15 | MIT | main |
| streamlink | streaming CLI | 42,338 | 11,828 | 2016-09-16 | BSD-2-Clause | master |
| sqlfluff | static analysis | 54,239 | 9,937 | 2018-11-01 | MIT | main |
| xarray | scientific computing | 55,111 | 4,207 | 2013-09-30 | Apache-2.0 | main |
| sphinx | documentation | 109,445 | 8,057 | 2015-01-02 | NOASSERTION | master |
| reflex | web framework | 137,861 | 28,939 | 2022-10-25 | Apache-2.0 | main |
| astropy | scientific computing | 187,774 | 5,327 | 2011-07-21 | BSD-3-Clause | main |
| scikit-learn | scientific computing | 188,290 | 67,479 | 2010-08-17 | BSD-3-Clause | main |
| sympy | scientific computing | 195,484 | 14,992 | 2010-04-30 | NOASSERTION | master |
| django | web framework | 283,187 | 91,338 | 2012-04-28 | BSD-3-Clause | main |
| matplotlib | scientific computing | 491,437 | 23,328 | 2011-02-19 | not retrieved | main |

## 3. Proposed selection rule (to be frozen in the protocol)

1. Flask and Requests are fixed (they are the pilot repositories and already have questions and indexes).
2. Candidates are the other 13 projects. Web and HTTP projects (django, reflex) are excluded because Flask and Requests already
   cover that domain.
3. Pick one project from each of four domains: testing, static analysis, documentation, scientific computing.
4. Size balance: two medium (< 60,000 KB) and two large (>= 100,000 KB) projects among the four.
5. Within a domain and tier, choose the candidate with the fewest GitHub stars. Stars are used **only as a deterministic tie-breaker**; they
   are not a validated measure of model memorisation and no memorisation claim is made from this choice.

Projects are selected before any model execution to provide domain and size diversity.

Applying the rule: testing has one candidate (pytest, medium); documentation has one candidate (sphinx, large); static analysis has only
medium candidates (pylint 5,730 stars < sqlfluff 9,937); scientific computing must supply the second large project (astropy 5,327 stars <
sympy 14,992 < matplotlib 23,328 < scikit-learn 67,479).

**Proposed four repositories: pytest, pylint, sphinx, astropy.** Together with Flask and Requests this gives 288 questions spanning
12-188 MB repositories and the domains web, HTTP, testing, static analysis, documentation and scientific computing.

Pre-specified replacement: if indexing astropy exceeds the resource budget fixed at the freeze, sympy (3c817ed) is the replacement.
The substitution is decided by the budget check before any model output exists, never after seeing results. For sphinx there is no
same-domain substitute; if it exceeds the budget, the tier requirement is relaxed and the change is logged in the deviation log.

## 4. Repository commits: the benchmark provides them (closes the open check)

The official repository `peng-weihan/SWE-QA-Bench` contains `repo_commit.txt` (URL and commit per project) and `clone_repos.sh`, which reads
that file and runs `git checkout <commit>` into `datas/repos/`. The README describes the questions as tied to pinned commits. The last
commit touching `repo_commit.txt` is `744a673b63bab77996096d278572da675defc0f9` (2026-04-07, "for camera-ready"). The dataset card on
Hugging Face states no commits, so `repo_commit.txt` is the authoritative source and replaces the date rule proposed earlier.

| Project | Official commit (short SHA) |
|---|---|
| flask (fixed) | 85c5d93 |
| requests (fixed) | 46e939b |
| pytest | 5989efe |
| pylint | 44740e5 |
| sphinx | 6c9e320 |
| astropy | 0a041d3 |
| sympy (prespecified replacement) | 3c817ed |

Integrity check: the Flask and Requests short SHAs match the full SHAs recorded in the v2 provenance (`85c5d93cbd04...`, `46e939b5525d...`), so the
pilot already used the official commits. The file was read through a fetch tool, not with a direct download (direct access to the raw file was
not available), so the four new SHAs are re-verified when the repositories are cloned: resolve each short SHA to a full SHA with `git rev-parse`,
check that it exists, and record the full SHA in `manifest.json`. A missing or ambiguous commit is a protocol deviation, not a silent fallback.
The repositories are cloned into a local directory only for indexing; no snapshot is redistributed.

## 5. Caveats for the protocol

- GitHub `size` is a crude proxy. Measure Python file and chunk counts after cloning, before the budget check; this does not change the rule.
- Licenses: pylint is GPL-2.0; sphinx and sympy report NOASSERTION. The repositories are used locally for indexing; do not redistribute
  snapshots (for example in a reviewer package).
- The dataset is Apache-2.0; the questions of Flask and Requests are identical to the pilot, so those two repositories are not new evidence
  about generalisation beyond the pilot repositories.
- Stars are only a deterministic tie-breaker; they do not measure contamination.
- The benchmark pins commits, but the reference answers were generated by LLMs and human-validated; any residual mismatch with code at that commit is part of the benchmark's own noise.

## 6. Status of open decisions

- D0: closed (revision `f048f07da234b39e32937f115d027c5b71acd54e`).
- D1: closed. Repositories pytest, pylint, sphinx, astropy at the official commits above; prespecified replacement sympy; commits re-verified
  at clone time; stars used only as a tie-breaker.
- D2, D4, D5: open. D3 and D6 are confirmed by the project owner as stated in the conversation.
