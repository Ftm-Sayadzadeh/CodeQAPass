"""Validate a returned SWE-QA v7 human-review score file against the package it was built from (protocol section 7). Never touches the blinding key.

    python scripts/evaluation/swe_qa_v7_human_validate.py --rater independent --task1 PATH --task2 PATH

Checks the columns, that every row is the package row (same ids and unchanged question, reference and answer text), integer scores 0-10, `refusal` yes/no,
`support` values, completeness, and that rows with a non-ok `status_flag` carry no scores. Prints counts only (no arm information, which this script cannot see).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "reports/evaluation/swe_qa_v7/human_review_v7"
SCORES = ("correctness", "completeness", "relevance", "groundedness")
SUPPORT = {"yes", "partial", "no", "not_applicable"}
MISSING = {"not_applicable", "source_identified", "source_missing"}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_task1(returned: list[dict], package: list[dict]) -> dict:
    problems: list[str] = []
    if [c for c in package[0]] != list(returned[0].keys()):
        problems.append(f"columns differ: {list(returned[0].keys())}")
    base = {r["item_id"]: r for r in package}
    seen = set()
    scored = empty = bad_scores = bad_refusal = scores_on_flagged = 0
    for r in returned:
        item = r["item_id"]
        if item in seen:
            problems.append(f"duplicate item {item}")
        seen.add(item)
        if item not in base:
            problems.append(f"unknown item {item}")
            continue
        for field in ("question_id", "question", "reference_answer", "answer", "status_flag"):
            if (r[field] or "").strip() != (base[item][field] or "").strip():
                problems.append(f"{item}: text of `{field}` was changed")
        values = [(r[c] or "").strip() for c in SCORES]
        if base[item]["status_flag"] != "ok":
            scores_on_flagged += any(values)
            continue
        if all(values):
            try:
                numbers = [int(v) for v in values]
                if not all(0 <= n <= 10 for n in numbers):
                    raise ValueError
                scored += 1
            except ValueError:
                bad_scores += 1
                problems.append(f"{item}: scores must be integers 0-10, got {values}")
        elif any(values):
            bad_scores += 1
            problems.append(f"{item}: only some of the four scores are filled")
        else:
            empty += 1
        if (r["refusal"] or "").strip().lower() not in {"yes", "no"} and all(values):
            bad_refusal += 1
    missing_items = sorted(set(base) - seen)
    return {"rows": len(returned), "expected": len(package), "scored_complete": scored, "no_scores": empty, "bad_scores": bad_scores, "bad_refusal_values": bad_refusal,
            "scores_on_flagged_rows": scores_on_flagged, "missing_items": len(missing_items), "problems": problems[:20], "problem_count": len(problems)}


def validate_task2(returned: list[dict], package: list[dict]) -> dict:
    problems: list[str] = []
    if [c for c in package[0]] != list(returned[0].keys()):
        problems.append(f"columns differ: {list(returned[0].keys())}")
    base = {r["citation_id"]: r for r in package}
    seen, labelled, bad = set(), 0, 0
    first_rows: dict[tuple[str, str], str] = {}
    for r in package:
        first_rows.setdefault((r["question_id"], r["system_label"]), r["citation_id"])
    missing_flag_filled = 0
    for r in returned:
        cid = r["citation_id"]
        if cid in seen:
            problems.append(f"duplicate citation {cid}")
        seen.add(cid)
        if cid not in base:
            problems.append(f"unknown citation {cid}")
            continue
        for field in ("item_id", "question_id", "question", "answer_being_attributed", "system_label", "cited_range"):
            if (r[field] or "").strip() != (base[cid][field] or "").strip():
                problems.append(f"{cid}: `{field}` was changed")
        support = (r["support"] or "").strip().lower()
        if support:
            labelled += 1
            if support not in SUPPORT:
                bad += 1
                problems.append(f"{cid}: invalid support value {support!r}")
        flag = (r["missing_key_source"] or "").strip().lower()
        if flag:
            missing_flag_filled += 1
            if flag not in MISSING:
                problems.append(f"{cid}: invalid missing_key_source {flag!r}")
    groups = {(r["question_id"], r["system_label"]) for r in package}
    flagged_groups = {(r["question_id"], r["system_label"]) for r in returned if (r["missing_key_source"] or "").strip()}
    return {"rows": len(returned), "expected": len(package), "support_filled": labelled, "support_empty": len(returned) - labelled, "invalid_support": bad,
            "question_label_groups": len(groups), "groups_with_missing_key_source": len(flagged_groups), "missing_items": len(set(base) - seen), "problems": problems[:20], "problem_count": len(problems)}


def content_check(task1: list[dict], task2: list[dict]) -> dict:
    """For files edited in place (no blank original left): the question, reference and answer texts must still be the benchmark and run texts.
    Per question the four returned answers must be exactly the four arm answers (compared as a multiset, so no arm is identified)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import swe_qa_v7_human_package as package  # noqa: E402
    import swe_qa_v7_runner as runner  # noqa: E402

    rows = runner.question_rows()
    answers = package.read_answers()
    problems: list[str] = []
    expected = {q: sorted(runner.strip_citations(answers[arm][q].get("answer")) for arm in package.ARMS) for q in {r["question_id"] for r in task1}}
    by_question: dict[str, list[str]] = {}
    for r in task1:
        q = r["question_id"]
        if (r["question"] or "").strip() != (rows[q]["question"] or "").strip() or (r["reference_answer"] or "").strip() != (rows[q].get("answer") or "").strip():
            problems.append(f"{r['item_id']}: question or reference text differs from the benchmark")
        by_question.setdefault(q, []).append(r["answer"])
    for q, got in by_question.items():
        if sorted(a.strip() for a in got) != [a.strip() for a in expected[q]]:
            problems.append(f"{q}: the four answers differ from the run records")
    cited = {q: {runner.strip_citations(answers[arm][q].get("answer")).strip() for arm in package.CITED_ARMS} for q in by_question}
    for r in task2:
        if r["answer_being_attributed"].strip() not in cited[r["question_id"]]:
            problems.append(f"{r['citation_id']}: attributed answer differs from the run records")
    return {"questions": len(by_question), "problem_count": len(problems), "problems": problems[:10]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content-check", action="store_true", help="for files edited in place: compare texts with the benchmark and the run records")
    parser.add_argument("--rater", choices=("author", "independent"), required=True)
    parser.add_argument("--task1", type=Path, required=True)
    parser.add_argument("--task2", type=Path, required=True)
    args = parser.parse_args()
    folder = PACKAGE / args.rater
    returned1, returned2 = read(args.task1), read(args.task2)
    result = {"rater_role": args.rater, "task1": validate_task1(returned1, read(folder / "task1_items.csv")), "task2": validate_task2(returned2, read(folder / "task2_items.csv"))}
    if args.content_check:
        result["content_check"] = content_check(returned1, returned2)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0 if not result["task1"]["problem_count"] and not result["task2"]["problem_count"] and not result.get("content_check", {}).get("problem_count") else 1


if __name__ == "__main__":
    raise SystemExit(main())
