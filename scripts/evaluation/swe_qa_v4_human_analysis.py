"""Validate and analyse the two v4 human-review CSVs."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "reports/evaluation/swe_qa_v4/human_review_v4"
FROZEN = ROOT / "reports/evaluation/swe_qa_v4/frozen"
OUT = FROZEN / "human_review_analysis.json"
SCORE_FIELDS = ("correctness", "completeness", "relevance", "groundedness")
ARMS = ("a0", "a1", "b0_k5")
PATH_LINE = re.compile(r"(?<!\w)(?:[\w./-]+\.py:\d+(?:-\d+)?)(?!\w)")
SEED = 20260930


def read_csv(name: str) -> list[dict[str, str]]:
    with (PACKAGE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stripped(text: str) -> str:
    return PATH_LINE.sub("[source citation]", text or "")


def validate(t1: list[dict[str, str]], t2: list[dict[str, str]]) -> None:
    if len(t1) != 90 or len(t2) != 133:
        raise SystemExit(f"unexpected row counts: task1={len(t1)} task2={len(t2)}")
    if len({r["item_id"] for r in t1}) != 90 or len({r["citation_id"] for r in t2}) != 133:
        raise SystemExit("duplicate item or citation IDs")
    for row in t1:
        for field in SCORE_FIELDS:
            try:
                value = int(row[field])
            except ValueError as error:
                raise SystemExit(f"invalid {field} in {row['item_id']}") from error
            if not 0 <= value <= 10:
                raise SystemExit(f"out-of-range {field} in {row['item_id']}")
        if row["refusal"] not in {"yes", "no"}:
            raise SystemExit(f"invalid refusal in {row['item_id']}")
    if any(row["support"] not in {"yes", "partial", "no", "not_applicable"} for row in t2):
        raise SystemExit("invalid Task 2 support value")
    if any(row["missing_key_source"] not in {"not_applicable", "source_identified", "source_missing"} for row in t2):
        raise SystemExit("invalid Task 2 missing_key_source value")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            result[order[k]] = rank
        i = j + 1
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    den = (sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry)) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den if den else None


def weighted_kappa(x: list[int], y: list[int], maximum: int) -> float | None:
    if not x:
        return None
    observed = sum((1 - ((a - b) / maximum) ** 2) for a, b in zip(x, y)) / len(x)
    px, py = Counter(x), Counter(y)
    expected = sum((1 - ((a - b) / maximum) ** 2) * px[a] * py[b] for a in px for b in py) / (len(x) ** 2)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def bootstrap(values: list[float], groups: dict[str, list[float]] | None = None) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(SEED)
    if groups:
        samples = [statistics.mean([rng.choice(group) for group in groups.values()]) for _ in range(20000)]
    else:
        samples = [statistics.mean([rng.choice(values) for _ in values]) for _ in range(20000)]
    samples.sort()
    return [samples[500], samples[19499]]


def main() -> int:
    t1 = read_csv("task1_items_independent.csv")
    t2 = read_csv("task2_items_independent.csv")
    validate(t1, t2)
    author_t1 = {r["item_id"]: r for r in read_csv("task1_items.csv")}
    author_t2 = {r["citation_id"]: r for r in read_csv("task2_items.csv")}
    agreement = {}
    for field in SCORE_FIELDS:
        a, b = [int(author_t1[r["item_id"]][field]) for r in t1], [int(r[field]) for r in t1]
        agreement[field] = {"quadratic_weighted_kappa": weighted_kappa(a, b, 10), "spearman": spearman(a, b)}
    agreement["overall_mean"] = {"spearman": spearman([statistics.mean(int(author_t1[r["item_id"]][f]) for f in SCORE_FIELDS) for r in t1], [statistics.mean(int(r[f]) for f in SCORE_FIELDS) for r in t1])}
    raw_by_arm = {arm: {r["question_id"]: r for r in read_jsonl(FROZEN / f"main_{arm}_combined.jsonl")} for arm in ARMS}
    arm_for_item = {}
    for row in t1:
        candidates = [arm for arm in ARMS if stripped(raw_by_arm[arm][row["question_id"]].get("answer", "")) == row["answer"]]
        if len(candidates) != 1:
            raise SystemExit(f"cannot uniquely map item to arm: {row['item_id']} {candidates}")
        arm_for_item[row["item_id"]] = candidates[0]
    by_arm = {}
    per_question = defaultdict(dict)
    for row in t1:
        arm = arm_for_item[row["item_id"]]; score = statistics.mean(int(row[f]) for f in SCORE_FIELDS)
        per_question[row["question_id"]][arm] = score
    for arm in ARMS:
        values = [q[arm] for q in per_question.values()]
        by_arm[arm] = {"n": len(values), "mean": statistics.mean(values), "bootstrap_ci95": bootstrap(values)}
    comparisons = {}
    for left, right in (("a1", "a0"), ("a1", "b0_k5"), ("b0_k5", "a0")):
        diffs = [q[left] - q[right] for q in per_question.values()]
        comparisons[f"{left}_minus_{right}"] = {"n": len(diffs), "mean": statistics.mean(diffs), "bootstrap_ci95": bootstrap(diffs)}
    a1_raw = {r["question_id"]: r for r in read_jsonl(FROZEN / "main_a1_combined.jsonl")}
    a2_raw = {r["question_id"]: r for r in read_jsonl(FROZEN / "a2_model_citations_retry.jsonl")}
    label_map: dict[tuple[str, str], str] = {}
    ambiguous_questions: list[str] = []
    for qid in sorted({r["question_id"] for r in t2}):
        expected = {
            "a1": {f"{c['source_file']}:{c['start_line']}-{c['end_line']}" for c in a1_raw[qid].get("citations", [])},
            "a2": {f"{c['source_file']}:{c['start_line']}-{c['end_line']}" for c in a2_raw[qid].get("citations", [])},
        }
        observed = {label: {r["cited_range"] for r in t2 if r["question_id"] == qid and r["system_label"] == label} for label in ("X", "Y")}
        for label in ("X", "Y"):
            matches = [arm for arm, ranges in expected.items() if ranges == observed[label]]
            if len(matches) == 1:
                label_map[(qid, label)] = matches[0]
        if len([key for key in label_map if key[0] == qid]) != 2:
            ambiguous_questions.append(qid)
    task2 = {}
    for label in ("X", "Y"):
        rows = [r for r in t2 if r["system_label"] == label]
        counted = [r for r in rows if r["support"] != "not_applicable"]
        strict = sum(r["support"] == "yes" for r in counted) / len(counted)
        lenient = sum(r["support"] in {"yes", "partial"} for r in counted) / len(counted)
        claim_groups = {(r["question_id"], r["system_label"]): r["missing_key_source"] for r in rows}
        claims = [value for value in claim_groups.values() if value != "not_applicable"]
        coverage = sum(value == "source_identified" for value in claims) / len(claims) if claims else None
        task2[label] = {"citations": len(rows), "strict_precision": strict, "lenient_precision": lenient, "claim_questions": len(claims), "claim_coverage": coverage}
    task2_by_arm = {}
    for arm in ("a1", "a2"):
        rows = [r for r in t2 if label_map.get((r["question_id"], r["system_label"])) == arm]
        counted = [r for r in rows if r["support"] != "not_applicable"]
        claim_groups = {(r["question_id"], r["system_label"]): r["missing_key_source"] for r in rows}
        claims = [value for value in claim_groups.values() if value != "not_applicable"]
        task2_by_arm[arm] = {"questions_resolved": len({r["question_id"] for r in rows}), "citations": len(rows), "strict_precision": sum(r["support"] == "yes" for r in counted) / len(counted) if counted else None, "lenient_precision": sum(r["support"] in {"yes", "partial"} for r in counted) / len(counted) if counted else None, "claim_questions": len(claims), "claim_coverage": sum(value == "source_identified" for value in claims) / len(claims) if claims else None}
    support_agreement = {}
    for field in ("support", "missing_key_source"):
        support_agreement[field] = {"spearman": None, "exact_agreement": sum(author_t2[r["citation_id"]][field] == r[field] for r in t2) / len(t2)}
    judge_key = json.loads((FROZEN / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    latest_judge = {}
    for name in ("judge_scores.jsonl", "judge_scores_retry.jsonl", "judge_scores_retry2.jsonl"):
        path = FROZEN / name
        if path.exists():
            latest_judge.update({row["blind_id"]: row for row in read_jsonl(path)})
    judge_values = defaultdict(dict)
    for blind_id, row in latest_judge.items():
        if row.get("status") == "usable":
            info = judge_key[blind_id]
            judge_values[info["question_id"]][info["arm"]] = row["mean"]
    human_vs_judge = {}
    for arm in ARMS:
        pairs = [(per_question[qid][arm], judge_values[qid][arm]) for qid in per_question if qid in judge_values and arm in judge_values[qid]]
        human_vs_judge[arm] = {"n": len(pairs), "spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs])}
    result = {"protocol": "swe_qa_v4", "task1": {"n": len(t1), "by_arm": by_arm, "comparisons": comparisons}, "task1_rater_agreement": agreement, "human_vs_automatic_judge": human_vs_judge, "task2": task2, "task2_by_arm": task2_by_arm, "task2_mapping": {"resolved_questions": 30 - len(ambiguous_questions), "ambiguous_questions": ambiguous_questions}, "task2_rater_agreement": support_agreement, "provenance": {"task1_author_sha256": hashlib.sha256((PACKAGE / "task1_items.csv").read_bytes()).hexdigest(), "task1_independent_sha256": hashlib.sha256((PACKAGE / "task1_items_independent.csv").read_bytes()).hexdigest(), "task2_author_sha256": hashlib.sha256((PACKAGE / "task2_items.csv").read_bytes()).hexdigest(), "task2_independent_sha256": hashlib.sha256((PACKAGE / "task2_items_independent.csv").read_bytes()).hexdigest(), "bootstrap_seed": SEED}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
