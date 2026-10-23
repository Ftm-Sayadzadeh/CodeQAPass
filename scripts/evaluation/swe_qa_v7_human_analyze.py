"""SWE-QA v7 Study 4 analysis (protocol section 7): human scores per arm, rater agreement, agreement with the automatic judge, citation support.

    python scripts/evaluation/swe_qa_v7_human_analyze.py

Run only after both score files are submitted and validated. The blinding key is decrypted in memory (OpenSSL) and never written to disk. Writes
`human_review_v7/human_analysis_v7.json` once. The independent rater is the primary human analysis; the author is a sensitivity analysis (the author also rated).
"""

from __future__ import annotations

import csv
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v7_analyze as an  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "reports/evaluation/swe_qa_v7/human_review_v7"
ARMS = ("a0", "b0_k5", "a1_v4", "a1_v5a")
CITED = ("a1_v4", "a1_v5a")
CRITERIA = ("correctness", "completeness", "relevance", "groundedness")
RATERS = ("independent", "author")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_key() -> dict:
    out = subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-in", str(PACKAGE / "custodian/key.enc"), "-pass", f"file:{PACKAGE / 'custodian/passphrase.txt'}"],
                         capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8"))


def quadratic_weighted_kappa(a: list[int], b: list[int], categories: range = range(0, 11)) -> float | None:
    cats = list(categories)
    n = len(a)
    if not n:
        return None
    k = len(cats)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[cats.index(x)][cats.index(y)] += 1
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[i][j]
            den += w * hist_a[i] * hist_b[j] / n
    return round(1 - num / den, 4) if den else None


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else None


def main() -> int:
    target = PACKAGE / "human_analysis_v7.json"
    if target.exists():
        raise SystemExit("human_analysis_v7.json already exists; it is written once")
    key = load_key()
    rows = an.question_rows()
    judge_units, _ = an.new_batch_scores()
    repo_of = lambda q: rows[q]["repo"]  # noqa: E731

    # ---- Task 1: scores per (rater, question, arm)
    scores: dict[str, dict[tuple[str, str], dict]] = {}
    refusal: dict[str, dict[tuple[str, str], str]] = {}
    for rater in RATERS:
        rated = read(PACKAGE / "submitted" / rater / "task1_items_scored.csv")
        scores[rater], refusal[rater] = {}, {}
        for r in rated:
            info = key[rater]["task1"][r["item_id"]]
            vals = {c: int(r[c]) for c in CRITERIA}
            scores[rater][(info["question_id"], info["arm"])] = {**vals, "mean": sum(vals.values()) / 4}
            refusal[rater][(info["question_id"], info["arm"])] = r["refusal"].strip().lower()
    questions = sorted({q for q, _ in scores["independent"]})
    result: dict = {"questions": len(questions), "note": "the independent rater is primary; the author is a sensitivity analysis; n = 30 questions per arm, descriptive"}

    def arm_summary(rater: str) -> dict:
        out = {}
        for arm in ARMS:
            groups: dict[str, list[float]] = {}
            for q in questions:
                groups.setdefault(repo_of(q), []).append(scores[rater][(q, arm)]["mean"])
            vals = [x for g in groups.values() for x in g]
            out[arm] = {"mean": round(stats.mean(vals), 3), "ci95": an.bootstrap_interval(groups), "criteria": {c: round(stats.mean(scores[rater][(q, arm)][c] for q in questions), 3) for c in CRITERIA},
                        "refusal_rate": round(sum(refusal[rater][(q, arm)] == "yes" for q in questions) / len(questions), 3)}
        return out

    def contrasts(rater: str) -> dict:
        out = {}
        for i, a in enumerate(ARMS):
            for b in ARMS[:i]:
                out[f"{a}_minus_{b}"] = an.paired_summary([(q, repo_of(q), scores[rater][(q, a)]["mean"], scores[rater][(q, b)]["mean"]) for q in questions])
        return out

    result["arms_primary_independent"] = arm_summary("independent")
    result["contrasts_primary_independent"] = contrasts("independent")
    result["arms_sensitivity_author"] = arm_summary("author")
    result["contrasts_sensitivity_author"] = contrasts("author")
    # mean of the two raters
    pooled = {(q, arm): {"mean": (scores["independent"][(q, arm)]["mean"] + scores["author"][(q, arm)]["mean"]) / 2} for q in questions for arm in ARMS}
    result["contrasts_mean_of_two_raters"] = {f"{a}_minus_{b}": an.paired_summary([(q, repo_of(q), pooled[(q, a)]["mean"], pooled[(q, b)]["mean"]) for q in questions])
                                              for i, a in enumerate(ARMS) for b in ARMS[:i]}

    # ---- agreement between the two raters and with the automatic judge
    keys = [(q, arm) for q in questions for arm in ARMS]
    agreement = {c: {"weighted_kappa_quadratic": quadratic_weighted_kappa([scores["independent"][k][c] for k in keys], [scores["author"][k][c] for k in keys])} for c in CRITERIA}
    overall_i = [scores["independent"][k]["mean"] for k in keys]
    overall_a = [scores["author"][k]["mean"] for k in keys]
    agreement["overall_mean"] = {"spearman": an.spearman(overall_i, overall_a), "mean_abs_difference": round(stats.mean(abs(x - y) for x, y in zip(overall_i, overall_a)), 3)}
    agreement["refusal"] = {"cohen_kappa": cohen_kappa([refusal["independent"][k] for k in keys], [refusal["author"][k] for k in keys]),
                            "exact_agreement": round(sum(refusal["independent"][k] == refusal["author"][k] for k in keys) / len(keys), 3)}
    result["rater_agreement"] = agreement
    result["agreement_with_judge"] = {}
    for rater in RATERS:
        per_arm = {arm: an.spearman([scores[rater][(q, arm)]["mean"] for q in questions if (q, arm) in judge_units], [judge_units[(q, arm)]["mean"] for q in questions if (q, arm) in judge_units])
                   for arm in ARMS}
        allk = [k for k in keys if k in judge_units]
        result["agreement_with_judge"][rater] = {"overall_spearman": an.spearman([scores[rater][k]["mean"] for k in allk], [judge_units[k]["mean"] for k in allk]), "per_arm_spearman": per_arm, "items": len(allk)}

    # ---- Task 2: citation support (the two A1 arms, each list attached to its own answer)
    support: dict[str, dict[tuple[str, str, str, int], str]] = {}
    missing: dict[str, dict[tuple[str, str], str]] = {}
    for rater in RATERS:
        rated = read(PACKAGE / "submitted" / rater / "task2_items_scored.csv")
        support[rater], missing[rater] = {}, {}
        seen: dict[tuple[str, str, str], int] = {}
        for r in rated:
            info = key[rater]["task2"]["citations"][r["citation_id"]]
            tag = (info["question_id"], info["arm"], r["cited_range"])
            idx = seen.get(tag, 0)
            seen[tag] = idx + 1
            support[rater][tag + (idx,)] = r["support"].strip().lower()
            if r["missing_key_source"].strip():
                missing[rater][(info["question_id"], info["arm"])] = r["missing_key_source"].strip().lower()

    def precision(rater: str, arm: str, lenient: bool) -> dict:
        per_q: dict[str, list[int]] = {}
        for (q, a, _, _), s in support[rater].items():
            if a == arm and s != "not_applicable":
                per_q.setdefault(q, []).append(1 if (s == "yes" or (lenient and s == "partial")) else 0)
        groups: dict[str, list[float]] = {}
        for q, v in per_q.items():
            groups.setdefault(repo_of(q), []).append(sum(v) / len(v))
        total = sum(len(v) for v in per_q.values())
        pooled_value = sum(sum(v) for v in per_q.values()) / total if total else None
        return {"citations": total, "pooled": round(pooled_value, 4) if pooled_value is not None else None, "mean_of_question_precision": round(stats.mean(x for g in groups.values() for x in g), 4) if groups else None,
                "ci95_question_bootstrap": an.bootstrap_interval(groups) if groups else None, "questions": len(per_q)}

    result["citation_support"] = {rater: {arm: {"strict": precision(rater, arm, False), "lenient": precision(rater, arm, True),
                                                "labels": {s: sum(1 for (q, a, _, _), v in support[rater].items() if a == arm and v == s) for s in ("yes", "partial", "no", "not_applicable")},
                                                "missing_key_source": {s: sum(1 for (q, a), v in missing[rater].items() if a == arm and v == s) for s in ("source_identified", "source_missing", "not_applicable")}}
                                          for arm in CITED} for rater in RATERS}
    common = sorted(set(support["independent"]) & set(support["author"]))
    result["citation_rater_agreement"] = {"matched_citations": len(common), "of": [len(support["independent"]), len(support["author"])],
                                          "exact_agreement": round(sum(support["independent"][k] == support["author"][k] for k in common) / len(common), 4) if common else None,
                                          "cohen_kappa": cohen_kappa([support["independent"][k] for k in common], [support["author"][k] for k in common])}
    pairs = []
    for q in questions:
        ps = {}
        for arm in CITED:
            vals = [1 if v == "yes" else 0 for (qq, a, _, _), v in support["independent"].items() if qq == q and a == arm and v != "not_applicable"]
            ps[arm] = sum(vals) / len(vals) if vals else None
        if all(ps[a] is not None for a in CITED):
            pairs.append((q, repo_of(q), ps["a1_v5a"], ps["a1_v4"]))
    result["citation_strict_precision_v5a_minus_v4_independent"] = an.paired_summary(pairs) if len(pairs) > 1 else {"pairs": len(pairs), "status": "insufficient"}
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("written", target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
