"""SWE-QA v7 post-hoc analysis: judge scores split by explicit refusal (no model call, no network; exploratory, not part of the frozen protocol).

    python scripts/evaluation/swe_qa_v7_refusal_conditional.py

Refusals are detected with the same lexical detector as the frozen behaviour analysis (`swe_qa_v5_confirm.is_refusal`). For every arm it reports the judge mean over
answered and over refused questions, and for the main contrasts the paired difference restricted to questions that both compared arms answered. It reads only the
frozen v4 answers, the v7 generation files and the new-batch judge scores, and writes `reports/evaluation/swe_qa_v7/main/refusal_conditional_v7.json` once.
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v7_analyze as an  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/evaluation/swe_qa_v7/main/refusal_conditional_v7.json"
ARMS = ("a0", "b0_k5", "a1_v4", "a1_v5a", "a1_v6e")
CONTRASTS = (("a1_v5a", "a1_v4"), ("a1_v5a", "b0_k5"), ("a1_v6e", "a1_v5a"), ("a1_v6e", "b0_k5"), ("a0", "a1_v5a"), ("a0", "b0_k5"), ("a0", "a1_v4"))


def split_scores(scores: dict[tuple[str, str], float], refused: dict[str, set[str]], arm: str) -> tuple[list[float], list[float]]:
    """Judge scores of one arm split into (answered, refused) lists."""
    answered, declined = [], []
    for (question, name), score in scores.items():
        if name == arm:
            (declined if question in refused[arm] else answered).append(score)
    return answered, declined


def paired_answered(scores: dict[tuple[str, str], float], refused: dict[str, set[str]], rows: dict, left: str, right: str) -> list[tuple[str, str, float, float]]:
    """(question, repo, left score, right score) for questions that neither arm refused."""
    return [(q, rows[q]["repo"], scores[(q, left)], scores[(q, right)]) for q in sorted(rows)
            if (q, left) in scores and (q, right) in scores and q not in refused[left] and q not in refused[right]]


def describe(values: list[float]) -> dict:
    return {"n": len(values), "mean": round(stats.mean(values), 3) if values else None}


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT.name} already exists; it is written once")
    rows = an.question_rows()
    records = an.answers()
    units, _ = an.new_batch_scores()
    scores = {key: unit["mean"] for key, unit in units.items()}
    refused = {arm: {q for q in rows if q in records[arm] and an.c5.is_refusal(records[arm][q])} for arm in ARMS}
    result: dict = {"note": "post-hoc exploratory analysis of the frozen new-batch judge scores; refusal = frozen lexical detector", "arms": {}, "paired_both_answered": {}}
    for arm in ARMS:
        answered, declined = split_scores(scores, refused, arm)
        result["arms"][arm] = {"all": describe(answered + declined), "answered": describe(answered), "refused": describe(declined)}
    for left, right in CONTRASTS:
        pairs = paired_answered(scores, refused, rows, left, right)
        summary = an.paired_summary(pairs) if len(pairs) > 1 else {"pairs": len(pairs)}
        summary["median_difference"] = round(stats.median(a - b for _, _, a, b in pairs), 3) if pairs else None
        result["paired_both_answered"][f"{left}_minus_{right}"] = summary
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"arms": result["arms"], "paired": {k: {m: v.get(m) for m in ("pairs", "mean_difference", "ci95", "median_difference", "wilcoxon_p")} for k, v in result["paired_both_answered"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
