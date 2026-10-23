"""SWE-QA v7 post-hoc extras (no model call, no network; exploratory, not part of the frozen protocol).

    python scripts/evaluation/swe_qa_v7_posthoc_extras.py

1. Judge scores on the set of questions that every retrieval configuration answered (no explicit refusal in any of them), so that conditional quality is compared on one
   fixed question set.
2. Study 2 latency split into the query-embedding call and the remaining retrieval-plus-assembly time (median over questions of the per-question median over repeats).
3. Cost per good answer recomputed from the provider-reported cost of Table VI (mean of per-question medians) and the Study 1 good-answer rate.

Reads only frozen run records and writes `reports/evaluation/swe_qa_v7/main/posthoc_extras_v7.json` once.
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v7_analyze as an  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "reports/evaluation/swe_qa_v7/main"
OUTPUT = MAIN / "posthoc_extras_v7.json"
RETRIEVAL_ARMS = ("b0_k5", "a1_v4", "a1_v5a", "a1_v6e")


def common_answered(scores: dict[tuple[str, str], float], refused: dict[str, set[str]], questions: list[str]) -> dict:
    """Mean judge score of every retrieval arm on the questions that none of them refused."""
    common = [q for q in sorted(questions) if all((q, arm) in scores and q not in refused[arm] for arm in RETRIEVAL_ARMS)]
    return {"questions": len(common), "means": {arm: round(stats.mean(scores[(q, arm)] for q in common), 3) for arm in RETRIEVAL_ARMS} if common else {}}


def latency_split(rows: list[dict]) -> dict:
    """Per arm: median of per-question medians of embedding seconds, local retrieval-plus-assembly seconds (total minus embedding) and the total."""
    out: dict = {}
    for arm in RETRIEVAL_ARMS:
        per_question: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            if row["arm"] != arm or row["status"] != "ok" or row["warmup"]:
                continue
            total, embedding = row.get("retrieval_assembly_seconds"), row.get("embedding_seconds")
            if total is None or embedding is None:
                continue
            per_question[row["question_id"]].append((embedding, max(total - embedding, 0.0)))
        if not per_question:
            continue
        embed = [stats.median(e for e, _ in v) for v in per_question.values()]
        local = [stats.median(l for _, l in v) for v in per_question.values()]
        out[arm] = {"questions": len(per_question), "embedding_seconds": round(stats.median(embed), 3), "local_retrieval_assembly_seconds": round(stats.median(local), 3)}
    return out


def cost_per_good_answer(provider_cost: dict[str, float], good_rate: dict[str, float]) -> dict:
    return {arm: round(provider_cost[arm] / good_rate[arm] * 1e4, 2) for arm in RETRIEVAL_ARMS if good_rate.get(arm)}


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT.name} already exists; it is written once")
    rows = an.question_rows()
    records = an.answers()
    units, _ = an.new_batch_scores()
    scores = {key: unit["mean"] for key, unit in units.items()}
    refused = {arm: {q for q in rows if q in records[arm] and an.c5.is_refusal(records[arm][q])} for arm in RETRIEVAL_ARMS}
    analysis = json.loads((MAIN / "analysis_v7.json").read_text(encoding="utf-8"))
    provider_cost = {arm: analysis["efficiency"]["arms"][arm]["provider_cost_unit"]["mean_of_question_medians"] for arm in RETRIEVAL_ARMS}
    good_rate = {arm: analysis["behaviour"]["arms"][arm]["good_answer"]["rate"] for arm in RETRIEVAL_ARMS}
    timing = [json.loads(line) for line in (MAIN / "study2_timing.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {"note": "post-hoc exploratory; frozen records only",
              "common_answered_all_retrieval_arms": common_answered(scores, refused, list(rows)),
              "latency_split_seconds": latency_split(timing),
              "cost_per_good_answer_1e-4_units": {"provider_cost": provider_cost, "good_answer_rate_study1": good_rate, "value": cost_per_good_answer(provider_cost, good_rate)}}
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
