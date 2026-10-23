"""Offline context-budget sensitivity diagnostics on frozen semantic rankings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import swe_qa_v5_dev_context as ctx

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, nargs="+", default=[6000, 8000, 10000])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = ctx.load_rows()
    rankings = ctx.load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_semantic.jsonl")
    sample = ctx.select_sample(rows)
    indexes = {repo: ctx.Index(repo) for repo in ctx.REPOS}
    original_budget = ctx.MAX_CONTEXT_CHARS
    summaries: list[dict] = []
    try:
        for budget in args.budget:
            if budget <= 0:
                raise SystemExit("budget must be positive")
            ctx.MAX_CONTEXT_CHARS = budget
            report: list[dict] = []
            for row in sample:
                ranking = rankings[row["question_id"]]
                index = indexes[row["repo"]]
                candidates = [candidate for rank, item in enumerate(ranking["ranking"], 1) for candidate in index.expand(item, rank, row["question"], split_large=False)]
                packed = ctx.pack(ctx.focus(row["question"], candidates), "baseline")
                gold = set(ranking["gold_files"])
                files = {block.source_file for block in packed["blocks"]}
                report.append({"question_id": row["question_id"], "gold": len(gold & files), "full": gold <= files, "blocks": len(packed["blocks"]), "chars": packed["total_chars"], "omitted": packed["omitted_count"]})
            summaries.append({"budget": budget, "sample_size": len(report), "gold_files": sum(r["gold"] for r in report), "full_questions": sum(r["full"] for r in report), "avg_blocks": statistics.mean(r["blocks"] for r in report), "avg_chars": statistics.mean(r["chars"] for r in report), "avg_omitted": statistics.mean(r["omitted"] for r in report), "rows": report})
    finally:
        ctx.MAX_CONTEXT_CHARS = original_budget
        for index in indexes.values():
            index.close()
    print(json.dumps(summaries, indent=2))
    if args.write:
        ctx.OUT.mkdir(parents=True, exist_ok=True)
        (ctx.OUT / "budget_sensitivity.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
