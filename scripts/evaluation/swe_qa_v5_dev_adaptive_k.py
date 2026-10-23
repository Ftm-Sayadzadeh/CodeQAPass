"""Offline adaptive retrieval-limit diagnostics on the frozen v4 dev sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_qa_v5_dev_context import OUT, REPOS, SAMPLE_PER_REPO, Index, focus, load_rankings, load_rows, pack, select_sample

ROOT = Path(__file__).resolve().parents[2]


def evaluate(row: dict, ranking: dict, index: Index, limit: int) -> dict:
    items = ranking["ranking"][:limit]
    candidates = [candidate for rank, item in enumerate(items, 1) for candidate in index.expand(item, rank, row["question"], split_large=False)]
    packed = pack(focus(row["question"], candidates), "baseline")
    gold = set(ranking["gold_files"])
    files = {block.source_file for block in packed["blocks"]}
    return {
        "context_files": sorted(files),
        "blocks": len(packed["blocks"]),
        "chars": packed["total_chars"],
        "omitted": packed["omitted_count"],
        "gold_files_in_context": len(gold & files),
        "full_gold": gold <= files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--min-omitted", type=int, default=3)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = load_rows()
    rankings = load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_semantic.jsonl")
    sample = select_sample(rows)
    indexes = {repo: Index(repo) for repo in REPOS}
    report: list[dict] = []
    try:
        for row in sample:
            qid = row["question_id"]
            index = indexes[row["repo"]]
            k5 = evaluate(row, rankings[qid], index, 5)
            use_k10 = k5["blocks"] < args.min_blocks or k5["omitted"] >= args.min_omitted
            chosen = evaluate(row, rankings[qid], index, 10) if use_k10 else k5
            report.append({"question_id": qid, "repo": row["repo"], "used_k": 10 if use_k10 else 5, "k5": k5, "chosen": chosen})
    finally:
        for index in indexes.values():
            index.close()
    summary = {
        "sample_size": len(report),
        "sample_per_repo": SAMPLE_PER_REPO,
        "base_k": 5,
        "fallback_k": 10,
        "min_blocks": args.min_blocks,
        "min_omitted": args.min_omitted,
        "fallback_count": sum(r["used_k"] == 10 for r in report),
        "rows": report,
    }
    print(json.dumps(summary, indent=2))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"adaptive_k_b5_f10_b{args.min_blocks}_o{args.min_omitted}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
