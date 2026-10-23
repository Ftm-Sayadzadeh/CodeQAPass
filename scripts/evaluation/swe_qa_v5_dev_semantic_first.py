"""Offline semantic-first/lexical-rescue diagnostics on the v4 dev sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_qa_v5_dev_context import (
    OUT,
    REPOS,
    SAMPLE_PER_REPO,
    Index,
    focus,
    load_rankings,
    load_rows,
    pack,
    select_sample,
)

ROOT = Path(__file__).resolve().parents[2]


def semantic_first(semantic: dict, lexical: dict, semantic_k: int) -> list[dict]:
    """Keep semantic's head, then append lexical-only candidates."""
    selected: list[dict] = []
    seen: set[str] = set()
    for chunk in semantic["ranking"][:semantic_k] + lexical["ranking"]:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            selected.append(chunk)
    return selected[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-k", type=int, default=5)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.semantic_k <= 10:
        raise SystemExit("semantic-k must be between 1 and 10")

    rows = load_rows()
    lexical = load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_lexical.jsonl")
    semantic = load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_semantic.jsonl")
    sample = select_sample(rows)
    indexes = {repo: Index(repo) for repo in REPOS}
    report: list[dict] = []
    try:
        for row in sample:
            qid = row["question_id"]
            ranking = semantic_first(semantic[qid], lexical[qid], args.semantic_k)
            index = indexes[row["repo"]]
            candidates = [candidate for rank, item in enumerate(ranking, 1) for candidate in index.expand(item, rank, row["question"], split_large=False)]
            packed = pack(focus(row["question"], candidates), "baseline")
            gold = set(semantic[qid]["gold_files"])
            files = {block.source_file for block in packed["blocks"]}
            report.append({
                "question_id": qid,
                "repo": row["repo"],
                "gold_files": sorted(gold),
                "context_files": sorted(files),
                "blocks": len(packed["blocks"]),
                "chars": packed["total_chars"],
                "omitted": packed["omitted_count"],
                "gold_files_in_context": len(gold & files),
                "ranking": ranking,
            })
    finally:
        for index in indexes.values():
            index.close()

    summary = {"sample_size": len(report), "sample_per_repo": SAMPLE_PER_REPO, "semantic_k": args.semantic_k, "rows": report}
    print(json.dumps(summary, indent=2))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"semantic_first_k{args.semantic_k}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
