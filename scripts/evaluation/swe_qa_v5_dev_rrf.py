"""Offline weighted-RRF diagnostics on the frozen SWE-QA v4 dev sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_qa_v5_dev_context import (
    INDEX_ROOT,
    OUT,
    REPOS,
    SAMPLE_PER_REPO,
    Candidate,
    Index,
    focus,
    load_rankings,
    load_rows,
    pack,
    select_sample,
)

ROOT = Path(__file__).resolve().parents[2]
RRF_K = 60


def weighted_fusion(lexical: dict, semantic: dict, lexical_weight: float, semantic_weight: float) -> list[dict]:
    """Fuse frozen rank lists using weighted reciprocal rank fusion."""
    chunks: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for item, weight in ((lexical, lexical_weight), (semantic, semantic_weight)):
        for rank, chunk in enumerate(item["ranking"], 1):
            chunk_id = chunk["chunk_id"]
            chunks.setdefault(chunk_id, chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank)
    return [
        chunks[chunk_id]
        for chunk_id in sorted(
            chunks,
            key=lambda chunk_id: (-scores[chunk_id], chunks[chunk_id].get("source_file", ""), chunk_id),
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.lexical_weight < 0 or args.semantic_weight < 0 or not (args.lexical_weight + args.semantic_weight):
        raise SystemExit("weights must be non-negative and not both zero")

    rows = load_rows()
    lexical = load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_lexical.jsonl")
    semantic = load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_semantic.jsonl")
    sample = select_sample(rows)
    indexes = {repo: Index(repo) for repo in REPOS}
    report: list[dict] = []
    try:
        for row in sample:
            qid = row["question_id"]
            ranking = weighted_fusion(lexical[qid], semantic[qid], args.lexical_weight, args.semantic_weight)
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

    summary = {
        "sample_size": len(report),
        "sample_per_repo": SAMPLE_PER_REPO,
        "lexical_weight": args.lexical_weight,
        "semantic_weight": args.semantic_weight,
        "rrf_k": RRF_K,
        "rows": report,
    }
    print(json.dumps(summary, indent=2))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        name = f"rrf_l{args.lexical_weight:g}_s{args.semantic_weight:g}.json"
        (OUT / name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
