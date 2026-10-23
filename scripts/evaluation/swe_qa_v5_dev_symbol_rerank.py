"""Offline symbol/name reranking diagnostics on frozen semantic rankings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from swe_qa_v5_dev_context import OUT, REPOS, SAMPLE_PER_REPO, Index, focus, load_rankings, load_rows, pack, select_sample

ROOT = Path(__file__).resolve().parents[2]


def terms(question: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", question) if len(token) > 2}


def rerank(row: dict, ranking: dict, index: Index, strength: float) -> list[dict]:
    query_terms = terms(row["question"])
    scored: list[tuple[float, int, dict]] = []
    for original_rank, item in enumerate(ranking["ranking"][:10], 1):
        chunk = index.chunks[item["chunk_id"]]
        names = " ".join(str(chunk[key] or "") for key in ("relative_path", "symbol_name", "qualified_name")).lower()
        overlap = len(query_terms & set(re.findall(r"[a-z_][a-z0-9_]+", names)))
        scored.append((strength * overlap - original_rank * 0.001, original_rank, item))
    return [item for _, _, item in sorted(scored, key=lambda value: (-value[0], value[1]))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=1.0)
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
            ranking = rankings[qid]
            index = indexes[row["repo"]]
            reordered = rerank(row, ranking, index, args.strength)
            candidates = [candidate for rank, item in enumerate(reordered, 1) for candidate in index.expand(item, rank, row["question"], split_large=False)]
            packed = pack(focus(row["question"], candidates), "baseline")
            gold = set(ranking["gold_files"])
            files = {block.source_file for block in packed["blocks"]}
            report.append({"question_id": qid, "repo": row["repo"], "context_files": sorted(files), "blocks": len(packed["blocks"]), "chars": packed["total_chars"], "omitted": packed["omitted_count"], "gold_files_in_context": len(gold & files), "full_gold": gold <= files, "ranking": reordered})
    finally:
        for index in indexes.values():
            index.close()
    summary = {"sample_size": len(report), "sample_per_repo": SAMPLE_PER_REPO, "strength": args.strength, "rows": report}
    print(json.dumps(summary, indent=2))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"symbol_rerank_s{args.strength:g}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
