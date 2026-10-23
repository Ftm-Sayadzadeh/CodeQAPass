"""Offline query-relevant method selection for large class chunks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from swe_qa_v5_dev_context import OUT, REPOS, SAMPLE_PER_REPO, Index, focus, load_rankings, load_rows, pack, select_sample

ROOT = Path(__file__).resolve().parents[2]


def query_terms(question: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", question) if len(x) > 2}


def select_candidates(row: dict, ranking: dict, index: Index, methods_per_class: int) -> list:
    terms = query_terms(row["question"])
    selected = []
    for rank, item in enumerate(ranking["ranking"], 1):
        expanded = index.expand(item, rank, row["question"], split_large=True)
        method_parts = [c for c in expanded if "#" in c.chunk_id and not c.chunk_id.endswith("#header")]
        if method_parts:
            header = [c for c in expanded if c.chunk_id.endswith("#header")]
            method_parts.sort(key=lambda c: (-sum(term in (c.symbol_name or "").lower() for term in terms), c.start_line))
            selected.extend(header + method_parts[:methods_per_class])
        else:
            selected.extend(expanded)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--methods-per-class", type=int, default=2); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    if args.methods_per_class < 1: raise SystemExit("methods-per-class must be positive")
    rows = load_rows(); rankings = load_rankings(ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_semantic.jsonl"); sample = select_sample(rows)
    indexes = {repo: Index(repo) for repo in REPOS}; report = []
    try:
        for row in sample:
            ranking = rankings[row["question_id"]]; candidates = select_candidates(row, ranking, indexes[row["repo"]], args.methods_per_class)
            packed = pack(focus(row["question"], candidates), "baseline"); gold = set(ranking["gold_files"]); files = {b.source_file for b in packed["blocks"]}
            report.append({"question_id": row["question_id"], "repo": row["repo"], "context_files": sorted(files), "blocks": len(packed["blocks"]), "chars": packed["total_chars"], "omitted": packed["omitted_count"], "gold_files_in_context": len(gold & files), "full_gold": gold <= files})
    finally:
        for index in indexes.values(): index.close()
    summary = {"sample_size": len(report), "sample_per_repo": SAMPLE_PER_REPO, "methods_per_class": args.methods_per_class, "rows": report}
    print(json.dumps(summary, indent=2))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True); (OUT / f"method_select_{args.methods_per_class}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
