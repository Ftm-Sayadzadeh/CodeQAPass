"""Offline dev-sample diagnostics for SWE-QA v5 context packing experiments.

This script never calls a model and never modifies the frozen v4 artifacts. It
replays frozen hybrid rankings against the v4 SQLite indexes and compares the
current greedy context builder with a method-split packing candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/swe_qa_v5_dev"
DATASET = ROOT / "data/benchmarks/swe_qa_v4/dataset"
RANKINGS = ROOT / "reports/evaluation/swe_qa_v4/frozen/rankings_hybrid.jsonl"
INDEX_ROOT = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2/index"
REPO_ROOT = ROOT / "data/benchmarks/swe_qa_v4/repos"
REPOS = ("flask", "requests", "pytest", "pylint")
SAMPLE_PER_REPO = 10
MAX_CONTEXT_CHARS = 6000
MAX_BLOCK_CHARS = 1600


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    source_file: str
    symbol_name: str | None
    qualified_name: str | None
    start_line: int
    end_line: int
    code: str
    score_rank: int


def _mentions_symbol(text: str, qualified_name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(qualified_name)}(?!\w)", text, re.IGNORECASE) is not None


def _mentions_code_name(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", text) is not None


def load_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for repo in REPOS:
        with (DATASET / f"{repo}.jsonl").open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row = {"question_id": f"{repo}-{number}", "repo": repo, **row}
                rows[row["question_id"]] = row
    return rows


def select_sample(rows: dict[str, dict]) -> list[dict]:
    selected: list[dict] = []
    by_repo = {repo: [] for repo in REPOS}
    with RANKINGS.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["question_id"] in rows and len(by_repo.get(item["repo"], [])) < SAMPLE_PER_REPO:
                by_repo[item["repo"]].append(item["question_id"])
    for repo in REPOS:
        selected.extend(rows[qid] for qid in by_repo[repo])
    if len(selected) != SAMPLE_PER_REPO * len(REPOS):
        raise RuntimeError(f"expected {SAMPLE_PER_REPO * len(REPOS)} sample rows, found {len(selected)}")
    return selected


def load_rankings(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {item["question_id"]: item for item in map(json.loads, handle)}


class Index:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.connection = sqlite3.connect(INDEX_ROOT / f"{repo}.sqlite")
        self.connection.row_factory = sqlite3.Row
        self.chunks = {
            row["chunk_id"]: row
            for row in self.connection.execute(
                    """
                    SELECT c.chunk_id, c.chunk_type, c.file_id, c.symbol_id,
                           c.start_line, c.end_line, c.code, sf.relative_path,
                           s.name AS symbol_name, s.qualified_name,
                           s.parent_qualified_name
                    FROM chunks c
                    JOIN source_files sf ON sf.id = c.file_id
                    LEFT JOIN symbols s ON s.id = c.symbol_id
                    """
            )
        }
        self.methods_by_class: dict[tuple[int, str], list[sqlite3.Row]] = {}
        for row in self.connection.execute(
            "SELECT file_id, kind, name, qualified_name, parent_qualified_name, start_line, end_line "
            "FROM symbols WHERE kind IN ('method', 'function') ORDER BY start_line"
        ):
            if row["kind"] == "method" and row["parent_qualified_name"]:
                self.methods_by_class.setdefault((row["file_id"], row["parent_qualified_name"]), []).append(row)

    def close(self) -> None:
        self.connection.close()

    def _source(self, relative_path: str) -> list[str]:
        return (REPO_ROOT / self.repo / relative_path).read_text(encoding="utf-8").splitlines(keepends=True)

    def expand(self, item: dict, rank: int, question: str, split_large: bool = True) -> list[Candidate]:
        row = self.chunks.get(item["chunk_id"])
        if row is None:
            raise RuntimeError(f"missing chunk {item['chunk_id']} in {self.repo} index")
        base = Candidate(
            chunk_id=row["chunk_id"], source_file=row["relative_path"], symbol_name=row["symbol_name"],
            qualified_name=row["qualified_name"], start_line=row["start_line"], end_line=row["end_line"],
            code=row["code"], score_rank=rank,
        )
        methods = self.methods_by_class.get((row["file_id"], row["qualified_name"]), [])
        if not split_large or row["chunk_type"] != "class_overview" or not methods or len(row["code"]) <= MAX_CONTEXT_CHARS:
            return [base]
        source = self._source(row["relative_path"])
        pieces: list[Candidate] = []
        header_end = methods[0]["start_line"] - 1
        if header_end >= row["start_line"]:
            pieces.append(Candidate(
                chunk_id=f"{row['chunk_id']}#header", source_file=row["relative_path"],
                symbol_name=row["symbol_name"], qualified_name=row["qualified_name"],
                start_line=row["start_line"], end_line=header_end,
                code="".join(source[row["start_line"] - 1:header_end]), score_rank=rank,
            ))
        query_terms = {term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", question)}
        methods = sorted(
            methods,
            key=lambda method: (
                -sum(term in method["name"].lower() for term in query_terms),
                method["start_line"],
            ),
        )
        for method in methods:
            pieces.append(Candidate(
                chunk_id=f"{row['chunk_id']}#{method['qualified_name']}", source_file=row["relative_path"],
                symbol_name=method["name"], qualified_name=method["qualified_name"],
                start_line=method["start_line"], end_line=method["end_line"],
                code="".join(source[method["start_line"] - 1:method["end_line"]]), score_rank=rank,
            ))
        return pieces or [base]


def focus(question: str, candidates: list[Candidate]) -> list[Candidate]:
    exact = {
        c.qualified_name for c in candidates
        if c.qualified_name and "." in c.qualified_name and _mentions_symbol(question, c.qualified_name)
    }
    if exact:
        return [c for c in candidates if c.qualified_name in exact]
    roots = {
        c.qualified_name.split(".", 1)[0] for c in candidates
        if c.qualified_name and _mentions_code_name(question, c.qualified_name.split(".", 1)[0])
    }
    if roots:
        return [c for c in candidates if c.qualified_name and c.qualified_name.split(".", 1)[0] in roots]
    symbols = {
        c.symbol_name for c in candidates
        if c.symbol_name and not c.symbol_name.startswith("__") and _mentions_code_name(question, c.symbol_name)
    }
    if symbols:
        return [c for c in candidates if c.symbol_name in symbols]
    return candidates


def round_robin(candidates: list[Candidate]) -> list[Candidate]:
    """Give each original retrieval rank a turn before expanding any one rank."""
    groups: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.score_rank].append(candidate)
    ordered: list[Candidate] = []
    for rank in sorted(groups):
        groups[rank].sort(key=lambda candidate: (candidate.start_line, candidate.chunk_id))
    for offset in range(max((len(group) for group in groups.values()), default=0)):
        ordered.extend(groups[rank][offset] for rank in sorted(groups) if offset < len(groups[rank]))
    return ordered


def pack(candidates: list[Candidate], mode: str) -> dict:
    blocks: list[Candidate] = []
    used = 0
    omitted = 0
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)
        remaining = MAX_CONTEXT_CHARS - used
        code = candidate.code
        if mode == "quota":
            code = code[: min(MAX_BLOCK_CHARS, remaining)]
        if len(code) <= remaining and (mode != "quota" or code):
            blocks.append(candidate)
            if mode == "quota" and len(code) != len(candidate.code):
                candidate = Candidate(**{**asdict(candidate), "code": code, "end_line": candidate.start_line + code.count("\n")})
            blocks[-1] = candidate
            used += len(code)
        elif mode == "split" and remaining > 0:
            code = candidate.code[:remaining]
            end_line = candidate.start_line + code.count("\n")
            blocks.append(Candidate(**{**asdict(candidate), "code": code, "end_line": end_line}))
            used += len(code)
        else:
            omitted += 1
    return {"blocks": blocks, "total_chars": used, "omitted_count": omitted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the sample and diagnostic JSON")
    parser.add_argument("--method", choices=("hybrid", "semantic"), default="hybrid")
    args = parser.parse_args()
    rows = load_rows()
    rankings = load_rankings(ROOT / f"reports/evaluation/swe_qa_v4/frozen/rankings_{args.method}.jsonl")
    sample = select_sample(rows)
    report: list[dict] = []
    indexes = {repo: Index(repo) for repo in REPOS}
    try:
        for row in sample:
            ranking = rankings[row["question_id"]]["ranking"]
            original = [candidate for rank, item in enumerate(ranking, 1) for candidate in indexes[row["repo"]].expand(item, rank, row["question"], split_large=False)]
            candidates = [candidate for rank, item in enumerate(ranking, 1) for candidate in indexes[row["repo"]].expand(item, rank, row["question"], split_large=True)]
            baseline = pack(focus(row["question"], original), "baseline")
            focused = focus(row["question"], candidates)
            split = pack(round_robin(focused), "split")
            quota = pack(focused, "quota")
            nofocus = pack(original, "baseline")
            fallback = nofocus if len(baseline["blocks"]) < 2 and len(focused) < len(original) else baseline
            gold = set(rankings[row["question_id"]]["gold_files"])
            report.append({
                "question_id": row["question_id"], "repo": row["repo"], "gold_files": sorted(gold),
                "baseline": {"context_files": sorted({b.source_file for b in baseline["blocks"]}), "blocks": len(baseline["blocks"]), "chars": baseline["total_chars"], "omitted": baseline["omitted_count"], "gold_files_in_context": len(gold & {b.source_file for b in baseline["blocks"]})},
                "split": {"context_files": sorted({b.source_file for b in split["blocks"]}), "blocks": len(split["blocks"]), "chars": split["total_chars"], "omitted": split["omitted_count"], "gold_files_in_context": len(gold & {b.source_file for b in split["blocks"]})},
                "quota": {"context_files": sorted({b.source_file for b in quota["blocks"]}), "blocks": len(quota["blocks"]), "chars": quota["total_chars"], "omitted": quota["omitted_count"], "gold_files_in_context": len(gold & {b.source_file for b in quota["blocks"]})},
                "nofocus": {"context_files": sorted({b.source_file for b in nofocus["blocks"]}), "blocks": len(nofocus["blocks"]), "chars": nofocus["total_chars"], "omitted": nofocus["omitted_count"], "gold_files_in_context": len(gold & {b.source_file for b in nofocus["blocks"]})},
                "fallback": {"context_files": sorted({b.source_file for b in fallback["blocks"]}), "blocks": len(fallback["blocks"]), "chars": fallback["total_chars"], "omitted": fallback["omitted_count"], "gold_files_in_context": len(gold & {b.source_file for b in fallback["blocks"]})},
            })
    finally:
        for index in indexes.values():
            index.close()
    summary = {"sample_size": len(report), "sample_per_repo": SAMPLE_PER_REPO, "context_budget": MAX_CONTEXT_CHARS, "rows": report}
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "dev_sample.json").write_text(json.dumps([r["question_id"] for r in sample], indent=2) + "\n", encoding="utf-8")
        (OUT / f"context_packing_{args.method}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
