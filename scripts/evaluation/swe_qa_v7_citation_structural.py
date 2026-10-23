"""SWE-QA v7 structural citation check for the A1 arms (no model call, no network).

    python scripts/evaluation/swe_qa_v7_citation_structural.py

For every citation of a usable answer it checks (1) the schema (chunk id, source file, integer start/end lines), (2) positive, ordered line ranges, (3) that the chunk
exists in the repository's canonical SQLite index, (4) that the cited file equals the canonical file and the cited range lies inside the canonical range, and (5) for
v7 records, that the cited chunk is one of the chunks admitted to the context. This is a provenance check, not semantic support. Writes
`reports/evaluation/swe_qa_v7/main/citation_structural_v7.json` once (never overwrites).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2/index"
SOURCES = {
    "a1_v4": ROOT / "reports/evaluation/swe_qa_v4/frozen/main_a1_combined.jsonl",
    "a1_v5a": ROOT / "reports/evaluation/swe_qa_v7/main/study1_generation_a1_v5a.jsonl",
    "a1_v6e": ROOT / "reports/evaluation/swe_qa_v7/main/study1_generation_a1_v6e.jsonl",
}
OUTPUT = ROOT / "reports/evaluation/swe_qa_v7/main/citation_structural_v7.json"
CHECKS = ("schema_ok", "lines_positive_ordered", "chunk_in_sqlite", "file_matches_sqlite", "range_inside_sqlite_range", "range_equals_sqlite_range", "chunk_in_context")


def check_citation(citation: dict, canonical: tuple[str, int, int] | None, context_ids: set[str] | None) -> dict[str, bool | None]:
    """Return the per-check outcome for one citation; `chunk_in_context` is None when the record has no context list."""
    start, end = citation.get("start_line"), citation.get("end_line")
    schema = bool(citation.get("chunk_id")) and bool(citation.get("source_file")) and isinstance(start, int) and isinstance(end, int)
    ordered = schema and 1 <= start <= end
    found = canonical is not None
    path_ok = found and canonical[0] == citation.get("source_file")
    inside = bool(found and ordered and canonical[1] <= start and end <= canonical[2])
    exact = bool(found and ordered and canonical[1] == start and canonical[2] == end)
    return {
        "schema_ok": schema, "lines_positive_ordered": bool(ordered), "chunk_in_sqlite": found, "file_matches_sqlite": bool(path_ok),
        "range_inside_sqlite_range": inside, "range_equals_sqlite_range": exact,
        "chunk_in_context": None if context_ids is None else citation.get("chunk_id") in context_ids,
    }


def parent_id(citation: dict) -> str:
    return citation.get("parent_chunk_id") or str(citation.get("chunk_id", "")).split("#")[0]


def summarize(records: list[dict], lookup) -> dict:
    """Aggregate the per-citation checks over the usable answers of one arm. `lookup(repo, chunk_id)` returns (path, start, end) or None."""
    usable = [r for r in records if r.get("status") == "usable"]
    totals = {name: 0 for name in CHECKS}
    context_checked = 0
    citations = answers_with_citations = answers_all_ok = 0
    for record in usable:
        cites = record.get("citations") or []
        context = record.get("context_chunk_ids")
        context_ids = set(context) if isinstance(context, list) else None
        outcomes = [check_citation(c, lookup(record["repo"], parent_id(c)), context_ids) for c in cites]
        citations += len(cites)
        answers_with_citations += bool(cites)
        for outcome in outcomes:
            for name in CHECKS:
                if outcome[name] is not None:
                    totals[name] += bool(outcome[name])
            context_checked += outcome["chunk_in_context"] is not None
        required = [n for n in CHECKS if n not in ("range_equals_sqlite_range", "chunk_in_context")]
        answers_all_ok += bool(cites) and all(all(o[n] for n in required) and o["chunk_in_context"] is not False for o in outcomes)
    return {"records": len(records), "usable_answers": len(usable), "answers_with_citations": answers_with_citations, "answers_all_citations_valid": answers_all_ok,
            "citations": citations, "citations_passing": {name: totals[name] for name in CHECKS}, "chunk_in_context_checked": context_checked}


def sqlite_lookup():
    connections: dict[str, sqlite3.Connection] = {}

    def lookup(repo: str, chunk_id: str):
        if repo not in connections:
            connections[repo] = sqlite3.connect(f"file:{INDEX / (repo + '.sqlite')}?mode=ro", uri=True)
        row = connections[repo].execute(
            "SELECT f.relative_path, c.start_line, c.end_line FROM chunks c JOIN source_files f ON f.id = c.file_id WHERE c.chunk_id = ?", (chunk_id,)).fetchone()
        return tuple(row) if row else None

    return lookup


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT.name} already exists; it is written once")
    lookup = sqlite_lookup()
    result = {"note": "structural/provenance check only; not semantic support", "checks": list(CHECKS), "arms": {arm: summarize(read_jsonl(path), lookup) for arm, path in SOURCES.items()}}
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["arms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
