"""Tests for the SWE-QA v7 structural citation check (in-memory data only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v7_citation_structural as check  # noqa: E402

CANON = {("flask", "c1"): ("src/a.py", 10, 40)}


def lookup(repo: str, chunk_id: str):
    return CANON.get((repo, chunk_id))


def cite(**overrides) -> dict:
    return {"chunk_id": "c1", "parent_chunk_id": "c1", "source_file": "src/a.py", "start_line": 10, "end_line": 40, **overrides}


def test_check_citation_accepts_exact_and_excerpt_ranges_and_rejects_bad_ones() -> None:
    canonical = ("src/a.py", 10, 40)
    assert check.check_citation(cite(), canonical, {"c1"}) == {
        "schema_ok": True, "lines_positive_ordered": True, "chunk_in_sqlite": True, "file_matches_sqlite": True,
        "range_inside_sqlite_range": True, "range_equals_sqlite_range": True, "chunk_in_context": True}
    excerpt = check.check_citation(cite(start_line=12, end_line=20), canonical, None)
    assert excerpt["range_inside_sqlite_range"] and not excerpt["range_equals_sqlite_range"] and excerpt["chunk_in_context"] is None
    assert not check.check_citation(cite(start_line=30, end_line=5), canonical, None)["lines_positive_ordered"]
    assert not check.check_citation(cite(end_line=99), canonical, None)["range_inside_sqlite_range"]
    assert not check.check_citation(cite(source_file="other.py"), canonical, None)["file_matches_sqlite"]
    assert not check.check_citation(cite(), None, None)["chunk_in_sqlite"]
    assert check.check_citation(cite(), canonical, {"zzz"})["chunk_in_context"] is False


def test_summarize_counts_usable_answers_only_and_flags_invalid_answers() -> None:
    good = {"status": "usable", "repo": "flask", "citations": [cite()], "context_chunk_ids": ["c1"]}
    bad = {"status": "usable", "repo": "flask", "citations": [cite(chunk_id="missing", parent_chunk_id="missing")], "context_chunk_ids": ["c1"]}
    no_evidence = {"status": "no_evidence", "repo": "flask", "citations": []}
    summary = check.summarize([good, bad, no_evidence], lookup)
    assert summary["usable_answers"] == 2 and summary["citations"] == 2 and summary["answers_with_citations"] == 2
    assert summary["answers_all_citations_valid"] == 1
    assert summary["citations_passing"]["chunk_in_sqlite"] == 1 and summary["chunk_in_context_checked"] == 2


def test_parent_id_falls_back_to_the_chunk_id_prefix() -> None:
    assert check.parent_id({"chunk_id": "abc#L3-9"}) == "abc"
    assert check.parent_id({"chunk_id": "abc#L3-9", "parent_chunk_id": "abc"}) == "abc"
