"""Tests for the SWE-QA v6 assembly helpers (no model, network, or index access)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v5_dev_context_gate as gate  # noqa: E402
import swe_qa_v6_assembly as v6  # noqa: E402
from codecompass.qa.prompt import SYSTEM_PROMPT  # noqa: E402

SOURCE = (
    ["    def load(self, path, strict=False):", '        """Load the file."""']
    + [f"        filler_{n} = {n}" for n in range(60)]
    + ["        encoding = guess_encoding(path)", "        return parse(path, encoding)"]
    + [f"        tail_{n} = {n}" for n in range(60)]
)


def piece(chunk_id: str, lines: list[str], kind: str = "method", start: int = 100, qualified: str = "A.load") -> gate.Piece:
    return gate.Piece(chunk_id, "pkg/mod.py", qualified, qualified.split(".")[-1], kind, start, start + len(lines) - 1, "\n".join(lines), 1)


def test_excerpt_is_verbatim_with_exact_line_ranges_and_keeps_the_relevant_lines() -> None:
    original = piece("c1", SOURCE)
    parts = v6.excerpt(original, "How does load pick the encoding via guess_encoding?")
    assert len(parts) == 2  # signature piece plus a window that does not start at the body
    for part in parts:
        expected = "\n".join(SOURCE[part.start_line - 100:part.end_line - 100 + 1])
        assert part.code == expected and part.chunk_id.startswith("c1#L") and v6.parent_chunk_id(part.chunk_id) == "c1"
    assert parts[0].start_line == 100 and "def load" in parts[0].code
    assert any("guess_encoding" in part.code for part in parts)
    assert sum(len(part.code) for part in parts) <= v6.EXCERPT_CHARS + v6.SIGNATURE_CHARS


def test_short_class_and_non_verbatim_chunks_are_never_excerpted() -> None:
    short = piece("s", SOURCE[:5])
    assert v6.excerpt(short, "load") == [short]
    overview = piece("o", SOURCE, kind="class_overview")
    assert v6.excerpt(overview, "load") == [overview]
    mismatched = gate.Piece("m", "pkg/mod.py", "A.load", "load", "method", 100, 105, "\n".join(SOURCE), 1)  # line count disagrees with metadata
    assert v6.excerpt(mismatched, "load") == [mismatched]


def test_v6_pieces_drop_hubs_and_prefer_focus_matches() -> None:
    hub = gate.Piece("hub", "pkg/mod.py", "Big", "Big", "class_overview", 1, 900, "x" * (gate.HUB_CHARS + 1), 1)
    other = piece("other", ["def other():", "    return 1"], kind="function", qualified="other")
    named = piece("named", ["def target():", "    return 2"], kind="function", qualified="target")
    result = v6.v6_pieces("What does target return?", [hub, other, named])
    assert [p.chunk_id for p in result] == ["named", "other"]


def test_v6_keeps_at_most_the_configured_number_of_chunks() -> None:
    many = [piece(f"c{n}", [f"def f{n}():", "    return 1"], kind="function", qualified=f"f{n}") for n in range(12)]
    assert len(v6.v6_pieces("anything", many)) == v6.CHUNKS_KEPT


def test_partial_prompt_changes_only_the_insufficiency_rule() -> None:
    assert v6.INSUFFICIENT_RULE in SYSTEM_PROMPT and v6.INSUFFICIENT_RULE not in v6.PARTIAL_SYSTEM_PROMPT
    assert v6.PARTIAL_RULE in v6.PARTIAL_SYSTEM_PROMPT
    assert v6.PARTIAL_SYSTEM_PROMPT.replace(v6.PARTIAL_RULE, v6.INSUFFICIENT_RULE) == SYSTEM_PROMPT
    assert "Do not invent file paths, symbols, or line numbers." in v6.PARTIAL_SYSTEM_PROMPT
