"""Tests for the SWE-QA v5 offline context gate helpers (no model, network, or index access)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v5_dev_context_gate as gate  # noqa: E402

CLASS_CODE = '''class Provider:
    """Serialises objects."""

    ensure_ascii = True

    def dumps(self, obj):
        """Dump to JSON."""
        return str(obj)

    def response(self, obj):
        """Build a response."""
        return obj

''' + "\n".join(f"    def filler_{n}(self):\n        return {n}" for n in range(300))


def piece(chunk_id: str, code: str, start: int = 10, end: int = 20, kind: str = "method", qualified: str | None = "A.m") -> gate.Piece:
    return gate.Piece(chunk_id, "pkg/mod.py", qualified, qualified.split(".")[-1] if qualified else None, kind, start, end, code, 1)


def test_outline_keeps_signatures_and_relevant_method_with_exact_lines() -> None:
    hub = piece("c1", CLASS_CODE, start=100, end=100 + CLASS_CODE.count("\n"), kind="class_overview", qualified="Provider")
    result = gate.outline(hub, "What does Provider.dumps do with ensure_ascii?")
    assert result is not None
    assert "class Provider:" in result.code and "ensure_ascii = True" in result.code
    assert "def filler_5(self):" in result.code
    assert "return str(obj)" in result.code  # relevant method body included
    assert len(result.code) <= gate.HUB_CHARS
    start, end = result.covered[0]
    assert (start, end) == (105, 107)  # dumps spans source lines 6-8, offset by chunk start 100


def test_outline_is_deterministic_and_rejects_non_class_code() -> None:
    hub = piece("c1", CLASS_CODE, kind="class_overview", qualified="Provider")
    assert gate.outline(hub, "dumps") == gate.outline(hub, "dumps")
    assert gate.outline(piece("c2", "x = = 1"), "dumps") is None
    assert gate.outline(piece("c3", "def f():\n    return 1\n"), "f") is None


def test_pack_skips_oversized_and_already_covered_pieces() -> None:
    big = piece("big", "x" * (gate.BUDGET + 1))
    outline_piece = gate.Piece("hub#outline", "pkg/mod.py", "Provider", "Provider", "class_outline", 1, 500, "outline", 1, ((10, 20),))
    inside = piece("inner", "body", start=12, end=15)
    outside = piece("outer", "body", start=30, end=40)
    chosen, omitted = gate.pack([big, outline_piece, inside, outside])
    assert [p.chunk_id for p in chosen] == ["hub#outline", "outer"]
    assert omitted == 1


def test_focus_matches_returns_empty_when_nothing_named() -> None:
    pieces = [piece("a", "x", qualified="Foo.bar")]
    assert gate.focus_matches("Explain Foo.bar please", pieces) == pieces
    assert gate.focus_matches("something unrelated", pieces) == []


def test_identifiers_extracts_backticked_names_only() -> None:
    found = gate.identifiers("The `DefaultJSONProvider.ensure_ascii` flag and `dumps()` matter, but not plain words like True.")
    assert {"defaultjsonprovider", "ensure_ascii", "dumps"} <= found
    assert "plain" not in found
