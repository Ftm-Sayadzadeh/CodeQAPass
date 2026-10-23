"""Tests for the SWE-QA v5 confirmation runner helpers (no network, no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v5_confirm as confirm  # noqa: E402

FREEZE = {"pilot_sample": {"question_ids": ["conan-20", "conan-7", "conan-1"]}}


def test_t1_ids_exclude_pilot_questions_and_keep_numeric_order() -> None:
    ids = confirm.t1_ids(FREEZE)
    assert len(ids) == 45 and "conan-20" not in ids and "conan-7" not in ids and "conan-1" not in ids
    assert ids[0] == "conan-2" and ids[-1] == "conan-48"


def test_real_freeze_gives_33_t1_questions() -> None:
    assert len(confirm.t1_ids(confirm.v4.load_freeze())) == 33


@pytest.mark.parametrize("record,expected", [
    ({"status": "no_evidence", "answer": "Not enough retrieved evidence to answer."}, True),
    ({"status": "usable", "answer": "There is not enough retrieved evidence to fully answer this."}, True),
    ({"status": "usable", "answer": "Not enough retrieved evidence for X, but Y is shown."}, True),
    ({"status": "usable", "answer": "The flag `x` is set. There is not enough retrieved evidence for Y."}, False),
    ({"status": "usable", "answer": None}, False),
])
def test_refusal_rule_only_counts_leading_refusals(record: dict, expected: bool) -> None:
    assert confirm.is_refusal(record) is expected


def test_b0_arm_context_respects_budget_and_k() -> None:
    meta = [{"file": f"f{i}.py", "symbol": f"s{i}", "qualified": f"s{i}", "start_line": 1, "end_line": 2, "text": "x" * size}
            for i, size in enumerate([4000, 3000, 1000, 500, 200, 100])]
    ranking = [{"chunk_id": f"b0-conan-{i}"} for i in range(6)]
    blocks5, omitted5 = confirm.b0_arm_context({}, ranking, meta, 5)
    assert [b["chunk_id"] for b in blocks5] == ["b0-conan-0", "b0-conan-2", "b0-conan-3", "b0-conan-4"] and omitted5 == 1
    blocks10, _ = confirm.b0_arm_context({}, ranking, meta, 10)
    assert blocks10[-1]["chunk_id"] == "b0-conan-5"


def test_context_from_blocks_keeps_metadata_identity() -> None:
    blocks = [{"chunk_id": "c1", "source_file": "a.py", "symbol_name": "f", "qualified_name": "A.f", "start_line": 3, "end_line": 9, "code": "pass"}]
    context = confirm.context_from_blocks(blocks, 2)
    assert context.blocks[0].citation.start_line == 3 and context.omitted_count == 2 and context.total_chars == 4


def test_spearman_and_compare_handle_ties_and_missing_pairs() -> None:
    assert confirm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert confirm.spearman([1, 2], [1, 2]) is None
    units = {(q, arm): {"mean": value} for q, (left, right) in {"q1": (5.0, 4.0), "q2": (6.0, 5.0), "q3": (7.0, 7.0)}.items()
             for arm, value in (("x", left), ("y", right))}
    result = confirm.compare(units, "x", "y", ["q1", "q2", "q3", "q4"])
    assert result["pairs"] == 3 and result["better"] == 2 and result["worse"] == 0
    assert confirm.compare(units, "x", "y", ["q4"])["status"] == "insufficient"


def test_append_new_refuses_to_overwrite_non_empty_files(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    with confirm.append_new(path) as handle:
        handle.write("{}\n")
    with pytest.raises(SystemExit):
        confirm.append_new(path)
