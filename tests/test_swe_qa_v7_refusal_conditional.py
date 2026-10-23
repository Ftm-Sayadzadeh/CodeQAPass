"""Tests for the SWE-QA v7 refusal-conditional analysis helpers (in-memory data only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v7_refusal_conditional as cond  # noqa: E402

ROWS = {"q1": {"repo": "flask"}, "q2": {"repo": "flask"}, "q3": {"repo": "pytest"}}
SCORES = {("q1", "a"): 15.0, ("q2", "a"): 4.0, ("q3", "a"): 12.0, ("q1", "b"): 10.0, ("q2", "b"): 11.0, ("q3", "b"): 4.0}
REFUSED = {"a": {"q2"}, "b": {"q3"}}


def test_split_scores_separates_answered_and_refused() -> None:
    assert cond.split_scores(SCORES, REFUSED, "a") == ([15.0, 12.0], [4.0])
    assert cond.split_scores(SCORES, REFUSED, "b") == ([10.0, 11.0], [4.0])


def test_paired_answered_keeps_only_questions_neither_arm_refused() -> None:
    assert cond.paired_answered(SCORES, REFUSED, ROWS, "a", "b") == [("q1", "flask", 15.0, 10.0)]


def test_describe_handles_empty_lists() -> None:
    assert cond.describe([]) == {"n": 0, "mean": None}
    assert cond.describe([1.0, 2.0]) == {"n": 2, "mean": 1.5}
