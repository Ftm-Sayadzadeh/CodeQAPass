"""Tests for the SWE-QA v6 runner helpers (no network, no model calls, no index access)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v6_confirm as confirm  # noqa: E402
from codecompass.qa.prompt import SYSTEM_PROMPT  # noqa: E402


def test_arm_tables_are_consistent() -> None:
    assert set(confirm.CONTEXT_OF) == set(confirm.ARMS)
    assert confirm.PARTIAL_ARMS <= set(confirm.ARMS)
    assert set(confirm.CONTEXT_OF.values()) <= set(confirm.CONTEXT_KEYS)
    # the decomposition pair shares the v6 context and differs only in the prompt
    assert confirm.CONTEXT_OF["a1_v6e"] == confirm.CONTEXT_OF["a1_v6"]
    assert "a1_v6e" not in confirm.PARTIAL_ARMS and "a1_v6" in confirm.PARTIAL_ARMS


def test_prompt_builders_follow_the_arm() -> None:
    empty = confirm.c5.context_from_blocks([], 0)
    strict, _ = confirm.prompt_builder("a1_v5a").build("q", empty)
    partial, _ = confirm.prompt_builder("a1_v6").build("q", empty)
    assert strict == SYSTEM_PROMPT and partial != SYSTEM_PROMPT
    assert confirm.prompt_builder("b0_k5_p").build("q", empty)[0] == partial
    assert confirm.prompt_builder("a1_v6e").build("q", empty)[0] == SYSTEM_PROMPT


def test_unsupported_identifiers_counts_only_names_absent_from_context_and_question() -> None:
    answer = "It uses `load_config()` and `Missing.helper`; see `Question_Term` and `abc`."
    unsupported, total = confirm.unsupported_identifiers(answer, "def load_config(): pass", "What is Question_Term?")
    assert total == 4 and unsupported == 2  # missing, helper; short names and builtins are ignored


def test_hedged_excludes_leading_refusals() -> None:
    assert confirm.is_hedged({"status": "usable", "answer": "X works. There is not enough retrieved evidence for Y."})
    assert not confirm.is_hedged({"status": "usable", "answer": "There is not enough retrieved evidence to answer."})
    assert not confirm.is_hedged({"status": "usable", "answer": "X works."})


def test_compare_is_stratified_and_reports_pairs() -> None:
    units = {(f"{repo}-{n}", arm): {"mean": value + n % 3} for repo in ("a", "b") for n in range(6) for arm, value in (("x", 10.0), ("y", 8.0))}
    rows = [{"question_id": f"{repo}-{n}", "repo": repo} for repo in ("a", "b") for n in range(6)]
    result = confirm.compare(units, "x", "y", rows)
    assert result["pairs"] == 12 and result["mean_difference"] == 2.0 and result["better"] == 12
    assert result["bootstrap_ci95"][0] == result["bootstrap_ci95"][1] == 2.0


def test_pending_rows_include_unavailable_and_never_attempted_questions() -> None:
    rows = [{"question_id": f"q{n}"} for n in range(5)]
    original = [{"question_id": "q0", "status": "usable"}, {"question_id": "q1", "status": "unavailable"},
                {"question_id": "q2", "status": "no_evidence"}, {"question_id": "q3", "status": "truncated"}]
    assert [r["question_id"] for r in confirm.pending_rows(rows, original)] == ["q1", "q3", "q4"]


def test_credit_errors_are_recognised_and_rate_limits_are_not() -> None:
    assert confirm.is_credit_error('HTTP 429 {"message": "Your account credit has been exhausted", "code": "quota_exceeded"}')
    assert confirm.is_credit_error("You have insufficient credit to complete this request")
    assert not confirm.is_credit_error("Rate limit reached for requests. Check your quota and upgrade")
