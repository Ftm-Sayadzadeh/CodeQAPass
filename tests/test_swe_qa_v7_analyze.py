"""Tests for the SWE-QA v7 analysis helpers (synthetic data only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v7_analyze as an  # noqa: E402


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean() -> None:
    groups = {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 4.0, 5.0]}
    low, high = an.bootstrap_interval(groups, resamples=500)
    assert (low, high) == tuple(an.bootstrap_interval(groups, resamples=500)) and low < 3.0 < high
    wide = an.bootstrap_interval(groups, level=0.975, resamples=500)
    assert wide[0] <= low and wide[1] >= high


def test_exact_tests_and_wilson_interval() -> None:
    assert an.sign_test_p([1, 1, 1, 1, 1, -1]) == pytest.approx(0.21875)
    assert an.sign_test_p([0, 0]) is None
    assert an.mcnemar_exact_p(5, 0) == pytest.approx(0.0625) and an.mcnemar_exact_p(0, 0) is None
    low, high = an.wilson(8, 10)
    assert 0.49 < low < 0.5 and 0.94 < high < 0.97 and an.wilson(0, 0) is None


def test_percentile_interpolates() -> None:
    assert an.percentile([1, 2, 3, 4], 0.5) == 2.5 and an.percentile([], 0.5) is None
    assert an.percentile(list(range(101)), 0.95) == 95


def test_paired_summary_and_decision_rules() -> None:
    pairs = [(f"r-{i}", "r", 12.0 + (i % 3), 10.0) for i in range(30)]
    summary = an.paired_summary(pairs)
    assert summary["wins"] == 30 and summary["mean_difference"] == pytest.approx(3.0) and summary["ci95"][0] > 0
    summary["holm_adjusted_p"] = 0.001
    assert an.decide(summary)["superiority_supported"] is True
    assert an.decide(summary, 1.5)["non_inferior"] is True
    summary["holm_adjusted_p"] = 0.2
    assert an.decide(summary)["superiority_supported"] is False
    flat = an.paired_summary([(f"r-{i}", "r", 10.0, 11.0 if i % 2 else 9.0) for i in range(20)])
    flat["holm_adjusted_p"] = 0.9
    assert an.decide(flat, 1.5)["non_inferior"] is True and an.decide(flat)["superiority_supported"] is False


def test_holm_adjustment_orders_p_values() -> None:
    adjusted = an.holm({"H1": 0.01, "H2": 0.04})
    assert adjusted["H1"] == pytest.approx(0.02) and adjusted["H2"] == pytest.approx(0.04)


def test_list_cost_separates_gross_and_cache_adjusted_and_respects_missing_usage() -> None:
    prices = {"pricing": {"glm-5.3-flash": {"input": 0.15, "cached_input": 0.03, "output": 0.5}, "gemini-embedding-2": {"input": 0.2}}}
    usage = {"usage_available": True, "prompt_tokens": 1000, "completion_tokens": 200, "cached_tokens": 400}
    embedding = {"usage_available": True, "prompt_tokens": 50}
    gross = an.list_cost(usage, embedding, prices, False)
    adjusted = an.list_cost(usage, embedding, prices, True)
    assert gross == pytest.approx((1000 * 0.15 + 200 * 0.5 + 50 * 0.2) / 1e6)
    assert adjusted == pytest.approx((600 * 0.15 + 400 * 0.03 + 200 * 0.5 + 50 * 0.2) / 1e6) and adjusted < gross
    assert an.list_cost({"usage_available": False}, embedding, prices, False) is None


def test_unsupported_share_counts_names_absent_from_context_and_question() -> None:
    unsupported, total = an.unsupported_share("Uses `load_config()` and `Missing.helper` and `Question_Term`.", "def load_config(): pass", "What is Question_Term?")
    assert (unsupported, total) == (2, 4)


def test_hedged_excludes_leading_refusals() -> None:
    assert an.is_hedged({"status": "usable", "answer": "X works. There is not enough retrieved evidence for Y."})
    assert not an.is_hedged({"status": "usable", "answer": "There is not enough retrieved evidence."})
