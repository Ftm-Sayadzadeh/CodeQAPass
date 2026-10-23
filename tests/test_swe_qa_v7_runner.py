"""Tests for the SWE-QA v7 runner (fake providers only: no network, no model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v7_runner as runner  # noqa: E402


class FakeLLM:
    """Provider-shaped fake: `generate` goes through `_post_json`, like the real provider."""

    def __init__(self, usage: dict | None, cost: dict | None = None, fail: str | None = None) -> None:
        self.usage, self.cost, self.fail = usage, cost, fail

    def _post_json(self, payload: dict) -> dict:
        if self.fail:
            raise RuntimeError(self.fail)
        return {"usage": self.usage, "estimated_cost": self.cost, "model": "fake"}

    def generate(self, request):
        self._post_json({})
        return SimpleNamespace(text='{"correctness": 10, "completeness": 10, "relevance": 10, "clarity": 10, "reasoning": 10}', model="fake", finish_reason="stop")


USAGE = {"prompt_tokens": 100, "completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 5}, "prompt_tokens_details": {"cached_tokens": 20}}


def test_recorder_captures_usage_per_thread_and_usage_of_sums_it() -> None:
    llm = FakeLLM(USAGE, {"unit": "0.0004"})
    recorder = runner.Recorder(llm)
    llm.generate(None)
    llm.generate(None)
    summary = runner.usage_of(recorder.take())
    assert summary["usage_available"] and summary["http_calls"] == 2
    assert (summary["prompt_tokens"], summary["completion_tokens"], summary["reasoning_tokens"], summary["cached_tokens"]) == (200, 80, 10, 40)
    assert summary["provider_cost_unit"] == pytest.approx(0.0008)
    assert runner.usage_of(recorder.take())["usage_available"] is False  # nothing recorded since the last take


def test_missing_usage_is_unavailable_never_zero() -> None:
    llm = FakeLLM(None, None)
    recorder = runner.Recorder(llm)
    llm.generate(None)
    summary = runner.usage_of(recorder.take())
    assert summary["usage_available"] is False and summary["prompt_tokens"] is None and summary["provider_cost_unit"] is None


def test_judge_one_scores_and_records_usage_and_stops_on_credit_errors() -> None:
    row = {"blind_id": "N0001", "question": "q", "reference_answer": "r", "candidate": "c"}
    good = FakeLLM(USAGE, {"unit": "0.001"})
    result = runner.judge_one(good, runner.Recorder(good), row, 1)
    assert result["status"] == "usable" and result["mean"] == 10 and result["usage"]["provider_cost_unit"] == 0.001
    broke = FakeLLM(None, None, fail="HTTP 429 Your account credit has been exhausted")
    failed = runner.judge_one(broke, runner.Recorder(broke), row, 1)
    assert failed["status"] == "unavailable" and failed["error_type"] == "CreditExhausted" and failed["attempts"] == 1


def test_credit_errors_are_not_confused_with_rate_limits() -> None:
    assert runner.is_credit_error("quota_exceeded") and runner.is_credit_error("You have insufficient credit")
    assert not runner.is_credit_error("Rate limit reached for requests")


def test_study1_call_builds_the_record_fields(monkeypatch) -> None:
    llm = FakeLLM(USAGE, {"unit": "0.0003"})
    recorder = runner.Recorder(llm)
    block = {"chunk_id": "c1#L3-9", "source_file": "a.py", "symbol_name": "f", "qualified_name": "A.f", "start_line": 3, "end_line": 9, "code": "def f(): pass"}
    context = {"question_id": "q-1", "arm": "a1_v5a", "blocks": [block], "omitted_context_count": 2, "context_sha256": "abc"}
    out = runner.study1_call("a1_v5a", llm, recorder, {"question": "What does f do?"}, context)()
    assert out["context_chunk_ids"] == ["c1#L3-9"] and out["context_sha256"] == "abc" and out["omitted_context_count"] == 2 and out["arm_context"] == "a1_v5a"
    assert out["citations"][0]["parent_chunk_id"] == "c1" and len(out["prompt_sha256"]) == 64 and out["usage"]["provider_cost_unit"] == 0.0003
    empty = runner.study1_call("a1_v5a", llm, recorder, {"question": "q"}, {**context, "blocks": []})()
    assert empty["no_evidence"] is True


def test_strip_citations_uses_the_v4_expression_only() -> None:
    assert runner.strip_citations("See pkg/mod.py:12-34 and x.py:7; keep `code`.") == "See  and ; keep `code`."
    assert runner.strip_citations(None) == ""


def test_second_pass_sample_is_stratified_and_reproducible() -> None:
    mapping = {f"N{n:04d}": {"arm": arm, "repo": repo, "question_id": f"{repo}-{i}", "source": "v7"}
               for n, (arm, repo, i) in enumerate(((a, r, i) for a in runner.ALL_ARMS for r in runner.REPOS for i in range(48)), 1)}
    sample = runner.second_pass_sample(mapping)
    assert sample == runner.second_pass_sample(mapping) and len(sample) == 5 * 4 * 12
    cells = {}
    for blind in sample:
        cells[(mapping[blind]["arm"], mapping[blind]["repo"])] = cells.get((mapping[blind]["arm"], mapping[blind]["repo"]), 0) + 1
    assert set(cells.values()) == {12} and len(cells) == 20


def test_pending_rows_and_separate_retry_files() -> None:
    rows = [{"question_id": f"q{n}"} for n in range(4)]
    original = [{"question_id": "q0", "status": "usable"}, {"question_id": "q1", "status": "unavailable"}, {"question_id": "q2", "status": "no_evidence"}]
    assert [r["question_id"] for r in runner.pending_rows(rows, original)] == ["q1", "q3"]
    assert runner.generation_path("a1_v5a", True) != runner.generation_path("a1_v5a", False)


def test_timing_plan_is_deterministic_balanced_and_shuffled() -> None:
    subset = {repo: [f"{repo}-{n}" for n in range(1, 13)] for repo in runner.REPOS}
    warmup, timed = runner.timing_plan(subset)
    assert (warmup, timed) == runner.timing_plan(subset) and len(timed) == 48 * 5 * 3 and len(warmup) == 15
    assert all(sum(1 for t in timed if t["arm"] == arm) == 48 * 3 for arm in runner.ALL_ARMS)
    first_orders = {tuple(t["arm"] for t in timed if t["question_id"] == q and t["repeat"] == 1) for q in (f"flask-{n}" for n in range(1, 13))}
    assert len(first_orders) > 1  # arm order differs between questions


def test_require_freeze_demands_the_file_and_detects_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "FREEZE", tmp_path / "FREEZE_V7.json")
    monkeypatch.setattr(runner, "ADDENDUM", tmp_path / "FREEZE_V7_ADDENDUM.json")
    with pytest.raises(SystemExit):
        runner.require_freeze()
    (tmp_path / "runner.py").write_text("v1", encoding="utf-8")
    (tmp_path / "protocol.md").write_text("p1", encoding="utf-8")
    monkeypatch.setattr(runner, "RUNNER_PATH", "runner.py")
    (tmp_path / "FREEZE_V7.json").write_text(json.dumps({"sha256": {"runner.py": runner.sha256_file(tmp_path / "runner.py"), "protocol.md": runner.sha256_file(tmp_path / "protocol.md")}}), encoding="utf-8")
    runner.require_freeze()
    (tmp_path / "protocol.md").write_text("p2", encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.require_freeze()
    (tmp_path / "protocol.md").write_text("p1", encoding="utf-8")
    (tmp_path / "runner.py").write_text("v2", encoding="utf-8")  # a runner change needs an addendum
    with pytest.raises(SystemExit):
        runner.require_freeze()
    (tmp_path / "FREEZE_V7_ADDENDUM.json").write_text(json.dumps({"sha256": {"runner.py": runner.sha256_file(tmp_path / "runner.py")}}), encoding="utf-8")
    runner.require_freeze()


def test_a_429_that_the_probe_confirms_as_credit_stops_generation_and_judging(monkeypatch) -> None:
    broke = FakeLLM(None, None, fail="OpenAI-compatible request failed with HTTP 429")
    context = {"question_id": "q-1", "arm": "a1_v5a", "omitted_context_count": 0, "context_sha256": "h",
               "blocks": [{"chunk_id": "c1", "source_file": "a.py", "symbol_name": "f", "qualified_name": "f", "start_line": 1, "end_line": 2, "code": "def f(): pass"}]}
    monkeypatch.setattr(runner, "credit_probe", lambda: True)
    with pytest.raises(runner.CreditStop):
        runner.study1_call("a1_v5a", broke, runner.Recorder(broke), {"question": "q"}, context)()
    row = {"blind_id": "N0001", "question": "q", "reference_answer": "r", "candidate": "c"}
    assert runner.judge_one(broke, runner.Recorder(broke), row, 1)["error_type"] == "CreditExhausted"
    # a 429 that is only a rate limit is an ordinary failure that the retry loop handles
    monkeypatch.setattr(runner, "credit_probe", lambda: False)
    with pytest.raises(RuntimeError):
        runner.study1_call("a1_v5a", broke, runner.Recorder(broke), {"question": "q"}, context)()


def test_credit_stop_passes_through_the_retry_loop() -> None:
    def go() -> dict:
        raise runner.CreditStop("no credit")

    with pytest.raises(runner.CreditStop):
        runner.attempt_rate_limited(go)
