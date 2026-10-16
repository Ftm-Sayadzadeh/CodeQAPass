"""Tests for the SWE-QA v4 runner (fake providers only: no network, no model calls)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v4_runner as runner  # noqa: E402
from codecompass.qa.models import NO_EVIDENCE_ANSWER, QAAnswer, QACitation  # noqa: E402
from codecompass.qa.prompt import SYSTEM_PROMPT  # noqa: E402

PROTOCOL = Path(__file__).resolve().parents[1] / "reports/evaluation/swe_qa_v4/PROTOCOL_V4_DRAFT.md"
ALL_REPOS = (*runner.MAIN_REPOS, runner.PILOT_REPO)


def make_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for repo in ALL_REPOS:
        lines = [json.dumps({"question": f"{repo} question {n}", "answer": f"{repo} answer {n}"}) for n in range(1, 49)]
        (directory / f"{repo}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


class FakeLLM:
    def __init__(self, text: str = "answer", finish_reason: str = "stop") -> None:
        self.requests: list = []
        self.text, self.finish_reason = text, finish_reason

    def generate(self, request):
        self.requests.append(request)
        return SimpleNamespace(text=self.text, finish_reason=self.finish_reason, model="fake-model", provider="fake")


def test_commits_match_protocol_table() -> None:
    table = {m.group(1): m.group(2) for m in re.finditer(r"\|\s*(\w+)[^|]*\|\s*`([0-9a-f]{40})`", PROTOCOL.read_text(encoding="utf-8"))}
    for repo, sha in runner.COMMITS.items():
        assert table[repo] == sha


def test_reasoning_effort_guard() -> None:
    with pytest.raises(SystemExit):
        runner.require_frozen_settings({})
    with pytest.raises(SystemExit):
        runner.require_frozen_settings({"CODECOMPASS_LLM_REASONING_EFFORT": "high"})
    runner.require_frozen_settings({"CODECOMPASS_LLM_REASONING_EFFORT": "low"})


def test_embedding_identity_guard_refuses_mixed_provider_artifacts(tmp_path: Path) -> None:
    google = {
        "CODECOMPASS_GEMINI2_EMBEDDING_PROVIDER": "openai_compatible",
        "CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL": "https://google.example/v1",
        "CODECOMPASS_GEMINI2_EMBEDDING_MODEL": "gemini-embedding-2",
        "CODECOMPASS_GEMINI2_EMBEDDING_DIMENSIONS": "3072",
    }
    first = runner.ensure_embedding_run_identity(tmp_path, google)
    assert runner.ensure_embedding_run_identity(tmp_path, google) == first
    assert "google.example" not in (tmp_path / "embedding_identity.json").read_text(encoding="utf-8")
    avalai = dict(google, CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL="https://avalai.example/v1")
    with pytest.raises(SystemExit):
        runner.ensure_embedding_run_identity(tmp_path, avalai)


def test_snapshot_verification_stops_on_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "PATH": __import__("os").environ["PATH"], "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=repo, check=True, env=env)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert runner.verify_snapshot(repo, head) == head
    with pytest.raises(SystemExit):
        runner.verify_snapshot(repo, "0" * 40)


def test_sample_sizes_follow_the_freeze(tmp_path: Path) -> None:
    freeze = runner.load_freeze()
    dataset = make_dataset(tmp_path / "ds")
    assert len(runner.select_rows("pilot", dataset, freeze)) == 15
    assert len(runner.select_rows("main", dataset, freeze)) == 192
    human = runner.select_rows("human", dataset, freeze)
    assert [r["question_id"] for r in human] == freeze["human_review_sample"]["ids_in_order"] and len(human) == 30
    assert len(runner.select_rows("persian", dataset, freeze)) == 32


def test_missing_dataset_file_stops(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.select_rows("pilot", tmp_path / "missing", runner.load_freeze())


def test_status_mapping() -> None:
    assert runner.status_of("text", "stop") == "usable"
    assert runner.status_of("text", "length") == "truncated"
    assert runner.status_of("", "stop") == "unavailable"
    assert runner.status_of("text", None) == "unavailable"


def test_attempt_retries_then_records_failure_without_a_score() -> None:
    calls = []

    def flaky() -> dict:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return {"text": "ok", "finish_reason": "stop", "model": "m"}

    assert runner.attempt(flaky, sleep=lambda _s: None)["status"] == "usable" and len(calls) == 3

    def always_fails() -> dict:
        raise RuntimeError("down")

    failed = runner.attempt(always_fails, sleep=lambda _s: None)
    assert failed["status"] == "unavailable" and failed["answer"] is None and failed["attempts"] == runner.MAX_ATTEMPTS
    assert "score" not in failed


def test_a0_uses_neutral_prompt_and_frozen_settings() -> None:
    llm = FakeLLM()
    row = {"question": "Why?", "repo": "flask"}
    runner.call_a0(llm, row)()
    request = llm.requests[0]
    assert request.temperature == 0.0 and request.max_tokens == 1200
    assert request.system_prompt == runner.v2.A0_NEUTRAL_SYSTEM and "flask" in request.prompt


def test_a1_goes_through_the_service_with_frozen_settings() -> None:
    seen = {}

    class Service:
        def answer(self, request):
            seen["request"] = request
            citation = QACitation("c1", "a.py", "f", "f", 1, 3)
            return QAAnswer(request.question, "ans", (citation,), request.retrieval_method, 0, "m", "p", "stop")

    outcome = runner.attempt(runner.call_a1(Service(), 7, {"question": "q", "repo": "flask"}))
    req = seen["request"]
    assert (req.project_id, req.retrieval_method, req.retrieval_limit, req.max_context_chars, req.temperature, req.max_tokens) == (
        7, "hybrid", 5, 6000, 0.0, 1200)
    assert outcome["status"] == "usable" and outcome["citations"][0]["source_file"] == "a.py"


def test_a1_empty_context_is_recorded_as_no_evidence_not_unavailable() -> None:
    class Service:
        def answer(self, request):
            return QAAnswer(request.question, NO_EVIDENCE_ANSWER, (), "hybrid", 0, None, None, None)

    assert runner.attempt(runner.call_a1(Service(), 1, {"question": "q", "repo": "r"}))["status"] == "no_evidence"


def test_b0_uses_top5_the_product_prompt_builder_and_attaches_no_citations() -> None:
    meta = [{"chunk_id": f"b0-x-{i}", "file": f"f{i}.py", "symbol": f"s{i}", "qualified": f"s{i}", "start_line": 1, "end_line": 2,
             "text": f"def s{i}(): pass"} for i in range(8)]
    matrix = np.eye(8, 4, dtype=np.float32) + 0.01
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    llm = FakeLLM()
    embed = lambda texts: np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # noqa: E731
    outcome = runner.attempt(runner.call_b0(llm, embed, (meta, matrix), {"question": "q", "repo": "x"}))
    request = llm.requests[0]
    assert request.system_prompt == SYSTEM_PROMPT and request.max_tokens == 1200 and request.temperature == 0.0
    assert len(outcome["context_chunk_ids"]) <= 5 and outcome["citations"] == [] and outcome["status"] == "usable"


def test_run_arm_is_append_only_and_retry_goes_to_a_separate_file(tmp_path: Path) -> None:
    rows = [{"question_id": f"flask-{n}", "repo": "flask", "question": "q", "answer": "a"} for n in (1, 2)]
    shas = {"flask": "s"}
    results = iter([{"text": "ok", "finish_reason": "stop", "model": "m"}, None])

    def make_call(row):
        def go():
            item = next(results, None)
            if item is None:
                raise RuntimeError("down")
            return item
        return go

    runner.run_arm("a0", "pilot", rows, tmp_path, False, make_call, shas, sleep=lambda _s: None)
    first = (tmp_path / "pilot_a0.jsonl").read_text(encoding="utf-8")
    assert [json.loads(line)["status"] for line in first.splitlines()] == ["usable", "unavailable"]
    with pytest.raises(SystemExit):
        runner.run_arm("a0", "pilot", rows, tmp_path, False, make_call, shas)
    runner.run_arm("a0", "pilot", rows, tmp_path, True,
                   lambda row: (lambda: {"text": "ok2", "finish_reason": "stop", "model": "m"}), shas, sleep=lambda _s: None)
    retry = [json.loads(line) for line in (tmp_path / "pilot_a0_retry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["question_id"] for r in retry] == ["flask-2"]
    assert (tmp_path / "pilot_a0.jsonl").read_text(encoding="utf-8") == first


def test_dry_run_makes_no_provider_call(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset = make_dataset(tmp_path / "ds")

    def boom(*_a, **_k):
        raise AssertionError("provider must not be created in a dry run")

    monkeypatch.setattr(runner.v2, "llm_from_env", boom)
    assert runner.main(["generate", "--arm", "a0", "--mode", "pilot", "--dry-run", "--dataset-dir", str(dataset)]) == 0
    assert "15 generation calls" in capsys.readouterr().out


def test_generate_without_the_reasoning_effort_stops_before_any_provider(tmp_path: Path, monkeypatch) -> None:
    dataset = make_dataset(tmp_path / "ds")
    monkeypatch.delenv("CODECOMPASS_LLM_REASONING_EFFORT", raising=False)

    def boom(*_a, **_k):
        raise AssertionError("provider must not be created")

    monkeypatch.setattr(runner.v2, "llm_from_env", boom)
    with pytest.raises(SystemExit):
        runner.main(["generate", "--arm", "a0", "--mode", "pilot", "--dataset-dir", str(dataset)])


def test_b0_index_refuses_repositories_over_the_chunk_cap(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("def f():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(runner, "repo_root", lambda _repo: root)
    monkeypatch.setattr(runner, "verify_snapshot", lambda *_a: "sha")
    monkeypatch.setattr(runner.b0lib, "chunk_source", lambda *_a: [{"text": "x"}] * (runner.CHUNK_CAP_PER_REPO + 1))

    def embed(_texts):
        raise AssertionError("embedding must not run over the cap")

    with pytest.raises(SystemExit):
        runner.index_b0("flask", embed)


def test_rate_limit_backoff_retries_only_on_429() -> None:
    waits: list[float] = []
    calls = {"n": 0}

    def embed(_texts):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("request failed with HTTP 429")
        return "vectors"

    assert runner.embed_with_rate_limit_backoff(embed, ["x"], sleep=waits.append) == "vectors" and waits == [30, 60]
    with pytest.raises(RuntimeError):
        runner.embed_with_rate_limit_backoff(lambda _t: (_ for _ in ()).throw(RuntimeError("bad key")), ["x"], sleep=waits.append)


def test_b0_index_resumes_from_stored_batches_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text("def f():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(runner, "repo_root", lambda _repo: root)
    monkeypatch.setattr(runner, "verify_snapshot", lambda *_a: "sha")
    artifact_root = tmp_path / "data"
    chunks = [{"file": "m.py", "symbol": f"s{i}", "qualified": f"s{i}", "start_line": 1, "end_line": 2, "text": f"t{i}", "kind": "function"}
              for i in range(70)]
    monkeypatch.setattr(runner.b0lib, "chunk_source", lambda *_a: [dict(c) for c in chunks])
    calls = []

    def failing_embed(texts):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("HTTP 429")
        return np.ones((len(texts), 4), dtype=np.float32)

    sleeps: list[float] = []
    with pytest.raises(RuntimeError):
        runner.index_b0("flask", failing_embed, delay_seconds=1.0, sleep=sleeps.append, artifact_root=artifact_root)
    assert (tmp_path / "data" / "b0" / "_partial_flask" / "batch_00000.npy").exists()
    assert not (tmp_path / "data" / "b0" / "flask_vectors.npy").exists()

    calls.clear()
    count = runner.index_b0("flask", lambda texts: np.ones((len(texts), 4), dtype=np.float32), delay_seconds=1.0,
                            sleep=sleeps.append, artifact_root=artifact_root)
    assert count == 70
    assert (tmp_path / "data" / "b0" / "flask_vectors.npy").exists()
    assert not (tmp_path / "data" / "b0" / "_partial_flask").exists()
