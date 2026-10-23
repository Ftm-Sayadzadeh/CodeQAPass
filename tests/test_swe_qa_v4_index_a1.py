"""Tests for the cached A1 indexing driver (fake providers only: no network, no model calls)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import swe_qa_v4_index_a1 as driver  # noqa: E402
from codecompass.embeddings.base import EmbeddingProviderError, EmbeddingResult  # noqa: E402

DIM = 4


def vector_for(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i] + 1) / 300.0 for i in range(DIM)]


class FakeProvider:
    """Deterministic provider; counts calls and records the texts it was asked to embed."""

    def __init__(self, fail_on_call: int | None = None, bad: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call
        self.bad = bad
        self.model = "fake-model"
        self.truncate = False

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise EmbeddingProviderError("fake", "fake-model", "HTTPError", "HTTP 429")
        if self.bad == "short":
            texts = texts[:-1]
        results = []
        for text in texts:
            vector = vector_for(text)
            if self.bad == "nan":
                vector[0] = float("nan")
            if self.bad == "dim":
                vector = vector[:-1]
            results.append(EmbeddingResult(vector=vector, model="fake-model", dimensions=len(vector)))
        return tuple(results)


def wrap(inner, cache: Path, pace: float = 0.0, sleep=lambda _s: None, clock=lambda: 0.0, namespace: str = "ns1"):
    return driver.CachingEmbeddingProvider(inner, cache, namespace, DIM, pace, sleep, clock)


def test_cache_miss_calls_once_and_hit_calls_never(tmp_path: Path) -> None:
    inner = FakeProvider()
    provider = wrap(inner, tmp_path)
    first = provider.embed_texts(["a", "b", "c"])
    assert len(inner.calls) == 1 and inner.calls[0] == ["a", "b", "c"]
    again = provider.embed_texts(["a", "b", "c"])
    assert len(inner.calls) == 1 and again == first
    reloaded = wrap(FakeProvider(), tmp_path)
    assert reloaded.embed_texts(["a", "b", "c"]) == first and reloaded.stats["inner_calls"] == 0
    assert reloaded.stats["cached_vectors_at_start"] == 3


def test_partial_hit_sends_only_missing_texts_and_keeps_order(tmp_path: Path) -> None:
    inner = FakeProvider()
    provider = wrap(inner, tmp_path)
    provider.embed_texts(["a", "c"])
    result = provider.embed_texts(["a", "b", "c", "d"])
    assert inner.calls[1] == ["b", "d"]
    assert [r.vector for r in result] == [vector_for(t) for t in ["a", "b", "c", "d"]]


def test_cached_output_is_identical_to_direct_output(tmp_path: Path) -> None:
    texts = ["x", "y", "z"]
    direct = FakeProvider().embed_texts(texts)
    cached = wrap(FakeProvider(), tmp_path).embed_texts(texts)
    reloaded = wrap(FakeProvider(), tmp_path).embed_texts(texts)
    assert direct == cached == reloaded


def test_interrupted_run_resends_only_the_remaining_batches(tmp_path: Path) -> None:
    batches = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    inner = FakeProvider(fail_on_call=2)
    provider = wrap(inner, tmp_path)
    provider.embed_texts(batches[0])
    with pytest.raises(EmbeddingProviderError):
        provider.embed_texts(batches[1])
    assert provider.stats["failed_inner_calls"] == 1

    resumed_inner = FakeProvider()
    resumed = wrap(resumed_inner, tmp_path)
    for batch in batches:
        resumed.embed_texts(batch)
    assert resumed_inner.calls == [["b1", "b2"], ["c1", "c2"]]


@pytest.mark.parametrize("mode", ["short", "nan", "dim"])
def test_invalid_response_is_not_stored(tmp_path: Path, mode: str) -> None:
    provider = wrap(FakeProvider(bad=mode), tmp_path)
    with pytest.raises(EmbeddingProviderError):
        provider.embed_texts(["a", "b"])
    assert not list((tmp_path / "ns1").glob("batch_*.npz"))
    assert wrap(FakeProvider(), tmp_path).stats["cached_vectors_at_start"] == 0


def test_namespaces_are_separate(tmp_path: Path) -> None:
    first = FakeProvider()
    wrap(first, tmp_path, namespace="model-a").embed_texts(["a"])
    second = FakeProvider()
    wrap(second, tmp_path, namespace="model-b").embed_texts(["a"])
    assert len(second.calls) == 1
    assert driver.namespace_hash("openai_compatible", "https://h/v1", "m1", 4) != driver.namespace_hash("openai_compatible", "https://h/v1", "m2", 4)
    assert driver.namespace_hash("openai_compatible", "https://h/v1", "m1", 4) != driver.namespace_hash("openai_compatible", "https://h/v1", "m1", 8)
    assert driver.namespace_hash("openai_compatible", "https://h/v1", "m1", 4) != driver.namespace_hash("openai_compatible", "https://other/v1", "m1", 4)


def test_pacing_applies_to_real_calls_only(tmp_path: Path) -> None:
    sleeps: list[float] = []
    provider = wrap(FakeProvider(), tmp_path, pace=2.0, sleep=sleeps.append, clock=lambda: 0.0)
    provider.embed_texts(["a"])
    assert sleeps == []
    provider.embed_texts(["b"])
    assert sleeps == [2.0]
    provider.embed_texts(["a", "b"])
    assert sleeps == [2.0]


def test_duplicate_texts_in_one_batch_are_requested_once(tmp_path: Path) -> None:
    inner = FakeProvider()
    result = wrap(inner, tmp_path).embed_texts(["a", "a", "b"])
    assert inner.calls == [["a", "b"]] and result[0] == result[1]


def test_official_cli_runs_through_the_cache_and_reuses_vectors(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "PATH": __import__("os").environ["PATH"], "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=repo, check=True, env=env)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    inner = FakeProvider()
    monkeypatch.setattr(driver, "real_factory", lambda _config: inner)
    monkeypatch.setenv("CODECOMPASS_API_KEY", "test-key")

    def argv(name: str) -> list[str]:
        return ["--repository", str(repo), "--expected-commit", head, "--database", str(tmp_path / f"{name}.sqlite"),
                "--chroma", str(tmp_path / f"{name}_chroma"), "--collection", f"c-{name}", "--project-name", "demo",
                "--provider", "openai_compatible", "--base-url", "https://example.invalid/v1", "--embedding-model", "fake-model",
                "--embedding-dimensions", str(DIM), "--embedding-delay-seconds", "0"]

    cache = tmp_path / "cache"
    code, wrappers = driver.run(argv("one"), cache, 0.0)
    assert code == 0 and len(inner.calls) >= 1
    calls_after_first = len(inner.calls)
    assert wrappers[0].stats["cached_vectors_at_start"] == 0

    code, wrappers = driver.run(argv("two"), cache, 0.0)
    assert code == 0
    assert len(inner.calls) == calls_after_first  # fresh index, same texts: every vector came from the cache
    assert wrappers[0].stats["inner_calls"] == 0 and wrappers[0].stats["hits"] > 0


def test_argv_uses_the_frozen_commit_and_collection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(driver.v4, "DATA", tmp_path)
    artifacts = tmp_path / "google"
    argv = driver.build_argv("conan", {"CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL": "https://h/v1"}, 2.0, artifacts, 8)
    assert argv[argv.index("--expected-commit") + 1] == driver.v4.COMMITS["conan"]
    assert argv[argv.index("--collection") + 1] == "swe-qa-v4-conan"
    assert Path(argv[argv.index("--database") + 1]).parent == artifacts / "index"
    assert argv[argv.index("--batch-size") + 1] == "8"
    assert argv[argv.index("--embedding-dimensions") + 1] == "3072" and argv[argv.index("--embedding-delay-seconds") + 1] == "0"
