from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codecompass.embeddings import EmbeddingProviderError, EmbeddingResult, embedding_identity
from codecompass.indexing import IndexingCoordinatorError, RepositoryIndexCoordinator
from codecompass.indexing.models import IndexingError, IndexingStats
from codecompass.indexing.service import PreparedRepositoryIndex
from codecompass.indexing import cli as indexing_cli
from codecompass.storage import SQLiteMetadataStore
from codecompass.vector_index import ChromaVectorIndex, VectorIndexError


class FakeEmbeddingProvider:
    truncate = False

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_texts((text,))[0]

    def embed_texts(self, texts) -> tuple[EmbeddingResult, ...]:
        if self.fail:
            raise EmbeddingProviderError("fake", "fake", "TimeoutError", "private provider detail")
        return tuple(EmbeddingResult([float(len(text)), 1.0], "fake", 2) for text in texts)


def repository(tmp_path: Path) -> Path:
    path = tmp_path / "repository"
    path.mkdir()
    (path / "app.py").write_text("def current():\n    return 1\n", encoding="utf-8")
    return path


def cli_args(tmp_path: Path, repo: Path) -> list[str]:
    return [
        "--repository",
        str(repo),
        "--expected-commit",
        "a" * 40,
        "--database",
        str(tmp_path / "metadata.sqlite"),
        "--chroma",
        str(tmp_path / "chroma"),
        "--collection",
        "cli-safe-index",
        "--embedding-model",
        "fake",
    ]


def managed_index(tmp_path: Path, project_id: int) -> ChromaVectorIndex:
    return ChromaVectorIndex(
        tmp_path / "chroma",
        "cli-safe-index",
        managed=True,
        project_id=project_id,
    )


def configure_cli(monkeypatch, provider: FakeEmbeddingProvider) -> None:
    monkeypatch.setattr(indexing_cli, "_validate_repository", lambda *_args: "a" * 40)
    monkeypatch.setattr(indexing_cli, "create_embedding_provider", lambda _config: provider)


def test_cli_same_identity_reindex_is_a_true_noop(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = repository(tmp_path)
    configure_cli(monkeypatch, FakeEmbeddingProvider())

    assert indexing_cli.main(cli_args(tmp_path, repo)) == 0
    capsys.readouterr()
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    project = store.get_project_by_root(repo)
    assert project is not None
    first = managed_index(tmp_path, project.id)
    first.list_ids(project.id)
    old_active = first._active_name

    assert indexing_cli.main(cli_args(tmp_path, repo)) == 0
    second = managed_index(tmp_path, project.id)
    assert second.list_ids(project.id)
    assert second._active_name == old_active
    output = capsys.readouterr().out
    assert "\"complete\": true" in output
    assert "\"strategy\": \"incremental\"" in output
    assert "\"no_changes\": true" in output


def test_allowed_structural_errors_are_narrow_and_evaluation_only(tmp_path: Path) -> None:
    coordinator = RepositoryIndexCoordinator(
        SQLiteMetadataStore(tmp_path / "metadata.sqlite"),
        FakeEmbeddingProvider(),
        embedding_identity("fake", "https://example.test/v1", "fake"),
        lambda project_id: managed_index(tmp_path, project_id),
        allowed_structural_error_paths=frozenset({"fixtures/invalid.py"}),
    )
    prepared = PreparedRepositoryIndex(
        tmp_path,
        (),
        (),
        (),
        (),
        IndexingStats(parse_errors=2),
        (
            IndexingError("parse", "fixtures/invalid.py", "SyntaxError", "expected"),
            IndexingError("parse", "unexpected.py", "SyntaxError", "unexpected"),
        ),
    )
    filtered = coordinator._filter_allowed_structural_errors(prepared)
    assert [error.relative_path for error in filtered.errors] == ["unexpected.py"]


def test_cli_embedding_failure_preserves_previous_index(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = repository(tmp_path)
    configure_cli(monkeypatch, FakeEmbeddingProvider())
    assert indexing_cli.main(cli_args(tmp_path, repo)) == 0
    capsys.readouterr()
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    project = store.get_project_by_root(repo)
    assert project is not None
    old_chunks = tuple(chunk.chunk_id for chunk in store.list_chunks(project.id))
    old_vectors = managed_index(tmp_path, project.id).list_ids(project.id)
    (repo / "app.py").write_text("def changed():\n    return 2\n", encoding="utf-8")
    monkeypatch.setattr(indexing_cli, "create_embedding_provider", lambda _config: FakeEmbeddingProvider(fail=True))

    assert indexing_cli.main(cli_args(tmp_path, repo)) == 1

    assert tuple(chunk.chunk_id for chunk in store.list_chunks(project.id)) == old_chunks
    assert managed_index(tmp_path, project.id).list_ids(project.id) == old_vectors
    assert "private provider detail" not in capsys.readouterr().out


def test_cli_activation_failure_preserves_previous_index(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = repository(tmp_path)
    configure_cli(monkeypatch, FakeEmbeddingProvider())
    assert indexing_cli.main(cli_args(tmp_path, repo)) == 0
    capsys.readouterr()
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    project = store.get_project_by_root(repo)
    assert project is not None
    old_chunks = tuple(chunk.chunk_id for chunk in store.list_chunks(project.id))
    old_index = managed_index(tmp_path, project.id)
    old_vectors = old_index.list_ids(project.id)
    old_pointer = old_index.active_pointer.read_text(encoding="utf-8")
    old_names = {item.name for item in old_index._client.list_collections()}
    (repo / "app.py").write_text("def changed():\n    return 3\n", encoding="utf-8")

    def fail_activation(self, replacement) -> None:
        raise VectorIndexError("activation failed")

    monkeypatch.setattr(ChromaVectorIndex, "activate_staged", fail_activation)
    assert indexing_cli.main(cli_args(tmp_path, repo)) == 1

    reopened = managed_index(tmp_path, project.id)
    assert tuple(chunk.chunk_id for chunk in store.list_chunks(project.id)) == old_chunks
    assert reopened.active_pointer.read_text(encoding="utf-8") == old_pointer
    assert reopened.list_ids(project.id) == old_vectors
    assert {item.name for item in reopened._client.list_collections()} == old_names


class CommitFailingConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self) -> None:
        raise sqlite3.OperationalError("simulated commit failure")


def test_pointer_switch_is_compensated_when_sqlite_commit_fails(tmp_path: Path, monkeypatch) -> None:
    repo = repository(tmp_path)
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite")
    identity = embedding_identity("fake", "https://example.test/v1", "fake")
    factory = lambda project_id: ChromaVectorIndex(
        tmp_path / "chroma",
        "coordinator-safe-index",
        managed=True,
        project_id=project_id,
    )
    coordinator = RepositoryIndexCoordinator(store, FakeEmbeddingProvider(), identity, factory)
    initial = coordinator.index_repository(repo)
    old_chunks = tuple(chunk.chunk_id for chunk in store.list_chunks(initial.project_id))
    old_index = factory(initial.project_id)
    old_vectors = old_index.list_ids(initial.project_id)
    old_pointer = old_index.active_pointer.read_text(encoding="utf-8")
    old_names = {item.name for item in old_index._client.list_collections()}
    (repo / "app.py").write_text("def changed():\n    return 4\n", encoding="utf-8")

    switched: list[str] = []
    original_activate = ChromaVectorIndex.activate_staged

    def record_activation(self, replacement) -> None:
        original_activate(self, replacement)
        switched.append(replacement.staging_name)

    monkeypatch.setattr(ChromaVectorIndex, "activate_staged", record_activation)
    original_replace = store.apply_incremental_project_index

    def fail_commit(*args, **kwargs):
        original_connect = store._connect
        monkeypatch.setattr(store, "_connect", lambda: CommitFailingConnection(original_connect()))
        try:
            return original_replace(*args, **kwargs)
        finally:
            monkeypatch.setattr(store, "_connect", original_connect)

    monkeypatch.setattr(store, "apply_incremental_project_index", fail_commit)
    with pytest.raises(IndexingCoordinatorError) as raised:
        coordinator.index_repository(repo)

    reopened = factory(initial.project_id)
    assert raised.value.code == "indexing_failed"
    assert switched
    assert tuple(chunk.chunk_id for chunk in store.list_chunks(initial.project_id)) == old_chunks
    assert reopened.active_pointer.read_text(encoding="utf-8") == old_pointer
    assert reopened.list_ids(initial.project_id) == old_vectors
    assert {item.name for item in reopened._client.list_collections()} == old_names
