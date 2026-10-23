from __future__ import annotations

import json
from pathlib import Path

import pytest

from codecompass.vector_index import ChromaVectorIndex, VectorIndexError, VectorIndexStateError, VectorRecord


def index(tmp_path: Path, name: str = "codecompass_test") -> ChromaVectorIndex:
    vector_index = ChromaVectorIndex(tmp_path / "chroma", name)
    vector_index.initialize()
    return vector_index


def record(chunk_id: str, vector: list[float], content_hash: str = "hash") -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        vector=vector,
        metadata={
            "project_id": 1,
            "content_hash": content_hash,
            "embedding_model": "fake-embed",
        },
    )


def managed_index(tmp_path: Path, name: str = "codecompass-project-1", project_id: int = 1) -> ChromaVectorIndex:
    vector_index = ChromaVectorIndex(tmp_path / "chroma", name, managed=True, project_id=project_id)
    vector_index.initialize()
    return vector_index


def pointer(vector_index: ChromaVectorIndex) -> dict[str, object]:
    return json.loads(vector_index.active_pointer.read_text(encoding="utf-8"))


def write_pointer(vector_index: ChromaVectorIndex, value: dict[str, object]) -> None:
    vector_index.active_pointer.write_text(json.dumps(value), encoding="utf-8")


def test_managed_first_initialization_creates_bound_pointer(tmp_path: Path) -> None:
    first = managed_index(tmp_path)
    state = pointer(first)

    assert state == {
        "schema_version": 1,
        "logical_collection": "codecompass-project-1",
        "active_collection": first._active_name,
        "generation": first.get_index_metadata()["codecompass:generation"],
    }
    assert first.get_index_metadata()["codecompass:project_id"] == 1
    assert managed_index(tmp_path)._active_name == first._active_name


def test_export_vectors_returns_exact_embeddings_and_safe_metadata(tmp_path: Path) -> None:
    vector_index = managed_index(tmp_path)
    vector_index.upsert((record("chunk-a", [1.0, 2.0]), record("chunk-b", [3.0, 4.0], "hash-b")))

    exported = vector_index.get_vectors(("chunk-b", "chunk-a"))

    assert [item.chunk_id for item in exported] == ["chunk-a", "chunk-b"]
    assert exported[0].vector == [1.0, 2.0]
    assert exported[1].metadata["content_hash"] == "hash-b"
    assert set(exported[0].metadata) == {"chunk_id", "project_id", "content_hash", "embedding_model", "dimensions"}


def test_upsert_splits_large_batches_for_chroma_limits(tmp_path: Path) -> None:
    vector_index = index(tmp_path, "large_batch_index")

    class RecordingCollection:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def upsert(self, *, ids, embeddings, metadatas) -> None:
            self.calls.append(len(ids))
            assert len(ids) == len(embeddings) == len(metadatas)

        def get(self, *, include, limit=None):
            return {"metadatas": []}

    collection = RecordingCollection()
    vector_index._collection = collection
    records = tuple(record(f"chunk-{number}", [1.0, 2.0]) for number in range(5001))

    vector_index.upsert(records)

    assert collection.calls == [500] * 10 + [1]


def test_missing_managed_pointer_fails_closed(tmp_path: Path) -> None:
    first = managed_index(tmp_path)
    names = {collection.name for collection in first._client.list_collections()}
    first.active_pointer.unlink()

    reopened = ChromaVectorIndex(tmp_path / "chroma", first.collection_name, managed=True, project_id=1)
    with pytest.raises(VectorIndexStateError, match="pointer is missing"):
        reopened.initialize()

    assert {collection.name for collection in reopened._client.list_collections()} == names
    assert first.collection_name not in names


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"schema_version": 1}),
        json.dumps({
            "schema_version": 2,
            "logical_collection": "codecompass-project-1",
            "active_collection": "unbound_collection",
            "generation": "0" * 32,
        }),
        json.dumps({
            "schema_version": 1,
            "logical_collection": "codecompass-project-1",
            "active_collection": "unbound_collection",
            "generation": "0" * 32,
            "extra": True,
        }),
    ],
)
def test_malformed_managed_pointer_fails_closed(tmp_path: Path, content: str) -> None:
    first = managed_index(tmp_path)
    names = {collection.name for collection in first._client.list_collections()}
    first.active_pointer.write_text(content, encoding="utf-8")

    with pytest.raises(VectorIndexStateError):
        managed_index(tmp_path)

    assert {collection.name for collection in first._client.list_collections()} == names


def test_pointer_with_wrong_logical_collection_is_rejected(tmp_path: Path) -> None:
    first = managed_index(tmp_path)
    state = pointer(first)
    state["logical_collection"] = "codecompass-project-2"
    write_pointer(first, state)

    with pytest.raises(VectorIndexStateError, match="binding is invalid"):
        managed_index(tmp_path)


def test_pointer_to_missing_collection_is_rejected_without_creation(tmp_path: Path) -> None:
    first = managed_index(tmp_path)
    state = pointer(first)
    generation = "0" * 32
    missing = first._physical_name("stage", generation)
    state.update(active_collection=missing, generation=generation)
    write_pointer(first, state)

    with pytest.raises(VectorIndexStateError, match="collection is missing"):
        managed_index(tmp_path)

    assert missing not in {collection.name for collection in first._client.list_collections()}


def test_pointer_cannot_activate_another_projects_collection(tmp_path: Path) -> None:
    first = managed_index(tmp_path, "codecompass-project-1", 1)
    second = managed_index(tmp_path, "codecompass-project-2", 2)
    foreign = pointer(second)
    local = pointer(first)
    local.update(active_collection=foreign["active_collection"], generation=foreign["generation"])
    write_pointer(first, local)

    with pytest.raises(VectorIndexStateError, match="binding is invalid"):
        managed_index(tmp_path, "codecompass-project-1", 1)


def test_pointer_generation_must_match_collection_binding(tmp_path: Path) -> None:
    first = managed_index(tmp_path)
    state = pointer(first)
    state["generation"] = "0" * 32
    write_pointer(first, state)

    with pytest.raises(VectorIndexStateError):
        managed_index(tmp_path)


def test_orphan_collection_does_not_override_valid_pointer(tmp_path: Path) -> None:
    first = managed_index(tmp_path)
    original = first._active_name
    generation = "0" * 32
    orphan_name = first._physical_name("stage", generation)
    first._create_physical_index(orphan_name, generation)

    reopened = managed_index(tmp_path)

    assert reopened._active_name == original
    assert orphan_name in {collection.name for collection in reopened._client.list_collections()}


def test_orphan_staging_cleanup_keeps_only_active_collection(tmp_path: Path) -> None:
    vector_index = managed_index(tmp_path)
    active_name = vector_index._active_name
    generation = "0" * 32
    orphan_name = vector_index._physical_name("stage", generation)
    vector_index._create_physical_index(orphan_name, generation)

    vector_index.cleanup_orphan_staging()

    names = {collection.name for collection in vector_index._client.list_collections()}
    assert active_name in names
    assert orphan_name not in names


def test_create_upsert_get_and_search(tmp_path: Path) -> None:
    vector_index = index(tmp_path)

    vector_index.upsert((record("chunk-a", [1.0, 0.0]), record("chunk-b", [0.0, 1.0]),))
    stored = vector_index.get(("chunk-a",))
    results = vector_index.search([1.0, 0.0], limit=2)

    assert stored[0].chunk_id == "chunk-a"
    assert stored[0].metadata == {
        "chunk_id": "chunk-a",
        "project_id": 1,
        "content_hash": "hash",
        "embedding_model": "fake-embed",
        "dimensions": 2,
    }
    assert results[0].chunk_id == "chunk-a"
    assert results[0].distance <= results[1].distance


def test_chroma_metadata_stays_minimal(tmp_path: Path) -> None:
    vector_index = index(tmp_path)

    vector_index.upsert((record("chunk-a", [1.0, 0.0]),))

    metadata = vector_index.get(("chunk-a",))[0].metadata
    assert set(metadata) == {"chunk_id", "project_id", "content_hash", "embedding_model", "dimensions"}
    assert "code" not in metadata
    assert "relative_path" not in metadata
    assert "symbol" not in metadata
    assert "start_line" not in metadata
    assert "end_line" not in metadata


def test_persistence_after_reopening(tmp_path: Path) -> None:
    first = index(tmp_path)
    first.upsert((record("chunk-a", [1.0, 0.0]),))

    reopened = index(tmp_path)

    assert reopened.get(("chunk-a",))[0].metadata["dimensions"] == 2
    assert reopened.search([1.0, 0.0], limit=1)[0].chunk_id == "chunk-a"


def test_collection_metadata_is_merged_and_persisted(tmp_path: Path) -> None:
    first = index(tmp_path)
    first.set_index_metadata({"codecompass:embedding_provider": "fake"})

    reopened = index(tmp_path)

    assert reopened.get_index_metadata()["codecompass:embedding_provider"] == "fake"


def test_duplicate_upsert_updates_without_duplicate_ids(tmp_path: Path) -> None:
    vector_index = index(tmp_path)
    vector_index.upsert((record("chunk-a", [1.0, 0.0], "old"),))

    vector_index.upsert((record("chunk-a", [0.0, 1.0], "new"),))

    stored = vector_index.get(("chunk-a",))
    assert len(stored) == 1
    assert stored[0].metadata["content_hash"] == "new"
    assert vector_index.search([0.0, 1.0], limit=1)[0].chunk_id == "chunk-a"


def test_delete_removes_vector(tmp_path: Path) -> None:
    vector_index = index(tmp_path)
    vector_index.upsert((record("chunk-a", [1.0, 0.0]),))

    vector_index.delete(("chunk-a",))

    assert vector_index.get(("chunk-a",)) == ()


def test_list_ids_can_be_scoped_by_project(tmp_path: Path) -> None:
    vector_index = index(tmp_path)
    other = record("chunk-b", [0.0, 1.0])
    other = VectorRecord(other.chunk_id, other.vector, {**other.metadata, "project_id": 2})
    vector_index.upsert((record("chunk-a", [1.0, 0.0]), other))

    assert vector_index.list_ids() == ("chunk-a", "chunk-b")
    assert vector_index.list_ids(1) == ("chunk-a",)
    assert vector_index.list_ids(2) == ("chunk-b",)


def test_empty_operations_are_noops(tmp_path: Path) -> None:
    vector_index = index(tmp_path)

    vector_index.upsert(())
    vector_index.delete(())

    assert vector_index.get(()) == ()


@pytest.mark.parametrize("name", ["AB", "bad name", "bad..name", "192.168.1.1", "-bad"])
def test_invalid_collection_name_raises(tmp_path: Path, name: str) -> None:
    with pytest.raises(VectorIndexError):
        index(tmp_path, name)


@pytest.mark.parametrize(
    "bad_record",
    [
        VectorRecord("", [1.0], {}),
        VectorRecord("chunk-a", [], {}),
        VectorRecord("chunk-a", [True], {}),
        VectorRecord("chunk-a", ["nope"], {}),
        VectorRecord("chunk-a", [1.0], {"relative_path": "app.py"}),
        VectorRecord("chunk-a", [1.0], {"project_id": object()}),
    ],
)
def test_invalid_records_raise(tmp_path: Path, bad_record: VectorRecord) -> None:
    with pytest.raises(VectorIndexError):
        index(tmp_path).upsert((bad_record,))


def test_duplicate_ids_in_one_batch_raise(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexError):
        index(tmp_path).upsert((record("chunk-a", [1.0]), record("chunk-a", [2.0]),))


def test_inconsistent_batch_dimensions_raise(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexError):
        index(tmp_path).upsert((record("chunk-a", [1.0]), record("chunk-b", [1.0, 2.0]),))


def test_upsert_dimension_mismatch_raises(tmp_path: Path) -> None:
    vector_index = index(tmp_path)
    vector_index.upsert((record("chunk-a", [1.0, 0.0]),))

    with pytest.raises(VectorIndexError, match="dimension mismatch"):
        vector_index.upsert((record("chunk-b", [1.0, 0.0, 0.0]),))


def test_query_dimension_mismatch_raises(tmp_path: Path) -> None:
    vector_index = index(tmp_path)
    vector_index.upsert((record("chunk-a", [1.0, 0.0]),))

    with pytest.raises(VectorIndexError, match="dimension mismatch"):
        vector_index.search([1.0, 0.0, 0.0], limit=1)


def test_invalid_search_inputs_raise(tmp_path: Path) -> None:
    vector_index = index(tmp_path)
    vector_index.upsert((record("chunk-a", [1.0]),))

    with pytest.raises(VectorIndexError):
        vector_index.search([], limit=1)
    with pytest.raises(VectorIndexError):
        vector_index.search([1.0], limit=0)
