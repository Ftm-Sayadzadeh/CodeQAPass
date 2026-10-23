"""Reliable embedding and vector-index orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from codecompass.chunker import Chunk
from codecompass.embeddings import EmbeddingIdentity, EmbeddingProvider, EmbeddingProviderError, EmbeddingResult
from codecompass.indexing.models import (
    TruncatedEmbedding,
    VectorIndexingError,
    VectorIndexingResult,
    VectorIndexingStats,
)
from codecompass.storage import SQLiteMetadataStore, StoredChunk
from codecompass.vector_index import VectorIndex, VectorIndexError, VectorRecord


@dataclass(frozen=True, slots=True)
class _EmbeddedChunk:
    chunk: StoredChunk | Chunk
    embedding: EmbeddingResult


@dataclass(frozen=True, slots=True)
class PreparedEmbeddings:
    """Embedded structural chunks that have not changed vector storage."""

    embedded: tuple[_EmbeddedChunk, ...]
    truncated: tuple[TruncatedEmbedding, ...]
    errors: tuple[VectorIndexingError, ...]
    identity: EmbeddingIdentity | None
    retries: int

    @property
    def expected_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.chunk.chunk_id for item in self.embedded))

    def records(self, project_id: int) -> tuple[VectorRecord, ...]:
        return tuple(
            VectorRecord(
                chunk_id=item.chunk.chunk_id,
                vector=item.embedding.vector,
                metadata={
                    "project_id": project_id,
                    "content_hash": item.chunk.content_hash,
                    "embedding_model": item.embedding.model,
                    "dimensions": item.embedding.dimensions,
                },
            )
            for item in self.embedded
        )


class VectorIndexingService:
    """Embed canonical SQLite chunks and verify an exact vector-index mirror."""

    def __init__(
        self,
        store: SQLiteMetadataStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex | None,
        batch_size: int = 32,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
        inter_batch_delay_seconds: float = 0.0,
        embedding_identity: EmbeddingIdentity | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        if getattr(embedding_provider, "truncate", False):
            raise ValueError("Reliable indexing requires provider truncation to be disabled and observable")
        if max_retries < 0 or retry_delay_seconds < 0 or inter_batch_delay_seconds < 0:
            raise ValueError("Retry limits and delay must be non-negative")
        self.store = store
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.inter_batch_delay_seconds = inter_batch_delay_seconds
        self.embedding_identity = embedding_identity
        self._embedding_retries = 0
        self._progress: Callable[[dict[str, int]], None] | None = None

    def prepare_chunks(
        self,
        chunks: Sequence[StoredChunk | Chunk],
        progress: Callable[[dict[str, int]], None] | None = None,
    ) -> PreparedEmbeddings:
        """Embed chunks without changing the active vector collection."""
        self._embedding_retries = 0
        self._progress = progress
        embedded, truncated, errors = self._embed_chunks(chunks)
        target_identity = None
        if self.embedding_identity is not None and embedded:
            dimensions = {item.embedding.dimensions for item in embedded}
            if len(dimensions) != 1:
                errors.append(VectorIndexingError("vector", (), "DimensionMismatch", "Embedding dimensions differ"))
            else:
                target_identity = self.embedding_identity.with_dimensions(dimensions.pop())
        self._progress = None
        return PreparedEmbeddings(
            tuple(embedded),
            tuple(truncated),
            tuple(errors),
            target_identity,
            self._embedding_retries,
        )

    def index_project(self, project_id: int) -> VectorIndexingResult:
        """Replace one project's vectors and verify exact chunk-id equality."""
        if self.vector_index is None:
            raise ValueError("Vector index is required for project activation")
        if self.store.get_project(project_id) is None:
            raise ValueError(f"Unknown project id: {project_id}")
        chunks = self.store.list_chunks(project_id)
        sqlite_ids = tuple(sorted(chunk.chunk_id for chunk in chunks))
        prepared = self.prepare_chunks(chunks)
        embedded = list(prepared.embedded)
        truncated = list(prepared.truncated)
        errors = list(prepared.errors)

        if errors:
            vector_ids, vector_errors = self._current_ids(project_id)
            errors.extend(vector_errors)
            return self._result(project_id, sqlite_ids, vector_ids, embedded, truncated, errors)

        target_identity = prepared.identity
        records = prepared.records(project_id)
        vector_ids: tuple[str, ...] = ()
        try:
            replacement = bool(getattr(self.vector_index, "managed", False))
            if replacement:
                metadata = self.identity_metadata(target_identity) if target_identity is not None else {
                    "codecompass:embedding_schema": 1,
                    "codecompass:embedding_dimensions": 0,
                }
                self.vector_index.replace_collection(records, metadata, sqlite_ids)
            else:
                existing = set(self.vector_index.list_ids(project_id))
                self.vector_index.upsert(records)
                stale = tuple(sorted(existing - set(sqlite_ids)))
                self.vector_index.delete(stale)
            vector_ids = self.vector_index.list_ids(project_id)
        except VectorIndexError as error:
            errors.append(VectorIndexingError("vector", (), type(error).__name__, str(error)))
            vector_ids, list_errors = self._current_ids(project_id)
            errors.extend(list_errors)

        if not errors and set(vector_ids) != set(sqlite_ids):
            difference = tuple(sorted(set(vector_ids) ^ set(sqlite_ids)))
            errors.append(
                VectorIndexingError(
                    "vector",
                    difference,
                    "IncompleteIndex",
                    "SQLite and vector-index chunk ids do not match",
                )
            )
        if not errors and target_identity is not None and not replacement:
            try:
                self.vector_index.set_index_metadata(self.identity_metadata(target_identity))
            except VectorIndexError as error:
                errors.append(VectorIndexingError("vector", (), type(error).__name__, str(error)))
        return self._result(project_id, sqlite_ids, vector_ids, embedded, truncated, errors)

    def identity_metadata(self, identity: EmbeddingIdentity) -> dict[str, str | int]:
        """Return trusted collection metadata for an embedding identity."""
        return {
            "codecompass:index_schema_version": 1,
            "codecompass:embedding_schema": 1,
            "codecompass:embedding_provider": identity.provider,
            "codecompass:embedding_endpoint_sha256": identity.endpoint_sha256,
            "codecompass:embedding_model": identity.model,
            "codecompass:embedding_dimensions": identity.dimensions or 0,
        }

    def _embed_chunks(
        self,
        chunks: Sequence[StoredChunk | Chunk],
    ) -> tuple[list[_EmbeddedChunk], list[TruncatedEmbedding], list[VectorIndexingError]]:
        embedded: list[_EmbeddedChunk] = []
        truncated: list[TruncatedEmbedding] = []
        errors: list[VectorIndexingError] = []
        for batch_number, start in enumerate(range(0, len(chunks), self.batch_size)):
            if batch_number and self.inter_batch_delay_seconds:
                time.sleep(self.inter_batch_delay_seconds)
            batch_embedded, batch_truncated, batch_errors = self._embed_batch(chunks[start : start + self.batch_size])
            embedded.extend(batch_embedded)
            truncated.extend(batch_truncated)
            errors.extend(batch_errors)
            if self._progress is not None:
                self._progress(
                    {
                        "embeddings_completed": len(embedded),
                        "embedding_retries": self._embedding_retries,
                        "compacted_embeddings": len(truncated),
                    }
                )
        return embedded, truncated, errors

    def _embed_batch(
        self,
        chunks: Sequence[StoredChunk | Chunk],
    ) -> tuple[list[_EmbeddedChunk], list[TruncatedEmbedding], list[VectorIndexingError]]:
        if not chunks:
            return [], [], []
        try:
            results = self._embed_with_retry(tuple(chunk.embedding_text for chunk in chunks))
        except EmbeddingProviderError as error:
            if error.error_type == "InputTooLong" and len(chunks) > 1:
                midpoint = len(chunks) // 2
                left = self._embed_batch(chunks[:midpoint])
                right = self._embed_batch(chunks[midpoint:])
                return [*left[0], *right[0]], [*left[1], *right[1]], [*left[2], *right[2]]
            if error.error_type == "InputTooLong":
                return self._embed_oversized(chunks[0])
            return [], [], [self._embedding_error(chunks, error)]
        return [_EmbeddedChunk(chunk, result) for chunk, result in zip(chunks, results, strict=True)], [], []

    def _embed_oversized(
        self,
        chunk: StoredChunk | Chunk,
    ) -> tuple[list[_EmbeddedChunk], list[TruncatedEmbedding], list[VectorIndexingError]]:
        original = chunk.embedding_text
        try:
            embedded_text, embedding = self._embed_compacted_lines(chunk)
        except EmbeddingProviderError as error:
            return [], [], [self._embedding_error((chunk,), error)]
        diagnostic = TruncatedEmbedding(
            chunk_id=chunk.chunk_id,
            relative_path=self._relative_path(chunk),
            qualified_name=self._qualified_name(chunk),
            original_chars=len(original),
            embedded_chars=len(embedded_text),
            strategy="head_tail_lines",
        )
        return [_EmbeddedChunk(chunk, embedding)], [diagnostic], []

    def _embed_compacted_lines(self, chunk: StoredChunk | Chunk) -> tuple[str, EmbeddingResult]:
        source_section = f"\nsource:\n{chunk.code}"
        if not chunk.embedding_text.endswith(source_section):
            raise EmbeddingProviderError(
                "indexing",
                None,
                "InvalidInput",
                "Embedding text does not end with canonical chunk source",
            )
        metadata = chunk.embedding_text[: -len(source_section)]
        lines = chunk.code.splitlines(keepends=True)
        if not lines:
            raise EmbeddingProviderError("indexing", None, "InputTooLong", "Oversized input has no compactable source")

        best: tuple[str, EmbeddingResult] | None = None
        low, high = 1, max(1, len(lines) - 1)
        while low <= high:
            keep = (low + high) // 2
            head_count = (keep + 1) // 2
            tail_count = keep // 2
            tail = lines[-tail_count:] if tail_count else []
            candidate = "".join(
                (
                    metadata,
                    "\nsource:\n",
                    *lines[:head_count],
                    "# ... source omitted from embedding input ...\n",
                    *tail,
                )
            )
            try:
                embedding = self._embed_with_retry((candidate,))[0]
            except EmbeddingProviderError as error:
                if error.error_type != "InputTooLong":
                    raise
                high = keep - 1
            else:
                best = candidate, embedding
                low = keep + 1
        if best is None:
            raise EmbeddingProviderError("indexing", None, "InputTooLong", "Could not compact embedding input")
        return best

    def _embed_with_retry(self, texts: tuple[str, ...]) -> tuple[EmbeddingResult, ...]:
        transient = {
            # urllib transport failures are normalized by the provider layer
            # to this stable name; retry them before declaring the batch failed.
            "ConnectionError",
            "HTTPError",
            "ConnectionAbortedError",
            "ConnectionResetError",
            "RemoteDisconnected",
            "TimeoutError",
            "URLError",
        }
        for attempt in range(self.max_retries + 1):
            try:
                return self.embedding_provider.embed_texts(texts)
            except EmbeddingProviderError as error:
                if error.error_type not in transient or attempt == self.max_retries:
                    raise
                self._embedding_retries += 1
                delay = self.retry_delay_seconds * (2**attempt)
                if delay:
                    time.sleep(delay)
        raise AssertionError("unreachable")

    def _embedding_error(
        self,
        chunks: Sequence[StoredChunk | Chunk],
        error: EmbeddingProviderError,
    ) -> VectorIndexingError:
        return VectorIndexingError(
            "embedding",
            tuple(chunk.chunk_id for chunk in chunks),
            error.error_type,
            error.message,
        )

    def _relative_path(self, chunk: StoredChunk | Chunk) -> str:
        return chunk.relative_path if isinstance(chunk, StoredChunk) else chunk.source_file.relative_path

    def _qualified_name(self, chunk: StoredChunk | Chunk) -> str | None:
        return chunk.qualified_name if isinstance(chunk, StoredChunk) else chunk.symbol.qualified_name

    def _current_ids(self, project_id: int) -> tuple[tuple[str, ...], list[VectorIndexingError]]:
        if self.vector_index is None:
            return (), [VectorIndexingError("vector", (), "VectorIndexUnavailable", "Vector index is unavailable")]
        try:
            return self.vector_index.list_ids(project_id), []
        except VectorIndexError as error:
            return (), [VectorIndexingError("vector", (), type(error).__name__, str(error))]

    def _result(
        self,
        project_id: int,
        sqlite_ids: tuple[str, ...],
        vector_ids: tuple[str, ...],
        embedded: Sequence[_EmbeddedChunk],
        truncated: Sequence[TruncatedEmbedding],
        errors: Sequence[VectorIndexingError],
    ) -> VectorIndexingResult:
        embedding_failures = sum(len(error.chunk_ids) or 1 for error in errors if error.stage == "embedding")
        vector_failures = sum(1 for error in errors if error.stage == "vector")
        complete = (
            not errors
            and len(embedded) == len(sqlite_ids)
            and set(vector_ids) == set(sqlite_ids)
        )
        stats = VectorIndexingStats(
            chunks_expected=len(sqlite_ids),
            embeddings_generated=len(embedded),
            vectors_stored=len(vector_ids),
            truncated_embeddings=len(truncated),
            embedding_retries=self._embedding_retries,
            embedding_failures=embedding_failures,
            vector_failures=vector_failures,
            complete=complete,
        )
        return VectorIndexingResult(
            project_id=project_id,
            stats=stats,
            sqlite_chunk_ids=sqlite_ids,
            vector_chunk_ids=tuple(sorted(vector_ids)),
            truncated=tuple(truncated),
            errors=tuple(errors),
        )
