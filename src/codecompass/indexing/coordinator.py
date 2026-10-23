"""Safe full and incremental repository indexing shared by every entry point."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from codecompass.embeddings import EmbeddingIdentity, EmbeddingProvider, EmbeddingProviderError, OllamaEmbeddingProvider
from codecompass.indexing.models import IndexingStats, TruncatedEmbedding
from codecompass.indexing.service import IndexingService, PreparedRepositoryIndex
from codecompass.indexing.vectors import PreparedEmbeddings, VectorIndexingService
from codecompass.scanner import RepositoryPathError, ScanResult, SourceFile
from codecompass.storage import ProjectRecord, SQLiteMetadataStore, SourceFileRecord, StorageError, StoredChunk
from codecompass.vector_index import ChromaVectorIndex, StagedVectorReplacement, VectorIndexError, VectorIndexStateError, VectorRecord

INDEX_SCHEMA_VERSION = 1
ProgressCallback = Callable[[str, dict[str, int]], None]
CollectionFactory = Callable[[int], ChromaVectorIndex]
ActivationCallback = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class IndexingFailure:
    """Safe indexing failure metadata suitable for API and CLI adapters."""

    stage: str
    error_type: str


class IndexingCoordinatorError(Exception):
    """A failed indexing run with safe, provider-neutral diagnostics."""

    def __init__(
        self,
        code: str,
        failures: tuple[IndexingFailure, ...],
        *,
        structural_stats: IndexingStats | None = None,
        truncated: tuple[TruncatedEmbedding, ...] = (),
    ) -> None:
        self.code = code
        self.failures = failures
        self.structural_stats = structural_stats
        self.truncated = truncated
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class FileChangePlan:
    """Deterministic file-level delta against canonical SQLite metadata."""

    unchanged: tuple[SourceFile, ...]
    added: tuple[SourceFile, ...]
    modified: tuple[SourceFile, ...]
    deleted_paths: tuple[str, ...]

    @property
    def changed(self) -> tuple[SourceFile, ...]:
        return tuple(sorted((*self.added, *self.modified), key=lambda item: item.relative_path))

    @property
    def no_changes(self) -> bool:
        return not self.added and not self.modified and not self.deleted_paths

    def counters(self) -> dict[str, int]:
        return {
            "files_unchanged": len(self.unchanged),
            "files_added": len(self.added),
            "files_modified": len(self.modified),
            "files_deleted": len(self.deleted_paths),
        }


@dataclass(frozen=True, slots=True)
class _IncrementalBase:
    project: ProjectRecord
    vector_index: ChromaVectorIndex
    chunks: tuple[StoredChunk, ...]
    generation: str
    dimensions: int
    metadata: dict[str, str | int]


@dataclass(frozen=True, slots=True)
class CoordinatedIndexingResult:
    """Verified metadata and vector generation produced by one safe run."""

    project_id: int
    operation: str
    strategy: str
    no_changes: bool
    structural_stats: IndexingStats
    chunks_expected: int
    embeddings_generated: int
    vectors_stored: int
    chunks_reused: int
    vectors_reused: int
    vectors_deleted: int
    file_changes: dict[str, int]
    truncated: tuple[TruncatedEmbedding, ...]
    embedding_retries: int
    largest_embedding_input_chars: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int | None
    expected_ids: tuple[str, ...]
    vector_ids: tuple[str, ...]

    def api_result(self) -> dict[str, object]:
        """Return the backward-compatible API indexing response with optional M24 detail."""
        return {
            "project_id": self.project_id,
            "operation": self.operation,
            "strategy": self.strategy,
            "no_changes": self.no_changes,
            "complete": True,
            "structural_stats": {**asdict(self.structural_stats), **self.file_changes},
            "vector_stats": {
                "chunks_expected": self.chunks_expected,
                "embeddings_generated": self.embeddings_generated,
                "vectors_stored": self.vectors_stored,
                "chunks_reused": self.chunks_reused,
                "vectors_reused": self.vectors_reused,
                "vectors_deleted": self.vectors_deleted,
                "truncated_embeddings": len(self.truncated),
                "embedding_retries": self.embedding_retries,
                "embedding_failures": 0,
                "vector_failures": 0,
                "largest_embedding_input_chars": self.largest_embedding_input_chars,
                "complete": True,
            },
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
            },
        }


class RepositoryIndexCoordinator:
    """Select and run the only supported safe full or incremental workflow."""

    def __init__(
        self,
        store: SQLiteMetadataStore,
        embedding_provider: EmbeddingProvider,
        embedding_identity: EmbeddingIdentity,
        collection_factory: CollectionFactory,
        *,
        batch_size: int = 32,
        inter_batch_delay_seconds: float = 0.0,
        begin_activation: ActivationCallback | None = None,
        end_activation: ActivationCallback | None = None,
        allowed_structural_error_paths: frozenset[str] = frozenset(),
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.embedding_identity = embedding_identity
        self.collection_factory = collection_factory
        self.batch_size = batch_size
        self.inter_batch_delay_seconds = inter_batch_delay_seconds
        self.begin_activation = begin_activation or (lambda _project_id: None)
        self.end_activation = end_activation or (lambda _project_id: None)
        # Evaluation-only exception for repositories that intentionally ship
        # invalid Python fixtures. Unexpected structural errors remain fatal.
        self.allowed_structural_error_paths = allowed_structural_error_paths

    def index_repository(
        self,
        path: Path,
        project_name: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> CoordinatedIndexingResult:
        """Automatically run a full rebuild, incremental update, or true no-op."""
        emit = progress or (lambda _state, _values: None)
        service = IndexingService(self.store)
        scan = self._scan(service, path, emit)
        existing = self.store.get_project_by_root(scan.root_path)
        stored_files = self.store.list_source_files(existing.id) if existing is not None else ()
        plan = plan_file_changes(scan.files, stored_files)
        if existing is not None:
            base = self._incremental_base(existing)
            if base is not None:
                emit("scanning", {"files_discovered": len(scan.files), **plan.counters()})
                return self._run_incremental(service, scan, base, plan, project_name, emit)
        return self._run_full(service, scan, existing, plan, project_name, emit)

    def _scan(self, service: IndexingService, path: Path, emit: ProgressCallback) -> ScanResult:
        try:
            scan = service.scan_repository(path, emit)
        except (RepositoryPathError, StorageError) as error:
            stage = "storage" if isinstance(error, StorageError) else "scan"
            raise IndexingCoordinatorError(
                "indexing_failed",
                (IndexingFailure(stage, type(error).__name__),),
                structural_stats=IndexingStats(scan_errors=int(stage == "scan"), storage_errors=int(stage == "storage")),
            ) from error
        if scan.errors:
            raise IndexingCoordinatorError(
                "indexing_failed",
                tuple(IndexingFailure("scan", error.error_type) for error in scan.errors),
                structural_stats=IndexingStats(files_discovered=len(scan.files), scan_errors=len(scan.errors)),
            )
        return scan

    def _incremental_base(self, project: ProjectRecord) -> _IncrementalBase | None:
        if project.index_schema_version != INDEX_SCHEMA_VERSION or project.vector_generation is None:
            return None
        vector_index = self.collection_factory(project.id)
        try:
            generation = vector_index.active_generation()
            metadata = dict(vector_index.get_index_metadata())
        except VectorIndexError as error:
            raise self._state_error(error) from error
        if generation != project.vector_generation:
            raise self._state_error(VectorIndexStateError("SQLite and Chroma generations do not match"))
        if metadata.get("codecompass:index_schema_version") != INDEX_SCHEMA_VERSION:
            raise self._state_error(VectorIndexStateError("SQLite and Chroma index schemas do not match"))
        dimensions = metadata.get("codecompass:embedding_dimensions")
        if not isinstance(dimensions, int) or dimensions < 0:
            raise self._state_error(VectorIndexStateError("Active embedding dimensions are invalid"))
        stored_identity = (
            metadata.get("codecompass:embedding_provider"),
            metadata.get("codecompass:embedding_endpoint_sha256"),
            metadata.get("codecompass:embedding_model"),
        )
        requested_identity = (
            self.embedding_identity.provider,
            self.embedding_identity.endpoint_sha256,
            self.embedding_identity.model,
        )
        if stored_identity != requested_identity:
            return None
        chunks = self.store.list_chunks(project.id)
        expected_ids = tuple(sorted(chunk.chunk_id for chunk in chunks))
        if expected_ids and dimensions < 1:
            raise self._state_error(VectorIndexStateError("Active embedding dimensions are invalid"))
        if not expected_ids and dimensions != 0:
            raise self._state_error(VectorIndexStateError("Empty vector index dimensions are invalid"))
        if (
            expected_ids
            and self.embedding_identity.dimensions is not None
            and self.embedding_identity.dimensions != dimensions
        ):
            return None
        try:
            vector_ids = vector_index.list_ids()
        except VectorIndexError as error:
            raise self._state_error(error) from error
        if vector_ids != expected_ids:
            return None
        return _IncrementalBase(
            project,
            vector_index,
            chunks,
            generation,
            dimensions,
            {key: value for key, value in metadata.items() if isinstance(value, (str, int))},
        )

    def _run_full(
        self,
        service: IndexingService,
        scan: ScanResult,
        existing: ProjectRecord | None,
        plan: FileChangePlan,
        project_name: str | None,
        emit: ProgressCallback,
    ) -> CoordinatedIndexingResult:
        old_chunk_ids = {chunk.chunk_id for chunk in self.store.list_chunks(existing.id)} if existing is not None else set()
        prepared = service.prepare_files(scan, scan.files, emit)
        prepared = self._filter_allowed_structural_errors(prepared)
        self._require_prepared(prepared)
        if prepared.chunks:
            emit("preflight", {"files_discovered": len(scan.files)})
            preflight_embedding(self.embedding_provider)
        vector_service, embedded, largest_chars = self._embed(prepared, emit)
        project_identity = existing or self.store.upsert_project(project_name or scan.root_path.name, scan.root_path)
        created_identity = existing is None
        vector_index = self.collection_factory(project_identity.id)
        if not vector_index.managed:
            self._cleanup_reserved_project(project_identity.id, vector_index, created_identity)
            raise IndexingCoordinatorError(
                "vector_indexing_failed",
                (IndexingFailure("vector", "VectorIndexError"),),
                structural_stats=prepared.stats,
                truncated=embedded.truncated,
            )
        staged: StagedVectorReplacement | None = None
        activation_project_id: int | None = None

        def activate(project_id: int) -> str:
            nonlocal activation_project_id
            if project_id != project_identity.id or staged is None:
                raise VectorIndexStateError("Prepared candidate project binding changed")
            activation_project_id = project_id
            self.begin_activation(project_id)
            vector_index.activate_staged(staged)
            return staged.generation

        def rollback() -> None:
            if vector_index is not None and staged is not None:
                vector_index.rollback_staged(staged)

        try:
            emit("verifying", {"vectors_stored": 0})
            metadata = vector_service.identity_metadata(embedded.identity or self.embedding_identity)
            if not embedded.expected_ids:
                metadata["codecompass:embedding_dimensions"] = 0
            staged = vector_index.stage_replacement(
                embedded.records(project_identity.id),
                metadata,
                embedded.expected_ids,
            )
            final_scan = self._scan(service, scan.root_path, lambda _state, _values: None)
            if file_snapshot(final_scan.files) != file_snapshot(scan.files):
                vector_index.discard_staged(staged)
                staged = None
                raise IndexingCoordinatorError(
                    "repository_changed_during_index",
                    (IndexingFailure("scan", "RepositoryChanged"),),
                    structural_stats=prepared.stats,
                    truncated=embedded.truncated,
                )
            project = self.store.replace_project_index(
                project_name or scan.root_path.name,
                scan.root_path,
                prepared.files,
                prepared.parse_results,
                prepared.chunks,
                before_commit=activate,
                on_rollback=rollback,
                index_schema_version=INDEX_SCHEMA_VERSION,
            )
            emit("activating", {"vectors_stored": len(embedded.expected_ids)})
            if staged is None:
                raise VectorIndexError("Vector replacement was not prepared")
            vector_index.finalize_staged(staged)
            actual_ids = vector_index.list_ids()
            if actual_ids != embedded.expected_ids or project.vector_generation != vector_index.active_generation():
                raise VectorIndexStateError("Activated metadata and vector generation do not match")
        except Exception as error:
            if activation_project_id is None and staged is not None:
                try:
                    vector_index.discard_staged(staged)
                except VectorIndexError:
                    pass
            self._cleanup_reserved_project(project_identity.id, vector_index, created_identity)
            self._raise_run_error(error, prepared.stats, embedded.truncated)
            raise AssertionError("unreachable")
        finally:
            if activation_project_id is not None:
                self.end_activation(activation_project_id)
        return self._result(
            project,
            "full",
            False,
            prepared.stats,
            embedded,
            actual_ids,
            operation="reindexed" if existing is not None else "indexed",
            file_changes=plan.counters(),
            chunks_reused=0,
            vectors_deleted=len(old_chunk_ids - set(actual_ids)),
            largest_chars=largest_chars,
        )

    def _cleanup_reserved_project(
        self,
        project_id: int,
        vector_index: ChromaVectorIndex,
        created_identity: bool,
    ) -> None:
        if not created_identity:
            return
        try:
            project = self.store.get_project(project_id)
        except StorageError:
            return
        if project is None or project.index_schema_version is not None or project.vector_generation is not None:
            return
        try:
            vector_index.discard_empty_managed_index()
        except VectorIndexError:
            return
        try:
            self.store.delete_empty_project(project_id)
        except StorageError:
            pass

    def _run_incremental(
        self,
        service: IndexingService,
        scan: ScanResult,
        base: _IncrementalBase,
        plan: FileChangePlan,
        project_name: str | None,
        emit: ProgressCallback,
    ) -> CoordinatedIndexingResult:
        if plan.no_changes:
            stats = IndexingStats(files_discovered=len(scan.files))
            ids = tuple(sorted(chunk.chunk_id for chunk in base.chunks))
            return CoordinatedIndexingResult(
                project_id=base.project.id,
                operation="reindexed",
                strategy="incremental",
                no_changes=True,
                structural_stats=stats,
                chunks_expected=len(ids),
                embeddings_generated=0,
                vectors_stored=len(ids),
                chunks_reused=len(ids),
                vectors_reused=len(ids),
                vectors_deleted=0,
                file_changes=plan.counters(),
                truncated=(),
                embedding_retries=0,
                largest_embedding_input_chars=0,
                embedding_provider=self.embedding_identity.provider,
                embedding_model=self.embedding_identity.model,
                embedding_dimensions=base.dimensions or None,
                expected_ids=ids,
                vector_ids=ids,
            )

        if plan.changed:
            prepared = service.prepare_files(scan, plan.changed, emit)
            self._require_prepared(prepared)
            if prepared.chunks:
                emit("preflight", {"files_discovered": len(scan.files), **plan.counters()})
                preflight_embedding(self.embedding_provider)
            vector_service, embedded, largest_chars = self._embed(prepared, emit)
        else:
            prepared = PreparedRepositoryIndex(
                scan.root_path,
                (),
                (),
                (),
                (),
                IndexingStats(files_discovered=len(scan.files)),
                (),
            )
            vector_service = VectorIndexingService(
                self.store,
                self.embedding_provider,
                None,
                batch_size=self.batch_size,
                inter_batch_delay_seconds=self.inter_batch_delay_seconds,
                embedding_identity=self.embedding_identity,
            )
            embedded = PreparedEmbeddings((), (), (), None, 0)
            largest_chars = 0

        changed_paths = {file.relative_path for file in plan.changed}
        deleted_paths = set(plan.deleted_paths)
        retained_chunks = tuple(
            chunk
            for chunk in base.chunks
            if chunk.relative_path not in changed_paths and chunk.relative_path not in deleted_paths
        )
        expected_ids = tuple(sorted((*[chunk.chunk_id for chunk in retained_chunks], *embedded.expected_ids)))
        vectors_deleted = len({chunk.chunk_id for chunk in base.chunks} - set(expected_ids))
        staged: StagedVectorReplacement | None = None
        try:
            reused = self._reuse_vectors(base, retained_chunks)
            new_records = self._new_records(embedded, base.project.id, base.dimensions)
            candidate_records = tuple(sorted((*reused, *new_records), key=lambda record: record.chunk_id))
            target_dimensions = self._target_dimensions(base.dimensions, candidate_records)
            target_identity = self.embedding_identity.with_dimensions(target_dimensions) if target_dimensions else self.embedding_identity
            metadata = vector_service.identity_metadata(target_identity)
            if not candidate_records:
                metadata["codecompass:embedding_dimensions"] = 0
            emit(
                "verifying",
                {
                    **plan.counters(),
                    "chunks_expected": len(expected_ids),
                    "chunks_reused": len(retained_chunks),
                    "vectors_reused": len(reused),
                    "vectors_deleted": vectors_deleted,
                },
            )
            staged = base.vector_index.stage_replacement(candidate_records, metadata, expected_ids)
            final_scan = self._scan(service, scan.root_path, lambda _state, _values: None)
            if file_snapshot(final_scan.files) != file_snapshot(scan.files):
                base.vector_index.discard_staged(staged)
                raise IndexingCoordinatorError(
                    "repository_changed_during_index",
                    (IndexingFailure("scan", "RepositoryChanged"),),
                    structural_stats=prepared.stats,
                    truncated=embedded.truncated,
                )
        except IndexingCoordinatorError:
            if staged is not None:
                try:
                    base.vector_index.discard_staged(staged)
                except VectorIndexError:
                    pass
            raise
        except Exception as error:
            if staged is not None:
                try:
                    base.vector_index.discard_staged(staged)
                except VectorIndexError:
                    pass
            self._raise_run_error(error, prepared.stats, embedded.truncated)
            raise AssertionError("unreachable")

        activation_started = False
        if staged is None:
            raise AssertionError("incremental candidate was not prepared")

        def activate(project_id: int) -> None:
            nonlocal activation_started
            activation_started = True
            self.begin_activation(project_id)
            base.vector_index.activate_staged(staged)

        def rollback() -> None:
            base.vector_index.rollback_staged(staged)

        try:
            project = self.store.apply_incremental_project_index(
                base.project.id,
                project_name or base.project.name,
                plan.changed,
                plan.deleted_paths,
                prepared.parse_results,
                prepared.chunks,
                index_schema_version=INDEX_SCHEMA_VERSION,
                vector_generation=staged.generation,
                expected_chunk_ids=expected_ids,
                before_commit=activate,
                on_rollback=rollback,
            )
            emit("activating", {"vectors_stored": len(expected_ids), **plan.counters()})
            base.vector_index.finalize_staged(staged)
            actual_ids = base.vector_index.list_ids()
            if actual_ids != expected_ids or project.vector_generation != base.vector_index.active_generation():
                raise VectorIndexStateError("Activated metadata and vector generation do not match")
        except Exception as error:
            if not activation_started:
                try:
                    base.vector_index.discard_staged(staged)
                except VectorIndexError:
                    pass
            self._raise_run_error(error, prepared.stats, embedded.truncated)
            raise AssertionError("unreachable")
        finally:
            if activation_started:
                self.end_activation(base.project.id)

        return self._result(
            project,
            "incremental",
            False,
            prepared.stats,
            embedded,
            actual_ids,
            operation="reindexed",
            file_changes=plan.counters(),
            chunks_reused=len(retained_chunks),
            vectors_deleted=vectors_deleted,
            largest_chars=largest_chars,
        )

    def _embed(
        self,
        prepared: PreparedRepositoryIndex,
        emit: ProgressCallback,
    ) -> tuple[VectorIndexingService, PreparedEmbeddings, int]:
        largest_chars = max((len(chunk.embedding_text) for chunk in prepared.chunks), default=0)
        emit(
            "embedding",
            {
                "files_discovered": prepared.stats.files_discovered,
                "files_parsed": prepared.stats.files_parsed,
                "files_chunked": prepared.stats.files_parsed,
                "symbols_extracted": prepared.stats.symbols_extracted,
                "chunks_generated": prepared.stats.chunks_generated,
                "chunks_expected": len(prepared.chunks),
                "largest_embedding_input_chars": largest_chars,
            },
        )
        service = VectorIndexingService(
            self.store,
            self.embedding_provider,
            None,
            batch_size=self.batch_size,
            inter_batch_delay_seconds=self.inter_batch_delay_seconds,
            embedding_identity=self.embedding_identity,
        )
        embedded = service.prepare_chunks(prepared.chunks, progress=lambda values: emit("embedding", values))
        if embedded.errors:
            raise IndexingCoordinatorError(
                "vector_indexing_failed",
                tuple(IndexingFailure(error.stage, error.error_type) for error in embedded.errors),
                structural_stats=prepared.stats,
                truncated=embedded.truncated,
            )
        return service, embedded, largest_chars

    def _reuse_vectors(self, base: _IncrementalBase, chunks: Sequence[StoredChunk]) -> tuple[VectorRecord, ...]:
        if not chunks:
            return ()
        exported = {record.chunk_id: record for record in base.vector_index.get_vectors([chunk.chunk_id for chunk in chunks])}
        records: list[VectorRecord] = []
        for chunk in chunks:
            record = exported.get(chunk.chunk_id)
            if record is None or record.metadata.get("project_id") != base.project.id:
                raise VectorIndexStateError("Reusable vector project binding is invalid")
            if record.metadata.get("content_hash") != chunk.content_hash:
                raise VectorIndexStateError("Reusable vector content hash is invalid")
            if record.metadata.get("embedding_model") != self.embedding_identity.model:
                raise VectorIndexStateError("Reusable vector model binding is invalid")
            if record.metadata.get("dimensions") != base.dimensions or len(record.vector) != base.dimensions:
                raise VectorIndexStateError("Reusable vector dimensions are invalid")
            records.append(
                VectorRecord(
                    chunk.chunk_id,
                    record.vector,
                    candidate_vector_metadata(chunk, base.project.id, self.embedding_identity.model, base.dimensions),
                )
            )
        return tuple(records)

    def _new_records(
        self,
        embedded: PreparedEmbeddings,
        project_id: int,
        active_dimensions: int,
    ) -> tuple[VectorRecord, ...]:
        records = embedded.records(project_id)
        if not records:
            return ()
        dimensions = len(records[0].vector)
        if active_dimensions and dimensions != active_dimensions:
            raise VectorIndexStateError("New and reusable vector dimensions differ")
        return tuple(
            VectorRecord(
                record.chunk_id,
                record.vector,
                {
                    "project_id": project_id,
                    "content_hash": record.metadata["content_hash"],
                    "embedding_model": self.embedding_identity.model,
                    "dimensions": dimensions,
                },
            )
            for record in records
        )

    def _target_dimensions(self, active_dimensions: int, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        dimensions = {len(record.vector) for record in records}
        if len(dimensions) != 1:
            raise VectorIndexStateError("Candidate vector dimensions differ")
        value = dimensions.pop()
        if active_dimensions and value != active_dimensions:
            raise VectorIndexStateError("Candidate and active vector dimensions differ")
        return value

    def _require_prepared(self, prepared: PreparedRepositoryIndex) -> None:
        if not prepared.succeeded or prepared.root_path is None:
            raise IndexingCoordinatorError(
                "indexing_failed",
                tuple(IndexingFailure(error.stage, error.error_type) for error in prepared.errors),
                structural_stats=prepared.stats,
            )

    def _filter_allowed_structural_errors(self, prepared: PreparedRepositoryIndex) -> PreparedRepositoryIndex:
        """Drop only explicitly allow-listed parse/chunk errors for evaluation fixtures."""
        if not self.allowed_structural_error_paths or not prepared.errors:
            return prepared
        remaining = tuple(
            error
            for error in prepared.errors
            if not (
                error.stage in {"parse", "chunk"}
                and error.relative_path in self.allowed_structural_error_paths
            )
        )
        if len(remaining) == len(prepared.errors):
            return prepared
        return PreparedRepositoryIndex(
            prepared.root_path,
            prepared.files,
            prepared.parse_results,
            prepared.chunk_results,
            prepared.chunks,
            prepared.stats,
            remaining,
        )

    def _result(
        self,
        project: ProjectRecord,
        strategy: str,
        no_changes: bool,
        stats: IndexingStats,
        embedded: PreparedEmbeddings,
        actual_ids: tuple[str, ...],
        *,
        operation: str,
        file_changes: dict[str, int],
        chunks_reused: int,
        vectors_deleted: int,
        largest_chars: int,
    ) -> CoordinatedIndexingResult:
        dimensions = _vector_index_dimensions(self.collection_factory(project.id))
        return CoordinatedIndexingResult(
            project_id=project.id,
            operation=operation,
            strategy=strategy,
            no_changes=no_changes,
            structural_stats=stats,
            chunks_expected=len(actual_ids),
            embeddings_generated=len(embedded.embedded),
            vectors_stored=len(actual_ids),
            chunks_reused=chunks_reused,
            vectors_reused=chunks_reused,
            vectors_deleted=vectors_deleted,
            file_changes=file_changes,
            truncated=embedded.truncated,
            embedding_retries=embedded.retries,
            largest_embedding_input_chars=largest_chars,
            embedding_provider=self.embedding_identity.provider,
            embedding_model=self.embedding_identity.model,
            embedding_dimensions=dimensions,
            expected_ids=actual_ids,
            vector_ids=actual_ids,
        )

    def _state_error(self, error: Exception) -> IndexingCoordinatorError:
        return IndexingCoordinatorError(
            "vector_index_state_invalid",
            (IndexingFailure("vector", type(error).__name__),),
        )

    def _raise_run_error(
        self,
        error: Exception,
        stats: IndexingStats,
        truncated: tuple[TruncatedEmbedding, ...],
    ) -> None:
        if isinstance(error, IndexingCoordinatorError):
            raise error
        if isinstance(error, StorageError):
            raise IndexingCoordinatorError(
                "indexing_failed",
                (IndexingFailure("storage", type(error).__name__),),
                structural_stats=stats,
                truncated=truncated,
            ) from error
        if isinstance(error, VectorIndexStateError):
            raise IndexingCoordinatorError(
                "vector_index_state_invalid",
                (IndexingFailure("vector", type(error).__name__),),
                structural_stats=stats,
                truncated=truncated,
            ) from error
        if isinstance(error, VectorIndexError):
            raise IndexingCoordinatorError(
                "vector_indexing_failed",
                (IndexingFailure("vector", type(error).__name__),),
                structural_stats=stats,
                truncated=truncated,
            ) from error
        raise error


def plan_file_changes(current: Sequence[SourceFile], stored: Sequence[SourceFileRecord]) -> FileChangePlan:
    """Classify a raw-byte file snapshot by relative path and SHA-256."""
    current_by_path = {item.relative_path: item for item in current}
    stored_by_path = {item.relative_path: item for item in stored}
    unchanged = tuple(
        current_by_path[path]
        for path in sorted(current_by_path.keys() & stored_by_path.keys())
        if current_by_path[path].sha256 == stored_by_path[path].sha256
    )
    modified = tuple(
        current_by_path[path]
        for path in sorted(current_by_path.keys() & stored_by_path.keys())
        if current_by_path[path].sha256 != stored_by_path[path].sha256
    )
    added = tuple(current_by_path[path] for path in sorted(current_by_path.keys() - stored_by_path.keys()))
    deleted = tuple(sorted(stored_by_path.keys() - current_by_path.keys()))
    return FileChangePlan(unchanged, added, modified, deleted)


def file_snapshot(files: Sequence[SourceFile]) -> tuple[tuple[str, str], ...]:
    """Return the path/hash identity used for best-effort source rechecks."""
    return tuple(sorted((item.relative_path, item.sha256) for item in files))


def candidate_vector_metadata(
    chunk: StoredChunk,
    project_id: int,
    model: str,
    dimensions: int,
) -> dict[str, str | int]:
    """Build vector metadata from canonical SQLite chunk identity."""
    return {
        "project_id": project_id,
        "content_hash": chunk.content_hash,
        "embedding_model": model,
        "dimensions": dimensions,
    }


def preflight_embedding(provider: EmbeddingProvider) -> None:
    """Run only the approved local Ollama model-availability preflight."""
    if not isinstance(provider, OllamaEmbeddingProvider):
        return
    try:
        provider.preflight()
    except EmbeddingProviderError as error:
        if error.error_type == "ModelNotFound":
            raise IndexingCoordinatorError(
                "embedding_model_unavailable",
                (IndexingFailure("preflight", "ModelNotFound"),),
            ) from error
        raise IndexingCoordinatorError(
            "embedding_provider_unavailable",
            (IndexingFailure("preflight", error.error_type),),
        ) from error


def _vector_index_dimensions(index: ChromaVectorIndex) -> int | None:
    value = index.get_index_metadata().get("codecompass:embedding_dimensions")
    return value if isinstance(value, int) and value > 0 else None
