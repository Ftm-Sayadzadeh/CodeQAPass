"""FastAPI application exposing existing CodeCompass domain services."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from codecompass.api.evaluation import EvaluationArtifactError
from codecompass.api.runtime import APIError, APIRuntime, APISettings
from codecompass.api.schemas import (
    AskRequest,
    AskResponse,
    CitationResponse,
    DocumentationRequest,
    DocumentationResponse,
    ErrorEnvelope,
    EvaluationResponse,
    HealthResponse,
    IndexProjectRequest,
    IndexProjectResponse,
    IndexJobRequest,
    IndexJobResponse,
    ProjectResponse,
    RetrievedChunkResponse,
    SearchRequest,
    SearchResponse,
    SourceContentResponse,
    SourceFileResponse,
    SymbolResponse,
)
from codecompass.documentation import DocumentationError, FunctionDocumentationService
from codecompass.qa import GroundedQAService, QAError, QAPromptBuilder, QARequest
from codecompass.rag import RAGContextBuilder
from codecompass.retrieval import RetrievalError, RetrievalQuery
from codecompass.storage import IndexingJobRecord, StorageError, StoredChunk
from codecompass.vector_index import VectorIndexStateError


def get_runtime(request: Request) -> APIRuntime:
    return request.app.state.runtime


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Create the local API without making external provider calls."""
    error_responses = {
        status: {"model": ErrorEnvelope, "description": "Safe CodeQAPass error"}
        for status in (400, 404, 409, 422, 500, 502, 503, 504)
    }
    app = FastAPI(title="CodeQAPass API", version="1.0.0", responses=error_responses)
    app.state.runtime = APIRuntime(settings or APISettings.from_environment())

    @app.exception_handler(APIError)
    def api_error(_: Request, error: APIError) -> JSONResponse:
        return _error(error.status, error.code, error.message, error.details)

    @app.exception_handler(RequestValidationError)
    def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            {"location": [str(part) for part in item.get("loc", ())], "type": str(item.get("type", "invalid")), "message": "Invalid request value"}
            for item in error.errors()
        ]
        return _error(422, "validation_error", "Request validation failed", {"fields": fields})

    @app.exception_handler(DocumentationError)
    def documentation_error(_: Request, error: DocumentationError) -> JSONResponse:
        status = {"invalid_request": 422, "not_found": 404, "ambiguous": 409, "insufficient_evidence": 422, "provider_timeout": 504}.get(error.code, 502)
        details = {"candidates": [asdict(item) for item in error.candidates]} if error.candidates else {}
        if error.provider_error_type:
            details["provider_error_type"] = error.provider_error_type
        return _error(status, f"documentation_{error.code}", error.message, details)

    @app.exception_handler(QAError)
    def qa_error(_: Request, error: QAError) -> JSONResponse:
        status = 422 if error.stage == "request" else 502 if error.stage == "llm" else 500
        return _error(status, f"qa_{error.stage}_failed", "Grounded answer generation failed")

    @app.exception_handler(RetrievalError)
    def retrieval_error(_: Request, error: RetrievalError) -> JSONResponse:
        if error.stage == "embedding_configuration_mismatch":
            return _error(409, "embedding_configuration_mismatch", "Embedding configuration does not match the index; re-index is required")
        if error.stage == "vector_index_state_invalid":
            return _error(409, "vector_index_state_invalid", "Vector index state is invalid; re-index storage is required")
        status = 422 if error.stage == "query" else 502 if error.stage == "embedding" else 500
        return _error(status, f"retrieval_{error.stage}_failed", "Retrieval failed")

    @app.exception_handler(VectorIndexStateError)
    def vector_index_state_error(_: Request, __: VectorIndexStateError) -> JSONResponse:
        return _error(409, "vector_index_state_invalid", "Vector index state is invalid; re-index storage is required")

    @app.exception_handler(EvaluationArtifactError)
    def evaluation_error(_: Request, __: EvaluationArtifactError) -> JSONResponse:
        return _error(503, "evaluation_unavailable", "Evaluation artifact is unavailable")

    @app.exception_handler(ValueError)
    def configuration_error(_: Request, __: ValueError) -> JSONResponse:
        return _error(422, "provider_configuration_invalid", "Provider configuration is invalid")

    @app.exception_handler(StorageError)
    def storage_error(_: Request, __: StorageError) -> JSONResponse:
        return _error(500, "storage_failure", "Metadata storage failed")

    @app.exception_handler(Exception)
    def unexpected_error(_: Request, __: Exception) -> JSONResponse:
        return _error(500, "internal_error", "An internal error occurred")

    @app.get("/health", response_model=HealthResponse)
    def health(runtime: APIRuntime = Depends(get_runtime)) -> HealthResponse:
        runtime.store.initialize()
        return HealthResponse()

    @app.get("/projects", response_model=list[ProjectResponse])
    def projects(runtime: APIRuntime = Depends(get_runtime)) -> list[ProjectResponse]:
        return [_project(item) for item in runtime.store.list_projects()]

    @app.get("/projects/{project_id}", response_model=ProjectResponse)
    def project(project_id: int, runtime: APIRuntime = Depends(get_runtime)) -> ProjectResponse:
        item = runtime.require_project(project_id)
        files = runtime.store.list_source_files(project_id)
        symbols = tuple(symbol for source in files for symbol in runtime.store.list_symbols(source.id))
        chunks = runtime.store.list_chunks(project_id)
        generation_matches = runtime.vector_generation_matches(item)
        vectors = runtime.collection(project_id).list_ids() if generation_matches else ()
        return ProjectResponse(
            **_project(item).model_dump(exclude={"files", "symbols", "chunks", "vector_complete"}),
            files=len(files),
            symbols=len(symbols),
            chunks=len(chunks),
            vector_complete=(
                generation_matches
                and set(vectors) == {chunk.chunk_id for chunk in chunks}
            ),
        )

    @app.post("/projects/index", response_model=IndexProjectResponse)
    def index_project(body: IndexProjectRequest, runtime: APIRuntime = Depends(get_runtime)) -> dict[str, Any]:
        return runtime.index(body.repository_path, body.project_name, body.embedding)

    @app.post("/projects/index-jobs", response_model=IndexJobResponse, status_code=202)
    def start_index_job(body: IndexJobRequest, runtime: APIRuntime = Depends(get_runtime)) -> IndexJobResponse:
        return _index_job(
            runtime.start_index_job(
                repository_path=body.repository_path,
                project_id=body.project_id,
                project_name=body.project_name,
                override=body.embedding,
            )
        )

    @app.get("/projects/index-jobs/active", response_model=IndexJobResponse)
    def active_index_job(runtime: APIRuntime = Depends(get_runtime)) -> IndexJobResponse | Response:
        job = runtime.active_indexing_job()
        return _index_job(job) if job is not None else Response(status_code=204)

    @app.get("/projects/index-jobs/{job_id}", response_model=IndexJobResponse)
    def index_job(job_id: str, runtime: APIRuntime = Depends(get_runtime)) -> IndexJobResponse:
        return _index_job(runtime.indexing_job(job_id))

    @app.get("/projects/{project_id}/files", response_model=list[SourceFileResponse])
    def files(project_id: int, runtime: APIRuntime = Depends(get_runtime)) -> list[SourceFileResponse]:
        runtime.require_project(project_id)
        return [SourceFileResponse(id=item.id, relative_path=item.relative_path, size_bytes=item.size_bytes, sha256=item.sha256, status=item.status) for item in runtime.store.list_source_files(project_id)]

    @app.get("/projects/{project_id}/files/{file_id}/content", response_model=SourceContentResponse)
    def file_content(project_id: int, file_id: int, runtime: APIRuntime = Depends(get_runtime)) -> dict[str, Any]:
        return runtime.source_content(project_id, file_id)

    @app.get("/projects/{project_id}/symbols", response_model=list[SymbolResponse])
    def symbols(project_id: int, file_id: int | None = Query(default=None, ge=1), runtime: APIRuntime = Depends(get_runtime)) -> list[SymbolResponse]:
        runtime.require_project(project_id)
        files = runtime.store.list_source_files(project_id)
        if file_id is not None:
            files = tuple(item for item in files if item.id == file_id)
            if not files:
                raise APIError(404, "file_not_found", "Source file was not found")
        return [
            SymbolResponse(id=item.id, file_id=item.file_id, kind=item.kind, name=item.name, qualified_name=item.qualified_name, is_async=item.is_async, start_line=item.start_line, end_line=item.end_line, parameters=list(item.parameters), returns=item.returns)
            for source in files for item in runtime.store.list_symbols(source.id)
        ]

    @app.post("/projects/{project_id}/search", response_model=SearchResponse)
    def search(project_id: int, body: SearchRequest, runtime: APIRuntime = Depends(get_runtime)) -> SearchResponse:
        config = runtime.embedding_config(body.embedding)
        service = runtime.retrieval(project_id, config, compatible=body.method != "lexical")
        result = getattr(service, f"search_{body.method}")(RetrievalQuery(body.query, project_id, body.limit))
        chunks = runtime.citation_chunks(project_id, tuple(item.chunk_id for item in result.results))
        return SearchResponse(
            query=body.query,
            method=body.method,
            results=[
                RetrievedChunkResponse(
                    **_citation(chunks[item.chunk_id]).model_dump(),
                    score=item.score,
                    code=chunks[item.chunk_id].code,
                    retrieval_method=item.retrieval_method,
                )
                for item in result.results
            ],
        )

    @app.post("/projects/{project_id}/ask", response_model=AskResponse)
    def ask(project_id: int, body: AskRequest, runtime: APIRuntime = Depends(get_runtime)) -> AskResponse:
        retrieval = runtime.retrieval(project_id, runtime.embedding_config(body.embedding), compatible=body.method != "lexical")
        service = GroundedQAService(retrieval, RAGContextBuilder(), QAPromptBuilder(), create_llm(runtime, body.llm))
        answer = service.answer(QARequest(question=body.question, project_id=project_id, retrieval_method=body.method, max_tokens=body.max_tokens))
        chunks = runtime.citation_chunks(project_id, tuple(item.chunk_id for item in answer.citations))
        return AskResponse(question=answer.question, answer=answer.answer, method=answer.retrieval_method, citations=[_citation(chunks[item.chunk_id]) for item in answer.citations], omitted_context_count=answer.omitted_context_count, llm_model=answer.llm_model, llm_provider=answer.llm_provider, finish_reason=answer.finish_reason)

    @app.post("/projects/{project_id}/documentation", response_model=DocumentationResponse)
    def documentation(project_id: int, body: DocumentationRequest, runtime: APIRuntime = Depends(get_runtime)) -> dict[str, Any]:
        runtime.require_project(project_id)
        result = FunctionDocumentationService(runtime.store, create_llm(runtime, body.llm)).document_symbol(project_id, body.identifier, language=body.language, max_tokens=body.max_tokens)
        response = asdict(result)
        chunks = runtime.citation_chunks(project_id, tuple(item.chunk_id for item in result.citations))
        project_name = runtime.require_project(project_id).name
        response["citations"] = [
            _documentation_citation(chunks[item.chunk_id], project_name)
            for item in result.citations
        ]
        target = chunks[result.extracted.citation.chunk_id]
        response["extracted"]["citation"] = _documentation_citation(target, project_name)
        return response

    @app.get("/evaluation/summary", response_model=EvaluationResponse)
    def evaluation_summary(runtime: APIRuntime = Depends(get_runtime)) -> EvaluationResponse:
        digest, data = runtime.evaluation(performance=False)
        return EvaluationResponse(artifact_sha256=digest, data=data)

    @app.get("/evaluation/performance", response_model=EvaluationResponse)
    def evaluation_performance(runtime: APIRuntime = Depends(get_runtime)) -> EvaluationResponse:
        digest, data = runtime.evaluation(performance=True)
        return EvaluationResponse(artifact_sha256=digest, data=data)

    @app.get("/evaluation/final-thesis", response_model=EvaluationResponse)
    def final_thesis_evaluation(runtime: APIRuntime = Depends(get_runtime)) -> EvaluationResponse:
        digest, data = runtime.final_thesis_evaluation()
        return EvaluationResponse(artifact_sha256=digest, data=data)

    @app.get("/evaluation/context-strategy", response_model=EvaluationResponse)
    def context_strategy_evaluation(runtime: APIRuntime = Depends(get_runtime)) -> EvaluationResponse:
        digest, data = runtime.context_strategy_evaluation()
        return EvaluationResponse(artifact_sha256=digest, data=data)

    return app


def create_llm(runtime: APIRuntime, override: Any):
    from codecompass.providers import create_llm_provider

    return create_llm_provider(runtime.llm_config(override))


def _project(item: Any) -> ProjectResponse:
    return ProjectResponse(id=item.id, name=item.name, created_at=item.created_at, updated_at=item.updated_at)


def _citation(chunk: StoredChunk) -> CitationResponse:
    return CitationResponse(
        file_id=chunk.file_id,
        symbol_id=chunk.symbol_id,
        chunk_id=chunk.chunk_id,
        source_file=chunk.relative_path,
        symbol_name=chunk.qualified_name.rsplit(".", 1)[-1] if chunk.qualified_name else None,
        qualified_name=chunk.qualified_name,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
    )


def _documentation_citation(chunk: StoredChunk, project_name: str) -> dict[str, Any]:
    if chunk.symbol_id is None or chunk.qualified_name is None:
        raise APIError(500, "citation_metadata_missing", "Trusted citation metadata is unavailable")
    return {
        "project_id": chunk.project_id,
        "project_name": project_name,
        "file_id": chunk.file_id,
        "symbol_id": chunk.symbol_id,
        "chunk_id": chunk.chunk_id,
        "qualified_name": chunk.qualified_name,
        "relative_source_path": chunk.relative_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "content_hash": chunk.content_hash,
    }


def _index_job(job: IndexingJobRecord) -> IndexJobResponse:
    started = datetime.fromisoformat(job.started_at)
    ended = datetime.fromisoformat(job.completed_at) if job.completed_at else datetime.now(timezone.utc)
    return IndexJobResponse(
        id=job.id,
        state=job.state,
        operation=job.operation,
        project_id=job.project_id,
        counters=job.counters,
        observed_stages=list(job.observed_stages),
        started_at=job.started_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        elapsed_seconds=max(0.0, (ended - started).total_seconds()),
        previous_index_preserved=job.previous_index_preserved,
        result=job.result,
        error=job.error,
    )


def _error(status: int, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details or {}}})
