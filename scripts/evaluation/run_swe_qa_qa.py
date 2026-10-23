"""Run grounded semantic QA over one SWE-QA JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from codecompass.providers import ProviderConfig, create_embedding_provider, create_llm_provider
from codecompass.qa import GroundedQAService, QAPromptBuilder, QARequest
from codecompass.rag import RAGContextBuilder
from codecompass.retrieval import RetrievalService
from codecompass.storage import SQLiteMetadataStore
from codecompass.vector_index import ChromaVectorIndex


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--chroma", type=Path, required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--retrieval-method", choices=("lexical", "semantic", "hybrid"), default="semantic")
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--resume", action="store_true", help="Append after completed question records")
    p.add_argument("--only-failed-from", type=Path, help="Run only question indexes previously recorded as errors")
    p.add_argument("--qa-retries", type=int, default=2)
    p.add_argument("--qa-retry-delay", type=float, default=2.0)
    args = p.parse_args()

    embedding = create_embedding_provider(ProviderConfig(
        provider="openai_compatible",
        base_url=os.environ["CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL"],
        api_key=os.environ["CODECOMPASS_GEMINI2_EMBEDDING_API_KEY"],
        embedding_model=os.environ.get("CODECOMPASS_GEMINI2_EMBEDDING_MODEL", "gemini-embedding-2"),
        embedding_dimensions=int(os.environ.get("CODECOMPASS_GEMINI2_EMBEDDING_DIMENSIONS", "3072")),
        timeout_seconds=120,
    ))
    llm = create_llm_provider(ProviderConfig(
        provider="openai_compatible",
        base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
        api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"],
        llm_model=os.environ["CODECOMPASS_COMPARE_MODEL"],
        timeout_seconds=120,
    ))
    store = SQLiteMetadataStore(args.database)
    project = store.get_project_by_root(args.repository)
    if project is None:
        raise SystemExit("repository is not indexed in the supplied SQLite database")
    service = GroundedQAService(
        RetrievalService(
            store,
            embedding,
            ChromaVectorIndex(args.chroma, args.collection, managed=True, project_id=project.id),
        ),
        RAGContextBuilder(), QAPromptBuilder(), llm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as previous:
            completed = {json.loads(line)["index"] for line in previous if line.strip()}
    failed_only = set()
    if args.only_failed_from:
        with args.only_failed_from.open(encoding="utf-8") as previous:
            failed_only = {
                item["index"] for line in previous if line.strip()
                for item in [json.loads(line)] if item.get("status") == "error"
            }
    mode = "a" if args.resume else "w"
    with args.questions.open(encoding="utf-8") as source, args.output.open(mode, encoding="utf-8") as target:
        for index, line in enumerate(source, start=1):
            if index in completed:
                continue
            if failed_only and index not in failed_only:
                continue
            row = json.loads(line)
            started = time.perf_counter()
            record = {"index": index, "question": row["question"], "reference_answer": row.get("answer")}
            try:
                answer = None
                last_error = None
                for attempt in range(args.qa_retries + 1):
                    try:
                        answer = service.answer(QARequest(
                            question=row["question"], project_id=project.id, retrieval_method=args.retrieval_method,
                            retrieval_limit=args.limit, max_context_chars=6000, temperature=0, max_tokens=args.max_tokens,
                        ))
                        break
                    except Exception as error:
                        last_error = error
                        if attempt == args.qa_retries:
                            raise
                        time.sleep(args.qa_retry_delay * (2**attempt))
                if answer is None:
                    raise last_error or RuntimeError("QA produced no answer")
                record.update({
                    "status": "usable", "answer": answer.answer, "retrieval_method": answer.retrieval_method,
                    "citations": [c.__dict__ if hasattr(c, "__dict__") else {
                        "chunk_id": c.chunk_id, "source_file": c.source_file, "symbol_name": c.symbol_name,
                        "qualified_name": c.qualified_name, "start_line": c.start_line, "end_line": c.end_line,
                    } for c in answer.citations],
                    "llm_model": answer.llm_model, "llm_provider": answer.llm_provider,
                    "finish_reason": answer.finish_reason, "omitted_context_count": answer.omitted_context_count,
                })
            except Exception as error:  # preserve every failed case for audit
                record.update({"status": "error", "error_type": type(error).__name__, "error": str(error)[:300]})
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(json.dumps({"index": index, "status": record["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
