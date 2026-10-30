"""Command-line demo for grounded CodeCompass questions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from codecompass.providers import (
    OPENAI_COMPATIBLE,
    OLLAMA,
    ProviderConfig,
    create_embedding_provider,
    create_llm_provider,
)
from codecompass.qa import GroundedQAService, QAAnswer, QACitation, QAError, QAPromptBuilder, QARequest
from codecompass.rag import RAGContext, RAGContextBuilder
from codecompass.retrieval import RetrievalService
from codecompass.storage import SQLiteMetadataStore, StorageError
from codecompass.vector_index import ChromaVectorIndex

DISPLAY_CITATION_LIMIT = 3
DEMO_RESPONSE_INSTRUCTIONS = (
    "Begin with the function that directly answers the question, using exactly two concise natural-language sentences. "
    "If the question is Persian, answer in Persian."
)
NO_DISPLAYABLE_ANSWER = "The local model returned no displayable answer."


class _DemoPromptBuilder(QAPromptBuilder):
    def build(self, question: str, context: RAGContext) -> tuple[str, str]:
        system_prompt, prompt = super().build(question, context)
        return system_prompt, f"Response style:\n{DEMO_RESPONSE_INSTRUCTIONS}\n\n{prompt}"


def format_answer(answer: QAAnswer) -> str:
    """Format a grounded answer for terminal display."""
    lines = [
        "================================",
        "CodeQAPass Grounded QA Demo",
        "================================",
        "",
        "Question:",
        answer.question,
        "",
        "Answer:",
        _clean_answer(answer.answer),
        "",
        "Model:",
        answer.llm_model or "not used",
        "",
        "Provider:",
        answer.llm_provider or "not used",
        "",
        "Verified Sources:",
    ]
    if not answer.citations:
        lines.append("None")
    citations = _display_citations(answer)
    for number, citation in enumerate(citations[:DISPLAY_CITATION_LIMIT], start=1):
        lines.extend(
            (
                f"{number}. Symbol: {citation.qualified_name or citation.symbol_name or 'unknown'}",
                f"   File: {citation.source_file}",
                f"   Lines: {citation.start_line}-{citation.end_line}",
                "",
            )
        )
    omitted = len(answer.citations) - DISPLAY_CITATION_LIMIT
    if omitted > 0:
        lines.append(f"Additional evidence blocks not shown: {omitted}")
    return "\n".join(lines)


def _display_citations(answer: QAAnswer) -> tuple[QACitation, ...]:
    question_terms = set(re.findall(r"\w+", answer.question.casefold()))

    def key(item: tuple[int, QACitation]) -> tuple[bool, int]:
        index, citation = item
        symbol = (citation.symbol_name or citation.qualified_name or "").casefold().rsplit(".", 1)[-1]
        mentioned = bool(symbol) and (symbol in question_terms or f"{symbol}s" in question_terms)
        return not mentioned, index

    return tuple(citation for _, citation in sorted(enumerate(answer.citations), key=key))


def _clean_answer(text: str) -> str:
    cleaned = re.sub(r"```[A-Za-z0-9_+-]*", "", text).strip()
    cleaned = re.sub(r"(?m)^\s*\[/?S\d+\]\s*$", "", cleaned).strip()
    if cleaned.casefold().startswith("answer:"):
        cleaned = cleaned[len("answer:") :].lstrip()
    if not cleaned:
        return NO_DISPLAYABLE_ANSWER
    return " ".join(re.split(r"(?<=[.!?])\s+", cleaned)[:2])


def main(argv: Sequence[str] | None = None) -> int:
    """Run one grounded question against an already indexed project."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = ProviderConfig.from_environment(
            provider=args.provider,
            base_url=args.base_url or (args.ollama_url if args.provider == OLLAMA else None),
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            timeout_seconds=args.timeout_seconds,
            embedding_dimensions=args.embedding_dimensions,
        )
        embedding_provider = create_embedding_provider(config)
        llm_provider = create_llm_provider(config)
    except ValueError as error:
        parser.error(str(error))
    store = SQLiteMetadataStore(args.database)

    try:
        project = store.get_project_by_root(args.repository)
        if project is None:
            parser.error("repository is not indexed in the supplied SQLite database")

        retrieval = RetrievalService(
            store,
            embedding_provider,
            ChromaVectorIndex(args.chroma, args.collection),
        )
        service = GroundedQAService(
            retrieval,
            RAGContextBuilder(),
            _DemoPromptBuilder(),
            llm_provider,
        )
        answer = service.answer(
            QARequest(
                question=args.question,
                project_id=project.id,
                retrieval_method=args.retrieval_method,
                retrieval_limit=args.limit,
                max_context_chars=args.max_context_chars,
                max_tokens=args.max_tokens,
            )
        )
    except (QAError, StorageError) as error:
        parser.exit(1, f"CodeQAPass demo failed: {error}\n")

    print(format_answer(answer))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask one grounded question about an indexed Python repository.")
    parser.add_argument("--repository", required=True, type=Path, help="Indexed repository root")
    parser.add_argument("--database", required=True, type=Path, help="SQLite metadata database")
    parser.add_argument("--chroma", required=True, type=Path, help="Chroma persistence directory")
    parser.add_argument("--collection", required=True, help="Existing Chroma collection name")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--provider", choices=(OLLAMA, OPENAI_COMPATIBLE), default=OLLAMA)
    parser.add_argument("--base-url", help="Provider base URL; required for openai_compatible")
    parser.add_argument("--llm-model", required=True, help="Generation model identifier")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--retrieval-method", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
