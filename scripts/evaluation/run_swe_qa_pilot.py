"""Run a retrieval-only SWE-QA pilot without touching frozen evaluations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from codecompass.indexing import IndexingService
from codecompass.retrieval import LexicalRetriever, RetrievalQuery
from codecompass.storage import SQLiteMetadataStore

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "which", "what", "where", "when",
    "how", "does", "are", "is", "can", "into", "their", "they", "then", "than", "use", "used",
    "function", "class", "method", "file", "code", "return", "returns", "object", "value", "called",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = SQLiteMetadataStore(args.database)
    store.initialize()
    prepared = IndexingService(store).prepare_repository(args.repo)
    if not prepared.succeeded or prepared.root_path is None:
        raise RuntimeError(f"structural indexing failed: {prepared.errors}")
    project = store.replace_project_index(
        args.repo.name,
        prepared.root_path,
        prepared.files,
        prepared.parse_results,
        prepared.chunks,
    )
    retriever = LexicalRetriever(store)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        question = str(record["question"])
        answer = str(record["answer"])
        results = retriever.search(RetrievalQuery(question, project.id, 5))
        answer_terms = {term.casefold() for term in IDENTIFIER.findall(answer)} - STOPWORDS
        hits = []
        for item in results:
            names = {item.symbol_name or "", item.qualified_name or "", item.source_file}
            names |= {part for name in tuple(names) for part in re.split(r"[./\\:]", name)}
            hits.append(bool(answer_terms & {name.casefold() for name in names}))
        rows.append(
            {
                "id": f"{args.repo.name}-{index:03d}",
                "question": question,
                "reference_answer": answer,
                "retrieved": [
                    {
                        "score": item.score,
                        "file": item.source_file,
                        "symbol": item.qualified_name,
                        "lines": [item.start_line, item.end_line],
                    }
                    for item in results
                ],
                "answer_identifier_hit_at_5": any(hits),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    hit_rate = sum(bool(row["answer_identifier_hit_at_5"]) for row in rows) / len(rows) if rows else 0.0
    print(json.dumps({"questions": len(rows), "files": len(prepared.files), "chunks": len(prepared.chunks), "answer_identifier_hit_at_5": hit_rate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
