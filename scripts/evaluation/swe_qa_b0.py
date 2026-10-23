"""B0: SWE-QA-style function-chunk RAG baseline, re-implemented inside this project's evaluation setup.

The official SWE-QA `rag_function_chunk` baseline embeds function and class chunks with Voyage `voyage-code-3`, searches a
FAISS index, and answers with DeepSeek-V3. This re-implementation keeps the chunking idea and the dense top-k search but
uses the embedding provider and generation settings of the other arms, so only chunking and retrieval differ.

    python scripts/evaluation/swe_qa_b0.py index       # chunk repositories and embed chunks (embedding calls)
    python scripts/evaluation/swe_qa_b0.py retrieve    # rank chunks per question, write rankings and contexts
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swe_qa_v2 as run  # noqa: E402
from codecompass.providers import ProviderConfig, create_embedding_provider  # noqa: E402
from codecompass.rag import RAGContextBuilder  # noqa: E402
from codecompass.retrieval import RetrievalQuery, RetrievalResult, RetrievedChunk  # noqa: E402

OUT = run.OUT / "b0"
MAX_CHUNK_CHARS = 32768  # same truncation limit as the official function-chunk baseline
BATCH_SIZE = 32
TOP_K = 10
CONFIGS = {"b0_k5": (5, 6000), "b0_k10": (10, 10**9)}  # (chunks, character budget); b0_k5 matches the A1 context budget


def chunk_source(path: str, text: str) -> list[dict]:
    """Function/method chunks hold their code; class chunks hold only the class name and docstring (SWE-QA style)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    chunks: list[dict] = []

    def add(kind: str, name: str, qualified: str, start: int, end: int, body: str) -> None:
        chunks.append({"file": path, "kind": kind, "symbol": name, "qualified": qualified,
                       "start_line": start, "end_line": end, "text": body[:MAX_CHUNK_CHARS]})

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                qualified = f"{prefix}{child.name}"
                add("function", child.name, qualified, start, end, "\n".join(lines[start - 1:end]))
                visit(child, f"{qualified}.")
            elif isinstance(child, ast.ClassDef):
                qualified = f"{prefix}{child.name}"
                doc = ast.get_docstring(child) or ""
                add("class", child.name, qualified, child.lineno, child.end_lineno or child.lineno,
                    f"class_name: {child.name}\nclass_docstring: {doc}")
                visit(child, f"{qualified}.")

    visit(tree, "")
    return chunks


def embedding_provider():
    return create_embedding_provider(ProviderConfig(
        provider="openai_compatible", base_url=os.environ["CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL"],
        api_key=os.environ["CODECOMPASS_GEMINI2_EMBEDDING_API_KEY"],
        embedding_model=os.environ.get("CODECOMPASS_GEMINI2_EMBEDDING_MODEL", "gemini-embedding-2"),
        embedding_dimensions=3072, timeout_seconds=180))


def embed_with_retry(provider, texts: list[str]) -> np.ndarray:
    for attempt in range(1, 5):
        try:
            return np.array([r.vector for r in provider.embed_texts(texts)], dtype=np.float32)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def top_chunks(query_vector: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    scores = matrix @ query_vector
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
    return [(i, float(scores[i])) for i in order]


def index_repo(repo: str, provider) -> None:
    root = run.REPOS[repo][1]
    chunks: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        chunks.extend(chunk_source(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="replace")))
    for number, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"b0-{repo}-{number}"
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path, vec_path = OUT / f"{repo}_chunks.jsonl", OUT / f"{repo}_vectors.npy"
    if meta_path.exists() or vec_path.exists():
        raise SystemExit(f"refusing to overwrite B0 index for {repo}")
    vectors = [embed_with_retry(provider, [c["text"] for c in chunks[i:i + BATCH_SIZE]]) for i in range(0, len(chunks), BATCH_SIZE)]
    np.save(vec_path, normalize(np.vstack(vectors)))
    meta_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")
    print(repo, len(chunks), "chunks", flush=True)


def retrieve() -> None:
    provider = embedding_provider()
    rows = sum((run.load_rows(path, repo) for repo, (path, *_rest) in run.REPOS.items()), [])
    indexes = {}
    for repo in run.REPOS:
        meta = [json.loads(line) for line in (OUT / f"{repo}_chunks.jsonl").open(encoding="utf-8") if line.strip()]
        indexes[repo] = (meta, np.load(OUT / f"{repo}_vectors.npy"))
    builder = RAGContextBuilder()
    with run.open_new(OUT / "rankings.jsonl") as rankings, run.open_new(OUT / "contexts.jsonl") as contexts:
        for row in rows:
            meta, matrix = indexes[row["repo"]]
            query = normalize(embed_with_retry(provider, [row["question"]]))[0]
            hits = top_chunks(query, matrix, TOP_K)
            retrieved = tuple(RetrievedChunk(chunk_id=meta[i]["chunk_id"], score=score, source_file=meta[i]["file"],
                                             symbol_name=meta[i]["symbol"], qualified_name=meta[i]["qualified"],
                                             start_line=meta[i]["start_line"], end_line=meta[i]["end_line"],
                                             code=meta[i]["text"], retrieval_method="semantic") for i, score in hits)
            rankings.write(json.dumps({"question_id": row["question_id"], "repo": row["repo"], "ranking": [
                {"chunk_id": c.chunk_id, "file": c.source_file, "start_line": c.start_line, "end_line": c.end_line, "score": c.score}
                for c in retrieved]}, ensure_ascii=False) + "\n")
            record = {"question_id": row["question_id"], "repo": row["repo"], "question": row["question"], "configs": {}}
            for name, (limit, budget) in CONFIGS.items():
                context = builder.build(RetrievalResult(RetrievalQuery(row["question"], 0, limit), retrieved[:limit]), budget)
                text = run.context_text(context)
                record["configs"][name] = {"chunk_ids": [b.chunk_id for b in context.blocks],
                                           "blocks": [asdict(b.citation) for b in context.blocks],
                                           "context": text, "context_sha256": run.sha(text), "omitted_count": context.omitted_count}
            contexts.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["question_id"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("index", "retrieve"))
    args = parser.parse_args()
    if args.phase == "index":
        provider = embedding_provider()
        for repo_name in run.REPOS:
            index_repo(repo_name, provider)
    else:
        retrieve()
