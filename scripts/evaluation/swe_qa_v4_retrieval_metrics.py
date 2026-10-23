"""Run the pre-specified v4 retrieval ablation and paired statistics.

The script never changes an index or a raw answer.  It writes one immutable
ranking file per method and a derived summary.  Gold files are explicitly
labelled as reference-answer-derived silver evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codecompass.rag import RAGContextBuilder
from codecompass.retrieval import RetrievalQuery, RetrievalResult, RetrievedChunk
from codecompass.storage import SQLiteMetadataStore
from codecompass.vector_index import ChromaVectorIndex

import swe_qa_v4_runner as v4
import swe_qa_b0 as b0

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2"
FROZEN = ROOT / "reports/evaluation/swe_qa_v4/frozen"
METHODS = ("lexical", "semantic", "hybrid", "b0")
CHUNKS_PER_QUERY = 10
KS = (1, 3, 5, 10)
PATH_RE = re.compile(r"(?<![\w/.-])([\w./-]+\.py)\b")
SEED = 20260930
BOOTSTRAPS = 20_000


class QueryCache:
    """Cache query embeddings within this analysis without changing the provider."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.cache: dict[str, object] = {}

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def embed_text(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.inner.embed_text(text)
        return self.cache[text]

    def embed_texts(self, texts):
        missing = [text for text in texts if text not in self.cache]
        if missing:
            for text, result in zip(missing, self.inner.embed_texts(missing)):
                self.cache[text] = result
        return tuple(self.cache[text] for text in texts)


def load_rows() -> list[dict]:
    return [json.loads(line) for line in (FROZEN / "main_a0_combined.jsonl").open(encoding="utf-8") if line.strip()]


def repository_files(repo: str) -> list[str]:
    root = v4.repo_root(repo)
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py") if ".git" not in path.parts)


def gold_files(answer: str, files: list[str]) -> list[str]:
    known = set(files)
    gold: set[str] = set()
    for raw in PATH_RE.findall(answer or ""):
        path = raw.lstrip("./")
        if path in known:
            gold.add(path)
        elif f"src/{path}" in known:
            gold.add(f"src/{path}")
        else:
            matches = [name for name in files if name == path or name.endswith(f"/{path}")]
            if len(matches) == 1:
                gold.add(matches[0])
    return sorted(gold)


def make_services():
    provider = QueryCache(b0.embedding_provider())
    services = {}
    for repo in v4.MAIN_REPOS:
        store = SQLiteMetadataStore(ARTIFACT_ROOT / "index" / f"{repo}.sqlite")
        project = store.get_project_by_root(v4.repo_root(repo))
        if project is None:
            raise SystemExit(f"missing A1 project for {repo}")
        vector = ChromaVectorIndex(ARTIFACT_ROOT / "index" / f"{repo}_chroma", f"swe-qa-v4-{repo}", managed=True, project_id=project.id)
        from codecompass.retrieval import RetrievalService
        services[repo] = (RetrievalService(store, provider, vector), project.id)
    return services, provider


def ranking_from_chunks(chunks: list[dict]) -> list[dict]:
    return [{"chunk_id": c["chunk_id"], "file": c["file"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in chunks]


def write_rankings(rows: list[dict], services, provider) -> dict[str, Path]:
    paths = {method: FROZEN / f"rankings_{method}.jsonl" for method in METHODS}
    if any(path.exists() for path in paths.values()):
        raise SystemExit("ranking artifact already exists; raw retrieval records are append-only")
    files = {repo: repository_files(repo) for repo in v4.MAIN_REPOS}
    indexes = {}
    for repo in v4.MAIN_REPOS:
        indexes[repo] = v4.load_b0_index(repo, ARTIFACT_ROOT)
    handles = {method: path.open("w", encoding="utf-8") for method, path in paths.items()}
    try:
        for number, row in enumerate(rows, 1):
            repo = row["repo"]
            service, project_id = services[repo]
            query = RetrievalQuery(row["question"], project_id, CHUNKS_PER_QUERY)
            results = {
                "lexical": service.search_lexical(query).results,
                "semantic": service.search_semantic(query).results,
                "hybrid": service.search_hybrid(query).results,
            }
            meta, matrix = indexes[repo]
            query_vector = b0.normalize(np.asarray(provider.embed_text(row["question"]).vector, dtype=np.float32)[None, :])[0]
            hits = b0.top_chunks(query_vector, matrix, CHUNKS_PER_QUERY)
            results["b0"] = tuple(
                RetrievedChunk(chunk_id=meta[i]["chunk_id"], score=score, source_file=meta[i]["file"],
                               symbol_name=meta[i]["symbol"], qualified_name=meta[i]["qualified"],
                               start_line=meta[i]["start_line"], end_line=meta[i]["end_line"],
                               code=meta[i]["text"], retrieval_method="semantic")
                for i, score in hits
            )
            gold = gold_files(row.get("reference_answer", ""), files[repo])
            for method, chunks in results.items():
                record = {"question_id": row["question_id"], "repo": repo, "gold_files": gold,
                          "ranking": ranking_from_chunks([{
                              "chunk_id": c.chunk_id, "file": c.source_file, "start_line": c.start_line,
                              "end_line": c.end_line
                          } for c in chunks])}
                handles[method].write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 10 == 0:
                for handle in handles.values():
                    handle.flush()
                print(f"retrieval {number}/{len(rows)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return paths


def distinct_files(ranking: list[dict]) -> list[str]:
    seen: list[str] = []
    for item in ranking:
        if item["file"] not in seen:
            seen.append(item["file"])
    return seen


def question_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    result = {}
    for k in KS:
        top = ranked[:k]
        hits = len(gold.intersection(top))
        result[f"hit@{k}"] = float(hits > 0)
        result[f"recall@{k}"] = hits / len(gold)
        result[f"precision@{k}"] = hits / len(top) if top else 0.0
    first = next((index for index, path in enumerate(ranked[:10], 1) if path in gold), None)
    result["mrr@10"] = 1.0 / first if first else 0.0
    return result


def wilcoxon_p(diffs: list[float]) -> float | None:
    nonzero = [value for value in diffs if value]
    if len(nonzero) < 10:
        return None
    order = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * len(nonzero)
    tie_sum = 0.0
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and abs(nonzero[order[end + 1]]) == abs(nonzero[order[index]]):
            end += 1
        rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        size = end - index + 1
        tie_sum += size**3 - size
        index = end + 1
    w_plus = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    variance = len(nonzero) * (len(nonzero) + 1) * (2 * len(nonzero) + 1) / 24 - tie_sum / 48
    if variance <= 0:
        return None
    z = (abs(w_plus - len(nonzero) * (len(nonzero) + 1) / 4) - 0.5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def bootstrap_stratified(values: dict[str, list[float]], seed: int = SEED, n: int = BOOTSTRAPS) -> list[float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        picked = [rng.choice(group) for group in values.values() for _ in group]
        samples.append(sum(picked) / len(picked))
    samples.sort()
    return [samples[int(0.025 * n)], samples[int(0.975 * n) - 1]]


def summarize(paths: dict[str, Path], rows: list[dict]) -> dict:
    records = {method: {json.loads(line)["question_id"]: json.loads(line) for line in path.open(encoding="utf-8")}
               for method, path in paths.items()}
    ids = [row["question_id"] for row in rows]
    missing = {method: sorted(set(ids) - set(data)) for method, data in records.items()}
    if any(missing.values()):
        raise SystemExit(f"missing retrieval rows: {missing}")
    per_method: dict[str, dict[str, list[float]]] = {method: {} for method in METHODS}
    per_repo: dict[str, dict[str, dict[str, list[float]]]] = {method: {} for method in METHODS}
    scored_ids = []
    for row in rows:
        gold = set(records["lexical"][row["question_id"]]["gold_files"])
        if not gold:
            continue
        scored_ids.append(row["question_id"])
        for method in METHODS:
            values = question_metrics(distinct_files(records[method][row["question_id"]]["ranking"]), gold)
            for key, value in values.items():
                per_method[method].setdefault(key, []).append(value)
                per_repo[method].setdefault(row["repo"], {}).setdefault(key, []).append(value)
    summary = {"protocol": "swe_qa_v4", "questions": len(rows), "scored_questions": len(scored_ids),
               "questions_without_gold": sorted(set(ids) - set(scored_ids)), "silver_definition": "Python files named in reference answers",
               "chunks_per_query": CHUNKS_PER_QUERY, "bootstrap": {"resamples": BOOTSTRAPS, "seed": SEED}, "methods": {}, "paired": {}, "by_repo": {}}
    for method in METHODS:
        summary["methods"][method] = {key: {"mean": statistics.mean(values), "n": len(values)} for key, values in per_method[method].items()}
        summary["by_repo"][method] = {repo: {key: {"mean": statistics.mean(values), "n": len(values)} for key, values in metrics.items()}
                                       for repo, metrics in per_repo[method].items()}
    comparisons = (("hybrid", "b0"), ("hybrid", "semantic"), ("semantic", "lexical"), ("hybrid", "lexical"), ("semantic", "b0"), ("lexical", "b0"))
    for left, right in comparisons:
        result = {}
        for key in per_method[left]:
            diffs_by_repo = {repo: [a - b for a, b in zip(per_repo[left][repo][key], per_repo[right][repo][key])]
                             for repo in v4.MAIN_REPOS if repo in per_repo[left] and repo in per_repo[right]}
            diffs = [value for group in diffs_by_repo.values() for value in group]
            result[key] = {"difference_mean": statistics.mean(diffs), "ci95": bootstrap_stratified(diffs_by_repo),
                           "wilcoxon_p": wilcoxon_p(diffs), "n": len(diffs)}
        summary["paired"][f"{left}_minus_{right}"] = result
    return summary


def main() -> int:
    rows = load_rows()
    paths = {method: FROZEN / f"rankings_{method}.jsonl" for method in METHODS}
    existing = [path.exists() for path in paths.values()]
    if all(existing):
        for method, path in paths.items():
            count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            if count != len(rows):
                raise SystemExit(f"{method}: incomplete ranking file ({count}/{len(rows)})")
    elif any(existing):
        raise SystemExit("partial ranking artifact set exists; preserve it and investigate before retrying")
    else:
        services, provider = make_services()
        paths = write_rankings(rows, services, provider)
    summary = summarize(paths, rows)
    output = FROZEN / "retrieval_metrics.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "scored_questions": summary["scored_questions"], "methods": list(paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
