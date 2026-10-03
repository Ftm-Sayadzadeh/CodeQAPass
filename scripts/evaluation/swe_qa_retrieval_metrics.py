"""File-level retrieval metrics (Hit/Recall/Precision@k, MRR) for the SWE-QA v2 questions.

Gold files are the repository Python files named in each SWE-QA reference answer. Retrieval returns chunks; a
question's ranking is the list of distinct files in chunk order. No LLM is called; the semantic and hybrid
methods embed each question once per method.

    python scripts/evaluation/swe_qa_retrieval_metrics.py retrieve [--methods lexical ...]   # raw rankings; reruns only append missing questions
    python scripts/evaluation/swe_qa_retrieval_metrics.py metrics    # reads raw rankings, writes metrics JSON
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swe_qa_v2 as run  # noqa: E402
from codecompass.retrieval import RetrievalQuery  # noqa: E402
from codecompass.retrieval.models import RetrievalError  # noqa: E402

OUT = run.OUT / "retrieval_metrics"
METHODS = ("lexical", "semantic", "hybrid")
CHUNKS_PER_QUERY = 20
KS = (1, 3, 5, 10)
PATH_RE = re.compile(r"(?<![\w/.-])([\w./-]+\.py)\b")


def repo_files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if ".git" not in p.parts)


def gold_files(answer: str, files: list[str]) -> list[str]:
    """Existing repository files named in a reference answer (exact, src/-prefixed, or unique suffix match)."""
    known = set(files)
    gold: set[str] = set()
    for raw in PATH_RE.findall(answer):
        path = raw.lstrip("./")
        if path in known:
            gold.add(path)
        elif f"src/{path}" in known:
            gold.add(f"src/{path}")
        else:
            matches = [f for f in files if f == path or f.endswith(f"/{path}")]
            if len(matches) == 1:
                gold.add(matches[0])
    return sorted(gold)


def raw_path(method: str) -> Path:
    return OUT / f"rankings_{method}.jsonl"


def retrieve(methods: tuple[str, ...]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sum((run.load_rows(path, repo) for repo, (path, *_rest) in run.REPOS.items()), [])
    services = {repo: run.build_services(repo) for repo in run.REPOS}
    files = {repo: repo_files(cfg[1]) for repo, cfg in run.REPOS.items()}
    for method in methods:
        retrieve_method(method, rows, services, files)


def retrieve_method(method: str, rows: list[dict], services: dict, files: dict) -> None:
    """Append rankings for questions not yet recorded; completed records are never rewritten."""
    path = raw_path(method)
    done = {json.loads(line)["question_id"] for line in path.open(encoding="utf-8") if line.strip()} if path.exists() else set()
    with path.open("a", encoding="utf-8") as target:
        for row in rows:
            if row["question_id"] in done:
                continue
            service, project_id = services[row["repo"]]
            query = RetrievalQuery(row["question"], project_id, CHUNKS_PER_QUERY)
            for attempt in range(1, 5):
                try:
                    result = getattr(service, f"search_{method}")(query)
                    break
                except RetrievalError:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            record = {"question_id": row["question_id"], "repo": row["repo"],
                      "gold_files": gold_files(row["answer"], files[row["repo"]]),
                      "ranking": [{"chunk_id": c.chunk_id, "file": c.source_file, "start_line": c.start_line, "end_line": c.end_line}
                                  for c in result.results]}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(method, row["question_id"], flush=True)


def distinct_files(ranking: list[dict]) -> list[str]:
    seen: list[str] = []
    for item in ranking:
        if item["file"] not in seen:
            seen.append(item["file"])
    return seen


def question_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in KS:
        top = ranked[:k]
        hits = len(gold.intersection(top))
        out[f"hit@{k}"] = float(hits > 0)
        out[f"recall@{k}"] = hits / len(gold)
        out[f"precision@{k}"] = hits / len(top) if top else 0.0
    first = next((i for i, f in enumerate(ranked[:10], 1) if f in gold), None)
    out["mrr@10"] = 1.0 / first if first else 0.0
    return out


def bootstrap(values: list[float], seed: int = 20260930, n: int = 20000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def wilcoxon_p(diffs: list[float]) -> float | None:
    nz = [d for d in diffs if d]
    n = len(nz)
    if n < 10:
        return None
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    ties, i = 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (abs(w_plus - n * (n + 1) / 4) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2))


def metrics() -> None:
    available = tuple(m for m in METHODS if raw_path(m).exists())
    by_method = {m: {r["question_id"]: r for r in map(json.loads, raw_path(m).open(encoding="utf-8"))} for m in available}
    records = list(by_method[available[0]].values())
    scored = [r for r in records if r["gold_files"]]
    per: dict[str, dict[str, list[float]]] = {m: {} for m in available}
    for r in scored:
        gold = set(r["gold_files"])
        for method in available:
            ranking = by_method[method][r["question_id"]]["ranking"]
            for key, value in question_metrics(distinct_files(ranking), gold).items():
                per[method].setdefault(key, []).append(value)
    summary: dict = {"questions": len(records), "questions_without_gold": [r["question_id"] for r in records if not r["gold_files"]],
                     "scored_questions": len(scored), "chunks_per_query": CHUNKS_PER_QUERY,
                     "definition": "file-level: ranking = distinct files in chunk order; gold = repo .py files named in the reference answer",
                     "methods": {}, "paired": {}, "by_repo": {}}
    for method in available:
        summary["methods"][method] = {key: {"mean": stats.mean(v), "ci95": bootstrap(v)} for key, v in per[method].items()}
    for a, b in (("hybrid", "lexical"), ("hybrid", "semantic"), ("semantic", "lexical")):
        if a not in per or b not in per:
            continue
        summary["paired"][f"{a}_minus_{b}"] = {}
        for key in per[a]:
            diffs = [x - y for x, y in zip(per[a][key], per[b][key])]
            summary["paired"][f"{a}_minus_{b}"][key] = {"mean_diff": stats.mean(diffs), "ci95": bootstrap(diffs), "wilcoxon_p": wilcoxon_p(diffs)}
    for repo in run.REPOS:
        idx = [i for i, r in enumerate(scored) if r["repo"] == repo]
        summary["by_repo"][repo] = {m: {k: stats.mean(per[m][k][i] for i in idx) for k in ("hit@5", "recall@5", "precision@5", "mrr@10")} for m in available}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({m: {k: round(v["mean"], 3) for k, v in summary["methods"][m].items()} for m in available}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("retrieve", "metrics"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    retrieve(tuple(args.methods)) if args.phase == "retrieve" else metrics()
