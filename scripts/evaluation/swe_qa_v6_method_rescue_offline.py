"""Offline method-rescue gate on the frozen SWE-QA v6 rankings.

This is a development diagnostic only. It does not call a model, alter frozen
records, or change the production QA path. It compares v5a with the existing
deterministic class-outline/method-body rescue variant on the v6 questions.
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v5_dev_context_gate as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V6 = ROOT / "reports/evaluation/swe_qa_v6/fresh"
V6_DATASET = ROOT / "data/benchmarks/swe_qa_v6/dataset"
V6_INDEX = ROOT / "data/benchmarks/swe_qa_v6_avalai_gemini2/index"
V6_REPOS = ROOT / "data/benchmarks/swe_qa_v6/repos"
OUT = ROOT / "reports/evaluation/swe_qa_v6/dev_method_rescue"
REPOS = ("streamlink", "sqlfluff")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for repo in REPOS:
        for number, row in enumerate(read_jsonl(V6_DATASET / f"{repo}.jsonl"), 1):
            result[f"{repo}-{number}"] = {"repo": repo, **row}
    return result


def repo_files(repo: str) -> list[str]:
    root = V6_REPOS / repo
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if ".git" not in p.parts)


def gold_files(answer: str, files: list[str]) -> set[str]:
    # Use the benchmark's established answer-to-file extraction, not a new rule.
    from swe_qa_retrieval_metrics import gold_files as extract

    return set(extract(answer, files))


def summarize(items: list[dict]) -> dict:
    keys = ("gold_cov", "ident_cov", "blocks", "chars", "omitted")
    return {"n": len(items), **{
        key: round(stats.mean(x[key] for x in items if x[key] is not None), 4)
        if any(x[key] is not None for x in items) else None
        for key in keys
    }}


def main() -> None:
    # Reuse the exact v5/v6 assembly and metadata lookup code with v6 inputs.
    gate.DATASET = V6_DATASET
    gate.INDEX_ROOT = V6_INDEX
    gate.REPOS = REPOS
    question_rows = rows()
    semantic = {r["question_id"]: r["ranking"] for r in read_jsonl(V6 / "rankings_semantic.jsonl")}
    lexical = {r["question_id"]: r["ranking"] for r in read_jsonl(V6 / "rankings_lexical.jsonl")}
    indexes = {repo: gate.Index(repo) for repo in REPOS}
    file_cache = {repo: repo_files(repo) for repo in REPOS}
    records: dict[str, list[dict]] = {"v5a": [], "method_rescue_k10": []}

    for qid, row in question_rows.items():
        ranked10 = gate.fuse(indexes[row["repo"]], semantic[qid], lexical[qid], 10)
        built = gate.variants(row["question"], gate.fuse(indexes[row["repo"]], semantic[qid], lexical[qid], 5), ranked10)
        candidates = {"v5a": built["hubless_soft_k5"], "method_rescue_k10": built["outline_soft_k10"]}
        gold = gold_files(row.get("answer") or "", file_cache[row["repo"]])
        ids = gate.identifiers(row.get("answer") or "")
        for name, pieces in candidates.items():
            chosen, omitted = gate.pack(pieces)
            records[name].append({"question_id": qid, "repo": row["repo"], **gate.metrics(chosen, omitted, gold, ids)})

    summary = {name: summarize(items) for name, items in records.items()}
    base = {item["question_id"]: item for item in records["v5a"]}
    paired = {}
    for key in ("gold_cov", "ident_cov", "blocks", "chars", "omitted"):
        deltas = [item[key] - base[item["question_id"]][key] for item in records["method_rescue_k10"]
                  if item[key] is not None and base[item["question_id"]][key] is not None]
        paired[key] = {"mean_delta": round(stats.mean(deltas), 4), "better": sum(x > 0 for x in deltas),
                       "worse": sum(x < 0 for x in deltas), "n": len(deltas)}
    report = {
        "questions": len(question_rows),
        "constants": {"retrieval_depth": 10, "methods_per_hub": gate.METHODS_PER_HUB,
                      "hub_chars": gate.HUB_CHARS, "outline_chars": gate.OUTLINE_CHARS,
                      "budget_chars": gate.BUDGET},
        "summary": summary,
        "method_rescue_k10_vs_v5a": paired,
        "gate": {
            "rule": "method rescue must improve both gold_cov and ident_cov; generation is required before adoption",
            "offline_pass": summary["method_rescue_k10"]["gold_cov"] > summary["v5a"]["gold_cov"]
            and summary["method_rescue_k10"]["ident_cov"] > summary["v5a"]["ident_cov"],
        },
    }
    print(json.dumps(report, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "method_rescue_summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (OUT / "method_rescue_rows.json").write_text(json.dumps(records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
