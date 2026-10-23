"""SWE-QA v7 preflight (offline, no model or embedding call): subsets, Study 1 contexts, Study 3 retrieval and delivered-evidence metrics.

    python scripts/evaluation/swe_qa_v7_preflight.py

Reads only frozen v4 artifacts and the v4 indexes; writes new files under reports/evaluation/swe_qa_v7/preflight/ and refuses to overwrite them.
See reports/evaluation/swe_qa_v7/PROTOCOL_V7_DRAFT.md (sections 2, 4, 6, 10).
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v5_confirm as c5  # noqa: E402
import swe_qa_v5_dev_context_gate as gate  # noqa: E402
import swe_qa_v6_assembly as v6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/swe_qa_v7/preflight_final"  # the earlier `preflight/` folder is superseded (it lacked the chunk-depth retrieval metrics)
V4 = gate.V4
REPOS = gate.REPOS
SUBSET_SEED = "20261022"
BOOTSTRAP_SEED = 20260930
RESAMPLES = 20_000
PER_REPO_SUBSET = 12
ARMS = ("a1_v4", "a1_v5a", "a1_v6e", "b0_k5")


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def study2_subset() -> dict[str, list[str]]:
    """12 question ids per repository, fixed seed, drawn before any v7 run."""
    return {repo: [f"{repo}-{n}" for n in sorted(random.Random(f"{SUBSET_SEED}:{repo}").sample(range(1, 49), PER_REPO_SUBSET))] for repo in REPOS}


# ---------------------------------------------------------------- retrieval metrics (file level, chunk depth)
def distinct_files(chunks: list[dict], depth: int) -> list[str]:
    out: list[str] = []
    for item in chunks[:depth]:
        if item["file"] not in out:
            out.append(item["file"])
    return out


def retrieval_row(ranking: list[dict], gold: set[str]) -> dict:
    f10 = distinct_files(ranking, 10)  # the v4 convention: the ranking is the distinct files of the first 10 chunks; Recall@5 and Hit@5 use its first 5 files
    f5 = f10[:5]
    first = next((i for i, f in enumerate(f10, 1) if f in gold), None)
    chunk5_files = {item["file"] for item in ranking[:5]}  # equal chunk depth: files inside the first 5 chunks
    return {"recall5": len(set(f5) & gold) / len(gold), "recall10": len(set(f10) & gold) / len(gold), "hit5": float(bool(set(f5) & gold)),
            "mrr10": 1 / first if first else 0.0, "recall5_chunks": len(chunk5_files & gold) / len(gold), "hit5_chunks": float(bool(chunk5_files & gold))}


def bootstrap(groups: dict[str, list[float]], resamples: int = RESAMPLES) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(stats.mean(x for g in groups.values() for x in rng.choices(g, k=len(g))) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)]


def summarize(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["repo"], []).append(r[key])
    return {"mean": round(stats.mean(r[key] for r in rows), 4), "ci95": bootstrap(groups), "n": len(rows)}


def paired(a: list[dict], b: list[dict], key: str) -> dict:
    index = {r["question_id"]: r for r in b}
    groups: dict[str, list[float]] = {}
    for r in a:
        groups.setdefault(r["repo"], []).append(r[key] - index[r["question_id"]][key])
    deltas = [d for g in groups.values() for d in g]
    return {"mean_difference": round(stats.mean(deltas), 4), "ci95": bootstrap(groups), "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas), "n": len(deltas)}


def main() -> int:
    questions: dict[str, dict] = {}
    for repo in REPOS:
        for number, row in enumerate(jl(gate.DATASET / f"{repo}.jsonl"), 1):
            questions[f"{repo}-{number}"] = {"repo": repo, **row}
    ranks = {m: {r["question_id"]: r for r in jl(V4 / f"rankings_{m}.jsonl")} for m in ("semantic", "lexical", "hybrid", "b0")}
    a1_records = {r["question_id"]: r for r in jl(V4 / "main_a1_combined.jsonl")}
    indexes = {repo: gate.Index(repo) for repo in REPOS}
    metas = {repo: jl(gate.B0_ROOT / f"{repo}_chunks.jsonl") for repo in REPOS}
    contexts: list[dict] = []
    coverage: dict[str, list[dict]] = {a: [] for a in ARMS}
    retrieval: dict[str, list[dict]] = {k: [] for k in ("lexical", "semantic", "hybrid_v4", "hybrid_k5", "hybrid_k10", "b0")}
    replica_ok = 0
    for qid, q in questions.items():
        repo, question = q["repo"], q["question"]
        gold = set(ranks["hybrid"][qid]["gold_files"])
        ids = gate.identifiers(q.get("answer") or "")
        sem, lex = ranks["semantic"][qid]["ranking"], ranks["lexical"][qid]["ranking"]
        ranked5, depth10 = gate.fuse(indexes[repo], sem, lex, 5), gate.fuse(indexes[repo], sem, lex, 10)
        built = gate.variants(question, ranked5, depth10)
        packed = {"a1_v4": gate.pack(built["baseline"]), "a1_v5a": gate.pack(built["hubless_soft_k5"]), "a1_v6e": gate.pack(v6.v6_pieces(question, depth10))}
        blocks_by_arm = {arm: (c5.blocks_from_pieces(chosen), omitted) for arm, (chosen, omitted) in packed.items()}
        blocks_by_arm["b0_k5"] = c5.b0_arm_context(q, ranks["b0"][qid]["ranking"], metas[repo], 5)
        replica_ok += [b["chunk_id"] for b in blocks_by_arm["a1_v4"][0]] == [c["chunk_id"] for c in a1_records[qid]["citations"]]
        for arm, (blocks, omitted) in blocks_by_arm.items():
            text = "\n".join(b["code"] for b in blocks)
            contexts.append({"question_id": qid, "arm": arm, "blocks": blocks, "omitted_context_count": omitted,
                             "context_sha256": hashlib.sha256(text.encode()).hexdigest()})
            low = text.lower()
            coverage[arm].append({"question_id": qid, "repo": repo, "gold_cov": len({b["source_file"] for b in blocks} & gold) / len(gold) if gold else None,
                                  "ident_cov": sum(n in low for n in ids) / len(ids) if ids else None, "blocks": len(blocks), "chars": len(text), "omitted": omitted})
        if gold:
            for key, ranking in (("lexical", lex), ("semantic", sem), ("hybrid_v4", ranks["hybrid"][qid]["ranking"]), ("hybrid_k5", [{"file": p.source_file} for p in ranked5]),
                                 ("hybrid_k10", [{"file": p.source_file} for p in depth10]), ("b0", ranks["b0"][qid]["ranking"])):
                retrieval[key].append({"question_id": qid, "repo": repo, **retrieval_row(ranking, gold)})
    subset = study2_subset()
    human = json.loads((ROOT / "reports/evaluation/swe_qa_v4/SAMPLE_FREEZE.json").read_text(encoding="utf-8"))["human_review_sample"]["ids_in_order"]
    study3 = {"replica_a1_v4_matches_frozen": f"{replica_ok}/{len(questions)}",
              "retrieval": {k: {m: summarize(rows, m) for m in ("recall5", "recall10", "hit5", "mrr10", "recall5_chunks", "hit5_chunks")} for k, rows in retrieval.items()},
              "retrieval_paired_hybrid_v4_minus_b0": {m: paired(retrieval["hybrid_v4"], retrieval["b0"], m) for m in ("recall5", "mrr10")},
              "retrieval_paired_hybrid_k10_minus_b0": {m: paired(retrieval["hybrid_k10"], retrieval["b0"], m) for m in ("recall5", "mrr10")},
              "retrieval_paired_chunk_depth_hybrid_k5_minus_b0": {m: paired(retrieval["hybrid_k5"], retrieval["b0"], m) for m in ("recall5_chunks", "hit5_chunks")},
              "delivered_evidence": {arm: {m: summarize([r for r in rows if r[m] is not None], m) for m in ("gold_cov", "ident_cov", "blocks", "chars", "omitted")} for arm, rows in coverage.items()},
              "delivered_paired_vs_b0": {arm: {m: paired([r for r in coverage[arm] if r[m] is not None], [r for r in coverage["b0_k5"] if r[m] is not None], m)
                                               for m in ("gold_cov", "ident_cov")} for arm in ("a1_v4", "a1_v5a", "a1_v6e")}}
    write_once(OUT / "subsets.json", json.dumps({"seed": SUBSET_SEED, "study2_subset": subset, "study2_total": sum(map(len, subset.values())),
                                                  "human_review_sample": human}, indent=1) + "\n")
    write_once(OUT / "contexts_study1.jsonl", "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in contexts))
    write_once(OUT / "study3_offline.json", json.dumps(study3, indent=1) + "\n")
    print(json.dumps(study3, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
