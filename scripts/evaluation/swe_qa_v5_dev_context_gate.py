"""Offline context-assembly gate for SWE-QA v5 development (no model, no network).

Replays the frozen v4 hybrid rankings against the frozen v4 SQLite indexes and compares how much gold evidence
actually reaches the generation prompt under the current context assembly and under pre-declared alternatives.
Nothing under reports/evaluation/swe_qa_v4/ or data/ is modified; output is written once to a new directory.

    python scripts/evaluation/swe_qa_v5_dev_context_gate.py            # print summary
    python scripts/evaluation/swe_qa_v5_dev_context_gate.py --write    # also write the gate JSON (refuses to overwrite)

Metrics are diagnostics against the benchmark's gold files and reference-answer identifiers. They are never used
at retrieval or generation time, and file paths/line ranges in every variant come from index metadata.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sqlite3
import statistics as stats
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "reports/evaluation/swe_qa_v4/frozen"
INDEX_ROOT = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2/index"
B0_ROOT = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2/b0"
DATASET = ROOT / "data/benchmarks/swe_qa_v4/dataset"
OUT = ROOT / "reports/evaluation/swe_qa_v5_dev/offline_gate"
REPOS = ("flask", "requests", "pytest", "pylint")
BUDGET = 6000  # same character budget as the frozen v4 A1 arm
HUB_CHARS = 2400  # a class chunk above this is replaced by an outline (pre-declared, not tuned)
OUTLINE_CHARS = 1400
METHODS_PER_HUB = 2
PASS_COVERAGE = 0.62  # pre-declared gate: delivered gold-file coverage must reach this
IDENT_SKIP = {"self", "cls", "true", "false", "none", "str", "int", "dict", "list", "bool", "type"}


@dataclass(frozen=True)
class Piece:
    """One retrieved chunk (or an outline derived from one) with metadata-sourced identity."""

    chunk_id: str
    source_file: str
    qualified_name: str | None
    symbol_name: str | None
    chunk_type: str
    start_line: int
    end_line: int
    code: str
    rank: int
    covered: tuple[tuple[int, int], ...] = ()


def mentions_symbol(text: str, qualified_name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(qualified_name)}(?!\w)", text, flags=re.IGNORECASE) is not None


def mentions_code_name(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", text) is not None


def focus_matches(question: str, pieces: list[Piece]) -> list[Piece]:
    """Pieces selected by the production `_focus_on_named_symbol` rules (exact, class root, symbol name)."""
    exact = {p.qualified_name for p in pieces if p.qualified_name and "." in p.qualified_name and mentions_symbol(question, p.qualified_name)}
    if exact:
        return [p for p in pieces if p.qualified_name in exact]
    roots = {p.qualified_name.split(".", 1)[0] for p in pieces if p.qualified_name and mentions_code_name(question, p.qualified_name.split(".", 1)[0])}
    if roots:
        return [p for p in pieces if p.qualified_name and p.qualified_name.split(".", 1)[0] in roots]
    symbols = {p.symbol_name for p in pieces if p.symbol_name and not p.symbol_name.startswith("__") and mentions_code_name(question, p.symbol_name)}
    if symbols:
        return [p for p in pieces if p.symbol_name in symbols]
    return []


def words(text: str) -> set[str]:
    """Lower-case identifier parts (snake_case and camelCase split) of length >= 3."""
    parts: set[str] = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for piece in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", token):
            if len(piece) >= 3:
                parts.add(piece.lower())
    return parts


def outline(piece: Piece, question: str) -> Piece | None:
    """Deterministic AST outline of a large class chunk: signature, docstring, class attributes, method signatures,
    plus the most question-relevant method bodies. Returns None if the chunk cannot be parsed."""
    source = textwrap.dedent(piece.code)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if node is None:
        return None
    lines = source.splitlines()
    first_body = node.body[0].lineno if node.body else node.lineno + 1
    head = lines[node.lineno - 1:first_body - 1] if first_body > node.lineno else [lines[node.lineno - 1]]
    out: list[str] = list(head)
    doc = ast.get_docstring(node)
    if doc:
        out.append(f'    """{doc.splitlines()[0][:200]}"""')
    qwords = words(question)
    scored: list[tuple[int, int, ast.AST]] = []
    for child in node.body:
        if isinstance(child, (ast.Assign, ast.AnnAssign)) and (child.end_lineno or child.lineno) - child.lineno < 3:
            out.append(lines[child.lineno - 1])
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([child.lineno] + [d.lineno for d in child.decorator_list])
            body_start = child.body[0].lineno
            signature = lines[start - 1:body_start - 1] if body_start > start else [lines[child.lineno - 1]]
            out.extend(signature + ["        ..."])
            overlap = len(qwords & words(child.name + " " + (ast.get_docstring(child) or "")))
            if overlap:
                scored.append((-overlap, start, child))
    text = "\n".join(out)
    if len(text) > OUTLINE_CHARS:
        text = text[:OUTLINE_CHARS].rsplit("\n", 1)[0] + "\n    # ... (outline truncated)"
    covered: list[tuple[int, int]] = []
    budget = HUB_CHARS - len(text)
    chosen = 0
    for _, start, child in sorted(scored, key=lambda item: (item[0], item[1])):
        body = "\n".join(lines[start - 1:child.end_lineno])
        if chosen >= METHODS_PER_HUB or len(body) + 1 > budget:
            continue
        text += "\n\n" + body
        budget -= len(body) + 2
        chosen += 1
        covered.append((piece.start_line + start - 1, piece.start_line + (child.end_lineno or start) - 1))
    return Piece(piece.chunk_id + "#outline", piece.source_file, piece.qualified_name, piece.symbol_name, "class_outline",
                 piece.start_line, piece.end_line, text, piece.rank, tuple(covered))


def pack(pieces: list[Piece]) -> tuple[list[Piece], int]:
    """Greedy packing identical to RAGContextBuilder (skip what does not fit); also drops pieces that a chosen
    piece already covers entirely. Returns chosen pieces and the number omitted for size."""
    chosen: list[Piece] = []
    used = omitted = 0
    for piece in pieces:
        if any(c.chunk_id == piece.chunk_id for c in chosen):
            continue
        if any(c.source_file == piece.source_file and any(s <= piece.start_line and piece.end_line <= e for s, e in c.covered) for c in chosen):
            continue
        if used + len(piece.code) > BUDGET:
            omitted += 1
            continue
        chosen.append(piece)
        used += len(piece.code)
    return chosen, omitted


class Index:
    def __init__(self, repo: str) -> None:
        connection = sqlite3.connect(f"file:{INDEX_ROOT / (repo + '.sqlite')}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        self.chunks = {
            row["chunk_id"]: row for row in connection.execute(
                "SELECT c.chunk_id, c.chunk_type, c.start_line, c.end_line, c.code, sf.relative_path, "
                "s.name AS symbol_name, s.qualified_name FROM chunks c JOIN source_files sf ON sf.id = c.file_id "
                "LEFT JOIN symbols s ON s.id = c.symbol_id")
        }
        connection.close()

    def piece(self, chunk_id: str, rank: int) -> Piece:
        row = self.chunks[chunk_id]
        return Piece(chunk_id, row["relative_path"], row["qualified_name"], row["symbol_name"], row["chunk_type"],
                     row["start_line"], row["end_line"], row["code"], rank)


def fuse(index: Index, semantic: list[dict], lexical: list[dict], limit: int, rrf_k: int = 60) -> list[Piece]:
    """Exact offline replay of HybridRetriever at a given retrieval limit (each list contributes its first `limit` hits)."""
    scores: dict[str, float] = {}
    for results in (semantic[:limit], lexical[:limit]):
        for rank, item in enumerate(results, 1):
            scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
    pieces = [index.piece(chunk_id, 0) for chunk_id in scores]
    pieces.sort(key=lambda p: (-scores[p.chunk_id], p.source_file, p.start_line, p.chunk_id))
    return [Piece(**{**p.__dict__, "rank": rank}) for rank, p in enumerate(pieces[:limit], 1)]


def variants(question: str, ranked: list[Piece], depth10: list[Piece]) -> dict[str, list[Piece]]:
    """Candidate pieces, in packing order, for every declared variant."""

    def expand(items: list[Piece]) -> list[Piece]:
        out: list[Piece] = []
        for p in items:
            if p.chunk_type == "class_overview" and len(p.code) > HUB_CHARS:
                replacement = outline(p, question)
                out.append(replacement or p)
            else:
                out.append(p)
        return out

    def soft(items: list[Piece]) -> list[Piece]:
        matched = {p.chunk_id for p in focus_matches(question, items)}
        return [p for p in items if p.chunk_id in matched] + [p for p in items if p.chunk_id not in matched]

    hard = focus_matches(question, ranked) or ranked
    hubless = [p for p in depth10 if not (p.chunk_type == "class_overview" and len(p.code) > HUB_CHARS)]
    return {
        "baseline": hard,
        "nofocus": ranked,
        "soft_focus": soft(ranked),
        "outline": expand(hard),
        "outline_soft": expand(soft(ranked)),
        "outline_soft_k10": expand(soft(depth10)),
        "hubless_soft_k5": soft(hubless[:5]),  # exploratory: retrieve 10, drop undeliverable hubs, keep 5
        "hubless_soft_k10": soft(hubless),
    }


def identifiers(reference: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"`([^`\n]+)`", reference):
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", token.replace("()", "")):
            if len(name) >= 4 and name.lower() not in IDENT_SKIP:
                out.add(name.lower())
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for repo in REPOS:
        for number, row in enumerate(load_jsonl(DATASET / f"{repo}.jsonl"), 1):
            rows[f"{repo}-{number}"] = {"repo": repo, **row}
    return rows


def b0_text(repo: str, chunks: list[dict], ranking: list[dict]) -> tuple[list[str], set[str]]:
    """B0-k5 delivered text and files (dense top-5, same greedy 6000-char budget)."""
    used = 0
    texts: list[str] = []
    files: set[str] = set()
    for item in ranking[:5]:
        chunk = chunks[int(item["chunk_id"].rsplit("-", 1)[1])]
        if used + len(chunk["text"]) > BUDGET:
            continue
        used += len(chunk["text"])
        texts.append(chunk["text"])
        files.add(chunk["file"])
    return texts, files


def metrics(pieces: list[Piece], omitted: int, gold: set[str], idents: set[str]) -> dict:
    files = {p.source_file for p in pieces}
    text = "\n".join(p.code for p in pieces).lower()
    return {
        "gold_cov": len(files & gold) / len(gold) if gold else None,
        "gold_hit": bool(files & gold) if gold else None,
        "ident_cov": sum(name in text for name in idents) / len(idents) if idents else None,
        "blocks": len(pieces),
        "single_block": len(pieces) <= 1,
        "chars": sum(len(p.code) for p in pieces),
        "omitted": omitted,
    }


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return round(stats.mean(values), 4) if values else None

    return {"n": len(rows), "gold_cov": mean("gold_cov"), "gold_hit": mean("gold_hit"), "ident_cov": mean("ident_cov"),
            "blocks": mean("blocks"), "single_block_rate": mean("single_block"), "chars": mean("chars"), "omitted": mean("omitted")}


def bootstrap_ci(deltas: list[float], seed: int = 20260930, resamples: int = 5000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(stats.mean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples)], 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for output files, so a later run never overwrites an earlier one")
    args = parser.parse_args()
    questions = question_rows()
    rankings = {r["question_id"]: r for r in load_jsonl(V4 / "rankings_hybrid.jsonl")}  # supplies gold files only
    semantic = {r["question_id"]: r["ranking"] for r in load_jsonl(V4 / "rankings_semantic.jsonl")}
    lexical = {r["question_id"]: r["ranking"] for r in load_jsonl(V4 / "rankings_lexical.jsonl")}
    b0_rankings = {r["question_id"]: r for r in load_jsonl(V4 / "rankings_b0.jsonl")}
    a1_records = {r["question_id"]: r for r in load_jsonl(V4 / "main_a1_combined.jsonl")}
    indexes = {repo: Index(repo) for repo in REPOS}
    b0_chunks = {repo: load_jsonl(B0_ROOT / f"{repo}_chunks.jsonl") for repo in REPOS}
    per_variant: dict[str, list[dict]] = {}
    replica_match = replica_total = 0
    for qid, ranking in rankings.items():
        repo, question = ranking["repo"], questions[qid]["question"]
        gold = set(ranking["gold_files"])
        idents = identifiers(questions[qid].get("answer") or "")
        index = indexes[repo]
        built = variants(question, fuse(index, semantic[qid], lexical[qid], 5), fuse(index, semantic[qid], lexical[qid], 10))
        for name, candidates in built.items():
            chosen, omitted = pack(candidates)
            per_variant.setdefault(name, []).append({"question_id": qid, "repo": repo, **metrics(chosen, omitted, gold, idents)})
            if name == "baseline":
                actual = {c["chunk_id"] for c in a1_records[qid]["citations"]}
                replica_total += 1
                replica_match += actual == {p.chunk_id for p in chosen}
        texts, files = b0_text(repo, b0_chunks[repo], b0_rankings[qid]["ranking"])
        lowered = "\n".join(texts).lower()
        per_variant.setdefault("b0_k5_reference", []).append({
            "question_id": qid, "repo": repo, "gold_cov": len(files & gold) / len(gold) if gold else None,
            "gold_hit": bool(files & gold) if gold else None,
            "ident_cov": sum(n in lowered for n in idents) / len(idents) if idents else None,
            "blocks": len(texts), "single_block": len(texts) <= 1, "chars": sum(map(len, texts)), "omitted": 0})
    summary = {name: summarize(rows) for name, rows in per_variant.items()}
    by_repo = {name: {repo: summarize([r for r in rows if r["repo"] == repo]) for repo in REPOS} for name, rows in per_variant.items()}
    base = {r["question_id"]: r for r in per_variant["baseline"]}
    paired: dict[str, dict] = {}
    for name, rows in per_variant.items():
        if name in ("baseline", "b0_k5_reference"):
            continue
        for key in ("gold_cov", "ident_cov"):
            deltas = [r[key] - base[r["question_id"]][key] for r in rows if r[key] is not None and base[r["question_id"]][key] is not None]
            paired.setdefault(name, {})[key] = {"mean_delta": round(stats.mean(deltas), 4), "ci95": bootstrap_ci(deltas),
                                                "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas), "n": len(deltas)}
    best = max((n for n in summary if n not in ("baseline", "b0_k5_reference")), key=lambda n: summary[n]["gold_cov"])
    verdict = {"rule": f"gold_cov >= {PASS_COVERAGE} and ident_cov not below baseline (pre-declared)", "best_variant": best,
               "passed": summary[best]["gold_cov"] >= PASS_COVERAGE and summary[best]["ident_cov"] >= summary["baseline"]["ident_cov"]}
    report = {"budget_chars": BUDGET, "hub_chars": HUB_CHARS, "outline_chars": OUTLINE_CHARS, "methods_per_hub": METHODS_PER_HUB,
              "replica_exact_match": f"{replica_match}/{replica_total}", "summary": summary, "by_repo": by_repo,
              "paired_vs_baseline": paired, "verdict": verdict}
    print(json.dumps(report, indent=1))
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""
        target = OUT / f"gate_summary{suffix}.json"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        target.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        (OUT / f"gate_rows{suffix}.json").write_text(json.dumps(per_variant) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
