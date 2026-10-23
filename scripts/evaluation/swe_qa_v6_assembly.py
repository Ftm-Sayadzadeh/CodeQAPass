"""SWE-QA v6 development arm: v5a context assembly plus evidence-dense excerpts and a partial-answer prompt (no model call here).

v6 = v5 hub-free soft-focus assembly (retrieve 10, drop class hubs, focus matches first)
   + idea 1: long function/method chunks are cut to a question-relevant excerpt and more chunks (K=8) fit the same budget
   + idea 2: the system prompt answers supported parts and names what is missing, refusing only when nothing is relevant.

Trust boundary: every excerpt is a verbatim contiguous slice of an indexed function/method chunk; its file, symbol and
exact start/end lines come from chunk metadata plus line counting, never from model text. `class_overview` chunks are not
verbatim slices of the source, so they are never excerpted. Parameters are fixed constants below (not tuned per question).

    python scripts/evaluation/swe_qa_v6_assembly.py            # offline coverage report on the frozen v4 questions (no API calls)
"""

from __future__ import annotations

import json
import re
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v5_dev_context_gate as gate  # noqa: E402
from codecompass.qa.prompt import SYSTEM_PROMPT, QAPromptBuilder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_DEPTH = 10
CHUNKS_KEPT = 8  # chunks kept after hub removal (v5a keeps 5)
EXCERPT_CHARS = 1200  # chunks longer than this are excerpted (signature + best contiguous window)
SIGNATURE_CHARS = 300
STOP = {"the", "and", "that", "this", "with", "for", "how", "what", "does", "are", "from", "which", "when", "where", "why", "its",
        "into", "their", "than", "between", "through", "class", "method", "function"}
EXCERPTABLE = {"function", "method"}  # verbatim slices only

INSUFFICIENT_RULE = "If the context is insufficient, say that there is not enough retrieved evidence."
PARTIAL_RULE = (
    "If the context supports only part of the answer, answer the supported parts and state explicitly which parts cannot be "
    "established from the retrieved code. Say that there is not enough retrieved evidence only when none of the context is "
    "relevant to the question."
)
PARTIAL_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(INSUFFICIENT_RULE, PARTIAL_RULE)
assert PARTIAL_SYSTEM_PROMPT != SYSTEM_PROMPT, "base prompt changed: update the partial-answer rule"


class PartialPromptBuilder(QAPromptBuilder):
    """Same user prompt as production; only the insufficiency rule in the system prompt differs."""

    def build(self, question: str, context) -> tuple[str, str]:
        _, prompt = super().build(question, context)
        return PARTIAL_SYSTEM_PROMPT, prompt


def is_hub(piece: gate.Piece) -> bool:
    return piece.chunk_type == "class_overview" and len(piece.code) > gate.HUB_CHARS


def line_scores(lines: list[str], question: str) -> list[int]:
    """Per-line relevance: exact question identifiers weigh 3, shared identifier parts weigh 1."""
    exact = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question) if ("_" in t or re.search(r"[a-z][A-Z]", t)) and len(t) >= 4}
    parts = gate.words(question) - STOP
    scores = []
    for line in lines:
        tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)}
        scores.append(3 * len(exact & tokens) + len(gate.words(line) & parts))
    return scores


def signature_length(lines: list[str]) -> int:
    """Number of leading lines through the first line that ends the def/decorator header with a colon."""
    used = 0
    for number, line in enumerate(lines, 1):
        used += len(line) + 1
        if line.rstrip().endswith(":") or used > SIGNATURE_CHARS:
            return number
    return len(lines)


def excerpt(piece: gate.Piece, question: str) -> list[gate.Piece]:
    """Signature plus the contiguous window of lines with the highest relevance, as exact-range pieces."""
    if piece.chunk_type not in EXCERPTABLE or len(piece.code) <= EXCERPT_CHARS:
        return [piece]
    lines = piece.code.splitlines()
    if len(lines) != piece.end_line - piece.start_line + 1:
        return [piece]  # not a verbatim slice: never invent line numbers
    head = signature_length(lines)
    budget = EXCERPT_CHARS - sum(len(x) + 1 for x in lines[:head])
    body = list(range(head, len(lines)))
    scores = line_scores(lines, question)
    best, best_start, best_end = -1, head, head
    start, used, total = head, 0, 0
    for end in body:  # sliding window over lines under the character budget
        used += len(lines[end]) + 1
        total += scores[end]
        while used > budget and start <= end:
            used -= len(lines[start]) + 1
            total -= scores[start]
            start += 1
        if start <= end and total > best:
            best, best_start, best_end = total, start, end
    if best <= 0:  # nothing matches the question: keep the beginning of the body (docstring and first statements)
        best_start, best_end, used = head, head, 0
        for end in body:
            if used + len(lines[end]) + 1 > budget:
                break
            used += len(lines[end]) + 1
            best_end = end
    spans = [(0, head - 1)] if best_start > head else []
    spans.append((0 if best_start == head else best_start, best_end))
    pieces = []
    for first, last in spans:
        text = "\n".join(lines[first:last + 1])
        a, b = piece.start_line + first, piece.start_line + last
        pieces.append(gate.Piece(f"{piece.chunk_id}#L{a}-{b}", piece.source_file, piece.qualified_name, piece.symbol_name,
                                 piece.chunk_type, a, b, text, piece.rank, ()))
    return pieces


def v6_pieces(question: str, depth10: list[gate.Piece]) -> list[gate.Piece]:
    """Ordered candidate pieces for the v6 arm (before budget packing)."""
    hubless = [p for p in depth10 if not is_hub(p)][:CHUNKS_KEPT]
    matched = {p.chunk_id for p in gate.focus_matches(question, hubless)}
    ordered = [p for p in hubless if p.chunk_id in matched] + [p for p in hubless if p.chunk_id not in matched]
    return [piece for p in ordered for piece in excerpt(p, question)]


def parent_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("#L", 1)[0]


def main() -> None:
    questions = gate.question_rows()
    rankings = {r["question_id"]: r for r in gate.load_jsonl(gate.V4 / "rankings_hybrid.jsonl")}
    semantic = {r["question_id"]: r["ranking"] for r in gate.load_jsonl(gate.V4 / "rankings_semantic.jsonl")}
    lexical = {r["question_id"]: r["ranking"] for r in gate.load_jsonl(gate.V4 / "rankings_lexical.jsonl")}
    indexes = {repo: gate.Index(repo) for repo in gate.REPOS}
    rows: dict[str, list[dict]] = {}
    for qid, ranking in rankings.items():
        question, gold = questions[qid]["question"], set(ranking["gold_files"])
        idents = gate.identifiers(questions[qid].get("answer") or "")
        index = indexes[ranking["repo"]]
        ranked5, depth10 = gate.fuse(index, semantic[qid], lexical[qid], 5), gate.fuse(index, semantic[qid], lexical[qid], 10)
        built = gate.variants(question, ranked5, depth10)
        candidates = {"a1_cur": built["baseline"], "v5a": built["hubless_soft_k5"], "v5b": built["hubless_soft_k10"],
                      "v6": v6_pieces(question, depth10)}
        for name, pieces in candidates.items():
            chosen, omitted = gate.pack(pieces)
            rows.setdefault(name, []).append({"question_id": qid, "repo": ranking["repo"], **gate.metrics(chosen, omitted, gold, idents)})
    summary = {name: gate.summarize(r) for name, r in rows.items()}
    by_repo = {name: {repo: gate.summarize([x for x in r if x["repo"] == repo]) for repo in gate.REPOS} for name, r in rows.items()}
    base = {r["question_id"]: r for r in rows["v5a"]}
    paired = {}
    for key in ("gold_cov", "ident_cov"):
        deltas = [r[key] - base[r["question_id"]][key] for r in rows["v6"] if r[key] is not None and base[r["question_id"]][key] is not None]
        paired[key] = {"mean_delta_vs_v5a": round(stats.mean(deltas), 4), "ci95": gate.bootstrap_ci(deltas),
                       "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas)}
    print(json.dumps({"constants": {"depth": RETRIEVAL_DEPTH, "chunks_kept": CHUNKS_KEPT, "excerpt_chars": EXCERPT_CHARS, "budget": gate.BUDGET},
                      "offline_v4_questions": summary, "by_repo": by_repo, "v6_vs_v5a": paired}, indent=1))


if __name__ == "__main__":
    main()
