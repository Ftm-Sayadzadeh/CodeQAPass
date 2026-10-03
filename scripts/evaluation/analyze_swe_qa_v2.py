"""Validate SWE-QA v2 citations and run the official blind LLM judge."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import time
from pathlib import Path

from codecompass.llm import LLMRequest
from codecompass.providers import ProviderConfig, create_llm_provider

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/swe_qa_v2"
REPO_ROOTS = {"flask": ROOT / "data/benchmarks/flask_repo", "requests": ROOT / "data/benchmarks/requests_repo"}
ARMS = ("a0_closed_book", "a0_neutral", "a1_codeqapass", "a2_model_citations")
JUDGED_ARMS = ("a0_neutral", "a1_codeqapass")
INVARIANT_ARMS = {"a1_codeqapass"}  # citations are the retrieved blocks themselves; zero errors holds by construction
CLOSED_BOOK_ARMS = {"a0_closed_book", "a0_neutral"}
REFUSAL_RE = re.compile(r"(insufficient|does not (contain|include|provide)|not (provided|included|supplied|present)|can't|cannot|no code)", re.I)
CITATION_RE = re.compile(r"[\w./-]+\.py:\d+(?:-\d+)?")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def arm_rows(arm: str) -> list[dict]:
    if not (OUT / f"{arm}.jsonl").exists():
        return []
    rows = read_jsonl(OUT / f"{arm}.jsonl")
    retry = OUT / f"{arm}_retry.jsonl"
    if retry.exists():
        rows += read_jsonl(retry)
    latest = {}
    for row in rows:
        latest[row["question_id"]] = row
    return list(latest.values())


def official_prompt(question: str, reference: str, candidate: str) -> str:
    source = (OUT / "official_llm_as_a_judge.py").read_text(encoding="utf-8")
    template = re.search(r"prompt = f\"\"\"(.*?)\"\"\"", source, flags=re.S)
    if not template:
        raise RuntimeError("official SWE-QA judge prompt not found")
    # unescape the f-string braces in the template first, then fill placeholders in one pass so that
    # braces or placeholder-like text inside question/reference/candidate are never altered
    text = template.group(1).replace("{{", "{").replace("}}", "}")
    values = {"question": question, "reference": reference, "candidate": candidate}
    return re.sub(r"\{(question|reference|candidate)\}", lambda m: values[m.group(1)], text)


def spans(path: Path) -> list[tuple[int, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    return [(n.lineno, getattr(n, "end_lineno", n.lineno)) for n in ast.walk(tree)
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]


def overlap(a: int, b: int, c: int, d: int) -> bool:
    return max(a, c) <= min(b, d)


def validate_citations() -> None:
    contexts = {r["question_id"]: r for r in read_jsonl(OUT / "contexts.jsonl")}
    with (OUT / "citation_checks.jsonl").open("w", encoding="utf-8") as target:
        for arm in ARMS:
            for row in arm_rows(arm):
                evidence = contexts[row["question_id"]]["blocks"]
                for number, citation in enumerate(row.get("citations", []), 1):
                    raw_path = str(citation.get("source_file", "")).replace("\\", "/").lstrip("/")
                    start, end = citation.get("start_line"), citation.get("end_line")
                    path = REPO_ROOTS[row["repo"]] / raw_path
                    if raw_path and not path.is_file() and (REPO_ROOTS[row["repo"]] / "src" / raw_path).is_file():
                        raw_path = f"src/{raw_path}"  # closed-book models often omit the src/ layout prefix
                        path = REPO_ROOTS[row["repo"]] / raw_path
                    if not raw_path or not isinstance(start, int) or not isinstance(end, int):
                        label = "unparsable"
                    elif not path.is_file():
                        label = "nonexistent-file"
                    else:
                        line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
                        in_file = 1 <= start <= end <= line_count
                        in_evidence = any(raw_path == str(e["source_file"]).replace("\\", "/").lstrip("/") and
                                          overlap(start, end, e["start_line"], e["end_line"]) for e in evidence)
                        in_symbol = any(overlap(start, end, s, e) for s, e in spans(path))
                        if not in_file:
                            label = "valid-file-wrong-lines"
                        elif arm not in CLOSED_BOOK_ARMS and not in_evidence:
                            label = "valid-but-outside-evidence"
                        elif not in_symbol:
                            label = "valid-file-no-symbol-overlap"
                        else:
                            label = "valid-in-evidence" if arm not in CLOSED_BOOK_ARMS else "valid-file-and-symbol"
                    target.write(json.dumps({"arm": arm, "question_id": row["question_id"], "citation_number": number,
                                             "citation": citation, "classification": label}, ensure_ascii=False) + "\n")


def judge() -> None:
    provider = create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
        api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model="gemini-2.5-flash", timeout_seconds=180))
    records = [{"arm": arm, **row} for arm in JUDGED_ARMS
               for row in arm_rows(arm) if row["status"] == "usable"]
    random.Random(20260930).shuffle(records)
    completed = {r["blind_id"] for r in read_jsonl(OUT / "judge_scores.jsonl") if r.get("status") == "usable"} if (OUT / "judge_scores.jsonl").exists() else set()
    with (OUT / "judge_scores.jsonl").open("a", encoding="utf-8") as target:
        for i, row in enumerate(records, 1):
            blind_id = f"J{i:03d}"
            if blind_id in completed:
                continue
            candidate = CITATION_RE.sub("", row["answer"])
            prompt = official_prompt(row["question"], row["reference_answer"], candidate)
            result = {"blind_id": blind_id, "question_id": row["question_id"], "arm": row["arm"], "judge_model": "gemini-2.5-flash"}
            for attempt in range(1, 5):
                try:
                    response = provider.generate(LLMRequest(prompt=prompt, temperature=0, max_tokens=500, response_format="json"))
                    scores = json.loads(response.text)
                    if all(isinstance(scores.get(k), int) and 1 <= scores[k] <= 20 for k in ("correctness", "completeness", "relevance", "clarity", "reasoning")):
                        result.update({"status": "usable", "scores": scores, "mean": sum(scores.values()) / 5, "attempts": attempt})
                        break
                except Exception as exc:
                    result.update({"status": "unavailable", "error": str(exc)[:300], "attempts": attempt})
                time.sleep(2 ** (attempt - 1))
            target.write(json.dumps(result, ensure_ascii=False) + "\n"); target.flush()
            print(f"{blind_id} {result.get('status')}", flush=True)


def wilson(errors: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.96; p = errors / total; d = 1 + z*z/total
    centre = (p + z*z/(2*total))/d; half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [centre-half, centre+half]


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test (normal approximation with tie and continuity correction).

    Zero differences are dropped. Suitable for n >= ~20 non-zero pairs; returns p=None otherwise.
    """
    nz = [d for d in diffs if d]
    n = len(nz)
    if n < 20:
        return {"n_nonzero": n, "statistic": None, "p_value": None}
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    tie_term, i = 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var) if var > 0 else 0.0
    return {"n_nonzero": n, "statistic": w_plus, "p_value": math.erfc(z / math.sqrt(2))}


def judge_summary() -> None:
    """Paired A0-neutral vs A1 judge comparison; a later usable record supersedes an earlier unavailable one."""
    latest = {}
    for row in read_jsonl(OUT / "judge_scores.jsonl"):
        latest[row["blind_id"]] = row
    scores = {(r["question_id"], r["arm"]): r for r in latest.values() if r.get("status") == "usable"}
    questions = sorted(q for q, arm in scores if arm == "a0_neutral" and (q, "a1_codeqapass") in scores)
    metrics = {}
    for name in ("correctness", "completeness", "relevance", "clarity", "reasoning", "mean"):
        def value(q: str, arm: str) -> float:
            row = scores[(q, arm)]
            return row["mean"] if name == "mean" else row["scores"][name]
        a0 = [value(q, "a0_neutral") for q in questions]
        a1 = [value(q, "a1_codeqapass") for q in questions]
        diffs = [y - x for x, y in zip(a0, a1)]
        rng = random.Random(20260930)
        boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(20000))
        metrics[name] = {"a0_neutral_mean": sum(a0) / len(a0), "a1_mean": sum(a1) / len(a1), "mean_difference": sum(diffs) / len(diffs),
                         "bootstrap_95": [boot[500], boot[19499]], "a1_better": sum(d > 0 for d in diffs),
                         "a1_worse": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
                         "wilcoxon": wilcoxon_signed_rank(diffs)}
    summary = {"judge_model": "gemini-2.5-flash", "n_pairs": len(questions), "bootstrap": {"resamples": 20000, "seed": 20260930},
               "metrics": metrics, "note": "automatic judge, not human quality scores"}
    (OUT / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize() -> None:
    """Write summary.json: run availability, citation coverage/omission, and citation validity (never quality scores)."""
    checks = read_jsonl(OUT / "citation_checks.jsonl")
    contexts = {r["question_id"]: r for r in read_jsonl(OUT / "contexts.jsonl")}
    summary: dict = {"note": "Structural checks only. Zero errors for A1 is an invariant of deterministic citation construction, "
                             "and A2 mostly copies context headers; neither shows that citations support any claim.",
                     "arms": {}, "citation_metrics": {}}
    for arm in ARMS:
        rows = arm_rows(arm)
        if not rows:
            continue
        summary["arms"][arm] = {s_: sum(r["status"] == s_ for r in rows) for s_ in ("usable", "truncated", "unavailable")}
        arm_checks = [r for r in checks if r["arm"] == arm]
        valid = {"valid-in-evidence", "valid-file-and-symbol"}
        errors = sum(r["classification"] not in valid for r in arm_checks)
        usable = [r for r in rows if r["status"] == "usable"]
        with_citation = sum(bool(r.get("citations")) for r in usable)
        metrics = {"kind": "invariant" if arm in INVARIANT_ARMS else "measured",
                   "citations": len(arm_checks), "errors": errors,
                   "error_rate": errors / len(arm_checks) if arm_checks else None, "wilson_95": wilson(errors, len(arm_checks)),
                   "questions_with_citation": with_citation, "questions_without_citation": len(usable) - with_citation,
                   "citation_omission_rate": (len(usable) - with_citation) / len(usable) if usable else None,
                   "refusal_like_answers": sum(bool(REFUSAL_RE.search(r.get("answer", "")[:300])) for r in usable)}
        if arm in ("a1_codeqapass", "a2_model_citations"):
            blocks = {q: c["blocks"] for q, c in contexts.items()}
            exact = sum(any(c["source_file"] == b["source_file"] and c["start_line"] == b["start_line"] and c["end_line"] == b["end_line"]
                            for b in blocks[r["question_id"]]) for r in usable for c in r.get("citations", []))
            metrics["citations_identical_to_a_context_block"] = exact
            total_blocks = sum(len(blocks[r["question_id"]]) for r in usable)
            cited = sum(any(c["source_file"] == b["source_file"] and overlap(c["start_line"], c["end_line"], b["start_line"], b["end_line"])
                            for c in r.get("citations", [])) for r in usable for b in blocks[r["question_id"]])
            metrics["context_blocks_cited"] = f"{cited}/{total_blocks}"
        summary["citation_metrics"][arm] = metrics
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("citations", "judge", "summary", "judge-summary")); phase = parser.parse_args().phase
    {"citations": validate_citations, "judge": judge, "summary": summarize, "judge-summary": judge_summary}[phase]()
