"""SWE-QA v5 confirmation runner (tier T1: the 33 unused conan questions). See reports/evaluation/swe_qa_v5/PROTOCOL_V5_DRAFT.md.

Phases (all outputs are append-only and never overwritten; retries go to separate files):

    python scripts/evaluation/swe_qa_v5_confirm.py check
    python scripts/evaluation/swe_qa_v5_confirm.py rank                 # embedding calls only: rankings + service-equivalence capture
    python scripts/evaluation/swe_qa_v5_confirm.py contexts             # offline: build and store every arm's context
    python scripts/evaluation/swe_qa_v5_confirm.py generate --arm ARM   # LLM calls (a0 b0_k5 b0_k10 a1_cur a1_v5a a1_v5b) [--retry]
    python scripts/evaluation/swe_qa_v5_confirm.py judge-prepare        # blind and shuffle once
    python scripts/evaluation/swe_qa_v5_confirm.py judge-run --pass 1 [--limit N] [--delay S]
    python scripts/evaluation/swe_qa_v5_confirm.py analyze

Source identity (file, symbol, line range, chunk id, citations) comes from SQLite chunk metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics as stats
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_b0 as b0lib  # noqa: E402
import swe_qa_v4_judge as v4judge  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
import swe_qa_v5_dev_context_gate as gate  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.qa import QAPromptBuilder  # noqa: E402
from codecompass.qa.models import NO_EVIDENCE_ANSWER, QARequest  # noqa: E402
from codecompass.rag import ContextBlock, ContextCitation, RAGContext  # noqa: E402
from codecompass.retrieval import RetrievalQuery  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/swe_qa_v5/t1_conan"
ARTIFACT_ROOT = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2"
REPO = "conan"
ARMS = ("a0", "b0_k5", "b0_k10", "a1_cur", "a1_v5a", "a1_v5b")
BUDGET = gate.BUDGET
SEED = 20260930
BOOTSTRAP = 20_000
NI_MARGIN = 1.5
REFUSAL_RE = re.compile(r"^\W*(there is not enough retrieved evidence|not enough retrieved evidence)", re.IGNORECASE)
SCORE_KEYS = v4judge.SCORE_KEYS


# ---------------------------------------------------------------- selection and helpers
def t1_ids(freeze: dict) -> list[str]:
    """Conan questions that were not part of the v4 pilot, in numeric order."""
    pilot = set(freeze["pilot_sample"]["question_ids"])
    return [f"{REPO}-{n}" for n in range(1, 49) if f"{REPO}-{n}" not in pilot]


def t1_rows() -> list[dict]:
    freeze = v4.load_freeze()
    rows = {r["question_id"]: r for r in v4.load_dataset(v4.DATA / "dataset", REPO)}
    return [rows[q] for q in t1_ids(freeze)]


def load_dotenv() -> None:
    v4judge.load_dotenv()
    os.environ.setdefault("CODECOMPASS_LLM_REASONING_EFFORT", "low")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_new(path: Path):
    """Open a raw-record file for appending only when it does not exist yet (whole-file append-only discipline)."""
    if path.exists() and path.stat().st_size:
        raise SystemExit(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def is_refusal(record: dict) -> bool:
    return record.get("status") == "no_evidence" or bool(REFUSAL_RE.match((record.get("answer") or "").strip()))


class CaptureLLM:
    """Fake provider used to capture the exact prompt the production service would send; makes no network call."""

    def __init__(self) -> None:
        self.request: LLMRequest | None = None

    def generate(self, request: LLMRequest):  # noqa: D401
        from types import SimpleNamespace

        self.request = request
        return SimpleNamespace(text="captured", model="capture", provider="capture", finish_reason="stop")


def blocks_from_pieces(pieces: list[gate.Piece]) -> list[dict]:
    return [{"chunk_id": p.chunk_id, "source_file": p.source_file, "symbol_name": p.symbol_name, "qualified_name": p.qualified_name,
             "start_line": p.start_line, "end_line": p.end_line, "code": p.code} for p in pieces]


def context_from_blocks(blocks: list[dict], omitted: int) -> RAGContext:
    made = tuple(ContextBlock(chunk_id=b["chunk_id"], citation=ContextCitation(b["source_file"], b["symbol_name"], b["qualified_name"],
                                                                           b["start_line"], b["end_line"]),
                              code=b["code"], score=0.0, retrieval_method="hybrid") for b in blocks)
    return RAGContext(blocks=made, total_chars=sum(len(b["code"]) for b in blocks), omitted_count=omitted)


# ---------------------------------------------------------------- phase: rank (embedding calls only)
def with_backoff(call: Callable[[], object], waits: tuple[int, ...] = (5, 15, 45, 120)) -> object:
    """Retry transient provider failures (rate limit, timeout) with growing waits; the last error is raised unchanged."""
    for wait in waits:
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - only provider/transport failures are expected here
            if "insufficient" in str(error).lower():
                raise
            print(f"transient failure ({type(error).__name__}); waiting {wait}s", flush=True)
            time.sleep(wait)
    return call()

def phase_rank() -> int:
    load_dotenv()
    v4.ensure_embedding_run_identity(ARTIFACT_ROOT, dict(os.environ))
    v4.verify_snapshot(v4.repo_root(REPO), v4.COMMITS[REPO])
    rows = t1_rows()
    capture = CaptureLLM()
    service, project_id = v4.build_a1_service(REPO, capture, ARTIFACT_ROOT)
    retrieval = service.retrieval_service
    meta, matrix = v4.load_b0_index(REPO, ARTIFACT_ROOT)
    provider = b0lib.embedding_provider()
    paths = {name: OUT / f"rankings_{name}.jsonl" for name in ("semantic", "lexical", "b0")}
    paths["service"] = OUT / "a1_cur_service_context.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set.intersection(*({r["question_id"] for r in read_jsonl(path)} if path.exists() else set() for path in paths.values()))
    handles = {name: path.open("a", encoding="utf-8") for name, path in paths.items()}
    try:
        for row in rows:
            if row["question_id"] in done:
                continue
            question = row["question"]
            lines: dict[str, dict] = {}
            for name, search in (("semantic", retrieval.search_semantic), ("lexical", retrieval.search_lexical)):
                result = with_backoff(lambda: search(RetrievalQuery(question, project_id, 10)))
                items = [{"chunk_id": c.chunk_id, "file": c.source_file, "start_line": c.start_line, "end_line": c.end_line} for c in result.results]
                lines[name] = {"question_id": row["question_id"], "repo": REPO, "ranking": items}
            vector = b0lib.normalize(with_backoff(lambda: b0lib.embed_with_retry(provider, [question])))[0]
            hits = b0lib.top_chunks(vector, matrix, 10)
            lines["b0"] = {"question_id": row["question_id"], "repo": REPO, "ranking": [
                {"chunk_id": meta[i]["chunk_id"], "file": meta[i]["file"], "start_line": meta[i]["start_line"], "end_line": meta[i]["end_line"],
                 "score": score} for i, score in hits]}
            answer = with_backoff(lambda: service.answer(QARequest(question=question, project_id=project_id, retrieval_method="hybrid",
                                                                   retrieval_limit=v4.RETRIEVAL_LIMIT, max_context_chars=BUDGET, temperature=0.0,
                                                                   max_tokens=v4.MAX_TOKENS)))
            lines["service"] = {"question_id": row["question_id"], "chunk_ids": [c.chunk_id for c in answer.citations],
                                "prompt_sha256": hashlib.sha256((capture.request.prompt if capture.request else "").encode()).hexdigest(),
                                "omitted_context_count": answer.omitted_context_count}
            for name, line in lines.items():  # written together so a failed question leaves no partial record
                handles[name].write(json.dumps(line) + "\n")
                handles[name].flush()
            print("ranked", row["question_id"], flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return 0


# ---------------------------------------------------------------- phase: contexts (offline)
def b0_arm_context(row: dict, ranking: list[dict], meta: list[dict], k: int) -> tuple[list[dict], int]:
    used, omitted, blocks = 0, 0, []
    for item in ranking[:k]:
        chunk = meta[int(item["chunk_id"].rsplit("-", 1)[1])]
        if used + len(chunk["text"]) > BUDGET:
            omitted += 1
            continue
        used += len(chunk["text"])
        blocks.append({"chunk_id": item["chunk_id"], "source_file": chunk["file"], "symbol_name": chunk["symbol"], "qualified_name": chunk["qualified"],
                       "start_line": chunk["start_line"], "end_line": chunk["end_line"], "code": chunk["text"]})
    return blocks, omitted


def phase_contexts() -> int:
    rows = t1_rows()
    semantic = {r["question_id"]: r["ranking"] for r in read_jsonl(OUT / "rankings_semantic.jsonl")}
    lexical = {r["question_id"]: r["ranking"] for r in read_jsonl(OUT / "rankings_lexical.jsonl")}
    b0_rank = {r["question_id"]: r["ranking"] for r in read_jsonl(OUT / "rankings_b0.jsonl")}
    service = {r["question_id"]: r for r in read_jsonl(OUT / "a1_cur_service_context.jsonl")}
    meta, _ = v4.load_b0_index(REPO, ARTIFACT_ROOT)
    index = gate.Index(REPO)
    mismatches = []
    with append_new(OUT / "contexts.jsonl") as target:
        for row in rows:
            qid, question = row["question_id"], row["question"]
            ranked5 = gate.fuse(index, semantic[qid], lexical[qid], 5)
            depth10 = gate.fuse(index, semantic[qid], lexical[qid], 10)
            built = gate.variants(question, ranked5, depth10)
            contexts: dict[str, tuple[list[dict], int]] = {}
            for arm, name in (("a1_cur", "baseline"), ("a1_v5a", "hubless_soft_k5"), ("a1_v5b", "hubless_soft_k10")):
                chosen, omitted = gate.pack(built[name])
                contexts[arm] = (blocks_from_pieces(chosen), omitted)
            if [b["chunk_id"] for b in contexts["a1_cur"][0]] != service[qid]["chunk_ids"]:
                mismatches.append(qid)
            contexts["b0_k5"] = b0_arm_context(row, b0_rank[qid], meta, 5)
            contexts["b0_k10"] = b0_arm_context(row, b0_rank[qid], meta, 10)
            for arm, (blocks, omitted) in contexts.items():
                text = "\n".join(b["code"] for b in blocks)
                target.write(json.dumps({"question_id": qid, "arm": arm, "blocks": blocks, "omitted_context_count": omitted,
                                         "context_sha256": hashlib.sha256(text.encode()).hexdigest()}, ensure_ascii=False) + "\n")
    print(json.dumps({"questions": len(rows), "a1_cur_replica_mismatches": mismatches}))
    if mismatches:
        raise SystemExit("A1-cur replica differs from the production service context; investigate before generating")
    return 0


# ---------------------------------------------------------------- phase: generate
def load_contexts() -> dict[tuple[str, str], dict]:
    return {(r["question_id"], r["arm"]): r for r in read_jsonl(OUT / "contexts.jsonl")}


def call_for(arm: str, llm, row: dict, contexts: dict) -> Callable[[], dict]:
    if arm == "a0":
        return v4.call_a0(llm, row)
    record = contexts[(row["question_id"], arm)]

    def go() -> dict:
        blocks = record["blocks"]
        if not blocks:
            return {"text": NO_EVIDENCE_ANSWER, "finish_reason": None, "model": None, "citations": [], "context_chunk_ids": [],
                    "omitted_context_count": record["omitted_context_count"], "no_evidence": True}
        system_prompt, prompt = QAPromptBuilder().build(row["question"], context_from_blocks(blocks, record["omitted_context_count"]))
        response = llm.generate(LLMRequest(prompt=prompt, system_prompt=system_prompt, temperature=v4.TEMPERATURE, max_tokens=v4.MAX_TOKENS))
        citations = [{"chunk_id": b["chunk_id"], "source_file": b["source_file"], "symbol_name": b["symbol_name"],
                      "qualified_name": b["qualified_name"], "start_line": b["start_line"], "end_line": b["end_line"]} for b in blocks] if arm.startswith("a1") else []
        return {"text": response.text, "finish_reason": response.finish_reason, "model": response.model, "citations": citations,
                "context_chunk_ids": [b["chunk_id"] for b in blocks], "omitted_context_count": record["omitted_context_count"],
                "context_sha256": record["context_sha256"]}
    return go


def phase_generate(arm: str, retry: bool) -> int:
    load_dotenv()
    v4.require_frozen_settings()
    rows = t1_rows()
    sha = v4.verify_snapshot(v4.repo_root(REPO), v4.COMMITS[REPO])
    contexts = load_contexts() if arm != "a0" else {}
    llm = v2.llm_from_env()
    path = OUT / f"generation_{arm}{'_retry' if retry else ''}.jsonl"
    if retry:
        failed = {r["question_id"] for r in read_jsonl(OUT / f"generation_{arm}.jsonl") if r.get("status") == "unavailable"}
        rows = [r for r in rows if r["question_id"] in failed]
    with append_new(path) as target:
        for row in rows:
            outcome = v4.attempt(call_for(arm, llm, row, contexts))
            record = {"protocol": "swe_qa_v5", "tier": "t1_conan", "arm": arm, "question_id": row["question_id"], "repo": REPO, "question": row["question"],
                      "reference_answer": row.get("answer"), "snapshot_commit": sha,
                      "settings": {"temperature": v4.TEMPERATURE, "max_tokens": v4.MAX_TOKENS, "max_context_chars": BUDGET,
                                   "reasoning_effort": v4.REQUIRED_REASONING_EFFORT}, **outcome}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(arm, row["question_id"], record["status"], flush=True)
    return 0


# ---------------------------------------------------------------- phase: judge
def latest_generation(arm: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for suffix in ("", "_retry"):
        path = OUT / f"generation_{arm}{suffix}.jsonl"
        if path.exists():
            for r in read_jsonl(path):
                if suffix == "" or r.get("status") != "unavailable":
                    out[r["question_id"]] = r
    return out


def phase_judge_prepare() -> int:
    key_path, input_path = OUT / "judge_key.json", OUT / "judge_input_blinded.jsonl"
    if key_path.exists() or input_path.exists():
        raise SystemExit("judge input already exists; frozen inputs are immutable")
    records = []
    for arm in ARMS:
        for qid, r in sorted(latest_generation(arm).items()):
            if r["status"] in {"usable", "no_evidence"}:
                records.append({"arm": arm, "question_id": qid, "question": r["question"], "reference_answer": r.get("reference_answer") or "",
                                "candidate": v4judge.CITATION_RE.sub("", r.get("answer") or "")})
    random.Random(SEED).shuffle(records)
    mapping, inputs = {}, []
    for number, r in enumerate(records, 1):
        blind = f"K{number:03d}"
        inputs.append({"blind_id": blind, "question": r["question"], "reference_answer": r["reference_answer"], "candidate": r["candidate"]})
        mapping[blind] = {"question_id": r["question_id"], "arm": r["arm"]}
    input_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs), encoding="utf-8")
    key_path.write_text(json.dumps({"seed": SEED, "records": len(records), "mapping": mapping}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}))
    return 0


def phase_judge_run(pass_number: int, limit: int | None, delay: float) -> int:
    load_dotenv()
    from codecompass.providers import ProviderConfig, create_llm_provider

    rows = read_jsonl(OUT / "judge_input_blinded.jsonl")
    out_path = OUT / f"judge_pass{pass_number}.jsonl"
    done = {r["blind_id"] for r in read_jsonl(out_path) if r.get("status") == "usable"} if out_path.exists() else set()
    todo = [r for r in rows if r["blind_id"] not in done][:limit]
    provider = create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
                                                  api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model=v4judge.MODEL, timeout_seconds=180))
    with out_path.open("a", encoding="utf-8") as target:
        for row in todo:
            result = {"blind_id": row["blind_id"], "pass": pass_number, "judge_model": v4judge.MODEL, "temperature": v4judge.TEMPERATURE,
                      "max_tokens": v4judge.MAX_TOKENS}
            prompt = v4judge.official_prompt(row["question"], row["reference_answer"], row["candidate"])
            for attempt in range(1, 5):
                try:
                    response = provider.generate(LLMRequest(prompt=prompt, temperature=v4judge.TEMPERATURE, max_tokens=v4judge.MAX_TOKENS, response_format="json"))
                    scores = v4judge.parse_scores(response.text)
                    result.update({"status": "usable", "scores": scores, "mean": sum(scores.values()) / len(SCORE_KEYS), "attempts": attempt,
                                   "finish_reason": response.finish_reason})
                    break
                except Exception as error:  # noqa: BLE001 - preserved as unavailable, never scored
                    result.update({"status": "unavailable", "error_type": type(error).__name__, "error": str(error)[:300], "attempts": attempt})
                    if attempt < 4:
                        time.sleep((5, 30, 90)[attempt - 1] if "429" in str(error) else 2 ** (attempt - 1))
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(row["blind_id"], result["status"], flush=True)
            time.sleep(delay)
    return 0


# ---------------------------------------------------------------- phase: analyze
def unit_scores() -> dict[tuple[str, str], dict]:
    key = json.loads((OUT / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    per: dict[tuple[str, str], dict[int, dict]] = {}
    for pass_number in (1, 2):
        path = OUT / f"judge_pass{pass_number}.jsonl"
        if not path.exists():
            continue
        latest: dict[str, dict] = {}
        for r in read_jsonl(path):
            if r.get("status") == "usable":
                latest[r["blind_id"]] = r
        for blind, r in latest.items():
            info = key[blind]
            per.setdefault((info["question_id"], info["arm"]), {})[pass_number] = r
    return {unit: {"mean": stats.mean(p["mean"] for p in passes.values()), "passes": len(passes),
                   **{k: stats.mean(p["scores"][k] for p in passes.values()) for k in SCORE_KEYS}} for unit, passes in per.items()}


def paired(units: dict, left: str, right: str, qids: list[str], metric: str = "mean") -> list[float]:
    return [units[(q, left)][metric] - units[(q, right)][metric] for q in qids if (q, left) in units and (q, right) in units]


def bootstrap_ci(deltas: list[float]) -> list[float]:
    rng = random.Random(SEED)
    means = sorted(stats.mean(rng.choices(deltas, k=len(deltas))) for _ in range(BOOTSTRAP))
    return [round(means[int(0.025 * BOOTSTRAP)], 3), round(means[int(0.975 * BOOTSTRAP) - 1], 3)]


def compare(units: dict, left: str, right: str, qids: list[str]) -> dict:
    deltas = paired(units, left, right, qids)
    if len(deltas) < 2:
        return {"pairs": len(deltas), "status": "insufficient"}
    return {"pairs": len(deltas), "mean_difference": round(stats.mean(deltas), 3), "bootstrap_ci95": bootstrap_ci(deltas),
            "wilcoxon_p": v4judge.wilcoxon_p(deltas), "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas)}


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks, i = [0.0] * len(values), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                ranks[order[k]] = (i + j) / 2
            i = j + 1
        return ranks

    ra, rb = rank(a), rank(b)
    ma, mb = stats.mean(ra), stats.mean(rb)
    denominator = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / denominator, 3) if denominator else None


def phase_analyze() -> int:
    rows = t1_rows()
    qids = [r["question_id"] for r in rows]
    units = unit_scores()
    gen = {arm: latest_generation(arm) for arm in ARMS}
    contexts = load_contexts()
    root = v4.repo_root(REPO)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if ".git" not in p.parts)
    from swe_qa_retrieval_metrics import gold_files  # noqa: E402

    summary: dict = {"questions": len(qids), "arms": {}}
    for arm in ARMS:
        usable = [q for q in qids if (q, arm) in units]
        record = {"judged": len(usable), "mean_score": round(stats.mean(units[(q, arm)]["mean"] for q in usable), 3) if usable else None,
                  "criteria": {k: round(stats.mean(units[(q, arm)][k] for q in usable), 3) for k in SCORE_KEYS} if usable else None,
                  "status_counts": {s: sum(1 for r in gen[arm].values() if r["status"] == s) for s in sorted({r["status"] for r in gen[arm].values()})},
                  "refusal_rate": round(sum(is_refusal(r) for r in gen[arm].values()) / max(len(gen[arm]), 1), 3)}
        if arm != "a0":
            cov, ident, blocks = [], [], []
            for row in rows:
                ctx = contexts[(row["question_id"], arm)]["blocks"]
                gold = set(gold_files(row.get("answer") or "", files))
                text = "\n".join(b["code"] for b in ctx).lower()
                ids = gate.identifiers(row.get("answer") or "")
                blocks.append(len(ctx))
                if gold:
                    cov.append(len({b["source_file"] for b in ctx} & gold) / len(gold))
                if ids:
                    ident.append(sum(name in text for name in ids) / len(ids))
            record["delivered"] = {"gold_cov": round(stats.mean(cov), 3), "ident_cov": round(stats.mean(ident), 3), "blocks": round(stats.mean(blocks), 2)}
        summary["arms"][arm] = record
    h1, h2 = compare(units, "a1_v5a", "a1_cur", qids), compare(units, "a1_v5a", "b0_k5", qids)
    summary["primary"] = {"H1_v5a_minus_a1_cur": h1, "H2_v5a_minus_b0_k5": h2}
    if h2.get("bootstrap_ci95"):
        h2["non_inferior_margin_1.5"] = h2["bootstrap_ci95"][0] > -NI_MARGIN
        h2["superior"] = h2["bootstrap_ci95"][0] > 0
    holm = v4judge.holm({"H1": h1.get("wilcoxon_p"), "H2": h2.get("wilcoxon_p")})
    summary["primary"]["holm_adjusted_p"] = holm
    summary["exploratory"] = {f"{a}_minus_{b}": compare(units, a, b, qids) for a, b in (("a1_v5b", "a1_v5a"), ("a1_v5b", "b0_k10"), ("a1_cur", "b0_k5"),
                                                                                         ("a1_cur", "a0"), ("b0_k5", "a0"), ("a1_v5a", "a0"))}
    p1 = {r["blind_id"]: r["mean"] for r in read_jsonl(OUT / "judge_pass1.jsonl") if r.get("status") == "usable"} if (OUT / "judge_pass1.jsonl").exists() else {}
    p2 = {r["blind_id"]: r["mean"] for r in read_jsonl(OUT / "judge_pass2.jsonl") if r.get("status") == "usable"} if (OUT / "judge_pass2.jsonl").exists() else {}
    common = sorted(set(p1) & set(p2))
    summary["judge_pass_agreement"] = {"n": len(common), "spearman": spearman([p1[b] for b in common], [p2[b] for b in common]),
                                       "mean_abs_difference": round(stats.mean(abs(p1[b] - p2[b]) for b in common), 3) if common else None}
    summary["note"] = "T1 has 33 questions: results are labelled promising / not supported, never established (protocol section 6)."
    target = OUT / "analysis_t1.json"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target.name}; delete only with owner approval or write a tagged copy")
    target.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def phase_check() -> int:
    rows = t1_rows()
    print(f"T1 questions: {len(rows)}; first {rows[0]['question_id']}, last {rows[-1]['question_id']}")
    print("conan snapshot", v4.verify_snapshot(v4.repo_root(REPO), v4.COMMITS[REPO]))
    for name in ("conan.sqlite", "conan_chroma"):
        print(name, (ARTIFACT_ROOT / "index" / name).exists())
    print("b0 index", (ARTIFACT_ROOT / "b0" / "conan_chunks.jsonl").exists(), (ARTIFACT_ROOT / "b0" / "conan_vectors.npy").exists())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("check", "rank", "contexts", "generate", "judge-prepare", "judge-run", "analyze"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--pass", dest="pass_number", type=int, choices=(1, 2))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args(argv)
    if args.phase == "check":
        return phase_check()
    if args.phase == "rank":
        return phase_rank()
    if args.phase == "contexts":
        return phase_contexts()
    if args.phase == "generate":
        if not args.arm:
            parser.error("generate needs --arm")
        return phase_generate(args.arm, args.retry)
    if args.phase == "judge-prepare":
        return phase_judge_prepare()
    if args.phase == "judge-run":
        if not args.pass_number:
            parser.error("judge-run needs --pass")
        return phase_judge_run(args.pass_number, args.limit, args.delay)
    return phase_analyze()


if __name__ == "__main__":
    raise SystemExit(main())
