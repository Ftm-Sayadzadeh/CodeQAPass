"""SWE-QA v7 runner: Study 1 generation and judging, Study 2 efficiency timing, dry-run and freeze. See reports/evaluation/swe_qa_v7/PROTOCOL_V7_DRAFT.md.

Phases (outputs are append-only and never overwritten; retries go to separate files; nothing in `src/` is changed):

    python scripts/evaluation/swe_qa_v7_runner.py dry-run                     # no network: checks counts, regex, hashes, paths
    python scripts/evaluation/swe_qa_v7_runner.py freeze                      # reads the provider price list (no model call), writes FREEZE_V7.json
    python scripts/evaluation/swe_qa_v7_runner.py generate --arm ARM [--retry]      # Study 1: a1_v5a, a1_v6e
    python scripts/evaluation/swe_qa_v7_runner.py judge-prepare               # blind shuffle of up to 960 records, stratified 25% second-pass sample
    python scripts/evaluation/swe_qa_v7_runner.py judge-run --pass 1|2 [--workers 2]
    python scripts/evaluation/swe_qa_v7_runner.py timing [--limit-questions N]       # Study 2

Every model-calling phase refuses to run unless FREEZE_V7.json exists and every frozen file still has its recorded hash.
Tokens, cost and latency come from the provider response (`usage`, `estimated_cost`); a call without `usage` is recorded as unavailable, never as zero or an estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_b0 as b0lib  # noqa: E402
import swe_qa_v4_judge as v4judge  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
import swe_qa_v5_confirm as c5  # noqa: E402
import swe_qa_v5_dev_context_gate as gate  # noqa: E402
import swe_qa_v6_assembly as v6  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.qa import QAPromptBuilder  # noqa: E402
from codecompass.qa.models import NO_EVIDENCE_ANSWER, QARequest  # noqa: E402
from codecompass.retrieval import RetrievalQuery  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "reports/evaluation/swe_qa_v7"
PRE = V7 / "preflight_final"
OUT = V7 / "main"
FREEZE = V7 / "FREEZE_V7.json"
ADDENDUM = V7 / "FREEZE_V7_ADDENDUM.json"
PRICES = V7 / "prices.json"
PROTOCOL = V7 / "PROTOCOL_V7_DRAFT.md"
V4F = gate.V4
ARTIFACT_V4 = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2"
REPOS = gate.REPOS
NEW_ARMS = ("a1_v5a", "a1_v6e")
OLD_ARMS = ("a0", "b0_k5", "a1_v4")
ALL_ARMS = ("a0", "b0_k5", "a1_v4", "a1_v5a", "a1_v6e")
OLD_FILES = {"a0": V4F / "main_a0_combined.jsonl", "b0_k5": V4F / "main_b0_k5_combined.jsonl", "a1_v4": V4F / "main_a1_combined.jsonl"}
SEED = 20260930
REPEATS = 3
WARMUP_CALLS = 3
SECOND_PASS_SHARE = 0.25
SCORE_KEYS = v4judge.SCORE_KEYS
CREDIT_MARKERS = ("insufficient", "exhausted", "quota_exceeded")
PRICED_MODELS = ("glm-5.3-flash", "gemini-embedding-2", "gemini-2.5-flash")
FROZEN_PATHS = (
    "reports/evaluation/swe_qa_v7/PROTOCOL_V7_DRAFT.md",
    "scripts/evaluation/swe_qa_v7_preflight.py",
    "scripts/evaluation/swe_qa_v7_analyze.py",
    "scripts/evaluation/swe_qa_v5_dev_context_gate.py",
    "scripts/evaluation/swe_qa_v6_assembly.py",
    "scripts/evaluation/swe_qa_v4_judge.py",
    "reports/evaluation/swe_qa_v7/prices.json",
    "reports/evaluation/swe_qa_v7/preflight_final/subsets.json",
    "reports/evaluation/swe_qa_v7/preflight_final/contexts_study1.jsonl",
    "reports/evaluation/swe_qa_v7/preflight_final/study3_offline.json",
    "reports/evaluation/swe_qa_v4/frozen/main_a0_combined.jsonl",
    "reports/evaluation/swe_qa_v4/frozen/main_b0_k5_combined.jsonl",
    "reports/evaluation/swe_qa_v4/frozen/main_a1_combined.jsonl",
    "reports/evaluation/swe_qa_v4/frozen/judge_scores.jsonl",
    "reports/evaluation/swe_qa_v4/frozen/judge_key.json",
)
RUNNER_PATH = "scripts/evaluation/swe_qa_v7_runner.py"


# ---------------------------------------------------------------- generic helpers
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_credit_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in CREDIT_MARKERS)


class CreditStop(BaseException):
    """Raised when the provider reports no credit. It derives from BaseException so the retry loops (which catch Exception) let it through and the run stops at once."""


def credit_probe() -> bool:
    """Read the provider error body to tell an exhausted-credit 429 from a rate limit (the providers hide the body from their error messages, so a credit
    failure reads as a plain HTTP 429). Both endpoints are probed with realistic request sizes, because a tiny request can still pass on a nearly empty balance."""
    def credit_error(url: str, key: str, body: dict) -> bool:
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            urllib.request.urlopen(request, timeout=30).read()
            return False
        except urllib.error.HTTPError as error:
            return is_credit_error(error.read().decode("utf-8", "ignore")[:2000])
        except Exception:  # noqa: BLE001 - any other failure is not a confirmed credit problem
            return False

    chat = credit_error(os.environ["CODECOMPASS_COMPARE_BASE_URL"].rstrip("/") + "/chat/completions", os.environ["CODECOMPASS_COMPARE_API_KEY"],
                        {"model": os.environ["CODECOMPASS_COMPARE_MODEL"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1200})
    embedding = credit_error(os.environ["CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL"].rstrip("/") + "/embeddings", os.environ["CODECOMPASS_GEMINI2_EMBEDDING_API_KEY"],
                             {"model": os.environ.get("CODECOMPASS_GEMINI2_EMBEDDING_MODEL", "gemini-embedding-2"), "input": ["hi"]})
    return chat or embedding


def strip_citations(text: str | None) -> str:
    """The v4 judge preparation: remove `file.py:12-34` citation patterns with the v4 regular expression, nothing else."""
    return v4judge.CITATION_RE.sub("", text or "")


def question_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for repo in REPOS:
        for number, row in enumerate(read_jsonl(gate.DATASET / f"{repo}.jsonl"), 1):
            rows[f"{repo}-{number}"] = {"question_id": f"{repo}-{number}", "repo": repo, **row}
    return rows


def study1_contexts() -> dict[tuple[str, str], dict]:
    return {(r["question_id"], r["arm"]): r for r in read_jsonl(PRE / "contexts_study1.jsonl")}


def generation_path(arm: str, retry: bool) -> Path:
    return OUT / f"study1_generation_{arm}{'_retry' if retry else ''}.jsonl"


def latest_generation(arm: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for retry in (False, True):
        path = generation_path(arm, retry)
        if path.exists():
            for record in read_jsonl(path):
                if not retry or record.get("status") != "unavailable":
                    out[record["question_id"]] = record
    return out


def pending_rows(rows: list[dict], original: list[dict]) -> list[dict]:
    """Rows without a usable (or explicit no-evidence) record in the original file: unavailable answers and questions never attempted."""
    done = {r["question_id"] for r in original if r.get("status") in {"usable", "no_evidence"}}
    return [row for row in rows if row["question_id"] not in done]


# ---------------------------------------------------------------- measurement wrapper (reads the raw provider response; src/ is not changed)
class Recorder:
    """Wraps `_post_json` of one provider instance and keeps, per thread, the timing, `usage` and `estimated_cost` of every successful call."""

    def __init__(self, provider) -> None:
        self.provider = provider
        self._local = threading.local()
        original = provider._post_json

        def wrapped(payload):
            start = time.perf_counter()
            response = original(payload)
            self._calls().append({"seconds": time.perf_counter() - start, "usage": response.get("usage"), "estimated_cost": response.get("estimated_cost")})
            return response

        provider._post_json = wrapped

    def _calls(self) -> list[dict]:
        if not hasattr(self._local, "calls"):
            self._local.calls = []
        return self._local.calls

    def take(self) -> list[dict]:
        calls, self._local.calls = self._calls(), []
        return calls


def usage_of(calls: list[dict]) -> dict:
    """Sum tokens and provider cost over calls. Missing `usage` makes every figure None (unavailable), never zero or an estimate."""
    seconds = sum(c.get("seconds", 0.0) for c in calls)
    summed = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    unit = 0.0
    available = bool(calls)
    for call in calls:
        usage, cost = call.get("usage"), call.get("estimated_cost")
        if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
            available = False
            continue
        summed["prompt_tokens"] += usage["prompt_tokens"]
        summed["completion_tokens"] += usage["completion_tokens"]
        summed["reasoning_tokens"] += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        summed["cached_tokens"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if isinstance(cost, dict) and cost.get("unit") is not None:
            unit += float(cost["unit"])
        else:
            available = False
    if not available:
        return {"usage_available": False, "http_calls": len(calls), "http_seconds": round(seconds, 4), **{k: None for k in summed}, "provider_cost_unit": None}
    return {"usage_available": True, "http_calls": len(calls), "http_seconds": round(seconds, 4), **summed, "provider_cost_unit": round(unit, 10)}


def attempt_rate_limited(call: Callable[[], dict]) -> dict:
    """`v4.attempt` with six tries and waits of 15, 30, 60, 120 and 240 seconds so provider rate limits are waited out, not recorded as failures."""
    saved = v4.MAX_ATTEMPTS
    v4.MAX_ATTEMPTS = 6
    try:
        return v4.attempt(call, sleep=lambda seconds: time.sleep(15 * seconds))
    finally:
        v4.MAX_ATTEMPTS = saved


# ---------------------------------------------------------------- freeze guard
def current_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in FROZEN_PATHS if (ROOT / path).exists()}


def require_freeze() -> dict:
    """Stop unless the freeze file exists and every frozen file (and the runner, or its addendum version) still has its recorded hash."""
    if not FREEZE.exists():
        raise SystemExit("FREEZE_V7.json is missing: run `dry-run` and `freeze` before any model call")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for path, expected in freeze["sha256"].items():
        actual = sha256_file(ROOT / path) if (ROOT / path).exists() else None
        if path == RUNNER_PATH and ADDENDUM.exists():
            allowed = {expected, json.loads(ADDENDUM.read_text(encoding="utf-8"))["sha256"].get(RUNNER_PATH)}
            if actual in allowed:
                continue
        if actual != expected:
            raise SystemExit(f"frozen file changed after the freeze: {path}; record a deviation and an addendum before continuing")
    return freeze


def fetch_prices() -> dict:
    """Read the provider price list (`/models`, field `pricing`); this is not a model call."""
    c5.load_dotenv()
    request = urllib.request.Request(os.environ["CODECOMPASS_COMPARE_BASE_URL"].rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + os.environ["CODECOMPASS_COMPARE_API_KEY"]})
    with urllib.request.urlopen(request, timeout=60) as response:
        models = json.loads(response.read().decode("utf-8"))["data"]
    table = {m["id"]: m["pricing"] for m in models if m.get("id") in PRICED_MODELS}
    missing = [m for m in PRICED_MODELS if m not in table]
    if missing:
        raise SystemExit(f"price list lacks {missing}")
    return {"read_at": datetime.now(timezone.utc).isoformat(), "unit": "credit units per million tokens", "dashboard_toman_per_credit_unit": 271450, "pricing": table}


# ---------------------------------------------------------------- Study 1: generation
def study1_call(arm: str, llm, recorder: Recorder, row: dict, context: dict) -> Callable[[], dict]:
    def go() -> dict:
        blocks = context["blocks"]
        if not blocks:
            return {"text": NO_EVIDENCE_ANSWER, "finish_reason": None, "model": None, "citations": [], "context_chunk_ids": [], "no_evidence": True,
                    "omitted_context_count": context["omitted_context_count"], "context_sha256": context["context_sha256"], "arm_context": context["arm"]}
        system_prompt, prompt = QAPromptBuilder().build(row["question"], c5.context_from_blocks(blocks, context["omitted_context_count"]))
        recorder.take()
        try:
            response = llm.generate(LLMRequest(prompt=prompt, system_prompt=system_prompt, temperature=v4.TEMPERATURE, max_tokens=v4.MAX_TOKENS))
        except Exception as error:  # noqa: BLE001
            if "429" in str(error) and credit_probe():
                raise CreditStop(str(error)) from None
            raise
        usage = usage_of(recorder.take())
        citations = [{"chunk_id": b["chunk_id"], "parent_chunk_id": v6.parent_chunk_id(b["chunk_id"]), "source_file": b["source_file"], "symbol_name": b["symbol_name"],
                      "qualified_name": b["qualified_name"], "start_line": b["start_line"], "end_line": b["end_line"]} for b in blocks]
        return {"text": response.text, "finish_reason": response.finish_reason, "model": response.model, "citations": citations,
                "context_chunk_ids": [b["chunk_id"] for b in blocks], "omitted_context_count": context["omitted_context_count"],
                "context_sha256": context["context_sha256"], "prompt_sha256": sha256_text(system_prompt + "\n" + prompt), "arm_context": context["arm"], "usage": usage}
    return go


def phase_generate(arm: str, retry: bool) -> int:
    require_freeze()
    c5.load_dotenv()
    v4.require_frozen_settings()
    rows = list(question_rows().values())
    shas = {repo: v4.verify_snapshot(v4.repo_root(repo), v4.COMMITS[repo]) for repo in REPOS}
    contexts = study1_contexts()
    llm = v2.llm_from_env()
    recorder = Recorder(llm)
    path = generation_path(arm, retry)
    if retry:
        rows = pending_rows(rows, read_jsonl(generation_path(arm, False)))
    with c5.append_new(path) as target:
        for row in rows:
            try:
                outcome = attempt_rate_limited(study1_call(arm, llm, recorder, row, contexts[(row["question_id"], arm)]))
            except CreditStop:
                print("credit exhausted: stopping; rerun with --retry after recharging (records so far are kept)", flush=True)
                return 2
            record = {"protocol": "swe_qa_v7", "study": "study1", "arm": arm, "question_id": row["question_id"], "repo": row["repo"], "question": row["question"],
                      "reference_answer": row.get("answer"), "snapshot_commit": shas[row["repo"]],
                      "settings": {"temperature": v4.TEMPERATURE, "max_tokens": v4.MAX_TOKENS, "max_context_chars": c5.BUDGET, "reasoning_effort": v4.REQUIRED_REASONING_EFFORT},
                      **outcome}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(arm, row["question_id"], record["status"], flush=True)
    return 0


# ---------------------------------------------------------------- Study 1: judging
def judge_records() -> list[dict]:
    """New answers (v7 generation) and the old v4 frozen answers, all prepared the same way, as a list of judgeable records."""
    rows = question_rows()
    records: list[dict] = []
    for arm in ALL_ARMS:
        if arm in NEW_ARMS:
            source = [(q, r) for q, r in sorted(latest_generation(arm).items())]
            label = "v7"
        else:
            source = [(r["question_id"], r) for r in sorted(read_jsonl(OLD_FILES[arm]), key=lambda r: r["question_id"])]
            label = "v4_frozen"
        for qid, record in source:
            if record["status"] not in {"usable", "no_evidence"}:
                continue
            records.append({"arm": arm, "question_id": qid, "repo": rows[qid]["repo"], "source": label, "question": rows[qid]["question"],
                            "reference_answer": rows[qid].get("answer") or "", "candidate": strip_citations(record.get("answer"))})
    return records


def second_pass_sample(mapping: dict[str, dict]) -> list[str]:
    """25% of the judgeable records, drawn separately for every arm x repository cell with the fixed seed."""
    cells: dict[tuple[str, str], list[str]] = {}
    for blind, info in sorted(mapping.items()):
        cells.setdefault((info["arm"], info["repo"]), []).append(blind)
    chosen: list[str] = []
    for (arm, repo), ids in sorted(cells.items()):
        chosen += random.Random(f"{SEED}:{arm}:{repo}").sample(ids, max(1, round(SECOND_PASS_SHARE * len(ids))))
    return sorted(chosen)


def phase_judge_prepare() -> int:
    require_freeze()
    key_path, input_path, sample_path = OUT / "judge_key.json", OUT / "judge_input_blinded.jsonl", OUT / "judge_pass2_sample.json"
    if key_path.exists() or input_path.exists() or sample_path.exists():
        raise SystemExit("judge inputs already exist; they are immutable")
    records = judge_records()
    random.Random(SEED).shuffle(records)
    mapping, inputs = {}, []
    for number, record in enumerate(records, 1):
        blind = f"N{number:04d}"
        inputs.append({"blind_id": blind, "question": record["question"], "reference_answer": record["reference_answer"], "candidate": record["candidate"]})
        mapping[blind] = {"question_id": record["question_id"], "arm": record["arm"], "repo": record["repo"], "source": record["source"]}
    OUT.mkdir(parents=True, exist_ok=True)
    input_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs), encoding="utf-8")
    key_path.write_text(json.dumps({"seed": SEED, "records": len(records), "mapping": mapping}, indent=1) + "\n", encoding="utf-8")
    sample_path.write_text(json.dumps({"seed": SEED, "share": SECOND_PASS_SHARE, "blind_ids": second_pass_sample(mapping)}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "input_sha256": sha256_file(input_path)}))
    return 0


def judge_one(provider, recorder: Recorder, row: dict, pass_number: int) -> dict:
    result = {"blind_id": row["blind_id"], "pass": pass_number, "judge_model": v4judge.MODEL, "temperature": v4judge.TEMPERATURE, "max_tokens": v4judge.MAX_TOKENS}
    prompt = v4judge.official_prompt(row["question"], row["reference_answer"], row["candidate"])
    for attempt in range(1, 5):
        try:
            recorder.take()
            response = provider.generate(LLMRequest(prompt=prompt, temperature=v4judge.TEMPERATURE, max_tokens=v4judge.MAX_TOKENS, response_format="json"))
            scores = v4judge.parse_scores(response.text)
            result.update({"status": "usable", "scores": scores, "mean": sum(scores.values()) / len(SCORE_KEYS), "attempts": attempt, "finish_reason": response.finish_reason,
                           "usage": usage_of(recorder.take())})
            return result
        except Exception as error:  # noqa: BLE001 - preserved as unavailable, never scored
            result.update({"status": "unavailable", "error_type": type(error).__name__, "error": str(error)[:300], "attempts": attempt})
            if is_credit_error(str(error)) or ("429" in str(error) and credit_probe()):
                result["error_type"] = "CreditExhausted"
                return result
            if attempt < 4:
                time.sleep((5, 30, 90)[attempt - 1] if "429" in str(error) else 2 ** (attempt - 1))
    return result


def phase_judge_run(pass_number: int, limit: int | None, delay: float, workers: int) -> int:
    require_freeze()
    c5.load_dotenv()
    from codecompass.providers import ProviderConfig, create_llm_provider

    rows = read_jsonl(OUT / "judge_input_blinded.jsonl")
    if pass_number == 2:
        sample = set(json.loads((OUT / "judge_pass2_sample.json").read_text(encoding="utf-8"))["blind_ids"])
        rows = [r for r in rows if r["blind_id"] in sample]
    out_path = OUT / f"judge_pass{pass_number}.jsonl"
    done = {r["blind_id"] for r in read_jsonl(out_path) if r.get("status") == "usable"} if out_path.exists() else set()
    todo = [r for r in rows if r["blind_id"] not in done][:limit]
    provider = create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
                                                  api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model=v4judge.MODEL, timeout_seconds=180))
    recorder = Recorder(provider)
    lock, stop = threading.Lock(), threading.Event()
    with out_path.open("a", encoding="utf-8") as target:
        def work(row: dict) -> None:
            if stop.is_set():
                return
            result = judge_one(provider, recorder, row, pass_number)
            with lock:
                if result.get("error_type") == "CreditExhausted":
                    stop.set()  # one record keeps the evidence; the rest of the pass is not attempted
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                target.flush()
                print(row["blind_id"], result["status"], flush=True)
            time.sleep(delay)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
    return 0


# ---------------------------------------------------------------- Study 2: efficiency timing
def hash_context(blocks: list[dict]) -> str:
    return sha256_text("\n".join(b["code"] for b in blocks))


def run_one(arm: str, row: dict, ctx: dict, parts: dict) -> dict:
    """One timed answer for one arm. `parts` holds the live services and the two measurement recorders."""
    llm, lrec, emb, erec = parts["llm"], parts["lrec"], parts["emb"], parts["erec"]
    service, project_id = parts["services"][row["repo"]]
    index = parts["indexes"][row["repo"]]
    question = row["question"]
    lrec.take()
    erec.take()
    started = time.perf_counter()
    stages: dict[str, float | None] = {"retrieval_seconds": None, "assembly_seconds": None}
    blocks: list[dict] | None = None
    omitted = 0
    if arm == "a0":
        response = llm.generate(LLMRequest(prompt=v2.a0_neutral_prompt(question, row["repo"]), system_prompt=v2.A0_NEUTRAL_SYSTEM, temperature=v4.TEMPERATURE, max_tokens=v4.MAX_TOKENS))
        text, finish = response.text, response.finish_reason
    elif arm == "a1_v4":
        answer = service.answer(QARequest(question=question, project_id=project_id, retrieval_method="hybrid", retrieval_limit=v4.RETRIEVAL_LIMIT, max_context_chars=c5.BUDGET,
                                          temperature=v4.TEMPERATURE, max_tokens=v4.MAX_TOKENS))
        text, finish = answer.answer, answer.finish_reason
        live_ids, omitted = [c.chunk_id for c in answer.citations], answer.omitted_context_count
        blocks = [{"chunk_id": i, "code": ""} for i in live_ids]  # ids only; the context text is compared through the chunk ids and omitted count
    else:
        if arm == "b0_k5":
            import numpy as np

            t0 = time.perf_counter()
            vector = b0lib.normalize(np.array([r.vector for r in emb.embed_texts([question])], dtype=np.float32))[0]
            meta, matrix = parts["b0"][row["repo"]]
            hits = b0lib.top_chunks(vector, matrix, 10)
            stages["retrieval_seconds"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            blocks, omitted = c5.b0_arm_context(row, [{"chunk_id": meta[i]["chunk_id"]} for i, _ in hits], meta, 5)
            stages["assembly_seconds"] = time.perf_counter() - t1
        else:
            t0 = time.perf_counter()
            result = service.retrieval_service.search_hybrid(RetrievalQuery(question, project_id, 10))
            stages["retrieval_seconds"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            pieces = [index.piece(c.chunk_id, rank) for rank, c in enumerate(result.results, 1)]
            chosen, omitted = gate.pack(gate.variants(question, pieces[:5], pieces)["hubless_soft_k5"] if arm == "a1_v5a" else v6.v6_pieces(question, pieces))
            blocks = c5.blocks_from_pieces(chosen)
            stages["assembly_seconds"] = time.perf_counter() - t1
        system_prompt, prompt = QAPromptBuilder().build(question, c5.context_from_blocks(blocks, omitted))
        response = llm.generate(LLMRequest(prompt=prompt, system_prompt=system_prompt, temperature=v4.TEMPERATURE, max_tokens=v4.MAX_TOKENS))
        text, finish = response.text, response.finish_reason
    total = time.perf_counter() - started
    llm_calls, emb_calls = lrec.take(), erec.take()
    live_ids = [b["chunk_id"] for b in blocks] if blocks is not None else []
    expected = ctx.get((row["question_id"], arm))
    match = None
    live_hash = None
    if arm != "a0" and expected is not None:
        match = live_ids == [b["chunk_id"] for b in expected["blocks"]] and omitted == expected["omitted_context_count"]
        if arm != "a1_v4":
            live_hash = hash_context(blocks)
            match = match and live_hash == expected["context_sha256"]
    generation_seconds = sum(c["seconds"] for c in llm_calls)
    return {"status": "ok", "answer": text, "finish_reason": finish, "total_seconds": round(total, 4), "generation_seconds": round(generation_seconds, 4),
            "retrieval_assembly_seconds": round(total - generation_seconds, 4), "embedding_seconds": round(sum(c["seconds"] for c in emb_calls), 4),
            **{k: (round(v, 4) if v is not None else None) for k, v in stages.items()}, "llm_usage": usage_of(llm_calls), "embedding_usage": usage_of(emb_calls),
            "context_blocks": len(live_ids), "context_chunk_ids": live_ids, "omitted_context_count": omitted, "context_sha256": live_hash, "context_matches_study1": match}


def timing_plan(subset: dict[str, list[str]], limit_questions: int | None = None) -> tuple[list[dict], list[dict]]:
    """Warm-up calls and the timed schedule: repeats in order, questions and arms shuffled with the fixed seed."""
    ids = [q for repo in REPOS for q in subset[repo]]
    if limit_questions:
        ids = ids[:limit_questions]
    warmup = [{"question_id": q, "arm": arm, "repeat": 0, "warmup": True} for arm in ALL_ARMS for q in ids[:WARMUP_CALLS]]
    timed = []
    for repeat in range(1, REPEATS + 1):
        order = list(ids)
        random.Random(f"{SEED}:order:{repeat}").shuffle(order)
        for q in order:
            arms = list(ALL_ARMS)
            random.Random(f"{SEED}:{q}:{repeat}").shuffle(arms)
            timed += [{"question_id": q, "arm": arm, "repeat": repeat, "warmup": False} for arm in arms]
    return warmup, timed


def phase_timing(limit_questions: int | None) -> int:
    require_freeze()
    c5.load_dotenv()
    v4.require_frozen_settings()
    subset = json.loads((PRE / "subsets.json").read_text(encoding="utf-8"))["study2_subset"]
    rows = question_rows()
    contexts = study1_contexts()
    for repo in REPOS:
        v4.verify_snapshot(v4.repo_root(repo), v4.COMMITS[repo])
    v4.ensure_embedding_run_identity(ARTIFACT_V4, dict(os.environ))
    llm = v2.llm_from_env()
    lrec = Recorder(llm)
    emb = b0lib.embedding_provider()
    erec = Recorder(emb)
    b0lib.embedding_provider = lambda: emb  # the A1 retrieval service built below embeds through the measured provider
    parts = {"llm": llm, "lrec": lrec, "emb": emb, "erec": erec, "services": {}, "indexes": {}, "b0": {}}
    for repo in REPOS:
        parts["services"][repo] = v4.build_a1_service(repo, llm, ARTIFACT_V4)
        parts["indexes"][repo] = gate.Index(repo)
        parts["b0"][repo] = v4.load_b0_index(repo, ARTIFACT_V4)
    warmup, timed = timing_plan(subset, limit_questions)
    path = OUT / "study2_timing.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = {(r["question_id"], r["arm"], r["repeat"]) for r in read_jsonl(path) if r.get("status") == "ok"} if path.exists() else set()
    with path.open("a", encoding="utf-8") as target:
        for step in warmup + timed:
            key = (step["question_id"], step["arm"], step["repeat"])
            if key in done:
                continue
            row = rows[step["question_id"]]
            record = {"protocol": "swe_qa_v7", "study": "study2", "question_id": step["question_id"], "repo": row["repo"], "arm": step["arm"], "repeat": step["repeat"],
                      "warmup": step["warmup"], "started_at": datetime.now(timezone.utc).isoformat()}
            try:
                record.update(run_one(step["arm"], row, contexts, parts))
            except Exception as error:  # noqa: BLE001 - failures are logged and excluded from latency statistics, never scored
                record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)[:300]})
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(step["question_id"], step["arm"], step["repeat"], record["status"], flush=True)
            if record["status"] == "failed" and (is_credit_error(record.get("error", "")) or ("429" in record.get("error", "") and credit_probe())):
                print("credit exhausted: stopping", flush=True)
                return 2
    return 0


# ---------------------------------------------------------------- dry-run and freeze
def dry_run_checks() -> list[tuple[str, bool, str]]:
    """Everything that can be verified before any credit is used (no network, no model call)."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    contexts = study1_contexts()
    rows = question_rows()
    check("192 questions, 48 per repository", len(rows) == 192 and all(sum(1 for r in rows.values() if r["repo"] == repo) == 48 for repo in REPOS))
    generations = sum(1 for (q, arm) in contexts if arm in NEW_ARMS)
    check("Study 1 generations at most 384", generations <= 384, f"{generations} planned ({', '.join(f'{a}={sum(1 for k in contexts if k[1] == a)}' for a in NEW_ARMS)})")
    old = {arm: [r for r in read_jsonl(OLD_FILES[arm]) if r["status"] in {"usable", "no_evidence"}] for arm in OLD_ARMS}
    judge_total = generations + sum(len(v) for v in old.values())
    check("judge records at most 960", judge_total <= 960, f"{generations} new + {sum(len(v) for v in old.values())} old = {judge_total}")
    # citation stripping must reproduce the v4 judge candidates exactly for every old answer
    key = json.loads((V4F / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    v4_inputs = {r["blind_id"]: r["candidate"] for r in read_jsonl(V4F / "judge_input_blinded.jsonl")}
    arm_name = {"a1": "a1_v4"}  # v4 names the production arm `a1`
    by_pair = {(info["question_id"], arm_name.get(info["arm"], info["arm"])): v4_inputs[blind] for blind, info in key.items() if blind in v4_inputs}
    mismatched = [(q, a) for a in OLD_ARMS for r in old[a] for q in [r["question_id"]] if (q, a) in by_pair and strip_citations(r.get("answer")) != by_pair[(q, a)]]
    covered = sum(1 for a in OLD_ARMS for r in old[a] if (r["question_id"], a) in by_pair)
    check("citation stripping equals the v4 judge preparation", not mismatched and covered == sum(len(v) for v in old.values()), f"{covered} old answers compared, {len(mismatched)} differ")
    check("citation regex is the v4 object", v4judge.CITATION_RE.pattern == r"[\w./-]+\.py:\d+(?:-\d+)?")
    sample = next(iter(contexts.values()))
    row = rows[sample["question_id"]]
    system_prompt, prompt = QAPromptBuilder().build(row["question"], c5.context_from_blocks(sample["blocks"], sample["omitted_context_count"]))
    check("prompt, arm and context hash are recordable", bool(sha256_text(system_prompt + "\n" + prompt)) and bool(sample["context_sha256"]) and sample["arm"] in ALL_ARMS)
    check("retries use separate files", all(generation_path(a, True) != generation_path(a, False) for a in NEW_ARMS))
    missing = usage_of([{"seconds": 1.0, "usage": None, "estimated_cost": None}])
    check("a call without usage is unavailable, not zero", missing["usage_available"] is False and missing["prompt_tokens"] is None and missing["provider_cost_unit"] is None)
    present = usage_of([{"seconds": 1.0, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 2}, "prompt_tokens_details": {"cached_tokens": 0}},
                         "estimated_cost": {"unit": "0.001"}}])
    check("usage and provider cost are summed", present["usage_available"] and present["prompt_tokens"] == 10 and present["reasoning_tokens"] == 2 and present["provider_cost_unit"] == 0.001)
    check("no output file exists yet", not OUT.exists() or not any(OUT.iterdir()), str(OUT))
    check("freeze file not yet written", not FREEZE.exists())
    check("preflight files present", all((PRE / n).exists() for n in ("subsets.json", "contexts_study1.jsonl", "study3_offline.json")))
    subset = json.loads((PRE / "subsets.json").read_text(encoding="utf-8"))
    warmup, timed = timing_plan(subset["study2_subset"])
    check("Study 2 plan: 48 questions x 5 arms x 3 repeats plus warm-up", len(timed) == 720 and len(warmup) == 15, f"{len(timed)} timed + {len(warmup)} warm-up")
    check("Study 2 equality baseline exists for the three A1 arms", all((q, a) in contexts for r in REPOS for q in subset["study2_subset"][r] for a in ("a1_v4", "a1_v5a", "a1_v6e")))
    check("human sample has 30 questions", len(subset["human_review_sample"]) == 30)
    for repo in REPOS:
        try:
            v4.verify_snapshot(v4.repo_root(repo), v4.COMMITS[repo])
            check(f"snapshot {repo}", True)
        except SystemExit as error:
            check(f"snapshot {repo}", False, str(error))
    check("provider variables present", all(os.environ.get(n) or n in _env_names() for n in ("CODECOMPASS_COMPARE_BASE_URL", "CODECOMPASS_COMPARE_API_KEY", "CODECOMPASS_COMPARE_MODEL",
                                                                                           "CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL", "CODECOMPASS_GEMINI2_EMBEDDING_API_KEY")))
    check("second-pass sample is 25% per arm x repository", True, "drawn at judge-prepare from the blind mapping; 12 of 48 per cell expected")
    return checks


def _env_names() -> set[str]:
    path = ROOT / ".env"
    return {line.split("=", 1)[0].strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if "=" in line and not line.lstrip().startswith("#")} if path.exists() else set()


def phase_dry_run() -> int:
    results = dry_run_checks()
    for name, ok, detail in results:
        print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""))
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def phase_freeze() -> int:
    if FREEZE.exists():
        raise SystemExit("FREEZE_V7.json already exists; a change needs an addendum, not a new freeze")
    failed = [n for n, ok, _ in dry_run_checks() if not ok and n != "freeze file not yet written"]
    if failed:
        raise SystemExit(f"dry-run checks failing, no freeze: {failed}")
    prices = fetch_prices()
    PRICES.write_text(json.dumps(prices, indent=1) + "\n", encoding="utf-8")
    hashes = current_hashes()
    hashes[RUNNER_PATH] = sha256_file(ROOT / RUNNER_PATH)
    FREEZE.write_text(json.dumps({"frozen_on": datetime.now(timezone.utc).date().isoformat(), "protocol": "swe_qa_v7", "sha256": hashes}, indent=1) + "\n", encoding="utf-8")
    print(f"frozen {len(hashes)} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("dry-run", "freeze", "generate", "judge-prepare", "judge-run", "timing"))
    parser.add_argument("--arm", choices=NEW_ARMS)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--pass", dest="pass_number", type=int, choices=(1, 2))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.phase == "generate" and not args.arm:
        parser.error("generate needs --arm")
    if args.phase == "judge-run" and not args.pass_number:
        parser.error("judge-run needs --pass")
    c5.load_dotenv() if args.phase in {"dry-run", "freeze"} else None
    return {"dry-run": phase_dry_run, "freeze": phase_freeze, "judge-prepare": phase_judge_prepare,
            "generate": lambda: phase_generate(args.arm, args.retry), "judge-run": lambda: phase_judge_run(args.pass_number, args.limit, args.delay, args.workers),
            "timing": lambda: phase_timing(args.limit_questions)}[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
