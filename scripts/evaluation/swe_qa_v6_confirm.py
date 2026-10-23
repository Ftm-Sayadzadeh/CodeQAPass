"""SWE-QA v6 runner: eight arms on two fresh repositories (streamlink, sqlfluff). See reports/evaluation/swe_qa_v6/PROTOCOL_V6_DRAFT.md.

Phases (outputs are append-only and never overwritten; retries go to separate files):

    python scripts/evaluation/swe_qa_v6_confirm.py check
    python scripts/evaluation/swe_qa_v6_confirm.py rank                  # embedding calls only (resumable)
    python scripts/evaluation/swe_qa_v6_confirm.py contexts              # offline: build/store every arm's context, replica check, coverage
    python scripts/evaluation/swe_qa_v6_confirm.py generate --arm ARM    # LLM calls [--retry]
    python scripts/evaluation/swe_qa_v6_confirm.py judge-prepare
    python scripts/evaluation/swe_qa_v6_confirm.py judge-run --pass 1 [--workers 3] [--limit N] [--delay S]
    python scripts/evaluation/swe_qa_v6_confirm.py analyze

Source identity (file, symbol, line range, chunk id) comes from SQLite chunk metadata and line counting only.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v6_env as env  # noqa: E402

env.configure()
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

OUT = env.OUT / "fresh"
ARMS = ("b0_k5_p", "a1_v5a", "a1_v6e", "a1_v6")  # cost-reduced design, see protocol section 3
CONTEXT_OF = {"b0_k5_p": "b0", "a1_v5a": "a1_v5a", "a1_v6e": "a1_v6", "a1_v6": "a1_v6"}
PARTIAL_ARMS = {"b0_k5_p", "a1_v6"}
CONTEXT_KEYS = ("b0", "a1_cur", "a1_v5a", "a1_v6")
SEED = 20260930
BOOTSTRAP = 20_000
NI_MARGIN = 1.5
SAFETY_MARGIN = 0.05
HEDGE_RE = re.compile(r"not enough retrieved evidence|not enough evidence|insufficient|cannot be (determined|established)|"
                      r"not (shown|visible|included|provided) in (the )?(provided|retrieved)", re.IGNORECASE)
BUILTIN_NAMES = gate.IDENT_SKIP | {"print", "return", "class", "def", "import", "from", "lambda", "yield", "async", "await"}
SCORE_KEYS = v4judge.SCORE_KEYS


# ---------------------------------------------------------------- helpers
def rows_all() -> list[dict]:
    return [row for repo in env.REPOS for row in v4.load_dataset(env.DATA / "dataset", repo)]


def prompt_builder(arm: str):
    return v6.PartialPromptBuilder() if arm in PARTIAL_ARMS else QAPromptBuilder()


def unsupported_identifiers(answer: str, context_text: str, question: str) -> tuple[int, int]:
    """(unsupported, total) unique backticked identifiers of an answer that appear neither in the delivered context nor in the question."""
    names = {n.lower() for token in re.findall(r"`([^`\n]+)`", answer or "") for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", token.replace("()", ""))
             if len(n) >= 4 and n.lower() not in BUILTIN_NAMES}
    haystack = (context_text + "\n" + question).lower()
    return sum(1 for n in names if n not in haystack), len(names)


def is_hedged(record: dict) -> bool:
    """Answers that contain insufficiency language but are not leading refusals."""
    return not c5.is_refusal(record) and bool(HEDGE_RE.search(record.get("answer") or ""))


def read_jsonl(path: Path) -> list[dict]:
    return c5.read_jsonl(path)


# ---------------------------------------------------------------- phase: rank (embedding calls only)
def phase_rank() -> int:
    c5.load_dotenv()
    v4.ensure_embedding_run_identity(env.ARTIFACT_ROOT, dict(os.environ))
    rows = rows_all()
    provider = b0lib.embedding_provider()
    paths = {name: OUT / f"rankings_{name}.jsonl" for name in ("semantic", "lexical", "b0")}
    paths["service"] = OUT / "a1_cur_service_context.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set.intersection(*({r["question_id"] for r in read_jsonl(p)} if p.exists() else set() for p in paths.values()))
    handles = {name: path.open("a", encoding="utf-8") for name, path in paths.items()}
    services, b0_indexes = {}, {}
    try:
        for row in rows:
            qid, repo, question = row["question_id"], row["repo"], row["question"]
            if qid in done:
                continue
            if repo not in services:
                v4.verify_snapshot(v4.repo_root(repo), v4.COMMITS[repo])
                capture = c5.CaptureLLM()
                service, project_id = v4.build_a1_service(repo, capture, env.ARTIFACT_ROOT)
                services[repo] = (service, project_id, capture)
                b0_indexes[repo] = v4.load_b0_index(repo, env.ARTIFACT_ROOT)
            service, project_id, capture = services[repo]
            meta, matrix = b0_indexes[repo]
            lines: dict[str, dict] = {}
            for name, search in (("semantic", service.retrieval_service.search_semantic), ("lexical", service.retrieval_service.search_lexical)):
                result = c5.with_backoff(lambda s=search: s(RetrievalQuery(question, project_id, 10)))
                lines[name] = {"question_id": qid, "repo": repo, "ranking": [
                    {"chunk_id": c.chunk_id, "file": c.source_file, "start_line": c.start_line, "end_line": c.end_line} for c in result.results]}
            vector = b0lib.normalize(c5.with_backoff(lambda: b0lib.embed_with_retry(provider, [question])))[0]
            lines["b0"] = {"question_id": qid, "repo": repo, "ranking": [
                {"chunk_id": meta[i]["chunk_id"], "file": meta[i]["file"], "start_line": meta[i]["start_line"], "end_line": meta[i]["end_line"], "score": s}
                for i, s in b0lib.top_chunks(vector, matrix, 10)]}
            answer = c5.with_backoff(lambda: service.answer(QARequest(question=question, project_id=project_id, retrieval_method="hybrid",
                                                                      retrieval_limit=v4.RETRIEVAL_LIMIT, max_context_chars=c5.BUDGET, temperature=0.0,
                                                                      max_tokens=v4.MAX_TOKENS)))
            lines["service"] = {"question_id": qid, "chunk_ids": [c.chunk_id for c in answer.citations], "omitted_context_count": answer.omitted_context_count}
            for name, line in lines.items():
                handles[name].write(json.dumps(line) + "\n")
                handles[name].flush()
            print("ranked", qid, flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    return 0


# ---------------------------------------------------------------- phase: contexts (offline)
def coverage(rows: list[dict], contexts: dict, key: str) -> dict:
    files_cache: dict[str, list[str]] = {}
    cov, ident, blocks, chars = [], [], [], []
    from swe_qa_retrieval_metrics import gold_files

    for row in rows:
        repo = row["repo"]
        if repo not in files_cache:
            root = v4.repo_root(repo)
            files_cache[repo] = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if ".git" not in p.parts)
        ctx = contexts[(row["question_id"], key)]["blocks"]
        gold = set(gold_files(row.get("answer") or "", files_cache[repo]))
        ids = gate.identifiers(row.get("answer") or "")
        text = "\n".join(b["code"] for b in ctx).lower()
        blocks.append(len(ctx))
        chars.append(len(text))
        if gold:
            cov.append(len({b["source_file"] for b in ctx} & gold) / len(gold))
        if ids:
            ident.append(sum(n in text for n in ids) / len(ids))
    return {"gold_cov": round(stats.mean(cov), 3) if cov else None, "gold_questions": len(cov), "ident_cov": round(stats.mean(ident), 3) if ident else None,
            "blocks": round(stats.mean(blocks), 2), "chars": round(stats.mean(chars))}


def phase_contexts() -> int:
    rows = rows_all()
    semantic = {r["question_id"]: r["ranking"] for r in read_jsonl(OUT / "rankings_semantic.jsonl")}
    lexical = {r["question_id"]: r["ranking"] for r in read_jsonl(OUT / "rankings_lexical.jsonl")}
    b0_rank = {r["question_id"]: r["ranking"] for r in read_jsonl(OUT / "rankings_b0.jsonl")}
    service = {r["question_id"]: r for r in read_jsonl(OUT / "a1_cur_service_context.jsonl")}
    indexes = {repo: gate.Index(repo) for repo in env.REPOS}
    metas = {repo: v4.load_b0_index(repo, env.ARTIFACT_ROOT)[0] for repo in env.REPOS}
    mismatches = []
    with c5.append_new(OUT / "contexts.jsonl") as target:
        for row in rows:
            qid, repo, question = row["question_id"], row["repo"], row["question"]
            ranked5 = gate.fuse(indexes[repo], semantic[qid], lexical[qid], 5)
            depth10 = gate.fuse(indexes[repo], semantic[qid], lexical[qid], 10)
            built = gate.variants(question, ranked5, depth10)
            contexts = {}
            for key, pieces in (("a1_cur", built["baseline"]), ("a1_v5a", built["hubless_soft_k5"]), ("a1_v6", v6.v6_pieces(question, depth10))):
                chosen, omitted = gate.pack(pieces)
                contexts[key] = (c5.blocks_from_pieces(chosen), omitted)
            if [b["chunk_id"] for b in contexts["a1_cur"][0]] != service[qid]["chunk_ids"]:
                mismatches.append(qid)
            contexts["b0"] = c5.b0_arm_context(row, b0_rank[qid], metas[repo], 5)
            for key, (blocks, omitted) in contexts.items():
                text = "\n".join(b["code"] for b in blocks)
                target.write(json.dumps({"question_id": qid, "arm": key, "blocks": blocks, "omitted_context_count": omitted,
                                         "context_sha256": hashlib.sha256(text.encode()).hexdigest()}, ensure_ascii=False) + "\n")
    stored = {(r["question_id"], r["arm"]): r for r in read_jsonl(OUT / "contexts.jsonl")}
    report = {"questions": len(rows), "a1_cur_replica_mismatches": mismatches,
              "offline_coverage": {key: coverage(rows, stored, key) for key in CONTEXT_KEYS},
              "by_repo": {repo: {key: coverage([r for r in rows if r["repo"] == repo], stored, key) for key in CONTEXT_KEYS} for repo in env.REPOS}}
    (OUT / "offline_coverage.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    if mismatches:
        raise SystemExit("A1-cur replica differs from the production service context; investigate before generating")
    return 0


# ---------------------------------------------------------------- phase: generate
def load_contexts() -> dict[tuple[str, str], dict]:
    return {(r["question_id"], r["arm"]): r for r in read_jsonl(OUT / "contexts.jsonl")}


def call_for(arm: str, llm, row: dict, contexts: dict):
    if arm == "a0":
        return v4.call_a0(llm, row)
    record = contexts[(row["question_id"], CONTEXT_OF[arm])]
    builder = prompt_builder(arm)

    def go() -> dict:
        blocks = record["blocks"]
        if not blocks:
            return {"text": NO_EVIDENCE_ANSWER, "finish_reason": None, "model": None, "citations": [], "context_chunk_ids": [],
                    "omitted_context_count": record["omitted_context_count"], "no_evidence": True}
        system_prompt, prompt = builder.build(row["question"], c5.context_from_blocks(blocks, record["omitted_context_count"]))
        response = llm.generate(LLMRequest(prompt=prompt, system_prompt=system_prompt, temperature=v4.TEMPERATURE, max_tokens=v4.MAX_TOKENS))
        citations = [{"chunk_id": b["chunk_id"], "parent_chunk_id": v6.parent_chunk_id(b["chunk_id"]), "source_file": b["source_file"],
                      "symbol_name": b["symbol_name"], "qualified_name": b["qualified_name"], "start_line": b["start_line"], "end_line": b["end_line"]}
                     for b in blocks] if arm.startswith("a1") else []
        return {"text": response.text, "finish_reason": response.finish_reason, "model": response.model, "citations": citations,
                "context_chunk_ids": [b["chunk_id"] for b in blocks], "omitted_context_count": record["omitted_context_count"],
                "context_sha256": record["context_sha256"], "prompt": "partial" if arm in PARTIAL_ARMS else "strict"}
    return go


def pending_rows(rows: list[dict], original: list[dict]) -> list[dict]:
    """Rows without a usable (or explicit no-evidence) record in the original file: unavailable answers and questions never attempted."""
    done = {r["question_id"] for r in original if r.get("status") in {"usable", "no_evidence"}}
    return [row for row in rows if row["question_id"] not in done]


def attempt_rate_limited(call):
    """`v4.attempt` with six tries and waits of 15, 30, 60, 120 and 240 seconds, so provider rate limits (HTTP 429) are waited out, not recorded as failures."""
    saved = v4.MAX_ATTEMPTS
    v4.MAX_ATTEMPTS = 6
    try:
        return v4.attempt(call, sleep=lambda seconds: time.sleep(15 * seconds))
    finally:
        v4.MAX_ATTEMPTS = saved


def phase_generate(arm: str, retry: bool) -> int:
    c5.load_dotenv()
    v4.require_frozen_settings()
    rows = rows_all()
    shas = {repo: v4.verify_snapshot(v4.repo_root(repo), v4.COMMITS[repo]) for repo in env.REPOS}
    contexts = load_contexts() if arm != "a0" else {}
    llm = v2.llm_from_env()
    path = OUT / f"generation_{arm}{'_retry' if retry else ''}.jsonl"
    if retry:
        rows = pending_rows(rows, read_jsonl(OUT / f"generation_{arm}.jsonl"))
    with c5.append_new(path) as target:
        for row in rows:
            outcome = attempt_rate_limited(call_for(arm, llm, row, contexts))
            record = {"protocol": "swe_qa_v6", "arm": arm, "question_id": row["question_id"], "repo": row["repo"], "question": row["question"],
                      "reference_answer": row.get("answer"), "snapshot_commit": shas[row["repo"]],
                      "settings": {"temperature": v4.TEMPERATURE, "max_tokens": v4.MAX_TOKENS, "max_context_chars": c5.BUDGET,
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
        blind = f"M{number:04d}"
        inputs.append({"blind_id": blind, "question": r["question"], "reference_answer": r["reference_answer"], "candidate": r["candidate"]})
        mapping[blind] = {"question_id": r["question_id"], "arm": r["arm"]}
    input_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs), encoding="utf-8")
    key_path.write_text(json.dumps({"seed": SEED, "records": len(records), "mapping": mapping}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}))
    return 0


CREDIT_MARKERS = ("insufficient", "exhausted", "quota_exceeded")


def is_credit_error(message: str) -> bool:
    """True for provider messages meaning the account has no credit (retrying cannot help)."""
    lowered = message.lower()
    return any(marker in lowered for marker in CREDIT_MARKERS)


def judge_one(provider, row: dict, pass_number: int) -> dict:
    result = {"blind_id": row["blind_id"], "pass": pass_number, "judge_model": v4judge.MODEL, "temperature": v4judge.TEMPERATURE,
              "max_tokens": v4judge.MAX_TOKENS}
    prompt = v4judge.official_prompt(row["question"], row["reference_answer"], row["candidate"])
    for attempt in range(1, 5):
        try:
            response = provider.generate(LLMRequest(prompt=prompt, temperature=v4judge.TEMPERATURE, max_tokens=v4judge.MAX_TOKENS, response_format="json"))
            scores = v4judge.parse_scores(response.text)
            result.update({"status": "usable", "scores": scores, "mean": sum(scores.values()) / len(SCORE_KEYS), "attempts": attempt,
                           "finish_reason": response.finish_reason})
            return result
        except Exception as error:  # noqa: BLE001 - preserved as unavailable, never scored
            result.update({"status": "unavailable", "error_type": type(error).__name__, "error": str(error)[:300], "attempts": attempt})
            if is_credit_error(str(error)):
                result["error_type"] = "CreditExhausted"
                return result
            if attempt < 4:
                time.sleep((5, 30, 90)[attempt - 1] if "429" in str(error) else 2 ** (attempt - 1))
    return result


def phase_judge_run(pass_number: int, limit: int | None, delay: float, workers: int) -> int:
    c5.load_dotenv()
    from codecompass.providers import ProviderConfig, create_llm_provider

    rows = read_jsonl(OUT / "judge_input_blinded.jsonl")
    out_path = OUT / f"judge_pass{pass_number}.jsonl"
    done = {r["blind_id"] for r in read_jsonl(out_path) if r.get("status") == "usable"} if out_path.exists() else set()
    todo = [r for r in rows if r["blind_id"] not in done][:limit]
    provider = create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"],
                                                  api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model=v4judge.MODEL, timeout_seconds=180))
    lock = threading.Lock()
    stop = threading.Event()
    with out_path.open("a", encoding="utf-8") as target:
        def work(row: dict) -> None:
            if stop.is_set():
                return
            result = judge_one(provider, row, pass_number)
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


# ---------------------------------------------------------------- phase: analyze
def unit_scores() -> dict[tuple[str, str], dict]:
    key = json.loads((OUT / "judge_key.json").read_text(encoding="utf-8"))["mapping"]
    per: dict[tuple[str, str], dict[int, dict]] = {}
    for pass_number in (1, 2):
        path = OUT / f"judge_pass{pass_number}.jsonl"
        if not path.exists():
            continue
        latest = {r["blind_id"]: r for r in read_jsonl(path) if r.get("status") == "usable"}
        for blind, r in latest.items():
            info = key[blind]
            per.setdefault((info["question_id"], info["arm"]), {})[pass_number] = r
    return {unit: {"mean": stats.mean(p["mean"] for p in passes.values()), "passes": len(passes),
                   **{k: stats.mean(p["scores"][k] for p in passes.values()) for k in SCORE_KEYS}} for unit, passes in per.items()}


def compare(units: dict, left: str, right: str, rows: list[dict], metric: str = "mean") -> dict:
    groups: dict[str, list[float]] = {}
    for row in rows:
        a, b = units.get((row["question_id"], left)), units.get((row["question_id"], right))
        if a and b:
            groups.setdefault(row["repo"], []).append(a[metric] - b[metric])
    deltas = [d for g in groups.values() for d in g]
    if len(deltas) < 2:
        return {"pairs": len(deltas), "status": "insufficient"}
    low, high = v4judge.bootstrap_stratified(groups, BOOTSTRAP)
    return {"pairs": len(deltas), "mean_difference": round(stats.mean(deltas), 3), "bootstrap_ci95": [round(low, 3), round(high, 3)],
            "wilcoxon_p": v4judge.wilcoxon_p(deltas), "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas)}


def phase_analyze() -> int:
    rows = rows_all()
    units = unit_scores()
    gen = {arm: latest_generation(arm) for arm in ARMS}
    contexts = load_contexts()
    summary: dict = {"questions": len(rows), "arms": {}}
    for arm in ARMS:
        usable = [r for r in rows if (r["question_id"], arm) in units]
        record = {"judged": len(usable), "mean_score": round(stats.mean(units[(r["question_id"], arm)]["mean"] for r in usable), 3) if usable else None,
                  "criteria": {k: round(stats.mean(units[(r["question_id"], arm)][k] for r in usable), 3) for k in SCORE_KEYS} if usable else None,
                  "by_repo": {repo: round(stats.mean(units[(r["question_id"], arm)]["mean"] for r in usable if r["repo"] == repo), 3)
                              for repo in env.REPOS if any(r["repo"] == repo for r in usable)},
                  "status_counts": {s: sum(1 for x in gen[arm].values() if x["status"] == s) for s in sorted({x["status"] for x in gen[arm].values()})},
                  "refusal_rate": round(sum(c5.is_refusal(x) for x in gen[arm].values()) / max(len(gen[arm]), 1), 3),
                  "hedged_rate": round(sum(is_hedged(x) for x in gen[arm].values()) / max(len(gen[arm]), 1), 3)}
        if arm != "a0":
            unsupported = total = 0
            for r in rows:
                answer = gen[arm].get(r["question_id"])
                if not answer or answer["status"] != "usable":
                    continue
                ctx = contexts[(r["question_id"], CONTEXT_OF[arm])]["blocks"]
                u, t = unsupported_identifiers(answer["answer"], "\n".join(b["code"] for b in ctx), r["question"])
                unsupported += u
                total += t
            record["unsupported_identifier_share"] = round(unsupported / total, 4) if total else None
            record["identifiers_checked"] = total
        summary["arms"][arm] = record
    h1, h2 = compare(units, "a1_v6", "a1_v5a", rows), compare(units, "a1_v6", "b0_k5_p", rows)
    if h2.get("bootstrap_ci95"):
        h2["non_inferior_margin_1.5"] = h2["bootstrap_ci95"][0] > -NI_MARGIN
        h2["superior"] = h2["bootstrap_ci95"][0] > 0
    holm = v4judge.holm({"H1": h1.get("wilcoxon_p"), "H2": h2.get("wilcoxon_p")})
    share = {arm: summary["arms"][arm].get("unsupported_identifier_share") for arm in ("a1_v5a", "a1_v6")}
    flag = share["a1_v6"] is not None and share["a1_v5a"] is not None and share["a1_v6"] - share["a1_v5a"] > SAFETY_MARGIN
    summary["primary"] = {"H1_v6_minus_v5a": h1, "H2_v6_minus_b0_k5_p": h2, "holm_adjusted_p": holm,
                          "safety_flag_raised_unsupported_share_above_v5a_by_5pp": flag}
    pairs = (("a1_v6e", "a1_v5a"), ("a1_v6", "a1_v6e"), ("a1_v5a", "b0_k5_p"), ("a1_v6e", "b0_k5_p"))
    summary["exploratory"] = {f"{a}_minus_{b}": compare(units, a, b, rows) for a, b in pairs}
    summary["exploratory_criteria"] = {f"{a}_minus_{b}_{k}": compare(units, a, b, rows, k) for a, b in (("a1_v6", "a1_v5a"), ("a1_v6", "b0_k5_p"))
                                       for k in ("correctness", "completeness")}
    stored = {(r["question_id"], r["arm"]): r for r in read_jsonl(OUT / "contexts.jsonl")}
    summary["delivered_coverage"] = {key: coverage(rows, stored, key) for key in CONTEXT_KEYS}
    pass1 = {r["blind_id"]: r["mean"] for r in read_jsonl(OUT / "judge_pass1.jsonl") if r.get("status") == "usable"} if (OUT / "judge_pass1.jsonl").exists() else {}
    pass2 = {r["blind_id"]: r["mean"] for r in read_jsonl(OUT / "judge_pass2.jsonl") if r.get("status") == "usable"} if (OUT / "judge_pass2.jsonl").exists() else {}
    common = sorted(set(pass1) & set(pass2))
    summary["judge_pass_agreement"] = {"n": len(common), "spearman": c5.spearman([pass1[b] for b in common], [pass2[b] for b in common]),
                                       "mean_abs_difference": round(stats.mean(abs(pass1[b] - pass2[b]) for b in common), 3) if common else None}
    target = OUT / "analysis_fresh.json"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target.name}")
    target.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def phase_check() -> int:
    rows = rows_all()
    counts = {repo: sum(1 for x in rows if x["repo"] == repo) for repo in env.REPOS}
    print(f"fresh questions: {len(rows)} {counts}")
    for repo in env.REPOS:
        print(repo, "snapshot", v4.verify_snapshot(v4.repo_root(repo), v4.COMMITS[repo]), "| indexed:",
              (env.ARTIFACT_ROOT / "index" / f"{repo}.sqlite").exists(), "| b0:", (env.ARTIFACT_ROOT / "b0" / f"{repo}_vectors.npy").exists())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("check", "rank", "contexts", "generate", "judge-prepare", "judge-run", "analyze"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--pass", dest="pass_number", type=int, choices=(1, 2))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.phase == "generate" and not args.arm:
        parser.error("generate needs --arm")
    if args.phase == "judge-run" and not args.pass_number:
        parser.error("judge-run needs --pass")
    return {"check": phase_check, "rank": phase_rank, "contexts": phase_contexts, "judge-prepare": phase_judge_prepare, "analyze": phase_analyze,
            "generate": lambda: phase_generate(args.arm, args.retry),
            "judge-run": lambda: phase_judge_run(args.pass_number, args.limit, args.delay, args.workers)}[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
