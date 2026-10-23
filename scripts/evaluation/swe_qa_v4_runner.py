"""SWE-QA protocol v4 runner: answer generation for A0, B0-k5 and A1 under the frozen settings.

Offline phases (no network, no provider call):
    python scripts/evaluation/swe_qa_v4_runner.py check          # dataset files, samples, snapshot SHAs
    python scripts/evaluation/swe_qa_v4_runner.py count-b0       # eligible function-chunk counts per repository (budget proxy)
    python scripts/evaluation/swe_qa_v4_runner.py generate --arm a0 --mode pilot --dry-run

Phases that call providers (run only after explicit approval; the pilot needs a separate approval):
    python scripts/evaluation/swe_qa_v4_runner.py index-b0 --repos conan
    python scripts/evaluation/swe_qa_v4_runner.py generate --arm {a0,b0_k5,a1} --mode {pilot,main,human,persian} [--retry]

Frozen settings come from PROTOCOL_V4_DRAFT.md: temperature 0, max_tokens 1200, top-5 / 6,000 characters, reasoning effort `low`.
A1 runs through GroundedQAService. A0 and B0 use the same generation call settings; B0 builds its prompt with the same QAPromptBuilder.
Raw records are append-only; retries go to separate files; unavailable and truncated answers are never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_b0 as b0lib  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.qa import QAPromptBuilder  # noqa: E402
from codecompass.qa.models import NO_EVIDENCE_ANSWER, QARequest  # noqa: E402
from codecompass.qa.service import GroundedQAService  # noqa: E402
from codecompass.rag import RAGContextBuilder  # noqa: E402
from codecompass.retrieval import RetrievalQuery, RetrievalResult, RetrievedChunk  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/swe_qa_v4"
FREEZE_PATH = OUT / "SAMPLE_FREEZE.json"
DATA = ROOT / "data/benchmarks/swe_qa_v4"

MAIN_REPOS = ("flask", "requests", "pytest", "pylint")
PILOT_REPO = "conan"
# Official SWE-QA commits (full SHAs), identical to the table in PROTOCOL_V4_DRAFT.md section 2; a test keeps them in sync.
COMMITS = {
    "flask": "85c5d93cbd049c4bd0679c36fd1ddcae8c37b642",
    "requests": "46e939b5525d9c72b677340985582b04b128478a",
    "pytest": "5989efe3efec31a9d14f550d89903b6311dfc828",
    "pylint": "44740e5e5cdd2b80321a4f0ed4254213d058b9f6",
    "sphinx": "6c9e3209c4ebc81c7c09f6a30eeaa4651f32f461",
    "astropy": "0a041d38fbd0014298d4cf340d9f891c4522093f",
    "sympy": "3c817ed8ab551d71e36907ce869afe5e4ca789ff",
    "conan": "52f43d98fe44ab92d01317110488d0670c3e9e64",
}
REQUIRED_REASONING_EFFORT = "low"
TEMPERATURE = 0.0
MAX_TOKENS = 1200
RETRIEVAL_LIMIT = 5
MAX_CONTEXT_CHARS = 6000
CHUNK_CAP_PER_REPO = 50_000
MAX_ATTEMPTS = 4
ARMS = ("a0", "b0_k5", "a1")
MODES = ("pilot", "main", "human", "persian")


def embedding_run_identity(env: dict[str, str]) -> dict[str, object]:
    """Return the non-secret embedding identity that owns one artifact root."""
    base_url = env["CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL"].rstrip("/")
    return {
        "provider": env.get("CODECOMPASS_GEMINI2_EMBEDDING_PROVIDER", "openai_compatible"),
        "endpoint_sha256": hashlib.sha256(base_url.encode("utf-8")).hexdigest(),
        "model": env.get("CODECOMPASS_GEMINI2_EMBEDDING_MODEL", "gemini-embedding-2"),
        "dimensions": int(env.get("CODECOMPASS_GEMINI2_EMBEDDING_DIMENSIONS", "3072")),
    }


def ensure_embedding_run_identity(artifact_root: Path, env: dict[str, str]) -> dict[str, object]:
    """Create the identity guard once and refuse to mix another embedding configuration into the root."""
    expected = embedding_run_identity(env)
    path = artifact_root / "embedding_identity.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(f"embedding identity mismatch in {artifact_root}; refusing to mix artifacts")
        return actual
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return expected


# ---------------------------------------------------------------- guards and selection
def require_frozen_settings(env: dict[str, str] | None = None) -> None:
    """Stop unless the frozen reasoning effort is set; without it answers are not comparable with the other arms."""
    actual = (os.environ if env is None else env).get("CODECOMPASS_LLM_REASONING_EFFORT")
    if actual != REQUIRED_REASONING_EFFORT:
        raise SystemExit(f"CODECOMPASS_LLM_REASONING_EFFORT must be '{REQUIRED_REASONING_EFFORT}' (got {actual!r})")


def verify_snapshot(root: Path, expected_sha: str) -> str:
    """`git rev-parse HEAD` must equal the frozen SHA; on mismatch stop. Nothing is substituted automatically."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot read the commit of {root}: {error}") from error
    if head != expected_sha:
        raise SystemExit(f"snapshot mismatch for {root.name}: HEAD {head} != frozen {expected_sha}; log a deviation, do not substitute")
    return head


def load_freeze(path: Path = FREEZE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_dir: Path, repo: str) -> list[dict]:
    path = dataset_dir / f"{repo}.jsonl"
    if not path.exists():
        raise SystemExit(f"dataset file missing: {path}")
    rows = v2.load_rows(path, repo)
    if len(rows) != 48:
        raise SystemExit(f"{repo}: expected 48 questions, found {len(rows)}")
    return rows


def select_rows(mode: str, dataset_dir: Path, freeze: dict, repos: tuple[str, ...] | None = None) -> list[dict]:
    """Rows for a mode: pilot (15 conan ids), main (all 48 of each main repository), human (30 ids), persian (48 ids)."""
    if mode == "main":
        chosen = repos or MAIN_REPOS
        return [row for repo in chosen for row in load_dataset(dataset_dir, repo)]
    ids = {
        "pilot": freeze["pilot_sample"]["question_ids"],
        "human": freeze["human_review_sample"]["ids_in_order"],
        "persian": freeze["persian_sample"]["ids_in_order"],
    }[mode]
    needed = sorted({qid.rsplit("-", 1)[0] for qid in ids})
    by_id = {row["question_id"]: row for repo in needed for row in load_dataset(dataset_dir, repo)}
    missing = [qid for qid in ids if qid not in by_id]
    if missing:
        raise SystemExit(f"question ids not in the dataset: {missing}")
    return [by_id[qid] for qid in ids]


def status_of(text: str | None, finish_reason: str | None) -> str:
    if text and text.strip() and finish_reason == "stop":
        return "usable"
    return "truncated" if finish_reason == "length" else "unavailable"


# ---------------------------------------------------------------- arms
def attempt(call: Callable[[], dict], sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run a generation call with up to MAX_ATTEMPTS tries; failures are recorded, never converted to scores."""
    started = time.perf_counter()
    last: Exception | None = None
    for number in range(1, MAX_ATTEMPTS + 1):
        try:
            result = call()
        except Exception as error:  # noqa: BLE001 - provider and service errors must be preserved in the record
            last = error
            if number < MAX_ATTEMPTS:
                sleep(2 ** (number - 1))
            continue
        status = "no_evidence" if result.get("no_evidence") else status_of(result["text"], result["finish_reason"])
        return {"status": status, "answer": result["text"], "finish_reason": result["finish_reason"], "model": result.get("model"),
                "attempts": number, "latency_seconds": round(time.perf_counter() - started, 3),
                **{k: v for k, v in result.items() if k not in ("text", "finish_reason", "model", "no_evidence")}}
    return {"status": "unavailable", "answer": None, "finish_reason": None, "error_type": type(last).__name__,
            "error": str(last)[:300], "attempts": MAX_ATTEMPTS, "latency_seconds": round(time.perf_counter() - started, 3)}


def call_a0(llm, row: dict) -> Callable[[], dict]:
    def go() -> dict:
        response = llm.generate(LLMRequest(prompt=v2.a0_neutral_prompt(row["question"], row["repo"]), system_prompt=v2.A0_NEUTRAL_SYSTEM,
                                           temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
        return {"text": response.text, "finish_reason": response.finish_reason, "model": response.model, "citations": []}
    return go


def call_a1(service, project_id: int, row: dict) -> Callable[[], dict]:
    def go() -> dict:
        answer = service.answer(QARequest(question=row["question"], project_id=project_id, retrieval_method="hybrid",
                                          retrieval_limit=RETRIEVAL_LIMIT, max_context_chars=MAX_CONTEXT_CHARS,
                                          temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
        return {"text": answer.answer, "finish_reason": answer.finish_reason, "model": answer.llm_model,
                "citations": [asdict(c) for c in answer.citations], "omitted_context_count": answer.omitted_context_count,
                "no_evidence": answer.llm_model is None and answer.answer == NO_EVIDENCE_ANSWER}
    return go


def b0_context(row: dict, query_vector, index: tuple[list[dict], object], k: int = RETRIEVAL_LIMIT, budget: int = MAX_CONTEXT_CHARS):
    """Dense top-k over function and class chunks, built into a context with the same builder and budget as A1."""
    meta, matrix = index
    hits = b0lib.top_chunks(query_vector, matrix, k)
    retrieved = tuple(
        RetrievedChunk(chunk_id=meta[i]["chunk_id"], score=score, source_file=meta[i]["file"], symbol_name=meta[i]["symbol"],
                       qualified_name=meta[i]["qualified"], start_line=meta[i]["start_line"], end_line=meta[i]["end_line"],
                       code=meta[i]["text"], retrieval_method="semantic")
        for i, score in hits)
    return RAGContextBuilder().build(RetrievalResult(RetrievalQuery(row["question"], 0, k), retrieved), budget)


def call_b0(llm, embed: Callable[[list[str]], object], index: tuple[list[dict], object], row: dict) -> Callable[[], dict]:
    def go() -> dict:
        query = b0lib.normalize(embed([row["question"]]))[0]
        context = b0_context(row, query, index)
        chunk_ids = [block.chunk_id for block in context.blocks]
        if not context.blocks:
            return {"text": NO_EVIDENCE_ANSWER, "finish_reason": None, "model": None, "citations": [], "context_chunk_ids": [],
                    "no_evidence": True}
        system_prompt, prompt = QAPromptBuilder().build(row["question"], context)
        response = llm.generate(LLMRequest(prompt=prompt, system_prompt=system_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
        return {"text": response.text, "finish_reason": response.finish_reason, "model": response.model, "citations": [],
                "context_chunk_ids": chunk_ids, "omitted_context_count": context.omitted_count}
    return go


# ---------------------------------------------------------------- records and output
def make_record(arm: str, mode: str, row: dict, outcome: dict, snapshot_sha: str) -> dict:
    return {"protocol": "swe_qa_v4", "mode": mode, "arm": arm, "question_id": row["question_id"], "repo": row["repo"],
            "question": row["question"], "reference_answer": row.get("answer"), "snapshot_commit": snapshot_sha,
            "settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "retrieval_limit": RETRIEVAL_LIMIT,
                         "max_context_chars": MAX_CONTEXT_CHARS, "reasoning_effort": REQUIRED_REASONING_EFFORT},
            **outcome}


def output_path(out_dir: Path, mode: str, arm: str, retry: bool) -> Path:
    return out_dir / f"{mode}_{arm}{'_retry' if retry else ''}.jsonl"


def todo_rows(rows: list[dict], out_dir: Path, mode: str, arm: str, retry: bool) -> list[dict]:
    """Normal run: every row. Retry run: only rows whose original record is `unavailable` (retried once, separate file)."""
    if not retry:
        return rows
    original = v2.read_jsonl(output_path(out_dir, mode, arm, False))
    failed = {r["question_id"] for r in original if r.get("status") == "unavailable"}
    return [row for row in rows if row["question_id"] in failed]


def run_arm(arm: str, mode: str, rows: list[dict], out_dir: Path, retry: bool, make_call: Callable[[dict], Callable[[], dict]],
            snapshot_shas: dict[str, str], sleep: Callable[[float], None] = time.sleep) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = todo_rows(rows, out_dir, mode, arm, retry)
    with v2.open_new(output_path(out_dir, mode, arm, retry)) as target:
        for row in todo:
            record = make_record(arm, mode, row, attempt(make_call(row), sleep), snapshot_shas[row["repo"]])
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{arm} {row['question_id']} {record['status']}", flush=True)
    return len(todo)


# ---------------------------------------------------------------- offline and provider phases
def repo_root(repo: str) -> Path:
    return DATA / "repos" / repo


def count_b0_chunks(root: Path) -> int:
    """Eligible B0 function/class chunk count (a budget proxy; the A1 count comes from the product indexer's own scan)."""
    total = 0
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        total += len(b0lib.chunk_source(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="replace")))
    return total


def load_b0_index(repo: str, artifact_root: Path | None = None) -> tuple[list[dict], object]:
    import numpy as np

    base = (DATA if artifact_root is None else artifact_root) / "b0"
    meta = [json.loads(line) for line in (base / f"{repo}_chunks.jsonl").open(encoding="utf-8") if line.strip()]
    return meta, np.load(base / f"{repo}_vectors.npy")


def embed_with_rate_limit_backoff(embed: Callable[[list[str]], object], texts: list[str],
                                  sleep: Callable[[float], None] = time.sleep, waits: tuple[int, ...] = (30, 60, 120, 240)):
    """Retry an embedding call on HTTP 429 with long waits; other errors are raised unchanged."""
    for wait in waits:
        try:
            return embed(texts)
        except Exception as error:  # noqa: BLE001
            if "429" not in str(error):
                raise
            sleep(wait)
    return embed(texts)


def index_b0(repo: str, embed: Callable[[list[str]], object], delay_seconds: float = 1.0,
             sleep: Callable[[float], None] = time.sleep, artifact_root: Path | None = None) -> int:
    """Build the B0 index for one repository after the chunk cap check; refuses to overwrite an existing index.

    Each batch is stored in a partial directory as soon as it is embedded, so a later failure (for example rate limiting) does not
    repeat already paid embedding calls; the partial files are removed after the final index is written.
    """
    import shutil

    import numpy as np

    root = repo_root(repo)
    verify_snapshot(root, COMMITS[repo])
    chunks: list[dict] = []
    for path in sorted(root.rglob("*.py")):
        if ".git" not in path.parts:
            chunks.extend(b0lib.chunk_source(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="replace")))
    if len(chunks) > CHUNK_CAP_PER_REPO:
        raise SystemExit(f"{repo}: {len(chunks)} chunks exceed the cap {CHUNK_CAP_PER_REPO}; use the prespecified substitute before any generation")
    for number, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"b0-{repo}-{number}"
    base = (DATA if artifact_root is None else artifact_root) / "b0"
    base.mkdir(parents=True, exist_ok=True)
    meta_path, vec_path = base / f"{repo}_chunks.jsonl", base / f"{repo}_vectors.npy"
    if meta_path.exists() or vec_path.exists():
        raise SystemExit(f"refusing to overwrite the B0 index of {repo}")
    partial = base / f"_partial_{repo}"
    partial.mkdir(exist_ok=True)
    batches = []
    for number, start in enumerate(range(0, len(chunks), b0lib.BATCH_SIZE)):
        batch_file = partial / f"batch_{number:05d}.npy"
        if batch_file.exists():
            batches.append(np.load(batch_file))
            continue
        if number and delay_seconds:
            sleep(delay_seconds)
        vectors = np.asarray(embed([c["text"] for c in chunks[start:start + b0lib.BATCH_SIZE]]), dtype=np.float32)
        np.save(batch_file, vectors)
        batches.append(vectors)
    np.save(vec_path, b0lib.normalize(np.vstack(batches)))
    meta_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")
    shutil.rmtree(partial, ignore_errors=True)
    return len(chunks)


def build_a1_service(repo: str, llm, artifact_root: Path | None = None):
    from codecompass.retrieval import RetrievalService
    from codecompass.storage import SQLiteMetadataStore
    from codecompass.vector_index import ChromaVectorIndex

    root = DATA if artifact_root is None else artifact_root
    store = SQLiteMetadataStore(root / "index" / f"{repo}.sqlite")
    project = store.get_project_by_root(repo_root(repo))
    if project is None:
        raise SystemExit(f"not indexed with the product pipeline: {repo}")
    retrieval = RetrievalService(store, b0lib.embedding_provider(),
                                 ChromaVectorIndex(root / "index" / f"{repo}_chroma", f"swe-qa-v4-{repo}", managed=True, project_id=project.id))
    return GroundedQAService(retrieval, RAGContextBuilder(), QAPromptBuilder(), llm), project.id


def dry_run_report(arm: str, mode: str, rows: list[dict]) -> str:
    lines = [f"DRY RUN arm={arm} mode={mode}: {len(rows)} questions = {len(rows)} generation calls (embedding calls for B0/A1 retrieval not counted)"]
    by_repo: dict[str, int] = {}
    for row in rows:
        by_repo[row["repo"]] = by_repo.get(row["repo"], 0) + 1
    lines.append("per repository: " + ", ".join(f"{k}={v}" for k, v in sorted(by_repo.items())))
    if arm == "a0" and rows:
        lines.append("first A0 prompt:\n" + v2.a0_neutral_prompt(rows[0]["question"], rows[0]["repo"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("check", "count-b0", "index-b0", "generate"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--mode", choices=MODES, default="pilot")
    parser.add_argument("--repos", nargs="+")
    parser.add_argument("--dataset-dir", type=Path, default=DATA / "dataset")
    parser.add_argument("--artifact-root", type=Path, default=DATA,
                        help="Provider-specific root for B0/A1 indexes; its embedding identity is guarded")
    parser.add_argument("--out-dir", type=Path, default=OUT / "runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    freeze = load_freeze()
    repos = tuple(args.repos) if args.repos else None

    if args.phase == "check":
        for mode in MODES:
            rows = select_rows(mode, args.dataset_dir, freeze, repos if mode == "main" else None)
            print(f"{mode}: {len(rows)} questions")
        for repo in (*MAIN_REPOS, PILOT_REPO):
            if repo_root(repo).exists():
                print(repo, "snapshot", verify_snapshot(repo_root(repo), COMMITS[repo]))
            else:
                print(repo, "not cloned yet")
        return 0
    if args.phase == "count-b0":
        for repo in repos or (*MAIN_REPOS, PILOT_REPO):
            count = count_b0_chunks(repo_root(repo))
            print(f"{repo}: {count} eligible B0 chunks (cap {CHUNK_CAP_PER_REPO})", "OVER CAP" if count > CHUNK_CAP_PER_REPO else "")
        return 0
    if args.phase == "index-b0":
        require_frozen_settings()
        ensure_embedding_run_identity(args.artifact_root, dict(os.environ))
        provider = b0lib.embedding_provider()
        embed = lambda texts: embed_with_rate_limit_backoff(lambda t: b0lib.embed_with_retry(provider, t), texts)  # noqa: E731
        for repo in repos or (PILOT_REPO,):
            print(repo, index_b0(repo, embed, artifact_root=args.artifact_root), "chunks")
        return 0

    if not args.arm:
        parser.error("generate needs --arm")
    rows = select_rows(args.mode, args.dataset_dir, freeze, repos if args.mode == "main" else None)
    if args.dry_run:
        print(dry_run_report(args.arm, args.mode, rows))
        return 0
    require_frozen_settings()
    if args.arm in ("b0_k5", "a1"):
        ensure_embedding_run_identity(args.artifact_root, dict(os.environ))
    shas = {repo: verify_snapshot(repo_root(repo), COMMITS[repo]) for repo in sorted({r["repo"] for r in rows})}
    llm = v2.llm_from_env()
    if args.arm == "a0":
        make_call = lambda row: call_a0(llm, row)  # noqa: E731
    elif args.arm == "b0_k5":
        provider = b0lib.embedding_provider()
        embed = lambda texts: b0lib.embed_with_retry(provider, texts)  # noqa: E731
        indexes = {repo: load_b0_index(repo, args.artifact_root) for repo in shas}
        make_call = lambda row: call_b0(llm, embed, indexes[row["repo"]], row)  # noqa: E731
    else:
        services = {repo: build_a1_service(repo, llm, args.artifact_root) for repo in shas}
        make_call = lambda row: call_a1(services[row["repo"]][0], services[row["repo"]][1], row)  # noqa: E731
    run_arm(args.arm, args.mode, rows, args.out_dir, args.retry, make_call, shas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
