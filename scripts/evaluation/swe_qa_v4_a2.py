"""Generate v4 A2 post-hoc citations for the frozen 30-question human sample."""
from __future__ import annotations

import json
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_qa_v2 as v2  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.providers import ProviderConfig, create_llm_provider  # type: ignore  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/evaluation/swe_qa_v4/frozen/a2_model_citations.jsonl"
RETRY = ROOT / "reports/evaluation/swe_qa_v4/frozen/a2_model_citations_retry.jsonl"
SAMPLE = ROOT / "reports/evaluation/swe_qa_v4/SAMPLE_FREEZE.json"
A1 = ROOT / "reports/evaluation/swe_qa_v4/frozen/main_a1_combined.jsonl"
DATA = ROOT / "data/benchmarks/swe_qa_v4"
ARTIFACT = ROOT / "data/benchmarks/swe_qa_v4_avalai_gemini2"
REPOS = {name: DATA / "repos" / name for name in ("flask", "requests", "pytest", "pylint")}
COMMITS = {"flask": "85c5d93cbd049c4bd0679c36fd1ddcae8c37b642", "requests": "46e939b5525d9c72b677340985582b04b128478a8", "pytest": "5989efe3efec31a9d14f550d89903b6311dfc828", "pylint": "44740e5e5cdd2b80321a4f0ed4254213d058b9f6"}
CITATION_RE = re.compile(r"(?P<path>[\w./-]+\.py):(?P<start>\d+)(?:-(?P<end>\d+))?")


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_citations(text: str) -> list[dict]:
    return [{"source_file": m.group("path"), "start_line": int(m.group("start")), "end_line": int(m.group("end") or m.group("start"))} for m in CITATION_RE.finditer(text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    output = RETRY if args.retry else OUT
    if output.exists() and output.stat().st_size:
        raise SystemExit(f"refusing to overwrite existing raw output: {output}")
    if not all(os.environ.get(k) for k in ("CODECOMPASS_COMPARE_BASE_URL", "CODECOMPASS_COMPARE_API_KEY", "CODECOMPASS_COMPARE_MODEL")):
        raise SystemExit("provider configuration is incomplete; no A2 calls were made")
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["human_review_sample"]["ids_in_order"]
    a1 = {r["question_id"]: r for r in (json.loads(line) for line in A1.read_text(encoding="utf-8").splitlines())}
    def fixed_context(row: dict) -> str:
        """Reconstruct the fixed A1 evidence from its deterministic citation metadata, offline."""
        blocks = []
        root = REPOS[row["repo"]]
        for number, citation in enumerate(row.get("citations", []), 1):
            path = root / citation["source_file"]
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, int(citation["start_line"]))
            end = min(len(lines), int(citation["end_line"]))
            code = "\n".join(lines[start - 1:end])
            blocks.append(f"[S{number}]\nfile: {citation['source_file']}\nlines: {start}-{end}\ncode:\n{code}")
        return "\n\n".join(blocks)
    llm = create_llm_provider(ProviderConfig(provider="openai_compatible", base_url=os.environ["CODECOMPASS_COMPARE_BASE_URL"], api_key=os.environ["CODECOMPASS_COMPARE_API_KEY"], llm_model=os.environ["CODECOMPASS_COMPARE_MODEL"], timeout_seconds=180))
    prior = {json.loads(line)["question_id"] for line in output.open(encoding="utf-8") if line.strip()} if output.exists() else set()
    failed = {json.loads(line)["question_id"] for line in OUT.open(encoding="utf-8") if line.strip() and json.loads(line).get("status") != "usable"} if args.retry else set()
    targets = [qid for qid in ids if qid in failed and qid not in prior] if args.retry else ids
    with output.open("a", encoding="utf-8") as handle:
        for qid in targets:
            base = a1[qid]; repo = base["repo"]
            prompt = v2.a2_prompt(base["question"], base.get("answer") or "", fixed_context(base))
            started = time.perf_counter(); last = None
            for attempt in range(1, 5):
                try:
                    response = llm.generate(LLMRequest(prompt=prompt, system_prompt=v2.A2_SYSTEM, temperature=0, max_tokens=1200))
                    status = "usable" if response.text.strip() and response.finish_reason == "stop" else ("truncated" if response.finish_reason == "length" else "unavailable")
                    record = {"protocol": "swe_qa_v4", "mode": "human", "arm": "a2_model_citations", "question_id": qid, "repo": repo, "question": base["question"], "fixed_a1_sha256": __import__("hashlib").sha256((base.get("answer") or "").encode()).hexdigest(), "status": status, "answer": response.text, "finish_reason": response.finish_reason, "model": response.model, "citations": parse_citations(response.text), "attempts": attempt, "latency_seconds": round(time.perf_counter() - started, 3)}
                    break
                except Exception as error:  # preserve provider failure
                    last = error
                    if attempt < 4: time.sleep(2 ** (attempt - 1))
            else:
                record = {"protocol": "swe_qa_v4", "mode": "human", "arm": "a2_model_citations", "question_id": qid, "repo": repo, "question": base["question"], "status": "unavailable", "error_type": type(last).__name__, "error": str(last)[:300], "attempts": 4, "citations": []}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush(); print(qid, record["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
