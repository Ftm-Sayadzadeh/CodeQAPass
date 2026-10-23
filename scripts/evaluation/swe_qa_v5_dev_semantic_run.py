"""Paired hybrid versus semantic-only dev run on the fixed 8-question sample."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.qa.models import QARequest  # noqa: E402


def load_dotenv() -> None:
    for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv()
    os.environ.setdefault("CODECOMPASS_LLM_REASONING_EFFORT", "low")
    sample_all = json.loads((ROOT / "reports/evaluation/swe_qa_v5_dev/dev_sample.json").read_text(encoding="utf-8"))
    qids = [qid for repo in ("flask", "requests", "pytest", "pylint") for qid in [x for x in sample_all if x.startswith(f"{repo}-")][:2]]
    freeze = v4.load_freeze()
    rows = {row["question_id"]: row for row in v4.select_rows("main", v4.DATA / "dataset", freeze)}
    selected = [rows[qid] for qid in qids]
    llm = v2.llm_from_env()
    services = {repo: v4.build_a1_service(repo, llm, v4.DATA / "../swe_qa_v4_avalai_gemini2") for repo in sorted({row["repo"] for row in selected})}
    out = ROOT / "reports/evaluation/swe_qa_v5_dev/semantic_smoke"
    out.mkdir(parents=True, exist_ok=True)
    for method in ("hybrid", "semantic"):
        path = out / f"{method}.jsonl"
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite {path}")
        with path.open("w", encoding="utf-8") as target:
            for row in selected:
                service, project_id = services[row["repo"]]
                answer = service.answer(QARequest(question=row["question"], project_id=project_id, retrieval_method=method,
                                                  retrieval_limit=5, max_context_chars=6000, temperature=0.0, max_tokens=1200))
                record = {"method": method, "question_id": row["question_id"], "repo": row["repo"], "question": row["question"],
                          "reference_answer": row.get("answer"), "answer": answer.answer, "omitted_context_count": answer.omitted_context_count,
                          "citations": [{"source_file": c.source_file, "qualified_name": c.qualified_name, "start_line": c.start_line, "end_line": c.end_line} for c in answer.citations],
                          "model": answer.llm_model, "finish_reason": answer.finish_reason}
                target.write(json.dumps(record, ensure_ascii=False) + "\n"); target.flush()
                print(method, row["question_id"], answer.finish_reason or "no_evidence", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
