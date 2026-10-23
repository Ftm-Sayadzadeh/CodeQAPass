"""Small semantic smoke test for partial-answer prompting."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.qa.models import QARequest  # noqa: E402
from codecompass.qa.prompt import QAPromptBuilder, SYSTEM_PROMPT  # noqa: E402

PARTIAL_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "If the context is insufficient, say that there is not enough retrieved evidence.",
    "If the context is incomplete but contains relevant evidence, answer the supported parts and explicitly state what cannot be established. Refuse only when there is no relevant evidence at all.",
)

class PartialPromptBuilder(QAPromptBuilder):
    def build(self, question, context):
        _, prompt = super().build(question, context)
        return PARTIAL_SYSTEM_PROMPT, prompt

def load_dotenv() -> None:
    for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))

def main() -> int:
    load_dotenv(); os.environ.setdefault("CODECOMPASS_LLM_REASONING_EFFORT", "low")
    qids = [qid for repo in ("flask", "requests", "pytest", "pylint") for qid in [x for x in json.loads((ROOT / "reports/evaluation/swe_qa_v5_dev/dev_sample.json").read_text()) if x.startswith(f"{repo}-")][:2]]
    freeze = v4.load_freeze(); rows = {r["question_id"]: r for r in v4.select_rows("main", v4.DATA / "dataset", freeze)}; selected = [rows[q] for q in qids]
    llm = v2.llm_from_env(); services = {}
    for repo in sorted({r["repo"] for r in selected}):
        service, project_id = v4.build_a1_service(repo, llm, v4.DATA / "../swe_qa_v4_avalai_gemini2"); service.prompt_builder = PartialPromptBuilder(); services[repo] = (service, project_id)
    out = ROOT / "reports/evaluation/swe_qa_v5_dev/partial_prompt"; out.mkdir(parents=True, exist_ok=True); path = out / "partial.jsonl"
    if path.exists() and path.stat().st_size: raise SystemExit(f"refusing to overwrite {path}")
    with path.open("w", encoding="utf-8") as target:
        for row in selected:
            service, project_id = services[row["repo"]]; answer = service.answer(QARequest(question=row["question"], project_id=project_id, retrieval_method="semantic", retrieval_limit=5, max_context_chars=6000, temperature=0.0, max_tokens=1200))
            target.write(json.dumps({"method": "partial_prompt", "question_id": row["question_id"], "repo": row["repo"], "question": row["question"], "reference_answer": row.get("answer"), "answer": answer.answer, "omitted_context_count": answer.omitted_context_count, "citations": [{"source_file": c.source_file, "qualified_name": c.qualified_name, "start_line": c.start_line, "end_line": c.end_line} for c in answer.citations], "model": answer.llm_model, "finish_reason": answer.finish_reason}, ensure_ascii=False) + "\n"); target.flush(); print(row["question_id"], answer.finish_reason or "no_evidence", flush=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
