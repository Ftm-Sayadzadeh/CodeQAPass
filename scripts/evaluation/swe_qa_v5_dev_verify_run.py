"""Small two-pass semantic smoke test with post-generation verification."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.qa.models import QARequest  # noqa: E402

VERIFY_SYSTEM = "You verify grounded answers about Python code. Use only the provided code context. Remove or correct claims not supported by the context. Preserve supported detail and answer the question directly. Return only the revised answer, with no commentary about the verification process."

class CaptureProvider:
    def __init__(self, inner): self.inner=inner; self.last_request=None
    def generate(self, request): self.last_request=request; return self.inner.generate(request)

def load_dotenv():
    for raw in (ROOT/'.env').read_text(encoding='utf-8').splitlines():
        line=raw.strip()
        if line and not line.startswith('#') and '=' in line:
            name,value=line.split('=',1); os.environ.setdefault(name.strip(),value.strip().strip('"').strip("'"))

def main():
    load_dotenv(); os.environ.setdefault('CODECOMPASS_LLM_REASONING_EFFORT','low'); sample=json.loads((ROOT/'reports/evaluation/swe_qa_v5_dev/dev_sample.json').read_text(encoding='utf-8')); qids=[qid for repo in ('flask','requests','pytest','pylint') for qid in [x for x in sample if x.startswith(f'{repo}-')][:2]]; freeze=v4.load_freeze(); rows={r['question_id']:r for r in v4.select_rows('main',v4.DATA/'dataset',freeze)}; selected=[rows[q] for q in qids]; base_llm=v2.llm_from_env(); services={}
    for repo in sorted({r['repo'] for r in selected}):
        capture=CaptureProvider(base_llm); service,project=v4.build_a1_service(repo,capture,v4.DATA/'../swe_qa_v4_avalai_gemini2'); services[repo]=(service,project,capture)
    out=ROOT/'reports/evaluation/swe_qa_v5_dev/verify_smoke'; out.mkdir(parents=True,exist_ok=True); path=out/'verified.jsonl'
    if path.exists() and path.stat().st_size: raise SystemExit(f'refusing to overwrite {path}')
    with path.open('w',encoding='utf-8') as target:
        for row in selected:
            service,project,capture=services[row['repo']]; answer=service.answer(QARequest(question=row['question'],project_id=project,retrieval_method='semantic',retrieval_limit=5,max_context_chars=6000,temperature=0.0,max_tokens=1200)); request=capture.last_request; verify_prompt=f"Original question and code context:\n{request.prompt}\n\nDraft answer:\n{answer.answer}\n\nRewrite the draft only where needed so every factual statement is supported by the code context."
            verified=base_llm.generate(LLMRequest(prompt=verify_prompt,system_prompt=VERIFY_SYSTEM,temperature=0.0,max_tokens=1200)); target.write(json.dumps({'method':'verified','question_id':row['question_id'],'repo':row['repo'],'question':row['question'],'reference_answer':row.get('answer'),'answer':verified.text,'draft_answer':answer.answer,'citations':[{'source_file':c.source_file,'qualified_name':c.qualified_name,'start_line':c.start_line,'end_line':c.end_line} for c in answer.citations]},ensure_ascii=False)+'\n'); target.flush(); print(row['question_id'],verified.finish_reason or 'stop',flush=True)
if __name__=='__main__': main()
