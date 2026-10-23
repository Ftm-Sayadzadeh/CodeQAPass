"""Small semantic multi-query smoke test for long, multi-part questions."""
from __future__ import annotations
import json, os, re, sys
from dataclasses import replace
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.qa.models import QARequest  # noqa: E402
from codecompass.retrieval import RetrievalQuery, RetrievalResult  # noqa: E402

def load_dotenv():
    for raw in (ROOT/'.env').read_text(encoding='utf-8').splitlines():
        line=raw.strip()
        if line and not line.startswith('#') and '=' in line:
            name,value=line.split('=',1); os.environ.setdefault(name.strip(),value.strip().strip('"').strip("'"))

def subqueries(question: str) -> list[str]:
    if len(question) < 120: return [question]
    parts = [p.strip(' ,?') for p in re.split(r'\s+(?:and|while)\s+', question, flags=re.I) if p.strip()]
    if len(parts) < 2: return [question]
    return [p if p.lower().startswith(('what','how','why','which','does','is','are')) else f"What {p}?" for p in parts]

def main():
    load_dotenv(); os.environ.setdefault('CODECOMPASS_LLM_REASONING_EFFORT','low'); sample=json.loads((ROOT/'reports/evaluation/swe_qa_v5_dev/dev_sample.json').read_text(encoding='utf-8')); qids=[qid for repo in ('flask','requests','pytest','pylint') for qid in [x for x in sample if x.startswith(f'{repo}-')][:2]]; freeze=v4.load_freeze(); rows={r['question_id']:r for r in v4.select_rows('main',v4.DATA/'dataset',freeze)}; selected=[rows[q] for q in qids]; llm=v2.llm_from_env(); services={}
    for repo in sorted({r['repo'] for r in selected}):
        service,project=v4.build_a1_service(repo,llm,v4.DATA/'../swe_qa_v4_avalai_gemini2'); services[repo]=(service,project)
    out=ROOT/'reports/evaluation/swe_qa_v5_dev/multquery_smoke'; out.mkdir(parents=True,exist_ok=True); path=out/'multquery.jsonl'
    if path.exists() and path.stat().st_size: raise SystemExit(f'refusing to overwrite {path}')
    with path.open('w',encoding='utf-8') as target:
        for row in selected:
            service,project=services[row['repo']]; original_search=service._search('semantic'); queries=subqueries(row['question']); merged={}
            for query in queries:
                result=original_search(RetrievalQuery(query,project,3))
                for rank,chunk in enumerate(result.results,1): merged.setdefault(chunk.chunk_id,replace(chunk,score=chunk.score-0.001*rank))
            original_retrieve=service._retrieve
            merged_chunks = tuple(sorted(merged.values(), key=lambda c: (-c.score, c.source_file, c.start_line, c.chunk_id)))
            def multi_retrieve(request):
                return RetrievalResult(query=RetrievalQuery(request.question, request.project_id, 5), results=merged_chunks)
            service._retrieve = multi_retrieve
            try: answer=service.answer(QARequest(question=row['question'],project_id=project,retrieval_method='semantic',retrieval_limit=5,max_context_chars=6000,temperature=0.0,max_tokens=1200))
            finally: service._retrieve=original_retrieve
            target.write(json.dumps({'method':'multquery','question_id':row['question_id'],'repo':row['repo'],'question':row['question'],'subqueries':queries,'reference_answer':row.get('answer'),'answer':answer.answer,'omitted_context_count':answer.omitted_context_count,'citations':[{'source_file':c.source_file,'qualified_name':c.qualified_name,'start_line':c.start_line,'end_line':c.end_line} for c in answer.citations]},ensure_ascii=False)+'\n'); target.flush(); print(row['question_id'],len(queries),answer.finish_reason or 'stop',flush=True)
if __name__=='__main__': main()
