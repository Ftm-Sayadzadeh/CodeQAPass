"""Run paired semantic and manually decomposed multi-query evaluation."""
from __future__ import annotations
import json, os, sys
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
def main():
    load_dotenv(); os.environ.setdefault('CODECOMPASS_LLM_REASONING_EFFORT','low'); manifest=json.loads((ROOT/'reports/evaluation/swe_qa_v5_dev/controlled_multquery/manifest.json').read_text(encoding='utf-8')); freeze=v4.load_freeze(); rows={r['question_id']:r for r in v4.select_rows('main',v4.DATA/'dataset',freeze)}; selected=[rows[x['question_id']]|{'subqueries':x['subqueries']} for x in manifest['questions']]; llm=v2.llm_from_env(); services={}
    for repo in sorted({r['repo'] for r in selected}): services[repo]=v4.build_a1_service(repo,llm,v4.DATA/'../swe_qa_v4_avalai_gemini2')
    out=ROOT/'reports/evaluation/swe_qa_v5_dev/controlled_multquery'; paths={m:out/f'{m}.jsonl' for m in ('semantic','multquery')}
    if any(p.exists() and p.stat().st_size for p in paths.values()): raise SystemExit('refusing to overwrite controlled outputs')
    handles={m:p.open('w',encoding='utf-8') for m,p in paths.items()}
    try:
      for row in selected:
        service,project=services[row['repo']]
        for method in ('semantic','multquery'):
          original_retrieve=service._retrieve
          if method=='multquery':
            merged={}; search=service._search('semantic')
            for query in row['subqueries']:
              result=search(RetrievalQuery(query,project,3))
              for rank,chunk in enumerate(result.results,1): merged.setdefault(chunk.chunk_id,replace(chunk,score=chunk.score-0.001*rank))
            merged_chunks=tuple(sorted(merged.values(),key=lambda c:(-c.score,c.source_file,c.start_line,c.chunk_id)))
            service._retrieve=lambda request, chunks=merged_chunks: RetrievalResult(query=RetrievalQuery(request.question,request.project_id,5),results=chunks)
          try: answer=service.answer(QARequest(question=row['question'],project_id=project,retrieval_method='semantic',retrieval_limit=5,max_context_chars=6000,temperature=0.0,max_tokens=1200))
          finally: service._retrieve=original_retrieve
          handles[method].write(json.dumps({'method':method,'question_id':row['question_id'],'repo':row['repo'],'question':row['question'],'reference_answer':row.get('answer'),'subqueries':row['subqueries'],'answer':answer.answer,'omitted_context_count':answer.omitted_context_count,'citations':[{'source_file':c.source_file,'qualified_name':c.qualified_name,'start_line':c.start_line,'end_line':c.end_line} for c in answer.citations]},ensure_ascii=False)+'\n'); handles[method].flush(); print(method,row['question_id'],flush=True)
    finally:
      for h in handles.values(): h.close()
if __name__=='__main__': main()
