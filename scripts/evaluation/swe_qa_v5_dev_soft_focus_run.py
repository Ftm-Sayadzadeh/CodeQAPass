"""Small semantic smoke test for soft focus with one rescue chunk per file."""
from __future__ import annotations
import json, os, sys
from dataclasses import replace
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.qa.models import QARequest  # noqa: E402
from codecompass.retrieval import RetrievalQuery, RetrievalResult  # noqa: E402
from codecompass.retrieval.models import RetrievedChunk  # noqa: E402
from swe_qa_v5_dev_context import Index, load_rankings  # noqa: E402
def load_dotenv():
    for raw in (ROOT/'.env').read_text(encoding='utf-8').splitlines():
        line=raw.strip()
        if line and not line.startswith('#') and '=' in line:
            name,value=line.split('=',1); os.environ.setdefault(name.strip(),value.strip().strip('"').strip("'"))
def main():
    load_dotenv(); os.environ.setdefault('CODECOMPASS_LLM_REASONING_EFFORT','low'); sample=json.loads((ROOT/'reports/evaluation/swe_qa_v5_dev/dev_sample.json').read_text(encoding='utf-8')); qids=[qid for repo in ('flask','requests','pytest','pylint') for qid in [x for x in sample if x.startswith(f'{repo}-')][:2]]; freeze=v4.load_freeze(); rows={r['question_id']:r for r in v4.select_rows('main',v4.DATA/'dataset',freeze)}; selected=[rows[q] for q in qids]; llm=v2.llm_from_env(); services={}
    for repo in sorted({r['repo'] for r in selected}): services[repo]=v4.build_a1_service(repo,llm,v4.DATA/'../swe_qa_v4_avalai_gemini2')
    rankings=load_rankings(ROOT/'reports/evaluation/swe_qa_v4/frozen/rankings_semantic.jsonl'); indexes={repo:Index(repo) for repo in services}
    out=ROOT/'reports/evaluation/swe_qa_v5_dev/soft_focus_smoke'; out.mkdir(parents=True,exist_ok=True); path=out/'soft_focus.jsonl'
    if path.exists() and path.stat().st_size: raise SystemExit(f'refusing to overwrite {path}')
    with path.open('w',encoding='utf-8') as target:
        for row in selected:
            service,project=services[row['repo']]; ranking=rankings[row['question_id']]['ranking'][:5]; index=indexes[row['repo']]; chunks=[]
            for rank,item in enumerate(ranking,1):
                db=index.chunks[item['chunk_id']]; chunks.append(RetrievedChunk(chunk_id=db['chunk_id'],score=1.0/rank,source_file=db['relative_path'],symbol_name=db['symbol_name'],qualified_name=db['qualified_name'],start_line=db['start_line'],end_line=db['end_line'],code=db['code'],retrieval_method='semantic'))
            result=RetrievalResult(query=RetrievalQuery(row['question'],project,5),results=tuple(chunks)); original_result=service._focus_on_named_symbol(result); original=original_result.results; focused_ids={c.chunk_id for c in original}; extras=[]; files=set()
            for chunk in result.results:
                if chunk.chunk_id not in focused_ids and chunk.source_file not in files: extras.append(chunk); files.add(chunk.source_file)
                if len(extras)>=1: break
            merged=RetrievalResult(query=result.query,results=tuple(list(original)+extras)); old_retrieve,old_focus=service._retrieve,service._focus_on_named_symbol; service._retrieve=lambda request,merged=merged: merged; service._focus_on_named_symbol=lambda r:r
            try: answer=service.answer(QARequest(question=row['question'],project_id=project,retrieval_method='semantic',retrieval_limit=5,max_context_chars=6000,temperature=0.0,max_tokens=1200))
            finally: service._retrieve,service._focus_on_named_symbol=old_retrieve,old_focus
            target.write(json.dumps({'method':'soft_focus','question_id':row['question_id'],'repo':row['repo'],'question':row['question'],'reference_answer':row.get('answer'),'answer':answer.answer,'omitted_context_count':answer.omitted_context_count,'citations':[{'source_file':c.source_file,'qualified_name':c.qualified_name,'start_line':c.start_line,'end_line':c.end_line} for c in answer.citations]},ensure_ascii=False)+'\n'); target.flush(); print(row['question_id'],answer.finish_reason or 'stop',flush=True)
    for index in indexes.values(): index.close()
if __name__=='__main__': main()
