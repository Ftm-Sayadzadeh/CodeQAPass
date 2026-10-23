"""Small semantic smoke test for query-relevant method selection."""
from __future__ import annotations
import ast, json, os, re, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import run_swe_qa_v2 as v2  # noqa: E402
import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.qa.models import QARequest  # noqa: E402
from codecompass.rag import RAGContext  # noqa: E402
from codecompass.rag.models import ContextBlock, ContextCitation  # noqa: E402

def load_dotenv():
    for raw in (ROOT/'.env').read_text(encoding='utf-8').splitlines():
        line=raw.strip()
        if line and not line.startswith('#') and '=' in line:
            name,value=line.split('=',1); os.environ.setdefault(name.strip(),value.strip().strip('"').strip("'"))

class MethodSelectiveBuilder:
    def __init__(self): self.question=''
    def build(self, retrieval_result, max_chars):
        terms={x.lower() for x in re.findall(r'[A-Za-z_][A-Za-z0-9_]+', self.question) if len(x)>2}; blocks=[]; used=omitted=0
        for chunk in sorted(retrieval_result.results, key=lambda c:(-c.score,c.source_file,c.start_line,c.chunk_id)):
            pieces=[chunk]
            if len(chunk.code)>max_chars:
                try:
                    tree=ast.parse(chunk.code); lines=chunk.code.splitlines(keepends=True)
                    methods=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                    methods.sort(key=lambda n:(-sum(t in n.name.lower() for t in terms),n.lineno))
                    if methods:
                        first=methods[0].lineno-1; pieces=[]
                        if first>0: pieces.append((chunk.code[:sum(len(x) for x in lines[:first])],chunk.start_line,chunk.start_line+first-1,chunk.symbol_name,chunk.qualified_name))
                        for node in methods[:3]:
                            code=''.join(lines[node.lineno-1:node.end_lineno]); pieces.append((code,chunk.start_line+node.lineno-1,chunk.start_line+node.end_lineno-1,node.name,f'{chunk.qualified_name}.{node.name}' if chunk.qualified_name else node.name))
                except SyntaxError: pass
            for piece in pieces:
                if isinstance(piece,tuple): code,start,end,symbol,qualified=piece
                else: code,start,end,symbol,qualified=piece.code,piece.start_line,piece.end_line,piece.symbol_name,piece.qualified_name
                if used+len(code)>max_chars: omitted+=1; continue
                blocks.append(ContextBlock(chunk_id=chunk.chunk_id, citation=ContextCitation(source_file=chunk.source_file,symbol_name=symbol,qualified_name=qualified,start_line=start,end_line=end),code=code,score=chunk.score,retrieval_method=chunk.retrieval_method)); used+=len(code)
        return RAGContext(blocks=tuple(blocks), total_chars=used, omitted_count=omitted)

def main():
    load_dotenv(); os.environ.setdefault('CODECOMPASS_LLM_REASONING_EFFORT','low')
    sample=json.loads((ROOT/'reports/evaluation/swe_qa_v5_dev/dev_sample.json').read_text(encoding='utf-8')); qids=[qid for repo in ('flask','requests','pytest','pylint') for qid in [x for x in sample if x.startswith(f'{repo}-')][:2]]
    freeze=v4.load_freeze(); rows={r['question_id']:r for r in v4.select_rows('main',v4.DATA/'dataset',freeze)}; selected=[rows[q] for q in qids]; llm=v2.llm_from_env(); services={}; builders={}
    for repo in sorted({r['repo'] for r in selected}):
        service,project=v4.build_a1_service(repo,llm,v4.DATA/'../swe_qa_v4_avalai_gemini2'); builder=MethodSelectiveBuilder(); service.context_builder=builder; services[repo]=(service,project); builders[repo]=builder
    out=ROOT/'reports/evaluation/swe_qa_v5_dev/method_select_smoke'; out.mkdir(parents=True,exist_ok=True); path=out/'method_select.jsonl'
    if path.exists() and path.stat().st_size: raise SystemExit(f'refusing to overwrite {path}')
    with path.open('w',encoding='utf-8') as target:
        for row in selected:
            service,project=services[row['repo']]; builders[row['repo']].question=row['question']; answer=service.answer(QARequest(question=row['question'],project_id=project,retrieval_method='semantic',retrieval_limit=5,max_context_chars=6000,temperature=0.0,max_tokens=1200)); target.write(json.dumps({'method':'method_select','question_id':row['question_id'],'repo':row['repo'],'question':row['question'],'reference_answer':row.get('answer'),'answer':answer.answer,'omitted_context_count':answer.omitted_context_count,'citations':[{'source_file':c.source_file,'qualified_name':c.qualified_name,'start_line':c.start_line,'end_line':c.end_line} for c in answer.citations],'model':answer.llm_model,'finish_reason':answer.finish_reason},ensure_ascii=False)+'\n'); target.flush(); print(row['question_id'],answer.finish_reason or 'no_evidence',flush=True)

if __name__=='__main__': main()
