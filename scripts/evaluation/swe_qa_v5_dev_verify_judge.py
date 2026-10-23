"""Judge semantic drafts versus verified two-pass answers."""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import swe_qa_v4_judge as official  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.providers import ProviderConfig, create_llm_provider  # noqa: E402
def main():
    official.load_dotenv(); folder=ROOT/'reports/evaluation/swe_qa_v5_dev'; out=folder/'verify_smoke/judge.jsonl'
    if out.exists() and out.stat().st_size: raise SystemExit('refusing to overwrite')
    data={'semantic':{json.loads(x)['question_id']:json.loads(x) for x in (folder/'semantic_smoke/semantic.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()},'verified':{json.loads(x)['question_id']:json.loads(x) for x in (folder/'verify_smoke/verified.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()}}
    provider=create_llm_provider(ProviderConfig(provider='openai_compatible',base_url=os.environ['CODECOMPASS_COMPARE_BASE_URL'],api_key=os.environ['CODECOMPASS_COMPARE_API_KEY'],llm_model='gemini-2.5-flash',timeout_seconds=180))
    with out.open('w',encoding='utf-8') as target:
        for qid in sorted(data['semantic']):
            for method in ('semantic','verified'):
                row=data[method][qid]; prompt=official.official_prompt(row['question'],row.get('reference_answer',''),re.sub(r'[\w./-]+\.py:\d+(?:-\d+)?','',row['answer'])); result={'question_id':qid,'method':method}
                try:
                    response=provider.generate(LLMRequest(prompt=prompt,temperature=0.0,max_tokens=2000,response_format='json')); scores=official.parse_scores(response.text); result.update(status='usable',scores=scores,mean=sum(scores.values())/5)
                except Exception as error: result.update(status='unavailable',error=str(error)[:300])
                target.write(json.dumps(result)+'\n'); target.flush(); print(method,qid,result['status'],flush=True)
if __name__=='__main__': main()
