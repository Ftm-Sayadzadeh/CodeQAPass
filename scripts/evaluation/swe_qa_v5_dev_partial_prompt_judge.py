"""Judge semantic versus partial-answer prompt smoke outputs."""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import swe_qa_v4_judge as official  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.providers import ProviderConfig, create_llm_provider  # noqa: E402

def main() -> int:
    official.load_dotenv(); folder=ROOT/'reports/evaluation/swe_qa_v5_dev'; out=folder/'partial_prompt/judge.jsonl'
    if out.exists() and out.stat().st_size: raise SystemExit('refusing to overwrite dev judge records')
    partial={json.loads(x)['question_id']:json.loads(x) for x in (folder/'partial_prompt/partial.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()}
    semantic={json.loads(x)['question_id']:json.loads(x) for x in (folder/'semantic_smoke/semantic.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()}
    provider=create_llm_provider(ProviderConfig(provider='openai_compatible',base_url=os.environ['CODECOMPASS_COMPARE_BASE_URL'],api_key=os.environ['CODECOMPASS_COMPARE_API_KEY'],llm_model='gemini-2.5-flash',timeout_seconds=180))
    with out.open('w',encoding='utf-8') as target:
        for qid in sorted(semantic):
            for method,data in (('semantic',semantic[qid]),('partial_prompt',partial[qid])):
                prompt=official.official_prompt(data['question'],data.get('reference_answer',''),re.sub(r'[\w./-]+\.py:\d+(?:-\d+)?','',data['answer']))
                result={'question_id':qid,'method':method,'model':'gemini-2.5-flash'}
                try:
                    response=provider.generate(LLMRequest(prompt=prompt,temperature=0.0,max_tokens=2000,response_format='json')); scores=official.parse_scores(response.text); result.update(status='usable',scores=scores,mean=sum(scores.values())/5,finish_reason=response.finish_reason)
                except Exception as error: result.update(status='unavailable',error_type=type(error).__name__,error=str(error)[:300])
                target.write(json.dumps(result,ensure_ascii=False)+'\n'); target.flush(); print(method,qid,result['status'],flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
