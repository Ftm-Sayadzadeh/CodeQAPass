"""Paired judge for the controlled 24-question outputs."""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
import swe_qa_v4_judge as official  # noqa: E402
from codecompass.llm import LLMRequest  # noqa: E402
from codecompass.providers import ProviderConfig, create_llm_provider  # noqa: E402
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--rep',type=int,default=1); args=parser.parse_args(); official.load_dotenv(); folder=ROOT/'reports/evaluation/swe_qa_v5_dev/controlled_multquery'; out=folder/f'judge_{args.rep}.jsonl'
    if out.exists() and out.stat().st_size: raise SystemExit('refusing to overwrite')
    data={m:{json.loads(x)['question_id']:json.loads(x) for x in (folder/f'{m}.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()} for m in ('semantic','multquery')}; provider=create_llm_provider(ProviderConfig(provider='openai_compatible',base_url=os.environ['CODECOMPASS_COMPARE_BASE_URL'],api_key=os.environ['CODECOMPASS_COMPARE_API_KEY'],llm_model='gemini-2.5-flash',timeout_seconds=180))
    with out.open('w',encoding='utf-8') as target:
        for qid in sorted(data['semantic']):
            for method in ('semantic','multquery'):
                row=data[method][qid]; prompt=official.official_prompt(row['question'],row.get('reference_answer',''),re.sub(r'[\w./-]+\.py:\d+(?:-\d+)?','',row['answer'])); result={'question_id':qid,'method':method,'rep':args.rep}
                try:
                    response=provider.generate(LLMRequest(prompt=prompt,temperature=0.0,max_tokens=2000,response_format='json')); scores=official.parse_scores(response.text); result.update(status='usable',scores=scores,mean=sum(scores.values())/5)
                except Exception as error: result.update(status='unavailable',error=str(error)[:300])
                target.write(json.dumps(result)+'\n'); target.flush(); print(method,qid,result['status'],flush=True)
if __name__=='__main__': main()
