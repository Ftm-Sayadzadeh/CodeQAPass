"""Offline soft-focus diagnostics; never changes production or frozen v4."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import swe_qa_v5_dev_context as ctx

ROOT = Path(__file__).resolve().parents[2]

def soft_focus(question, original, rescue):
    focused = ctx.focus(question, original)
    focused_ids = {c.chunk_id for c in focused}
    extras = []
    files = set()
    for candidate in original:
        if candidate.chunk_id in focused_ids or candidate.source_file in files:
            continue
        extras.append(candidate)
        files.add(candidate.source_file)
        if len(extras) >= rescue:
            break
    return focused + extras

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--method', choices=('semantic','hybrid'), default='semantic'); parser.add_argument('--rescue', type=int, default=1); parser.add_argument('--write', action='store_true'); args = parser.parse_args()
    rows = ctx.load_rows(); rankings = ctx.load_rankings(ROOT / f'reports/evaluation/swe_qa_v4/frozen/rankings_{args.method}.jsonl'); sample = ctx.select_sample(rows); indexes = {repo: ctx.Index(repo) for repo in ctx.REPOS}; report=[]
    try:
        for row in sample:
            ranking=rankings[row['question_id']]['ranking']; original=[c for rank,item in enumerate(ranking,1) for c in indexes[row['repo']].expand(item,rank,row['question'],split_large=False)]; packed=ctx.pack(soft_focus(row['question'],original,args.rescue),'baseline'); gold=set(rankings[row['question_id']]['gold_files']); files={b.source_file for b in packed['blocks']}; report.append({'question_id':row['question_id'],'repo':row['repo'],'gold':len(gold&files),'full':gold<=files,'blocks':len(packed['blocks']),'chars':packed['total_chars'],'omitted':packed['omitted_count']})
    finally:
        for index in indexes.values(): index.close()
    summary={'method':args.method,'rescue':args.rescue,'sample_size':len(report),'gold_files':sum(r['gold'] for r in report),'full_questions':sum(r['full'] for r in report),'avg_blocks':statistics.mean(r['blocks'] for r in report),'avg_chars':statistics.mean(r['chars'] for r in report),'avg_omitted':statistics.mean(r['omitted'] for r in report),'rows':report}; print(json.dumps(summary,indent=2))
    if args.write:
        ctx.OUT.mkdir(parents=True,exist_ok=True); (ctx.OUT/f'soft_focus_{args.method}_r{args.rescue}.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__': main()
