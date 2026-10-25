#!/usr/bin/env python3
"""Export unique low-margin judged responses for human review."""
import argparse, csv, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.judge_pass import collect
from src.abstention_classifier import classify_texts
from src.config import CFG
LABELS={"ANSWER","ABSTAIN","DEGENERATE"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',type=Path)
    p.add_argument('--threshold',type=float,default=1.0)
    p.add_argument('--output',type=Path,default=CFG.results_dir/'judge_low_margin_review.csv')
    a=p.parse_args()
    paths=a.input or [Path(x) for x in CFG.judge_inputs if Path(x).exists()]
    if not paths: raise FileNotFoundError('No judge inputs found')
    items,_,_=collect(paths)
    records=classify_texts(items,offline=True)
    rows=[]
    for item,rec in zip(items,records):
        margin=rec.get('judge_margin')
        if margin is None or float(margin)>=a.threshold: continue
        rows.append({'cache_key':rec['cache_key'],'question':item['question'],'response':item['response'],
                     'judge_label':rec['judge_label'],'judge_margin':margin,'conditions':';'.join(item['conditions']),
                     'human_label':'','notes':''})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cache_key','question','response','judge_label','judge_margin','conditions','human_label','notes'])
        w.writeheader();w.writerows(rows)
    print(f'Exported {len(rows)} unique items with margin < {a.threshold:g} nat to {a.output}')
if __name__=='__main__':main()
