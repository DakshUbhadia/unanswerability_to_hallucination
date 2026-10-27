#!/usr/bin/env python3
"""Export/verify the blinded 100-item fresh-PopQA grader transfer audit."""
import argparse,csv,json,random
from pathlib import Path
from src.config import CFG
from src.eval_common import read_jsonl
from src.popqa_fresh import verify_grader_transfer

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['export','verify']);p.add_argument('--input',type=Path,default=CFG.data_dir/'popqa_fresh.jsonl');p.add_argument('--sheet',type=Path,default=CFG.data_dir/'fresh_grader_transfer.csv');p.add_argument('--key',type=Path,default=CFG.results_dir/'fresh_grader_transfer_key.json');p.add_argument('--human-labels',type=Path,default=CFG.data_dir/'fresh_grader_transfer.csv');a=p.parse_args();rows=read_jsonl(a.input)
 if a.mode=='export':
  rng=random.Random(CFG.grader_validation_seed);chosen=sorted(rng.sample(rows,min(100,len(rows))),key=lambda r:r['hf_row_idx'])
  if len(chosen)!=100:raise RuntimeError(f'Need 100 fresh rows; found {len(chosen)}')
  a.sheet.parent.mkdir(parents=True,exist_ok=True)
  with a.sheet.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=['uid','question','response','human_grade','notes']);w.writeheader()
   for r in chosen:w.writerow({'uid':r['popqa_id'],'question':r['question'],'response':r['response'],'human_grade':'','notes':''})
  a.key.write_text(json.dumps({r['popqa_id']:{'grade':r['grade'],'hf_row_idx':r['hf_row_idx']} for r in chosen},indent=2)+'\n',encoding='utf-8')
  print(f'Exported blinded 100-item sheet: {a.sheet}; keep key private until labels are complete')
 else:
  key=json.loads(a.key.read_text(encoding='utf-8'));auto=[];human=[]
  with a.human_labels.open(newline='',encoding='utf-8-sig') as f:
   for r in csv.DictReader(f):
    if r['uid'] not in key:raise ValueError(f"unexpected UID {r['uid']}")
    if not r['human_grade'].strip():raise ValueError(f"missing human label for {r['uid']}")
    auto.append({'grade':key[r['uid']]['grade']});human.append(r['human_grade'])
  result=verify_grader_transfer(human,auto);out=CFG.results_dir/'fresh_grader_transfer.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
  if not result['passed']:raise SystemExit('Grader transfer gate failed; do not run Stage 8 v5')
if __name__=='__main__':main()
