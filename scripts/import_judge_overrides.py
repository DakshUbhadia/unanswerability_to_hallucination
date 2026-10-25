#!/usr/bin/env python3
"""Import reviewed low-margin labels as auditable cache overrides."""
import argparse,csv,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.abstention_classifier import LABELS,_load_cache
from src.config import CFG

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=CFG.results_dir/'judge_low_margin_review.csv')
    p.add_argument('--cache',type=Path,default=CFG.judge_cache_path)
    a=p.parse_args()
    cache=_load_cache(a.cache); applied=0; seen=set()
    with a.csv.open(encoding='utf-8-sig',newline='') as f:
        for line,row in enumerate(csv.DictReader(f),2):
            label=(row.get('human_label') or '').strip().upper()
            if not label: continue
            if label not in LABELS: raise ValueError(f'{a.csv}:{line}: invalid human_label {label!r}')
            key=row.get('cache_key','')
            if not key or key not in cache: raise KeyError(f'{a.csv}:{line}: cache_key is absent from {a.cache}')
            if key in seen: raise ValueError(f'{a.csv}:{line}: duplicate cache_key {key}')
            seen.add(key); record=cache[key]
            margin=record.get('judge_margin')
            if margin is None or float(margin)>=1.0: raise ValueError(f'{a.csv}:{line}: only items with judge_margin < 1 nat may be overridden')
            if record.get('human_override') and record['human_override']!=label:
                raise ValueError(f'{a.csv}:{line}: conflicts with existing human override')
            record.update(human_override=label,final_label_source='human',human_override_note=row.get('notes',''),
                          human_override_source=str(a.csv))
            applied+=1
    a.cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.cache.with_suffix(a.cache.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        for key,record in cache.items(): f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    tmp.replace(a.cache)
    summary_path=CFG.judge_pass_summary_path
    if summary_path.exists() and a.cache.resolve()==Path(CFG.judge_cache_path).resolve():
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        import hashlib
        summary["judge_cache_sha256"]=hashlib.sha256(a.cache.read_bytes()).hexdigest()
        summary["human_override_count"]=sum(bool(r.get("human_override")) for r in cache.values())
        summary_path.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(f'Imported {applied} human overrides into {a.cache}')
if __name__=='__main__':main()
