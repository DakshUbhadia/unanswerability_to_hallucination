#!/usr/bin/env python3
"""Verify fresh-sample exclusions, grading records, and relation-exact matching."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from src.config import CFG
from src.eval_common import read_jsonl
from src.pilot_check import compute_log10_s_pop

def verify(fresh_path=CFG.data_dir/'popqa_fresh.jsonl',archive_path=CFG.data_dir/'popqa_sample.jsonl'):
 fresh=read_jsonl(fresh_path);archive=read_jsonl(archive_path);assert len(fresh)<=CFG.n_popqa_fresh_cap
 old={int(r['hf_row_idx']) for r in archive};ids=[int(r['hf_row_idx']) for r in fresh]
 assert len(ids)==len(set(ids)) and not (old&set(ids)), 'fresh data overlaps archived hf_row_idx'
 assert all(r.get('input_ids_sha256') and r.get('label_source')=='canonical_v1_batch1_judged' for r in fresh)
 pairs={}
 for r in fresh:
  pid=r.get('matched_pair_id_v5')
  if pid:pairs.setdefault(pid,[]).append(r)
 for pid,group in pairs.items():
  assert len(group)==2,(pid,len(group));assert {x['grade'] for x in group}=={'correct','wrong'}
  assert len({x.get('relation',x.get('prop')) for x in group})==1
  d=abs(compute_log10_s_pop(group[0]['s_pop'])-compute_log10_s_pop(group[1]['s_pop']))
  assert d<=.25*float(json.loads(Path(CFG.results_dir/'popqa_fresh_sampling.json').read_text())['global_sd'])+1e-8
 return {'fresh_rows':len(fresh),'archived_overlap':0,'matched_pairs':len(pairs),'all_hashes_and_judgements_present':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fresh',type=Path,default=CFG.data_dir/'popqa_fresh.jsonl');p.add_argument('--archive',type=Path,default=CFG.data_dir/'popqa_sample.jsonl');a=p.parse_args();print(json.dumps(verify(a.fresh,a.archive),indent=2))
