#!/usr/bin/env python3
"""Generate and grade preregistered fresh PopQA batches for Tier 2 (Kaggle/GPU)."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import torch
from src.config import CFG
from src.dataset_builder import load_popqa_rows
from src.popqa_fresh import sample_fresh_batches,match_relation_exact
from src.eval_common import load_runtime_contract,load_stage_model,read_json,read_jsonl
from src.pilot_check import grade_popqa
from src.abstention_classifier import classify_texts,require_judge_validation
from src.tokenization_contract import CONTRACT_VERSION,canonical_input_ids,chat_template_sha256,input_ids_sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--dtype',default='auto');p.add_argument('--offline',action='store_true');p.add_argument('--force',action='store_true')
 for n,d in [('stage0',CFG.stage0_out_path),('stage2',CFG.stage2_out_path),('archive',CFG.data_dir/'popqa_sample.jsonl'),('output',CFG.data_dir/'popqa_fresh.jsonl'),('checkpoint',CFG.results_dir/'popqa_fresh_checkpoint.jsonl')]:p.add_argument('--'+n,type=Path,default=d)
 a=p.parse_args()
 from scripts.tier2_preflight import verify_preflight
 verify_preflight();require_judge_validation();contract=load_runtime_contract(a.stage0,a.stage2);model,tok,_=load_stage_model(contract,a.device,a.dtype);template_hash=chat_template_sha256(tok)
 old=read_jsonl(a.archive);excluded={int(r['hf_row_idx']) for r in old};rows=read_jsonl(a.checkpoint) if a.checkpoint.exists() and not a.force else []
 excluded.update(int(r['hf_row_idx']) for r in rows);all_rows=load_popqa_rows();batch_idx=len(rows)//CFG.n_popqa_fresh_batch
 while len(rows)<CFG.n_popqa_fresh_cap:
  batch=sample_fresh_batches(all_rows,excluded,batch_idx,CFG.n_popqa_fresh_batch,CFG.seed_popqa_fresh)
  if not batch:break
  for row in batch:
   ids=canonical_input_ids(tok,row['question'],contract['system_prompt_used'],contract['suffix_token_ids']);digest=input_ids_sha256(ids,contract['model_name'],template_hash)
   dev=next(model.parameters()).device;enc={'input_ids':torch.tensor([ids],device=dev),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long,device=dev),'position_ids':torch.arange(len(ids),dtype=torch.long,device=dev).unsqueeze(0)}
   with torch.no_grad():out=model.generate(**enc,max_new_tokens=CFG.max_new_tokens,do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   response=tok.decode(out[0,len(ids):],skip_special_tokens=True)
   judged=classify_texts([{'uid':f"popqa_fresh:{row['hf_row_idx']}",'question':row['question'],'response':response}],offline=a.offline)[0]
   grade=grade_popqa(response,row.get('possible_answers',[]),row.get('obj',''),judged)
   row.update(popqa_id=f"fresh_{int(row['hf_row_idx']):06d}",relation=str(row.get('prop','')),response=response,baseline_response=response,
    input_ids_sha256=digest,chat_template_sha256=template_hash,tokenization_contract_version=CONTRACT_VERSION,_generation_batch_size=1,
    judged=judged,grade=grade['grade'],grade_reason=grade['reason'],grade_record=grade,label_source='canonical_v1_batch1_judged')
   rows.append(row);excluded.add(int(row['hf_row_idx']))
  a.checkpoint.parent.mkdir(parents=True,exist_ok=True);a.checkpoint.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
  match=match_relation_exact(rows)
  print(f"batch={batch_idx} fresh={len(rows)} relation-exact pairs={match['n_pairs']}",flush=True)
  batch_idx+=1
  if match['n_pairs']>=CFG.n_popqa_matched_target:break
 match=match_relation_exact(rows);n=match['n_pairs']
 if len(rows)>=CFG.n_popqa_fresh_cap and n<300:raise RuntimeError(f"Stop condition: cap reached with only {n} pairs; report to owner")
 for row in rows:
  row['matched_pair_id_v5']=row.pop('matched_pair_id_v5',None);row['is_matched_v5']=bool(row.get('is_matched_v5'))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
 meta={'seed':CFG.seed_popqa_fresh,'batch_size':CFG.n_popqa_fresh_batch,'n_fresh':len(rows),'n_matched_pairs':n,'matched_target':CFG.n_popqa_matched_target,'cap':CFG.n_popqa_fresh_cap,'caliper':match['caliper'],'global_sd':match['global_sd'],'excluded_relations':match['excluded_relations'],'passed_target':n>=CFG.n_popqa_matched_target,'output':str(a.output)}
 (CFG.results_dir/'popqa_fresh_sampling.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
