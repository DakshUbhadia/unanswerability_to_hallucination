#!/usr/bin/env python3
"""Extract and analyze preregistered v5 PopQA H4'/H5' features."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import numpy as np,torch,pandas as pd
from sklearn.metrics import roc_auc_score
from src.config import CFG
from src.eval_common import (build_chat_batches,capture_residuals_and_logits,load_runtime_contract,load_stage_model,
    read_json,read_jsonl,sha256_file)
from src.tokenization_contract import canonical_input_ids,chat_template_sha256,input_ids_sha256
from src.stage8_v5 import analyze_h4_v5,analyze_h5_v5

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--dtype',default='auto');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--n-perm',type=int,default=CFG.n_perm_stratified);p.add_argument('--n-boot',type=int,default=1000)
 for n,d in [('stage0',CFG.stage0_out_path),('stage2',CFG.stage2_out_path),('stage4',CFG.stage4_out_path),('stage5',CFG.stage5_out_path),('fresh',CFG.data_dir/'popqa_fresh.jsonl'),('nulls',CFG.results_dir/'stage4b_nulls.pt'),('features',CFG.results_dir/'stage8v5_acts_lstar.pt'),('output',CFG.results_dir/'stage8v5.json')]:p.add_argument('--'+n,type=Path,default=d)
 a=p.parse_args()
 from scripts.tier2_preflight import verify_preflight
 verify_preflight()
 sampling=read_json(CFG.results_dir/'popqa_fresh_sampling.json');transfer=read_json(CFG.results_dir/'fresh_grader_transfer.json')
 if not transfer.get('passed'):raise RuntimeError('fresh PopQA grader transfer gate has not passed')
 if not sampling.get('n_matched_pairs',0):raise RuntimeError('fresh PopQA matching produced no matched pairs')
 rows=read_jsonl(a.fresh);matched=[r for r in rows if r.get('is_matched_v5')];abstained=[r for r in rows if r.get('grade')=='abstain'];eligible=matched+[r for r in abstained if r not in matched];n_matched=len(matched)
 if len({r['hf_row_idx'] for r in matched})!=len(matched):raise ValueError('duplicate hf_row_idx in matched fresh data')
 contract=load_runtime_contract(a.stage0,a.stage2);selection=read_json(a.stage5);model,tok,_=load_stage_model(contract,a.device,a.dtype);th=chat_template_sha256(tok);prompts=[r['question'] for r in eligible]
 for row in eligible:
  ids=canonical_input_ids(tok,row['question'],contract['system_prompt_used'],contract['suffix_token_ids'])
  got=input_ids_sha256(ids,contract['model_name'],th)
  if got!=row['input_ids_sha256']:raise RuntimeError(f"input hash mismatch for {row['hf_row_idx']}")
 batches=build_chat_batches(tok,prompts,contract['system_prompt_used'],contract['suffix_token_ids'],a.batch_size)
 l=int(selection['l_star']);pos=int(selection['pos_idx']);acts,logits=capture_residuals_and_logits(model,batches,contract,list(range(int(contract['n_layers']))),pos)
 acts=acts.numpy().astype(np.float32);logits=logits.numpy().astype(np.float32);rdata=torch.load(a.stage4,map_location='cpu',weights_only=False);rhat=rdata['unit_directions'][l,pos].float().numpy()
 probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True);unknown=np.array(contract['unknown_token_ids'],int);pabs=np.clip(probs[:,unknown].sum(1),1e-7,1-1e-7)
 nulls=torch.load(a.nulls,map_location='cpu',weights_only=False);vecs={'covariance':nulls['covariance']['unit'].numpy(),'template_partition':nulls['template_partition']['unit'].numpy(),'label_free':nulls['label_free']['unit'].numpy()}
 rec={'acts':torch.from_numpy(acts),'rows':[{'hf_row_idx':r['hf_row_idx'],'input_ids_sha256':r['input_ids_sha256'],'is_matched_v5':bool(r.get('is_matched_v5')),'grade':r['grade']} for r in eligible],'logits':torch.from_numpy(logits)};torch.save(rec,a.features)
 feat={'proj':acts[:,l,:]@rhat,'maxp':probs.max(1),'entropy':-(probs*np.log(np.clip(probs,1e-30,1))).sum(1),'log_odds':np.log(pabs/(1-pabs)),'p_abs':pabs}
 y=np.array([int(r['grade']=='wrong') for r in matched]);strata=np.array([str(r.get('relation',r.get('prop',''))) for r in matched]);pairs=np.array([r['matched_pair_id_v5'] for r in matched]);
 nullscores={k:v@acts[:n_matched,l,:].T for k,v in vecs.items()}
 h4=analyze_h4_v5(feat['proj'][:n_matched],y,strata,pairs,nullscores['covariance'],nullscores['template_partition'],nullscores['label_free'],a.n_perm,CFG.seed_popqa_fresh,a.n_boot)
 frame=pd.DataFrame({k:v[:n_matched] for k,v in feat.items()});frame['y']=y;frame['matched_pair_id']=pairs;frame['relation']=strata;h5=analyze_h5_v5(frame,a.n_boot,CFG.seed_incremental_cv)
 correct_ix=np.array([i for i,r in enumerate(matched) if r['grade']=='correct']);abstain_ix=np.arange(n_matched,len(eligible))
 abstain_vs_correct=roc_auc_score(np.r_[np.ones(len(abstain_ix)),np.zeros(len(correct_ix))],np.r_[feat['proj'][abstain_ix],feat['proj'][correct_ix]]) if len(abstain_ix) and len(correct_ix) else None
 rel_scores=[]
 for i,row in enumerate(matched):
  peers=[j for j,x in enumerate(matched) if j!=i and str(x.get('relation',x.get('prop','')))==str(row.get('relation',row.get('prop','')))]
  rel_scores.append(float(np.mean([y[j] for j in peers])) if peers else .5)
 relation_only_auc=float(roc_auc_score(y,rel_scores))
 out={'stage':'8v5','n_matched_items':n_matched,'n_abstained_items':len(eligible)-n_matched,'n_matched_pairs':len(np.unique(pairs)),'feature_sha256':sha256_file(a.features),
  'h4':h4,'h5':h5,'per_relation':h4['stratified']['strata'],'relation_only_leave_one_out_auroc':relation_only_auc,
  'abstain_vs_correct_projection_auroc':abstain_vs_correct,'all_layer_proj':(acts@rhat).mean(0).tolist(),
  'input_hashes_verified':True,'fresh_sampling':sampling,'grader_transfer':transfer}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(f"Wrote {a.output}; H4' AUROC={h4['stratified']['auroc']:.3f}, H5' delta={h5['delta_auroc_cv']:.3f}")
if __name__=='__main__':main()
