#!/usr/bin/env python3
"""Stage 6b: evaluate Tier 2 null directions and judged learned generations."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.config import CFG
from src.eval_common import (ConditionCheckpoint,build_chat_batches,ensure_stage4_directions,generate_with_hooks,
    load_runtime_contract,load_stage_model,read_json,read_jsonl,evaluate_condition,sha256_file)
from src.abstention_classifier import classify_texts
from src.stats_utils import wilcoxon_greater,cluster_bootstrap

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--device',default='cuda:0');p.add_argument('--dtype',default='auto');p.add_argument('--batch-size',type=int,default=CFG.batch_size_forward);p.add_argument('--force',action='store_true');p.add_argument('--offline',action='store_true')
 for n,d in [('stage0',CFG.stage0_out_path),('stage2',CFG.stage2_out_path),('stage4',CFG.stage4_out_path),('stage4-manifest',CFG.stage4_manifest_path),('stage5',CFG.stage5_out_path),('pairs',CFG.data_dir/'pairs.jsonl'),('splits',CFG.data_dir/'splits.json'),('nulls',CFG.results_dir/'stage4b_nulls.pt'),('output',CFG.results_dir/'stage6b_null_eval.json'),('checkpoint',CFG.results_dir/'stage6b_checkpoint.pt'),('generations',CFG.results_dir/'stage6b_generations.jsonl'),('resid-norms',CFG.results_dir/'stage4b_resid_norms.json')]:p.add_argument('--'+n,type=Path,default=d)
 a=p.parse_args()
 from scripts.tier2_preflight import verify_preflight
 verify_preflight()
 contract=load_runtime_contract(a.stage0,a.stage2);directions,stage4_sha=ensure_stage4_directions(a.stage4,a.stage4_manifest,a.stage0);selection=read_json(a.stage5);l=int(selection['l_star']);pos=int(selection['pos_idx']);rnorm=float(directions['norms'][l,pos]);r=directions['unit_directions'][l,pos].float();r_steer=r*rnorm
 pairs=sorted([x for x in read_jsonl(a.pairs) if x['split']=='test'],key=lambda x:x['pair_id']);ans=[x['ans_prompt'] for x in pairs];unans=[x['unans_prompt'] for x in pairs]
 model,tok,_=load_stage_model(contract,a.device,a.dtype);ba=build_chat_batches(tok,ans,contract['system_prompt_used'],contract['suffix_token_ids'],a.batch_size);bu=build_chat_batches(tok,unans,contract['system_prompt_used'],contract['suffix_token_ids'],a.batch_size)
 from src.tokenization_contract import canonical_input_ids,chat_template_sha256,input_ids_sha256
 template_hash=chat_template_sha256(tok);input_hashes={}
 for pair in pairs:
  input_hashes[pair['pair_id']]={'ans':input_ids_sha256(canonical_input_ids(tok,pair['ans_prompt'],contract['system_prompt_used'],contract['suffix_token_ids']),contract['model_name'],template_hash),
    'unans':input_ids_sha256(canonical_input_ids(tok,pair['unans_prompt'],contract['system_prompt_used'],contract['suffix_token_ids']),contract['model_name'],template_hash)}
 lib=torch.load(a.nulls,map_location='cpu',weights_only=False);families=[]
 for fam,key in [('covariance','covariance'),('label_free','label_free'),('template_partition','template_partition')]:families.extend((fam,j,x) for j,x in enumerate(lib[key]['unit']))
 families.extend(('concept',j,x['unit']) for j,x in enumerate(lib['concepts']))
 families.extend(('pca',j,x) for j,x in enumerate(lib['pca']['unit']))
 families.extend(('variance_matched',j,x) for j,x in enumerate(lib['variance_matched']['directions']))
 fp_payload=stage4_sha+sha256_file(a.nulls)+json.dumps([(x['pair_id'],x['ans_prompt'],x['unans_prompt']) for x in pairs],ensure_ascii=False)
 fp=hashlib.sha256(fp_payload.encode()).hexdigest();cp=ConditionCheckpoint(a.checkpoint,fp,a.force,every=25)
 for family,j,vec in families:
  key=f'{family}:{j}'
  if key in cp:continue
  vec=torch.as_tensor(vec,dtype=torch.float32);steer=vec*rnorm
  row={}
  for name,batches,mode,alpha,unit in [('nec_ablate',bu,'ablate_all',0.,vec),('nec_steer',bu,'steer_layer',-1.,steer),('suf_steer',ba,'steer_layer',1.,steer)]:
   ev=evaluate_condition(model,batches,contract,mode,unit,l if mode=='steer_layer' else None,alpha)
   row[name]={'p_abs':ev['p_abs'].tolist(),'mean_p_abs':float(np.mean(ev['p_abs']))}
  row.update(family=family,index=j);cp.put(key,row)
 cp.flush()
 base_u=evaluate_condition(model,bu,contract);base_a=evaluate_condition(model,ba,contract)
 def generate_judged(key,batches,prompts,part,mode='none',vec=None,layer=None,alpha=0.):
  ck='generation:'+key
  if ck in cp:return cp.get(ck)['texts'],cp.get(ck)['judged']
  texts=generate_with_hooks(model,tok,batches,contract,mode,vec,layer,alpha)
  judged=classify_texts([{'uid':f'stage6b:{key}:{i}','question':q,'response':t} for i,(q,t) in enumerate(zip(prompts,texts))],offline=a.offline)
  cp.put(ck,{'texts':texts,'judged':judged});cp.flush();return texts,judged
 # Primary learned directions; generation labels use Tier 1 judge and exact input prompts.
 gen_rows=[]
 for part,batches,prompts in [('unans',bu,unans),('ans',ba,ans)]:
  baseline_texts,baseline_judged=generate_judged(f'{part}:baseline',batches,prompts,part)
  for i,(q,t,jd) in enumerate(zip(prompts,baseline_texts,baseline_judged)):
   gen_rows.append({'part':part,'condition':'baseline','pair_id':pairs[i]['pair_id'],'template_id':pairs[i]['template_id'],'prompt':q,'input_ids_sha256':input_hashes[pairs[i]['pair_id']]['ans' if part=='ans' else 'unans'],'response':t,'judged':jd})
  conds=[('ablate_all','ablate_all',r,None,0.)]
  conds += [('steer_neg1','steer_layer',r_steer,l,-1.)] if part=='unans' else [('steer_pos1','steer_layer',r_steer,l,1.)]
  for cname,mode,vec,layer,alpha in conds:
   texts,judged=generate_judged(f'{part}:{cname}',batches,prompts,part,mode,vec,layer,alpha)
   for i,(q,t,jd) in enumerate(zip(prompts,texts,judged)):gen_rows.append({'part':part,'condition':cname,'pair_id':pairs[i]['pair_id'],'template_id':pairs[i]['template_id'],'prompt':q,'input_ids_sha256':input_hashes[pairs[i]['pair_id']]['ans' if part=='ans' else 'unans'],'response':t,'judged':jd})
 # Generation subset: first 19 covariance nulls at registered necessity/sufficiency points.
 null_gen=[]
 for j,vec in enumerate(lib['covariance']['unit'][:CFG.n_null_gen_subset]):
  vec=torch.as_tensor(vec,dtype=torch.float32)
  for part,batches,prompts,mode,alpha,unit in [('unans',bu,unans,'ablate_all',0.,vec),('ans',ba,ans,'steer_layer',1.,vec.float()*rnorm)]:
   layer=l if mode=='steer_layer' else None;texts,judged=generate_judged(f'null:{j}:{part}',batches,prompts,part,mode,unit,layer,alpha)
  for i,(t,jd) in enumerate(zip(texts,judged)):null_gen.append({'null_index':j,'part':part,'condition':'ablate_all' if mode=='ablate_all' else 'steer_pos1','pair_id':pairs[i]['pair_id'],'input_ids_sha256':input_hashes[pairs[i]['pair_id']]['ans' if part=='ans' else 'unans'],'response':t,'judged':jd})
 # Dose response for learned direction and first 20 covariance nulls, probability scale.
 doses=[];null20=lib['covariance']['unit'][:CFG.n_null_dose_subset]
 for name,vec in [('learned',r)]+[(f'covariance_{i}',v) for i,v in enumerate(null20)]:
  steer=torch.as_tensor(vec,dtype=torch.float32)*rnorm
  for part,batches,base,alphas in [('unans',bu,base_u['p_abs'],CFG.grid_alpha_nec),('ans',ba,base_a['p_abs'],CFG.grid_alpha_suf)]:
   for alpha in alphas:
    dkey=f"dose:{name}:{part}:{alpha:+.2f}"
    if dkey in cp:row=cp.get(dkey)
    else:
     if alpha==0:mean=float(np.mean(base))
     else:
      ev=evaluate_condition(model,batches,contract,'steer_layer',steer,l,float(alpha));mean=float(np.mean(ev['p_abs']))
     row={'direction':name,'part':part,'alpha':float(alpha),'mean_p_abs':mean,'delta_p_abs':mean-float(np.mean(base))}
     cp.put(dkey,row)
    doses.append(row)
  cp.flush()
 norm_stats=read_json(a.resid_norms);scale=__import__('src.steering_scale',fromlist=['alpha_scale_table']).alpha_scale_table(
  sorted(set(CFG.grid_alpha_nec+CFG.grid_alpha_suf)),rnorm,norm_stats)
 dose_summary=[]
 for part,alphas in (('unans',CFG.grid_alpha_nec),('ans',CFG.grid_alpha_suf)):
  for alpha in alphas:
   vals=[x['delta_p_abs'] for x in doses if x['part']==part and x['alpha']==alpha and x['direction'].startswith('covariance_')]
   learned=next(x['delta_p_abs'] for x in doses if x['part']==part and x['alpha']==alpha and x['direction']=='learned')
   dose_summary.append({'part':part,'alpha':alpha,'learned_delta_p_abs':learned,'null_mean':float(np.mean(vals)) if vals else None,
    'null_p2_5':float(np.percentile(vals,2.5)) if vals else None,'null_p97_5':float(np.percentile(vals,97.5)) if vals else None,
    'net_effect':learned-float(np.mean(vals)) if vals else None})
 # H1'/H2': judged generation changes clustered by test template, with probability-scale IUTs.
 from collections import defaultdict
 gen_lookup={(r['part'],r['condition'],r['pair_id']):r['judged'].get('final_label')=='ABSTAIN' for r in gen_rows}
 def generated_cluster_effect(part,condition,orient):
  vals=defaultdict(list)
  for pair in pairs:
   b=gen_lookup[(part,'baseline',pair['pair_id'])];v=gen_lookup[(part,condition,pair['pair_id'])]
   vals[pair['template_id']].append((float(b)-float(v) if part=='unans' else float(v)-float(b))*orient)
  return np.array([np.mean(v) for v in vals.values()],float)
 generated={}
 for hid,part,condition,orient,nullkey in [('H1_prime','unans','ablate_all',1.,'nec_ablate'),('H2_prime','ans','steer_pos1',1.,'suf_steer')]:
  cl=generated_cluster_effect(part,condition,orient); learned=float(cl.mean());w=wilcoxon_greater(cl)
  component={'wilcoxon_cluster':float(w['p_value'])};null_summary={}
  for family in ('covariance','label_free','template_partition'):
   effects=[]
   for j in range(len(lib[family]['unit'])):
    row=cp.get(f'{family}:{j}');p0=np.asarray(base_u['p_abs'] if part=='unans' else base_a['p_abs']);p1=np.asarray(row[nullkey]['p_abs'])
    delta=(p0-p1) if part=='unans' else (p1-p0)
    by=defaultdict(list)
    for val,pair in zip(delta,pairs):by[pair['template_id']].append(float(val))
    effects.append(float(np.mean([np.mean(v) for v in by.values()])))
   pnull=(1+sum(x>=learned for x in effects))/(len(effects)+1);component[family+'_null']=pnull
   null_summary[family]={'p_value':pnull,'mean_effect':float(np.mean(effects)),'p95_effect':float(np.percentile(effects,95)),'effects':effects}
  concept_effects=[]
  for j in range(len(lib['concepts'])):
   row=cp.get(f'concept:{j}');p0=np.asarray(base_u['p_abs'] if part=='unans' else base_a['p_abs']);p1=np.asarray(row[nullkey]['p_abs']);delta=(p0-p1) if part=='unans' else (p1-p0)
   by=defaultdict(list)
   for val,pair in zip(delta,pairs):by[pair['template_id']].append(float(val))
   concept_effects.append({'name':lib['concepts'][j]['attribute']+':'+lib['concepts'][j]['level'],'effect':float(np.mean([np.mean(v) for v in by.values()]))})
  largest=max((x['effect'] for x in concept_effects),default=0.)
  confounded=learned<=0 or max(null_summary['template_partition']['p95_effect'],largest)>.25*learned
  generated[hid]={'learned_cluster_effect':learned,'wilcoxon':w,'components':component,'iut_p':max(component.values()),
    'binding_component':max(component,key=component.get),'nulls':null_summary,'named_concept_effects':concept_effects,
    'concept_confounded':bool(confounded),'cluster_effects':cl.tolist()}
 # Answer retention for learned all-layer ablation on answerable prompts.
 ans_ablate={r['pair_id']:r for r in gen_rows if r['part']=='ans' and r['condition']=='ablate_all'}
 from src.pilot_check import normalize_popqa_text
 retention=[]
 for pair in pairs:
  rec=ans_ablate[pair['pair_id']];label=rec['judged'].get('final_label');retention.append(bool(label=='ANSWER' and normalize_popqa_text(rec['response'])==normalize_popqa_text(pair['ground_truth'])))
 retention_ci=cluster_bootstrap(np.asarray(retention,dtype=float),[x['template_id'] for x in pairs],CFG.n_bootstrap,CFG.seed_bootstrap)
 null_generation_comparison={}
 for part,learned_condition,null_condition,orient in [('unans','ablate_all','ablate_all',-1.),('ans','steer_pos1','steer_pos1',1.)]:
  base_rate=float(np.mean([gen_lookup[(part,'baseline',pair['pair_id'])] for pair in pairs]))
  learned_rate=float(np.mean([gen_lookup[(part,learned_condition,pair['pair_id'])] for pair in pairs]))
  ldelta=(base_rate-learned_rate) if part=='unans' else (learned_rate-base_rate)
  vals=[]
  for j in range(CFG.n_null_gen_subset):
   subset=[x for x in null_gen if x['part']==part and x['null_index']==j]
   rate=float(np.mean([x['judged'].get('final_label')=='ABSTAIN' for x in subset]))
   vals.append((base_rate-rate) if part=='unans' else (rate-base_rate))
  null_generation_comparison[part]={'baseline_rate':base_rate,'learned_rate':learned_rate,'learned_delta':ldelta,
    'null_deltas':vals,'rank_out_of_20':1+sum(v>ldelta for v in vals),'p_rank':(1+sum(v>=ldelta for v in vals))/(len(vals)+1)}
 a.generations.parent.mkdir(parents=True,exist_ok=True);a.generations.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gen_rows+null_gen),encoding='utf-8')
 result={'stage':'6b','stage4_sha256':stage4_sha,'null_library_sha256':sha256_file(a.nulls),'n_directions':len(families),'direction_results':cp.results,'generated_primary':generated,'null_generation_comparison':null_generation_comparison,
  'baseline_p_abs':{'unans':base_u['p_abs'].tolist(),'ans':base_a['p_abs'].tolist()},'dose_curves':doses,'dose_summary':dose_summary,'steering_scale':scale,
  'answer_retention':{'fraction_answer_and_exact_match':float(np.mean(retention)),'n_test':len(retention),'correct_answers':sum(retention),'template_cluster_ci95':retention_ci},
  'learned_generations':str(a.generations),'generation_rows':len(gen_rows),'null_generation_rows':len(null_gen),'generation_judge':'Tier 1 classify_texts','input_ids_sha256':input_hashes,'test_pair_ids':[x['pair_id'] for x in pairs]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(f'Wrote {a.output}; {len(families)} directions')
if __name__=='__main__':main()
