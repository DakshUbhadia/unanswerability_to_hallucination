#!/usr/bin/env python3
"""Tier 2 Stage 4b: activation pool GPU pass and CPU null-library build.

Run `--gpu-pass` on Kaggle, then `--build` on CPU (or the same machine).
"""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import CFG
from src.null_directions import (fit_activation_covariance,sample_covariance_null,top_pca_directions,
    labelfree_null,template_partition_null,variance_matched_subset)
from src.concept_attributes import derive_attribute_rows,make_contrast_directions,save_attribute_rows
from src.steering_scale import residual_norm_stats,alpha_scale_table
from src.eval_common import (load_runtime_contract,load_stage_model,read_json,read_jsonl,
    build_chat_batches,capture_residuals_and_logits,sha256_file)

POOL=CFG.results_dir/'stage4b_pool_acts.pt'; NORMS=CFG.results_dir/'stage4b_resid_norms.json'
NULLPT=CFG.results_dir/'stage4b_nulls.pt';NULLJSON=CFG.results_dir/'stage4b_nulls.json'

def gpu_pass(args):
    """Capture fixed-coordinate acts for train/val/archive pools and test residual norms."""
    from scripts.tier2_preflight import verify_preflight
    verify_preflight()
    from src.activation_cache import load_and_validate_train_pairs
    from src.tokenization_contract import canonical_input_ids,chat_template_sha256,input_ids_sha256
    contract=load_runtime_contract(args.stage0,args.stage2); model,tok,_=load_stage_model(contract,args.device,args.dtype)
    tok.padding_side='left'; selection=read_json(args.stage5); l=int(selection['l_star']); p=int(selection['pos_idx'])
    pairs=read_jsonl(args.pairs);train=load_and_validate_train_pairs(args.pairs,args.splits)
    val=sorted([r for r in pairs if r['split']=='val'],key=lambda r:r['pair_id'])
    arch=read_jsonl(args.archive_popqa); test=sorted([r for r in pairs if r['split']=='test'],key=lambda r:r['pair_id'])
    template_hash=chat_template_sha256(tok); meta=[]; chunks=[]
    # Stage 3 contains the exact training tensors and avoids re-running locked Stage 3.
    cache=torch.load(args.stage3,map_location='cpu',weights_only=False)
    for label,key in [('answerable','train_pair_acts_ans'),('unanswerable','train_pair_acts_unans')]:
        acts=cache[key][:,l,p,:].float().numpy()
        prompt_key='ans_prompt' if label=='answerable' else 'unans_prompt'
        for row,a in zip(train,acts):
            ids=canonical_input_ids(tok,row[prompt_key],contract['system_prompt_used'],contract['suffix_token_ids'])
            meta.append({'source':'train','source_id':row['pair_id']+':'+label,'label':label,'template_id':row['template_id'],'family':row['family'],
                'input_ids_sha256':input_ids_sha256(ids,contract['model_name'],template_hash)});chunks.append(a)
    for source,rows,prompt_keys in [('val',val,('ans_prompt','unans_prompt')),('archive_popqa',arch,('question',))]:
        prompts=[];owners=[]
        for row in rows:
            for label,key in ((('answerable','ans_prompt'),('unanswerable','unans_prompt')) if source=='val' else ((row.get('baseline_partition','unknown'),'question'),)):
                prompts.append(row[key]);owners.append((row,label,key))
        batch=build_chat_batches(tok,prompts,contract['system_prompt_used'],contract['suffix_token_ids'],args.batch_size)
        acts,_=capture_residuals_and_logits(model,batch,contract,[l],p);acts=acts[:,0,:].numpy()
        for (row,label,key),a in zip(owners,acts):
            ids=canonical_input_ids(tok,row[key],contract['system_prompt_used'],contract['suffix_token_ids'])
            meta.append({'source':source,'source_id':str(row.get('pair_id',row.get('hf_row_idx')))+':'+str(label),'label':label,
                'template_id':row.get('template_id'),'family':row.get('family'),'hf_row_idx':row.get('hf_row_idx'),
                'input_ids_sha256':input_ids_sha256(ids,contract['model_name'],template_hash)});chunks.append(a)
    acts=np.asarray(chunks,dtype=np.float32)
    if not np.isfinite(acts).all():raise ValueError('non-finite pool activation')
    test_ids={r['pair_id'] for r in test}
    if any(m['source']=='test' or m['source_id'].split(':')[0] in test_ids for m in meta):raise AssertionError('test prompt leaked into covariance pool')
    torch.save({'acts':torch.from_numpy(acts),'rows':meta,'layer':l,'position_index':p},POOL)
    # Capture all residual norms in one unmodified forward per test batch.
    from src.activation_cache import format_and_verify_batch_inputs
    all_prompts=[p[k] for p in test for k in ('ans_prompt','unans_prompt')]
    bos_norms=[];nonbos_norms=[];istar_norms=[];device=next(model.parameters()).device
    layer=model.model.layers[l]
    for start in range(0,len(all_prompts),args.batch_size):
        prompts=all_prompts[start:start+args.batch_size]
        enc=format_and_verify_batch_inputs(tok,prompts,contract['system_prompt_used'],contract['suffix_token_ids'],str(device))
        saved={}
        def hook(_module,_inputs,output):saved['x']=output[0] if bool(contract['is_layer_output_tuple']) else output
        handle=layer.register_forward_hook(hook)
        try:
            with torch.no_grad():model(**enc,use_cache=False)
        finally:handle.remove()
        x=saved['x'].float(); norms=torch.linalg.vector_norm(x,dim=-1).cpu().numpy();mask=enc['attention_mask'].cpu().numpy()
        for z,m in zip(norms,mask):
            nz=np.flatnonzero(m);bos_norms.append(float(z[nz[0]]));nonbos_norms.extend(float(z[j]) for j in nz[1:])
            if len(nz)>=abs(int(selection['i_star'])):istar_norms.append(float(z[int(selection['i_star'])]))
    summaries=residual_norm_stats(nonbos_norms,istar_norms,bos_norms);summaries['n_test_prompts']=len(all_prompts)
    NORMS.write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(meta)} pool activations to {POOL}; test prompts excluded by construction')

def build(args):
    from scripts.tier2_preflight import verify_preflight
    verify_preflight()
    pool=torch.load(args.pool,map_location='cpu',weights_only=False); acts=pool['acts'].numpy().astype(np.float64); rows=pool['rows']
    selection=read_json(args.stage5);directions=torch.load(args.stage4,map_location='cpu',weights_only=False)
    l=int(selection['l_star']);p=int(selection['pos_idx']);r_hat=directions['unit_directions'][l,p].float().numpy();r_norm=float(directions['norms'][l,p])
    fit=fit_activation_covariance(acts);cov=fit['covariance']
    covnull=sample_covariance_null(cov,CFG.n_null_cov,CFG.seed_cov_null,r_norm,r_hat)
    pca=top_pca_directions(cov,CFG.n_pca_null,r_hat,r_norm)
    train_rows=[i for i,x in enumerate(rows) if x['source']=='train']
    t_ids=list(dict.fromkeys(rows[i]['template_id'] for i in train_rows));lookup={(rows[i]['source_id'].split(':')[0],rows[i]['label']):acts[i] for i in train_rows}
    train_pairs=read_jsonl(args.pairs);byid={r['pair_id']:r for r in train_pairs if r['split']=='train'}
    ans=np.stack([[lookup[(pid,'answerable')] for pid in sorted(byid) if byid[pid]['template_id']==t] for t in t_ids])
    unans=np.stack([[lookup[(pid,'unanswerable')] for pid in sorted(byid) if byid[pid]['template_id']==t] for t in t_ids])
    template_order=[next(byid[pid]['family'] for pid in sorted(byid) if byid[pid]['template_id']==t) for t in t_ids]
    lf=labelfree_null(ans,unans,n=CFG.n_null_labelfree,seed=CFG.seed_labelfree_null,r_norm=r_norm)
    tp=template_partition_null(ans,unans,None,template_order,r_hat,r_norm,CFG.n_null_template_partition,CFG.seed_template_partition_null)
    from transformers import AutoTokenizer
    stage2=read_json(args.stage2);tokenizer=AutoTokenizer.from_pretrained(stage2['model_name'])
    attr=derive_attribute_rows(args.pairs,args.templates,args.domains,tokenizer=tokenizer)
    # align training rows with activation order (pair sorted, answerable then unanswerable)
    attracts=np.stack([lookup[(r['pair_id'],r['label'])] for r in attr])
    concepts=make_contrast_directions(attracts,attr,r_hat,r_norm)
    save_attribute_rows(attr,args.attributes_out)
    vm=variance_matched_subset(covnull,cov,r_hat)
    nulls={'covariance':covnull,'pca':pca,'label_free':lf,'template_partition':tp,'concepts':concepts,'variance_matched':vm}
    torch.save(nulls,args.output_pt)
    projected=float(r_hat@cov@r_hat)
    metadata={'layer':l,'position_index':p,'n_pool':len(rows),'covariance_shrinkage':fit['shrinkage'],'participation_ratio':fit['participation_ratio'],
      'covariance_null_n':CFG.n_null_cov,'label_free_n':CFG.n_null_labelfree,'template_partition_n':CFG.n_null_template_partition,
      'pca_n':CFG.n_pca_null,'n_concepts':len(concepts),'r_projected_variance':projected,'r_variance_percentile':covnull['metadata']['r_variance_percentile'],
      'r_hat':r_hat.tolist(),'r_norm':r_norm,
      'concept_contrasts':[{'attribute':x['attribute'],'level':x['level'],'group_sizes':x['group_sizes'],'pre_orthogonalization_cosine':x['pre_orthogonalization_cosine']} for x in concepts],
      'label_free_mean_abs_cosine':float(np.mean(np.abs(lf['unit']@r_hat))),'template_max_abs_cosine':float(np.max(np.abs(tp['unit']@r_hat))),
      'steer_norm_max_error':float(max(np.max(np.abs(np.linalg.norm(x['steer'],axis=-1)-r_norm)) for x in [covnull,pca,lf,tp])),
      'source_counts':dict((s,sum(r['source']==s for r in rows)) for s in sorted({r['source'] for r in rows})),
      'pool_rows':rows,'pool_file_sha256':sha256_file(args.pool)}
    if metadata['label_free_mean_abs_cosine']>=.1:raise RuntimeError('label-free null cosine gate failed')
    if metadata['template_max_abs_cosine']>=1e-5:raise RuntimeError('template null orthogonality gate failed')
    norm_stats=json.loads(Path(args.norms).read_text(encoding='utf-8'))
    metadata['steering_scale']=alpha_scale_table(sorted(set(CFG.grid_alpha_nec+CFG.grid_alpha_suf)),r_norm,norm_stats)
    args.output_json.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Built null library: {args.output_pt}; {len(concepts)} named contrasts')

def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='mode',required=True)
    g=sp.add_parser('gpu-pass');g.add_argument('--device',default='cuda:0');g.add_argument('--dtype',default='auto');g.add_argument('--batch-size',type=int,default=CFG.batch_size_forward)
    for name,default in [('stage0',CFG.stage0_out_path),('stage2',CFG.stage2_out_path),('stage3',CFG.stage3_out_path),('stage4',CFG.stage4_out_path),('stage5',CFG.stage5_out_path),('pairs',CFG.data_dir/'pairs.jsonl'),('splits',CFG.data_dir/'splits.json'),('archive-popqa',CFG.results_dir/'archive_v42/data/popqa_sample.jsonl')]:g.add_argument('--'+name, type=Path, default=default)
    b=sp.add_parser('build');
    for name,default in [('pool',POOL),('pairs',CFG.data_dir/'pairs.jsonl'),('templates',CFG.data_dir/'templates.jsonl'),('domains',CFG.data_dir/'domain_categories.json'),('stage2',CFG.stage2_out_path),('stage4',CFG.stage4_out_path),('stage5',CFG.stage5_out_path),('norms',NORMS),('output-pt',NULLPT),('output-json',NULLJSON),('attributes-out',CFG.data_dir/'concept_attributes.jsonl')]:b.add_argument('--'+name,type=Path,default=default)
    a=ap.parse_args();gpu_pass(a) if a.mode=='gpu-pass' else build(a)
if __name__=='__main__':main()
