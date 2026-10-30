#!/usr/bin/env python3
"""GPU Part C: measure vocabulary changes under the exact Stage 6 steering operator."""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src.config import CFG
from src.eval_common import (build_chat_batches, ensure_stage4_directions, evaluate_condition,
    load_locked_selection, load_runtime_contract, load_stage_model, read_json, ConditionCheckpoint)
from src.exploratory import assert_not_fresh, require_tier1_complete, write_exploratory_json
from src.exploratory.e2_data import load_e1_cache

def stage6_steer_kwargs(vectors, alpha):
    """Return the Stage 6 mode, raw learned_diff vector, layer and alpha convention."""
    return {"mode":"steer_layer","vec":vectors["learned_diff"],"target_layer":11,"alpha":float(alpha)}


def verify_dom_vector(vector_path, results_path):
    vector_path, results_path = Path(vector_path), Path(results_path)
    if not vector_path.is_file():
        raise FileNotFoundError(f"Missing PopQA geometry direction: {vector_path}")
    if not results_path.is_file():
        raise FileNotFoundError("Missing e2_results.json containing the Part G vector checksum")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    expected = results.get("parts", {}).get("G", {}).get("dom_vector_sha256")
    actual = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"Part G DOM vector checksum mismatch: expected {expected}, got {actual}")
    return actual

def _run_one(model,batches,contract,base_logits,groups,name,vec,alpha,tokenizer):
    total=np.zeros(np.asarray(base_logits).shape[1],dtype=np.float64); total_n=0; kl_sum=0.; split_sums={}; offset=0
    for batch in batches:
        changed=evaluate_condition(model,[batch],contract,"steer_layer",vec,11,alpha,return_logits=True)["logits"]
        n=len(changed); base=torch.as_tensor(base_logits[offset:offset+n],dtype=torch.float32); changed=torch.as_tensor(changed,dtype=torch.float32); offset+=n
        base_log=F.log_softmax(base,dim=-1); changed_log=F.log_softmax(changed,dim=-1); delta=changed_log-base_log
        total += delta.sum(0).cpu().numpy(); total_n += n
        kl_sum += float((F.softmax(base,dim=-1)*(base_log-changed_log)).sum(-1).sum().item())
        batch_groups=np.asarray(groups[offset-n:offset])
        for label in ("wrong","correct","abstain"):
            mask=batch_groups==label
            if np.any(mask):
                amount=int(mask.sum()); split_sums[label]=split_sums.get(label,0.)+float(delta[mask][:,[14109,22435]].sum().item())
                split_sums[label+"_n"]=split_sums.get(label+"_n",0)+amount
    mean=(total/max(1,total_n)).astype(np.float32); kl=kl_sum/max(1,total_n)
    split={label:{"n":int(split_sums[label+"_n"]),"mean_delta_unknown":split_sums[label]/(2*split_sums[label+"_n"])}
           for label in ("wrong","correct","abstain") if label in split_sums}
    order=np.argsort(mean); unknown_ranks={str(i):int(np.sum(mean>mean[i])+1) for i in (14109,22435)}
    top=[{"token_id":int(i),"token":tokenizer.decode([int(i)]),"mean_delta":float(mean[i])} for i in order[-20:][::-1]]
    return {"name":name,"best_unknown_rank":min(unknown_ranks.values()),"unknown_token_ranks":unknown_ranks,
            "mean_delta_unknown":float(mean[[14109,22435]].mean()),"mean_kl_base_to_steered":float(kl),
            "top20":top,"group_splits":split,"delta":mean}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--features-dir",type=Path,default=ROOT/"results/exploratory/e1_features"); ap.add_argument("--dom-vector",type=Path,required=True); ap.add_argument("--batch-size",type=int,default=16); ap.add_argument("--max-random",type=int,default=99); ap.add_argument("--output",type=Path,default=ROOT/"results/exploratory/e2_causal_vocab.json"); ap.add_argument("--mean-delta",type=Path,default=ROOT/"results/exploratory/e2_causal_vocab_meanDelta.npy"); ap.add_argument("--push-hf",action="store_true"); args=ap.parse_args(argv)
    assert_not_fresh(args.features_dir); assert_not_fresh(args.dom_vector); assert_not_fresh(args.output); require_tier1_complete(ROOT)
    if not 1<=args.max_random<=99: raise ValueError("--max-random must be 1..99")
    verify_dom_vector(args.dom_vector, ROOT/"results/exploratory/e2_results.json")
    cache=load_e1_cache(args.features_dir,dry_run=False); meta=cache["meta"]
    graded=meta[meta["is_popularity_matched"].astype(bool) & meta["matched_pair_id"].notna() & meta["baseline_partition"].isin(["wrong","correct"]) ]
    abstain=meta[meta["baseline_partition"]=="abstain"]
    rows=pd.concat([graded,abstain]).drop_duplicates("popqa_id").reset_index(drop=True)
    contract=load_runtime_contract(); from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(contract["model_name"]); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    from src.popqa_uncertainty import _verify_popqa_input_identity
    _verify_popqa_input_identity(tok,rows.to_dict("records"),contract["system_prompt_used"],contract["model_name"],contract["suffix_token_ids"])
    batches=build_chat_batches(tok,rows["question"].astype(str).tolist(),contract["system_prompt_used"],contract["suffix_token_ids"],args.batch_size)
    dirs,stage4_sha=ensure_stage4_directions(CFG.stage4_out_path,CFG.stage4_manifest_path,CFG.stage0_out_path)
    selection=load_locked_selection(CFG.stage5_out_path,stage4_sha)
    if (int(selection["l_star"]),int(selection["pos_idx"]))!=(11,1): raise RuntimeError("Stage 5 coordinate differs from registered (11,1)")
    learned=dirs["diff_in_means"][11,1].float(); learned_norm=torch.linalg.vector_norm(learned).item()
    if abs(learned_norm-3.807962656)>=1e-4: raise RuntimeError(f"Stage 4 steering norm mismatch: {learned_norm}")
    vectors={"learned_diff":learned,"random_scaled":dirs["random_controls_unit"][:args.max_random,11,1].float()*learned_norm,
             "concept_scaled":dirs["r_concept_ortho_scaled"][11,1].float()}
    dom=torch.as_tensor(np.load(args.dom_vector),dtype=torch.float32); dom=dom/torch.linalg.vector_norm(dom)*learned_norm
    model,tok,_dtype=load_stage_model(contract,device="cuda:0")
    base=evaluate_condition(model,batches,contract,return_logits=True)["logits"]
    groups=rows["baseline_partition"].astype(str).to_numpy(); results=[]
    fingerprint=hashlib.sha256((stage4_sha+"|"+"|".join(rows.popqa_id.astype(str))+f"|{args.max_random}").encode()).hexdigest()
    checkpoint=ConditionCheckpoint(ROOT/"results/exploratory/e2_causal_vocab_checkpoint.pt",fingerprint,every=1)
    plan=[(f"learned_alpha_{alpha:g}",vectors["learned_diff"],alpha) for alpha in (.25,.5,1.0)]
    plan += [(f"random_{i+1}",vec,1.0) for i,vec in enumerate(vectors["random_scaled"])]
    plan += [("concept",vectors["concept_scaled"],1.0),("popqa_dom",dom,1.0)]
    for name,vec,alpha in plan:
        if name in checkpoint: result=checkpoint.get(name)
        else:
            result=_run_one(model,batches,contract,base,groups,name,vec,alpha,tok)
            checkpoint.put(name,result)
        results.append(dict(result))
    checkpoint.flush()
    rand=np.asarray([x["mean_delta_unknown"] for x in results if x["name"].startswith("random_")])
    for name in ("learned_alpha_1","popqa_dom"):
        item=next(x for x in results if x["name"]==name)
        specific=item["best_unknown_rank"]<=10 and item["mean_delta_unknown"]>np.percentile(rand,99)
        item["label"]=("C: promotes_unknown_specifically" if specific else
                       "C: promotes_unknown_nonspecific" if item["best_unknown_rank"]<=10 else "C: does_not_promote_unknown")
    learned_specific=next(x for x in results if x["name"]=="learned_alpha_1")["label"]=="C: promotes_unknown_specifically"
    dom_specific=next(x for x in results if x["name"]=="popqa_dom")["label"]=="C: promotes_unknown_specifically"
    matrix=np.stack([item.pop("delta") for item in results]).astype(np.float16)
    args.mean_delta.parent.mkdir(parents=True,exist_ok=True); np.save(args.mean_delta,matrix)
    payload={"n_prompts":len(rows),"n_wrong":int(np.sum(groups=="wrong")),"n_correct":int(np.sum(groups=="correct")),"n_abstain":int(np.sum(groups=="abstain")),
             "learned_norm":learned_norm,"max_random":args.max_random,"conditions":results,
             "C_dom_label":"C_dom: dissociated" if learned_specific and not dom_specific else "C_dom: also_promotes_unknown"}
    write_exploratory_json(args.output,payload,ROOT)
    if args.push_hf:
        import os
        from huggingface_hub import HfApi
        repo=os.environ.get("HF_TENSOR_REPO_ID")
        if not repo: raise RuntimeError("HF_TENSOR_REPO_ID is required for --push-hf")
        HfApi().upload_file(path_or_fileobj=str(args.mean_delta),path_in_repo="exploratory/e2_causal_vocab_meanDelta.npy",repo_id=repo,repo_type="dataset")
    print(f"Wrote {args.output} and {args.mean_delta}")
    return 0
if __name__=="__main__": raise SystemExit(main())
