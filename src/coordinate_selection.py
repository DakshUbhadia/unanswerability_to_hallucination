"""Stage 5 validation sweep and deterministic coordinate selection."""
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.activation_cache import format_and_verify_batch_inputs
from src.config import CFG
from src.direction_extraction import verify_stage4_outputs
from src.model_utils import (apply_norm_scaled_steering, apply_orthogonal_projection,
                             forward_with_intervention)

TABLE_COLUMNS = ["tau_kl", "rho_depth", "selected_l_star", "selected_i_star",
                 "val_ablation_log_odds", "val_steering_log_odds", "val_kl_nats",
                 "composite_j_score", "matches_default_coordinate"]
REQUIRED_KEYS = {"stage", "model_name", "n_layers", "k_suffix", "d_model", "suffix_indices",
 "n_val_pairs", "alpha_val_select", "l_star", "i_star", "pos_idx", "r_norm_at_star",
 "cos_concept_pre_ortho_at_star", "default_tau_kl", "default_rho_depth", "active_tau_kl",
 "active_rho_depth", "filter_relaxed", "relaxation_reason", "val_unans_baseline_log_odds",
 "val_ans_baseline_log_odds", "selected_ablation_log_odds", "selected_steering_log_odds",
 "selected_kl_val", "selected_composite_j", "z_score_stats", "ablation_scores_grid",
 "steering_scores_grid", "kl_val_grid", "composite_j_grid", "sensitivity_table",
 "source_stage4_sha256", "elapsed_sec", "passed_acceptance_gate"}


def load_and_validate_val_pairs(pairs_path=CFG.data_dir / "pairs.jsonl", splits_path=CFG.data_dir / "splits.json"):
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    val_templates = set(splits["val_template_ids"])
    rows = [json.loads(line) for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = sorted((p for p in rows if p.get("split") == "val"), key=lambda p: p["pair_id"])
    assert len(pairs) == 4 * len(val_templates) == CFG.n_pairs_val
    assert len({p["pair_id"] for p in pairs}) == len(pairs)
    assert all(p["template_id"] in val_templates for p in pairs)
    assert [p["family"] for p in pairs].count("A") == [p["family"] for p in pairs].count("B") == 48
    assert all("ans_prompt" in p and "unans_prompt" in p for p in pairs)
    assert not (val_templates & set(splits.get("train_template_ids", [])))
    assert not (val_templates & set(splits.get("test_template_ids", [])))
    return pairs


def compute_composite_j_grid(ablation_scores_grid, steering_scores_grid, eps=CFG.eps_norm):
    a, s = np.asarray(ablation_scores_grid, dtype=np.float64), np.asarray(steering_scores_grid, dtype=np.float64)
    assert a.ndim == s.ndim == 2 and a.shape == s.shape and np.isfinite(a).all() and np.isfinite(s).all()
    am, ast = float(a.mean()), float(a.std(ddof=0)); sm, sst = float(s.mean()), float(s.std(ddof=0))
    return (s - sm) / (sst + eps) - (a - am) / (ast + eps), {"abl_mean": am, "abl_std": ast, "steer_mean": sm, "steer_std": sst}


def select_optimal_coordinate_for_threshold(composite_j_grid, kl_val_grid, tau_kl, rho_depth,
                                             suffix_indices=(-5,-4,-3,-2,-1), tie_tol=CFG.tie_tol_j):
    j, kl = np.asarray(composite_j_grid), np.asarray(kl_val_grid)
    assert j.ndim == kl.ndim == 2 and j.shape == kl.shape and len(suffix_indices) == j.shape[1]
    mask = (np.arange(j.shape[0])[:, None] < math.floor(float(rho_depth) * j.shape[0])) & (kl <= tau_kl)
    if not mask.any(): return None
    max_j = float(j[mask].max())
    ties = [(l, p) for l, p in zip(*np.where(mask)) if max_j - float(j[l,p]) <= tie_tol]
    l, p = min(ties, key=lambda pair: (pair[0], -suffix_indices[pair[1]]))
    return int(l), int(suffix_indices[p]), int(p)


def select_primary_coordinate_with_branch_rules(ablation_scores_grid, steering_scores_grid, kl_val_grid,
 composite_j_grid, suffix_indices=(-5,-4,-3,-2,-1), default_tau_kl=CFG.default_tau_kl,
 default_rho_depth=CFG.default_rho_depth, tie_tol=CFG.tie_tol_j):
    j, kl = np.asarray(composite_j_grid), np.asarray(kl_val_grid)
    attempts = [(default_tau_kl, default_rho_depth, "none"), (.25, default_rho_depth, "relaxed_tau_kl_to_0.25"),
                (.25, .95, "relaxed_tau_kl_0.25_and_rho_depth_0.95")]
    for idx, (tau, rho, reason) in enumerate(attempts):
        coord = select_optimal_coordinate_for_threshold(j, kl, tau, rho, suffix_indices, tie_tol)
        if coord is not None:
            l, i, p = coord
            break
    else:
        limit = math.floor(default_rho_depth * j.shape[0])
        eligible = [(l,p) for l in range(limit) for p in range(j.shape[1])]
        topn = min(18, max(1, math.ceil(.10*j.size)))
        top = sorted(eligible, key=lambda x: (-float(j[x]), x[0], -suffix_indices[x[1]]))[:topn]
        bestkl = min(float(kl[x]) for x in top)
        l,p = min((x for x in top if float(kl[x]) == bestkl), key=lambda x:(x[0],-suffix_indices[x[1]]))
        i=suffix_indices[p]; tau=.25; rho=default_rho_depth; reason="top10pct_j_min_kl_fallback"
    return {"l_star":int(l),"i_star":int(i),"pos_idx":int(p),"filter_relaxed":reason!="none",
            "active_tau_kl":float(tau),"active_rho_depth":float(rho),"relaxation_reason":reason,
            "selected_ablation_log_odds":float(np.asarray(ablation_scores_grid)[l,p]),
            "selected_steering_log_odds":float(np.asarray(steering_scores_grid)[l,p]),
            "selected_kl_val":float(kl[l,p]),"selected_composite_j":float(j[l,p])}


def build_sensitivity_grid_table(ablation_scores_grid, steering_scores_grid, kl_val_grid, composite_j_grid,
 suffix_indices, primary_l_star, primary_i_star, grid_tau_kl=CFG.grid_tau_kl,
 grid_rho_depth=CFG.grid_rho_depth, tie_tol=CFG.tie_tol_j):
    rows=[]
    for tau in grid_tau_kl:
        for rho in grid_rho_depth:
            coord=select_optimal_coordinate_for_threshold(composite_j_grid,kl_val_grid,tau,rho,suffix_indices,tie_tol)
            if coord is None: d={"l_star":None,"i_star":None,"pos_idx":None,"val_ablation_log_odds":None,"val_steering_log_odds":None,"val_kl_nats":None,"composite_j_score":None}
            else:
                l,i,p=coord; d={"l_star":l,"i_star":i,"pos_idx":p,"val_ablation_log_odds":float(np.asarray(ablation_scores_grid)[l,p]),"val_steering_log_odds":float(np.asarray(steering_scores_grid)[l,p]),"val_kl_nats":float(np.asarray(kl_val_grid)[l,p]),"composite_j_score":float(np.asarray(composite_j_grid)[l,p])}
            rows.append({"tau_kl":float(tau),"rho_depth":float(rho),"selected_l_star":d["l_star"],"selected_i_star":d["i_star"],
                "val_ablation_log_odds":d["val_ablation_log_odds"],"val_steering_log_odds":d["val_steering_log_odds"],"val_kl_nats":d["val_kl_nats"],"composite_j_score":d["composite_j_score"],
                "matches_default_coordinate": d["l_star"]==primary_l_star and d["i_star"]==primary_i_star,
                "l_star":d["l_star"],"i_star":d["i_star"],"pos_idx":d["pos_idx"]})
    return rows


def _log_odds(logits, unknown_ids):
    probs=F.softmax(logits.float(),dim=-1)
    mass=probs[..., list(dict.fromkeys(map(int,unknown_ids)))].sum(-1).clamp(1e-7,1-1e-7)
    return torch.log(mass)-torch.log1p(-mass)


def _native_forward(llm, inputs, mode="none", vector=None, target_layer=None, alpha=0., is_tuple=True):
    model=llm._model; handles=[]
    if mode != "none":
        layers=model.model.layers
        selected=range(len(layers)) if mode=="ablate_all" else [target_layer]
        for idx in selected:
            def hook(module, args, output, idx=idx):
                hidden=output[0] if is_tuple else output
                changed=(apply_orthogonal_projection(hidden,vector) if mode=="ablate_all" else apply_norm_scaled_steering(hidden,vector,alpha))
                return (changed, *output[1:]) if is_tuple else changed
            handles.append(layers[idx].register_forward_hook(hook))
    try:
        out=model(**inputs, use_cache=False)
        return out.logits[:, -1, :].detach()
    finally:
        for h in handles: h.remove()


def run_validation_coordinate_sweep(llm, contract, directions_dict, val_pairs, batch_size=CFG.batch_size_forward,
 checkpoint_path=CFG.stage5_checkpoint_path, force=False, stage2_record=None, source_sha256=""):
    started=time.perf_counter(); L,K,D=map(int,(contract["n_layers"],contract["k_suffix"],contract["d_model"]))
    suffix=list(range(-K,0)); device=str(next(llm._model.parameters()).device); tokenizer=llm.tokenizer
    tokenizer.padding_side="left"; suffix_ids=list(contract["suffix_token_ids"]); unknown=[contract["unknown_token_id_nospace"],contract["unknown_token_id_space"]]
    batches={}
    for name,key in (("ans","ans_prompt"),("unans","unans_prompt")):
        prompts=[p[key] for p in val_pairs]; batches[name]=[]
        for st in range(0,len(prompts),batch_size):
            batches[name].append(format_and_verify_batch_inputs(tokenizer,prompts[st:st+batch_size],contract["system_prompt_used"],suffix_ids,"cpu"))
    # Baseline full vocabulary log probabilities are cached on CPU once.
    baseline={}; baseline_lp=[]
    probe={k:v.to(device) for k,v in batches["ans"][0].items()}
    native_probe=_native_forward(llm,probe)
    traced_probe=forward_with_intervention(llm,bool(contract["is_layer_output_tuple"]),probe,mode="none")
    assert float((native_probe.float()-traced_probe.float()).abs().max()) <= 1e-5, "Native hook baseline differs from nnsight"
    for name in ("unans","ans"):
        logits=[]
        for b in batches[name]: logits.append(_native_forward(llm,{k:v.to(device) for k,v in b.items()}))
        baseline[name]=torch.cat(logits); baseline[name+"_odds"]=_log_odds(baseline[name],unknown).mean().item()
        if name=="ans": baseline_lp=F.log_softmax(baseline[name].float(),-1).cpu()
    if stage2_record:
        for n,key in (("unans","val_unans_mean_log_odds"),("ans","val_ans_mean_log_odds")):
            assert abs(baseline[n+"_odds"]-float(stage2_record[key]))<=1e-2, f"{n} baseline differs from Stage 2"
    ab=np.full((L,K),np.nan); st=np.full((L,K),np.nan); kl=np.full((L,K),np.nan); aux=np.full((L,K),np.nan); done=-1
    checkpoint_path=Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists() and not force:
        cp=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        if cp.get("shape")==[L,K] and cp.get("source_sha256")==source_sha256:
            ab,st,kl,aux,done=cp["ablation"],cp["steering"],cp["kl"],cp["ans_ablation"],int(cp["completed_layer"])
    for l in range(done+1,L):
        for p in range(K):
            u=directions_dict["unit_directions"][l,p]; v=directions_dict["diff_in_means"][l,p]
            logits_u=[]; logits_a=[]; logits_s=[]
            for b in batches["unans"]: logits_u.append(_native_forward(llm,{k:x.to(device) for k,x in b.items()},"ablate_all",u,is_tuple=contract["is_layer_output_tuple"]))
            for b in batches["ans"]:
                logits_a.append(_native_forward(llm,{k:x.to(device) for k,x in b.items()},"ablate_all",u,is_tuple=contract["is_layer_output_tuple"]))
                logits_s.append(_native_forward(llm,{k:x.to(device) for k,x in b.items()},"steer_layer",v,l,CFG.alpha_val_select,contract["is_layer_output_tuple"]))
            lu=torch.cat(logits_u); la=torch.cat(logits_a); ls=torch.cat(logits_s)
            ab[l,p]=_log_odds(lu,unknown).mean().item(); aux[l,p]=_log_odds(la,unknown).mean().item(); st[l,p]=_log_odds(ls,unknown).mean().item()
            abl_lp=F.log_softmax(la.float(),-1).cpu()
            kl[l,p]=F.kl_div(abl_lp,baseline_lp,reduction="none",log_target=True).sum(-1).clamp(min=0).mean().item()
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=checkpoint_path.with_suffix(".pt.tmp"); torch.save({"shape":[L,K],"source_sha256":source_sha256,"completed_layer":l,"ablation":ab,"steering":st,"kl":kl,"ans_ablation":aux},tmp); tmp.replace(checkpoint_path)
    assert np.isfinite(ab).all() and np.isfinite(st).all() and np.isfinite(kl).all()
    j,z=compute_composite_j_grid(ab,st); primary=select_primary_coordinate_with_branch_rules(ab,st,kl,j,suffix)
    table=build_sensitivity_grid_table(ab,st,kl,j,suffix,primary["l_star"],primary["i_star"])
    l,p=primary["l_star"],primary["pos_idx"]
    return {"stage":5,"model_name":contract["model_name"],"n_layers":L,"k_suffix":K,"d_model":D,"suffix_indices":suffix,"n_val_pairs":len(val_pairs),"alpha_val_select":CFG.alpha_val_select,
        **primary,"r_norm_at_star":float(directions_dict["norms"][l,p]),"cos_concept_pre_ortho_at_star":float(directions_dict["cos_pre_ortho"][l,p]),
        "default_tau_kl":CFG.default_tau_kl,"default_rho_depth":CFG.default_rho_depth,"val_unans_baseline_log_odds":float(baseline["unans_odds"]),"val_ans_baseline_log_odds":float(baseline["ans_odds"]),
        "z_score_stats":z,"ablation_scores_grid":ab.tolist(),"steering_scores_grid":st.tolist(),"kl_val_grid":kl.tolist(),"composite_j_grid":j.tolist(),"sensitivity_table":table,
        "source_stage4_sha256":source_sha256,"elapsed_sec":time.perf_counter()-started,"passed_acceptance_gate":True}


def save_stage5_artifacts(selection_dict, selection_path, table2_csv_path, table2_md_path, checkpoint_path=None):
    selection_path,table2_csv_path,table2_md_path=map(Path,(selection_path,table2_csv_path,table2_md_path))
    for p in (selection_path,table2_csv_path,table2_md_path): p.parent.mkdir(parents=True,exist_ok=True)
    tmp=selection_path.with_suffix(selection_path.suffix+".tmp"); tmp.write_text(json.dumps(selection_dict,indent=2,allow_nan=False)+"\n",encoding="utf-8"); tmp.replace(selection_path)
    rows=selection_dict["sensitivity_table"]
    with table2_csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=TABLE_COLUMNS,extrasaction="ignore"); w.writeheader()
        for row in rows:
            d={k:row.get(k) for k in TABLE_COLUMNS}
            for k in TABLE_COLUMNS[4:8]:
                if d[k] is not None: d[k]=f"{float(d[k]):.4f}"
            w.writerow(d)
    def fmt(v): return "" if v is None else (f"{v:.4f}" if isinstance(v,float) else str(v))
    lines=["| "+" | ".join(TABLE_COLUMNS)+" |","|"+"|".join(["---"]*len(TABLE_COLUMNS))+"|"]
    lines += ["| "+" | ".join(fmt(r.get(c)) for c in TABLE_COLUMNS)+" |" for r in rows]
    table2_md_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    cp=Path(checkpoint_path) if checkpoint_path else CFG.stage5_checkpoint_path
    if cp.exists(): cp.unlink()


def verify_stage5_outputs(selection_path, table2_csv_path, table2_md_path, stage0_path=None, stage4_manifest_path=None):
    paths=list(map(Path,(selection_path,table2_csv_path,table2_md_path)))
    assert all(p.is_file() and p.stat().st_size>0 for p in paths)
    d=json.loads(paths[0].read_text(encoding="utf-8")); assert REQUIRED_KEYS<=d.keys()
    L,K=int(d["n_layers"]),int(d["k_suffix"]); suff=list(d["suffix_indices"])
    arrays={k:np.asarray(d[k],dtype=np.float64) for k in ("ablation_scores_grid","steering_scores_grid","kl_val_grid","composite_j_grid")}
    for name,a in arrays.items(): assert a.shape==(L,K) and np.isfinite(a).all(),name
    assert (arrays["kl_val_grid"]>=0).all()
    j,z=compute_composite_j_grid(arrays["ablation_scores_grid"],arrays["steering_scores_grid"])
    assert np.allclose(j,arrays["composite_j_grid"],atol=1e-5,rtol=1e-5)
    p=select_primary_coordinate_with_branch_rules(arrays["ablation_scores_grid"],arrays["steering_scores_grid"],arrays["kl_val_grid"],j,suff)
    assert (p["l_star"],p["i_star"],p["pos_idx"])==(d["l_star"],d["i_star"],d["pos_idx"])
    table=build_sensitivity_grid_table(arrays["ablation_scores_grid"],arrays["steering_scores_grid"],arrays["kl_val_grid"],j,suff,p["l_star"],p["i_star"])
    assert len(d["sensitivity_table"])==12 and d["sensitivity_table"]==table
    with paths[1].open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f); rows=list(reader); assert reader.fieldnames==TABLE_COLUMNS
    assert len(rows)==12
    for csvrow,row in zip(rows,table):
        for k in TABLE_COLUMNS:
            expected=row.get(k)
            if expected is None: assert csvrow[k]==""
            elif isinstance(expected,float): assert abs(float(csvrow[k])-expected)<5.1e-5
            elif isinstance(expected,bool): assert csvrow[k].lower()==str(expected).lower()
            else: assert int(csvrow[k])==expected
    assert len(paths[2].read_text(encoding="utf-8").splitlines())==14
    assert d["passed_acceptance_gate"] is True and d["pos_idx"]==suff.index(d["i_star"])
    assert d["stage"] == 5 and d["n_val_pairs"] == 96
    assert d["r_norm_at_star"] > 1e-3
    assert d["selected_kl_val"] >= 0.0
    assert d["selected_ablation_log_odds"] < d["val_unans_baseline_log_odds"]
    assert d["selected_steering_log_odds"] > d["val_ans_baseline_log_odds"]
    assert len(d["source_stage4_sha256"]) == 64
    if stage0_path and Path(stage0_path).exists():
        c=json.loads(Path(stage0_path).read_text()); assert (L,K,int(d["d_model"]))==(int(c["n_layers"]),int(c["k_suffix"]),int(c["d_model"]))
    if stage4_manifest_path and Path(stage4_manifest_path).exists():
        m=json.loads(Path(stage4_manifest_path).read_text()); assert d["source_stage4_sha256"]==m["directions_sha256"]
    return d
