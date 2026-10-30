#!/usr/bin/env python3
"""Run registered CPU E2 analyses R, G, L and A; Part C is a separate GPU job."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score
from src.exploratory import assert_not_fresh, exploratory_header, require_tier1_complete, write_exploratory_json
from src.exploratory.e2_data import load_e1_cache, load_rhat, population, projection_sign_check
from src.exploratory.e2_analysis import (crossfit_residual, stratified_confidence_auroc, partial_spearman,
    fit_dom, fit_logistic_pattern, cosine_spaces, geometry_nulls, load_unembedding_and_norm,
    static_lens, amnesic_auroc, label_part_R, label_part_G, label_part_L, label_part_A, combined_verdict)

def _auc_ci(y,s,g,n,seed):
    from src.exploratory.e2_analysis import _cluster_boot_auc
    return {"auroc":float(roc_auc_score(y,s)),"ci95":_cluster_boot_auc(y,s,g,n,seed)}

def _features_at(cache, rows, pos=1):
    return np.asarray(cache["suf"][rows.index.to_numpy(dtype=int),pos,:,:],dtype=np.float64)

def _run_r(cache, directions, nboot, seed, dry):
    meta=cache["meta"]; rows,y,groups,units,base=population(meta,"P1"); idx=rows.index.to_numpy(dtype=int)
    X=np.asarray(cache["suf"][idx,1,11,:],dtype=np.float64); s=X@directions["unit"][11,1]
    stage8=json.loads((ROOT/"results/stage8_popqa_core.json").read_text(encoding="utf-8"))
    # Stage 8 uses its own registered matched-set metric; fail closed on any disagreement.
    sign_info={"flipped":False,"auroc":float(roc_auc_score(y,s))}
    if not dry:
        # The helper checks only the matched Stage 8 population and flips the projection if needed.
        # E2's input projection is explicitly assembled in metadata row order.
        stage8_proj=np.full(len(meta),np.nan); stage8_proj[idx]=s
        checked,sign_info=projection_sign_check(meta,stage8_proj,stage8)
        s=np.asarray(checked)[idx]
    auc=float(roc_auc_score(y,s))
    conf=rows[["log_odds","maxp","entropy"]].to_numpy(float)
    primary=crossfit_residual(s,conf,y,groups,"spline_ridge",5,5,20260414)
    secondary=crossfit_residual(s,conf,y,groups,"linear",5,5,20260414)
    p1=_auc_ci(y,primary["residual"],units,nboot,seed)
    p2=_auc_ci(y,secondary["residual"],units,nboot,seed)
    strat=stratified_confidence_auroc(s,y,rows.log_odds.to_numpy(float),units,nboot,seed)
    partial=partial_spearman(s,y,conf,nboot,seed,units)
    reverse=crossfit_residual(rows.log_odds.to_numpy(float),s[:,None],y,groups,"spline_ridge",5,5,20260414)
    out={"n":len(rows),"projection_auroc":auc,"stage8_auroc":stage8.get("auroc_proj"),"projection_sign_check":sign_info,
         "primary_spline_ridge":{**{k:v for k,v in primary.items() if k not in ("residual","prediction","oof_by_repeat","fold_assignments")},**p1},
         "secondary_linear":{**{k:v for k,v in secondary.items() if k not in ("residual","prediction","oof_by_repeat","fold_assignments")},**p2},
         "stratified":strat,"partial_spearman":partial,
         "reverse_log_odds_residual_auroc":_auc_ci(y,reverse["residual"],units,nboot,seed),
         "oof_residual":primary["residual"],"labels":rows.baseline_partition.astype(str).tolist(),
         "interpretation":label_part_R([p1["ci95"]["low"],p1["ci95"]["high"]])}
    out["ci_includes_0_5"]=bool(p1["ci95"]["low"]<=.5<=p1["ci95"]["high"])
    # E1 mirror predictions are optional until the additive hook has been run.
    oof=ROOT/("results/exploratory/dryrun/e1_oof/P1_suf-4_logistic.npz" if dry else "results/exploratory/e1_oof/P1_suf-4_logistic.npz")
    if oof.exists():
        saved=np.load(oof,allow_pickle=False); byid={str(k):v for k,v in zip(saved["popqa_id"],saved["oof_score"])}
        mirror=np.asarray([byid[str(i)] for i in rows.popqa_id]); mr=crossfit_residual(mirror,conf,y,groups,"spline_ridge",5,5,20260414)
        out["e1_probe_mirror"]={**_auc_ci(y,mr["residual"],units,nboot,seed),"crossfit_r2":mr["crossfit_r2"]}
    else: out["e1_probe_mirror"]={"pending":"Run E1 with --save-oof first"}
    # Secondary P2 analysis adds popularity and relation controls as preregistered.
    p2rows,p2y,p2groups,p2units,_=population(meta,"P2"); p2idx=p2rows.index.to_numpy(int)
    p2x=np.asarray(cache["suf"][p2idx,1,11,:],dtype=np.float64)
    p2s=p2x@directions["unit"][11,1]
    relation=np.asarray(__import__("pandas").get_dummies(p2rows["prop_grouped"].astype(str),dtype=float))
    p2conf=np.column_stack([p2rows[["log_odds","maxp","entropy","log_s_pop"]].to_numpy(float),relation])
    p2res=crossfit_residual(p2s,p2conf,p2y,p2groups,"spline_ridge",5,5,20260414)
    out["P2_secondary"]={"n":len(p2rows),"spline_ridge":{**_auc_ci(p2y,p2res["residual"],p2units,nboot,seed),
        "crossfit_r2":p2res["crossfit_r2"],"auroc":p2res["auroc"]}}
    return out,primary["residual"]

def _run_g(cache,directions):
    meta=cache["meta"]; rows,y,groups,units,base=population(meta,"P2"); idx=rows.index.to_numpy(int)
    heat={k:np.full((28,5),np.nan) for k in ("raw","outlier_removed","whitened")}; locked=None
    outliers=np.zeros((28,5,8),dtype=np.int32)
    # Geometry directions are fitted on P2 and never scored as probes.
    for pos in range(5):
        for layer in range(28):
            X=np.asarray(cache["suf"][idx,pos,layer,:],dtype=np.float64)
            d=fit_dom(X,y); dims=np.argsort(np.mean(np.abs(X),axis=0))[-8:]
            outliers[layer,pos]=dims
            c=cosine_spaces(directions["unit"][layer,pos],d,X,dims)
            for space in heat: heat[space][layer,pos]=c[space]
            if layer==11 and pos==1:
                from src.exploratory.e2_analysis import _scale_equivariant_lw_covariance
                cov=_scale_equivariant_lw_covariance(X)
                nulls=geometry_nulls(directions["unit"][11,1],X,y,cov)
                w,pattern=fit_logistic_pattern(X,y,C=.1,cov=cov)
                dims=np.argsort(np.mean(np.abs(X),axis=0))[-8:]
                locked={"cosines":c,"nulls":nulls,"dom":d,
                        "logistic_raw_weight_cosines":cosine_spaces(directions["unit"][11,1],w,X,dims),
                        "logistic_haufe_pattern_cosines":cosine_spaces(directions["unit"][11,1],pattern,X,dims),
                        "label":label_part_G(c["whitened"],nulls["covariance_null"],nulls["permutation_null"])}
    rhat=directions["unit"][11,1]; cross={"suf-1":[],"ans_first":[]}
    for layer in range(28):
      x=np.asarray(cache["suf"][idx,4,layer,:],dtype=np.float64); cross["suf-1"].append(float(np.dot(rhat,fit_dom(x,y))/(np.linalg.norm(rhat)*np.linalg.norm(fit_dom(x,y)))))
      xa=np.asarray(cache["ans"][idx,0,layer,:],dtype=np.float64); da=fit_dom(xa,y); cross["ans_first"].append(float(np.dot(rhat,da)/(np.linalg.norm(rhat)*np.linalg.norm(da))))
    dom_path=assert_not_fresh(ROOT/"results/exploratory/e2_dom_11_suf-4.npy")
    dom_path.parent.mkdir(parents=True,exist_ok=True); np.save(dom_path,locked["dom"].astype(np.float32))
    dom_sha=hashlib.sha256(dom_path.read_bytes()).hexdigest()
    return {"heatmaps":{k:v.tolist() for k,v in heat.items()},"outlier_dimensions":outliers.tolist(),"cross_position_alignment":cross,
            "locked":{k:v for k,v in locked.items() if k!="dom"},"dom_vector_path":str(dom_path.relative_to(ROOT)),
            "dom_vector_sha256":dom_sha}

def _run_a(cache,directions,nboot,seed,contract):
    rows,y,groups,units,base=population(cache["meta"],"P1"); idx=rows.index.to_numpy(int)
    X=np.asarray(cache["suf"][idx,1,11,:],dtype=np.float64)
    removals={"r_hat":directions["unit"][11,1]}
    U,g,_meta=load_unembedding_and_norm(contract["model_name"])
    unk=np.asarray(contract["unknown_token_ids"],dtype=int)
    uunk=g.astype(np.float64)*(U[unk[0]].astype(np.float64)+U[unk[1]].astype(np.float64))/2
    removals["unknown_unembedding"]=uunk/np.linalg.norm(uunk)
    removals.update({f"random_{i+1}":directions["random_unit"][i,11,1] for i in range(99)})
    out=amnesic_auroc(X,y,groups,removals,{"folds":5,"repeats":5,"seed":20260414})
    drops=[v["drop"] for k,v in out["after"].items() if k.startswith("random_")]
    out["random_drop_percentiles"]={"p95":float(np.percentile(drops,95)),"p99":float(np.percentile(drops,99))}
    rng=np.random.default_rng(seed); unique=np.unique(groups); by={u:np.flatnonzero(groups==u) for u in unique}; boot=[]
    a=out["before_scores"]; b=out["after"]["r_hat"]["scores"]
    for _ in range(int(nboot)):
        ix=np.concatenate([by[u] for u in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))==2: boot.append(float(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix])))
    ci=[float(v) for v in np.percentile(boot,[2.5,97.5])]
    for val in out["after"].values(): val.pop("scores",None)
    out.pop("before_scores",None); out["r_hat_drop_ci95"]=ci
    out["label"]=label_part_A(out["after"]["r_hat"]["drop"],ci,drops)
    return out

def _run_l(cache,directions,contract,revision=None):
    from transformers import AutoTokenizer
    U,g,meta=load_unembedding_and_norm(contract["model_name"],revision); tok=AutoTokenizer.from_pretrained(contract["model_name"],revision=meta["revision"])
    unknown=contract["unknown_token_ids"]; words=["unknown"," unknown","UNKNOWN","I"," I","Sorry","There","N"]
    extra=list(dict.fromkeys(int(tok.encode(w,add_special_tokens=False)[0]) for w in words if tok.encode(w,add_special_tokens=False)))
    table=[]; rand=[]
    for layer in range(28):
      for pos in range(5):
        v=static_lens(U,g,directions["unit"][layer,pos],unknown,extra,tokenizer=tok)
        table.append({"layer":layer,"pos_idx":pos,**v})
        if layer==11 and pos==1:
          for j,rv in enumerate(directions["random_unit"][:,layer,pos]): rand.append(static_lens(U,g,rv,unknown,top_k=1)["best_unknown_rank"])
    label=label_part_L(table[11*5+1]["best_unknown_rank"],rand)
    return {"revision":meta["revision"],"approximation":"Final RMSNorm elementwise gain; omitted positive scalar denominator does not affect ranking.",
            "rows":table,"random_best_unknown_ranks":rand,"label":label,
            "caveat":"At (11,-4), r_hat is 16 layers and 3 suffix positions from the readout; a negative static lens result does not contradict Part R. Part C is decisive."}

def _plots(results,figdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figdir.mkdir(parents=True,exist_ok=True)
    if "R" in results:
      rr=results["R"]; s=rr["oof_residual"]; y=np.asarray([x=="wrong" for x in rr["labels"]]); fig,ax=plt.subplots(figsize=(5,4)); ax.violinplot([s[~y],s[y]],showmeans=True); ax.set_xticks([1,2], ["correct","wrong"]); ax.set_title(f"Residual AUROC={rr['primary_spline_ridge']['auroc']:.3f}"); fig.tight_layout(); fig.savefig(figdir/"e2_R_residual.png",dpi=150); plt.close(fig)
      st=rr["stratified"]; fig,ax=plt.subplots(figsize=(6,4)); q=[r["quintile"] for r in st["strata"] if r["auroc"] is not None]; v=[r["auroc"] for r in st["strata"] if r["auroc"] is not None]; ax.plot(q,v,"o-"); ax.axhline(.5,color="black",lw=.8); ax.set(xlabel="log_odds quintile",ylabel="AUROC"); fig.tight_layout(); fig.savefig(figdir/"e2_R_stratified.png",dpi=150); plt.close(fig)
    if "G" in results:
      fig,axs=plt.subplots(1,3,figsize=(12,8));
      for ax,(k,v) in zip(axs,results["G"]["heatmaps"].items()): im=ax.imshow(v,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm"); ax.set_title(k); ax.set_xlabel("suffix position"); ax.set_ylabel("layer")
      fig.colorbar(im,ax=axs.ravel().tolist()); fig.tight_layout(); fig.savefig(figdir/"e2_G_cosine_heatmaps.png",dpi=150); plt.close(fig)
      locked=results["G"]["locked"]; null=locked["nulls"]
      fig,ax=plt.subplots(figsize=(6,4)); ax.hist(null["covariance_null"],bins=25,alpha=.55,label="covariance matched"); ax.hist(null["permutation_null"],bins=25,alpha=.55,label="label permutation"); ax.axvline(locked["cosines"]["whitened"],color="black",lw=2,label="observed whitened cosine"); ax.legend(); ax.set_xlabel("whitened cosine"); fig.tight_layout(); fig.savefig(figdir/"e2_G_null.png",dpi=150); plt.close(fig)
    if "L" in results:
      rows=results["L"]["rows"]; ranks=np.asarray([r["best_unknown_rank"] for r in rows]).reshape(28,5); ctl=results["L"]["random_best_unknown_ranks"]
      lo,hi=np.percentile(ctl,[1,99]); fig,ax=plt.subplots(figsize=(8,8)); im=ax.imshow(np.log10(ranks),aspect="auto",cmap="viridis"); ax.scatter([1],[11],marker="x",color="red"); ax.set(xticks=range(5),xticklabels=["suf-5","suf-4","suf-3","suf-2","suf-1"],ylabel="layer",title=f"Unknown rank; random-control 1–99%: {lo:.0f}–{hi:.0f}"); fig.colorbar(im,ax=ax,label="log10 rank"); fig.tight_layout(); fig.savefig(figdir/"e2_L_unknown_rank.png",dpi=150); plt.close(fig)
    if "A" in results:
      aa=results["A"]; names=["before","r_hat","unknown_unembedding"]; vals=[aa["before"],aa["after"]["r_hat"]["auroc"],aa["after"]["unknown_unembedding"]["auroc"]]; rand=[v["auroc"] for k,v in aa["after"].items() if k.startswith("random_")]
      fig,ax=plt.subplots(figsize=(7,4)); ax.bar(names,vals); ax.axhspan(np.percentile(rand,2.5),np.percentile(rand,97.5),alpha=.2,label="random-removal 95% band"); ax.set_ylim(.4,1); ax.set_ylabel("nested OOF AUROC"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"e2_A_amnesic.png",dpi=150); plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--features-dir",type=Path,default=ROOT/"results/exploratory/e1_features"); p.add_argument("--parts",default="R,G,L,A"); p.add_argument("--population",default="P1,P2"); p.add_argument("--n-jobs",type=int,default=1); p.add_argument("--dry-run",action="store_true"); p.add_argument("--bootstrap",type=int,default=1000); p.add_argument("--model-revision"); args=p.parse_args(argv)
    parts={x.strip().upper() for x in args.parts.split(",") if x.strip()};
    if args.dry_run and parts!={"R"}: raise ValueError("--dry-run permits only Part R")
    assert_not_fresh(args.features_dir)
    from threadpoolctl import threadpool_limits
    if args.n_jobs < 1: raise ValueError("--n-jobs must be positive")
    threadpool_limits(limits=args.n_jobs)
    cache=load_e1_cache(args.features_dir,args.dry_run); directions=load_rhat()
    if not args.dry_run: require_tier1_complete(ROOT)
    prereg=ROOT/"results/exploratory/e2_preregistration.json"; prereg_sha=hashlib.sha256(prereg.read_bytes()).hexdigest()
    target=ROOT/("results/exploratory/dryrun" if args.dry_run else "results/exploratory")
    existing=target/"e2_results.json"
    if existing.exists():
      prior=json.loads(existing.read_text(encoding="utf-8"))
      if prior.get("preregistration_sha256")!=prereg_sha: raise RuntimeError("Existing E2 output was generated from a different preregistration")
      payload={**prior,"parts":dict(prior.get("parts",{}))}
    else: payload={"preregistration_sha256":prereg_sha,"dry_run":args.dry_run,"parts":{}}
    if "R" in parts: payload["parts"]["R"],_=_run_r(cache,directions,args.bootstrap,20260407,args.dry_run)
    if "G" in parts: payload["parts"]["G"]=_run_g(cache,directions)
    if "A" in parts:
      from src.eval_common import load_runtime_contract
      payload["parts"]["A"]=_run_a(cache,directions,args.bootstrap,20260407,load_runtime_contract())
    if "L" in parts:
      from src.eval_common import load_runtime_contract
      payload["parts"]["L"]=_run_l(cache,directions,load_runtime_contract(),args.model_revision)
    if "C" in parts:
      cpath=assert_not_fresh(ROOT/"results/exploratory/e2_causal_vocab.json")
      if not cpath.exists(): raise FileNotFoundError("Part C GPU output is missing; run e2_causal_vocab.py on the owner GPU first")
      payload["parts"]["C"]=json.loads(cpath.read_text(encoding="utf-8"))
      payload["parts"]["C"]["label"]=next(x["label"] for x in payload["parts"]["C"]["conditions"] if x["name"]=="learned_alpha_1")
    labels={k:v.get("label",v.get("interpretation")) for k,v in payload["parts"].items()}
    e1_c1=e1_c2=None; e1_path=ROOT/("results/exploratory/dryrun/e1_results.json" if args.dry_run else "results/exploratory/e1_results.json")
    if e1_path.exists():
      e1=json.loads(e1_path.read_text(encoding="utf-8")); c=e1.get("populations",{}).get("P1",{}).get("contrasts",{}); e1_c1=c.get("C1_pregen",{}).get("interpretation"); e1_c2=c.get("C2_postans",{}).get("interpretation")
    payload["combined_verdict"]=combined_verdict(labels,e1_c1,e1_c2)
    target.mkdir(parents=True,exist_ok=True)
    # Keep arrays out of JSON; detailed residuals are available in figures and NPZ.
    results=payload["parts"].get("R",{}); residual=results.pop("oof_residual",None)
    if residual is not None: np.savez_compressed(target/"e2_R_oof.npz",residual=residual)
    write_exploratory_json(target/"e2_results.json",payload,ROOT)
    _plots(payload["parts"], ROOT/"figures/exploratory" if not args.dry_run else target/"figures")
    summary=["# E2 decomposition summary","","Exploratory post-registration results. Dry-run outputs are not reportable.","","| Part | Statistic | CI or null comparison | Label |","|---|---|---|---|"]
    labels={}
    for name,item in payload["parts"].items():
      label=item.get("label",item.get("interpretation","pending")); labels[name]=label
      if name=="R": statistic=f"residual AUROC {item['primary_spline_ridge']['auroc']:.3f}"; uncertainty=f"[{item['primary_spline_ridge']['ci95']['low']:.3f}, {item['primary_spline_ridge']['ci95']['high']:.3f}]"
      elif name=="G": statistic=f"whitened cosine {item['locked']['cosines']['whitened']:.3f}"; uncertainty="covariance and full-label permutation nulls"
      elif name=="L": statistic=f"best Unknown rank {item['rows'][56]['best_unknown_rank']}"; uncertainty="99 random controls"
      elif name=="C": statistic="mean Δ log-softmax for Unknown"; uncertainty="99 random controls"
      elif name=="A": statistic=f"AUROC drop {item['after']['r_hat']['drop']:.3f}"; uncertainty=f"pair-bootstrap [{item['r_hat_drop_ci95'][0]:.3f}, {item['r_hat_drop_ci95'][1]:.3f}] and random-removal null"
      else: statistic=""; uncertainty=""
      summary.append(f"| {name} | {statistic} | {uncertainty} | {label} |")
    summary += ["","Combined verdict: "+payload["combined_verdict"]["verdict"],"","The static lens omits RMSNorm's positive scalar denominator; at (11, -4) it is three suffix positions and sixteen layers from the output readout, so Part C is decisive.",""]
    for fig in ("e2_R_residual.png","e2_R_stratified.png","e2_G_cosine_heatmaps.png","e2_G_null.png","e2_L_unknown_rank.png","e2_A_amnesic.png"):
      summary.append(f"- [{fig}](../../figures/exploratory/{fig})")
    (target/"e2_summary.md").write_text("\n".join(summary),encoding="utf-8"); print(f"Wrote {target/'e2_results.json'}")
    return 0
if __name__=="__main__": raise SystemExit(main())
