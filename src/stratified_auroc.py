"""Stratum-aware AUROC and matched-pair bootstrap utilities."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def stratified_auroc(scores, labels, strata) -> dict:
    s=np.asarray(scores,float); y=np.asarray(labels,int); g=np.asarray(strata)
    if not (len(s)==len(y)==len(g)): raise ValueError("scores, labels, strata must align")
    pieces=[]; dropped=0
    for name in dict.fromkeys(g.tolist()):
        ix=g==name; pos=int(np.sum(y[ix]==1)); neg=int(np.sum(y[ix]==0))
        if not pos or not neg: dropped+=1; continue
        auc=float(roc_auc_score(y[ix],s[ix])); w=pos*neg; pieces.append((name,auc,w,pos,neg))
    if not pieces: raise ValueError("no stratum contains both classes")
    den=sum(x[2] for x in pieces)
    return {"auroc":float(sum(a*w for _,a,w,_,_ in pieces)/den),"strata_used":len(pieces),"strata_dropped":dropped,
            "strata":[{"stratum":str(n),"auroc":a,"weight":w,"n_pos":p,"n_neg":q} for n,a,w,p,q in pieces]}


def stratified_permutation_test(scores, labels, strata, n_perm=9999, seed=0, alternative="greater") -> dict:
    s=np.asarray(scores,float); y=np.asarray(labels,int); g=np.asarray(strata); obs=stratified_auroc(s,y,g)["auroc"]
    if alternative not in {"greater","less","two-sided"}: raise ValueError("invalid alternative")
    rng=np.random.default_rng(seed); count=int(n_perm);num=np.zeros(count,float);den=0
    # Within each stratum, the label count is fixed. Mann–Whitney U is a rank
    # sum, so batch permutations and avoid thousands of repeated roc_auc_score calls.
    for name in dict.fromkeys(g.tolist()):
        ix=np.flatnonzero(g==name); yy=y[ix];pos=int(yy.sum());neg=len(yy)-pos
        if not pos or not neg:continue
        ranks=rankdata(s[ix]);weight=pos*neg;den+=weight
        order=np.argsort(rng.random((count,len(ix))),axis=1)
        perm_labels=yy[order]
        num+=(ranks[None,:]*perm_labels).sum(1)-pos*(pos+1)/2
    stats=num/den
    if alternative=="greater":ge=int(np.sum(stats>=obs))
    elif alternative=="less":ge=int(np.sum(stats<=obs))
    else:ge=int(np.sum(np.abs(stats-.5)>=abs(obs-.5)))
    return {"statistic":obs,"p_value":(1+ge)/(int(n_perm)+1),"n_perm":int(n_perm),"seed":int(seed),"alternative":alternative}


def cluster_bootstrap_stratified_auroc(scores, labels, strata, pair_ids, n_boot=1000, seed=0) -> dict:
    s=np.asarray(scores,float); y=np.asarray(labels,int); g=np.asarray(strata); pairs=np.asarray(pair_ids)
    uniq=np.unique(pairs); rng=np.random.default_rng(seed); vals=[]
    for _ in range(int(n_boot)):
        sampled=rng.choice(uniq,len(uniq),replace=True)
        ix=np.concatenate([np.flatnonzero(pairs==p) for p in sampled])
        try: vals.append(stratified_auroc(s[ix],y[ix],g[ix])["auroc"])
        except ValueError: continue
    if not vals: raise ValueError("bootstrap produced no valid replicate")
    lo,hi=np.percentile(vals,[2.5,97.5]); point=stratified_auroc(s,y,g)["auroc"]
    return {"auroc":point,"ci95_low":float(lo),"ci95_high":float(hi),"n_boot_valid":len(vals),"n_boot":int(n_boot)}
