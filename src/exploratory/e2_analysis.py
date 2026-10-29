"""Statistical helpers for the five pre-registered E2 analyses (CPU only)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler


def _auc(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else 0.5


def _splits(y, groups, folds, repeats, seed):
    y, groups = np.asarray(y, int), np.asarray(groups).astype(str)
    n = min(int(folds), *(len(np.unique(groups[y == k])) for k in (0, 1)))
    if n < 2:
        raise ValueError("Grouped cross-fitting requires two groups per class")
    for repeat in range(int(repeats)):
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=int(seed) + repeat)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            if set(groups[tr]) & set(groups[te]):
                raise AssertionError("A group crossed residualization folds")
            yield repeat, fold, tr, te


def _fit_regressor(x, y, kind):
    x = np.asarray(x, float)
    if kind in ("spline_ridge", "spline"):
        transformer = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        z = transformer.fit_transform(x)
        model = RidgeCV(alphas=[0.1, 1, 10]).fit(z, y)
        return transformer, model
    if kind in ("linear", "ols"):
        return None, LinearRegression().fit(x, y)
    raise ValueError(f"Unknown residual regressor: {kind}")


def _predict_regressor(fitted, x):
    transformer, model = fitted
    return model.predict(transformer.transform(x) if transformer is not None else x)


def crossfit_residual(s, confounds, y, groups, regressor="spline_ridge", folds=5, repeats=5, seed=20260414):
    s, x, y, groups = np.asarray(s, float), np.asarray(confounds, float), np.asarray(y, int), np.asarray(groups).astype(str)
    if x.ndim == 1:
        x = x[:, None]
    if len(s) != len(y) or len(s) != len(x) or not np.isfinite(s).all() or not np.isfinite(x).all():
        raise ValueError("Residualization inputs must be finite, aligned arrays")
    residual = np.full((int(repeats), len(y)), np.nan)
    pred = np.full_like(residual, np.nan)
    fold_map = np.full((int(repeats), len(y)), -1, dtype=np.int16)
    for repeat, fold, train, test in _splits(y, groups, folds, repeats, seed):
        fitted = _fit_regressor(x[train], s[train], regressor)
        pred[repeat, test] = _predict_regressor(fitted, x[test])
        residual[repeat, test] = s[test] - pred[repeat, test]
        fold_map[repeat, test] = fold
    if not np.isfinite(residual).all():
        raise RuntimeError("Cross-fitted residual scores are incomplete")
    mean_resid, mean_pred = residual.mean(0), pred.mean(0)
    r2 = 1 - np.sum((s - mean_pred) ** 2) / max(np.sum((s - s.mean()) ** 2), 1e-15)
    return {"residual": mean_resid, "prediction": mean_pred, "oof_by_repeat": residual,
            "fold_assignments": fold_map, "auroc": _auc(y, mean_resid), "crossfit_r2": float(r2),
            "regressor": regressor}


def _cluster_boot_auc(y, score, groups, n_boot, seed):
    y, score, groups = np.asarray(y, int), np.asarray(score, float), np.asarray(groups).astype(str)
    unique = np.unique(groups)
    ix = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([ix[g] for g in chosen])
        if len(np.unique(y[idx])) == 2:
            vals.append(_auc(y[idx], score[idx]))
    if not vals:
        raise ValueError("No valid cluster bootstrap samples")
    return {"low": float(np.percentile(vals, 2.5)), "high": float(np.percentile(vals, 97.5)), "n_valid": len(vals)}


def stratified_confidence_auroc(s, y, log_odds, pair_ids, n_boot=1000, seed=20260407):
    s, y, conf, groups = np.asarray(s, float), np.asarray(y, int), np.asarray(log_odds, float), np.asarray(pair_ids).astype(str)
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, 6)))
    strata = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, max(0, len(edges)-2))
    per = []
    for q in np.unique(strata):
        idx = np.flatnonzero(strata == q)
        per.append({"quintile": int(q+1), "n": len(idx), "n_wrong": int(y[idx].sum()),
                    "auroc": _auc(y[idx], s[idx]) if len(np.unique(y[idx])) == 2 else None})
    # Weighted within-stratum concordance; pair bootstrap keeps matched observations together.
    valid = [r for r in per if r["auroc"] is not None]
    score = sum(r["auroc"] * r["n"] for r in valid) / sum(r["n"] for r in valid) if valid else 0.5
    rng = np.random.default_rng(seed); unique = np.unique(groups); ix = {g: np.flatnonzero(groups == g) for g in unique}; boot=[]
    for _ in range(int(n_boot)):
        ids = np.concatenate([ix[g] for g in rng.choice(unique, len(unique), replace=True)])
        vals=[]
        for q in np.unique(strata[ids]):
            sub=ids[strata[ids] == q]
            if len(np.unique(y[sub])) == 2: vals.append((_auc(y[sub], s[sub]), len(sub)))
        if vals: boot.append(sum(a*n for a,n in vals)/sum(n for _,n in vals))
    return {"auroc": float(score), "ci95": [float(x) for x in np.percentile(boot, [2.5,97.5])] if boot else None,
            "strata": per, "quintile_edges": edges.tolist(), "bootstrap_n": len(boot)}


def partial_spearman(s, y, confounds, n_boot=1000, seed=20260407, groups=None):
    s, y, x = np.asarray(s, float), np.asarray(y, float), np.asarray(confounds, float)
    if x.ndim == 1: x = x[:,None]
    def calc(ix):
        a, b = rankdata(s[ix]), rankdata(y[ix]); z = np.column_stack([rankdata(x[ix,j]) for j in range(x.shape[1])])
        ar = a - LinearRegression().fit(z,a).predict(z); br = b - LinearRegression().fit(z,b).predict(z)
        return float(spearmanr(ar, br).statistic)
    value = calc(np.arange(len(y)))
    if groups is None: groups=np.arange(len(y)).astype(str)
    groups=np.asarray(groups).astype(str); unique=np.unique(groups); ixmap={g:np.flatnonzero(groups==g) for g in unique}
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(int(n_boot)):
        ix=np.concatenate([ixmap[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(ix)>2: vals.append(calc(ix))
    return {"rho": value, "ci95": [float(v) for v in np.percentile(vals,[2.5,97.5])] if vals else None,
            "bootstrap_n":len(vals)}


def fit_dom(X, y):
    X, y = np.asarray(X, float), np.asarray(y, int)
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def fit_logistic_pattern(X, y, C=0.1, cov=None):
    X, y = np.asarray(X,float), np.asarray(y,int); scaler=StandardScaler().fit(X); z=scaler.transform(X)
    model=LogisticRegression(C=float(C),solver="lbfgs",max_iter=3000).fit(z,y)
    w=model.coef_[0]/np.where(scaler.scale_==0,1,scaler.scale_)
    if cov is None:
        from src.null_directions import fit_activation_covariance
        cov=fit_activation_covariance(X)["covariance"]
    return w, np.asarray(cov,float) @ w


def _cos(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else float("nan")


def _scale_equivariant_lw_covariance(X):
    """Ledoit-Wolf correlation shrinkage mapped back to units; preserves diagonal scale changes."""
    from src.null_directions import fit_activation_covariance
    X=np.asarray(X,dtype=np.float64); scale=np.std(X,axis=0,ddof=1); scale=np.where(scale>1e-15,scale,1.)
    standardized=(X-X.mean(0))/scale
    cov_std=fit_activation_covariance(standardized)["covariance"]
    return cov_std*scale[:,None]*scale[None,:]


def cosine_spaces(u, v, X_for_cov, outlier_dims):
    from src.null_directions import fit_activation_covariance
    u,v=np.asarray(u,float),np.asarray(v,float); X=np.asarray(X_for_cov,float)
    keep=np.ones(X.shape[1],bool); keep[np.asarray(outlier_dims,int)]=False
    cov=_scale_equivariant_lw_covariance(X)
    vals,vecs=np.linalg.eigh((cov+cov.T)/2); floor=1e-6*np.trace(cov)/cov.shape[0]
    invroot=(vecs*(1/np.sqrt(np.maximum(vals,floor))))@vecs.T
    out={"raw":_cos(u,v),"outlier_removed":_cos(u[keep],v[keep]),"whitened":_cos(invroot@u,invroot@v),
         "outlier_dims":np.asarray(outlier_dims,int).tolist()}
    if not np.isfinite(list(out[k] for k in ("raw","outlier_removed","whitened"))).all():
        raise RuntimeError("Whitening or cosine produced non-finite values")
    return out


def geometry_nulls(rhat, X, y, cov, n_cov=499, n_perm=199, seed=20260410):
    from src.null_directions import sample_covariance_null
    cov=_scale_equivariant_lw_covariance(X)
    vals,vecs=np.linalg.eigh((np.asarray(cov,float)+np.asarray(cov,float).T)/2)
    floor=1e-6*np.trace(cov)/len(cov); invroot=(vecs*(1/np.sqrt(np.maximum(vals,floor))))@vecs.T
    observed=fit_dom(X,y); cos_obs=_cos(invroot@rhat,invroot@observed)
    null=sample_covariance_null(cov,n_cov,seed)["unit"]
    cov_dist=np.asarray([_cos(invroot@v,invroot@observed) for v in null])
    rng=np.random.default_rng(seed+1); perm=[]
    for yp in (rng.permutation(y) for _ in range(int(n_perm))): perm.append(_cos(invroot@rhat,invroot@fit_dom(X,yp)))
    # Item-bootstrap refits dom and uses fixed rhat.
    vals=[]; groups=np.arange(len(y));
    for _ in range(1000):
        ix=rng.choice(groups,len(groups),replace=True)
        if len(np.unique(y[ix]))==2: vals.append(_cos(invroot@rhat,invroot@fit_dom(X[ix],y[ix])))
    return {"observed_cosine":cos_obs,"covariance_null":cov_dist.tolist(),"permutation_null":perm,
            "covariance_null_abs_p":float((1+np.sum(np.abs(cov_dist)>=abs(cos_obs)))/(len(cov_dist)+1)),
            "permutation_null_abs_p":float((1+np.sum(np.abs(perm)>=abs(cos_obs)))/(len(perm)+1)),
            "bootstrap_ci95":[float(v) for v in np.percentile(vals,[2.5,97.5])] if vals else None}


def load_unembedding_and_norm(model_name, revision=None):
    """Fetch only final RMSNorm and output embedding tensors using safetensors."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(model_name,revision=revision); rev=revision or getattr(config,"_commit_hash",None)
    norm_key="model.norm.weight"; out_key="model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    needed={norm_key,out_key}; got={}
    try:
        index_path=hf_hub_download(model_name,"model.safetensors.index.json",revision=rev)
    except Exception:
        index_path=None
    if index_path:
        import json
        index=json.loads(open(index_path,encoding="utf-8").read()); wm=index["weight_map"]
        for shard in sorted({wm[k] for k in needed}):
            path=hf_hub_download(model_name,shard,revision=rev)
            with safe_open(path,framework="np",device="cpu") as f:
                for key in needed:
                    if key in f.keys(): got[key]=f.get_tensor(key)
    else:
        path=hf_hub_download(model_name,"model.safetensors",revision=rev)
        with safe_open(path,framework="np",device="cpu") as f:
            for key in needed: got[key]=f.get_tensor(key)
    return np.asarray(got[out_key]),np.asarray(got[norm_key]),{"revision":rev,"norm_key":norm_key,"unembedding_key":out_key}


def static_lens(U, g, vec, unknown_ids, extra_ids=(), top_k=20, chunk=16384, tokenizer=None):
    U,g,vec=np.asarray(U),np.asarray(g),np.asarray(vec); direction=(g*vec).astype(np.float32)
    logits=np.empty(U.shape[0],np.float32)
    for start in range(0,len(logits),int(chunk)): logits[start:start+chunk]=np.asarray(U[start:start+chunk],np.float32)@direction
    order=np.argsort(logits); best={str(int(t)):int(np.sum(logits>logits[int(t)])+1) for t in unknown_ids}
    def token(i): return tokenizer.decode([int(i)]) if tokenizer is not None else str(int(i))
    z={str(int(t)):float((logits[int(t)]-logits.mean())/(logits.std()+1e-12)) for t in unknown_ids}
    return {"unknown_token_ranks":best,"best_unknown_rank":min(best.values()),"unknown_zscores":z,
            "promoted_top": [{"id":int(i),"token":token(i),"score":float(logits[i])} for i in order[-top_k:][::-1]],
            "demoted_top": [{"id":int(i),"token":token(i),"score":float(logits[i])} for i in order[:top_k]],
            "extra_token_ranks":{str(int(t)):int(np.sum(logits>logits[int(t)])+1) for t in extra_ids}}


def amnesic_auroc(X,y,groups,removal_vecs,e1_cv_kwargs=None):
    from src.exploratory.analysis import nested_group_probe_oof
    kwargs=e1_cv_kwargs or {}; folds=kwargs.get("folds",5); repeats=kwargs.get("repeats",5); seed=kwargs.get("seed",20260414)
    X,y=np.asarray(X,float),np.asarray(y,int)
    before=nested_group_probe_oof(X,y,groups,folds,repeats,seed)[0].mean(0); result={"before":_auc(y,before),"before_scores":before,"after":{}}
    for name,v in removal_vecs.items():
        v=np.asarray(v,float); v=v/np.linalg.norm(v); removed=X-(X@v)[:,None]*v[None,:]
        pred=nested_group_probe_oof(removed,y,groups,folds,repeats,seed)[0].mean(0)
        result["after"][name]={"auroc":_auc(y,pred),"drop":result["before"]-_auc(y,pred),"scores":pred}
    return result


def label_part_R(ci):
    lo,hi=map(float,ci)
    if (lo>.5 or hi<.5) and lo>=.45 and hi<=.55: return "R: tiny_residual_below_margin"
    if lo>=.45 and hi<=.55: return "R: residual_null_equivalent"
    if lo>.55: return "R: residual_signal_present"
    return "R: inconclusive"


def label_part_G(observed, cov_null, perm_null):
    a,b=np.percentile(np.abs(cov_null),[95,99]); c,d=np.percentile(np.abs(perm_null),[95,99]); x=abs(float(observed))
    if x>d and x>b:return "G: aligned"
    if x<=a and x<=c:return "G: not_aligned"
    return "G: weak_or_inconclusive"


def label_part_L(best_rank, random_ranks):
    return "L: unknown_aligned" if best_rank < np.percentile(random_ranks,1) else "L: not_aligned"


def label_part_A(drop, ci, random_drops):
    if drop>np.percentile(random_drops,99) and ci[0]>0:return "A: r_hat_carries_popqa_info"
    if drop<=np.percentile(random_drops,95):return "A: r_hat_removal_irrelevant"
    return "A: inconclusive"


def combined_verdict(labels, e1_c1=None, e1_c2=None):
    labels=dict(labels)
    if labels.get("R")=="R: residual_null_equivalent" and labels.get("C")=="C: promotes_unknown_specifically":
        verdict="(b) abstention-intent readout: r_hat's PopQA separation is explained by output confidence, and r_hat causally promotes 'Unknown'"
    elif labels.get("R")=="R: residual_signal_present" and labels.get("G")=="G: aligned":
        verdict="(a) shared knowledge-state component: r_hat carries PopQA wrong-vs-correct information beyond output confidence and aligns with PopQA error directions"
    elif (labels.get("R")=="R: residual_null_equivalent" and labels.get("G")=="G: not_aligned" and
          e1_c1=="knowledge_state_signal_beyond_output_confidence"):
        verdict="(c) separate representations: knowledge-state information exists (E1) but is not along r_hat"
    else: verdict="mixed/inconclusive"
    rendered="; ".join(f"{k}={v}" for k,v in labels.items())
    return {"labels":labels,"e1_C1":e1_c1,"e1_C2":e1_c2,"verdict":f"{verdict}; labels: {rendered}; E1 C1={e1_c1}; E1 C2={e1_c2}"}
