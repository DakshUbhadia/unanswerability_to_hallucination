"""Pre-registered repeated group-CV incremental-value test."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.config import CFG
BASE_FEATURES=["maxp","entropy","log_odds"]
FULL_FEATURES=BASE_FEATURES+["proj"]


def cv_oof_auroc(X,y,groups,folds=5,repeats=20,seed=0):
    X=np.asarray(X,float); y=np.asarray(y,int); groups=np.asarray(groups)
    if X.ndim!=2 or len(X)!=len(y) or len(y)!=len(groups): raise ValueError("unaligned inputs")
    preds=np.zeros((int(repeats),len(y)),float); seen=np.zeros_like(preds,dtype=bool)
    for rep in range(int(repeats)):
        cv=StratifiedGroupKFold(n_splits=int(folds),shuffle=True,random_state=int(seed)+rep)
        for tr,te in cv.split(X,y,groups):
            if set(groups[tr]) & set(groups[te]): raise AssertionError("a pair was split across folds")
            scaler=StandardScaler().fit(X[tr]); model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000)
            model.fit(scaler.transform(X[tr]),y[tr]); preds[rep,te]=model.predict_proba(scaler.transform(X[te]))[:,1];seen[rep,te]=True
    if not seen.all(): raise RuntimeError("OOF scores missing")
    return float(roc_auc_score(y,preds.mean(0)))


def incremental_value_test(df,n_boot=1000,seed=0,sesoi=CFG.sesoi_delta_auroc):
    import pandas as pd
    if not isinstance(df,pd.DataFrame): df=pd.DataFrame(df)
    req=set(FULL_FEATURES+['y','matched_pair_id'])
    if not req.issubset(df.columns): raise ValueError(f"missing fields: {sorted(req-set(df.columns))}")
    y=df.y.to_numpy(int); groups=df.matched_pair_id.to_numpy()
    def calc(d, sd, repeats=CFG.n_cv_repeats):
        yy=d.y.to_numpy(int); gg=d.matched_pair_id.to_numpy()
        b=cv_oof_auroc(d[BASE_FEATURES].to_numpy(float),yy,gg,CFG.n_cv_folds,repeats,sd)
        f=cv_oof_auroc(d[FULL_FEATURES].to_numpy(float),yy,gg,CFG.n_cv_folds,repeats,sd)
        return f-b
    delta=calc(df,int(seed)); uniq=np.unique(groups); rng=np.random.default_rng(seed); boots=[]
    for b in range(int(n_boot)):
        picked=rng.choice(uniq,len(uniq),replace=True); chunks=[]
        for j,p in enumerate(picked):
            part=df[df.matched_pair_id==p].copy();part['matched_pair_id']=f"{j}:{p}";chunks.append(part)
        sample=pd.concat(chunks,ignore_index=True)
        try: boots.append(calc(sample,int(seed)+b+1,repeats=1))
        except ValueError: continue
    if not boots: raise ValueError("no valid pair bootstrap replicates")
    lo,hi=np.percentile(boots,[2.5,97.5]); p=(1+sum(x<=0 for x in boots))/(len(boots)+1)
    return {"delta_auroc_cv":float(delta),"ci95_low":float(lo),"ci95_high":float(hi),"p_one_sided":float(p),
            "n_boot":len(boots),"sesoi":float(sesoi),"equivalent_below_sesoi":bool(hi<sesoi),"seed":int(seed)}
