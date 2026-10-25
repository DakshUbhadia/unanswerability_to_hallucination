"""Interpretable perturbation scales for residual-stream steering."""
import numpy as np

def residual_norm_stats(norms_all_positions,norms_at_istar,bos_norms) -> dict:
    def summary(x):
        a=np.asarray(x,float).reshape(-1)
        if not len(a) or not np.isfinite(a).all(): raise ValueError("norm arrays must be finite and nonempty")
        q=np.percentile(a,[25,50,75]);return {"median":float(q[1]),"q25":float(q[0]),"q75":float(q[2]),"iqr":float(q[2]-q[0]),"n":len(a)}
    return {"non_bos":summary(norms_all_positions),"i_star":summary(norms_at_istar),"bos":summary(bos_norms)}

def alpha_scale_table(alphas,r_norm,stats) -> list[dict]:
    a=float(stats["non_bos"]["median"]); b=float(stats["i_star"]["median"])
    if a<=0 or b<=0: raise ValueError("median residual norms must be positive")
    return [{"alpha":float(x),"alpha_r_norm":float(x)*float(r_norm),"fraction_median_non_bos":float(x)*float(r_norm)/a,
             "fraction_median_i_star":float(x)*float(r_norm)/b} for x in alphas]
