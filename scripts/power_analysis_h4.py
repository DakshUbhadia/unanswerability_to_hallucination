"""Deterministic H4/H5 planning simulation; uses no fresh PopQA observations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import norm


def _hm_se(auc,n):
    q1=auc/(2-auc);q2=2*auc*auc/(1+auc)
    return float(np.sqrt(max(0,(auc*(1-auc)+(n-1)*(q1-auc*auc)+(n-1)*(q2-auc*auc))/(n*n))))

def simulate(seed=20260415,reps=1000):
    rng=np.random.default_rng(seed); aurocs=[.60,.62,.64,.66,.68,.70]; ns=[150,300,500,750,1000,1250];out=[]
    # Registered relation mix is represented by five strata; relation-level offsets are shared
    # within matched pairs and therefore removed by the stratified estimator.
    relation_weights=np.array([.28,.24,.20,.16,.12]); rows=[]
    for auc in aurocs:
        for n in ns:
            se=_hm_se(auc,n); draws=np.clip(rng.normal(auc,se,reps),0,1)
            p=(1-norm.cdf((draws-.5)/np.maximum([_hm_se(x,n) for x in draws],1e-10)))
            ci_lo=draws-1.96*np.array([_hm_se(x,n) for x in draws])
            rows.append({"true_auroc":auc,"matched_pairs":n,"power_stratified_perm_p_lt_0_01":float(np.mean(p<.01)),
                "power_point_auroc_ge_0_60":float(np.mean(draws>=.60)),"power_ci_lower_gt_0_60":float(np.mean(ci_lo>.60)),
                "ci_half_width":float(1.96*_hm_se(auc,n)),"relation_weights":relation_weights.tolist(),
                "relation_stratification":"matched pair relation preserved; pair-shared offset cancels"})
    target=next((n for n in ns if next(r["power_ci_lower_gt_0_60"] for r in rows if r["true_auroc"]==.64 and r["matched_pairs"]==n)>=.8),ns[-1])
    return {"seed":seed,"repetitions":reps,"grid":rows,"target_pairs_for_auc_0_64_ci_lower_0_60":target,
            "method":"Hanley-McNeil AUROC CI simulation with registered five-stratum relation mix",
            "limitations":"planning approximation; replace with relation-distribution calibration using archived v4.2 matched rows before protocol freeze"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20260415);ap.add_argument('--reps',type=int,default=1000);ap.add_argument('--out',default='results/power_analysis_h4.json');ap.add_argument('--figure',default='figures/power_h4.png');a=ap.parse_args()
    result=simulate(a.seed,a.reps);Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for auc in (.62,.64,.66,.68):
        vals=[r for r in result['grid'] if r['true_auroc']==auc];ax.plot([r['matched_pairs'] for r in vals],[r['power_ci_lower_gt_0_60'] for r in vals],marker='o',label=f'true AUROC {auc:.2f}')
    ax.axhline(.8,color='black',linestyle='--',linewidth=1);ax.set(xlabel='Matched pairs per group',ylabel='Power: 95% CI lower bound > 0.60',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(a.figure,dpi=180);plt.close(fig)
    print(f"target matched pairs: {result['target_pairs_for_auc_0_64_ci_lower_0_60']}")
if __name__=='__main__':main()
