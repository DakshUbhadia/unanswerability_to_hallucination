"""CPU analysis for preregistered Stage 8 v5 metrics."""
from __future__ import annotations
import numpy as np
from src.stratified_auroc import stratified_auroc, stratified_permutation_test, cluster_bootstrap_stratified_auroc
from src.incremental_value import incremental_value_test
from src.intersection_union import iut_p
from src.config import CFG


def classify_interpretation_v5(holm_rows, h1_pass, h2_pass, h3_pass, concept_confounded,
                               h4_pass, h5_pass, h5_ci_upper, stratified_auroc, abstain_vs_correct,
                               overall_v4=None, overall_v41=None):
    """Apply the six ordered v5 decisions and retain the earlier verdicts."""
    if not (h1_pass and h2_pass and h3_pass) or concept_confounded:
        verdict="Inconclusive: prerequisites not met"
    elif abs(float(stratified_auroc)-0.5)<=0.05 and float(abstain_vs_correct)<=0.5:
        verdict="Inconclusive: domain-transfer failure"
    elif h4_pass and h5_pass:
        verdict="Hypothesis B (internal uncertainty)"
    elif not h5_pass and float(h5_ci_upper)<CFG.sesoi_delta_auroc:
        verdict="Hypothesis A: output-confidence readout" if h4_pass else "Hypothesis A: input-property detector"
    elif not h5_pass and float(h5_ci_upper)>=CFG.sesoi_delta_auroc:
        verdict="Inconclusive: underpowered for incremental value"
    elif not h4_pass and h5_pass:
        verdict="Mixed"
    else:
        verdict="Inconclusive: prerequisites not met"
    return {"overall_v4":overall_v4,"overall_v41":overall_v41,"overall_v5":verdict,
            "rule_version":"v5-preregistered","prerequisites_met":bool(h1_pass and h2_pass and h3_pass and not concept_confounded),
            "concept_confound_flag":bool(concept_confounded),"h4_pass":bool(h4_pass),"h5_pass":bool(h5_pass),
            "h5_ci_upper":float(h5_ci_upper),"sesoi":float(CFG.sesoi_delta_auroc),
            "holm_rows":holm_rows}


def analyze_h4_v5(scores, labels, strata, pair_ids, covariance_scores=None, template_partition_scores=None,
                  label_free_scores=None, n_perm=CFG.n_perm_stratified, seed=CFG.seed_popqa_fresh, n_boot=1000):
    """Compute registered H4 point estimate, permutation, bootstrap, and null IUT."""
    observed=stratified_auroc(scores,labels,strata)
    perm=stratified_permutation_test(scores,labels,strata,n_perm,seed,"greater")
    boot=cluster_bootstrap_stratified_auroc(scores,labels,strata,pair_ids,n_boot,seed+1)
    y=np.asarray(labels,int);g=np.asarray(strata);s=np.asarray(scores,float)
    def auc_for(z): return stratified_auroc(s*y*0 + z,y,g)["auroc"]
    # Null arrays contain [M,N] projected scores for the same fresh rows.
    def null_p(array):
        if array is None: raise ValueError("registered null score arrays are required")
        vals=np.array([auc_for(row) for row in np.asarray(array)])
        return {"p_value":float((1+np.sum(np.abs(vals-.5)>=abs(observed["auroc"]-.5)))/(len(vals)+1)),
                "n":int(len(vals)),"mean_auroc":float(vals.mean()),"sd_auroc":float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                "aurocs":vals.tolist()}
    cov=null_p(covariance_scores);tp=null_p(template_partition_scores)
    components={"stratified_permutation":perm["p_value"],"covariance_null":cov["p_value"],"template_partition_null":tp["p_value"]}
    return {"stratified":observed,"permutation":perm,"pair_cluster_bootstrap":boot,"covariance_null":cov,
            "template_partition_null":tp,"label_free_null":null_p(label_free_scores) if label_free_scores is not None else None,
            "iut":iut_p(components),"point_criterion":bool(observed["auroc"]>=.60)}


def analyze_h5_v5(frame, n_boot=1000, seed=CFG.seed_incremental_cv):
    return incremental_value_test(frame,n_boot,seed,CFG.sesoi_delta_auroc)
