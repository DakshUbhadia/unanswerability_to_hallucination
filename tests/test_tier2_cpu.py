import numpy as np
from src.null_directions import (fit_activation_covariance,sample_covariance_null,labelfree_null,template_partition_null)
from src.stratified_auroc import stratified_auroc,stratified_permutation_test
from src.intersection_union import iut_p
from src.steering_scale import residual_norm_stats,alpha_scale_table
from src.stage8_v5 import classify_interpretation_v5


def test_covariance_null_raw_covariance_and_norms():
    cov=np.array([[2.,.7],[.7,1.]])
    out=sample_covariance_null(cov,20000,7,r_norm=3.2,r_hat=np.array([1.,0.]))
    empirical=np.cov(out['raw_samples'].T);assert np.linalg.norm(empirical-cov,'fro')/np.linalg.norm(cov,'fro')<.05
    assert np.max(np.abs(np.linalg.norm(out['steer'],axis=1)-3.2))<1e-5
    fit=fit_activation_covariance(np.random.default_rng(2).normal(size=(100,4)))
    assert fit['covariance'].shape==(4,4) and 0<=fit['shrinkage']<=1


def test_label_free_and_template_partition_are_orthogonal_and_label_balanced():
    rng=np.random.default_rng(4);t,d=48,256;r=np.zeros(d);r[0]=1
    base=rng.normal(size=(t,1,d));noise=rng.normal(scale=.2,size=(t,4,d))
    ans=base+.7*r+noise;unans=base-.7*r+noise
    lf=labelfree_null(ans,unans,n=199,seed=3)
    assert np.mean(np.abs(lf['unit']@r))<.1
    fam=['A']*24+['B']*24
    tp=template_partition_null(ans,unans,None,fam,r,2.,199,5)
    assert np.max(np.abs(tp['unit']@r))<1e-6
    assert np.max(np.abs(np.linalg.norm(tp['steer'],axis=1)-2.))<1e-5
    tid_family={i:('A' if i<24 else 'B') for i in range(48)}
    assert tp['metadata']['label_balance_per_group']=={'answerable':96,'unanswerable':96}
    for part in tp['metadata']['partitions']:
        assert len(part['group1'])==len(part['group2'])==24
        assert sum(tid_family[t]=='A' for t in part['group1'])==12
        assert sum(tid_family[t]=='B' for t in part['group1'])==12
        assert sum(tid_family[t]=='A' for t in part['group2'])==12
        assert sum(tid_family[t]=='B' for t in part['group2'])==12


def test_within_pair_sign_flip_retains_axis_while_label_free_does_not():
    rng=np.random.default_rng(31);n=48;d=256;r=np.zeros(d);r[0]=1.
    diffs=2*r+rng.normal(scale=.05,size=(n,d))
    signs=rng.choice([-1.,1.],size=n)
    old=np.mean(diffs*signs[:,None],axis=0);old/=np.linalg.norm(old)
    ans=np.broadcast_to((rng.normal(size=(n,1,d))+.5*r),(n,4,d)).copy()+rng.normal(scale=.01,size=(n,4,d))
    unans=np.broadcast_to((rng.normal(size=(n,1,d))-.5*r),(n,4,d)).copy()+rng.normal(scale=.01,size=(n,4,d))
    new=labelfree_null(ans,unans,n=199,seed=8)
    assert abs(float(old@r))>.3
    assert np.mean(np.abs(new['unit']@r))<.1


def test_stratified_auc_single_stratum_and_weighted():
    s=np.array([.1,.4,.35,.8,.2,.9]);y=np.array([0,1,0,1,0,1]);g=np.array(['a']*6)
    assert stratified_auroc(s,y,g)['auroc']==1.0
    g=np.array(['a','a','a','a','b','b']);y=np.array([0,1,0,1,0,1]);s=np.array([0,1,1,0,0,1])
    out=stratified_auroc(s,y,g);assert np.isclose(out['auroc'],.6)


def test_iut_binding_and_steering_scale():
    assert iut_p({'a':.01,'b':.04})=={'components':{'a':.01,'b':.04},'p_composite':.04,'binding_component':'b'}
    stats=residual_norm_stats([4,6,8],[2,4,6],[1,1])
    row=alpha_scale_table([1],2,stats)[0];assert row['alpha_r_norm']==2 and np.isclose(row['fraction_median_non_bos'],1/3)


def test_interpretation_v5_ordered_rows():
    base=dict(holm_rows=[],h1_pass=True,h2_pass=True,h3_pass=True,concept_confounded=False,
      h4_pass=True,h5_pass=True,h5_ci_upper=.03,stratified_auroc=.7,abstain_vs_correct=.6)
    assert classify_interpretation_v5(**base)['overall_v5'].startswith('Hypothesis B')
    assert classify_interpretation_v5(**{**base,'concept_confounded':True})['overall_v5'].startswith('Inconclusive: prerequisites')
    assert classify_interpretation_v5(**{**base,'h5_pass':False,'h5_ci_upper':.01})['overall_v5'].endswith('output-confidence readout')
    assert classify_interpretation_v5(**{**base,'h4_pass':False,'h5_pass':False,'h5_ci_upper':.04})['overall_v5'].startswith('Inconclusive: underpowered')
    assert classify_interpretation_v5(**{**base,'h4_pass':False,'h5_pass':True})['overall_v5']=='Mixed'
    assert classify_interpretation_v5(**{**base,'stratified_auroc':.51,'abstain_vs_correct':.5})['overall_v5'].startswith('Inconclusive: domain')


def test_named_concept_attribute_derivation_is_deterministic_and_balanced():
    from src.concept_attributes import derive_attribute_rows, make_contrast_directions
    rows1=derive_attribute_rows();rows2=derive_attribute_rows()
    assert rows1==rows2 and len(rows1)==384
    assert {r['label'] for r in rows1}=={'answerable','unanswerable'}
    rng=np.random.default_rng(8);acts=rng.normal(size=(len(rows1),16));rh=np.zeros(16);rh[0]=1
    contrasts=make_contrast_directions(acts,rows1,rh,3.)
    family=[x for x in contrasts if x['attribute']=='family']
    assert len(family)==2 and all(min(v.values())>=32 for v in family[0]['group_sizes'].values())
    assert all(np.isfinite(x['pre_orthogonalization_cosine']) for x in contrasts)


def test_between_stratum_effect_is_removed():
    from src.stratified_auroc import stratified_auroc
    # Labels are separated only because positive examples are concentrated in high-score relation b.
    scores=np.array([.1]*100+[.9]*100);labels=np.array([1]*10+[0]*90+[1]*90+[0]*10)
    strata=np.array(['a']*100+['b']*100)
    assert stratified_auroc(scores,labels,strata)['auroc']==.5
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(labels,scores)>.8


def test_stratified_permutation_type_one_rate_on_400_null_datasets():
    rng=np.random.default_rng(72);rejects=0
    strata=np.repeat(np.arange(10),8);labels=np.tile([0,0,0,0,1,1,1,1],10)
    for i in range(400):
        scores=rng.normal(size=len(labels))
        result=stratified_permutation_test(scores,labels,strata,n_perm=199,seed=1000+i)
        rejects+=result['p_value']<.05
    rate=rejects/400
    assert .03<=rate<=.07


def test_group_cv_keeps_matched_pairs_together():
    from src.incremental_value import cv_oof_auroc
    rng=np.random.default_rng(10);groups=np.repeat(np.arange(30),2);y=np.tile([1,0],30)
    X=np.column_stack([y+rng.normal(0,.5,len(y)),rng.normal(size=len(y))])
    auc=cv_oof_auroc(X,y,groups,folds=5,repeats=2,seed=3)
    assert .5<auc<=1


def test_incremental_value_noise_and_independent_signal_are_separated():
    from src.incremental_value import cv_oof_auroc
    rng=np.random.default_rng(44);groups=np.repeat(np.arange(120),2);y=np.tile([1,0],120)
    base=np.column_stack([rng.normal(size=240),rng.normal(size=240),rng.normal(size=240)])
    noise=np.column_stack([base,rng.normal(size=240)])
    signal=np.column_stack([base,y+rng.normal(scale=.3,size=240)])
    auc_noise=cv_oof_auroc(noise,y,groups,folds=5,repeats=3,seed=5)
    auc_signal=cv_oof_auroc(signal,y,groups,folds=5,repeats=3,seed=5)
    assert .4<auc_noise<.6
    assert auc_signal>auc_noise+.2


def test_archive_v42_manifest_verifies():
    import json
    from pathlib import Path
    from scripts.verify_archive_v42 import verify
    normalized = verify()
    manifest = json.loads(Path("results/archive_v42/MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 55
    assert set(normalized).issubset(manifest["files"])
