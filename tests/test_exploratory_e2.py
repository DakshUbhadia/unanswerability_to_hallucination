import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.exploratory.e2_analysis import (crossfit_residual, fit_logistic_pattern,
    cosine_spaces, static_lens, amnesic_auroc)


def test_spline_residualization_removes_nonlinear_confounds_and_keeps_added_signal():
    rng=np.random.default_rng(301); n=1200; x=rng.uniform(-2,2,n); y=(x+rng.normal(0,.7,n)>0).astype(int)
    groups=np.arange(n).astype(str); nonlinear=x**3
    confound=crossfit_residual(nonlinear+rng.normal(0,.03,n),x,y,groups,"spline_ridge",5,3,14)
    linear=crossfit_residual(nonlinear+rng.normal(0,.03,n),x,y,groups,"linear",5,3,14)
    auc=roc_auc_score(y,confound["residual"]); linear_auc=roc_auc_score(y,linear["residual"])
    assert .45 <= auc <= .55
    assert abs(linear_auc-.5) > .05  # Linear residual leaves the nonlinear confidence artifact.
    z=rng.normal(size=n); y2=(z+rng.normal(0,.7,n)>0).astype(int)
    signal=crossfit_residual(nonlinear+1.2*z+rng.normal(0,.1,n),np.column_stack([x]),y2,groups,"spline_ridge",5,3,15)
    assert roc_auc_score(y2,signal["residual"])>.6


def test_crossfit_splits_keep_groups_disjoint():
    from src.exploratory.e2_analysis import _splits
    y=np.tile([0,1],30); groups=np.repeat(np.arange(30).astype(str),2)
    for _,_,train,test in _splits(y,groups,5,2,22):
        assert not set(groups[train]) & set(groups[test])


def test_haufe_pattern_recovers_correlated_signal_axis():
    rng=np.random.default_rng(11); latent=rng.normal(size=1000); noise=rng.normal(size=(1000,3))
    X=np.column_stack([latent+noise[:,0]*.1,latent+noise[:,1]*.1,noise[:,2]])
    y=(latent+rng.normal(0,.5,1000)>0).astype(int); cov=np.cov(X,rowvar=False)
    w,pattern=fit_logistic_pattern(X,y,C=.1,cov=cov); truth=np.array([1,1,0.])
    cosine=lambda a,b: np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    assert cosine(pattern,truth)>cosine(w,truth)


def test_cosine_whitening_isotropic_and_anisotropic():
    rng=np.random.default_rng(8); X=rng.normal(size=(500,5)); u=np.array([1.,0,1,0,0]); v=np.array([0,1.,1,0,0])
    isotropic=cosine_spaces(u,v,X,[])
    assert isotropic["whitened"]==pytest.approx(isotropic["raw"],abs=.08)
    scale=np.array([100.,1,1,1,1]); anis=cosine_spaces(u*scale,v*scale,X*scale,[])
    assert anis["raw"] != pytest.approx(isotropic["raw"],abs=.05)
    assert anis["whitened"]==pytest.approx(isotropic["whitened"],abs=.08)


def test_static_lens_ranks_matching_toy_unembedding_first():
    rng=np.random.default_rng(2); g=np.array([2.,.5,1]); v=np.array([1.,-1.,.25]); U=rng.normal(size=(20,3)).astype(np.float32); U[7]=g*v
    result=static_lens(U,g,v,[7],[3],top_k=3)
    assert result["best_unknown_rank"]==1


def test_amnesic_removal_eliminates_signal_direction():
    rng=np.random.default_rng(5); n_pairs=50; y=np.tile([0,1],n_pairs); groups=np.repeat(np.arange(n_pairs).astype(str),2)
    v=np.array([1.,0,0,0]); X=rng.normal(size=(len(y),4)); X[:,0]+=y*2
    out=amnesic_auroc(X,y,groups,{"signal":v,"random":np.array([0,0,1.,0])},{"folds":5,"repeats":2,"seed":9})
    assert out["after"]["signal"]["auroc"]==pytest.approx(.5,abs=.1)
    assert out["after"]["random"]["auroc"]>.8


def test_stage6_steering_kwargs_match_tiny_model_logits():
    torch=pytest.importorskip("torch")
    from tests._tiny_model import tiny_model,tiny_contract,StubTokenizer
    from scripts.exploratory.e2_causal_vocab import stage6_steer_kwargs
    from src.eval_common import build_chat_batches,evaluate_condition
    from src.model_utils import forward_with_native_hooks
    tok=StubTokenizer(); model=tiny_model(n_layers=12,d_model=32,seed=81); contract=tiny_contract(tok,12,32)
    batch=build_chat_batches(tok,["question one","question two"],contract["system_prompt_used"],contract["suffix_token_ids"],2)
    vec=torch.randn(32); kwargs=stage6_steer_kwargs({"learned_diff":vec},.5)
    actual=evaluate_condition(model,batch,contract,kwargs["mode"],kwargs["vec"],kwargs["target_layer"],kwargs["alpha"],return_logits=True)["logits"]
    expected=np.concatenate([forward_with_native_hooks(model,b,contract["is_layer_output_tuple"],
        kwargs["mode"],kwargs["vec"],kwargs["target_layer"],kwargs["alpha"]).float().numpy() for b in batch])
    assert np.allclose(actual,expected,atol=1e-6,rtol=0)


def test_dry_run_rejects_non_r_part():
    from scripts.exploratory.e2_analyze import main
    with pytest.raises(ValueError,match="only Part R"):
        main(["--dry-run","--parts","R,G"])


def test_e1_oof_artifact_reproduces_nested_auc(tmp_path):
    from scripts.exploratory.e1_analyze import _save_oof
    y=np.tile([0,1],20); scores=np.stack([y*.8,y*.7]); folds=np.tile(np.arange(5).repeat(8),(2,1))
    path=tmp_path/"oof.npz"; expected=_save_oof(path,[f"id{i}" for i in range(len(y))],scores,y,folds)
    saved=np.load(path); assert expected==pytest.approx(roc_auc_score(saved["y"],saved["oof_score"]),abs=1e-9)
