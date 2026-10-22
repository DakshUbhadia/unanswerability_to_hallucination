import json
import numpy as np
import pytest

from src.coordinate_selection import (build_sensitivity_grid_table, compute_composite_j_grid,
 load_and_validate_val_pairs, save_stage5_artifacts, select_optimal_coordinate_for_threshold,
 select_primary_coordinate_with_branch_rules, verify_stage5_outputs)


def test_load_and_validate_val_pairs_real_repo_data():
    rows=load_and_validate_val_pairs()
    assert len(rows)==96 and [x['pair_id'] for x in rows]==sorted(x['pair_id'] for x in rows)
    assert [x['family'] for x in rows].count('A')==[x['family'] for x in rows].count('B')==48


def test_compute_composite_j_grid_standardization():
    a=np.arange(140,dtype=float).reshape(28,5); s=np.flip(a,axis=1).copy()
    s[0,0]=1000; a[0,0]=-1000
    j,z=compute_composite_j_grid(a,s)
    assert np.isclose(((a-z['abl_mean'])/z['abl_std']).mean(),0,atol=1e-12)
    assert np.isclose(((a-z['abl_mean'])/z['abl_std']).std(),1)
    assert j.shape==(28,5) and j[0,0]==j.max()


def test_branch_rule3_tie_breaking_earlier_layer_then_later_position():
    j=np.zeros((28,5)); kl=np.ones_like(j)*.1
    j[10,1]=1; j[10,3]=1-5e-7; j[14,4]=1-2e-7
    assert select_optimal_coordinate_for_threshold(j,kl,.15,.85)==(10,-2,3)


def test_branch_rule2_filter_relaxation_all_four_steps():
    ab=np.zeros((28,5)); st=np.zeros_like(ab); j=np.arange(140).reshape(28,5).astype(float)
    def run(k): return select_primary_coordinate_with_branch_rules(ab,st,k,j)
    k=np.ones((28,5))*.3; k[0,0]=.1; assert run(k)['filter_relaxed'] is False
    k[:23]=.18; assert (run(k)['active_tau_kl'],run(k)['active_rho_depth'])==(.25,.85)
    k[:]=.4; k[24,0]=.22; out=run(k); assert (out['active_tau_kl'],out['active_rho_depth'])==(.25,.95)
    k[:]=.4; out=run(k); assert out['relaxation_reason']=='top10pct_j_min_kl_fallback'


def _payload():
    a=np.arange(140,dtype=float).reshape(28,5)/100; s=np.flip(a,axis=0).copy()
    j,z=compute_composite_j_grid(a,s); kl=np.ones((28,5))*.1
    prim=select_primary_coordinate_with_branch_rules(a,s,kl,j)
    tab=build_sensitivity_grid_table(a,s,kl,j,[-5,-4,-3,-2,-1],prim['l_star'],prim['i_star'])
    return {'stage':5,'model_name':'model','n_layers':28,'k_suffix':5,'d_model':3072,'suffix_indices':[-5,-4,-3,-2,-1],
      'n_val_pairs':96,'alpha_val_select':1.,**prim,'r_norm_at_star':1.,'cos_concept_pre_ortho_at_star':.2,'default_tau_kl':.15,'default_rho_depth':.85,
      'val_unans_baseline_log_odds':4.,'val_ans_baseline_log_odds':-4.,'z_score_stats':z,'ablation_scores_grid':a.tolist(),'steering_scores_grid':s.tolist(),
      'kl_val_grid':kl.tolist(),'composite_j_grid':j.tolist(),'sensitivity_table':tab,'source_stage4_sha256':'a'*64,'elapsed_sec':1.,'passed_acceptance_gate':True}


def test_build_sensitivity_grid_table_and_exports(tmp_path):
    d=_payload(); assert len(d['sensitivity_table'])==12
    cp=tmp_path/'checkpoint.pt'; cp.touch()
    save_stage5_artifacts(d,tmp_path/'out.json',tmp_path/'table.csv',tmp_path/'table.md',cp)
    assert not cp.exists(); assert len((tmp_path/'table.csv').read_text().splitlines())==13
    assert '| tau_kl |' in (tmp_path/'table.md').read_text()


def test_verify_stage5_outputs_accepts_valid_and_rejects_tampered(tmp_path):
    d=_payload(); out=tmp_path/'out.json'; csv=tmp_path/'table.csv'; md=tmp_path/'table.md'; cp=tmp_path/'cp'
    save_stage5_artifacts(d,out,csv,md,cp); verify_stage5_outputs(out,csv,md)
    bad=dict(d); bad['l_star']=bad['l_star']+1; out.write_text(json.dumps(bad));
    with pytest.raises(AssertionError): verify_stage5_outputs(out,csv,md)
    bad=dict(d); bad['kl_val_grid']=[r[:] for r in d['kl_val_grid']]; bad['kl_val_grid'][0][0]=-1; out.write_text(json.dumps(bad))
    with pytest.raises(AssertionError): verify_stage5_outputs(out,csv,md)
    save_stage5_artifacts(d,out,csv,md,cp); csv.write_text('\n'.join(csv.read_text().splitlines()[:-1])+"\n")
    with pytest.raises(AssertionError): verify_stage5_outputs(out,csv,md)


def test_stage5_cli_skips_when_already_verified(tmp_path,monkeypatch):
    d=_payload(); out=tmp_path/'out.json'; csv=tmp_path/'table.csv'; md=tmp_path/'table.md'; save_stage5_artifacts(d,out,csv,md,tmp_path/'cp')
    from scripts import stage5_select_coordinate as cli
    assert cli.main(['--output-path',str(out),'--table2-csv-path',str(csv),'--table2-md-path',str(md),
                     '--stage0-path',str(tmp_path/'missing-stage0.json'),
                     '--stage4-manifest-path',str(tmp_path/'missing-manifest.json')])==0


@pytest.mark.integration
def test_stage5_real_artifacts_verified():
    from pathlib import Path
    out=Path('results/stage5_selection.json')
    if not out.exists(): pytest.skip('Stage 5 runtime artifact is not present')
    d=verify_stage5_outputs(out,'tables/table2_sensitivity_grid.csv','tables/table2_sensitivity_grid.md')
    assert d['selected_ablation_log_odds']<d['val_unans_baseline_log_odds']
    assert d['selected_steering_log_odds']>d['val_ans_baseline_log_odds']
