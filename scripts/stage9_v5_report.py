#!/usr/bin/env python3
"""Combine Tier 2 confirmatory results with Holm v5 and interpretation v5."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from src.config import CFG
from src.eval_common import read_json
from src.stats_utils import holm_bonferroni
from src.stage8_v5 import classify_interpretation_v5

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage6b',type=Path,default=CFG.results_dir/'stage6b_null_eval.json');p.add_argument('--stage8v5',type=Path,default=CFG.results_dir/'stage8v5.json');p.add_argument('--stage6',type=Path,default=CFG.stage6_out_path);p.add_argument('--stage8v4',type=Path,default=CFG.stage8_out_path);p.add_argument('--stage4b',type=Path,default=CFG.results_dir/'stage4b_nulls.json');p.add_argument('--out',type=Path,default=CFG.results_dir/'stage9_v5_summary.json');p.add_argument('--readme',type=Path,default=CFG.readme_path);a=p.parse_args()
 s6b=read_json(a.stage6b);s8=read_json(a.stage8v5);s6=read_json(a.stage6);s8v4=read_json(a.stage8v4)
 g=s6b['generated_primary'];h1=g['H1_prime'];h2=g['H2_prime'];h3=s6['primary_tests']['H3'];h4=s8['h4'];h5=s8['h5']
 pvals={'H1_prime':h1['iut_p'],'H2_prime':h2['iut_p'],'H3':float(h3['p_wilcoxon']),'H4_prime':float(h4['iut']['p_composite']),'H5_prime':float(h5['p_one_sided'])}
 bctrl=float(h3.get('b_ctrl_first_token_abstention_rate',1.0));criteria={'H1_prime':not h1['concept_confounded'],'H2_prime':not h2['concept_confounded'],'H3':bctrl<=CFG.b_ctrl_abstention_max,'H4_prime':bool(h4['point_criterion']),'H5_prime':True}
 holm=holm_bonferroni(pvals,CFG.alpha_fwer,criteria);by={r['test_id']:r for r in holm}
 interp=classify_interpretation_v5(holm,by['H1_prime']['confirmed'],by['H2_prime']['confirmed'],by['H3']['confirmed'],
  h1['concept_confounded'] or h2['concept_confounded'],by['H4_prime']['confirmed'],by['H5_prime']['confirmed'],h5['ci95_high'],
  h4['stratified']['auroc'],s8.get('abstain_vs_correct_projection_auroc') or .5,
  overall_v4=s8v4.get('interpretation',{}).get('overall'),overall_v41=s8v4.get('interpretation',{}).get('overall_as_registered_v4'))
 result={'stage':'9-v5','confirmatory_family':pvals,'holm_v5':holm,'interpretation':interp,'h1_prime':h1,'h2_prime':h2,'h3_tier1':h3,'h4_prime':h4,'h5_prime':h5,
  'stage6b_sha256':__import__('src.eval_common',fromlist=['sha256_file']).sha256_file(a.stage6b),'stage8v5_sha256':__import__('src.eval_common',fromlist=['sha256_file']).sha256_file(a.stage8v5)}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(f"v5 verdict: {interp['overall_v5']}")
 # Human-readable Stage 9 outputs, written only after the v5 result files exist.
 binding={'H1_prime':h1['binding_component'],'H2_prime':h2['binding_component'],'H3':'Tier 1 Wilcoxon','H4_prime':h4['iut']['binding_component'],'H5_prime':'paired pair-cluster bootstrap'}
 components={'H1_prime':h1['components'],'H2_prime':h2['components'],'H3':{'wilcoxon':h3['p_wilcoxon']},'H4_prime':h4['iut']['components'],'H5_prime':{'pair_cluster_bootstrap':h5['p_one_sided']}}
 table1=['| Test | Component p-values | Composite p | Binding component | Holm threshold | Decision |','|---|---|---:|---|---:|---|']
 for row in holm:table1.append(f"| {row['test_id']} | `{json.dumps(components[row['test_id']],sort_keys=True)}` | {row['raw_p_value']:.6g} | {binding[row['test_id']]} | {row['holm_threshold']:.4g} | {'pass' if row['confirmed'] else 'fail'} |")
 (CFG.tables_dir/'table1_v5.md').write_text('# Table 1 v5 — confirmatory tests\n\n'+'\n'.join(table1)+'\n',encoding='utf-8')
 secondary=h5['secondary'];table8=['| Metric | Result |','|---|---:|',f"| H4' relation-stratified AUROC | {h4['stratified']['auroc']:.4f} |",f"| H4' 95% pair-cluster CI | [{h4['pair_cluster_bootstrap']['ci95_low']:.4f}, {h4['pair_cluster_bootstrap']['ci95_high']:.4f}] |",f"| Relation-only leave-one-out AUROC | {s8['relation_only_leave_one_out_auroc']:.4f} |",f"| H5' delta AUROC | {h5['delta_auroc_cv']:.4f} |",f"| H5' 95% CI | [{h5['ci95_low']:.4f}, {h5['ci95_high']:.4f}] |",f"| H5' upper CI below SESOI 0.02 | {h5['equivalent_below_sesoi']} |",f"| BASE+relation FE CV AUROC | {secondary['relation_fixed_effects']['base_auroc_cv']:.4f} |",f"| FULL+relation FE CV AUROC | {secondary['relation_fixed_effects']['full_auroc_cv']:.4f} |",f"| LRT adding proj p-value | {secondary['likelihood_ratio_add_proj']['p_value']:.6g} |"]
 (CFG.tables_dir/'table8_popqa_v5.md').write_text('# Table 8 — PopQA v5\n\n'+'\n'.join(table8)+'\n\nPer-relation results are in `results/stage8v5.json`.\n',encoding='utf-8')
 stage4b=read_json(a.stage4b);table6=['| Attribute | Level | Group sizes | Pre-orthogonalization cosine | H1′ effect | H1′ ratio | H2′ effect | H2′ ratio |','|---|---|---|---:|---:|---:|---:|---:|']
 effects1={x['name']:x['effect'] for x in h1.get('named_concept_effects',[])};effects2={x['name']:x['effect'] for x in h2.get('named_concept_effects',[])}
 for c in stage4b.get('concept_contrasts',[]):
  key=c['attribute']+':'+c['level'];e1=effects1.get(key,0.);e2=effects2.get(key,0.)
  table6.append(f"| {c['attribute']} | {c['level']} | {json.dumps(c['group_sizes'],sort_keys=True)} | {c['pre_orthogonalization_cosine']:.5f} | {e1:.5f} | {e1/(h1['learned_cluster_effect'] or 1.):.4f} | {e2:.5f} | {e2/(h2['learned_cluster_effect'] or 1.):.4f} |")
 (CFG.tables_dir/'table6_named_concepts.md').write_text('# Table 6 — Named concept controls\n\n'+'\n'.join(table6)+'\n',encoding='utf-8')
 table7=['| Part | alpha | alpha·||r|| / median non-BOS norm | Learned ΔP_abs | Null mean | Net effect |','|---|---:|---:|---:|---:|---:|']
 scales={x['alpha']:x for x in s6b['steering_scale']}
 for row in s6b['dose_summary']:table7.append(f"| {row['part']} | {row['alpha']:.2f} | {scales[row['alpha']]['fraction_median_non_bos']:.5f} | {row['learned_delta_p_abs']:.5f} | {row['null_mean']} | {row['net_effect']} |")
 (CFG.tables_dir/'table7_steering_scale.md').write_text('# Table 7 — Steering scale and net effects\n\n'+'\n'.join(table7)+'\n',encoding='utf-8')
 table5=['| Null family | N | H1′ p | H2′ p | H4′ p | H4′ mean AUROC | H4′ SD |','|---|---:|---:|---:|---:|---:|---:|']
 for family in ('covariance','label_free','template_partition'):
  n=len(h1['nulls'][family]['effects']);h4n=h4.get(family+'_null',{});table5.append(f"| {family} | {n} | {h1['nulls'][family]['p_value']:.5g} | {h2['nulls'][family]['p_value']:.5g} | {h4n.get('p_value',float('nan')):.5g} | {h4n.get('mean_auroc',float('nan')):.4f} | {h4n.get('sd_auroc',float('nan')):.4f} |")
 table5.append(f"\nLearned direction projected-variance percentile: {stage4b['r_variance_percentile']:.2f}.")
 (CFG.tables_dir/'table5_nulls.md').write_text('# Table 5 — Null families\n\n'+'\n'.join(table5)+'\n',encoding='utf-8')
 # Replace only a clearly delimited generated summary block; preserve the prior v4 findings.
 marker='<!-- BEGIN_TIER2_V5_SUMMARY -->';end='<!-- END_TIER2_V5_SUMMARY -->';readme=a.readme.read_text(encoding='utf-8');block=(marker+'\n\n'+f"- **Tier 2 v5 verdict:** {interp['overall_v5']}\n- **Confirmatory:** H4′ and H5′ use fresh PopQA; H1′/H2′ are re-analyses on seen test templates; H3 is the Tier 1 judged control.\n\n"+end)
 if marker in readme:
  start=readme.index(marker);stop=readme.index(end,start)+len(end);readme=readme[:start]+block+readme[stop:]
 else:
  pos=readme.find('\n## Status')
  readme=readme[:pos]+'\n'+block+'\n'+readme[pos:] if pos>=0 else readme+'\n'+block+'\n'
 a.readme.write_text(readme,encoding='utf-8')
 # Figure outputs are generated from saved JSON, never by revisiting fresh data.
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(6.2,4.2))
 for i,(part,label) in enumerate([('unans','H1′ necessity'),('ans','H2′ sufficiency')]):
  x=s6b['null_generation_comparison'][part];base=x['baseline_rate'];learned=x['learned_rate'];ax.bar(i-.18,base,.34,label='Baseline' if i==0 else None);ax.bar(i+.18,learned,.34,label='Learned' if i==0 else None)
  null_rates=[base-v if part=='unans' else base+v for v in x['null_deltas']];ax.errorbar(i+.52,np.mean(null_rates),yerr=[[np.mean(null_rates)-np.percentile(null_rates,2.5)],[np.percentile(null_rates,97.5)-np.mean(null_rates)]],fmt='o',capsize=4,label='Covariance-null 95% envelope' if i==0 else None)
 ax.set_xticks([0,1],['H1′ unanswerable','H2′ answerable']);ax.set_ylabel('Judged generation abstention rate');ax.legend();fig.tight_layout();fig.savefig(CFG.figures_dir/'fig1_v5.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(15,6.8))
 plot_specs=[('H1′ covariance',h1['nulls']['covariance']['effects'],h1['learned_cluster_effect']),('H1′ label-free',h1['nulls']['label_free']['effects'],h1['learned_cluster_effect']),('H1′ template-partition',h1['nulls']['template_partition']['effects'],h1['learned_cluster_effect']),('H2′ covariance',h2['nulls']['covariance']['effects'],h2['learned_cluster_effect']),('H2′ label-free',h2['nulls']['label_free']['effects'],h2['learned_cluster_effect']),('H2′ template-partition',h2['nulls']['template_partition']['effects'],h2['learned_cluster_effect']),('H4′ covariance',h4['covariance_null']['aurocs'],h4['stratified']['auroc']),('H4′ template-partition',h4['template_partition_null']['aurocs'],h4['stratified']['auroc'])]
 for ax,(title,vals,obs) in zip(axs.flat,plot_specs):
  ax.hist(vals,bins=24,color='#8aaed0',edgecolor='white');ax.axvline(obs,color='#b33434',lw=2,label='Learned');ax.set_title(title);ax.legend()
 fig.tight_layout();fig.savefig(CFG.figures_dir/'fig3_v5_nulls.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4.2))
 for part,label,color in [('unans','Unanswerable','#345995'),('ans','Answerable','#e40066')]:
  rr=[r for r in s6b['dose_summary'] if r['part']==part];xx=[scales[r['alpha']]['fraction_median_non_bos'] for r in rr];ax.plot(xx,[r['learned_delta_p_abs'] for r in rr],marker='o',label=label,color=color);ax.fill_between(xx,[r['null_p2_5'] for r in rr],[r['null_p97_5'] for r in rr],alpha=.18,color=color)
  ax.plot(xx,[r['net_effect'] for r in rr],linestyle='--',color=color,alpha=.85,label=label+' net')
 ax.axhline(0,color='black',lw=.7);ax.set(xlabel='α·||r|| / median non-BOS residual norm',ylabel='Change in P_abs');ax.legend();fig.tight_layout();fig.savefig(CFG.figures_dir/'fig4_v5_dose.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5.4,4.2));vals=[h5['base_auroc_cv'],h5['full_auroc_cv']];ax.bar([0,1],vals,color=['#8aaed0','#345995']);ax.errorbar(1,h5['full_auroc_cv'],yerr=[[h5['full_auroc_cv']-(h5['base_auroc_cv']+h5['ci95_low'])],[h5['base_auroc_cv']+h5['ci95_high']-h5['full_auroc_cv']]],fmt='none',ecolor='black',capsize=4);ax.set_xticks([0,1],['BASE','FULL']);ax.set_ylabel('Repeated group-CV AUROC');ax.axhspan(h5['base_auroc_cv'],h5['base_auroc_cv']+.02,color='#f4cc6b',alpha=.25,label='SESOI ΔAUROC=0.02');ax.legend();fig.tight_layout();fig.savefig(CFG.figures_dir/'fig5_v5_incremental.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
