#!/usr/bin/env python3
"""CPU gate for the Stage 4b pool and null-library invariants."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from src.config import CFG
from scripts.verify_archive_v42 import verify as verify_archive

def verify(pool_path=CFG.results_dir/'stage4b_pool_acts.pt',null_path=CFG.results_dir/'stage4b_nulls.pt',meta_path=CFG.results_dir/'stage4b_nulls.json'):
 pool=torch.load(pool_path,map_location='cpu',weights_only=False);nulls=torch.load(null_path,map_location='cpu',weights_only=False);meta=json.loads(Path(meta_path).read_text(encoding='utf-8'))
 acts=pool['acts'];rows=pool['rows'];assert acts.ndim==2 and torch.isfinite(acts).all();assert len(rows)==len(acts)
 ids=[r['source_id'] for r in rows];assert len(ids)==len(set(ids))
 assert all(r.get('input_ids_sha256') for r in rows),'pool input hash missing'
 assert not any(r['source']=='test' or r['source']=='b_ctrl' or r['source']=='fresh_popqa' for r in rows)
 assert int(meta['covariance_null_n'])==CFG.n_null_cov and len(nulls['covariance']['unit'])==CFG.n_null_cov
 assert len(nulls['label_free']['unit'])==CFG.n_null_labelfree and len(nulls['template_partition']['unit'])==CFG.n_null_template_partition
 assert len(nulls['pca']['unit'])==CFG.n_pca_null
 learned=float(meta['label_free_mean_abs_cosine']);assert learned<.1
 rh_cos=float(meta['template_max_abs_cosine']);assert rh_cos<1e-5
 rh=np.asarray(meta['r_hat'],dtype=float);rh/=np.linalg.norm(rh)
 assert np.max(np.abs(nulls['template_partition']['unit']@rh))<1e-5
 rnorm=float(nulls['covariance']['steer'].norm(dim=1).median());
 for family in ('covariance','label_free','template_partition','pca'):
  assert torch.max(torch.abs(torch.linalg.vector_norm(nulls[family]['steer'],dim=1)-rnorm))<1e-5
 assert len(nulls['concepts'])>=1
 for row in nulls['concepts']:
  assert abs(float(np.dot(row['unit'],rh)))<1e-5
  assert abs(float(np.linalg.norm(row['steer']))-float(meta['r_norm']))<1e-5
 return {'pool_rows':len(rows),'d_model':acts.shape[1],'covariance_n':len(nulls['covariance']['unit']),'label_free_cosine':learned,'template_max_abs_cosine':rh_cos,'archive_files_verified':verify_archive()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,default=CFG.results_dir/'stage4b_pool_acts.pt');p.add_argument('--nulls',type=Path,default=CFG.results_dir/'stage4b_nulls.pt');p.add_argument('--metadata',type=Path,default=CFG.results_dir/'stage4b_nulls.json');a=p.parse_args();print(json.dumps(verify(a.pool,a.nulls,a.metadata),indent=2))
