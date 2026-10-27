"""Registered Tier 2 null-direction construction at the locked residual slice."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np


def fit_activation_covariance(acts: np.ndarray) -> dict:
    """Fit centered Ledoit-Wolf covariance and report effective dimensionality."""
    from sklearn.covariance import LedoitWolf
    x = np.asarray(acts, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("acts must be a finite [N,d] matrix with N >= 2")
    model = LedoitWolf(assume_centered=False).fit(x)
    vals = np.clip(np.linalg.eigvalsh(model.covariance_), 0, None)
    pr = float(vals.sum() ** 2 / np.square(vals).sum()) if np.square(vals).sum() else 0.0
    return {"mean": model.location_, "covariance": model.covariance_, "shrinkage": float(model.shrinkage_),
            "eigenvalues": vals, "participation_ratio": pr}


def _direction_set(directions: np.ndarray, r_norm: float, metadata: dict | None = None) -> dict:
    x = np.asarray(directions, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-12): raise ValueError("cannot normalize a zero direction")
    unit = (x / norms[:, None]).astype(np.float32)
    return {"unit": unit, "steer": (unit * float(r_norm)).astype(np.float32), "metadata": metadata or {}}


def sample_covariance_null(cov: np.ndarray, n: int, seed: int, r_norm: float = 1.0,
                           r_hat: np.ndarray | None = None) -> dict:
    """Sample Gaussian directions in covariance geometry, then unit-normalize."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all(): raise ValueError("cov must be finite square matrix")
    c = (c + c.T) / 2; rng = np.random.default_rng(int(seed))
    try: root = np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(c); root = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None)))
    z = rng.standard_normal((int(n), c.shape[0])) @ root.T
    out = _direction_set(z, r_norm, {"family": "covariance", "seed": int(seed), "n": int(n)})
    out["raw_samples"] = z.astype(np.float32)
    variances = np.einsum("ni,ij,nj->n", out["unit"], c, out["unit"])
    out["metadata"]["projected_variance"] = variances.tolist()
    if r_hat is not None:
        rh = np.asarray(r_hat, dtype=np.float64); rh = rh / np.linalg.norm(rh); rv = float(rh @ c @ rh)
        out["metadata"].update(r_variance=rv, r_variance_percentile=float(100 * np.mean(variances <= rv)))
    return out


def top_pca_directions(cov: np.ndarray, k: int = 50, r_hat: np.ndarray | None = None,
                       r_norm: float = 1.0) -> dict:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64)); order = np.argsort(vals)[::-1][:int(k)]
    dirs = vecs[:, order].T
    metadata = {"family": "pca_descriptive", "eigenvalues": vals[order].tolist()}
    if r_hat is not None:
        rh = np.asarray(r_hat, dtype=np.float64); rh /= np.linalg.norm(rh)
        cos = dirs @ rh; dirs *= np.where(cos < 0, -1., 1.)[:, None]
        all_cos = np.abs(vecs.T @ rh); metadata["r_rank"] = int(np.sum(all_cos > np.max(np.abs(cos)))) + 1
        metadata["r_cosines"] = np.abs(dirs @ rh).tolist()
    return _direction_set(dirs, r_norm, metadata)


def variance_matched_subset(null: dict, cov: np.ndarray, r_hat: np.ndarray, tolerance: float = .25) -> dict:
    rh = np.asarray(r_hat, dtype=np.float64); rh /= np.linalg.norm(rh); c = np.asarray(cov, dtype=np.float64)
    target = float(rh @ c @ rh); var = np.einsum("ni,ij,nj->n", null["unit"], c, null["unit"])
    keep = np.abs(var-target) <= abs(target)*tolerance
    return {"indices": np.flatnonzero(keep), "directions": null["unit"][keep], "target_variance": target, "variances": var[keep]}


def _as_template_tensor(acts: np.ndarray, template_ids: Sequence[Any] | None = None) -> tuple[np.ndarray, list[Any]]:
    a = np.asarray(acts, dtype=np.float64)
    if a.ndim == 3:
        if a.shape[1] != 4: raise ValueError("template tensor must have four instances")
        return a, list(range(a.shape[0]))
    if a.ndim != 2 or template_ids is None or len(template_ids) != len(a): raise ValueError("flat activations require template ids")
    tids = list(dict.fromkeys(template_ids)); groups = [np.flatnonzero(np.asarray(template_ids) == t) for t in tids]
    if any(len(ix) != 4 for ix in groups): raise ValueError("each template must have four instances")
    return np.stack([a[ix] for ix in groups]), tids


def labelfree_null(acts_ans: np.ndarray, acts_unans: np.ndarray, template_ids: Sequence[Any] | None = None,
                   n: int = 199, seed: int = 0, r_norm: float = 1.0) -> dict:
    """Random 2/2 splits within each template and label, averaging labels equally."""
    ans, tids = _as_template_tensor(acts_ans, template_ids); unans, tids_u = _as_template_tensor(acts_unans, template_ids)
    if ans.shape != unans.shape or tids != tids_u: raise ValueError("answerable/unanswerable templates differ")
    rng = np.random.default_rng(int(seed)); draws=[]
    for _ in range(int(n)):
        d=np.zeros(ans.shape[-1])
        for a,u in zip(ans,unans):
            ia=rng.permutation(4); iu=rng.permutation(4)
            d += ((a[ia[:2]].mean(0)-a[ia[2:]].mean(0)) + (u[iu[:2]].mean(0)-u[iu[2:]].mean(0)))/2
        draws.append(d/ans.shape[0])
    return _direction_set(np.asarray(draws), r_norm, {"family":"label_free", "seed":int(seed), "n":int(n)})


def template_partition_null(acts_ans: np.ndarray, acts_unans: np.ndarray, template_ids: Sequence[Any] | None,
                            families: Sequence[str], r_hat: np.ndarray, r_norm: float, n: int, seed: int) -> dict:
    """Family-stratified template partitions; label balance and learned-axis orthogonalization."""
    ans,tids=_as_template_tensor(acts_ans,template_ids); unans,tids_u=_as_template_tensor(acts_unans,template_ids); fam=np.asarray(families)
    if ans.shape!=unans.shape or tids!=tids_u or len(fam)!=len(tids): raise ValueError("template metadata mismatch")
    if len(tids)!=48 or len(set(fam))!=2 or any(np.sum(fam==f)!=24 for f in set(fam)): raise ValueError("requires 48 templates, 24 per family")
    rh=np.asarray(r_hat,dtype=np.float64); rh/=np.linalg.norm(rh); rng=np.random.default_rng(int(seed)); draws=[];partitions=[]
    for _ in range(int(n)):
        g1=[];g2=[]
        for f in sorted(set(fam)):
            ix=np.flatnonzero(fam==f);rng.shuffle(ix);g1.extend(ix[:12]);g2.extend(ix[12:])
        partitions.append({'group1':[tids[i] for i in g1],'group2':[tids[i] for i in g2]})
        d=(ans[g1].mean((0,1))+unans[g1].mean((0,1))-ans[g2].mean((0,1))-unans[g2].mean((0,1)))/2
        draws.append(d-np.dot(d,rh)*rh)
    out=_direction_set(np.asarray(draws),r_norm,{"family":"template_partition","seed":int(seed),"n":int(n),"groups_per_family":12,"template_ids":tids,
        "partitions":partitions,"label_balance_per_group":{"answerable":96,"unanswerable":96}})
    out["metadata"]["max_abs_r_dot"]=float(np.max(np.abs(out["unit"]@rh)))
    return out
