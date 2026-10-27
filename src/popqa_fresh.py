"""Deterministic, relation-exact fresh PopQA sampling and matching for Tier 2."""
from __future__ import annotations
import random
from collections import defaultdict

import numpy as np

from src.config import CFG
from src.pilot_check import compute_log10_s_pop


def sample_fresh_batches(all_rows, archived_hf_row_idxs, batch_index, n=CFG.n_popqa_fresh_batch,
                         seed=CFG.seed_popqa_fresh):
    """Draw one weighted decile batch, excluding every archived row index."""
    excluded={int(x) for x in archived_hf_row_idxs}
    ordered=sorted(all_rows,key=lambda r:(float(r["s_pop"]),int(r["hf_row_idx"])))
    deciles=[[] for _ in range(10)]
    for i,row in enumerate(ordered): deciles[min(9,i*10//len(ordered))].append(row)
    rng=random.Random(int(seed)+int(batch_index)); result=[]
    for di,weight in enumerate([.06]*5+[.14]*5):
        pool=[r for r in deciles[di] if int(r["hf_row_idx"]) not in excluded]
        k=min(round(n*weight),len(pool)); result.extend(rng.sample(pool,k))
    return sorted(result,key=lambda r:int(r["hf_row_idx"]))


def match_relation_exact(rows, min_relation_pool=CFG.min_relation_pool, caliper_fraction=.25,
                         seed=CFG.seed_popqa_fresh):
    """Greedy 1:1 nearest-neighbor correct/wrong matches within exact relation."""
    eligible=defaultdict(list)
    for row in rows: eligible[str(row.get("relation",row.get("prop","")))].append(row)
    excluded=[]; candidates={}
    for rel,group in eligible.items():
        if len(group)<min_relation_pool: excluded.append(rel);continue
        candidates[rel]=group
    vals=np.array([compute_log10_s_pop(r["s_pop"]) for g in candidates.values() for r in g],float)
    sd=float(vals.std(ddof=0)) if len(vals) else 0.
    caliper=float(caliper_fraction*sd)
    rng=random.Random(int(seed));tie={str(r.get("popqa_id",r["hf_row_idx"])):rng.random() for g in candidates.values() for r in g}
    matches=[]
    for rel,group in sorted(candidates.items()):
        wrong=[r for r in group if r.get("grade")=="wrong"]
        correct=[r for r in group if r.get("grade")=="correct"]
        edges=[]
        for w in wrong:
            for c in correct:
                d=abs(compute_log10_s_pop(w["s_pop"])-compute_log10_s_pop(c["s_pop"]))
                if d<=caliper:
                    wi=str(w.get("popqa_id",w["hf_row_idx"]));ci=str(c.get("popqa_id",c["hf_row_idx"]))
                    edges.append((d,tie[wi],tie[ci],wi,ci,w,c))
        used=set()
        for _,_,_,wi,ci,w,c in sorted(edges):
            if wi in used or ci in used:continue
            used|={wi,ci};matches.append((rel,w,c))
    for r in rows:r["is_matched_v5"]=False;r["matched_pair_id_v5"]=None
    for i,(rel,w,c) in enumerate(matches,1):
        pair_id=f"v5_{i:05d}"
        w["is_matched_v5"]=c["is_matched_v5"]=True
        w["matched_pair_id_v5"]=c["matched_pair_id_v5"]=pair_id
    return {"pairs":matches,"n_pairs":len(matches),"caliper":caliper,"global_sd":sd,
            "excluded_relations":sorted(excluded),"relations_used":len(candidates)}


def verify_grader_transfer(human_labels, graded_rows):
    """Check 100 blind labels against the automated correct/not-correct decision."""
    if len(human_labels)!=100:raise ValueError("exactly 100 blinded labels are required")
    tp=fp=fn=0
    for human,row in zip(human_labels,graded_rows):
        label=str(human).strip().lower()
        if label not in {"correct","wrong","incorrect","abstain","ambiguous","true","false","1","0"}:
            raise ValueError(f"invalid blinded human grade: {human!r}")
        h=label in {"correct","true","1"};g=row.get("grade")=="correct"
        tp+=h and g;fp+=(not h) and g;fn+=h and (not g)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return {"n":100,"precision_correct":precision,"recall_correct":recall,
            "passed":precision>=.95 and recall>=.90}
