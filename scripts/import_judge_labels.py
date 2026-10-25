#!/usr/bin/env python3
"""Score a completed blinded held-out judge-validation sheet."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from src.config import CFG
LABELS=["ANSWER","ABSTAIN","DEGENERATE"]

def _group_stats(ix, human, pred):
    if not ix: return {"n":0,"cohen_kappa":None,"abstain_recall":None}
    h=np.asarray(human)[ix]; q=np.asarray(pred)[ix]
    recall=precision_recall_fscore_support(h,q,labels=LABELS,zero_division=0)[1]
    return {"n":len(ix),"cohen_kappa":float(cohen_kappa_score(h,q,labels=LABELS)),
            "abstain_recall":float(recall[LABELS.index("ABSTAIN")])}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv",type=Path,default=CFG.results_dir/"judge_validation_test.csv")
    p.add_argument("--output",type=Path,default=CFG.results_dir/"judge_validation.json"); a=p.parse_args()
    rows=list(csv.DictReader(a.csv.open(encoding="utf-8",newline="")))
    if not rows or any(r.get("human_label") not in LABELS for r in rows):
        raise ValueError("Every held-out row must have human_label ANSWER, ABSTAIN, or DEGENERATE")
    key=json.loads((a.csv.parent/"judge_validation_key.json").read_text(encoding="utf-8"))
    manifest_path=a.csv.parent/"judge_validation_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    from src.abstention_classifier import JUDGE_PROMPT_SHA256, JUDGE_PROMPT_VERSION, _revision
    if manifest and (manifest.get("judge_model_revision")!=_revision() or manifest.get("judge_prompt_sha256")!=JUDGE_PROMPT_SHA256 or manifest.get("judge_prompt_version")!=JUDGE_PROMPT_VERSION):
        raise ValueError("Validation sheet was prepared for a different judge revision or prompt")
    pred=[key[r["uid"]]["final_label"] for r in rows]; human=[r["human_label"] for r in rows]
    margins=[key[r["uid"]].get("judge_margin") for r in rows]
    if any(x not in LABELS for x in pred): raise ValueError("Judge labels contain invalid values")
    k=cohen_kappa_score(human,pred,labels=LABELS); rng=np.random.default_rng(CFG.judge_validation_seed); boots=[]
    for _ in range(1000):
        ix=rng.integers(0,len(rows),len(rows)); boots.append(cohen_kappa_score(np.array(human)[ix],np.array(pred)[ix],labels=LABELS))
    pr,rc,_,_=precision_recall_fscore_support(human,pred,labels=LABELS,zero_division=0)
    regex=[key[r["uid"]]["regex_label"] for r in rows]
    low=[i for i,m in enumerate(margins) if m is not None and float(m)<1.0]
    high=[i for i,m in enumerate(margins) if m is not None and float(m)>=1.0]
    low_stats=_group_stats(low,human,pred); high_stats=_group_stats(high,human,pred)
    split="dev" if "dev" in a.csv.stem.lower() else "test"
    out={"split":split,"n":len(rows),"labels":LABELS,"judge_model_id":manifest.get("judge_model_id"),
         "judge_model_revision":manifest.get("judge_model_revision"),"judge_prompt_version":manifest.get("judge_prompt_version"),
         "judge_prompt_sha256":manifest.get("judge_prompt_sha256"),"cohen_kappa":float(k),
         "kappa_bootstrap_ci":[float(x) for x in np.percentile(boots,[2.5,97.5])],
         "precision":dict(zip(LABELS,map(float,pr))),"recall":dict(zip(LABELS,map(float,rc))),
         "confusion_matrix_rows_human_cols_judge":confusion_matrix(human,pred,labels=LABELS).tolist(),
         "regex_human_agreement":float(np.mean(np.array(regex)==np.array(human))),
         "margin_threshold_nat":1.0,"low_margin":low_stats,"high_margin":high_stats,
         "gate_passed":bool(split=="test" and k>=.90 and rc[LABELS.index("ABSTAIN")]>=.95),
         "low_margin_gate_passed":(None if low_stats["n"]==0 else bool(low_stats["cohen_kappa"]>=.90 and low_stats["abstain_recall"]>=.95))}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
