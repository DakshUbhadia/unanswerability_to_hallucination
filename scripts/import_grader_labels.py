#!/usr/bin/env python3
"""Score held-out human labels for the PopQA response grader."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from src.config import CFG
LABELS=["CORRECT","INCORRECT","ABSTAIN","UNGRADEABLE"]
TO_GRADE={"CORRECT":"correct","INCORRECT":"wrong","ABSTAIN":"abstain","UNGRADEABLE":"ambiguous"}
def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",type=Path,default=CFG.results_dir/"grader_validation_test.csv");p.add_argument("--output",type=Path,default=CFG.results_dir/"grader_validation.json");p.add_argument("--key",type=Path,default=CFG.results_dir/"grader_validation_key.json");a=p.parse_args()
    rows=list(csv.DictReader(a.csv.open(encoding="utf-8",newline="")))
    if not rows or any(r.get("human_label") not in LABELS for r in rows):raise ValueError("Every held-out row needs CORRECT, INCORRECT, ABSTAIN, or UNGRADEABLE")
    key=json.loads(a.key.read_text(encoding="utf-8"))
    human=[r["human_label"] for r in rows]; pred=[{"correct":"CORRECT","wrong":"INCORRECT","abstain":"ABSTAIN","ambiguous":"UNGRADEABLE"}[key[r["uid"]]["grade"]] for r in rows]
    pr,rc,_,_=precision_recall_fscore_support(human,pred,labels=LABELS,zero_division=0)
    correct=[i for i,x in enumerate(pred) if x=="CORRECT"];tp=sum(human[i]=="CORRECT" for i in correct)
    precision=tp/len(correct) if correct else 0.; actual=sum(x=="CORRECT" for x in human);recall=tp/actual if actual else 0.
    by=defaultdict(lambda:{"n":0,"correct":0,"human_correct":0})
    for r,h,predicted in zip(rows,human,pred):
        reason=key[r["uid"]]["grade_reason"];by[reason]["n"]+=1;by[reason]["correct"]+=predicted=="CORRECT";by[reason]["human_correct"]+=h=="CORRECT"
    ambiguous=[i for i,r in enumerate(rows) if key[r["uid"]]["grade"]=="ambiguous"]
    out={"n":len(rows),"precision_correct":precision,"recall_correct":recall,"accuracy":float(np.mean(np.array(human)==np.array(pred))),
      "precision_by_label":dict(zip(LABELS,map(float,pr))),"recall_by_label":dict(zip(LABELS,map(float,rc))),
      "confusion_matrix_rows_human_cols_grader":confusion_matrix(human,pred,labels=LABELS).tolist(),
      "by_grade_reason":dict(by),"ambiguous_n":len(ambiguous),"ambiguous_human_resolution":{human[i]:sum(human[j]==human[i] for j in ambiguous) for i in ambiguous},
      "gate_passed":bool(precision>=.95 and recall>=.90)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
