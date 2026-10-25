#!/usr/bin/env python3
"""Export blinded, stratified PopQA grader dev and held-out sheets."""
import argparse,csv,json,random
from collections import defaultdict
from pathlib import Path
from src.config import CFG
from src.eval_common import read_jsonl
LABEL={"correct":"CORRECT","wrong":"INCORRECT","abstain":"ABSTAIN","ambiguous":"UNGRADEABLE"}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=CFG.results_dir);a=ap.parse_args()
    rows=read_jsonl(CFG.data_dir/"popqa_sample.jsonl")
    if not rows or any(not r.get("grade_reason") for r in rows):raise RuntimeError("Requires canonically regenerated and graded PopQA rows")
    rng=random.Random(CFG.grader_validation_seed)
    amb=[r for r in rows if r["baseline_partition"]=="ambiguous"]
    chosen=rng.sample(amb,min(40,len(amb)))
    rest=[r for r in rows if r not in chosen]
    groups=defaultdict(list)
    for r in rest:groups[(r["baseline_partition"],r["grade_reason"])].append(r)
    quota=max(0,200-len(chosen)); keys=list(groups); base={k:quota//len(keys) for k in keys}
    for k in keys[:quota%len(keys)]:base[k]+=1
    for k in keys:chosen.extend(rng.sample(groups[k],min(base[k],len(groups[k]))))
    if len(chosen)<200:
        remain=[r for r in rows if r not in chosen];chosen.extend(rng.sample(remain,min(200-len(chosen),len(remain))))
    # Split by grade/reason strata while preserving the overall 60/140 split.
    split={"dev":[],"test":[]}; by=defaultdict(list)
    for r in chosen:by[(r["baseline_partition"],r["grade_reason"])].append(r)
    for vals in by.values():
        rng.shuffle(vals); n=round(len(vals)*.30);split["dev"].extend(vals[:n]);split["test"].extend(vals[n:])
    while len(split["dev"])<min(60,len(chosen)):
        take=split["test"].pop(0);split["dev"].append(take)
    while len(split["dev"])>60 and split["test"]:
        split["test"].append(split["dev"].pop())
    a.output_dir.mkdir(parents=True,exist_ok=True);key={}
    for name,items in split.items():
        rng.shuffle(items);out=[]
        for r in items:
            uid=f"popqa:{r['popqa_id']}";key[uid]={"grade":r["baseline_partition"],"grade_reason":r["grade_reason"]}
            out.append({"uid":uid,"question":r["question"],"possible_answers":json.dumps(r.get("possible_answers",[]),ensure_ascii=False),
                        "response":r["baseline_response"],"human_label":"","notes":""})
        with (a.output_dir/f"grader_validation_{name}.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    (a.output_dir/"grader_validation_key.json").write_text(json.dumps(key,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (a.output_dir/"grader_validation_manifest.json").write_text(json.dumps({"seed":CFG.grader_validation_seed,"dev_uids":[f"popqa:{r['popqa_id']}" for r in split["dev"]],"test_uids":[f"popqa:{r['popqa_id']}" for r in split["test"]]},indent=2)+"\n",encoding="utf-8")
    print(f"Wrote grader sheets: {len(split['dev'])} dev, {len(split['test'])} held-out")
if __name__=="__main__":main()
